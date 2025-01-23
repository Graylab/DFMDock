import copy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
from torch.utils import data
from torch_geometric.loader import DataLoader
from transformers import AutoModelForMaskedLM
from omegaconf import DictConfig
from dfmdock.models.score_net_attn import Score_Net
from dfmdock.utils.so3_diffuser import SO3Diffuser 
from dfmdock.utils.r3_diffuser import R3Diffuser 
from dfmdock.utils.geometry import axis_angle_to_matrix
from dfmdock.utils.crop import get_crop
from dfmdock.datasets.ppi_dataset import PPIDataset


#----------------------------------------------------------------------------
# Main wrapper for training the model

class Score_Model(pl.LightningModule):
    def __init__(
        self,
        model,
        diffuser,
        experiment,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = experiment.lr
        self.weight_decay = experiment.weight_decay
        self.crop_size = experiment.crop_size

        # energy
        self.grad_energy = experiment.grad_energy
        self.separate_energy_loss = experiment.separate_energy_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        # interface 
        self.use_interface_loss = experiment.use_interface_loss

        # contrastive
        self.use_contrastive_loss = experiment.use_contrastive_loss

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # Load esmfold
        self.esm_model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_large', trust_remote_code=True).eval().to(self.device)
        self.tokenizer = self.esm_model.tokenizer

        # net
        self.net = Score_Net(model)
    
    def forward(self, batch):
        outputs = self.net(batch, predict=True)
        return outputs

    def loss_fn(self, batch, eps=1e-5):
        with torch.no_grad():
            # uniformly sample a timestep
            t = torch.rand(1, device=self.device) * (1. - eps) + eps
            batch["t"] = t

            # sample perturbation for translation and rotation
            if self.perturb_tr:
                tr_score_scale = self.r3_diffuser.score_scaling(t.item())
                tr_sigma = torch.tensor(self.r3_diffuser.sigma(t.item())).float().to(self.device)
                tr_update, tr_score_gt = self.r3_diffuser.forward_marginal(t.item())
                tr_update = torch.from_numpy(tr_update).float().to(self.device)
                tr_score_gt = torch.from_numpy(tr_score_gt).float().to(self.device)
            else:
                tr_update = np.zeros(3)
                tr_update = torch.from_numpy(tr_update).float().to(self.device)

            if self.perturb_rot:
                rot_score_scale = self.so3_diffuser.score_scaling(t.item())
                rot_sigma = torch.tensor(self.so3_diffuser.sigma(t.item())).float().to(self.device)
                rot_update, rot_score_gt = self.so3_diffuser.forward_marginal(t.item())
                rot_update = torch.from_numpy(rot_update).float().to(self.device)
                rot_score_gt = torch.from_numpy(rot_score_gt).float().to(self.device)
            else:
                rot_update = np.zeros(3)
                rot_update = torch.from_numpy(rot_update).float().to(self.device)

            batch = get_crop(batch, crop_size=self.crop_size)

            # update poses          
            batch["lig_pos"] = self.modify_coords(batch["lig_pos"], rot_update, tr_update)

        # predict score based on the current state
        if self.grad_energy:
            outputs = self.net(batch)

            # grab some outputs
            tr_score = outputs["tr_score"]
            rot_score = outputs["rot_score"]
            f = outputs["f"]
            dedx = outputs["dedx"]
            energy_noised = outputs["energy"]

            # energy conservation loss
            if self.separate_energy_loss:
                f_angle = torch.norm(f, dim=-1, keepdim=True)
                f_axis = f / (f_angle + 1e-6)

                dedx_angle = torch.norm(dedx, dim=-1, keepdim=True)
                dedx_axis = dedx / (dedx_angle + 1e-6)

                ec_axis_loss = torch.mean((f_axis - dedx_axis)**2)
                ec_angle_loss = torch.mean((f_angle - dedx_angle)**2)
                ec_loss = 0.5 * (ec_axis_loss + ec_angle_loss)
                
            else:
                ec_loss = torch.mean((dedx - f)**2)
        else:
            outputs = self.net(batch, predict=True)

            # grab some outputs
            tr_score = outputs["tr_score"]
            rot_score = outputs["rot_score"]
            energy_noised = outputs["energy"]
            
            # energy conservation loss
            ec_loss = torch.tensor(0.0, device=self.device)

        # translation loss
        if self.perturb_tr:
            if self.separate_tr_loss:
                gt_tr_angle = torch.norm(tr_score_gt, dim=-1, keepdim=True)
                gt_tr_axis = tr_score_gt / (gt_tr_angle + 1e-6)

                pred_tr_angle = torch.norm(tr_score, dim=-1, keepdim=True)
                pred_tr_axis = tr_score / (pred_tr_angle + 1e-6)

                tr_axis_loss = torch.mean((pred_tr_axis - gt_tr_axis)**2)
                tr_angle_loss = torch.mean((pred_tr_angle - gt_tr_angle)**2 / tr_score_scale**2)
                tr_loss = 0.5 * (tr_axis_loss + tr_angle_loss)

            else:
                tr_loss = torch.mean((tr_score - tr_score_gt)**2 / tr_score_scale**2)
        else:
            tr_loss = torch.tensor(0.0, device=self.device)

        # rotation loss
        if self.perturb_rot:
            if self.separate_rot_loss:
                gt_rot_angle = torch.norm(rot_score_gt, dim=-1, keepdim=True)
                gt_rot_axis = rot_score_gt / (gt_rot_angle + 1e-6)

                pred_rot_angle = torch.norm(rot_score, dim=-1, keepdim=True)
                pred_rot_axis = rot_score / (pred_rot_angle + 1e-6)

                rot_axis_loss = torch.mean((pred_rot_axis - gt_rot_axis)**2)
                rot_angle_loss = torch.mean((pred_rot_angle - gt_rot_angle)**2 / rot_score_scale**2)
                rot_loss = 0.5 * (rot_axis_loss + rot_angle_loss)

            else:
                rot_loss = torch.mean((rot_score - rot_score_gt)**2 / rot_score_scale**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)
        
        # interface loss
        bce_logits_loss = nn.BCEWithLogitsLoss()
        if self.use_interface_loss:
            ires_loss = bce_logits_loss(outputs['ires'], batch['ires'])
        else:
            ires_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + ec_loss + ires_loss         
        losses = {"tr_loss": tr_loss, "rot_loss": rot_loss, "ec_loss": ec_loss, "ires_loss": ires_loss, "loss": loss}

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        cen = lig_pos[..., 1, :].mean(dim=0)
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - cen) @ rot.T + cen
        lig_pos = lig_pos + tr
        return lig_pos

    def get_esm_rep(self, sequence):
        tokenized = self.tokenizer(sequence, padding=False, return_tensors='pt')
        tokenized = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
        with torch.no_grad():
            output = self.esm_model(**tokenized, output_attentions=True)
        single_rep = output.last_hidden_state[0, 1:-1, :]
        pair_rep = torch.cat(output.attentions, dim=1).permute(0, 2, 3, 1)[0, 1:-1, 1:-1, :]
        return single_rep, pair_rep

    def step(self, batch, batch_idx):
        rec_seq = batch['rec_seq'][0]
        lig_seq = batch['lig_seq'][0]
        rec_onehot = batch['rec_onehot'].squeeze(0)
        lig_onehot = batch['lig_onehot'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        position_matrix = batch['position_matrix'].squeeze(0)
        ires = batch['ires'].squeeze(0)
        
        # get esm
        x, pair_matrix = self.get_esm_rep(rec_seq + lig_seq)
        rec_x = torch.cat([x[:rec_pos.size(0)], rec_onehot], dim=-1)
        lig_x = torch.cat([x[rec_pos.size(0):], lig_onehot], dim=-1)

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "position_matrix": position_matrix,
            "pair_matrix": pair_matrix,
            "ires": ires,
        }

        # get losses
        losses = self.loss_fn(batch)
        return losses
    
    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"train/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
    
    def on_validation_model_train(self, *args, **kwargs):
        super().on_validation_model_train(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"val/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"test/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer

#----------------------------------------------------------------------------
# Helpers

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd

def distogram_loss(
    logits,
    coords,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    """
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2


    dists = torch.sum(
        (coords[:, None, 1, :] - coords[None, :, 1, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        F.one_hot(true_bins, no_bins),
    )

    loss = torch.mean(errors)

    return loss

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss
#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="score_model_attn.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='dips_train',
    )

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = Score_Model(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='auto', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
