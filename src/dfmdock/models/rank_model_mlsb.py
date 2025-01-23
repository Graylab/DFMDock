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
from omegaconf import DictConfig
from dfmdock.models.rank_net_mlsb import Rank_Net
from dfmdock.utils.so3_diffuser import SO3Diffuser 
from dfmdock.utils.r3_diffuser import R3Diffuser 
from dfmdock.utils.geometry import axis_angle_to_matrix
from dfmdock.utils.dockq import get_DockQ
from dfmdock.datasets.ppi_mlsb_dataset import PPIDataset

#----------------------------------------------------------------------------
# Main wrapper for training the model

class Rank_Model(pl.LightningModule):
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

        # net
        self.net = Rank_Net(model)
    
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
                tr_update, tr_score_gt = self.r3_diffuser.forward_marginal(t.item())
                tr_update = torch.from_numpy(tr_update).float().to(self.device)
                tr_score_gt = torch.from_numpy(tr_score_gt).float().to(self.device)
            else:
                tr_update = np.zeros(3)
                tr_update = torch.from_numpy(tr_update).float().to(self.device)

            if self.perturb_rot:
                rot_score_scale = self.so3_diffuser.score_scaling(t.item())
                rot_update, rot_score_gt = self.so3_diffuser.forward_marginal(t.item())
                rot_update = torch.from_numpy(rot_update).float().to(self.device)
                rot_score_gt = torch.from_numpy(rot_score_gt).float().to(self.device)
            else:
                rot_update = np.zeros(3)
                rot_update = torch.from_numpy(rot_update).float().to(self.device)

            # save gt state
            batch_gt = copy.deepcopy(batch)

            # update poses          
            batch["lig_pos"] = self.modify_coords(batch["lig_pos"], rot_update, tr_update)

            # get lrmsd
            lrmsd = get_rmsd(batch["lig_pos"][..., 1, :], batch_gt["lig_pos"][..., 1, :])
        
        logits = self.net(batch)

        # interface loss
        bce_logits_loss = nn.BCEWithLogitsLoss()
        loss = bce_logits_loss(logits, (lrmsd < 5.0).float())

        # total losses
        losses = {"loss": loss}

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        cen = lig_pos[..., 1, :].mean(dim=0)
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - cen) @ rot.T + cen
        lig_pos = lig_pos + tr
        return lig_pos

    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        position_matrix = batch['position_matrix'].squeeze(0)
        ires = batch['ires'].squeeze(0)

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "position_matrix": position_matrix,
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

#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="rank_model_mlsb.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='dips_train',
        crop_size=500,
    )

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = Rank_Model(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
