import copy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
import importlib
from torch.utils import data
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation
from omegaconf import DictConfig
from dfmdock.utils.so3_diffuser import SO3Diffuser 
from dfmdock.utils.r3_diffuser import R3Diffuser 
from dfmdock.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from dfmdock.datasets.ppi_mlsb_dataset import PPIDataset

#----------------------------------------------------------------------------
# Main wrapper for training the model

class Score_Model(pl.LightningModule):
    def __init__(
        self,
        model,
        diffuser,
        experiment,
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        if self.debug:
            self.automatic_optimization = False
        self.save_hyperparameters()
        self.lr = experiment.lr
        self.weight_decay = experiment.weight_decay

        # energy
        self.grad_energy = experiment.grad_energy
        
        # translation
        self.perturb_tr = experiment.perturb_tr

        # rotation
        self.perturb_rot = experiment.perturb_rot

        # contrastive
        self.use_contrastive_loss = experiment.use_contrastive_loss

        # interface 
        self.use_interface_loss = experiment.use_interface_loss

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # net
        module = importlib.import_module(f"dfmdock.models.{model.file_name}")
        self.net = module.Score_Net(model)
    
    def forward(self, batch):
        outputs = self.net(batch, predict=True)
        return outputs

    def get_energy(self, batch):
        energy = self.net(batch, return_energy=True)
        return energy

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

        # predict score based on the current state
        if self.grad_energy:
            outputs = self.net(batch)

            # grab some outputs
            tr_score = outputs["tr_score"]
            rot_score = outputs["rot_score"]
            tr_grad = outputs["tr_grad"]
            rot_grad = outputs["rot_score"]
            energy_noised = outputs["energy"]
        else:
            outputs = self.net(batch, predict=True)

            # grab some outputs
            tr_score = outputs["tr_score"]
            rot_score = outputs["rot_score"]
            energy_noised = outputs["energy"]
            
        # translation loss
        if self.perturb_tr:
            # gt
            gt_tr_mag = torch.norm(tr_score_gt, dim=-1, keepdim=True)
            gt_tr_dir = tr_score_gt / (gt_tr_mag + 1e-6)

            # score
            pred_tr_mag = torch.norm(tr_score, dim=-1, keepdim=True)
            pred_tr_dir = tr_score / (pred_tr_mag + 1e-6)
            tr_dir_loss = torch.mean((pred_tr_dir - gt_tr_dir)**2)
            tr_mag_loss = torch.mean((pred_tr_mag - gt_tr_mag)**2 / tr_score_scale**2)
            tr_loss = 0.5 * tr_dir_loss + 0.5 * tr_mag_loss
            
            # grad
            pred_tr_grad_mag = torch.norm(tr_grad, dim=-1, keepdim=True)
            pred_tr_grad_dir = tr_grad / (pred_tr_grad_mag + 1e-6)
            tr_grad_dir_loss = torch.mean((pred_tr_grad_dir - gt_tr_dir)**2)
            tr_grad_mag_loss = torch.mean((pred_tr_grad_mag - gt_tr_mag)**2 / tr_score_scale**2)
            tr_grad_loss = 0.5 * tr_grad_dir_loss + 0.5 * tr_grad_mag_loss
        else:
            tr_loss = torch.tensor(0.0, device=self.device)
            tr_grad_loss = torch.tensor(0.0, device=self.device)

        # rotation loss
        if self.perturb_rot:
            # gt
            gt_rot_mag = torch.norm(rot_score_gt, dim=-1, keepdim=True)
            gt_rot_dir = rot_score_gt / (gt_rot_mag + 1e-6)

            # score 
            pred_rot_mag = torch.norm(rot_score, dim=-1, keepdim=True)
            pred_rot_dir = rot_score / (pred_rot_mag + 1e-6)
            rot_dir_loss = torch.mean((pred_rot_dir - gt_rot_dir)**2)
            rot_mag_loss = torch.mean((pred_rot_mag - gt_rot_mag)**2 / rot_score_scale**2)
            rot_loss = 0.5 * rot_dir_loss + 0.5 * rot_mag_loss

            # grad
            pred_rot_grad_mag = torch.norm(rot_grad, dim=-1, keepdim=True)
            pred_rot_grad_dir = rot_grad / (pred_rot_grad_mag + 1e-6)
            rot_grad_dir_loss = torch.mean((pred_rot_grad_dir - gt_rot_dir)**2)
            rot_grad_mag_loss = torch.mean((pred_rot_grad_mag - gt_rot_mag)**2 / rot_score_scale**2)
            rot_grad_loss = 0.5 * rot_grad_dir_loss + 0.5 * rot_grad_mag_loss             
        else:
            rot_loss = torch.tensor(0.0, device=self.device)
            rot_grad_loss = torch.tensor(0.0, device=self.device) 
        
        # contrastive loss
        # modified from https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/denoising_diffusion_pytorch_1d.py
        if self.use_contrastive_loss:
            energy_gt = self.net(batch_gt, return_energy=True)
            energy_stack = torch.stack([energy_gt, energy_noised], dim=-1)
            target = torch.zeros([], device=energy_stack.device)
            el_loss = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')
        else: 
            el_loss = torch.tensor(0.0, device=self.device) 

        # interface loss
        bce_logits_loss = nn.BCEWithLogitsLoss()
        if self.use_interface_loss:
            ires_loss = bce_logits_loss(outputs['ires'], batch['ires'])
        else:
            ires_loss = torch.tensor(0.0, device=self.device)
        
        # total losses
        loss = tr_loss + rot_loss + tr_grad_loss + rot_grad_loss + el_loss + ires_loss
        losses = {
            "tr_loss": tr_loss, 
            "rot_loss": rot_loss, 
            "tr_grad_loss": tr_grad_loss, 
            "rot_grad_loss": rot_grad_loss, 
            "el_loss": el_loss, 
            "ires_loss": ires_loss,
            "loss": loss,
        }

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
        
        # debug
        if self.debug:
            optimizer = self.optimizers()

            loss = losses["loss"]
            optimizer.zero_grad()
            self.manual_backward(loss)

            # Print gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad norm: {param.grad.norm()}")
                else:
                    print(f"{name} has NO gradient!")
            
            optimizer.step()  # Update weights
        

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
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="score_model_base.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='dips_train_hetero',
        crop_size=1200,
    )
    index = random.randint(0, len(dataset) - 1)

    subset_indices = [index]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = Score_Model(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, inference_mode=False)
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
