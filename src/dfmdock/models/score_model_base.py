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
from dfmdock.utils.dockq import get_DockQ
from dfmdock.datasets.pp_docking_dataset import PPDockingDataset
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
        self.separate_energy_loss = experiment.separate_energy_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        # contrastive
        self.use_contrastive_loss = experiment.use_contrastive_loss

        # interface 
        self.use_interface_loss = experiment.use_interface_loss

        # contact
        self.use_contact_loss = experiment.use_contact_loss

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

            # get dockq
            dockq = get_DockQ((batch["rec_pos"], batch["lig_pos"]), (batch_gt["rec_pos"], batch_gt["lig_pos"]))

        # predict score based on the current state
        if self.grad_energy:
            outputs = self.net(batch)

            # grab some outputs
            tr_score = outputs["tr_score"]
            rot_score = outputs["rot_score"]
            force = outputs["force"]
            torque = outputs["torque"]
            grad_force = outputs["grad_force"]
            grad_torque = outputs["grad_torque"]
            energy_noised = outputs["energy"]

            #print(energy_noised)
            #print(force.norm())
            #print(torque.norm())
            #print(grad_force.norm())
            #print(grad_torque.norm())

            # energy conservation loss
            if self.separate_energy_loss:
                f_mag = torch.norm(f, dim=-1, keepdim=True)
                f_dir = f / (f_mag + 1e-6)

                dedx_mag = torch.norm(dedx, dim=-1, keepdim=True)
                dedx_dir = dedx / (dedx_mag + 1e-6)

                ec_dir_loss = torch.mean((f_dir - dedx_dir)**2)
                ec_mag_loss = torch.mean((f_mag - dedx_mag)**2)
                ec_loss = 0.5 * (ec_dir_loss + ec_mag_loss)
                #ec_loss = ec_dir_loss + ec_mag_loss
                
            else:
                ec_tr_loss = torch.mean((grad_force - force)**2)
                ec_rot_loss = torch.mean((grad_torque - torque)**2)
                ec_loss = ec_tr_loss + ec_rot_loss
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
                gt_tr_mag = torch.norm(tr_score_gt, dim=-1, keepdim=True)
                gt_tr_dir = tr_score_gt / (gt_tr_mag + 1e-6)

                pred_tr_mag = torch.norm(tr_score, dim=-1, keepdim=True)
                pred_tr_dir = tr_score / (pred_tr_mag + 1e-6)

                tr_dir_loss = torch.mean((pred_tr_dir - gt_tr_dir)**2)
                tr_mag_loss = torch.mean((pred_tr_mag - gt_tr_mag)**2 / tr_score_scale**2)
                tr_loss = 0.5 * (tr_dir_loss + tr_mag_loss)
                #tr_loss = tr_dir_loss + 0.1 * tr_mag_loss

            else:
                tr_loss = torch.mean((tr_score - tr_score_gt)**2 / tr_score_scale**2)
        else:
            tr_loss = torch.tensor(0.0, device=self.device)

        # rotation loss
        if self.perturb_rot:
            if self.separate_rot_loss:
                gt_rot_mag = torch.norm(rot_score_gt, dim=-1, keepdim=True)
                gt_rot_dir = rot_score_gt / (gt_rot_mag + 1e-6)

                pred_rot_mag = torch.norm(rot_score, dim=-1, keepdim=True)
                pred_rot_dir = rot_score / (pred_rot_mag + 1e-6)

                rot_dir_loss = torch.mean((pred_rot_dir - gt_rot_dir)**2)
                rot_mag_loss = torch.mean((pred_rot_mag - gt_rot_mag)**2 / rot_score_scale**2)
                rot_loss = 0.5 * (rot_dir_loss + rot_mag_loss)
                #rot_loss = rot_dir_loss + 0.1 * rot_mag_loss

            else:
                rot_loss = torch.mean((rot_score - rot_score_gt)**2 / rot_score_scale**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)
        
        # contrastive loss
        # modified from https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/denoising_diffusion_pytorch_1d.py
        if self.use_contrastive_loss:
            energy_gt = self.net(batch_gt, return_energy=True)
            energy_stack = torch.stack([energy_gt, energy_noised], dim=-1)
            target = torch.zeros([], device=energy_stack.device)
            el_loss = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')
            # weight by dockq
            el_loss = el_loss * (1.0 - dockq)
        else: 
            el_loss = torch.tensor(0.0, device=self.device) 

        # interface loss
        bce_logits_loss = nn.BCEWithLogitsLoss()
        if self.use_interface_loss:
            ires_loss = bce_logits_loss(outputs['ires'], batch['ires'])
        else:
            ires_loss = torch.tensor(0.0, device=self.device)
        
        # contact loss
        if self.use_contact_loss:
            gt_dist = torch.norm(batch_gt["rec_pos"][:, None, 1, :] - batch_gt["lig_pos"][None, :, 1, :], dim=-1)
            cut_off = 10.0
            gt_contact = (gt_dist < cut_off).float()
            contact_loss = bce_logits_loss(outputs['contact'], gt_contact.unsqueeze(-1))
        else:
            contact_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + 0.1 * (ec_loss + el_loss + ires_loss + contact_loss)
        losses = {
            "tr_loss": tr_loss, 
            "rot_loss": rot_loss, 
            "ec_loss": ec_loss, 
            "el_loss": el_loss, 
            "ires_loss": ires_loss,
            "contact_loss": contact_loss,
            "loss": loss,
        }

        if (self.perturb_tr and self.separate_tr_loss):
            losses["tr_dir_loss"] = tr_dir_loss
            losses["tr_mag_loss"] = tr_mag_loss
        if (self.perturb_rot and self.separate_rot_loss):
            losses["rot_dir_loss"] = rot_dir_loss
            losses["rot_mag_loss"] = rot_mag_loss

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
        #dockq = self.inference(batch)
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
    
    def compute_dockq(self, batch):
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

        _rec_pos, _lig_pos = self.Euler_Maruyama_sampler(batch)
        dockq = get_DockQ((_rec_pos, _lig_pos), (rec_pos, lig_pos))
        return dockq

    def Euler_Maruyama_sampler(
        self,
        batch,
        batch_size=1, 
        eps=1e-3,
        num_steps=40,
    ):
        # initialize time steps
        t = torch.ones(batch_size, device=self.device)
        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        dt = time_steps[0] - time_steps[1]

        # get initial pose
        rec_pos = batch["rec_pos"] 
        lig_pos = batch["lig_pos"] 

        # randomly initialize coordinates
        rec_pos, lig_pos, rot_update, tr_update = self.randomize_pose(rec_pos, lig_pos)
        
        # run reverse sde 
        with torch.no_grad():
            for i, time_step in enumerate(time_steps):  
                # get current time step 
                is_last = i == time_steps.size(0) - 1   
                t = torch.ones(batch_size, device=self.device) * time_step

                batch["t"] = t
                batch["rec_pos"] = rec_pos.detach().clone()
                batch["lig_pos"] = lig_pos.detach().clone()

                # get predictions
                output = self.net(batch, predict=True) 

                if not is_last:
                    tr_noise_scale = 0.5
                    rot_noise_scale = 0.5
                else:
                    tr_noise_scale = 0.0
                    rot_noise_scale = 0.0

                if self.perturb_rot:
                    rot = self.so3_diffuser.torch_reverse(
                        score_t=output["rot_score"].detach(),
                        t=t.item(),
                        dt=dt,
                        noise_scale=rot_noise_scale,
                    )
                else:
                    rot = torch.zeros((1, 3), device=self.device)

                if self.perturb_tr:
                    tr = self.r3_diffuser.torch_reverse(
                        score_t=output["tr_score"].detach(),
                        t=t.item(),
                        dt=dt,
                        noise_scale=tr_noise_scale,
                    )
                else:
                    tr = torch.zeros((1, 3), device=self.device)

                lig_pos = self.modify_coords(lig_pos, rot, tr)
                
        return rec_pos, lig_pos

    def randomize_pose(self, x1, x2):
        # get center of mass
        c1 = torch.mean(x1[..., 1, :], dim=0)
        c2 = torch.mean(x2[..., 1, :], dim=0)

        # get rotat update
        rot_update = torch.from_numpy(Rotation.random().as_matrix()).float().to(self.device)

        # get trans update
        tr_update = torch.normal(0.0, 30.0, size=(1, 3), device=self.device)
        #tr_update = torch.from_numpy(sample_sphere(radius=50.0)).float().to(self.device)

        # move to origin
        x1 = x1 - c1
        x2 = x2 - c2

        # init rotation
        if self.perturb_rot:
            x2 = x2 @ rot_update.T

        # init translation
        if self.perturb_tr:
            x2 = x2 + tr_update 

        # convert to axis angle
        rot_update = matrix_to_axis_angle(rot_update.unsqueeze(0))

        return x1, x2, rot_update, tr_update

#----------------------------------------------------------------------------
# Helpers

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd

#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="score_model_base.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='pinder_train',
        crop_size=500,
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
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, inference_mode=False)
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
