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
from dfmdock.models.score_model import Score_Model
from dfmdock.utils.so3_diffuser import SO3Diffuser 
from dfmdock.utils.r3_diffuser import R3Diffuser 
from dfmdock.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from dfmdock.utils.dockq import get_dockq
from dfmdock.datasets.ppi_mlsb_dataset import PPIDataset

#----------------------------------------------------------------------------
# Main wrapper for training the model

class Rank_Model(pl.LightningModule):
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

        # load score model
        if self.training:
            self.score_model = Score_Model.load_from_checkpoint(
                experiment.ckpt, 
                map_location=self.device,
            )
            self.score_model.eval()
            self.score_model.to(self.device)
        
        # net
        module = importlib.import_module(f"dfmdock.models.{model.file_name}")
        self.net = module.Rank_Net(model)
    
    def forward(self, batch):
        outputs = self.net(batch, predict=True)
        return outputs

    def get_energy(self, batch):
        energy = self.net(batch, return_energy=True)
        return energy

    def loss_fn(self, batch, eps=1e-5):
        with torch.no_grad():
            # save gt state
            batch_gt = copy.deepcopy(batch)

            # sample pose
            batch["rec_pos"], batch["lig_pos"] = self.Euler_Maruyama_sampler(batch)

            # get dockq
            dockq, i_rmsd, l_rmsd, fnat = get_dockq((batch["rec_pos"], batch["lig_pos"]), (batch_gt["rec_pos"], batch_gt["lig_pos"]))
            print(dockq, i_rmsd, l_rmsd, fnat)

        confidence = self.net(batch)
        bce_logits_loss = nn.BCEWithLogitsLoss()
        loss = bce_logits_loss(confidence, (l_rmsd < 5.0).float())

        losses = {
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
    
    def Euler_Maruyama_sampler(
            self,
            batch,
            batch_size=1, 
            num_steps=40,
        ):

            # initialize time steps
            t = torch.ones(batch_size, device=self.device)
            time_steps = torch.linspace(1., 0., num_steps, device=self.device)
            dt = time_steps[0] - time_steps[1]

            # get initial pose
            rec_pos = batch["rec_pos"] 
            lig_pos = batch["lig_pos"] 

            # randomly initialize coordinates
            rec_pos, lig_pos = self.initialize_ligand_far_from_receptor(rec_pos, lig_pos)
            
            # run reverse sde 
            with torch.no_grad():
                for i, time_step in enumerate((time_steps[:-1])):  
                    # get current time step 
                    is_last = i == time_steps.size(0) - 2
                    t = torch.ones(batch_size, device=self.device) * time_step

                    batch["t"] = t
                    batch["rec_pos"] = rec_pos.detach().clone()
                    batch["lig_pos"] = lig_pos.detach().clone()

                    # get predictions
                    output = self.score_model(batch) 

                    if not is_last:
                        tr_noise_scale = 0.5
                        rot_noise_scale = 0.5
                    else:
                        tr_noise_scale = 0.0
                        rot_noise_scale = 0.0

                    if self.perturb_rot:
                        rot = self.score_model.so3_diffuser.torch_reverse(
                            score_t=output["rot_score"].detach(),
                            t=t.item(),
                            dt=dt,
                            noise_scale=rot_noise_scale,
                        )
                    else:
                        rot = torch.zeros((1, 3), device=self.device)

                    if self.perturb_tr:
                        tr = self.score_model.r3_diffuser.torch_reverse(
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

    def initialize_ligand_far_from_receptor(self, x1, x2, max_iter=1000, min_distance=8.0, step=1.0):
        # get center of mass
        c1 = torch.mean(x1[..., 1, :], dim=0)
        c2 = torch.mean(x2[..., 1, :], dim=0)

        # move to origin
        x1 = x1 - c1
        x2 = x2 - c2

        # init rotation
        if self.perturb_rot:
            # get rotat update
            rot_update = torch.from_numpy(Rotation.random().as_matrix()).float().to(self.device)
            x2 = x2 @ rot_update.T

        # init translation
        if self.perturb_tr:
            # Sample random unit direction
            direction = F.normalize(torch.randn(1, 3, device=self.device))

            # Move ligand until min distance is satisfied
            distance = torch.tensor(step, device=self.device)
            for _ in range(max_iter):
                x2 = x2 + distance * direction
                dists = torch.cdist(x1[..., 1, :], x2[..., 1, :])
                min_dist = dists.min()
                if min_dist >= min_distance:
                    break
                distance += step

        return x1, x2

#----------------------------------------------------------------------------
# Helpers

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd

#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="rank_model.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='dips_train_hetero',
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
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, inference_mode=False)
    trainer.fit(model, dataloader)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
