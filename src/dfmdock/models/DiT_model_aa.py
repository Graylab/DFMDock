import os
import esm
import copy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
from scipy.spatial.transform import Rotation 
from torch.utils import data
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from dataclasses import dataclass
from models.DiT_net_aa import Score_Net
from datasets.ppi_mlsb_dataset import PPIDataset
from utils.metrics import find_rigid_alignment
from utils.loss import violation_loss
from utils.pdb import save_PDB, place_fourth_atom 

#----------------------------------------------------------------------------
# Data class for pose

@dataclass
class pose():
    rec_seq: str
    lig_seq: str
    rec_pos: torch.FloatTensor
    lig_pos: torch.FloatTensor

#----------------------------------------------------------------------------
# Main wrapper for training the model

class Score_Model(pl.LightningModule):
    def __init__(
        self,
        model,
        experiment,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = experiment.lr
        self.weight_decay = experiment.weight_decay

        # energy
        self.grad_energy = experiment.grad_energy

        # violation
        self.use_violation_loss = experiment.use_violation_loss
        
        # net
        self.net = Score_Net(model)

        # edm
        P_mean = -1.2
        P_std = 1.2
        sigma_data = 0.5
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
    
    def forward(self, batch):
        outputs = self.net(batch, predict=True)

        return outputs["pos"]

    def loss_fn(self, batch):
        # edm   
        rnd_normal = torch.randn(1, device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        batch["sigma"] = sigma

        # noise
        batch["pos"] = centre_random_augmentation(batch["pos"])
        batch_gt = copy.deepcopy(batch)
        n = torch.randn_like(batch["pos"]) * sigma
        batch["pos"] = batch["pos"] + n
        
        # predict score based on the current state
        if self.grad_energy:
            outputs = self.net(batch)

            # grab some outputs
            f = outputs["f"]
            dedx = outputs["dedx"]

            # energy conservation loss
            ec_loss = torch.mean((dedx - f)**2)
        else:
            outputs = self.net(batch, predict=True)

            # energy conservation loss
            ec_loss = torch.tensor(0.0, device=self.device)

        # denoising score matching
        batch_gt["pos"] = self.align_coords(batch_gt["pos"], outputs["pos"])
        edm_loss = weight * (outputs["pos"] - batch_gt["pos"]) ** 2
        edm_loss = edm_loss.mean() / 3

        # violation loss
        if self.use_violation_loss:
            v_loss = weight * violation_loss(outputs["pos"].view(-1, 3, 3), batch["rec_x"].size(0))
        else:
            v_loss = torch.tensor(0.0, device=self.device)

        # COM loss
        """
        n = batch_gt["rec_x"].size(0)
        ci_pred = outputs["pos"].view(-1, 3, 3)[:n, 1, :].mean(dim=0)
        cj_pred = outputs["pos"].view(-1, 3, 3)[n:, 1, :].mean(dim=0)
        ci_gt = batch_gt["pos"].view(-1, 3, 3)[:n, 1, :].mean(dim=0)
        cj_gt = batch_gt["pos"].view(-1, 3, 3)[n:, 1, :].mean(dim=0)
        diff = ((ci_pred - cj_pred).norm() - (ci_gt - cj_gt).norm() + 4.0) / 20.0
        com_loss = 0.5 * diff.clamp(max=0.0) ** 2
        """

        """
        p = batch_gt["pos"].view(-1, 3, 3)
        rec_pos = p[:batch_gt["rec_x"].size(0)]
        lig_pos = p[batch_gt["rec_x"].size(0):]
        pred = pose(
            rec_seq=batch_gt["rec_seq"],
            lig_seq=batch_gt["lig_seq"],
            rec_pos=rec_pos,
            lig_pos=lig_pos,
        )
        save_pdb(pred)
        """

        # total losses
        loss = edm_loss + ec_loss + v_loss 
        losses = {"edm_loss": edm_loss,"v_loss": v_loss, "ec_loss": ec_loss, "loss": loss}

        return losses

    def align_coords(self, gt, pred):
        R, t = find_rigid_alignment(gt, pred)
        return ((R.mm(gt.T)).T + t).detach()

    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_seq = batch['rec_seq'][0]
        lig_seq = batch['lig_seq'][0]
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        position_matrix = batch['position_matrix'].squeeze(0)

        x = torch.cat([rec_x, lig_x], dim=0)
        pos = torch.cat([rec_pos, lig_pos], dim=0).view(-1, 3)

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_seq": rec_seq,
            "lig_seq": lig_seq,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "pos": pos,
            "position_matrix": position_matrix,
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

def centre_random_augmentation(pos):
    cen = pos.mean(dim=0)
    rot = torch.from_numpy(Rotation.random().as_matrix()).float().to(pos.device)
    tr = torch.randn(1, 3).to(pos.device)
    pos = (pos - cen) @ rot.T + tr
    return pos


def save_pdb(pred):

    # set output directory
    out_pdb =  'test.pdb'

    # output trajectory
    if os.path.exists(out_pdb):
        os.remove(out_pdb)
        
    seq1 = pred.rec_seq
    seq2 = pred.lig_seq
        
    coords = torch.cat([pred.rec_pos, pred.lig_pos], dim=0)
    coords = get_full_coords(coords)

    # get total len
    total_len = coords.size(0)

    # check seq len
    assert len(seq1) + len(seq2) == total_len

    # get pdb
    save_PDB(out_pdb=out_pdb, coords=coords, seq=seq1+seq2, delim=len(seq1)-1)

def get_full_coords(coords):
    #get full coords
    N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
    # Infer CB coordinates.
    b = CA - N
    c = C - CA
    a = b.cross(c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    
    O = place_fourth_atom(torch.roll(N, -1, 0),
                                    CA, C,
                                    torch.tensor(1.231),
                                    torch.tensor(2.108),
                                    torch.tensor(-3.142))
    full_coords = torch.stack(
        [N, CA, C, O, CB], dim=1)
    
    return full_coords
#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="DiT_model_aa.yaml")
def main(conf: DictConfig):
    dataset = PPIDataset(
        dataset='dips_single',
        crop_size=500,
    )

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = Score_Model(
        model=conf.model, 
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
