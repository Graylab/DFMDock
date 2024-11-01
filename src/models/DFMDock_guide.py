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
from datasets.pinder_dataset import PinderDataset
from datasets.docking_dataset import DockingDataset
from models.egnn_net import EGNN_Net
from utils.so3_diffuser import SO3Diffuser 
from utils.r3_diffuser import R3Diffuser 
from utils.geometry import axis_angle_to_matrix
from utils.crop import get_crop_idxs, get_crop, get_position_matrix
from utils.loss import distogram_loss

#----------------------------------------------------------------------------
# Main wrapper for training the model

class DFMDock(pl.LightningModule):
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

        # crop size
        self.crop_size = experiment.crop_size

        # confidence model
        self.use_confidence_loss = experiment.use_confidence_loss

        # dist model
        self.use_dist_loss = experiment.use_dist_loss

        # interface residue model
        self.use_interface_loss = experiment.use_interface_loss

        # energy
        self.grad_energy = experiment.grad_energy
        self.separate_energy_loss = experiment.separate_energy_loss
        self.use_contrastive_loss = experiment.use_contrastive_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # net
        self.net = EGNN_Net(model)
    
    def forward(self, batch):
        # move lig center to origin
        self.move_to_lig_center(batch)

        # predict
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

            # get crop_idxs
            crop_idxs = get_crop_idxs(batch_gt, crop_size=self.crop_size)
            
            # pre crop
            batch = get_crop(batch, crop_idxs)
            batch_gt = get_crop(batch_gt, crop_idxs)

            # noised pose          
            batch["lig_pos"] = self.modify_coords(batch["lig_pos"], rot_update, tr_update)

            # get LRMSD
            l_rmsd = get_rmsd(batch["lig_pos"][..., 1, :], batch_gt["lig_pos"][..., 1, :])

            # move lig center to origin
            self.move_to_lig_center(batch)
            self.move_to_lig_center(batch_gt)

            # post crop
            #batch = get_crop(batch, crop_idxs)
            #batch_gt = get_crop(batch_gt, crop_idxs)

            # get gt contact
            gt_dist = torch.norm((batch_gt["rec_pos"][:, None, 1, :] - batch_gt["lig_pos"][None, :, 1, :]), dim=-1, keepdim=True)
            gt_contact = torch.where(gt_dist < 8.0, 1.0, 0.0)
            if random.random() < 0.1:
                contact_matrix = get_contact_matrix(gt_contact, num_samples=random.randint(0, 3)).unsqueeze(-1)
            else:
                n = gt_contact.size(0) + gt_contact.size(1)
                contact_matrix = torch.zeros((n, n), device=self.device).unsqueeze(-1)
                
            batch["position_matrix"] = torch.cat([batch["position_matrix"], contact_matrix], dim=-1)
            batch_gt["position_matrix"] = torch.cat([batch_gt["position_matrix"], contact_matrix], dim=-1)

            # get gt interface
            gt_ires = get_interface_residue_tensors(batch_gt["rec_pos"][:, 1, :], batch_gt["lig_pos"][:, 1, :])
            if random.random() < 0.1:
                ires = randomly_mask_ones(gt_ires)
            else:
                ires = torch.zeros_like(gt_ires)

            batch["rec_x"] = torch.cat([batch["rec_x"], ires[:gt_contact.size(0)]], dim=-1)
            batch["lig_x"] = torch.cat([batch["lig_x"], ires[gt_contact.size(0):]], dim=-1)
            batch_gt["rec_x"] = torch.cat([batch_gt["rec_x"], ires[:gt_contact.size(0)]], dim=-1)
            batch_gt["lig_x"] = torch.cat([batch_gt["lig_x"], ires[gt_contact.size(0):]], dim=-1)
        
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

        mse_loss_fn = nn.MSELoss()
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
        
        # contrastive loss
        # modified from https://github.com/yilundu/ired_code_release/blob/main/diffusion_lib/denoising_diffusion_pytorch_1d.py
        if self.use_contrastive_loss:
            energy_gt = self.net(batch_gt, return_energy=True)
            energy_stack = torch.stack([energy_gt, energy_noised], dim=-1)
            target = torch.zeros([], device=energy_stack.device)
            el_loss = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')
        else: 
            el_loss = torch.tensor(0.0, device=self.device)

        bce_logits_loss = nn.BCEWithLogitsLoss()
        # distogram loss
        if self.use_dist_loss:
            gt_dist = torch.norm((batch_gt["rec_pos"][:, None, 1, :] - batch_gt["lig_pos"][None, :, 1, :]), dim=-1, keepdim=True)
            dist_loss = distogram_loss(outputs["dist_logits"], gt_dist)
        else:
            dist_loss = torch.tensor(0.0, device=self.device)

        # interface loss
        if self.use_interface_loss:
            gt_ires = get_interface_residue_tensors(batch_gt["rec_pos"][:, 1, :], batch_gt["lig_pos"][:, 1, :])
            ires_loss = bce_logits_loss(outputs["ires_logits"], gt_ires)
        else:
            ires_loss = torch.tensor(0.0, device=self.device)

        # confidence loss
        if self.use_confidence_loss:
            label = (l_rmsd < 5.0).float()
            conf_loss = bce_logits_loss(outputs["confidence_logits"], label)
        else:
            conf_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + 0.1 * (ec_loss + el_loss+ conf_loss + dist_loss + ires_loss)
        losses = {
            "tr_loss": tr_loss, 
            "rot_loss": rot_loss, 
            "ec_loss": ec_loss, 
            "el_loss": el_loss, 
            "dist_loss": dist_loss, 
            "ires_loss": ires_loss,
            "conf_loss": conf_loss,
            "loss": loss,
        }

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        cen = lig_pos.mean(dim=(0, 1))
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - cen) @ rot.T + cen
        lig_pos = lig_pos + tr
        return lig_pos

    def move_to_lig_center(self, batch):
        center = batch["lig_pos"].mean(dim=(0, 1))
        batch["rec_pos"] = batch["rec_pos"] - center
        batch["lig_pos"] = batch["lig_pos"] - center

    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        is_homomer = batch['is_homomer']

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "is_homomer": is_homomer,
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
# helper functions

def get_interface_residue_tensors(set1, set2, threshold=8.0):
    device = set1.device
    n1_len = set1.shape[0]
    n2_len = set2.shape[0]
    
    # Calculate the Euclidean distance between each pair of points from the two sets
    dists = torch.cdist(set1, set2)

    # Find the indices where the distance is less than the threshold
    close_points = dists < threshold

    # Create indicator tensors initialized to 0
    indicator_set1 = torch.zeros((n1_len, 1), device=device)
    indicator_set2 = torch.zeros((n2_len, 1), device=device)

    # Set the corresponding indices to 1 where the points are close
    indicator_set1[torch.any(close_points, dim=1)] = 1.0
    indicator_set2[torch.any(close_points, dim=0)] = 1.0

    return torch.cat([indicator_set1, indicator_set2], dim=0)

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd
    
def randomly_mask_ones(tensor, n=4):
    # Get the indices of all the ones in the tensor
    one_indices = torch.nonzero(tensor == 1).squeeze()

    # Get the total number of ones in the tensor
    num_ones = one_indices.size(0)

    if num_ones == 0:
        # Return the tensor as is if there are no ones
        return tensor
    
    # Randomly shuffle the indices of the ones
    perm = torch.randperm(num_ones)

    # Select a number between 1 and the smaller of 4 or the total number of ones
    num_ones_to_keep = torch.randint(1, min(n, num_ones) + 1, (1,)).item()

    # Keep only the first `num_ones_to_keep` ones
    indices_to_keep = one_indices[perm[:num_ones_to_keep]]

    # Create a mask that zeros out all other ones
    masked_tensor = torch.zeros_like(tensor)
    masked_tensor[indices_to_keep[:, 0], indices_to_keep[:, 1]] = 1

    return masked_tensor

def get_contact_matrix(contact_pairs, num_samples=None):
    """
    Constructs a contact matrix for two sets of residues with 1 indicating sampled contact pairs.
    
    :param num_samples: Number of contact pairs to sample. If None, use all valid contacts.
    :return: PyTorch tensor of shape [(n1+n2), (n1+n2)] representing the contact matrix with sampled contact pairs
    """
    device = contact_pairs.device
    n1 = contact_pairs.size(0)
    n2 = contact_pairs.size(1)
    
    # Get indices of valid contact pairs
    contact_indices = contact_pairs.nonzero(as_tuple=False)
    
    # Initialize the contact matrix with zeros
    contact_matrix = torch.zeros((n1 + n2, n1 + n2), device=device)

    # Determine the number of samples
    if num_samples is None or num_samples > contact_indices.size(0):
        num_samples = contact_indices.size(0)
    
    if num_samples > 0:
        # Sample contact indices uniformly
        sampled_indices = contact_indices[torch.randint(0, contact_indices.size(0), (num_samples,))]
        
        # Fill in the contact matrix for the sampled contacts
        contact_matrix[sampled_indices[:, 0], sampled_indices[:, 1] + n1] = 1.0
        contact_matrix[sampled_indices[:, 1] + n1, sampled_indices[:, 0]] = 1.0
    
    return contact_matrix
    

#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs/model", config_name="DFMDock_guide.yaml")
def main(conf: DictConfig):
    dataset = PinderDataset(
        data_dir='/scratch4/jgray21/lchu11/data/pinder/train',
        training=True,
        use_esm=True,
    )
    
    #dataset = DockingDataset(
    #    dataset='dips_train',
    #    training=True,
    #    use_esm=True,
    #)

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = DFMDock(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
