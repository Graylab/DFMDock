import os
import csv
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from scipy.spatial.transform import Rotation 
from utils import residue_constants

#----------------------------------------------------------------------------
# Helper functions

def get_interface_residues(coords, asym_id, interface_threshold=10.0):
    coord_diff = coords[..., None, :, :] - coords[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    mask = diff_chain_mask[..., None].bool()
    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)
    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_idxs

def get_spatial_crop_idx(coords, asym_id, crop_size=256, interface_threshold=10.0):
    interface_residues = get_interface_residues(coords, asym_id, interface_threshold=interface_threshold)

    if not torch.any(interface_residues):
        return get_contiguous_crop_idx(asym_id, crop_size)

    target_res_idx = randint(lower=0, upper=interface_residues.shape[-1] - 1)
    target_res = interface_residues[target_res_idx]

    ca_positions = coords[..., 1, :]
    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    to_target_distances = ca_pairwise_dists[target_res]

    break_tie = (
            torch.arange(
                0, to_target_distances.shape[-1] 
            ).float()
            * 1e-3
    )
    to_target_distances += break_tie
    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values

def get_contiguous_crop_idx(asym_id, crop_size):
    unique_asym_ids, chain_idxs, chain_lens = asym_id.unique(dim=-1,
                                                             return_inverse=True,
                                                             return_counts=True)
    
    shuffle_idx = torch.randperm(chain_lens.shape[-1])
    

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]), dim=0)
    asym_offsets = idx_sorted[cum_sum]

    num_budget = crop_size
    num_remaining = len(chain_idxs)

    crop_idxs = []
    for i, idx in enumerate(shuffle_idx):
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        if i == 0:
            crop_size_max = min(num_budget - 50, chain_len)
            crop_size_min = min(chain_len, 50)
        else:
            crop_size_max = min(num_budget, chain_len)
            crop_size_min = min(chain_len, max(50, num_budget - num_remaining))

        chain_crop_size = randint(lower=crop_size_min,
                                  upper=crop_size_max)

        num_budget -= chain_crop_size

        chain_start = randint(lower=0,
                              upper=chain_len - chain_crop_size)

        asym_offset = asym_offsets[idx]
        crop_idxs.append(
            torch.arange(asym_offset + chain_start, asym_offset + chain_start + chain_crop_size)
        )

    return torch.concat(crop_idxs).sort().values

def randint(lower, upper):
    return int(torch.randint(
        lower,
        upper + 1,
        (1,),
    )[0])


def get_interface_residue_tensors(set1, set2, threshold=8.0):
    n1_len = set1.shape[0]
    n2_len = set2.shape[0]
    
    # Calculate the Euclidean distance between each pair of points from the two sets
    dists = torch.cdist(set1, set2)

    # Find the indices where the distance is less than the threshold
    close_points = dists < threshold

    # Create indicator tensors initialized to 0
    indicator_set1 = torch.zeros((n1_len, 1), dtype=torch.float32)
    indicator_set2 = torch.zeros((n2_len, 1), dtype=torch.float32)

    # Set the corresponding indices to 1 where the points are close
    indicator_set1[torch.any(close_points, dim=1)] = 1.0
    indicator_set2[torch.any(close_points, dim=0)] = 1.0

    return indicator_set1, indicator_set2

def get_sampled_contact_matrix(set1, set2, threshold=8.0, num_samples=None):
    """
    Constructs a contact matrix for two sets of residues with 1 indicating sampled contact pairs.
    
    :param set1: PyTorch tensor of shape [n1, 3] for residues in set 1
    :param set2: PyTorch tensor of shape [n2, 3] for residues in set 2
    :param threshold: Distance threshold to define contact residues
    :param num_samples: Number of contact pairs to sample. If None, use all valid contacts.
    :return: PyTorch tensor of shape [(n1+n2), (n1+n2)] representing the contact matrix with sampled contact pairs
    """
    n1 = set1.size(0)
    n2 = set2.size(0)
    
    # Compute the pairwise distances between set1 and set2
    dists = torch.cdist(set1, set2)
    
    # Find pairs where distances are less than or equal to the threshold
    contact_pairs = (dists <= threshold)
    
    # Get indices of valid contact pairs
    contact_indices = contact_pairs.nonzero(as_tuple=False)
    
    # Initialize the contact matrix with zeros
    contact_matrix = torch.zeros((n1 + n2, n1 + n2))

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

def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return F.one_hot(am, num_classes=len(v_bins)).float()

def relpos(res_id, asym_id, use_chain_relative=True):
    max_relative_idx = 32
    pos = res_id
    asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
    offset = pos[..., None] - pos[..., None, :]

    clipped_offset = torch.clamp(
        offset + max_relative_idx, 0, 2 * max_relative_idx
    )

    rel_feats = []
    if use_chain_relative:
        final_offset = torch.where(
            asym_id_same, 
            clipped_offset,
            (2 * max_relative_idx + 1) * 
            torch.ones_like(clipped_offset)
        )

        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 2
        )
        rel_pos = one_hot(
            final_offset,
            boundaries,
        )

        rel_feats.append(rel_pos)

    else:
        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 1
        )
        rel_pos = one_hot(
            clipped_offset, boundaries,
        )
        rel_feats.append(rel_pos)

    rel_feat = torch.cat(rel_feats, dim=-1).float()

    return rel_feat

def random_rotation(rec_pos, lig_pos):
    rot = torch.from_numpy(Rotation.random().as_matrix()).float()
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    cen = pos[..., 1, :].mean(dim=0)
    pos = (pos - cen) @ rot.T
    rec_pos_out = pos[:rec_pos.size(0)]
    lig_pos_out = pos[rec_pos.size(0):]
    return rec_pos_out, lig_pos_out

#----------------------------------------------------------------------------
# Dataset class

class PPIDataset(Dataset):
    def __init__(
        self, 
        dataset: str,
        training: bool = True,
        use_interface: bool = False,
        use_esm: bool = True,
        crop_size: int = 1500,
    ):
        self.dataset = dataset 
        self.training = training
        self.use_interface = use_interface
        self.use_esm = use_esm
        self.crop_size = crop_size

        parent_dir = Path(__file__).resolve().parents[2]

        if dataset == 'db5_test':
            self.data_dir = f"{parent_dir}/data/db5_test"
            self.data_list = f"{parent_dir}/data/db5_test/test.txt"
            
        with open(self.data_list, 'r') as f:
            lines = f.readlines()
        self.file_list = [line.strip() for line in lines] 

    def __getitem__(self, idx: int):
        # Get info from file_list 
        if self.dataset[:4] == 'dips':
            _id = self.file_list[idx]
            split_string = _id.split('/')
            _id = split_string[0] + '_' + split_string[1].rsplit('.', 1)[0]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))
        else:
            _id = self.file_list[idx]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))

        rec_x = data['receptor'].x
        rec_seq = data['receptor'].seq
        rec_pos = data['receptor'].pos
        lig_x = data['ligand'].x
        lig_seq = data['ligand'].seq
        lig_pos = data['ligand'].pos

        # One-Hot embeddings
        rec_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=rec_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        lig_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=lig_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        # ESM embeddings
        if self.use_esm:
            rec_x = torch.cat([rec_x, rec_onehot], dim=-1)
            lig_x = torch.cat([lig_x, lig_onehot], dim=-1)
        else:
            rec_x = rec_onehot
            lig_x = lig_onehot

        # Shuffle and Crop for training
        if self.training:
            # Shuffle the order of rec and lig
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]
            random.shuffle(vars_list)
            rec_x, rec_seq, rec_pos = vars_list[0]
            lig_x, lig_seq, lig_pos = vars_list[1]

            # Crop to crop_size
            rec_x, lig_x, rec_pos, lig_pos, res_id, asym_id = self.crop_to_size(rec_x, lig_x, rec_pos, lig_pos)  
        else:
            # get res_id and asym_id
            n = rec_x.size(0) + lig_x.size(0)
            res_id = torch.arange(n).long()
            asym_id = torch.zeros(n).long()
            asym_id[rec_x.size(0):] = 1

        # Positional embeddings
        position_matrix = relpos(res_id, asym_id)

        # Random rotation augmentation
        rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

        # Interface residues
        rec_ires, lig_ires = get_interface_residue_tensors(rec_pos[..., 1, :], lig_pos[..., 1, :])
        ires = torch.cat([rec_ires, lig_ires], dim=0) 

        # Output
        output = {
            'id': _id,
            'rec_seq': rec_seq,
            'lig_seq': lig_seq,
            'rec_x': rec_x,
            'lig_x': lig_x,
            'rec_pos': rec_pos,
            'lig_pos': lig_pos,
            'position_matrix': position_matrix,
            'ires': ires,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)

    def crop_to_size(self, rec_x, lig_x, rec_pos, lig_pos):

        n = rec_x.size(0) + lig_x.size(0)
        res_id = torch.arange(n).long()
        asym_id = torch.zeros(n).long()
        asym_id[rec_x.size(0):] = 1

        x = torch.cat([rec_x, lig_x], dim=0)
        pos = torch.cat([rec_pos, lig_pos], dim=0)

        #use_spatial_crop = random.random() < 0.5
        use_spatial_crop = True
        num_res = asym_id.size(0)

        if num_res <= self.crop_size:
            crop_idxs = torch.arange(num_res)
        elif use_spatial_crop:
            crop_idxs = get_spatial_crop_idx(pos, asym_id, crop_size=self.crop_size)
        else:
            crop_idxs = get_contiguous_crop_idx(asym_id, crop_size=self.crop_size)

        res_id = torch.index_select(res_id, 0, crop_idxs)
        asym_id = torch.index_select(asym_id, 0, crop_idxs)
        x = torch.index_select(x, 0, crop_idxs)
        pos = torch.index_select(pos, 0, crop_idxs)

        sep = asym_id.tolist().index(1)
        rec_x = x[:sep]
        lig_x = x[sep:]
        rec_pos = pos[:sep]
        lig_pos = pos[sep:]

        return rec_x, lig_x, rec_pos, lig_pos, res_id, asym_id


if __name__ == '__main__':
    dataset = PPIDataset(dataset='db5_test')
