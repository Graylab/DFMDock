import os
import csv
import h5py
import gzip
import random
import pickle
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
import biotite.structure.io as strucio
import biotite.structure as struc
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset, Subset, DistributedSampler
from scipy.spatial.transform import Rotation 
from dfmdock.utils import residue_constants
from pinder.core.index.utils import get_index

#----------------------------------------------------------------------------
# Helper functions

def load_dict_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        dict_data = pickle.load(f)
    return dict_data

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

#----------------------------------------------------------------------------
# Dataset class

class PPIDataset(Dataset):
    def __init__(
        self, 
        dataset: str,
        training: bool = True,
        crop_size: int = 1200,
    ):
        self.dataset = dataset 
        self.training = training
        self.crop_size = crop_size

        # Training sets
        if dataset == 'dips_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/train.txt" 

        elif dataset == 'dips_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/val.txt" 

        elif dataset == 'dips_train_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_train_hetero.txt" 

        elif dataset == 'dips_train_hetero_sub':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_train_hetero_sub.txt" 

        elif dataset == 'dips_val_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_val_hetero.txt" 

        elif dataset == 'dips_val_hetero_sub':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_val_hetero_sub.txt" 

        # PINDER
        elif dataset == 'pinder_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/train"
            self.file_list = [f.name.split('.')[0] for f in Path(self.data_dir).iterdir()]

        elif dataset == 'pinder_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/val"
            self.file_list = [f.name.split('.')[0] for f in Path(self.data_dir).iterdir()]

        elif dataset == 'pinder_train_sub':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/train"
            with open("/scratch4/jgray21/lchu11/graylab_repos/DFMDock/src/dfmdock/data/pinder_train/pinder_train_sub.txt", 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        # PPI3D
        elif dataset == 'ppi3d_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/ppi3d/gz_files"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/train_less_than_1200.txt"

        elif dataset == 'ppi3d_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/ppi3d/gz_files"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/val_less_than_1200.txt"

        elif dataset == 'ppi3d_train_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/ppi3d/gz_files"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/train_hetero_less_than_1200.txt"

        elif dataset == 'ppi3d_val_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/ppi3d/gz_files"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/val_hetero_less_than_1200.txt"

        # Testing sets

        # DIPS
        elif dataset == 'dips_test':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_test"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/test.txt" 

        # DB5
        elif dataset == 'db5_test':
            #self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_bound"
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_test_bound"
            self.data_list = "/scratch4/jgray21/lchu11/data/db5/test.txt"
            
        elif dataset == 'db5_all':
            #self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_bound"
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_test_bound"
            self.data_list = "/scratch4/jgray21/lchu11/data/db5/all.txt"

        elif dataset == 'db5_ab_ag':
            #self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_bound"
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_test_bound"
            self.data_list = "/scratch4/jgray21/lchu11/data/db5/ab_ag.txt"

        # AB-AG
        elif dataset == 'ab_ag':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/ab_ag"
            self.data_list = "/scratch4/jgray21/lchu11/data/ab_ag/Yin/af2.3_benchmark/test.txt"
        
        # PINDER
        elif dataset == 'pinder_s':
            pindex = get_index()
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/test" 
            self.file_list = list(pindex.query('pinder_s == True').id)

        elif dataset == 'pinder_af2':
            pindex = get_index()
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/test" 
            self.file_list = list(pindex.query('pinder_af2 == True').id)

        elif dataset == 'pinder_xl':
            pindex = get_index()
            self.data_dir = "/scratch4/jgray21/lchu11/data/pinder/test" 
            self.file_list = list(pindex.query('pinder_xl == True').id)
        
        if self.dataset[:6] != 'pinder':
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
            rec_seq = data['receptor'].seq
            rec_pos = data['receptor'].pos.float()
            lig_seq = data['ligand'].seq
            lig_pos = data['ligand'].pos.float()

        elif self.dataset[:6] == 'pinder':
            data = load_dict_data(os.path.join(self.data_dir, f'{self.file_list[idx]}.pkl.gz'))
            _id = data['id']
            rec_seq = data['rec_seq']
            lig_seq = data['lig_seq']
            rec_pos = torch.from_numpy(data['rec_pos']).float()
            lig_pos = torch.from_numpy(data['lig_pos']).float()
        
        elif self.dataset[:5] == 'ppi3d':
            data_path = os.path.join(self.data_dir, f'{self.file_list[idx]}.pth.gz')
            with gzip.open(data_path, "rb") as f:
                data = torch.load(f)
            _id = data['id']
            rec_seq = data['rec_seq']
            lig_seq = data['lig_seq']
            rec_pos = data['rec_pos'][..., :3, :]
            lig_pos = data['lig_pos'][..., :3, :]
            
        else:
            _id = self.file_list[idx]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))
            rec_seq = data['receptor'].seq
            rec_pos = data['receptor'].pos.float()
            lig_seq = data['ligand'].seq
            lig_pos = data['ligand'].pos.float()
        
        # One-Hot embeddings
        rec_x = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=rec_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        lig_x = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=lig_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        # Shuffle and Crop for training
        if self.training:
            # Shuffle the order of rec and lig
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]
            random.shuffle(vars_list)
            rec_x, rec_seq, rec_pos = vars_list[0]
            lig_x, lig_seq, lig_pos = vars_list[1]

            # Crop to crop_size
            rec_x, lig_x, rec_seq, lig_seq, rec_pos, lig_pos, res_id, asym_id = self.crop_to_size(rec_x, lig_x, rec_seq, lig_seq, rec_pos, lig_pos)  
        else:
            # make the smaller one ligand
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]
            if len(rec_x) < len(lig_x):
                rec_x, rec_seq, rec_pos = vars_list[1]
                lig_x, lig_seq, lig_pos = vars_list[0]

            # get res_id and asym_id
            n = rec_x.size(0) + lig_x.size(0)
            res_id = torch.arange(n).long()
            asym_id = torch.zeros(n).long()
            asym_id[rec_x.size(0):] = 1

        # Positional embeddings
        position_matrix = relpos(res_id, asym_id)

        # Random rotation augmentation
        rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

        # move lig center to origin
        center = lig_pos[..., 1, :].mean(dim=0)
        rec_pos -= center
        lig_pos -= center

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

    def crop_to_size(self, rec_x, lig_x, rec_seq, lig_seq, rec_pos, lig_pos):
        n = rec_x.size(0) + lig_x.size(0)
        res_id = torch.arange(n).long()
        asym_id = torch.zeros(n).long()
        asym_id[rec_x.size(0):] = 1
        
        x = torch.cat([rec_x, lig_x], dim=0)
        pos = torch.cat([rec_pos, lig_pos], dim=0)
        seq = rec_seq + lig_seq

        use_spatial_crop = random.random() < 0.5
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
        seq = ''.join([seq[i] for i in crop_idxs.tolist()])

        sep = asym_id.tolist().index(1)
        rec_x = x[:sep]
        lig_x = x[sep:]
        rec_pos = pos[:sep]
        lig_pos = pos[sep:]
        rec_seq = seq[:sep]
        lig_seq = seq[sep:]

        return rec_x, lig_x, rec_seq, lig_seq, rec_pos, lig_pos, res_id, asym_id
        
#----------------------------------------------------------------------------
# DataModule class

class PPIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: str = "data/",
        val_dataset: str = "data/",
        crop_size: int = 1200,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = PPIDataset(
            dataset=self.train_dataset, 
            crop_size=self.crop_size,
        )
        self.data_val = PPIDataset(
            dataset=self.val_dataset, 
            crop_size=self.crop_size,
        )

    def train_dataloader(self):
        sampler = DistributedSampler(self.data_train) if self.trainer.strategy == "ddp" else None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    dataset = PPIDataset(
        dataset="ppi3d_train",
        crop_size=100,
    )
    print(len(dataset))
    print(dataset[0])
