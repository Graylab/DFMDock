import os
import h5py
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import gzip
import pickle
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation 
from pinder.core.index.utils import get_index
from utils import residue_constants


#----------------------------------------------------------------------------
# Helper functions

def load_dict_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        dict_data = pickle.load(f)
    return dict_data

def random_rotation(rec_pos, lig_pos):
    rot = torch.from_numpy(Rotation.random().as_matrix()).float()
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    cen = pos.mean(dim=(0, 1))
    pos = (pos - cen) @ rot.T
    rec_pos_out = pos[:rec_pos.size(0)]
    lig_pos_out = pos[rec_pos.size(0):]
    return rec_pos_out, lig_pos_out

#----------------------------------------------------------------------------
# Dataset class

class PinderDataset(Dataset):
    def __init__(
        self, 
        data_dir,
        test_split: str = 'pinder_s',
        training: bool = True,
        use_esm: bool = False,
    ):
        self.training = training
        self.use_esm = use_esm

        # Load the dictionary data
        self.data_dir = data_dir
        if training:
            self.data_list = [f.name.split('.')[0] for f in Path(self.data_dir).iterdir()]
        else:
            pindex = get_index()
            self.data_list = list(pindex.query(f'{test_split} == True').id)

        if self.use_esm:
            self.h5f = h5py.File('/scratch16/jgray21/lchu11/data/h5_files/pinder_combined.h5', 'r')

    def __getitem__(self, idx: int):
        data = load_dict_data(os.path.join(self.data_dir, f'{self.data_list[idx]}.pkl.gz'))

        _id = data['id']
        rec_seq = data['rec_seq']
        lig_seq = data['lig_seq']
        rec_pos = torch.from_numpy(data['rec_pos']).float()
        lig_pos = torch.from_numpy(data['lig_pos']).float()

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

        # ESM embeddings
        if self.use_esm:
            group = self.h5f[_id]
            rec_esm = torch.tensor(group['rec_esm'][:])
            lig_esm = torch.tensor(group['lig_esm'][:])

            rec_x = torch.cat([rec_esm, rec_x], dim=-1)
            lig_x = torch.cat([lig_esm, lig_x], dim=-1)

        if self.training:
            # shuffle the order of rec and lig
            vars_list = [(rec_x, rec_pos), (lig_x, lig_pos)]
            random.shuffle(vars_list)
            rec_x, rec_pos = vars_list[0]
            lig_x, lig_pos = vars_list[1]


        # random rotation augmentation
        rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

        # Output
        output = {
            'id': _id,
            'rec_seq': rec_seq,
            'lig_seq': lig_seq,
            'rec_x': rec_x,
            'lig_x': lig_x,
            'rec_pos': rec_pos,
            'lig_pos': lig_pos,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.data_list)


#----------------------------------------------------------------------------
# DataModule class

class PinderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        use_esm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.use_esm = use_esm
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = PinderDataset(
            data_dir='/scratch4/jgray21/lchu11/data/pinder/train',
            use_esm=self.use_esm,
        )
        self.data_val = PinderDataset(
            data_dir='/scratch4/jgray21/lchu11/data/pinder/val',
            use_esm=self.use_esm,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
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
    dataset = PinderDataset(
        data_dir='/scratch4/jgray21/lchu11/data/pinder/train',
        test_split='pinder_s',
        training=True,
        use_esm=True,
    )
    print(dataset[0])
