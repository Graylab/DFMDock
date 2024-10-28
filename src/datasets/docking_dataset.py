import os
import csv
import esm
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from scipy.spatial.transform import Rotation 
from utils import residue_constants
from utils.crop import get_crop_idxs, get_crop

#----------------------------------------------------------------------------
# Helper functions

def random_rotation(rec_pos, lig_pos):
    rot = torch.from_numpy(Rotation.random().as_matrix()).float()
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    cen = pos.mean(dim=(0, 1))
    pos = (pos - cen) @ rot.T
    rec_pos_out = pos[:rec_pos.size(0)]
    lig_pos_out = pos[rec_pos.size(0):]
    return rec_pos_out, lig_pos_out

def get_esm_attn(seq_prim, batch_converter, esm_model, device):
    # Use ESM-1b format.
    # The length of tokens is:
    # L (sequence length) + 2 (start and end tokens)
    seq = [
        ("seq", seq_prim)
    ]
    out = batch_converter(seq)
    with torch.no_grad():
        results = esm_model(out[-1].to(device), repr_layers=[33], return_contacts=True)
        attn = results["attentions"].squeeze(0)[:, :, 1:-1, 1:-1].cpu()
        output = attn.permute(2, 3, 0, 1).flatten(2, 3)

    return output

#----------------------------------------------------------------------------
# Dataset class

class DockingDataset(Dataset):
    def __init__(
        self, 
        dataset: str,
        training: bool = True,
        use_esm: bool = True,
        use_attn: bool = False,
    ):
        self.dataset = dataset 
        self.training = training
        self.use_esm = use_esm
        self.use_attn = use_attn

        if self.use_attn:
            # Load esm
            self.esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/lchu11/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
            self.batch_converter = alphabet.get_batch_converter()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.esm_model = self.esm_model.to(self.device).eval()

        if dataset == 'dips_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/train.txt" 

        elif dataset == 'dips_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/val.txt" 

        elif dataset == 'dips_testing':
            self.data_dir = "/scratch4/jgray21/lchu11/data/dips/pt_clean"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/testing.txt" 

        elif dataset == 'dips_train_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_train_hetero.txt" 

        elif dataset == 'dips_val_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/diffdock-pp/dips_val_hetero.txt" 
            
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
        rec_pos = data['receptor'].pos.float()
        lig_x = data['ligand'].x
        lig_seq = data['ligand'].seq
        lig_pos = data['ligand'].pos.float()

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

        if self.use_attn:
            seq = rec_seq + lig_seq
            attn = get_esm_attn(seq, self.batch_converter, self.esm_model, self.device)

        # Shuffle and Crop for training
        if self.training:
            # Shuffle the order of rec and lig
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]

            if random.random() > 0.5:
                if self.use_attn:
                    n = rec_x.size(0)
                    attn_upper_left = attn[:n, :n]
                    attn_upper_right = attn[:n, n:]
                    attn_lower_left = attn[n:, :n]
                    attn_lower_right = attn[n:, n:]
                    attn_upper = torch.cat([attn_lower_right, attn_lower_left], dim=1)
                    attn_lower = torch.cat([attn_upper_right, attn_upper_left], dim=1)
                    attn = torch.cat([attn_upper, attn_lower], dim=0)

                rec_x, rec_seq, rec_pos = vars_list[1]
                lig_x, lig_seq, lig_pos = vars_list[0]

        # Random rotation augmentation
        rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

        if self.use_attn:
            # Output
            output = {
                'id': _id,
                'rec_seq': rec_seq,
                'lig_seq': lig_seq,
                'rec_x': rec_x,
                'lig_x': lig_x,
                'rec_pos': rec_pos,
                'lig_pos': lig_pos,
                'attn': attn,
            }

        else:
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
        return len(self.file_list)


#----------------------------------------------------------------------------
# DataModule class

class DockingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set: str = 'dips_train',
        val_set: str = 'dips_val',
        batch_size: int = 1,
        use_esm: bool = True,
        use_attn: bool = False,
        **kwargs
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.use_esm = use_esm
        self.use_attn = use_attn
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = DockingDataset(
            dataset=self.train_set,
            use_esm=self.use_esm,
            use_attn=self.use_attn,
        )
        self.data_val = DockingDataset(
            dataset=self.val_set,
            use_esm=self.use_esm,
            use_attn=self.use_attn,
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


if __name__ == '__main__':
    dataset = DockingDataset(dataset='dips_train')
    print(dataset[0])
    """
    dataloader = DataLoader(dataset, batch_size=1, num_workers=6)

    with open('dips_train_size.txt', 'w') as f:
        for batch in dataloader:
            n = batch['rec_x'].size(1) + batch['lig_x'].size(1)
            f.write(str(n) + '\n')

    """

