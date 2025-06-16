import os
import hydra
import csv
import torch
import torch.nn.functional as F
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils import data
from dfmdock.models.confidence_model import Confidence_Model
from dfmdock.utils import residue_constants

def set_seed(seed=42):
    # Set seed for Python's random library
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)

def read_pdb_backbone(pdb_path):
    res_dict = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_id = (line[21], int(line[22:26]))  # (chain_id, res_seq)
                if atom_name in ('N', 'CA', 'C'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_dict.setdefault(res_id, {})[atom_name] = [x, y, z]

    # Sort residues and collect backbone atoms
    coords_list = []
    for res_id in sorted(res_dict.keys(), key=lambda x: (x[0], x[1])):
        atom_dict = res_dict[res_id]
        if all(atom in atom_dict for atom in ('N', 'CA', 'C')):
            coords_list.append([atom_dict['N'], atom_dict['CA'], atom_dict['C']])
    return np.array(coords_list)  # shape: [n_res, 3, 3]

def read_pdb_backbone_per_chain(pdb_path):
    chain_res_dict = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                chain_id = line[21]
                res_seq = int(line[22:26])
                res_id = (chain_id, res_seq)

                if atom_name in ('N', 'CA', 'C'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chain_res_dict.setdefault(chain_id, {}).setdefault(res_id, {})[atom_name] = [x, y, z]

    # Collect coords per chain
    chain_coords = {}
    for chain_id, res_dict in chain_res_dict.items():
        coords_list = []
        for res_id in sorted(res_dict.keys(), key=lambda x: x[1]):
            atom_dict = res_dict[res_id]
            if all(atom in atom_dict for atom in ('N', 'CA', 'C')):
                coords_list.append([atom_dict['N'], atom_dict['CA'], atom_dict['C']])
        if coords_list:
            chain_coords[chain_id] = np.array(coords_list)  # [n_res, 3, 3]

    return chain_coords  # dict: {chain_id: ndarray [n_res, 3, 3]}

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

def run(config):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model = Confidence_Model.load_from_checkpoint(
        config.ckpt, 
        map_location=device,
    )
    model.eval()
    model.to(device)

    # load data
    pdb_dir = config.pdb_dir
    data_dir = config.data_dir

    pdb_list = os.listdir(pdb_dir)
    pdb_id_list = list(set([_.split("_")[0] for _ in pdb_list]))

    result = []
    num_samples = config.num_samples
    for pdb_id in tqdm(pdb_id_list):
        # esm embedding
        data = torch.load(os.path.join(data_dir, pdb_id+'.pt'))
        rec_esm = data['receptor'].x.float()
        rec_seq = data['receptor'].seq
        lig_esm = data['ligand'].x.float()
        lig_seq = data['ligand'].seq

        # one-hot embedding
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

        rec_x = torch.cat([rec_esm, rec_onehot], dim=-1)
        lig_x = torch.cat([lig_esm, lig_onehot], dim=-1)

        # Positional embeddings
        n = rec_x.size(0) + lig_x.size(0)
        res_id = torch.arange(n).long()
        asym_id = torch.zeros(n).long()
        asym_id[rec_x.size(0):] = 1
        position_matrix = relpos(res_id, asym_id)

        batch = {
            "rec_x": rec_x.to(device),
            "lig_x": lig_x.to(device),
            "position_matrix": position_matrix.to(device),
        }

        # run
        for idx in range(num_samples):
            # read pdb
            pdb_path = os.path.join(pdb_dir, f'{pdb_id}_p{idx}.pdb')
            pos = read_pdb_backbone_per_chain(pdb_path)

            # to torch tensor
            rec_pos = pos["A"]
            lig_pos = pos["B"]

            # check matching
            vars_list = [rec_pos, lig_pos]
            if len(rec_x) == len(lig_pos) and len(lig_x) == len(rec_pos):
                rec_pos = vars_list[1]
                lig_pos = vars_list[0]

            # to torch tensor
            rec_pos = torch.from_numpy(rec_pos).float().to(device)
            lig_pos = torch.from_numpy(lig_pos).float().to(device)

            # batch
            batch["t"] = torch.zeros(1, device=device)
            batch["rec_pos"] = rec_pos
            batch["lig_pos"] = lig_pos
            logits = model(batch).detach()

            metrics = {'id': pdb_id, 'index': idx, 'logits': logits.item()}
            result.append(metrics)

    return result
    

#----------------------------------------------------------------------------
# Main
@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs", config_name="inference_confidence") 
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    set_seed()

    results = run(config)

    output_dir = config.out_csv_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set output directory
    output_filename =  os.path.join(output_dir, config.out_csv)

    with open(output_filename, "w", newline="") as csvfile:
        # Write header row to CSV file
        header = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
