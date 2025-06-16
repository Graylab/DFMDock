import torch
import numpy as np
from pathlib import Path
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure

def load_input(target_dir, pdb_id):
    # Load the structure
    structure = np.load(f"{target_dir}/structures/{pdb_id}.npz")

    # In order to add cyclic_period to chains if it does not exist
    # Extract the chains array
    chains = structure["chains"]
    # Check if the field exists
    if "cyclic_period" not in chains.dtype.names:
        # Create a new dtype with the additional field
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        # Create a new array with the new dtype
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        # Copy over existing fields
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        # Set the new field to 0
        new_chains["cyclic_period"] = 0
        # Replace old chains array with new one
        chains = new_chains

    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    return structure


class PPIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_dir,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.target_dir = target_dir
        self.filenames = [f.stem for f in Path(f"{target_dir}/structures").iterdir() if f.is_file()]

    def __getitem__(self, idx: int):
        pdb_id = self.filenames[idx]
        structure = load_input(self.target_dir, pdb_id)
        print(pdb_id)
        chains = [c for c in structure.chains if c["mol_type"] == 0]
        #print(chains)
        chain_sequences = {}
        chain_backbones = {}
        for chain in chains:
            chain_tag = chain["name"]
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            residues = structure.residues[res_start:res_end]

            seq = []
            backbone_coords = []
            for residue in residues:
                resname = residue["res_type"] 
                
                atom_start = residue["atom_idx"]
                atom_end = residue["atom_idx"] + residue["atom_num"]
                atoms = structure.atoms[atom_start:atom_end]

                seq.append(resname)
                coords = [atoms[i]["coords"] for i in range(3)]
                backbone_coords.append(coords)

            chain_sequences[chain_tag] = seq
            chain_backbones[chain_tag] = np.array(backbone_coords)
        
        print(chain_sequences)
        print(chain_backbones)
 
    def __len__(self) -> int:
        return len(self.filenames)

if __name__ == '__main__':
    dataset = PPIDataset(target_dir="/scratch16/jgray21/lchu11/data/rcsb_processed_targets")
    print(len(dataset))
    dataset[1]
