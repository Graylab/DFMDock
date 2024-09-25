import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from utils.residue_constants import restype_3to1

file_path = '1A2K_r_b.pdb'  # Replace with your PDB file path

# Load the structure from the PDB file
structure = strucio.load_structure(file_path)

# Get the atomic coordinates as a NumPy array
coordinates = structure.coord  # Coordinates are stored in the 'coord' attribute

# Get the residue names (three-letter codes)
numbering, resn = struc.get_residues(structure)
seq_list = [restype_3to1.get(three, "X") for three in resn]
seq = ''.join(seq_list)


def get_bb_coords(structure):
    # Filter atoms by names 'N', 'CA', 'C'
    n_atoms = structure[structure.atom_name == "N"]
    ca_atoms = structure[structure.atom_name == "CA"]
    c_atoms = structure[structure.atom_name == "C"]
    print(len(n_atoms))
    print(len(ca_atoms))
    print(len(c_atoms))

    # Ensure that the number of N, CA, and C atoms are the same and correspond to residues
    n_res = min(len(n_atoms), len(ca_atoms), len(c_atoms))

    # Create an array of shape (n_res, 3, 3) to hold [N, CA, C] for each residue
    coords = np.zeros((n_res, 3, 3))

    # Assign coordinates for N, CA, and C atoms in the correct order
    coords[:, 0, :] = n_atoms.coord[:n_res]  # N
    coords[:, 1, :] = ca_atoms.coord[:n_res]  # CA
    coords[:, 2, :] = c_atoms.coord[:n_res]  # C
    
    return coords

bb_coords = get_bb_coords(structure)
