import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import csv
import esm
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm
from scipy.spatial.transform import Rotation 
from utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from utils.residue_constants import restype_3to1, sequence_to_onehot, restype_order_with_x
from utils.pdb import save_PDB, place_fourth_atom 
from utils.metrics import compute_metrics
from models.DFMDock import DFMDock


#----------------------------------------------------------------------------
# output functions

def save_pdb(output, out_pdb):
    if os.path.exists(out_pdb):
        os.remove(out_pdb)
        
    seq1 = output["rec_seq"]
    seq2 = output["lig_seq"]
        
    coords = torch.cat([output["rec_pos"], output["lig_pos"]], dim=0)
    coords = get_full_coords(coords)

    # get total len
    total_len = coords.size(0)

    # check seq len
    assert len(seq1) + len(seq2) == total_len

    # get pdb
    save_PDB(out_pdb=out_pdb, coords=coords, seq=seq1+seq2, delim=len(seq1)-1)

def combine_atom_arrays(atom_array_1, atom_array_2):
    """
    Combine two AtomArray objects and save as a single PDB file.
    
    Parameters:
    - atom_array_1: First AtomArray object.
    - atom_array_2: Second AtomArray object.
    - output_file: Name of the output PDB file.
    """
    # Check if both AtomArray objects have the same number of atoms
    if atom_array_1.coord.shape[1] != 3 or atom_array_2.coord.shape[1] != 3:
        raise ValueError("Both AtomArray objects must have 3D coordinates (Nx3 arrays)")

    # Concatenate coordinates and attributes manually
    combined_coords = np.concatenate([atom_array_1.coord, atom_array_2.coord])

    # Create a new AtomArray with the combined coordinates
    combined_structure = struc.AtomArray(len(combined_coords))
    combined_structure.coord = combined_coords

    # Transfer additional attributes from the original AtomArray objects
    combined_structure.element = np.concatenate([atom_array_1.element, atom_array_2.element])

    # Handle optional attributes like atom names, residue names, residue IDs, chain IDs, etc.
    combined_structure.atom_name = np.concatenate([atom_array_1.atom_name, atom_array_2.atom_name])
    combined_structure.res_name = np.concatenate([atom_array_1.res_name, atom_array_2.res_name])
    combined_structure.res_id = np.concatenate([atom_array_1.res_id, atom_array_2.res_id])
    combined_structure.chain_id = np.concatenate([atom_array_1.chain_id, atom_array_2.chain_id])

    return combined_structure


#----------------------------------------------------------------------------
# pre-process functions

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

def get_info_from_pdb(pdb_path):
    # Load the structure from the PDB file
    structure = strucio.load_structure(pdb_path)

    # Filter for only ATOM lines
    structure = structure[~structure.hetero]

    # Get the atomic coordinates as a NumPy array
    aa_coords = structure.coord  # all-atom coordinates

    # Get the residue names (three-letter codes)
    numbering, resn = struc.get_residues(structure)
    seq_list = [restype_3to1.get(three, "X") for three in resn]
    seq = ''.join(seq_list)

    # Filter atoms by names 'N', 'CA', 'C'
    n_atoms = structure[structure.atom_name == "N"]
    ca_atoms = structure[structure.atom_name == "CA"]
    c_atoms = structure[structure.atom_name == "C"]

    # Ensure that the number of N, CA, and C atoms are the same and correspond to residues
    n_res = min(len(n_atoms), len(ca_atoms), len(c_atoms))

    # Create an array of shape (n_res, 3, 3) to hold [N, CA, C] for each residue
    bb_coords = np.zeros((n_res, 3, 3)) # back-bone coords

    # Assign coordinates for N, CA, and C atoms in the correct order
    bb_coords[:, 0, :] = n_atoms.coord[:n_res]  # N
    bb_coords[:, 1, :] = ca_atoms.coord[:n_res]  # CA
    bb_coords[:, 2, :] = c_atoms.coord[:n_res]  # C
    
    return {"structure": structure, "seq":seq, "aa_coords":aa_coords, "bb_coords":bb_coords}

def get_batch_from_inputs(inputs, batch_converter, esm_model, device):
    # One-Hot embeddings
    rec_onehot = torch.from_numpy(sequence_to_onehot(
        sequence=inputs['receptor']['seq'],
        mapping=restype_order_with_x,
        map_unknown_to_x=True,
    )).float()

    lig_onehot = torch.from_numpy(sequence_to_onehot(
        sequence=inputs['ligand']['seq'],
        mapping=restype_order_with_x,
        map_unknown_to_x=True,
    )).float()
    
    # ESM embeddings
    rec_esm = get_esm_rep(inputs['receptor']['seq'], batch_converter, esm_model, device)
    lig_esm = get_esm_rep(inputs['ligand']['seq'], batch_converter, esm_model, device)

    # node embeddings
    rec_x = torch.cat([rec_esm, rec_onehot], dim=-1).to(device)
    lig_x = torch.cat([lig_esm, lig_onehot], dim=-1).to(device)

    # coords
    rec_pos = torch.from_numpy(inputs['receptor']['bb_coords']).float().to(device)
    lig_pos = torch.from_numpy(inputs['ligand']['bb_coords']).float().to(device)

    batch = {
        'rec_x': rec_x,
        'lig_x': lig_x,
        'rec_pos': rec_pos,
        'lig_pos': lig_pos,
    }

    return batch

def get_esm_rep(seq_prim, batch_converter, esm_model, device):
    # Use ESM-1b format.
    # The length of tokens is:
    # L (sequence length) + 2 (start and end tokens)
    seq = [
        ("seq", seq_prim)
    ]
    out = batch_converter(seq)
    with torch.no_grad():
        results = esm_model(out[-1].to(device), repr_layers = [33])
        rep = results["representations"][33].cpu()
    
    return rep[0, 1:-1, :]

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

def compute_tm(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(logits.size(0) + logits.size(1), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = F.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    max_sum = max(torch.mean(predicted_tm_term, dim=0).max(), torch.mean(predicted_tm_term, dim=1).max())

    return max_sum


#----------------------------------------------------------------------------
# coords functions

def rot_compose(r1, r2):
    R1 = axis_angle_to_matrix(r1)
    R2 = axis_angle_to_matrix(r2)
    R = torch.einsum('b i j, b j k -> b i k', R2, R1)
    r = matrix_to_axis_angle(R)
    return r
     
def randomize_pose(x1, x2):
    device=x1.device

    # get center of mass
    c1 = torch.mean(x1, dim=(0, 1))
    c2 = torch.mean(x2, dim=(0, 1))

    # get rotat update
    rot_update = torch.from_numpy(Rotation.random().as_matrix()).float().to(device)

    # get trans update
    tr_update = torch.normal(0.0, 30.0, size=(1, 3), device=device) - c2 + c1

    # init rotation
    x2 = (x2 - c2) @ rot_update.T + c2

    # init translation
    x2 = x2 + tr_update 

    # convert rot_update to axis angle
    rot_update = matrix_to_axis_angle(rot_update.unsqueeze(0))

    return x2, tr_update, rot_update

def modify_coords(x, rot, tr):
    center = torch.mean(x, dim=(0, 1))
    rot = axis_angle_to_matrix(rot).squeeze()
    
    # update rotation
    x = (x - center) @ rot.T + center 
    
    # update translation
    x = x + tr
    
    return x

def modify_aa_coords(x, rot, tr):
    center = x.mean(axis=0)
    rot = axis_angle_to_matrix(rot).squeeze().cpu().numpy()
    
    # update rotation
    x = (x - center) @ rot.T + center 
    
    # update translation
    x = x + tr.cpu().numpy()
    
    return x

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

def get_clash_force(rec_pos, lig_pos):
    rec_pos = rec_pos.view(-1, 3)
    lig_pos = lig_pos.view(-1, 3)

    with torch.set_grad_enabled(True):
        lig_pos.requires_grad_(True)
        # get distance matrix 
        D = torch.norm((rec_pos[:, None, :] - lig_pos[None, :, :]), dim=-1)

        def rep_fn(x):
            x0, p, w_rep = 4, 1.5, 5
            rep = torch.where(x < x0, (torch.abs(x0 - x) ** p) / (p * x * (p - 1)), torch.tensor(0.0, device=x.device)) 
            return -w_rep * torch.sum(rep)

        rep = rep_fn(D)

        force = torch.autograd.grad(rep, lig_pos, retain_graph=False)[0]

    return force.mean(dim=0).detach()


#----------------------------------------------------------------------------
# Sampler

def Euler_Maruyama_sampler(
    model,
    batch,
    num_steps=40,
    device='cpu',
    batch_size=1, 
    eps=1e-3,
    use_clash_force=False,
    noise_annealing=False,
    tr_noise_scale=0.5,
    rot_noise_scale=0.5,
):

    # initialize time steps
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]

    # get initial pose
    rec_pos = batch["rec_pos"] 
    lig_pos = batch["lig_pos"] 


    # randomly initialize coordinates
    lig_pos, tr_update, rot_update = randomize_pose(rec_pos, lig_pos)
    
    # run reverse sde 
    with torch.no_grad():
        for i, time_step in enumerate(tqdm((time_steps))):  
            # get current time step 
            is_last = i == time_steps.size(0) - 1   
            t = torch.ones(batch_size, device=device) * time_step

            batch["t"] = t
            batch["rec_pos"] = rec_pos.clone().detach()
            batch["lig_pos"] = lig_pos.clone().detach()

            # get predictions
            output = model(batch) 

            if noise_annealing:
                tr_noise_scale = time_step
                rot_noise_scale = time_step
            else:
                if not is_last:
                    tr_noise_scale = tr_noise_scale
                    rot_noise_scale = rot_noise_scale
                else:
                    tr_noise_scale = 0.0
                    rot_noise_scale = 0.0

            rot = model.so3_diffuser.torch_reverse(
                score_t=output["rot_score"].detach(),
                t=t.item(),
                dt=dt,
                noise_scale=rot_noise_scale,
            )

            tr = model.r3_diffuser.torch_reverse(
                score_t=output["tr_score"].detach(),
                t=t.item(),
                dt=dt,
                noise_scale=tr_noise_scale,
            )

            lig_pos = modify_coords(lig_pos, rot, tr)

            tr_update = tr_update + tr
            rot_update = rot_compose(rot_update, rot)

            if use_clash_force:
                clash_force = get_clash_force(rec_pos.detach().clone(), lig_pos.detach().clone())
                lig_pos = lig_pos + clash_force
                tr_update = tr_update + clash_force

            if is_last:
                batch["rec_pos"] = rec_pos.clone().detach()
                batch["lig_pos"] = lig_pos.clone().detach()
                output = model(batch) 

    return rec_pos, lig_pos, rot_update, tr_update, output["energy"], output["num_clashes"], output["tm_logits"]


#----------------------------------------------------------------------------
# run function

def run(args, model, inputs, batch, device):
    metrics_list = []
    for i in range(args.num_samples):
        rec_pos, lig_pos, rot_update, tr_update, energy, num_clashes, tm_logits = Euler_Maruyama_sampler(
            model=model, 
            batch=batch, 
            num_steps=args.num_steps,
            device=device,
            use_clash_force=args.use_clash_force,
            noise_annealing=args.noise_annealing,
            tr_noise_scale=args.tr_noise_scale,
            rot_noise_scale=args.rot_noise_scale,
        )
        
        id = inputs["id"]

        # get metrics
        metrics = {'id': id, 'index': str(i)}
        pred = (rec_pos.detach().cpu(), lig_pos.detach().cpu())
        native = (torch.from_numpy(inputs['receptor']['bb_coords']).float(), torch.from_numpy(inputs['ligand']['bb_coords']).float())
        metrics.update(compute_metrics(pred, native))
        metrics.update({'energy': energy.item()})
        metrics.update({'num_clashes': num_clashes.item()})
        metrics.update({'iptm': compute_tm(tm_logits).item()})
        metrics_list.append(metrics)

        # get aa structure
        lig_aa_coords = modify_aa_coords(inputs["ligand"]["aa_coords"], rot_update, tr_update)
        rec_structure = inputs["receptor"]["structure"]
        lig_structure = inputs["ligand"]["structure"]
        lig_structure.coord = lig_aa_coords
        complex_structure = combine_atom_arrays(rec_structure, lig_structure)
        
        # output
        out_pdb =  os.path.join(args.out_dir, f'{id}_{i}.pdb')
        file = PDBFile()
        file.set_structure(complex_structure)
        file.write(out_pdb)

    return metrics_list 


#----------------------------------------------------------------------------
# main function

def main(args):
    paths_list = []
    if args.paths:
        id, in_pdb_1, in_pdb_2 = args.paths
        if os.path.exists(in_pdb_1) and os.path.exists(in_pdb_2):
            paths_list.append((id, in_pdb_1, in_pdb_2))
        else:
            print("One or both paths do not exist.")
    elif args.csv:
        if os.path.exists(args.csv):
            with open(args.csv, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    id, in_pdb_1, in_pdb_2 = row[0], row[1], row[2]
                    paths_list.append((id, in_pdb_1, in_pdb_2))
        else:
            print("CSV file does not exist.")

    # create output directory if not exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load esm model
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device).eval()
    
    # load score model
    model = DFMDock.load_from_checkpoint(
        args.ckpt, 
        map_location=device,
    )

    model.to(device).eval()

    results = []
    # inference
    for id, in_pdb_1, in_pdb_2 in paths_list:
        receptor = get_info_from_pdb(in_pdb_1)
        ligand = get_info_from_pdb(in_pdb_2)

        # prepare inputs 
        inputs = {
            "id": id,
            "receptor": receptor,
            "ligand": ligand,
        }

        batch = get_batch_from_inputs(inputs, batch_converter, esm_model, device)

        # Move the batch to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}

        # run
        metrics_list = run(args, model, inputs, batch, device)

        results.extend(metrics_list)

    # write metrics to csv
    if not os.path.exists(args.out_csv_dir):
        os.makedirs(args.out_csv_dir)

    # set output directory
    output_filename =  os.path.join(args.out_csv_dir, args.out_csv)

    with open(output_filename, "w", newline="") as csvfile:
        # Write header row to CSV file
        header = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A description of what your program does")

    # Add arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--paths', nargs=3, metavar=('id', 'in_pdb_1', 'in_pdb_2'), help='Input two pdb paths')
    group.add_argument('--csv', type=str, help='Input a CSV file containing pdb paths')
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint file", default='../checkpoints/dips/model_0.ckpt') 
    parser.add_argument("--out_dir", type=str, help="Path to the output file", default='./pdbs') 
    parser.add_argument("--out_csv_dir", type=str, help="Path to the output file", default='./csv_files') 
    parser.add_argument("--out_csv", type=str, help="Path to the output file", default='./test.csv') 
    parser.add_argument("--num_samples", type=int, help="Number of sample poses", default=1) 
    parser.add_argument("--num_steps", type=int, help="Number of sde steps", default=40) 
    parser.add_argument("--tr_noise_scale", type=float, help="Translation noise scale", default=0.5) 
    parser.add_argument("--rot_noise_scale", type=int, help="Rotation noise scale", default=0.5) 
    parser.add_argument("--use_clash_force", action='store_true', help="Use clash force") 
    parser.add_argument("--noise_annealing", action='store_true', help="Use clash force") 

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
