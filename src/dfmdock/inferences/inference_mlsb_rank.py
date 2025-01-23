import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import packages
import os
import csv
import torch
import numpy as np
import hydra
import random
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils import data
from scipy.spatial.transform import Rotation 
from dfmdock.models.score_model_mlsb import Score_Model
from dfmdock.models.rank_model_mlsb import Rank_Model
from dfmdock.datasets.ppi_mlsb_dataset import PPIDataset
from dfmdock.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from dfmdock.utils.pdb import save_PDB, place_fourth_atom 
from dfmdock.utils.metrics import compute_metrics

#----------------------------------------------------------------------------
# Data class for pose

@dataclass
class pose():
    _id: str
    rec_seq: str
    lig_seq: str
    rec_pos: torch.FloatTensor
    lig_pos: torch.FloatTensor
    index: str = None

#----------------------------------------------------------------------------
# Helper functions

def set_seed(seed=42):
    # Set seed for Python's random library
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)

def sample_sphere(radius=1):
    """Samples a random point on the surface of a sphere.

    Args:
        radius: The radius of the sphere.

    Returns:
        A 3D NumPy array representing the sampled point.
    """

    # Generate two random numbers in the range [0, 1).
    u = np.random.rand()
    v = np.random.rand()

    # Compute the azimuthal and polar angles.
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    # Compute the Cartesian coordinates of the sampled point.
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return np.array([x, y, z])

def rot_compose(r1, r2):
    R1 = axis_angle_to_matrix(r1)
    R2 = axis_angle_to_matrix(r2)
    R = torch.einsum('b i j, b j k -> b i k', R2, R1)
    r = matrix_to_axis_angle(R)
    return r
     
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
# Sampler

class Sampler:
    def __init__(
        self,
        conf: DictConfig,
    ):
        self.data_conf = conf.data
        self.perturb_tr = self.data_conf.perturb_tr
        self.perturb_rot = self.data_conf.perturb_rot

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load score model
        self.score_model = Score_Model.load_from_checkpoint(
            self.data_conf.score_model, 
            map_location=self.device,
        )
        self.score_model.eval()
        self.score_model.to(self.device)

        # load rank model
        self.rank_model = Rank_Model.load_from_checkpoint(
            self.data_conf.rank_model, 
            map_location=self.device,
        )
        self.rank_model.eval()
        self.rank_model.to(self.device)
        
        # get testset
        testset = PPIDataset(
            dataset=self.data_conf.dataset, 
            training=False, 
            use_esm=self.data_conf.use_esm,
        )

        # load dataset
        if self.data_conf.test_all:
            self.test_dataloader = data.DataLoader(testset, batch_size=1, num_workers=6)
        else:
            # get subset
            subset_indices = [0]
            subset = data.Subset(testset, subset_indices)
            self.test_dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    
    def get_metrics(self, pred, label):
        metrics = compute_metrics(pred, label)
        return metrics
    
    def save_trj(self, pred):
        # create output directory if not exist
        if not os.path.exists(self.data_conf.out_trj_dir):
            os.makedirs(self.data_conf.out_trj_dir)

        # set output directory
        out_pdb =  os.path.join(self.data_conf.out_trj_dir, pred._id + '_p' + pred.index + '.pdb')

        # output trajectory
        if os.path.exists(out_pdb):
            os.remove(out_pdb)
            
        seq1 = pred.rec_seq
        seq2 = pred.lig_seq
            
        for i, (x1, x2) in enumerate(zip(pred.rec_pos, pred.lig_pos)):
            coords = torch.cat([x1, x2], dim=0)
            coords = get_full_coords(coords)

            #get total len
            total_len = coords.size(0)

            #check seq len
            assert len(seq1) + len(seq2) == total_len

            #get pdb
            f = open(out_pdb, 'a')
            f.write("MODEL        " + str(i) + "\n")
            f.close()
            save_PDB(out_pdb=out_pdb, coords=coords, seq=seq1+seq2, delim=len(seq1)-1)

    def save_pdb(self, pred):
        # create output directory if not exist
        if not os.path.exists(self.data_conf.out_pdb_dir):
            os.makedirs(self.data_conf.out_pdb_dir)

        # set output directory
        out_pdb =  os.path.join(self.data_conf.out_pdb_dir, pred._id + '_p' + pred.index + '.pdb')

        # output trajectory
        if os.path.exists(out_pdb):
            os.remove(out_pdb)
            
        seq1 = pred.rec_seq
        seq2 = pred.lig_seq
            
        coords = torch.cat([pred.rec_pos[-1], pred.lig_pos[-1]], dim=0)
        coords = get_full_coords(coords)

        # get total len
        total_len = coords.size(0)

        # check seq len
        assert len(seq1) + len(seq2) == total_len

        # get pdb
        save_PDB(out_pdb=out_pdb, coords=coords, seq=seq1+seq2, delim=len(seq1)-1)
    
    def run_sampling(self):
        metrics_list = []
        transforms_list = []
        for batch in tqdm(self.test_dataloader):
            # get batch from testset loader
            _id = batch['id'][0]
            rec_seq = batch['rec_seq'][0]
            lig_seq = batch['lig_seq'][0]
            rec_x = batch['rec_x'].to(self.device).squeeze(0)
            lig_x = batch['lig_x'].to(self.device).squeeze(0)
            rec_pos = batch['rec_pos'].to(self.device).squeeze(0)
            lig_pos = batch['lig_pos'].to(self.device).squeeze(0)
            position_matrix = batch['position_matrix'].to(self.device).squeeze(0)

            batch = {
                "rec_x": rec_x,
                "lig_x": lig_x,
                "rec_pos": rec_pos.clone().detach(),
                "lig_pos": lig_pos.clone().detach(),
                "position_matrix": position_matrix,
            }

            # get ground truth pose
            label = pose(
                _id=_id,
                rec_seq=rec_seq,
                lig_seq=lig_seq,
                rec_pos=rec_pos,
                lig_pos=lig_pos
            )

            if self.data_conf.get_gt_energy:
                batch["t"] = torch.zeros(1, device=self.device) + 1e-5
                output = self.score_model(batch)

                metrics = {'id': _id}
                metrics.update(self.get_metrics([rec_pos, lig_pos], [rec_pos, lig_pos]))
                metrics.update({'energy': output["energy"].item()})
                metrics.update({'num_clashes': output["num_clashes"].item()})
                metrics_list.append(metrics)
            
            else:
                # run 
                for i in range(self.data_conf.num_samples):
                    _rec_pos, _lig_pos, energy, num_clashes, logits = self.Euler_Maruyama_sampler(
                        batch=batch,
                        batch_size=1,
                        eps=1e-3,
                        ode=self.data_conf.ode,
                    )
                    
                    # get predicted pose
                    pred = pose(
                        _id=_id,
                        rec_seq=rec_seq,
                        lig_seq=lig_seq,
                        rec_pos=_rec_pos,
                        lig_pos=_lig_pos,
                        index=str(i)
                    )

                    # get metrics
                    metrics = {'id': _id, 'index': str(i)}
                    metrics.update(self.get_metrics([_rec_pos[-1], _lig_pos[-1]], [rec_pos, lig_pos]))
                    metrics.update({'energy': energy.item()})
                    metrics.update({'num_clashes': num_clashes.item()})
                    metrics.update({'logits': logits.item()})
                    metrics_list.append(metrics)

                    if self.data_conf.out_trj:
                        self.save_trj(pred)

                    if self.data_conf.out_pdb:
                        self.save_pdb(pred)

        return metrics_list


    def Euler_Maruyama_sampler(
        self,
        batch,
        batch_size=1, 
        eps=1e-3,
        ode=False,
    ):
        # coordinates and energy saver
        rec_trj = []
        lig_trj = []

        # initialize time steps
        t = torch.ones(batch_size, device=self.device)
        time_steps = torch.linspace(1., eps, self.data_conf.num_steps, device=self.device)
        dt = time_steps[0] - time_steps[1]

        # get initial pose
        rec_pos = batch["rec_pos"] 
        lig_pos = batch["lig_pos"] 

        # randomly initialize coordinates
        rec_pos, lig_pos, rot_update, tr_update = self.randomize_pose(rec_pos, lig_pos)
        
        # save initial coordinates 
        rec_trj.append(rec_pos)
        lig_trj.append(lig_pos)

        # run reverse sde 
        with torch.no_grad():
            for i, time_step in enumerate(tqdm((time_steps))):  
                # get current time step 
                is_last = i == time_steps.size(0) - 1   
                t = torch.ones(batch_size, device=self.device) * time_step

                batch["t"] = t
                batch["rec_pos"] = rec_pos.clone().detach()
                batch["lig_pos"] = lig_pos.clone().detach()

                # get predictions
                output = self.score_model(batch) 

                if not is_last:
                    tr_noise_scale = self.data_conf.tr_noise_scale
                    rot_noise_scale = self.data_conf.rot_noise_scale
                else:
                    tr_noise_scale = 0.0
                    rot_noise_scale = 0.0

                if self.perturb_rot:
                    rot = self.score_model.so3_diffuser.torch_reverse(
                        score_t=output["rot_score"].detach(),
                        t=t.item(),
                        dt=dt,
                        noise_scale=rot_noise_scale,
                        ode=ode,
                    )
                else:
                    rot = torch.zeros((1, 3), device=self.device)

                if self.perturb_tr:
                    tr = self.score_model.r3_diffuser.torch_reverse(
                        score_t=output["tr_score"].detach(),
                        t=t.item(),
                        dt=dt,
                        noise_scale=tr_noise_scale,
                        ode=ode,
                    )
                else:
                    tr = torch.zeros((1, 3), device=self.device)

                lig_pos = self.modify_coords(lig_pos, rot, tr)

                # clash
                if self.data_conf.use_clash_force:
                    clash_force = self.clash_force(rec_pos.clone().detach(), lig_pos.clone().detach())
                    lig_pos = lig_pos + clash_force

                if is_last:
                    batch["rec_pos"] = rec_pos.clone().detach()
                    batch["lig_pos"] = lig_pos.clone().detach()
                    logits = self.rank_model(batch) 

                # save coordinates
                rec_trj.append(rec_pos)         
                lig_trj.append(lig_pos)
                
        return rec_trj, lig_trj, output["energy"], output["num_clashes"], logits

    def randomize_pose(self, x1, x2):
        # get center of mass
        c1 = torch.mean(x1[..., 1, :], dim=0)
        c2 = torch.mean(x2[..., 1, :], dim=0)

        # get rotat update
        rot_update = torch.from_numpy(Rotation.random().as_matrix()).float().to(self.device)

        # get trans update
        tr_update = torch.normal(0.0, 30.0, size=(1, 3), device=self.device)
        #tr_update = torch.from_numpy(sample_sphere(radius=50.0)).float().to(self.device)

        # move to origin
        x1 = x1 - c1
        x2 = x2 - c2

        # init rotation
        if self.perturb_rot:
            x2 = x2 @ rot_update.T

        # init translation
        if self.perturb_tr:
            x2 = x2 + tr_update 

        # convert to axis angle
        rot_update = matrix_to_axis_angle(rot_update.unsqueeze(0))

        return x1, x2, rot_update, tr_update

    def modify_coords(self, x, rot, tr):
        center = torch.mean(x[..., 1, :], dim=0, keepdim=True)
        rot = axis_angle_to_matrix(rot).squeeze()
        # update rotation
        if self.perturb_rot:
            x = (x - center) @ rot.T + center 
        # update translation
        if self.perturb_tr:
            x = x + tr
        
        return x
    
    def clash_force(self, rec_pos, lig_pos):
        rec_pos = rec_pos.view(-1, 3)
        lig_pos = lig_pos.view(-1, 3)

        with torch.set_grad_enabled(True):
            lig_pos.requires_grad_(True)
            # get distance matrix 
            D = torch.norm((rec_pos[:, None, :] - lig_pos[None, :, :]), dim=-1)

            def rep_fn(x):
                x0, p, w_rep = 4, 1.5, 5
                rep = torch.where(x < x0, (torch.abs(x0 - x) ** p) / (p * x * (p - 1)), torch.tensor(0.0, device=x.device)) 
                return - w_rep * torch.sum(rep)

            rep = rep_fn(D)

            force = torch.autograd.grad(rep, lig_pos, retain_graph=False)[0]

        return force.mean(dim=0).detach()
    
#----------------------------------------------------------------------------
# Main
@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DFMDock/configs", config_name="inference_rank") 
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    set_seed()
    sampler = Sampler(config)

    output_dir = config.data.out_csv_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set output directory
    output_filename =  os.path.join(output_dir, config.data.out_csv)

    with open(output_filename, "w", newline="") as csvfile:
        results = sampler.run_sampling()

        # Write header row to CSV file
        header = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
