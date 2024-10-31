import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from einops import repeat
from models.egnn import E_GCL
from utils.coords6d import get_coords6d
from utils.frame import get_pairs

#----------------------------------------------------------------------------
# Data class for model config

@dataclass
class ModelConfig:
    lm_embed_dim: int
    positional_embed_dim: int
    spatial_embed_dim: int
    node_dim: int
    edge_dim: int
    inner_dim: int
    depth: int
    dropout: float = 0.0
    cut_off: float = 20.0
    normalize: bool = True
    agg: str = 'mean'

#----------------------------------------------------------------------------
# Helper functions

def get_spatial_matrix(coord):
    dist, omega, theta, phi = get_coords6d(coord)

    mask = dist < 20.0
    
    num_dist_bins = 40
    num_omega_bins = 24
    num_theta_bins = 24
    num_phi_bins = 12
    dist_bin = get_bins(dist, 3.25, 50.75, num_dist_bins)
    omega_bin = get_bins(omega, -180.0, 180.0, num_omega_bins)
    theta_bin = get_bins(theta, -180.0, 180.0, num_theta_bins)
    phi_bin = get_bins(phi, 0.0, 180.0, num_phi_bins)

    def mask_mat(mat, num_bins):
        mat[~mask] = 0
        mat.fill_diagonal_(0)
        return mat

    omega_bin = mask_mat(omega_bin, num_omega_bins)
    theta_bin = mask_mat(theta_bin, num_theta_bins)
    phi_bin = mask_mat(phi_bin, num_phi_bins)

    # to onehot
    dist = F.one_hot(dist_bin, num_classes=num_dist_bins).float()
    omega = F.one_hot(omega_bin, num_classes=num_omega_bins).float() 
    theta = F.one_hot(theta_bin, num_classes=num_theta_bins).float() 
    phi = F.one_hot(phi_bin, num_classes=num_phi_bins).float() 
    
    return torch.cat([dist, omega, theta, phi], dim=-1)

def get_bins(x, min_bin, max_bin, num_bins):
    # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        num_bins - 1,
        device=x.device,
    )
    bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]
    return bins

def get_clashes(distance):
    return torch.sum(distance <= 3.0)

def sample_indices(matrix, num_samples):
    n, m = matrix.shape
    # Generate random permutations of indices for each row
    permuted_indices = torch.argsort(torch.rand(n, m, device=matrix.device), dim=1)

    # Select the first num_samples indices from each permutation
    sampled_indices = permuted_indices[:, :num_samples]

    return sampled_indices

def get_knn_and_sample(points, knn=20, sample_size=40, epsilon=1e-10):
    device = points.device
    n_points = points.size(0)

    if n_points < knn:
        knn = n_points
        sample_size = 0

    if n_points < knn + sample_size:
        sample_size = n_points - knn
    
    # Step 1: Compute pairwise distances
    dist_matrix = torch.cdist(points, points)
    
    # Step 2: Find the 20 nearest neighbors (including the point itself)
    _, knn_indices = torch.topk(dist_matrix, k=knn, largest=False)
    
    if sample_size > 0:
        # Step 3: Create a mask for the non-knn points
        mask = torch.ones(n_points, n_points, dtype=torch.bool, device=device)
        mask.scatter_(1, knn_indices, False)
        
        # Select the non-knn distances and compute inverse cubic distances
        non_knn_distances = dist_matrix[mask].view(n_points, -1)
        
        # Replace zero distances with a small value to avoid division by zero
        non_knn_distances = torch.where(non_knn_distances < epsilon, torch.tensor(epsilon, device=device), non_knn_distances)
        
        inv_cubic_distances = 1 / torch.pow(non_knn_distances, 3)
        
        # Normalize the inverse cubic distances to get probabilities
        probabilities = inv_cubic_distances / inv_cubic_distances.sum(dim=1, keepdim=True)
        
        # Ensure there are no NaNs or negative values
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        probabilities = torch.clamp(probabilities, min=0)
        
        # Normalize again to ensure it's a proper probability distribution
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        
        # Generate a tensor of indices excluding knn_indices
        all_indices = torch.arange(n_points, device=device).expand(n_points, n_points)
        non_knn_indices = all_indices[mask].view(n_points, -1)
        
        # Step 4: Sample 40 indices based on the probability distribution
        sample_indices = torch.multinomial(probabilities, sample_size, replacement=False)
        sampled_points_indices = non_knn_indices.gather(1, sample_indices)
    else:
        sampled_points_indices = None
    
    return knn_indices, sampled_points_indices

#----------------------------------------------------------------------------
# Edge seletion functions

def get_knn_and_sample_graph(x, e, knn=20, sample_size=40):
    knn_indices, sampled_points_indices = get_knn_and_sample(x, knn=knn, sample_size=sample_size)
    if sampled_points_indices is not None:
        indices = torch.cat([knn_indices, sampled_points_indices], dim=-1)
    else:
        indices = knn_indices
    n_points, n_samples = indices.shape

    # edge src and dst
    edge_src = torch.arange(start=0, end=n_points, device=x.device)[..., None].repeat(1, n_samples).reshape(-1)
    edge_dst = indices.reshape(-1)

    # combine graphs
    edge_index = [edge_src, edge_dst]
    edge_indices = torch.stack(edge_index, dim=1)
    edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

    return edge_index, edge_attr

#----------------------------------------------------------------------------
# nn Modules

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=1.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EGNNLayer(nn.Module):
    def __init__(
        self, 
        node_dim, 
        edge_dim=0, 
        act_fn=nn.SiLU(), 
        residual=True, 
        attention=False, 
        normalize=False, 
        tanh=False, 
        update_coords=False,
        coord_weights_clamp_value=2.0,
        dropout=0.0,
    ):
        super(EGNNLayer, self).__init__()
        self.egcl = E_GCL(
            input_nf=node_dim, 
            output_nf=node_dim, 
            hidden_nf=node_dim, 
            edges_in_d=edge_dim,
            act_fn=act_fn, 
            residual=residual, 
            attention=attention,
            normalize=normalize, 
            tanh=tanh, 
            update_coords=update_coords,
            coord_weights_clamp_value=coord_weights_clamp_value,
            dropout=dropout,
        )

    def forward(self, h, x, edges, edge_attr=None, lig_mask=None):
        h, x, edge_attr = self.egcl(h, edges, x, edge_attr=edge_attr, lig_mask=lig_mask)
        return h, x, edge_attr


class EGNN(nn.Module):
    def __init__(
        self, 
        node_dim, 
        edge_dim=0, 
        act_fn=nn.SiLU(), 
        depth=4, 
        residual=True, 
        attention=False, 
        normalize=False, 
        tanh=False,
        dropout=0.0,
    ):
        super(EGNN, self).__init__()
        self.depth = depth
        for i in range(depth):
            self.add_module("EGNN_%d" % i, EGNNLayer(
                node_dim=node_dim, 
                edge_dim=edge_dim,
                act_fn=act_fn, 
                residual=residual, 
                attention=attention,
                normalize=normalize, 
                tanh=tanh,
                dropout=dropout,
                update_coords=False,
            )
        )

    def forward(self, h, x, edges, edge_attr=None, lig_mask=None):
        for i in range(self.depth):
            h, x, edge_attr = self._modules["EGNN_%d" % i](h, x, edges, edge_attr=edge_attr, lig_mask=lig_mask)
        return h


#----------------------------------------------------------------------------
# Main score network

class EGNN_Net(nn.Module):
    """EGNN backbone for translation and rotation scores"""
    def __init__(
        self, 
        conf,
    ):
        super().__init__()
        lm_embed_dim = conf.lm_embed_dim
        spatial_embed_dim = conf.spatial_embed_dim
        positional_embed_dim = conf.positional_embed_dim
        node_dim = conf.node_dim
        edge_dim = conf.edge_dim
        inner_dim = conf.inner_dim
        depth = conf.depth
        dropout = conf.dropout
        normalize = conf.normalize
        
        self.agg = conf.agg
        self.cut_off = conf.cut_off
        
        # single init embedding
        self.single_embed = nn.Linear(lm_embed_dim, node_dim, bias=False)

        # pair init embedding
        self.spatial_embed = nn.Linear(spatial_embed_dim, edge_dim, bias=False)
        self.positional_embed = nn.Linear(positional_embed_dim, edge_dim, bias=False)

        # denoising score network
        self.network = EGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=True, 
            normalize=normalize, 
            tanh=False,
            dropout=dropout,
        )

        # energy head
        self.to_energy = nn.Sequential(
            nn.Linear(2*node_dim + 1, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
        )

        # force head 
        self.to_force = nn.Sequential(
            nn.Linear(2*node_dim + 1, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
        )

        # dist head
        self.to_dist = nn.Sequential(
            nn.Linear(2*node_dim + 1, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 64, bias=False),
        )

        # confidence head
        self.to_confidence = nn.Sequential(
            nn.Linear(2*node_dim + 1, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
        )

        # interface residue head
        self.to_ires = nn.Sequential(
            nn.Linear(node_dim, 2*node_dim),
            nn.SiLU(),
            nn.Linear(2*node_dim, 2*node_dim),
            nn.SiLU(),
            nn.Linear(2*node_dim, 1),
        )

        # timestep embedding
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=inner_dim),
            nn.Linear(inner_dim, inner_dim, bias=False),
            nn.Sigmoid(),
        )

        # tr_scale mlp
        self.tr_scale = nn.Sequential(
            nn.Linear(inner_dim + 1, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(inner_dim, 1, bias=False),
            nn.Softplus()
        )

        # rot_scale mlp
        self.rot_scale = nn.Sequential(
            nn.Linear(inner_dim + 1, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(inner_dim, 1, bias=False),
            nn.Softplus()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, batch, predict=False, return_energy=False):
        # get inputs
        rec_x = batch["rec_x"] 
        lig_x = batch["lig_x"] 
        rec_pos = batch["rec_pos"] 
        lig_pos = batch["lig_pos"] 
        position_matrix = batch["position_matrix"]
        t = batch["t"]

        # get ca distance matrix 
        vec = rec_pos[:, None, 1, :] - lig_pos[None, :, 1, :] 
        unit_vec = F.normalize(vec, dim=-1)
        D = torch.norm(vec, dim=-1)
        
        # get current complex pose
        lig_pos.requires_grad_()
        pos = torch.cat([rec_pos, lig_pos], dim=0)

        # node feature embedding
        x = torch.cat([rec_x, lig_x], dim=0)
        node = self.single_embed(x) # [n, c]

        # edge feature embedding
        spatial_matrix = get_spatial_matrix(pos)
        edge = self.spatial_embed(spatial_matrix) + self.positional_embed(position_matrix)

        # sample edge_index and get edge_attr
        edge_index, edge_attr = get_knn_and_sample_graph(pos[..., 1, :], edge)

        # main network 
        node = self.network(node, pos[..., 1, :], edge_index, edge_attr) # [R+L, H]

        # energy
        h_rec = repeat(node[:rec_pos.size(0)], 'n h -> n m h', m=lig_pos.size(0))
        h_lig = repeat(node[rec_pos.size(0):], 'm h -> n m h', n=rec_pos.size(0))
        interaction = torch.cat([h_rec, h_lig, D.unsqueeze(-1)], dim=-1)
        energy = self.to_energy(interaction).squeeze(-1) # [R, L]
        mask_2D = (D < self.cut_off).float() # [R, L]

        if self.agg == 'mean':
            energy = (energy * mask_2D).sum() / mask_2D.sum().clamp(min=1)
        else:
            energy = (energy * mask_2D).sum() 

        if return_energy:
            return energy

        # confidence head
        confidence_logits = self.to_confidence(interaction).mean()

        # contact head
        dist_logits = self.to_dist(interaction)

        # interface residue head
        ires_logits = self.to_ires(node)

        # force
        fij = unit_vec * self.to_force(interaction) 
        if self.agg == 'mean':
            f = fij.mean(dim=0)
        else:
            f = fij.sum(dim=0)

        # translation
        if self.agg == 'mean':
            tr_pred = f.mean(dim=0, keepdim=True)
        else:
            tr_pred = f.sum(dim=0, keepdim=True)

        # rotation
        r = lig_pos[..., 1, :].detach()
        if self.agg == 'mean':
            rot_pred = torch.cross(r, f, dim=-1).mean(dim=0, keepdim=True)
        else:
            rot_pred = torch.cross(r, f, dim=-1).sum(dim=0, keepdim=True)

        # scale
        t = self.t_embed(t)
        tr_norm = torch.linalg.vector_norm(tr_pred, keepdim=True)
        tr_score = tr_pred / (tr_norm + 1e-6) * self.tr_scale(torch.cat([tr_norm, t], dim=-1))
        rot_norm = torch.linalg.vector_norm(rot_pred, keepdim=True)
        rot_score = rot_pred / (rot_norm + 1e-6) * self.rot_scale(torch.cat([rot_norm, t], dim=-1))

        if predict:
            num_clashes = get_clashes(D)

            outputs = {
                "tr_score": tr_score,
                "rot_score": rot_score,
                "energy": energy,
                "f": f,
                "num_clashes": num_clashes,
                "dist_logits": dist_logits,
                "ires_logits": ires_logits,
                "confidence_logits": confidence_logits,
            }

            return outputs

        # dedx
        dedx = torch.autograd.grad(
            outputs=energy, 
            inputs=lig_pos, 
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training, 
            retain_graph=self.training,
            only_inputs=True, 
            allow_unused=True,
        )[0]

        dedx = -dedx[..., 1, :] # F / kT
        
        outputs = {
            "tr_score": tr_score,
            "rot_score": rot_score,
            "energy": energy,
            "f": f,
            "dedx": dedx,
            "dist_logits": dist_logits,
            "ires_logits": ires_logits,
            "confidence_logits": confidence_logits,
        }

        return outputs
    
#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    conf = ModelConfig(
        lm_embed_dim=1280,
        positional_embed_dim=66,
        spatial_embed_dim=100,
        node_dim=24,
        edge_dim=12,
        inner_dim=24,
        depth=2,
    )

    model = EGNN_Net(conf)

    rec_x = torch.randn(40, 1280)
    lig_x = torch.randn(5, 1280)
    rec_pos = torch.randn(40, 3, 3)
    lig_pos = torch.randn(5, 3, 3)
    position_matrix = torch.randn(45, 45, 66)
    t = torch.tensor([0.5])

    batch = {
        "rec_x": rec_x,
        "lig_x": lig_x,
        "rec_pos": rec_pos,
        "lig_pos": lig_pos,
        "position_matrix": position_matrix,
        "t": t,
    }

    out = model(batch)
    print(out)
