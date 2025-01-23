import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from einops import repeat
from dfmdock.models.egnn import E_GCL
from dfmdock.utils.coords6d import get_coords6d

#----------------------------------------------------------------------------
# Data class for model config

@dataclass
class ModelConfig:
    lm_embed_dim: int
    positional_embed_dim: int
    spatial_embed_dim: int
    contact_embed_dim: int
    node_dim: int
    edge_dim: int
    inner_dim: int
    depth: int
    dropout: float = 0.0
    cut_off: float = 30.0
    normalize: bool = False

#----------------------------------------------------------------------------
# Helper functions

def get_spatial_matrix(coord):
    dist, omega, theta, phi = get_coords6d(coord)

    mask = dist < 22.0
    
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

def get_cross_graph(x, e, sep, num_self, num_cross):
    """cross graph from the complex pose"""

    # distance matrix
    d = torch.norm((x[:, None, :] - x[None, :, :]), dim=-1)

    # make sure the knn not exceed the size
    rec_len = sep
    lig_len = x.size(0) - sep

    # self and cross
    num_self_lig = num_self
    num_cross_lig = num_cross
    num_self_rec = num_self
    num_cross_rec = num_cross

    if num_self_lig > lig_len:
        num_self_lig = lig_len
    if num_cross_lig > rec_len:
        num_cross_lig = rec_len
    if num_self_rec > rec_len:
        num_self_rec = rec_len
    if num_cross_rec > lig_len:
        num_cross_rec = lig_len

    # intra and inter topk
    nbhd_ranking_ii, nbhd_indices_ii = d[..., :sep, :sep].topk(num_self_rec, dim=-1, largest=False)
    nbhd_ranking_jj, nbhd_indices_jj = d[..., sep:, sep:].topk(num_self_lig, dim=-1, largest=False)
    nbhd_ranking_ij, nbhd_indices_ij = d[..., :sep, sep:].topk(num_cross_rec, dim=-1, largest=False)
    nbhd_ranking_ji, nbhd_indices_ji = d[..., sep:, :sep].topk(num_cross_lig, dim=-1, largest=False)

    # edge src and dst
    edge_src_rec = torch.arange(start=0, end=rec_len, device=x.device)[..., None].repeat(1, num_self_rec+num_cross_rec)
    edge_src_lig = torch.arange(start=rec_len, end=rec_len+lig_len, device=x.device)[..., None].repeat(1, num_self_lig+num_cross_lig)
    edge_dst_rec = torch.cat([nbhd_indices_ii, nbhd_indices_ij + rec_len], dim=1)
    edge_dst_lig = torch.cat([nbhd_indices_ji, nbhd_indices_jj + rec_len], dim=1)
    edge_src = torch.cat([edge_src_rec.reshape(-1), edge_src_lig.reshape(-1)])
    edge_dst = torch.cat([edge_dst_rec.reshape(-1), edge_dst_lig.reshape(-1)])

    # combine graphs
    edge_index = [edge_src, edge_dst]
    edge_indices = torch.stack(edge_index, dim=1)
    edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

    return edge_index, edge_attr

#----------------------------------------------------------------------------
# nn Modules

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

class Rank_Net(nn.Module):
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
        self.to_logits = nn.Sequential(
            nn.Linear(2*node_dim, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
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
        t = batch["t"]
        position_matrix = batch["position_matrix"]

        # get the current complex pose
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
        node_out = self.network(node, pos[..., 1, :], edge_index, edge_attr) # [R+L, H]

        # confidence score
        h_rec = repeat(node_out[:rec_pos.size(0)], 'n h -> n m h', m=lig_pos.size(0))
        h_lig = repeat(node_out[rec_pos.size(0):], 'm h -> n m h', n=rec_pos.size(0))
        logits = self.to_logits(torch.cat([h_rec, h_lig], dim=-1)).mean()

        return logits

    
#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    conf = ModelConfig(
        lm_embed_dim=1280,
        positional_embed_dim=68,
        spatial_embed_dim=100,
        contact_embed_dim=1,
        node_dim=24,
        edge_dim=12,
        inner_dim=24,
        depth=2,
    )

    model = Rank_Net(conf)

    rec_x = torch.randn(40, 1280)
    lig_x = torch.randn(5, 1280)
    rec_pos = torch.randn(40, 3, 3)
    lig_pos = torch.randn(5, 3, 3)
    t = torch.tensor([0.5])
    contact_matrix = torch.zeros(45, 45)
    position_matrix = torch.zeros(45, 45, 68)

    batch = {
        "rec_x": rec_x,
        "lig_x": lig_x,
        "rec_pos": rec_pos,
        "lig_pos": lig_pos,
        "t": t,
        "contact_matrix": contact_matrix,
        "position_matrix": position_matrix,
    }

    out = model(batch)
    print(out)
