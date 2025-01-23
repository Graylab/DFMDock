import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from einops import repeat
from models.egnn import E_GCL
from models.DiT import DiTModule
from utils.coords6d import get_coords6d

#----------------------------------------------------------------------------
# Data class for model config

@dataclass
class ModelConfig:
    lm_embed_dim: int
    positional_embed_dim: int
    node_dim: int
    edge_dim: int
    inner_dim: int
    encoder_depth: int
    decoder_depth: int
    num_heads: int
    dropout: float = 0.0

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

def get_knn_and_sample_graph(x, e=None, knn=20, sample_size=40):
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

    if e is None:
        return edge_index

    edge_indices = torch.stack(edge_index, dim=1)
    edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

    return edge_index, edge_attr

#----------------------------------------------------------------------------
# nn Module

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


class EGNN_self(nn.Module):
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
        super(EGNN_self, self).__init__()
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


class CrossAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(CrossAttention, self).__init__()
        #MultiHead
        self.MultiHead_1 = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)

    def forward(self, input1, input2):
        output_1, attention_weights_1 = self.MultiHead_1(input1, input2, input2)
        return output_1, attention_weights_1


class Encoder(nn.Module):
    def __init__(self, node_dim, depth, dropout, num_heads=8):
        super(Encoder, self).__init__()
        self.intra_network = EGNN_self(
            node_dim=node_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=True, 
            normalize=True, 
            tanh=False,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(node_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(node_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm(node_dim)
        self.dropout3 = nn.Dropout(p=dropout)

        self.cross_attn = CrossAttention(node_dim, num_heads=num_heads)

        self.ff = nn.Sequential(
            nn.Linear(node_dim, 2*node_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(node_dim*2, node_dim),
        )

    def forward(self, rec_x, lig_x, rec_pos, lig_pos):
        rec_edge_index = get_knn_and_sample_graph(rec_pos[:, 1])
        lig_edge_index = get_knn_and_sample_graph(lig_pos[:, 1])
        _rec_x = rec_x
        _lig_x = lig_x
        rec_x = self.norm1(self.dropout1(self.intra_network(rec_x, rec_pos[:, 1], rec_edge_index)) + _rec_x)
        lig_x = self.norm1(self.dropout1(self.intra_network(lig_x, lig_pos[:, 1], lig_edge_index)) + _lig_x)

        # Cross-Attention
        _rec_x = rec_x
        _lig_x = lig_x
        rec_attn, rec_attn_weights = self.cross_attn(rec_x, lig_x)
        lig_attn, lig_attn_weights = self.cross_attn(lig_x, rec_x)
        rec_x = self.norm2(self.dropout2(rec_attn) + _rec_x)
        lig_x = self.norm2(self.dropout2(lig_attn) + _lig_x)

        # Feed-Forward
        _rec_x = rec_x
        _lig_x = lig_x
        rec_x = self.norm3(self.dropout3(self.ff(rec_x)) + _rec_x)
        lig_x = self.norm3(self.dropout3(self.ff(lig_x)) + _lig_x)
        x = torch.cat([rec_x, lig_x], dim=0)

        return x


#----------------------------------------------------------------------------
# Main score network

class Score_Net(nn.Module):
    """EGNN backbone for translation and rotation scores"""
    def __init__(
        self, 
        conf,
    ):
        super().__init__()
        lm_embed_dim = conf.lm_embed_dim
        positional_embed_dim = conf.positional_embed_dim
        node_dim = conf.node_dim
        edge_dim = conf.edge_dim
        inner_dim = conf.inner_dim
        encoder_depth = conf.encoder_depth
        decoder_depth = conf.decoder_depth
        num_heads = conf.num_heads
        dropout = conf.dropout
        
        # node embedding
        self.single_embed = nn.Linear(lm_embed_dim, node_dim, bias=False)

        # edge embedding
        self.edge_embed = nn.Linear(positional_embed_dim, edge_dim, bias=False)

        # sigma embedding
        self.sigma_data = 0.5

        # encoder
        self.encoder = Encoder(
            node_dim=node_dim, 
            depth=encoder_depth, 
            dropout=dropout,
        )

        # transition
        self.node_embed = nn.Sequential(
            nn.LayerNorm(node_dim + 3),
            nn.Linear(node_dim + 3, node_dim, bias=False),
        )
        self.pos_to_node = nn.Linear(3, node_dim, bias=False)

        # decoder
        self.decoder = DiTModule(
            dim=node_dim, 
            pairwise_state_dim=edge_dim, 
            num_heads=num_heads, 
            depth=decoder_depth,
        )

        # energy head
        self.to_energy = nn.Sequential(
            nn.Linear(node_dim, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
        )

        # force head
        self.to_force = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, 3, bias=False),
        )

    def forward(self, batch, predict=False, return_energy=False):
        # get inputs
        rec_x = batch["rec_x"] 
        lig_x = batch["lig_x"] 
        rec_pos = batch["rec_pos"] 
        lig_pos = batch["lig_pos"] 
        pos = batch["pos"] 
        sigma = batch["sigma"]
        position_matrix = batch["position_matrix"]

        # single embed
        rec_x = self.single_embed(rec_x)
        lig_x = self.single_embed(lig_x)

        # encoder
        x = self.encoder(rec_x, lig_x, rec_pos, lig_pos)

        # pre-conditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # get the current complex pose
        pos = pos.view(-1, 3)
        pos.requires_grad_()

        # node feature embedding
        x = x.unsqueeze(1).repeat(1, 3, 1)
        one_hot = torch.eye(3, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)

        # node embedding
        node = torch.cat([x, one_hot], dim=-1)
        node = self.node_embed(node) # [n, 3, c]
        node = node.view(-1, node.size(-1)) # [3n, c]

        # scale pos
        pos_in = c_in * pos
        node = node + self.pos_to_node(pos_in)

        # edge feature embedding
        position_matrix = position_matrix.repeat(3, 3, 1) # [3n, 3n, c]
        edge = self.edge_embed(position_matrix) # [3n, 3n, c]

        # decoder
        node = self.decoder(node, c_noise, bias=edge) # [R+L, H]

        # energy
        energy = self.to_energy(node).sum()

        if return_energy:
            return energy

        # force
        f = self.to_force(node)

        # denoised pos
        denoised_pos = c_skip * pos + c_out * f

        if predict:
            outputs = {
                "energy": energy,
                "f": f,
                "pos": denoised_pos,
            }

            return outputs

        # dedx
        gradient = torch.autograd.grad(
            outputs=energy, 
            inputs=pos, 
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training, 
            retain_graph=self.training,
            only_inputs=True, 
            allow_unused=True,
        )[0]

        dedx = -gradient # F / kT
        
        outputs = {
            "energy": energy,
            "f": f,
            "dedx": dedx,
            "pos": denoised_pos,
        }

        return outputs
    
#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    conf = ModelConfig(
        lm_embed_dim=1280,
        positional_embed_dim=68,
        node_dim=24,
        edge_dim=12,
        inner_dim=24,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=8,
    )

    model = Score_Net(conf)

    rec_x = torch.randn(45, 1280)
    lig_x = torch.randn(5, 1280)
    rec_pos = torch.randn(45, 3, 3)
    lig_pos = torch.randn(5, 3, 3)
    pos = torch.randn(50, 3, 3)
    sigma = torch.tensor([0.5])
    position_matrix = torch.zeros(50, 50, 68)

    batch = {
        "rec_x": rec_x,
        "lig_x": lig_x,
        "rec_pos": rec_pos,
        "lig_pos": lig_pos,
        "pos": pos,
        "sigma": sigma,
        "position_matrix": position_matrix,
    }

    out = model(batch)
    print(out)
