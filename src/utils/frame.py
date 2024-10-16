import torch
import torch.nn.functional as F
from einops import repeat
from utils.geometry import matrix_to_rotation_6d


def get_rotat(coords):
    # Get backbone coordinates. 
    n_coords = coords[:, 0, :]
    ca_coords = coords[:, 1, :]
    c_coords = coords[:, 2, :]

    # Gram-Schmidt process.
    v1 = c_coords - ca_coords 
    v2 = n_coords - ca_coords
    e1 = F.normalize(v1) 
    u2 = v2 - e1 * (torch.einsum('b i, b i -> b', e1, v2).unsqueeze(-1))
    e2 = F.normalize(u2) 
    e3 = torch.cross(e1, e2, dim=-1)

    # Get rotations.
    rotations=torch.stack([e1, e2, e3], dim=-1)
    return rotations

def get_trans(coords):
    return coords[:, 1, :]

def get_pair_dist(coords):
    coords = coords[:, 1, :]
    vec = repeat(coords, 'i c -> i j c', j=coords.size(0)) - repeat(coords, 'j c -> i j c', i=coords.size(0))
    dist = torch.norm(vec, dim=-1, keepdim=True)
    dist = rbf(dist, 2.0, 22.0, n_bins=16)
    return dist

def get_pair_direct(coords):
    ca_coords = coords[:, 1, :]
    vec = repeat(ca_coords, 'i c -> i j c', j=coords.size(0)) - repeat(ca_coords, 'j c -> i j c', i=coords.size(0))
    direct = F.normalize(vec, dim=-1)
    rotat = get_rotat(coords)
    rotat = repeat(rotat, 'r i j -> r c i j', c=rotat.size(0))
    direct = torch.einsum('r c i j, r c j -> r c i', rotat.transpose(-1, -2), direct)
    return direct

def get_pair_orient(coords):
    rotat = get_rotat(coords)
    rotat_i = repeat(rotat, 'r i j -> r c i j', c=rotat.size(0))
    rotat_j = repeat(rotat, 'c i j -> r c i j', r=rotat.size(0))
    orient = torch.einsum('r c i j, r c j k -> r c i k', rotat_i.transpose(-1, -2), rotat_j)
    orient = matrix_to_rotation_6d(orient)
    return orient

def get_pairs(coords):
    dists = get_pair_dist(coords)
    direct = get_pair_direct(coords)
    orient = get_pair_orient(coords)
    pair = torch.cat([dists, direct, orient], dim=-1)
    return pair

def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    #v_expand = torch.unsqueeze(values, -1)
    z = ((values.unsqueeze(-1) - rbf_centers) / rbf_std).squeeze(-2)
    return torch.exp(-z ** 2)