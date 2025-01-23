import torch

def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return torch.nn.functional.one_hot(am, num_classes=len(v_bins)).float()

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
            start=0, end=2 * max_relative_idx + 2, device=res_id.device
        )
        rel_pos = one_hot(
            final_offset,
            boundaries,
        )

        rel_feats.append(rel_pos)

    else:
        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 1, device=res_id.device
        )
        rel_pos = one_hot(
            clipped_offset, boundaries,
        )
        rel_feats.append(rel_pos)

    rel_feat = torch.cat(rel_feats, dim=-1).float()

    return rel_feat

def get_interface_residues(coords, asym_id, interface_threshold=10.0):
    coord_diff = coords[..., None, :, :] - coords[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    mask = diff_chain_mask[..., None].bool()
    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)
    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_idxs

def get_spatial_crop_idx(coords, asym_id, crop_size=256, interface_threshold=10.0):
    interface_residues = get_interface_residues(coords, asym_id, interface_threshold=interface_threshold)

    if not torch.any(interface_residues):
        return get_contiguous_crop_idx(asym_id, crop_size)

    target_res_idx = randint(lower=0, upper=interface_residues.shape[-1] - 1)
    target_res = interface_residues[target_res_idx]

    ca_positions = coords[..., 1, :]
    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    to_target_distances = ca_pairwise_dists[target_res]

    break_tie = (
            torch.arange(
                0, to_target_distances.shape[-1], device=coords.device
            ).float()
            * 1e-3
    )
    to_target_distances += break_tie
    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values

def get_contiguous_crop_idx(asym_id, crop_size):
    unique_asym_ids, chain_idxs, chain_lens = asym_id.unique(dim=-1,
                                                             return_inverse=True,
                                                             return_counts=True)
    
    shuffle_idx = torch.randperm(chain_lens.shape[-1])
    

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]), dim=0)
    asym_offsets = idx_sorted[cum_sum]

    num_budget = crop_size
    num_remaining = len(chain_idxs)

    crop_idxs = []
    for i, idx in enumerate(shuffle_idx):
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        if i == 0:
            crop_size_max = min(num_budget - 50, chain_len)
            crop_size_min = min(chain_len, 50)
        else:
            crop_size_max = min(num_budget, chain_len)
            crop_size_min = min(chain_len, max(50, num_budget - num_remaining))

        chain_crop_size = randint(lower=crop_size_min,
                                  upper=crop_size_max)

        num_budget -= chain_crop_size

        chain_start = randint(lower=0,
                              upper=chain_len - chain_crop_size)

        asym_offset = asym_offsets[idx]
        crop_idxs.append(
            torch.arange(asym_offset + chain_start, asym_offset + chain_start + chain_crop_size)
        )

    return torch.concat(crop_idxs).sort().values

def randint(lower, upper):
    return int(torch.randint(
        lower,
        upper + 1,
        (1,),
    )[0])

def get_crop_idxs(batch, crop_size):
    rec_pos = batch["rec_pos"]
    lig_pos = batch["lig_pos"]
    n = rec_pos.size(0) + lig_pos.size(0)
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    asym_id = torch.zeros(n, device=pos.device).long()
    asym_id[rec_pos.size(0):] = 1

    use_spatial_crop = True
    num_res = asym_id.size(0)

    if num_res <= crop_size:
        crop_idxs = torch.arange(num_res)
    elif use_spatial_crop:
        crop_idxs = get_spatial_crop_idx(pos, asym_id, crop_size=crop_size)
    else:
        crop_idxs = get_contiguous_crop_idx(asym_id, crop_size=crop_size)

    crop_idxs = crop_idxs.to(pos.device)

    return crop_idxs

def get_crop(batch, crop_size):
    crop_idxs = get_crop_idxs(batch, crop_size)

    rec_x = batch["rec_x"]
    lig_x = batch["lig_x"]
    rec_pos = batch["rec_pos"]
    lig_pos = batch["lig_pos"]
    ires = batch["ires"]
    pair_matrix = batch["pair_matrix"]

    n = rec_x.size(0) + lig_x.size(0)
    x = torch.cat([rec_x, lig_x], dim=0)
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    res_id = torch.arange(n, device=x.device).long()
    asym_id = torch.zeros(n, device=x.device).long()
    asym_id[rec_x.size(0):] = 1

    res_id = torch.index_select(res_id, 0, crop_idxs)
    asym_id = torch.index_select(asym_id, 0, crop_idxs)
    x = torch.index_select(x, 0, crop_idxs)
    pos = torch.index_select(pos, 0, crop_idxs)
    ires = torch.index_select(ires, 0, crop_idxs)
    pair_matrix = torch.index_select(pair_matrix, 0, crop_idxs)
    pair_matrix = torch.index_select(pair_matrix, 1, crop_idxs)

    sep = asym_id.tolist().index(1)
    rec_x = x[:sep]
    lig_x = x[sep:]
    rec_pos = pos[:sep]
    lig_pos = pos[sep:]

    # Positional embeddings
    position_matrix = relpos(res_id, asym_id).to(x.device)

    batch["rec_x"] = rec_x
    batch["lig_x"] = lig_x
    batch["rec_pos"] = rec_pos
    batch["lig_pos"] = lig_pos
    batch["position_matrix"] = position_matrix
    batch["pair_matrix"] = pair_matrix
    batch["ires"] = ires

    return batch

def get_crop_no_pair(batch, crop_size):
    crop_idxs = get_crop_idxs(batch, crop_size)

    rec_x = batch["rec_x"]
    lig_x = batch["lig_x"]
    rec_pos = batch["rec_pos"]
    lig_pos = batch["lig_pos"]
    ires = batch["ires"]

    n = rec_x.size(0) + lig_x.size(0)
    x = torch.cat([rec_x, lig_x], dim=0)
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    res_id = torch.arange(n, device=x.device).long()
    asym_id = torch.zeros(n, device=x.device).long()
    asym_id[rec_x.size(0):] = 1

    res_id = torch.index_select(res_id, 0, crop_idxs)
    asym_id = torch.index_select(asym_id, 0, crop_idxs)
    x = torch.index_select(x, 0, crop_idxs)
    pos = torch.index_select(pos, 0, crop_idxs)
    ires = torch.index_select(ires, 0, crop_idxs)

    sep = asym_id.tolist().index(1)
    rec_x = x[:sep]
    lig_x = x[sep:]
    rec_pos = pos[:sep]
    lig_pos = pos[sep:]

    # Positional embeddings
    position_matrix = relpos(res_id, asym_id).to(x.device)

    batch["rec_x"] = rec_x
    batch["lig_x"] = lig_x
    batch["rec_pos"] = rec_pos
    batch["lig_pos"] = lig_pos
    batch["position_matrix"] = position_matrix
    batch["ires"] = ires

    return batch

def get_position_matrix(batch):
    rec_x = batch["rec_x"]
    lig_x = batch["lig_x"]
    x = torch.cat([rec_x, lig_x], dim=0)
    
    res_id = torch.arange(x.size(0), device=x.device).long()
    asym_id = torch.zeros(x.size(0), device=x.device).long()
    asym_id[rec_x.size(0):] = 1

    # Positional embeddings
    position_matrix = relpos(res_id, asym_id).to(x.device)

    batch["position_matrix"] = position_matrix

    return batch
