import torch


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

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

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    max_sum = max(torch.mean(predicted_tm_term, dim=0).max(), torch.mean(predicted_tm_term, dim=1).max())

    return max_sum

def get_tm_loss(
    logits,
    sq_diff,
    max_bin=31,
    no_bins=64,
):
    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    loss = torch.mean(errors)

    return loss

def distogram_loss(
    logits,
    dists,
    min_bin=3.25, 
    max_bin=50.75,
    no_bins=64,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    """
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    true_bins = torch.sum(dists ** 2 > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    loss = torch.mean(errors)

    return loss

def between_residue_bond_loss(
    pred_coords: torch.Tensor,
    eps: float = 1e-6,
    tolerance_factor_soft: float = 12.0
) -> torch.Tensor:
    """
    """
    this_ca_pos = pred_coords[:-1, 1]
    this_c_pos = pred_coords[:-1, 2]
    next_n_pos = pred_coords[1:, 0]
    next_ca_pos = pred_coords[1:, 1]
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )
    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    gt_length = 1.329
    gt_stddev = 0.014
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    c_n_loss = torch.mean(c_n_loss_per_residue)

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = -0.4473
    gt_stddev = 0.014
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    ca_c_n_loss = torch.mean(ca_c_n_loss_per_residue)

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = -0.5203
    gt_stddev = 0.03
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    c_n_ca_loss = torch.mean(c_n_ca_loss_per_residue)

    loss = c_n_loss + ca_c_n_loss + c_n_ca_loss
    return loss

def violation_loss(
    pred_coords: torch.Tensor,
    sep: int,
) -> torch.Tensor:
    """
    """
    pred_1 = pred_coords[:sep]
    pred_2 = pred_coords[sep:]
    loss_1 = between_residue_bond_loss(pred_1)
    loss_2 = between_residue_bond_loss(pred_2)
    loss = (loss_1 + loss_2) * 0.5
    return loss
