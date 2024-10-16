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