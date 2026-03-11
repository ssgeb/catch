"""
Mask utilities adapted for query-based instance segmentation.
"""

import torch
import torch.nn.functional as F


def point_sample(input, point_coords, align_corners=False):
    """
    Sample features at [0, 1] normalized point coordinates.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(
        input,
        2.0 * point_coords - 1.0,
        mode="bilinear",
        align_corners=align_corners,
    )

    if add_dim:
        output = output.squeeze(3)

    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, 1 - targets)
    return loss / hw


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits: torch.Tensor):
    """
    Higher uncertainty corresponds to values closer to zero.
    """
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
):
    if coarse_logits.shape[0] == 0:
        return coarse_logits.new_empty((0, num_points, 2))

    num_boxes = coarse_logits.shape[0]
    num_sampled = max(int(num_points * oversample_ratio), num_points)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    uncertainties = uncertainty_func(point_logits).squeeze(1)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(uncertainties, k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, device=coarse_logits.device)[:, None]
    idx = (idx + shift).reshape(-1)
    point_coords = point_coords.reshape(-1, 2)[idx].reshape(num_boxes, num_uncertain_points, 2)

    if num_random_points > 0:
        random_coords = torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)
        point_coords = torch.cat([point_coords, random_coords], dim=1)

    return point_coords
