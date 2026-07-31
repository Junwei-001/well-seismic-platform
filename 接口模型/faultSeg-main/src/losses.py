"""Loss functions for sparse voxel-wise fault segmentation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, target.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * soft_dice_loss(
        logits, target
    )


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    false_positive_weight: float = 0.7,
    false_negative_weight: float = 0.3,
) -> torch.Tensor:
    """Sparse segmentation loss with extra weight on false positives."""
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, target.ndim))
    true_positive = (probability * target).sum(dim=dims)
    false_positive = (probability * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probability) * target).sum(dim=dims)
    score = (true_positive + 1.0) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + 1.0
    )
    return (1.0 - score).mean()


def conservative_bce_tversky_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * tversky_loss(
        logits, target
    )


def balanced_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.sum()
    total = target.new_tensor(target.numel())
    if positive.item() == 0:
        return logits.sum() * 0.0
    beta = (total - positive) / total
    positive_weight = beta / (1.0 - beta)
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight, reduction="mean"
    ) * (1.0 - beta)
