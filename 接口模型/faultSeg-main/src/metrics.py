"""Streaming segmentation metrics accumulated over one or more volumes."""

from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.counts: torch.Tensor | None = None

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = torch.sigmoid(logits.detach()) >= self.threshold
        truth = target >= 0.5
        counts = torch.stack(
            (
                (prediction & truth).sum(),
                (prediction & ~truth).sum(),
                (~prediction & truth).sum(),
                (~prediction & ~truth).sum(),
            )
        )
        self.counts = counts if self.counts is None else self.counts + counts

    def compute(self) -> dict[str, float]:
        if self.counts is None:
            raise RuntimeError("no predictions were accumulated")
        tp, fp, fn, tn = (int(value) for value in self.counts.cpu().tolist())
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return {
            "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
            "precision": precision,
            "recall": recall,
            "dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
        }
