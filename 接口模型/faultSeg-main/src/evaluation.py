"""Reusable histogram-based evaluation utilities for binary fault segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METRIC_NAMES = ("precision", "recall", "dice", "iou", "specificity", "youden", "accuracy")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    """Compute common binary segmentation metrics from a confusion matrix."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    total = tp + fp + fn + tn
    return {
        "precision": precision,
        "recall": recall,
        "dice": 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "specificity": specificity,
        "youden": recall + specificity - 1.0,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


@dataclass
class ProbabilityHistogram:
    """Positive/negative probability histograms with threshold curve helpers."""

    bins: int = 1000

    def __post_init__(self) -> None:
        if self.bins < 20:
            raise ValueError("bins must be at least 20")
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, probability: np.ndarray, truth: np.ndarray) -> None:
        probability = np.asarray(probability)
        truth = np.asarray(truth, dtype=bool)
        if probability.shape != truth.shape:
            raise ValueError("probability and truth must have the same shape")
        if not np.all(np.isfinite(probability)):
            raise ValueError("probability contains non-finite values")
        index = np.minimum((np.clip(probability, 0.0, 1.0) * self.bins).astype(np.int32), self.bins - 1)
        self.positive += np.bincount(index[truth], minlength=self.bins)
        self.negative += np.bincount(index[~truth], minlength=self.bins)

    def merge(self, other: "ProbabilityHistogram") -> None:
        if other.bins != self.bins:
            raise ValueError("cannot merge histograms with different bin counts")
        self.positive += other.positive
        self.negative += other.negative

    def curves(self) -> dict[str, np.ndarray]:
        tp = np.cumsum(self.positive[::-1], dtype=np.float64)[::-1]
        fp = np.cumsum(self.negative[::-1], dtype=np.float64)[::-1]
        positives = float(self.positive.sum())
        negatives = float(self.negative.sum())
        fn = positives - tp
        tn = negatives - fp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        return {
            "threshold": np.arange(self.bins, dtype=np.float64) / self.bins,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "dice": _safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
            "iou": _safe_divide(tp, tp + fp + fn),
            "specificity": specificity,
            "youden": recall + specificity - 1.0,
            "accuracy": _safe_divide(tp + tn, tp + fp + fn + tn),
        }

    def best(self, metric: str = "dice") -> tuple[float, dict[str, float]]:
        if metric not in ("dice", "iou", "youden"):
            raise ValueError("metric must be dice, iou, or youden")
        curves = self.curves()
        index = int(np.argmax(curves[metric]))
        return float(curves["threshold"][index]), self.metrics_at_index(index)

    def index_for_threshold(self, threshold: float) -> int:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return min(int(np.ceil(threshold * self.bins)), self.bins - 1)

    def metrics_at(self, threshold: float) -> dict[str, float]:
        return self.metrics_at_index(self.index_for_threshold(threshold))

    def metrics_at_index(self, index: int) -> dict[str, float]:
        curves = self.curves()
        result = {name: float(curves[name][index]) for name in METRIC_NAMES}
        result.update({name: int(curves[name][index]) for name in ("tp", "fp", "fn", "tn")})
        return result

