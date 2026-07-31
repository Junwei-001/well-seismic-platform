"""Threshold selection and probability-volume summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    value: float
    method: str


def validate_threshold(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability threshold must be between 0 and 1")
    return value


def otsu_threshold(probability: np.ndarray, bins: int = 512) -> float:
    """Return an unsupervised Otsu threshold for finite probabilities."""
    values = np.asarray(probability, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot select a threshold from an empty probability volume")
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_left = np.cumsum(histogram, dtype=np.float64)
    weight_right = values.size - weight_left
    mean_left = np.cumsum(histogram * centers, dtype=np.float64)
    total_mean = mean_left[-1]
    valid = (weight_left > 0) & (weight_right > 0)
    between = np.full(histogram.shape, -np.inf, dtype=np.float64)
    left_mean = np.zeros(histogram.shape, dtype=np.float64)
    right_mean = np.zeros(histogram.shape, dtype=np.float64)
    left_mean[valid] = mean_left[valid] / weight_left[valid]
    right_mean[valid] = (total_mean - mean_left[valid]) / weight_right[valid]
    between[valid] = weight_left[valid] * weight_right[valid] * (
        left_mean[valid] - right_mean[valid]
    ) ** 2
    return validate_threshold(float(centers[int(np.argmax(between))]))


def select_threshold(
    spec: str | float | None,
    probability: np.ndarray,
    profile_default: float,
) -> ThresholdSelection:
    """Resolve fixed, profile, Otsu, or quantile threshold specifications."""
    if spec is None or str(spec).lower() in {"profile", "default"}:
        return ThresholdSelection(validate_threshold(profile_default), "profile")
    text = str(spec).strip().lower()
    if text == "otsu":
        return ThresholdSelection(otsu_threshold(probability), "otsu")
    if text.startswith("quantile:"):
        quantile = float(text.split(":", 1)[1])
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile threshold must use a fraction between 0 and 1")
        values = np.asarray(probability)
        value = float(np.quantile(values[np.isfinite(values)], quantile))
        return ThresholdSelection(validate_threshold(value), f"quantile:{quantile:g}")
    return ThresholdSelection(validate_threshold(float(text)), "fixed")


def probability_summary(probability: np.ndarray, threshold: float) -> dict[str, object]:
    values = np.asarray(probability, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("probability volume contains no finite values")
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "percentiles": {
            str(percentile): float(np.percentile(finite, percentile))
            for percentile in (50, 90, 95, 98, 99, 99.5)
        },
        "threshold": float(threshold),
        "positive_fraction": float((finite >= threshold).mean()),
    }
