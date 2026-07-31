"""Fault-oriented seismic preprocessing filters."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def normalized_trace_similarity(volume: np.ndarray, half_window: int = 7) -> np.ndarray:
    """Similarity of opposite crossline neighbours over a vertical window."""
    if half_window < 1:
        raise ValueError("half_window must be positive")
    source = np.asarray(volume, dtype=np.float32)
    filled = np.nan_to_num(source, nan=0.0)
    left = np.empty_like(filled)
    right = np.empty_like(filled)
    left[:, :, 1:] = filled[:, :, :-1]
    left[:, :, 0] = filled[:, :, 0]
    right[:, :, :-1] = filled[:, :, 1:]
    right[:, :, -1] = filled[:, :, -1]
    size = 2 * half_window + 1
    mean_left = uniform_filter1d(left, size=size, axis=0, mode="nearest")
    mean_right = uniform_filter1d(right, size=size, axis=0, mode="nearest")
    covariance = uniform_filter1d(left * right, size=size, axis=0, mode="nearest") - mean_left * mean_right
    variance_left = uniform_filter1d(left * left, size=size, axis=0, mode="nearest") - mean_left * mean_left
    variance_right = uniform_filter1d(right * right, size=size, axis=0, mode="nearest") - mean_right * mean_right
    denominator = np.sqrt(np.maximum(variance_left * variance_right, 0.0))
    similarity = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
    return np.clip(similarity, -1.0, 1.0)


def fault_enhancement_filter(
    volume: np.ndarray,
    *,
    half_window: int = 7,
    similarity_gate: float = 0.85,
    lateral_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the non-steered form of OpendTect's Fault Enhancing Filter.

    The original F3 graph uses dip steering. This implementation preserves its
    similarity/median/minimum-position/gate sequence and is the appropriate
    approximation when a standalone SGY does not contain a steering cube.
    """
    if not -1.0 <= similarity_gate <= 1.0:
        raise ValueError("similarity_gate must be in [-1, 1]")
    if lateral_radius < 1:
        raise ValueError("lateral_radius must be positive")
    source = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(source)
    filled = np.nan_to_num(source, nan=0.0)
    similarity = normalized_trace_similarity(filled, half_window)
    window = 2 * lateral_radius + 1
    local_median = median_filter(filled, size=(1, window, window), mode="nearest")
    minimum = np.full(source.shape, np.inf, dtype=np.float32)
    diffusion = np.array(local_median, copy=True)
    padded_similarity = np.pad(similarity, ((0, 0), (lateral_radius, lateral_radius), (lateral_radius, lateral_radius)), mode="edge")
    padded_median = np.pad(local_median, ((0, 0), (lateral_radius, lateral_radius), (lateral_radius, lateral_radius)), mode="edge")
    nz, ny, nx = source.shape
    for dy in range(window):
        for dx in range(window):
            candidate = padded_similarity[:, dy:dy + ny, dx:dx + nx]
            update = candidate < minimum
            minimum[update] = candidate[update]
            values = padded_median[:, dy:dy + ny, dx:dx + nx]
            diffusion[update] = values[update]
    enhanced = np.where(similarity < similarity_gate, diffusion, local_median).astype(np.float32)
    enhanced[~valid] = np.nan
    return enhanced, similarity

