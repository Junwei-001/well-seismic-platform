"""Headless QC visualizations for seismic fault probabilities and masks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


def _plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _display_range(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1, 99))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def save_orthogonal_preview(
    seismic: np.ndarray,
    probability: np.ndarray,
    path: str | Path,
    threshold: float,
) -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    indices = tuple(size // 2 for size in seismic.shape)
    names = ("Z", "Y", "X")
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    probability_image = None
    for column, (axis, index, name) in enumerate(zip(range(3), indices, names)):
        seismic_plane = _plane(seismic, axis, index)
        probability_plane = _plane(probability, axis, index)
        low, high = _display_range(seismic_plane)
        axes[0, column].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[0, column].set_title(f"Seismic · {name}={index}")
        probability_image = axes[1, column].imshow(
            probability_plane, cmap="magma", vmin=0, vmax=1, aspect="auto"
        )
        axes[1, column].set_title(f"Fault probability · {name}={index}")
        axes[2, column].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
        mask = np.ma.masked_where(probability_plane < threshold, probability_plane)
        axes[2, column].imshow(mask, cmap="autumn", vmin=threshold, vmax=1, alpha=0.72, aspect="auto")
        axes[2, column].set_title(f"Segmentation · p ≥ {threshold:.4g} · {name}={index}")
        for row in range(3):
            axes[row, column].axis("off")
    if probability_image is not None:
        figure.colorbar(probability_image, ax=axes[1:, :], shrink=0.75, label="Fault probability")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_threshold_sweep(
    seismic: np.ndarray,
    probability: np.ndarray,
    path: str | Path,
    thresholds: Sequence[float],
) -> None:
    """Compare several thresholds on all three central orthogonal planes."""
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds:
        raise ValueError("threshold sweep requires at least one threshold")
    path = Path(path)
    indices = tuple(size // 2 for size in seismic.shape)
    names = ("Time slice", "Inline section", "Crossline section")
    columns = 2 + len(thresholds)
    figure, axes = plt.subplots(3, columns, figsize=(3.6 * columns, 12), constrained_layout=True)
    for row, (axis, index, name) in enumerate(zip(range(3), indices, names)):
        seismic_plane = _plane(seismic, axis, index)
        probability_plane = _plane(probability, axis, index)
        low, high = _display_range(seismic_plane)
        axes[row, 0].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 0].set_title(f"{name}\nseismic")
        axes[row, 1].imshow(probability_plane, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[row, 1].set_title(f"{name}\nprobability")
        for column, threshold in enumerate(thresholds, start=2):
            axes[row, column].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
            binary = np.ma.masked_where(probability_plane < threshold, probability_plane)
            axes[row, column].imshow(
                binary, cmap="autumn", vmin=threshold, vmax=1, alpha=0.75, aspect="auto"
            )
            fraction = float((probability_plane >= threshold).mean())
            axes[row, column].set_title(f"p ≥ {threshold:.4g}\nplane positive={fraction:.2%}")
        for column in range(columns):
            axes[row, column].axis("off")
    figure.suptitle("Fault segmentation threshold sweep", fontsize=17)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_labeled_preview(
    seismic: np.ndarray,
    probability: np.ndarray,
    truth: np.ndarray,
    path: str | Path,
    threshold: float,
) -> None:
    """Save central input/label/probability/overlay planes for labeled inference."""
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    indices = tuple(int(np.argmax(np.moveaxis(truth, axis, 0).sum(axis=(1, 2)))) for axis in range(3))
    names = ("Time", "Inline", "Crossline")
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for row, (axis, index, name) in enumerate(zip(range(3), indices, names)):
        seismic_plane = _plane(seismic, axis, index)
        probability_plane = _plane(probability, axis, index)
        truth_plane = _plane(truth, axis, index)
        low, high = _display_range(seismic_plane)
        axes[row, 0].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 0].set_title(f"{name} {index} · seismic")
        axes[row, 1].imshow(truth_plane, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(probability_plane, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[row, 2].set_title("Probability")
        axes[row, 3].imshow(seismic_plane, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 3].imshow(
            np.ma.masked_where(probability_plane < threshold, probability_plane),
            cmap="autumn", vmin=threshold, vmax=1, alpha=0.7, aspect="auto",
        )
        axes[row, 3].contour(truth_plane, levels=[0.5], colors="cyan", linewidths=0.7)
        axes[row, 3].set_title(f"p ≥ {threshold:.3f} · cyan=truth")
        for axis_object in axes[row]:
            axis_object.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
