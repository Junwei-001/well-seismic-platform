"""Shared device selection and overlap-tiled inference utilities."""

from __future__ import annotations

import itertools
from typing import Callable, Tuple

import numpy as np
import torch

from .data import normalize_seismic


ProgressCallback = Callable[[int, int, Tuple[int, int, int]], None]


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def patch_starts(length: int, patch: int, overlap: int) -> list[int]:
    if patch > length:
        raise ValueError(f"patch dimension {patch} exceeds volume dimension {length}")
    if patch % 8:
        raise ValueError("each patch dimension must be divisible by 8")
    stride = patch - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch size")
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def _normalize_patch(array: np.ndarray, invalid_value: float | None) -> np.ndarray:
    valid = np.isfinite(array)
    if invalid_value is not None and np.isfinite(invalid_value):
        valid &= array != invalid_value
    if valid.all():
        return normalize_seismic(array)
    normalized = np.zeros(array.shape, dtype=np.float32)
    if valid.any():
        values = array[valid]
        std = float(values.std())
        if std > 0.0 and np.isfinite(std):
            normalized[valid] = (values - float(values.mean())) / std
    return normalized


def predict_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: tuple[int, int, int] | None,
    overlap: tuple[int, int, int] = (32, 32, 32),
    amp_enabled: bool = True,
    *,
    normalize_patches: bool = False,
    weighted_blending: bool = False,
    invalid_value: float | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Predict a complete ``[Z,Y,X]`` volume with optional overlap blending."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    if patch_size is None:
        if any(size % 8 for size in volume.shape):
            raise ValueError("volume dimensions must be divisible by 8")
        patch_size = volume.shape
        overlap = (0, 0, 0)
    grids = [patch_starts(n, p, o) for n, p, o in zip(volume.shape, patch_size, overlap)]
    probabilities = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)
    total = int(np.prod([len(grid) for grid in grids]))
    weight = np.ones(patch_size, dtype=np.float32)
    if weighted_blending and total > 1:
        vectors = [
            np.maximum(np.hanning(size + 2)[1:-1], 0.1).astype(np.float32)
            for size in patch_size
        ]
        weight = (
            vectors[0][:, None, None]
            * vectors[1][None, :, None]
            * vectors[2][None, None, :]
        )
    with torch.inference_mode():
        for index, origin in enumerate(itertools.product(*grids), start=1):
            slices = tuple(
                slice(start, start + size) for start, size in zip(origin, patch_size)
            )
            patch_array = volume[slices]
            if normalize_patches:
                patch_array = _normalize_patch(patch_array, invalid_value)
            patch = torch.from_numpy(np.ascontiguousarray(patch_array)[None, None]).to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = torch.sigmoid(model(patch))
            result = prediction[0, 0].float().cpu().numpy()
            probabilities[slices] += result * weight
            counts[slices] += weight
            if progress is not None:
                progress(index, total, origin)
    if np.any(counts == 0):
        raise RuntimeError("some output voxels were not covered by a valid patch")
    return probabilities / counts


def print_progress(index: int, total: int, origin: tuple[int, int, int]) -> None:
    print(f"patch {index}/{total} origin={origin}", flush=True)
