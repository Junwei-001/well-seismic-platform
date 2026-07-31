"""Seismic-volume I/O and PyTorch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


Shape3D = Tuple[int, int, int]


def parse_shape(text: str) -> Shape3D:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("shape must contain three positive integers, e.g. 128,128,128")
    return values  # type: ignore[return-value]


def read_raw(path: str | Path, storage_shape: Shape3D) -> np.ndarray:
    path = Path(path)
    array = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(storage_shape))
    if array.size != expected:
        raise ValueError(
            f"{path} contains {array.size} float32 values; expected {expected} "
            f"for shape {storage_shape}"
        )
    return array.reshape(storage_shape)


def to_model_order(array: np.ndarray) -> np.ndarray:
    # Seismic files use [n3, n2, n1]; Conv3d consumes [D, H, W].
    return np.ascontiguousarray(array.transpose())


def normalize_seismic(array: np.ndarray) -> np.ndarray:
    # Normalize each survey volume independently.
    mean = float(array.mean())
    std = float(array.std())
    if not np.isfinite(std) or std == 0.0:
        raise ValueError("seismic volume has zero or invalid standard deviation")
    return np.asarray((array - mean) / std, dtype=np.float32)


class FaultVolumeDataset(Dataset):
    """Paired seismic/fault `.dat` volumes."""

    def __init__(
        self,
        seismic_dir: str | Path,
        fault_dir: str | Path,
        ids: Sequence[int],
        storage_shape: Shape3D = (128, 128, 128),
        augment: bool = False,
    ) -> None:
        self.seismic_dir = Path(seismic_dir)
        self.fault_dir = Path(fault_dir)
        self.ids = list(ids)
        self.storage_shape = storage_shape
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume_id = self.ids[index]
        seismic = to_model_order(
            read_raw(self.seismic_dir / f"{volume_id}.dat", self.storage_shape)
        )
        fault = to_model_order(
            read_raw(self.fault_dir / f"{volume_id}.dat", self.storage_shape)
        )
        seismic = normalize_seismic(seismic)
        if self.augment:
            # Synthetic volumes have no privileged horizontal direction.
            for axis in (0, 1, 2):
                if torch.rand(()) < 0.5:
                    seismic = np.flip(seismic, axis=axis)
                    fault = np.flip(fault, axis=axis)
            seismic = np.ascontiguousarray(seismic)
            fault = np.ascontiguousarray(fault)
        return torch.from_numpy(seismic[None]), torch.from_numpy(fault[None])


def discover_ids(seismic_dir: str | Path, fault_dir: str | Path) -> list[int]:
    seismic = {int(p.stem) for p in Path(seismic_dir).glob("*.dat") if p.stem.isdigit()}
    fault = {int(p.stem) for p in Path(fault_dir).glob("*.dat") if p.stem.isdigit()}
    return sorted(seismic & fault)
