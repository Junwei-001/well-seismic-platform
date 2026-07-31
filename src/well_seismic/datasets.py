from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from .output_schema import sample_to_internal


class JsonlMultimodalDataset:
    """Lazy JSONL dataset usable by traditional ML and framework adapters."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return sample_to_internal(json.loads(handle.readline().decode("utf-8")))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def to_numpy(self, curve_order: list[str], require_seismic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        well_rows, seismic_rows, quality_rows = [], [], []
        for sample in self:
            seismic = sample.get("seismic_window")
            if require_seismic and seismic is None:
                continue
            well_rows.append([sample.get("well_features", {}).get(curve, np.nan) for curve in curve_order])
            seismic_rows.append(seismic or [])
            quality_rows.append([sample.get("horizontal_confidence", 0.0), sample.get("vertical_confidence", 0.0)])
        max_seismic = max((len(row) for row in seismic_rows), default=0)
        seismic_array = np.full((len(seismic_rows), max_seismic), np.nan, dtype=np.float32)
        for i, row in enumerate(seismic_rows):
            seismic_array[i, :len(row)] = row
        return np.asarray(well_rows, dtype=np.float32), seismic_array, np.asarray(quality_rows, dtype=np.float32)


class TorchDatasetAdapter:
    """Optional adapter that imports torch only when explicitly requested."""
    def __init__(self, dataset: JsonlMultimodalDataset, curve_order: list[str]):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is optional; install it to use TorchDatasetAdapter") from exc
        self.torch = torch
        self.dataset = dataset
        self.curve_order = curve_order

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        well = [sample.get("well_features", {}).get(curve, float("nan")) for curve in self.curve_order]
        seismic = sample.get("seismic_window") or []
        return {
            "well": self.torch.tensor(well, dtype=self.torch.float32),
            "seismic": self.torch.tensor(seismic, dtype=self.torch.float32),
            "horizontal_confidence": self.torch.tensor(sample.get("horizontal_confidence", 0.0)),
            "vertical_confidence": self.torch.tensor(sample.get("vertical_confidence", 0.0)),
            "metadata": sample,
        }
