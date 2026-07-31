"""FaultSeg-first seismic volume preparation.

The public contract deliberately follows FaultSeg rather than the legacy
single-trace sample representation: model-order arrays are ``[Z, Y, X]`` and
model tensors are ``[N, 1, Z, Y, X]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .io.segy import SegyReader


Shape3D = tuple[int, int, int]


@dataclass(frozen=True)
class FaultSegInputSpec:
    patch_size: Shape3D = (32, 32, 32)
    overlap: Shape3D = (8, 8, 8)
    patch_multiple: int = 8
    threshold: float = 0.5

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "FaultSegInputSpec":
        values = (config or {}).get("faultseg", config or {})
        return cls(
            patch_size=_shape(values.get("patch_size", (32, 32, 32))),
            overlap=_shape(values.get("overlap", (8, 8, 8))),
            patch_multiple=int(values.get("patch_multiple", 8)),
            threshold=float(values.get("threshold", 0.5)),
        ).validated()

    def validated(self) -> "FaultSegInputSpec":
        if self.patch_multiple <= 0:
            raise ValueError("patch_multiple must be positive")
        if any(size <= 0 or size % self.patch_multiple for size in self.patch_size):
            raise ValueError(f"FaultSeg patch dimensions must be positive multiples of {self.patch_multiple}")
        if any(value < 0 or value >= size for value, size in zip(self.overlap, self.patch_size)):
            raise ValueError("FaultSeg overlap must be non-negative and smaller than patch size")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("FaultSeg threshold must be between 0 and 1")
        return self


@dataclass
class FaultSegVolume:
    data: np.ndarray
    valid_traces: np.ndarray
    inline_values: np.ndarray
    crossline_values: np.ndarray
    sample_slice: slice
    provenance: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> np.ndarray:
        """Return FaultSeg-compatible patch z-score data as contiguous float32."""
        array = np.asarray(self.data, dtype=np.float32)
        valid = np.isfinite(array)
        values = array[valid]
        if values.size == 0:
            raise ValueError("FaultSeg volume contains no finite seismic samples")
        mean = float(values.mean())
        std = float(values.std())
        result = np.zeros(array.shape, dtype=np.float32)
        if np.isfinite(std) and std > 0.0:
            result[valid] = (values - mean) / std
        return np.ascontiguousarray(result)

    def tensor(self) -> np.ndarray:
        """Return the framework-neutral FaultSeg tensor ``[1,1,Z,Y,X]``."""
        return self.normalized()[None, None]


def build_faultseg_volume(
    reader: SegyReader,
    *,
    sample_slice: slice,
    inline_slice: slice,
    crossline_slice: slice,
) -> FaultSegVolume:
    """Build a regular FaultSeg-order crop from the system SEG-Y reader."""
    geometry = reader.geometry or reader.inspect()
    if geometry.inline is None or geometry.crossline is None:
        raise ValueError("FaultSeg requires resolved 3D inline and crossline geometry")

    inline_all = np.unique(geometry.inline)
    crossline_all = np.unique(geometry.crossline)
    inline_values = inline_all[inline_slice]
    crossline_values = crossline_all[crossline_slice]
    sample_indices = np.arange(geometry.samples_per_trace)[sample_slice]
    if not len(sample_indices) or not len(inline_values) or not len(crossline_values):
        raise ValueError("FaultSeg crop is empty")

    lookup = {
        (int(inline), int(crossline)): index
        for index, (inline, crossline) in enumerate(zip(geometry.inline, geometry.crossline))
    }
    data = np.full(
        (len(sample_indices), len(inline_values), len(crossline_values)),
        np.nan,
        dtype=np.float32,
    )
    valid_traces = np.zeros((len(inline_values), len(crossline_values)), dtype=bool)
    for yi, inline in enumerate(inline_values):
        for xi, crossline in enumerate(crossline_values):
            trace_index = lookup.get((int(inline), int(crossline)))
            if trace_index is None:
                continue
            values = reader.read_trace(trace_index, sample_slice)
            if len(values) != len(sample_indices):
                raise ValueError("SEG-Y trace crop length does not match requested FaultSeg Z dimension")
            data[:, yi, xi] = values
            valid_traces[yi, xi] = True

    return FaultSegVolume(
        data=data,
        valid_traces=valid_traces,
        inline_values=inline_values,
        crossline_values=crossline_values,
        sample_slice=sample_slice,
        provenance={
            "source": str(reader.path),
            "model_order": ["Z", "INLINE", "CROSSLINE"],
            "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
            "geometry_profile": geometry.profile,
            "geometry_confidence": geometry.confidence,
            "geometry_issues": list(geometry.issues),
        },
    )


def patch_starts(length: int, patch: int, overlap: int) -> list[int]:
    if patch > length:
        raise ValueError(f"patch dimension {patch} exceeds volume dimension {length}")
    stride = patch - overlap
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def iter_faultseg_patches(
    volume: FaultSegVolume,
    spec: FaultSegInputSpec,
) -> Iterator[tuple[Shape3D, np.ndarray]]:
    """Yield normalized ``[1,1,Z,Y,X]`` patches and their model-order origins."""
    spec = spec.validated()
    grids = [patch_starts(n, p, o) for n, p, o in zip(volume.data.shape, spec.patch_size, spec.overlap)]
    for z in grids[0]:
        for y in grids[1]:
            for x in grids[2]:
                pz, py, px = spec.patch_size
                patch = FaultSegVolume(
                    volume.data[z:z + pz, y:y + py, x:x + px],
                    volume.valid_traces[y:y + py, x:x + px],
                    volume.inline_values[y:y + py],
                    volume.crossline_values[x:x + px],
                    slice(z, z + pz),
                    volume.provenance,
                )
                yield (z, y, x), patch.tensor()


def _shape(value: Any) -> Shape3D:
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError("expected three FaultSeg dimensions")
    return result  # type: ignore[return-value]
