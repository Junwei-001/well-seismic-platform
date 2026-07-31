"""Unified readers for seismic volumes used by inference and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .cbvs import CBVSVolume
from .data import Shape3D, read_raw, to_model_order


@dataclass
class VolumeData:
    data: np.ndarray
    valid_traces: np.ndarray
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_format(path: str | Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested.lower()
    suffix = Path(path).suffix.lower()
    formats = {
        ".tif": "tiff", ".tiff": "tiff", ".cbvs": "cbvs",
        ".dat": "dat", ".sgy": "sgy", ".segy": "sgy",
    }
    if suffix not in formats:
        raise ValueError(f"cannot infer volume format from {path}; use --format")
    return formats[suffix]


def _read_sgy(path: Path, inline_byte: int, crossline_byte: int) -> VolumeData:
    try:
        import segyio
    except ImportError as error:
        raise ImportError("SGY support requires segyio: pip install segyio") from error

    with segyio.open(str(path), "r", ignore_geometry=True) as stream:
        inlines = np.asarray(stream.attributes(inline_byte)[:], dtype=np.int64)
        crosslines = np.asarray(stream.attributes(crossline_byte)[:], dtype=np.int64)
        inline_values = np.unique(inlines)
        crossline_values = np.unique(crosslines)
        inline_index = np.searchsorted(inline_values, inlines)
        crossline_index = np.searchsorted(crossline_values, crosslines)
        lookup = np.full((len(inline_values), len(crossline_values)), -1, dtype=np.int32)
        lookup[inline_index, crossline_index] = np.arange(stream.tracecount, dtype=np.int32)
        traces = np.asarray(stream.trace.raw[:], dtype=np.float32)
        volume = np.full(
            (traces.shape[1], len(inline_values), len(crossline_values)),
            np.nan,
            dtype=np.float32,
        )
        volume[:, inline_index, crossline_index] = traces.T
        sample_values = np.asarray(stream.samples, dtype=np.float64)
        sample_step = float(np.median(np.diff(sample_values))) if len(sample_values) > 1 else None
        metadata = {
            "inline_byte": inline_byte,
            "crossline_byte": crossline_byte,
            "inline_values": inline_values,
            "crossline_values": crossline_values,
            "trace_inline_index": inline_index,
            "trace_crossline_index": crossline_index,
            "sample_start": float(sample_values[0]),
            "sample_step": sample_step,
            "trace_count": int(stream.tracecount),
        }
    return VolumeData(volume, lookup >= 0, "sgy", metadata)


def read_volume(
    path: str | Path,
    *,
    format: str = "auto",
    shape: Shape3D | None = None,
    component: int | str = 0,
    inline_byte: int = 189,
    crossline_byte: int = 193,
) -> VolumeData:
    """Read a volume into model order ``[Z, inline, crossline]``."""
    path = Path(path)
    volume_format = detect_format(path, format)
    if volume_format == "tiff":
        data = np.asarray(tifffile.imread(path), dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"expected 3D TIFF, got shape {data.shape}")
        valid = np.any(np.isfinite(data) & (data != 0), axis=0)
        return VolumeData(data, valid, volume_format)
    if volume_format == "cbvs":
        source = CBVSVolume(path)
        data, valid = source.read_crop(slice(None), slice(None), slice(None), component)
        item = source.components[int(component)] if isinstance(component, int) else next(
            value for value in source.components if value.name == component
        )
        return VolumeData(data, valid, volume_format, {
            "component": item.name,
            "sample_start_ms": item.start_ms,
            "sample_step_ms": item.step_ms,
        })
    if volume_format == "dat":
        if shape is None:
            raise ValueError("--shape is required for headerless DAT input")
        data = to_model_order(read_raw(path, shape))
        return VolumeData(data, np.ones(data.shape[1:], dtype=bool), volume_format, {
            "storage_shape": list(shape), "storage_order": "n3,n2,n1",
        })
    if volume_format == "sgy":
        return _read_sgy(path, inline_byte, crossline_byte)
    raise ValueError(f"unsupported volume format: {volume_format}")


def write_sgy_like(source_path: str | Path, output_path: str | Path, volume: VolumeData) -> None:
    """Copy an SGY and replace traces from a processed model-order volume."""
    import shutil

    try:
        import segyio
    except ImportError as error:
        raise ImportError("SGY support requires segyio: pip install segyio") from error
    if volume.format != "sgy":
        raise ValueError("write_sgy_like requires metadata from an SGY volume")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    yi = volume.metadata["trace_inline_index"]
    xi = volume.metadata["trace_crossline_index"]
    with segyio.open(str(output_path), "r+", ignore_geometry=True) as stream:
        for index in range(stream.tracecount):
            stream.trace[index] = np.asarray(volume.data[:, yi[index], xi[index]], dtype=np.float32)

