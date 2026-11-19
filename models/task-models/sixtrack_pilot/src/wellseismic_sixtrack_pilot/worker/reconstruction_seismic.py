"""Original P21 seismic attributes at explicitly aligned grid locations.

The caller owns geometry/time alignment. No targets, well-property labels or
feature scaling are read here. This preserves build_dataset's nearest-sample,
five-sample RMS and central vertical difference arithmetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio


def sample_attributes(
    seismic_path: Path, inline: np.ndarray, crossline: np.ndarray,
    twt_ms: np.ndarray, index: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    inline, crossline = np.asarray(inline), np.asarray(crossline)
    twt_ms = np.asarray(twt_ms)
    if inline.ndim != 1 or not len(inline) or crossline.shape != inline.shape or twt_ms.shape != inline.shape:
        raise ValueError("Seismic sampling requires aligned non-empty location arrays")
    if not np.issubdtype(inline.dtype, np.integer) or not np.issubdtype(crossline.dtype, np.integer):
        raise ValueError("Inline/crossline must contain integer grid addresses")
    if not np.isfinite(twt_ms).all():
        raise ValueError("TWT samples must be finite")
    if np.any(inline < int(index["il_min"])) or np.any(inline > int(index["il_max"])) or np.any(crossline < int(index["xl_min"])) or np.any(crossline > int(index["xl_max"])):
        raise ValueError("Seismic location is outside the declared survey")
    samples_ms = np.asarray(index["samples_ms"], dtype=np.float64)
    if samples_ms.ndim != 1 or len(samples_ms) < 3 or not np.isfinite(samples_ms).all() or not np.all(np.diff(samples_ms) > 0):
        raise ValueError("Seismic sample times must be finite and strictly increasing")
    time_index = np.searchsorted(samples_ms, twt_ms)
    time_index = np.clip(time_index, 1, samples_ms.size - 2)
    left = np.abs(samples_ms[time_index - 1] - twt_ms)
    right = np.abs(samples_ms[time_index] - twt_ms)
    time_index = np.where(left < right, time_index - 1, time_index).astype(np.int32)
    if np.any(time_index == 0):
        raise ValueError("Selected sample lacks a preceding sample for central vertical gradient")
    trace_index = (inline - int(index["il_min"])) * int(index["n_xl"]) + (crossline - int(index["xl_min"]))
    unique_trace, inverse = np.unique(trace_index, return_inverse=True)
    traces = np.empty((unique_trace.size, samples_ms.size), dtype=np.float32)
    with segyio.open(str(seismic_path), "r", ignore_geometry=True) as segy:
        if segy.tracecount != int(index["n_traces"]) or len(segy.samples) != len(samples_ms):
            raise ValueError("SEG-Y geometry differs from the declared seismic index")
        for row, trace_id in enumerate(unique_trace):
            traces[row] = np.asarray(segy.trace[int(trace_id)], dtype=np.float32)
    offsets = np.arange(-2, 3, dtype=np.int32)
    windows = traces[inverse[:, None], np.clip(time_index[:, None] + offsets, 0, samples_ms.size - 1)]
    amplitude = traces[inverse, time_index]
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    gradient = traces[inverse, time_index + 1] - traces[inverse, time_index - 1]
    attributes = np.column_stack([amplitude, rms, gradient])
    if not np.isfinite(attributes).all():
        raise ValueError("Non-finite seismic attributes encountered")
    return attributes, {
        "rows": len(inline), "unique_traces_read": len(unique_trace),
        "time_index_range": [int(time_index.min()), int(time_index.max())],
        "twt_ms_range": [float(twt_ms.min()), float(twt_ms.max())],
        "attribute_order": ["amplitude", "local_rms", "vertical_gradient"],
        "query_targets_consumed": False,
    }
