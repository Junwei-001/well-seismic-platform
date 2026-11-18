"""Isolated worker: P78-retained ``fault_local_logistic_audited_v2`` on a SEG-Y box.

Runs inside the dedicated model interpreter (``WELL_SEISMIC_SIXTRACK_PYTHON``),
never inside the platform process.  It imports only numpy/scipy/segyio and the
stdlib helpers of this worker package, so it cannot reach platform state.

Numerical contract (reproduces the competition project exactly):

* traces are addressed by the INLINE_3D / CROSSLINE_3D headers, the same
  trace map the P30 development-box builder uses (last duplicate wins);
* raw float32 amplitudes, no normalisation, array axes ``[Z, INLINE, CROSSLINE]``;
* each inline slice becomes a ``[1, 1, CROSSLINE, Z]`` patch, five local
  features are computed (``FaultLocalLogistic._features``), and the portable
  SGD-logistic coefficients give ``sigmoid(clip(features @ coef + b, ±80))`` in
  float64, stored as float32 — the same arithmetic as ``decision_batch`` /
  ``predict_batch`` of the canonical class.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import segyio
from scipy.ndimage import sobel, uniform_filter

from ..manifest import atomic_json, fail, load_request, sha256_file, verify_payload

AXES = ("Z", "INLINE", "CROSSLINE")
MIN_BOX_EDGE = 3
MAX_BOX_VOXELS = 128 * 1024 * 1024


def local_features(patches: np.ndarray) -> np.ndarray:
    """Verbatim copy of ``_models/fault/fault_local_logistic.py::_features``."""

    array = np.asarray(patches, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W] seismic patches, received {array.shape}")
    features: list[np.ndarray] = []
    for patch in array[:, 0]:
        if not np.isfinite(patch).all():
            raise ValueError("seismic features must be finite")
        local_mean = uniform_filter(patch, size=3, mode="nearest")
        local_sq_mean = uniform_filter(patch * patch, size=3, mode="nearest")
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        features.append(np.column_stack((
            patch.ravel(), local_mean.ravel(), local_std.ravel(),
            (sobel(patch, axis=0, mode="nearest") / 8.0).ravel(),
            (sobel(patch, axis=1, mode="nearest") / 8.0).ravel(),
        )))
    return np.concatenate(features).astype(np.float32)


def load_coefficients(portable_path: Path, metrics_path: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if portable.get("source_metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("portable coefficients are not bound to the packaged metrics file")
    estimator = portable["estimator"]
    coef = np.asarray(estimator["coef"], dtype=np.float64)
    intercept = np.asarray(estimator["intercept"], dtype=np.float64)
    if coef.shape != (1, 5) or intercept.shape != (1,) or estimator.get("classes") != [0, 1]:
        raise ValueError("portable coefficients do not describe a 5-feature binary logistic")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return coef[0], float(intercept[0]), {
        "feature_contract": portable.get("feature_contract"),
        "source_joblib_sha256": portable.get("source_joblib_sha256"),
        "threshold": float(metrics["threshold"]),
        "threshold_source": metrics.get("threshold_source"),
        "test_metrics": metrics.get("test_metrics"),
    }


def build_trace_index(segy_path: Path) -> dict[str, Any]:
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        inline = handle.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
        crossline = handle.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
        samples_ms = np.asarray(handle.samples, dtype=np.float64)
    il_min, il_max = int(inline.min()), int(inline.max())
    xl_min, xl_max = int(crossline.min()), int(crossline.max())
    trace_map = np.full((il_max - il_min + 1, xl_max - xl_min + 1), -1, dtype=np.int32)
    trace_map[inline - il_min, crossline - xl_min] = np.arange(len(inline), dtype=np.int32)
    return {
        "il_min": il_min,
        "il_max": il_max,
        "xl_min": xl_min,
        "xl_max": xl_max,
        "n_traces": int(len(inline)),
        "samples_ms": samples_ms,
        "trace_map": trace_map,
    }


def resolve_box(index: dict[str, Any], request: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inclusive inline, crossline and sample-index vectors."""

    options = request.get("options") or {}
    ranges = [options.get(key) for key in ("inline_range", "crossline_range", "sample_range")]
    crop_start, crop_size = request.get("crop_start"), request.get("crop_size")
    n_samples = len(index["samples_ms"])
    if any(item is not None for item in ranges):
        if crop_start is not None or crop_size is not None:
            raise ValueError("options.*_range and crop_start/crop_size are mutually exclusive")
        if any(item is None for item in ranges):
            raise ValueError("inline_range, crossline_range and sample_range must be given together")
        (il0, il1), (xl0, xl1), (s0, s1) = (tuple(int(v) for v in item) for item in ranges)
    elif crop_start is not None or crop_size is not None:
        if crop_start is None or crop_size is None:
            raise ValueError("crop_start and crop_size must be given together")
        z0, i0, x0 = (int(v) for v in crop_start)
        dz, di, dx = (int(v) for v in crop_size)
        s0, s1 = z0, z0 + dz - 1
        il0, il1 = index["il_min"] + i0, index["il_min"] + i0 + di - 1
        xl0, xl1 = index["xl_min"] + x0, index["xl_min"] + x0 + dx - 1
    else:
        raise ValueError("an explicit box is required (options.*_range or crop_start/crop_size)")
    if not (index["il_min"] <= il0 <= il1 <= index["il_max"]):
        raise ValueError(f"inline box {il0}..{il1} outside survey {index['il_min']}..{index['il_max']}")
    if not (index["xl_min"] <= xl0 <= xl1 <= index["xl_max"]):
        raise ValueError(f"crossline box {xl0}..{xl1} outside survey {index['xl_min']}..{index['xl_max']}")
    if not (0 <= s0 <= s1 < n_samples):
        raise ValueError(f"sample box {s0}..{s1} outside trace length {n_samples}")
    shape = (s1 - s0 + 1, il1 - il0 + 1, xl1 - xl0 + 1)
    if min(shape) < MIN_BOX_EDGE:
        raise ValueError(f"box {shape} is smaller than {MIN_BOX_EDGE} samples on some axis")
    if int(np.prod(shape)) > MAX_BOX_VOXELS:
        raise ValueError(f"box {shape} exceeds {MAX_BOX_VOXELS} voxels; split the request")
    return (
        np.arange(il0, il1 + 1, dtype=np.int32),
        np.arange(xl0, xl1 + 1, dtype=np.int32),
        np.arange(s0, s1 + 1, dtype=np.int32),
    )


def read_box(segy_path: Path, index: dict[str, Any], inlines: np.ndarray, crosslines: np.ndarray, samples: np.ndarray) -> np.ndarray:
    t0, t1 = int(samples[0]), int(samples[-1])
    cube = np.empty((len(samples), len(inlines), len(crosslines)), dtype=np.float32)
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        for il_i, inline in enumerate(inlines.tolist()):
            row = index["trace_map"][int(inline) - index["il_min"]]
            for xl_i, crossline in enumerate(crosslines.tolist()):
                trace_index = int(row[int(crossline) - index["xl_min"]])
                if trace_index < 0:
                    raise ValueError(f"missing trace at inline={inline}, crossline={crossline}")
                trace = np.asarray(handle.trace[trace_index][t0 : t1 + 1], dtype=np.float32)
                cube[:, il_i, xl_i] = trace
    if not np.isfinite(cube).all():
        raise ValueError("requested seismic box contains non-finite amplitudes")
    return cube


def predict_probability(seismic: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    probability = np.empty_like(seismic, dtype=np.float32)
    for inline_index in range(seismic.shape[1]):
        patch = np.asarray(seismic[:, inline_index, :], dtype=np.float32).T[None, None]
        features = local_features(patch).astype(np.float64)
        logits = features @ coef + intercept
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        probability[:, inline_index, :] = values.reshape(patch.shape[-2:]).T.astype(np.float32)
    return probability


def run(request_path: Path) -> dict[str, Any]:
    request = load_request(request_path)
    payload = verify_payload(request["model_dir"])
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    segy_path = Path(request["seismic_path"])
    coef, intercept, model_info = load_coefficients(
        payload["portable_coefficients"], payload["training_metrics"]
    )
    threshold = request.get("threshold")
    threshold = model_info["threshold"] if threshold is None else float(threshold)
    index = build_trace_index(segy_path)
    inlines, crosslines, samples = resolve_box(index, request)
    seismic = read_box(segy_path, index, inlines, crosslines, samples)
    probability = predict_probability(seismic, coef, intercept)
    mask = (probability >= np.float32(threshold)).astype(np.uint8)

    probability_path = output_dir / "fault_probability.npy"
    mask_path = output_dir / "fault_mask.npy"
    grid_path = output_dir / "grid.json"
    np.save(probability_path, probability, allow_pickle=False)
    np.save(mask_path, mask, allow_pickle=False)
    samples_ms = index["samples_ms"][samples]
    grid = {
        "axis_order": list(AXES),
        "shape_zyx": list(probability.shape),
        "inline_range": [int(inlines[0]), int(inlines[-1])],
        "crossline_range": [int(crosslines[0]), int(crosslines[-1])],
        "sample_range": [int(samples[0]), int(samples[-1])],
        "sample_time_ms_range": [float(samples_ms[0]), float(samples_ms[-1])],
        "sample_interval_ms": float(index["samples_ms"][1] - index["samples_ms"][0]),
        "survey_inline_range": [index["il_min"], index["il_max"]],
        "survey_crossline_range": [index["xl_min"], index["xl_max"]],
        "survey_trace_count": index["n_traces"],
        "survey_samples_per_trace": int(len(index["samples_ms"])),
        "crop_start_zyx": [int(samples[0]), int(inlines[0] - index["il_min"]), int(crosslines[0] - index["xl_min"])],
    }
    atomic_json(grid_path, grid)
    elapsed = time.perf_counter() - started
    summary = {
        "probability": {
            "min": float(probability.min()),
            "max": float(probability.max()),
            "mean": float(probability.mean()),
        },
        "threshold": threshold,
        "threshold_source": (
            model_info["threshold_source"] if request.get("threshold") is None else "request"
        ),
        "mask_positive_fraction": float(mask.mean()),
        "voxels": int(probability.size),
        "elapsed_seconds": elapsed,
    }
    return {
        "status": "completed",
        "outputs": {
            "probability_npy": str(probability_path),
            "mask_npy": str(mask_path),
            "grid_json": str(grid_path),
        },
        "grid": grid,
        "summary": summary,
        "model": model_info,
        "payload_sha256": {role: sha256_file(path) for role, path in payload.items()},
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "segyio_trace_addressing": "INLINE_3D/CROSSLINE_3D headers, ignore_geometry=True",
            "device": "cpu",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.request)
    except Exception as exc:  # reported to the platform as a structured failure
        return fail(args.result, exc)
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
