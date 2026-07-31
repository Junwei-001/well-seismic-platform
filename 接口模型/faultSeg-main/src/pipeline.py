"""Software-facing prediction and optional single-volume evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile

from .checkpoint import load_model
from .data import Shape3D
from .evaluation import ProbabilityHistogram
from .filters import fault_enhancement_filter
from .inference import choose_device, predict_volume
from .profiles import DEFAULT_CONFIG, load_profile
from .thresholds import ThresholdSelection, probability_summary, select_threshold
from .visualization import save_labeled_preview, save_orthogonal_preview, save_threshold_sweep
from .volumes import read_volume


@dataclass
class PredictionRequest:
    input_path: Path
    output_dir: Path
    input_format: str = "auto"
    shape: Shape3D | None = None
    label_path: Path | None = None
    label_format: str = "auto"
    label_shape: Shape3D | None = None
    profile: str = "auto"
    config_path: Path = DEFAULT_CONFIG
    checkpoint: Path | None = None
    component: int | str = 0
    threshold: str = "profile"
    optimize: str = "dice"
    patch_size: Shape3D = (128, 128, 128)
    overlap: Shape3D = (32, 32, 32)
    preprocessing: str = "none"
    similarity_gate: float = 0.85
    similarity_half_window: int = 7
    inline_byte: int = 189
    crossline_byte: int = 193
    device: str = "auto"
    amp: bool = True


def run_prediction(
    request: PredictionRequest,
    progress: Callable[[int, int, tuple[int, int, int]], None] | None = None,
) -> dict[str, Any]:
    """Run one prediction and return JSON-serializable result metadata."""
    source = read_volume(
        request.input_path, format=request.input_format, shape=request.shape,
        component=request.component, inline_byte=request.inline_byte,
        crossline_byte=request.crossline_byte,
    )
    seismic = source.data
    filter_metadata: dict[str, Any] | None = None
    if request.preprocessing == "fault-enhancement":
        seismic, similarity = fault_enhancement_filter(
            seismic,
            half_window=request.similarity_half_window,
            similarity_gate=request.similarity_gate,
        )
        filter_metadata = {
            "name": "fault-enhancement",
            "mode": "non-steered-approximation",
            "similarity_gate": request.similarity_gate,
            "half_window_samples": request.similarity_half_window,
            "similarity_percentiles": {
                str(q): float(np.nanpercentile(similarity, q)) for q in (1, 5, 50, 95, 99)
            },
        }
    if any(size < patch for size, patch in zip(seismic.shape, request.patch_size)):
        raise ValueError(f"volume shape {seismic.shape} is smaller than patch {request.patch_size}")
    requested_profile = "fault-enhanced" if request.preprocessing == "fault-enhancement" and request.profile == "auto" else request.profile
    profile = load_profile(requested_profile, request.input_path, request.config_path)
    checkpoint = request.checkpoint or profile.checkpoint
    device = choose_device(request.device)
    model, _ = load_model(checkpoint, device)
    probability = predict_volume(
        model, seismic, device, request.patch_size, request.overlap,
        amp_enabled=device.type == "cuda" and request.amp,
        normalize_patches=True, weighted_blending=True,
        progress=progress,
    )
    probability[:, ~source.valid_traces] = 0.0
    valid_probability = probability[:, source.valid_traces]
    truth: np.ndarray | None = None
    histogram: ProbabilityHistogram | None = None
    if request.label_path is not None:
        label = read_volume(
            request.label_path, format=request.label_format,
            shape=request.label_shape or request.shape,
            inline_byte=request.inline_byte, crossline_byte=request.crossline_byte,
        )
        if label.data.shape != probability.shape:
            raise ValueError(f"label shape {label.data.shape} differs from prediction {probability.shape}")
        truth = np.nan_to_num(label.data, nan=0.0) >= 0.5
        histogram = ProbabilityHistogram(1000)
        histogram.update(probability[:, source.valid_traces], truth[:, source.valid_traces])
    if request.threshold.lower() in {"best", "auto"}:
        if histogram is None:
            raise ValueError("--threshold best/auto requires --label; use otsu without labels")
        threshold_value, _ = histogram.best(request.optimize)
        selection = ThresholdSelection(threshold_value, f"best-{request.optimize}")
    else:
        selection = select_threshold(request.threshold, valid_probability, profile.threshold)
    mask = (probability >= selection.value) & source.valid_traces[None, :, :]
    output = request.output_dir
    output.mkdir(parents=True, exist_ok=True)
    probability_path = output / "probability.tif"
    mask_path = output / "mask.tif"
    preview_path = output / "preview.png"
    sweep_path = output / "thresholds.png"
    tifffile.imwrite(probability_path, probability.astype(np.float32), bigtiff=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(mask_path, mask.astype(np.uint8), bigtiff=True, metadata={"axes": "ZYX"})
    if request.preprocessing == "fault-enhancement":
        tifffile.imwrite(output / "enhanced_input.tif", seismic.astype(np.float32), bigtiff=True, metadata={"axes": "ZYX"})
    display = np.nan_to_num(seismic, nan=0.0)
    save_orthogonal_preview(display, probability, preview_path, selection.value)
    thresholds = tuple(sorted(set((*profile.threshold_grid, selection.value))))
    save_threshold_sweep(display, probability, sweep_path, thresholds)
    metrics = histogram.metrics_at(selection.value) if histogram is not None else None
    labeled_preview: str | None = None
    if truth is not None:
        labeled_path = output / "labeled_comparison.png"
        save_labeled_preview(display, probability, truth, labeled_path, selection.value)
        labeled_preview = labeled_path.name
    result = {
        "source": str(request.input_path.resolve()),
        "source_format": source.format,
        "shape_zyx": list(probability.shape),
        "valid_trace_fraction": float(source.valid_traces.mean()),
        "profile": profile.name,
        "checkpoint": str(checkpoint.resolve()),
        "device": str(device),
        "preprocessing": filter_metadata,
        "threshold": {"value": selection.value, "method": selection.method},
        "probability": probability_summary(valid_probability, selection.value),
        "evaluation": None if metrics is None else {
            "label": str(request.label_path.resolve()), "optimized_metric": request.optimize,
            "metrics": metrics,
        },
        "outputs": {
            "probability": probability_path.name, "mask": mask_path.name,
            "preview": preview_path.name, "threshold_sweep": sweep_path.name,
            "labeled_comparison": labeled_preview,
            "enhanced_input": "enhanced_input.tif" if filter_metadata else None,
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
