#!/usr/bin/env python3
"""Offline SEG-Y stratigraphic instance segmentation with the bundled weights."""

from __future__ import annotations

import argparse
import colorsys
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SwinConfig,
)

try:
    import segyio
except ImportError as error:  # pragma: no cover - exercised by the CLI
    raise SystemExit(
        "Missing dependency 'segyio'. Run: "
        "python -m pip install -r minimal_sgy/requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
TRAINING_MIN = -1215.0
TRAINING_MAX = 1930.0
TRAINING_MEAN = 0.3865
TRAINING_STD = 0.03485
CONTRAST_FACTOR = 25.0
MODEL_SIZE = (512, 512)


@dataclass
class SegyGeometry:
    """Trace-to-cube mapping needed to write predictions back to SEG-Y."""

    inline_values: list[int]
    xline_values: list[int]
    trace_inline_indices: list[int]
    trace_xline_indices: list[int]
    sample_count: int
    trace_count: int
    missing_trace_count: int
    sample_interval_us: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a post-stack 3-D SEG-Y volume, segment ordered strata with "
            "the bundled HPC checkpoints, and write masks plus color figures."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .sgy/.segy file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/sgy_inference")
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when available",
    )
    parser.add_argument("--segformer-batch-size", type=int, default=16)
    parser.add_argument(
        "--mask2former-batch-size",
        type=int,
        default=4,
        help="Reduce to 1 if GPU memory is insufficient",
    )
    parser.add_argument(
        "--amplitude-mode",
        choices=("auto", "training", "robust"),
        default="auto",
        help="auto keeps training scaling when compatible, otherwise uses percentiles",
    )
    parser.add_argument("--query-threshold", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument(
        "--inline-count",
        type=int,
        help="Fallback inline count for SEG-Y files without valid 3-D headers",
    )
    parser.add_argument(
        "--max-inlines",
        type=int,
        help="Smoke-test only: process the first N inlines and omit mask.sgy",
    )
    parser.add_argument(
        "--no-mask-sgy",
        action="store_true",
        help="Do not create the SEG-Y copy containing integer mask labels",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _ordered_unique(values: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.unique(values.astype(np.int64, copy=False))
    return unique, {int(value): index for index, value in enumerate(unique)}


def read_segy(
    path: Path,
    inline_count: int | None,
    max_inlines: int | None = None,
) -> tuple[np.ndarray, SegyGeometry]:
    """Read a regular post-stack SEG-Y into [inline, xline, sample].

    When ``max_inlines`` is provided, only traces belonging to the first N
    Inline slices are materialized.  Header arrays remain lightweight and keep
    the full trace-to-grid mapping for provenance and optional SEG-Y export.
    """

    if not path.is_file():
        raise FileNotFoundError(f"SEG-Y input does not exist: {path}")
    if max_inlines is not None and max_inlines < 1:
        raise ValueError("--max-inlines must be positive")
    with segyio.open(str(path), "r", strict=False, ignore_geometry=True) as source:
        source.mmap()
        trace_count = int(source.tracecount)
        sample_count = int(len(source.samples))
        inline_headers = np.asarray(
            source.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int64
        )
        xline_headers = np.asarray(
            source.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int64
        )
        try:
            sample_interval = int(
                source.bin[segyio.BinField.Interval]
            )
        except (KeyError, RuntimeError):
            sample_interval = None

        inlines, inline_lookup = _ordered_unique(inline_headers)
        xlines, xline_lookup = _ordered_unique(xline_headers)
        regular_headers = (
            len(inlines) > 0
            and len(xlines) > 0
            and len(inlines) * len(xlines) >= trace_count
            and not (
                len(inlines) == 1
                and len(xlines) == 1
                and trace_count > 1
            )
        )
        # An explicit Inline count is a platform geometry decision and must
        # override coincidentally non-constant garbage in bytes 189/193.
        regular_headers = regular_headers and inline_count is None
        if regular_headers:
            trace_inline = np.asarray(
                [inline_lookup[int(value)] for value in inline_headers], dtype=np.int64
            )
            trace_xline = np.asarray(
                [xline_lookup[int(value)] for value in xline_headers], dtype=np.int64
            )
        else:
            if inline_count is None:
                raise ValueError(
                    "The SEG-Y inline/xline headers do not define a 3-D grid. "
                    "Pass --inline-count N so traces can be reshaped as N inlines."
                )
            if inline_count < 1 or trace_count % inline_count:
                raise ValueError(
                    f"--inline-count {inline_count} must divide {trace_count} traces"
                )
            xline_count = trace_count // inline_count
            inlines = np.arange(inline_count, dtype=np.int64)
            xlines = np.arange(xline_count, dtype=np.int64)
            trace_inline = np.repeat(np.arange(inline_count), xline_count)
            trace_xline = np.tile(np.arange(xline_count), inline_count)

        pair_codes = trace_inline * max(len(xlines), 1) + trace_xline
        if np.unique(pair_codes).size != trace_count:
            raise ValueError(
                "Duplicate inline/xline trace headers were found; "
                "pre-stack SEG-Y is not supported."
            )

        selected_inline_count = (
            len(inlines)
            if max_inlines is None
            else min(int(max_inlines), len(inlines))
        )
        selected_indices = np.flatnonzero(trace_inline < selected_inline_count)
        if selected_indices.size == trace_count:
            traces = np.asarray(source.trace.raw[:], dtype=np.float32)
        elif selected_indices.size and np.all(np.diff(selected_indices) == 1):
            start = int(selected_indices[0])
            stop = int(selected_indices[-1]) + 1
            traces = np.asarray(source.trace.raw[start:stop], dtype=np.float32)
        else:
            traces = np.empty((selected_indices.size, sample_count), dtype=np.float32)
            for target_index, trace_index in enumerate(selected_indices):
                traces[target_index] = source.trace.raw[int(trace_index)]

    cube = np.full(
        (selected_inline_count, len(xlines), sample_count), np.nan, dtype=np.float32
    )
    occupied = np.zeros(cube.shape[:2], dtype=bool)
    selected_inline = trace_inline[selected_indices]
    selected_xline = trace_xline[selected_indices]
    for trace, inline_index, xline_index in zip(
        traces,
        selected_inline,
        selected_xline,
        strict=True,
    ):
        cube[inline_index, xline_index] = trace
        occupied[inline_index, xline_index] = True

    selected_missing = int((~occupied).sum())
    if selected_missing:
        # Missing grid cells are neutral during inference and remain -1 in the mask.
        cube[~occupied] = 0.0
    missing = int(len(inlines) * len(xlines) - trace_count)
    geometry = SegyGeometry(
        inline_values=[int(value) for value in inlines],
        xline_values=[int(value) for value in xlines],
        trace_inline_indices=[int(value) for value in trace_inline],
        trace_xline_indices=[int(value) for value in trace_xline],
        sample_count=sample_count,
        trace_count=trace_count,
        missing_trace_count=missing,
        sample_interval_us=sample_interval,
    )
    return cube, geometry


def choose_amplitude_scaling(
    cube: np.ndarray, requested: str
) -> tuple[str, float, float]:
    finite = cube[np.isfinite(cube)]
    if not finite.size:
        raise ValueError("The SEG-Y volume has no finite amplitude samples")
    robust_low, robust_high = np.percentile(finite, (1.0, 99.0))
    spread_ratio = (robust_high - robust_low) / (TRAINING_MAX - TRAINING_MIN)
    center_compatible = robust_low > -4000 and robust_high < 5000
    if requested == "training" or (
        requested == "auto" and 0.08 <= spread_ratio <= 3.0 and center_compatible
    ):
        return "training", TRAINING_MIN, TRAINING_MAX
    if robust_high <= robust_low:
        raise ValueError("The SEG-Y amplitudes are constant; segmentation is impossible")
    return "robust", float(robust_low), float(robust_high)


def scale_raw_slice(raw_time_xline: np.ndarray, low: float, high: float) -> torch.Tensor:
    image = torch.from_numpy(raw_time_xline.astype(np.float32, copy=False))
    image = ((image - low) / (high - low)).clamp(0.0, 1.0)
    mean = image.mean()
    return ((image - mean) * CONTRAST_FACTOR + mean).clamp(0.0, 1.0)


def normalize_and_resize(channels: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        channels[None], size=MODEL_SIZE, mode="bilinear", align_corners=False
    )[0]
    return (resized - TRAINING_MEAN) / TRAINING_STD


def segformer_config() -> SegformerConfig:
    return SegformerConfig(
        num_labels=1,
        num_channels=3,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        drop_path_rate=0.1,
    )


def mask2former_config() -> Mask2FormerConfig:
    backbone = SwinConfig(
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.3,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    return Mask2FormerConfig(
        backbone_config=backbone.to_dict(),
        num_labels=1,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        encoder_feedforward_dim=1024,
        dim_feedforward=2048,
        encoder_layers=6,
        decoder_layers=10,
        num_attention_heads=8,
        num_queries=100,
        feature_strides=[4, 8, 16, 32],
        class_weight=1.0,
        mask_weight=1.0,
        dice_weight=10.0,
        dropout=0.0,
        use_auxiliary_loss=True,
    )


def load_checkpoint(model: torch.nn.Module, path: Path, expected_stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch 2.1
        payload = torch.load(path, map_location="cpu")
    stage = payload.get("stage")
    if stage != expected_stage:
        raise ValueError(
            f"{path} contains stage={stage!r}, expected {expected_stage!r}"
        )
    model.load_state_dict(payload["model"], strict=True)
    return {
        "path": str(path.resolve()),
        "stage": stage,
        "epoch": int(payload["epoch"]),
        "best_metric": float(payload["best_metric"]),
    }


def make_segformer_batch(
    cube: np.ndarray,
    indexes: Sequence[int],
    prior: np.ndarray | None,
    low: float,
    high: float,
) -> torch.Tensor:
    images = []
    for index in indexes:
        raw = scale_raw_slice(cube[index].T, low, high)
        if prior is None:
            channels = torch.stack((raw, raw, raw))
        else:
            mask = torch.from_numpy(prior[index].T.astype(np.float32, copy=False))
            channels = torch.stack((raw, mask, raw))
        images.append(normalize_and_resize(channels))
    return torch.stack(images)


@torch.inference_mode()
def infer_segformer_volume(
    model: SegformerForSemanticSegmentation,
    cube: np.ndarray,
    prior: np.ndarray | None,
    low: float,
    high: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result = np.empty(cube.shape, dtype=np.uint8)
    for start in range(0, len(cube), batch_size):
        indexes = list(range(start, min(start + batch_size, len(cube))))
        images = make_segformer_batch(cube, indexes, prior, low, high).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(pixel_values=images).logits
        logits = F.interpolate(
            logits.float(),
            size=(cube.shape[2], cube.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        masks = (torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy()
        result[indexes] = masks.transpose(0, 2, 1).astype(np.uint8)
        print(
            f"  SegFormer: {indexes[-1] + 1}/{len(cube)} inlines",
            flush=True,
        )
    return result


def order_layers_top_to_bottom(segmentation: np.ndarray) -> np.ndarray:
    ordered = np.full(segmentation.shape, -1, dtype=np.int16)
    layers = []
    for label in np.unique(segmentation):
        if label < 0:
            continue
        positions = np.nonzero(segmentation == label)
        layers.append((float(np.median(positions[0])), int(label)))
    for ordered_id, (_, original_id) in enumerate(sorted(layers)):
        ordered[segmentation == original_id] = ordered_id
    return ordered


def postprocess_instances(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_size: tuple[int, int],
    query_threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    probabilities = class_logits.softmax(dim=-1)
    scores, labels = probabilities.max(dim=-1)
    keep = (labels != 1) & (scores >= query_threshold)
    if not torch.any(keep):
        # Always retain the strongest geological query to produce a valid mask.
        best = probabilities[:, 0].argmax()
        keep[best] = True
    kept_scores = probabilities[keep, 0]
    masks = F.interpolate(
        mask_logits[keep, None].float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0].sigmoid()
    weighted = masks * kept_scores[:, None, None]

    initial_assignment = weighted.argmax(dim=0)
    consistent = []
    for query_index in range(len(weighted)):
        assigned_area = (initial_assignment == query_index).sum()
        original_area = (masks[query_index] >= mask_threshold).sum()
        if original_area > 0 and assigned_area / original_area >= 0.5:
            consistent.append(query_index)
    if not consistent:
        consistent = [int(kept_scores.argmax())]
    weighted = weighted[consistent]
    assignment = weighted.argmax(dim=0)
    confidence = weighted.max(dim=0).values
    segmentation = order_layers_top_to_bottom(
        assignment.cpu().numpy().astype(np.int16)
    )
    confidence_array = confidence.cpu().numpy().astype(np.float32)
    unique, counts = np.unique(segmentation, return_counts=True)
    largest_fraction = float(counts.max() / counts.sum()) if counts.size else 1.0
    stats: dict[str, float | int] = {
        "instances": int(len(unique)),
        "mean_confidence": float(confidence_array.mean()),
        "largest_instance_fraction": largest_fraction,
    }
    return segmentation, confidence_array, stats


@torch.inference_mode()
def infer_instances(
    model: Mask2FormerForUniversalSegmentation,
    cube: np.ndarray,
    refine: np.ndarray,
    low: float,
    high: float,
    device: torch.device,
    query_threshold: float,
    mask_threshold: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    labels = np.empty(cube.shape, dtype=np.int16)
    confidence = np.empty(cube.shape, dtype=np.float16)
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(cube), batch_size):
        indexes = list(range(start, min(start + batch_size, len(cube))))
        images = []
        for index in indexes:
            raw = scale_raw_slice(cube[index].T, low, high)
            prior = torch.from_numpy(refine[index].T.astype(np.float32, copy=False))
            images.append(normalize_and_resize(torch.stack((prior, raw, prior))))
        image = torch.stack(images).to(device)
        pixel_mask = torch.ones(
            (len(indexes), *MODEL_SIZE), dtype=torch.long, device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(pixel_values=image, pixel_mask=pixel_mask)
        for offset, index in enumerate(indexes):
            prediction, score, stats = postprocess_instances(
                output.class_queries_logits[offset].float(),
                output.masks_queries_logits[offset].float(),
                target_size=(cube.shape[2], cube.shape[1]),
                query_threshold=query_threshold,
                mask_threshold=mask_threshold,
            )
            labels[index] = prediction.T
            confidence[index] = score.T.astype(np.float16)
            rows.append({"inline_index": index, **stats})
        print(f"  Mask2Former: {indexes[-1] + 1}/{len(cube)} inlines", flush=True)
    return labels, confidence, rows


def write_mask_segy(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
    geometry: SegyGeometry,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    with segyio.open(str(output_path), "r+", strict=False, ignore_geometry=True) as target:
        for trace_index, (inline_index, xline_index) in enumerate(
            zip(
                geometry.trace_inline_indices,
                geometry.trace_xline_indices,
                strict=True,
            )
        ):
            target.trace[trace_index] = mask[inline_index, xline_index].astype(
                np.float32, copy=False
            )
        target.flush()


def color_map(max_label: int) -> ListedColormap:
    colors = [(0.05, 0.05, 0.05, 1.0)]
    golden_ratio = 0.618033988749895
    for label in range(max_label + 1):
        hue = (0.07 + label * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        colors.append((*rgb, 1.0))
    return ListedColormap(colors)


def select_visualizations(
    rows: list[dict[str, float | int]], count: int
) -> list[int]:
    count = max(1, min(count, len(rows)))
    selected = []
    for bin_index in range(count):
        start = bin_index * len(rows) // count
        stop = (bin_index + 1) * len(rows) // count
        candidates = rows[start:stop] or rows

        def quality(row: dict[str, float | int]) -> float:
            instances = int(row["instances"])
            diversity = min(instances / 8.0, 1.0)
            balance = 1.0 - float(row["largest_instance_fraction"])
            confidence = float(row["mean_confidence"])
            return 0.45 * diversity + 0.35 * balance + 0.20 * confidence

        selected.append(int(max(candidates, key=quality)["inline_index"]))
    return selected


def render_slice(
    cube: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    index: int,
    inline_value: int,
    path: Path,
    cmap: ListedColormap,
    color_max: int,
) -> None:
    raw = cube[index].T
    mask = labels[index].T
    score = confidence[index].T
    colored = cmap(mask + 1)
    grayscale = plt.get_cmap("gray")(
        np.clip((raw - np.percentile(raw, 1)) / max(np.ptp(np.percentile(raw, (1, 99))), 1e-6), 0, 1)
    )
    overlay = 0.48 * grayscale + 0.52 * colored
    figure, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
    panels = (
        (raw, "Seismic", "gray"),
        (mask + 1, "Colored layer mask", cmap),
        (overlay, "Overlay", None),
        (score, "Confidence", "viridis"),
    )
    for axis, (array, title, panel_cmap) in zip(axes, panels, strict=True):
        options: dict[str, Any] = {
            "aspect": "auto",
            "interpolation": "nearest",
            "cmap": panel_cmap,
        }
        if title == "Colored layer mask":
            options.update(vmin=0, vmax=color_max + 1)
        axis.imshow(array, **options)
        axis.set_title(title)
        axis.set_xlabel("Crossline")
        axis.set_ylabel("Sample")
    figure.suptitle(f"Inline {inline_value} (array index {index})", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_overview(
    cube: np.ndarray,
    labels: np.ndarray,
    indexes: Sequence[int],
    inline_values: Sequence[int],
    path: Path,
    cmap: ListedColormap,
    color_max: int,
) -> None:
    figure, axes = plt.subplots(
        2, len(indexes), figsize=(4 * len(indexes), 8), squeeze=False,
        constrained_layout=True,
    )
    for column, index in enumerate(indexes):
        axes[0, column].imshow(cube[index].T, cmap="gray", aspect="auto")
        axes[1, column].imshow(
            labels[index].T + 1,
            cmap=cmap,
            vmin=0,
            vmax=color_max + 1,
            aspect="auto",
        )
        axes[0, column].set_title(f"Inline {inline_values[index]}")
        axes[0, column].set_ylabel("Seismic")
        axes[1, column].set_ylabel("Colored mask")
        for axis in axes[:, column]:
            axis.set_xlabel("Crossline")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_inference(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    models_dir: str | Path | None = None,
    device: str = "auto",
    segformer_batch_size: int = 16,
    mask2former_batch_size: int = 4,
    amplitude_mode: str = "auto",
    query_threshold: float = 0.35,
    mask_threshold: float = 0.5,
    num_visualizations: int = 5,
    inline_count: int | None = None,
    max_inlines: int | None = None,
    write_mask_sgy: bool = True,
) -> dict[str, Any]:
    """Run SEG-Y inference and return the same metadata saved to summary.json.

    Parameters
    ----------
    input_path:
        Regular post-stack 3-D SEG-Y. The public array convention is
        ``[inline, xline, sample]``.
    output_dir:
        Destination for ``mask.npy``, ``mask.sgy``, ``confidence.npy``,
        ``overview.png``, per-inline visualizations, and ``summary.json``.
    models_dir:
        Directory containing ``segformer-base/best.pt``,
        ``segformer-refine/best.pt``, and ``mask2former/best.pt``.

    Returns
    -------
    dict
        JSON-serializable run summary, including output paths and volume shape.
    """

    args = argparse.Namespace(
        input=Path(input_path),
        output_dir=Path(output_dir),
        models_dir=Path(models_dir) if models_dir is not None else ROOT / "models",
        device=device,
        segformer_batch_size=segformer_batch_size,
        mask2former_batch_size=mask2former_batch_size,
        amplitude_mode=amplitude_mode,
        query_threshold=query_threshold,
        mask_threshold=mask_threshold,
        num_visualizations=num_visualizations,
        inline_count=inline_count,
        max_inlines=max_inlines,
        no_mask_sgy=not write_mask_sgy,
    )
    return run_with_options(args)


def run_with_options(args: argparse.Namespace) -> dict[str, Any]:
    """Internal shared implementation for the CLI and Python API."""

    started = time.time()
    if args.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if args.amplitude_mode not in {"auto", "training", "robust"}:
        raise ValueError(
            "amplitude_mode must be one of: auto, training, robust"
        )
    if args.segformer_batch_size < 1:
        raise ValueError("--segformer-batch-size must be positive")
    if args.mask2former_batch_size < 1:
        raise ValueError("--mask2former-batch-size must be positive")
    if not 0.0 <= args.query_threshold <= 1.0:
        raise ValueError("--query-threshold must be in [0, 1]")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold must be in [0, 1]")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[1/6] Reading {args.input}", flush=True)
    cube, geometry = read_segy(
        args.input.resolve(),
        args.inline_count,
        max_inlines=args.max_inlines,
    )
    full_inline_count = len(geometry.inline_values)
    amplitude_mode, scale_low, scale_high = choose_amplitude_scaling(
        cube, args.amplitude_mode
    )
    print(
        f"  shape={tuple(cube.shape)}, device={device}, "
        f"amplitude_mode={amplitude_mode} [{scale_low:.4g}, {scale_high:.4g}]",
        flush=True,
    )

    checkpoints: dict[str, dict[str, Any]] = {}
    print("[2/6] Loading SegFormer Base", flush=True)
    base = SegformerForSemanticSegmentation(segformer_config())
    checkpoints["segformer_base"] = load_checkpoint(
        base, args.models_dir / "segformer-base" / "best.pt", "segformer-base"
    )
    base.to(device).eval()
    base_mask = infer_segformer_volume(
        base,
        cube,
        prior=None,
        low=scale_low,
        high=scale_high,
        batch_size=args.segformer_batch_size,
        device=device,
    )
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[3/6] Loading SegFormer Refine", flush=True)
    refine_model = SegformerForSemanticSegmentation(segformer_config())
    checkpoints["segformer_refine"] = load_checkpoint(
        refine_model,
        args.models_dir / "segformer-refine" / "best.pt",
        "segformer-refine",
    )
    refine_model.to(device).eval()
    # The HPC training used a prompt-free SAM prior, but no SAM checkpoint is
    # bundled. Base SegFormer is the deterministic, fully offline substitute.
    refine_mask = infer_segformer_volume(
        refine_model,
        cube,
        prior=base_mask,
        low=scale_low,
        high=scale_high,
        batch_size=args.segformer_batch_size,
        device=device,
    )
    del refine_model, base_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[4/6] Loading Mask2Former and segmenting layers", flush=True)
    instance_model = Mask2FormerForUniversalSegmentation(mask2former_config())
    checkpoints["mask2former"] = load_checkpoint(
        instance_model,
        args.models_dir / "mask2former" / "best.pt",
        "mask2former",
    )
    instance_model.to(device).eval()
    labels, confidence, slice_rows = infer_instances(
        instance_model,
        cube,
        refine_mask,
        scale_low,
        scale_high,
        device,
        args.query_threshold,
        args.mask_threshold,
        args.mask2former_batch_size,
    )
    del instance_model, refine_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid_grid = np.zeros(labels.shape[:2], dtype=bool)
    for inline_index, xline_index in zip(
        geometry.trace_inline_indices,
        geometry.trace_xline_indices,
        strict=True,
    ):
        if inline_index < len(labels):
            valid_grid[inline_index, xline_index] = True
    labels[~valid_grid] = -1
    confidence[~valid_grid] = 0

    print("[5/6] Writing masks", flush=True)
    np.save(output_dir / "mask.npy", labels)
    np.save(output_dir / "confidence.npy", confidence)
    mask_sgy_written = False
    if (
        not args.no_mask_sgy
        and args.max_inlines is None
        and len(labels) == full_inline_count
    ):
        write_mask_segy(args.input.resolve(), output_dir / "mask.sgy", labels, geometry)
        mask_sgy_written = True

    print("[6/6] Rendering representative color slices", flush=True)
    selected = select_visualizations(slice_rows, args.num_visualizations)
    max_label = int(labels.max(initial=0))
    cmap = color_map(max_label)
    visualization_dir = output_dir / "visualizations"
    for index in selected:
        render_slice(
            cube,
            labels,
            confidence,
            index,
            geometry.inline_values[index],
            visualization_dir / f"inline_{geometry.inline_values[index]}.png",
            cmap,
            max_label,
        )
    render_overview(
        cube,
        labels,
        selected,
        geometry.inline_values,
        output_dir / "overview.png",
        cmap,
        max_label,
    )

    summary = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "volume_shape": list(labels.shape),
        "mask_dtype": str(labels.dtype),
        "label_range": [int(labels.min()), int(labels.max())],
        "device": str(device),
        "amplitude_scaling": {
            "requested": args.amplitude_mode,
            "effective": amplitude_mode,
            "low": scale_low,
            "high": scale_high,
        },
        "thresholds": {
            "query": args.query_threshold,
            "mask": args.mask_threshold,
        },
        "prior_compatibility_mode": "segformer-base-as-refine-prior",
        "geometry": {
            "inline_count": len(geometry.inline_values),
            "inline_range": [
                min(geometry.inline_values),
                max(geometry.inline_values),
            ],
            "xline_count": len(geometry.xline_values),
            "xline_range": [
                min(geometry.xline_values),
                max(geometry.xline_values),
            ],
            "sample_count": geometry.sample_count,
            "trace_count": geometry.trace_count,
            "missing_trace_count": geometry.missing_trace_count,
            "sample_interval_us": geometry.sample_interval_us,
        },
        "checkpoints": checkpoints,
        "selected_visualizations": [
            {
                **slice_rows[index],
                "inline_value": geometry.inline_values[index],
                "file": str(
                    (
                        visualization_dir
                        / f"inline_{geometry.inline_values[index]}.png"
                    ).resolve()
                ),
            }
            for index in selected
        ],
        "artifacts": {
            "mask_npy": str((output_dir / "mask.npy").resolve()),
            "mask_sgy": (
                str((output_dir / "mask.sgy").resolve())
                if mask_sgy_written
                else None
            ),
            "confidence_npy": str((output_dir / "confidence.npy").resolve()),
            "overview": str((output_dir / "overview.png").resolve()),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_with_options(parse_args())
    print(json.dumps(summary["artifacts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
