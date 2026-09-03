#!/usr/bin/env python3
"""SEG-Y stratigraphic instance-segmentation interface (weights supplied externally)."""

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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SwinConfig,
)

try:
    from .horizon_reconciliation import (
        DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
        HORIZON_DISPLAY_GATE_SCHEMA,
        UNKNOWN_LABEL,
        extract_horizon_surfaces,
        reconcile_global_packages,
    )
except ImportError:  # ``python minimal_sgy/infer.py``
    from horizon_reconciliation import (  # type: ignore[no-redef]
        DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
        HORIZON_DISPLAY_GATE_SCHEMA,
        UNKNOWN_LABEL,
        extract_horizon_surfaces,
        reconcile_global_packages,
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    _MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    # The isolated CUDA runtime intentionally carries only inference-critical
    # packages.  Pillow is its small, deterministic rendering fallback.
    plt = None
    ListedColormap = None
    _MATPLOTLIB_AVAILABLE = False

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
CHECKPOINT_NATIVE_PREPROCESSING_POLICY = "checkpoint-native-anisotropic-resize"
EXPERIMENTAL_ASPECT_PRESERVING_POLICY = (
    "experimental-bounded-aspect-preserving-overlap"
)


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


@dataclass(frozen=True)
class InferenceTile:
    """Mapping from one native section window to the checkpoint input square."""

    top: int
    left: int
    bottom: int
    right: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    uniform_scale: float | None
    scale_y: float | None = None
    scale_x: float | None = None

    @property
    def native_height(self) -> int:
        return self.bottom - self.top

    @property
    def native_width(self) -> int:
        return self.right - self.left


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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Pixels below this final confidence abstain with label -1",
    )
    parser.add_argument(
        "--preprocessing-policy",
        choices=(
            CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
            EXPERIMENTAL_ASPECT_PRESERVING_POLICY,
        ),
        default=CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
        help=(
            "Sealed checkpoint-native whole-Inline resize, or the explicit "
            "experimental aspect-preserving window path"
        ),
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.0,
        help="Experimental aspect-preserving window overlap",
    )
    parser.add_argument(
        "--max-tiles-per-inline",
        type=int,
        default=1,
        help="Must be 1 for checkpoint-native policy; experimental tile budget otherwise",
    )
    parser.add_argument(
        "--minimum-uniform-scale",
        type=float,
        default=0.35,
        help="Fail instead of silently over-compressing a native tile",
    )
    parser.add_argument(
        "--association-max-gap",
        type=int,
        default=2,
        help="Number of missing inlines tolerated by global package matching",
    )
    parser.add_argument(
        "--minimum-horizon-finite-trace-fraction",
        type=float,
        default=DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
        help="Minimum finite trace support required to display one horizon",
    )
    parser.add_argument(
        "--minimum-horizon-largest-component-fraction",
        type=float,
        default=DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
        help=(
            "Minimum fraction of a horizon's finite traces in its largest "
            "4-connected component"
        ),
    )
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument(
        "--inline-count",
        type=int,
        help="Fallback inline count for SEG-Y files without valid 3-D headers",
    )
    parser.add_argument(
        "--inline-byte",
        type=int,
        help="Resolved SEG-Y trace-header byte containing Inline numbers",
    )
    parser.add_argument(
        "--crossline-byte",
        type=int,
        help="Resolved SEG-Y trace-header byte containing Crossline numbers",
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
    *,
    inline_byte: int | None = None,
    crossline_byte: int | None = None,
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
    if (inline_byte is None) != (crossline_byte is None):
        raise ValueError("--inline-byte and --crossline-byte must be provided together")
    for option_name, byte_value in (
        ("--inline-byte", inline_byte),
        ("--crossline-byte", crossline_byte),
    ):
        if byte_value is not None and not 1 <= int(byte_value) <= 237:
            raise ValueError(f"{option_name} must stay within SEG-Y 4-byte header starts 1..237")
    with segyio.open(str(path), "r", strict=False, ignore_geometry=True) as source:
        source.mmap()
        trace_count = int(source.tracecount)
        sample_count = int(len(source.samples))
        inline_field = (
            segyio.TraceField.INLINE_3D if inline_byte is None else int(inline_byte)
        )
        crossline_field = (
            segyio.TraceField.CROSSLINE_3D
            if crossline_byte is None
            else int(crossline_byte)
        )
        inline_headers = np.asarray(source.attributes(inline_field)[:], dtype=np.int64)
        xline_headers = np.asarray(source.attributes(crossline_field)[:], dtype=np.int64)
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


def _axis_bounds(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    if count == 1:
        return [(0, length)]
    coverage_factor = count - (count - 1) * overlap
    span = min(length, int(np.ceil(length / coverage_factor)))
    starts = np.rint(np.linspace(0, length - span, count)).astype(np.int64)
    return [(int(start), min(int(start) + span, length)) for start in starts]


def plan_inference_tiles(
    native_shape: tuple[int, int],
    *,
    preprocessing_policy: str = CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
    overlap: float = 0.0,
    max_tiles: int = 1,
    minimum_uniform_scale: float = 0.35,
) -> tuple[list[InferenceTile], dict[str, Any]]:
    """Plan the sealed checkpoint preprocessing or an explicit experiment.

    The released checkpoints were trained by resizing each complete Inline to
    512x512 independently along sample and crossline.  That anisotropic mapping
    is therefore the production default.  The bounded aspect-preserving window
    planner remains available only through its explicitly experimental policy.

    Candidate row/column grids are exhaustively compared under the small tile
    budget.  The winning grid preserves the largest native scale available
    within that sealed budget; tile count is only a tie-breaker.  This avoids
    collapsing a long Inline into one heavily letterboxed model window merely
    because that single window clears a permissive minimum-scale floor.
    """

    height, width = (int(native_shape[0]), int(native_shape[1]))
    if height < 1 or width < 1:
        raise ValueError("native inference shape must be positive")
    if preprocessing_policy == CHECKPOINT_NATIVE_PREPROCESSING_POLICY:
        if overlap != 0.0 or max_tiles != 1:
            raise ValueError(
                "checkpoint-native-anisotropic-resize requires overlap=0 and "
                "max_tiles=1"
            )
        scale_y = MODEL_SIZE[0] / height
        scale_x = MODEL_SIZE[1] / width
        tile = InferenceTile(
            top=0,
            left=0,
            bottom=height,
            right=width,
            resized_height=MODEL_SIZE[0],
            resized_width=MODEL_SIZE[1],
            pad_top=0,
            pad_left=0,
            uniform_scale=None,
            scale_y=float(scale_y),
            scale_x=float(scale_x),
        )
        receipt = {
            "schema_version": "surface-seg.window-inference.v1",
            "preprocessing_policy": CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
            "mode": "whole-inline-anisotropic-resize",
            "native_shape_sample_xline": [height, width],
            "checkpoint_native_shape": list(MODEL_SIZE),
            "model_window": list(MODEL_SIZE),
            "overlap_fraction": 0.0,
            "max_tiles_per_inline": 1,
            "tile_selection_policy": "whole-inline-single-window",
            "tile_count_per_inline": 1,
            "whole_inline": True,
            "tile_stitching": False,
            "aspect_preserving": False,
            "nonuniform_resize": True,
            "training_preprocess_compatible": True,
            "degraded": False,
            "mapping": [
                {
                    "source_bounds_sample_xline": [0, 0, height, width],
                    "native_shape": [height, width],
                    "resized_shape": list(MODEL_SIZE),
                    "restore_shape": [height, width],
                    "scale_y": float(scale_y),
                    "scale_x": float(scale_x),
                    "padding_top_left_bottom_right": [0, 0, 0, 0],
                }
            ],
        }
        return [tile], receipt
    if preprocessing_policy != EXPERIMENTAL_ASPECT_PRESERVING_POLICY:
        raise ValueError(f"unsupported preprocessing policy: {preprocessing_policy}")
    if not 0.0 <= overlap < 0.5:
        raise ValueError("window overlap must be in [0, 0.5)")
    if max_tiles < 1:
        raise ValueError("max tiles per inline must be positive")
    candidates: list[tuple[float, int, int, int]] = []
    for rows in range(1, max_tiles + 1):
        for columns in range(1, max_tiles // rows + 1):
            count = rows * columns
            y_bounds = _axis_bounds(height, rows, overlap)
            x_bounds = _axis_bounds(width, columns, overlap)
            worst_height = max(stop - start for start, stop in y_bounds)
            worst_width = max(stop - start for start, stop in x_bounds)
            scale = min(
                MODEL_SIZE[0] / worst_height,
                MODEL_SIZE[1] / worst_width,
                1.0,
            )
            candidates.append((scale, count, rows, columns))
    eligible = [item for item in candidates if item[0] >= minimum_uniform_scale]
    if not eligible:
        scale, count, rows, columns = max(candidates)
        raise RuntimeError(
            "Native section cannot meet the bounded aspect-preserving window "
            f"contract: shape={native_shape}, best_uniform_scale={scale:.4f}, "
            f"minimum={minimum_uniform_scale:.4f}, max_tiles={max_tiles}. "
            "Increase --max-tiles-per-inline explicitly; inference was not run."
        )
    # Quality is the primary policy for this checkpoint family.  A previous
    # minimum-tile policy compressed F3 (462x951) into one 249x512 letterboxed
    # window and produced a single-package collapse.  Prefer the highest
    # uniform scale, then the fewest tiles and the deterministic grid order.
    scale, count, rows, columns = min(
        eligible, key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    y_bounds = _axis_bounds(height, rows, overlap)
    x_bounds = _axis_bounds(width, columns, overlap)
    tiles: list[InferenceTile] = []
    for top, bottom in y_bounds:
        for left, right in x_bounds:
            native_height, native_width = bottom - top, right - left
            uniform_scale = min(
                MODEL_SIZE[0] / native_height,
                MODEL_SIZE[1] / native_width,
                1.0,
            )
            resized_height = max(1, min(MODEL_SIZE[0], round(native_height * uniform_scale)))
            resized_width = max(1, min(MODEL_SIZE[1], round(native_width * uniform_scale)))
            tiles.append(
                InferenceTile(
                    top=top,
                    left=left,
                    bottom=bottom,
                    right=right,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    pad_top=(MODEL_SIZE[0] - resized_height) // 2,
                    pad_left=(MODEL_SIZE[1] - resized_width) // 2,
                    uniform_scale=float(uniform_scale),
                )
            )
    receipt = {
        "schema_version": "surface-seg.window-inference.v1",
        "preprocessing_policy": EXPERIMENTAL_ASPECT_PRESERVING_POLICY,
        "mode": "bounded-aspect-preserving-overlap",
        "native_shape_sample_xline": [height, width],
        "model_window": list(MODEL_SIZE),
        "overlap_fraction": float(overlap),
        "max_tiles_per_inline": int(max_tiles),
        "tile_selection_policy": "maximum-uniform-scale-within-tile-budget",
        "tile_count_per_inline": len(tiles),
        "whole_inline": len(tiles) == 1,
        "tile_stitching": len(tiles) > 1,
        "aspect_preserving": True,
        "nonuniform_resize": False,
        "training_preprocess_compatible": False,
        "degraded": False,
        "minimum_uniform_scale": float(minimum_uniform_scale),
        "minimum_effective_scale": float(min(item.uniform_scale for item in tiles)),
        "mapping": [
            {
                "source_bounds_sample_xline": [
                    item.top,
                    item.left,
                    item.bottom,
                    item.right,
                ],
                "native_shape": [item.native_height, item.native_width],
                "resized_shape": [item.resized_height, item.resized_width],
                "uniform_scale": item.uniform_scale,
                "padding_top_left_bottom_right": [
                    item.pad_top,
                    item.pad_left,
                    MODEL_SIZE[0] - item.pad_top - item.resized_height,
                    MODEL_SIZE[1] - item.pad_left - item.resized_width,
                ],
            }
            for item in tiles
        ],
    }
    return tiles, receipt


def _prepare_tile(channels: torch.Tensor, tile: InferenceTile) -> tuple[torch.Tensor, torch.Tensor]:
    native = channels[:, tile.top : tile.bottom, tile.left : tile.right]
    if (tile.resized_height, tile.resized_width) != (
        tile.native_height,
        tile.native_width,
    ):
        native = F.interpolate(
            native[None],
            size=(tile.resized_height, tile.resized_width),
            mode="bilinear",
            align_corners=False,
        )[0]
    pad_bottom = MODEL_SIZE[0] - tile.pad_top - tile.resized_height
    pad_right = MODEL_SIZE[1] - tile.pad_left - tile.resized_width
    padded = F.pad(
        native,
        (tile.pad_left, pad_right, tile.pad_top, pad_bottom),
        mode="replicate",
    )
    pixel_mask = torch.zeros(MODEL_SIZE, dtype=torch.long)
    pixel_mask[
        tile.pad_top : tile.pad_top + tile.resized_height,
        tile.pad_left : tile.pad_left + tile.resized_width,
    ] = 1
    return (padded - TRAINING_MEAN) / TRAINING_STD, pixel_mask


def _restore_tile_tensor(
    values: torch.Tensor,
    tile: InferenceTile,
    *,
    mode: str,
) -> torch.Tensor:
    cropped = values[
        ...,
        tile.pad_top : tile.pad_top + tile.resized_height,
        tile.pad_left : tile.pad_left + tile.resized_width,
    ]
    if tuple(cropped.shape[-2:]) != (tile.native_height, tile.native_width):
        options: dict[str, Any] = {"size": (tile.native_height, tile.native_width), "mode": mode}
        if mode != "nearest":
            options["align_corners"] = False
        cropped = F.interpolate(cropped[None] if cropped.ndim == 3 else cropped, **options)
        if values.ndim == 3:
            cropped = cropped[0]
    return cropped


def _blend_weight(height: int, width: int) -> np.ndarray:
    vertical = np.hanning(max(height, 3))[:height]
    horizontal = np.hanning(max(width, 3))[:width]
    return np.clip(np.outer(vertical, horizontal), 0.05, None).astype(np.float32)


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


class _UnionFind:
    def __init__(self, items: Sequence[tuple[int, int]]) -> None:
        self.parent = {item: item for item in items}
        self.member_tiles = {item: {item[0]} for item in items}

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return True
        # A reconciled component may contain at most one query from each tile.
        # This prevents transitive cycles in a 2x2 overlap grid from swallowing
        # two distinct layers emitted by the same model evaluation.
        if self.member_tiles[left_root] & self.member_tiles[right_root]:
            return False
        parent, child = min(left_root, right_root), max(left_root, right_root)
        self.parent[child] = parent
        self.member_tiles[parent].update(self.member_tiles.pop(child))
        return True


def _native_tile_result(
    prediction: np.ndarray,
    confidence: np.ndarray,
    tile: InferenceTile,
) -> tuple[np.ndarray, np.ndarray]:
    label_tensor = torch.from_numpy(prediction.astype(np.float32, copy=False))[None]
    score_tensor = torch.from_numpy(confidence.astype(np.float32, copy=False))[None]
    native_labels = _restore_tile_tensor(label_tensor, tile, mode="nearest")[0]
    native_scores = _restore_tile_tensor(score_tensor, tile, mode="bilinear")[0]
    return (
        native_labels.numpy().astype(np.int16),
        native_scores.numpy().astype(np.float32),
    )


def _stitch_inline_instances(
    tile_labels: Sequence[np.ndarray],
    tile_scores: Sequence[np.ndarray],
    tiles: Sequence[InferenceTile],
    native_shape: tuple[int, int],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    nodes = [
        (tile_index, int(label))
        for tile_index, labels in enumerate(tile_labels)
        for label in np.unique(labels)
        if label >= 0
    ]
    if not nodes:
        return (
            np.full(native_shape, UNKNOWN_LABEL, dtype=np.int16),
            np.zeros(native_shape, dtype=np.float32),
            {
                "instances": 0,
                "mean_confidence": 0.0,
                "largest_instance_fraction": 0.0,
                "unknown_fraction": 1.0,
                "ambiguous_union_rejection_count": 0,
            },
        )
    unions = _UnionFind(nodes)
    ambiguous_union_rejections = 0
    # Match overlapping tile instances by overlap coefficient.  Greedy
    # one-to-one selection prevents one broad query from swallowing layers.
    for left_index, left_tile in enumerate(tiles):
        for right_index in range(left_index + 1, len(tiles)):
            right_tile = tiles[right_index]
            top, bottom = max(left_tile.top, right_tile.top), min(left_tile.bottom, right_tile.bottom)
            left, right = max(left_tile.left, right_tile.left), min(left_tile.right, right_tile.right)
            if top >= bottom or left >= right:
                continue
            left_y = slice(top - left_tile.top, bottom - left_tile.top)
            left_x = slice(left - left_tile.left, right - left_tile.left)
            right_y = slice(top - right_tile.top, bottom - right_tile.top)
            right_x = slice(left - right_tile.left, right - right_tile.left)
            left_labels = tile_labels[left_index][left_y, left_x]
            right_labels = tile_labels[right_index][right_y, right_x]
            valid = (
                (left_labels >= 0)
                & (right_labels >= 0)
                & (tile_scores[left_index][left_y, left_x] >= confidence_threshold)
                & (tile_scores[right_index][right_y, right_x] >= confidence_threshold)
            )
            candidates: list[tuple[float, int, int]] = []
            for left_label in np.unique(left_labels[valid]):
                left_area = int(np.count_nonzero(valid & (left_labels == left_label)))
                for right_label in np.unique(right_labels[valid & (left_labels == left_label)]):
                    intersection = int(
                        np.count_nonzero(
                            valid
                            & (left_labels == left_label)
                            & (right_labels == right_label)
                        )
                    )
                    right_area = int(np.count_nonzero(valid & (right_labels == right_label)))
                    coefficient = intersection / max(min(left_area, right_area), 1)
                    if coefficient >= 0.35:
                        candidates.append((coefficient, int(left_label), int(right_label)))
            used_left: set[int] = set()
            used_right: set[int] = set()
            for _, left_label, right_label in sorted(candidates, reverse=True):
                if left_label in used_left or right_label in used_right:
                    continue
                if not unions.union(
                    (left_index, left_label), (right_index, right_label)
                ):
                    ambiguous_union_rejections += 1
                    continue
                used_left.add(left_label)
                used_right.add(right_label)

    component_depths: dict[tuple[int, int], list[float]] = {}
    for tile_index, label in nodes:
        positions = np.nonzero(
            (tile_labels[tile_index] == label)
            & (tile_scores[tile_index] >= confidence_threshold)
        )
        if positions[0].size:
            component_depths.setdefault(unions.find((tile_index, label)), []).append(
                float(np.median(positions[0] + tiles[tile_index].top))
            )
    roots = sorted(
        component_depths,
        key=lambda item: (float(np.median(component_depths[item])), item),
    )
    root_to_ordered = {root: index for index, root in enumerate(roots)}
    output = np.full(native_shape, UNKNOWN_LABEL, dtype=np.int16)
    output_score = np.zeros(native_shape, dtype=np.float32)
    best_vote = np.zeros(native_shape, dtype=np.float32)
    for tile_index, tile in enumerate(tiles):
        labels = tile_labels[tile_index]
        scores = tile_scores[tile_index]
        weight = _blend_weight(tile.native_height, tile.native_width)
        for local_label in np.unique(labels):
            if local_label < 0:
                continue
            root = unions.find((tile_index, int(local_label)))
            if root not in root_to_ordered:
                continue
            candidate = (labels == local_label) & (scores >= confidence_threshold)
            vote = scores * weight
            target_vote = best_vote[tile.top : tile.bottom, tile.left : tile.right]
            update = candidate & (vote > target_vote)
            target_labels = output[tile.top : tile.bottom, tile.left : tile.right]
            target_scores = output_score[tile.top : tile.bottom, tile.left : tile.right]
            target_labels[update] = root_to_ordered[root]
            target_scores[update] = scores[update]
            target_vote[update] = vote[update]

    known_scores = output_score[output >= 0]
    unique, counts = np.unique(output[output >= 0], return_counts=True)
    known_count = int(counts.sum()) if counts.size else 0
    stats: dict[str, float | int] = {
        "instances": int(len(unique)),
        "mean_confidence": float(known_scores.mean()) if known_scores.size else 0.0,
        "largest_instance_fraction": (
            float(counts.max() / known_count) if known_count else 0.0
        ),
        "unknown_fraction": float(np.count_nonzero(output < 0) / output.size),
        "ambiguous_union_rejection_count": ambiguous_union_rejections,
    }
    return output, output_score, stats


def _apply_confidence_abstention(
    prediction: np.ndarray,
    score: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Apply the production abstention gate without tile reconciliation."""

    output = prediction.astype(np.int16, copy=True)
    output_score = score.astype(np.float32, copy=True)
    unknown = (output < 0) | (output_score < confidence_threshold)
    output[unknown] = UNKNOWN_LABEL
    output_score[unknown] = 0.0
    known_scores = output_score[output >= 0]
    unique, counts = np.unique(output[output >= 0], return_counts=True)
    known_count = int(counts.sum()) if counts.size else 0
    return output, output_score, {
        "instances": int(len(unique)),
        "mean_confidence": float(known_scores.mean()) if known_scores.size else 0.0,
        "largest_instance_fraction": (
            float(counts.max() / known_count) if known_count else 0.0
        ),
        "unknown_fraction": float(np.count_nonzero(unknown) / output.size),
        "ambiguous_union_rejection_count": 0,
    }


@torch.inference_mode()
def infer_segformer_volume(
    model: SegformerForSemanticSegmentation,
    cube: np.ndarray,
    prior: np.ndarray | None,
    low: float,
    high: float,
    batch_size: int,
    device: torch.device,
    tiles: Sequence[InferenceTile],
    preprocessing_policy: str = CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
) -> np.ndarray:
    result = np.empty(cube.shape, dtype=np.uint8)
    if preprocessing_policy == CHECKPOINT_NATIVE_PREPROCESSING_POLICY:
        if len(tiles) != 1:
            raise RuntimeError("checkpoint-native preprocessing requires one window")
        tile = tiles[0]
        for start in range(0, len(cube), batch_size):
            indexes = list(range(start, min(start + batch_size, len(cube))))
            prepared: list[torch.Tensor] = []
            for index in indexes:
                raw = scale_raw_slice(cube[index].T, low, high)
                if prior is None:
                    channels = torch.stack((raw, raw, raw))
                else:
                    mask = torch.from_numpy(
                        prior[index].T.astype(np.float32, copy=False)
                    )
                    channels = torch.stack((raw, mask, raw))
                prepared.append(_prepare_tile(channels, tile)[0])
            images = torch.stack(prepared).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixel_values=images).logits
            # Match the checkpoint training/replay path exactly: restore model
            # logits directly to the complete native Inline, with no tile blend.
            logits = F.interpolate(
                logits.float(),
                size=(cube.shape[2], cube.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            masks = (torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy()
            result[indexes] = masks.transpose(0, 2, 1).astype(np.uint8)
            print(
                f"  SegFormer: {indexes[-1] + 1}/{len(cube)} inlines "
                "(checkpoint-native whole Inline)",
                flush=True,
            )
        return result

    for index in range(len(cube)):
        raw = scale_raw_slice(cube[index].T, low, high)
        if prior is None:
            channels = torch.stack((raw, raw, raw))
        else:
            mask = torch.from_numpy(prior[index].T.astype(np.float32, copy=False))
            channels = torch.stack((raw, mask, raw))
        prepared = [_prepare_tile(channels, tile)[0] for tile in tiles]
        accumulated = np.zeros(raw.shape, dtype=np.float32)
        weights = np.zeros(raw.shape, dtype=np.float32)
        for start in range(0, len(prepared), batch_size):
            tile_indexes = list(range(start, min(start + batch_size, len(prepared))))
            images = torch.stack([prepared[item] for item in tile_indexes]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixel_values=images).logits
            logits = F.interpolate(
                logits.float(), size=MODEL_SIZE, mode="bilinear", align_corners=False
            )[:, 0]
            for offset, tile_index in enumerate(tile_indexes):
                tile = tiles[tile_index]
                native_logits = _restore_tile_tensor(
                    logits[offset][None], tile, mode="bilinear"
                )[0].cpu().numpy()
                weight = _blend_weight(tile.native_height, tile.native_width)
                accumulated[tile.top : tile.bottom, tile.left : tile.right] += (
                    native_logits * weight
                )
                weights[tile.top : tile.bottom, tile.left : tile.right] += weight
        blended_logits = np.clip(
            accumulated / np.maximum(weights, 1e-6), -50.0, 50.0
        )
        mask = (1.0 / (1.0 + np.exp(-blended_logits))) > 0.5
        result[index] = mask.T.astype(np.uint8)
        print(
            f"  SegFormer: {index + 1}/{len(cube)} inlines "
            f"({len(tiles)} experimental aspect-preserving windows/inline)",
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
    valid_pixel_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    probabilities = class_logits.softmax(dim=-1)
    scores, labels = probabilities.max(dim=-1)
    keep = (labels != 1) & (scores >= query_threshold)
    if not torch.any(keep):
        segmentation = np.full(target_size, UNKNOWN_LABEL, dtype=np.int16)
        confidence = np.zeros(target_size, dtype=np.float32)
        return segmentation, confidence, {
            "instances": 0,
            "mean_confidence": 0.0,
            "largest_instance_fraction": 0.0,
            "unknown_fraction": 1.0,
        }
    kept_scores = probabilities[keep, 0]
    masks = F.interpolate(
        mask_logits[keep, None].float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0].sigmoid()
    weighted = masks * kept_scores[:, None, None]
    if valid_pixel_mask is None:
        valid = torch.ones(target_size, dtype=torch.bool, device=weighted.device)
    else:
        valid = valid_pixel_mask.to(device=weighted.device, dtype=torch.bool)
        if tuple(valid.shape) != target_size:
            valid = F.interpolate(
                valid[None, None].float(), size=target_size, mode="nearest"
            )[0, 0].bool()
    weighted[:, ~valid] = 0

    initial_assignment = weighted.argmax(dim=0)
    consistent = []
    for query_index in range(len(weighted)):
        assigned_area = ((initial_assignment == query_index) & valid).sum()
        original_area = ((masks[query_index] >= mask_threshold) & valid).sum()
        if original_area > 0 and assigned_area / original_area >= 0.5:
            consistent.append(query_index)
    if not consistent:
        segmentation = np.full(target_size, UNKNOWN_LABEL, dtype=np.int16)
        confidence = np.zeros(target_size, dtype=np.float32)
        return segmentation, confidence, {
            "instances": 0,
            "mean_confidence": 0.0,
            "largest_instance_fraction": 0.0,
            "unknown_fraction": 1.0,
        }
    weighted = weighted[consistent]
    assignment = weighted.argmax(dim=0)
    confidence = weighted.max(dim=0).values
    segmentation = order_layers_top_to_bottom(
        assignment.cpu().numpy().astype(np.int16)
    )
    confidence_array = confidence.cpu().numpy().astype(np.float32)
    valid_array = valid.cpu().numpy()
    segmentation[~valid_array] = UNKNOWN_LABEL
    confidence_array[~valid_array] = 0.0
    known = valid_array & (segmentation >= 0)
    unique, counts = np.unique(segmentation[known], return_counts=True)
    largest_fraction = float(counts.max() / counts.sum()) if counts.size else 1.0
    stats: dict[str, float | int] = {
        "instances": int(len(unique)),
        "mean_confidence": float(confidence_array[valid_array].mean()),
        "largest_instance_fraction": largest_fraction,
        "unknown_fraction": float(
            np.count_nonzero(valid_array & (segmentation < 0))
            / max(np.count_nonzero(valid_array), 1)
        ),
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
    confidence_threshold: float,
    batch_size: int,
    tiles: Sequence[InferenceTile],
    preprocessing_policy: str = CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    labels = np.full(cube.shape, UNKNOWN_LABEL, dtype=np.int16)
    confidence = np.zeros(cube.shape, dtype=np.float16)
    rows: list[dict[str, float | int]] = []
    if preprocessing_policy == CHECKPOINT_NATIVE_PREPROCESSING_POLICY:
        if len(tiles) != 1:
            raise RuntimeError("checkpoint-native preprocessing requires one window")
        tile = tiles[0]
        for start in range(0, len(cube), batch_size):
            indexes = list(range(start, min(start + batch_size, len(cube))))
            images: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            for index in indexes:
                raw = scale_raw_slice(cube[index].T, low, high)
                prior = torch.from_numpy(
                    refine[index].T.astype(np.float32, copy=False)
                )
                image, pixel_mask = _prepare_tile(
                    torch.stack((prior, raw, prior)), tile
                )
                images.append(image)
                masks.append(pixel_mask)
            image_batch = torch.stack(images).to(device)
            pixel_mask_batch = torch.stack(masks).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    pixel_values=image_batch,
                    pixel_mask=pixel_mask_batch,
                )
            for offset, index in enumerate(indexes):
                prediction, score, _ = postprocess_instances(
                    output.class_queries_logits[offset].float(),
                    output.masks_queries_logits[offset].float(),
                    target_size=(cube.shape[2], cube.shape[1]),
                    query_threshold=query_threshold,
                    mask_threshold=mask_threshold,
                )
                prediction, score, stats = _apply_confidence_abstention(
                    prediction,
                    score,
                    confidence_threshold=confidence_threshold,
                )
                labels[index] = prediction.T
                confidence[index] = score.T.astype(np.float16)
                rows.append({"inline_index": index, **stats})
            print(
                f"  Mask2Former: {indexes[-1] + 1}/{len(cube)} inlines "
                "(checkpoint-native whole Inline; no tile stitching)",
                flush=True,
            )
        return labels, confidence, rows

    for index in range(len(cube)):
        raw = scale_raw_slice(cube[index].T, low, high)
        prior = torch.from_numpy(refine[index].T.astype(np.float32, copy=False))
        channels = torch.stack((prior, raw, prior))
        prepared = [_prepare_tile(channels, tile) for tile in tiles]
        tile_labels: list[np.ndarray] = []
        tile_scores: list[np.ndarray] = []
        for start in range(0, len(prepared), batch_size):
            tile_indexes = list(range(start, min(start + batch_size, len(prepared))))
            image = torch.stack([prepared[item][0] for item in tile_indexes]).to(device)
            pixel_mask = torch.stack([prepared[item][1] for item in tile_indexes]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(pixel_values=image, pixel_mask=pixel_mask)
            for offset, tile_index in enumerate(tile_indexes):
                prediction, score, _ = postprocess_instances(
                    output.class_queries_logits[offset].float(),
                    output.masks_queries_logits[offset].float(),
                    target_size=MODEL_SIZE,
                    query_threshold=query_threshold,
                    mask_threshold=mask_threshold,
                    valid_pixel_mask=pixel_mask[offset],
                )
                native_labels, native_score = _native_tile_result(
                    prediction, score, tiles[tile_index]
                )
                tile_labels.append(native_labels)
                tile_scores.append(native_score)
        prediction, score, stats = _stitch_inline_instances(
            tile_labels,
            tile_scores,
            tiles,
            native_shape=(cube.shape[2], cube.shape[1]),
            confidence_threshold=confidence_threshold,
        )
        labels[index] = prediction.T
        confidence[index] = score.T.astype(np.float16)
        rows.append({"inline_index": index, **stats})
        print(
            f"  Mask2Former: {index + 1}/{len(cube)} inlines "
            f"({len(tiles)} experimental aspect-preserving windows/inline)",
            flush=True,
        )
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


def color_map(max_label: int) -> Any:
    colors = [(0.05, 0.05, 0.05, 1.0)]
    golden_ratio = 0.618033988749895
    for label in range(max_label + 1):
        hue = (0.07 + label * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        colors.append((*rgb, 1.0))
    if _MATPLOTLIB_AVAILABLE:
        assert ListedColormap is not None
        return ListedColormap(colors)
    return np.asarray(colors, dtype=np.float32)


def _pillow_grayscale(array: np.ndarray) -> np.ndarray:
    finite = np.asarray(array, dtype=np.float32)
    valid = finite[np.isfinite(finite)]
    if not valid.size:
        return np.zeros((*finite.shape, 3), dtype=np.uint8)
    low, high = np.percentile(valid, (1.0, 99.0))
    if high <= low:
        scaled = np.zeros(finite.shape, dtype=np.uint8)
    else:
        scaled = np.clip((finite - low) / (high - low), 0.0, 1.0)
        scaled = np.nan_to_num(scaled, nan=0.0)
        scaled = np.rint(scaled * 255.0).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=-1)


def _pillow_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indexes = np.asarray(mask, dtype=np.int64) + 1
    indexes = np.clip(indexes, 0, len(palette) - 1)
    return np.rint(palette[indexes, :3] * 255.0).astype(np.uint8)


def _pillow_confidence(score: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    value = np.nan_to_num(value, nan=0.0)
    red = np.clip(2.0 * value - 0.5, 0.0, 1.0)
    green = np.clip(2.0 * value, 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * value, 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(
        np.uint8
    )


def _save_pillow_grid(
    panels: Sequence[tuple[np.ndarray, str, bool]],
    *,
    rows: int,
    columns: int,
    path: Path,
) -> None:
    panel_width = 480
    panel_height = 520
    title_height = 28
    canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + title_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (array, title, categorical) in enumerate(panels):
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
        resampling = (
            Image.Resampling.NEAREST
            if categorical
            else Image.Resampling.BILINEAR
        )
        image = image.resize((panel_width, panel_height), resample=resampling)
        row, column = divmod(index, columns)
        left = column * panel_width
        top = row * (panel_height + title_height)
        canvas.paste(image, (left, top + title_height))
        draw.text((left + 8, top + 7), title, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


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
    cmap: Any,
    color_max: int,
) -> None:
    raw = cube[index].T
    mask = labels[index].T
    score = confidence[index].T
    if not _MATPLOTLIB_AVAILABLE:
        grayscale = _pillow_grayscale(raw)
        colored = _pillow_mask(mask, np.asarray(cmap))
        overlay = np.rint(
            0.48 * grayscale.astype(np.float32)
            + 0.52 * colored.astype(np.float32)
        ).astype(np.uint8)
        _save_pillow_grid(
            (
                (grayscale, "Seismic", False),
                (colored, "Colored layer mask", True),
                (overlay, "Overlay", False),
                (_pillow_confidence(score), "Confidence", False),
            ),
            rows=1,
            columns=4,
            path=path,
        )
        return
    assert plt is not None
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
    cmap: Any,
    color_max: int,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        panels: list[tuple[np.ndarray, str, bool]] = []
        palette = np.asarray(cmap)
        for index in indexes:
            panels.append(
                (
                    _pillow_grayscale(cube[index].T),
                    f"Inline {inline_values[index]} seismic",
                    False,
                )
            )
        for index in indexes:
            panels.append(
                (
                    _pillow_mask(labels[index].T, palette),
                    f"Inline {inline_values[index]} colored mask",
                    True,
                )
            )
        _save_pillow_grid(
            panels,
            rows=2,
            columns=len(indexes),
            path=path,
        )
        return
    assert plt is not None
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
    confidence_threshold: float = 0.35,
    preprocessing_policy: str = CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
    window_overlap: float = 0.0,
    max_tiles_per_inline: int = 1,
    minimum_uniform_scale: float = 0.35,
    association_max_gap: int = 2,
    minimum_horizon_finite_trace_fraction: float = (
        DEFAULT_MINIMUM_FINITE_TRACE_FRACTION
    ),
    minimum_horizon_largest_component_fraction: float = (
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
    ),
    num_visualizations: int = 5,
    inline_count: int | None = None,
    inline_byte: int | None = None,
    crossline_byte: int | None = None,
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
        confidence_threshold=confidence_threshold,
        preprocessing_policy=preprocessing_policy,
        window_overlap=window_overlap,
        max_tiles_per_inline=max_tiles_per_inline,
        minimum_uniform_scale=minimum_uniform_scale,
        association_max_gap=association_max_gap,
        minimum_horizon_finite_trace_fraction=(
            minimum_horizon_finite_trace_fraction
        ),
        minimum_horizon_largest_component_fraction=(
            minimum_horizon_largest_component_fraction
        ),
        num_visualizations=num_visualizations,
        inline_count=inline_count,
        inline_byte=inline_byte,
        crossline_byte=crossline_byte,
        max_inlines=max_inlines,
        no_mask_sgy=not write_mask_sgy,
    )
    return run_with_options(args)


def run_with_options(args: argparse.Namespace) -> dict[str, Any]:
    """Internal shared implementation for the CLI and Python API."""

    started = time.time()
    # ``getattr`` keeps Namespace objects created by older Python callers valid.
    confidence_threshold = float(getattr(args, "confidence_threshold", 0.35))
    preprocessing_policy = str(
        getattr(
            args,
            "preprocessing_policy",
            CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
        )
    )
    window_overlap = float(getattr(args, "window_overlap", 0.0))
    max_tiles_per_inline = int(getattr(args, "max_tiles_per_inline", 1))
    minimum_uniform_scale = float(getattr(args, "minimum_uniform_scale", 0.35))
    association_max_gap = int(getattr(args, "association_max_gap", 2))
    minimum_horizon_finite_trace_fraction = float(
        getattr(
            args,
            "minimum_horizon_finite_trace_fraction",
            DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
        )
    )
    minimum_horizon_largest_component_fraction = float(
        getattr(
            args,
            "minimum_horizon_largest_component_fraction",
            DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
        )
    )
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
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if preprocessing_policy not in {
        CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
        EXPERIMENTAL_ASPECT_PRESERVING_POLICY,
    }:
        raise ValueError("unsupported --preprocessing-policy")
    if not 0.0 <= window_overlap < 0.5:
        raise ValueError("--window-overlap must be in [0, 0.5)")
    if max_tiles_per_inline < 1:
        raise ValueError("--max-tiles-per-inline must be positive")
    if not 0.0 < minimum_uniform_scale <= 1.0:
        raise ValueError("--minimum-uniform-scale must be in (0, 1]")
    if (
        preprocessing_policy == CHECKPOINT_NATIVE_PREPROCESSING_POLICY
        and (window_overlap != 0.0 or max_tiles_per_inline != 1)
    ):
        raise ValueError(
            "checkpoint-native-anisotropic-resize requires --window-overlap 0 "
            "and --max-tiles-per-inline 1"
        )
    if association_max_gap < 0:
        raise ValueError("--association-max-gap cannot be negative")
    if not (
        DEFAULT_MINIMUM_FINITE_TRACE_FRACTION
        <= minimum_horizon_finite_trace_fraction
        <= 1.0
    ):
        raise ValueError(
            "--minimum-horizon-finite-trace-fraction must stay within [0.10, 1]"
        )
    if not (
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
        <= minimum_horizon_largest_component_fraction
        <= 1.0
    ):
        raise ValueError(
            "--minimum-horizon-largest-component-fraction must stay within [0.05, 1]"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[1/6] Reading {args.input}", flush=True)
    cube, geometry = read_segy(
        args.input.resolve(),
        args.inline_count,
        max_inlines=args.max_inlines,
        inline_byte=getattr(args, "inline_byte", None),
        crossline_byte=getattr(args, "crossline_byte", None),
    )
    full_inline_count = len(geometry.inline_values)
    amplitude_mode, scale_low, scale_high = choose_amplitude_scaling(
        cube, args.amplitude_mode
    )
    tiles, window_receipt = plan_inference_tiles(
        (cube.shape[2], cube.shape[1]),
        preprocessing_policy=preprocessing_policy,
        overlap=window_overlap,
        max_tiles=max_tiles_per_inline,
        minimum_uniform_scale=minimum_uniform_scale,
    )
    window_receipt["inline_count"] = len(cube)
    window_receipt["model_evaluation_count"] = len(cube) * len(tiles) * 3
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
        tiles=tiles,
        preprocessing_policy=preprocessing_policy,
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
        tiles=tiles,
        preprocessing_policy=preprocessing_policy,
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
        confidence_threshold,
        args.mask2former_batch_size,
        tiles,
        preprocessing_policy,
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
    labels[~valid_grid] = UNKNOWN_LABEL
    confidence[~valid_grid] = 0

    # A successful global reconciliation may abstain on crossing local evidence
    # and therefore shares neither labels nor confidence semantics with the
    # pre-reconciliation mask.  Do not publish a misleading local sidecar.  On
    # degraded runs mask.npy itself remains the untouched local fallback.
    local_mask_written = False
    global_labels, reconciliation = reconcile_global_packages(
        labels,
        confidence,
        max_gap_inlines=association_max_gap,
        in_place=True,
    )
    global_display_ready = bool(reconciliation.get("global_display_ready", False))
    valid_voxel_count = int(np.count_nonzero(valid_grid) * labels.shape[2])
    unknown_voxel_count = 0
    for inline_index in range(len(global_labels)):
        invalid = global_labels[inline_index] < 0
        confidence[inline_index][invalid] = 0
        unknown_voxel_count += int(
            np.count_nonzero(invalid & valid_grid[inline_index, :, None])
        )
    abstention = {
        "schema_version": "surface-seg.abstention.v1",
        "unknown_label": UNKNOWN_LABEL,
        "confidence_threshold": confidence_threshold,
        "valid_voxel_count": valid_voxel_count,
        "unknown_voxel_count": unknown_voxel_count,
        "unknown_fraction": (
            float(unknown_voxel_count / valid_voxel_count)
            if valid_voxel_count
            else 1.0
        ),
        "invalid_grid_voxel_count": int(global_labels.size - valid_voxel_count),
    }
    horizon_payload: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ] | None = None
    horizon_display_receipt: dict[str, Any] = {
        "schema_version": HORIZON_DISPLAY_GATE_SCHEMA,
        "minimum_finite_trace_fraction": (
            minimum_horizon_finite_trace_fraction
        ),
        "minimum_largest_component_fraction": (
            minimum_horizon_largest_component_fraction
        ),
        "surface_connectivity": 4,
        "volume_connectivity_analogue": 6,
        "finite_trace_fraction_denominator": "dense_inline_xline_grid",
        "axis_support_fraction_denominator": "dense_axis_count",
        "largest_component_fraction_denominator": "finite_trace_count",
        "raw_horizon_count": 0,
        "display_horizon_count": 0,
        "suppressed_horizon_ids": [],
        "horizon_surface_receipts": [],
    }
    if global_display_ready:
        horizon_payload = extract_horizon_surfaces(
            global_labels,
            confidence,
            int(reconciliation["global_package_count"]),
            minimum_finite_trace_fraction=(
                minimum_horizon_finite_trace_fraction
            ),
            minimum_largest_component_fraction=(
                minimum_horizon_largest_component_fraction
            ),
        )
        horizon_display_receipt = horizon_payload[4]
    reconciliation["horizon_display_gate"] = {
        key: horizon_display_receipt[key]
        for key in (
            "schema_version",
            "minimum_finite_trace_fraction",
            "minimum_largest_component_fraction",
            "surface_connectivity",
            "volume_connectivity_analogue",
            "finite_trace_fraction_denominator",
            "axis_support_fraction_denominator",
            "largest_component_fraction_denominator",
        )
    }
    reconciliation["horizon_surface_receipts"] = horizon_display_receipt[
        "horizon_surface_receipts"
    ]
    reconciliation["display_horizon_count"] = horizon_display_receipt[
        "display_horizon_count"
    ]
    reconciliation["suppressed_horizon_ids"] = horizon_display_receipt[
        "suppressed_horizon_ids"
    ]

    print("[5/6] Writing masks", flush=True)
    # Backward-compatible artifact name; its semantics are now the more useful
    # globally reconciled package ids.  The global alias points to this same file
    # in metadata to avoid doubling large-volume disk use.
    np.save(output_dir / "mask.npy", global_labels)
    np.save(output_dir / "confidence.npy", confidence)
    horizon_path: Path | None = None
    if horizon_payload is not None:
        (
            horizon_depths,
            horizon_confidence,
            horizon_ids,
            lower_package_ids,
            _horizon_display_receipt,
        ) = horizon_payload
        horizon_path = output_dir / "horizon_surfaces.npz"
        np.savez_compressed(
            horizon_path,
            depth_samples=horizon_depths,
            confidence=horizon_confidence,
            horizon_ids=horizon_ids,
            lower_package_ids=lower_package_ids,
            inline_values=np.asarray(
                geometry.inline_values[: len(global_labels)], dtype=np.int64
            ),
            xline_values=np.asarray(geometry.xline_values, dtype=np.int64),
            sample_interval_us=np.asarray(
                geometry.sample_interval_us
                if geometry.sample_interval_us is not None
                else -1,
                dtype=np.int64,
            ),
        )
    reconciliation["artifacts"] = {
        "global_mask_npy": (
            str((output_dir / "mask.npy").resolve()) if global_display_ready else None
        ),
        "local_mask_npy": (
            str((output_dir / "local_mask.npy").resolve())
            if local_mask_written
            else None
        ),
        "horizon_surfaces_npz": (
            str(horizon_path.resolve()) if horizon_path is not None else None
        ),
    }
    (output_dir / "global_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mask_sgy_written = False
    if (
        not args.no_mask_sgy
        and args.max_inlines is None
        and len(labels) == full_inline_count
    ):
        write_mask_segy(
            args.input.resolve(), output_dir / "mask.sgy", global_labels, geometry
        )
        mask_sgy_written = True

    print("[6/6] Rendering representative color slices", flush=True)
    selected = select_visualizations(slice_rows, args.num_visualizations)
    labels = global_labels
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
            "confidence_abstention": confidence_threshold,
            "minimum_horizon_finite_trace_fraction": (
                minimum_horizon_finite_trace_fraction
            ),
            "minimum_horizon_largest_component_fraction": (
                minimum_horizon_largest_component_fraction
            ),
        },
        "preprocessing_policy": preprocessing_policy,
        "nonuniform_resize": bool(window_receipt["nonuniform_resize"]),
        "aspect_preserving": bool(window_receipt["aspect_preserving"]),
        "training_preprocess_compatible": bool(
            window_receipt["training_preprocess_compatible"]
        ),
        "label_semantics": {
            "mask_npy": (
                "global_ordered_package_id"
                if global_display_ready
                else "local_per_inline_package_id"
            ),
            "local_mask_npy": "not_persisted; mask_npy_is_local_only_on_degraded_runs",
            "unknown_label": UNKNOWN_LABEL,
        },
        "window_inference": window_receipt,
        "abstention": abstention,
        "global_reconciliation": reconciliation,
        "prior_compatibility_mode": "segformer-base-as-refine-prior",
        "visualization_backend": (
            "matplotlib" if _MATPLOTLIB_AVAILABLE else "pillow"
        ),
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
            "local_mask_npy": (
                str((output_dir / "local_mask.npy").resolve())
                if local_mask_written
                else None
            ),
            "global_mask_npy": (
                str((output_dir / "mask.npy").resolve())
                if global_display_ready
                else None
            ),
            "mask_sgy": (
                str((output_dir / "mask.sgy").resolve())
                if mask_sgy_written
                else None
            ),
            "confidence_npy": str((output_dir / "confidence.npy").resolve()),
            "horizon_surfaces_npz": (
                str(horizon_path.resolve()) if horizon_path is not None else None
            ),
            "global_reconciliation_json": str(
                (output_dir / "global_reconciliation.json").resolve()
            ),
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
