"""CIGVis payloads for seismic stratigraphic surface segmentation.

The upstream model writes ``mask.npy`` and ``confidence.npy`` in
``[INLINE, CROSSLINE, SAMPLE]`` order.  The platform and its lightweight
visualization cache use ``[Z, INLINE, CROSSLINE]`` instead.  This module keeps
the model outputs memory-mapped, samples only the indices requested by
``SegySliceCache``, and performs the axis conversion on that bounded sample.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .faultseg_visualization import (
    AXES_ZYX,
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
)
from .surface_horizon_display_contract import (
    validate_surface_horizon_display_contract,
)


AXES_ICS = ("INLINE", "CROSSLINE", "SAMPLE")
SURFACE_SEG_MODEL_ID = "seismic_surface_seg"
GLOBAL_RECONCILIATION_SCHEMA = "surface-seg.global-reconciliation.v1"
_HORIZON_COLORS = (
    "#ef4444",
    "#f59e0b",
    "#10b981",
    "#06b6d4",
    "#3b82f6",
    "#8b5cf6",
    "#ec4899",
    "#84cc16",
)

_DISPLAY_CONTINUITY_SCHEMA = "surface-seg.display-continuity.v2"
_DISPLAY_CONTINUITY_MAX_ENDPOINT_DELTA_SAMPLES = 8.0
_DISPLAY_CONTINUITY_MIN_SPIKE_RESIDUAL_SAMPLES = 16.0
_DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES = 8.0
_DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS = 1

_MAXIMUM_UNIFORM_SCALE_POLICY = "maximum-uniform-scale-within-tile-budget"
_CHECKPOINT_NATIVE_PREPROCESSING_POLICY = "checkpoint-native-anisotropic-resize"
_EXPERIMENTAL_ASPECT_PRESERVING_POLICY = (
    "experimental-bounded-aspect-preserving-overlap"
)
_WindowPlanItem = tuple[
    tuple[int, int, int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    float,
]


def _window_axis_bounds(
    length: int, count: int, overlap: float
) -> list[tuple[int, int]]:
    """Mirror the sealed producer's deterministic axis partitioning."""

    if count == 1:
        return [(0, length)]
    coverage_factor = count - (count - 1) * overlap
    span = min(length, int(np.ceil(length / coverage_factor)))
    starts = np.rint(np.linspace(0, length - span, count)).astype(np.int64)
    return [(int(start), min(int(start) + span, length)) for start in starts]


def _maximum_uniform_scale_plan(
    native_shape: tuple[int, int],
    model_window: tuple[int, int],
    *,
    overlap: float,
    max_tiles: int,
    minimum_uniform_scale: float,
) -> list[_WindowPlanItem] | None:
    """Independently reproduce the producer's quality-first tile plan.

    This implementation intentionally lives in the verifier module instead of
    importing the inference runtime. A receipt therefore cannot attest its
    own policy merely by copying the producer's policy string.
    """

    height, width = native_shape
    model_height, model_width = model_window
    candidates: list[tuple[float, int, int, int]] = []
    for rows in range(1, max_tiles + 1):
        for columns in range(1, max_tiles // rows + 1):
            count = rows * columns
            y_bounds = _window_axis_bounds(height, rows, overlap)
            x_bounds = _window_axis_bounds(width, columns, overlap)
            worst_height = max(stop - start for start, stop in y_bounds)
            worst_width = max(stop - start for start, stop in x_bounds)
            scale = min(
                model_height / worst_height,
                model_width / worst_width,
                1.0,
            )
            candidates.append((scale, count, rows, columns))
    eligible = [item for item in candidates if item[0] >= minimum_uniform_scale]
    if not eligible:
        return None
    _, _, rows, columns = min(
        eligible, key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    plan: list[_WindowPlanItem] = []
    for top, bottom in _window_axis_bounds(height, rows, overlap):
        for left, right in _window_axis_bounds(width, columns, overlap):
            tile_height, tile_width = bottom - top, right - left
            scale = min(
                model_height / tile_height,
                model_width / tile_width,
                1.0,
            )
            resized_height = max(1, min(model_height, round(tile_height * scale)))
            resized_width = max(1, min(model_width, round(tile_width * scale)))
            pad_top = (model_height - resized_height) // 2
            pad_left = (model_width - resized_width) // 2
            plan.append(
                (
                    (top, left, bottom, right),
                    (tile_height, tile_width),
                    (resized_height, resized_width),
                    (
                        pad_top,
                        pad_left,
                        model_height - pad_top - resized_height,
                        model_width - pad_left - resized_width,
                    ),
                    float(scale),
                )
            )
    return plan


def _validate_checkpoint_native_window_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_native_shape: tuple[int, int] | None,
    expected_max_tiles: int | None,
    expected_overlap: float | None,
    expected_tile_selection_policy: str | None,
) -> list[str]:
    """Independently verify the checkpoint's whole-Inline 512x512 transform."""

    reasons: list[str] = []

    def integers(value: Any, length: int, *, positive: bool) -> tuple[int, ...] | None:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != length
        ):
            return None
        parsed: list[int] = []
        for raw in value:
            if isinstance(raw, (bool, np.bool_)):
                return None
            try:
                item = int(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if item != raw or (item <= 0 if positive else item < 0):
                return None
            parsed.append(item)
        return tuple(parsed)

    if receipt.get("schema_version") != "surface-seg.window-inference.v1":
        reasons.append("window_inference_schema_invalid")
    if receipt.get("mode") != "whole-inline-anisotropic-resize":
        reasons.append("window_inference_mode_invalid")
    if (
        expected_tile_selection_policy is not None
        and receipt.get("tile_selection_policy") != expected_tile_selection_policy
    ):
        reasons.append("window_inference_selection_policy_invalid")
    if receipt.get("tile_selection_policy") != "whole-inline-single-window":
        reasons.append("window_inference_selection_policy_invalid")
    if (
        receipt.get("whole_inline") is not True
        or receipt.get("tile_stitching") is not False
        or receipt.get("aspect_preserving") is not False
        or receipt.get("nonuniform_resize") is not True
        or receipt.get("training_preprocess_compatible") is not True
        or receipt.get("degraded") is not False
    ):
        reasons.append("window_inference_attestation_invalid")

    native_shape = integers(receipt.get("native_shape_sample_xline"), 2, positive=True)
    checkpoint_native_shape = integers(
        receipt.get("checkpoint_native_shape"), 2, positive=True
    )
    model_window = integers(receipt.get("model_window"), 2, positive=True)
    if native_shape is None or (
        expected_native_shape is not None
        and native_shape != tuple(int(value) for value in expected_native_shape)
    ):
        reasons.append("window_inference_native_shape_invalid")
    if checkpoint_native_shape != (512, 512) or model_window != (512, 512):
        reasons.append("window_inference_model_window_invalid")

    raw_maximum_tiles = receipt.get("max_tiles_per_inline")
    raw_tile_count = receipt.get("tile_count_per_inline")
    try:
        maximum_tiles = int(raw_maximum_tiles)
        tile_count = int(raw_tile_count)
        overlap_fraction = float(receipt.get("overlap_fraction"))
    except (TypeError, ValueError, OverflowError):
        maximum_tiles = tile_count = -1
        overlap_fraction = float("nan")
    if (
        isinstance(raw_maximum_tiles, (bool, np.bool_))
        or isinstance(raw_tile_count, (bool, np.bool_))
        or maximum_tiles != raw_maximum_tiles
        or tile_count != raw_tile_count
        or maximum_tiles != 1
        or tile_count != 1
        or (expected_max_tiles is not None and expected_max_tiles != 1)
    ):
        reasons.append("window_inference_tile_budget_invalid")
    if (
        not np.isfinite(overlap_fraction)
        or overlap_fraction != 0.0
        or (expected_overlap is not None and expected_overlap != 0.0)
    ):
        reasons.append("window_inference_overlap_invalid")

    mapping = receipt.get("mapping")
    if (
        not isinstance(mapping, Sequence)
        or isinstance(mapping, (str, bytes))
        or len(mapping) != 1
        or native_shape is None
    ):
        reasons.append("window_inference_mapping_invalid")
        return list(dict.fromkeys(reasons))
    raw_mapping = mapping[0]
    if not isinstance(raw_mapping, Mapping):
        reasons.append("window_inference_mapping_invalid")
        return list(dict.fromkeys(reasons))
    bounds = integers(raw_mapping.get("source_bounds_sample_xline"), 4, positive=False)
    mapped_native = integers(raw_mapping.get("native_shape"), 2, positive=True)
    resized = integers(raw_mapping.get("resized_shape"), 2, positive=True)
    restored = integers(raw_mapping.get("restore_shape"), 2, positive=True)
    padding = integers(
        raw_mapping.get("padding_top_left_bottom_right"), 4, positive=False
    )
    try:
        scale_y = float(raw_mapping.get("scale_y"))
        scale_x = float(raw_mapping.get("scale_x"))
    except (TypeError, ValueError, OverflowError):
        scale_y = scale_x = float("nan")
    height, width = native_shape
    if (
        bounds != (0, 0, height, width)
        or mapped_native != native_shape
        or resized != (512, 512)
        or restored != native_shape
        or padding != (0, 0, 0, 0)
    ):
        reasons.append("window_inference_mapping_geometry_invalid")
    if (
        not np.isfinite(scale_y)
        or not np.isfinite(scale_x)
        or not np.isclose(scale_y, 512.0 / height, rtol=1e-9, atol=1e-12)
        or not np.isclose(scale_x, 512.0 / width, rtol=1e-9, atol=1e-12)
    ):
        reasons.append("window_inference_anisotropic_scale_invalid")
    return list(dict.fromkeys(reasons))


def validate_surface_window_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_native_shape: tuple[int, int] | None = None,
    expected_max_tiles: int | None = None,
    expected_overlap: float | None = None,
    expected_minimum_uniform_scale: float | None = None,
    expected_preprocessing_policy: str | None = None,
    expected_tile_selection_policy: str | None = None,
    hard_max_tiles: int = 4,
) -> list[str]:
    """Recompute the selected preprocessing transform fail-closed."""

    reasons: list[str] = []

    preprocessing_policy = receipt.get("preprocessing_policy")
    if (
        expected_preprocessing_policy is not None
        and preprocessing_policy != expected_preprocessing_policy
    ):
        reasons.append("window_inference_preprocessing_policy_invalid")
    if preprocessing_policy == _CHECKPOINT_NATIVE_PREPROCESSING_POLICY:
        reasons.extend(
            _validate_checkpoint_native_window_receipt(
                receipt,
                expected_native_shape=expected_native_shape,
                expected_max_tiles=expected_max_tiles,
                expected_overlap=expected_overlap,
                expected_tile_selection_policy=expected_tile_selection_policy,
            )
        )
        return list(dict.fromkeys(reasons))
    if preprocessing_policy != _EXPERIMENTAL_ASPECT_PRESERVING_POLICY:
        reasons.append("window_inference_preprocessing_policy_invalid")

    def integer_sequence(
        value: Any,
        length: int,
        *,
        positive: bool,
    ) -> tuple[int, ...] | None:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != length
        ):
            return None
        parsed: list[int] = []
        for raw in value:
            if isinstance(raw, (bool, np.bool_)):
                return None
            try:
                item = int(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if item != raw or (item <= 0 if positive else item < 0):
                return None
            parsed.append(item)
        return tuple(parsed)

    if receipt.get("schema_version") != "surface-seg.window-inference.v1":
        reasons.append("window_inference_schema_invalid")
    if receipt.get("mode") != "bounded-aspect-preserving-overlap":
        reasons.append("window_inference_mode_invalid")
    if (
        expected_tile_selection_policy is not None
        and receipt.get("tile_selection_policy") != expected_tile_selection_policy
    ):
        reasons.append("window_inference_selection_policy_invalid")
    if (
        receipt.get("aspect_preserving") is not True
        or receipt.get("nonuniform_resize") is not False
        or receipt.get("training_preprocess_compatible") is not False
        or receipt.get("degraded") is not False
    ):
        reasons.append("window_inference_attestation_invalid")

    native_shape = integer_sequence(
        receipt.get("native_shape_sample_xline"), 2, positive=True
    )
    model_window = integer_sequence(receipt.get("model_window"), 2, positive=True)
    if native_shape is None or (
        expected_native_shape is not None
        and native_shape != tuple(int(value) for value in expected_native_shape)
    ):
        reasons.append("window_inference_native_shape_invalid")
    if model_window != (512, 512):
        reasons.append("window_inference_model_window_invalid")

    raw_maximum_tiles = receipt.get("max_tiles_per_inline")
    raw_tile_count = receipt.get("tile_count_per_inline")
    try:
        maximum_tiles = int(raw_maximum_tiles)
        tile_count = int(raw_tile_count)
        overlap_fraction = float(receipt.get("overlap_fraction"))
        minimum_scale = float(receipt.get("minimum_uniform_scale"))
        effective_scale = float(receipt.get("minimum_effective_scale"))
    except (TypeError, ValueError, OverflowError):
        maximum_tiles = tile_count = -1
        overlap_fraction = minimum_scale = effective_scale = float("nan")
    if (
        isinstance(raw_maximum_tiles, (bool, np.bool_))
        or isinstance(raw_tile_count, (bool, np.bool_))
        or maximum_tiles != raw_maximum_tiles
        or tile_count != raw_tile_count
        or maximum_tiles < 1
        or maximum_tiles > hard_max_tiles
        or (expected_max_tiles is not None and maximum_tiles != expected_max_tiles)
        or tile_count < 1
        or tile_count > maximum_tiles
    ):
        reasons.append("window_inference_tile_budget_invalid")
    if (
        not np.isfinite(overlap_fraction)
        or not 0.0 <= overlap_fraction < 0.5
        or (
            expected_overlap is not None
            and not np.isclose(overlap_fraction, expected_overlap, rtol=0.0, atol=1e-9)
        )
    ):
        reasons.append("window_inference_overlap_invalid")
    if (
        not np.isfinite(minimum_scale)
        or not 0.0 < minimum_scale <= 1.0
        or (
            expected_minimum_uniform_scale is not None
            and not np.isclose(
                minimum_scale,
                expected_minimum_uniform_scale,
                rtol=0.0,
                atol=1e-9,
            )
        )
        or not np.isfinite(effective_scale)
        or not minimum_scale <= effective_scale <= 1.0
    ):
        reasons.append("window_inference_scale_policy_invalid")

    mapping = receipt.get("mapping")
    if (
        not isinstance(mapping, Sequence)
        or isinstance(mapping, (str, bytes))
        or len(mapping) != tile_count
        or native_shape is None
        or model_window is None
    ):
        reasons.append("window_inference_mapping_invalid")
        return list(dict.fromkeys(reasons))

    native_height, native_width = native_shape
    model_height, model_width = model_window
    rectangles: list[tuple[int, int, int, int]] = []
    observed_scales: list[float] = []
    observed_plan: list[_WindowPlanItem] = []
    for raw_tile in mapping:
        if not isinstance(raw_tile, Mapping):
            reasons.append("window_inference_mapping_invalid")
            continue
        bounds = integer_sequence(
            raw_tile.get("source_bounds_sample_xline"), 4, positive=False
        )
        tile_native = integer_sequence(raw_tile.get("native_shape"), 2, positive=True)
        resized = integer_sequence(raw_tile.get("resized_shape"), 2, positive=True)
        padding = integer_sequence(
            raw_tile.get("padding_top_left_bottom_right"), 4, positive=False
        )
        try:
            scale = float(raw_tile.get("uniform_scale"))
        except (TypeError, ValueError, OverflowError):
            scale = float("nan")
        if bounds is None or tile_native is None or resized is None or padding is None:
            reasons.append("window_inference_mapping_invalid")
            continue
        top, left, bottom, right = bounds
        tile_height, tile_width = tile_native
        resized_height, resized_width = resized
        pad_top, pad_left, pad_bottom, pad_right = padding
        if (
            not 0 <= top < bottom <= native_height
            or not 0 <= left < right <= native_width
            or tile_native != (bottom - top, right - left)
            or resized_height > model_height
            or resized_width > model_width
            or pad_top + resized_height + pad_bottom != model_height
            or pad_left + resized_width + pad_right != model_width
            or abs(pad_top - pad_bottom) > 1
            or abs(pad_left - pad_right) > 1
        ):
            reasons.append("window_inference_mapping_geometry_invalid")
            continue
        expected_scale = min(
            model_height / tile_height,
            model_width / tile_width,
            1.0,
        )
        if (
            not np.isfinite(scale)
            or not minimum_scale <= scale <= 1.0
            or not np.isclose(scale, expected_scale, rtol=1e-6, atol=1e-9)
            or abs(resized_height - tile_height * scale) > 1.0
            or abs(resized_width - tile_width * scale) > 1.0
        ):
            reasons.append("window_inference_aspect_mapping_invalid")
            continue
        rectangles.append((top, left, bottom, right))
        observed_scales.append(scale)
        observed_plan.append((bounds, tile_native, resized, padding, scale))

    if len(rectangles) != tile_count:
        reasons.append("window_inference_mapping_invalid")
    else:
        y_edges = sorted(
            {0, native_height}
            | {edge for top, _, bottom, _ in rectangles for edge in (top, bottom)}
        )
        coverage_valid = y_edges[0] == 0 and y_edges[-1] == native_height
        for band_top, band_bottom in zip(y_edges, y_edges[1:]):
            intervals = sorted(
                (left, right)
                for top, left, bottom, right in rectangles
                if top <= band_top and bottom >= band_bottom
            )
            cursor = 0
            for left, right in intervals:
                if left > cursor:
                    coverage_valid = False
                    break
                cursor = max(cursor, right)
            if cursor < native_width:
                coverage_valid = False
            if not coverage_valid:
                break
        if not coverage_valid:
            reasons.append("window_inference_native_coverage_incomplete")
    if observed_scales and not np.isclose(
        effective_scale,
        min(observed_scales),
        rtol=1e-6,
        atol=1e-9,
    ):
        reasons.append("window_inference_effective_scale_invalid")

    claims_maximum_uniform_scale = (
        receipt.get("tile_selection_policy") == _MAXIMUM_UNIFORM_SCALE_POLICY
        or expected_tile_selection_policy == _MAXIMUM_UNIFORM_SCALE_POLICY
    )
    planning_inputs_valid = (
        native_shape is not None
        and model_window is not None
        and 1 <= maximum_tiles <= hard_max_tiles
        and np.isfinite(overlap_fraction)
        and 0.0 <= overlap_fraction < 0.5
        and np.isfinite(minimum_scale)
        and 0.0 < minimum_scale <= 1.0
    )
    if claims_maximum_uniform_scale and planning_inputs_valid:
        expected_plan = _maximum_uniform_scale_plan(
            native_shape,
            model_window,
            overlap=overlap_fraction,
            max_tiles=maximum_tiles,
            minimum_uniform_scale=minimum_scale,
        )
        plan_matches = expected_plan is not None and len(observed_plan) == len(
            expected_plan
        )
        if plan_matches:
            for observed, expected in zip(observed_plan, expected_plan, strict=True):
                if observed[:4] != expected[:4] or not np.isclose(
                    observed[4], expected[4], rtol=1e-6, atol=1e-9
                ):
                    plan_matches = False
                    break
        if not plan_matches:
            reasons.append("window_inference_selection_plan_invalid")

    return list(dict.fromkeys(reasons))


def _shape3(value: Sequence[Any], name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three values")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integers, not booleans")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain integers") from exc
        if integer != item or integer <= 0:
            raise ValueError(f"{name} must contain positive integers")
        result.append(integer)
    return tuple(result)  # type: ignore[return-value]


def _result_document(
    result_or_metadata: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(result_or_metadata, Mapping):
        result: Mapping[str, Any] = result_or_metadata
        if "prediction" in result and "model_id" not in result:
            nested = result.get("prediction")
            if not isinstance(nested, Mapping):
                raise ValueError("prediction wrapper must contain a mapping")
            result = nested
        return result, None
    metadata_path = Path(result_or_metadata).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SurfaceSeg metadata does not exist: {metadata_path}")
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("SurfaceSeg metadata JSON must contain an object")
    return document, metadata_path.parent


def _metadata_path(
    value: Any,
    *,
    base: Path | None,
    label: str,
    suffixes: set[str],
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"SurfaceSeg result is missing {label}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise FileNotFoundError(f"invalid {label}: {path}")
    return path


def _encode_uint8(array: np.ndarray, **metadata: Any) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "axes": list(AXES_ZYX),
        "encoding": "base64-uint8",
        "values": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        **metadata,
    }


def _encode_discrete_labels(
    labels_zyx: np.ndarray,
    *,
    invalid_label: int,
    reported_max: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric = np.asarray(labels_zyx)
    if not np.issubdtype(numeric.dtype, np.integer):
        if not np.issubdtype(numeric.dtype, np.number):
            raise ValueError("sampled SurfaceSeg mask must be numeric")
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
            raise ValueError("sampled SurfaceSeg mask must contain integer labels")
    labels = numeric.astype(np.int64, copy=False)
    if np.any((labels < 0) & (labels != invalid_label)):
        raise ValueError(
            f"sampled SurfaceSeg mask contains a negative label other than {invalid_label}"
        )
    valid = labels != invalid_label
    if np.any(labels[valid] > reported_max):
        raise ValueError("sampled SurfaceSeg mask exceeds segmentation.label_range")

    # Reserve code 0 for missing traces.  Valid labels occupy [32, 255], which
    # lets CIGVis use jet + nearest interpolation while ``excpt=min`` makes
    # only the missing-trace sentinel transparent.  Keeping label 0 away from
    # the transparent endpoint also makes the first valid stratum clearly
    # visible in the whole-volume WebGL transfer function.  If there are more
    # than 224 labels the display is intentionally quantized, while the .npy
    # file
    # remains the authoritative int16 result.
    encoded = np.zeros(labels.shape, dtype=np.uint8)
    if np.any(valid):
        if reported_max <= 0:
            encoded[valid] = np.uint8(255)
        else:
            scaled = 32.0 + labels[valid].astype(np.float64) * (223.0 / reported_max)
            encoded[valid] = np.clip(np.rint(scaled), 32, 255).astype(np.uint8)
    return encoded, {
        "invalidLabel": int(invalid_label),
        "invalidDisplayCode": 0,
        "unknownLabel": int(invalid_label),
        "unknownTransparent": True,
        "labelValueRange": [0, int(reported_max)],
        "displayCodeRange": [32, 255],
        "displayQuantized": bool(reported_max > 223),
    }


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _surface_display_scope(
    segmentation: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    output_inline_count: int,
    horizon_display_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed from local Inline labels to reconciled global labels."""

    raw_reconciliation = segmentation.get("global_reconciliation")
    has_reconciliation = isinstance(raw_reconciliation, Mapping) and bool(
        raw_reconciliation
    )
    reconciliation = raw_reconciliation
    declared_consistent = segmentation.get("cross_inline_consistent") is True
    reasons: list[str] = []
    if not declared_consistent:
        reasons.append("cross_inline_consistency_not_declared")
    if not isinstance(reconciliation, Mapping):
        reasons.append("global_reconciliation_missing")
        reconciliation = {}
    elif reconciliation.get("schema_version") != GLOBAL_RECONCILIATION_SCHEMA:
        reasons.append("global_reconciliation_schema_invalid")
    if reconciliation.get("non_crossing_verified") is not True:
        reasons.append("global_non_crossing_not_verified")
    if reconciliation.get("order_graph_acyclic") is not True:
        reasons.append("global_order_graph_not_acyclic")
    if reconciliation.get("global_display_ready") is not True:
        reasons.append("global_display_ready_not_attested")
    if reconciliation.get("output_semantics") != "global_ordered_package_id":
        reasons.append("global_output_semantics_invalid")
    if not str(reconciliation.get("association_scope") or "").strip():
        reasons.append("global_association_scope_missing")

    try:
        package_count = _nonnegative_int(
            reconciliation.get("global_package_count"),
            "segmentation.global_reconciliation.global_package_count",
        )
        horizon_count = _nonnegative_int(
            reconciliation.get("global_horizon_count"),
            "segmentation.global_reconciliation.global_horizon_count",
        )
    except ValueError:
        package_count = 0
        horizon_count = 0
        reasons.append("global_reconciliation_counts_invalid")
    # A single package has no interface to draw.  Treating 1 package / 0
    # horizons as a successful global interpretation produced the uniform blue
    # screen that hid the seismic reflectors on real F3 data.
    if package_count < 2 or horizon_count < 1 or horizon_count != package_count - 1:
        reasons.append("global_reconciliation_counts_invalid")
    try:
        processed_inline_count = _nonnegative_int(
            reconciliation.get("processed_inline_count"),
            "segmentation.global_reconciliation.processed_inline_count",
        )
        matched_transition_count = _nonnegative_int(
            reconciliation.get("matched_transition_count"),
            "segmentation.global_reconciliation.matched_transition_count",
        )
    except ValueError:
        processed_inline_count = matched_transition_count = 0
    transition_receipts = reconciliation.get("transition_receipts")
    if (
        output_inline_count < 2
        or processed_inline_count != output_inline_count
        or matched_transition_count <= 0
        or not isinstance(transition_receipts, Sequence)
        or isinstance(transition_receipts, (str, bytes))
        or len(transition_receipts) != output_inline_count
    ):
        reasons.append("global_transition_evidence_insufficient")
    if not outputs.get("global_mask_npy"):
        reasons.append("global_mask_artifact_missing")
    if horizon_count > 0 and not outputs.get("horizon_surfaces_npz"):
        reasons.append("global_horizon_artifact_missing")

    display_contract = dict(horizon_display_contract or {})
    if reconciliation.get("global_display_ready") is True:
        if display_contract.get("valid") is not True:
            reasons.extend(
                list(
                    display_contract.get(
                        "reason_codes", ["horizon_display_contract_invalid"]
                    )
                )
            )
        try:
            display_horizon_count = _nonnegative_int(
                display_contract.get("display_horizon_count"),
                "horizon_display_contract.display_horizon_count",
            )
        except ValueError:
            display_horizon_count = 0
            reasons.append("horizon_display_counts_invalid")
        suppressed_horizon_ids = list(
            display_contract.get("suppressed_horizon_ids") or []
        )
    else:
        display_horizon_count = 0
        suppressed_horizon_ids = []
    no_display_eligible_horizons = bool(
        display_contract.get("valid") is True
        and horizon_count > 0
        and display_horizon_count == 0
    )
    if no_display_eligible_horizons:
        reasons.append("no_display_eligible_horizons")

    global_consistent = not reasons
    sealed_local_fallback = bool(
        has_reconciliation
        and reconciliation.get("global_display_ready") is False
        and reconciliation.get("output_semantics") == "local_inline_fallback"
        and (outputs.get("local_mask_npy") or outputs.get("mask_npy"))
    )
    insufficient_global_horizons = bool(
        has_reconciliation and (package_count < 2 or horizon_count < 1)
    )
    background_only = bool(
        (insufficient_global_horizons or no_display_eligible_horizons)
        and not global_consistent
        and not sealed_local_fallback
    )
    labels_renderable = not background_only
    if background_only and no_display_eligible_horizons:
        display_notice = (
            f"全部 {horizon_count} 个原始层位面均未通过逐面支持门；"
            "原始 NPZ 与层位 ID 已保留，当前仅显示灰度地震背景。"
        )
    elif background_only:
        display_notice = (
            "层位结果未通过全局层序与层界校验；已隐藏标签和置信度，"
            "当前仅显示灰度地震背景，请重新预测。"
        )
    elif suppressed_horizon_ids:
        display_notice = (
            f"显示 {display_horizon_count}/{horizon_count} 个全局层位面；"
            f"已抑制 {len(suppressed_horizon_ids)} 个低支持面，原始 ID 保留。"
        )
    else:
        display_notice = ""
    return {
        "globalConsistent": global_consistent,
        "labelsRenderable": labels_renderable,
        "backgroundOnly": background_only,
        "displayNotice": display_notice,
        "labelScope": (
            "global_packages"
            if global_consistent
            else "unavailable"
            if background_only
            else "inline_local"
        ),
        "legendTitle": (
            "全局层间编号"
            if global_consistent
            else "层位结果不可显示"
            if background_only
            else "局部分层（仅当前 Inline）"
        ),
        # The model predicts per-Inline stratigraphic regions.  Even after
        # global reconciliation, the clearest primary interpretation remains
        # the seismic Inline with interval fill and horizon boundaries.  A
        # single-horizon plan map is useful QC, but must not replace that
        # primary segmentation view.
        "defaultPlane": "i",
        "allowedPlanes": (
            ["horizon", "z", "i", "x", "interval-i", "interval-x"]
            if global_consistent
            else ["z", "i", "x"]
            if background_only
            else ["i"]
        ),
        "disabledPlaneReasons": (
            {}
            if global_consistent
            else {
                "z": "未完成跨 Inline 全局关联，时间平面标签没有统一语义",
                "x": "未完成跨 Inline 全局关联，Crossline 标签没有统一语义",
                "horizon": "没有通过非交叉校验的全局层位面",
                "interval-i": "没有通过非交叉校验的全局层位面",
                "interval-x": "没有通过非交叉校验的全局层位面",
            }
        ),
        "reasonCodes": list(dict.fromkeys(reasons)),
        "globalPackageCount": package_count if global_consistent else 0,
        "globalHorizonCount": horizon_count if global_consistent else 0,
        "rawGlobalHorizonCount": horizon_count,
        "displayHorizonCount": display_horizon_count,
        "suppressedHorizonIds": suppressed_horizon_ids,
        "eligibleHorizonIds": list(display_contract.get("eligible_horizon_ids") or []),
        "horizonDisplayContractValid": display_contract.get("valid") is True,
        "reconciliation": dict(reconciliation),
    }


def _adjacent_inline_stats(values: np.ndarray) -> dict[str, int | float | None]:
    deltas = np.abs(np.diff(np.asarray(values, dtype=np.float64), axis=1))
    finite = deltas[np.isfinite(deltas)]
    if not finite.size:
        return {
            "pairCount": 0,
            "p50Samples": None,
            "p95Samples": None,
            "maxSamples": None,
        }
    return {
        "pairCount": int(finite.size),
        "p50Samples": float(np.percentile(finite, 50.0)),
        "p95Samples": float(np.percentile(finite, 95.0)),
        "maxSamples": float(np.max(finite)),
    }


def _derive_inline_display_continuity(
    sampled_depths: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair only isolated Inline display artifacts on the bounded grid.

    The sealed horizon artifact is never mutated.  A finite candidate is
    accepted only when its immediate Inline neighbours agree within a small
    source-sample tolerance, and its correction is hard-limited.  Missing
    values are never filled.  Consequently the operation cannot bridge a gap
    or a fault-like jump.  Every accepted value is also checked against the raw
    horizons immediately above and below; the complete derived stack is then
    rechecked for strict non-crossing before it is returned.
    """

    raw = np.asarray(sampled_depths, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(
            "SurfaceSeg sampled horizon depths must use [HORIZON, INLINE, CROSSLINE]"
        )
    display = raw.copy()
    spike_counts = np.zeros(raw.shape[0], dtype=np.int64)
    rejected_counts = np.zeros(raw.shape[0], dtype=np.int64)

    if raw.shape[1] >= 3:
        for horizon_index in range(raw.shape[0]):
            left = raw[horizon_index, :-2]
            center = raw[horizon_index, 1:-1]
            right = raw[horizon_index, 2:]
            neighbour_target = (left + right) * 0.5
            neighbours_continue = (
                np.isfinite(left)
                & np.isfinite(right)
                & (
                    np.abs(left - right)
                    <= _DISPLAY_CONTINUITY_MAX_ENDPOINT_DELTA_SAMPLES
                )
            )
            spike_candidate = (
                neighbours_continue
                & np.isfinite(center)
                & (
                    np.abs(center - neighbour_target)
                    >= _DISPLAY_CONTINUITY_MIN_SPIKE_RESIDUAL_SAMPLES
                )
                & (
                    np.abs(center - left)
                    >= _DISPLAY_CONTINUITY_MIN_SPIKE_RESIDUAL_SAMPLES
                )
                & (
                    np.abs(center - right)
                    >= _DISPLAY_CONTINUITY_MIN_SPIKE_RESIDUAL_SAMPLES
                )
            )
            correction = np.clip(
                neighbour_target - center,
                -_DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES,
                _DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES,
            )
            candidate = center + correction
            proposed = spike_candidate
            safe = proposed.copy()
            if horizon_index:
                upper = raw[horizon_index - 1, 1:-1]
                safe &= ~np.isfinite(upper) | (candidate > upper)
            if horizon_index + 1 < raw.shape[0]:
                lower = raw[horizon_index + 1, 1:-1]
                safe &= ~np.isfinite(lower) | (candidate < lower)

            target = display[horizon_index, 1:-1]
            target[safe] = candidate[safe]
            spike_counts[horizon_index] = int(np.count_nonzero(safe & spike_candidate))
            rejected_counts[horizon_index] = int(np.count_nonzero(proposed & ~safe))

    comparable = np.isfinite(display[:-1]) & np.isfinite(display[1:])
    non_crossing = not bool(np.any(display[1:][comparable] <= display[:-1][comparable]))
    finite_values = display[np.isfinite(display)]
    finite_values_safe = not bool(np.any(np.isinf(display))) and bool(
        not finite_values.size
        or (np.min(finite_values) >= 0.0 and np.max(finite_values) <= np.nanmax(raw))
    )
    failed_closed = not (non_crossing and finite_values_safe)
    if failed_closed:
        display = raw.copy()
        spike_counts.fill(0)

    per_horizon = []
    for horizon_index in range(raw.shape[0]):
        per_horizon.append(
            {
                "horizonIndex": horizon_index,
                "rawFiniteCount": int(
                    np.count_nonzero(np.isfinite(raw[horizon_index]))
                ),
                "displayFiniteCount": int(
                    np.count_nonzero(np.isfinite(display[horizon_index]))
                ),
                "spikeReplacementCount": int(spike_counts[horizon_index]),
                "unsafeCandidateRejectionCount": int(rejected_counts[horizon_index]),
                "rawAdjacentInline": _adjacent_inline_stats(
                    raw[horizon_index : horizon_index + 1]
                ),
                "displayAdjacentInline": _adjacent_inline_stats(
                    display[horizon_index : horizon_index + 1]
                ),
            }
        )

    repair_count = int(spike_counts.sum())
    metadata: dict[str, Any] = {
        "schemaVersion": _DISPLAY_CONTINUITY_SCHEMA,
        "mode": "bounded-isolated-inline-spike-display-repair",
        "axis": "INLINE",
        "applied": bool(repair_count and not failed_closed),
        "rawArtifactUnchanged": True,
        "rawHorizonIdsUnchanged": True,
        "confidenceGridUnchanged": True,
        "finiteSupportPreserved": bool(
            np.array_equal(np.isfinite(raw), np.isfinite(display))
        ),
        "cumulative": False,
        "sourceInput": "sealed-raw-grid-on-every-build",
        "parameters": {
            "passes": 1,
            "gapFillEnabled": False,
            "maxEndpointDeltaSamples": (_DISPLAY_CONTINUITY_MAX_ENDPOINT_DELTA_SAMPLES),
            "minSpikeResidualSamples": (_DISPLAY_CONTINUITY_MIN_SPIKE_RESIDUAL_SAMPLES),
            "maxCorrectionSamples": _DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES,
            "target": "mean-of-immediate-inline-neighbours",
        },
        "safety": {
            "failedClosed": failed_closed,
            "nonCrossingVerified": non_crossing,
            "derivedCrossSurfaceStrictOrderVerified": non_crossing,
            "finiteMaskIdentityVerified": bool(
                np.array_equal(np.isfinite(raw), np.isfinite(display))
            ),
            "sourceGridGapFillEnabled": False,
            "displayPolylineSingleGapBridgeOrderCheckRequired": True,
            "displayPolylineEndpointExtrapolationEnabled": False,
        },
        "metrics": {
            "rawFiniteCount": int(np.count_nonzero(np.isfinite(raw))),
            "displayFiniteCount": int(np.count_nonzero(np.isfinite(display))),
            "spikeReplacementCount": int(spike_counts.sum()),
            "replacementFractionOfFinite": (
                float(spike_counts.sum() / np.count_nonzero(np.isfinite(raw)))
                if np.count_nonzero(np.isfinite(raw))
                else 0.0
            ),
            "replacementPercentageOfFinite": (
                float(100.0 * spike_counts.sum() / np.count_nonzero(np.isfinite(raw)))
                if np.count_nonzero(np.isfinite(raw))
                else 0.0
            ),
            "unsafeCandidateRejectionCount": int(rejected_counts.sum()),
            "rawAdjacentInline": _adjacent_inline_stats(raw),
            "displayAdjacentInline": _adjacent_inline_stats(display),
        },
        "perHorizon": per_horizon,
    }
    return display, metadata


def _sample_global_horizon_surfaces(
    path: Path,
    *,
    output_shape_ics: tuple[int, int, int],
    z_indices: np.ndarray,
    inline_indices: np.ndarray,
    crossline_indices: np.ndarray,
    preview_inline_values: np.ndarray,
    preview_crossline_values: np.ndarray,
    expected_horizon_count: int,
    eligible_horizon_ids: Sequence[int],
    horizon_surface_receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the bounded CIGVis grid from a reconciled horizon NPZ."""

    required = {
        "depth_samples",
        "confidence",
        "horizon_ids",
        "lower_package_ids",
        "inline_values",
        "xline_values",
        "sample_interval_us",
    }
    with np.load(path, allow_pickle=False) as document:
        missing = required.difference(document.files)
        if missing:
            raise ValueError(
                "SurfaceSeg horizon surface artifact is missing keys: "
                + ", ".join(sorted(missing))
            )
        depths = np.asarray(document["depth_samples"], dtype=np.float32)
        confidence = np.asarray(document["confidence"], dtype=np.float32)
        horizon_ids = np.asarray(document["horizon_ids"], dtype=np.int64)
        lower_package_ids = np.asarray(document["lower_package_ids"], dtype=np.int64)
        inline_values = np.asarray(document["inline_values"])
        xline_values = np.asarray(document["xline_values"])

    inline_count, crossline_count, _ = output_shape_ics
    if depths.ndim != 3 or depths.shape[1:] != (inline_count, crossline_count):
        raise ValueError(
            "SurfaceSeg depth_samples must use [HORIZON, INLINE, CROSSLINE]"
        )
    if confidence.shape != depths.shape:
        raise ValueError(
            "SurfaceSeg horizon confidence shape differs from depth_samples"
        )
    horizon_count = int(depths.shape[0])
    if horizon_count != expected_horizon_count:
        raise ValueError(
            "SurfaceSeg horizon count differs from global reconciliation contract"
        )
    if horizon_ids.shape != (horizon_count,) or lower_package_ids.shape != (
        horizon_count,
    ):
        raise ValueError("SurfaceSeg horizon identifiers have invalid shape")
    if len(np.unique(horizon_ids)) != horizon_count:
        raise ValueError("SurfaceSeg horizon identifiers must be unique")
    if inline_values.shape != (inline_count,) or xline_values.shape != (
        crossline_count,
    ):
        raise ValueError("SurfaceSeg horizon coordinate arrays do not match the volume")
    if not np.array_equal(
        inline_values[inline_indices], np.asarray(preview_inline_values)
    ) or not np.array_equal(
        xline_values[crossline_indices], np.asarray(preview_crossline_values)
    ):
        raise ValueError(
            "SurfaceSeg horizon coordinate values do not align with the SEG-Y preview"
        )
    if np.any(np.isinf(depths)) or np.any(np.isinf(confidence)):
        raise ValueError(
            "SurfaceSeg horizon arrays may contain NaN gaps but not infinity"
        )
    finite_confidence = np.isfinite(confidence)
    if np.any(confidence[finite_confidence] < 0.0) or np.any(
        confidence[finite_confidence] > 1.0
    ):
        raise ValueError("SurfaceSeg horizon confidence must stay within [0, 1]")
    for index in range(1, horizon_count):
        upper = depths[index - 1]
        lower = depths[index]
        comparable = np.isfinite(upper) & np.isfinite(lower)
        if np.any(lower[comparable] <= upper[comparable]):
            raise ValueError("SurfaceSeg global horizons cross or touch")

    sampled_depths = depths[:, inline_indices][:, :, crossline_indices]
    sampled_confidence = confidence[:, inline_indices][:, :, crossline_indices]
    display_depths, continuity = _derive_inline_display_continuity(sampled_depths)
    z_positions = np.arange(len(z_indices), dtype=np.float64)
    z_source = np.asarray(z_indices, dtype=np.float64)
    z_steps = np.diff(z_source)
    positive_z_steps = z_steps[np.isfinite(z_steps) & (z_steps > 0.0)]
    native_samples_per_preview_cell = (
        float(np.median(positive_z_steps)) if positive_z_steps.size else 1.0
    )
    max_correction_preview_cells = float(
        _DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES / native_samples_per_preview_cell
    )
    continuity["parameters"].update(
        {
            "nativeSamplesPerPreviewZCell": native_samples_per_preview_cell,
            "maxCorrectionPreviewCells": max_correction_preview_cells,
            "shortGapBridgeMaxMissingPreviewCells": (
                _DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS
            ),
            "profileSingleGapConnection": {
                "plane": "x",
                "displayOnly": True,
                "interpolation": "linear-between-immediate-finite-neighbours",
                "orderSafe": True,
                "strictOrderScope": "all-rendered-horizons-at-gap",
                "endpointExtrapolation": False,
                "intervalXGridFillModified": False,
                "sourceGridModified": False,
            },
        }
    )
    surfaces: list[dict[str, Any]] = []
    eligible_ids = {int(value) for value in eligible_horizon_ids}
    receipts_by_id = {
        int(item["horizon_id"]): dict(item) for item in horizon_surface_receipts
    }
    for index in range(horizon_count):
        raw_depth = sampled_depths[index].astype(np.float64, copy=False)
        display_depth = display_depths[index]
        raw_valid = (
            np.isfinite(raw_depth)
            & (raw_depth >= z_source[0])
            & (raw_depth <= z_source[-1])
        )
        display_valid = (
            np.isfinite(display_depth)
            & (display_depth >= z_source[0])
            & (display_depth <= z_source[-1])
        )
        raw_preview_z = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        display_preview_z = np.full(display_depth.shape, np.nan, dtype=np.float64)
        raw_preview_z[raw_valid] = np.interp(
            raw_depth[raw_valid], z_source, z_positions
        )
        display_preview_z[display_valid] = np.interp(
            display_depth[display_valid], z_source, z_positions
        )
        raw_grid = [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in raw_preview_z
        ]
        cross_inline_display_grid = [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in display_preview_z
        ]
        confidence_grid = [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in sampled_confidence[index]
        ]
        horizon_id = int(horizon_ids[index])
        if horizon_id not in eligible_ids:
            continue
        surfaces.append(
            {
                "id": f"global-horizon-{horizon_id}",
                "name": f"全局层位 H{horizon_id + 1}",
                "kind": "surface",
                "labelScope": "global_horizon",
                "grid": raw_grid,
                "crossInlineDisplayGrid": cross_inline_display_grid,
                "shortGapBridgeMaxMissingPreviewCells": (
                    _DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS
                ),
                "crossInlineMaxCorrectionPreviewCells": (max_correction_preview_cells),
                "confidenceGrid": confidence_grid,
                "lowerPackageId": int(lower_package_ids[index]),
                "color": _HORIZON_COLORS[index % len(_HORIZON_COLORS)],
                "source": str(path),
                "supportReceipt": receipts_by_id[horizon_id],
                "supportReceiptScope": "raw_horizon_artifact",
                "displayContinuity": {
                    **dict(continuity["perHorizon"][index]),
                    "maxCorrectionSamples": (
                        _DISPLAY_CONTINUITY_MAX_CORRECTION_SAMPLES
                    ),
                    "maxCorrectionPreviewCells": max_correction_preview_cells,
                    "shortGapBridgeMaxMissingPreviewCells": (
                        _DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS
                    ),
                    "profileSingleGapConnection": {
                        "plane": "x",
                        "displayOnly": True,
                        "interpolation": (
                            "linear-between-immediate-finite-neighbours"
                        ),
                        "orderSafe": True,
                        "strictOrderScope": "all-rendered-horizons-at-gap",
                        "endpointExtrapolation": False,
                        "intervalXGridFillModified": False,
                        "sourceGridModified": False,
                    },
                },
            }
        )
    return surfaces, continuity


def build_surface_seg_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
    overlay_layer: str = "mask",
    verified_horizon_display_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sparse seismic background with an aligned stratigraphic layer.

    ``overlay_layer`` is intentionally restricted to ``"mask"``.  Confidence
    remains available under ``surfaceSeg`` for bounded quality-control reads,
    but it is not a geological interpretation layer and must never replace the
    deterministic interval labels in the primary viewer.
    """

    result, metadata_base = _result_document(result_or_metadata)
    if result.get("model_id") != SURFACE_SEG_MODEL_ID:
        raise ValueError(
            f"visualization result must use model_id={SURFACE_SEG_MODEL_ID}"
        )
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    segmentation = result.get("segmentation")
    inference = result.get("inference")
    if not isinstance(input_metadata, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("SurfaceSeg result must contain input and outputs mappings")
    if not isinstance(segmentation, Mapping) or not isinstance(inference, Mapping):
        raise ValueError(
            "SurfaceSeg result must contain segmentation and inference mappings"
        )
    input_axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    segmentation_axes = tuple(
        str(axis).upper()
        for axis in segmentation.get("axes", input_metadata.get("axes", ()))
    )
    if input_axes != AXES_ICS or segmentation_axes != AXES_ICS:
        raise ValueError(
            f"SurfaceSeg axes must be {AXES_ICS}, got input={input_axes}, "
            f"segmentation={segmentation_axes}"
        )

    output_shape_ics = _shape3(
        segmentation.get("shape_ics", input_metadata.get("shape_ics", ())),
        "segmentation.shape_ics",
    )
    input_shape_ics = _shape3(
        input_metadata.get("shape_ics", output_shape_ics),
        "input.shape_ics",
    )
    if input_shape_ics != output_shape_ics:
        raise ValueError(
            "SurfaceSeg input.shape_ics and segmentation.shape_ics must match"
        )
    source_shape_zyx = _shape3(
        input_metadata.get("source_shape_zyx", ()),
        "input.source_shape_zyx",
    )
    output_inline_count, output_crossline_count, output_sample_count = output_shape_ics
    output_shape_zyx = (
        output_sample_count,
        output_inline_count,
        output_crossline_count,
    )
    if (
        output_sample_count != source_shape_zyx[0]
        or output_crossline_count != source_shape_zyx[2]
        or output_inline_count > source_shape_zyx[1]
    ):
        raise ValueError(
            "SurfaceSeg output shape does not align with the SEG-Y "
            f"source shape: ICS={output_shape_ics}, source ZYX={source_shape_zyx}"
        )
    max_inlines_value = inference.get("max_inlines")
    if max_inlines_value is None:
        if output_inline_count != source_shape_zyx[1]:
            raise ValueError(
                "partial Inline output requires inference.max_inlines so the "
                "first-Inline smoke-test alignment is explicit"
            )
        smoke_mode = False
    else:
        try:
            max_inlines = int(max_inlines_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "inference.max_inlines must be a positive integer"
            ) from exc
        if (
            isinstance(max_inlines_value, (bool, np.bool_))
            or max_inlines <= 0
            or max_inlines != max_inlines_value
        ):
            raise ValueError("inference.max_inlines must be a positive integer")
        expected_inline_count = min(max_inlines, source_shape_zyx[1])
        if output_inline_count != expected_inline_count:
            raise ValueError(
                "SurfaceSeg smoke output must contain the first "
                f"{expected_inline_count} Inline slices, got {output_inline_count}"
            )
        smoke_mode = output_inline_count < source_shape_zyx[1]

    source = _metadata_path(
        input_metadata.get("source"),
        base=metadata_base,
        label="input.source",
        suffixes={".sgy", ".segy"},
    )
    horizon_display_contract = copy.deepcopy(
        dict(verified_horizon_display_contract or {})
    )
    reconciliation = segmentation.get("global_reconciliation")
    if (
        isinstance(reconciliation, Mapping)
        and reconciliation.get("global_display_ready") is True
    ):
        if not horizon_display_contract:
            horizon_contract_path = _metadata_path(
                outputs.get("horizon_surfaces_npz"),
                base=metadata_base,
                label="outputs.horizon_surfaces_npz",
                suffixes={".npz"},
            )
            global_contract_mask_path = _metadata_path(
                outputs.get("global_mask_npy"),
                base=metadata_base,
                label="outputs.global_mask_npy",
                suffixes={".npy"},
            )
            horizon_display_contract = validate_surface_horizon_display_contract(
                reconciliation,
                horizon_contract_path,
                global_mask_path=global_contract_mask_path,
                expected_shape_ics=output_shape_ics,
            )
        if not horizon_display_contract["valid"]:
            raise ValueError(
                "SurfaceSeg horizon display receipt is inconsistent with raw "
                "NPZ/mask artifacts: "
                + ", ".join(horizon_display_contract["reason_codes"])
            )
    display_scope = _surface_display_scope(
        segmentation,
        outputs,
        output_inline_count=output_inline_count,
        horizon_display_contract=horizon_display_contract,
    )
    if display_scope["globalConsistent"]:
        mask_output = outputs.get("global_mask_npy")
        mask_output_label = "outputs.global_mask_npy"
    elif display_scope["labelsRenderable"]:
        reconciliation = segmentation.get("global_reconciliation")
        if isinstance(reconciliation, Mapping) and reconciliation:
            if (
                reconciliation.get("global_display_ready") is not False
                or reconciliation.get("output_semantics") != "local_inline_fallback"
            ):
                raise ValueError(
                    "SurfaceSeg global reconciliation is not display-ready and has no "
                    "sealed local Inline fallback"
                )
            if outputs.get("local_mask_npy"):
                mask_output = outputs.get("local_mask_npy")
                mask_output_label = "outputs.local_mask_npy"
            else:
                mask_output = outputs.get("mask_npy")
                mask_output_label = "outputs.mask_npy"
        else:
            mask_output = outputs.get("mask_npy")
            mask_output_label = "outputs.mask_npy"
    else:
        # Keep the seismic context available for diagnosis, but do not expose
        # a collapsed/malformed global label volume as an interpretation.
        mask_output = outputs.get("mask_npy") or outputs.get("global_mask_npy")
        mask_output_label = "outputs.mask_npy"
    mask_path = _metadata_path(
        mask_output,
        base=metadata_base,
        label=mask_output_label,
        suffixes={".npy"},
    )
    confidence_path = _metadata_path(
        outputs.get("confidence_npy"),
        base=metadata_base,
        label="outputs.confidence_npy",
        suffixes={".npy"},
    )
    mask_ics = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    confidence_ics = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    if mask_ics.ndim != 3 or tuple(mask_ics.shape) != output_shape_ics:
        raise ValueError(
            f"mask array shape {mask_ics.shape} does not match {output_shape_ics}"
        )
    if confidence_ics.ndim != 3 or tuple(confidence_ics.shape) != output_shape_ics:
        raise ValueError(
            "confidence array shape "
            f"{confidence_ics.shape} does not match {output_shape_ics}"
        )
    if not np.issubdtype(mask_ics.dtype, np.number):
        raise ValueError("SurfaceSeg mask array must be numeric")
    if not np.issubdtype(confidence_ics.dtype, np.number):
        raise ValueError("SurfaceSeg confidence array must be numeric")

    label_range_raw = segmentation.get("label_range", ())
    if (
        not isinstance(label_range_raw, Sequence)
        or isinstance(label_range_raw, (str, bytes))
        or len(label_range_raw) != 2
    ):
        raise ValueError("segmentation.label_range must contain [min, max]")
    try:
        reported_min, reported_max = (
            int(label_range_raw[0]),
            int(label_range_raw[1]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("segmentation.label_range must contain integers") from exc
    invalid_label = int(segmentation.get("invalid_label", -1))
    if invalid_label >= 0:
        raise ValueError("segmentation.invalid_label must be negative")
    if reported_min < invalid_label or reported_max < 0 or reported_min > reported_max:
        raise ValueError("invalid segmentation.label_range")

    background, cache_hit = cache.get_crop(
        source,
        crop_start_zyx=(0, 0, 0),
        crop_size_zyx=output_shape_zyx,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=segy_options,
        expected_source_shape_zyx=source_shape_zyx,
    )
    z_indices, inline_indices, crossline_indices = background.sample_indices_zyx
    # NPY is [I, X, S]; select in that native order first and transpose only
    # the bounded sample into the platform's [Z, I, X] contract.
    selection_ics = np.ix_(inline_indices, crossline_indices, z_indices)
    mask_sample_zyx = np.transpose(
        np.asarray(mask_ics[selection_ics]),
        (2, 0, 1),
    )
    confidence_sample_zyx = np.transpose(
        np.asarray(confidence_ics[selection_ics], dtype=np.float32),
        (2, 0, 1),
    )
    if (
        mask_sample_zyx.shape != background.cube_int8.shape
        or confidence_sample_zyx.shape != background.cube_int8.shape
    ):
        raise ValueError(
            "SurfaceSeg overlay sampling is not aligned with the SEG-Y background"
        )
    if (
        not np.all(np.isfinite(confidence_sample_zyx))
        or np.any(confidence_sample_zyx < 0.0)
        or np.any(confidence_sample_zyx > 1.0)
    ):
        raise ValueError("sampled SurfaceSeg confidence contains values outside [0, 1]")

    encoded_labels, label_metadata = _encode_discrete_labels(
        mask_sample_zyx,
        invalid_label=invalid_label,
        reported_max=reported_max,
    )
    confidence_uint8 = np.rint(confidence_sample_zyx * 255.0).astype(np.uint8)
    mask_spec = _encode_uint8(
        encoded_labels,
        valueRange=[0.0, 1.0],
        source=str(mask_path),
        **label_metadata,
    )
    confidence_spec = _encode_uint8(
        confidence_uint8,
        valueRange=[0.0, 1.0],
        source=str(confidence_path),
    )

    abstention = segmentation.get("abstention")
    if abstention is not None and not isinstance(abstention, Mapping):
        raise ValueError("segmentation.abstention must be a mapping")
    abstention_contract = dict(abstention or {})
    if abstention_contract:
        unknown_label = int(abstention_contract.get("unknown_label", invalid_label))
        if unknown_label != invalid_label:
            raise ValueError(
                "segmentation.abstention.unknown_label differs from invalid_label"
            )
        threshold = float(
            abstention_contract.get(
                "confidence_threshold", inference.get("query_threshold", 0.35)
            )
        )
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("abstention confidence_threshold must stay within [0, 1]")
        for field in ("unknown_voxel_count", "valid_voxel_count"):
            if field in abstention_contract:
                _nonnegative_int(
                    abstention_contract[field], f"segmentation.abstention.{field}"
                )
        if "unknown_fraction" in abstention_contract:
            unknown_fraction = float(abstention_contract["unknown_fraction"])
            if not np.isfinite(unknown_fraction) or not 0.0 <= unknown_fraction <= 1.0:
                raise ValueError("segmentation.abstention.unknown_fraction is invalid")
    else:
        abstention_contract = {
            "unknown_label": invalid_label,
            "confidence_threshold": float(inference.get("query_threshold", 0.35)),
            "semantics": "legacy_invalid_trace_only",
        }

    normalized_layer = str(overlay_layer).strip().lower()
    if normalized_layer != "mask":
        raise ValueError(
            "SurfaceSeg confidence is quality-control evidence, not a display layer; "
            "overlay_layer must be 'mask'"
        )
    overlay = {
        "id": "surface_seg_labels",
        "name": str(display_scope["legendTitle"]),
        "kind": "labels",
        "labelScope": str(display_scope["labelScope"]),
        "globalConsistent": bool(display_scope["globalConsistent"]),
        "volume": mask_spec,
        "clim": [0.0, 1.0],
        "cmap": "categorical",
        "alpha": 0.20,
        "excpt": "min",
        "interpolation": "nearest",
        "renderStyle": "seismic_gray_interval_fill_with_boundaries",
        "boundaryColor": "#f59e0b",
        "boundaryAlpha": 0.98,
    }

    payload = background.as_payload(cache_hit=cache_hit)
    surfaces: list[dict[str, Any]] = []
    display_continuity: dict[str, Any] = {
        "schemaVersion": _DISPLAY_CONTINUITY_SCHEMA,
        "mode": "bounded-isolated-inline-spike-display-repair",
        "axis": "INLINE",
        "applied": False,
        "rawArtifactUnchanged": True,
        "reasonCode": "global_horizon_display_unavailable",
    }
    if display_scope["globalConsistent"] and display_scope["globalHorizonCount"]:
        horizon_path = _metadata_path(
            outputs.get("horizon_surfaces_npz"),
            base=metadata_base,
            label="outputs.horizon_surfaces_npz",
            suffixes={".npz"},
        )
        surfaces, display_continuity = _sample_global_horizon_surfaces(
            horizon_path,
            output_shape_ics=output_shape_ics,
            z_indices=np.asarray(z_indices),
            inline_indices=np.asarray(inline_indices),
            crossline_indices=np.asarray(crossline_indices),
            preview_inline_values=np.asarray(background.inline_values),
            preview_crossline_values=np.asarray(background.crossline_values),
            expected_horizon_count=int(display_scope["globalHorizonCount"]),
            eligible_horizon_ids=display_scope["eligibleHorizonIds"],
            horizon_surface_receipts=list(
                display_scope["reconciliation"].get("horizon_surface_receipts", [])
            ),
        )

    payload["contractVersion"] = "surface-seg-cigvis-v2"
    payload["name"] = (
        (
            f"{source.name} · 全局层位"
            + (
                f"（抑制 {len(display_scope['suppressedHorizonIds'])} 面）"
                if display_scope["suppressedHorizonIds"]
                else ""
            )
        )
        if display_scope["globalConsistent"]
        else f"{source.name} · 层位结果退化（仅地震背景）"
        if display_scope["backgroundOnly"]
        else f"{source.name} · Inline 局部分层"
    )
    continuity_notice = (
        "仅 Crossline 层位折线使用有界的显示派生去尖峰，并在全部显示层位"
        "严格有序时跨越至多 1 个连续缺失预览点；2 个及以上缺口不连接；"
        "Inline、层位平面、Inline 层间分区、原始标签体、置信度和层位 NPZ "
        "保持不变；该连接仅作用于折线，不填充任何网格单元"
        if display_continuity.get("applied") is True
        else (
            "Crossline 层位折线仅在全部显示层位严格有序时跨越至多 1 个连续"
            "缺失预览点；2 个及以上缺口不连接；Inline、层位平面、Inline "
            "层间分区、原始标签体、置信度和层位 NPZ 保持不变"
            if display_scope["globalConsistent"]
            else ""
        )
    )
    payload["sliceViewContract"] = {
        "modelId": SURFACE_SEG_MODEL_ID,
        "defaultPlane": display_scope["defaultPlane"],
        "allowedPlanes": list(display_scope["allowedPlanes"]),
        "disabledPlaneReasons": dict(display_scope["disabledPlaneReasons"]),
        "labelScope": display_scope["labelScope"],
        "globalConsistent": display_scope["globalConsistent"],
        "unknownTransparent": True,
        "displayMode": (
            "background_only" if display_scope["backgroundOnly"] else "interpretation"
        ),
        "displayNotice": " · ".join(
            value
            for value in (display_scope["displayNotice"], continuity_notice)
            if value
        ),
        "preferSlice": True,
    }
    payload["surfaceSeg"] = {
        "modelId": SURFACE_SEG_MODEL_ID,
        "sourceAxes": list(AXES_ICS),
        "platformAxes": list(AXES_ZYX),
        "transposeICSToZYX": [2, 0, 1],
        "shapeICS": list(output_shape_ics),
        "shapeZYX": list(output_shape_zyx),
        "processedInlineRange": [0, output_inline_count - 1],
        "smokeMode": smoke_mode,
        "crossInlineConsistent": bool(display_scope["globalConsistent"]),
        "labelScope": display_scope["labelScope"],
        "globalReconciliation": display_scope["reconciliation"],
        "globalDisplayGate": {
            "accepted": display_scope["globalConsistent"],
            "reasonCodes": display_scope["reasonCodes"],
            "allowedPlanes": display_scope["allowedPlanes"],
            "labelsRenderable": display_scope["labelsRenderable"],
            "backgroundOnly": display_scope["backgroundOnly"],
            "rawHorizonCount": display_scope["rawGlobalHorizonCount"],
            "displayHorizonCount": display_scope["displayHorizonCount"],
            "suppressedHorizonCount": len(display_scope["suppressedHorizonIds"]),
            "suppressedHorizonIds": display_scope["suppressedHorizonIds"],
        },
        "abstention": abstention_contract,
        "windowInference": dict(inference.get("window_inference") or {}),
        "mask": mask_spec,
        "confidence": confidence_spec,
        "display": {
            "backgroundCmap": "gray",
            "preferredLayer": ("none" if display_scope["backgroundOnly"] else "mask"),
            "maskCmap": "categorical",
            "maskDiscrete": True,
            "confidenceRole": "quality_control_only",
            "confidenceSelectable": False,
            "alpha": float(overlay["alpha"]),
            "renderStyle": overlay["renderStyle"],
            "boundaryColor": overlay["boundaryColor"],
            "excludeMinimum": True,
            "unknownTransparent": True,
            "legendTitle": display_scope["legendTitle"],
            "rawHorizonCount": display_scope["rawGlobalHorizonCount"],
            "displayHorizonCount": display_scope["displayHorizonCount"],
            "suppressedHorizonIds": display_scope["suppressedHorizonIds"],
            "horizonContinuity": display_continuity,
        },
        "cigvis": {
            "method": "add_mask",
            "sourceAxes": list(AXES_ZYX),
            "transposeZYXToLineFirst": [1, 2, 0],
        },
    }
    payload["overlays"] = [] if display_scope["backgroundOnly"] else [overlay]
    payload["surfaces"] = surfaces
    payload["preview"]["cacheStats"] = cache.stats
    payload["preview"]["vertical_note"] = (
        "地层标签/置信度由[Inline,Crossline,Sample]转为"
        "[Z,Inline,Crossline]，并与背景体共用同一稀疏采样索引；"
        + (
            "全局层位已通过跨Inline关联与非交叉校验；"
            f"逐面支持门显示 {display_scope['displayHorizonCount']}/"
            f"{display_scope['rawGlobalHorizonCount']} 面；"
            + (
                "仅Crossline层位折线使用有界显示派生去尖峰；"
                "严格层序校验通过时可跨越1个缺失预览点，2个及以上不连接；"
                "Inline/层位平面/Inline层间分区/原始标签体/置信度/NPZ不变；"
                "仅连接折线，不填充网格"
                if display_continuity.get("applied") is True
                else "Crossline层位折线未触发尖峰校正；严格层序校验通过时可跨越"
                "1个缺失预览点，2个及以上不连接；"
                "Inline/层位平面/Inline层间分区/原始标签体/置信度/NPZ不变"
            )
            if display_scope["globalConsistent"]
            else "全局层位结果退化，标签与置信度已隐藏，仅保留灰度地震背景"
            if display_scope["backgroundOnly"]
            else "当前仅具备Inline局部分层语义，跨方向标签视图已关闭"
        )
    )
    return payload


__all__ = [
    "AXES_ICS",
    "SURFACE_SEG_MODEL_ID",
    "build_surface_seg_visualization_payload",
]
