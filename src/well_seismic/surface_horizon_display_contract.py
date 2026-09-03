"""Independent verifier for SurfaceSeg per-horizon display support.

The producer lives in a self-contained offline bundle, so this verifier
deliberately reimplements the small deterministic 4-neighbour calculation.
Candidate metadata cannot make itself displayable merely by copying claimed
fractions or ``display_eligible`` flags into a JSON receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


HORIZON_DISPLAY_GATE_SCHEMA = "surface-seg.horizon-display-gate.v1"
DEFAULT_MINIMUM_FINITE_TRACE_FRACTION = 0.10
DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION = 0.05
SURFACE_COMPONENT_CONNECTIVITY = 4


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        return None
    result = int(value)
    return result if result >= minimum else None


def _fraction(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) and 0.0 <= result <= 1.0 else None


def _four_connected_component_sizes(mask: np.ndarray) -> list[int]:
    supported = np.asarray(mask, dtype=bool)
    if supported.ndim != 2:
        raise ValueError("surface support mask must use [INLINE, CROSSLINE]")
    parents: list[int] = []
    sizes: list[int] = []

    def find(node: int) -> int:
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            parent = parents[node]
            parents[node] = root
            node = parent
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        root, child = sorted((first_root, second_root))
        parents[child] = root
        sizes[root] += sizes[child]
        sizes[child] = 0

    previous_runs: list[tuple[int, int, int]] = []
    for row in supported:
        padded = np.pad(row, (1, 1), constant_values=False)
        transitions = np.diff(padded.astype(np.int8, copy=False))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        current_runs: list[tuple[int, int, int]] = []
        previous_cursor = 0
        for start_raw, stop_raw in zip(starts, stops, strict=True):
            start, stop = int(start_raw), int(stop_raw)
            node = len(parents)
            parents.append(node)
            sizes.append(stop - start)
            while (
                previous_cursor < len(previous_runs)
                and previous_runs[previous_cursor][1] <= start
            ):
                previous_cursor += 1
            overlap_cursor = previous_cursor
            while (
                overlap_cursor < len(previous_runs)
                and previous_runs[overlap_cursor][0] < stop
            ):
                union(node, previous_runs[overlap_cursor][2])
                overlap_cursor += 1
            current_runs.append((start, stop, node))
        previous_runs = current_runs
    return [sizes[index] for index, parent in enumerate(parents) if parent == index]


def surface_support_metrics(depth_samples: np.ndarray) -> dict[str, int | float]:
    """Recompute one surface's finite support with deterministic 4-connectivity."""

    depth = np.asarray(depth_samples)
    if depth.ndim != 2:
        raise ValueError("one horizon must use [INLINE, CROSSLINE]")
    inline_count, xline_count = depth.shape
    finite = np.isfinite(depth)
    finite_count = int(np.count_nonzero(finite))
    total_count = int(finite.size)
    inline_support_count = int(np.count_nonzero(np.any(finite, axis=1)))
    xline_support_count = int(np.count_nonzero(np.any(finite, axis=0)))
    component_sizes = _four_connected_component_sizes(finite)
    largest_count = max(component_sizes, default=0)
    return {
        "finite_trace_count": finite_count,
        "total_trace_count": total_count,
        "finite_trace_fraction": (
            float(finite_count / total_count) if total_count else 0.0
        ),
        "inline_support_count": inline_support_count,
        "inline_support_fraction": (
            float(inline_support_count / inline_count) if inline_count else 0.0
        ),
        "xline_support_count": xline_support_count,
        "xline_support_fraction": (
            float(xline_support_count / xline_count) if xline_count else 0.0
        ),
        "connected_component_count": len(component_sizes),
        "largest_component_trace_count": largest_count,
        "largest_component_fraction": (
            float(largest_count / finite_count) if finite_count else 0.0
        ),
    }


def _validate_depths_against_global_mask(
    depths: np.ndarray,
    lower_package_ids: np.ndarray,
    global_mask_path: str | Path,
    *,
    expected_shape_ics: tuple[int, int, int] | None,
) -> str | None:
    """Replay the producer's first-lower-package extraction in bounded slabs."""

    path = Path(global_mask_path).expanduser()
    if path.suffix.casefold() != ".npy" or not path.is_file():
        return "global_mask_artifact_missing"
    try:
        labels = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError):
        return "global_mask_artifact_unreadable"
    horizon_count, inline_count, xline_count = depths.shape
    expected_shape = (
        expected_shape_ics
        if expected_shape_ics is not None
        else (inline_count, xline_count, int(labels.shape[2]) if labels.ndim == 3 else 0)
    )
    if (
        labels.ndim != 3
        or tuple(labels.shape) != tuple(expected_shape)
        or not np.issubdtype(labels.dtype, np.integer)
    ):
        return "global_mask_artifact_shape_or_dtype_invalid"
    if not np.array_equal(
        lower_package_ids.astype(np.int64, copy=False),
        np.arange(1, horizon_count + 1, dtype=np.int64),
    ):
        return "horizon_surfaces_artifact_values_invalid"
    sample_count = int(labels.shape[2])
    trace_indexes = np.repeat(np.arange(xline_count, dtype=np.int64), sample_count)
    sample_indexes = np.tile(np.arange(sample_count, dtype=np.int32), xline_count)
    for inline_index in range(inline_count):
        slab = np.asarray(labels[inline_index])
        if np.any(slab < -1) or np.any(slab > horizon_count):
            return "global_mask_label_range_invalid"
        flat = slab.reshape(-1)
        known = (flat >= 1) & (flat <= horizon_count)
        first = np.full(horizon_count * xline_count, sample_count, dtype=np.int32)
        if np.any(known):
            positions = (
                (flat[known].astype(np.int64, copy=False) - 1) * xline_count
                + trace_indexes[known]
            )
            np.minimum.at(first, positions, sample_indexes[known])
        first = first.reshape(horizon_count, xline_count)
        present = first < sample_count
        observed = depths[:, inline_index, :]
        if not np.array_equal(np.isfinite(observed), present):
            return "horizon_surfaces_global_mask_semantics_mismatch"
        if np.any(observed[present] != first[present]):
            return "horizon_surfaces_global_mask_semantics_mismatch"
    return None


def _expected_display_reasons(
    metrics: Mapping[str, int | float],
    *,
    minimum_finite_trace_fraction: float,
    minimum_largest_component_fraction: float,
) -> list[str]:
    reasons: list[str] = []
    if float(metrics["finite_trace_fraction"]) < minimum_finite_trace_fraction:
        reasons.append("finite_trace_fraction_below_minimum")
    if (
        float(metrics["largest_component_fraction"])
        < minimum_largest_component_fraction
    ):
        reasons.append("largest_component_fraction_below_minimum")
    return reasons


def validate_surface_horizon_display_contract(
    reconciliation: Mapping[str, Any],
    horizon_path: str | Path,
    *,
    global_mask_path: str | Path | None = None,
    expected_shape_ics: tuple[int, int, int] | None = None,
    expected_minimum_finite_trace_fraction: float | None = None,
    expected_minimum_largest_component_fraction: float | None = None,
) -> dict[str, Any]:
    """Verify the receipt against every raw NPZ surface and fail closed on lies."""

    reasons: list[str] = []
    gate_raw = reconciliation.get("horizon_display_gate")
    gate = gate_raw if isinstance(gate_raw, Mapping) else {}
    if gate.get("schema_version") != HORIZON_DISPLAY_GATE_SCHEMA:
        reasons.append("horizon_display_gate_schema_invalid")
    minimum_finite = _fraction(gate.get("minimum_finite_trace_fraction"))
    minimum_largest = _fraction(
        gate.get("minimum_largest_component_fraction")
    )
    if (
        minimum_finite is None
        or minimum_largest is None
        or minimum_finite < DEFAULT_MINIMUM_FINITE_TRACE_FRACTION
        or minimum_largest < DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
    ):
        reasons.append("horizon_display_gate_threshold_invalid")
    else:
        if (
            expected_minimum_finite_trace_fraction is not None
            and not np.isclose(
                minimum_finite,
                expected_minimum_finite_trace_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            reasons.append("horizon_display_gate_threshold_mismatch")
        if (
            expected_minimum_largest_component_fraction is not None
            and not np.isclose(
                minimum_largest,
                expected_minimum_largest_component_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            reasons.append("horizon_display_gate_threshold_mismatch")
    if (
        _integer(gate.get("surface_connectivity"), minimum=1)
        != SURFACE_COMPONENT_CONNECTIVITY
        or _integer(gate.get("volume_connectivity_analogue"), minimum=1) != 6
        or gate.get("finite_trace_fraction_denominator")
        != "dense_inline_xline_grid"
        or gate.get("axis_support_fraction_denominator") != "dense_axis_count"
        or gate.get("largest_component_fraction_denominator")
        != "finite_trace_count"
    ):
        reasons.append("horizon_display_gate_connectivity_invalid")

    path = Path(horizon_path).expanduser()
    required = {
        "depth_samples",
        "confidence",
        "horizon_ids",
        "lower_package_ids",
        "inline_values",
        "xline_values",
        "sample_interval_us",
    }
    if path.suffix.casefold() != ".npz" or not path.is_file():
        reasons.append("horizon_surfaces_artifact_missing")
        return {
            "valid": False,
            "reason_codes": list(dict.fromkeys(reasons)),
            "raw_horizon_count": 0,
            "display_horizon_count": 0,
            "eligible_horizon_ids": [],
            "suppressed_horizon_ids": [],
        }
    try:
        with np.load(path, allow_pickle=False) as document:
            missing = required.difference(document.files)
            if missing:
                raise KeyError(",".join(sorted(missing)))
            depths = np.asarray(document["depth_samples"])
            confidence = np.asarray(document["confidence"])
            horizon_ids = np.asarray(document["horizon_ids"])
            lower_package_ids = np.asarray(document["lower_package_ids"])
            inline_values = np.asarray(document["inline_values"])
            xline_values = np.asarray(document["xline_values"])
    except (KeyError, OSError, TypeError, ValueError):
        reasons.append("horizon_surfaces_artifact_unreadable")
        return {
            "valid": False,
            "reason_codes": list(dict.fromkeys(reasons)),
            "raw_horizon_count": 0,
            "display_horizon_count": 0,
            "eligible_horizon_ids": [],
            "suppressed_horizon_ids": [],
        }

    shape_valid = depths.ndim == 3 and confidence.shape == depths.shape
    horizon_count = int(depths.shape[0]) if depths.ndim == 3 else 0
    if shape_valid:
        inline_count, xline_count = depths.shape[1:]
        shape_valid = (
            horizon_ids.shape == (horizon_count,)
            and lower_package_ids.shape == (horizon_count,)
            and inline_values.shape == (inline_count,)
            and xline_values.shape == (xline_count,)
            and (
                expected_shape_ics is None
                or depths.shape[1:] == expected_shape_ics[:2]
            )
        )
    if not shape_valid:
        reasons.append("horizon_surfaces_artifact_shape_mismatch")
        horizon_count = 0
    artifact_values_valid = bool(shape_valid)
    if shape_valid and (
        not np.issubdtype(horizon_ids.dtype, np.integer)
        or not np.issubdtype(lower_package_ids.dtype, np.integer)
        or len(np.unique(horizon_ids)) != horizon_count
        or not np.array_equal(
            horizon_ids.astype(np.int64, copy=False),
            np.arange(horizon_count, dtype=np.int64),
        )
        or not np.array_equal(
            lower_package_ids.astype(np.int64, copy=False),
            np.arange(1, horizon_count + 1, dtype=np.int64),
        )
        or np.any(np.isinf(depths))
        or np.any(np.isinf(confidence))
        or not np.array_equal(np.isfinite(depths), np.isfinite(confidence))
        or np.any(depths[np.isfinite(depths)] < 0.0)
        or (
            expected_shape_ics is not None
            and np.any(
                depths[np.isfinite(depths)] >= expected_shape_ics[2]
            )
        )
        or np.any(
            depths[np.isfinite(depths)]
            != np.rint(depths[np.isfinite(depths)])
        )
        or np.any(confidence[np.isfinite(confidence)] < 0.0)
        or np.any(confidence[np.isfinite(confidence)] > 1.0)
    ):
        artifact_values_valid = False
        reasons.append("horizon_surfaces_artifact_values_invalid")
    if horizon_count > 1 and artifact_values_valid:
        jointly_finite = np.isfinite(depths[:-1]) & np.isfinite(depths[1:])
        if np.any(depths[1:][jointly_finite] <= depths[:-1][jointly_finite]):
            artifact_values_valid = False
            reasons.append("horizon_surfaces_non_crossing_invalid")
    if horizon_count and artifact_values_valid and global_mask_path is not None:
        semantic_error = _validate_depths_against_global_mask(
            depths,
            lower_package_ids,
            global_mask_path,
            expected_shape_ics=expected_shape_ics,
        )
        if semantic_error:
            reasons.append(semantic_error)

    raw_count = _integer(reconciliation.get("global_horizon_count"))
    if raw_count != horizon_count:
        reasons.append("horizon_display_raw_count_mismatch")
    receipts_raw = reconciliation.get("horizon_surface_receipts")
    receipts = (
        list(receipts_raw)
        if isinstance(receipts_raw, Sequence)
        and not isinstance(receipts_raw, (str, bytes))
        else []
    )
    if len(receipts) != horizon_count or any(
        not isinstance(item, Mapping) for item in receipts
    ):
        reasons.append("horizon_surface_receipts_invalid")

    eligible_ids: list[int] = []
    suppressed_ids: list[int] = []
    if (
        horizon_count
        and len(receipts) == horizon_count
        and all(isinstance(item, Mapping) for item in receipts)
        and minimum_finite is not None
        and minimum_largest is not None
        and artifact_values_valid
    ):
        integer_fields = {
            "finite_trace_count",
            "total_trace_count",
            "inline_support_count",
            "xline_support_count",
            "connected_component_count",
            "largest_component_trace_count",
        }
        fraction_fields = {
            "finite_trace_fraction",
            "inline_support_fraction",
            "xline_support_fraction",
            "largest_component_fraction",
        }
        for index, item_raw in enumerate(receipts):
            assert isinstance(item_raw, Mapping)
            item = item_raw
            horizon_id = int(horizon_ids[index])
            lower_package_id = int(lower_package_ids[index])
            metrics = surface_support_metrics(depths[index])
            if (
                _integer(item.get("horizon_id")) != horizon_id
                or _integer(item.get("lower_package_id")) != lower_package_id
            ):
                reasons.append("horizon_surface_id_order_mismatch")
            metrics_match = True
            for field in integer_fields:
                if _integer(item.get(field)) != int(metrics[field]):
                    metrics_match = False
            for field in fraction_fields:
                observed = _fraction(item.get(field))
                if observed is None or not np.isclose(
                    observed, float(metrics[field]), rtol=0.0, atol=1e-12
                ):
                    metrics_match = False
            if not metrics_match:
                reasons.append("horizon_surface_metrics_mismatch")
            expected_reasons = _expected_display_reasons(
                metrics,
                minimum_finite_trace_fraction=minimum_finite,
                minimum_largest_component_fraction=minimum_largest,
            )
            raw_item_reasons = item.get("reasons")
            item_reasons = (
                list(raw_item_reasons)
                if isinstance(raw_item_reasons, Sequence)
                and not isinstance(raw_item_reasons, (str, bytes))
                else None
            )
            expected_eligible = not expected_reasons
            if (
                item.get("display_eligible") is not expected_eligible
                or item_reasons != expected_reasons
            ):
                reasons.append("horizon_display_eligibility_mismatch")
            if expected_eligible:
                eligible_ids.append(horizon_id)
            else:
                suppressed_ids.append(horizon_id)

    display_count = _integer(reconciliation.get("display_horizon_count"))
    raw_suppressed = reconciliation.get("suppressed_horizon_ids")
    suppressed_items = (
        [_integer(value) for value in raw_suppressed]
        if isinstance(raw_suppressed, Sequence)
        and not isinstance(raw_suppressed, (str, bytes))
        else None
    )
    reported_suppressed = (
        [int(value) for value in suppressed_items]
        if suppressed_items is not None
        and all(value is not None for value in suppressed_items)
        else None
    )
    if (
        display_count != len(eligible_ids)
        or reported_suppressed != suppressed_ids
        or len(eligible_ids) + len(suppressed_ids) != horizon_count
    ):
        reasons.append("horizon_display_counts_invalid")
    return {
        "valid": not reasons,
        "reason_codes": list(dict.fromkeys(reasons)),
        "raw_horizon_count": horizon_count,
        "display_horizon_count": len(eligible_ids),
        "eligible_horizon_ids": eligible_ids,
        "suppressed_horizon_ids": suppressed_ids,
    }


__all__ = [
    "DEFAULT_MINIMUM_FINITE_TRACE_FRACTION",
    "DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION",
    "HORIZON_DISPLAY_GATE_SCHEMA",
    "SURFACE_COMPONENT_CONNECTIVITY",
    "surface_support_metrics",
    "validate_surface_horizon_display_contract",
]
