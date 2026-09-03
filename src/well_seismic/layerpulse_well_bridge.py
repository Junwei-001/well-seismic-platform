from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .io.las import read_las
from .knowledge import CurveKnowledgeBase
from .modeling.input_adapters import _prepared_view_aligned_well_inputs
from .models import WellLog
from .registration_contract import RegistrationPointV3, read_registration_points_v3


LAYERPULSE_WELL_BUNDLE_SCHEMA = "well-seismic.layerpulse-well-bundle.v1"
LAYERPULSE_WELL_INPUT_CONTRACT = "prepared-view-md-trajectory-no-td.v1"
LAYERPULSE_MAX_WELL_STATIONS = 256

# The order is the frozen nine-channel LayerPulse well-encoder contract.  The
# aliases cover the common names before and after WellFuse KB standardisation.
LAYERPULSE_CURVE_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GR", ("GR", "SGR", "CGR")),
    ("SP", ("SP",)),
    ("CAL", ("CAL", "CALI", "CALS", "CALI_FDC", "CALI_MSFL")),
    ("AC/DT", ("DT", "AC", "DTC")),
    ("DEN/RHOB", ("RHOB", "DEN", "RHOZ", "ZDEN")),
    ("CNL/NPHI", ("NPHI", "CNL", "TNPH")),
    ("RXO", ("RXO", "MSFL", "M2R1")),
    ("RT", ("RT", "ILD", "LLD", "LLD_NORM", "AT90", "M2RX")),
    ("PE", ("PE", "PEF", "PEFZ")),
)

LAYERPULSE_WELL_ARRAY_KEYS = frozenset(
    {
        "wells",
        "well_curve_mask",
        "well_mask",
        "well_md",
        "well_md_m",
        "trajectory",
        "trajectory_metric_xyz",
        "trajectory_tvd_m",
        "trajectory_mask",
        "well_parent_index",
        "well_kickoff_md_m",
    }
)


def _identity_key(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _axis_values(
    provenance: Mapping[str, Any],
    *,
    values_key: str,
    range_key: str,
) -> np.ndarray:
    raw_values = provenance.get(values_key)
    values = np.asarray(raw_values if raw_values is not None else (), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"input_provenance.{values_key} must be one-dimensional")
    values = values[np.isfinite(values)]
    if values.size:
        values = np.unique(values)
    else:
        raw_range = provenance.get(range_key)
        if (
            not isinstance(raw_range, Sequence)
            or isinstance(raw_range, (str, bytes))
            or len(raw_range) != 2
        ):
            raise ValueError(
                f"input_provenance requires {values_key} or a two-value {range_key}"
            )
        values = np.asarray(raw_range, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"input_provenance.{range_key} must be finite")
        values.sort()
    if values.size == 0:
        raise ValueError(f"input_provenance.{values_key} is empty")
    return values


def _normalise_axis(value: float, axis: np.ndarray) -> float:
    if axis.size < 2 or math.isclose(float(axis[-1]), float(axis[0])):
        return 0.0
    coordinate = np.interp(
        float(value),
        axis,
        np.linspace(-1.0, 1.0, axis.size, dtype=np.float64),
    )
    return float(np.clip(coordinate, -1.0, 1.0))


def _finite_point_geometry(point: RegistrationPointV3) -> bool:
    physical = (point.md_m, point.tvd_m, point.x, point.y)
    return all(value is not None and math.isfinite(float(value)) for value in physical)


def _projection_from_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = provenance.get("xy_to_grid_projection")
    if not isinstance(raw, Mapping):
        return None
    projection = dict(raw)
    if projection.get("schema_version") != "well-seismic.xy-to-grid-affine.v1":
        return None
    origin = np.asarray(projection.get("origin_xy") or (), dtype=np.float64)
    inline = np.asarray(projection.get("inline_coefficients") or (), dtype=np.float64)
    crossline = np.asarray(
        projection.get("crossline_coefficients") or (), dtype=np.float64
    )
    if (
        origin.shape != (2,)
        or inline.shape != (3,)
        or crossline.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(inline).all()
        or not np.isfinite(crossline).all()
    ):
        return None
    projection["origin_xy"] = origin
    projection["inline_coefficients"] = inline
    projection["crossline_coefficients"] = crossline
    return projection


def _resolved_grid_point(
    point: RegistrationPointV3,
    projection: Mapping[str, Any] | None,
) -> tuple[RegistrationPointV3 | None, bool]:
    if (
        point.inline is not None
        and point.crossline is not None
        and math.isfinite(float(point.inline))
        and math.isfinite(float(point.crossline))
    ):
        return point, False
    if projection is None or point.x is None or point.y is None:
        return None, False
    projection_crs = str(projection.get("horizontal_crs_id") or "").strip().casefold()
    point_crs = str(point.horizontal_crs_id or "").strip().casefold()
    if not projection_crs or not point_crs or projection_crs != point_crs:
        return None, False
    origin = np.asarray(projection["origin_xy"], dtype=np.float64)
    vector = np.asarray(
        [float(point.x) - origin[0], float(point.y) - origin[1], 1.0],
        dtype=np.float64,
    )
    inline = float(vector @ np.asarray(projection["inline_coefficients"]))
    crossline = float(vector @ np.asarray(projection["crossline_coefficients"]))
    if not math.isfinite(inline) or not math.isfinite(crossline):
        return None, False
    # RegistrationPointV3 is frozen.  The local replacement is an inference
    # view only; the sealed registration CSV is never rewritten or backfilled.
    return replace(point, inline=inline, crossline=crossline), True


def _fusion_points_by_well(
    points: Sequence[RegistrationPointV3],
    *,
    projection: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[RegistrationPointV3]], int]:
    grouped: dict[str, list[RegistrationPointV3]] = defaultdict(list)
    projected_count = 0
    for point in points:
        resolved, projected = _resolved_grid_point(point, projection)
        if (
            point.valid_mask
            and point.inference_eligible
            and point.fusion_ready
            and _finite_point_geometry(point)
            and resolved is not None
        ):
            grouped[point.well_uid].append(resolved)
            projected_count += int(projected)
    output: dict[str, list[RegistrationPointV3]] = {}
    for well_uid, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (float(row.md_m), int(row.point_index)))
        if any(
            float(current.md_m) <= float(previous.md_m)
            for previous, current in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError(
                f"Registration V3 fusion trajectory MD is not strictly increasing: {well_uid}"
            )
        output[well_uid] = ordered
    return output, projected_count


def _well_log_aliases(log: WellLog, path: Path) -> set[str]:
    aliases = {_identity_key(log.well_name), _identity_key(path.stem)}
    aliases.update(_identity_key(identifier) for identifier in log.identifiers)
    return {alias for alias in aliases if alias}


def _curve_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _select_slot_curve(
    log: WellLog,
    aliases: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    alias_tokens = {_curve_token(alias) for alias in aliases}
    candidates: list[tuple[int, int, str, np.ndarray, np.ndarray]] = []
    for standard_name, raw_values in log.curves.items():
        info = log.curve_info.get(standard_name)
        original_name = info.original_name if info is not None else ""
        tokens = {_curve_token(standard_name), _curve_token(original_name)}
        if not tokens.intersection(alias_tokens):
            continue
        values = np.asarray(raw_values, dtype=np.float64)
        mask = np.asarray(log.masks.get(standard_name, np.isfinite(values)), dtype=bool)
        mask = mask & np.isfinite(values)
        canonical = int(_curve_token(standard_name) == _curve_token(aliases[0]))
        candidates.append(
            (canonical, int(mask.sum()), str(standard_name), values, mask)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2].casefold()))
    return candidates[0][3], candidates[0][4]


def _interpolate_curve(
    source_md: np.ndarray,
    source_values: np.ndarray,
    source_mask: np.ndarray,
    target_md: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(source_md, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    declared = np.asarray(source_mask, dtype=bool)
    if depth.ndim != 1 or values.shape != depth.shape or declared.shape != depth.shape:
        raise ValueError("canonical LAS depth, curve, and mask shapes must agree")
    finite_depth = np.isfinite(depth)
    finite_curve = finite_depth & declared & np.isfinite(values)
    output = np.zeros(target_md.shape, dtype=np.float64)
    output_mask = np.zeros(target_md.shape, dtype=bool)
    if int(finite_curve.sum()) < 2:
        return output, output_mask

    valid_md = depth[finite_curve]
    valid_values = values[finite_curve]
    output[:] = np.interp(target_md, valid_md, valid_values)

    # Do not bridge a LAS null gap merely because valid values exist above and
    # below it.  A target is admitted only when its immediate source bracket is
    # declared valid (or it exactly matches a valid source station).
    right = np.searchsorted(depth, target_md, side="left")
    clipped_right = np.clip(right, 0, depth.size - 1)
    tolerance = 1.0e-7 * max(1.0, float(np.nanmax(np.abs(depth[finite_depth]))))
    exact = np.isclose(
        depth[clipped_right], target_md, rtol=0.0, atol=tolerance
    )
    exact_valid = exact & finite_curve[clipped_right]
    bracketed = (right > 0) & (right < depth.size)
    safe_right = np.clip(right, 1, depth.size - 1)
    left = safe_right - 1
    bracket_valid = (
        bracketed
        & finite_curve[left]
        & finite_curve[safe_right]
        & (target_md >= depth[left])
        & (target_md <= depth[safe_right])
    )
    output_mask = exact_valid | bracket_valid
    output[~output_mask] = 0.0
    return output, output_mask


def _interpolate_slots(
    log: WellLog,
    target_md: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channel_count = len(LAYERPULSE_CURVE_SLOTS)
    values = np.zeros((channel_count, target_md.size), dtype=np.float64)
    masks = np.zeros_like(values, dtype=bool)
    for channel, (_name, aliases) in enumerate(LAYERPULSE_CURVE_SLOTS):
        selected = _select_slot_curve(log, aliases)
        if selected is None:
            continue
        values[channel], masks[channel] = _interpolate_curve(
            np.asarray(log.depth, dtype=np.float64),
            selected[0],
            selected[1],
            target_md,
        )
    return values, masks


def _normalise_curves(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    normalised = np.zeros(values.shape, dtype=np.float32)
    for channel in range(values.shape[0]):
        valid = masks[channel].copy()
        selected = values[channel].copy()
        if channel in (6, 7):
            valid &= selected > 0.0
            masks[channel] = valid
            selected[valid] = np.log10(selected[valid])
        if not valid.any():
            masks[channel] = False
            continue
        observed = selected[valid]
        median = float(np.median(observed))
        scale = float(1.4826 * np.median(np.abs(observed - median)))
        if not math.isfinite(scale) or scale < 1.0e-6:
            scale = max(float(np.std(observed, dtype=np.float64)), 1.0e-6)
        normalised[channel, valid] = np.clip(
            (observed - median) / scale, -6.0, 6.0
        ).astype(np.float32)
    return normalised


def _station_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    # Endpoint-preserving deterministic thinning retains MD order for long
    # horizontal wells and branches represented as their own registration well.
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _destination_path(destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    if path.suffix.casefold() != ".npz":
        path = path / "layerpulse_wells_md_trajectory.npz"
    return path


def _degraded_receipt(
    reason: str,
    *,
    prepared_receipt: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        **dict(prepared_receipt or {}),
        # A validated PreparedView may still contribute no in-patch well token.
        # Keep validation distinct from actual model consumption.
        "prepared_view_validated": bool(prepared_receipt),
        "prepared_view_consumed": False,
        "schema_version": LAYERPULSE_WELL_BUNDLE_SCHEMA,
        "input_contract": LAYERPULSE_WELL_INPUT_CONTRACT,
        "status": "degraded",
        "reason": reason,
        "bundle_materialized": False,
        "well_count": 0,
        "valid_station_count": 0,
        "well_channels": len(LAYERPULSE_CURVE_SLOTS),
        "curve_slot_names": [name for name, _aliases in LAYERPULSE_CURVE_SLOTS],
        "time_depth_table_consumed": False,
        "registration_time_fields_consumed": False,
        "warnings": list(warnings),
    }


def materialize_layerpulse_well_bundle(
    options: Mapping[str, Any],
    *,
    platform_config: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
    destination: str | Path,
) -> tuple[Path | None, dict[str, Any]]:
    """Materialise the sealed PreparedView well input for one LayerPulse patch.

    Registration V3 contributes physical MD/TVD/XY and preview grid location.
    Canonical LAS contributes nine curve slots and missingness masks.  Seismic
    time alignment remains an internal Backbone inference problem: registration
    time fields and external time/depth tables are neither consumed nor emitted.
    """

    aligned = _prepared_view_aligned_well_inputs(options)
    if aligned is None:
        return None, _degraded_receipt("prepared_view_aligned_wells_unavailable")
    prepared_receipt = dict(aligned.get("receipt") or {})

    inline_axis = _axis_values(
        input_provenance,
        values_key="inline_values",
        range_key="inline_range",
    )
    crossline_axis = _axis_values(
        input_provenance,
        values_key="crossline_values",
        range_key="crossline_range",
    )

    points, validation = read_registration_points_v3(aligned["registration_points_path"])
    projection = _projection_from_provenance(input_provenance)
    grouped, projected_grid_point_count = _fusion_points_by_well(
        points,
        projection=projection,
    )
    if not grouped:
        return None, _degraded_receipt(
            (
                "fusion_ready_points_missing_inline_crossline_projection"
                if projection is None
                else "no_fusion_ready_registration_trajectory"
            ),
            prepared_receipt=prepared_receipt,
        )

    knowledge = CurveKnowledgeBase(dict(platform_config))
    preprocessing = platform_config.get("preprocessing")
    preprocessing = dict(preprocessing) if isinstance(preprocessing, Mapping) else {}
    logs: list[tuple[Path, WellLog]] = []
    for raw_path in aligned["las_paths"]:
        path = Path(raw_path).expanduser().resolve()
        logs.append(
            (
                path,
                read_las(path, knowledge, preprocessing=preprocessing),
            )
        )

    logs_by_alias: dict[str, list[tuple[Path, WellLog]]] = defaultdict(list)
    for path, log in logs:
        for alias in _well_log_aliases(log, path):
            logs_by_alias[alias].append((path, log))

    selected_wells: list[dict[str, Any]] = []
    unmatched_wells: list[str] = []
    for well_uid in sorted(grouped, key=str.casefold):
        full_track = grouped[well_uid]
        well_name = full_track[0].well_name
        match_keys = {_identity_key(well_uid), _identity_key(well_name)}
        candidates = {
            str(path): (path, log)
            for key in match_keys
            for path, log in logs_by_alias.get(key, ())
        }
        if not candidates:
            unmatched_wells.append(well_uid)
            continue
        # Canonical PreparedViews normally contain one LAS per well.  If legacy
        # aliases resolve more than one, prefer the file with most observed data.
        path, log = max(
            candidates.values(),
            key=lambda item: (
                sum(int(np.asarray(mask, dtype=bool).sum()) for mask in item[1].masks.values()),
                str(item[0]).casefold(),
            ),
        )

        patch_points = [
            point
            for point in full_track
            if float(inline_axis[0]) <= float(point.inline) <= float(inline_axis[-1])
            and float(crossline_axis[0])
            <= float(point.crossline)
            <= float(crossline_axis[-1])
        ]
        if not patch_points:
            continue

        target_md = np.asarray([point.md_m for point in patch_points], dtype=np.float64)
        curve_values, curve_masks = _interpolate_slots(log, target_md)
        curve_station_valid = curve_masks.any(axis=0)
        if not curve_station_valid.any():
            continue
        patch_points = [
            point
            for point, valid in zip(patch_points, curve_station_valid, strict=True)
            if bool(valid)
        ]
        curve_values = curve_values[:, curve_station_valid]
        curve_masks = curve_masks[:, curve_station_valid]

        eligible_in_patch_station_count = len(patch_points)
        keep = _station_indices(len(patch_points), LAYERPULSE_MAX_WELL_STATIONS)
        patch_points = [patch_points[int(index)] for index in keep]
        curve_values = curve_values[:, keep]
        curve_masks = curve_masks[:, keep]
        curve_values = _normalise_curves(curve_values, curve_masks)

        full_tvd = np.asarray([point.tvd_m for point in full_track], dtype=np.float64)
        tvd_min = float(full_tvd.min())
        tvd_max = float(full_tvd.max())
        selected_wells.append(
            {
                "well_uid": well_uid,
                "well_name": well_name,
                "las_path": str(path),
                "points": patch_points,
                "curves": curve_values,
                "curve_masks": curve_masks,
                "eligible_in_patch_station_count": eligible_in_patch_station_count,
                "tvd_min": tvd_min,
                "tvd_max": tvd_max,
            }
        )

    if not selected_wells:
        return None, _degraded_receipt(
            "no_fusion_ready_well_curves_in_preview",
            prepared_receipt=prepared_receipt,
            warnings=(
                (["registration_wells_without_matching_canonical_las:" + ",".join(unmatched_wells)])
                if unmatched_wells
                else ()
            ),
        )

    well_count = len(selected_wells)
    station_capacity = max(len(item["points"]) for item in selected_wells)
    channel_count = len(LAYERPULSE_CURVE_SLOTS)
    wells = np.zeros((well_count, channel_count, station_capacity), dtype=np.float32)
    curve_mask = np.zeros(wells.shape, dtype=bool)
    well_mask = np.zeros((well_count, station_capacity), dtype=bool)
    well_md_m = np.zeros((well_count, station_capacity), dtype=np.float32)
    trajectory = np.zeros((well_count, station_capacity, 3), dtype=np.float32)
    metric_xyz = np.zeros_like(trajectory)
    trajectory_tvd_m = np.zeros((well_count, station_capacity), dtype=np.float32)
    trajectory_mask = np.zeros_like(well_mask)

    well_summaries: list[dict[str, Any]] = []
    for well_index, item in enumerate(selected_wells):
        selected_points = item["points"]
        count = len(selected_points)
        wells[well_index, :, :count] = item["curves"]
        curve_mask[well_index, :, :count] = item["curve_masks"]
        well_mask[well_index, :count] = True
        trajectory_mask[well_index, :count] = True
        md = np.asarray([point.md_m for point in selected_points], dtype=np.float32)
        tvd = np.asarray([point.tvd_m for point in selected_points], dtype=np.float32)
        x = np.asarray([point.x for point in selected_points], dtype=np.float32)
        y = np.asarray([point.y for point in selected_points], dtype=np.float32)
        well_md_m[well_index, :count] = md
        trajectory_tvd_m[well_index, :count] = tvd
        metric_xyz[well_index, :count] = np.column_stack((x, y, tvd))
        tvd_span = float(item["tvd_max"] - item["tvd_min"])
        normalised_tvd = (
            2.0 * (tvd.astype(np.float64) - float(item["tvd_min"])) / tvd_span - 1.0
            if tvd_span > 1.0e-6
            else np.zeros(count, dtype=np.float64)
        )
        trajectory[well_index, :count, 0] = np.clip(normalised_tvd, -1.0, 1.0)
        trajectory[well_index, :count, 1] = np.asarray(
            [_normalise_axis(float(point.inline), inline_axis) for point in selected_points],
            dtype=np.float32,
        )
        trajectory[well_index, :count, 2] = np.asarray(
            [
                _normalise_axis(float(point.crossline), crossline_axis)
                for point in selected_points
            ],
            dtype=np.float32,
        )
        well_summaries.append(
            {
                "well_uid": item["well_uid"],
                "well_name": item["well_name"],
                "station_count": count,
                "observed_curve_fraction": float(item["curve_masks"].mean()),
            }
        )

    arrays: dict[str, np.ndarray] = {
        "wells": wells,
        "well_curve_mask": curve_mask,
        "well_mask": well_mask,
        "well_md": well_md_m.copy(),
        "well_md_m": well_md_m,
        "trajectory": trajectory,
        "trajectory_metric_xyz": metric_xyz,
        "trajectory_tvd_m": trajectory_tvd_m,
        "trajectory_mask": trajectory_mask,
        # Registration V3 does not declare a branch-parent graph.  Each sealed
        # trajectory is therefore an explicit ROOT until such lineage exists.
        "well_parent_index": np.full((well_count,), -1, dtype=np.int64),
        "well_kickoff_md_m": np.zeros((well_count,), dtype=np.float32),
    }
    if set(arrays) != LAYERPULSE_WELL_ARRAY_KEYS:
        raise AssertionError("internal LayerPulse well-bundle key contract drifted")
    for key, array in arrays.items():
        if key not in {"well_curve_mask", "well_mask", "trajectory_mask"} and not np.isfinite(array).all():
            raise ValueError(f"LayerPulse well bundle contains non-finite values: {key}")

    output_path = _destination_path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    valid_station_count = int(well_mask.sum())
    eligible_in_patch_station_count = sum(
        int(item["eligible_in_patch_station_count"]) for item in selected_wells
    )
    observed = int(curve_mask.sum())
    receipt = {
        **prepared_receipt,
        "schema_version": LAYERPULSE_WELL_BUNDLE_SCHEMA,
        "input_contract": LAYERPULSE_WELL_INPUT_CONTRACT,
        "status": "ready",
        "bundle_materialized": True,
        "bundle_path": str(output_path),
        "well_count": well_count,
        "station_capacity": station_capacity,
        "valid_station_count": valid_station_count,
        "eligible_in_patch_station_count": eligible_in_patch_station_count,
        "well_channels": channel_count,
        "curve_slot_names": [name for name, _aliases in LAYERPULSE_CURVE_SLOTS],
        "observed_curve_fraction": float(observed / max(curve_mask.size, 1)),
        "curve_normalization": "per_well_robust_mad_clip_6;RXO_RT_log10",
        "trajectory_order": "normalized_physical_tvd_preview_inline_preview_xline",
        "trajectory_metric_xyz_order": ["X_m", "Y_m", "TVD_down_m"],
        "maximum_stations_per_well": LAYERPULSE_MAX_WELL_STATIONS,
        "root_topology_defaulted": True,
        "registration_validation": {
            "point_count": int(validation.get("point_count") or len(points)),
            "well_count": int(validation.get("well_count") or len(grouped)),
        },
        "grid_location_source": (
            "sealed_segy_xy_to_grid_affine"
            if projected_grid_point_count
            else "registration_inline_crossline"
        ),
        "projected_grid_point_count": projected_grid_point_count,
        "xy_to_grid_projection": (
            {
                key: value
                for key, value in dict(input_provenance.get("xy_to_grid_projection") or {}).items()
                if key
                not in {
                    "inline_coefficients",
                    "crossline_coefficients",
                    "origin_xy",
                }
            }
            if projected_grid_point_count
            else None
        ),
        "wells": well_summaries,
        "time_depth_table_consumed": False,
        "registration_time_fields_consumed": False,
        "warnings": (
            ["registration_wells_without_matching_canonical_las:" + ",".join(unmatched_wells)]
            if unmatched_wells
            else []
        ),
    }
    return output_path, receipt


__all__ = [
    "LAYERPULSE_CURVE_SLOTS",
    "LAYERPULSE_MAX_WELL_STATIONS",
    "LAYERPULSE_WELL_ARRAY_KEYS",
    "LAYERPULSE_WELL_BUNDLE_SCHEMA",
    "LAYERPULSE_WELL_INPUT_CONTRACT",
    "materialize_layerpulse_well_bundle",
]
