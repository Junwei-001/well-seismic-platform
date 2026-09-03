from __future__ import annotations

import base64
import csv
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import numpy as np


NINE_CURVES = (
    ("SP", "自然电位 SP", "#d97706", False),
    ("GR", "自然伽马 GR", "#16a34a", False),
    ("CAL", "井径 CAL", "#64748b", False),
    ("DT", "声波时差 AC", "#2563eb", False),
    ("NPHI", "中子 CNL", "#7c3aed", False),
    ("RHOB", "密度 DEN", "#dc2626", False),
    ("MSFL", "微球聚焦 MSFL", "#0891b2", True),
    ("RS", "浅侧向 LLS", "#ea580c", True),
    ("RT", "深侧向 LLD", "#be123c", True),
)


def _sample_positions(size: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, max(0, size - 1), min(size, count), dtype=np.int64))


def _finite_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values if np.isfinite(value)]


def _nullable_list(values: np.ndarray) -> list[float | None]:
    return [round(float(value), 5) if np.isfinite(value) else None for value in values]


WELL_GEOMETRY_LABELS = {
    "vertical": "直井",
    "deviated": "斜井",
    "horizontal": "水平井",
}


def _well_key(value: str) -> str:
    """Return a conservative key used only to match frozen per-well products."""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _align_prediction_roots(pipeline: Any) -> list[Path]:
    """Locate only products explicitly bound to the current task manifest."""
    candidates: list[Path] = []
    manifest = getattr(pipeline, "manifest", {}) or {}
    explicit = manifest.get("align_predictions_root")
    if explicit:
        candidates.append(Path(str(explicit)).expanduser())
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key not in seen and resolved.is_dir():
            result.append(resolved)
            seen.add(key)
    return result


@lru_cache(maxsize=8)
def _align_product_index(root_text: str) -> dict[str, Path]:
    root = Path(root_text)
    return {
        _well_key(item.name): item / "time_depth.csv"
        for item in root.iterdir()
        if item.is_dir() and (item / "time_depth.csv").is_file()
    }


@lru_cache(maxsize=256)
def _load_align_registration(path_text: str) -> tuple[np.ndarray, ...]:
    """Load only the frozen inference columns required by visualization."""
    path = Path(path_text)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    lookup = {str(name).strip().casefold(): index for index, name in enumerate(header)}
    required = ("md", "tvdss", "x", "y", "twt_mean", "twt_std", "quality")
    if any(name not in lookup for name in required):
        raise ValueError(f"Align CSV缺少字段：{path}")
    usecols = tuple(lookup[name] for name in required)
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, ndmin=2)
    md, tvdss, x, y, twt, uncertainty, quality = (
        np.asarray(values[:, index], dtype=float) for index in range(7)
    )
    valid = (
        np.isfinite(md)
        & np.isfinite(tvdss)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(twt)
        & np.isfinite(uncertainty)
        & np.isfinite(quality)
    )
    md, tvdss, x, y, twt, uncertainty, quality = (
        item[valid] for item in (md, tvdss, x, y, twt, uncertainty, quality)
    )
    if md.size < 2:
        raise ValueError(f"Align CSV有效样点不足：{path}")
    order = np.argsort(md)
    md, unique = np.unique(md[order], return_index=True)
    indices = order[unique]
    return (
        md,
        tvdss[indices],
        x[indices],
        y[indices],
        twt[indices],
        uncertainty[indices],
        quality[indices],
    )


def _align_path_for_well(pipeline: Any, well_name: str) -> Path | None:
    key = _well_key(well_name)
    for root in _align_prediction_roots(pipeline):
        path = _align_product_index(str(root)).get(key)
        if path is not None:
            return path
    return None


def _interpolate_generated_registration(
    generated: dict[str, Any],
    query_md: np.ndarray,
    *,
    registration_path: str,
) -> dict[str, Any] | None:
    source_md_all = np.asarray(generated.get("md", []), dtype=float)
    source_twt_all = np.asarray(generated.get("twtMean", []), dtype=float)
    source_std_all = np.asarray(generated.get("twtStd", []), dtype=float)
    source_quality_all = np.asarray(
        generated.get("registrationQuality", []), dtype=float
    )
    if not (
        source_md_all.shape
        == source_twt_all.shape
        == source_std_all.shape
        == source_quality_all.shape
    ):
        return None
    declared_valid = np.asarray(
        generated.get(
            "validMask", np.ones(source_md_all.shape, dtype=bool)
        ),
        dtype=bool,
    )
    if declared_valid.shape != source_md_all.shape:
        return None
    core_valid = (
        declared_valid
        & np.isfinite(source_md_all)
        & np.isfinite(source_twt_all)
    )
    if int(np.sum(core_valid)) < 2:
        return None

    source_md = source_md_all[core_valid]
    order = np.argsort(source_md)
    source_md = source_md[order]
    source_twt = source_twt_all[core_valid][order]
    source_md, unique_indices = np.unique(source_md, return_index=True)
    source_twt = source_twt[unique_indices]
    if source_md.size < 2:
        return None
    inside = (
        np.isfinite(query_md)
        & (query_md >= source_md[0])
        & (query_md <= source_md[-1])
    )
    twt_mean = np.full(query_md.shape, np.nan, dtype=float)
    twt_std = np.full(query_md.shape, np.nan, dtype=float)
    quality = np.full(query_md.shape, np.nan, dtype=float)
    twt_mean[inside] = np.interp(query_md[inside], source_md, source_twt)

    std_valid = core_valid & np.isfinite(source_std_all) & (source_std_all >= 0.0)
    if int(np.sum(std_valid)) >= 2:
        std_order = np.argsort(source_md_all[std_valid])
        std_md = source_md_all[std_valid][std_order]
        std_values = source_std_all[std_valid][std_order]
        std_md, std_unique = np.unique(std_md, return_index=True)
        std_values = std_values[std_unique]
        std_inside = (
            inside & (query_md >= std_md[0]) & (query_md <= std_md[-1])
        )
        twt_std[std_inside] = np.interp(
            query_md[std_inside], std_md, std_values
        )

    quality_valid = (
        core_valid
        & np.isfinite(source_quality_all)
        & (source_quality_all >= 0.0)
        & (source_quality_all <= 1.0)
    )
    if int(np.sum(quality_valid)) >= 2:
        quality_order = np.argsort(source_md_all[quality_valid])
        quality_md = source_md_all[quality_valid][quality_order]
        quality_values = source_quality_all[quality_valid][quality_order]
        quality_md, quality_unique = np.unique(quality_md, return_index=True)
        quality_values = quality_values[quality_unique]
        quality_inside = (
            inside
            & (query_md >= quality_md[0])
            & (query_md <= quality_md[-1])
        )
        quality[quality_inside] = np.interp(
            query_md[quality_inside], quality_md, quality_values
        )

    visualization_only = bool(generated.get("visualizationOnly", False))
    formal_registration = bool(
        generated.get("formalRegistration", not visualization_only)
    )
    return {
        "twtMean": _nullable_list(twt_mean),
        "twtStd": _nullable_list(twt_std),
        "registrationQuality": _nullable_list(quality),
        "registrationSource": str(
            generated.get("registrationSource", "well_tie")
        ),
        "registrationStatus": str(
            generated.get("registrationStatus", "estimated_tie")
        ),
        "registrationCoverage": round(float(np.mean(inside)), 4),
        "registrationPath": registration_path,
        "trajectoryTimePolicy": str(
            generated.get("trajectoryTimePolicy")
            or (generated.get("diagnostics") or {}).get(
                "trajectory_time_policy"
            )
            or "strict_md_twt_v1"
        ),
        "visualizationOnly": visualization_only,
        "formalRegistration": formal_registration,
        "inferenceEligible": bool(generated.get("inferenceEligible", False)),
        "fusionReady": bool(generated.get("fusionReady", False)),
        "supervisionEligible": bool(
            generated.get("supervisionEligible", False)
        ),
        "trainingEligible": bool(generated.get("trainingEligible", False)),
        # A probabilistic formal product may be valid for protected fusion
        # without ever becoming a supervised label. Preview candidates can
        # never become accepted through this presentation path.
        "registrationAccepted": bool(
            not visualization_only
            and generated.get(
                "registrationAccepted",
                generated.get(
                    "fusionReady", generated.get("trainingEligible", False)
                ),
            )
        ),
    }


def _registration_for_well(
    pipeline: Any,
    well_name: str,
    query_md: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    allow_spatial_product: bool = True,
) -> dict[str, Any] | None:
    generated_candidates = (
        (
            getattr(pipeline, "registration_tracks", {}).get(well_name),
            "current_registration_task",
        ),
        (
            getattr(pipeline, "visualization_registration_tracks", {}).get(
                well_name
            ),
            "current_visualization_candidate",
        ),
    )
    for generated, registration_path in generated_candidates:
        if not generated:
            continue
        interpolated = _interpolate_generated_registration(
            generated,
            query_md,
            registration_path=registration_path,
        )
        if interpolated is not None:
            return interpolated

    # A frozen spatial tie carries absolute XYZ.  Never bind it to a display
    # trajectory whose horizontal position was assigned inside the viewer.
    if not allow_spatial_product:
        return None

    key = _well_key(well_name)
    for root in _align_prediction_roots(pipeline):
        path = _align_product_index(str(root)).get(key)
        if path is None:
            continue
        try:
            md, _, source_x, source_y, twt, uncertainty, quality = _load_align_registration(str(path))
        except (OSError, ValueError):
            continue
        overlap = (query_md >= md[0]) & (query_md <= md[-1])
        if np.any(overlap):
            expected_x = np.interp(query_md[overlap], md, source_x)
            expected_y = np.interp(query_md[overlap], md, source_y)
            coordinate_gap = np.hypot(expected_x - query_x[overlap], expected_y - query_y[overlap])
            if not np.any(np.isfinite(coordinate_gap)) or float(np.nanmedian(coordinate_gap)) > 100.0:
                continue
        inside = np.isfinite(query_md) & (query_md >= md[0]) & (query_md <= md[-1])
        twt_mean = np.full(query_md.shape, np.nan, dtype=float)
        twt_std = np.full(query_md.shape, np.nan, dtype=float)
        registration_quality = np.full(query_md.shape, np.nan, dtype=float)
        twt_mean[inside] = np.interp(query_md[inside], md, twt)
        twt_std[inside] = np.interp(query_md[inside], md, uncertainty)
        registration_quality[inside] = np.interp(query_md[inside], md, quality)
        coverage = float(np.mean(inside)) if inside.size else 0.0
        return {
            "twtMean": _nullable_list(twt_mean),
            "twtStd": _nullable_list(twt_std),
            "registrationQuality": _nullable_list(registration_quality),
            "registrationSource": "wellfuse_align_prediction",
            "registrationStatus": "estimated_tie",
            "registrationCoverage": round(coverage, 4),
            "registrationPath": str(path),
            "visualizationOnly": False,
            "formalRegistration": True,
            "inferenceEligible": True,
            "registrationAccepted": True,
        }
    return None


def _explicit_time_depth_preview(
    entity: Any,
    query_md: np.ndarray,
    query_tvd: np.ndarray,
) -> dict[str, Any] | None:
    """Interpolate a parsed time-depth table for display without promoting it.

    The result is deliberately visualization-only.  Formal registration still
    has to pass the normal snapshot and registration gates elsewhere.
    """

    best: tuple[float, dict[str, Any]] | None = None
    for table in getattr(entity, "time_depth", []) or []:
        try:
            source_depth = np.asarray(table.depth, dtype=float).reshape(-1)
            source_time = np.asarray(table.time, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            continue
        count = min(source_depth.size, source_time.size)
        if count < 2:
            continue
        valid = np.isfinite(source_depth[:count]) & np.isfinite(source_time[:count])
        if int(np.sum(valid)) < 2:
            continue
        source_depth = source_depth[:count][valid]
        source_time = source_time[:count][valid]
        order = np.argsort(source_depth)
        source_depth = source_depth[order]
        source_time = source_time[order]
        source_depth, unique_indices = np.unique(source_depth, return_index=True)
        source_time = source_time[unique_indices]
        if source_depth.size < 2:
            continue

        time_unit = str(getattr(table, "time_unit", "unknown") or "unknown").lower()
        if time_unit in {"s", "sec", "second", "seconds"}:
            source_time = source_time * 1000.0
            resolved_unit = "ms"
        elif time_unit in {"us", "microsecond", "microseconds"}:
            source_time = source_time / 1000.0
            resolved_unit = "ms"
        elif time_unit == "ms":
            resolved_unit = "ms"
        else:
            # Common field tables are either seconds near single digits or
            # milliseconds in the hundreds/thousands.  This affects display
            # coordinates only and is retained in provenance below.
            if float(np.nanpercentile(np.abs(source_time), 95.0)) <= 20.0:
                source_time = source_time * 1000.0
                resolved_unit = "ms_from_numeric_scale"
            else:
                resolved_unit = "ms_from_numeric_scale"

        depth_domain = str(
            getattr(table, "depth_domain", "unknown") or "unknown"
        ).lower()
        query_depth = query_tvd if depth_domain in {"tvd", "tvdss"} else query_md
        inside = (
            np.isfinite(query_depth)
            & (query_depth >= source_depth[0])
            & (query_depth <= source_depth[-1])
        )
        if int(np.sum(inside)) < 2:
            continue
        twt_mean = np.full(query_depth.shape, np.nan, dtype=float)
        twt_mean[inside] = np.interp(
            query_depth[inside], source_depth, source_time
        )
        coverage = float(np.mean(inside))
        payload = {
            "twtMean": _nullable_list(twt_mean),
            "twtStd": _nullable_list(np.full(query_depth.shape, np.nan)),
            "registrationQuality": _nullable_list(
                np.where(inside, 0.6, np.nan)
            ),
            "registrationSource": "explicit_time_depth_display",
            "registrationStatus": "display_ready",
            "registrationCoverage": round(coverage, 4),
            "registrationPath": str(getattr(table, "source", "")),
            "trajectoryTimePolicy": "explicit_time_depth_display_v1",
            "timeDepthDomain": depth_domain,
            "timeDepthTimeDomain": str(
                getattr(table, "time_domain", "unknown") or "unknown"
            ).upper(),
            "timeDepthDisplayUnit": resolved_unit,
            "visualizationOnly": True,
            "formalRegistration": False,
            "inferenceEligible": False,
            "fusionReady": False,
            "supervisionEligible": False,
            "trainingEligible": False,
            "registrationAccepted": False,
        }
        if best is None or coverage > best[0]:
            best = (coverage, payload)
    return best[1] if best is not None else None


def _classify_well_geometry(
    md: np.ndarray,
    tvd: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    inclination: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Classify measured geometry from trajectory evidence, never from a well name."""
    count = min(len(md), len(x), len(y))
    valid = (
        np.isfinite(md[:count])
        & np.isfinite(x[:count])
        & np.isfinite(y[:count])
    )
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return {
            "geometryType": "vertical",
            "geometryLabel": "直井",
            "maxInclinationDeg": 0.0,
            "lateralDisplacementM": 0.0,
        }
    radial = np.hypot(x[indices] - x[indices[0]], y[indices] - y[indices[0]])
    lateral = float(np.nanmax(radial))
    md_span = max(float(np.nanmax(md[indices]) - np.nanmin(md[indices])), 1.0)
    tvd_count = min(count, len(tvd))
    tvd_valid = np.flatnonzero(np.isfinite(tvd[:tvd_count]))
    tvd_span = (
        max(float(np.nanmax(tvd[tvd_valid]) - np.nanmin(tvd[tvd_valid])), 1.0)
        if tvd_valid.size >= 2
        else max(float(np.sqrt(max(md_span**2 - lateral**2, 1.0))), 1.0)
    )
    max_inclination = float("nan")
    if inclination is not None:
        inc = np.asarray(inclination, dtype=float)[:count]
        inc = inc[np.isfinite(inc)]
        if inc.size:
            max_inclination = float(np.nanmax(np.abs(inc)))
    if not np.isfinite(max_inclination):
        max_inclination = float(np.degrees(np.arctan2(lateral, tvd_span)))

    if max_inclination >= 75.0 or lateral / tvd_span >= 0.45:
        geometry_type = "horizontal"
    elif max_inclination >= 15.0 or lateral / md_span >= 0.08:
        geometry_type = "deviated"
    else:
        geometry_type = "vertical"
    return {
        "geometryType": geometry_type,
        "geometryLabel": WELL_GEOMETRY_LABELS[geometry_type],
        "maxInclinationDeg": round(max_inclination, 3),
        "lateralDisplacementM": round(lateral, 3),
    }


def _well_log_payloads(pipeline: Any, max_points: int = 720) -> list[dict[str, Any]]:
    """Expose real, standardized conventional-nine curves as a lightweight preview."""
    result: list[dict[str, Any]] = []
    for entity in pipeline.registry.entities.values():
        for log_index, log in enumerate(entity.logs):
            depth = np.asarray(log.depth, dtype=float)
            if not depth.size:
                continue
            indices = _sample_positions(depth.size, max_points)
            sampled_depth = depth[indices]
            curves: list[dict[str, Any]] = []
            for curve_id, label, color, logarithmic in NINE_CURVES:
                if curve_id not in log.curves:
                    continue
                values = np.asarray(log.curves[curve_id], dtype=float)[indices]
                mask_source = np.asarray(log.masks.get(curve_id, np.isfinite(log.curves[curve_id])), dtype=bool)[indices]
                values = np.where(mask_source, values, np.nan)
                valid_count = int(np.sum(np.isfinite(values)))
                if valid_count == 0:
                    continue
                info = log.curve_info.get(curve_id)
                curves.append({
                    "id": curve_id,
                    "label": label,
                    "unit": info.standard_unit if info is not None else "",
                    "color": color,
                    "scale": "log" if logarithmic else "linear",
                    "values": _nullable_list(values),
                    "validCount": valid_count,
                })
            if not curves:
                continue
            source_name = Path(log.source).name
            display_name = entity.canonical_name if len(entity.logs) == 1 else f"{entity.canonical_name} · {source_name}"
            result.append({
                "id": f"{entity.well_uid}:{log_index}",
                "name": display_name,
                "wellName": entity.canonical_name,
                "source": log.source,
                "version": log.version,
                "depthUnit": "m",
                "depth": _nullable_list(sampled_depth),
                "curves": curves,
                "coverage": f"{len(curves)}/9",
            })
    return result


def _trajectory_payloads(pipeline: Any, max_points: int = 160) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for entity in pipeline.registry.entities.values():
        head = entity.preferred_head
        preferred = entity.preferred_trajectory
        entity_trajectories = list(getattr(entity, "trajectories", []) or [])
        trajectory_candidates = (
            [
                preferred,
                *(item for item in entity_trajectories if item is not preferred),
            ]
            if preferred is not None
            else entity_trajectories
        )
        for trajectory in trajectory_candidates:
            try:
                md = np.asarray(trajectory.md, dtype=float).reshape(-1)
                if trajectory.x is not None and trajectory.y is not None:
                    x = np.asarray(trajectory.x, dtype=float).reshape(-1)
                    y = np.asarray(trajectory.y, dtype=float).reshape(-1)
                    horizontal_geometry_available = True
                    horizontal_placement_source = "measured_absolute_xy"
                elif (
                    head is not None
                    and head.x is not None
                    and head.y is not None
                ):
                    x = float(head.x) + np.asarray(
                        trajectory.x_offset, dtype=float
                    ).reshape(-1)
                    y = float(head.y) + np.asarray(
                        trajectory.y_offset, dtype=float
                    ).reshape(-1)
                    horizontal_geometry_available = True
                    horizontal_placement_source = "measured_offsets_from_wellhead"
                else:
                    x = np.asarray(
                        getattr(trajectory, "x_offset", np.zeros(md.shape)),
                        dtype=float,
                    ).reshape(-1)
                    y = np.asarray(
                        getattr(trajectory, "y_offset", np.zeros(md.shape)),
                        dtype=float,
                    ).reshape(-1)
                    if x.size != md.size or y.size != md.size:
                        x = np.zeros(md.shape, dtype=float)
                        y = np.zeros(md.shape, dtype=float)
                    horizontal_geometry_available = False
                    horizontal_placement_source = "deterministic_interior_grid"
                tvd_source = np.asarray(
                    trajectory.tvd, dtype=float
                ).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                continue
            # XY/MD is enough to keep a measured path visible.  TVD belongs to
            # a stricter vertical-placement tier and may be incomplete without
            # making the whole well disappear from the viewer.
            count = min(len(md), len(x), len(y))
            if count < 2:
                continue
            tvd = np.full(count, np.nan, dtype=float)
            tvd_count = min(count, tvd_source.size)
            if tvd_count:
                tvd[:tvd_count] = tvd_source[:tvd_count]
            valid = np.isfinite(md[:count]) & np.isfinite(x[:count]) & np.isfinite(y[:count])
            indices = np.flatnonzero(valid)
            if indices.size < 2:
                continue
            raw_md = md[indices]
            order = np.argsort(raw_md)
            raw_md = raw_md[order]
            raw_tvd = tvd[indices][order]
            raw_x = x[indices][order]
            raw_y = y[indices][order]
            unique_md, unique_indices = np.unique(raw_md, return_index=True)
            raw_md = unique_md
            raw_tvd = raw_tvd[unique_indices]
            raw_x = raw_x[unique_indices]
            raw_y = raw_y[unique_indices]
            if raw_md.size < 2:
                continue
            vertical_geometry_available = int(np.sum(np.isfinite(raw_tvd))) >= 2
            if not vertical_geometry_available:
                # MD is retained as a display-depth axis so an otherwise usable
                # well is not reduced to a point.  It remains explicitly
                # display-only and cannot satisfy a formal trajectory gate.
                raw_tvd = raw_md - float(raw_md[0])
            display_count = min(max_points, max(2, int(np.ceil(raw_md[-1] - raw_md[0])) + 1))
            output_md = np.linspace(raw_md[0], raw_md[-1], display_count)
            output_x = np.interp(output_md, raw_md, raw_x)
            output_y = np.interp(output_md, raw_md, raw_y)
            valid_tvd = np.isfinite(raw_tvd)
            output_tvd = np.full(output_md.shape, np.nan, dtype=float)
            if int(np.sum(valid_tvd)) >= 2:
                tvd_md = raw_md[valid_tvd]
                tvd_values = raw_tvd[valid_tvd]
                inside_tvd = (output_md >= tvd_md[0]) & (output_md <= tvd_md[-1])
                output_tvd[inside_tvd] = np.interp(
                    output_md[inside_tvd], tvd_md, tvd_values
                )
            try:
                inclination = (
                    np.asarray(trajectory.inclination, dtype=float).reshape(-1)
                    if trajectory.inclination is not None
                    else None
                )
            except (TypeError, ValueError, OverflowError):
                inclination = None
            geometry_classification = _classify_well_geometry(
                md,
                tvd,
                x,
                y,
                inclination,
            )
            geometry_authoritative = bool(
                horizontal_geometry_available and vertical_geometry_available
            )
            if geometry_authoritative:
                geometry_tier = "measured_xyz"
                geometry_method = "测斜轨迹"
            elif horizontal_geometry_available:
                geometry_tier = "measured_xy_md_vertical"
                geometry_method = "测斜平面轨迹 + MD直井垂向"
            elif vertical_geometry_available:
                geometry_tier = "measured_relative_xy_interior"
                geometry_method = "测斜形态 + 体内井位"
            else:
                geometry_tier = "measured_md_interior_vertical"
                geometry_method = "MD直井轨迹"
            payload = {
                "wellUid": str(entity.well_uid),
                "wellId": entity.canonical_name,
                "name": entity.canonical_name,
                "confidence": round(float(trajectory.confidence), 4),
                "x": _finite_list(output_x),
                "y": _finite_list(output_y),
                "tvd": _nullable_list(output_tvd),
                "md": _finite_list(output_md),
                "geometryMethod": geometry_method,
                "geometryTier": geometry_tier,
                "geometryAuthoritative": geometry_authoritative,
                "measuredTrajectory": True,
                "displayOnlyGeometry": not geometry_authoritative,
                "horizontalGeometryAvailable": horizontal_geometry_available,
                "horizontalPlacementSource": horizontal_placement_source,
                "verticalGeometryAvailable": vertical_geometry_available,
                "verticalDisplayAvailable": True,
                **geometry_classification,
            }
            registration = _registration_for_well(
                pipeline,
                entity.canonical_name,
                output_md,
                output_x,
                output_y,
                allow_spatial_product=horizontal_geometry_available,
            )
            if registration is None:
                registration = _explicit_time_depth_preview(
                    entity,
                    output_md,
                    output_tvd,
                )
            if registration is not None:
                payload.update(registration)
            else:
                payload.update({
                    "registrationSource": "none",
                    "registrationCoverage": 0.0,
                })
            trajectories.append(payload)
            break
        else:
            # An unusable preferred trajectory must not suppress a usable
            # secondary survey.  With no DEV at all, keep a display-only
            # vertical path based on the depth support already parsed for the
            # same well.  A volume-specific interior position is assigned
            # later when the wellhead has no usable XY.
            log_depths = [
                np.asarray(log.depth, dtype=float)
                for log in (getattr(entity, "logs", []) or [])
                if len(log.depth)
            ]
            time_depth_depths = [
                np.asarray(table.depth, dtype=float)
                for table in (getattr(entity, "time_depth", []) or [])
                if len(table.depth)
            ]
            finite_parts = [
                depth[np.isfinite(depth)]
                for depth in [*log_depths, *time_depth_depths]
                if np.any(np.isfinite(depth))
            ]
            finite_depths = (
                np.concatenate(finite_parts)
                if finite_parts
                else np.asarray([], dtype=float)
            )
            total_depth = (
                float(np.max(finite_depths))
                if finite_depths.size
                else float(
                    (head.total_depth_md if head is not None else None) or 1.0
                )
            )
            total_depth = max(total_depth, 1.0)
            display_count = min(max_points, 48)
            output_md = np.linspace(0.0, total_depth, display_count)
            has_head_xy = bool(
                head is not None and head.x is not None and head.y is not None
            )
            head_x = float(head.x) if has_head_xy else 0.0
            head_y = float(head.y) if has_head_xy else 0.0
            payload = {
                "wellUid": str(entity.well_uid),
                "wellId": entity.canonical_name,
                "name": entity.canonical_name,
                "confidence": round(
                    min(float(head.confidence), 0.5)
                    if head is not None
                    else 0.25,
                    4,
                ),
                "x": [head_x] * display_count,
                "y": [head_y] * display_count,
                "tvd": _finite_list(output_md),
                "md": _finite_list(output_md),
                "geometryMethod": "MD直井轨迹",
                "geometryTier": (
                    "synthetic_vertical_from_wellhead"
                    if has_head_xy
                    else "synthetic_vertical_interior"
                ),
                "geometryAuthoritative": False,
                "measuredTrajectory": False,
                "displayOnlyGeometry": True,
                "horizontalGeometryAvailable": has_head_xy,
                "horizontalPlacementSource": (
                    "measured_wellhead"
                    if has_head_xy
                    else "deterministic_interior_grid"
                ),
                "verticalGeometryAvailable": True,
                "verticalDisplayAvailable": True,
                "geometryType": "vertical",
                "geometryLabel": "直井",
                "maxInclinationDeg": 0.0,
                "lateralDisplacementM": 0.0,
            }
            registration = _registration_for_well(
                pipeline,
                entity.canonical_name,
                output_md,
                np.full(output_md.shape, head_x),
                np.full(output_md.shape, head_y),
                allow_spatial_product=has_head_xy,
            )
            if registration is None:
                registration = _explicit_time_depth_preview(
                    entity,
                    output_md,
                    output_md,
                )
            payload.update(
                registration
                or {
                    "registrationSource": "none",
                    "registrationCoverage": 0.0,
                }
            )
            trajectories.append(payload)
    return trajectories


def _clip_segment_to_unit_square(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float] | None:
    """Return the visible parameter interval for one 2-D line segment."""

    dx = x1 - x0
    dy = y1 - y0
    lower = 0.0
    upper = 1.0
    for coefficient, boundary in (
        (-dx, x0),
        (dx, 1.0 - x0),
        (-dy, y0),
        (dy, 1.0 - y0),
    ):
        if abs(coefficient) <= 1e-12:
            if boundary < 0.0:
                return None
            continue
        ratio = boundary / coefficient
        if coefficient < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return lower, upper


def _visible_polyline_positions(
    normalized_x: np.ndarray,
    normalized_y: np.ndarray,
) -> np.ndarray:
    """Return station parameters for the longest portion intersecting the cube."""

    runs: list[list[tuple[int, float, float]]] = []
    current: list[tuple[int, float, float]] = []
    for index in range(max(0, normalized_x.size - 1)):
        clipped = _clip_segment_to_unit_square(
            float(normalized_x[index]),
            float(normalized_y[index]),
            float(normalized_x[index + 1]),
            float(normalized_y[index + 1]),
        )
        if clipped is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append((index, clipped[0], clipped[1]))
    if current:
        runs.append(current)
    if not runs:
        in_bounds = np.flatnonzero(
            (normalized_x >= 0.0)
            & (normalized_x <= 1.0)
            & (normalized_y >= 0.0)
            & (normalized_y <= 1.0)
        )
        return in_bounds[:1].astype(float)
    visible = max(
        runs,
        key=lambda run: (
            sum(max(0.0, end - start) for _, start, end in run),
            len(run),
        ),
    )
    positions: list[float] = []
    for index, start, end in visible:
        for position in (index + start, index + end):
            if not positions or abs(position - positions[-1]) > 1e-9:
                positions.append(position)
    return np.asarray(positions, dtype=float)


def _interpolate_polyline_series(
    values: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Interpolate a station series at clipping points without filling NaNs."""

    if values.ndim != 1 or values.size < 1:
        return np.asarray([], dtype=float)
    result = np.full(positions.shape, np.nan, dtype=float)
    for output_index, position in enumerate(positions):
        left = int(np.floor(position))
        right = min(left + 1, values.size - 1)
        fraction = float(position - left)
        if left < 0 or left >= values.size:
            continue
        if right == left or fraction <= 1e-12:
            result[output_index] = values[left]
        elif np.isfinite(values[left]) and np.isfinite(values[right]):
            result[output_index] = (
                values[left] + fraction * (values[right] - values[left])
            )
    return result


def _embedded_wells(
    geometry: Any,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    trajectories: list[dict[str, Any]],
    time_values: np.ndarray | None = None,
    *,
    allow_time_registration: bool = True,
    time_axis_domain: str = "TWT",
    twt_axis_verified: bool = False,
) -> list[dict[str, Any]]:
    if geometry.x is None or geometry.y is None or geometry.inline is None or geometry.crossline is None:
        return []
    x = np.asarray(geometry.x, dtype=float)
    y = np.asarray(geometry.y, dtype=float)
    inline = np.asarray(geometry.inline, dtype=float)
    crossline = np.asarray(geometry.crossline, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(inline) & np.isfinite(crossline)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 3:
        return []
    fit_indices = valid_indices[_sample_positions(valid_indices.size, 5000)]
    design = np.column_stack((x[fit_indices], y[fit_indices], np.ones(fit_indices.size)))
    try:
        inline_coefficients = np.linalg.lstsq(design, inline[fit_indices], rcond=None)[0]
        crossline_coefficients = np.linalg.lstsq(design, crossline[fit_indices], rcond=None)[0]
    except np.linalg.LinAlgError:
        return []

    inline_min, inline_max = float(inline_values[0]), float(inline_values[-1])
    crossline_min, crossline_max = float(crossline_values[0]), float(crossline_values[-1])
    inline_span = max(abs(inline_max - inline_min), 1.0)
    crossline_span = max(abs(crossline_max - crossline_min), 1.0)
    all_tvd: list[float] = []
    for trajectory in trajectories:
        try:
            values = np.asarray(trajectory.get("tvd", []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        all_tvd.extend(float(value) for value in values[np.isfinite(values)])
    global_tvd_min = min(0.0, float(np.min(all_tvd))) if all_tvd else 0.0
    global_tvd_max = float(np.max(all_tvd)) if all_tvd else 1.0
    seismic_time = np.asarray(time_values if time_values is not None else [], dtype=float)
    seismic_time = seismic_time[np.isfinite(seismic_time)]
    time_min = float(np.min(seismic_time)) if seismic_time.size else float("nan")
    time_max = float(np.max(seismic_time)) if seismic_time.size else float("nan")
    time_span = max(time_max - time_min, 1e-6) if seismic_time.size else float("nan")
    affine_inline = design @ inline_coefficients
    affine_crossline = design @ crossline_coefficients
    fit_rmse = float(np.sqrt(np.mean(
        np.square(affine_inline - inline[fit_indices])
        + np.square(affine_crossline - crossline[fit_indices])
    )))
    full_xy = np.column_stack((x[valid_indices], y[valid_indices]))
    try:
        from scipy.spatial import cKDTree

        trace_tree: Any | None = cKDTree(full_xy)
    except (ImportError, ValueError):
        trace_tree = None

    def placement_key(item: dict[str, Any]) -> str:
        return str(item.get("wellUid") or item.get("wellId") or item.get("name"))

    unplaced = sorted(
        (
            item
            for item in trajectories
            if not bool(item.get("horizontalGeometryAvailable", True))
        ),
        key=placement_key,
    )
    interior_grid: dict[str, tuple[float, float]] = {}
    if unplaced:
        columns = max(1, int(np.ceil(np.sqrt(len(unplaced)))))
        rows = max(1, int(np.ceil(len(unplaced) / columns)))
        for index, item in enumerate(unplaced):
            column = index % columns
            row = index // columns
            grid_x = (
                0.5
                if columns == 1
                else 0.22 + 0.56 * column / (columns - 1)
            )
            grid_y = (
                0.5
                if rows == 1
                else 0.22 + 0.56 * row / (rows - 1)
            )
            interior_grid[placement_key(item)] = (grid_x, grid_y)
    result: list[dict[str, Any]] = []
    for trajectory in trajectories:
        try:
            tx = np.asarray(trajectory.get("x", []), dtype=float).reshape(-1)
            ty = np.asarray(trajectory.get("y", []), dtype=float).reshape(-1)
            tvd = np.asarray(
                [
                    np.nan if value is None else value
                    for value in trajectory.get("tvd", [])
                ],
                dtype=float,
            ).reshape(-1)
            md = np.asarray(
                trajectory.get("md", []), dtype=float
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            continue
        if tx.size < 2 or tx.size != ty.size or tx.size != tvd.size:
            continue
        station_count = tx.size
        horizontal_geometry_available = bool(
            trajectory.get("horizontalGeometryAvailable", True)
        )
        grid_position: tuple[float, float] | None = None
        if horizontal_geometry_available:
            points = np.column_stack((tx, ty, np.ones(tx.size)))
            mapped_inline = points @ inline_coefficients
            mapped_crossline = points @ crossline_coefficients
            normalized_x = (mapped_crossline - crossline_min) / crossline_span
            normalized_y = (mapped_inline - inline_min) / inline_span
        else:
            grid_position = interior_grid.get(
                placement_key(trajectory), (0.5, 0.5)
            )
            offset_x = tx - float(tx[0])
            offset_y = ty - float(ty[0])
            delta_inline = (
                offset_x * inline_coefficients[0]
                + offset_y * inline_coefficients[1]
            )
            delta_crossline = (
                offset_x * crossline_coefficients[0]
                + offset_y * crossline_coefficients[1]
            )
            normalized_x = grid_position[0] + delta_crossline / crossline_span
            normalized_y = grid_position[1] + delta_inline / inline_span
            mapped_inline = inline_min + normalized_y * inline_span
            mapped_crossline = crossline_min + normalized_x * crossline_span
        visible_positions = _visible_polyline_positions(normalized_x, normalized_y)
        if visible_positions.size == 0:
            continue
        tx = _interpolate_polyline_series(tx, visible_positions)
        ty = _interpolate_polyline_series(ty, visible_positions)
        tvd = _interpolate_polyline_series(tvd, visible_positions)
        md = (
            _interpolate_polyline_series(md, visible_positions)
            if md.size == station_count
            else np.asarray([])
        )
        mapped_inline = _interpolate_polyline_series(
            mapped_inline, visible_positions
        )
        mapped_crossline = _interpolate_polyline_series(
            mapped_crossline, visible_positions
        )
        normalized_x = _interpolate_polyline_series(
            normalized_x, visible_positions
        )
        normalized_y = _interpolate_polyline_series(
            normalized_y, visible_positions
        )
        try:
            twt_mean = np.asarray(
                [
                    np.nan if value is None else value
                    for value in trajectory.get("twtMean", [])
                ],
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            twt_mean = np.asarray([], dtype=float)
        try:
            twt_std = np.asarray(
                [
                    np.nan if value is None else value
                    for value in trajectory.get("twtStd", [])
                ],
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            twt_std = np.asarray([], dtype=float)
        try:
            registration_quality = np.asarray(
                [
                    np.nan if value is None else value
                    for value in trajectory.get("registrationQuality", [])
                ],
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            registration_quality = np.asarray([], dtype=float)
        registration_source = str(
            trajectory.get("registrationSource", "none")
        )
        registration_status = str(
            trajectory.get("registrationStatus", "unregistered")
        )
        registration_coverage = float(
            trajectory.get("registrationCoverage", 0.0) or 0.0
        )
        visualization_only = bool(
            trajectory.get("visualizationOnly", False)
        )
        formal_registration = bool(
            trajectory.get("formalRegistration", False)
        )
        inference_eligible = bool(
            trajectory.get("inferenceEligible", False)
        )
        fusion_ready = bool(trajectory.get("fusionReady", False))
        supervision_eligible = bool(
            trajectory.get("supervisionEligible", False)
        )
        training_eligible = bool(
            trajectory.get("trainingEligible", False)
        )
        trajectory_time_policy = str(
            trajectory.get("trajectoryTimePolicy", "strict_md_twt_v1")
        )
        display_only_geometry = bool(
            trajectory.get("displayOnlyGeometry", False)
        )
        if display_only_geometry:
            # Viewer geometry can improve coverage, but must never inherit the
            # scientific permissions of a measured, spatially verified DEV.
            visualization_only = True
            formal_registration = False
            inference_eligible = False
            fusion_ready = False
            supervision_eligible = False
            training_eligible = False
        twt_mean = (
            _interpolate_polyline_series(twt_mean, visible_positions)
            if twt_mean.size == station_count
            else np.asarray([])
        )
        twt_std = (
            _interpolate_polyline_series(twt_std, visible_positions)
            if twt_std.size == station_count
            else np.asarray([])
        )
        registration_quality = (
            _interpolate_polyline_series(
                registration_quality, visible_positions
            )
            if registration_quality.size == station_count
            else np.asarray([])
        )
        display_velocity_m_s: float | None = None
        # Wide Viewer gate: any finite display-depth path can receive a
        # low-confidence constant-velocity placement on the current native
        # seismic axis.  This is presentation state only; it never changes
        # formal registration, inference, fusion or training flags.
        if (
            allow_time_registration
            and seismic_time.size >= 2
            and bool(
                trajectory.get(
                    "verticalDisplayAvailable",
                    int(np.sum(np.isfinite(tvd))) >= 2,
                )
            )
            and (
                twt_mean.size != tvd.size
                or int(np.sum(np.isfinite(twt_mean))) < 2
            )
            and int(np.sum(np.isfinite(tvd))) >= 2
        ):
            velocity_m_s = float(
                trajectory.get("displayVelocityMps", 2000.0) or 2000.0
            )
            if not np.isfinite(velocity_m_s) or velocity_m_s <= 0.0:
                velocity_m_s = 2000.0
            display_velocity_m_s = velocity_m_s
            depth_origin = min(0.0, float(np.nanmin(tvd)))
            travel_factor = 1000.0 if time_axis_domain == "OWT" else 2000.0
            rough_time = (
                time_min
                + travel_factor * (tvd - depth_origin) / velocity_m_s
            )
            rough_valid = (
                np.isfinite(rough_time)
                & (rough_time >= time_min)
                & (rough_time <= time_max)
            )
            rough_indices = np.flatnonzero(rough_valid)
            rough_runs = (
                np.split(
                    rough_indices,
                    np.flatnonzero(np.diff(rough_indices) > 1) + 1,
                )
                if rough_indices.size
                else []
            )
            rough_segment = max(rough_runs, key=len) if rough_runs else np.asarray([])
            if len(rough_segment) >= 2:
                keep = np.zeros(tvd.shape, dtype=bool)
                keep[rough_segment] = True
                twt_mean = np.where(keep, rough_time, np.nan)
                twt_std = np.where(
                    keep,
                    max(250.0, time_span * 0.15),
                    np.nan,
                )
                registration_quality = np.where(keep, 0.12, np.nan)
                registration_source = "constant_velocity_visualization_preview"
                registration_status = "visualization_preview"
                registration_coverage = float(np.mean(keep))
                visualization_only = True
                formal_registration = False
                inference_eligible = False
                fusion_ready = False
                supervision_eligible = False
                training_eligible = False
                if np.any(np.diff(tvd[rough_segment]) <= 0.0):
                    trajectory_time_policy = "trajectory_stationwise_twt_v1"
        time_registered = (
            allow_time_registration
            and
            seismic_time.size >= 2
            and twt_mean.size == tvd.size
            and int(np.sum(np.isfinite(twt_mean))) >= 2
        )
        guide_payload = None
        plan_guides: list[dict[str, Any]] = []
        if time_registered:
            valid_time = np.isfinite(twt_mean) & (twt_mean >= time_min) & (twt_mean <= time_max)
            if int(np.sum(valid_time)) >= 2:
                invalid_indices = np.flatnonzero(~valid_time)
                invalid_runs = (
                    np.split(
                        invalid_indices,
                        np.flatnonzero(np.diff(invalid_indices) > 1) + 1,
                    )
                    if invalid_indices.size
                    else []
                )
                for invalid_run in invalid_runs:
                    if invalid_run.size == 0:
                        continue
                    start = max(0, int(invalid_run[0]) - 1)
                    stop = min(normalized_x.size - 1, int(invalid_run[-1]) + 1)
                    guide_indices = np.arange(start, stop + 1)
                    if guide_indices.size < 2:
                        continue
                    plan_guides.append({
                        "x": np.clip(
                            normalized_x[guide_indices], 0.0, 1.0
                        ).round(5).tolist(),
                        "y": np.clip(
                            normalized_y[guide_indices], 0.0, 1.0
                        ).round(5).tolist(),
                        "z": np.zeros(guide_indices.size, dtype=float).tolist(),
                        "meaning": (
                            "未覆盖TWT的真实DEV水平段，仅投影在地震顶面；未伪造TWT"
                            if time_axis_domain == "TWT"
                            else "未覆盖时间候选的真实DEV水平段，仅投影在地震顶面"
                        ),
                    })
                first_registered = int(np.flatnonzero(valid_time)[0])
                first_registered_z = float((twt_mean[first_registered] - time_min) / time_span)
                if first_registered > 0 and first_registered_z > 0.005:
                    guide_indices = np.arange(first_registered + 1)
                    guide_payload = {
                        "x": np.clip(normalized_x[guide_indices], 0.0, 1.0).round(5).tolist(),
                        "y": np.clip(normalized_y[guide_indices], 0.0, 1.0).round(5).tolist(),
                        "z": np.zeros(guide_indices.size, dtype=float).tolist(),
                        "meaning": (
                            "标定覆盖之上的真实DEV水平轨迹，仅投影在地震顶面；未伪造TWT"
                            if time_axis_domain == "TWT"
                            else "候选覆盖之上的真实DEV水平轨迹，仅投影在地震顶面；未伪造时间值"
                        ),
                    }
                tx, ty, tvd = tx[valid_time], ty[valid_time], tvd[valid_time]
                md = md[valid_time] if md.size == valid_time.size else np.asarray([])
                mapped_inline = mapped_inline[valid_time]
                mapped_crossline = mapped_crossline[valid_time]
                normalized_x = normalized_x[valid_time]
                normalized_y = normalized_y[valid_time]
                twt_mean = twt_mean[valid_time]
                twt_std = twt_std[valid_time] if twt_std.size == valid_time.size else np.full(twt_mean.shape, np.nan)
                registration_quality = (
                    registration_quality[valid_time]
                    if registration_quality.size == valid_time.size
                    else np.full(twt_mean.shape, np.nan)
                )
                normalized_z = (twt_mean - time_min) / time_span
                z_low = (twt_mean - twt_std - time_min) / time_span
                z_high = (twt_mean + twt_std - time_min) / time_span
                registration_accepted = bool(
                    not visualization_only
                    and formal_registration
                    and twt_axis_verified
                    and time_axis_domain == "TWT"
                    and trajectory.get("registrationAccepted", False)
                )
                alignment_mode = (
                    "time_registered"
                    if registration_accepted and time_axis_domain == "TWT"
                    else "time_registration_candidate"
                )
                registration_status = str(
                    registration_status or "estimated_tie"
                )
                registration_source = str(
                    registration_source or "well_tie"
                )
                axis_name = "TWT" if time_axis_domain == "TWT" else "SEG-Y采样轴"
                alignment_status = (
                    f"{axis_name}{'已冻结' if alignment_mode == 'time_registered' else '候选'}"
                    f"（{registration_status} / {registration_source}）"
                )
                vertical_display_mode = (
                    "registered_twt"
                    if alignment_mode == "time_registered"
                    else "native_time_candidate"
                )
            else:
                time_registered = False
        if not time_registered:
            # The visual gate is intentionally more permissive than the
            # scientific MD->TWT gate.  A finite TVD path is placed through the
            # cube by relative depth so the measured trajectory remains
            # inspectable; this placement is explicitly not a time tie and can
            # never be used for inference or interval projection.
            finite_tvd = np.isfinite(tvd)
            global_tvd_span = global_tvd_max - global_tvd_min
            if int(np.sum(finite_tvd)) >= 2 and global_tvd_span > 1e-6:
                normalized_z = np.full(tvd.shape, np.nan, dtype=float)
                normalized_z[finite_tvd] = (
                    tvd[finite_tvd] - global_tvd_min
                ) / global_tvd_span
                alignment_mode = "depth_normalized_preview"
                vertical_display_mode = "relative_tvd_preview"
                alignment_status = (
                    "按当前井组TVD范围嵌入，仅作轨迹几何参考；非TWT标定"
                )
            else:
                normalized_z = np.zeros(tvd.shape, dtype=float)
                alignment_mode = "xy_plan_projection"
                vertical_display_mode = "xy_plan_projection"
                alignment_status = {
                    "TWT": "仅水平XY配准；等待井震标定后进入TWT体",
                    "OWT": "仅水平XY配准；当前为OWT轴，需显式换算或相应注册后进入时间体",
                    "UNKNOWN_TIME": "仅水平XY配准；等待原生SEG-Y时间轴候选后进入预览",
                }.get(time_axis_domain, "仅水平XY配准；等待时间轴合同闭合")
            z_low = np.full(normalized_z.shape, np.nan)
            z_high = np.full(normalized_z.shape, np.nan)
        if not horizontal_geometry_available:
            head_distance: float | None = None
            horizontal_alignment = {
                "method": "deterministic_interior_grid",
                "fitRmseTraceUnits": None,
                "nearestTraceDistanceM": None,
                "placementDerived": True,
                "gridPosition": (
                    [round(grid_position[0], 5), round(grid_position[1], 5)]
                    if grid_position is not None
                    else [0.5, 0.5]
                ),
            }
        elif trace_tree is not None:
            head_distance = float(trace_tree.query([float(tx[0]), float(ty[0])], k=1)[0])
            horizontal_alignment = {
                "method": "SEG-Y XY道头最小二乘仿射映射",
                "fitRmseTraceUnits": round(fit_rmse, 4),
                "nearestTraceDistanceM": round(head_distance, 3),
                "placementDerived": False,
            }
        else:
            head_distance = float(np.min(np.hypot(full_xy[:, 0] - tx[0], full_xy[:, 1] - ty[0])))
            horizontal_alignment = {
                "method": "SEG-Y XY道头最小二乘仿射映射",
                "fitRmseTraceUnits": round(fit_rmse, 4),
                "nearestTraceDistanceM": round(head_distance, 3),
                "placementDerived": False,
            }
        if display_only_geometry:
            alignment_status = (
                "时间轴预览" if time_registered else "井轨迹预览"
            )
        clipped_z = np.clip(np.nan_to_num(normalized_z, nan=0.0), 0.0, 1.0)
        result.append({
            "wellUid": str(trajectory.get("wellUid") or ""),
            "wellId": str(trajectory.get("wellId") or trajectory["name"]),
            "name": trajectory["name"],
            "x": np.clip(normalized_x, 0.0, 1.0).round(5).tolist(),
            "y": np.clip(normalized_y, 0.0, 1.0).round(5).tolist(),
            "z": clipped_z.round(5).tolist(),
            "mdM": np.round(md, 4).tolist() if md.size == clipped_z.size else [],
            "zLow": (
                np.clip(z_low, 0.0, 1.0).round(5).tolist()
                if time_registered and np.any(np.isfinite(z_low))
                else []
            ),
            "zHigh": (
                np.clip(z_high, 0.0, 1.0).round(5).tolist()
                if time_registered and np.any(np.isfinite(z_high))
                else []
            ),
            "geometryMethod": (
                trajectory["geometryMethod"]
                if display_only_geometry
                else (
                    f"{trajectory['geometryMethod']}；XY按SEG-Y道头自动映射；"
                    + (
                        "垂向采用当前注册任务的TWT_mean/std"
                        if time_axis_domain == "TWT"
                        else "垂向采用当前任务的原生SEG-Y时间轴候选（轴域未核验）"
                    )
                )
                if time_registered
                else (
                    f"{trajectory['geometryMethod']}；XY按SEG-Y道头自动映射；"
                    + (
                        "垂向按井组TVD范围归一显示（非TWT）"
                        if alignment_mode == "depth_normalized_preview"
                        else "缺少可用TVD，仅投影到地震顶面"
                    )
                )
            ),
            "geometryConfidence": round(
                float(trajectory["confidence"])
                * (float(np.nanmedian(registration_quality)) if time_registered and np.any(np.isfinite(registration_quality)) else 0.45),
                4,
            ),
            "geometryType": trajectory.get("geometryType", "vertical"),
            "geometryLabel": trajectory.get("geometryLabel", "直井"),
            "maxInclinationDeg": trajectory.get("maxInclinationDeg", 0.0),
            "lateralDisplacementM": trajectory.get("lateralDisplacementM", 0.0),
            "geometryTier": trajectory.get("geometryTier", "measured_xyz"),
            "measuredTrajectory": bool(
                trajectory.get("measuredTrajectory", True)
            ),
            "displayOnlyGeometry": display_only_geometry,
            "horizontalGeometryAvailable": horizontal_geometry_available,
            "horizontalPlacementSource": trajectory.get(
                "horizontalPlacementSource", "measured_absolute_xy"
            ),
            "globalTvdRangeM": [round(global_tvd_min, 3), round(global_tvd_max, 3)],
            "distance": round(head_distance, 3) if head_distance is not None else None,
            "horizontalAlignment": horizontal_alignment,
            "alignmentMode": alignment_mode,
            "alignmentStatus": alignment_status,
            "verticalDisplayMode": vertical_display_mode,
            "timeAxisDomain": time_axis_domain,
            "provenTwt": bool(
                twt_axis_verified and time_axis_domain == "TWT"
            ),
            "registrationSource": registration_source,
            "registrationCoverage": round(registration_coverage, 4),
            "visualizationOnly": visualization_only,
            "formalRegistration": formal_registration,
            "inferenceEligible": inference_eligible,
            "fusionReady": fusion_ready,
            "supervisionEligible": supervision_eligible,
            "trainingEligible": training_eligible,
            "trajectoryTimePolicy": trajectory_time_policy,
            "timeDepthDomain": trajectory.get("timeDepthDomain"),
            "timeDepthTimeDomain": trajectory.get("timeDepthTimeDomain"),
            "timeDepthDisplayUnit": trajectory.get("timeDepthDisplayUnit"),
            "displayVelocityMps": (
                round(display_velocity_m_s, 3)
                if display_velocity_m_s is not None
                else None
            ),
            "medianTwtStdMs": (
                round(float(np.nanmedian(twt_std)), 3)
                if time_registered
                and time_axis_domain == "TWT"
                and np.any(np.isfinite(twt_std))
                else None
            ),
            "medianTimeStdMs": (
                round(float(np.nanmedian(twt_std)), 3)
                if time_registered and np.any(np.isfinite(twt_std))
                else None
            ),
            "topGuide": guide_payload,
            "planGuides": plan_guides,
        })
    return result


def _vertical_axis_contract(
    pipeline: Any,
    asset: Any,
    values: np.ndarray,
) -> dict[str, Any]:
    """Describe the SEG-Y vertical axis without guessing TWT on a new survey."""
    metadata = getattr(pipeline, "time_reference_metadata", {}).get(str(asset.path))
    raw_domain = str(getattr(metadata, "time_domain", "unknown") or "unknown").upper()
    domain = raw_domain if raw_domain in {"TWT", "OWT"} else "UNKNOWN_TIME"
    reference = str(getattr(metadata, "time_reference", "unknown") or "unknown")
    correction_state = str(
        getattr(metadata, "correction_state", "unknown") or "unknown"
    )
    twt_verified = (
        domain == "TWT"
        and reference == "SRD"
        and correction_state == "corrected_to_srd"
    )
    label = {
        "TWT": "双程时 TWT" if twt_verified else "双程时 TWT（基准未完全核验）",
        "OWT": "单程时 OWT",
        "UNKNOWN_TIME": "采样时间（TWT/OWT未核验）",
    }[domain]
    return {
        "contractVersion": "well-seismic.vertical-axis.v2",
        "domain": domain,
        "label": label,
        "unit": "ms",
        "reference": reference,
        "correctionState": correction_state,
        "direction": "increasing_downward",
        "top": round(float(values[0]), 3),
        "bottom": round(float(values[-1]), 3),
        "defaultView": "top_oblique",
        "twtCandidate": domain == "TWT",
        "provenTwt": twt_verified,
        "twtVerified": twt_verified,
    }


def _horizontal_extent_contract(geometry: Any) -> dict[str, float | str]:
    """Estimate real horizontal spans from SEG-Y trace-header XY coordinates."""
    if any(
        value is None
        for value in (geometry.x, geometry.y, geometry.inline, geometry.crossline)
    ):
        return {"method": "preview_shape_fallback"}
    x = np.asarray(geometry.x, dtype=float)
    y = np.asarray(geometry.y, dtype=float)
    inline = np.asarray(geometry.inline, dtype=float)
    crossline = np.asarray(geometry.crossline, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(inline) & np.isfinite(crossline)
    indices = np.flatnonzero(valid)
    if indices.size < 4:
        return {"method": "preview_shape_fallback"}
    indices = indices[_sample_positions(indices.size, 10000)]
    design = np.column_stack((inline[indices], crossline[indices], np.ones(indices.size)))
    try:
        x_coefficients = np.linalg.lstsq(design, x[indices], rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design, y[indices], rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"method": "preview_shape_fallback"}
    inline_step = float(np.hypot(x_coefficients[0], y_coefficients[0]))
    crossline_step = float(np.hypot(x_coefficients[1], y_coefficients[1]))
    inline_span = inline_step * float(np.nanmax(inline[indices]) - np.nanmin(inline[indices]))
    crossline_span = crossline_step * float(
        np.nanmax(crossline[indices]) - np.nanmin(crossline[indices])
    )
    if min(inline_span, crossline_span) <= 0.0:
        return {"method": "preview_shape_fallback"}
    return {
        "method": "SEG-Y_trace_header_XY_affine",
        "inlineSpanM": round(inline_span, 3),
        "crosslineSpanM": round(crossline_span, 3),
        "inlineStepM": round(inline_step, 4),
        "crosslineStepM": round(crossline_step, 4),
    }


def _build_line_preview(
    asset: Any,
    reader: Any,
    geometry: Any,
    *,
    max_trace_samples: int,
    max_time_samples: int,
) -> dict[str, Any] | None:
    trace_indices = _sample_positions(int(geometry.trace_count), max_trace_samples)
    time_indices = _sample_positions(int(geometry.samples_per_trace), max_time_samples)
    if trace_indices.size < 2 or time_indices.size < 2:
        return None

    image = np.zeros((time_indices.size, trace_indices.size), dtype=np.float32)
    for column, trace_index in enumerate(trace_indices):
        trace = np.asarray(reader.read_trace(int(trace_index)), dtype=np.float32)
        image[:, column] = trace[time_indices]

    finite = np.abs(image[np.isfinite(image)])
    scale = float(np.percentile(finite, 99.0)) if finite.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    encoded = np.clip(np.nan_to_num(image / scale) * 127.0, -127, 127).astype(np.int8)

    inline = np.asarray(geometry.inline, dtype=float) if geometry.inline is not None else np.asarray([])
    crossline = np.asarray(geometry.crossline, dtype=float) if geometry.crossline is not None else np.asarray([])
    inline_count = int(np.unique(inline[np.isfinite(inline)]).size) if inline.size else 0
    crossline_count = int(np.unique(crossline[np.isfinite(crossline)]).size) if crossline.size else 0
    if inline_count >= crossline_count and inline.size:
        line_axis = "Inline"
        trace_values = inline[trace_indices]
    elif crossline.size:
        line_axis = "Crossline"
        trace_values = crossline[trace_indices]
    else:
        line_axis = "Trace"
        trace_values = trace_indices.astype(float)

    distance_values: np.ndarray | None = None
    if geometry.x is not None and geometry.y is not None:
        x = np.asarray(geometry.x, dtype=float)[trace_indices]
        y = np.asarray(geometry.y, dtype=float)[trace_indices]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            steps = np.hypot(np.diff(x), np.diff(y))
            distance_values = np.r_[0.0, np.cumsum(steps)]

    time_values = np.asarray(geometry.time_axis, dtype=float)[time_indices]
    return {
        "name": asset.path.name,
        "path": str(Path(asset.path)),
        "lineAxis": line_axis,
        "traceValues": _nullable_list(trace_values),
        "distanceValues": _nullable_list(distance_values) if distance_values is not None else [],
        "timeValues": _nullable_list(time_values),
        "image": {
            "shape": [int(value) for value in encoded.shape],
            "encoding": "base64-int8",
            "values": base64.b64encode(encoded.tobytes(order="C")).decode("ascii"),
        },
        "preview": {
            "sampled_traces": int(trace_indices.size),
            "source_trace_count": int(geometry.trace_count),
            "amplitude_scale_p99": scale,
        },
    }


def build_visualization_preview(
    pipeline: Any,
    *,
    max_volumes: int = 3,
    max_lines: int = 12,
    max_time_samples: int = 72,
    max_inline_samples: int = 24,
    max_crossline_samples: int = 32,
    max_line_time_samples: int = 320,
    max_line_trace_samples: int = 480,
) -> dict[str, Any]:
    """从当前任务稀疏读取真实SEG-Y，生成模型无关的二维/三维预览。"""
    trajectories = _trajectory_payloads(pipeline)
    volumes: list[dict[str, Any]] = []
    lines2d: list[dict[str, Any]] = []
    issues: list[str] = []
    for asset, reader in pipeline.seismic:
        geometry = reader.geometry
        if geometry is None:
            continue
        unique_inline = (
            np.unique(np.asarray(geometry.inline, dtype=np.int64))
            if geometry.inline is not None
            else np.asarray([], dtype=np.int64)
        )
        unique_crossline = (
            np.unique(np.asarray(geometry.crossline, dtype=np.int64))
            if geometry.crossline is not None
            else np.asarray([], dtype=np.int64)
        )
        if unique_inline.size < 2 or unique_crossline.size < 2:
            if len(lines2d) >= max_lines:
                continue
            try:
                line_preview = _build_line_preview(
                    asset,
                    reader,
                    geometry,
                    max_trace_samples=max_line_trace_samples,
                    max_time_samples=max_line_time_samples,
                )
            except Exception as exc:
                issues.append(f"{asset.path.name}轻量二维预览失败：{exc}")
                continue
            if line_preview is not None:
                line_values = np.asarray(
                    [value for value in line_preview.get("timeValues", []) if value is not None],
                    dtype=float,
                )
                if line_values.size:
                    line_preview["verticalAxis"] = _vertical_axis_contract(
                        pipeline,
                        asset,
                        line_values,
                    )
                lines2d.append(line_preview)
            continue
        if len(volumes) >= max_volumes:
            continue
        inline_values = unique_inline[_sample_positions(unique_inline.size, max_inline_samples)]
        crossline_values = unique_crossline[_sample_positions(unique_crossline.size, max_crossline_samples)]
        time_indices = _sample_positions(int(geometry.samples_per_trace), max_time_samples)
        pair_to_trace: dict[tuple[int, int], int] = {}
        for trace_index, (inline_value, crossline_value) in enumerate(zip(geometry.inline, geometry.crossline)):
            pair_to_trace.setdefault((int(inline_value), int(crossline_value)), trace_index)

        cube = np.zeros((time_indices.size, inline_values.size, crossline_values.size), dtype=np.float32)
        loaded_traces = 0
        try:
            for inline_index, inline_value in enumerate(inline_values):
                for crossline_index, crossline_value in enumerate(crossline_values):
                    trace_index = pair_to_trace.get((int(inline_value), int(crossline_value)))
                    if trace_index is None:
                        continue
                    trace = np.asarray(reader.read_trace(trace_index), dtype=np.float32)
                    cube[:, inline_index, crossline_index] = trace[time_indices]
                    loaded_traces += 1
        except Exception as exc:
            issues.append(f"{asset.path.name}轻量三维预览失败：{exc}")
            continue
        if loaded_traces == 0:
            issues.append(f"{asset.path.name}未形成可读取的Inline/Crossline稀疏网格")
            continue
        finite = np.abs(cube[np.isfinite(cube)])
        scale = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        encoded_cube = np.clip(np.nan_to_num(cube / scale) * 127.0, -127, 127).astype(np.int8)
        time_values = np.asarray(geometry.time_axis, dtype=float)[time_indices]
        vertical_axis = _vertical_axis_contract(pipeline, asset, time_values)
        time_axis_domain = str(vertical_axis["domain"])
        proven_twt = bool(vertical_axis["provenTwt"])
        if proven_twt:
            vertical_note = (
                "已核验井使用当前任务的TWT_mean/std；其他井以明确标记的时间候选或"
                "相对TVD参考轨迹显示，不参与精确层段投影"
            )
        elif time_axis_domain == "OWT":
            vertical_note = (
                "当前SEG-Y轴为OWT；有实测DEV的井可作低置信恒速时间预览，"
                "但未显式换算前不写入或宣称为TWT"
            )
        else:
            vertical_note = (
                "具备相对时间候选的井按当前原生SEG-Y采样轴显示，仅供审阅；"
                "TWT/OWT尚未核验，不称TWT"
            )
        horizontal_extent = _horizontal_extent_contract(geometry)
        volumes.append({
            "name": asset.path.name,
            "path": str(Path(asset.path)),
            "inline": int(inline_values[len(inline_values) // 2]),
            "crossline": int(crossline_values[len(crossline_values) // 2]),
            "time": float(time_values[len(time_values) // 2]),
            "timeMax": float(time_values[-1]),
            "inlineRange": [int(inline_values[0]), int(inline_values[-1])],
            "crosslineRange": [int(crossline_values[0]), int(crossline_values[-1])],
            "inlineValues": [int(value) for value in inline_values],
            "crosslineValues": [int(value) for value in crossline_values],
            "timeValues": [round(float(value), 3) for value in time_values],
            "defaultIndices": [int(time_values.size // 2), int(inline_values.size // 2), int(crossline_values.size // 2)],
            "cube": {
                "shape": [int(value) for value in encoded_cube.shape],
                "encoding": "base64-int8",
                "values": base64.b64encode(encoded_cube.tobytes(order="C")).decode("ascii"),
            },
            "embeddedWells": _embedded_wells(
                geometry,
                inline_values,
                crossline_values,
                trajectories,
                time_values,
                allow_time_registration=time_axis_domain in {
                    "TWT",
                    "OWT",
                    "UNKNOWN_TIME",
                },
                time_axis_domain=time_axis_domain,
                twt_axis_verified=bool(vertical_axis.get("twtVerified", False)),
            ),
            "verticalAxis": vertical_axis,
            "horizontalExtent": horizontal_extent,
            "preview": {
                "loaded_traces": loaded_traces,
                "source_trace_count": int(geometry.trace_count),
                "amplitude_scale_p99": scale,
                "vertical_note": vertical_note,
            },
        })
    geometry_counts = {
        geometry_type: sum(1 for item in trajectories if item.get("geometryType") == geometry_type)
        for geometry_type in WELL_GEOMETRY_LABELS
    }
    embedded_wells = [
        well
        for volume in volumes
        for well in volume.get("embeddedWells", [])
    ]
    registration_candidate_wells = len(
        {
            str(item.get("wellUid") or item.get("wellId") or item.get("name"))
            for item in embedded_wells
            if item.get("registrationSource") not in {None, "", "none"}
            and float(item.get("registrationCoverage", 0.0)) > 0.0
        }
    )
    formal_registration_candidates = len(
        {
            str(item.get("wellUid") or item.get("wellId") or item.get("name"))
            for item in embedded_wells
            if bool(item.get("formalRegistration", False))
            and float(item.get("registrationCoverage", 0.0)) > 0.0
        }
    )
    visualization_only_candidates = len(
        {
            str(item.get("wellUid") or item.get("wellId") or item.get("name"))
            for item in embedded_wells
            if bool(item.get("visualizationOnly", False))
            and float(item.get("registrationCoverage", 0.0)) > 0.0
        }
    )
    time_mounted_well_names = {
        str(well.get("name"))
        for volume in volumes
        for well in volume.get("embeddedWells", [])
        if well.get("alignmentMode")
        in {"time_registered", "time_registration_candidate"}
    }
    aligned_wells = len(time_mounted_well_names)
    embedded_well_names = {
        str(well.get("wellUid") or well.get("wellId") or well.get("name"))
        for well in embedded_wells
    }
    depth_preview_well_names = {
        str(well.get("wellUid") or well.get("wellId") or well.get("name"))
        for well in embedded_wells
        if well.get("alignmentMode") == "depth_normalized_preview"
    }
    time_axis_domains = sorted({
        str(vertical_axis.get("domain", "UNKNOWN_TIME"))
        for item in [*volumes, *lines2d]
        if isinstance((vertical_axis := item.get("verticalAxis")), dict)
    })
    if len(time_axis_domains) == 1:
        time_axis_domain = time_axis_domains[0]
    elif time_axis_domains:
        time_axis_domain = "MIXED"
    else:
        time_axis_domain = "UNAVAILABLE"
    vertical_axis_contracts = [
        vertical_axis
        for item in [*volumes, *lines2d]
        if isinstance((vertical_axis := item.get("verticalAxis")), dict)
    ]
    proven_twt = bool(vertical_axis_contracts) and all(
        bool(contract.get("twtVerified", False))
        for contract in vertical_axis_contracts
    )
    if proven_twt:
        vertical_scale = "XYZ/TVD始终来自当前快照DEV；只有完成TWT标定的井才进入TWT时间体"
        alignment_policy = "XY不依赖时深表；垂向必须先产生可审计TWT注册，禁止把TVD直接冒充TWT"
        alignment_meaning = "proven_twt_registration"
    elif time_axis_domain == "OWT":
        vertical_scale = "XYZ/TVD始终来自当前快照DEV；当前时间体为OWT，未显式换算前不称TWT"
        alignment_policy = "XY不依赖时深表；OWT垂向需使用相应注册或显式换算，禁止冒充TWT"
        alignment_meaning = "owt_registration_candidate"
    else:
        vertical_scale = (
            "XYZ/TVD始终来自当前快照DEV；相对时间候选仅按原生SEG-Y采样轴显示，"
            "轴域未核验时不称TWT"
        )
        alignment_policy = (
            "XY不依赖时深表；原生时间候选可审阅，但TWT/OWT未核验前不得作为TWT注册成果"
        )
        alignment_meaning = "native_time_candidate"
    return {
        "contractVersion": "well-seismic.visualization-preview.v2",
        "volumes": volumes,
        "lines2d": lines2d,
        "trajectories": trajectories,
        "wellGeometrySummary": {
            "counts": geometry_counts,
            "labels": WELL_GEOMETRY_LABELS,
            "classification": "实测井斜优先；无井斜时按XYZ水平位移与TVD分类",
            "verticalScale": vertical_scale,
        },
        "wellTimeAlignmentSummary": {
            "aligned": aligned_wells,
            "horizontalOnly": max(0, len(trajectories) - aligned_wells),
            "embedded": len(embedded_well_names),
            "depthNormalizedPreviews": len(depth_preview_well_names),
            "registrationCandidates": registration_candidate_wells,
            "formalRegistrationCandidates": formal_registration_candidates,
            "visualizationOnlyCandidates": visualization_only_candidates,
            "preferredSource": "current_registration_task_then_visualization_candidate",
            "timeAxisDomain": time_axis_domain,
            "timeAxisDomains": time_axis_domains,
            "provenTwt": proven_twt,
            "alignmentMeaning": alignment_meaning,
            "policy": alignment_policy,
        },
        "wellLogs": _well_log_payloads(pipeline),
        "issues": issues,
        "source": "当前任务真实数据的稀疏降采样预览",
    }
