"""Plan-view well-to-seismic registration without a depth-to-time transform.

This module deliberately has no dependency on Registration V3.  It answers one
scientifically narrower question: where does each measured trajectory station
fall relative to the sealed SEG-Y trace grid?  It never invents TWT, SRD, a
vertical static, or a fusion/training eligibility decision.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .alignment.spatial import build_spatial_aligner
from .content_identity import canonical_sha256, file_sha256


HORIZONTAL_REGISTRATION_CONTRACT_VERSION = "well-seismic.horizontal-registration.v1"
HORIZONTAL_VISUALIZATION_CONTRACT_VERSION = (
    "well-seismic.horizontal-registration-visualization.v1"
)


@dataclass(frozen=True)
class HorizontalRegistrationProduct:
    manifest: dict[str, Any]
    manifest_path: Path
    points_path: Path
    wells_path: Path
    visualization_path: Path
    manifest_sha256: str
    points_sha256: str
    wells_sha256: str
    visualization_sha256: str
    product_sha256: str


@dataclass(frozen=True)
class _GridFootprint:
    asset: Any
    reader: Any
    summary: dict[str, Any]
    hull: np.ndarray | None
    hull_equations: np.ndarray | None
    bounds: tuple[float, float, float, float]
    coverage_radius_m: float

    def contains(self, x: float, y: float) -> bool:
        """Test the sampled convex footprint with one trace-cell tolerance."""

        if self.hull_equations is not None:
            normals = self.hull_equations[:, :2]
            offsets = self.hull_equations[:, 2]
            norm = np.linalg.norm(normals, axis=1)
            signed_distance = (
                normals @ np.asarray([x, y], dtype=float) + offsets
            ) / np.maximum(norm, 1e-12)
            return bool(np.all(signed_distance <= self.coverage_radius_m))
        x_min, x_max, y_min, y_max = self.bounds
        return bool(
            x_min - self.coverage_radius_m <= x <= x_max + self.coverage_radius_m
            and y_min - self.coverage_radius_m <= y <= y_max + self.coverage_radius_m
        )


_POINT_FIELDS = (
    "well_uid",
    "well_name",
    "geometry_mode",
    "geometry_source",
    "station_index",
    "md_m",
    "tvd_m",
    "md_tvd_semantics",
    "x_m",
    "y_m",
    "trajectory_source",
    "trajectory_xy_source",
    "trajectory_crs_id",
    "seismic_asset_id",
    "seismic_source",
    "trace_index",
    "inline",
    "crossline",
    "trace_x_m",
    "trace_y_m",
    "nearest_trace_distance_m",
    "grid_coverage_radius_m",
    "inside_grid_footprint",
    "within_distance_tolerance",
    "covered_by_seismic_grid",
    "horizontal_confidence",
    "qc_status",
)

_WELL_FIELDS = (
    "well_uid",
    "well_name",
    "geometry_mode",
    "geometry_source",
    "md_tvd_semantics",
    "trajectory_source",
    "log_count",
    "station_count",
    "valid_xy_station_count",
    "covered_station_count",
    "coverage_fraction",
    "nearest_distance_median",
    "nearest_distance_p95",
    "nearest_distance_max",
    "horizontal_status",
    "plan_view_usable",
    "fusion_ready",
    "training_eligible",
)

_HEAD_ONLY_MD_TVD_SEMANTICS = (
    "zero_placeholder_no_trajectory_or_vertical_meaning"
)
_TRAJECTORY_MD_TVD_SEMANTICS = "measured_trajectory_coordinates"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _group_neighbor_distances(
    points: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> np.ndarray:
    order = np.lexsort((secondary, primary))
    ordered_points = points[order]
    ordered_primary = primary[order]
    same_group = ordered_primary[1:] == ordered_primary[:-1]
    distances = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
    return distances[same_group & np.isfinite(distances) & (distances > 1e-9)]


def _structured_boundary_indices(
    inline: np.ndarray,
    crossline: np.ndarray,
) -> np.ndarray:
    """Return endpoints of every inline and crossline without rasterizing."""

    selected: list[np.ndarray] = []
    for primary, secondary in ((inline, crossline), (crossline, inline)):
        order = np.lexsort((secondary, primary))
        ordered_primary = primary[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_primary) != 0) + 1]
        stops = np.r_[starts[1:] - 1, len(order) - 1]
        selected.extend((order[starts], order[stops]))
    if not selected:
        return np.asarray([], dtype=int)
    return np.unique(np.concatenate(selected).astype(int))


def _unstructured_spacing(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=float)
    limit = 20_000
    if len(points) > limit:
        sample = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    else:
        sample = points
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(sample).query(sample, k=2)
        nearest = np.asarray(distances, dtype=float)[:, 1]
        return nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    except (ImportError, ValueError):
        order = np.lexsort((sample[:, 1], sample[:, 0]))
        distances = np.linalg.norm(np.diff(sample[order], axis=0), axis=1)
        return distances[np.isfinite(distances) & (distances > 1e-9)]


def _grid_footprint(
    asset: Any,
    reader: Any,
    *,
    maximum_distance_m: float,
) -> _GridFootprint | None:
    geometry = getattr(reader, "geometry", None)
    if geometry is None or geometry.x is None or geometry.y is None:
        return None
    x = np.asarray(geometry.x, dtype=float)
    y = np.asarray(geometry.y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & ~((x == 0.0) & (y == 0.0))
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    points = np.column_stack((x[valid], y[valid]))
    inline_values: np.ndarray | None = None
    crossline_values: np.ndarray | None = None
    structured_headers_available = bool(
        geometry.inline is not None
        and geometry.crossline is not None
        and np.asarray(geometry.inline).shape == x.shape
        and np.asarray(geometry.crossline).shape == x.shape
    )
    structured = False
    grid_structure_reason = "inline_crossline_headers_unavailable"
    spacing_parts: list[np.ndarray] = []
    boundary_local: np.ndarray
    inline_count = 0
    crossline_count = 0
    grid_occupancy = None
    if structured_headers_available:
        inline_values = np.asarray(geometry.inline, dtype=float)[valid]
        crossline_values = np.asarray(geometry.crossline, dtype=float)[valid]
        finite_grid = np.isfinite(inline_values) & np.isfinite(crossline_values)
        points = points[finite_grid]
        valid_indices = valid_indices[finite_grid]
        inline_values = inline_values[finite_grid]
        crossline_values = crossline_values[finite_grid]
        if len(points) == 0:
            return None
        inline_count = int(np.unique(inline_values).size)
        crossline_count = int(np.unique(crossline_values).size)
        expected_cells = inline_count * crossline_count
        grid_occupancy = (
            min(1.0, float(len(points) / expected_cells)) if expected_cells else None
        )
        inline_neighbor_spacing = _group_neighbor_distances(
            points,
            inline_values,
            crossline_values,
        )
        crossline_neighbor_spacing = _group_neighbor_distances(
            points,
            crossline_values,
            inline_values,
        )
        repeated_axes = bool(
            inline_count < len(points) and crossline_count < len(points)
        )
        # A connected set of occupied inline/crossline cells needs at least
        # I + X - 1 observations.  Values below that lower bound are usually
        # unrelated per-trace identifiers that merely happened to be parsed
        # from the nominal inline/xline bytes.
        occupancy_plausible = bool(
            expected_cells
            and len(points) >= inline_count + crossline_count - 1
        )
        bidirectional_neighbors = bool(
            inline_neighbor_spacing.size and crossline_neighbor_spacing.size
        )
        structured = bool(
            repeated_axes and occupancy_plausible and bidirectional_neighbors
        )
        if structured:
            grid_structure_reason = "repeated_axes_and_occupancy_plausible"
            spacing_parts.extend(
                (inline_neighbor_spacing, crossline_neighbor_spacing)
            )
            boundary_local = _structured_boundary_indices(
                inline_values,
                crossline_values,
            )
        else:
            if not repeated_axes:
                grid_structure_reason = "inline_or_crossline_unique_per_trace"
            elif not occupancy_plausible:
                grid_structure_reason = "occupancy_below_connected_grid_minimum"
            else:
                grid_structure_reason = "missing_bidirectional_grid_neighbors"
            boundary_limit = 50_000
            boundary_local = (
                np.linspace(0, len(points) - 1, boundary_limit, dtype=int)
                if len(points) > boundary_limit
                else np.arange(len(points), dtype=int)
            )
            spacing_parts.append(_unstructured_spacing(points))
    else:
        boundary_limit = 50_000
        boundary_local = (
            np.linspace(0, len(points) - 1, boundary_limit, dtype=int)
            if len(points) > boundary_limit
            else np.arange(len(points), dtype=int)
        )
        spacing_parts.append(_unstructured_spacing(points))

    nonempty_spacing = [item for item in spacing_parts if item.size]
    spacing = (
        np.concatenate(nonempty_spacing)
        if nonempty_spacing
        else np.asarray([], dtype=float)
    )
    median_spacing = float(np.median(spacing)) if spacing.size else None
    p95_spacing = float(np.percentile(spacing, 95)) if spacing.size else None
    cell_scale = max(
        1.0,
        float(median_spacing or 0.0),
        float(p95_spacing or 0.0),
    )
    coverage_radius = min(
        float(maximum_distance_m),
        max(1.0, math.sqrt(2.0) * cell_scale),
    )
    boundary_points = points[boundary_local]
    hull: np.ndarray | None = None
    equations: np.ndarray | None = None
    if len(boundary_points) >= 3:
        try:
            from scipy.spatial import ConvexHull

            convex = ConvexHull(boundary_points)
            hull = boundary_points[convex.vertices]
            equations = np.asarray(convex.equations, dtype=float)
        except (ImportError, ValueError):
            hull = None
            equations = None
    bounds = (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )
    summary = {
        "asset_id": str(asset.asset_id),
        "source": str(asset.path),
        "profile": str(geometry.profile),
        "trace_count": int(geometry.trace_count),
        "valid_xy_trace_count": int(len(points)),
        "inline_count": inline_count,
        "crossline_count": crossline_count,
        "grid_occupancy": grid_occupancy,
        "grid_structure": "structured" if structured else "unstructured",
        "grid_structure_reason": grid_structure_reason,
        "spacing_method": (
            "inline_crossline_neighbors"
            if structured
            else "xy_nearest_neighbors"
        ),
        "median_trace_spacing_m": median_spacing,
        "p95_trace_spacing_m": p95_spacing,
        "coverage_radius_m": coverage_radius,
        "bounds_m": {
            "x_min": bounds[0],
            "x_max": bounds[1],
            "y_min": bounds[2],
            "y_max": bounds[3],
        },
        "footprint_method": (
            "structured_inline_crossline_endpoints_convex_hull"
            if equations is not None and structured
            else (
                "sampled_trace_convex_hull"
                if equations is not None
                else "trace_bounds_fallback"
            )
        ),
        "geometry_confidence": float(geometry.confidence),
        "issues": list(geometry.issues),
        "source_crs": geometry.source_crs,
        "horizontal_crs": geometry.horizontal_crs,
    }
    return _GridFootprint(
        asset=asset,
        reader=reader,
        summary=summary,
        hull=hull,
        hull_equations=equations,
        bounds=bounds,
        coverage_radius_m=coverage_radius,
    )


def _trajectory_coordinates(entity: Any) -> tuple[np.ndarray, np.ndarray, str]:
    trajectory = entity.preferred_trajectory
    if trajectory is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), "missing"
    md = np.asarray(trajectory.md, dtype=float)
    if trajectory.x is not None and trajectory.y is not None:
        x = np.asarray(trajectory.x, dtype=float)
        y = np.asarray(trajectory.y, dtype=float)
        if x.shape == md.shape and y.shape == md.shape:
            return x, y, "trajectory_absolute_xy"
    head = entity.preferred_head
    x_offset = np.asarray(trajectory.x_offset, dtype=float)
    y_offset = np.asarray(trajectory.y_offset, dtype=float)
    if (
        head is not None
        and head.x is not None
        and head.y is not None
        and x_offset.shape == md.shape
        and y_offset.shape == md.shape
    ):
        return (
            float(head.x) + x_offset,
            float(head.y) + y_offset,
            "wellhead_plus_trajectory_offsets",
        )
    return (
        np.full(md.shape, np.nan, dtype=float),
        np.full(md.shape, np.nan, dtype=float),
        "unresolved",
    )


def _valid_well_head_xy(head: Any) -> bool:
    if head is None or str(getattr(head, "horizontal_unit", "")).lower() != "m":
        return False
    return _finite_float(getattr(head, "x", None)) is not None and _finite_float(
        getattr(head, "y", None)
    ) is not None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else None


def build_horizontal_registration(
    pipeline: Any,
    *,
    source_snapshot_id: str,
    source_snapshot_fingerprint: str | None,
    horizontal_crs_id: str,
    horizontal_unit: str,
    horizontal_axis_order: str,
    maximum_distance_m: float | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Build full-resolution plan-view matches from an already ingested pipeline."""

    coordinate_reference = pipeline.config.get("matching", {}).get(
        "coordinate_reference", {}
    )
    if coordinate_reference.get("verified") is not True:
        raise ValueError("horizontal registration requires a verified CRS contract")
    if str(horizontal_crs_id or "").strip() == "":
        raise ValueError("horizontal registration requires horizontal_crs_id")
    if str(horizontal_unit).lower() != "m":
        raise ValueError("horizontal registration requires canonical metre XY")
    if str(horizontal_axis_order).upper() not in {"XY", "YX"}:
        raise ValueError("horizontal registration requires an explicit XY axis order")
    if any(entity.time_depth for entity in pipeline.registry.entities.values()):
        raise ValueError(
            "horizontal registration no-time-depth contract detected a TWT/time-depth "
            "curve; create a snapshot that excludes those sources"
        )

    matching = pipeline.config.get("matching", {})
    max_distance = float(
        maximum_distance_m
        if maximum_distance_m is not None
        else matching.get("max_horizontal_distance", 500.0)
    )
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("maximum horizontal distance must be finite and positive")
    seismic_sources = pipeline._selected_seismic_sources(matching)
    footprints = [
        footprint
        for asset, reader in seismic_sources
        if (
            footprint := _grid_footprint(
                asset,
                reader,
                maximum_distance_m=max_distance,
            )
        )
        is not None
    ]
    if not footprints:
        raise ValueError(
            "horizontal registration found no SEG-Y grid with valid XY headers"
        )
    footprints_by_asset = {id(item.asset): item for item in footprints}
    spatial_aligner = build_spatial_aligner({**matching, "neighbor_traces": 1}).fit(
        [(item.asset, item.reader) for item in footprints]
    )

    points: list[dict[str, Any]] = []
    wells: list[dict[str, Any]] = []
    excluded_wells: list[dict[str, Any]] = []
    entities = list(pipeline.registry.entities.values())
    for entity_index, entity in enumerate(entities):
        if progress:
            progress(
                entity_index,
                len(entities),
                f"正在配准井轨迹：{entity.canonical_name}",
            )
        trajectory = entity.preferred_trajectory
        head = entity.preferred_head
        head_xy_valid = _valid_well_head_xy(head)
        reasons: list[str] = []
        if not entity.logs:
            reasons.append("missing_las")
        if head is None:
            reasons.append("missing_well_head")
        elif not head_xy_valid:
            reasons.append("invalid_well_head_xy")
        if trajectory is None and not head_xy_valid:
            reasons.append("missing_trajectory")
        if reasons:
            excluded_wells.append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "reasons": reasons,
                }
            )
            continue
        if trajectory is None:
            # A measured well head is a legitimate single plan-view point.  MD
            # and TVD are explicit zero placeholders for the tabular schema;
            # they carry no trajectory or vertical-registration meaning.
            geometry_mode = "head_only"
            geometry_source = str(head.source)
            md_tvd_semantics = _HEAD_ONLY_MD_TVD_SEMANTICS
            md = np.asarray([0.0], dtype=float)
            tvd = np.asarray([0.0], dtype=float)
            x = np.asarray([float(head.x)], dtype=float)
            y = np.asarray([float(head.y)], dtype=float)
            xy_source = "well_head_xy"
            trajectory_source: str | None = None
            trajectory_crs = head.crs or horizontal_crs_id
            source_confidence = float(head.confidence)
        else:
            geometry_mode = "trajectory"
            geometry_source = str(trajectory.source)
            md_tvd_semantics = _TRAJECTORY_MD_TVD_SEMANTICS
            md = np.asarray(trajectory.md, dtype=float)
            tvd = np.asarray(trajectory.tvd, dtype=float)
            x, y, xy_source = _trajectory_coordinates(entity)
            trajectory_source = str(trajectory.source)
            trajectory_crs = (
                trajectory.horizontal_crs
                or trajectory.source_crs
                or getattr(head, "crs", None)
                or horizontal_crs_id
            )
            source_confidence = float(trajectory.confidence)
        if tvd.shape != md.shape or x.shape != md.shape or y.shape != md.shape:
            excluded_wells.append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "reasons": ["trajectory_column_length_mismatch"],
                }
            )
            continue
        well_points: list[dict[str, Any]] = []
        for station_index in range(len(md)):
            station_md = _finite_float(md[station_index])
            station_tvd = _finite_float(tvd[station_index])
            station_x = _finite_float(x[station_index])
            station_y = _finite_float(y[station_index])
            base = {
                "well_uid": entity.well_uid,
                "well_name": entity.canonical_name,
                "geometry_mode": geometry_mode,
                "geometry_source": geometry_source,
                "station_index": station_index,
                "md_m": station_md,
                "tvd_m": station_tvd,
                "md_tvd_semantics": md_tvd_semantics,
                "x_m": station_x,
                "y_m": station_y,
                "trajectory_source": trajectory_source,
                "trajectory_xy_source": xy_source,
                "trajectory_crs_id": trajectory_crs,
            }
            if station_x is None or station_y is None or station_md is None:
                row = {
                    **base,
                    "seismic_asset_id": None,
                    "seismic_source": None,
                    "trace_index": None,
                    "inline": None,
                    "crossline": None,
                    "trace_x_m": None,
                    "trace_y_m": None,
                    "nearest_trace_distance_m": None,
                    "grid_coverage_radius_m": None,
                    "inside_grid_footprint": False,
                    "within_distance_tolerance": False,
                    "covered_by_seismic_grid": False,
                    "horizontal_confidence": 0.0,
                    "qc_status": "invalid_trajectory_station",
                }
                well_points.append(row)
                points.append(row)
                continue
            nearest = spatial_aligner.match(station_x, station_y)
            if nearest is None:
                row = {
                    **base,
                    "seismic_asset_id": None,
                    "seismic_source": None,
                    "trace_index": None,
                    "inline": None,
                    "crossline": None,
                    "trace_x_m": None,
                    "trace_y_m": None,
                    "nearest_trace_distance_m": None,
                    "grid_coverage_radius_m": None,
                    "inside_grid_footprint": False,
                    "within_distance_tolerance": False,
                    "covered_by_seismic_grid": False,
                    "horizontal_confidence": 0.0,
                    "qc_status": "no_seismic_xy_index",
                }
                well_points.append(row)
                points.append(row)
                continue
            footprint = footprints_by_asset[id(nearest.asset)]
            geometry = nearest.reader.geometry
            assert (
                geometry is not None
                and geometry.x is not None
                and geometry.y is not None
            )
            trace_index = int(nearest.trace_index)
            distance = float(nearest.distance)
            inside = footprint.contains(station_x, station_y)
            within_tolerance = distance <= footprint.coverage_radius_m
            covered = bool(inside and within_tolerance)
            confidence = float(
                source_confidence
                * geometry.confidence
                * math.exp(-distance / max(footprint.coverage_radius_m, 1e-9))
            )
            if covered:
                status = "covered"
            elif not inside:
                status = "outside_grid_footprint"
            else:
                status = "grid_hole_or_sparse_geometry"
            row = {
                **base,
                "seismic_asset_id": str(nearest.asset.asset_id),
                "seismic_source": str(nearest.asset.path),
                "trace_index": trace_index,
                "inline": (
                    int(geometry.inline[trace_index])
                    if geometry.inline is not None
                    and np.isfinite(geometry.inline[trace_index])
                    else None
                ),
                "crossline": (
                    int(geometry.crossline[trace_index])
                    if geometry.crossline is not None
                    and np.isfinite(geometry.crossline[trace_index])
                    else None
                ),
                "trace_x_m": float(geometry.x[trace_index]),
                "trace_y_m": float(geometry.y[trace_index]),
                "nearest_trace_distance_m": distance,
                "grid_coverage_radius_m": footprint.coverage_radius_m,
                "inside_grid_footprint": inside,
                "within_distance_tolerance": within_tolerance,
                "covered_by_seismic_grid": covered,
                "horizontal_confidence": confidence,
                "qc_status": status,
            }
            well_points.append(row)
            points.append(row)

        valid_xy = [
            item
            for item in well_points
            if item["x_m"] is not None and item["y_m"] is not None
        ]
        covered_points = [item for item in valid_xy if item["covered_by_seismic_grid"]]
        distances = [
            float(item["nearest_trace_distance_m"])
            for item in valid_xy
            if item["nearest_trace_distance_m"] is not None
        ]
        coverage_fraction = (
            float(len(covered_points) / len(valid_xy)) if valid_xy else 0.0
        )
        if valid_xy and len(covered_points) == len(valid_xy):
            horizontal_status = "covered"
        elif covered_points:
            horizontal_status = "partial_coverage"
        else:
            horizontal_status = "outside_or_unresolved"
        wells.append(
            {
                "well_uid": entity.well_uid,
                "well_name": entity.canonical_name,
                "geometry_mode": geometry_mode,
                "geometry_source": geometry_source,
                "md_tvd_semantics": md_tvd_semantics,
                "trajectory_source": trajectory_source,
                "log_count": len(entity.logs),
                "station_count": len(well_points),
                "valid_xy_station_count": len(valid_xy),
                "covered_station_count": len(covered_points),
                "coverage_fraction": coverage_fraction,
                "nearest_distance_median": _percentile(distances, 50),
                "nearest_distance_p95": _percentile(distances, 95),
                "nearest_distance_max": max(distances) if distances else None,
                "horizontal_status": horizontal_status,
                "plan_view_usable": bool(covered_points),
                "fusion_ready": False,
                "training_eligible": False,
            }
        )

    if progress:
        progress(len(entities), len(entities), "井轨迹水平配准完成")
    usable_wells = sum(bool(item["plan_view_usable"]) for item in wells)
    if usable_wells == 0:
        raise ValueError(
            "horizontal registration found zero plan-view usable wells; "
            "refusing a completed result"
        )
    fully_covered_wells = sum(item["horizontal_status"] == "covered" for item in wells)
    if wells and fully_covered_wells == len(wells):
        business_status = "usable"
    elif usable_wells:
        business_status = "partially_usable"
    else:  # pragma: no cover - guarded by the zero-usable fail-closed check above
        raise AssertionError("unreachable horizontal registration business status")

    visualization_wells: list[dict[str, Any]] = []
    for well in wells:
        well_points = [
            point for point in points if point["well_uid"] == well["well_uid"]
        ]
        if len(well_points) > 500:
            keep = np.unique(
                np.linspace(0, len(well_points) - 1, 500, dtype=int)
            ).tolist()
            preview_points = [well_points[index] for index in keep]
        else:
            preview_points = well_points
        visualization_wells.append(
            {
                "well_uid": well["well_uid"],
                "well_name": well["well_name"],
                "geometry_mode": well["geometry_mode"],
                "geometry_source": well["geometry_source"],
                "md_tvd_semantics": well["md_tvd_semantics"],
                "horizontal_status": well["horizontal_status"],
                "coverage_fraction": well["coverage_fraction"],
                "path": [
                    {
                        "md_m": item["md_m"],
                        "tvd_m": item["tvd_m"],
                        "md_tvd_semantics": item["md_tvd_semantics"],
                        "geometry_mode": item["geometry_mode"],
                        "x_m": item["x_m"],
                        "y_m": item["y_m"],
                        "trace_x_m": item["trace_x_m"],
                        "trace_y_m": item["trace_y_m"],
                        "covered": item["covered_by_seismic_grid"],
                        "distance_m": item["nearest_trace_distance_m"],
                    }
                    for item in preview_points
                ],
            }
        )
    visualization = {
        "contract_version": HORIZONTAL_VISUALIZATION_CONTRACT_VERSION,
        "view": "plan_view_xy_only",
        "coordinate_reference": {
            "horizontal_crs_id": horizontal_crs_id,
            "horizontal_unit": "m",
            "horizontal_axis_order": horizontal_axis_order,
        },
        "vertical_display_policy": {
            "twt_inferred": False,
            "depth_to_time_transform": "not_performed",
            "seismic_vertical_coordinate": None,
            "well_vertical_coordinate": None,
            "reason": "no_time_depth_contract",
        },
        "seismic_grids": [
            {
                **footprint.summary,
                "footprint_xy_m": (
                    footprint.hull.round(6).tolist()
                    if footprint.hull is not None
                    else [
                        [footprint.bounds[0], footprint.bounds[2]],
                        [footprint.bounds[1], footprint.bounds[2]],
                        [footprint.bounds[1], footprint.bounds[3]],
                        [footprint.bounds[0], footprint.bounds[3]],
                    ]
                ),
            }
            for footprint in footprints
        ],
        "wells": visualization_wells,
    }
    geometry_modes = sorted({str(well["geometry_mode"]) for well in wells})
    operation = (
        "trajectory_station_to_nearest_segy_trace"
        if geometry_modes == ["trajectory"]
        else "well_geometry_point_to_nearest_segy_trace"
    )
    return {
        "contract_version": HORIZONTAL_REGISTRATION_CONTRACT_VERSION,
        "source_snapshot_id": source_snapshot_id,
        "source_snapshot_fingerprint": source_snapshot_fingerprint,
        "product_kind": "horizontal_registration",
        "coordinate_reference": {
            "verified": True,
            "horizontal_crs_id": horizontal_crs_id,
            "horizontal_unit": "m",
            "horizontal_axis_order": horizontal_axis_order,
        },
        "scientific_scope": {
            "operation": operation,
            "geometry_modes": geometry_modes,
            "head_only_md_tvd_policy": _HEAD_ONLY_MD_TVD_SEMANTICS,
            "time_depth_asset_count": 0,
            "time_depth_used": False,
            "depth_to_time_transform": "not_performed",
            "vertical_registration": "not_performed",
            "twt_generated": False,
            "registration_v3_generated": False,
            "can_build_multimodal_view": False,
            "fusion_ready": False,
            "training_eligible": False,
        },
        "matching_policy": {
            "method": "nearest_trace",
            "maximum_horizontal_distance_m": max_distance,
            "coverage_test": (
                "trace_grid_convex_footprint_with_one_cell_tolerance_and_"
                "nearest_trace_distance"
            ),
        },
        "summary": {
            "seismic_grid_count": len(footprints),
            "well_count": len(wells),
            "excluded_well_count": len(excluded_wells),
            "station_count": len(points),
            "covered_station_count": sum(
                bool(item["covered_by_seismic_grid"]) for item in points
            ),
            "fully_covered_well_count": fully_covered_wells,
            "plan_view_usable_well_count": usable_wells,
            "fusion_ready_well_count": 0,
            "training_eligible_well_count": 0,
            "business_status": business_status,
        },
        "seismic_grids": [item.summary for item in footprints],
        "wells": wells,
        "excluded_wells": excluded_wells,
        "points": points,
        "visualization": visualization,
    }


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row = {
                key: (
                    "true"
                    if value is True
                    else "false"
                    if value is False
                    else ""
                    if value is None
                    else value
                )
                for key, value in raw.items()
            }
            writer.writerow(row)


def write_horizontal_registration_product(
    output_directory: str | Path,
    result: dict[str, Any],
) -> HorizontalRegistrationProduct:
    """Write one immutable-by-hash plan-view product and its display payload."""

    if result.get("contract_version") != HORIZONTAL_REGISTRATION_CONTRACT_VERSION:
        raise ValueError("horizontal registration result contract is incompatible")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    points_path = output / "horizontal_registration_points.csv"
    wells_path = output / "horizontal_registration_wells.csv"
    visualization_path = output / "horizontal_registration_plan_view.json"
    manifest_path = output / "horizontal_registration_manifest.json"
    existing = [
        path
        for path in (points_path, wells_path, visualization_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "horizontal registration output already exists; use a new output "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    _write_csv(points_path, _POINT_FIELDS, result.get("points") or [])
    _write_csv(wells_path, _WELL_FIELDS, result.get("wells") or [])
    visualization_path.write_text(
        json.dumps(
            _json_safe(result.get("visualization") or {}), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    points_sha256 = file_sha256(points_path)
    wells_sha256 = file_sha256(wells_path)
    visualization_sha256 = file_sha256(visualization_path)
    manifest = {
        key: _json_safe(value)
        for key, value in result.items()
        if key not in {"points", "visualization"}
    }
    manifest["outputs"] = {
        "horizontal_registration_points": str(points_path),
        "horizontal_registration_wells": str(wells_path),
        "horizontal_registration_plan_view": str(visualization_path),
    }
    manifest["output_integrity"] = {
        "horizontal_registration_points_sha256": points_sha256,
        "horizontal_registration_wells_sha256": wells_sha256,
        "horizontal_registration_plan_view_sha256": visualization_sha256,
    }
    product_sha256 = canonical_sha256(
        {
            "contract_version": HORIZONTAL_REGISTRATION_CONTRACT_VERSION,
            "source_snapshot_id": manifest.get("source_snapshot_id"),
            "source_snapshot_fingerprint": manifest.get("source_snapshot_fingerprint"),
            **manifest["output_integrity"],
        }
    )
    manifest["horizontal_registration_product_sha256"] = product_sha256
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = file_sha256(manifest_path)
    return HorizontalRegistrationProduct(
        manifest=manifest,
        manifest_path=manifest_path,
        points_path=points_path,
        wells_path=wells_path,
        visualization_path=visualization_path,
        manifest_sha256=manifest_sha256,
        points_sha256=points_sha256,
        wells_sha256=wells_sha256,
        visualization_sha256=visualization_sha256,
        product_sha256=product_sha256,
    )


__all__ = [
    "HORIZONTAL_REGISTRATION_CONTRACT_VERSION",
    "HORIZONTAL_VISUALIZATION_CONTRACT_VERSION",
    "HorizontalRegistrationProduct",
    "build_horizontal_registration",
    "write_horizontal_registration_product",
]
