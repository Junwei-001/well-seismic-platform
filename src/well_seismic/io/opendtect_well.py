"""Strict reader for legacy OpenDtect ``.well`` and ASCII ``.track`` files.

OpenDtect's documented four-column track contract is absolute X, absolute Y,
TVDSS (positive down from MSL), and measured depth (MD from the well reference
datum).  These files do not contain a time-depth model.  This reader therefore
creates only a well head and a spatial MD/TVD trajectory and never fabricates
checkshot or TWT samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..coordinate_reference import (
    CoordinateReferenceError,
    canonical_crs_id,
    parse_crs,
    transform_xy,
)
from ..models import Trajectory, WellHead
from ..vertical_datum import length_to_metres
from .trajectory_canonicalization import canonicalize_station_rows


_SUPPORTED_SUFFIXES = frozenset({".well", ".track"})
_LENGTH_UNITS = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
}
_VERSION = re.compile(r"^dTect\s+V(?P<version>[0-9]+(?:\.[0-9]+)*)\s*$", re.I)
_SURFACE_COORDINATE = re.compile(
    r"^Surface\s+coordinate\s*:\s*\(\s*"
    r"(?P<x>[-+0-9.Ee]+)\s*,\s*(?P<y>[-+0-9.Ee]+)\s*\)\s*$",
    re.I,
)
_SURFACE_ELEVATION = re.compile(
    r"^Surface\s+elevation\s*:\s*(?P<value>[-+0-9.Ee]+)\s*$",
    re.I,
)
_UNIQUE_WELL_ID = re.compile(r"^Unique\s+Well\s+ID\s*:\s*(?P<value>.*?)\s*$", re.I)


@dataclass(frozen=True)
class OpenDtectWellTrack:
    """Canonical spatial products and their parse evidence."""

    head: WellHead
    trajectory: Trajectory
    evidence: tuple[str, ...]
    format_version: str


def _read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace").splitlines()


def _unit(value: Any) -> str | None:
    text = str(value or "").strip().casefold().replace(".", "")
    if text in {"", "unknown", "null", "none"}:
        return None
    return _LENGTH_UNITS.get(text)


def _declared_unit(
    options: dict[str, Any], *keys: str
) -> tuple[str | None, str | None]:
    declarations: list[tuple[str, str]] = []
    for key in keys:
        if key not in options:
            continue
        raw = options.get(key)
        unit = _unit(raw)
        if unit is None:
            if str(raw or "").strip().casefold() not in {"", "unknown", "null", "none"}:
                raise ValueError(f"OpenDtect {key} has unsupported unit {raw!r}")
            continue
        declarations.append((unit, f"options.{key}"))
    distinct = {unit for unit, _ in declarations}
    if len(distinct) > 1:
        evidence = ",".join(f"{source}={unit}" for unit, source in declarations)
        raise ValueError(f"OpenDtect length unit conflict: {evidence}")
    return declarations[0] if declarations else (None, None)


def _projected_crs_unit(crs_value: str) -> str:
    crs = parse_crs(crs_value, field="OpenDtect source CRS")
    if not crs.is_projected:
        raise CoordinateReferenceError(
            "OpenDtect X/Y track columns require a projected source CRS"
        )
    axes = tuple(crs.axis_info[:2])
    if len(axes) != 2 or any(axis.unit_conversion_factor is None for axis in axes):
        raise CoordinateReferenceError("OpenDtect source CRS lacks horizontal units")
    factors = tuple(float(axis.unit_conversion_factor) for axis in axes)
    if all(np.isclose(value, 1.0, rtol=0.0, atol=1e-12) for value in factors):
        return "m"
    if all(
        np.isclose(value, 0.3048, rtol=0.0, atol=1e-8)
        or np.isclose(value, 1200.0 / 3937.0, rtol=0.0, atol=1e-8)
        for value in factors
    ):
        return "ft"
    raise CoordinateReferenceError(
        "OpenDtect source CRS horizontal unit is neither metres nor feet"
    )


def _resolve_units(
    options: dict[str, Any],
) -> tuple[str, str, str, str, str | None, str | None]:
    source_crs = str(options.get("source_crs") or options.get("crs") or "").strip()
    target_crs = str(options.get("target_crs") or "").strip()
    if target_crs and not source_crs:
        raise CoordinateReferenceError(
            "OpenDtect target_crs cannot be applied without a declared source_crs"
        )
    coordinate_unit, coordinate_source = _declared_unit(
        options,
        "opendtect_coordinate_unit",
        "coordinate_unit",
        "horizontal_unit",
    )
    crs_unit = _projected_crs_unit(source_crs) if source_crs else None
    if coordinate_unit and crs_unit and coordinate_unit != crs_unit:
        raise CoordinateReferenceError(
            "OpenDtect coordinate unit conflicts with source CRS axis unit: "
            f"{coordinate_unit}!={crs_unit}"
        )
    if coordinate_unit is None:
        coordinate_unit = crs_unit
        coordinate_source = "source_crs_axis_unit" if crs_unit else None
    if coordinate_unit is None:
        raise ValueError(
            "OpenDtect X/Y unit is unknown; declare coordinate_unit or source_crs"
        )
    if source_crs and crs_unit != "m" and not target_crs:
        raise CoordinateReferenceError(
            "OpenDtect non-metre source CRS requires an explicit projected-metre target_crs"
        )

    depth_unit, depth_source = _declared_unit(
        options,
        "opendtect_depth_unit",
        "trajectory_length_unit",
        "depth_unit",
        "length_unit",
    )
    if depth_unit is None:
        # Legacy OpenDtect track files do not carry a unit token.  Reusing the
        # explicitly declared survey coordinate unit is deterministic for this
        # four-column format, and the provenance records that shared contract.
        depth_unit = coordinate_unit
        depth_source = f"{coordinate_source}:opendtect_shared_survey_unit"
    return (
        coordinate_unit,
        depth_unit,
        str(coordinate_source),
        str(depth_source),
        source_crs or None,
        target_crs or None,
    )


def _numeric_rows(lines: list[str], *, start: int, path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, raw in enumerate(lines[start:], start=start + 1):
        line = raw.strip()
        if not line or line == "!":
            continue
        tokens = re.split(r"[\s,]+", line)
        if len(tokens) != 4:
            raise ValueError(
                f"OpenDtect track row {line_number} must have X/Y/TVDSS/MD: {path}"
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"OpenDtect track row {line_number} is not numeric: {path}"
            ) from exc
        rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"OpenDtect track needs at least two stations: {path}")
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape[1] != 4 or not np.isfinite(matrix).all():
        raise ValueError(f"OpenDtect track contains invalid station values: {path}")
    return matrix


def _parse_document(path: Path) -> tuple[np.ndarray, dict[str, Any], str, list[str]]:
    lines = _read_lines(path)
    nonempty = [
        (index, line.strip()) for index, line in enumerate(lines) if line.strip()
    ]
    if not nonempty:
        raise ValueError(f"OpenDtect track is empty: {path}")
    metadata: dict[str, Any] = {}
    evidence: list[str] = []
    if path.suffix.casefold() == ".well":
        version_match = _VERSION.match(nonempty[0][1])
        if (
            version_match is None
            or len(nonempty) < 2
            or nonempty[1][1].casefold() != "well"
        ):
            raise ValueError(
                f".well file is missing the OpenDtect Well signature: {path}"
            )
        separators = [index for index, line in nonempty if line == "!"]
        if not separators:
            raise ValueError(f"OpenDtect .well header has no data separator: {path}")
        data_start = separators[-1] + 1
        header_lines = [line for index, line in nonempty if index < data_start]
        for line in header_lines:
            coordinate = _SURFACE_COORDINATE.match(line)
            if coordinate:
                metadata["surface_coordinate"] = (
                    float(coordinate.group("x")),
                    float(coordinate.group("y")),
                )
            elevation = _SURFACE_ELEVATION.match(line)
            if elevation:
                metadata["surface_elevation"] = float(elevation.group("value"))
            unique_id = _UNIQUE_WELL_ID.match(line)
            if unique_id and unique_id.group("value").strip():
                metadata["unique_well_id"] = unique_id.group("value").strip()
        version = version_match.group("version")
        evidence.append(f"opendtect_signature:dTect_V{version}")
    else:
        data_start = 0
        version = "ascii-track"
        # A .track file has no embedded signature, so every non-empty line must
        # satisfy the documented four-column numeric contract.
        evidence.append("opendtect_ascii_track_suffix_and_four_column_contract")
    matrix = _numeric_rows(lines, start=data_start, path=path)
    return matrix, metadata, version, evidence


def _reference_elevation(
    md_m: np.ndarray,
    tvdss_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    options: dict[str, Any],
) -> tuple[float, str]:
    explicit = options.get("well_reference_elevation_m")
    zero_indices = np.flatnonzero(np.isclose(md_m, 0.0, rtol=0.0, atol=1e-6))
    inferred_zero = -float(tvdss_m[zero_indices[0]]) if zero_indices.size else None
    if explicit is not None:
        reference = float(explicit)
        if not np.isfinite(reference):
            raise ValueError("well_reference_elevation_m must be finite")
        if inferred_zero is not None and not np.isclose(
            reference, inferred_zero, rtol=0.0, atol=0.05
        ):
            raise ValueError(
                "OpenDtect MD=0 TVDSS conflicts with well_reference_elevation_m"
            )
        return reference, "options.well_reference_elevation_m"
    if inferred_zero is not None:
        return inferred_zero, "opendtect_md_zero_tvdss"

    xy_span = float(np.hypot(np.ptp(x_m), np.ptp(y_m)))
    reference_samples = md_m - tvdss_m
    reference_spread = float(np.ptp(reference_samples))
    if xy_span <= 0.05 and reference_spread <= 0.05:
        return (
            float(np.median(reference_samples)),
            "opendtect_vertical_track_constant_md_minus_tvdss",
        )
    raise ValueError(
        "OpenDtect track has no MD=0 station and no declared reference elevation; "
        "only a provably vertical constant MD-TVDSS track can derive KB"
    )


def _validate_geometry(
    md_m: np.ndarray,
    tvd_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> None:
    if np.any(md_m < -1e-6) or np.any(np.diff(md_m) <= 0.0):
        raise ValueError("OpenDtect MD must be non-negative and strictly increasing")
    if np.any(tvd_m < -0.05):
        raise ValueError("OpenDtect TVD derived from TVDSS cannot be negative above KB")
    segment = np.sqrt(np.diff(x_m) ** 2 + np.diff(y_m) ** 2 + np.diff(tvd_m) ** 2)
    md_step = np.diff(md_m)
    tolerance = np.maximum(0.1, 0.01 * md_step)
    if np.any(segment > md_step + tolerance):
        raise ValueError("OpenDtect XYZ segment length exceeds its MD increment")


def read_opendtect_well_track(
    path: str | Path,
    options: dict[str, Any] | None = None,
) -> OpenDtectWellTrack:
    """Read one OpenDtect track without consulting time-depth assets."""

    source = Path(path)
    if source.suffix.casefold() not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported OpenDtect well suffix: {source.suffix}")
    raw, metadata, version, evidence = _parse_document(source)
    raw_x, raw_y, raw_tvdss, raw_md = raw.T
    raw_options = dict(options or {})
    (
        coordinate_unit,
        depth_unit,
        coordinate_unit_source,
        depth_unit_source,
        source_crs,
        target_crs,
    ) = _resolve_units(raw_options)
    md_m = np.asarray(length_to_metres(raw_md, depth_unit), dtype=float)
    tvdss_m = np.asarray(length_to_metres(raw_tvdss, depth_unit), dtype=float)

    coordinate_transform: dict[str, Any] = {}
    if source_crs and target_crs:
        transformed = transform_xy(
            raw_x,
            raw_y,
            source_crs=source_crs,
            target_crs=target_crs,
        )
        x_m, y_m = transformed.x, transformed.y
        resolved_crs = transformed.target_crs
        source_crs_id = transformed.source_crs
        coordinate_transform = transformed.provenance()
    else:
        x_m = np.asarray(length_to_metres(raw_x, coordinate_unit), dtype=float)
        y_m = np.asarray(length_to_metres(raw_y, coordinate_unit), dtype=float)
        source_crs_id = (
            canonical_crs_id(source_crs, field="OpenDtect source CRS")
            if source_crs
            else None
        )
        resolved_crs = source_crs_id
        if source_crs_id:
            coordinate_transform = {
                "source_crs": source_crs_id,
                "target_crs": source_crs_id,
                "transformed": False,
                "operation": "identity_then_canonical_unit_conversion",
                "axis_contract": "always_xy",
            }

    canonical = canonicalize_station_rows(
        md_m,
        {"tvdss": tvdss_m, "x": x_m, "y": y_m},
    )
    if canonical.conflicting_duplicate_md:
        raise ValueError(
            "OpenDtect duplicate MD has conflicting TVDSS/XY station geometry"
        )
    station_order = canonical.source_indices
    if station_order.size < 2:
        raise ValueError(
            "OpenDtect track needs at least two finite, unique MD stations"
        )
    raw_x = raw_x[station_order]
    raw_y = raw_y[station_order]
    raw_tvdss = raw_tvdss[station_order]
    raw_md = raw_md[station_order]
    md_m = md_m[station_order]
    tvdss_m = tvdss_m[station_order]
    x_m = x_m[station_order]
    y_m = y_m[station_order]
    if canonical.receipt is not None:
        if canonical.receipt["source_order_changed"]:
            evidence.append("station_order_canonicalized_by_md")
        dropped = canonical.receipt["dropped_nonfinite_md_source_rows_1_based"]
        if dropped:
            evidence.append(
                "nonfinite_md_stations_quarantined:source_rows="
                + ",".join(str(value) for value in dropped)
            )
        collapsed = canonical.receipt[
            "collapsed_congruent_duplicate_md_groups"
        ]
        if collapsed:
            evidence.append(
                f"congruent_duplicate_md_stations_collapsed:groups={len(collapsed)}"
            )

    declared_surface = metadata.get("surface_coordinate")
    if declared_surface is not None:
        surface_x_raw, surface_y_raw = declared_surface
        if source_crs and target_crs:
            surface = transform_xy(
                [surface_x_raw],
                [surface_y_raw],
                source_crs=source_crs,
                target_crs=target_crs,
            )
            surface_x_m, surface_y_m = float(surface.x[0]), float(surface.y[0])
        else:
            surface_x_m = float(length_to_metres(surface_x_raw, coordinate_unit))
            surface_y_m = float(length_to_metres(surface_y_raw, coordinate_unit))
        if np.hypot(surface_x_m - x_m[0], surface_y_m - y_m[0]) > 0.25:
            raise ValueError(
                "OpenDtect Surface coordinate conflicts with the first track station"
            )
        evidence.append("surface_coordinate_matches_first_station")
    else:
        surface_x_raw, surface_y_raw = float(raw_x[0]), float(raw_y[0])
        surface_x_m, surface_y_m = float(x_m[0]), float(y_m[0])

    reference_m, reference_source = _reference_elevation(
        md_m,
        tvdss_m,
        x_m,
        y_m,
        raw_options,
    )
    tvd_m = tvdss_m + reference_m
    _validate_geometry(md_m, tvd_m, x_m, y_m)
    well_name = str(raw_options.get("well_name") or source.stem).strip()
    if not well_name:
        raise ValueError("OpenDtect well name is empty")

    conversions: list[str] = []
    if coordinate_unit != "m":
        conversions.append(f"x/y:{coordinate_unit}->m")
    if depth_unit != "m":
        conversions.append(f"md/tvdss:{depth_unit}->m")
    confidence = 0.99 if source.suffix.casefold() == ".well" else 0.95
    vertical_semantics = {
        "source_depth": "TVDSS",
        "source_depth_axis": "positive_down",
        "source_datum": "MSL",
        "canonical_depth": "TVD_m_positive_down_from_MD0",
        "well_reference_elevation_m": reference_m,
        "well_reference_source": reference_source,
        "registration_eligible": True,
        "datum_status": "resolved_from_file_geometry",
    }
    if canonical.receipt is not None:
        vertical_semantics["station_normalization"] = canonical.receipt
    evidence.extend(
        [
            "columns:X,Y,TVDSS_positive_down_from_MSL,MD_from_KB",
            f"well_name_source:{'options.well_name' if raw_options.get('well_name') else 'filename_stem'}",
            f"coordinate_unit_source:{coordinate_unit_source}",
            f"depth_unit_source:{depth_unit_source}",
            f"well_reference_source:{reference_source}",
            "time_depth_model:not_read_not_created",
        ]
    )
    if metadata.get("unique_well_id"):
        evidence.append("unique_well_id_present_not_reinterpreted_as_well_name")

    trajectory = Trajectory(
        well_name=well_name,
        md=md_m,
        tvd=tvd_m,
        x_offset=x_m - float(x_m[0]),
        y_offset=y_m - float(y_m[0]),
        x=x_m,
        y=y_m,
        source=str(source),
        confidence=confidence,
        source_units={
            "md": depth_unit,
            "tvdss": depth_unit,
            "x": coordinate_unit,
            "y": coordinate_unit,
            "x_offset": coordinate_unit,
            "y_offset": coordinate_unit,
        },
        unit_conversions=conversions,
        unit_provenance={
            "md": depth_unit_source,
            "tvdss": depth_unit_source,
            "tvd": f"computed:{reference_source}",
            "x": coordinate_unit_source,
            "y": coordinate_unit_source,
            "x_offset": "computed:absolute_x_minus_first_x",
            "y_offset": "computed:absolute_y_minus_first_y",
        },
        vertical_semantics=vertical_semantics,
        source_crs=source_crs_id,
        horizontal_crs=resolved_crs,
        coordinate_transform=coordinate_transform,
    )
    ground_elevation = metadata.get("surface_elevation")
    ground_elevation_m = (
        None
        if ground_elevation is None
        else float(length_to_metres(float(ground_elevation), depth_unit))
    )
    head = WellHead(
        well_name=well_name,
        x=surface_x_m,
        y=surface_y_m,
        kb=reference_m,
        ground_elevation=ground_elevation_m,
        total_depth_md=float(md_m[-1]),
        crs=resolved_crs,
        source=str(source),
        confidence=confidence,
        vertical_datum_unit="m",
        horizontal_unit="m",
        source_x=float(surface_x_raw),
        source_y=float(surface_y_raw),
        source_crs=source_crs_id,
        coordinate_transform=coordinate_transform,
    )
    return OpenDtectWellTrack(
        head=head,
        trajectory=trajectory,
        evidence=tuple(evidence),
        format_version=version,
    )


__all__ = ["OpenDtectWellTrack", "read_opendtect_well_track"]
