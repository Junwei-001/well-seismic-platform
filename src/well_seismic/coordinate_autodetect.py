"""Low-interaction CRS evidence detection and cross-asset quality checks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .content_identity import canonical_sha256
from .coordinate_reference import (
    CoordinateReferenceError,
    canonical_crs_id,
    require_projected_metre_crs,
)

_EPSG_PATTERN = re.compile(r"\bEPSG\s*[,=: ]\s*(\d{4,6})\b", re.IGNORECASE)


def _files(paths: Iterable[str], *, recursive: bool, suffixes: set[str]) -> Iterable[Path]:
    visited = 0
    for raw in paths:
        path = Path(str(raw)).expanduser()
        if path.is_file():
            candidates = (path,)
        elif path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
        else:
            continue
        for candidate in candidates:
            if visited >= 20_000:
                return
            visited += 1
            if candidate.is_file() and candidate.suffix.casefold() in suffixes:
                yield candidate


def _epsg_from_bytes(raw: bytes) -> str | None:
    for encoding in ("ascii", "utf-8", "gb18030", "cp500", "cp037", "latin1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        match = _EPSG_PATTERN.search(text)
        if match:
            return f"EPSG:{match.group(1)}"
    return None


def detect_crs_evidence(
    *,
    well_paths: Iterable[str],
    seismic_paths: Iterable[str],
    recursive: bool,
) -> dict[str, Any]:
    evidence: list[dict[str, str]] = []
    for path in _files(well_paths, recursive=recursive, suffixes={".dev", ".prj"}):
        try:
            raw = path.read_bytes()[:128_000]
        except OSError:
            continue
        candidate = _epsg_from_bytes(raw)
        if candidate is None and path.suffix.casefold() == ".prj":
            try:
                candidate = canonical_crs_id(raw.decode("utf-8-sig"), field="PRJ CRS")
            except (UnicodeDecodeError, CoordinateReferenceError):
                candidate = None
        if candidate:
            evidence.append(
                {"kind": "well", "crs": canonical_crs_id(candidate), "source": str(path)}
            )
    for path in _files(seismic_paths, recursive=recursive, suffixes={".sgy", ".segy"}):
        try:
            with path.open("rb") as handle:
                raw = handle.read(3_200)
        except OSError:
            continue
        candidate = _epsg_from_bytes(raw)
        if candidate:
            evidence.append(
                {"kind": "seismic", "crs": canonical_crs_id(candidate), "source": str(path)}
            )
    well_candidates = sorted({item["crs"] for item in evidence if item["kind"] == "well"})
    seismic_candidates = sorted(
        {item["crs"] for item in evidence if item["kind"] == "seismic"}
    )
    return {
        "evidence": evidence,
        "well_candidates": well_candidates,
        "seismic_candidates": seismic_candidates,
        "conflict": len(well_candidates) > 1 or len(seismic_candidates) > 1,
    }


def choose_target_crs(
    detection: dict[str, Any],
    *,
    explicit_target: str | None,
    explicit_seismic_source: str | None,
) -> tuple[str | None, str]:
    if explicit_target and str(explicit_target).strip():
        try:
            crs = require_projected_metre_crs(explicit_target, field="目标CRS")
        except CoordinateReferenceError:
            # Competition inputs often name only a geodetic family (for
            # example WGS84 or Beijing 1954) without a projected zone.  Keep
            # that declaration as evidence, but do not invent an EPSG zone.
            # The post-ingest native-grid gate below may still bind both
            # modalities to one local numerical survey grid.
            return None, "explicit_target_requires_projected_zone_or_native_grid"
        return canonical_crs_id(crs), "explicit_target"
    if explicit_seismic_source and str(explicit_seismic_source).strip():
        try:
            crs = require_projected_metre_crs(
                explicit_seismic_source, field="SEG-Y源CRS"
            )
        except CoordinateReferenceError:
            return None, "explicit_source_requires_projected_target"
        return canonical_crs_id(crs), "explicit_seismic_source"
    seismic = list(detection.get("seismic_candidates") or ())
    if len(seismic) == 1:
        try:
            crs = require_projected_metre_crs(seismic[0], field="SEG-Y检测CRS")
        except CoordinateReferenceError:
            return None, "seismic_candidate_requires_projected_target"
        return canonical_crs_id(crs), "seismic_file_evidence"
    well = list(detection.get("well_candidates") or ())
    if len(well) == 1:
        try:
            crs = require_projected_metre_crs(well[0], field="井文件检测CRS")
        except CoordinateReferenceError:
            return None, "well_candidate_requires_projected_target"
        return canonical_crs_id(crs), "well_file_candidate"
    return None, "unresolved"


def _native_metre_grid_contract(pipeline: Any) -> dict[str, Any]:
    """Bind co-located raw XY arrays without pretending to know an EPSG zone.

    This is deliberately narrower than CRS inference.  It requires either
    canonical metre-valued well heads or retained raw well X/Y with one unique
    projected-looking SEG-Y bbox match.  It never invents a projection zone or
    transforms coordinates; a geographic, ambiguous or cross-grid input stays
    blocked.
    """

    geometries = [
        reader.geometry
        for _, reader in pipeline.seismic
        if reader.geometry is not None
        and reader.geometry.x is not None
        and reader.geometry.y is not None
    ]
    canonical_heads: list[tuple[str, Any, float, float]] = []
    raw_heads: list[tuple[str, Any, float, float]] = []
    for entity_key, entity in pipeline.registry.entities.items():
        head = entity.preferred_head
        if head is None:
            continue
        # Registry keys are the normalized, replay-stable well identity.  The
        # preferred physical source may legitimately change after an inferred
        # unit is sealed (for example from a standalone well-location table to
        # an otherwise identical, higher-confidence LAS header).  Source paths
        # therefore belong to provenance, not to the local-grid namespace.
        well_identity = str(entity_key)
        if (
            head.x is not None
            and head.y is not None
            and np.isfinite(float(head.x))
            and np.isfinite(float(head.y))
            and str(getattr(head, "horizontal_unit", "unknown")).casefold()
            == "m"
        ):
            canonical_heads.append(
                (well_identity, head, float(head.x), float(head.y))
            )
            continue
        # The tabular reader intentionally retains unknown-unit raw X/Y while
        # withholding canonical coordinates.  They may be promoted to metres
        # only after this cross-asset gate proves one unique native-grid
        # interpretation; no CRS zone or coordinate transform is inferred.
        source_x = getattr(head, "source_x", None)
        source_y = getattr(head, "source_y", None)
        if (
            source_x is not None
            and source_y is not None
            and np.isfinite(float(source_x))
            and np.isfinite(float(source_y))
        ):
            raw_heads.append(
                (well_identity, head, float(source_x), float(source_y))
            )
    canonical_heads.sort(key=lambda item: (item[0], item[2], item[3]))
    raw_heads.sort(key=lambda item: (item[0], item[2], item[3]))
    head_records = sorted(
        [*canonical_heads, *raw_heads],
        key=lambda item: (item[0], item[2], item[3]),
    )
    if not geometries or not head_records:
        return {
            "verified": False,
            "reason": "native_grid_qc_requires_wells_and_seismic",
            "target_crs": None,
            "well_count": len(head_records),
            "seismic_count": len(geometries),
        }
    geographic_raw_heads = [
        str(head.source)
        for _, head, x, y in raw_heads
        if abs(x) <= 180.0 and abs(y) <= 90.0
    ]
    if geographic_raw_heads:
        return {
            "verified": False,
            "reason": "native_grid_raw_well_xy_looks_geographic",
            "target_crs": None,
            "sources": geographic_raw_heads,
        }

    geometry_arrays = [
        (
            geometry,
            np.asarray(geometry.x, dtype=float).reshape(-1),
            np.asarray(geometry.y, dtype=float).reshape(-1),
        )
        for geometry in geometries
    ]
    if any(x.size != y.size for _, x, y in geometry_arrays):
        return {
            "verified": False,
            "reason": "native_grid_seismic_xy_shape_mismatch",
            "target_crs": None,
        }
    seismic_x_parts = [x[np.isfinite(x)] for _, x, _ in geometry_arrays]
    seismic_y_parts = [y[np.isfinite(y)] for _, _, y in geometry_arrays]
    if any(values.size == 0 for values in (*seismic_x_parts, *seismic_y_parts)):
        return {
            "verified": False,
            "reason": "native_grid_seismic_xy_empty",
            "target_crs": None,
        }
    seismic_x = np.concatenate(seismic_x_parts)
    seismic_y = np.concatenate(seismic_y_parts)
    xmin, xmax = float(np.min(seismic_x)), float(np.max(seismic_x))
    ymin, ymax = float(np.min(seismic_y)), float(np.max(seismic_y))
    span = max(xmax - xmin, ymax - ymin)
    # Longitude/latitude degrees are not a metre grid.  Extremely small or
    # planetary-scale spans are likewise outside the safe local-grid envelope.
    geographic_like = (
        np.all(np.abs(seismic_x) <= 180.0)
        and np.all(np.abs(seismic_y) <= 90.0)
    )
    if geographic_like or not 100.0 <= span <= 10_000_000.0:
        return {
            "verified": False,
            "reason": "native_grid_not_projected_metre_scale",
            "target_crs": None,
            "survey_bbox": [xmin, ymin, xmax, ymax],
        }
    margin = max(1_000.0, 0.1 * span)

    def _in_area(x: float, y: float) -> bool:
        return (
            xmin - margin <= x <= xmax + margin
            and ymin - margin <= y <= ymax + margin
        )

    xy_coordinates = [(x, y) for _, _, x, y in head_records]
    xy_matches = [_in_area(x, y) for x, y in xy_coordinates]
    required_matches = max(1, (len(head_records) + 1) // 2)
    swapped_coordinates = [
        (x, y) for _, _, x, y in canonical_heads
    ] + [
        (y, x) for _, _, x, y in raw_heads
    ]
    swapped_match_count = sum(_in_area(x, y) for x, y in swapped_coordinates)
    xy_match_count = sum(xy_matches)
    if xy_match_count < required_matches:
        return {
            "verified": False,
            "reason": "native_grid_well_seismic_bounding_boxes_do_not_overlap",
            "target_crs": None,
            "survey_bbox": [xmin, ymin, xmax, ymax],
            "well_xy": [[x, y] for x, y in xy_coordinates],
            "required_overlap_count": required_matches,
            "overlap_count": xy_match_count,
        }
    if raw_heads and swapped_match_count >= xy_match_count:
        return {
            "verified": False,
            "reason": "native_grid_raw_well_xy_axis_interpretation_not_unique",
            "target_crs": None,
            "survey_bbox": [xmin, ymin, xmax, ymax],
            "xy_overlap_count": xy_match_count,
            "yx_overlap_count": swapped_match_count,
        }

    geometry_semantics: list[dict[str, Any]] = []
    geometry_provenance: list[dict[str, str]] = []
    for geometry, x_values, y_values in geometry_arrays:
        finite_pairs = np.isfinite(x_values) & np.isfinite(y_values)
        digest = hashlib.sha256()
        digest.update(b"well-seismic.local-survey-xy-geometry.v1\0")
        digest.update(
            np.asarray([x_values.size], dtype="<u8").tobytes(order="C")
        )
        digest.update(finite_pairs.astype(np.uint8).tobytes(order="C"))
        digest.update(
            np.asarray(x_values[finite_pairs], dtype="<f8").tobytes(order="C")
        )
        digest.update(
            np.asarray(y_values[finite_pairs], dtype="<f8").tobytes(order="C")
        )
        if np.any(finite_pairs):
            geometry_bbox = [
                float(np.min(x_values[finite_pairs])),
                float(np.min(y_values[finite_pairs])),
                float(np.max(x_values[finite_pairs])),
                float(np.max(y_values[finite_pairs])),
            ]
        else:  # guarded by the non-empty checks above
            geometry_bbox = []
        semantic_record = {
            "profile": str(getattr(geometry, "profile", "unknown")),
            "trace_count": int(
                getattr(geometry, "trace_count", x_values.size)
            ),
            "coordinate_count": int(x_values.size),
            "finite_pair_count": int(np.sum(finite_pairs)),
            "coordinate_sha256": digest.hexdigest(),
            "bbox": geometry_bbox,
        }
        semantic_sha256 = canonical_sha256(semantic_record)
        geometry_semantics.append(semantic_record)
        geometry_provenance.append(
            {
                "geometry_semantics_sha256": semantic_sha256,
                "source": str(geometry.path),
            }
        )
    geometry_semantics.sort(key=canonical_sha256)
    geometry_provenance.sort(
        key=lambda item: (item["geometry_semantics_sha256"], item["source"])
    )
    well_semantics = [
        {
            "well_identity": well_identity,
            "x": x,
            "y": y,
            "horizontal_unit": "m",
        }
        for well_identity, _, x, y in head_records
    ]
    raw_head_ids = {id(head) for _, head, _, _ in raw_heads}
    well_provenance = [
        {
            "well_identity": well_identity,
            "well_name": str(getattr(head, "well_name", well_identity)),
            "selected_source": str(head.source),
            "coordinate_state": (
                "raw_unknown_unit" if id(head) in raw_head_ids
                else "canonical_m"
            ),
        }
        for well_identity, head, x, y in head_records
    ]
    evidence = {
        "contract_version": "well-seismic.local-survey-xy-evidence.v2",
        "survey_bbox": [xmin, ymin, xmax, ymax],
        "seismic_geometries": geometry_semantics,
        "well_xy": well_semantics,
        "wells_in_or_near_survey": int(xy_match_count),
        "required_overlap_count": required_matches,
        "policy": "same_raw_numeric_grid_no_coordinate_transform",
    }
    evidence_sha256 = canonical_sha256(evidence)
    local_crs = f"LOCAL_SURVEY_XY_{evidence_sha256[:12].upper()}"
    return {
        "verified": True,
        "reason": "native_metre_grid_and_cross_asset_bbox_qc",
        "target_crs": local_crs,
        "horizontal_unit": "m",
        "axis_order": "XY",
        "well_count": len(head_records),
        "seismic_count": len(geometries),
        "wells_in_or_near_survey": int(xy_match_count),
        "survey_bbox": [xmin, ymin, xmax, ymax],
        "coordinate_transform_applied": False,
        "derived_well_coordinate_source_unit": (
            "m" if raw_heads else None
        ),
        "unit_derivation_receipt": (
            {
                "rule": "raw_numeric_unique_native_grid_inference",
                "raw_well_count": len(raw_heads),
                "accepted_axis": "XY",
                "xy_overlap_count": xy_match_count,
                "yx_overlap_count": swapped_match_count,
                "no_projection_zone_inferred": True,
            }
            if raw_heads
            else None
        ),
        "evidence": evidence,
        "evidence_sha256": evidence_sha256,
        "provenance": {
            "contract_version": "well-seismic.local-survey-xy-provenance.v1",
            "seismic_geometry_sources": geometry_provenance,
            "well_coordinate_sources": well_provenance,
        },
    }


def native_grid_replay_is_consistent(
    first_pass: dict[str, Any],
    replay: dict[str, Any],
) -> bool:
    """Require exact scientific semantics across raw-unit normalization.

    Parser source selection is allowed to change and is recorded separately in
    ``provenance``.  Well identities, metre X/Y values and full SEG-Y geometry
    semantics must remain byte-for-byte equivalent, otherwise replay remains
    fail-closed.
    """

    for result in (first_pass, replay):
        evidence = result.get("evidence")
        if (
            result.get("verified") is not True
            or result.get("reason") != "native_metre_grid_and_cross_asset_bbox_qc"
            or result.get("horizontal_unit") != "m"
            or result.get("axis_order") != "XY"
            or result.get("coordinate_transform_applied") is not False
            or not isinstance(evidence, dict)
            or evidence.get("contract_version")
            != "well-seismic.local-survey-xy-evidence.v2"
            or canonical_sha256(evidence) != result.get("evidence_sha256")
        ):
            return False
    return bool(
        first_pass.get("derived_well_coordinate_source_unit") == "m"
        and replay.get("derived_well_coordinate_source_unit") is None
        and first_pass.get("target_crs") == replay.get("target_crs")
        and first_pass.get("evidence_sha256") == replay.get("evidence_sha256")
        and first_pass.get("evidence") == replay.get("evidence")
    )


def verify_pipeline_coordinate_contract(
    pipeline: Any,
    *,
    target_crs: str | None,
) -> dict[str, Any]:
    if not target_crs:
        return _native_metre_grid_contract(pipeline)
    try:
        target = canonical_crs_id(
            require_projected_metre_crs(target_crs, field="目标CRS")
        )
    except CoordinateReferenceError:
        # A datum-family label without projection parameters is evidence, not
        # an executable transform.  The only safe automatic alternative is a
        # verified same-grid local namespace.
        result = _native_metre_grid_contract(pipeline)
        result["declared_non_projected_reference"] = str(target_crs)
        return result
    geometries = [
        reader.geometry
        for _, reader in pipeline.seismic
        if reader.geometry is not None
        and reader.geometry.x is not None
        and reader.geometry.y is not None
    ]
    heads = [
        entity.preferred_head
        for entity in pipeline.registry.entities.values()
        if entity.preferred_head is not None
        and entity.preferred_head.x is not None
        and entity.preferred_head.y is not None
    ]
    mismatches: list[str] = []
    for geometry in geometries:
        if geometry.horizontal_crs != target:
            mismatches.append(f"seismic:{geometry.path}:{geometry.horizontal_crs or 'unknown'}")
    for head in heads:
        if head.crs != target:
            mismatches.append(f"well:{head.source}:{head.crs or 'unknown'}")
    if mismatches:
        return {
            "verified": False,
            "reason": "canonical_crs_mismatch",
            "target_crs": target,
            "mismatches": mismatches,
        }
    if not geometries or not heads:
        return {
            "verified": False,
            "reason": "cross_asset_qc_requires_wells_and_seismic",
            "target_crs": target,
            "well_count": len(heads),
            "seismic_count": len(geometries),
        }
    seismic_x = np.concatenate(
        [np.asarray(geometry.x, dtype=float)[np.isfinite(geometry.x)] for geometry in geometries]
    )
    seismic_y = np.concatenate(
        [np.asarray(geometry.y, dtype=float)[np.isfinite(geometry.y)] for geometry in geometries]
    )
    if seismic_x.size == 0 or seismic_y.size == 0:
        return {
            "verified": False,
            "reason": "seismic_xy_empty",
            "target_crs": target,
        }
    xmin, xmax = float(np.min(seismic_x)), float(np.max(seismic_x))
    ymin, ymax = float(np.min(seismic_y)), float(np.max(seismic_y))
    margin = max(1_000.0, 0.1 * max(xmax - xmin, ymax - ymin, 1.0))
    in_area = [
        xmin - margin <= float(head.x) <= xmax + margin
        and ymin - margin <= float(head.y) <= ymax + margin
        for head in heads
    ]
    if not any(in_area):
        return {
            "verified": False,
            "reason": "well_seismic_bounding_boxes_do_not_overlap",
            "target_crs": target,
            "survey_bbox": [xmin, ymin, xmax, ymax],
            "well_xy": [[float(head.x), float(head.y)] for head in heads],
        }
    return {
        "verified": True,
        "reason": "explicit_crs_transform_receipts_and_cross_asset_bbox_qc",
        "target_crs": target,
        "horizontal_unit": "m",
        "axis_order": "XY",
        "well_count": len(heads),
        "seismic_count": len(geometries),
        "wells_in_or_near_survey": int(sum(in_area)),
        "survey_bbox": [xmin, ymin, xmax, ymax],
    }
