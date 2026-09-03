"""Sealed, label-free SEG-Y trace support for frozen P13 inference.

The platform owns coordinate interpretation.  A downstream runtime must not
rebuild a spatial index from raw SEG-Y coordinates after the platform has
already transformed those coordinates into the verified survey CRS.  This
module therefore records the exact zero-based source-trace indices consumed by
P13, together with the immutable SourceSnapshot and canonical well inputs that
selected them.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import build_spatial_aligner
from .content_identity import (
    canonical_sha256,
    file_sha256,
    seismic_geometry_identity,
)

P13_TRACE_SUPPORT_SCHEMA = "wellfuse.p13_trace_support_v1"
P13_DEPTH_SAMPLE_COUNT = 512
P13_NEIGHBOR_TRACE_COUNT = 9
_LOCAL_SURVEY_CRS_PREFIX = "LOCAL_SURVEY_XY_"


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return canonical_sha256(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    )


def _sealed_endian(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {">", "big", "be", "big_endian", "big-endian"}:
        return "big"
    if normalized in {"<", "little", "le", "little_endian", "little-endian"}:
        return "little"
    raise ValueError("P13 trace support requires an explicit big/little SEG-Y endian")


def _sealed_time_axis(geometry: Any) -> np.ndarray:
    time_axis = np.ascontiguousarray(
        np.asarray(getattr(geometry, "time_axis", None), dtype="<f8")
    )
    expected = (int(getattr(geometry, "samples_per_trace", 0)),)
    if (
        time_axis.shape != expected
        or time_axis.size < 2
        or not np.isfinite(time_axis).all()
        or bool((np.diff(time_axis) <= 0.0).any())
    ):
        raise ValueError(
            "P13 trace support requires a finite, strictly increasing SEG-Y time axis"
        )
    return time_axis


def _resolved_coordinate_contract(
    geometry: Any,
    source_snapshot_context: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any], str]:
    """Resolve explicit CRS identities or a sealed local-grid identity default.

    ``LOCAL_SURVEY_XY_*`` is an identity namespace, not an executable pyproj
    transform. Readers therefore legitimately leave their CRS fields empty.
    When the sealed snapshot has already verified that wells and SEG-Y share the
    same metre/XY grid, use that immutable namespace for both sides instead of
    blocking P13 before inference.
    """

    transform = dict(getattr(geometry, "coordinate_transform", {}) or {})
    transform.setdefault("axis_contract", "always_xy")
    source_crs = str(getattr(geometry, "source_crs", "") or "").strip()
    target_crs = str(getattr(geometry, "horizontal_crs", "") or "").strip()
    resolution = "geometry"

    if not source_crs or not target_crs:
        semantics = source_snapshot_context.get("source_snapshot_semantics") or {}
        if isinstance(semantics, Mapping):
            local_crs = str(semantics.get("horizontal_crs_id") or "").strip()
            local_identity_verified = bool(
                local_crs.upper().startswith(_LOCAL_SURVEY_CRS_PREFIX)
                and semantics.get("coordinate_reference_verified") is True
                and str(semantics.get("horizontal_unit") or "").strip().casefold()
                == "m"
                and str(semantics.get("horizontal_axis_order") or "").strip().upper()
                == "XY"
            )
            declared_identities = {
                value for value in (source_crs, target_crs) if value
            }
            if local_identity_verified and declared_identities.issubset({local_crs}):
                source_crs = source_crs or local_crs
                target_crs = target_crs or local_crs
                transform.setdefault("operation", "identity")
                transform.setdefault("transformed", False)
                transform.setdefault(
                    "provenance",
                    "sealed_source_snapshot_verified_local_grid_identity",
                )
                resolution = "sealed_local_grid_default"

    if not source_crs or not target_crs:
        raise ValueError("P13 trace support requires source and target CRS identities")
    if source_crs != target_crs and not transform.get("operation"):
        raise ValueError("P13 non-identity coordinate transform lacks provenance")
    return source_crs, target_crs, transform, resolution


def _canonical_matrix(path: Path, *, columns: int, label: str) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read canonical P13 {label}: {path}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"canonical P13 {label} must have {columns} columns")
    if not np.isfinite(matrix).all():
        raise ValueError(f"canonical P13 {label} contains NaN/Inf")
    if matrix.shape[0] < 2 or bool((np.diff(matrix[:, 0]) <= 0.0).any()):
        raise ValueError(f"canonical P13 {label} MD must be strictly increasing")
    return matrix


def build_p13_trace_support_aligner(pipeline: Any) -> Any:
    """Build the same nine-neighbour spatial support required by P13."""

    matching = dict(pipeline.config.get("matching") or {})
    matching["neighbor_traces"] = P13_NEIGHBOR_TRACE_COUNT
    sources = pipeline._selected_seismic_sources(matching)
    return build_spatial_aligner(matching).fit(sources)


def _sealed_seismic_record(
    source_snapshot_context: Mapping[str, Any],
    *,
    asset: Any,
    geometry: Any,
) -> dict[str, Any]:
    snapshot_id = str(source_snapshot_context.get("source_snapshot_id") or "").strip()
    snapshot_sha256 = _require_sha256(
        source_snapshot_context.get("source_snapshot_fingerprint"),
        field="source snapshot fingerprint",
    )
    if not snapshot_id:
        raise ValueError("P13 trace support requires a sealed SourceSnapshot id")
    source = Path(asset.path).expanduser().resolve()
    matches = [
        dict(item)
        for item in (source_snapshot_context.get("snapshot_assets") or [])
        if isinstance(item, Mapping)
        and item.get("path")
        and os.path.normcase(str(Path(str(item["path"])).expanduser().resolve()))
        == os.path.normcase(str(source))
    ]
    if len(matches) != 1:
        raise ValueError("P13 seismic asset is not uniquely bound to SourceSnapshot")
    sealed = matches[0]
    source_sha256 = _require_sha256(sealed.get("sha256"), field="sealed SEG-Y SHA-256")
    options_sha256 = _require_sha256(
        sealed.get("asset_options_sha256"), field="sealed SEG-Y options SHA-256"
    )
    geometry_fingerprint = _require_sha256(
        sealed.get("geometry_fingerprint"),
        field="sealed SEG-Y geometry fingerprint",
    )
    if canonical_sha256(dict(asset.options or {})) != options_sha256:
        raise ValueError("P13 SEG-Y parser options differ from SourceSnapshot")
    observed_geometry = seismic_geometry_identity(geometry)
    if observed_geometry["geometry_fingerprint"] != geometry_fingerprint:
        raise ValueError("P13 SEG-Y geometry differs from SourceSnapshot")
    stat = source.stat()
    if int(sealed.get("size") or -1) != int(stat.st_size):
        raise ValueError("P13 SEG-Y size differs from SourceSnapshot")
    return {
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "snapshot_contract_version": str(
            source_snapshot_context.get("snapshot_contract_version") or ""
        ),
        "snapshot_manifest_sha256": str(
            source_snapshot_context.get("source_snapshot_manifest_sha256") or ""
        ),
        "source_path": source,
        "source_sha256": source_sha256,
        "source_size": int(stat.st_size),
        "asset_options_sha256": options_sha256,
        "geometry_fingerprint": geometry_fingerprint,
        "geometry_identity": observed_geometry,
    }


def _nine_neighbour_match(
    aligner: Any,
    *,
    x: float,
    y: float,
    asset: Any,
    trace_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    match = aligner.match(float(x), float(y), asset=asset)
    if match is None:
        raise ValueError("P13 trace support lies outside sealed SEG-Y geometry")
    indices = np.asarray(getattr(match, "neighbor_trace_indices", ()), dtype=np.int64)
    distances = np.asarray(getattr(match, "neighbor_distances", ()), dtype=np.float64)
    expected = (P13_NEIGHBOR_TRACE_COUNT,)
    if indices.shape != expected or distances.shape != expected:
        raise ValueError("P13 trace support requires exactly nine neighbours")
    if (
        indices[0] != int(match.trace_index)
        or indices.min() < 0
        or indices.max() >= int(trace_count)
        or not np.isfinite(distances).all()
        or bool((distances < 0.0).any())
        or bool((np.diff(distances) < -1e-9).any())
    ):
        raise ValueError("P13 trace support indices/distances are invalid")
    return indices, distances


def write_p13_trace_support(
    directory: str | Path,
    *,
    pipeline: Any,
    aligner: Any,
    item: Mapping[str, Any],
    acoustic_path: str | Path,
    trajectory_path: str | Path,
    coordinate_fields: Mapping[str, int],
    source_snapshot_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Write and return one immutable, label-free P13 trace-support receipt."""

    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    acoustic_source = Path(acoustic_path).expanduser().resolve()
    trajectory_source = Path(trajectory_path).expanduser().resolve()
    acoustic = _canonical_matrix(acoustic_source, columns=2, label="acoustic input")
    trajectory = _canonical_matrix(
        trajectory_source, columns=4, label="trajectory input"
    )
    lower = max(float(acoustic[0, 0]), float(trajectory[0, 0]))
    upper = min(float(acoustic[-1, 0]), float(trajectory[-1, 0]))
    if upper <= lower:
        raise ValueError("P13 acoustic/trajectory MD intersection is empty")
    trajectory_md = np.linspace(lower, upper, P13_DEPTH_SAMPLE_COUNT, dtype=np.float64)
    head = item["head"]
    trajectory_x = float(head.x) + np.interp(
        trajectory_md, trajectory[:, 0], trajectory[:, 2]
    )
    trajectory_y = float(head.y) + np.interp(
        trajectory_md, trajectory[:, 0], trajectory[:, 3]
    )
    if not np.isfinite(trajectory_x).all() or not np.isfinite(trajectory_y).all():
        raise ValueError("P13 interpolated trajectory coordinates contain NaN/Inf")

    reader = item["reader"]
    geometry = reader.geometry or reader.inspect()
    sealed = _sealed_seismic_record(
        source_snapshot_context, asset=item["asset"], geometry=geometry
    )
    surface_indices, surface_distances = _nine_neighbour_match(
        aligner,
        x=float(head.x),
        y=float(head.y),
        asset=item["asset"],
        trace_count=int(geometry.trace_count),
    )
    trajectory_indices = np.empty(
        (P13_DEPTH_SAMPLE_COUNT, P13_NEIGHBOR_TRACE_COUNT), dtype=np.int64
    )
    trajectory_distances = np.empty(
        (P13_DEPTH_SAMPLE_COUNT, P13_NEIGHBOR_TRACE_COUNT), dtype=np.float64
    )
    for index, (x_value, y_value) in enumerate(zip(trajectory_x, trajectory_y)):
        indices, distances = _nine_neighbour_match(
            aligner,
            x=float(x_value),
            y=float(y_value),
            asset=item["asset"],
            trace_count=int(geometry.trace_count),
        )
        trajectory_indices[index] = indices
        trajectory_distances[index] = distances

    maximum_distance = float(
        pipeline.config.get("matching", {}).get("max_horizontal_distance", 500.0)
    )
    observed_maximum = float(max(surface_distances.max(), trajectory_distances.max()))
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError("P13 maximum horizontal distance is invalid")
    if observed_maximum > maximum_distance:
        raise ValueError(
            "P13 nine-neighbour support exceeds maximum horizontal distance"
        )

    entity = item["entity"]
    safe_name = (
        "".join(
            char if char.isalnum() or char in "_.-" else "_"
            for char in str(entity.well_uid)
        ).strip("._")
        or "well"
    )
    arrays_path = target / f"{safe_name}_trace_support.npz"
    arrays_temporary = target / f".{safe_name}_trace_support.tmp"
    arrays = {
        "surface_trace_indices": surface_indices.astype("<i8", copy=False),
        "surface_trace_distances_m": surface_distances.astype("<f8", copy=False),
        "trajectory_md_m": trajectory_md.astype("<f8", copy=False),
        "trajectory_trace_indices": trajectory_indices.astype("<i8", copy=False),
        "trajectory_trace_distances_m": trajectory_distances.astype("<f8", copy=False),
    }
    with arrays_temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    arrays_temporary.replace(arrays_path)
    arrays_sha256 = file_sha256(arrays_path)

    source_crs, target_crs, geometry_transform, coordinate_identity_resolution = (
        _resolved_coordinate_contract(geometry, source_snapshot_context)
    )
    asset_options = dict(item["asset"].options or {})
    if canonical_sha256(asset_options) != sealed["asset_options_sha256"]:
        raise ValueError("P13 SEG-Y parser options changed before receipt serialization")
    seismic_endian = _sealed_endian(getattr(geometry, "endian", None))
    time_axis_ms = _sealed_time_axis(geometry)
    time_axis_identity = dict(
        (sealed["geometry_identity"].get("arrays") or {}).get("time_axis") or {}
    )
    if (
        time_axis_identity.get("dtype") != "float64"
        or time_axis_identity.get("shape") != [int(geometry.samples_per_trace)]
    ):
        raise ValueError(
            "P13 trace support requires a float64 SEG-Y time-axis identity"
        )

    receipt: dict[str, Any] = {
        "schema": P13_TRACE_SUPPORT_SCHEMA,
        "label_free": True,
        "supervision_asset_count": 0,
        "source_snapshot": {
            "id": sealed["snapshot_id"],
            "sha256": sealed["snapshot_sha256"],
            "contract_version": sealed["snapshot_contract_version"],
            "manifest_sha256": sealed["snapshot_manifest_sha256"] or None,
        },
        "seismic": {
            "path": str(sealed["source_path"]),
            "sha256": sealed["source_sha256"],
            "size_bytes": sealed["source_size"],
            "trace_count": int(geometry.trace_count),
            "samples_per_trace": int(geometry.samples_per_trace),
            "geometry_fingerprint": sealed["geometry_fingerprint"],
            "geometry_identity": sealed["geometry_identity"],
            "asset_options_sha256": sealed["asset_options_sha256"],
            "asset_options": asset_options,
            "endian": seismic_endian,
            "time_axis_ms": time_axis_ms.tolist(),
            "time_axis_identity": time_axis_identity,
            "profile": str(geometry.profile),
            "coordinate_fields": {
                key: int(value) for key, value in coordinate_fields.items()
            },
        },
        "well": {
            "well_uid": str(entity.well_uid),
            "well_id": str(entity.canonical_name),
            "surface_xy_m": [float(head.x), float(head.y)],
            "acoustic_sha256": file_sha256(acoustic_source),
            "trajectory_sha256": file_sha256(trajectory_source),
            "md_intersection_m": [lower, upper],
        },
        "coordinate_contract": {
            "source_crs": source_crs,
            "target_crs": target_crs,
            "horizontal_unit": "m",
            "axis_order": "XY",
            "identity_resolution": coordinate_identity_resolution,
            "coordinate_transform": geometry_transform,
        },
        "selection": {
            "neighbor_trace_count": P13_NEIGHBOR_TRACE_COUNT,
            "trajectory_depth_sample_count": P13_DEPTH_SAMPLE_COUNT,
            "index_semantics": "zero_based_original_segy_trace_order",
            "maximum_neighbor_distance_m": maximum_distance,
            "observed_maximum_neighbor_distance_m": observed_maximum,
            "surface_nearest_distance_m": float(surface_distances[0]),
            "trajectory_maximum_nearest_distance_m": float(
                trajectory_distances[:, 0].max()
            ),
        },
        "arrays": {
            "path": str(arrays_path),
            "sha256": arrays_sha256,
            "members": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
        },
    }
    receipt["support_identity_sha256"] = canonical_sha256(receipt)
    receipt_path = target / f"{safe_name}_trace_support.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_sha256 = file_sha256(receipt_path)
    all_indices = np.unique(
        np.concatenate(
            (
                surface_indices.reshape(-1),
                trajectory_indices.reshape(-1),
            )
        )
    ).astype(np.int64, copy=False)
    return {
        "request": {
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "support_identity_sha256": receipt["support_identity_sha256"],
        },
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "support_identity_sha256": receipt["support_identity_sha256"],
        "arrays_path": str(arrays_path),
        "arrays_sha256": arrays_sha256,
        "trace_indices": all_indices,
        "reader": reader,
    }


__all__ = [
    "P13_DEPTH_SAMPLE_COUNT",
    "P13_NEIGHBOR_TRACE_COUNT",
    "P13_TRACE_SUPPORT_SCHEMA",
    "build_p13_trace_support_aligner",
    "write_p13_trace_support",
]
