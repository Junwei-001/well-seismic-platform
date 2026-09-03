"""Runtime bridge for the promoted GeoPathTie-V1 candidate.

This module intentionally keeps the platform hook small. It consumes a sealed
Registration V3 product, an adapter-attested SEG-Y profile, and canonical LAS
curves before running the frozen ``geopath_full`` checkpoint. It does not
alter the default P13 registration path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..content_identity import canonical_sha256, seismic_geometry_identity
from ..io.las import read_las
from ..io.segy import SegyReader
from ..knowledge import CurveKnowledgeBase
from ..models import WellLog
from ..registration_contract import (
    read_registration_product_v3,
    write_registration_product_v3,
)
from .geopath_tie import (
    bedding_normals_from_time_dip,
    crossing_coefficient,
    extract_seismic_tube_features,
    score_shift_candidates,
    trajectory_geometry,
)
from .geopath_training import (
    GeoPathTensorContract,
    apply_geopath_acceptance_fallback,
    concatenate_contracts,
    contract_from_geopath_outputs,
    load_checkpoint,
    predict_geopath_model,
    regularize_geopath_full_paths,
)
from .well_tie import ricker_wavelet

MODEL_ID = "wellfuse_align_geopath_tie_v1"
DEFAULT_CHECKPOINT_RELATIVE = (
    "models/wellfuse/geopath_tie_v1/checkpoints/geopath_full/final_all.pt"
)

_CANONICAL_CURVES = ("GR", "SP", "CAL", "DT", "RHOB", "NPHI", "RT", "MSFL", "PE")
_HIGH_AUTHORITY_PRIORS = {
    "provided_checkshot_vsp",
    "provided_time_depth",
    "external_registration",
    "learned_geopath_human_accepted",
}
# Keep the acceptance gate consistent with GeoPathTieConfig's 20 ms maximum
# adjacent candidate step. A repair larger than one allowed path step means
# the deployed monotone track no longer represents the solved candidate path.
_MAX_ACCEPTABLE_MONOTONIC_REPAIR_MS = 20.0
_MIN_ACCEPTED_POINT_FRACTION = 0.60


class _GeoPathWellFallback(ValueError):
    """A scientifically safe, well-local fallback that must not abort a batch."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_label(inclination: np.ndarray) -> str:
    value = float(np.nanmedian(inclination))
    if value < 10.0:
        return "vertical"
    if value < 60.0:
        return "deviated"
    return "horizontal"


def _registration_geometry_label(rows: Sequence[Any]) -> str:
    selected = [
        row
        for row in sorted(rows, key=lambda item: item.point_index)
        if row.valid_mask
        and all(
            value is not None and np.isfinite(float(value))
            for value in (row.md_m, row.x, row.y, row.tvd_m)
        )
    ]
    if len(selected) < 2:
        return "unknown"
    try:
        trajectory = trajectory_geometry(
            np.asarray([row.md_m for row in selected], dtype=float),
            np.asarray(
                [[row.x, row.y, row.tvd_m] for row in selected], dtype=float
            ),
        )
    except ValueError:
        return "unknown"
    return _geometry_label(trajectory.inclination_deg)


def _sample_windows(
    values: np.ndarray, time_axis_ms: np.ndarray, centers_ms: np.ndarray, half_window_ms: float
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centers = np.asarray(centers_ms, dtype=float)
    dt_ms = float(np.median(np.diff(time_axis_ms)))
    radius = max(2, round(half_window_ms / dt_ms))
    center = np.rint((centers - time_axis_ms[0]) / dt_ms).astype(int)
    index = center[:, None] + np.arange(-radius, radius + 1)[None, :]
    valid = (index >= 0) & (index < values.size)
    result = values[np.clip(index, 0, values.size - 1)]
    result[~valid] = np.nan
    return result


def _unknown_unit(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"", "unknown", "null", "none"}


def _load_canonical_las(path: Path, config: Mapping[str, Any]) -> WellLog:
    """Use the platform curve knowledge/unit converter, never a private LAS parser."""

    configuration = dict(config)
    log = read_las(
        path,
        CurveKnowledgeBase(configuration),
        dict(configuration.get("preprocessing") or {}),
    )
    if "DT" not in log.curves or "DT" not in log.curve_info:
        raise ValueError(f"sealed LAS lacks a canonical DT curve: {path}")
    for curve_name in ("DT", "RHOB"):
        if curve_name not in log.curves:
            continue
        info = log.curve_info.get(curve_name)
        if info is None or _unknown_unit(info.original_unit):
            raise ValueError(
                f"sealed LAS {curve_name} unit is unknown; GeoPathTie fails closed: {path}"
            )
        conversion_issue = any(
            issue.startswith(f"{info.original_name}:unit_conversion_unavailable:")
            for issue in log.issues
        )
        if conversion_issue:
            raise ValueError(
                f"sealed LAS {curve_name} unit cannot be converted canonically: {path}"
            )
    if float(np.mean(np.asarray(log.masks["DT"], dtype=bool))) < 0.25:
        raise ValueError(f"sealed LAS lacks sufficient canonical DT coverage: {path}")
    return log


def _sample_canonical_las(
    path: Path,
    log: WellLog,
    md_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.full((len(_CANONICAL_CURVES), md_query.size), np.nan, dtype=float)
    masks = np.zeros_like(values, dtype=float)
    depth = np.asarray(log.depth, dtype=float)
    for channel, curve_name in enumerate(_CANONICAL_CURVES):
        if curve_name not in log.curves:
            continue
        source = np.asarray(log.curves[curve_name], dtype=float)
        source_mask = np.asarray(log.masks.get(curve_name), dtype=bool)
        finite = np.isfinite(depth) & np.isfinite(source) & source_mask
        if int(np.count_nonzero(finite)) < 2:
            continue
        unique_depth, unique_index = np.unique(depth[finite], return_index=True)
        source = source[finite][unique_index]
        inside = (md_query >= unique_depth[0]) & (md_query <= unique_depth[-1])
        values[channel, inside] = np.interp(md_query[inside], unique_depth, source)
        masks[channel, inside] = 1.0
    if float(np.mean(masks[3])) < 0.25:
        raise ValueError(f"sealed LAS lacks usable canonical DT samples: {path}")
    unit_receipt = {
        name: {
            "original_unit": log.curve_info[name].original_unit,
            "canonical_unit": log.curve_info[name].standard_unit,
        }
        for name in ("DT", "RHOB")
        if name in log.curve_info
    }
    return values, masks, {
        "path": str(path),
        "sha256": _sha256(path),
        "well_name": log.well_name,
        "unit_receipt": unit_receipt,
        "processing_steps": list(log.processing_steps),
        "issues": list(log.issues),
    }


def _identity_token(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _match_las_assets(
    grouped: Mapping[str, list[Any]],
    las_assets: Mapping[Path, WellLog],
) -> dict[str, Path]:
    """Return an exact, injective Registration-well to LAS mapping."""

    selected: dict[str, Path] = {}
    owner_by_path: dict[Path, str] = {}
    for well_uid, rows in grouped.items():
        well_tokens = {
            token
            for token in (
                _identity_token(well_uid),
                _identity_token(rows[0].well_uid),
                _identity_token(rows[0].well_name),
            )
            if token
        }
        matches = []
        for path, log in las_assets.items():
            las_tokens = {
                token
                for token in (
                    _identity_token(path.stem),
                    _identity_token(log.well_name),
                )
                if token
            }
            if well_tokens & las_tokens:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                f"Registration V3 well {well_uid} requires exactly one identity-matched LAS; "
                f"matched={len(matches)}"
            )
        path = matches[0]
        previous_owner = owner_by_path.get(path)
        if previous_owner is not None:
            raise ValueError(
                f"sealed LAS cannot be reused across wells: {path} matched "
                f"{previous_owner} and {well_uid}"
            )
        owner_by_path[path] = well_uid
        selected[well_uid] = path
    return selected


def _reader_from_profile_receipt(
    seismic_path: Path,
    receipt: Mapping[str, Any],
) -> tuple[SegyReader, Any]:
    if receipt.get("contract_version") != "well-seismic.segy-profile-receipt.v1":
        raise ValueError("GeoPathTie requires an adapter-resolved SEG-Y profile receipt")
    profile_name = str(receipt.get("profile_name") or "")
    profile_definition = dict(receipt.get("profile_definition") or {})
    if not profile_name or not profile_definition:
        raise ValueError("GeoPathTie SEG-Y profile receipt is incomplete")
    if canonical_sha256(profile_definition) != str(
        receipt.get("profile_definition_sha256") or ""
    ):
        raise ValueError("GeoPathTie SEG-Y profile receipt hash mismatch")
    resolved = dict(receipt.get("resolved_header_bytes") or {})
    if set(resolved) != {"inline", "crossline", "x", "y"}:
        raise ValueError("GeoPathTie SEG-Y receipt lacks resolved spatial header bytes")
    scalar_byte = receipt.get("coordinate_scalar_byte")
    if scalar_byte is None:
        raise ValueError("GeoPathTie SEG-Y receipt lacks coordinate scalar byte")
    options = {
        "profile": profile_name,
        **{f"{name}_byte": int(value) for name, value in resolved.items()},
        "coordinate_scalar_byte": int(scalar_byte),
    }
    reader = SegyReader(
        seismic_path,
        {"segy": {"profiles": {profile_name: profile_definition}}},
        options,
    )
    geometry = reader.inspect()
    minimum_confidence = float(receipt.get("minimum_geometry_confidence") or 0.0)
    if geometry.confidence < minimum_confidence:
        raise ValueError(
            "GeoPathTie runtime SEG-Y geometry confidence is below the adapter gate"
        )
    observed_identity = seismic_geometry_identity(geometry)
    if observed_identity["geometry_fingerprint"] != str(
        receipt.get("geometry_fingerprint") or ""
    ):
        raise ValueError("GeoPathTie runtime SEG-Y interpretation differs from adapter receipt")
    return reader, geometry


def _synthetic_trace(curve_values: np.ndarray, curve_masks: np.ndarray, reference_twt_ms: np.ndarray, time_axis_ms: np.ndarray) -> np.ndarray:
    acoustic = np.asarray(curve_values[3], dtype=float)
    density = np.asarray(curve_values[4], dtype=float)
    acoustic_mask = np.asarray(curve_masks[3] > 0.5, dtype=bool)
    density_mask = np.asarray(curve_masks[4] > 0.5, dtype=bool)
    sample = np.arange(acoustic.size)
    if int(acoustic_mask.sum()) >= 2:
        acoustic = np.interp(sample, sample[acoustic_mask], acoustic[acoustic_mask])
    else:
        acoustic = np.zeros_like(acoustic)
    if int(density_mask.sum()) >= 2:
        density = np.interp(sample, sample[density_mask], density[density_mask])
        log_impedance = -acoustic + 0.35 * density
    else:
        log_impedance = -acoustic
    log_impedance = np.convolve(log_impedance, np.ones(5) / 5.0, mode="same")
    reflectivity = np.zeros_like(log_impedance)
    reflectivity[1:] = np.tanh(0.5 * np.diff(log_impedance))
    dt_ms = float(np.median(np.diff(time_axis_ms)))
    impulse = np.zeros(time_axis_ms.size, dtype=float)
    index = np.rint((reference_twt_ms - time_axis_ms[0]) / dt_ms).astype(int)
    valid = np.isfinite(reference_twt_ms) & (index >= 0) & (index < impulse.size) & acoustic_mask
    np.add.at(impulse, index[valid], reflectivity[valid])
    synthetic = np.convolve(impulse, ricker_wavelet(30.0, dt_ms, duration_ms=128.0), mode="same")
    center = float(np.nanmedian(synthetic))
    scale = float(np.nanmedian(np.abs(synthetic - center)) * 1.4826)
    return (synthetic - center) / max(scale, 1e-6)


class _RuntimeRawTubeReader:
    """Physical-aperture raw tube reader equivalent to the training gather."""

    def __init__(self, reader: SegyReader, *, aperture_m: float = 55.0, max_neighbors: int = 17) -> None:
        geometry = reader.geometry or reader.inspect()
        if geometry.x is None or geometry.y is None:
            raise ValueError("SEG-Y lacks sealed XY geometry required for physical raw tube")
        from scipy.spatial import cKDTree
        self.reader = reader
        self.geometry = geometry
        self.x = np.asarray(geometry.x, dtype=float)
        self.y = np.asarray(geometry.y, dtype=float)
        self.tree = cKDTree(np.column_stack((self.x, self.y)))
        self.aperture_m = float(aperture_m)
        self.max_neighbors = int(max_neighbors)

    def gather(self, xy: np.ndarray, reference: np.ndarray, half_window_ms: float) -> tuple[Any, np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=float)
        distances, indices = self.tree.query(points, k=self.max_neighbors, distance_upper_bound=self.aperture_m)
        if indices.ndim == 1:
            indices, distances = indices[:, None], distances[:, None]
        valid = indices < self.x.size
        unique = np.unique(indices[valid])
        dt_ms = float(np.median(np.diff(self.geometry.time_axis)))
        radius = max(2, round(half_window_ms / dt_ms))
        centers = np.rint((reference - self.geometry.time_axis[0]) / dt_ms).astype(int)
        sample_index = centers[:, None] + np.arange(-radius, radius + 1)[None, :]
        valid_time = (sample_index >= 0) & (
            sample_index < self.geometry.samples_per_trace
        )
        sample_index = np.clip(
            sample_index, 0, self.geometry.samples_per_trace - 1
        )
        tube = np.full((points.shape[0], indices.shape[1], sample_index.shape[1]), np.nan, dtype=np.float32)
        offsets = np.full((points.shape[0], indices.shape[1], 2), np.nan, dtype=float)
        if unique.size:
            traces = np.stack(
                [self.reader.read_trace(int(index)) for index in unique]
            ).astype(np.float32)
            median = np.median(traces, axis=1, keepdims=True)
            mad = np.median(np.abs(traces - median), axis=1, keepdims=True) * 1.4826
            traces = (traces - median) / np.maximum(mad, 1e-6)
            for neighbor in range(indices.shape[1]):
                rows = np.flatnonzero(valid[:, neighbor])
                if not rows.size:
                    continue
                position = np.searchsorted(unique, indices[rows, neighbor])
                tube[rows, neighbor] = traces[
                    position[:, None], sample_index[rows]
                ]
                tube[rows, neighbor][~valid_time[rows]] = np.nan
                offsets[rows, neighbor, 0] = (
                    self.x[indices[rows, neighbor]] - points[rows, 0]
                )
                offsets[rows, neighbor, 1] = (
                    self.y[indices[rows, neighbor]] - points[rows, 1]
                )
        return extract_seismic_tube_features(tube, sample_interval_ms=dt_ms, trace_offsets_xy_m=offsets, center_trace_index=0), valid.sum(axis=1), distances[:, 0]


def _nearest_trace_indices(
    geometry: Any, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    available = geometry.x is not None and geometry.y is not None
    if not available:
        return (
            np.zeros(xy.shape[0], dtype=int),
            np.full(xy.shape[0], np.nan, dtype=float),
        )
    trace_xy = np.column_stack((geometry.x, geometry.y)).astype(float)
    valid = np.all(np.isfinite(trace_xy), axis=1)
    if not np.any(valid):
        return (
            np.zeros(xy.shape[0], dtype=int),
            np.full(xy.shape[0], np.nan, dtype=float),
        )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(trace_xy[valid])
        distance, local = tree.query(xy, k=1)
        indices = np.flatnonzero(valid)[np.asarray(local, dtype=int)]
        return np.asarray(indices, dtype=int), np.asarray(distance, dtype=float)
    except ImportError:
        indices = np.empty(xy.shape[0], dtype=int)
        distances = np.empty(xy.shape[0], dtype=float)
        valid_indices = np.flatnonzero(valid)
        for row, point in enumerate(xy):
            delta = trace_xy[valid] - point
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            choice = int(np.argmin(distance))
            indices[row] = int(valid_indices[choice])
            distances[row] = float(distance[choice])
        return indices, distances


def _station_windows(
    reader: SegyReader,
    trace_indices: np.ndarray,
    twt_ms: np.ndarray,
    *,
    half_window_ms: float = 120.0,
) -> tuple[np.ndarray, float]:
    geometry = reader.geometry or reader.inspect()
    time_axis = np.asarray(geometry.time_axis, dtype=float)
    dt_ms = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else 2.0
    radius = max(8, round(half_window_ms / max(dt_ms, 1.0e-6)))
    width = 2 * radius + 1
    windows = np.full((twt_ms.size, width), np.nan, dtype=np.float32)
    for index, (trace_index, center) in enumerate(zip(trace_indices, twt_ms, strict=True)):
        center_index = int(np.argmin(np.abs(time_axis - float(center))))
        start = max(0, center_index - radius)
        stop = min(geometry.samples_per_trace, center_index + radius + 1)
        values = reader.read_trace(int(trace_index), slice(start, stop))
        left = radius - (center_index - start)
        right = left + values.size
        windows[index, left:right] = values[: max(0, right - left)]
    return windows, dt_ms


def _build_well_contract(
    reader: SegyReader,
    well_uid: str,
    rows: list[Any],
    las_path: Path,
    las_log: WellLog,
) -> tuple[GeoPathTensorContract, dict[str, Any], np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda row: row.point_index)
    registration_valid = np.asarray([row.valid_mask for row in rows], dtype=bool)
    md_all = np.asarray([row.md_m for row in rows], dtype=float)
    x_all = np.asarray([row.x for row in rows], dtype=float)
    y_all = np.asarray([row.y for row in rows], dtype=float)
    tvd_all = np.asarray([row.tvd_m for row in rows], dtype=float)
    reference_all = np.asarray([row.twt_mean_ms for row in rows], dtype=float)
    quality_all = np.asarray([row.quality for row in rows], dtype=float)
    complete = (
        np.isfinite(md_all)
        & np.isfinite(x_all)
        & np.isfinite(y_all)
        & np.isfinite(tvd_all)
        & np.isfinite(reference_all)
    )
    if np.any(registration_valid & ~complete):
        raise _GeoPathWellFallback(
            "registration_valid_point_missing_xyz_tvd_or_twt"
        )
    valid = registration_valid & complete
    if int(np.count_nonzero(valid)) < 2:
        raise _GeoPathWellFallback("registration_has_too_few_complete_valid_points")
    source_point_indices = np.asarray(
        [row.point_index for row in rows], dtype=int
    )[valid]
    md, x, y, tvd, reference, quality = (
        values[valid]
        for values in (md_all, x_all, y_all, tvd_all, reference_all, quality_all)
    )
    xyz = np.column_stack((x, y, tvd))
    trajectory = trajectory_geometry(md, xyz)
    if not np.all(trajectory.valid_mask):
        raise _GeoPathWellFallback("registration_v3_trajectory_geometry_is_invalid")
    geometry_label = _geometry_label(trajectory.inclination_deg)
    curve_values, curve_masks, las_receipt = _sample_canonical_las(
        las_path, las_log, md
    )
    acoustic = curve_values[3]
    finite_acoustic = np.isfinite(acoustic)
    sonic = reference.copy()
    if int(np.count_nonzero(finite_acoustic)) >= 2:
        sonic[0] = reference[0]
        for index in range(1, md.size):
            dz = float(tvd[index] - tvd[index - 1])
            if (
                finite_acoustic[index - 1]
                and finite_acoustic[index]
                and dz >= 0.0
            ):
                mean_slowness_us_m = 0.5 * (
                    acoustic[index - 1] + acoustic[index]
                )
                sonic[index] = (
                    sonic[index - 1]
                    + 2.0 * mean_slowness_us_m * dz / 1000.0
                )
            else:
                sonic[index] = reference[index]
    raw_tube = _RuntimeRawTubeReader(reader, aperture_m=55.0, max_neighbors=17)
    tube, neighbor_count, distance = raw_tube.gather(np.column_stack((x, y)), reference, 220.0)
    trace_available = (neighbor_count > 0) & np.isfinite(distance) & (distance <= 55.0)
    if not np.any(trace_available):
        raise _GeoPathWellFallback("no_seismic_trace_within_55m_aperture")
    dt_ms = float(np.median(np.diff(reader.geometry.time_axis)))
    normal = bedding_normals_from_time_dip(
        tube.dip_x_ms_per_m, tube.dip_y_ms_per_m, vertical_velocity_m_s=3500.0
    )
    invalid_normal = ~np.all(np.isfinite(normal), axis=1)
    normal[invalid_normal] = np.asarray((0.0, 0.0, 1.0))
    kappa = np.nan_to_num(crossing_coefficient(trajectory.tangent_xyz, normal), nan=0.0)
    synthetic = _synthetic_trace(curve_values, curve_masks, reference, reader.geometry.time_axis)
    synthetic_window = _sample_windows(synthetic, reader.geometry.time_axis, reference, 220.0)
    observed_window = np.asarray(tube.robust_mean, dtype=float)
    finite_window = np.any(np.isfinite(observed_window), axis=1)
    shifts = np.arange(-120.0, 120.1, 4.0, dtype=float)
    scores = score_shift_candidates(
        synthetic_window,
        observed_window,
        shifts,
        sample_interval_ms=dt_ms,
        kappa=kappa,
        min_kappa=0.18,
        min_overlap_samples=12,
    )
    coherence = np.nanmedian(tube.coherence, axis=1)
    candidate_score_full = np.nan_to_num(scores.score * np.nan_to_num(coherence[:, None], nan=0.0), nan=0.0)
    candidate_corr_full = np.nan_to_num(scores.correlation, nan=0.0)
    candidate_score = np.zeros((md.size, 9), dtype=float)
    candidate_corr = np.zeros_like(candidate_score)
    candidate_shift = np.zeros_like(candidate_score)
    zero_index = int(np.argmin(np.abs(shifts)))
    for station in range(md.size):
        ranked = np.argsort(-candidate_score_full[station], kind="stable")
        ordered: list[int] = []
        for choice in (zero_index, *ranked.tolist()):
            if int(choice) not in ordered:
                ordered.append(int(choice))
            if len(ordered) == 9:
                break
        selected = np.asarray(ordered, dtype=int)
        candidate_shift[station] = shifts[selected]
        candidate_score[station] = candidate_score_full[station, selected]
        candidate_corr[station] = candidate_corr_full[station, selected]
    finite_window = np.isfinite(observed_window).any(axis=1)
    acceptance_eligible = trace_available & finite_window
    candidate_valid = np.broadcast_to(
        acceptance_eligible[:, None], candidate_score.shape
    ).copy()
    # The tensor contract requires one finite candidate per station. Stations
    # outside the aperture carry only a zero-shift placeholder and are forced
    # through the sealed-prior fallback after inference.
    candidate_valid[~acceptance_eligible] = False
    candidate_valid[~acceptance_eligible, 0] = True
    contract = contract_from_geopath_outputs(
        reference_twt_ms=reference,
        sonic_twt_ms=sonic,
        p13_twt_ms=reference,
        p13_confidence=np.nan_to_num(quality, nan=0.0),
        kappa=kappa,
        inclination_deg=np.nan_to_num(trajectory.inclination_deg, nan=0.0),
        dogleg_deg_per_30m=np.nan_to_num(trajectory.dogleg_deg_per_30m, nan=0.0),
        curve_quality=np.mean(curve_masks > 0.5, axis=0),
        tube_coherence=np.nan_to_num(tube.coherence, nan=0.0),
        tube_std=np.nan_to_num(tube.std, nan=0.0),
        tube_phase=np.nan_to_num(tube.phase_rad, nan=0.0),
        candidate_shift_ms=candidate_shift,
        candidate_score=candidate_score,
        candidate_correlation=candidate_corr,
        candidate_valid_mask=candidate_valid,
        target_twt_ms=np.full(md.size, np.nan, dtype=float),
        supervision_mask=np.zeros(md.size, dtype=bool),
        md_m=md,
        dominant_period_ms=np.full(md.size, 1000.0 / 30.0, dtype=float),
        well_id=np.full(md.size, well_uid),
        family_id=np.full(md.size, well_uid),
        geometry=np.full(md.size, geometry_label),
        anchor_distance_m=np.where(scores.seismic_evidence_mask, 0.0, 1000.0),
    )
    contract.validate()
    return contract, {
        "well_uid": well_uid,
        "geometry": geometry_label,
        "point_count": int(md.size),
        "registration_valid_point_count": int(registration_valid.sum()),
        "registration_invalid_point_count": int((~registration_valid).sum()),
        "trajectory_authority": "registration_v3_points",
        "las": las_receipt,
        "tube_neighbor_count_median": float(np.median(neighbor_count)),
        "kappa_median": float(np.median(kappa)),
        "nearest_trace_distance_median": float(
            np.median(distance[np.isfinite(distance)])
        ),
        "seismic_window_valid_fraction": float(np.mean(finite_window)),
        "aperture_eligible_fraction": float(np.mean(acceptance_eligible)),
    }, acceptance_eligible, source_point_indices


def run_geopath_tie_v1(
    *,
    seismic_path: Path,
    registration_manifest_path: Path,
    output_directory: Path,
    segy_profile_receipt: Mapping[str, Any],
    canonical_las_config: Mapping[str, Any],
    checkpoint_path: Path | None = None,
    device: str = "auto",
    source_snapshot_id: str | None = None,
    project_root: Path | None = None,
    las_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Run a real frozen GeoPath checkpoint and write a candidate V3 product."""

    product = read_registration_product_v3(registration_manifest_path)
    reader, geometry = _reader_from_profile_receipt(
        seismic_path, segy_profile_receipt
    )
    observed_seismic_sha256 = _sha256(seismic_path)
    expected_seismic_sha256 = str(
        segy_profile_receipt.get("source_asset_sha256") or ""
    )
    if (
        expected_seismic_sha256
        and expected_seismic_sha256 != observed_seismic_sha256
    ):
        raise ValueError("GeoPathTie SEG-Y bytes differ from the adapter receipt")

    grouped: dict[str, list[Any]] = {}
    for point in product.points:
        grouped.setdefault(point.well_uid or point.well_name, []).append(point)

    contracts: list[GeoPathTensorContract] = []
    diagnostics: list[dict[str, Any]] = []
    sealed_las = [Path(path).expanduser().resolve() for path in (las_paths or [])]
    if not sealed_las or not all(path.is_file() for path in sealed_las):
        raise FileNotFoundError("GeoPathTie requires sealed LAS paths for every Registration V3 well")
    if len(set(sealed_las)) != len(sealed_las):
        raise ValueError("GeoPathTie LAS inputs contain duplicate file paths")
    canonical_las = {
        path: _load_canonical_las(path, canonical_las_config)
        for path in sealed_las
    }
    las_by_well = _match_las_assets(grouped, canonical_las)
    acceptance_parts: list[np.ndarray] = []
    point_indices_by_well: dict[str, np.ndarray] = {}
    fallback_by_well: dict[str, str] = {}

    for well_uid, rows in grouped.items():
        source_authority = str(rows[0].source_authority)
        registration_geometry = _registration_geometry_label(rows)
        if source_authority in _HIGH_AUTHORITY_PRIORS:
            reason = f"protected_high_authority_prior:{source_authority}"
            fallback_by_well[well_uid] = reason
            diagnostics.append(
                {
                    "well_uid": well_uid,
                    "trajectory_authority": "registration_v3_points",
                    "runtime_status": "prior_fallback",
                    "fallback_reason": reason,
                    "source_authority": source_authority,
                    "point_count": len(rows),
                    "geometry": registration_geometry,
                    "accepted_fraction": 0.0,
                    "minimum_accepted_fraction": _MIN_ACCEPTED_POINT_FRACTION,
                    "aperture_eligible_fraction": 0.0,
                    "repair_status": "not_required_prior_preserved",
                    "repair_reason": reason,
                    "monotonic_repair_count": 0,
                    "monotonic_repair_max_ms": 0.0,
                    "monotonic_repair_total_ms": 0.0,
                    "acceptance_eligible": False,
                }
            )
            continue
        try:
            contract_part, detail, acceptance_eligible, source_point_indices = (
                _build_well_contract(
                    reader,
                    well_uid,
                    rows,
                    las_by_well[well_uid],
                    canonical_las[las_by_well[well_uid]],
                )
            )
        except _GeoPathWellFallback as exc:
            reason = str(exc)
            fallback_by_well[well_uid] = reason
            diagnostics.append(
                {
                    "well_uid": well_uid,
                    "trajectory_authority": "registration_v3_points",
                    "runtime_status": "prior_fallback",
                    "fallback_reason": reason,
                    "source_authority": source_authority,
                    "point_count": len(rows),
                    "geometry": registration_geometry,
                    "las_path": str(las_by_well[well_uid]),
                    "accepted_fraction": 0.0,
                    "minimum_accepted_fraction": _MIN_ACCEPTED_POINT_FRACTION,
                    "aperture_eligible_fraction": 0.0,
                    "repair_status": "not_required_prior_preserved",
                    "repair_reason": reason,
                    "monotonic_repair_count": 0,
                    "monotonic_repair_max_ms": 0.0,
                    "monotonic_repair_total_ms": 0.0,
                    "acceptance_eligible": False,
                }
            )
            continue
        contracts.append(contract_part)
        acceptance_parts.append(acceptance_eligible)
        point_indices_by_well[well_uid] = source_point_indices
        diagnostics.append(
            {
                **detail,
                "runtime_status": "modeled_candidate",
                "source_authority": source_authority,
                "prior_feature_role": (
                    "p13_registration_prior"
                    if source_authority.startswith("learned_p13_")
                    else "sealed_registration_prior_proxy"
                ),
            }
        )

    resolved_checkpoint = (
        checkpoint_path
        or (project_root or seismic_path.parent.parent) / DEFAULT_CHECKPOINT_RELATIVE
    ).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"GeoPathTie checkpoint not found: {resolved_checkpoint}")

    checkpoint: dict[str, Any] = {}
    contract: GeoPathTensorContract | None = None
    raw_twt = np.empty(0, dtype=float)
    final_twt = np.empty(0, dtype=float)
    accepted = np.empty(0, dtype=bool)
    sigma = np.empty(0, dtype=float)
    if contracts:
        contract = concatenate_contracts(contracts)
        acceptance_eligible = np.concatenate(acceptance_parts)
        model, checkpoint = load_checkpoint(resolved_checkpoint, device=device)
        prediction = predict_geopath_model(
            model,
            contract,
            np.ones(contract.point_count, dtype=bool),
            arm="geopath_full",
            device=device,
        )
        prediction = regularize_geopath_full_paths(contract, prediction)
        accepted = (
            np.asarray(prediction["accepted"], dtype=bool)
            & acceptance_eligible
        )
        raw_twt, final_twt = apply_geopath_acceptance_fallback(
            contract.reference_twt_ms,
            contract.p13_twt_ms,
            prediction["predicted_shift_ms"],
            accepted,
        )
        sigma = np.asarray(prediction["sigma_ms"], dtype=float)

    updated_tracks: list[dict[str, Any]] = []
    for identity, track in product.tracks.items():
        updated = dict(track)
        modeled = contract is not None and identity in point_indices_by_well
        if modeled:
            mask = np.asarray(contract.well_id, dtype=str) == str(identity)
            point_indices = point_indices_by_well[identity]
            candidate_twt_raw = np.asarray(final_twt[mask], dtype=float)
            candidate_twt = candidate_twt_raw.copy()
            for index in range(1, candidate_twt.size):
                candidate_twt[index] = max(
                    candidate_twt[index], candidate_twt[index - 1] + 1.0e-3
                )
            repair_delta = candidate_twt - candidate_twt_raw
            repaired = np.abs(repair_delta) > 1.0e-9
            full_twt = np.asarray(updated.get("twtMean"), dtype=float)
            full_std = np.asarray(updated.get("twtStd"), dtype=float)
            full_twt[point_indices] = candidate_twt
            full_std[point_indices] = sigma[mask]
            updated["twtMean"] = full_twt.astype(float).tolist()
            updated["twtStd"] = full_std.astype(float).tolist()
            updated["registrationSource"] = MODEL_ID
            updated["registrationStatus"] = "candidate"
            updated["sourceAuthority"] = "learned_geopath_candidate"
            updated["uncertaintyCalibrated"] = False
            updated["uncertaintySource"] = "geopath_v1_path_posterior"
            runtime_diagnostics = {
                "geopath_candidate": True,
                "runtime_status": "modeled_candidate",
                "accepted_fraction": float(np.mean(accepted[mask])),
                "raw_twt_range_ms": [
                    float(np.nanmin(raw_twt[mask])),
                    float(np.nanmax(raw_twt[mask])),
                ],
                "monotonic_repair_count": int(repaired.sum()),
                "monotonic_repair_max_ms": (
                    float(np.max(np.abs(repair_delta))) if repaired.any() else 0.0
                ),
                "monotonic_repair_total_ms": float(np.sum(np.abs(repair_delta))),
            }
            repair_max_ms = (
                float(np.max(np.abs(repair_delta))) if repaired.any() else 0.0
            )
            repair_is_safe = repair_max_ms <= _MAX_ACCEPTABLE_MONOTONIC_REPAIR_MS
            accepted_fraction = float(np.mean(accepted[mask]))
            for detail in diagnostics:
                if detail.get("well_uid") == identity:
                    detail.update(
                        {
                            "accepted_fraction": accepted_fraction,
                            "minimum_accepted_fraction": (
                                _MIN_ACCEPTED_POINT_FRACTION
                            ),
                            "repair_status": (
                                "blocked_excessive_monotonic_repair"
                                if not repair_is_safe
                                else (
                                    "monotonic_repaired_within_tolerance"
                                    if repaired.any()
                                    else "not_required"
                                )
                            ),
                            "repair_reason": (
                                "strict_registration_v3_twt_monotonicity"
                                if repaired.any()
                                else None
                            ),
                            "acceptance_eligible": bool(
                                accepted_fraction >= _MIN_ACCEPTED_POINT_FRACTION
                                and repair_is_safe
                            ),
                            "monotonic_repair_count": int(repaired.sum()),
                            "monotonic_repair_max_ms": repair_max_ms,
                            "monotonic_repair_total_ms": float(
                                np.sum(np.abs(repair_delta))
                            ),
                            "monotonic_repair_limit_ms": (
                                _MAX_ACCEPTABLE_MONOTONIC_REPAIR_MS
                            ),
                        }
                    )
                    break
        else:
            original_authority = str(updated.get("sourceAuthority") or "")
            if original_authority == "learned_p13_fusion_ready":
                updated["sourceAuthority"] = "learned_p13_candidate"
            updated["registrationStatus"] = "candidate_prior_fallback"
            runtime_diagnostics = {
                "geopath_candidate": True,
                "runtime_status": "prior_fallback",
                "fallback_reason": fallback_by_well.get(
                    identity, "well_not_modeled"
                ),
                "monotonic_repair_count": 0,
                "monotonic_repair_max_ms": 0.0,
                "monotonic_repair_total_ms": 0.0,
            }
        updated["fusionReady"] = False
        updated["diagnostics"] = {
            **dict(updated.get("diagnostics") or {}),
            **runtime_diagnostics,
        }
        updated_tracks.append(updated)

    if not updated_tracks:
        raise ValueError("GeoPath produced no Registration V3 tracks")

    manifest_well_diagnostics = [
        {
            "well_id": detail.get("well_uid"),
            **{
                key: detail.get(key)
                for key in (
                    "geometry",
                    "accepted_fraction",
                    "minimum_accepted_fraction",
                    "aperture_eligible_fraction",
                    "repair_status",
                    "repair_reason",
                    "monotonic_repair_count",
                    "monotonic_repair_max_ms",
                    "monotonic_repair_total_ms",
                    "monotonic_repair_limit_ms",
                    "acceptance_eligible",
                )
            },
        }
        for detail in diagnostics
    ]
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    candidate_directory = output_directory / "registration_v3_candidate"
    manifest = product.manifest
    coordinate = manifest.get("coordinate_contract") or {}
    vertical = manifest.get("vertical_contract") or {}
    timing = manifest.get("time_contract") or {}
    candidate_product = write_registration_product_v3(
        candidate_directory,
        updated_tracks,
        semantics={
            "horizontal_crs_id": coordinate.get("horizontal_crs_id"),
            "horizontal_unit": coordinate.get("horizontal_unit"),
            "horizontal_axis_order": coordinate.get("horizontal_axis_order"),
            "vertical_crs_id": vertical.get("vertical_crs_id"),
            "seismic_srd_elevation_m": vertical.get("seismic_srd_elevation_m"),
            "time_domain": timing.get("time_domain"),
            "time_reference": timing.get("time_reference") or "SRD",
            "correction_state": timing.get("correction_state"),
        },
        manifest_fields={
            "source_snapshot_id": source_snapshot_id or manifest.get("source_snapshot_id"),
            "source_snapshot_fingerprint": manifest.get("source_snapshot_fingerprint"),
            "registration_source_policy": "GeoPathTie-V1 candidate over sealed Registration V3 prior",
            "geopath_checkpoint": str(resolved_checkpoint),
            "geopath_checkpoint_sha256": _sha256(resolved_checkpoint),
            "geopath_checkpoint_policy": checkpoint.get("selected_epoch_policy"),
            "geopath_checkpoint_executed": bool(contracts),
            "trajectory_authority": "registration_v3_points",
            "segy_profile_receipt_sha256": canonical_sha256(
                dict(segy_profile_receipt)
            ),
            "geopath_well_diagnostics": manifest_well_diagnostics,
            "candidate_status": "candidate_not_promoted",
        },
    )
    diagnostics_path = output_directory / "geopath_runtime_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "checkpoint": str(resolved_checkpoint),
                "source_snapshot_id": source_snapshot_id,
                "trajectory_authority": "registration_v3_points",
                "segy_profile_receipt": dict(segy_profile_receipt),
                "wells": diagnostics,
                "accepted_point_fraction": (
                    float(np.mean(accepted)) if accepted.size else 0.0
                ),
                "modeled_well_count": len(point_indices_by_well),
                "fallback_well_count": len(fallback_by_well),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    joined_wells = sorted(str(item) for item in grouped)
    modeled_wells = sorted(point_indices_by_well)
    las_receipt = {
        "contract_version": "well-seismic.geopath-las-consumption.v1",
        "mapping_policy": "exact_identity_one_las_per_registration_well",
        "canonicalizer": "well_seismic.io.las.read_las",
        "well_to_las": {
            well_uid: {
                "path": str(path),
                "sha256": _sha256(path),
                "canonical_well_name": canonical_las[path].well_name,
            }
            for well_uid, path in sorted(las_by_well.items())
        },
        "unused_las_paths": sorted(
            str(path) for path in set(sealed_las) - set(las_by_well.values())
        ),
    }
    receipt = {
        "status": "consumed",
        "registration_consumed": True,
        "registration_manifest_sha256": _sha256(product.manifest_path),
        "registration_points_sha256": _sha256(product.points_path),
        "joined_well_ids": joined_wells,
        "modeled_well_ids": modeled_wells,
        "fallback_well_ids": sorted(fallback_by_well),
        "joined_row_count": len(product.points),
        "join_coverage_fraction": 1.0,
        "trajectory_authority": "registration_v3_points",
        "segy_profile_receipt": dict(segy_profile_receipt),
        "las_consumption_receipt": las_receipt,
        "feature_channels": [
            "sealed_registration_twt_prior",
            "registration_v3_xyz_tvd_trajectory",
            "segy_nearest_trace_tube",
            "geopath_candidate_path",
        ],
    }
    return {
        "model_id": MODEL_ID,
        "model_name": "轨迹感知井震校正（井震数据+完整轨迹→候选时深轨）",
        "model_executed": bool(contracts),
        "execution_status": (
            "candidate_registration_v3" if contracts else "prior_fallback_only"
        ),
        "device": device,
        "input": {
            "seismic_path": str(seismic_path.resolve()),
            "seismic_sha256": observed_seismic_sha256,
            "segy_profile_receipt": dict(segy_profile_receipt),
            "las_consumption_receipt": las_receipt,
            "trajectory_authority": "registration_v3_points",
            "registration_manifest_path": str(product.manifest_path),
            "registration_points_path": str(product.points_path),
            "registration_contract_version": product.manifest.get("contract_version"),
            "registration_consumed": True,
            "registration_consumption": receipt,
        },
        "provenance": {
            "source_snapshot_id": source_snapshot_id,
            "checkpoint_path": str(resolved_checkpoint),
            "checkpoint_sha256": _sha256(resolved_checkpoint),
            "registration_consumed": True,
            "registration_consumption": receipt,
            "segy_trace_count": int(geometry.trace_count),
            "segy_samples_per_trace": int(geometry.samples_per_trace),
            "segy_profile_receipt": dict(segy_profile_receipt),
            "las_consumption_receipt": las_receipt,
            "trajectory_authority": "registration_v3_points",
        },
        "outputs": {
            "registration_manifest": str(candidate_product.manifest_path),
            "registration_points": str(candidate_product.points_path),
            "registration_preview": str(candidate_product.preview_path),
            "checkpoint": str(resolved_checkpoint),
            "diagnostics": str(diagnostics_path),
        },
        "diagnostics": diagnostics,
        "accepted_point_fraction": (
            float(np.mean(accepted)) if accepted.size else 0.0
        ),
        "modeled_well_count": len(modeled_wells),
        "fallback_well_count": len(fallback_by_well),
        "candidate_twt_mae_unavailable_without_labels": True,
    }


__all__ = ["MODEL_ID", "run_geopath_tie_v1"]
