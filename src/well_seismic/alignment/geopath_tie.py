"""Trajectory-aware primitives for one unified vertical/deviated/horizontal tie.

The functions in this module deliberately operate on NumPy arrays and have no
training or SEG-Y dependency.  They form the deterministic geophysical part of
GeoPathTie: trajectory geometry, a local seismic tube, crossing-angle evidence,
shift likelihoods, and a k-best path through those likelihoods.

Positive candidate shift means that a synthetic event is moved to a later TWT.
Low-crossing-angle samples do not contribute reflectivity likelihood.  At those
samples the path is controlled by an interpolated anchor (when bracketed by
anchors) or by the physical prior.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeoPathTieConfig:
    """Numerical controls shared by scoring and path selection."""

    min_kappa: float = 0.18
    min_overlap_samples: int = 12
    max_shift_step_ms: float = 20.0
    smoothness_penalty: float = 0.025
    prior_penalty: float = 0.006
    anchor_penalty: float = 0.08
    top_k_paths: int = 5
    posterior_temperature: float = 0.25

    def validate(self) -> None:
        if not 0.0 <= self.min_kappa <= 1.0:
            raise ValueError("min_kappa must be in [0, 1]")
        if self.min_overlap_samples < 2:
            raise ValueError("min_overlap_samples must be at least 2")
        if self.max_shift_step_ms <= 0.0:
            raise ValueError("max_shift_step_ms must be positive")
        if (
            min(
                self.smoothness_penalty,
                self.prior_penalty,
                self.anchor_penalty,
            )
            < 0.0
        ):
            raise ValueError("path penalties cannot be negative")
        if self.top_k_paths < 1:
            raise ValueError("top_k_paths must be positive")
        if self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive")


@dataclass(frozen=True)
class TrajectoryGeometry:
    """Geometry sampled on monotonically increasing measured depth."""

    md_m: np.ndarray = field(repr=False)
    xyz_m: np.ndarray = field(repr=False)
    tangent_xyz: np.ndarray = field(repr=False)
    inclination_deg: np.ndarray = field(repr=False)
    dogleg_deg_per_30m: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SeismicTubeFeatures:
    """Trace-level attributes sampled along the well trajectory."""

    center: np.ndarray = field(repr=False)
    robust_mean: np.ndarray = field(repr=False)
    median: np.ndarray = field(repr=False)
    std: np.ndarray = field(repr=False)
    coherence: np.ndarray = field(repr=False)
    phase_rad: np.ndarray = field(repr=False)
    dip_x_ms_per_m: np.ndarray = field(repr=False)
    dip_y_ms_per_m: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    trace_count: int
    feature_source: str = "gathered_neighbor_traces"


@dataclass(frozen=True)
class ShiftScoreResult:
    """Per-station candidate likelihoods before path regularisation."""

    candidate_shift_ms: np.ndarray = field(repr=False)
    correlation: np.ndarray = field(repr=False)
    score: np.ndarray = field(repr=False)
    seismic_evidence_mask: np.ndarray = field(repr=False)
    overlap_samples: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class GeoPathCandidate:
    """One complete and auditable shift path."""

    score: float
    candidate_index: np.ndarray = field(repr=False)
    shift_ms: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class GeoPathResult:
    """K-best solution of the trajectory-wide tie."""

    accepted: bool
    status: str
    selected_shift_ms: np.ndarray | None = field(default=None, repr=False)
    selected_candidate_index: np.ndarray | None = field(default=None, repr=False)
    posterior_std_ms: np.ndarray | None = field(default=None, repr=False)
    evidence_source: np.ndarray | None = field(default=None, repr=False)
    candidates: tuple[GeoPathCandidate, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _as_float_array(values: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return result


def trajectory_geometry(md_m: np.ndarray, xyz_m: np.ndarray) -> TrajectoryGeometry:
    """Derive tangent, inclination and dogleg from one common trajectory.

    ``xyz_m[:, 2]`` is true vertical depth in metres, positive downward.  The
    inclination convention is zero for a vertical well and 90 degrees for a
    horizontal well.  Dogleg is reported in degrees per 30 metres.
    """

    md = _as_float_array(md_m, ndim=1, name="md_m")
    xyz = _as_float_array(xyz_m, ndim=2, name="xyz_m")
    if xyz.shape != (md.size, 3):
        raise ValueError("xyz_m must have shape (len(md_m), 3)")
    if md.size < 2:
        raise ValueError("trajectory needs at least two stations")
    if not np.all(np.isfinite(md)) or not np.all(np.diff(md) > 0.0):
        raise ValueError("md_m must be finite and strictly increasing")

    valid = np.all(np.isfinite(xyz), axis=1)
    tangent = np.full_like(xyz, np.nan, dtype=float)
    if int(np.sum(valid)) >= 2:
        for column in range(3):
            finite_values = np.interp(md, md[valid], xyz[valid, column])
            tangent[:, column] = np.gradient(finite_values, md, edge_order=1)
        norms = np.linalg.norm(tangent, axis=1)
        usable = norms > 1e-12
        tangent[usable] /= norms[usable, None]
        tangent[~usable] = np.nan
        valid &= usable

    inclination = np.full(md.shape, np.nan, dtype=float)
    inclination[valid] = np.degrees(
        np.arccos(np.clip(np.abs(tangent[valid, 2]), 0.0, 1.0))
    )
    dogleg = np.full(md.shape, np.nan, dtype=float)
    dogleg[0] = 0.0 if valid[0] else np.nan
    for index in range(1, md.size):
        if not (valid[index - 1] and valid[index]):
            continue
        angle_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(tangent[index - 1], tangent[index]),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        dogleg[index] = angle_deg * 30.0 / float(md[index] - md[index - 1])

    return TrajectoryGeometry(
        md_m=md.copy(),
        xyz_m=xyz.copy(),
        tangent_xyz=tangent,
        inclination_deg=inclination,
        dogleg_deg_per_30m=dogleg,
        valid_mask=valid,
    )


def _analytic_phase(traces: np.ndarray) -> np.ndarray:
    size = traces.shape[-1]
    spectrum = np.fft.fft(np.nan_to_num(traces, nan=0.0), axis=-1)
    multiplier = np.zeros(size, dtype=float)
    multiplier[0] = 1.0
    if size % 2 == 0:
        multiplier[1 : size // 2] = 2.0
        multiplier[size // 2] = 1.0
    else:
        multiplier[1 : (size + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier, axis=-1)
    phase = np.angle(analytic)
    phase[~np.isfinite(traces)] = np.nan
    return phase


def _trimmed_mean(values: np.ndarray, fraction: float) -> np.ndarray:
    station_count, _, sample_count = values.shape
    result = np.full((station_count, sample_count), np.nan, dtype=float)
    for station in range(station_count):
        for sample in range(sample_count):
            finite = np.sort(
                values[station, :, sample][np.isfinite(values[station, :, sample])]
            )
            if finite.size == 0:
                continue
            trim = min(int(np.floor(fraction * finite.size)), (finite.size - 1) // 2)
            result[station, sample] = float(np.mean(finite[trim : finite.size - trim]))
    return result


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(valid)) < 3:
        return float("nan")
    a = left[valid] - float(np.mean(left[valid]))
    b = right[valid] - float(np.mean(right[valid]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _estimate_tube_dip(
    traces: np.ndarray,
    offsets_xy_m: np.ndarray | None,
    sample_interval_ms: float,
    center_indices: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    station_count, trace_count, _ = traces.shape
    dip_x = np.full(station_count, np.nan, dtype=float)
    dip_y = np.full(station_count, np.nan, dtype=float)
    if offsets_xy_m is None or trace_count < 3:
        return dip_x, dip_y
    offsets = np.asarray(offsets_xy_m, dtype=float)
    if offsets.shape == (trace_count, 2):
        offsets = np.broadcast_to(offsets[None, :, :], (station_count, trace_count, 2))
    if offsets.shape != (station_count, trace_count, 2):
        raise ValueError("trace_offsets_xy_m must have shape (P,2) or (N,P,2)")

    for station in range(station_count):
        center_index = int(center_indices[station])
        reference = traces[station, center_index]
        relative_offsets = offsets[station] - offsets[station, center_index]
        rows: list[np.ndarray] = []
        lags_ms: list[float] = []
        weights: list[float] = []
        for trace_index in range(trace_count):
            if trace_index == center_index:
                continue
            candidate = traces[station, trace_index]
            best_lag = 0
            best_correlation = -np.inf
            for lag in range(-max_lag_samples, max_lag_samples + 1):
                shifted = _shift_samples(reference, lag)
                correlation = _normalized_correlation(shifted, candidate)
                if np.isfinite(correlation) and correlation > best_correlation:
                    best_lag = lag
                    best_correlation = correlation
            if not np.isfinite(best_correlation) or best_correlation <= 0.0:
                continue
            rows.append(relative_offsets[trace_index])
            lags_ms.append(float(best_lag) * sample_interval_ms)
            weights.append(best_correlation**2)
        if len(rows) < 2:
            continue
        design = np.asarray(rows, dtype=float)
        response = np.asarray(lags_ms, dtype=float)
        weight = np.sqrt(np.asarray(weights, dtype=float))
        weighted_design = design * weight[:, None]
        if np.linalg.matrix_rank(weighted_design) < 2:
            continue
        solution, *_ = np.linalg.lstsq(weighted_design, response * weight, rcond=None)
        dip_x[station], dip_y[station] = solution
    return dip_x, dip_y


def extract_seismic_tube_features(
    seismic_tube: np.ndarray,
    *,
    sample_interval_ms: float,
    trace_offsets_xy_m: np.ndarray | None = None,
    center_trace_index: int | np.ndarray | None = None,
    robust_trim_fraction: float = 0.1,
    dip_max_lag_ms: float = 24.0,
) -> SeismicTubeFeatures:
    """Summarise a local 3-D seismic tube sampled along a trajectory.

    Accepted input shapes are ``(station, trace, time)`` and
    ``(station, y, x, time)``.  The latter is flattened over ``(y, x)``; the
    supplied XY offsets must follow that same row-major order.
    """

    tube = np.asarray(seismic_tube, dtype=float)
    if tube.ndim == 4:
        tube = tube.reshape(tube.shape[0], -1, tube.shape[-1])
    if tube.ndim != 3:
        raise ValueError("seismic_tube must have shape (N,P,T) or (N,Y,X,T)")
    station_count, trace_count, sample_count = tube.shape
    if station_count < 1 or trace_count < 1 or sample_count < 4:
        raise ValueError("seismic_tube is too small")
    if sample_interval_ms <= 0.0:
        raise ValueError("sample_interval_ms must be positive")
    if not 0.0 <= robust_trim_fraction < 0.5:
        raise ValueError("robust_trim_fraction must be in [0, 0.5)")
    if center_trace_index is None:
        if trace_offsets_xy_m is not None:
            offsets = np.asarray(trace_offsets_xy_m, dtype=float)
            if offsets.shape == (trace_count, 2):
                center_indices = np.full(
                    station_count,
                    int(np.argmin(np.linalg.norm(offsets, axis=1))),
                    dtype=int,
                )
            elif offsets.shape == (station_count, trace_count, 2):
                center_indices = np.argmin(np.linalg.norm(offsets, axis=2), axis=1)
            else:
                raise ValueError("trace_offsets_xy_m must have shape (P,2) or (N,P,2)")
        else:
            center_indices = np.full(station_count, trace_count // 2, dtype=int)
    else:
        supplied_center = np.asarray(center_trace_index, dtype=int)
        if supplied_center.ndim == 0:
            center_indices = np.full(station_count, int(supplied_center), dtype=int)
        elif supplied_center.shape == (station_count,):
            center_indices = supplied_center.copy()
        else:
            raise ValueError("center_trace_index must be a scalar or have shape (N,)")
    if np.any((center_indices < 0) | (center_indices >= trace_count)):
        raise ValueError("center_trace_index is outside the tube")

    finite_count = np.sum(np.isfinite(tube), axis=1)
    trace_sum = np.nansum(tube, axis=1)
    energy_sum = np.nansum(np.square(tube), axis=1)
    denominator = finite_count * energy_sum
    coherence = np.full((station_count, sample_count), np.nan, dtype=float)
    usable = (finite_count >= 2) & (denominator > 1e-12)
    coherence[usable] = np.square(trace_sum[usable]) / denominator[usable]
    coherence[usable] = np.clip(coherence[usable], 0.0, 1.0)

    robust_mean = _trimmed_mean(tube, robust_trim_fraction)
    # A physical aperture can legitimately have no trace at one sample.  NaN
    # is the intended missing-evidence value and valid_mask carries that state.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(tube, axis=1)
        std = np.nanstd(tube, axis=1)
    valid = finite_count >= min(2, trace_count)
    center = tube[np.arange(station_count), center_indices, :].copy()
    phase = _analytic_phase(robust_mean)
    max_lag_samples = max(1, round(dip_max_lag_ms / sample_interval_ms))
    dip_x, dip_y = _estimate_tube_dip(
        tube,
        trace_offsets_xy_m,
        sample_interval_ms,
        center_indices,
        max_lag_samples,
    )
    return SeismicTubeFeatures(
        center=center,
        robust_mean=robust_mean,
        median=median,
        std=std,
        coherence=coherence,
        phase_rad=phase,
        dip_x_ms_per_m=dip_x,
        dip_y_ms_per_m=dip_y,
        valid_mask=valid,
        trace_count=trace_count,
        feature_source="gathered_neighbor_traces",
    )


def seismic_tube_features_from_cached_channels(
    trajectory_seismic: np.ndarray,
) -> SeismicTubeFeatures:
    """Adapt the existing Chengdu four-channel cache without rereading SEG-Y.

    The input contract is ``(4, station, time)`` with channels
    ``nearest_trace, distance_weighted_mean, weighted_std,
    nearest_minus_mean``.  Median and coherence are conservative proxies in
    this fast path; local dip is deliberately left unknown.  Raw neighbor
    gathers should use :func:`extract_seismic_tube_features` for final runs.
    """

    cached = np.asarray(trajectory_seismic, dtype=float)
    if cached.ndim != 3 or cached.shape[0] != 4:
        raise ValueError("trajectory_seismic must have shape (4,N,T)")
    center = cached[0].copy()
    weighted_mean = cached[1].copy()
    weighted_std = np.abs(cached[2])
    finite = (
        np.isfinite(center) & np.isfinite(weighted_mean) & np.isfinite(weighted_std)
    )
    amplitude_scale = np.maximum(np.abs(weighted_mean), 1e-6)
    coherence = 1.0 / (1.0 + np.square(weighted_std / amplitude_scale))
    coherence[~finite] = np.nan
    station_count = cached.shape[1]
    return SeismicTubeFeatures(
        center=center,
        robust_mean=weighted_mean,
        median=weighted_mean.copy(),
        std=weighted_std,
        coherence=coherence,
        phase_rad=_analytic_phase(weighted_mean),
        dip_x_ms_per_m=np.full(station_count, np.nan, dtype=float),
        dip_y_ms_per_m=np.full(station_count, np.nan, dtype=float),
        valid_mask=finite,
        trace_count=1,
        feature_source="chengdu_cached_four_channel_proxy",
    )


def bedding_normals_from_time_dip(
    dip_x_ms_per_m: np.ndarray,
    dip_y_ms_per_m: np.ndarray,
    *,
    vertical_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Convert TWT spatial gradients to unit normals in XYZ/TVD coordinates."""

    dip_x = _as_float_array(dip_x_ms_per_m, ndim=1, name="dip_x_ms_per_m")
    dip_y = _as_float_array(dip_y_ms_per_m, ndim=1, name="dip_y_ms_per_m")
    if dip_x.shape != dip_y.shape:
        raise ValueError("dip_x_ms_per_m and dip_y_ms_per_m must have equal shape")
    velocity = np.asarray(vertical_velocity_m_s, dtype=float)
    velocity = np.broadcast_to(velocity, dip_x.shape)
    if np.any(np.isfinite(velocity) & (velocity <= 0.0)):
        raise ValueError("vertical_velocity_m_s must be positive")
    scale = velocity / 2000.0
    normals = np.column_stack((-scale * dip_x, -scale * dip_y, np.ones(dip_x.size)))
    finite = np.all(np.isfinite(normals), axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid = finite & (norms > 1e-12)
    normals[valid] /= norms[valid, None]
    normals[~valid] = np.nan
    return normals


def crossing_coefficient(
    tangent_xyz: np.ndarray, bedding_normal_xyz: np.ndarray
) -> np.ndarray:
    """Return ``abs(tangent dot bedding_normal)`` in the interval [0, 1]."""

    tangent = _as_float_array(tangent_xyz, ndim=2, name="tangent_xyz")
    normal = _as_float_array(bedding_normal_xyz, ndim=2, name="bedding_normal_xyz")
    if tangent.shape != normal.shape or tangent.shape[1] != 3:
        raise ValueError("tangent and bedding normal must both have shape (N, 3)")
    tangent_norm = np.linalg.norm(tangent, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    valid = (
        np.all(np.isfinite(tangent), axis=1)
        & np.all(np.isfinite(normal), axis=1)
        & (tangent_norm > 1e-12)
        & (normal_norm > 1e-12)
    )
    kappa = np.full(tangent.shape[0], np.nan, dtype=float)
    numerator = np.sum(tangent[valid] * normal[valid], axis=1)
    kappa[valid] = np.clip(
        np.abs(numerator / (tangent_norm[valid] * normal_norm[valid])),
        0.0,
        1.0,
    )
    return kappa


def _shift_samples(trace: np.ndarray, lag_samples: float) -> np.ndarray:
    samples = np.arange(trace.size, dtype=float)
    return np.interp(samples - lag_samples, samples, trace, left=np.nan, right=np.nan)


def score_shift_candidates(
    synthetic_windows: np.ndarray,
    observed_windows: np.ndarray,
    candidate_shift_ms: np.ndarray,
    *,
    sample_interval_ms: float,
    kappa: np.ndarray | None = None,
    min_kappa: float = 0.18,
    min_overlap_samples: int = 12,
) -> ShiftScoreResult:
    """Score seismic shifts while suppressing along-MD evidence in bed-parallel zones."""

    synthetic = np.asarray(synthetic_windows, dtype=float)
    observed = np.asarray(observed_windows, dtype=float)
    if synthetic.ndim == 1:
        synthetic = synthetic[None, :]
    if observed.ndim == 1:
        observed = observed[None, :]
    if synthetic.shape != observed.shape or synthetic.ndim != 2:
        raise ValueError("synthetic and observed windows must have equal shape (N,T)")
    if sample_interval_ms <= 0.0:
        raise ValueError("sample_interval_ms must be positive")
    if min_overlap_samples < 2:
        raise ValueError("min_overlap_samples must be at least 2")
    station_count = synthetic.shape[0]
    shifts = np.asarray(candidate_shift_ms, dtype=float)
    if shifts.ndim == 1:
        shifts = np.broadcast_to(shifts[None, :], (station_count, shifts.size)).copy()
    if shifts.ndim != 2 or shifts.shape[0] != station_count:
        raise ValueError("candidate_shift_ms must have shape (K,) or (N,K)")
    if shifts.shape[1] < 1 or not np.all(np.isfinite(shifts)):
        raise ValueError("candidate shifts must be non-empty and finite")

    if kappa is None:
        kappa_values = np.ones(station_count, dtype=float)
    else:
        kappa_values = _as_float_array(kappa, ndim=1, name="kappa")
        if kappa_values.shape != (station_count,):
            raise ValueError("kappa must have shape (N,)")
    evidence = np.isfinite(kappa_values) & (kappa_values >= min_kappa)
    correlation = np.full(shifts.shape, np.nan, dtype=float)
    overlap = np.zeros(shifts.shape, dtype=int)
    for station in range(station_count):
        if not evidence[station]:
            continue
        for candidate_index, shift_ms in enumerate(shifts[station]):
            shifted = _shift_samples(
                synthetic[station],
                shift_ms / sample_interval_ms,
            )
            valid = np.isfinite(shifted) & np.isfinite(observed[station])
            overlap[station, candidate_index] = int(np.sum(valid))
            if overlap[station, candidate_index] < min_overlap_samples:
                continue
            correlation[station, candidate_index] = _normalized_correlation(
                shifted[valid],
                observed[station, valid],
            )
    score = np.where(np.isfinite(correlation), correlation, 0.0)
    score[~evidence] = 0.0
    return ShiftScoreResult(
        candidate_shift_ms=shifts,
        correlation=correlation,
        score=score,
        seismic_evidence_mask=evidence,
        overlap_samples=overlap,
    )


def _bracketed_anchor_fallback(
    coordinate: np.ndarray,
    prior: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(anchor)
    fallback = prior.copy()
    from_anchor = np.zeros(prior.size, dtype=bool)
    if int(np.sum(finite)) < 2:
        fallback[finite] = anchor[finite]
        from_anchor[finite] = True
        return fallback, from_anchor
    lower = float(np.min(coordinate[finite]))
    upper = float(np.max(coordinate[finite]))
    bracketed = (coordinate >= lower) & (coordinate <= upper)
    fallback[bracketed] = np.interp(
        coordinate[bracketed], coordinate[finite], anchor[finite]
    )
    from_anchor[bracketed] = True
    return fallback, from_anchor


def solve_geopath_path(
    local_score: np.ndarray,
    candidate_shift_ms: np.ndarray,
    *,
    kappa: np.ndarray,
    prior_shift_ms: np.ndarray | None = None,
    anchor_shift_ms: np.ndarray | None = None,
    coordinate_m: np.ndarray | None = None,
    config: GeoPathTieConfig | None = None,
) -> GeoPathResult:
    """Select a trajectory-consistent K-best path through shift candidates."""

    options = config or GeoPathTieConfig()
    options.validate()
    scores = _as_float_array(local_score, ndim=2, name="local_score")
    station_count, candidate_count = scores.shape
    shifts = np.asarray(candidate_shift_ms, dtype=float)
    if shifts.ndim == 1:
        shifts = np.broadcast_to(shifts[None, :], scores.shape).copy()
    if shifts.shape != scores.shape or not np.all(np.isfinite(shifts)):
        raise ValueError("candidate_shift_ms must be finite with shape (K,) or (N,K)")
    kappa_values = _as_float_array(kappa, ndim=1, name="kappa")
    if kappa_values.shape != (station_count,):
        raise ValueError("kappa must have shape (N,)")
    coordinate = (
        np.arange(station_count, dtype=float)
        if coordinate_m is None
        else _as_float_array(coordinate_m, ndim=1, name="coordinate_m")
    )
    if coordinate.shape != (station_count,) or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("coordinate_m must be strictly increasing with shape (N,)")
    prior = (
        np.zeros(station_count, dtype=float)
        if prior_shift_ms is None
        else _as_float_array(prior_shift_ms, ndim=1, name="prior_shift_ms")
    )
    if prior.shape != (station_count,) or not np.all(np.isfinite(prior)):
        raise ValueError("prior_shift_ms must be finite with shape (N,)")
    anchor = (
        np.full(station_count, np.nan, dtype=float)
        if anchor_shift_ms is None
        else _as_float_array(anchor_shift_ms, ndim=1, name="anchor_shift_ms")
    )
    if anchor.shape != (station_count,):
        raise ValueError("anchor_shift_ms must have shape (N,)")

    evidence = np.isfinite(kappa_values) & (kappa_values >= options.min_kappa)
    fallback, fallback_from_anchor = _bracketed_anchor_fallback(
        coordinate, prior, anchor
    )
    unary = np.zeros_like(scores, dtype=float)
    shift_scale = max(
        1.0,
        float(np.median(np.abs(np.diff(np.sort(np.unique(shifts))))))
        if np.unique(shifts).size > 1
        else 1.0,
    )
    for station in range(station_count):
        delta_prior = np.abs(shifts[station] - prior[station]) / shift_scale
        if evidence[station]:
            usable_score = np.where(np.isfinite(scores[station]), scores[station], -1e6)
            unary[station] = usable_score - options.prior_penalty * np.square(
                delta_prior
            )
        else:
            delta_fallback = np.abs(shifts[station] - fallback[station]) / shift_scale
            unary[station] = -options.anchor_penalty * np.square(delta_fallback)
        if np.isfinite(anchor[station]):
            delta_anchor = np.abs(shifts[station] - anchor[station]) / shift_scale
            unary[station] -= options.anchor_penalty * np.square(delta_anchor)

    rank_count = min(options.top_k_paths, max(1, candidate_count * options.top_k_paths))
    value = np.full((station_count, candidate_count, rank_count), -np.inf, dtype=float)
    back_state = np.full((station_count, candidate_count, rank_count), -1, dtype=int)
    back_rank = np.full((station_count, candidate_count, rank_count), -1, dtype=int)
    value[0, :, 0] = unary[0]
    for station in range(1, station_count):
        for current in range(candidate_count):
            transitions: list[tuple[float, int, int]] = []
            for previous in range(candidate_count):
                delta_ms = abs(shifts[station, current] - shifts[station - 1, previous])
                if delta_ms > options.max_shift_step_ms + 1e-9:
                    continue
                penalty = options.smoothness_penalty * np.square(delta_ms / shift_scale)
                for previous_rank in range(rank_count):
                    previous_value = value[station - 1, previous, previous_rank]
                    if np.isfinite(previous_value):
                        transitions.append(
                            (
                                float(
                                    previous_value + unary[station, current] - penalty
                                ),
                                previous,
                                previous_rank,
                            )
                        )
            transitions.sort(key=lambda item: item[0], reverse=True)
            for rank, (path_value, previous, previous_rank) in enumerate(
                transitions[:rank_count]
            ):
                value[station, current, rank] = path_value
                back_state[station, current, rank] = previous
                back_rank[station, current, rank] = previous_rank

    endings: list[tuple[float, int, int]] = []
    for state in range(candidate_count):
        for rank in range(rank_count):
            if np.isfinite(value[-1, state, rank]):
                endings.append((float(value[-1, state, rank]), state, rank))
    endings.sort(key=lambda item: item[0], reverse=True)
    if not endings:
        return GeoPathResult(
            accepted=False,
            status="no_consistent_path",
            diagnostics={
                "station_count": station_count,
                "candidate_count": candidate_count,
                "seismic_evidence_fraction": float(np.mean(evidence)),
            },
        )

    paths: list[GeoPathCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for path_score, state, rank in endings:
        indices = np.empty(station_count, dtype=int)
        indices[-1] = state
        current_state, current_rank = state, rank
        possible = True
        for station in range(station_count - 1, 0, -1):
            previous_state = back_state[station, current_state, current_rank]
            previous_rank = back_rank[station, current_state, current_rank]
            if previous_state < 0 or previous_rank < 0:
                possible = False
                break
            indices[station - 1] = previous_state
            current_state, current_rank = previous_state, previous_rank
        key = tuple(int(index) for index in indices)
        if not possible or key in seen:
            continue
        seen.add(key)
        path_shifts = shifts[np.arange(station_count), indices]
        paths.append(
            GeoPathCandidate(
                score=path_score,
                candidate_index=indices,
                shift_ms=path_shifts,
            )
        )
        if len(paths) >= options.top_k_paths:
            break

    best = paths[0]
    path_scores = np.asarray([path.score for path in paths], dtype=float)
    path_scores = (path_scores - np.max(path_scores)) / options.posterior_temperature
    weights = np.exp(np.clip(path_scores, -700.0, 0.0))
    weights /= np.sum(weights)
    path_shift_matrix = np.vstack([path.shift_ms for path in paths])
    mean_shift = np.sum(weights[:, None] * path_shift_matrix, axis=0)
    posterior_std = np.sqrt(
        np.sum(weights[:, None] * np.square(path_shift_matrix - mean_shift), axis=0)
    )
    source = np.full(station_count, "seismic_reflectivity", dtype=object)
    source[~evidence & fallback_from_anchor] = "anchor_interpolation"
    source[~evidence & ~fallback_from_anchor] = "physical_prior"
    return GeoPathResult(
        accepted=True,
        status="accepted",
        selected_shift_ms=best.shift_ms.copy(),
        selected_candidate_index=best.candidate_index.copy(),
        posterior_std_ms=posterior_std,
        evidence_source=source,
        candidates=tuple(paths),
        diagnostics={
            "station_count": station_count,
            "candidate_count": candidate_count,
            "seismic_evidence_fraction": float(np.mean(evidence)),
            "anchor_fallback_count": int(np.sum(~evidence & fallback_from_anchor)),
            "physical_prior_fallback_count": int(
                np.sum(~evidence & ~fallback_from_anchor)
            ),
            "best_path_score": float(best.score),
        },
    )


def geopath_tie(
    synthetic_windows: np.ndarray,
    observed_windows: np.ndarray,
    candidate_shift_ms: np.ndarray,
    *,
    sample_interval_ms: float,
    kappa: np.ndarray,
    prior_shift_ms: np.ndarray | None = None,
    anchor_shift_ms: np.ndarray | None = None,
    coordinate_m: np.ndarray | None = None,
    config: GeoPathTieConfig | None = None,
) -> tuple[ShiftScoreResult, GeoPathResult]:
    """Score and solve one common tie for any well geometry."""

    options = config or GeoPathTieConfig()
    options.validate()
    score = score_shift_candidates(
        synthetic_windows,
        observed_windows,
        candidate_shift_ms,
        sample_interval_ms=sample_interval_ms,
        kappa=kappa,
        min_kappa=options.min_kappa,
        min_overlap_samples=options.min_overlap_samples,
    )
    path = solve_geopath_path(
        score.score,
        score.candidate_shift_ms,
        kappa=kappa,
        prior_shift_ms=prior_shift_ms,
        anchor_shift_ms=anchor_shift_ms,
        coordinate_m=coordinate_m,
        config=options,
    )
    return score, path
