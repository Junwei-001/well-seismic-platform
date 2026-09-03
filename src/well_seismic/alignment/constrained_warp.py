"""Fail-closed, physically constrained refinement of a sonic well tie.

The operator in this module is intentionally independent from the production
pipeline.  It refines a *physical* sonic time-depth prior; it does not infer an
absolute time-depth relation from seismic alone.  A global bulk/stretch grid is
followed by a windowed-correlation dynamic programme whose transitions enforce
a strictly monotone, locally bounded time warp.  Optional interval-velocity
constraints are then projected (or rejected) on the full sonic track.

Rejected results never expose a ``corrected_twt_ms`` track.  Their ranked
candidates and diagnostics remain available for offline calibration and QC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ConstrainedWarpConfig:
    """Numerical and acceptance limits for :func:`constrained_well_tie`."""

    window_ms: float = 160.0
    window_step_ms: float = 48.0
    max_bulk_shift_ms: float = 64.0
    max_global_stretch: float = 0.06
    global_stretch_step: float = 0.01
    max_local_shift_ms: float = 28.0
    local_lag_step_ms: float | None = None
    min_local_slope: float = 0.78
    max_local_slope: float = 1.22
    smoothness_penalty: float = 0.10
    local_shift_penalty: float = 0.015
    min_overlap_samples: int = 32
    min_valid_windows: int = 5
    min_window_correlation: float = 0.25
    min_usable_window_fraction: float = 0.55
    min_global_correlation: float = 0.30
    min_refined_correlation: float = 0.42
    max_refined_correlation_loss: float = 0.01
    cycle_skip_separation_ms: float = 24.0
    min_global_peak_margin: float = 0.012
    min_candidate_score_margin: float = 0.006
    min_local_peak_margin: float = 0.004
    candidates_to_refine: int = 18
    top_k: int = 5
    uncertainty_temperature: float = 0.025
    min_interval_velocity_m_s: float = 1200.0
    max_interval_velocity_m_s: float = 6500.0
    velocity_policy: Literal["project", "reject"] = "project"
    max_velocity_projection_ms: float = 24.0

    def validate(self) -> None:
        positive = {
            "window_ms": self.window_ms,
            "window_step_ms": self.window_step_ms,
            "max_bulk_shift_ms": self.max_bulk_shift_ms,
            "global_stretch_step": self.global_stretch_step,
            "max_local_shift_ms": self.max_local_shift_ms,
            "min_overlap_samples": self.min_overlap_samples,
            "min_valid_windows": self.min_valid_windows,
            "candidates_to_refine": self.candidates_to_refine,
            "top_k": self.top_k,
            "uncertainty_temperature": self.uncertainty_temperature,
            "min_interval_velocity_m_s": self.min_interval_velocity_m_s,
            "max_interval_velocity_m_s": self.max_interval_velocity_m_s,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0.0]
        if invalid:
            raise ValueError(f"constrained-warp values must be positive: {invalid}")
        if self.local_lag_step_ms is not None and self.local_lag_step_ms <= 0.0:
            raise ValueError("local_lag_step_ms must be positive when provided")
        if not 0.0 <= self.max_global_stretch < 0.5:
            raise ValueError("max_global_stretch must be in [0, 0.5)")
        if not 0.0 < self.min_local_slope <= self.max_local_slope:
            raise ValueError("local slope bounds must be positive and ordered")
        if not 0.0 <= self.min_usable_window_fraction <= 1.0:
            raise ValueError("min_usable_window_fraction must be in [0, 1]")
        for name in (
            "min_window_correlation",
            "min_global_correlation",
            "min_refined_correlation",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_interval_velocity_m_s >= self.max_interval_velocity_m_s:
            raise ValueError("interval-velocity bounds must be ordered")
        if self.velocity_policy not in {"project", "reject"}:
            raise ValueError("velocity_policy must be 'project' or 'reject'")


@dataclass(frozen=True)
class WarpCandidate:
    """One auditable constrained-warp posterior candidate."""

    score: float
    correlation: float
    global_correlation: float
    bulk_shift_ms: float
    global_scale: float
    polarity: int
    local_knot_time_ms: np.ndarray = field(repr=False)
    local_shift_ms: np.ndarray = field(repr=False)
    corrected_twt_ms: np.ndarray = field(repr=False)
    selected_window_correlation: np.ndarray = field(repr=False)
    usable_window_fraction: float
    local_peak_margin: float
    physical_valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 6),
            "correlation": round(float(self.correlation), 6),
            "global_correlation": round(float(self.global_correlation), 6),
            "bulk_shift_ms": round(float(self.bulk_shift_ms), 6),
            "global_scale": round(float(self.global_scale), 8),
            "polarity": int(self.polarity),
            "window_count": int(self.local_knot_time_ms.size),
            "usable_window_fraction": round(float(self.usable_window_fraction), 6),
            "local_peak_margin": round(float(self.local_peak_margin), 6),
            "physical_valid": bool(self.physical_valid),
            **self.diagnostics,
        }


@dataclass
class ConstrainedWarpResult:
    """Result contract; rejected ties deliberately have no consumable track."""

    accepted: bool
    status: str
    prior_twt_ms: np.ndarray = field(repr=False)
    corrected_twt_ms: np.ndarray | None = field(default=None, repr=False)
    twt_std_ms: np.ndarray | None = field(default=None, repr=False)
    candidates: tuple[WarpCandidate, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "status": self.status,
            "track_point_count": int(self.prior_twt_ms.size),
            "candidate_count": len(self.candidates),
            "uncertainty_median_ms": (
                None
                if self.twt_std_ms is None
                else round(float(np.median(self.twt_std_ms)), 6)
            ),
            "uncertainty_p90_ms": (
                None
                if self.twt_std_ms is None
                else round(float(np.quantile(self.twt_std_ms, 0.9)), 6)
            ),
            "candidates": [candidate.summary() for candidate in self.candidates],
            **self.diagnostics,
        }


@dataclass(frozen=True)
class _GlobalCandidate:
    score: float
    correlation: float
    bulk_shift_ms: float
    scale: float
    polarity: int
    segment_correlation: float
    coverage: float


def _correlation(
    left: np.ndarray,
    right: np.ndarray,
    minimum_samples: int,
) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < int(minimum_samples):
        return float("nan")
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _regular_values(start: float, stop: float, step: float) -> np.ndarray:
    count = max(1, int(np.floor((stop - start) / step + 0.5)))
    values = start + np.arange(count + 1, dtype=float) * step
    values = values[values <= stop + 0.25 * step]
    if values.size == 0 or values[-1] < stop - 0.25 * step:
        values = np.append(values, stop)
    return values


def _global_mapping(time_ms: np.ndarray, scale: float, bulk_ms: float) -> np.ndarray:
    center = 0.5 * float(time_ms[0] + time_ms[-1])
    return center + float(scale) * (time_ms - center) + float(bulk_ms)


def _warp_trace(
    synthetic: np.ndarray,
    source_time_ms: np.ndarray,
    target_time_for_source_ms: np.ndarray,
    observed_time_ms: np.ndarray,
    polarity: int = 1,
) -> np.ndarray:
    finite = (
        np.isfinite(synthetic)
        & np.isfinite(source_time_ms)
        & np.isfinite(target_time_for_source_ms)
    )
    if int(np.sum(finite)) < 2:
        return np.full(observed_time_ms.shape, np.nan, dtype=float)
    return np.interp(
        observed_time_ms,
        target_time_for_source_ms[finite],
        float(polarity) * synthetic[finite],
        left=np.nan,
        right=np.nan,
    )


def _segment_correlation(
    predicted: np.ndarray,
    observed: np.ndarray,
    minimum_samples: int,
) -> float:
    size = predicted.size
    correlations: list[float] = []
    for start, stop in zip(np.linspace(0, size, 6)[:-1], np.linspace(0, size, 6)[1:]):
        left = int(round(start))
        right = int(round(stop))
        value = _correlation(
            predicted[left:right],
            observed[left:right],
            max(8, minimum_samples // 4),
        )
        if np.isfinite(value):
            correlations.append(value)
    return float(np.median(correlations)) if correlations else float("nan")


def _enumerate_global_candidates(
    synthetic: np.ndarray,
    observed: np.ndarray,
    time_ms: np.ndarray,
    config: ConstrainedWarpConfig,
) -> tuple[list[_GlobalCandidate], _GlobalCandidate | None]:
    dt_ms = float(np.median(np.diff(time_ms)))
    bulk_values = _regular_values(
        -config.max_bulk_shift_ms,
        config.max_bulk_shift_ms,
        dt_ms,
    )
    stretch_offsets = _regular_values(
        -config.max_global_stretch,
        config.max_global_stretch,
        config.global_stretch_step,
    )
    candidates: list[_GlobalCandidate] = []
    bulk_only: _GlobalCandidate | None = None
    for offset in stretch_offsets:
        scale = 1.0 + float(offset)
        for bulk in bulk_values:
            mapping = _global_mapping(time_ms, scale, float(bulk))
            raw = _warp_trace(synthetic, time_ms, mapping, time_ms)
            signed = _correlation(raw, observed, config.min_overlap_samples)
            if not np.isfinite(signed):
                continue
            polarity = 1 if signed >= 0.0 else -1
            predicted = float(polarity) * raw
            absolute = abs(float(signed))
            segment = _segment_correlation(
                predicted,
                observed,
                config.min_overlap_samples,
            )
            segment = max(0.0, float(segment)) if np.isfinite(segment) else 0.0
            coverage = float(np.mean(np.isfinite(predicted) & np.isfinite(observed)))
            stretch_penalty = (
                0.012 * abs(offset) / config.max_global_stretch
                if config.max_global_stretch > 0.0
                else 0.0
            )
            score = 0.72 * absolute + 0.28 * segment - stretch_penalty
            candidate = _GlobalCandidate(
                score=float(score),
                correlation=absolute,
                bulk_shift_ms=float(bulk),
                scale=scale,
                polarity=polarity,
                segment_correlation=segment,
                coverage=coverage,
            )
            candidates.append(candidate)
            if abs(offset) <= 1e-10 and (
                bulk_only is None or candidate.score > bulk_only.score
            ):
                bulk_only = candidate
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates, bulk_only


def _global_distance_ms(
    left: _GlobalCandidate,
    right: _GlobalCandidate,
    time_ms: np.ndarray,
) -> float:
    difference = _global_mapping(time_ms, left.scale, left.bulk_shift_ms) - _global_mapping(
        time_ms, right.scale, right.bulk_shift_ms
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def _global_peak_margin(
    candidates: list[_GlobalCandidate],
    time_ms: np.ndarray,
    separation_ms: float,
) -> float:
    if not candidates:
        return 0.0
    best = candidates[0]
    competitor = next(
        (
            item
            for item in candidates[1:]
            if _global_distance_ms(best, item, time_ms) >= separation_ms
        ),
        None,
    )
    return float("inf") if competitor is None else float(best.score - competitor.score)


def _window_centers(time_ms: np.ndarray, config: ConstrainedWarpConfig) -> np.ndarray:
    half = 0.5 * config.window_ms
    start = float(time_ms[0] + half)
    stop = float(time_ms[-1] - half)
    if stop <= start:
        return np.empty(0, dtype=float)
    return _regular_values(start, stop, config.window_step_ms)


def _local_correlation_matrix(
    synthetic: np.ndarray,
    observed: np.ndarray,
    time_ms: np.ndarray,
    global_candidate: _GlobalCandidate,
    centers_ms: np.ndarray,
    residual_lags_ms: np.ndarray,
    config: ConstrainedWarpConfig,
) -> tuple[np.ndarray, np.ndarray]:
    correlations = np.full((centers_ms.size, residual_lags_ms.size), np.nan, dtype=float)
    usable_centers = np.zeros(centers_ms.size, dtype=bool)
    half = 0.5 * config.window_ms
    for row, center in enumerate(centers_ms):
        mask = np.abs(time_ms - center) <= half
        if int(np.sum(mask)) < config.min_overlap_samples:
            continue
        source_time = time_ms[mask]
        source_amplitude = float(global_candidate.polarity) * synthetic[mask]
        base_target = _global_mapping(
            source_time,
            global_candidate.scale,
            global_candidate.bulk_shift_ms,
        )
        for column, residual in enumerate(residual_lags_ms):
            target = base_target + float(residual)
            seismic = np.interp(target, time_ms, observed, left=np.nan, right=np.nan)
            correlations[row, column] = _correlation(
                source_amplitude,
                seismic,
                config.min_overlap_samples,
            )
        usable_centers[row] = bool(np.any(np.isfinite(correlations[row])))
    return correlations[usable_centers], centers_ms[usable_centers]


def _best_local_path(
    correlations: np.ndarray,
    centers_ms: np.ndarray,
    residual_lags_ms: np.ndarray,
    global_scale: float,
    config: ConstrainedWarpConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows, columns = correlations.shape
    if rows < config.min_valid_windows:
        return None
    lag_limit = max(config.max_local_shift_ms, 1e-9)
    unary = np.where(
        np.isfinite(correlations),
        correlations
        - config.local_shift_penalty * np.square(residual_lags_ms / lag_limit)[None, :],
        -np.inf,
    )
    score = np.full((rows, columns), -np.inf, dtype=float)
    parent = np.full((rows, columns), -1, dtype=int)
    score[0] = unary[0]
    for row in range(1, rows):
        delta_source = float(centers_ms[row] - centers_ms[row - 1])
        for column, current_lag in enumerate(residual_lags_ms):
            if not np.isfinite(unary[row, column]):
                continue
            slopes = global_scale + (
                float(current_lag) - residual_lags_ms
            ) / delta_source
            allowed = (
                (slopes >= config.min_local_slope)
                & (slopes <= config.max_local_slope)
                & np.isfinite(score[row - 1])
            )
            if not np.any(allowed):
                continue
            transition = score[row - 1].copy()
            transition -= config.smoothness_penalty * (
                np.abs(float(current_lag) - residual_lags_ms) / lag_limit
            )
            transition[~allowed] = -np.inf
            best_previous = int(np.argmax(transition))
            score[row, column] = unary[row, column] + transition[best_previous]
            parent[row, column] = best_previous
    last = int(np.argmax(score[-1]))
    if not np.isfinite(score[-1, last]):
        return None
    indices = np.empty(rows, dtype=int)
    indices[-1] = last
    for row in range(rows - 1, 0, -1):
        indices[row - 1] = parent[row, indices[row]]
        if indices[row - 1] < 0:
            return None
    return residual_lags_ms[indices], correlations[np.arange(rows), indices]


def _local_peak_margin(
    correlations: np.ndarray,
    selected_lags: np.ndarray,
    residual_lags_ms: np.ndarray,
    separation_ms: float,
) -> float:
    margins: list[float] = []
    for row, selected_lag in enumerate(selected_lags):
        selected_index = int(np.argmin(np.abs(residual_lags_ms - selected_lag)))
        selected = correlations[row, selected_index]
        alternatives = correlations[row].copy()
        alternatives[np.abs(residual_lags_ms - selected_lag) < separation_ms] = np.nan
        finite = alternatives[np.isfinite(alternatives)]
        if np.isfinite(selected) and finite.size:
            margins.append(float(selected - np.max(finite)))
    return float(np.median(margins)) if margins else 0.0


def _evaluate_mapping(
    query_twt_ms: np.ndarray,
    global_candidate: _GlobalCandidate,
    knot_time_ms: np.ndarray,
    knot_shift_ms: np.ndarray,
) -> np.ndarray:
    base = _global_mapping(
        query_twt_ms,
        global_candidate.scale,
        global_candidate.bulk_shift_ms,
    )
    if knot_time_ms.size == 0:
        return base
    residual = np.interp(
        query_twt_ms,
        knot_time_ms,
        knot_shift_ms,
        left=float(knot_shift_ms[0]),
        right=float(knot_shift_ms[-1]),
    )
    return base + residual


def _velocity_constraint(
    twt_ms: np.ndarray,
    prior_twt_ms: np.ndarray,
    depth_m: np.ndarray | None,
    config: ConstrainedWarpConfig,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    values = np.asarray(twt_ms, dtype=float)
    prior_dt = np.diff(np.asarray(prior_twt_ms, dtype=float))
    if depth_m is None:
        return values, True, {
            "velocity_constraint_applied": False,
            "velocity_projection_applied": False,
        }
    depth = np.asarray(depth_m, dtype=float)
    dz = np.diff(depth)
    dt = np.diff(values)
    velocity_minimum_dt = 2000.0 * dz / config.max_interval_velocity_m_s
    velocity_maximum_dt = 2000.0 * dz / config.min_interval_velocity_m_s
    # The physical projection is the intersection of interval-velocity and
    # warp-slope bounds.  Projecting against velocity alone could otherwise
    # invalidate the local-warp contract exposed to downstream consumers.
    minimum_dt = np.maximum(
        velocity_minimum_dt,
        config.min_local_slope * prior_dt,
    )
    maximum_dt = np.minimum(
        velocity_maximum_dt,
        config.max_local_slope * prior_dt,
    )
    if np.any(minimum_dt > maximum_dt):
        return values, False, {
            "velocity_constraint_applied": True,
            "velocity_policy": config.velocity_policy,
            "velocity_projection_applied": False,
            "velocity_rejection_reason": "velocity_and_slope_constraints_incompatible",
            "incompatible_constraint_interval_count": int(
                np.sum(minimum_dt > maximum_dt)
            ),
        }
    tolerance = np.maximum(1e-6, 1e-3 * minimum_dt)
    violations = (dt < minimum_dt - tolerance) | (dt > maximum_dt + tolerance)
    raw_velocity = np.divide(
        2000.0 * dz,
        dt,
        out=np.full(dt.shape, np.nan, dtype=float),
        where=dt > 0.0,
    )
    diagnostics: dict[str, Any] = {
        "velocity_constraint_applied": True,
        "velocity_policy": config.velocity_policy,
        "interval_velocity_min_m_s_before": float(np.nanmin(raw_velocity)),
        "interval_velocity_max_m_s_before": float(np.nanmax(raw_velocity)),
        "velocity_violation_count": int(np.sum(violations)),
        "velocity_violation_fraction": float(np.mean(violations)),
        "velocity_projection_applied": False,
    }
    if not np.any(violations):
        return values, True, diagnostics
    if config.velocity_policy == "reject":
        diagnostics["velocity_rejection_reason"] = "interval_velocity_out_of_bounds"
        return values, False, diagnostics
    projected_dt = np.clip(dt, minimum_dt, maximum_dt)
    projected = np.concatenate(([values[0]], values[0] + np.cumsum(projected_dt)))
    correction = projected - values
    max_correction = float(np.max(np.abs(correction)))
    projected_velocity = 2000.0 * dz / np.diff(projected)
    diagnostics.update(
        {
            "velocity_projection_applied": True,
            "velocity_projection_rms_ms": float(
                np.sqrt(np.mean(np.square(correction)))
            ),
            "velocity_projection_max_ms": max_correction,
            "interval_velocity_min_m_s_after": float(np.min(projected_velocity)),
            "interval_velocity_max_m_s_after": float(np.max(projected_velocity)),
        }
    )
    accepted = max_correction <= config.max_velocity_projection_ms
    if not accepted:
        diagnostics["velocity_rejection_reason"] = "projection_exceeds_limit"
    return projected, accepted, diagnostics


def _refine_candidate(
    global_candidate: _GlobalCandidate,
    synthetic: np.ndarray,
    observed: np.ndarray,
    time_ms: np.ndarray,
    prior_twt_ms: np.ndarray,
    depth_m: np.ndarray | None,
    config: ConstrainedWarpConfig,
) -> WarpCandidate | None:
    dt_ms = float(np.median(np.diff(time_ms)))
    lag_step = float(config.local_lag_step_ms or dt_ms)
    residual_lags = _regular_values(
        -config.max_local_shift_ms,
        config.max_local_shift_ms,
        lag_step,
    )
    correlations, centers = _local_correlation_matrix(
        synthetic,
        observed,
        time_ms,
        global_candidate,
        _window_centers(time_ms, config),
        residual_lags,
        config,
    )
    if correlations.shape[0] < config.min_valid_windows:
        return None
    path = _best_local_path(
        correlations,
        centers,
        residual_lags,
        global_candidate.scale,
        config,
    )
    if path is None:
        return None
    selected_lags, selected_correlation = path
    mapping = _evaluate_mapping(
        time_ms,
        global_candidate,
        centers,
        selected_lags,
    )
    increments = np.diff(mapping)
    local_slopes = increments / np.diff(time_ms)
    monotone = bool(np.all(increments > 0.0))
    slope_valid = bool(
        np.all(local_slopes >= config.min_local_slope - 1e-9)
        and np.all(local_slopes <= config.max_local_slope + 1e-9)
    )
    if not monotone or not slope_valid:
        return None
    predicted = _warp_trace(
        synthetic,
        time_ms,
        mapping,
        time_ms,
        global_candidate.polarity,
    )
    refined_correlation = _correlation(
        predicted,
        observed,
        config.min_overlap_samples,
    )
    if not np.isfinite(refined_correlation):
        return None
    refined_correlation = max(0.0, float(refined_correlation))
    usable_fraction = float(
        np.mean(selected_correlation >= config.min_window_correlation)
    )
    positive_window_correlation = selected_correlation[
        selected_correlation >= config.min_window_correlation
    ]
    window_quality = (
        float(np.median(positive_window_correlation))
        if positive_window_correlation.size
        else 0.0
    )
    roughness = (
        float(np.mean(np.abs(np.diff(selected_lags)))) / config.max_local_shift_ms
        if selected_lags.size > 1
        else 0.0
    )
    score = (
        0.58 * refined_correlation
        + 0.27 * window_quality
        + 0.15 * usable_fraction
        - 0.018 * roughness
        - 0.008
        * abs(global_candidate.scale - 1.0)
        / max(config.max_global_stretch, 1e-9)
    )
    corrected = _evaluate_mapping(
        prior_twt_ms,
        global_candidate,
        centers,
        selected_lags,
    )
    corrected, physical_valid, velocity_diagnostics = _velocity_constraint(
        corrected,
        prior_twt_ms,
        depth_m,
        config,
    )
    peak_margin = _local_peak_margin(
        correlations,
        selected_lags,
        residual_lags,
        config.cycle_skip_separation_ms,
    )
    return WarpCandidate(
        score=float(score),
        correlation=refined_correlation,
        global_correlation=global_candidate.correlation,
        bulk_shift_ms=global_candidate.bulk_shift_ms,
        global_scale=global_candidate.scale,
        polarity=global_candidate.polarity,
        local_knot_time_ms=centers,
        local_shift_ms=selected_lags,
        corrected_twt_ms=corrected,
        selected_window_correlation=selected_correlation,
        usable_window_fraction=usable_fraction,
        local_peak_margin=peak_margin,
        physical_valid=physical_valid,
        diagnostics={
            "window_correlation_median": float(np.median(selected_correlation)),
            "window_correlation_min": float(np.min(selected_correlation)),
            "local_slope_min": float(np.min(local_slopes)),
            "local_slope_max": float(np.max(local_slopes)),
            "local_shift_rms_ms": float(
                np.sqrt(np.mean(np.square(selected_lags)))
            ),
            "local_shift_roughness": roughness,
            **velocity_diagnostics,
        },
    )


def _candidate_distance_ms(left: WarpCandidate, right: WarpCandidate) -> float:
    difference = left.corrected_twt_ms - right.corrected_twt_ms
    return float(np.sqrt(np.mean(np.square(difference))))


def _rank_diverse_candidates(
    candidates: list[WarpCandidate],
    top_k: int,
    minimum_distance_ms: float,
) -> list[WarpCandidate]:
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    selected: list[WarpCandidate] = []
    for candidate in ranked:
        if not selected or all(
            _candidate_distance_ms(candidate, prior) >= minimum_distance_ms
            for prior in selected
        ):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    if not selected and ranked:
        selected.append(ranked[0])
    return selected


def _posterior_uncertainty(
    candidates: list[WarpCandidate],
    sample_interval_ms: float,
    temperature: float,
) -> np.ndarray:
    scores = np.asarray([item.score for item in candidates], dtype=float)
    weights = np.exp((scores - float(np.max(scores))) / temperature)
    weights /= float(np.sum(weights))
    tracks = np.stack([item.corrected_twt_ms for item in candidates], axis=0)
    mean = np.sum(weights[:, None] * tracks, axis=0)
    variance = np.sum(weights[:, None] * np.square(tracks - mean), axis=0)
    # One sample is an explicit resolution floor, not a claim of calibration.
    return np.sqrt(variance + float(sample_interval_ms) ** 2)


def constrained_well_tie(
    synthetic_trace: np.ndarray,
    seismic_trace: np.ndarray,
    seismic_time_ms: np.ndarray,
    sonic_twt_ms: np.ndarray,
    *,
    depth_m: np.ndarray | None = None,
    config: ConstrainedWarpConfig | None = None,
) -> ConstrainedWarpResult:
    """Refine a physical sonic TWT prior with a bounded seismic correlation warp.

    ``synthetic_trace`` and ``seismic_trace`` must share the regular
    ``seismic_time_ms`` sampling.  ``sonic_twt_ms`` is the full-resolution
    physical time-depth prior to be corrected and can have a different length.
    When ``depth_m`` is supplied, it must be paired one-to-one with that prior.
    """

    options = config or ConstrainedWarpConfig()
    options.validate()
    synthetic = np.asarray(synthetic_trace, dtype=float)
    observed = np.asarray(seismic_trace, dtype=float)
    time_ms = np.asarray(seismic_time_ms, dtype=float)
    prior = np.asarray(sonic_twt_ms, dtype=float)
    if synthetic.ndim != 1 or observed.ndim != 1 or time_ms.ndim != 1:
        raise ValueError("synthetic, seismic and seismic time must be one-dimensional")
    if not (synthetic.shape == observed.shape == time_ms.shape):
        raise ValueError("synthetic and seismic traces must match the seismic time axis")
    if time_ms.size < max(options.min_overlap_samples, 4):
        raise ValueError("seismic trace is too short for constrained warping")
    if np.any(~np.isfinite(time_ms)) or np.any(np.diff(time_ms) <= 0.0):
        raise ValueError("seismic time axis must be finite and strictly increasing")
    time_steps = np.diff(time_ms)
    dt_ms = float(np.median(time_steps))
    if float(np.max(np.abs(time_steps - dt_ms))) > max(1e-6, 0.05 * dt_ms):
        raise ValueError("seismic time axis must be regular within five percent")
    if prior.ndim != 1 or prior.size < 2:
        raise ValueError("sonic_twt_ms must contain at least two points")
    if np.any(~np.isfinite(prior)) or np.any(np.diff(prior) <= 0.0):
        raise ValueError("sonic TWT prior must be finite and strictly increasing")
    depth: np.ndarray | None = None
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=float)
        if depth.shape != prior.shape:
            raise ValueError("depth_m must match the full-resolution sonic TWT prior")
        if np.any(~np.isfinite(depth)) or np.any(np.diff(depth) <= 0.0):
            raise ValueError("depth_m must be finite and strictly increasing")

    globals_ranked, bulk_only = _enumerate_global_candidates(
        synthetic,
        observed,
        time_ms,
        options,
    )
    if not globals_ranked or bulk_only is None:
        return ConstrainedWarpResult(
            accepted=False,
            status="rejected_insufficient_correlation_support",
            prior_twt_ms=prior.copy(),
            diagnostics={
                "rejection_reasons": ["no_finite_global_candidate"],
                "config": asdict(options),
            },
        )
    global_margin = _global_peak_margin(
        globals_ranked,
        time_ms,
        options.cycle_skip_separation_ms,
    )
    refined: list[WarpCandidate] = []
    for global_candidate in globals_ranked[: options.candidates_to_refine]:
        candidate = _refine_candidate(
            global_candidate,
            synthetic,
            observed,
            time_ms,
            prior,
            depth,
            options,
        )
        if candidate is not None:
            refined.append(candidate)
    physically_valid = [item for item in refined if item.physical_valid]
    diverse = _rank_diverse_candidates(
        physically_valid,
        options.top_k,
        max(dt_ms, 0.5 * options.cycle_skip_separation_ms),
    )
    if not diverse:
        return ConstrainedWarpResult(
            accepted=False,
            status="rejected_no_physical_candidate",
            prior_twt_ms=prior.copy(),
            diagnostics={
                "rejection_reasons": ["all_refined_candidates_failed_physical_constraints"],
                "global_peak_margin": global_margin,
                "refined_candidate_count": len(refined),
                "config": asdict(options),
            },
        )

    best = diverse[0]
    second = next(
        (
            item
            for item in sorted(physically_valid, key=lambda value: value.score, reverse=True)[1:]
            if _candidate_distance_ms(best, item)
            >= options.cycle_skip_separation_ms
        ),
        None,
    )
    candidate_margin = (
        float("inf") if second is None else float(best.score - second.score)
    )
    rejection_reasons: list[str] = []
    if best.global_correlation < options.min_global_correlation:
        rejection_reasons.append("global_correlation_below_threshold")
    if best.correlation < options.min_refined_correlation:
        rejection_reasons.append("refined_correlation_below_threshold")
    if best.usable_window_fraction < options.min_usable_window_fraction:
        rejection_reasons.append("insufficient_usable_windows")
    if best.correlation < bulk_only.correlation - options.max_refined_correlation_loss:
        rejection_reasons.append("refinement_degrades_bulk_only_tie")
    if global_margin < options.min_global_peak_margin:
        rejection_reasons.append("ambiguous_global_cycle")
    if candidate_margin < options.min_candidate_score_margin:
        rejection_reasons.append("ambiguous_refined_cycle")
    if best.local_peak_margin < options.min_local_peak_margin:
        rejection_reasons.append("ambiguous_local_windows")
    if abs(best.bulk_shift_ms) >= options.max_bulk_shift_ms - 0.25 * dt_ms:
        rejection_reasons.append("bulk_shift_at_search_boundary")

    common_diagnostics = {
        "operator": "sonic_prior_constrained_correlation_warp_v1",
        "production_default_enabled": False,
        "bulk_only_shift_ms": bulk_only.bulk_shift_ms,
        "bulk_only_correlation": bulk_only.correlation,
        "bulk_only_score": bulk_only.score,
        "global_peak_margin": global_margin,
        "candidate_score_margin": candidate_margin,
        "global_candidate_count": len(globals_ranked),
        "refined_candidate_count": len(refined),
        "physical_candidate_count": len(physically_valid),
        "selected_local_slope_bounds": [
            best.diagnostics["local_slope_min"],
            best.diagnostics["local_slope_max"],
        ],
        "configured_local_slope_bounds": [
            options.min_local_slope,
            options.max_local_slope,
        ],
        "rejection_reasons": rejection_reasons,
        "uncertainty_status": "candidate_spread_with_one_sample_floor_not_calibrated",
        "config": asdict(options),
    }
    if rejection_reasons:
        return ConstrainedWarpResult(
            accepted=False,
            status="rejected_fail_closed",
            prior_twt_ms=prior.copy(),
            candidates=tuple(diverse),
            diagnostics=common_diagnostics,
        )
    uncertainty = _posterior_uncertainty(
        diverse,
        dt_ms,
        options.uncertainty_temperature,
    )
    return ConstrainedWarpResult(
        accepted=True,
        status="accepted_constrained_warp_candidate",
        prior_twt_ms=prior.copy(),
        corrected_twt_ms=best.corrected_twt_ms.copy(),
        twt_std_ms=uncertainty,
        candidates=tuple(diverse),
        diagnostics=common_diagnostics,
    )


__all__ = [
    "ConstrainedWarpConfig",
    "ConstrainedWarpResult",
    "WarpCandidate",
    "constrained_well_tie",
]
