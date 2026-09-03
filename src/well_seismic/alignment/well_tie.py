from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..depth_time import DepthTimeTransform, SonicIntegratedTimeDepthTransform


@dataclass
class TimeDomainAlignment:
    """一次可审计的时间域垂向对齐结果。"""

    transform: DepthTimeTransform
    status: str
    method: str
    confidence: float
    uncertainty_ms: float | None
    training_eligible: bool
    depth_domain: str = "tvd"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    synthetic_trace: np.ndarray | None = field(default=None, repr=False)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "confidence": round(float(self.confidence), 6),
            "uncertainty_ms": None if self.uncertainty_ms is None else round(float(self.uncertainty_ms), 6),
            "training_eligible": bool(self.training_eligible),
            "depth_domain": self.depth_domain,
            **self.diagnostics,
        }


@dataclass(frozen=True)
class StaticShiftAssessment:
    """Correlation evidence for one bounded bulk-static search.

    ``alternate_*`` describes the strongest peak outside the exclusion zone
    around the winning lobe.  It lets the caller reject cycle skips instead of
    treating a barely higher peak as unique evidence.
    """

    lag_samples: int
    polarity: int
    absolute_correlation: float
    zero_lag_absolute_correlation: float
    alternate_lag_samples: int | None
    alternate_absolute_correlation: float | None
    peak_prominence: float
    search_limit_samples: int
    valid_candidate_count: int


def ricker_wavelet(frequency_hz: float, sample_interval_ms: float, duration_ms: float = 128.0) -> np.ndarray:
    """生成奇数长度、峰值归一化的零相位Ricker子波。"""

    frequency = float(frequency_hz)
    dt_ms = float(sample_interval_ms)
    if frequency <= 0 or dt_ms <= 0 or duration_ms <= 0:
        raise ValueError("Ricker子波频率、采样间隔和长度必须大于0")
    half_samples = max(1, int(round(float(duration_ms) / dt_ms / 2.0)))
    time_s = np.arange(-half_samples, half_samples + 1, dtype=float) * dt_ms / 1000.0
    phase = (np.pi * frequency * time_s) ** 2
    wavelet = (1.0 - 2.0 * phase) * np.exp(-phase)
    scale = float(np.max(np.abs(wavelet)))
    return wavelet / scale if scale > 0 else wavelet


def acoustic_reflectivity(velocity_m_s: np.ndarray, density_g_cm3: np.ndarray) -> np.ndarray:
    """由纵波速度和体积密度计算法向入射反射系数。"""

    velocity = np.asarray(velocity_m_s, dtype=float)
    density = np.asarray(density_g_cm3, dtype=float)
    if velocity.shape != density.shape:
        raise ValueError("速度与密度数组形状不一致")
    impedance = velocity * density
    denominator = impedance[1:] + impedance[:-1]
    reflectivity = np.full(max(0, impedance.size - 1), np.nan, dtype=float)
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    reflectivity[valid] = (impedance[1:][valid] - impedance[:-1][valid]) / denominator[valid]
    return reflectivity


def shift_trace(trace: np.ndarray, lag_samples: int, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(trace, dtype=float)
    shifted = np.full(values.shape, float(fill), dtype=float)
    lag = int(lag_samples)
    if lag == 0:
        shifted[:] = values
    elif 0 < lag < values.size:
        shifted[lag:] = values[:-lag]
    elif -values.size < lag < 0:
        shifted[:lag] = values[-lag:]
    return shifted


def assess_static_shift(
    synthetic: np.ndarray,
    seismic: np.ndarray,
    max_lag_samples: int,
    *,
    min_overlap_samples: int = 24,
    ambiguity_exclusion_samples: int = 0,
) -> StaticShiftAssessment:
    """Measure the winning static and whether it is a unique correlation peak.

    正lag表示将合成记录向更晚时间移动。相关计算只使用真实重叠区，避免循环移位。
    """

    left = np.asarray(synthetic, dtype=float)
    right = np.asarray(seismic, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("静态时移估计要求两个等长一维数组")
    limit = min(max(0, int(max_lag_samples)), max(0, left.size - 2))
    candidates: list[tuple[int, int, float]] = []
    for lag in range(-limit, limit + 1):
        if lag > 0:
            a, b = left[:-lag], right[lag:]
        elif lag < 0:
            a, b = left[-lag:], right[:lag]
        else:
            a, b = left, right
        valid = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(valid)) < int(min_overlap_samples):
            continue
        a, b = a[valid], b[valid]
        a, b = a - np.mean(a), b - np.mean(b)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-12:
            continue
        correlation = float(np.dot(a, b) / denominator)
        candidates.append((lag, 1 if correlation >= 0 else -1, abs(correlation)))

    if not candidates:
        return StaticShiftAssessment(
            lag_samples=0,
            polarity=1,
            absolute_correlation=0.0,
            zero_lag_absolute_correlation=0.0,
            alternate_lag_samples=None,
            alternate_absolute_correlation=None,
            peak_prominence=0.0,
            search_limit_samples=limit,
            valid_candidate_count=0,
        )

    # ``max`` is stable, preserving the historical earlier-lag tie break.
    best_lag, best_polarity, best_abs = max(candidates, key=lambda item: item[2])
    zero_lag_abs = next(
        (correlation for lag, _polarity, correlation in candidates if lag == 0),
        0.0,
    )
    exclusion = max(0, int(ambiguity_exclusion_samples))
    alternatives = [
        item for item in candidates if abs(item[0] - best_lag) > exclusion
    ]
    alternate = max(alternatives, key=lambda item: item[2]) if alternatives else None
    alternate_abs = None if alternate is None else float(alternate[2])
    prominence = (
        float(best_abs)
        if alternate_abs is None
        else max(0.0, float(best_abs) - alternate_abs)
    )
    return StaticShiftAssessment(
        lag_samples=int(best_lag),
        polarity=int(best_polarity),
        absolute_correlation=max(0.0, float(best_abs)),
        zero_lag_absolute_correlation=max(0.0, float(zero_lag_abs)),
        alternate_lag_samples=None if alternate is None else int(alternate[0]),
        alternate_absolute_correlation=alternate_abs,
        peak_prominence=prominence,
        search_limit_samples=limit,
        valid_candidate_count=len(candidates),
    )


def estimate_static_shift(
    synthetic: np.ndarray,
    seismic: np.ndarray,
    max_lag_samples: int,
    *,
    min_overlap_samples: int = 24,
) -> tuple[int, int, float]:
    """Return ``(lag, polarity, abs_correlation)`` for compatibility."""

    assessment = assess_static_shift(
        synthetic,
        seismic,
        max_lag_samples,
        min_overlap_samples=min_overlap_samples,
    )
    return (
        assessment.lag_samples,
        assessment.polarity,
        assessment.absolute_correlation,
    )


def static_shift_gate_rejection_reasons(
    assessment: StaticShiftAssessment,
    *,
    sample_interval_ms: float,
    minimum_correlation: float,
    minimum_peak_prominence: float,
    maximum_accepted_shift_ms: float,
    boundary_margin_ms: float,
) -> tuple[str, ...]:
    """Return fail-closed reasons without consulting any depth-time labels."""

    dt_ms = float(sample_interval_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("地震采样间隔必须是有限正数")
    candidate_shift_ms = float(assessment.lag_samples * dt_ms)
    boundary_margin_samples = int(
        np.ceil(max(0.0, float(boundary_margin_ms)) / dt_ms)
    )
    rejection_reasons: list[str] = []
    if assessment.absolute_correlation < float(minimum_correlation):
        rejection_reasons.append("correlation_below_threshold")
    if (
        assessment.alternate_absolute_correlation is not None
        and assessment.peak_prominence < float(minimum_peak_prominence)
    ):
        rejection_reasons.append("ambiguous_correlation_peak")
    if abs(candidate_shift_ms) > float(maximum_accepted_shift_ms) + 1e-9:
        rejection_reasons.append("static_shift_exceeds_physical_prior_limit")
    boundary_threshold = max(
        1, assessment.search_limit_samples - boundary_margin_samples
    )
    if (
        assessment.search_limit_samples > 0
        and abs(assessment.lag_samples) >= boundary_threshold
    ):
        rejection_reasons.append("correlation_peak_at_search_boundary")
    return tuple(rejection_reasons)


def robust_stack_traces(
    traces: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    clip_sigma: float = 3.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a local reference trace while suppressing incoherent amplitudes.

    The first axis is the local trace gather and the second axis is seismic
    time.  With at least three traces, a sample-wise median/MAD envelope
    winsorizes spikes before the weighted stack.  One trace remains an exact
    no-op, so existing single-trace behavior is unchanged.
    """

    values = np.asarray(traces, dtype=float)
    if values.ndim == 1:
        return values.copy(), {
            "trace_stack_method": "single_trace",
            "trace_stack_count": 1,
            "trace_stack_clipped_fraction": 0.0,
        }
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("井旁地震参考必须是一维道或二维局部道集")
    if not np.isfinite(clip_sigma) or float(clip_sigma) <= 0.0:
        raise ValueError("稳健道集截断系数必须是有限正数")
    count = values.shape[0]
    if weights is None:
        base_weights = np.ones(count, dtype=float)
    else:
        base_weights = np.asarray(weights, dtype=float)
        if base_weights.shape != (count,):
            raise ValueError("局部道权重必须与道数一致")
        if np.any(~np.isfinite(base_weights)) or np.any(base_weights < 0.0):
            raise ValueError("局部道权重必须是有限非负数")
        if not np.any(base_weights > 0.0):
            raise ValueError("局部道权重不能全部为0")

    finite = np.isfinite(values)
    prepared = values.copy()
    clipped_count = 0
    finite_count = int(np.sum(finite))
    method = "finite_weighted_mean"
    if count >= 3:
        method = "median_mad_winsorized_weighted_mean"
        with np.errstate(invalid="ignore"):
            center = np.nanmedian(np.where(finite, values, np.nan), axis=0)
            mad = np.nanmedian(np.abs(np.where(finite, values, np.nan) - center), axis=0)
        robust_scale = 1.4826 * mad
        positive_scale = robust_scale[np.isfinite(robust_scale) & (robust_scale > 1e-12)]
        scale_floor = float(np.median(positive_scale) * 0.05) if positive_scale.size else 0.0
        robust_scale = np.maximum(robust_scale, scale_floor)
        lower = center - float(clip_sigma) * robust_scale
        upper = center + float(clip_sigma) * robust_scale
        clip_mask = finite & (
            (prepared < lower[None, :]) | (prepared > upper[None, :])
        )
        clipped_count = int(np.sum(clip_mask))
        prepared = np.where(finite, np.minimum(np.maximum(prepared, lower), upper), np.nan)

    weighted = np.where(np.isfinite(prepared), prepared, 0.0) * base_weights[:, None]
    denominator = np.sum(np.isfinite(prepared) * base_weights[:, None], axis=0)
    stacked = np.divide(
        np.sum(weighted, axis=0),
        denominator,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=denominator > 0.0,
    )
    return stacked, {
        "trace_stack_method": method,
        "trace_stack_count": int(count),
        "trace_stack_clip_sigma": float(clip_sigma),
        "trace_stack_clipped_fraction": (
            float(clipped_count / finite_count) if finite_count else 0.0
        ),
        "trace_stack_valid_sample_fraction": float(np.mean(np.isfinite(stacked))),
    }


def _curve(curves: Mapping[str, np.ndarray], name: str, size: int) -> np.ndarray | None:
    if name not in curves:
        return None
    values = np.asarray(curves[name], dtype=float)
    return values if values.shape == (size,) else None


def _velocity_from_curves(
    curves: Mapping[str, np.ndarray],
    size: int,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray | None, str, float]:
    candidates: list[tuple[np.ndarray, str, float]] = []
    vp_curve = _curve(curves, "VP", size)
    if vp_curve is not None:
        candidates.append((vp_curve, "VP", 0.95))
    dt = _curve(curves, "DT", size)
    if dt is not None:
        from_dt = np.divide(1_000_000.0, dt, out=np.full(dt.shape, np.nan), where=np.isfinite(dt) & (dt > 0))
        candidates.append((from_dt, "DT", 0.9))
    impedance = _curve(curves, "PIMP", size)
    density = _curve(curves, "RHOB", size)
    if impedance is not None and density is not None:
        from_impedance = np.divide(
            impedance,
            density,
            out=np.full(impedance.shape, np.nan),
            where=np.isfinite(impedance) & np.isfinite(density) & (density > 0),
        )
        candidates.append((from_impedance, "PIMP/RHOB", 0.72))
    if not candidates:
        return None, "none", 0.0
    prepared: list[tuple[float, np.ndarray, str, float]] = []
    for values, source, quality in candidates:
        candidate = np.asarray(values, dtype=float).copy()
        candidate[(candidate < minimum) | (candidate > maximum)] = np.nan
        coverage = float(np.mean(np.isfinite(candidate)))
        prepared.append((coverage * quality, candidate, source, quality))
    _, velocity, source, quality = max(prepared, key=lambda item: item[0])
    return velocity, source, quality


def _density_from_curves(
    curves: Mapping[str, np.ndarray],
    velocity: np.ndarray,
    coefficient: float,
    exponent: float,
) -> tuple[np.ndarray, str, float, float]:
    size = velocity.size
    density = np.full(size, np.nan, dtype=float)
    source_parts: list[str] = []
    source_quality = 0.58

    rhob = _curve(curves, "RHOB", size)
    if rhob is not None:
        valid = np.isfinite(rhob) & (rhob > 0.5) & (rhob < 5.0)
        density[valid] = rhob[valid]
        if np.any(valid):
            source_parts.append("RHOB")
            source_quality = 0.95

    impedance = _curve(curves, "PIMP", size)
    if impedance is not None:
        missing = ~np.isfinite(density)
        valid = missing & np.isfinite(impedance) & np.isfinite(velocity) & (velocity > 0)
        density[valid] = impedance[valid] / velocity[valid]
        valid &= (density > 0.5) & (density < 5.0)
        density[~np.isfinite(density) | (density <= 0.5) | (density >= 5.0)] = np.nan
        if np.any(valid):
            source_parts.append("PIMP/VP")
            source_quality = min(source_quality, 0.82) if "RHOB" in source_parts else 0.82

    missing = ~np.isfinite(density) & np.isfinite(velocity) & (velocity > 0)
    if np.any(missing):
        density[missing] = float(coefficient) * np.power(velocity[missing], float(exponent))
        source_parts.append("Gardner")
        source_quality = min(source_quality, 0.62)

    coverage = float(np.mean(np.isfinite(density))) if density.size else 0.0
    return density, "+".join(source_parts) or "none", source_quality, coverage


def _synthetic_on_time_axis(
    depth_m: np.ndarray,
    velocity_m_s: np.ndarray,
    density_g_cm3: np.ndarray,
    transform: DepthTimeTransform,
    seismic_time_ms: np.ndarray,
    *,
    frequency_hz: float,
    wavelet_duration_ms: float,
    max_reflectivity_gap_ms: float,
) -> tuple[np.ndarray, tuple[float, float], int]:
    depth = np.asarray(depth_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    density = np.asarray(density_g_cm3, dtype=float)
    times = transform.depth_to_time(depth)
    valid = np.isfinite(depth) & np.isfinite(times) & np.isfinite(velocity) & np.isfinite(density)
    if int(np.sum(valid)) < 3:
        raise ValueError("有效声波/密度样点不足，无法生成合成地震")
    order = np.argsort(times[valid])
    times, velocity, density = times[valid][order], velocity[valid][order], density[valid][order]
    unique, unique_index = np.unique(times, return_index=True)
    times, velocity, density = unique, velocity[unique_index], density[unique_index]
    if times.size < 3:
        raise ValueError("时间域有效样点不足")

    reflectivity = acoustic_reflectivity(velocity, density)
    mid_time = 0.5 * (times[1:] + times[:-1])
    time_step = np.diff(times)
    valid_rc = np.isfinite(reflectivity) & (time_step > 0)
    if max_reflectivity_gap_ms > 0:
        valid_rc &= time_step <= float(max_reflectivity_gap_ms)
    reflectivity, mid_time = reflectivity[valid_rc], mid_time[valid_rc]
    if reflectivity.size < 2:
        raise ValueError("连续阻抗样点不足，无法计算反射系数")

    time_axis = np.asarray(seismic_time_ms, dtype=float)
    if time_axis.ndim != 1 or time_axis.size < 3 or not np.all(np.diff(time_axis) > 0):
        raise ValueError("地震时间轴必须是严格递增的一维毫秒数组")
    dt_ms = float(np.median(np.diff(time_axis)))
    reflectivity_time = np.zeros(time_axis.size, dtype=float)
    indices = np.rint((mid_time - time_axis[0]) / dt_ms).astype(int)
    inside = (indices >= 0) & (indices < time_axis.size)
    np.add.at(reflectivity_time, indices[inside], reflectivity[inside])
    if not np.any(np.abs(reflectivity_time) > 0):
        raise ValueError("声波曲线时间范围与地震时间轴不重叠")

    wavelet = ricker_wavelet(frequency_hz, dt_ms, wavelet_duration_ms)
    if wavelet.size > reflectivity_time.size:
        half = max(1, (reflectivity_time.size - 1) // 2)
        center = wavelet.size // 2
        wavelet = wavelet[center - half:center + half + 1]
    synthetic = np.convolve(reflectivity_time, wavelet, mode="same")
    return synthetic, (float(np.min(mid_time)), float(np.max(mid_time))), int(reflectivity.size)


def build_sonic_time_domain_alignment(
    depth_m: np.ndarray,
    curves: Mapping[str, np.ndarray],
    seismic_trace: np.ndarray,
    seismic_time_ms: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> TimeDomainAlignment | None:
    """用声波积分和有界静态相关构建时间域候选井震标定。

    实现遵循SEG ``well-tie calculus`` 的物理顺序：速度积分、阻抗差分、
    子波卷积，再用局部地震参考道做有限校正；不会执行无约束stretch/squeeze。
    """

    options = dict(config or {})
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 1 or depth.size < 3:
        return None
    minimum_velocity = float(options.get("min_velocity_m_s", 500.0))
    maximum_velocity = float(options.get("max_velocity_m_s", 8000.0))
    velocity, velocity_source, velocity_quality = _velocity_from_curves(
        curves, depth.size, minimum_velocity, maximum_velocity
    )
    if velocity is None or int(np.sum(np.isfinite(velocity))) < 3:
        return None

    coverage = float(np.mean(np.isfinite(depth) & np.isfinite(velocity)))
    density, density_source, density_quality, density_coverage = _density_from_curves(
        curves,
        velocity,
        float(options.get("gardner_coefficient", 0.31)),
        float(options.get("gardner_exponent", 0.25)),
    )
    datum_explicit = bool(options.get("datum_is_explicit", False))
    prior_confidence = (
        velocity_quality
        * density_quality
        * min(coverage, density_coverage)
        * (1.0 if datum_explicit else 0.72)
    )
    replacement_velocity = options.get("replacement_velocity_m_s")
    transform = SonicIntegratedTimeDepthTransform(
        depth,
        velocity,
        reference_depth_m=float(options.get("reference_depth_m", 0.0)),
        datum_time_ms=float(options.get("datum_time_ms", 0.0)),
        replacement_velocity_m_s=replacement_velocity,
        max_gap_m=float(options.get("max_gap_m", 30.0)),
        confidence=prior_confidence,
    )

    trace_input = np.asarray(seismic_trace, dtype=float)
    time_axis = np.asarray(seismic_time_ms, dtype=float)
    if trace_input.ndim == 1:
        trace, trace_stack_diagnostics = robust_stack_traces(trace_input)
    elif trace_input.ndim == 2:
        trace, trace_stack_diagnostics = robust_stack_traces(
            trace_input,
            options.get("trace_stack_weights"),
            clip_sigma=float(options.get("trace_stack_clip_sigma", 3.5)),
        )
    else:
        raise ValueError("井旁地震参考必须是一维道或二维局部道集")
    if time_axis.ndim != 1 or trace.shape != time_axis.shape:
        raise ValueError("井旁地震参考道必须与地震时间轴等长")
    diagnostics: dict[str, Any] = {
        "velocity_source": velocity_source,
        "velocity_coverage": round(coverage, 6),
        "density_source": density_source,
        "density_coverage": round(density_coverage, 6),
        "datum_is_explicit": datum_explicit,
        "datum_time_ms": float(options.get("datum_time_ms", 0.0)),
        "reference_depth_m": float(options.get("reference_depth_m", 0.0)),
        "replacement_velocity_m_s": (
            None if replacement_velocity is None else float(replacement_velocity)
        ),
        **trace_stack_diagnostics,
        "open_source_basis": "SEG tutorials-2014 / Bruges-compatible equations; original NumPy implementation",
    }
    try:
        synthetic, valid_time_range, reflectivity_count = _synthetic_on_time_axis(
            depth,
            velocity,
            density,
            transform,
            time_axis,
            frequency_hz=float(options.get("wavelet_frequency_hz", 30.0)),
            wavelet_duration_ms=float(options.get("wavelet_duration_ms", 128.0)),
            max_reflectivity_gap_ms=float(options.get("max_reflectivity_gap_ms", 30.0)),
        )
    except ValueError as exc:
        diagnostics["tie_issue"] = str(exc)
        return TimeDomainAlignment(
            transform=transform,
            status="vertical_initial",
            method="sonic_integrated",
            confidence=float(np.clip(prior_confidence * 0.5, 0.0, 1.0)),
            uncertainty_ms=None,
            training_eligible=False,
            diagnostics=diagnostics,
        )

    dt_ms = float(np.median(np.diff(time_axis)))
    margin = float(options.get("wavelet_duration_ms", 128.0)) / 2.0
    tie_mask = (time_axis >= valid_time_range[0] - margin) & (time_axis <= valid_time_range[1] + margin)
    minimum_overlap = int(options.get("min_overlap_samples", 24))
    maximum_shift_ms = float(options.get("max_static_shift_ms", 120.0))
    maximum_lag_samples = int(round(maximum_shift_ms / max(dt_ms, 1e-9)))
    ambiguity_exclusion_ms = float(
        options.get(
            "static_shift_ambiguity_exclusion_ms",
            1000.0 / max(float(options.get("wavelet_frequency_hz", 30.0)), 1e-9),
        )
    )
    ambiguity_exclusion_samples = int(
        np.ceil(max(0.0, ambiguity_exclusion_ms) / max(dt_ms, 1e-9))
    )
    if int(np.sum(tie_mask)) >= minimum_overlap:
        assessment = assess_static_shift(
            synthetic[tie_mask],
            trace[tie_mask],
            maximum_lag_samples,
            min_overlap_samples=minimum_overlap,
            ambiguity_exclusion_samples=ambiguity_exclusion_samples,
        )
    else:
        assessment = StaticShiftAssessment(
            lag_samples=0,
            polarity=1,
            absolute_correlation=0.0,
            zero_lag_absolute_correlation=0.0,
            alternate_lag_samples=None,
            alternate_absolute_correlation=None,
            peak_prominence=0.0,
            search_limit_samples=maximum_lag_samples,
            valid_candidate_count=0,
        )
    candidate_lag = assessment.lag_samples
    polarity = assessment.polarity
    correlation = assessment.absolute_correlation
    candidate_shift_ms = float(candidate_lag * dt_ms)
    minimum_correlation = float(options.get("min_correlation", 0.2))
    minimum_peak_prominence = float(options.get("min_peak_prominence", 0.05))
    maximum_accepted_shift_ms = float(
        options.get("max_accepted_static_shift_ms", maximum_shift_ms)
    )
    boundary_margin_ms = float(
        options.get("static_shift_boundary_margin_ms", 2.0 * dt_ms)
    )
    rejection_reasons = list(
        static_shift_gate_rejection_reasons(
            assessment,
            sample_interval_ms=dt_ms,
            minimum_correlation=minimum_correlation,
            minimum_peak_prominence=minimum_peak_prominence,
            maximum_accepted_shift_ms=maximum_accepted_shift_ms,
            boundary_margin_ms=boundary_margin_ms,
        )
    )
    accepted = not rejection_reasons
    lag = candidate_lag if accepted else 0
    shift_ms = candidate_shift_ms if accepted else 0.0
    if accepted:
        transform.shift_ms += shift_ms

    shift_quality = float(np.exp(-abs(shift_ms) / max(maximum_shift_ms, dt_ms)))
    tie_quality = correlation if accepted else 0.2 * correlation
    confidence = float(np.clip(prior_confidence * (0.35 + 0.65 * tie_quality) * shift_quality, 0.0, 0.88))
    uncertainty_ms = float(max(dt_ms, (1.0 - correlation) * maximum_shift_ms))
    status = "estimated_tie" if accepted else "vertical_initial"
    training_eligible = bool(
        accepted
        and options.get("allow_estimated_for_training", False)
        and correlation >= float(options.get("training_min_correlation", 0.55))
    )
    diagnostics.update({
        "valid_time_range_ms": [round(valid_time_range[0], 6), round(valid_time_range[1], 6)],
        "reflectivity_samples": reflectivity_count,
        "wavelet_frequency_hz": float(options.get("wavelet_frequency_hz", 30.0)),
        "static_shift_ms": round(shift_ms, 6),
        "candidate_static_shift_ms": round(candidate_shift_ms, 6),
        "candidate_lag_samples": int(candidate_lag),
        "polarity": polarity,
        "absolute_correlation": round(correlation, 6),
        "zero_lag_absolute_correlation": round(
            assessment.zero_lag_absolute_correlation, 6
        ),
        "correlation_gain_vs_zero_lag": round(
            correlation - assessment.zero_lag_absolute_correlation, 6
        ),
        "alternate_lag_samples": assessment.alternate_lag_samples,
        "alternate_absolute_correlation": (
            None
            if assessment.alternate_absolute_correlation is None
            else round(assessment.alternate_absolute_correlation, 6)
        ),
        "peak_prominence": round(assessment.peak_prominence, 6),
        "static_shift_gate_accepted": accepted,
        "static_shift_rejection_reasons": rejection_reasons,
        "correlation_threshold": minimum_correlation,
        "minimum_peak_prominence": minimum_peak_prominence,
        "static_shift_ambiguity_exclusion_ms": ambiguity_exclusion_ms,
        "max_accepted_static_shift_ms": maximum_accepted_shift_ms,
        "static_shift_boundary_margin_ms": boundary_margin_ms,
        "static_shift_search_limit_ms": maximum_shift_ms,
    })
    corrected_synthetic = shift_trace(float(polarity) * synthetic, lag) if accepted else synthetic
    return TimeDomainAlignment(
        transform=transform,
        status=status,
        method="sonic_integrated_static_tie" if accepted else "sonic_integrated",
        confidence=confidence,
        uncertainty_ms=uncertainty_ms,
        training_eligible=training_eligible,
        diagnostics=diagnostics,
        synthetic_trace=corrected_synthetic,
    )
