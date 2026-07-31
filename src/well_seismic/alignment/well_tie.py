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


def estimate_static_shift(
    synthetic: np.ndarray,
    seismic: np.ndarray,
    max_lag_samples: int,
    *,
    min_overlap_samples: int = 24,
) -> tuple[int, int, float]:
    """返回 ``(lag, polarity, abs_correlation)``。

    正lag表示将合成记录向更晚时间移动。相关计算只使用真实重叠区，避免循环移位。
    """

    left = np.asarray(synthetic, dtype=float)
    right = np.asarray(seismic, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("静态时移估计要求两个等长一维数组")
    best_lag, best_polarity, best_abs = 0, 1, -1.0
    limit = min(max(0, int(max_lag_samples)), max(0, left.size - 2))
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
        if abs(correlation) > best_abs:
            best_lag = lag
            best_polarity = 1 if correlation >= 0 else -1
            best_abs = abs(correlation)
    return best_lag, best_polarity, max(0.0, best_abs)


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
    transform = SonicIntegratedTimeDepthTransform(
        depth,
        velocity,
        reference_depth_m=float(options.get("reference_depth_m", 0.0)),
        datum_time_ms=float(options.get("datum_time_ms", 0.0)),
        replacement_velocity_m_s=options.get("replacement_velocity_m_s", 2000.0),
        max_gap_m=float(options.get("max_gap_m", 30.0)),
        confidence=prior_confidence,
    )

    trace = np.asarray(seismic_trace, dtype=float)
    time_axis = np.asarray(seismic_time_ms, dtype=float)
    if trace.ndim != 1 or trace.shape != time_axis.shape:
        raise ValueError("井旁地震参考道必须与地震时间轴等长")
    diagnostics: dict[str, Any] = {
        "velocity_source": velocity_source,
        "velocity_coverage": round(coverage, 6),
        "density_source": density_source,
        "density_coverage": round(density_coverage, 6),
        "datum_is_explicit": datum_explicit,
        "datum_time_ms": float(options.get("datum_time_ms", 0.0)),
        "reference_depth_m": float(options.get("reference_depth_m", 0.0)),
        "replacement_velocity_m_s": float(options.get("replacement_velocity_m_s", 2000.0)),
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
    if int(np.sum(tie_mask)) >= minimum_overlap:
        lag, polarity, correlation = estimate_static_shift(
            synthetic[tie_mask],
            trace[tie_mask],
            int(round(maximum_shift_ms / max(dt_ms, 1e-9))),
            min_overlap_samples=minimum_overlap,
        )
    else:
        lag, polarity, correlation = 0, 1, 0.0
    shift_ms = float(lag * dt_ms)
    minimum_correlation = float(options.get("min_correlation", 0.2))
    accepted = correlation >= minimum_correlation
    if accepted:
        transform.shift_ms += shift_ms
    else:
        shift_ms, lag = 0.0, 0

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
        "polarity": polarity,
        "absolute_correlation": round(correlation, 6),
        "correlation_threshold": minimum_correlation,
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
