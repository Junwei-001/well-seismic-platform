from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np


DepthDirection = Literal["positive_down", "positive_up"]


_SOURCE_AUTHORITY = {
    "checkshot": 50,
    "vsp": 50,
    "checkshot_vsp": 50,
    "provided_time_depth": 40,
    "well_twt_curve": 20,
    "sonic": 10,
    "unknown": 0,
}


@dataclass(frozen=True)
class ProvidedTimeDepthCandidate:
    """One already-normalized provided time-depth candidate.

    ``depth_m`` must be metres and ``twt_ms`` must be two-way time in
    milliseconds.  Datum and OWT/TWT normalization deliberately happen before
    this scientific QC so a plausible number cannot hide an unresolved
    coordinate contract.
    """

    source: str
    depth_m: np.ndarray
    twt_ms: np.ndarray
    depth_direction: DepthDirection = "positive_down"
    source_kind: str = "provided_time_depth"
    confidence: float = 0.95
    metadata_score: float = 1.0
    uncertainty_ms: float | np.ndarray | None = None


@dataclass
class ProvidedTimeDepthQC:
    source: str
    source_kind: str
    accepted: bool
    depth_m: np.ndarray
    twt_ms: np.ndarray
    uncertainty_ms: np.ndarray
    interval_velocity_m_s: np.ndarray
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata_score: float = 0.0

    @property
    def uncertainty_known(self) -> bool:
        return bool(self.uncertainty_ms.size and np.any(np.isfinite(self.uncertainty_ms)))

    @property
    def authority_key(self) -> tuple[float, ...]:
        return (
            float(_SOURCE_AUTHORITY.get(self.source_kind, -1)),
            float(np.clip(self.metadata_score, 0.0, 1.0)),
            float(np.clip(self.confidence, 0.0, 1.0)),
            float(self.diagnostics.get("depth_span_m", 0.0)),
            float(self.depth_m.size),
        )

    def to_metadata(self) -> dict[str, Any]:
        finite_uncertainty = self.uncertainty_ms[np.isfinite(self.uncertainty_ms)]
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "accepted": bool(self.accepted),
            "point_count": int(self.depth_m.size),
            "uncertainty_known": self.uncertainty_known,
            # Unknown uncertainty is null, never the misleading value 0 ms.
            "median_uncertainty_ms": (
                float(np.median(finite_uncertainty)) if finite_uncertainty.size else None
            ),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            **self.diagnostics,
        }


@dataclass
class ProvidedTimeDepthSelection:
    selected: ProvidedTimeDepthQC | None
    accepted: bool
    evaluations: list[ProvidedTimeDepthQC]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)


def _uncertainty_array(value: float | np.ndarray | None, size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=float)
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.full(size, float(raw), dtype=float)
    if raw.shape != (size,):
        raise ValueError("时深不确定度必须是标量或与控制点等长的一维数组")
    result = raw.copy()
    result[~np.isfinite(result) | (result < 0.0)] = np.nan
    return result


def _collapse_duplicate_depths(
    depth: np.ndarray,
    twt: np.ndarray,
    uncertainty: np.ndarray,
    *,
    tolerance_m: float,
    conflict_tolerance_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    if depth.size == 0:
        return depth, twt, uncertainty, [], 0
    groups: list[slice] = []
    start = 0
    for index in range(1, depth.size):
        # Compare with the first member, not the previous member, so a dense
        # sequence cannot be transitively collapsed into one giant group.
        if depth[index] - depth[start] > tolerance_m:
            groups.append(slice(start, index))
            start = index
    groups.append(slice(start, depth.size))

    clean_depth: list[float] = []
    clean_twt: list[float] = []
    clean_uncertainty: list[float] = []
    issues: list[str] = []
    duplicates = 0
    for group in groups:
        group_depth = depth[group]
        group_twt = twt[group]
        group_uncertainty = uncertainty[group]
        duplicates += max(0, int(group_depth.size) - 1)
        spread_ms = float(np.max(group_twt) - np.min(group_twt))
        if group_depth.size > 1 and spread_ms > conflict_tolerance_ms:
            issues.append(
                "duplicate_depth_conflict:"
                f"depth={float(np.median(group_depth)):.6g}m,time_spread={spread_ms:.6g}ms"
            )
        clean_depth.append(float(np.median(group_depth)))
        clean_twt.append(float(np.median(group_twt)))
        finite_uncertainty = group_uncertainty[np.isfinite(group_uncertainty)]
        if finite_uncertainty.size:
            # Preserve the most conservative stated uncertainty and account for
            # disagreement between duplicate observations.
            clean_uncertainty.append(
                max(float(np.max(finite_uncertainty)), 0.5 * spread_ms)
            )
        else:
            clean_uncertainty.append(np.nan)
    return (
        np.asarray(clean_depth, dtype=float),
        np.asarray(clean_twt, dtype=float),
        np.asarray(clean_uncertainty, dtype=float),
        issues,
        duplicates,
    )


def _interval_velocity(depth_down_m: np.ndarray, twt_ms: np.ndarray) -> np.ndarray:
    dz = np.diff(depth_down_m)
    dt = np.diff(twt_ms)
    return np.divide(
        2000.0 * dz,
        dt,
        out=np.full(dt.shape, np.nan, dtype=float),
        where=np.isfinite(dz) & np.isfinite(dt) & (dt > 0.0),
    )


def _repair_isolated_time_spikes(
    depth: np.ndarray,
    twt: np.ndarray,
    uncertainty: np.ndarray,
    *,
    minimum_velocity_m_s: float,
    maximum_velocity_m_s: float,
    residual_threshold_ms: float,
    maximum_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    original_size = depth.size
    removed: list[str] = []
    while depth.size >= 4 and len(removed) < max(1, int(np.floor(original_size * maximum_fraction))):
        velocity = _interval_velocity(depth, twt)
        valid_velocity = (
            np.isfinite(velocity)
            & (velocity >= minimum_velocity_m_s)
            & (velocity <= maximum_velocity_m_s)
        )
        repaired = False
        for index in range(1, depth.size - 1):
            if valid_velocity[index - 1] or valid_velocity[index]:
                continue
            bridge_dt = twt[index + 1] - twt[index - 1]
            bridge_dz = depth[index + 1] - depth[index - 1]
            if bridge_dt <= 0.0 or bridge_dz <= 0.0:
                continue
            bridge_velocity = 2000.0 * bridge_dz / bridge_dt
            if not minimum_velocity_m_s <= bridge_velocity <= maximum_velocity_m_s:
                continue
            fraction = (depth[index] - depth[index - 1]) / bridge_dz
            expected = twt[index - 1] + fraction * bridge_dt
            residual = abs(float(twt[index] - expected))
            if residual < residual_threshold_ms:
                continue
            removed.append(
                f"isolated_time_outlier_removed:index={index},depth={depth[index]:.6g}m,"
                f"residual={residual:.6g}ms"
            )
            keep = np.ones(depth.size, dtype=bool)
            keep[index] = False
            depth, twt, uncertainty = depth[keep], twt[keep], uncertainty[keep]
            repaired = True
            break
        if not repaired:
            break
    return depth, twt, uncertainty, removed


def quality_control_provided_time_depth(
    candidate: ProvidedTimeDepthCandidate,
    *,
    minimum_points: int = 3,
    duplicate_depth_tolerance_m: float = 0.01,
    duplicate_time_conflict_ms: float = 4.0,
    minimum_interval_velocity_m_s: float = 1200.0,
    maximum_interval_velocity_m_s: float = 6500.0,
    repair_isolated_outliers: bool = False,
    isolated_outlier_threshold_ms: float = 12.0,
    maximum_repair_fraction: float = 0.1,
) -> ProvidedTimeDepthQC:
    """Clean and physically validate a provided checkshot/VSP/time-depth table.

    Duplicate depths are explicitly collapsed.  All remaining controls must be
    strictly monotonic in both physical depth and TWT, and every interval must
    imply a velocity inside the configured physical envelope.  Optional spike
    repair is deliberately opt-in and is fully reported.
    """

    depth = np.asarray(candidate.depth_m, dtype=float)
    twt = np.asarray(candidate.twt_ms, dtype=float)
    if depth.ndim != 1 or twt.ndim != 1 or depth.shape != twt.shape:
        raise ValueError("时深控制点必须是两个等长的一维数组")
    if candidate.depth_direction not in {"positive_down", "positive_up"}:
        raise ValueError(f"未知深度方向：{candidate.depth_direction}")
    if duplicate_depth_tolerance_m < 0.0:
        raise ValueError("重复深度容差不得小于0")
    if not 0.0 <= maximum_repair_fraction < 1.0:
        raise ValueError("最大异常点修复比例必须位于[0,1)")
    if not 0.0 < minimum_interval_velocity_m_s < maximum_interval_velocity_m_s:
        raise ValueError("区间速度上下限无效")

    uncertainty = _uncertainty_array(candidate.uncertainty_ms, depth.size)
    finite = np.isfinite(depth) & np.isfinite(twt)
    removed_nonfinite = int(depth.size - np.sum(finite))
    depth, twt, uncertainty = depth[finite], twt[finite], uncertainty[finite]
    depth_down = depth if candidate.depth_direction == "positive_down" else -depth
    order = np.argsort(depth_down, kind="stable")
    depth_down, twt, uncertainty = depth_down[order], twt[order], uncertainty[order]

    depth_down, twt, uncertainty, duplicate_issues, duplicate_count = _collapse_duplicate_depths(
        depth_down,
        twt,
        uncertainty,
        tolerance_m=float(duplicate_depth_tolerance_m),
        conflict_tolerance_ms=float(duplicate_time_conflict_ms),
    )
    warnings: list[str] = []
    if repair_isolated_outliers and depth_down.size >= minimum_points + 1:
        depth_down, twt, uncertainty, repair_warnings = _repair_isolated_time_spikes(
            depth_down,
            twt,
            uncertainty,
            minimum_velocity_m_s=float(minimum_interval_velocity_m_s),
            maximum_velocity_m_s=float(maximum_interval_velocity_m_s),
            residual_threshold_ms=float(isolated_outlier_threshold_ms),
            maximum_fraction=float(maximum_repair_fraction),
        )
        warnings.extend(repair_warnings)

    issues = list(duplicate_issues)
    if depth_down.size < int(minimum_points):
        issues.append(f"insufficient_control_points:{depth_down.size}<{int(minimum_points)}")
    depth_step = np.diff(depth_down)
    time_step = np.diff(twt)
    if np.any(depth_step <= 0.0):
        issues.append("depth_not_strictly_monotonic_after_deduplication")
    non_monotonic = np.flatnonzero(time_step <= 0.0)
    if non_monotonic.size:
        issues.append(
            "twt_not_strictly_increasing:intervals="
            + ",".join(str(int(index)) for index in non_monotonic[:12])
        )

    velocity = _interval_velocity(depth_down, twt)
    bad_velocity = np.flatnonzero(
        ~np.isfinite(velocity)
        | (velocity < float(minimum_interval_velocity_m_s))
        | (velocity > float(maximum_interval_velocity_m_s))
    )
    if bad_velocity.size:
        issues.append(
            "interval_velocity_out_of_range:intervals="
            + ",".join(str(int(index)) for index in bad_velocity[:12])
        )

    returned_depth = depth_down if candidate.depth_direction == "positive_down" else -depth_down
    finite_velocity = velocity[np.isfinite(velocity)]
    diagnostics = {
        "depth_direction": candidate.depth_direction,
        "input_point_count": int(np.asarray(candidate.depth_m).size),
        "removed_nonfinite_count": removed_nonfinite,
        "collapsed_duplicate_count": duplicate_count,
        "repaired_outlier_count": len(warnings),
        "depth_span_m": (
            float(depth_down[-1] - depth_down[0]) if depth_down.size >= 2 else 0.0
        ),
        "minimum_interval_velocity_m_s": float(minimum_interval_velocity_m_s),
        "maximum_interval_velocity_m_s": float(maximum_interval_velocity_m_s),
        "observed_interval_velocity_min_m_s": (
            float(np.min(finite_velocity)) if finite_velocity.size else None
        ),
        "observed_interval_velocity_median_m_s": (
            float(np.median(finite_velocity)) if finite_velocity.size else None
        ),
        "observed_interval_velocity_max_m_s": (
            float(np.max(finite_velocity)) if finite_velocity.size else None
        ),
        "bad_interval_count": int(bad_velocity.size),
    }
    return ProvidedTimeDepthQC(
        source=str(candidate.source),
        source_kind=str(candidate.source_kind).casefold(),
        accepted=not issues,
        depth_m=returned_depth,
        twt_ms=twt,
        uncertainty_ms=uncertainty,
        interval_velocity_m_s=velocity,
        issues=issues,
        warnings=warnings,
        diagnostics=diagnostics,
        confidence=float(np.clip(candidate.confidence, 0.0, 1.0)),
        metadata_score=float(np.clip(candidate.metadata_score, 0.0, 1.0)),
    )


def _pairwise_disagreement(
    left: ProvidedTimeDepthQC,
    right: ProvidedTimeDepthQC,
    *,
    grid_points: int,
) -> dict[str, Any] | None:
    left_down = left.depth_m if left.diagnostics["depth_direction"] == "positive_down" else -left.depth_m
    right_down = right.depth_m if right.diagnostics["depth_direction"] == "positive_down" else -right.depth_m
    lower = max(float(np.min(left_down)), float(np.min(right_down)))
    upper = min(float(np.max(left_down)), float(np.max(right_down)))
    if not upper > lower:
        return None
    grid = np.linspace(lower, upper, max(3, int(grid_points)))
    left_twt = np.interp(grid, left_down, left.twt_ms)
    right_twt = np.interp(grid, right_down, right.twt_ms)
    absolute = np.abs(left_twt - right_twt)
    return {
        "left_source": left.source,
        "right_source": right.source,
        "overlap_depth_m": float(upper - lower),
        "median_absolute_disagreement_ms": float(np.median(absolute)),
        "p90_absolute_disagreement_ms": float(np.quantile(absolute, 0.9)),
        "maximum_absolute_disagreement_ms": float(np.max(absolute)),
    }


def select_authoritative_time_depth(
    candidates: Sequence[ProvidedTimeDepthCandidate],
    *,
    equal_authority_conflict_ms: float = 12.0,
    comparison_grid_points: int = 101,
    **qc_options: Any,
) -> ProvidedTimeDepthSelection:
    """QC all tables and select by explicit evidence authority, never row count.

    Conflicting candidates at the same authority level require review and do
    not silently pick the longer file.  A lower-authority disagreement is
    retained as an audit warning while the higher-authority candidate wins.
    """

    evaluations = [quality_control_provided_time_depth(item, **qc_options) for item in candidates]
    valid = [item for item in evaluations if item.accepted]
    if not valid:
        return ProvidedTimeDepthSelection(
            selected=None,
            accepted=False,
            evaluations=evaluations,
            issues=["no_physically_valid_provided_time_depth_table"],
        )
    valid.sort(key=lambda item: item.authority_key, reverse=True)
    selected = valid[0]
    issues: list[str] = []
    warnings: list[str] = []
    comparisons: list[dict[str, Any]] = []
    selected_priority = _SOURCE_AUTHORITY.get(selected.source_kind, -1)
    for other in valid[1:]:
        comparison = _pairwise_disagreement(
            selected,
            other,
            grid_points=comparison_grid_points,
        )
        if comparison is None:
            continue
        comparisons.append(comparison)
        if comparison["p90_absolute_disagreement_ms"] <= float(equal_authority_conflict_ms):
            continue
        other_priority = _SOURCE_AUTHORITY.get(other.source_kind, -1)
        message = (
            f"time_depth_table_disagreement:{selected.source}!={other.source}:"
            f"p90={comparison['p90_absolute_disagreement_ms']:.6g}ms"
        )
        if other_priority == selected_priority:
            issues.append(message)
        else:
            warnings.append("lower_authority_" + message)
    return ProvidedTimeDepthSelection(
        selected=selected,
        accepted=not issues,
        evaluations=evaluations,
        issues=issues,
        warnings=warnings,
        comparisons=comparisons,
    )
