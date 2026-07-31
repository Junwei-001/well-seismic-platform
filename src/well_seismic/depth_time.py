from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DepthTimeTransform(ABC):
    method = "none"
    confidence = 0.0

    @abstractmethod
    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoDepthTimeTransform(DepthTimeTransform):
    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(depth, dtype=float), np.nan)


class ProvidedTimeDepthTransform(DepthTimeTransform):
    method = "provided_time_depth"
    confidence = 0.95

    def __init__(self, depth: np.ndarray, time: np.ndarray):
        order = np.argsort(depth)
        self.depth = np.asarray(depth, dtype=float)[order]
        self.time = np.asarray(time, dtype=float)[order]

    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        values = np.interp(depth, self.depth, self.time, left=np.nan, right=np.nan)
        outside = (depth < self.depth[0]) | (depth > self.depth[-1])
        values[outside] = np.nan
        return values


class ConstantVelocityTransform(DepthTimeTransform):
    method = "constant_velocity"
    confidence = 0.2

    def __init__(self, velocity_m_s: float):
        self.velocity = float(velocity_m_s)

    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        return 2000.0 * np.asarray(depth, dtype=float) / self.velocity


class SonicIntegratedTimeDepthTransform(DepthTimeTransform):
    """由TVD域纵波速度积分得到的候选双程时间关系。

    该关系没有checkshot/VSP约束，只能作为 ``estimated_tie`` 的物理先验。
    ``reference_depth_m`` 与 ``datum_time_ms`` 显式保留，避免默认把声波起测点
    当作地震零时刻。
    """

    method = "sonic_integrated"
    depth_domain = "tvd"

    def __init__(
        self,
        depth_m: np.ndarray,
        velocity_m_s: np.ndarray,
        *,
        reference_depth_m: float = 0.0,
        datum_time_ms: float = 0.0,
        replacement_velocity_m_s: float | None = None,
        max_gap_m: float = 30.0,
        shift_ms: float = 0.0,
        confidence: float = 0.45,
    ):
        depth = np.asarray(depth_m, dtype=float)
        velocity = np.asarray(velocity_m_s, dtype=float)
        if depth.shape != velocity.shape:
            raise ValueError("深度与速度数组形状不一致")
        valid = np.isfinite(depth) & np.isfinite(velocity) & (velocity > 0)
        depth, velocity = depth[valid], velocity[valid]
        if depth.size < 2:
            raise ValueError("声波积分至少需要两个有效速度样点")

        order = np.argsort(depth)
        depth, velocity = depth[order], velocity[order]
        unique_depth, inverse = np.unique(depth, return_inverse=True)
        if unique_depth.size != depth.size:
            sums = np.zeros(unique_depth.size, dtype=float)
            counts = np.zeros(unique_depth.size, dtype=float)
            np.add.at(sums, inverse, velocity)
            np.add.at(counts, inverse, 1.0)
            depth, velocity = unique_depth, sums / np.maximum(counts, 1.0)
        if depth.size < 2:
            raise ValueError("去除重复深度后速度样点不足")

        replacement = None if replacement_velocity_m_s is None else float(replacement_velocity_m_s)
        if replacement is not None and replacement <= 0:
            raise ValueError("替代速度必须大于0")
        start_time = float(datum_time_ms)
        if replacement is not None:
            start_time += 2000.0 * (float(depth[0]) - float(reference_depth_m)) / replacement

        twt = np.full(depth.size, np.nan, dtype=float)
        twt[0] = start_time
        last = depth.size
        for index in range(1, depth.size):
            dz = float(depth[index] - depth[index - 1])
            if dz <= 0 or (max_gap_m > 0 and dz > float(max_gap_m)):
                last = index
                break
            slowness = 0.5 * (1.0 / velocity[index - 1] + 1.0 / velocity[index])
            twt[index] = twt[index - 1] + 2000.0 * dz * slowness

        self.depth = depth[:last]
        self.time = twt[:last]
        self.velocity = velocity[:last]
        if self.depth.size < 2 or not np.all(np.isfinite(self.time)):
            raise ValueError("有效声波连续段不足，无法积分时深关系")
        self.reference_depth_m = float(reference_depth_m)
        self.datum_time_ms = float(datum_time_ms)
        self.replacement_velocity_m_s = replacement
        self.max_gap_m = float(max_gap_m)
        self.shift_ms = float(shift_ms)
        self.confidence = float(np.clip(confidence, 0.0, 1.0))

    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        query = np.asarray(depth, dtype=float)
        values = np.interp(query, self.depth, self.time, left=np.nan, right=np.nan)
        outside = (query < self.depth[0]) | (query > self.depth[-1])
        values[outside] = np.nan
        return values + self.shift_ms

    def state_dict(self) -> dict[str, float | None]:
        return {
            "reference_depth_m": self.reference_depth_m,
            "datum_time_ms": self.datum_time_ms,
            "replacement_velocity_m_s": self.replacement_velocity_m_s,
            "max_gap_m": self.max_gap_m,
            "shift_ms": self.shift_ms,
            "confidence": self.confidence,
        }


class LearnableDepthTimeTransform(DepthTimeTransform):
    """Framework-neutral affine correction hook for downstream optimization."""
    method = "learnable_affine"
    confidence = 0.5

    def __init__(self, base: DepthTimeTransform, scale: float = 1.0, shift_ms: float = 0.0):
        self.base, self.scale, self.shift_ms = base, scale, shift_ms

    def depth_to_time(self, depth: np.ndarray) -> np.ndarray:
        return self.base.depth_to_time(depth) * self.scale + self.shift_ms

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "shift_ms": self.shift_ms}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])
        self.shift_ms = float(state["shift_ms"])
