from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialMatch:
    asset: Any
    reader: Any
    trace_index: int
    distance: float
    neighbor_trace_indices: tuple[int, ...] = ()
    neighbor_distances: tuple[float, ...] = ()
    interpolation_weights: tuple[float, ...] = ()


@dataclass
class _SpatialIndex:
    asset: Any
    reader: Any
    trace_indices: np.ndarray
    points: np.ndarray
    tree: Any = None


class NearestTraceSpatialAligner:
    """最近地震道基线；接口与样本构建解耦，后续可单独替换。"""

    name = "nearest_trace"
    version = "1.0"

    def __init__(self, neighbor_count: int = 1) -> None:
        self.indexes: list[_SpatialIndex] = []
        self.neighbor_count = max(1, int(neighbor_count))

    def fit(self, seismic_sources: list[tuple[Any, Any]]) -> "NearestTraceSpatialAligner":
        self.indexes = []
        for asset, reader in seismic_sources:
            geometry = reader.geometry
            if geometry is None or geometry.x is None or geometry.y is None:
                continue
            valid = (
                np.isfinite(geometry.x)
                & np.isfinite(geometry.y)
                & ~((geometry.x == 0) & (geometry.y == 0))
            )
            if not np.any(valid):
                continue
            trace_indices = np.flatnonzero(valid)
            points = np.column_stack([geometry.x[valid], geometry.y[valid]])
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(points)
            except ImportError:
                tree = None
            self.indexes.append(_SpatialIndex(asset, reader, trace_indices, points, tree))
        return self

    def match(self, x: float, y: float, asset: Any | None = None) -> SpatialMatch | None:
        best: SpatialMatch | None = None
        for index in self.indexes:
            if asset is not None and index.asset is not asset:
                continue
            count = min(self.neighbor_count, len(index.trace_indices))
            if index.tree is not None:
                distances, local = index.tree.query([x, y], k=count)
            else:
                all_distances = np.hypot(index.points[:, 0] - x, index.points[:, 1] - y)
                local = np.argsort(all_distances)[:count]
                distances = all_distances[local]
            local = np.atleast_1d(np.asarray(local, dtype=int))
            distances = np.atleast_1d(np.asarray(distances, dtype=float))
            order = np.argsort(distances)
            local, distances = local[order], distances[order]
            neighbors = index.trace_indices[local].astype(int)
            exact = distances <= 1e-9
            if np.any(exact):
                weights = exact.astype(float) / float(np.sum(exact))
            else:
                inverse = 1.0 / np.maximum(distances, 1e-9)
                weights = inverse / np.sum(inverse)
            candidate = SpatialMatch(
                asset=index.asset,
                reader=index.reader,
                trace_index=int(neighbors[0]),
                distance=float(distances[0]),
                neighbor_trace_indices=tuple(int(value) for value in neighbors),
                neighbor_distances=tuple(float(value) for value in distances),
                interpolation_weights=tuple(float(value) for value in weights),
            )
            if best is None or candidate.distance < best.distance:
                best = candidate
        return best


def build_spatial_aligner(config: dict[str, Any] | None = None) -> NearestTraceSpatialAligner:
    options = dict(config or {})
    method = str(options.get("method", "nearest_trace")).lower()
    if method != "nearest_trace":
        raise ValueError(f"未知井震空间对齐算法：{method}")
    return NearestTraceSpatialAligner(options.get("neighbor_traces", 1))
