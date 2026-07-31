from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import TimeDepthTable, Trajectory, WellHead, WellLog


def normalize_well_name(name: str) -> str:
    return re.sub(r"[\s\-_.]+", "", str(name).upper())


@dataclass
class WellEntity:
    well_uid: str
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    heads: list[WellHead] = field(default_factory=list)
    logs: list[WellLog] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    time_depth: list[TimeDepthTable] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def preferred_head(self) -> WellHead | None:
        if not self.heads:
            return None
        return sorted(self.heads, key=lambda h: (h.x is not None and h.y is not None, h.confidence), reverse=True)[0]

    @property
    def preferred_trajectory(self) -> Trajectory | None:
        return sorted(self.trajectories, key=lambda t: (len(t.md), t.confidence), reverse=True)[0] if self.trajectories else None


class WellRegistry:
    def __init__(self, aliases: dict[str, list[str]] | None = None):
        self.entities: dict[str, WellEntity] = {}
        self.alias_to_key: dict[str, str] = {}
        for canonical, names in (aliases or {}).items():
            key = normalize_well_name(canonical)
            self._entity(key, canonical)
            for name in [canonical, *names]:
                self.alias_to_key[normalize_well_name(name)] = key

    def _entity(self, key: str, display: str) -> WellEntity:
        if key not in self.entities:
            self.entities[key] = WellEntity(f"WELL-{len(self.entities)+1:05d}", display, {display})
        return self.entities[key]

    def resolve(self, name: str) -> WellEntity:
        normalized = normalize_well_name(name)
        key = self.alias_to_key.get(normalized, normalized)
        entity = self._entity(key, name)
        entity.aliases.add(name)
        self.alias_to_key[normalized] = key
        return entity

    def add_head(self, head: WellHead) -> None:
        entity = self.resolve(head.well_name)
        for existing in entity.heads:
            if existing.x is not None and head.x is not None and abs(existing.x - head.x) > 1:
                entity.conflicts.append(f"wellhead_x_conflict:{existing.x}!={head.x}")
            if existing.y is not None and head.y is not None and abs(existing.y - head.y) > 1:
                entity.conflicts.append(f"wellhead_y_conflict:{existing.y}!={head.y}")
        entity.heads.append(head)

    def add_log(self, log: WellLog) -> None:
        self.resolve(log.well_name).logs.append(log)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        self.resolve(trajectory.well_name).trajectories.append(trajectory)

    def add_time_depth(
        self,
        name: str,
        source: str,
        depth: Any,
        time: Any,
        *,
        depth_domain: str = "md",
        depth_unit: str = "m",
        time_unit: str = "ms",
        confidence: float = 0.95,
    ) -> None:
        domain = str(depth_domain).strip().lower()
        if domain not in {"md", "tvd", "tvdss"}:
            raise ValueError(f"未知时深表深度域：{depth_domain}")
        self.resolve(name).time_depth.append(
            TimeDepthTable(
                source=source,
                depth=depth,
                time=time,
                depth_domain=domain,
                depth_unit=str(depth_unit),
                time_unit=str(time_unit),
                confidence=float(confidence),
            )
        )
