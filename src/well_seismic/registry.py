from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import TimeDepthTable, Trajectory, WellHead, WellLog
from .well_identity import canonical_api12_values


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
    identifiers: set[str] = field(default_factory=set)

    @property
    def preferred_head(self) -> WellHead | None:
        if not self.heads:
            return None
        return sorted(self.heads, key=lambda h: (h.x is not None and h.y is not None, h.confidence), reverse=True)[0]

    @property
    def preferred_trajectory(self) -> Trajectory | None:
        if not self.trajectories:
            return None

        def authority_score(trajectory: Trajectory) -> tuple[float, ...]:
            md = np.asarray(trajectory.md, dtype=float)
            tvd = np.asarray(trajectory.tvd, dtype=float)
            valid = np.isfinite(md) & np.isfinite(tvd)
            valid_count = int(np.sum(valid))
            md_valid = md[valid]
            monotonic = float(np.mean(np.diff(md_valid) > 0.0)) if md_valid.size > 1 else 0.0
            span = float(np.nanmax(md_valid) - np.nanmin(md_valid)) if md_valid.size else 0.0

            def complete_pair(left: Any, right: Any) -> bool:
                if left is None or right is None:
                    return False
                try:
                    left_values = np.asarray(left, dtype=float)
                    right_values = np.asarray(right, dtype=float)
                except (TypeError, ValueError):
                    return False
                return bool(
                    left_values.ndim == 1
                    and right_values.ndim == 1
                    and left_values.shape == md.shape
                    and right_values.shape == md.shape
                    and np.isfinite(left_values).all()
                    and np.isfinite(right_values).all()
                )

            explicit_xy = float(
                complete_pair(trajectory.x, trajectory.y)
            )
            complete_offsets = complete_pair(
                trajectory.x_offset, trajectory.y_offset
            )
            registration_allowed = (
                trajectory.vertical_semantics.get("registration_eligible") is not False
            )
            formal_ready = float(
                registration_allowed
                and md.ndim == 1
                and tvd.ndim == 1
                and md.shape == tvd.shape
                and md.size >= 2
                and np.isfinite(md).all()
                and np.isfinite(tvd).all()
                and np.all(np.diff(np.round(md, decimals=8)) > 0.0)
                and (bool(explicit_xy) or complete_offsets)
            )
            source = str(trajectory.source).casefold()
            version_hint = 0.0
            if any(token in source for token in ("latest", "current", "final", "最新")):
                version_hint += 0.1
            if any(token in source for token in ("backup", "old", "history", "旧版", "历史")):
                version_hint -= 0.1
            return (
                formal_ready,
                explicit_xy,
                monotonic,
                float(trajectory.confidence) + version_hint - 0.03 * len(trajectory.issues),
                span,
                float(valid_count) / max(len(md), 1),
                float(valid_count),
            )

        return max(self.trajectories, key=authority_score)


class WellRegistry:
    def __init__(self, aliases: dict[str, list[str]] | None = None):
        self.entities: dict[str, WellEntity] = {}
        self.alias_to_key: dict[str, str] = {}
        self.identifier_to_key: dict[str, str] = {}
        self._next_well_id = 1
        self._next_identity_quarantine_id = 1
        for canonical, names in (aliases or {}).items():
            key = normalize_well_name(canonical)
            self._entity(key, canonical)
            for name in [canonical, *names]:
                self.alias_to_key[normalize_well_name(name)] = key

    def _entity(self, key: str, display: str) -> WellEntity:
        if key not in self.entities:
            well_uid = f"WELL-{self._next_well_id:05d}"
            self._next_well_id += 1
            self.entities[key] = WellEntity(well_uid, display, {display})
        return self.entities[key]

    def resolve(self, name: str) -> WellEntity:
        normalized = normalize_well_name(name)
        key = self.alias_to_key.get(normalized, normalized)
        entity = self._entity(key, name)
        entity.aliases.add(name)
        self.alias_to_key[normalized] = key
        return entity

    @staticmethod
    def _add_conflict(entity: WellEntity, conflict: str) -> None:
        if conflict not in entity.conflicts:
            entity.conflicts.append(conflict)

    def _identity_quarantine(
        self,
        name: str,
        identifiers: list[str],
        reason: str,
    ) -> WellEntity:
        normalized = normalize_well_name(name) or "UNNAMED"
        key = (
            f"{normalized}#IDENTITY-QUARANTINE-"
            f"{self._next_identity_quarantine_id:05d}"
        )
        self._next_identity_quarantine_id += 1
        entity = self._entity(key, name)
        entity.identifiers.update(identifiers)
        self._add_conflict(entity, reason)
        # Neither the ambiguous name nor any disputed identifier is published
        # as an alias.  The asset remains inspectable but cannot enter a
        # cross-file registration join.
        return entity

    def _resolve_asset(
        self,
        name: str,
        raw_identifiers: list[str],
        *,
        asset_kind: str,
    ) -> WellEntity:
        identifiers, invalid = canonical_api12_values(raw_identifiers)
        if invalid:
            return self._identity_quarantine(
                name,
                identifiers,
                f"{asset_kind}_well_identifier_invalid:{'|'.join(invalid)}",
            )
        if len(identifiers) > 1:
            return self._identity_quarantine(
                name,
                identifiers,
                f"{asset_kind}_well_identifier_conflict:{'|'.join(identifiers)}",
            )
        if not identifiers:
            return self.resolve(name)

        identifier = identifiers[0]
        normalized = normalize_well_name(name)
        identifier_key = self.identifier_to_key.get(identifier)
        name_key = self.alias_to_key.get(normalized)

        if identifier_key is not None:
            entity = self.entities[identifier_key]
            if name_key is not None and name_key != identifier_key:
                named = self.entities[name_key]
                if named.identifiers and identifier not in named.identifiers:
                    conflict = (
                        "well_name_maps_to_distinct_identifiers:"
                        f"{normalized}:{'|'.join(sorted(named.identifiers))}!="
                        f"{identifier}"
                    )
                    self._add_conflict(named, conflict)
                    self._add_conflict(entity, conflict)
                else:
                    entity = self._merge_entities(identifier_key, name_key)
                    identifier_key = identifier_key
            entity.identifiers.add(identifier)
            entity.aliases.add(name)
            if name_key in {None, identifier_key}:
                self.alias_to_key[normalized] = identifier_key
            return entity

        if name_key is not None:
            named = self.entities[name_key]
            if named.identifiers and identifier not in named.identifiers:
                conflict = (
                    "well_name_maps_to_distinct_identifiers:"
                    f"{normalized}:{'|'.join(sorted(named.identifiers))}!="
                    f"{identifier}"
                )
                self._add_conflict(named, conflict)
                key = f"{normalized}#{identifier}"
                entity = self._entity(key, name)
                entity.identifiers.add(identifier)
                self._add_conflict(entity, conflict)
                self.identifier_to_key[identifier] = key
                return entity
            entity = named
            key = name_key
        else:
            key = normalized
            entity = self._entity(key, name)
            self.alias_to_key[normalized] = key
        entity.identifiers.add(identifier)
        entity.aliases.add(name)
        self.identifier_to_key[identifier] = key
        return entity

    @staticmethod
    def _las_filename_alias(log: WellLog) -> str | None:
        """Return a conservative filename identity carried by one LAS file.

        Some public LAS headers prepend a field/survey name to the actual well
        name (for example ``PENOBSCOT B-41`` in ``b41.las``).  The filename is
        usable as an alias only when its *entire* normalized stem is the suffix
        of the normalized header name.  Partial/prefix matches are deliberately
        rejected.
        """

        filename_key = normalize_well_name(Path(str(log.source)).stem)
        header_key = normalize_well_name(log.well_name)
        if (
            not filename_key
            or not header_key
            or filename_key == header_key
            or not header_key.endswith(filename_key)
        ):
            return None
        return filename_key

    def _filename_log_entity_keys(self, filename_key: str) -> set[str]:
        return {
            key
            for key, entity in self.entities.items()
            if any(
                self._las_filename_alias(log) == filename_key
                for log in entity.logs
            )
        }

    def _merge_entities(self, target_key: str, source_key: str) -> WellEntity:
        """Merge a uniquely proven filename/header entity into an exact name."""

        if target_key == source_key:
            return self.entities[target_key]
        target = self.entities[target_key]
        source = self.entities[source_key]
        target.aliases.update(source.aliases)
        target.heads.extend(source.heads)
        target.logs.extend(source.logs)
        target.trajectories.extend(source.trajectories)
        target.time_depth.extend(source.time_depth)
        target.identifiers.update(source.identifiers)
        for conflict in source.conflicts:
            if conflict not in target.conflicts:
                target.conflicts.append(conflict)
        for alias, key in list(self.alias_to_key.items()):
            if key == source_key:
                self.alias_to_key[alias] = target_key
        for identifier, key in list(self.identifier_to_key.items()):
            if key == source_key:
                self.identifier_to_key[identifier] = target_key
        del self.entities[source_key]
        return target

    def add_head(self, head: WellHead) -> None:
        entity = self._resolve_asset(
            head.well_name,
            list(head.identifiers),
            asset_kind="well_head",
        )
        for existing in entity.heads:
            same_xy = bool(
                existing.x is not None
                and existing.y is not None
                and head.x is not None
                and head.y is not None
                and abs(existing.x - head.x) <= 0.05
                and abs(existing.y - head.y) <= 0.05
            )
            if same_xy and existing.crs and not head.crs:
                head.crs = existing.crs
                head.source_crs = existing.crs
                head.coordinate_transform = {
                    "source_crs": existing.crs,
                    "target_crs": existing.crs,
                    "transformed": False,
                    "operation": "inferred_from_authoritative_duplicate",
                    "evidence_source": existing.source,
                }
            elif same_xy and head.crs and not existing.crs:
                existing.crs = head.crs
                existing.source_crs = head.crs
                existing.coordinate_transform = {
                    "source_crs": head.crs,
                    "target_crs": head.crs,
                    "transformed": False,
                    "operation": "inferred_from_authoritative_duplicate",
                    "evidence_source": head.source,
                }
            if existing.x is not None and head.x is not None and abs(existing.x - head.x) > 1:
                entity.conflicts.append(f"wellhead_x_conflict:{existing.x}!={head.x}")
            if existing.y is not None and head.y is not None and abs(existing.y - head.y) > 1:
                entity.conflicts.append(f"wellhead_y_conflict:{existing.y}!={head.y}")
        entity.heads.append(head)

    def add_log(self, log: WellLog) -> None:
        if log.identifiers:
            entity = self._resolve_asset(
                log.well_name,
                list(log.identifiers),
                asset_kind="well_log",
            )
            entity.logs.append(log)
            return
        filename_key = self._las_filename_alias(log)
        header_key = normalize_well_name(log.well_name)
        exact_key = (
            self.alias_to_key.get(filename_key)
            if filename_key is not None
            else None
        )
        header_target = self.alias_to_key.get(header_key)
        filename_candidates = (
            self._filename_log_entity_keys(filename_key)
            if filename_key is not None
            else set()
        )
        if (
            filename_key is not None
            and exact_key in self.entities
            and header_target in {None, exact_key}
            and filename_candidates <= {exact_key}
        ):
            entity = self.entities[exact_key]
            entity.aliases.add(log.well_name)
            self.alias_to_key[header_key] = exact_key
            evidence = (
                "las_filename_header_suffix_alias:"
                f"{header_key}->{filename_key}"
            )
            if evidence not in log.processing_steps:
                log.processing_steps.append(evidence)
        else:
            # Do not publish the filename stem as an alias yet.  A later head or
            # trajectory may consume it only when exactly one LAS entity claims
            # the normalized filename identity.
            entity = self.resolve(log.well_name)
        entity.logs.append(log)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.identifiers:
            entity = self._resolve_asset(
                trajectory.well_name,
                list(trajectory.identifiers),
                asset_kind="trajectory",
            )
        else:
            trajectory_key = normalize_well_name(trajectory.well_name)
            filename_candidates = self._filename_log_entity_keys(trajectory_key)
            if len(filename_candidates) == 1:
                filename_entity_key = next(iter(filename_candidates))
                exact_entity_key = self.alias_to_key.get(trajectory_key)
                if exact_entity_key in self.entities:
                    entity = self._merge_entities(
                        exact_entity_key,
                        filename_entity_key,
                    )
                    target_key = exact_entity_key
                else:
                    entity = self.entities[filename_entity_key]
                    target_key = filename_entity_key
                entity.aliases.add(trajectory.well_name)
                self.alias_to_key[trajectory_key] = target_key
            else:
                entity = self.resolve(trajectory.well_name)
        head = entity.preferred_head
        if (
            trajectory.source_crs is None
            and trajectory.x is not None
            and trajectory.y is not None
            and trajectory.x.size
            and trajectory.y.size
            and head is not None
            and head.crs
            and head.x is not None
            and head.y is not None
            and np.isfinite(trajectory.x[0])
            and np.isfinite(trajectory.y[0])
            and abs(float(trajectory.x[0]) - head.x) <= 0.05
            and abs(float(trajectory.y[0]) - head.y) <= 0.05
        ):
            trajectory.source_crs = head.crs
            trajectory.horizontal_crs = head.crs
            trajectory.coordinate_transform = {
                "source_crs": head.crs,
                "target_crs": head.crs,
                "transformed": False,
                "operation": "inferred_from_authoritative_duplicate",
                "evidence_source": head.source,
            }
        reference = trajectory.vertical_semantics.get(
            "well_reference_elevation_m"
        )
        if reference is not None and np.isfinite(float(reference)):
            for existing in entity.trajectories:
                existing_reference = existing.vertical_semantics.get(
                    "well_reference_elevation_m"
                )
                if existing_reference is None or not np.isfinite(
                    float(existing_reference)
                ):
                    continue
                if np.isclose(
                    float(reference),
                    float(existing_reference),
                    rtol=0.0,
                    atol=0.25,
                ):
                    continue
                conflict = (
                    "trajectory_vertical_datum_conflict:"
                    f"{float(existing_reference):g}m@{existing.source}!="
                    f"{float(reference):g}m@{trajectory.source}"
                )
                if conflict not in entity.conflicts:
                    entity.conflicts.append(conflict)
                issue = "unresolved_trajectory_vertical_datum_conflict"
                for candidate, other in (
                    (existing, trajectory),
                    (trajectory, existing),
                ):
                    if issue not in candidate.issues:
                        candidate.issues.append(issue)
                    candidate.confidence = min(float(candidate.confidence), 0.49)
                    candidate.vertical_semantics.update(
                        {
                            "registration_eligible": False,
                            "datum_status": "conflict_unresolved",
                            "datum_conflict_with_source": other.source,
                            "datum_conflict_tolerance_m": 0.25,
                        }
                    )
        entity.trajectories.append(trajectory)

    def add_time_depth(
        self,
        name: str,
        source: str,
        depth: Any,
        time: Any,
        *,
        source_kind: str = "unknown",
        depth_domain: str = "md",
        depth_unit: str = "unknown",
        time_unit: str = "unknown",
        confidence: float = 0.95,
        depth_datum: str | None = None,
        depth_convention: str | None = None,
        time_reference: str = "unknown",
        time_domain: str = "unknown",
        correction_state: str = "unknown",
        replacement_velocity_mps: float | None = None,
        md_offset_to_trajectory_m: float | None = None,
    ) -> None:
        domain = str(depth_domain).strip().lower()
        if domain not in {"md", "tvd", "tvdss", "z_msl_m", "depth_below_msl_m"}:
            raise ValueError(f"未知时深表深度域：{depth_domain}")
        self.resolve(name).time_depth.append(
            TimeDepthTable(
                source=source,
                depth=depth,
                time=time,
                source_kind=str(source_kind or "unknown").strip().casefold(),
                depth_domain=domain,
                depth_unit=str(depth_unit),
                time_unit=str(time_unit),
                confidence=float(confidence),
                depth_datum=None if depth_datum is None else str(depth_datum).upper(),
                depth_convention=None if depth_convention is None else str(depth_convention),
                time_reference=(
                    "unknown" if str(time_reference or "unknown").casefold() == "unknown"
                    else str(time_reference).upper()
                ),
                time_domain=(
                    "unknown" if str(time_domain or "unknown").casefold() == "unknown"
                    else str(time_domain).upper()
                ),
                correction_state=str(correction_state or "unknown").lower(),
                replacement_velocity_mps=None if replacement_velocity_mps is None else float(replacement_velocity_mps),
                md_offset_to_trajectory_m=None if md_offset_to_trajectory_m is None else float(md_offset_to_trajectory_m),
            )
        )
