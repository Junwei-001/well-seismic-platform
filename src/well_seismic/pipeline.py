from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import (
    ProvidedTimeDepthCandidate,
    TimeDomainAlignment,
    build_sonic_time_domain_alignment,
    build_spatial_aligner,
    select_authoritative_time_depth,
)
from .catalog import build_catalog
from .config import load_config
from .content_identity import canonical_sha256, file_sha256, seismic_geometry_identity
from .coordinate_reference import (
    CoordinateReferenceError,
    canonical_crs_id,
    transform_offset_path,
    transform_xy,
)
from .depth_time import (
    ConstantVelocityTransform,
    NoDepthTimeTransform,
    ProvidedTimeDepthTransform,
)
from .fusion import WellSeismicFusion, build_fusion
from .io import (
    SegyReader,
    SourceFileIdentityError,
    apply_llm_metadata_decision,
    read_acoustic_text,
    read_adaptive_metadata,
    read_las,
    read_time_depth,
    read_trajectory,
    read_well_heads,
)
from .knowledge import CurveKnowledgeBase
from .llm import build_decision_resolver
from .llm.parse_repair import (
    REPAIR_CONTRACT_VERSION,
    repair_fingerprint,
    summarize_tabular_source,
    validate_trajectory_parse_patch,
    validate_trajectory_physics,
)
from .llm.privacy import sanitize_llm_text
from .llm.transformation import apply_active_transformations
from .models import Evidence, MatchRecord, SeismicGeometry, WellHead
from .output_schema import SAMPLE_FIELDS_ZH, sample_to_chinese
from .registry import WellRegistry, normalize_well_name
from .registration_contract import (
    STRICT_MD_TWT_POLICY,
    TRAJECTORY_STATIONWISE_TWT_POLICY,
    validate_trajectory_stationwise_twt,
)
from .trajectory import interpolate_trajectory
from .vertical_datum import (
    ContractEvidenceCandidate,
    DatumObservation,
    ResolvedVerticalDatum,
    TimeReferenceMetadata,
    absolute_elevation_from_tvd,
    correct_time_to_srd,
    depth_below_srd,
    extract_time_reference_metadata,
    extract_vertical_datum_observations,
    length_to_metres,
    observation_from_value,
    observations_from_asset_options,
    resolve_vertical_datum,
    time_to_milliseconds,
)
from .workflow import build_preparation_report


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


class WellSeismicPipeline:
    def __init__(
        self,
        manifest: str | Path | dict[str, Any],
        config_dir: str | Path,
        *,
        use_llm_fallback: bool = False,
        llm_provider: Any | None = None,
    ):
        self.config = load_config(config_dir, manifest)
        apply_active_transformations(self.config, config_dir)
        self.manifest = self.config["manifest"]
        self.assets, self.duplicates = build_catalog(
            self.manifest, self.config["manifest_path"]
        )
        self.asset_by_path = {str(asset.path): asset for asset in self.assets}
        self.knowledge = CurveKnowledgeBase(self.config)
        self.decision_resolver = build_decision_resolver(
            self.config,
            requested=use_llm_fallback,
            provider=llm_provider,
        )
        self.registry = WellRegistry(self.manifest.get("well_aliases", {}))
        self.seismic: list[tuple[Any, SegyReader]] = []
        self.errors: list[dict[str, str]] = []
        self.samples: list[dict[str, Any]] = []
        self.well_ties: list[dict[str, Any]] = []
        self.registration_tracks: dict[str, dict[str, Any]] = {}
        # Viewer candidates are deliberately separate from the sealed V3
        # registration product.  They make partially described surveys
        # inspectable without relaxing fusion/training eligibility.
        self.visualization_registration_tracks: dict[str, dict[str, Any]] = {}
        self.metadata_detection: list[dict[str, Any]] = []
        self.llm_parse_repairs: list[dict[str, Any]] = []
        self.deterministic_unit_inheritances: list[dict[str, Any]] = []
        self.vertical_datum_observations: list[DatumObservation] = []
        self.vertical_datum_resolutions: dict[str, dict[str, Any]] = {}
        self.time_reference_metadata: dict[str, TimeReferenceMetadata] = {}
        self.contract_evidence_candidates: list[ContractEvidenceCandidate] = []
        self._derived_vertical_crs_candidate: ContractEvidenceCandidate | None = None
        self.automatic_inventory: dict[str, Any] | None = None
        self.ingest_cache_receipt: dict[str, Any] = {
            "seismic_geometry_hits": 0,
            "seismic_geometry_misses": 0,
        }

    @classmethod
    def from_input_root(
        cls, input_root: str | Path, config_dir: str | Path
    ) -> "WellSeismicPipeline":
        from .auto_input import build_automatic_manifest

        manifest, inventory = build_automatic_manifest(input_root)
        instance = cls(manifest, config_dir)
        instance.automatic_inventory = inventory
        return instance

    @classmethod
    def from_input_paths(
        cls,
        seismic_directory: str | Path | list[str | Path],
        log_directory: str | Path | list[str | Path],
        config_dir: str | Path,
        metadata_directory: str | Path | list[str | Path] | None = None,
        auxiliary_directory: str | Path | list[str | Path] | None = None,
        recursive: bool = True,
    ) -> "WellSeismicPipeline":
        from .auto_input import build_explicit_paths_manifest

        manifest, inventory = build_explicit_paths_manifest(
            seismic_directory=seismic_directory,
            log_directory=log_directory,
            metadata_directory=metadata_directory,
            auxiliary_directory=auxiliary_directory,
            recursive=recursive,
        )
        instance = cls(manifest, config_dir)
        instance.automatic_inventory = inventory
        return instance

    def ingest(
        self,
        progress: Any = None,
        *,
        seismic_geometry_by_path: Mapping[str, SeismicGeometry] | None = None,
        source_stat_signatures_by_path: Mapping[
            str, tuple[int, int, int, int, int]
        ]
        | None = None,
    ) -> "WellSeismicPipeline":
        """Read source assets, optionally reusing verified sealed SEG-Y geometry.

        The caller owns cache verification.  This method accepts only geometry
        whose resolved source path matches the catalog asset, and records every
        hit/miss so derived task results can prove whether a full trace-header
        scan occurred.
        """

        field_aliases = self.config.get("well_schema", {}).get("fields", {})
        geometry_cache = {
            os.path.normcase(str(Path(path).expanduser().resolve())): geometry
            for path, geometry in (seismic_geometry_by_path or {}).items()
        }
        source_signatures = {
            os.path.normcase(str(Path(path).expanduser().resolve())): signature
            for path, signature in (source_stat_signatures_by_path or {}).items()
        }
        self.ingest_cache_receipt = {
            "seismic_geometry_hits": 0,
            "seismic_geometry_misses": 0,
        }
        ingest_roles = (
            "well_heads",
            "well_logs",
            "trajectories",
            "time_depth",
            "well_metadata",
            "seismic",
        )
        ingest_total = sum(len(self._assets(role)) for role in ingest_roles)
        ingested = 0

        def begin_asset(asset: Any) -> None:
            if progress:
                progress(ingested, ingest_total, asset)

        def finish_asset() -> None:
            nonlocal ingested
            ingested += 1
            if progress:
                progress(ingested, ingest_total, None)

        def report_asset_work(asset: Any, done: int, total: int) -> None:
            if progress:
                progress(ingested, ingest_total, asset, done, total)

        # Auxiliary survey contracts are advisory evidence only.  They are
        # intentionally not merged into the effective manifest in-place: the
        # resulting request_patch must be reviewed and used to create a fresh
        # source snapshot.  Large/binary auxiliary files are never opened here.
        for asset in self._assets("auxiliary"):
            try:
                if (
                    asset.path.suffix.casefold()
                    not in {".yaml", ".yml", ".json", ".txt", ".md"}
                    or asset.path.stat().st_size > 1024 * 1024
                ):
                    continue
                auxiliary_meta = extract_time_reference_metadata(asset.path)
                for candidate in auxiliary_meta.contract_candidates:
                    self.contract_evidence_candidates.append(
                        replace(
                            candidate,
                            status=(
                                "insufficient"
                                if candidate.status == "insufficient"
                                else "candidate"
                            ),
                            source=str(asset.path),
                            inference_source="rule_auxiliary_contract",
                            requires_human_confirmation=True,
                            auto_applied=False,
                        )
                    )
            except OSError:
                continue

        # Well heads first, then logs can fill gaps from LAS headers.
        for asset in self._assets("well_heads"):
            begin_asset(asset)
            try:
                options = {**asset.options, "field_aliases": field_aliases}
                for head in read_well_heads(asset.path, options):
                    self.registry.add_head(head)
                    self._register_head_datums(head)
                self._register_path_datums(asset.path, entity_kind="well")
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        for asset in self._assets("well_logs"):
            begin_asset(asset)
            try:
                if asset.path.suffix.casefold() == ".ac":
                    log = read_acoustic_text(
                        asset.path,
                        self.knowledge,
                        self.config.get("preprocessing", {}),
                    )
                else:
                    log = read_las(
                        asset.path,
                        self.knowledge,
                        self.config.get("preprocessing", {}),
                        self.decision_resolver,
                        asset.options,
                    )
                self.registry.add_log(log)
                self._register_las_datums(log)
                self._register_path_datums(
                    asset.path, entity_kind="well", entity_name=log.well_name
                )
                if "TWT" in log.curves:
                    valid = np.isfinite(log.depth) & np.isfinite(log.curves["TWT"])
                    if np.sum(valid) >= 2:
                        time_meta = self._time_metadata(log.source, asset.options)
                        self.registry.add_time_depth(
                            log.well_name,
                            log.source + "#TWT",
                            log.depth[valid],
                            log.curves["TWT"][valid],
                            source_kind="well_twt_curve",
                            depth_domain="md",
                            depth_unit="m",
                            time_unit="ms",
                            depth_datum=time_meta.depth_datum,
                            depth_convention=time_meta.depth_convention,
                            time_reference=time_meta.time_reference,
                            time_domain="TWT",
                            correction_state=time_meta.correction_state,
                            replacement_velocity_mps=time_meta.replacement_velocity_mps,
                            md_offset_to_trajectory_m=time_meta.md_offset_to_trajectory_m,
                        )
                x, x_unit = self._header_coordinate_m(
                    log.header, "XCRD", asset.options
                )
                y, y_unit = self._header_coordinate_m(
                    log.header, "YCRD", asset.options
                )
                kb = self._header_float(log.header, "EKB", "KB")
                stop = self._header_float(log.header, "STOP")
                if self._header_float(log.header, "XCRD") is not None and x is None:
                    log.issues.append(
                        "XCRD:horizontal_coordinate_unit_"
                        + ("unknown" if x_unit == "unknown" else f"unsupported:{x_unit}")
                    )
                if self._header_float(log.header, "YCRD") is not None and y is None:
                    log.issues.append(
                        "YCRD:horizontal_coordinate_unit_"
                        + ("unknown" if y_unit == "unknown" else f"unsupported:{y_unit}")
                    )
                if x is not None and y is not None:
                    if x_unit != "m" or y_unit != "m":
                        log.processing_steps.append(
                            f"wellhead_xy_unit_converted:{x_unit}/{y_unit}->m"
                        )
                    head = WellHead(
                        log.well_name,
                        x,
                        y,
                        kb=kb,
                        total_depth_md=stop,
                        source=log.source,
                        confidence=0.9,
                        horizontal_unit="m",
                    )
                    self.registry.add_head(head)
                    self._register_head_datums(head)
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        for asset in self._assets("trajectories"):
            begin_asset(asset)
            try:
                options = {**asset.options, "field_aliases": field_aliases}
                try:
                    trajectories = read_trajectory(asset.path, options)
                except Exception as exc:
                    trajectories = self._repair_trajectory_asset(
                        asset,
                        options,
                        exc,
                    )
                    if trajectories is None:
                        raise
                for trajectory in trajectories:
                    self.registry.add_trajectory(trajectory)
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        for asset in self._assets("time_depth"):
            begin_asset(asset)
            try:
                self._register_path_datums(asset.path, entity_kind="well")
                time_meta = self._time_metadata(asset.path, asset.options)
                for name, (depth, time) in read_time_depth(
                    asset.path, asset.options
                ).items():
                    self.registry.add_time_depth(
                        name,
                        str(asset.path),
                        depth,
                        time,
                        source_kind=asset.options.get("source_kind", "unknown"),
                        depth_domain=asset.options.get("depth_domain", "md"),
                        depth_unit=time_meta.depth_unit,
                        time_unit=time_meta.time_unit,
                        depth_datum=time_meta.depth_datum,
                        depth_convention=time_meta.depth_convention,
                        time_reference=time_meta.time_reference,
                        time_domain=time_meta.time_domain,
                        correction_state=time_meta.correction_state,
                        replacement_velocity_mps=time_meta.replacement_velocity_mps,
                        md_offset_to_trajectory_m=time_meta.md_offset_to_trajectory_m,
                    )
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        for asset in self._assets("well_metadata"):
            begin_asset(asset)
            try:
                self._register_path_datums(asset.path, entity_kind="well")
                time_meta = self._time_metadata(asset.path, asset.options)
                try:
                    detected = read_adaptive_metadata(
                        asset.path,
                        field_aliases,
                        asset.options,
                    )
                except Exception as exc:
                    detected = self._inherit_trajectory_md_unit_from_same_well_las(
                        asset,
                        {**asset.options, "field_aliases": field_aliases},
                        exc,
                    )
                    if detected is None:
                        repaired = self._repair_trajectory_asset(
                            asset,
                            {**asset.options, "field_aliases": field_aliases},
                            exc,
                        )
                        if repaired is None:
                            raise
                        for trajectory in repaired:
                            self.registry.add_trajectory(trajectory)
                        repair = self.llm_parse_repairs[-1]
                        self.metadata_detection.append(
                            {
                                "文件": str(asset.path),
                                "识别角色": ["井轨迹"],
                                "置信度": repair["confidence"],
                                "状态": "LLM补丁经规则复检后已识别",
                                "证据": [
                                    "原文件只读；结构化补丁仅作用于当前数据快照",
                                    "隔离内存重解析和轨迹物理门均通过",
                                    *repair["corroboration"],
                                ],
                                "时深深度域": "未明确",
                                "决策来源": "llm_structured_patch_validated",
                                "LLM判断": {
                                    "来源摘要": repair["source_hash"],
                                    "补丁摘要": repair["patch_sha256"],
                                    "提供方": repair["provider"],
                                    "模型": repair["model"],
                                },
                            }
                        )
                        continue
                llm_decision = None
                if not detected.accepted and detected.detected_roles:
                    llm_decision = self.decision_resolver.resolve_metadata(
                        asset.path, detected
                    )
                    if llm_decision and llm_decision.accepted:
                        detected = apply_llm_metadata_decision(
                            detected,
                            llm_decision.choice,
                            llm_decision.confidence,
                        )
                detected_time_depth_domain = (
                    asset.options.get("depth_domain")
                    or detected.time_depth_domain
                    or asset.options.get("default_depth_domain")
                )
                if detected.time_depth and not detected_time_depth_domain:
                    detected.evidence.append("时深表深度域未明确，未进入垂向标定")
                self.metadata_detection.append(
                    {
                        "文件": str(asset.path),
                        "识别角色": detected.detected_roles,
                        "置信度": detected.confidence,
                        "状态": detected.status,
                        "证据": detected.evidence,
                        "时深深度域": detected_time_depth_domain or "未明确",
                        "决策来源": detected.decision_source,
                        "LLM判断": llm_decision.to_audit_dict()
                        if llm_decision
                        else None,
                    }
                )
                # Uncertain headerless interpretations remain report-only and never silently enter matching.
                if detected.accepted:
                    for head in detected.heads:
                        head.confidence = detected.confidence
                        self.registry.add_head(head)
                        self._register_head_datums(head)
                    for trajectory in detected.trajectories:
                        trajectory.confidence = detected.confidence
                        self.registry.add_trajectory(trajectory)
                    for name, (depth, time) in (
                        detected.time_depth.items()
                        if detected_time_depth_domain
                        else ()
                    ):
                        self.registry.add_time_depth(
                            name,
                            str(asset.path),
                            depth,
                            time,
                            source_kind=asset.options.get("source_kind", "unknown"),
                            depth_domain=detected_time_depth_domain,
                            depth_unit=time_meta.depth_unit,
                            time_unit=time_meta.time_unit,
                            confidence=detected.confidence,
                            depth_datum=time_meta.depth_datum,
                            depth_convention=time_meta.depth_convention,
                            time_reference=time_meta.time_reference,
                            time_domain=time_meta.time_domain,
                            correction_state=time_meta.correction_state,
                            replacement_velocity_mps=time_meta.replacement_velocity_mps,
                            md_offset_to_trajectory_m=time_meta.md_offset_to_trajectory_m,
                        )
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        for asset in self._assets("seismic"):
            begin_asset(asset)
            try:
                self._register_path_datums(asset.path, entity_kind="seismic")
                self.vertical_datum_observations.extend(
                    observations_from_asset_options(
                        asset.options,
                        source=str(asset.path),
                        entity_kind="seismic",
                        entity_name=str(asset.path),
                    )
                )
                self._time_metadata(asset.path, asset.options, table_contract=False)
                reader = SegyReader(asset.path, self.config, asset.options)
                path_key = os.path.normcase(str(asset.path.resolve()))
                expected_source_signature = source_signatures.get(path_key)
                if source_stat_signatures_by_path is not None:
                    if expected_source_signature is None:
                        raise SourceFileIdentityError(
                            "sealed SEG-Y asset has no verified runtime file identity: "
                            f"{asset.path}"
                        )
                    reader.bind_expected_source_stat_signature(
                        expected_source_signature
                    )
                cached_geometry = geometry_cache.get(path_key)
                if cached_geometry is not None:
                    if os.path.normcase(
                        str(Path(cached_geometry.path).expanduser().resolve())
                    ) != path_key:
                        raise ValueError(
                            "cached SEG-Y geometry path does not match catalog asset"
                        )
                    reader.geometry = cached_geometry
                    self.ingest_cache_receipt["seismic_geometry_hits"] += 1
                else:
                    self.ingest_cache_receipt["seismic_geometry_misses"] += 1
                    reader.inspect(
                        progress=lambda done, total: report_asset_work(
                            asset,
                            done,
                            total,
                        )
                    )
                self.seismic.append((asset, reader))
            except SourceFileIdentityError:
                raise
            except Exception as exc:
                self._error(asset, exc)
            finally:
                finish_asset()
        return self

    def _time_metadata(
        self,
        path: str | Path,
        options: dict[str, Any] | None = None,
        *,
        table_contract: bool = True,
    ) -> TimeReferenceMetadata:
        source = str(path)
        detected = extract_time_reference_metadata(path)
        raw = dict(options or {})

        def upper_or_unknown(value: Any) -> str:
            text = str(value or "unknown").strip()
            return "unknown" if text.casefold() == "unknown" else text.upper()

        def declared_or_detected(field: str, detected_value: Any) -> Any:
            value = raw.get(field)
            unresolved = value is None or str(value).strip().casefold() in {
                "",
                "unknown",
                "无法确定",
            }
            if not unresolved:
                return value
            detected_unresolved = detected_value is None or str(
                detected_value
            ).strip().casefold() in {"", "unknown", "无法确定"}
            fallback = raw.get(f"default_{field}")
            if detected_unresolved and fallback is not None and str(
                fallback
            ).strip().casefold() not in {"", "unknown", "无法确定"}:
                return fallback
            return detected_value

        def finite_positive_velocity(value: Any) -> float | None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if np.isfinite(numeric) and numeric > 0 else None

        replacement_velocity_mps = (
            finite_positive_velocity(raw.get("replacement_velocity_mps"))
            or finite_positive_velocity(detected.replacement_velocity_mps)
            or finite_positive_velocity(raw.get("default_replacement_velocity_mps"))
        )

        result = replace(
            detected,
            time_reference=upper_or_unknown(
                declared_or_detected("time_reference", detected.time_reference)
            ),
            time_domain=upper_or_unknown(
                declared_or_detected("time_domain", detected.time_domain)
            ),
            correction_state=str(
                declared_or_detected("correction_state", detected.correction_state)
            ).lower(),
            depth_unit=str(
                declared_or_detected("depth_unit", detected.depth_unit)
            ).lower(),
            time_unit=str(
                declared_or_detected("time_unit", detected.time_unit)
            ).lower(),
            depth_datum=(
                detected.depth_datum
                if declared_or_detected("depth_datum", detected.depth_datum) is None
                else str(
                    declared_or_detected("depth_datum", detected.depth_datum)
                ).upper()
            ),
            depth_convention=declared_or_detected(
                "depth_convention", detected.depth_convention
            ),
            replacement_velocity_mps=replacement_velocity_mps,
            md_offset_to_trajectory_m=(
                detected.md_offset_to_trajectory_m
                if raw.get("md_offset_to_trajectory_m") is None
                else float(raw.get("md_offset_to_trajectory_m"))
            ),
            provenance=(f"{source}#asset_options" if raw else source),
        )
        if result.time_domain not in {"TWT", "OWT", "unknown"}:
            result = replace(result, time_domain="unknown")
        if result.time_reference not in {"SRD", "KB", "GL", "DF", "RT", "unknown"}:
            result = replace(result, time_reference="unknown")
        if result.correction_state not in {
            "corrected_to_srd",
            "uncorrected",
            "unknown",
        }:
            result = replace(result, correction_state="unknown")
        if (
            result.correction_state == "corrected_to_srd"
            and result.time_reference not in {"unknown", "SRD"}
        ):
            raise ValueError(
                "correction_state=corrected_to_srd与非SRD time_reference冲突"
            )
        if result.correction_state == "corrected_to_srd":
            result = replace(result, time_reference="SRD")
        candidates = list(result.contract_candidates)
        explicit_fields = (
            ("time_domain", "seismic_time_domain", result.time_domain),
            (
                "correction_state",
                "seismic_correction_state",
                result.correction_state,
            ),
            ("time_reference", "seismic_time_reference", result.time_reference),
        )
        for option_field, candidate_field, value in explicit_fields:
            raw_value = raw.get(option_field)
            if (
                raw_value is None
                or str(raw_value).strip().casefold()
                in {"", "unknown", "无法确定"}
            ):
                continue
            candidates.append(
                ContractEvidenceCandidate(
                    field=candidate_field,
                    value=value,
                    confidence=1.0,
                    status="verified",
                    source=f"{source}#asset_options",
                    evidence=(f"asset.options.{option_field}={raw_value}",),
                    inference_source="explicit_input",
                    requires_human_confirmation=False,
                    auto_applied=True,
                )
            )
        decision_specs = [
            (
                "time_domain",
                result.time_domain,
                ["TWT", "OWT", "无法确定"],
                "判断时间列是双程时TWT还是单程时OWT。",
                "seismic_time_domain",
            ),
            (
                "time_reference",
                result.time_reference,
                ["SRD", "KB", "GL", "DF", "RT", "无法确定"],
                "判断时间值当前参考的垂向基准。",
                "seismic_time_reference",
            ),
        ]
        if table_contract:
            decision_specs.extend(
                [
                    (
                        "depth_datum",
                        result.depth_datum or "unknown",
                        ["KB", "GL", "DF", "RT", "MSL", "无法确定"],
                        "判断MD/TVD深度列的起算面。",
                        "time_depth_depth_datum",
                    ),
                    (
                        "depth_unit",
                        result.depth_unit,
                        ["m", "ft", "cm", "mm", "dm", "km", "in", "yd", "无法确定"],
                        "判断深度列长度单位。",
                        "time_depth_depth_unit",
                    ),
                    (
                        "time_unit",
                        result.time_unit,
                        ["ms", "s", "us", "无法确定"],
                        "判断时间列单位。",
                        "time_depth_time_unit",
                    ),
                ]
            )
        field_evidence = "\n".join(result.evidence).upper()
        evidence_patterns = {
            "time_domain": r"\b(?:TWT|TWTT|OWT|PSTM)\b|TWO[- ]?WAY|ONE[- ]?WAY|双程|单程",
            "time_reference": r"\b(?:TIME\s+(?:REFERENCE|DATUM)|SRD|FINAL\s+DATUM|PROCESSING\s+DATUM)\b|时间基准",
            "depth_datum": r"\b(?:DEPTH\s+(?:REFERENCE|DATUM)|KB|GL|DF|RT|TVDSS)\b|深度(?:参考|基准)",
            "depth_unit": r"\b(?:DEPTH\s+UNIT|MD\s*[\[(]|TVD\s*[\[(])|深度单位",
            "time_unit": r"\b(?:TIME\s+UNIT|TWT\s*[\[(]|OWT\s*[\[(])|时间单位",
        }
        for field, current, choices, question, candidate_field in decision_specs:
            raw_value = raw.get(field)
            raw_is_explicit = (
                field in raw
                and raw_value is not None
                and str(raw_value).strip().casefold() not in {"", "unknown", "无法确定"}
            )
            if (
                not result.evidence
                or raw_is_explicit
                or current not in {"unknown", None}
                or not re.search(
                    evidence_patterns[field], field_evidence, re.IGNORECASE
                )
            ):
                continue
            decision = self.decision_resolver.resolve(
                "vertical_datum_semantics",
                question,
                choices,
                {"field": field, "header_evidence": list(result.evidence)},
            )
            if (
                decision is not None
                and decision.accepted
                and decision.choice != "无法确定"
            ):
                candidates.append(
                    ContractEvidenceCandidate(
                        field=candidate_field,
                        value=decision.choice,
                        confidence=float(decision.confidence),
                        status="llm_suggestion",
                        source=source,
                        evidence=tuple(
                            [
                                *result.evidence,
                                f"Kimi结构化建议:{decision.reason}",
                                "建议未写入有效合同，必须由显式来源证据或人工确认后重新准备",
                            ]
                        ),
                        inference_source="LLM",
                        requires_human_confirmation=True,
                        auto_applied=False,
                    )
                )
        result = replace(result, contract_candidates=tuple(candidates))
        if not table_contract:
            self.contract_evidence_candidates.extend(candidates)
        self.time_reference_metadata[source] = result
        return result

    def _register_path_datums(
        self,
        path: str | Path,
        *,
        entity_kind: str,
        entity_name: str | None = None,
    ) -> None:
        observations = extract_vertical_datum_observations(
            path,
            entity_kind=entity_kind,
            entity_name=entity_name
            or (str(path) if entity_kind == "seismic" else None),
        )
        reviewed: list[DatumObservation] = []
        for item in observations:
            if item.unit == "unknown":
                unit_decision = self.decision_resolver.resolve(
                    "vertical_datum_semantics",
                    "根据原始表头判断该高程数值的长度单位，只能从候选项选择。",
                    ["m", "ft", "cm", "mm", "dm", "km", "in", "yd", "无法确定"],
                    {
                        "datum": item.datum,
                        "value": item.value,
                        "raw_header_evidence": item.evidence,
                    },
                )
                if (
                    unit_decision is not None
                    and unit_decision.accepted
                    and unit_decision.choice != "无法确定"
                ):
                    item = replace(
                        item,
                        unit=unit_decision.choice,
                        confidence=min(float(unit_decision.confidence), 0.92),
                        evidence=f"{item.evidence}；Kimi单位识别:{unit_decision.choice}，{unit_decision.reason}",
                    )
            if item.review_required:
                decision = self.decision_resolver.resolve(
                    "vertical_datum_semantics",
                    "该字段是否表示相对平均海平面MSL、向上为正的绝对高程？",
                    ["相对MSL的绝对高程", "无法确定"],
                    {
                        "datum": item.datum,
                        "value": item.value,
                        "unit": item.unit,
                        "header_evidence": item.evidence,
                    },
                )
                if (
                    item.unit != "unknown"
                    and decision is not None
                    and decision.accepted
                    and decision.choice == "相对MSL的绝对高程"
                ):
                    if entity_kind == "seismic":
                        self.contract_evidence_candidates.append(
                            ContractEvidenceCandidate(
                                field="seismic_srd_elevation_m",
                                value=item.absolute_elevation_m,
                                confidence=min(float(decision.confidence), 0.95),
                                status="llm_suggestion",
                                source=str(path),
                                evidence=(
                                    item.evidence,
                                    f"Kimi结构化建议:{decision.reason}",
                                    "建议未改变SRD有效合同，需人工确认后重新准备",
                                ),
                                inference_source="LLM",
                                requires_human_confirmation=True,
                                auto_applied=False,
                            )
                        )
                    else:
                        item = replace(
                            item,
                            confidence=min(float(decision.confidence), 0.95),
                            review_required=False,
                            evidence=f"{item.evidence}；Kimi语义确认:{decision.reason}",
                        )
            reviewed.append(item)
            if entity_kind == "seismic" and item.absolute_elevation_m is not None:
                self.contract_evidence_candidates.append(
                    ContractEvidenceCandidate(
                        field="seismic_srd_elevation_m",
                        value=float(item.absolute_elevation_m),
                        confidence=float(item.confidence),
                        status="candidate" if item.review_required else "verified",
                        source=str(path),
                        evidence=(item.evidence,),
                        inference_source="rule",
                        requires_human_confirmation=bool(item.review_required),
                        auto_applied=not item.review_required,
                    )
                )
        self.vertical_datum_observations.extend(reviewed)

    def _register_head_datums(self, head: WellHead) -> None:
        unit = str(head.vertical_datum_unit or "unknown").strip()
        if head.kb is not None:
            item = observation_from_value(
                datum="KB",
                value=float(head.kb),
                source=head.source,
                entity_kind="well",
                entity_name=head.well_name,
                unit=unit,
                evidence=f"井位/井轨迹表KB字段；垂向单位={unit}",
                confidence=min(0.98, float(head.confidence) * 0.95),
                review_required=unit == "unknown",
            )
            self.vertical_datum_observations.append(
                self._review_tabular_datum_unit(item)
            )
        if head.ground_elevation is not None:
            item = observation_from_value(
                datum="GL",
                value=float(head.ground_elevation),
                source=head.source,
                entity_kind="well",
                entity_name=head.well_name,
                unit=unit,
                evidence=f"井位表明确GL/ground_elevation字段；垂向单位={unit}",
                confidence=min(0.98, float(head.confidence) * 0.95),
                review_required=unit == "unknown",
                is_depth_reference=False,
            )
            self.vertical_datum_observations.append(
                self._review_tabular_datum_unit(item)
            )

    def _review_tabular_datum_unit(self, item: DatumObservation) -> DatumObservation:
        if item.unit != "unknown":
            return item
        decision = self.decision_resolver.resolve(
            "vertical_datum_semantics",
            "识别井位/井轨迹表高程字段的长度单位，只能从候选项选择；证据不足必须选择无法确定。",
            ["m", "ft", "cm", "mm", "dm", "km", "in", "yd", "无法确定"],
            {
                "datum": item.datum,
                "value": item.value,
                "header_evidence": item.evidence,
            },
        )
        if decision is None or not decision.accepted or decision.choice == "无法确定":
            return item
        return replace(
            item,
            unit=decision.choice,
            review_required=False,
            confidence=min(float(decision.confidence), 0.9),
            evidence=f"{item.evidence}；Kimi单位识别:{decision.choice}，{decision.reason}",
        )

    def _register_las_datums(self, log: Any) -> None:
        # Bare ELEV/ELEVATION is intentionally excluded: in LAS exports it may
        # describe KB, DF, RT or GL and is not a safe synonym for ground level.
        aliases = (("KB", "EKB", "KB"), ("GL", "GL", "GLEV", "GROUND_ELEVATION"))
        for datum, *keys in aliases:
            for key in keys:
                evidence = log.header.get(key)
                if evidence is None:
                    continue
                try:
                    value = float(str(evidence.value).split()[0].replace(",", ""))
                except (TypeError, ValueError):
                    continue
                raw_unit = str(evidence.notes[0]).strip() if evidence.notes else ""
                description = (
                    str(evidence.notes[1]).strip() if len(evidence.notes) > 1 else ""
                )
                unit = raw_unit or "unknown"
                explicit_msl = bool(
                    re.search(
                        r"\bMSL\b|MEAN\s+SEA\s+LEVEL|海平面", description, re.IGNORECASE
                    )
                )
                item = observation_from_value(
                    datum=datum,
                    value=value,
                    source=log.source,
                    entity_kind="well",
                    entity_name=log.well_name,
                    unit=unit,
                    evidence=f"LAS头段{key}.{unit}={value}；说明={description or '空'}",
                    confidence=0.94 if explicit_msl else 0.78,
                    review_required=not explicit_msl,
                    is_depth_reference=datum == "KB",
                )
                if item.absolute_elevation_m is None:
                    unit_decision = self.decision_resolver.resolve(
                        "vertical_datum_semantics",
                        "识别LAS高程头字段的长度单位，只能从候选项选择。",
                        ["m", "ft", "cm", "mm", "dm", "km", "in", "yd", "无法确定"],
                        {
                            "datum": datum,
                            "value": value,
                            "raw_unit": unit,
                            "header": item.evidence,
                        },
                    )
                    if (
                        unit_decision is not None
                        and unit_decision.accepted
                        and unit_decision.choice != "无法确定"
                    ):
                        item = replace(
                            item,
                            unit=unit_decision.choice,
                            confidence=min(float(unit_decision.confidence), 0.92),
                            evidence=f"{item.evidence}；Kimi单位识别:{unit_decision.choice}，{unit_decision.reason}",
                        )
                if item.review_required:
                    semantic_decision = self.decision_resolver.resolve(
                        "vertical_datum_semantics",
                        "该LAS高程头字段是否明确表示相对MSL、向上为正的绝对高程？",
                        ["相对MSL的绝对高程", "无法确定"],
                        {
                            "datum": datum,
                            "value": value,
                            "unit": item.unit,
                            "header": item.evidence,
                        },
                    )
                    if (
                        item.unit != "unknown"
                        and semantic_decision is not None
                        and semantic_decision.accepted
                        and semantic_decision.choice == "相对MSL的绝对高程"
                    ):
                        item = replace(
                            item,
                            review_required=False,
                            confidence=min(float(semantic_decision.confidence), 0.92),
                            evidence=f"{item.evidence}；Kimi语义确认:{semantic_decision.reason}",
                        )
                self.vertical_datum_observations.append(item)
                break

    def _observations_for_well(self, entity: Any) -> list[DatumObservation]:
        sources = {
            str(item.source)
            for item in (
                *entity.heads,
                *entity.logs,
                *entity.trajectories,
                *entity.time_depth,
            )
            if getattr(item, "source", None)
        }
        aliases = {
            str(name).casefold() for name in entity.aliases | {entity.canonical_name}
        }
        observations: list[DatumObservation] = []
        for item in self.vertical_datum_observations:
            if item.entity_kind != "well":
                continue
            named = (
                item.entity_name is not None
                and str(item.entity_name).casefold() in aliases
            )
            # A tabular asset can contain many wells.  Once the parser has
            # attached an entity_name, the shared source path is not evidence
            # that every row belongs to every well.  Source fallback is only
            # valid for genuinely unnamed observations.
            unnamed_source_match = item.entity_name is None and item.source in sources
            if named or unnamed_source_match:
                observations.append(
                    item.with_entity(entity.canonical_name)
                    if item.entity_name is None
                    else item
                )
        return observations

    def _datum_config(self) -> dict[str, Any]:
        return self.config.get("vertical_datum", {})

    def _vertical_crs(self) -> dict[str, Any]:
        contract = {
            "id": "LOCAL_MSL_UNSPECIFIED",
            "unit": "m",
            "axis": "elevation_positive_up",
            **dict(self._datum_config().get("vertical_crs", {})),
            **dict(self.manifest.get("vertical_crs", {})),
        }
        contract["status"] = (
            "declared"
            if contract["id"] != "LOCAL_MSL_UNSPECIFIED"
            else "unknown_survey"
        )
        return contract

    def _suggest_vertical_crs_candidate(self) -> ContractEvidenceCandidate | None:
        """Create a stable local survey label without asserting a physical datum.

        The label is only an identity namespace for this seismic survey.  It
        does not claim that two local MSL realizations are interchangeable and
        does not resolve SRD elevation or correction state.
        """

        cached = getattr(self, "_derived_vertical_crs_candidate", None)
        if cached is not None:
            return cached
        seismic_assets = [asset for asset in self.assets if asset.role == "seismic"]
        if not seismic_assets:
            return None
        digest = hashlib.sha256(b"well-seismic.local-msl-survey-id.v1\0")
        evidence: list[str] = []
        for asset in sorted(seismic_assets, key=lambda item: str(item.path).casefold()):
            try:
                size = int(asset.path.stat().st_size)
                quick = hashlib.sha256()
                with asset.path.open("rb") as handle:
                    quick.update(handle.read(65536))
                    if size > 65536:
                        handle.seek(max(0, size - 65536))
                        quick.update(handle.read(65536))
            except OSError:
                return None
            digest.update(str(size).encode("ascii"))
            digest.update(quick.digest())
            evidence.append(
                f"seismic_quick_identity:size={size},sha256={quick.hexdigest()[:16]}"
            )
        candidate = ContractEvidenceCandidate(
            field="vertical_crs_id",
            value=f"LOCAL_MSL_SURVEY_{digest.hexdigest()[:12].upper()}",
            confidence=1.0,
            status="verified",
            source="platform:deterministic_survey_identity",
            evidence=tuple(
                [
                    *evidence,
                    "该值仅为本地MSL测区命名空间，不推断SRD高程或国家垂向基准",
                ]
            ),
            inference_source="rule",
            requires_human_confirmation=False,
            auto_applied=False,
        )
        self._derived_vertical_crs_candidate = candidate
        return candidate

    def survey_contract_candidates(self) -> list[dict[str, Any]]:
        candidates = list(getattr(self, "contract_evidence_candidates", []))
        for observation in getattr(self, "vertical_datum_observations", []):
            if (
                observation.entity_kind != "seismic"
                or observation.datum != "SRD"
                or observation.absolute_elevation_m is None
            ):
                continue
            candidates.append(
                ContractEvidenceCandidate(
                    field="seismic_srd_elevation_m",
                    value=float(observation.absolute_elevation_m),
                    confidence=float(observation.confidence),
                    status=(
                        "candidate" if observation.review_required else "verified"
                    ),
                    source=observation.source,
                    evidence=(observation.evidence,),
                    inference_source="rule",
                    requires_human_confirmation=bool(observation.review_required),
                    auto_applied=not observation.review_required,
                )
            )
        vertical = self._vertical_crs()
        if vertical.get("status") == "declared":
            candidates.append(
                ContractEvidenceCandidate(
                    field="vertical_crs_id",
                    value=vertical.get("id"),
                    confidence=1.0,
                    status="verified",
                    source="manifest.vertical_crs",
                    evidence=("输入合同已明确垂向CRS/本地MSL测区标识",),
                    inference_source="explicit_input",
                    requires_human_confirmation=False,
                    auto_applied=True,
                )
            )
        else:
            derived = self._suggest_vertical_crs_candidate()
            if derived is not None:
                candidates.append(derived)
        unique: list[ContractEvidenceCandidate] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in candidates:
            key = (
                item.field,
                json.dumps(item.value, ensure_ascii=False, sort_keys=True),
                item.status,
                item.source,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return [item.to_dict() for item in unique]

    def _well_datum(self, entity: Any) -> ResolvedVerticalDatum:
        cfg = self._datum_config()
        return resolve_vertical_datum(
            self._observations_for_well(entity),
            entity_kind="well",
            entity_name=entity.canonical_name,
            tolerance_m=float(cfg.get("consistency_tolerance_m", 0.5)),
            allow_gl_as_well_depth_reference=bool(
                cfg.get("allow_gl_as_well_depth_reference", False)
            ),
            maximum_kb_gl_offset_m=float(cfg.get("maximum_kb_gl_offset_m", 30.0)),
        )

    def _seismic_datum(self, asset: Any | None) -> ResolvedVerticalDatum:
        name = "未匹配地震" if asset is None else str(asset.path)
        observations = [
            item
            for item in self.vertical_datum_observations
            if item.entity_kind == "seismic"
            and asset is not None
            and item.source == str(asset.path)
        ]
        cfg = self._datum_config()
        return resolve_vertical_datum(
            observations,
            entity_kind="seismic",
            entity_name=name,
            tolerance_m=float(cfg.get("consistency_tolerance_m", 0.5)),
        )

    def vertical_datum_inventory(self) -> dict[str, Any]:
        well_entities = [
            entity for entity in self.registry.entities.values() if entity.logs
        ]
        wells = [self._well_datum(entity) for entity in well_entities]
        seismic = [self._seismic_datum(asset) for asset, _ in self.seismic]
        vertical_crs = self._vertical_crs()
        well_payload: list[dict[str, Any]] = []
        for entity, resolution in zip(well_entities, wells):
            observations = self._observations_for_well(entity)
            datum_objects: dict[str, Any] = {}
            for datum in ("KB", "DF", "RT", "GL"):
                candidates = [
                    item
                    for item in observations
                    if item.datum == datum and item.absolute_elevation_m is not None
                ]
                chosen = (
                    max(
                        candidates,
                        key=lambda item: (not item.review_required, item.confidence),
                    )
                    if candidates
                    else None
                )
                datum_objects[datum] = {
                    "elevation_msl_m": None
                    if chosen is None
                    else float(chosen.absolute_elevation_m),
                    "provenance": "" if chosen is None else chosen.source,
                    "status": "unknown"
                    if chosen is None
                    else ("assumed" if chosen.review_required else "verified"),
                    "uncertainty_m": None,
                }
            item = resolution.to_dict()
            item["vertical_crs"] = vertical_crs
            item["well_datums"] = datum_objects
            item["asset_references"] = {
                "trajectory": {
                    "depth_axis": "MD",
                    "datum": resolution.datum,
                    "status": "verified"
                    if entity.preferred_trajectory is not None
                    else "unknown",
                },
                "las": {
                    "depth_axis": "MD",
                    "datum": resolution.datum,
                    "status": "assumed",
                },
                "time_depth": [
                    {
                        "source": table.source,
                        "source_kind": self._time_depth_source_kind(table),
                        "depth_axis": table.depth_domain.upper(),
                        "datum": table.depth_datum,
                        "depth_convention": table.depth_convention,
                        "time_reference": table.time_reference,
                        "time_domain": table.time_domain,
                        "correction_state": table.correction_state,
                    }
                    for table in entity.time_depth
                ],
            }
            well_payload.append(item)
        seismic_payload: list[dict[str, Any]] = []
        ready_seismic_time = 0
        for (asset, _), resolution in zip(self.seismic, seismic):
            time_meta = self.time_reference_metadata.get(
                str(asset.path), TimeReferenceMetadata(provenance=str(asset.path))
            )
            time_ready = bool(
                time_meta.time_domain == "TWT"
                and time_meta.time_reference == "SRD"
                and time_meta.correction_state == "corrected_to_srd"
            )
            ready_seismic_time += int(time_ready)
            item = resolution.to_dict()
            item["seismic_reference"] = {
                "type": "SRD",
                "elevation_msl_m": resolution.absolute_elevation_m,
                "replacement_velocity_mps": time_meta.replacement_velocity_mps,
                "time_domain": time_meta.time_domain,
                "time_reference": time_meta.time_reference,
                "correction_state": time_meta.correction_state,
                "provenance": time_meta.provenance or resolution.source,
                "evidence": list(time_meta.evidence),
                "contract_candidates": [
                    candidate.to_dict()
                    for candidate in time_meta.contract_candidates
                ],
                "ready": time_ready,
            }
            seismic_payload.append(item)
        payload = {
            "vertical_crs": vertical_crs,
            "physical_space": {
                "field": "z_msl_m",
                "unit": "m",
                "axis": "elevation_positive_up",
            },
            "seismic_time_space": {"reference": "SRD", "supported_domain": "TWT"},
            "wells": well_payload,
            "seismic": seismic_payload,
            "ready_wells": sum(1 for item in wells if item.ready),
            "ready_seismic": sum(1 for item in seismic if item.ready),
            "ready_seismic_time": ready_seismic_time,
            "vertical_crs_ready": vertical_crs["status"] == "declared",
            "contract_candidates": self.survey_contract_candidates(),
            "conflicts": sum(len(item.conflicts) for item in (*wells, *seismic)),
        }
        self.vertical_datum_resolutions = {
            "wells": {item["entity_name"]: item for item in well_payload},
            "seismic": {item["entity_name"]: item for item in seismic_payload},
        }
        return payload

    @staticmethod
    def _external_registration_alignment(
        track: dict[str, Any],
        query_md: np.ndarray,
    ) -> tuple[TimeDomainAlignment, np.ndarray, np.ndarray, np.ndarray]:
        """Rehydrate a prior registration by MD without treating it as TD truth."""

        source_md = np.asarray(track.get("md", []), dtype=float)
        source_twt = np.asarray(track.get("twtMean", []), dtype=float)
        source_tvd = np.asarray(track.get("tvd", []), dtype=float)
        source_std = np.asarray(track.get("twtStd", []), dtype=float)
        source_quality = np.asarray(track.get("registrationQuality", []), dtype=float)
        declared_valid = np.asarray(
            track.get("validMask", np.ones(source_md.shape, dtype=bool)),
            dtype=bool,
        )
        if not (
            source_md.ndim == 1
            and source_md.size >= 2
            and source_twt.shape == source_md.shape
            and source_std.shape == source_md.shape
            and source_quality.shape == source_md.shape
            and declared_valid.shape == source_md.shape
            and np.all(np.isfinite(source_md))
            and np.all(np.diff(source_md) > 0.0)
        ):
            raise ValueError("外部井震标定轨迹不满足全分辨率、单调MD、同长度合同")
        valid = declared_valid & np.isfinite(source_twt)
        valid_md = source_md[valid]
        valid_twt = source_twt[valid]
        if valid_md.size < 2 or np.any(np.diff(valid_md) <= 0.0):
            raise ValueError("外部井震标定有效MD-TWT点不足或MD不单调")
        trajectory_time_policy = str(
            track.get("trajectoryTimePolicy")
            or (track.get("diagnostics") or {}).get("trajectory_time_policy")
            or STRICT_MD_TWT_POLICY
        )
        if trajectory_time_policy == STRICT_MD_TWT_POLICY:
            if np.any(np.diff(valid_twt) <= 0.0):
                raise ValueError("外部井震标定有效TWT不满足严格MD-TWT合同")
            stationwise_validation = None
        elif trajectory_time_policy == TRAJECTORY_STATIONWISE_TWT_POLICY:
            valid_indices = np.flatnonzero(declared_valid)
            if (
                source_tvd.shape != source_md.shape
                or valid_indices.size < 2
                or not np.array_equal(
                    valid_indices,
                    np.arange(int(valid_indices[0]), int(valid_indices[-1]) + 1),
                )
            ):
                raise ValueError("逐站点外部井震标定要求连续有效的全分辨率TVD支持域")
            raw_segment_ids = track.get("trajectorySegmentId")
            valid_segment_ids = None
            if raw_segment_ids is not None:
                segment_ids = list(raw_segment_ids)
                if len(segment_ids) != source_md.size:
                    raise ValueError("逐站点外部井震标定分段列长度与MD不一致")
                valid_segment_ids = [segment_ids[int(index)] for index in valid_indices]
            stationwise_validation = validate_trajectory_stationwise_twt(
                source_tvd[valid_indices],
                source_twt[valid_indices],
                segment_ids=valid_segment_ids,
                identity=str(track.get("well_uid") or track.get("well_name") or "外部轨迹"),
            )
        else:
            raise ValueError(f"外部井震标定包含不支持的轨迹时间策略：{trajectory_time_policy}")
        finite_std = np.isfinite(source_std) & (source_std >= 0.0) & valid
        finite_quality_mask = (
            np.isfinite(source_quality)
            & (source_quality >= 0.0)
            & (source_quality <= 1.0)
            & valid
        )
        transform = ProvidedTimeDepthTransform(valid_md, valid_twt)
        times = transform.depth_to_time(query_md)
        inside = (query_md >= valid_md[0]) & (query_md <= valid_md[-1])
        uncertainty = np.full(query_md.shape, np.nan, dtype=float)
        quality = np.full(query_md.shape, np.nan, dtype=float)
        if int(np.sum(finite_std)) >= 2:
            std_md = source_md[finite_std]
            std_inside = inside & (query_md >= std_md[0]) & (query_md <= std_md[-1])
            uncertainty[std_inside] = np.interp(
                query_md[std_inside], std_md, source_std[finite_std]
            )
        if int(np.sum(finite_quality_mask)) >= 2:
            quality_md = source_md[finite_quality_mask]
            quality_inside = (
                inside & (query_md >= quality_md[0]) & (query_md <= quality_md[-1])
            )
            quality[quality_inside] = np.interp(
                query_md[quality_inside],
                quality_md,
                source_quality[finite_quality_mask],
            )
        finite_quality = quality[np.isfinite(quality)]
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        diagnostics = {
            **dict(track.get("diagnostics") or {}),
            "external_registration_reused": True,
            "registration_is_time_depth_supervision": False,
            "registration_source": str(
                track.get("registrationSource", "external_registration")
            ),
            "inference_eligible": bool(track.get("inferenceEligible", True)),
            "fusion_ready": bool(
                track.get("fusionReady", track.get("trainingEligible", False))
            ),
            "supervision_eligible": bool(
                track.get("supervisionEligible", track.get("trainingEligible", False))
            ),
            "trajectory_time_policy": trajectory_time_policy,
            "trajectory_segment_count": (
                None
                if stationwise_validation is None
                else stationwise_validation["trajectory_segment_count"]
            ),
            "tvd_reversal_interval_count": (
                0
                if stationwise_validation is None
                else stationwise_validation["tvd_reversal_interval_count"]
            ),
        }
        alignment = TimeDomainAlignment(
            transform=transform,
            status=str(track.get("registrationStatus", "estimated_tie")),
            method=str(track.get("registrationSource", "external_registration")),
            confidence=(
                float(np.median(finite_quality)) if finite_quality.size else 0.0
            ),
            uncertainty_ms=(
                float(np.median(finite_uncertainty))
                if finite_uncertainty.size
                else None
            ),
            training_eligible=bool(track.get("trainingEligible", False)),
            depth_domain="md",
            diagnostics=diagnostics,
        )
        return alignment, times, uncertainty, quality

    def build_samples(
        self,
        *,
        emit_samples: bool = True,
        registration_tracks: dict[str, dict[str, Any]] | None = None,
        exclude_alignment_well_uids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        matching = self.config.get("matching", {})
        stride = max(1, int(matching.get("log_sample_stride", 128)))
        max_distance = float(matching.get("max_horizontal_distance", 500.0))
        distance_scale = float(matching.get("distance_confidence_scale", 100.0))
        window_size = max(1, int(matching.get("seismic_window_samples", 32)))
        minimum_horizontal = float(
            matching.get("min_horizontal_confidence_for_training", 0.35)
        )
        minimum_vertical = float(
            matching.get("min_vertical_confidence_for_training", 0.55)
        )
        coordinate_reference = matching.get("coordinate_reference", {})
        coordinate_reference_verified = bool(
            coordinate_reference.get("verified", False)
        )
        seismic_sources = self._selected_seismic_sources(matching)
        spatial_aligner = build_spatial_aligner(matching).fit(seismic_sources)

        samples: list[dict[str, Any]] = []
        self.well_ties = []
        self.registration_tracks = {}
        self.visualization_registration_tracks = {}
        external_tracks = registration_tracks or {}
        excluded = set(exclude_alignment_well_uids or ())
        for entity in self.registry.entities.values():
            if not emit_samples and entity.well_uid in excluded:
                continue
            head = entity.preferred_head
            if not entity.logs or head is None or head.x is None or head.y is None:
                continue
            trajectory = entity.preferred_trajectory
            well_datum = self._well_datum(entity)
            for log in entity.logs:
                md_full = np.asarray(log.depth, dtype=float)
                if trajectory is not None:
                    tvd_full = interpolate_trajectory(
                        md_full, trajectory.md, trajectory.tvd
                    )
                    if trajectory.x is not None and trajectory.y is not None:
                        xs_full = interpolate_trajectory(
                            md_full, trajectory.md, trajectory.x
                        )
                        ys_full = interpolate_trajectory(
                            md_full, trajectory.md, trajectory.y
                        )
                    elif (
                        trajectory.source_crs
                        and coordinate_reference.get("crs")
                        and head.source_x is not None
                        and head.source_y is not None
                        and trajectory.coordinate_transform.get("operation")
                        == "pending_absolute_path_from_wellhead_and_offsets"
                    ):
                        source_head_x = float(head.source_x)
                        source_head_y = float(head.source_y)
                        try:
                            if (
                                head.source_crs
                                and canonical_crs_id(head.source_crs, field="井位源CRS")
                                != canonical_crs_id(
                                    trajectory.source_crs, field="轨迹源CRS"
                                )
                            ):
                                converted_head = transform_xy(
                                    [source_head_x],
                                    [source_head_y],
                                    source_crs=head.source_crs,
                                    target_crs=trajectory.source_crs,
                                )
                                source_head_x = float(converted_head.x[0])
                                source_head_y = float(converted_head.y[0])
                            projected_path = transform_offset_path(
                                source_head_x,
                                source_head_y,
                                trajectory.x_offset,
                                trajectory.y_offset,
                                source_crs=trajectory.source_crs,
                                target_crs=str(coordinate_reference["crs"]),
                            )
                        except CoordinateReferenceError as exc:
                            conflict = f"horizontal_crs_transform_failed:{exc}"
                            if conflict not in entity.conflicts:
                                entity.conflicts.append(conflict)
                            projected_x = np.full_like(trajectory.md, np.nan)
                            projected_y = np.full_like(trajectory.md, np.nan)
                        else:
                            projected_x = projected_path.x
                            projected_y = projected_path.y
                            trajectory.coordinate_transform = projected_path.provenance()
                            trajectory.horizontal_crs = projected_path.target_crs
                        xs_full = interpolate_trajectory(
                            md_full, trajectory.md, projected_x
                        )
                        ys_full = interpolate_trajectory(
                            md_full, trajectory.md, projected_y
                        )
                    else:
                        xs_full = head.x + interpolate_trajectory(
                            md_full, trajectory.md, trajectory.x_offset
                        )
                        ys_full = head.y + interpolate_trajectory(
                            md_full, trajectory.md, trajectory.y_offset
                        )
                    trajectory_source = trajectory.source
                    position_confidence = trajectory.confidence
                else:
                    # MD is along-hole length. Without a trajectory or an explicit
                    # depth tie it cannot be converted to TVD/elevation safely.
                    tvd_full = np.full_like(md_full, np.nan)
                    xs_full = np.full_like(md_full, head.x)
                    ys_full = np.full_like(md_full, head.y)
                    trajectory_source = "horizontal_only_missing_md_to_tvd_tie"
                    position_confidence = 0.5
                z_msl_full = (
                    absolute_elevation_from_tvd(
                        float(well_datum.absolute_elevation_m), tvd_full
                    )
                    if well_datum.absolute_elevation_m is not None
                    else np.full_like(tvd_full, np.nan)
                )
                depth_below_msl_full = -z_msl_full
                positions = np.isfinite(xs_full) & np.isfinite(ys_full)
                reference_match = None
                if np.any(positions):
                    reference_match = spatial_aligner.match(
                        float(np.median(xs_full[positions])),
                        float(np.median(ys_full[positions])),
                    )
                    if (
                        reference_match is not None
                        and reference_match.distance > max_distance
                    ):
                        reference_match = None
                seismic_datum = self._seismic_datum(
                    None if reference_match is None else reference_match.asset
                )
                seismic_time_meta = self.time_reference_metadata.get(
                    "" if reference_match is None else str(reference_match.asset.path),
                    TimeReferenceMetadata(),
                )
                depth_below_srd_full = (
                    depth_below_srd(
                        float(seismic_datum.absolute_elevation_m), z_msl_full
                    )
                    if seismic_datum.absolute_elevation_m is not None
                    else np.full_like(tvd_full, np.nan)
                )
                external_track = external_tracks.get(entity.well_uid)
                if external_track is None:
                    external_track = external_tracks.get(entity.canonical_name)
                if external_track is not None:
                    (
                        alignment,
                        times_full,
                        uncertainty_full,
                        quality_full,
                    ) = self._external_registration_alignment(external_track, md_full)
                else:
                    alignment = self._vertical_alignment(
                        entity,
                        log,
                        md_full,
                        tvd_full,
                        z_msl_full,
                        depth_below_msl_full,
                        depth_below_srd_full,
                        reference_match,
                        well_datum,
                        seismic_datum,
                        seismic_time_meta,
                    )
                    alignment_depth = self._alignment_depth_values(
                        alignment.depth_domain,
                        md_full,
                        tvd_full,
                        z_msl_full,
                        depth_below_msl_full,
                        depth_below_srd_full,
                    )
                    times_full = alignment.transform.depth_to_time(alignment_depth)
                    uncertainty_full = np.full(
                        md_full.shape,
                        (
                            float(alignment.uncertainty_ms)
                            if alignment.uncertainty_ms is not None
                            # Unknown uncertainty is not zero uncertainty.  V3
                            # preserves this distinction as a nullable value.
                            else np.nan
                        ),
                        dtype=float,
                    )
                    quality_full = np.full(
                        md_full.shape, float(alignment.confidence), dtype=float
                    )
                track_valid = (
                    np.isfinite(md_full)
                    & np.isfinite(xs_full)
                    & np.isfinite(ys_full)
                    & np.isfinite(times_full)
                )
                formal_track_eligible, trajectory_time_qc = (
                    self._trajectory_aware_sonic_formal_gate(
                        md_full,
                        tvd_full,
                        times_full,
                        track_valid,
                        method=alignment.method,
                        depth_domain=alignment.depth_domain,
                        has_measured_trajectory=trajectory is not None,
                    )
                )
                md_formal_eligible, registration_md_qc = (
                    self._registration_md_formal_gate(md_full)
                )
                formal_track_eligible = (
                    formal_track_eligible and md_formal_eligible
                )
                if trajectory_time_qc is not None:
                    alignment.diagnostics["trajectory_aware_time_branch_qc"] = (
                        trajectory_time_qc
                    )
                if registration_md_qc is not None:
                    alignment.diagnostics["registration_md_qc"] = registration_md_qc
                tie_metadata = alignment.to_metadata()
                self.well_ties.append(
                    {
                        "well_uid": entity.well_uid,
                        "well_name": entity.canonical_name,
                        "log_source": log.source,
                        "seismic_source": None
                        if reference_match is None
                        else str(reference_match.asset.path),
                        **tie_metadata,
                    }
                )

                # The Viewer has a deliberately wider admission policy than
                # the sealed registration/fusion product.  Prefer an existing
                # finite tie candidate; if none is display-safe, fall back to a
                # low-confidence TVD/constant-velocity mapping on the native
                # seismic sample axis.  Neither path can acquire downstream
                # eligibility through this store.
                visualization_times = np.asarray(times_full, dtype=float).copy()
                visualization_uncertainty = np.asarray(
                    uncertainty_full, dtype=float
                ).copy()
                visualization_quality = np.asarray(quality_full, dtype=float).copy()
                visualization_source = str(alignment.method)
                visualization_status = str(alignment.status)
                visualization_valid = track_valid.copy()
                if reference_match is not None and reference_match.reader.geometry is not None:
                    native_axis = np.asarray(
                        reference_match.reader.geometry.time_axis,
                        dtype=float,
                    )
                    finite_native_axis = native_axis[np.isfinite(native_axis)]
                    if finite_native_axis.size >= 2:
                        visualization_valid &= (
                            visualization_times >= float(np.min(finite_native_axis))
                        ) & (
                            visualization_times <= float(np.max(finite_native_axis))
                        )
                visualization_support = self._longest_visualization_support(
                    md_full,
                    visualization_valid,
                )
                visualization_policy = self._visualization_time_policy(
                    tvd_full,
                    visualization_times,
                    visualization_support,
                )
                visualization_fallback_used = False
                if visualization_policy is None:
                    fallback = self._constant_velocity_visualization_preview(
                        tvd_full,
                        reference_match,
                    )
                    if fallback is not None:
                        (
                            visualization_times,
                            visualization_uncertainty,
                            visualization_quality,
                            visualization_source,
                        ) = fallback
                        visualization_status = "visualization_preview"
                        visualization_valid = (
                            np.isfinite(md_full)
                            & np.isfinite(xs_full)
                            & np.isfinite(ys_full)
                            & np.isfinite(visualization_times)
                        )
                        visualization_support = self._longest_visualization_support(
                            md_full,
                            visualization_valid,
                        )
                        visualization_policy = self._visualization_time_policy(
                            tvd_full,
                            visualization_times,
                            visualization_support,
                        )
                        visualization_fallback_used = True

                if (
                    visualization_policy is not None
                    and int(np.sum(visualization_support)) >= 2
                ):
                    coordinate_contract = matching.get("coordinate_reference", {})
                    vertical_crs = self._vertical_crs()
                    trajectory_time_policy, trajectory_segment_ids = (
                        visualization_policy
                    )
                    preview_track = {
                        "well_name": entity.canonical_name,
                        "well_uid": entity.well_uid,
                        "md": md_full.round(8).tolist(),
                        "tvd": tvd_full.round(8).tolist(),
                        "zMsl": z_msl_full.round(8).tolist(),
                        "depthBelowSrd": depth_below_srd_full.round(8).tolist(),
                        "x": xs_full.round(8).tolist(),
                        "y": ys_full.round(8).tolist(),
                        "twtMean": visualization_times.round(8).tolist(),
                        "twtStd": visualization_uncertainty.round(8).tolist(),
                        "registrationQuality": visualization_quality.round(8).tolist(),
                        "validMask": visualization_support.tolist(),
                        "trajectoryTimePolicy": trajectory_time_policy,
                        "trajectorySegmentId": trajectory_segment_ids,
                        "registrationSource": visualization_source,
                        "registrationStatus": visualization_status,
                        "registrationCoverage": round(
                            float(np.mean(visualization_support)), 6
                        ),
                        "wellDepthDatum": well_datum.datum,
                        "wellReferenceElevationM": well_datum.absolute_elevation_m,
                        "horizontalCrsId": coordinate_contract.get("crs"),
                        "horizontalUnit": coordinate_contract.get("horizontal_unit"),
                        "horizontalAxisOrder": coordinate_contract.get("axis_order"),
                        "verticalCrsId": vertical_crs.get("id"),
                        "seismicSrdElevationM": seismic_datum.absolute_elevation_m,
                        "timeDomain": seismic_time_meta.time_domain,
                        "timeReference": seismic_time_meta.time_reference,
                        "correctionState": seismic_time_meta.correction_state,
                        "visualizationOnly": True,
                        "formalRegistration": False,
                        "registrationAccepted": False,
                        "inferenceEligible": False,
                        "fusionReady": False,
                        "supervisionEligible": False,
                        "trainingEligible": False,
                        "diagnostics": {
                            "candidate_only": True,
                            "visualization_gate": "well-seismic.viewer-candidate.v1",
                            "formal_track_eligible": bool(formal_track_eligible),
                            "formal_gate_reason": (
                                None
                                if trajectory_time_qc is None
                                else trajectory_time_qc.get("reason")
                            ),
                            "registration_md_qc": registration_md_qc,
                            "constant_velocity_fallback": visualization_fallback_used,
                            "source_alignment": tie_metadata,
                            "invalid_intervals_bridged": False,
                        },
                    }
                    current_preview = self.visualization_registration_tracks.get(
                        entity.canonical_name
                    )
                    current_preview_score = (
                        -1.0
                        if current_preview is None
                        else (
                            float(
                                np.nanmean(
                                    current_preview.get(
                                        "registrationQuality", [0.0]
                                    )
                                )
                            )
                            * float(
                                current_preview.get("registrationCoverage", 0.0)
                            )
                        )
                    )
                    preview_score = float(
                        np.nanmean(
                            visualization_quality[visualization_support]
                        )
                    ) * float(np.mean(visualization_support))
                    if preview_score > current_preview_score:
                        self.visualization_registration_tracks[
                            entity.canonical_name
                        ] = preview_track

                track_indices = np.flatnonzero(track_valid)
                if formal_track_eligible and track_indices.size >= 2:
                    coordinate_contract = matching.get("coordinate_reference", {})
                    vertical_crs = self._vertical_crs()
                    trajectory_time_policy = str(
                        (
                            external_track.get("trajectoryTimePolicy")
                            if external_track is not None
                            else (
                                None
                                if trajectory_time_qc is None
                                else trajectory_time_qc.get("trajectory_time_policy")
                            )
                        )
                        or STRICT_MD_TWT_POLICY
                    )
                    trajectory_segment_ids = (
                        external_track.get("trajectorySegmentId")
                        if external_track is not None
                        else (
                            None
                            if trajectory_time_qc is None
                            else trajectory_time_qc.get("trajectory_segment_ids")
                        )
                    )
                    track = {
                        "well_name": entity.canonical_name,
                        "well_uid": entity.well_uid,
                        # Registration V3 is an authoritative model input, not
                        # a UI payload.  Keep every LAS-depth row here; the API
                        # writes a separate <=240 point preview artifact.
                        "md": md_full.round(8).tolist(),
                        "tvd": tvd_full.round(8).tolist(),
                        "zMsl": z_msl_full.round(8).tolist(),
                        "depthBelowSrd": depth_below_srd_full.round(8).tolist(),
                        "x": xs_full.round(8).tolist(),
                        "y": ys_full.round(8).tolist(),
                        "twtMean": times_full.round(8).tolist(),
                        "twtStd": uncertainty_full.round(8).tolist(),
                        "registrationQuality": quality_full.round(8).tolist(),
                        "validMask": track_valid.tolist(),
                        "trajectoryTimePolicy": trajectory_time_policy,
                        "trajectorySegmentId": trajectory_segment_ids,
                        "registrationSource": alignment.method,
                        "registrationStatus": alignment.status,
                        "registrationCoverage": round(float(np.mean(track_valid)), 6),
                        "wellDepthDatum": well_datum.datum,
                        "wellReferenceElevationM": well_datum.absolute_elevation_m,
                        "horizontalCrsId": coordinate_contract.get("crs"),
                        "horizontalUnit": coordinate_contract.get("horizontal_unit"),
                        "horizontalAxisOrder": coordinate_contract.get("axis_order"),
                        "verticalCrsId": vertical_crs.get("id"),
                        "seismicSrdElevationM": seismic_datum.absolute_elevation_m,
                        "timeDomain": seismic_time_meta.time_domain,
                        "timeReference": seismic_time_meta.time_reference,
                        "correctionState": seismic_time_meta.correction_state,
                        "inferenceEligible": bool(
                            external_track.get("inferenceEligible", True)
                            if external_track is not None
                            else True
                        ),
                        "fusionReady": bool(
                            external_track.get(
                                "fusionReady",
                                external_track.get("trainingEligible", False),
                            )
                            if external_track is not None
                            else alignment.training_eligible
                        ),
                        "supervisionEligible": bool(
                            external_track.get(
                                "supervisionEligible",
                                external_track.get("trainingEligible", False),
                            )
                            if external_track is not None
                            else alignment.training_eligible
                        ),
                        "trainingEligible": bool(alignment.training_eligible),
                        "diagnostics": tie_metadata,
                    }
                    current = self.registration_tracks.get(entity.canonical_name)
                    current_score = (
                        -1.0
                        if current is None
                        else (
                            float(np.nanmean(current.get("registrationQuality", [0.0])))
                            * float(current.get("registrationCoverage", 0.0))
                        )
                    )
                    track_score = float(np.nanmean(quality_full[track_valid])) * float(
                        np.mean(track_valid)
                    )
                    if track_score > current_score:
                        self.registration_tracks[entity.canonical_name] = track

                if not emit_samples:
                    continue

                sample_indices = np.arange(0, len(md_full), stride, dtype=int)
                for local, original_index in enumerate(sample_indices):
                    x, y = xs_full[original_index], ys_full[original_index]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    nearest = spatial_aligner.match(
                        float(x),
                        float(y),
                        asset=None
                        if reference_match is None
                        else reference_match.asset,
                    )
                    if nearest is None:
                        continue
                    asset = nearest.asset
                    reader = nearest.reader
                    trace_index = nearest.trace_index
                    distance = nearest.distance
                    if distance > max_distance:
                        continue
                    geom = reader.geometry
                    assert geom is not None
                    seismic_window = None
                    seismic_coordinate = None
                    seismic_window_valid = False
                    time_value = times_full[original_index]
                    if np.isfinite(time_value) and float(geom.time_axis[0]) <= float(
                        time_value
                    ) <= float(geom.time_axis[-1]):
                        center = int(np.argmin(np.abs(geom.time_axis - time_value)))
                        start = center - window_size // 2
                        stop = start + window_size
                        if start < 0:
                            start, stop = 0, window_size
                        if stop > geom.samples_per_trace:
                            stop = geom.samples_per_trace
                            start = stop - window_size
                        if (
                            start >= 0
                            and stop <= geom.samples_per_trace
                            and stop - start == window_size
                        ):
                            window = reader.read_trace(
                                trace_index, slice(start, stop)
                            ).astype(float)
                            seismic_window_valid = bool(
                                window.size == window_size
                                and np.all(np.isfinite(window))
                            )
                            seismic_window = window.tolist()
                            seismic_coordinate = float(geom.time_axis[center])
                    features = {
                        name: float(values[original_index])
                        for name, values in log.curves.items()
                        if np.isfinite(values[original_index])
                    }
                    masks = {
                        name: bool(mask[original_index])
                        for name, mask in log.masks.items()
                    }
                    horizontal_confidence = float(
                        position_confidence
                        * geom.confidence
                        * math.exp(-distance / max(distance_scale, 1e-9))
                    )
                    training_eligible = bool(
                        coordinate_reference_verified
                        and well_datum.ready
                        and seismic_datum.ready
                        and self._vertical_crs()["status"] == "declared"
                        and seismic_time_meta.time_domain == "TWT"
                        and seismic_time_meta.time_reference == "SRD"
                        and seismic_time_meta.correction_state == "corrected_to_srd"
                        and seismic_window_valid
                        and alignment.training_eligible
                        and horizontal_confidence >= minimum_horizontal
                        and np.isfinite(quality_full[original_index])
                        and float(quality_full[original_index]) >= minimum_vertical
                    )
                    record = MatchRecord(
                        well_uid=entity.well_uid,
                        well_name=entity.canonical_name,
                        log_source=log.source,
                        seismic_source=str(asset.path),
                        md=float(md_full[original_index]),
                        tvd=float(tvd_full[original_index])
                        if np.isfinite(tvd_full[original_index])
                        else None,
                        z_msl_m=(
                            float(z_msl_full[original_index])
                            if np.isfinite(z_msl_full[original_index])
                            else None
                        ),
                        depth_below_msl_m=(
                            float(depth_below_msl_full[original_index])
                            if np.isfinite(depth_below_msl_full[original_index])
                            else None
                        ),
                        depth_below_srd_m=(
                            float(depth_below_srd_full[original_index])
                            if np.isfinite(depth_below_srd_full[original_index])
                            else None
                        ),
                        x=float(x),
                        y=float(y),
                        trace_index=int(trace_index),
                        inline=int(geom.inline[trace_index])
                        if geom.inline is not None
                        else None,
                        crossline=int(geom.crossline[trace_index])
                        if geom.crossline is not None
                        else None,
                        distance=float(distance),
                        seismic_coordinate=seismic_coordinate,
                        horizontal_confidence=horizontal_confidence,
                        vertical_method=alignment.method,
                        vertical_confidence=(
                            float(quality_full[original_index])
                            if np.isfinite(quality_full[original_index])
                            else 0.0
                        ),
                        well_features=features,
                        well_mask=masks,
                        seismic_window=seismic_window,
                        provenance={
                            "well_head": head.source,
                            "trajectory": trajectory_source,
                            "log": log.source,
                            "seismic": str(asset.path),
                            "curve_mapping_version": "curve_knowledge_v1",
                            "segy_profile": geom.profile,
                            "segy_revision": geom.revision,
                            "log_asset_id": self.asset_by_path.get(log.source).asset_id
                            if self.asset_by_path.get(log.source)
                            else None,
                            "log_stage": self.asset_by_path.get(log.source).stage
                            if self.asset_by_path.get(log.source)
                            else "UNKNOWN",
                            "log_version": self.asset_by_path.get(log.source).version
                            if self.asset_by_path.get(log.source)
                            else log.version,
                            "seismic_asset_id": asset.asset_id,
                            "seismic_stage": asset.stage,
                            "seismic_version": asset.version,
                            "neighbor_trace_indices": list(
                                nearest.neighbor_trace_indices
                            ),
                            "neighbor_distances": list(nearest.neighbor_distances),
                            "neighbor_weights": list(nearest.interpolation_weights),
                            "coordinate_reference": {
                                "verified": coordinate_reference_verified,
                                "crs": coordinate_reference.get("crs"),
                                "horizontal_unit": coordinate_reference.get(
                                    "horizontal_unit", "m"
                                ),
                            },
                            "vertical_alignment": tie_metadata,
                            "vertical_datum": {
                                "canonical_reference": "MSL",
                                "positive_direction": "up",
                                "well": well_datum.to_dict(),
                                "seismic": seismic_datum.to_dict(),
                                "seismic_time_reference": seismic_time_meta.to_dict(),
                            },
                        },
                        vertical_status=alignment.status,
                        vertical_uncertainty_ms=(
                            float(uncertainty_full[original_index])
                            if np.isfinite(uncertainty_full[original_index])
                            else None
                        ),
                        seismic_window_valid=seismic_window_valid,
                        coordinate_reference_verified=coordinate_reference_verified,
                        vertical_datum_verified=bool(
                            well_datum.ready
                            and seismic_datum.ready
                            and self._vertical_crs()["status"] == "declared"
                            and seismic_time_meta.time_domain == "TWT"
                            and seismic_time_meta.time_reference == "SRD"
                            and seismic_time_meta.correction_state == "corrected_to_srd"
                        ),
                        training_eligible=training_eligible,
                    )
                    samples.append(record.to_dict())
        self.samples = samples
        return samples

    def calibrate_wells(
        self,
        *,
        exclude_well_uids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run horizontal registration and vertical well tie before fusion.

        This deliberately emits no multimodal training/inference samples.  It
        freezes only per-well registration evidence so the next stage can make
        an explicit quality decision.
        """
        self.build_samples(
            emit_samples=False,
            exclude_alignment_well_uids=exclude_well_uids,
        )
        return list(self.well_ties)

    def _selected_seismic_sources(
        self, matching: dict[str, Any]
    ) -> list[tuple[Any, SegyReader]]:
        sources = list(self.seismic)
        selection = matching.get("survey_selection", {})
        asset_id = selection.get("asset_id")
        path_contains = str(selection.get("path_contains") or "").strip().lower()
        if asset_id:
            sources = [item for item in sources if item[0].asset_id == asset_id]
        if path_contains:
            sources = [
                item for item in sources if path_contains in str(item[0].path).lower()
            ]
        mode = str(selection.get("mode", "nearest_3d")).lower()
        if not sources or mode == "all":
            return sources

        def is_3d(item: tuple[Any, SegyReader]) -> bool:
            geometry = item[1].geometry
            if (
                geometry is None
                or geometry.inline is None
                or geometry.crossline is None
            ):
                return False
            inline = geometry.inline[np.isfinite(geometry.inline)]
            crossline = geometry.crossline[np.isfinite(geometry.crossline)]
            return len(np.unique(inline)) > 1 and len(np.unique(crossline)) > 1

        candidates = [item for item in sources if is_3d(item)] or sources
        if mode == "nearest_3d":
            return candidates
        return [
            max(
                candidates,
                key=lambda item: item[1].geometry.trace_count
                if item[1].geometry
                else 0,
            )
        ]

    @staticmethod
    def _composite_trace(match: Any) -> np.ndarray:
        indices = match.neighbor_trace_indices or (match.trace_index,)
        weights = np.asarray(match.interpolation_weights or (1.0,), dtype=float)
        stack = np.vstack(
            [match.reader.read_trace(int(index)).astype(float) for index in indices]
        )
        finite = np.isfinite(stack)
        weighted = np.where(finite, stack, 0.0) * weights[:, None]
        denominator = np.sum(finite * weights[:, None], axis=0)
        return np.divide(
            np.sum(weighted, axis=0),
            denominator,
            out=np.zeros(stack.shape[1], dtype=float),
            where=denominator > 0,
        )

    @staticmethod
    def _alignment_depth_values(
        domain: str,
        md: np.ndarray,
        tvd: np.ndarray,
        z_msl_m: np.ndarray,
        depth_below_msl_m: np.ndarray,
        depth_below_srd_m: np.ndarray,
    ) -> np.ndarray:
        return {
            "md": md,
            "tvd": tvd,
            "z_msl_m": z_msl_m,
            "depth_below_msl_m": depth_below_msl_m,
            "depth_below_srd_m": depth_below_srd_m,
        }.get(str(domain).lower(), z_msl_m)

    @staticmethod
    def _longest_visualization_support(
        md: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Keep one honest, contiguous MD interval for a Viewer candidate.

        Preview interpolation must never bridge an invalid source interval or
        silently repair duplicate/reversed MD.  Unlike formal V3, the Viewer is
        allowed to show the longest independently valid interval and disclose
        the reduced coverage.
        """

        md_values = np.asarray(md, dtype=float)
        candidate = np.asarray(valid, dtype=bool).copy()
        if md_values.ndim != 1 or candidate.shape != md_values.shape:
            return np.zeros(md_values.shape, dtype=bool)
        candidate &= np.isfinite(md_values)
        indices = np.flatnonzero(candidate)
        if indices.size < 2:
            return np.zeros(md_values.shape, dtype=bool)

        breaks = np.flatnonzero(
            (np.diff(indices) != 1)
            | (np.diff(md_values[indices]) <= 0.0)
        ) + 1
        runs = [run for run in np.split(indices, breaks) if run.size >= 2]
        support = np.zeros(md_values.shape, dtype=bool)
        if not runs:
            return support
        selected = max(
            runs,
            key=lambda run: (
                int(run.size),
                float(md_values[int(run[-1])] - md_values[int(run[0])]),
            ),
        )
        support[selected] = True
        return support

    @staticmethod
    def _visualization_time_policy(
        tvd: np.ndarray,
        twt: np.ndarray,
        support: np.ndarray,
    ) -> tuple[str, list[int | None]] | None:
        """Validate time direction for display without promoting the result."""

        indices = np.flatnonzero(np.asarray(support, dtype=bool))
        if indices.size < 2:
            return None
        tvd_values = np.asarray(tvd, dtype=float)[indices]
        twt_values = np.asarray(twt, dtype=float)[indices]
        if not np.all(np.isfinite(twt_values)):
            return None

        # A strictly increasing time candidate is displayable even when TVD is
        # unavailable.  If measured TVD reverses or flattens, however, require
        # stationwise sign coherence so an arbitrary time reversal cannot be
        # painted as a physical horizontal-well trajectory.
        twt_increasing = bool(np.all(np.diff(twt_values) > 0.0))
        tvd_finite = bool(np.all(np.isfinite(tvd_values)))
        tvd_increasing = bool(
            tvd_finite and np.all(np.diff(tvd_values) > 0.0)
        )
        if twt_increasing and (not tvd_finite or tvd_increasing):
            return STRICT_MD_TWT_POLICY, [None] * len(support)
        if not tvd_finite:
            return None
        try:
            stationwise = validate_trajectory_stationwise_twt(
                tvd_values,
                twt_values,
                identity="visualization-only trajectory time candidate",
            )
        except ValueError:
            return None
        segment_ids: list[int | None] = [None] * len(support)
        for point_index, segment_id in zip(
            indices,
            stationwise["trajectory_segment_ids"],
        ):
            segment_ids[int(point_index)] = int(segment_id)
        return TRAJECTORY_STATIONWISE_TWT_POLICY, segment_ids

    def _constant_velocity_visualization_preview(
        self,
        tvd: np.ndarray,
        reference_match: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        """Build a low-confidence native-axis preview, never a formal tie.

        The mapping is only available when a measured TVD path can be located
        in a seismic survey with a finite native sample axis.  Values outside
        the seismic time extent remain invalid rather than being clipped.
        """

        if reference_match is None or reference_match.reader.geometry is None:
            return None
        time_axis = np.asarray(
            reference_match.reader.geometry.time_axis,
            dtype=float,
        )
        finite_axis = time_axis[np.isfinite(time_axis)]
        tvd_values = np.asarray(tvd, dtype=float)
        finite_tvd = tvd_values[np.isfinite(tvd_values)]
        if finite_axis.size < 2 or finite_tvd.size < 2:
            return None

        vertical = self.config.get("matching", {}).get("vertical", {})
        relative = vertical.get("relative_sonic_without_time_depth", {})
        velocity_m_s = float(
            relative.get(
                "visualization_preview_velocity_m_s",
                relative.get("initial_velocity_m_s", 3000.0),
            )
        )
        if not math.isfinite(velocity_m_s) or velocity_m_s <= 0.0:
            return None
        axis_min = float(np.min(finite_axis))
        axis_max = float(np.max(finite_axis))
        depth_origin = min(0.0, float(np.min(finite_tvd)))
        twt = axis_min + 2000.0 * (tvd_values - depth_origin) / velocity_m_s
        valid = (
            np.isfinite(tvd_values)
            & np.isfinite(twt)
            & (twt >= axis_min)
            & (twt <= axis_max)
        )
        if int(np.sum(valid)) < 2:
            return None
        axis_span = max(axis_max - axis_min, 1.0)
        uncertainty = np.full(tvd_values.shape, max(250.0, axis_span * 0.15))
        quality = np.full(tvd_values.shape, 0.12)
        return twt, uncertainty, quality, "constant_velocity_visualization_preview"

    @staticmethod
    def _registration_md_formal_gate(
        md: np.ndarray,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Fail-close one well whose source-order MD cannot satisfy V3.

        Registration V3 requires every emitted row to carry a finite,
        strictly increasing MD.  A non-finite or duplicate source MD cannot be
        repaired without inventing location along the borehole.  Exclude that
        well from the formal product while allowing independently valid wells
        in the same task to continue.  No row is sorted, dropped, interpolated
        or numerically nudged here.
        """

        md_values = np.asarray(md, dtype=float)
        if md_values.ndim != 1:
            return False, {
                "policy": "exclude_invalid_source_md_track_from_formal_v3_v1",
                "reason": "md_not_one_dimensional",
                "formal_track_eligible": False,
                "formal_track_excluded": True,
                "source_shape": list(md_values.shape),
                "rows_reordered": False,
                "rows_dropped": False,
                "md_values_rewritten": False,
                "downstream_gap_interpolation_allowed": False,
            }

        rounded_md = np.round(md_values, 8)
        finite = np.isfinite(rounded_md)
        nonfinite_indices = np.flatnonzero(~finite)
        nonincreasing_indices = (
            np.flatnonzero(np.diff(rounded_md) <= 0.0)
            if np.all(finite) and rounded_md.size >= 2
            else np.asarray([], dtype=int)
        )
        if (
            rounded_md.size >= 2
            and nonfinite_indices.size == 0
            and nonincreasing_indices.size == 0
        ):
            return True, None

        if nonfinite_indices.size:
            reason = "nonfinite_source_md"
        elif rounded_md.size < 2:
            reason = "fewer_than_two_source_md_rows"
        else:
            reason = "non_strictly_increasing_source_md_at_v3_precision"
        identity_values = [
            float(value) if math.isfinite(float(value)) else None
            for value in rounded_md
        ]
        return False, {
            "policy": "exclude_invalid_source_md_track_from_formal_v3_v1",
            "reason": reason,
            "formal_track_eligible": False,
            "formal_track_excluded": True,
            "source_row_count": int(rounded_md.size),
            "finite_md_count": int(np.sum(finite)),
            "nonfinite_md_count": int(nonfinite_indices.size),
            "first_nonfinite_source_row_index": (
                int(nonfinite_indices[0]) if nonfinite_indices.size else None
            ),
            "nonincreasing_interval_count": int(nonincreasing_indices.size),
            "first_nonincreasing_left_source_row_index": (
                int(nonincreasing_indices[0])
                if nonincreasing_indices.size
                else None
            ),
            "product_precision_decimals": 8,
            "raw_candidate_identity_sha256": canonical_sha256(identity_values),
            "rows_reordered": False,
            "rows_dropped": False,
            "md_values_rewritten": False,
            "downstream_gap_interpolation_allowed": False,
            "audit_meaning": (
                "源MD无法满足正式V3的有限且严格递增合同；该井整井排除，"
                "同批其他合格井继续，禁止排序、删行、插值或数值修补"
            ),
        }

    @staticmethod
    def _trajectory_aware_sonic_formal_gate(
        md: np.ndarray,
        tvd: np.ndarray,
        twt: np.ndarray,
        base_valid: np.ndarray,
        *,
        method: str,
        depth_domain: str,
        has_measured_trajectory: bool,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Select strict or trajectory-stationwise time semantics per sonic tie.

        A horizontal well can legitimately cross 90 degrees inclination, so
        TVD and a TVD-domain sonic time candidate can locally reverse while MD
        remains strictly increasing.  Such a curve is a valid *stationwise*
        trajectory-time mapping, but it is not a globally monotonic MD--TWT
        transform.  Preserve it only when every source row is valid and every
        TWT direction change follows the unchanged measured TVD direction.

        This remains limited to sonic candidates evaluated on a measured
        trajectory. Provided/checkshot, external and learned products keep
        their own policies. No values are sorted, accumulated, nudged, masked,
        or rewritten here.
        """

        valid = np.asarray(base_valid, dtype=bool).copy()
        if not (
            has_measured_trajectory
            and str(depth_domain).strip().casefold() == "tvd"
            and str(method).strip().casefold().startswith("sonic_integrated")
        ):
            return True, None

        md_values = np.asarray(md, dtype=float)
        tvd_values = np.asarray(tvd, dtype=float)
        # Apply the exact product precision here so a sub-precision increment
        # cannot become a duplicate only after V3 serialisation.
        twt_values = np.round(np.asarray(twt, dtype=float), 8)
        if not (
            md_values.ndim == 1
            and md_values.shape == tvd_values.shape
            and md_values.shape == twt_values.shape
            and md_values.shape == valid.shape
        ):
            raise ValueError(
                "trajectory-aware sonic QC columns must be equal 1-D arrays"
            )

        candidates = np.flatnonzero(
            valid
            & np.isfinite(md_values)
            & np.isfinite(tvd_values)
            & np.isfinite(twt_values)
        )
        if candidates.size < 2:
            return True, None
        candidate_md = md_values[candidates]
        if np.any(np.diff(candidate_md) <= 0.0):
            # Source-order MD is an upstream contract. Never reorder it here
            # to make a malformed track appear physically valid.
            return True, None

        candidate_tvd = tvd_values[candidates]
        candidate_twt = twt_values[candidates]
        tvd_diff = np.diff(candidate_tvd)
        twt_diff = np.diff(candidate_twt)
        nonincreasing = (tvd_diff <= 0.0) | (twt_diff <= 0.0)
        if not np.any(nonincreasing):
            return True, None
        affected = np.flatnonzero(nonincreasing)
        raw_identity = canonical_sha256(
            {
                "md_m": md_values[candidates].round(8).tolist(),
                "tvd_m": candidate_tvd.round(8).tolist(),
                "twt_ms": candidate_twt.tolist(),
                "valid_point_indices": candidates.tolist(),
            }
        )
        audit = {
            "scope": "trajectory_aware_sonic_tvd_candidate_only",
            "input_valid_count": int(np.sum(valid)),
            "nonincreasing_interval_count": int(np.sum(nonincreasing)),
            "decreasing_tvd_interval_count": int(np.sum(tvd_diff < 0.0)),
            "duplicate_tvd_interval_count": int(np.sum(tvd_diff == 0.0)),
            "decreasing_twt_interval_count": int(np.sum(twt_diff < 0.0)),
            "duplicate_twt_interval_count": int(np.sum(twt_diff == 0.0)),
            "first_affected_left_point_index": int(candidates[affected[0]]),
            "last_affected_right_point_index": int(candidates[affected[-1] + 1]),
            "raw_candidate_point_count": int(candidates.size),
            "raw_candidate_md_range_m": [
                float(candidate_md[0]),
                float(candidate_md[-1]),
            ],
            "raw_candidate_tvd_range_m": [
                float(np.min(candidate_tvd)),
                float(np.max(candidate_tvd)),
            ],
            "raw_candidate_twt_range_ms": [
                float(np.min(candidate_twt)),
                float(np.max(candidate_twt)),
            ],
            "raw_candidate_identity_sha256": raw_identity,
            "source_trajectory_and_horizontal_geometry_preserved": True,
            "raw_md_tvd_twt_values_preserved": True,
            "rows_reordered": False,
            "twt_values_rewritten": False,
        }
        contiguous_valid_support = bool(
            candidates.size >= 2
            and np.array_equal(
                candidates,
                np.arange(int(candidates[0]), int(candidates[-1]) + 1),
            )
        )
        if not contiguous_valid_support:
            return False, {
                **audit,
                "policy": "exclude_gapped_trajectory_stationwise_sonic_track_v1",
                "reason": "trajectory_stationwise_valid_support_contains_internal_gap",
                "formal_track_eligible": False,
                "formal_track_excluded": True,
                "formal_excluded_point_count": int(np.sum(valid)),
                "downstream_gap_interpolation_allowed": False,
                "audit_meaning": (
                    "逐站点时间轨迹存在无效行；整井时间产品失败关闭，禁止跨缺口插值"
                ),
            }
        try:
            stationwise = validate_trajectory_stationwise_twt(
                candidate_tvd,
                candidate_twt,
                identity="trajectory-aware sonic candidate",
            )
        except ValueError as exc:
            return False, {
                **audit,
                "policy": "exclude_incoherent_trajectory_stationwise_sonic_track_v1",
                "reason": "twt_direction_does_not_follow_measured_tvd",
                "validation_error": str(exc),
                "formal_track_eligible": False,
                "formal_track_excluded": True,
                "formal_excluded_point_count": int(np.sum(valid)),
                "downstream_gap_interpolation_allowed": False,
                "audit_meaning": (
                    "时间回折与实测TVD方向不一致；保留原始审计但不封存时间轨迹"
                ),
            }
        trajectory_segment_ids: list[int | None] = [None] * md_values.size
        for point_index, segment_id in zip(
            candidates,
            stationwise["trajectory_segment_ids"],
        ):
            trajectory_segment_ids[int(point_index)] = int(segment_id)
        return True, {
            **audit,
            "policy": "preserve_trajectory_stationwise_sonic_track_v1",
            "reason": "measured_tvd_reversal_preserved_stationwise",
            "trajectory_time_policy": TRAJECTORY_STATIONWISE_TWT_POLICY,
            "trajectory_segment_ids": trajectory_segment_ids,
            "trajectory_segment_count": stationwise["trajectory_segment_count"],
            "tvd_reversal_interval_count": stationwise[
                "tvd_reversal_interval_count"
            ],
            "formal_track_eligible": True,
            "formal_track_excluded": False,
            "formal_excluded_point_count": 0,
            "downstream_gap_interpolation_allowed": True,
            "interpolation_domain": "piecewise_values_over_strictly_increasing_md",
            "audit_meaning": (
                "按递增MD保留完整逐站点TVD与时间候选；局部时间回折仅跟随真实TVD回折，"
                "不排序、不修单调、不生成监督标签"
            ),
        }

    def _well_reference_elevation(self, entity: Any, datum: str | None) -> float | None:
        if datum is None:
            return None
        canonical = str(datum).upper()
        candidates = [
            item
            for item in self._observations_for_well(entity)
            if item.datum == canonical
            and item.absolute_elevation_m is not None
            and not item.review_required
        ]
        if not candidates:
            return None
        tolerance = float(self._datum_config().get("consistency_tolerance_m", 0.5))
        values = [float(item.absolute_elevation_m) for item in candidates]
        if max(values) - min(values) > tolerance:
            return None
        return float(
            max(candidates, key=lambda item: item.confidence).absolute_elevation_m
        )

    def _time_depth_absolute_elevation(
        self,
        entity: Any,
        domain: str,
        depth_m: np.ndarray,
        well_datum: ResolvedVerticalDatum,
        *,
        depth_datum: str | None,
        depth_convention: str | None,
        md_offset_to_trajectory_m: float | None,
    ) -> tuple[np.ndarray | None, str]:
        normalized = str(domain).strip().lower()
        if normalized == "tvd":
            elevation = self._well_reference_elevation(entity, depth_datum)
            if elevation is None:
                return None, "TVD深度起算面未声明或高程未确认"
            return absolute_elevation_from_tvd(
                elevation, depth_m
            ), f"TVD({depth_datum})->z_msl_m"
        if normalized == "tvdss":
            if depth_convention == "depth_below_msl_positive_down":
                return -depth_m, "输入TVDSS已声明为depth_below_msl_m"
            if depth_convention == "elevation_positive_up":
                return depth_m, "输入TVDSS已声明为z_msl_m"
            return None, "TVDSS符号约定未声明，禁止自动转换"
        if normalized in {"z_msl_m", "elevation"}:
            return depth_m, "输入已是z_msl_m"
        if normalized == "depth_below_msl_m":
            return -depth_m, "depth_below_msl_m->z_msl_m"
        if normalized == "md":
            trajectory = entity.preferred_trajectory
            if trajectory is None:
                return None, "MD不能按高程平移：缺少真实轨迹或人工深度tie"
            if depth_datum is None:
                return None, "MD深度起算面未声明"
            if depth_datum != well_datum.datum and md_offset_to_trajectory_m is None:
                return None, "MD起算面与轨迹不一致，缺少md_offset_to_trajectory_m"
            query_md = depth_m + float(md_offset_to_trajectory_m or 0.0)
            tvd = interpolate_trajectory(query_md, trajectory.md, trajectory.tvd)
            if int(np.sum(np.isfinite(tvd))) < 2:
                return None, "轨迹没有可用TVD控制点，禁止把MD自动当作TVD"
            if well_datum.absolute_elevation_m is None:
                return None, "轨迹垂向基准高程未确认"
            return absolute_elevation_from_tvd(
                float(well_datum.absolute_elevation_m), tvd
            ), "MD->真实轨迹TVD->z_msl_m"
        return None, f"unsupported_depth_domain:{domain}"

    @staticmethod
    def _time_depth_source_kind(table: Any) -> str:
        """Resolve an explicit scientific authority tier for one input table."""

        declared = (
            str(getattr(table, "source_kind", "unknown") or "unknown")
            .strip()
            .casefold()
        )
        declared = re.sub(r"[\s-]+", "_", declared)
        aliases = {
            "check_shot": "checkshot",
            "checkshots": "checkshot",
            "vertical_seismic_profile": "vsp",
            "checkshot_and_vsp": "checkshot_vsp",
            "las_twt": "well_twt_curve",
            "twt_curve": "well_twt_curve",
            "provided": "provided_time_depth",
            "time_depth": "provided_time_depth",
        }
        declared = aliases.get(declared, declared)
        if declared in {
            "checkshot",
            "vsp",
            "checkshot_vsp",
            "provided_time_depth",
            "well_twt_curve",
            "sonic",
        }:
            return declared

        source = str(getattr(table, "source", "")).casefold()
        if re.search(
            r"(?:^|[^a-z])vsp(?:[^a-z]|$)|vertical[ _-]*seismic[ _-]*profile", source
        ):
            return "vsp"
        if re.search(r"check[ _-]*shots?|checkshots?|校深", source):
            return "checkshot"
        if "#twt" in source or (source.endswith(".las") and "twt" in source):
            return "well_twt_curve"
        return "provided_time_depth"

    def _normalised_time_depth_candidates(
        self,
        entity: Any,
        well_datum: ResolvedVerticalDatum,
        seismic_datum: ResolvedVerticalDatum,
    ) -> tuple[
        list[ProvidedTimeDepthCandidate],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Canonicalise every table before scientific QC and authority selection."""

        candidates: list[ProvidedTimeDepthCandidate] = []
        metadata: dict[str, dict[str, Any]] = {}
        rejections: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}

        def reject(source: str, source_kind: str, stage: str, reason: str) -> None:
            rejections.append(
                {
                    "source": source,
                    "source_kind": source_kind,
                    "stage": stage,
                    "reason": reason,
                }
            )

        for table in entity.time_depth:
            source = str(table.source)
            source_kind = self._time_depth_source_kind(table)
            source_counts[source] = source_counts.get(source, 0) + 1
            suffix = source_counts[source]
            candidate_source = source if suffix == 1 else f"{source}#candidate-{suffix}"
            depth = np.asarray(table.depth, dtype=float)
            time = np.asarray(table.time, dtype=float)
            if depth.ndim != 1 or time.ndim != 1 or depth.shape != time.shape:
                reject(source, source_kind, "shape", "时深控制点必须是两个等长一维数组")
                continue
            try:
                depth = length_to_metres(depth, table.depth_unit)
                time = time_to_milliseconds(time, table.time_unit)
            except ValueError as exc:
                reject(source, source_kind, "unit_normalization", str(exc))
                continue

            control_elevation, conversion = self._time_depth_absolute_elevation(
                entity,
                table.depth_domain,
                depth,
                well_datum,
                depth_datum=table.depth_datum,
                depth_convention=table.depth_convention,
                md_offset_to_trajectory_m=table.md_offset_to_trajectory_m,
            )
            if control_elevation is None:
                reject(source, source_kind, "vertical_normalization", conversion)
                continue

            time_domain = str(table.time_domain).strip().upper()
            if time_domain not in {"OWT", "TWT"}:
                reject(source, source_kind, "time_normalization", "time_domain_unknown")
                continue
            correction_state = str(table.correction_state).strip().casefold()
            time_reference = str(table.time_reference).strip().upper()
            time_normalization: list[str] = []
            if correction_state == "corrected_to_srd":
                if time_reference != "SRD":
                    reject(
                        source,
                        source_kind,
                        "time_normalization",
                        "corrected_to_srd但time_reference不是SRD",
                    )
                    continue
                time_normalization.append("已声明corrected_to_srd，未重复静校正")
            elif correction_state == "uncorrected":
                source_elevation = self._well_reference_elevation(
                    entity, time_reference
                )
                if source_elevation is None or table.replacement_velocity_mps is None:
                    reject(
                        source,
                        source_kind,
                        "time_normalization",
                        "未校正时间缺少参考面高程或替换速度",
                    )
                    continue
                time = correct_time_to_srd(
                    time,
                    source_elevation_msl_m=source_elevation,
                    srd_elevation_msl_m=float(seismic_datum.absolute_elevation_m),
                    replacement_velocity_mps=float(table.replacement_velocity_mps),
                    time_domain=time_domain,
                )
                time_normalization.append("仅执行近地表datum静校正到SRD")
            else:
                reject(
                    source,
                    source_kind,
                    "time_normalization",
                    "time_correction_state_unknown",
                )
                continue
            if time_domain == "OWT":
                time = time * 2.0
                time_normalization.append("OWT->TWT")

            candidate = ProvidedTimeDepthCandidate(
                source=candidate_source,
                source_kind=source_kind,
                depth_m=np.asarray(control_elevation, dtype=float),
                twt_ms=np.asarray(time, dtype=float),
                depth_direction="positive_up",
                confidence=float(table.confidence),
                metadata_score=1.0,
                uncertainty_ms=getattr(table, "uncertainty_ms", None),
            )
            candidates.append(candidate)
            metadata[candidate_source] = {
                "source": source,
                "source_kind": source_kind,
                "source_depth_domain": table.depth_domain,
                "normalization": conversion,
                "depth_unit": "m",
                "time_unit": "ms",
                "time_reference": "SRD",
                "time_domain": "TWT",
                "correction_state": "corrected_to_srd",
                "time_reference_normalization": time_normalization,
            }
        return candidates, metadata, rejections

    def _vertical_alignment(
        self,
        entity: Any,
        log: Any,
        md: np.ndarray,
        tvd: np.ndarray,
        z_msl_m: np.ndarray,
        depth_below_msl_m: np.ndarray,
        depth_below_srd_m: np.ndarray,
        reference_match: Any | None,
        well_datum: ResolvedVerticalDatum,
        seismic_datum: ResolvedVerticalDatum,
        seismic_time_meta: TimeReferenceMetadata,
    ) -> TimeDomainAlignment:
        datum_metadata = {
            "canonical_reference": "MSL",
            "positive_direction": "up",
            "well_datum": well_datum.datum,
            "well_datum_elevation_m": well_datum.absolute_elevation_m,
            "well_datum_source": well_datum.source,
            "seismic_datum": seismic_datum.datum,
            "seismic_datum_elevation_m": seismic_datum.absolute_elevation_m,
            "seismic_datum_source": seismic_datum.source,
            "seismic_time_reference": seismic_time_meta.to_dict(),
        }
        require_datums = bool(
            self._datum_config().get("require_resolved_well_and_seismic_datums", True)
        )
        vertical_crs_ready = self._vertical_crs()["status"] == "declared"
        datum_reasons = []
        if not vertical_crs_ready:
            datum_reasons.append("vertical_crs_LOCAL_MSL_survey_id_unresolved")
        if not well_datum.ready:
            datum_reasons.append("well_datum_unresolved_or_conflicting")
        if not seismic_datum.ready:
            datum_reasons.append("seismic_srd_unresolved_or_conflicting")

        # A survey without checkshot/time-depth controls can still produce an
        # auditable DT/VP synthetic-tie candidate.  This happens before strict
        # datum acceptance because the correlation lives on the SEG-Y sample
        # axis; the candidate remains explicitly non-training-eligible.
        vertical_options = self.config.get("matching", {}).get("vertical", {})
        relative_options = vertical_options.get("relative_sonic_without_time_depth", {})
        if (
            not entity.time_depth
            and reference_match is not None
            and bool(relative_options.get("enabled", True))
        ):
            valid_tvd = tvd[np.isfinite(tvd)]
            geometry = reference_match.reader.geometry
            if valid_tvd.size >= 3 and geometry is not None:
                initial_velocity = float(
                    relative_options.get("initial_velocity_m_s", 3000.0)
                )
                first_depth = max(0.0, float(np.min(valid_tvd)))
                initial_twt = 2000.0 * first_depth / max(initial_velocity, 1.0)
                sonic_options = {
                    **vertical_options.get("sonic", {}),
                    **relative_options,
                    "reference_depth_m": first_depth,
                    "datum_time_ms": initial_twt,
                    "datum_is_explicit": False,
                    "replacement_velocity_m_s": None,
                }
                try:
                    candidate = build_sonic_time_domain_alignment(
                        tvd,
                        log.curves,
                        self._composite_trace(reference_match),
                        geometry.time_axis,
                        sonic_options,
                    )
                except (ValueError, FloatingPointError) as exc:
                    log.issues.append(
                        f"relative_sonic_well_tie_failed:{type(exc).__name__}:{exc}"
                    )
                    candidate = None
                if candidate is not None:
                    candidate.training_eligible = False
                    candidate.depth_domain = "tvd"
                    candidate.diagnostics.update(
                        {
                            **datum_metadata,
                            "candidate_only": True,
                            "blocking_reasons": datum_reasons,
                            "absolute_time_prior": "average_velocity_to_first_sonic_sample",
                            "initial_velocity_m_s": initial_velocity,
                            "initial_twt_ms": initial_twt,
                            "registration_policy": "no_time_depth_relative_sonic_then_bounded_static_tie",
                        }
                    )
                    return candidate

        if require_datums and datum_reasons:
            return TimeDomainAlignment(
                transform=NoDepthTimeTransform(),
                status="datum_unresolved",
                method="blocked_vertical_datum",
                confidence=0.0,
                uncertainty_ms=None,
                training_eligible=False,
                depth_domain="z_msl_m",
                diagnostics={**datum_metadata, "blocking_reasons": datum_reasons},
            )
        seismic_time_ready = bool(
            seismic_time_meta.time_domain == "TWT"
            and seismic_time_meta.time_reference == "SRD"
            and seismic_time_meta.correction_state == "corrected_to_srd"
        )
        if not seismic_time_ready:
            return TimeDomainAlignment(
                transform=NoDepthTimeTransform(),
                status="time_reference_unresolved",
                method="blocked_seismic_time_reference",
                confidence=0.0,
                uncertainty_ms=None,
                training_eligible=False,
                depth_domain="z_msl_m",
                diagnostics={
                    **datum_metadata,
                    "blocking_reasons": ["seismic_time_must_be_TWT_corrected_to_SRD"],
                },
            )

        provided_fallback_diagnostics: dict[str, Any] | None = None
        if entity.time_depth:
            candidates, candidate_metadata, normalization_rejections = (
                self._normalised_time_depth_candidates(
                    entity, well_datum, seismic_datum
                )
            )
            qc_config = dict(vertical_options.get("provided_time_depth_qc", {}))
            supported_qc_options = {
                "equal_authority_conflict_ms",
                "comparison_grid_points",
                "minimum_points",
                "duplicate_depth_tolerance_m",
                "duplicate_time_conflict_ms",
                "minimum_interval_velocity_m_s",
                "maximum_interval_velocity_m_s",
                "repair_isolated_outliers",
                "isolated_outlier_threshold_ms",
                "maximum_repair_fraction",
            }
            qc_options = {
                key: value
                for key, value in qc_config.items()
                if key in supported_qc_options
            }
            selection = select_authoritative_time_depth(candidates, **qc_options)
            qc_diagnostics = {
                "selection_policy": (
                    "checkshot_or_vsp>provided_time_depth>well_twt_curve>sonic;"
                    "never_select_by_row_count"
                ),
                "accepted": bool(selection.accepted),
                "selected_source": (
                    None if selection.selected is None else selection.selected.source
                ),
                "selected_source_kind": (
                    None
                    if selection.selected is None
                    else selection.selected.source_kind
                ),
                "evaluations": [item.to_metadata() for item in selection.evaluations],
                "normalization_rejections": normalization_rejections,
                "issues": list(selection.issues),
                "warnings": list(selection.warnings),
                "comparisons": list(selection.comparisons),
            }
            for warning in selection.warnings:
                log.issues.append(f"provided_time_depth_qc_warning:{warning}")
            if selection.accepted and selection.selected is not None:
                selected = selection.selected
                selected_metadata = candidate_metadata[selected.source]
                finite_uncertainty = selected.uncertainty_ms[
                    np.isfinite(selected.uncertainty_ms)
                ]
                uncertainty_ms = (
                    float(np.median(finite_uncertainty))
                    if finite_uncertainty.size
                    else None
                )
                return TimeDomainAlignment(
                    transform=ProvidedTimeDepthTransform(
                        selected.depth_m, selected.twt_ms
                    ),
                    status="provided_tie",
                    method="provided_time_depth",
                    confidence=float(selected.confidence),
                    uncertainty_ms=uncertainty_ms,
                    training_eligible=True,
                    depth_domain="z_msl_m",
                    diagnostics={
                        **selected_metadata,
                        "time_depth_qc": qc_diagnostics,
                        **datum_metadata,
                    },
                )

            # Two individually valid, equal-authority physical tables that
            # materially disagree are a review blocker.  Falling through to a
            # weaker sonic estimate would hide the conflict.
            if selection.selected is not None:
                log.issues.extend(
                    f"provided_time_depth_qc_blocked:{issue}"
                    for issue in selection.issues
                )
                return TimeDomainAlignment(
                    transform=NoDepthTimeTransform(),
                    status="provided_time_depth_conflict",
                    method="blocked_provided_time_depth_conflict",
                    confidence=0.0,
                    uncertainty_ms=None,
                    training_eligible=False,
                    depth_domain="z_msl_m",
                    diagnostics={
                        **datum_metadata,
                        "blocking_reasons": list(selection.issues),
                        "time_depth_qc": qc_diagnostics,
                    },
                )

            # If every supplied table fails canonicalisation or physical QC,
            # retain the complete rejection receipt and permit only the
            # already-audited sonic branch below as an explicit fallback.
            evaluation_issues = [
                f"{item.source}:{issue}"
                for item in selection.evaluations
                for issue in item.issues
            ]
            normalization_issues = [
                f"{item['source']}:{item['stage']}:{item['reason']}"
                for item in normalization_rejections
            ]
            blocking_reasons = list(
                dict.fromkeys(
                    [*selection.issues, *evaluation_issues, *normalization_issues]
                )
            )
            provided_fallback_diagnostics = {
                "fallback_from": "rejected_provided_time_depth",
                "blocking_reasons": blocking_reasons,
                "time_depth_qc": qc_diagnostics,
            }
            log.issues.append("provided_time_depth_rejected:sonic_fallback_considered")

        vertical = vertical_options
        sonic = vertical.get("sonic", {})
        preferred = [str(item).lower() for item in vertical.get("preferred", [])]
        if (
            reference_match is not None
            and sonic.get("enabled", True)
            and (not preferred or "sonic_integrated" in preferred)
        ):
            geometry = reference_match.reader.geometry
            if (
                geometry is not None
                and seismic_time_meta.replacement_velocity_mps is None
            ):
                log.issues.append(
                    "sonic_well_tie_blocked:replacement_velocity_mps_unknown"
                )
            elif geometry is not None:
                try:
                    sonic_options = {
                        **sonic,
                        "reference_depth_m": float(well_datum.absolute_elevation_m)
                        - float(seismic_datum.absolute_elevation_m),
                        "datum_time_ms": 0.0,
                        "datum_is_explicit": True,
                        "replacement_velocity_m_s": float(
                            seismic_time_meta.replacement_velocity_mps
                        ),
                    }
                    alignment = build_sonic_time_domain_alignment(
                        tvd,
                        log.curves,
                        self._composite_trace(reference_match),
                        geometry.time_axis,
                        sonic_options,
                    )
                    if alignment is not None:
                        alignment.diagnostics.update(
                            {
                                **datum_metadata,
                                "normalization": "KB/DF/RT与SRD先转MSL绝对高程，再计算SRD对应井深",
                            }
                        )
                        if provided_fallback_diagnostics is not None:
                            alignment.diagnostics["provided_time_depth_fallback"] = (
                                provided_fallback_diagnostics
                            )
                        return alignment
                except (ValueError, FloatingPointError) as exc:
                    log.issues.append(
                        f"sonic_well_tie_failed:{type(exc).__name__}:{exc}"
                    )

        if provided_fallback_diagnostics is not None:
            return TimeDomainAlignment(
                transform=NoDepthTimeTransform(),
                status="provided_time_depth_rejected",
                method="blocked_provided_time_depth_qc",
                confidence=0.0,
                uncertainty_ms=None,
                training_eligible=False,
                depth_domain="z_msl_m",
                diagnostics={
                    **datum_metadata,
                    **provided_fallback_diagnostics,
                },
            )

        velocity = vertical.get("constant_velocity")
        if velocity and vertical.get("allow_low_confidence_constant_velocity", False):
            transform = ConstantVelocityTransform(float(velocity))
            return TimeDomainAlignment(
                transform=transform,
                status="vertical_initial",
                method=transform.method,
                confidence=transform.confidence,
                uncertainty_ms=None,
                training_eligible=False,
                depth_domain="depth_below_srd_m",
                diagnostics={
                    "velocity_m_s": float(velocity),
                    "normalization": "MSL绝对高程->SRD以下深度",
                    **datum_metadata,
                },
            )
        return TimeDomainAlignment(
            transform=NoDepthTimeTransform(),
            status="horizontal_only",
            method="none",
            confidence=0.0,
            uncertainty_ms=None,
            training_eligible=False,
            depth_domain="z_msl_m",
            diagnostics=datum_metadata,
        )

    def fuse_samples(
        self,
        fusion: WellSeismicFusion | None = None,
        labels: Any = None,
    ) -> list[dict[str, Any]]:
        """对已匹配样本执行可替换融合算法；默认读取configs/fusion.yaml。"""
        if not self.samples:
            self.build_samples()
        algorithm = fusion or build_fusion(self.config.get("fusion", {}))
        self.samples = algorithm.fit_transform(self.samples, labels)
        return self.samples

    def write_outputs(
        self,
        output_dir: str | Path,
        *,
        sealed_assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = self.quality_report(sealed_assets=sealed_assets)
        report["preparation"] = build_preparation_report(self)
        json_path = output / "质量报告.json"
        json_path.write_text(
            json.dumps(
                _json_value(self._chinese_report(report)), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        md_path = output / "质量报告.md"
        md_path.write_text(self._markdown_report(report), encoding="utf-8")
        sample_jsonl = output / "多模态样本.jsonl"
        with sample_jsonl.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(
                    json.dumps(
                        _json_value(sample_to_chinese(sample)), ensure_ascii=False
                    )
                    + "\n"
                )
        index_path = output / "样本索引.csv"
        fields = [
            "well_uid",
            "well_name",
            "md",
            "tvd",
            "z_msl_m",
            "depth_below_msl_m",
            "depth_below_srd_m",
            "x",
            "y",
            "trace_index",
            "inline",
            "crossline",
            "distance",
            "seismic_coordinate",
            "horizontal_confidence",
            "vertical_method",
            "vertical_status",
            "vertical_confidence",
            "vertical_uncertainty_ms",
            "seismic_window_valid",
            "coordinate_reference_verified",
            "vertical_datum_verified",
            "training_eligible",
            "log_source",
            "seismic_source",
        ]
        with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
            chinese_fields = [SAMPLE_FIELDS_ZH[field] for field in fields]
            writer = csv.DictWriter(handle, fieldnames=chinese_fields)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(
                    {SAMPLE_FIELDS_ZH[key]: sample.get(key) for key in fields}
                )
        return {
            "质量报告JSON": json_path,
            "质量报告Markdown": md_path,
            "多模态样本": sample_jsonl,
            "样本索引": index_path,
        }

    @staticmethod
    def _chinese_report(report: dict[str, Any]) -> dict[str, Any]:
        summary = report["summary"]
        return {
            "汇总": {
                "数据资产数": summary["assets"],
                "跳过重复文件数": summary["duplicates_skipped"],
                "井实体数": summary["wells"],
                "地震文件数": summary["seismic_files"],
                "多模态样本数": summary["samples"],
                "读取错误数": summary["errors"],
            },
            "运行时输入识别": report.get("automatic_inventory"),
            "井相关文件识别": report.get("metadata_detection", []),
            "LLM受控判断": report.get("llm_decisions", []),
            "LLM结构化解析补丁": report.get("llm_parse_repairs", []),
            "LLM配置状态": report.get("llm_status", {}),
            "垂向基准统一": report.get("vertical_datum", {}),
            "预处理与对齐阶段": report.get("preparation", {}),
            "时间域井震标定": report.get("well_ties", []),
            "数据资产": report["assets"],
            "重复文件": report["duplicates"],
            "读取错误": report["errors"],
            "井数据": report["wells"],
            "地震数据": report["seismic"],
        }

    def quality_report(
        self,
        progress: Any = None,
        *,
        sealed_assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a report, reusing already verified sealed asset identities.

        ``sealed_assets`` is only valid after the API has verified every source
        SHA-256 for the immutable SourceSnapshot.  We still compare the current
        catalog size/options and reconstructed geometry fingerprint, but avoid
        hashing the same multi-gigabyte SEG-Y again for each derived view.
        """

        wells = []
        for entity in self.registry.entities.values():
            wells.append(
                {
                    "well_uid": entity.well_uid,
                    "name": entity.canonical_name,
                    "aliases": sorted(entity.aliases),
                    "identifiers": sorted(entity.identifiers),
                    "head_count": len(entity.heads),
                    "log_count": len(entity.logs),
                    "trajectory_count": len(entity.trajectories),
                    "conflicts": entity.conflicts,
                    "heads": [
                        {
                            "source": head.source,
                            "crs": head.crs,
                            "source_crs": head.source_crs,
                            "coordinate_transform": head.coordinate_transform,
                            "identifiers": list(head.identifiers),
                        }
                        for head in entity.heads
                    ],
                    "trajectories": [
                        {
                            "source": trajectory.source,
                            "source_crs": trajectory.source_crs,
                            "horizontal_crs": trajectory.horizontal_crs,
                            "coordinate_transform": trajectory.coordinate_transform,
                            "identifiers": list(trajectory.identifiers),
                        }
                        for trajectory in entity.trajectories
                    ],
                    "logs": [
                        {
                            "source": log.source,
                            "las_version": log.version,
                            "asset_version": self.asset_by_path.get(log.source).version
                            if self.asset_by_path.get(log.source)
                            else None,
                            "stage": self.asset_by_path.get(log.source).stage
                            if self.asset_by_path.get(log.source)
                            else "UNKNOWN",
                            "samples": len(log.depth),
                            "curves": sorted(log.curves),
                            "issues": log.issues,
                            "processing_steps": log.processing_steps,
                            "identifiers": list(log.identifiers),
                        }
                        for log in entity.logs
                    ],
                }
            )
        seismic = []
        for asset, reader in self.seismic:
            geom = reader.geometry
            if geom:
                seismic.append(
                    {
                        "asset": str(asset.path),
                        "revision": geom.revision,
                        "endian": geom.endian,
                        "sample_format": geom.sample_format,
                        "sample_interval_ms": geom.sample_interval,
                        "samples_per_trace": geom.samples_per_trace,
                        "trace_count": geom.trace_count,
                        "profile": geom.profile,
                        "confidence": geom.confidence,
                        "issues": geom.issues,
                        "source_crs": geom.source_crs,
                        "horizontal_crs": geom.horizontal_crs,
                        "coordinate_transform": geom.coordinate_transform,
                    }
                )
        geometry_by_path = {
            str(asset.path.resolve()): reader.geometry
            for asset, reader in self.seismic
            if reader.geometry is not None
        }
        assets = []
        asset_paths = [(asset, asset.path.resolve()) for asset in self.assets]
        asset_sizes = {str(path): path.stat().st_size for _, path in asset_paths}
        sealed_by_path: dict[str, dict[str, Any]] | None = None
        if sealed_assets is not None:
            sealed_by_path = {}
            for raw in sealed_assets:
                path = str(raw.get("path") or "").strip()
                if not path:
                    raise ValueError("sealed asset record is missing its path")
                key = os.path.normcase(str(Path(path).expanduser().resolve()))
                if key in sealed_by_path:
                    raise ValueError("sealed asset inventory contains duplicate paths")
                sealed_by_path[key] = dict(raw)
            current_keys = {
                os.path.normcase(str(path)) for _, path in asset_paths
            }
            if current_keys != set(sealed_by_path):
                raise ValueError(
                    "pipeline catalog differs from the verified sealed asset inventory"
                )
        hash_total = sum(asset_sizes.values())
        hash_completed = 0
        last_progress_report = 0
        progress_interval = 256 * 1024 * 1024
        if progress:
            progress(0, hash_total, None)
        for asset, path in asset_paths:
            size = asset_sizes[str(path)]

            def report_file_hash(file_completed: int, file_total: int) -> None:
                nonlocal last_progress_report
                overall = hash_completed + file_completed
                if (
                    overall - last_progress_report >= progress_interval
                    or file_completed >= file_total
                ):
                    last_progress_report = overall
                    if progress:
                        progress(overall, hash_total, path)

            geometry = geometry_by_path.get(str(path))
            geometry_identity = (
                seismic_geometry_identity(geometry) if geometry is not None else None
            )
            sealed = (
                sealed_by_path.get(os.path.normcase(str(path)))
                if sealed_by_path is not None
                else None
            )
            options_sha256 = canonical_sha256(_json_value(asset.options))
            if sealed is not None:
                if int(sealed.get("size") or -1) != size:
                    raise ValueError(
                        f"sealed asset size differs from current catalog: {path}"
                    )
                sha256 = str(sealed.get("sha256") or "").casefold()
                if len(sha256) != 64:
                    raise ValueError(f"sealed asset has no valid SHA-256: {path}")
                expected_options = str(
                    sealed.get("asset_options_sha256") or ""
                ).casefold()
                if expected_options and expected_options != options_sha256:
                    raise ValueError(
                        f"sealed asset parser options differ from current catalog: {path}"
                    )
                expected_geometry = str(
                    sealed.get("geometry_fingerprint") or ""
                ).casefold()
                observed_geometry = (
                    str(geometry_identity["geometry_fingerprint"]).casefold()
                    if geometry_identity is not None
                    else ""
                )
                if expected_geometry != observed_geometry:
                    raise ValueError(
                        f"sealed asset geometry differs from current pipeline: {path}"
                    )
            else:
                sha256 = file_sha256(
                    path,
                    progress=report_file_hash if progress else None,
                )
            assets.append(
                {
                    "id": asset.asset_id,
                    "role": asset.role,
                    "dataset": asset.dataset,
                    "stage": asset.stage,
                    "path": str(path),
                    "size": size,
                    "sha256": sha256,
                    "geometry_fingerprint": (
                        geometry_identity["geometry_fingerprint"]
                        if geometry_identity is not None
                        else None
                    ),
                    "geometry_identity": geometry_identity,
                    "asset_options_sha256": options_sha256,
                }
            )
            hash_completed += size
        return {
            "summary": {
                "assets": len(self.assets),
                "duplicates_skipped": len(self.duplicates),
                "wells": len(wells),
                "seismic_files": len(seismic),
                "samples": len(self.samples),
                "training_eligible_samples": sum(
                    1 for item in self.samples if item.get("training_eligible")
                ),
                "errors": len(self.errors),
            },
            "assets": assets,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "wells": wells,
            "seismic": seismic,
            "well_ties": self.well_ties,
            "automatic_inventory": self.automatic_inventory,
            "metadata_detection": self.metadata_detection,
            "llm_parse_repairs": list(self.llm_parse_repairs),
            "deterministic_unit_inheritances": list(
                self.deterministic_unit_inheritances
            ),
            "vertical_datum": self.vertical_datum_inventory(),
            "llm_decisions": list(self.decision_resolver.records),
            "llm_status": self.decision_resolver.settings.public_status(),
        }

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        s = report["summary"]
        lines = [
            "# 井震数据质量报告",
            "",
            "## 汇总",
            "",
            f"- 数据资产：{s['assets']}",
            f"- 去重文件：{s['duplicates_skipped']}",
            f"- 井实体：{s['wells']}",
            f"- 地震文件：{s['seismic_files']}",
            f"- 多模态样本：{s['samples']}",
            f"- 可训练样本：{s.get('training_eligible_samples', 0)}",
            f"- 读取错误：{s['errors']}",
            "",
            "## 井数据",
            "",
        ]
        for well in report["wells"]:
            curves = sorted({c for log in well["logs"] for c in log["curves"]})
            lines.append(
                f"- **{well['name']}**（{well['well_uid']}）：测井 {well['log_count']}，轨迹 {well['trajectory_count']}；曲线：{', '.join(curves) or '无'}"
            )
        lines.extend(["", "## 地震数据", ""])
        for seismic in report["seismic"]:
            lines.append(
                f"- `{seismic['asset']}`：{seismic['trace_count']} 道 × {seismic['samples_per_trace']} 样点，采样间隔 {seismic['sample_interval_ms']} ms，几何置信度 {seismic['confidence']:.3f}"
            )
        if report.get("well_ties"):
            lines.extend(["", "## 时间域井震标定", ""])
            for tie in report["well_ties"]:
                lines.append(
                    f"- **{tie['well_name']}**：{tie['status']} / {tie['method']}，"
                    f"置信度 {float(tie['confidence']):.3f}，可训练：{'是' if tie['training_eligible'] else '否'}"
                )
        if report["errors"]:
            lines.extend(["", "## 隔离的读取错误", ""])
            lines.extend(f"- `{x['path']}`：{x['error']}" for x in report["errors"])
        preparation = report.get("preparation", {})
        if preparation:
            lines.extend(["", "## 预处理与对齐阶段", ""])
            for stage in preparation.get("stages", []):
                lines.append(
                    f"- **{stage['name']}**：{stage['status']}，问题 {stage['issue_count']} 项"
                )
            blocking = [
                issue
                for issue in preparation.get("issues", [])
                if issue.get("blocking")
            ]
            if blocking:
                lines.extend(["", "### 阻断问题", ""])
                lines.extend(
                    f"- **{issue['title']}**：{issue['message']}" for issue in blocking
                )
        return "\n".join(lines) + "\n"

    def _assets(self, role: str):
        return [asset for asset in self.assets if asset.role == role]

    @staticmethod
    def _merge_parse_options(
        base: dict[str, Any],
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if key in {"columns", "field_units"}:
                current = dict(merged.get(key) or {})
                current.update(dict(value))
                merged[key] = current
            else:
                merged[key] = value
        return merged

    def _inherit_trajectory_md_unit_from_same_well_las(
        self,
        asset: Any,
        options: dict[str, Any],
        original_error: Exception,
    ) -> Any | None:
        """Resolve a missing trajectory MD unit from same-well LAS evidence.

        The source trajectory remains read-only.  Both supported length-unit
        interpretations are parsed in memory and compared with the already
        canonicalised MD extent of an *exactly named* LAS from the same input
        request.  A candidate is accepted only when it closely matches every
        well in the file and the counterfactual unit is clearly incompatible.
        This keeps a useful deterministic path for headerless survey exports
        without turning plausible numeric ranges into an implicit metre
        default.
        """

        error_text = f"{type(original_error).__name__}: {original_error}"
        if not re.search(
            r"trajectory\s+md\s+length\s+unit\s+is\s+unknown",
            error_text,
            flags=re.IGNORECASE,
        ):
            return None
        current_units = options.get("field_units")
        if isinstance(current_units, dict) and str(
            current_units.get("md") or ""
        ).strip().casefold() not in {"", "unknown"}:
            return None

        try:
            source_sha_before = file_sha256(Path(asset.path))
        except OSError:
            return None

        candidates: list[dict[str, Any]] = []
        for unit in ("m", "ft"):
            patched_options = self._merge_parse_options(
                options,
                {"field_units": {"md": unit}},
            )
            try:
                detected = read_adaptive_metadata(
                    asset.path,
                    dict(options.get("field_aliases") or {}),
                    patched_options,
                )
            except Exception:
                continue
            if not detected.accepted or not detected.trajectories:
                continue

            comparisons: list[dict[str, Any]] = []
            valid = True
            for trajectory in detected.trajectories:
                md = np.asarray(trajectory.md, dtype=float)
                x_offset = np.asarray(trajectory.x_offset, dtype=float)
                y_offset = np.asarray(trajectory.y_offset, dtype=float)
                if (
                    md.size < 2
                    or md.size != x_offset.size
                    or md.size != y_offset.size
                    or not np.isfinite(md).all()
                    or not np.isfinite(x_offset).all()
                    or not np.isfinite(y_offset).all()
                    or np.any(np.diff(md) < 0)
                    or trajectory.source_units.get("md") != unit
                    or trajectory.unit_provenance.get("md")
                    != "options.field_units.md"
                ):
                    valid = False
                    break

                normalized = normalize_well_name(trajectory.well_name)
                entity_key = self.registry.alias_to_key.get(normalized, normalized)
                entity = self.registry.entities.get(entity_key)
                exact_logs = (
                    []
                    if entity is None
                    else [
                        log
                        for log in entity.logs
                        if normalize_well_name(log.well_name) == normalized
                    ]
                )
                log_maxima = [
                    float(np.nanmax(log.depth))
                    for log in exact_logs
                    if np.asarray(log.depth).size
                    and np.isfinite(np.asarray(log.depth, dtype=float)).any()
                ]
                if not log_maxima:
                    valid = False
                    break
                trajectory_max_m = float(np.nanmax(md))
                las_max_m = float(np.median(log_maxima))
                if trajectory_max_m <= 0 or las_max_m <= 0:
                    valid = False
                    break
                comparisons.append(
                    {
                        "well_name": trajectory.well_name,
                        "trajectory_md_max_m": trajectory_max_m,
                        "same_well_las_md_max_m": las_max_m,
                        "relative_endpoint_error": abs(
                            trajectory_max_m - las_max_m
                        )
                        / las_max_m,
                        "same_well_las_count": len(exact_logs),
                    }
                )
            if valid and comparisons:
                candidates.append(
                    {
                        "unit": unit,
                        "detected": detected,
                        "options": patched_options,
                        "comparisons": comparisons,
                        "maximum_relative_endpoint_error": max(
                            item["relative_endpoint_error"]
                            for item in comparisons
                        ),
                    }
                )

        accepted = [
            candidate
            for candidate in candidates
            if candidate["maximum_relative_endpoint_error"] <= 0.05
        ]
        rejected_counterfactuals = [
            candidate
            for candidate in candidates
            if candidate["maximum_relative_endpoint_error"] >= 0.50
        ]
        if len(accepted) != 1 or len(rejected_counterfactuals) != 1:
            return None
        chosen = accepted[0]
        if chosen["unit"] == rejected_counterfactuals[0]["unit"]:
            return None

        try:
            source_sha_after = file_sha256(Path(asset.path))
        except OSError:
            return None
        if source_sha_after != source_sha_before:
            return None

        provenance = {
            "contract_version": "well-seismic.deterministic-unit-inheritance.v1",
            "status": "applied_to_current_snapshot",
            "rule_id": "same_well_las_md_endpoint_unique_m_or_ft.v1",
            "field": "md",
            "inherited_unit": chosen["unit"],
            "source_path": str(Path(asset.path).resolve()),
            "source_sha256_before": source_sha_before,
            "source_sha256_after": source_sha_after,
            "original_preserved": True,
            "original_error": error_text[:800],
            "acceptance_threshold_relative": 0.05,
            "counterfactual_rejection_threshold_relative": 0.50,
            "accepted_comparisons": chosen["comparisons"],
            "counterfactual_comparisons": rejected_counterfactuals[0][
                "comparisons"
            ],
        }
        trajectory_schema = getattr(
            chosen["detected"], "deterministic_trajectory_schema", None
        )
        if isinstance(trajectory_schema, dict):
            schema_without_hash = {
                key: value
                for key, value in trajectory_schema.items()
                if key != "evidence_sha256"
            }
            if (
                trajectory_schema.get("source_sha256_before") != source_sha_before
                or trajectory_schema.get("source_sha256_after") != source_sha_after
                or canonical_sha256(schema_without_hash)
                != trajectory_schema.get("evidence_sha256")
            ):
                return None
            # This nested receipt is frozen by the existing parse-repair hash.
            # Replay still applies only the independently proven MD unit; the
            # deterministic parser must re-close the source angles against
            # DX/DY instead of trusting a stored column mapping blindly.
            provenance["dependent_trajectory_schema_evidence"] = dict(
                trajectory_schema
            )
        provenance["evidence_sha256"] = canonical_sha256(provenance)
        effective_options = dict(chosen["options"])
        effective_options.pop("field_aliases", None)
        asset.options.clear()
        asset.options.update(effective_options)
        asset.options["deterministic_unit_inheritance"] = provenance
        self.deterministic_unit_inheritances.append(provenance)

        detected = chosen["detected"]
        detected.confidence = min(float(detected.confidence), 0.98)
        detected.decision_source = "deterministic_same_well_las_unit_inheritance"
        detected.evidence.extend(
            [
                "MD单位由同井LAS规范深度覆盖唯一继承，未读取时深表或监督标签",
                "m/ft双候选只读重解析；仅唯一通过端点门的候选进入当前快照",
                f"MD单位继承为{chosen['unit']}；证据摘要{provenance['evidence_sha256']}",
            ]
        )
        counterfactual_by_well = {
            item["well_name"]: item
            for item in provenance["counterfactual_comparisons"]
        }
        for comparison in provenance["accepted_comparisons"]:
            counterfactual = counterfactual_by_well[comparison["well_name"]]
            detected.evidence.append(
                "同井LAS端点佐证："
                f"{comparison['well_name']}，轨迹MD最大值"
                f"{comparison['trajectory_md_max_m']:.6g}m，LAS MD最大值"
                f"{comparison['same_well_las_md_max_m']:.6g}m，"
                f"相对差{comparison['relative_endpoint_error']:.6f}；"
                f"反事实{rejected_counterfactuals[0]['unit']}相对差"
                f"{counterfactual['relative_endpoint_error']:.6f}"
            )
        for trajectory in detected.trajectories:
            trajectory.confidence = min(float(trajectory.confidence), 0.98)
            trajectory.issues.append(
                "md_unit_inherited_from_same_well_las:"
                f"{chosen['unit']}:{provenance['evidence_sha256']}"
            )
        return detected

    def _repair_corroboration(
        self,
        trajectories: list[Any],
        *,
        base_options: dict[str, Any],
        patch_options: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Require independent evidence before accepting an inferred unit.

        A plausible depth range alone cannot distinguish metres from feet.  A
        unit proposed by the LLM is therefore applied only when corroborated by
        a manifest declaration or a same-well canonical LAS MD extent.  Cross-
        well evidence is intentionally excluded because one request may contain
        several blocks with different conventions.  Delimiter-only repairs do
        not introduce physical units and need no such semantic promotion.
        """

        inferred_units = dict(patch_options.get("field_units") or {})
        if not inferred_units:
            return True, ["未新增物理单位，仅修复表格结构"]

        explicitly_declared: dict[str, str] = {}
        for option_name in ("field_units", "trajectory_units", "units"):
            value = base_options.get(option_name)
            if isinstance(value, dict):
                explicitly_declared.update(
                    {
                        str(field): str(unit).casefold()
                        for field, unit in value.items()
                        if str(unit).casefold() in {"m", "ft"}
                    }
                )
        if len(set(inferred_units.values())) > 1:
            unsupported = sorted(
                field
                for field, unit in inferred_units.items()
                if explicitly_declared.get(field) != unit
            )
            if unsupported:
                return False, [
                    "同一轨迹出现混合m/ft单位，且这些异质字段未在manifest中逐字段显式声明："
                    + "、".join(unsupported)
                ]

        declared = dict(explicitly_declared)
        fallback_unit = str(
            base_options.get("trajectory_length_unit")
            or base_options.get("source_length_unit")
            or base_options.get("length_unit")
            or ""
        ).casefold()
        if fallback_unit in {"m", "ft"}:
            declared.setdefault("md", fallback_unit)
        if declared and all(
            declared.get(field, declared.get("md")) == unit
            for field, unit in inferred_units.items()
        ):
            return True, ["补丁单位与manifest显式长度单位一致"]

        # LAS depth has already been converted to canonical metres before
        # trajectories are ingested.  Require a clearly better endpoint match
        # than the counterfactual m/ft interpretation; weak matches are kept as
        # candidates rather than promoted.
        for trajectory in trajectories:
            normalized = normalize_well_name(trajectory.well_name)
            entity_key = self.registry.alias_to_key.get(normalized, normalized)
            entity = self.registry.entities.get(entity_key)
            if entity is None or not entity.logs:
                continue
            log_maxima = [
                float(np.nanmax(log.depth))
                for log in entity.logs
                if np.asarray(log.depth).size and np.isfinite(log.depth).any()
            ]
            if not log_maxima:
                continue
            trajectory_max = float(np.nanmax(trajectory.md))
            reference = float(np.median(log_maxima))
            if reference <= 0 or trajectory_max <= 0:
                continue
            unit = inferred_units.get("md")
            if unit not in {"m", "ft"}:
                continue
            proposed_error = abs(trajectory_max - reference) / reference
            alternate_max = trajectory_max * (0.3048 if unit == "m" else 1.0 / 0.3048)
            alternate_error = abs(alternate_max - reference) / reference
            if proposed_error <= 0.25 and alternate_error >= 0.50:
                return True, [
                    "补丁MD单位与同井LAS规范深度覆盖一致",
                    f"MD端点相对差={proposed_error:.3f}，反事实单位相对差={alternate_error:.3f}",
                ]

        return False, [
            "m/ft均可能通过单文件物理门，缺少manifest逐字段声明或同井LAS覆盖佐证"
        ]

    def _repair_trajectory_asset(
        self,
        asset: Any,
        options: dict[str, Any],
        original_error: Exception,
    ) -> list[Any] | None:
        """Generate, validate and apply one read-only trajectory parse patch."""

        try:
            evidence = summarize_tabular_source(asset.path)
        except (OSError, ValueError):
            return None
        original_error_text = f"{type(original_error).__name__}: {original_error}"
        model_error = sanitize_llm_text(
            original_error_text,
            known_paths=(asset.path, Path(asset.path).expanduser().resolve()),
        ).replace("<local-path>", "<source>")
        generated = self.decision_resolver.resolve_trajectory_parse_patch(
            original_error=model_error,
            evidence=evidence,
        )
        if generated is None:
            return None
        proposal, metadata = generated
        compiled, patch_errors = validate_trajectory_parse_patch(
            proposal,
            evidence=evidence,
            manifest_columns=(
                options.get("columns")
                if isinstance(options.get("columns"), dict)
                else None
            ),
        )
        confidence = float(compiled["confidence"])
        if confidence < self.decision_resolver.settings.min_confidence:
            patch_errors.append(
                "confidence_below_threshold:"
                f"{confidence:.3f}<{self.decision_resolver.settings.min_confidence:.3f}"
            )
        patch_options = dict(compiled["options"])
        patched_options = self._merge_parse_options(options, patch_options)
        trajectories: list[Any] = []
        physics_errors: list[str] = []
        physics_summary: dict[str, Any] = {"trajectory_count": 0}
        source_sha_before = str(evidence["source_sha256"])
        source_sha_after = source_sha_before
        if not patch_errors:
            try:
                trajectories = read_trajectory(asset.path, patched_options)
                physics_ok, physics_errors, physics_summary = (
                    validate_trajectory_physics(trajectories)
                )
                if not physics_ok:
                    trajectories = []
            except Exception as exc:
                physics_errors.append(f"reparse_failed:{type(exc).__name__}:{exc}")
                trajectories = []
            try:
                source_sha_after = file_sha256(Path(asset.path))
            except OSError as exc:
                physics_errors.append(f"source_rehash_failed:{exc}")
                trajectories = []
            if source_sha_after != source_sha_before:
                physics_errors.append("source_changed_during_repair")
                trajectories = []

        corroborated = False
        corroboration: list[str] = []
        if trajectories:
            corroborated, corroboration = self._repair_corroboration(
                trajectories,
                base_options=options,
                patch_options=patch_options,
            )
            if not corroborated:
                trajectories = []
        elif patch_errors or physics_errors:
            corroboration = ["补丁未通过结构或物理门，未进入语义佐证"]

        status = "applied_to_current_snapshot" if trajectories else "candidate_rejected"
        record: dict[str, Any] = {
            "contract_version": REPAIR_CONTRACT_VERSION,
            "status": status,
            "asset_id": asset.asset_id,
            "asset_role": asset.role,
            "source_path": str(Path(asset.path).resolve()),
            "source_sha256": source_sha_before,
            "source_hash": str(metadata.get("source_hash") or ""),
            "original_error": original_error_text[:800],
            "options_patch": patch_options,
            "confidence": confidence,
            "reason": compiled["reason"],
            "warnings": compiled["warnings"],
            "provider": metadata.get("provider", ""),
            "model": metadata.get("model", ""),
            "request_id": metadata.get("request_id", ""),
            "validation": {
                "mode": "read_only_in_memory_reparse",
                "original_sha256_before": source_sha_before,
                "original_sha256_after": source_sha_after,
                "original_preserved": source_sha_before == source_sha_after,
                "schema_errors": patch_errors,
                "physics_errors": physics_errors,
                "physics_summary": physics_summary,
                "corroborated": corroborated,
            },
            "corroboration": corroboration,
        }
        record["patch_sha256"] = repair_fingerprint(
            {
                "contract_version": REPAIR_CONTRACT_VERSION,
                "source_sha256": source_sha_before,
                "options_patch": patch_options,
            }
        )
        self.llm_parse_repairs.append(record)

        # Complete the resolver audit only after deterministic validation.
        for audit in reversed(self.decision_resolver.records):
            if (
                audit.get("判断类型") == "trajectory_parse_patch"
                and audit.get("来源摘要") == record["source_hash"]
            ):
                audit["是否采纳"] = bool(trajectories)
                audit["验证状态"] = (
                    "结构、物理与独立语义佐证通过"
                    if trajectories
                    else "结构、物理或独立语义佐证未通过"
                )
                audit["补丁摘要"] = record["patch_sha256"]
                break
        if not trajectories:
            return None

        provenance = {
            key: record[key]
            for key in (
                "contract_version",
                "source_sha256",
                "source_hash",
                "patch_sha256",
                "confidence",
                "provider",
                "model",
                "request_id",
                "validation",
                "corroboration",
            )
        }
        asset.options.clear()
        asset.options.update(
            self._merge_parse_options(options, patch_options)
        )
        # ``field_aliases`` is effective configuration rather than source
        # semantics and was only added to the call-local options above.
        asset.options.pop("field_aliases", None)
        asset.options["llm_parse_repair"] = provenance
        for trajectory in trajectories:
            trajectory.confidence = min(float(trajectory.confidence), confidence)
            trajectory.issues.append(
                f"llm_parse_repair_validated:{record['patch_sha256']}"
            )
        return trajectories

    def _error(self, asset: Any, exc: Exception) -> None:
        self.errors.append(
            {
                "asset_id": asset.asset_id,
                "path": str(asset.path),
                "role": asset.role,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    @staticmethod
    def _header_float(header: dict[str, Evidence], *keys: str) -> float | None:
        for key in keys:
            if key in header:
                try:
                    return float(str(header[key].value).split()[0])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _header_coordinate_m(
        header: dict[str, Evidence],
        key: str,
        options: dict[str, Any],
    ) -> tuple[float | None, str]:
        evidence = header.get(key)
        if evidence is None:
            return None, "unknown"
        try:
            value = float(str(evidence.value).split()[0])
        except (TypeError, ValueError):
            return None, "unknown"
        header_unit = str(evidence.notes[0]).strip() if evidence.notes else ""
        unit = (
            header_unit
            or str(options.get(f"{key.casefold()}_unit") or "").strip()
            or str(options.get("coordinate_unit") or "").strip()
            or str(options.get("horizontal_unit") or "").strip()
            or str(options.get("length_unit") or "").strip()
        )
        if not unit or unit.casefold() == "unknown":
            return None, "unknown"
        try:
            return float(length_to_metres(value, unit)), unit
        except ValueError:
            return None, unit
