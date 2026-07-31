from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import build_catalog
from .alignment import TimeDomainAlignment, build_sonic_time_domain_alignment, build_spatial_aligner
from .config import load_config
from .depth_time import ConstantVelocityTransform, NoDepthTimeTransform, ProvidedTimeDepthTransform
from .fusion import WellSeismicFusion, build_fusion
from .io import (
    SegyReader,
    apply_llm_metadata_decision,
    read_adaptive_metadata,
    read_las,
    read_time_depth,
    read_trajectory,
    read_well_heads,
)
from .knowledge import CurveKnowledgeBase
from .llm import build_decision_resolver
from .llm.transformation import apply_active_transformations
from .models import Evidence, MatchRecord, WellHead
from .registry import WellRegistry
from .trajectory import interpolate_trajectory
from .output_schema import SAMPLE_FIELDS_ZH, sample_to_chinese
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
        self.assets, self.duplicates = build_catalog(self.manifest, self.config["manifest_path"])
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
        self.metadata_detection: list[dict[str, Any]] = []
        self.automatic_inventory: dict[str, Any] | None = None

    @classmethod
    def from_input_root(cls, input_root: str | Path, config_dir: str | Path) -> "WellSeismicPipeline":
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

    def ingest(self) -> "WellSeismicPipeline":
        field_aliases = self.config.get("well_schema", {}).get("fields", {})
        # Well heads first, then logs can fill gaps from LAS headers.
        for asset in self._assets("well_heads"):
            try:
                options = {**asset.options, "field_aliases": field_aliases}
                for head in read_well_heads(asset.path, options):
                    self.registry.add_head(head)
            except Exception as exc:
                self._error(asset, exc)
        for asset in self._assets("well_logs"):
            try:
                log = read_las(
                    asset.path,
                    self.knowledge,
                    self.config.get("preprocessing", {}),
                    self.decision_resolver,
                )
                self.registry.add_log(log)
                if "TWT" in log.curves:
                    valid = np.isfinite(log.depth) & np.isfinite(log.curves["TWT"])
                    if np.sum(valid) >= 2:
                        self.registry.add_time_depth(
                            log.well_name,
                            log.source + "#TWT",
                            log.depth[valid],
                            log.curves["TWT"][valid],
                            depth_domain="md",
                            depth_unit="m",
                            time_unit="ms",
                        )
                x = self._header_float(log.header, "XCRD")
                y = self._header_float(log.header, "YCRD")
                kb = self._header_float(log.header, "EKB", "KB")
                stop = self._header_float(log.header, "STOP")
                if x is not None and y is not None:
                    self.registry.add_head(WellHead(log.well_name, x, y, kb=kb, total_depth_md=stop, source=log.source, confidence=0.9))
            except Exception as exc:
                self._error(asset, exc)
        for asset in self._assets("trajectories"):
            try:
                options = {**asset.options, "field_aliases": field_aliases}
                for trajectory in read_trajectory(asset.path, options):
                    self.registry.add_trajectory(trajectory)
            except Exception as exc:
                self._error(asset, exc)
        for asset in self._assets("time_depth"):
            try:
                for name, (depth, time) in read_time_depth(asset.path, asset.options).items():
                    self.registry.add_time_depth(
                        name,
                        str(asset.path),
                        depth,
                        time,
                        depth_domain=asset.options.get("depth_domain", "md"),
                        depth_unit=asset.options.get("depth_unit", "m"),
                        time_unit=asset.options.get("time_unit", "ms"),
                    )
            except Exception as exc:
                self._error(asset, exc)
        for asset in self._assets("well_metadata"):
            try:
                detected = read_adaptive_metadata(asset.path, field_aliases)
                llm_decision = None
                if not detected.accepted and detected.detected_roles:
                    llm_decision = self.decision_resolver.resolve_metadata(asset.path, detected)
                    if llm_decision and llm_decision.accepted:
                        detected = apply_llm_metadata_decision(
                            detected,
                            llm_decision.choice,
                            llm_decision.confidence,
                        )
                detected_time_depth_domain = (
                    asset.options.get("depth_domain") or detected.time_depth_domain
                )
                if detected.time_depth and not detected_time_depth_domain:
                    detected.evidence.append("时深表深度域未明确，未进入垂向标定")
                self.metadata_detection.append({
                    "文件": str(asset.path), "识别角色": detected.detected_roles,
                    "置信度": detected.confidence, "状态": detected.status, "证据": detected.evidence,
                    "时深深度域": detected_time_depth_domain or "未明确",
                    "决策来源": detected.decision_source,
                    "LLM判断": llm_decision.to_audit_dict() if llm_decision else None,
                })
                # Uncertain headerless interpretations remain report-only and never silently enter matching.
                if detected.accepted:
                    for head in detected.heads:
                        head.confidence = detected.confidence
                        self.registry.add_head(head)
                    for trajectory in detected.trajectories:
                        trajectory.confidence = detected.confidence
                        self.registry.add_trajectory(trajectory)
                    for name, (depth, time) in detected.time_depth.items() if detected_time_depth_domain else ():
                        self.registry.add_time_depth(
                            name,
                            str(asset.path),
                            depth,
                            time,
                            depth_domain=detected_time_depth_domain,
                            depth_unit=asset.options.get("depth_unit", "m"),
                            time_unit=asset.options.get("time_unit", "ms"),
                            confidence=detected.confidence,
                        )
            except Exception as exc:
                self._error(asset, exc)
        for asset in self._assets("seismic"):
            try:
                reader = SegyReader(asset.path, self.config, asset.options)
                geometry = reader.inspect()
                self.seismic.append((asset, reader))
            except Exception as exc:
                self._error(asset, exc)
        return self

    def build_samples(self) -> list[dict[str, Any]]:
        matching = self.config.get("matching", {})
        stride = max(1, int(matching.get("log_sample_stride", 128)))
        max_distance = float(matching.get("max_horizontal_distance", 500.0))
        distance_scale = float(matching.get("distance_confidence_scale", 100.0))
        window_size = max(1, int(matching.get("seismic_window_samples", 32)))
        minimum_horizontal = float(matching.get("min_horizontal_confidence_for_training", 0.35))
        minimum_vertical = float(matching.get("min_vertical_confidence_for_training", 0.55))
        coordinate_reference = matching.get("coordinate_reference", {})
        coordinate_reference_verified = bool(coordinate_reference.get("verified", False))
        seismic_sources = self._selected_seismic_sources(matching)
        spatial_aligner = build_spatial_aligner(matching).fit(seismic_sources)

        samples: list[dict[str, Any]] = []
        self.well_ties = []
        for entity in self.registry.entities.values():
            head = entity.preferred_head
            if not entity.logs or head is None or head.x is None or head.y is None:
                continue
            trajectory = entity.preferred_trajectory
            for log in entity.logs:
                md_full = np.asarray(log.depth, dtype=float)
                if trajectory is not None:
                    tvd_full = interpolate_trajectory(md_full, trajectory.md, trajectory.tvd)
                    if trajectory.x is not None and trajectory.y is not None:
                        xs_full = interpolate_trajectory(md_full, trajectory.md, trajectory.x)
                        ys_full = interpolate_trajectory(md_full, trajectory.md, trajectory.y)
                    else:
                        xs_full = head.x + interpolate_trajectory(md_full, trajectory.md, trajectory.x_offset)
                        ys_full = head.y + interpolate_trajectory(md_full, trajectory.md, trajectory.y_offset)
                    trajectory_source = trajectory.source
                    position_confidence = trajectory.confidence
                else:
                    tvd_full = md_full.copy()
                    xs_full = np.full_like(md_full, head.x)
                    ys_full = np.full_like(md_full, head.y)
                    trajectory_source = "vertical_well_fallback"
                    position_confidence = 0.5
                tvdss_full = (
                    tvd_full - float(head.kb)
                    if head.kb is not None
                    else np.full_like(tvd_full, np.nan)
                )
                positions = np.isfinite(xs_full) & np.isfinite(ys_full)
                reference_match = None
                if np.any(positions):
                    reference_match = spatial_aligner.match(
                        float(np.median(xs_full[positions])),
                        float(np.median(ys_full[positions])),
                    )
                    if reference_match is not None and reference_match.distance > max_distance:
                        reference_match = None
                alignment = self._vertical_alignment(
                    entity,
                    log,
                    md_full,
                    tvd_full,
                    tvdss_full,
                    reference_match,
                )
                alignment_depth = self._alignment_depth_values(
                    alignment.depth_domain,
                    md_full,
                    tvd_full,
                    tvdss_full,
                )
                times_full = alignment.transform.depth_to_time(alignment_depth)
                tie_metadata = alignment.to_metadata()
                self.well_ties.append({
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "log_source": log.source,
                    "seismic_source": None if reference_match is None else str(reference_match.asset.path),
                    **tie_metadata,
                })

                sample_indices = np.arange(0, len(md_full), stride, dtype=int)
                for local, original_index in enumerate(sample_indices):
                    x, y = xs_full[original_index], ys_full[original_index]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    nearest = spatial_aligner.match(
                        float(x),
                        float(y),
                        asset=None if reference_match is None else reference_match.asset,
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
                    if (
                        np.isfinite(time_value)
                        and float(geom.time_axis[0]) <= float(time_value) <= float(geom.time_axis[-1])
                    ):
                        center = int(np.argmin(np.abs(geom.time_axis - time_value)))
                        start = center - window_size // 2
                        stop = start + window_size
                        if start < 0:
                            start, stop = 0, window_size
                        if stop > geom.samples_per_trace:
                            stop = geom.samples_per_trace
                            start = stop - window_size
                        if start >= 0 and stop <= geom.samples_per_trace and stop - start == window_size:
                            window = reader.read_trace(trace_index, slice(start, stop)).astype(float)
                            seismic_window_valid = bool(window.size == window_size and np.all(np.isfinite(window)))
                            seismic_window = window.tolist()
                            seismic_coordinate = float(geom.time_axis[center])
                    features = {name: float(values[original_index]) for name, values in log.curves.items() if np.isfinite(values[original_index])}
                    masks = {name: bool(mask[original_index]) for name, mask in log.masks.items()}
                    horizontal_confidence = float(position_confidence * geom.confidence * math.exp(-distance / max(distance_scale, 1e-9)))
                    training_eligible = bool(
                        coordinate_reference_verified
                        and seismic_window_valid
                        and alignment.training_eligible
                        and horizontal_confidence >= minimum_horizontal
                        and alignment.confidence >= minimum_vertical
                    )
                    record = MatchRecord(
                        well_uid=entity.well_uid,
                        well_name=entity.canonical_name,
                        log_source=log.source,
                        seismic_source=str(asset.path),
                        md=float(md_full[original_index]),
                        tvd=float(tvd_full[original_index]) if np.isfinite(tvd_full[original_index]) else None,
                        tvdss=float(tvdss_full[original_index]) if np.isfinite(tvdss_full[original_index]) else None,
                        x=float(x),
                        y=float(y),
                        trace_index=int(trace_index),
                        inline=int(geom.inline[trace_index]) if geom.inline is not None else None,
                        crossline=int(geom.crossline[trace_index]) if geom.crossline is not None else None,
                        distance=float(distance),
                        seismic_coordinate=seismic_coordinate,
                        horizontal_confidence=horizontal_confidence,
                        vertical_method=alignment.method,
                        vertical_confidence=alignment.confidence,
                        well_features=features,
                        well_mask=masks,
                        seismic_window=seismic_window,
                        provenance={
                            "well_head": head.source, "trajectory": trajectory_source, "log": log.source,
                            "seismic": str(asset.path), "curve_mapping_version": "curve_knowledge_v1",
                            "segy_profile": geom.profile, "segy_revision": geom.revision,
                            "log_asset_id": self.asset_by_path.get(log.source).asset_id if self.asset_by_path.get(log.source) else None,
                            "log_stage": self.asset_by_path.get(log.source).stage if self.asset_by_path.get(log.source) else "UNKNOWN",
                            "log_version": self.asset_by_path.get(log.source).version if self.asset_by_path.get(log.source) else log.version,
                            "seismic_asset_id": asset.asset_id, "seismic_stage": asset.stage, "seismic_version": asset.version,
                            "neighbor_trace_indices": list(nearest.neighbor_trace_indices),
                            "neighbor_distances": list(nearest.neighbor_distances),
                            "neighbor_weights": list(nearest.interpolation_weights),
                            "coordinate_reference": {
                                "verified": coordinate_reference_verified,
                                "crs": coordinate_reference.get("crs"),
                                "horizontal_unit": coordinate_reference.get("horizontal_unit", "m"),
                            },
                            "vertical_alignment": tie_metadata,
                        },
                        vertical_status=alignment.status,
                        vertical_uncertainty_ms=alignment.uncertainty_ms,
                        seismic_window_valid=seismic_window_valid,
                        coordinate_reference_verified=coordinate_reference_verified,
                        training_eligible=training_eligible,
                    )
                    samples.append(record.to_dict())
        self.samples = samples
        return samples

    def _selected_seismic_sources(self, matching: dict[str, Any]) -> list[tuple[Any, SegyReader]]:
        sources = list(self.seismic)
        selection = matching.get("survey_selection", {})
        asset_id = selection.get("asset_id")
        path_contains = str(selection.get("path_contains") or "").strip().lower()
        if asset_id:
            sources = [item for item in sources if item[0].asset_id == asset_id]
        if path_contains:
            sources = [item for item in sources if path_contains in str(item[0].path).lower()]
        mode = str(selection.get("mode", "nearest_3d")).lower()
        if not sources or mode == "all":
            return sources

        def is_3d(item: tuple[Any, SegyReader]) -> bool:
            geometry = item[1].geometry
            if geometry is None or geometry.inline is None or geometry.crossline is None:
                return False
            inline = geometry.inline[np.isfinite(geometry.inline)]
            crossline = geometry.crossline[np.isfinite(geometry.crossline)]
            return len(np.unique(inline)) > 1 and len(np.unique(crossline)) > 1

        candidates = [item for item in sources if is_3d(item)] or sources
        if mode == "nearest_3d":
            return candidates
        return [max(candidates, key=lambda item: item[1].geometry.trace_count if item[1].geometry else 0)]

    @staticmethod
    def _composite_trace(match: Any) -> np.ndarray:
        indices = match.neighbor_trace_indices or (match.trace_index,)
        weights = np.asarray(match.interpolation_weights or (1.0,), dtype=float)
        stack = np.vstack([match.reader.read_trace(int(index)).astype(float) for index in indices])
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
        tvdss: np.ndarray,
    ) -> np.ndarray:
        return {"md": md, "tvd": tvd, "tvdss": tvdss}.get(str(domain).lower(), tvd)

    def _vertical_alignment(
        self,
        entity: Any,
        log: Any,
        md: np.ndarray,
        tvd: np.ndarray,
        tvdss: np.ndarray,
        reference_match: Any | None,
    ) -> TimeDomainAlignment:
        if entity.time_depth:
            table = max(entity.time_depth, key=lambda item: len(item.depth))
            depth = np.asarray(table.depth, dtype=float)
            time = np.asarray(table.time, dtype=float)
            if str(table.depth_unit).lower() in {"ft", "feet", "foot"}:
                depth = depth * 0.3048
            if str(table.time_unit).lower() in {"s", "sec", "second", "seconds"}:
                time = time * 1000.0
            transform = ProvidedTimeDepthTransform(depth, time)
            return TimeDomainAlignment(
                transform=transform,
                status="provided_tie",
                method="provided_time_depth",
                confidence=float(table.confidence),
                uncertainty_ms=None,
                training_eligible=True,
                depth_domain=table.depth_domain,
                diagnostics={
                    "source": table.source,
                    "depth_unit": "m",
                    "time_unit": "ms",
                },
            )

        vertical = self.config.get("matching", {}).get("vertical", {})
        sonic = vertical.get("sonic", {})
        preferred = [str(item).lower() for item in vertical.get("preferred", [])]
        if (
            reference_match is not None
            and sonic.get("enabled", True)
            and (not preferred or "sonic_integrated" in preferred)
        ):
            geometry = reference_match.reader.geometry
            if geometry is not None:
                try:
                    alignment = build_sonic_time_domain_alignment(
                        tvd,
                        log.curves,
                        self._composite_trace(reference_match),
                        geometry.time_axis,
                        sonic,
                    )
                    if alignment is not None:
                        return alignment
                except (ValueError, FloatingPointError) as exc:
                    log.issues.append(f"sonic_well_tie_failed:{type(exc).__name__}:{exc}")

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
                depth_domain="tvd",
                diagnostics={"velocity_m_s": float(velocity)},
            )
        return TimeDomainAlignment(
            transform=NoDepthTimeTransform(),
            status="horizontal_only",
            method="none",
            confidence=0.0,
            uncertainty_ms=None,
            training_eligible=False,
            depth_domain="tvd",
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

    def write_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = self.quality_report()
        report["preparation"] = build_preparation_report(self)
        json_path = output / "质量报告.json"
        json_path.write_text(json.dumps(_json_value(self._chinese_report(report)), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = output / "质量报告.md"
        md_path.write_text(self._markdown_report(report), encoding="utf-8")
        sample_jsonl = output / "多模态样本.jsonl"
        with sample_jsonl.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(_json_value(sample_to_chinese(sample)), ensure_ascii=False) + "\n")
        index_path = output / "样本索引.csv"
        fields = [
            "well_uid", "well_name", "md", "tvd", "tvdss", "x", "y", "trace_index",
            "inline", "crossline", "distance", "seismic_coordinate", "horizontal_confidence",
            "vertical_method", "vertical_status", "vertical_confidence", "vertical_uncertainty_ms",
            "seismic_window_valid", "coordinate_reference_verified", "training_eligible",
            "log_source", "seismic_source",
        ]
        with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
            chinese_fields = [SAMPLE_FIELDS_ZH[field] for field in fields]
            writer = csv.DictWriter(handle, fieldnames=chinese_fields)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow({SAMPLE_FIELDS_ZH[key]: sample.get(key) for key in fields})
        return {"质量报告JSON": json_path, "质量报告Markdown": md_path, "多模态样本": sample_jsonl, "样本索引": index_path}

    @staticmethod
    def _chinese_report(report: dict[str, Any]) -> dict[str, Any]:
        summary = report["summary"]
        return {
            "汇总": {
                "数据资产数": summary["assets"], "跳过重复文件数": summary["duplicates_skipped"],
                "井实体数": summary["wells"], "地震文件数": summary["seismic_files"],
                "多模态样本数": summary["samples"], "读取错误数": summary["errors"],
            },
            "运行时输入识别": report.get("automatic_inventory"),
            "井相关文件识别": report.get("metadata_detection", []),
            "LLM受控判断": report.get("llm_decisions", []),
            "LLM配置状态": report.get("llm_status", {}),
            "预处理与对齐阶段": report.get("preparation", {}),
            "时间域井震标定": report.get("well_ties", []),
            "数据资产": report["assets"], "重复文件": report["duplicates"],
            "读取错误": report["errors"], "井数据": report["wells"], "地震数据": report["seismic"],
        }

    def quality_report(self) -> dict[str, Any]:
        wells = []
        for entity in self.registry.entities.values():
            wells.append({
                "well_uid": entity.well_uid, "name": entity.canonical_name, "aliases": sorted(entity.aliases),
                "head_count": len(entity.heads), "log_count": len(entity.logs), "trajectory_count": len(entity.trajectories),
                "conflicts": entity.conflicts,
                "logs": [{
                    "source": log.source, "las_version": log.version,
                    "asset_version": self.asset_by_path.get(log.source).version if self.asset_by_path.get(log.source) else None,
                    "stage": self.asset_by_path.get(log.source).stage if self.asset_by_path.get(log.source) else "UNKNOWN",
                    "samples": len(log.depth), "curves": sorted(log.curves), "issues": log.issues,
                    "processing_steps": log.processing_steps,
                } for log in entity.logs],
            })
        seismic = []
        for asset, reader in self.seismic:
            geom = reader.geometry
            if geom:
                seismic.append({
                    "asset": str(asset.path), "revision": geom.revision, "endian": geom.endian,
                    "sample_format": geom.sample_format, "sample_interval_ms": geom.sample_interval,
                    "samples_per_trace": geom.samples_per_trace, "trace_count": geom.trace_count,
                    "profile": geom.profile, "confidence": geom.confidence, "issues": geom.issues,
                })
        return {
            "summary": {
                "assets": len(self.assets), "duplicates_skipped": len(self.duplicates),
                "wells": len(wells), "seismic_files": len(seismic), "samples": len(self.samples),
                "training_eligible_samples": sum(1 for item in self.samples if item.get("training_eligible")),
                "errors": len(self.errors),
            },
            "assets": [{"id": a.asset_id, "role": a.role, "dataset": a.dataset, "stage": a.stage, "path": str(a.path), "size": a.path.stat().st_size} for a in self.assets],
            "duplicates": self.duplicates, "errors": self.errors, "wells": wells, "seismic": seismic,
            "well_ties": self.well_ties,
            "automatic_inventory": self.automatic_inventory,
            "metadata_detection": self.metadata_detection,
            "llm_decisions": list(self.decision_resolver.records),
            "llm_status": self.decision_resolver.settings.public_status(),
        }

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        s = report["summary"]
        lines = ["# 井震数据质量报告", "", "## 汇总", "", f"- 数据资产：{s['assets']}", f"- 去重文件：{s['duplicates_skipped']}", f"- 井实体：{s['wells']}", f"- 地震文件：{s['seismic_files']}", f"- 多模态样本：{s['samples']}", f"- 可训练样本：{s.get('training_eligible_samples', 0)}", f"- 读取错误：{s['errors']}", "", "## 井数据", ""]
        for well in report["wells"]:
            curves = sorted({c for log in well["logs"] for c in log["curves"]})
            lines.append(f"- **{well['name']}**（{well['well_uid']}）：测井 {well['log_count']}，轨迹 {well['trajectory_count']}；曲线：{', '.join(curves) or '无'}")
        lines.extend(["", "## 地震数据", ""])
        for seismic in report["seismic"]:
            lines.append(f"- `{seismic['asset']}`：{seismic['trace_count']} 道 × {seismic['samples_per_trace']} 样点，采样间隔 {seismic['sample_interval_ms']} ms，几何置信度 {seismic['confidence']:.3f}")
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
                issue for issue in preparation.get("issues", []) if issue.get("blocking")
            ]
            if blocking:
                lines.extend(["", "### 阻断问题", ""])
                lines.extend(
                    f"- **{issue['title']}**：{issue['message']}"
                    for issue in blocking
                )
        return "\n".join(lines) + "\n"

    def _assets(self, role: str):
        return [asset for asset in self.assets if asset.role == role]

    def _error(self, asset: Any, exc: Exception) -> None:
        self.errors.append({"asset_id": asset.asset_id, "path": str(asset.path), "role": asset.role, "error": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _header_float(header: dict[str, Evidence], *keys: str) -> float | None:
        for key in keys:
            if key in header:
                try:
                    return float(str(header[key].value).split()[0])
                except ValueError:
                    pass
        return None
