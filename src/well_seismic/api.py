from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import logging
import os
import shutil
import threading
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .api_models import (
    ActiveSnapshotRequest,
    AssistantChatRequest,
    BatchIssueActionRequest,
    HorizontalRegistrationRequest,
    InputDiscoveryRequest,
    InspectionRequest,
    IssueConfirmationRequest,
    PredictionRequest,
    PreprocessingRequest,
    RegistrationCandidateAcceptanceRequest,
    RegistrationPreflightFailureDetail,
    RegistrationPreflightRequest,
    RegistrationPreflightResponse,
    RuntimeContractConfirmationRequest,
    RuntimeContractConfirmationResponse,
    RuntimeContractValues,
    SystemCacheClearRequest,
    TaskCreated,
    TransformationActivationRequest,
    ViserLayerModeRequest,
    ViserSliceRequest,
)
from .auto_input import build_explicit_paths_manifest, discover_input_root
from .candidate_visualization import (
    CANDIDATE_DISPLAY_CONTRACT_VERSION,
    SUPPORTED_CANDIDATE_MODEL_IDS,
    evaluate_candidate_visualization,
)
from .cigvis_adapter import (
    plotly_javascript,
    public_visualization_text,
    render_cigvis_workbench,
    update_viser_layer_mode,
    update_viser_slices,
)
from .config import load_config
from .content_identity import canonical_sha256, file_sha256, snapshot_assets_fingerprint
from .coordinate_autodetect import (
    choose_target_crs,
    detect_crs_evidence,
    native_grid_replay_is_consistent,
    verify_pipeline_coordinate_contract,
)
from .coordinate_reference import (
    CoordinateReferenceError,
    require_projected_metre_crs,
)
from .data_flow import build_model_data_flow_specs
from .direct12b_runtime import (
    DIRECT12B_MODEL_ID,
    validate_direct12b_request_options,
)
from .fault_models import is_fault_volume_model_id
from .faultseg_visualization import (
    DEFAULT_FAULTSEG_MASK_SAMPLE_CACHE,
    DEFAULT_FAULTSEG_SLICE_CACHE,
)
from .fusion import build_default_fusion_registry
from .horizontal_registration import (
    HORIZONTAL_REGISTRATION_CONTRACT_VERSION,
    build_horizontal_registration,
    write_horizontal_registration_product,
)
from .interpretation import build_default_interpretation_registry
from .layerpulse_contract import LAYERPULSE_MODEL_ID
from .layerpulse_exports import (
    layerpulse_class_legend_artifact_key,
    layerpulse_segy_artifact_key,
    materialize_layerpulse_common_exports,
    resolve_layerpulse_output_spec,
)
from .layerpulse_visualization import evaluate_layerpulse_visualization
from .llm import build_decision_resolver, build_structured_generator, load_llm_settings
from .llm.parse_repair import REPAIR_CONTRACT_VERSION, repair_fingerprint
from .llm.privacy import issue_local_paths, sanitize_llm_payload
from .llm.transformation import activate_transformation, create_transformation_draft
from .modeling import (
    ModelInputRequest,
    build_default_input_adapters,
    build_default_registry,
)
from .modeling.input_adapters import (
    ARCHIVED_WELL_PROPERTY_COMPLETION_MODEL_IDS,
    FAULTSEG_REPRESENTATIVE_SCOPE,
    SNAPSHOT_ONLY_DOWNSTREAM_WELL_MODEL_IDS,
)
from .p13_registration import (
    arbitrate_registration_tracks,
    build_p13_fusion_feature_tracks,
    load_registration_points,
    registration_evidence_priority,
    run_p13_registration_candidates,
)
from .pipeline import WellSeismicPipeline
from .platform_capabilities import build_platform_capabilities
from .platform_mode import INTERFACES_ONLY_MODE, interface_only_enabled
from .prediction import build_default_prediction_runners
from .prediction_input_attestation import (
    attest_prediction_registration_consumption,
)
from .prediction_visualization import (
    build_prediction_visualization_payload,
    supported_prediction_visualization_models,
)
from .prepared_view import (
    validate_prepared_view_manifest,
    write_prepared_view_manifest,
)
from .registration_contract import (
    FUSION_FEATURE_TRACK_CONTRACT_VERSION,
    read_registration_product_v3,
    write_registration_product_v3,
)
from .releases import ReleaseCatalog, build_release_catalog
from .result_display_acceptance import (
    CONTRACT_VERSION as RESULT_DISPLAY_ACCEPTANCE_CONTRACT_VERSION,
)
from .result_display_acceptance import (
    comparison_panels_from_layers,
    evaluate_result_display_acceptance,
)
from .snapshot_contract import (
    REGISTRATION_EVIDENCE_CONTRACT_VERSION,
    SOURCE_SNAPSHOT_CONTRACT_VERSION,
    SYSTEM_EVIDENCE_ALLOWED_RULE_IDS,
    SYSTEM_EVIDENCE_MIN_CONFIDENCE,
    build_source_snapshot_manifest,
    survey_attestation_reuse_key,
    validate_snapshot_request_semantics,
)
from .sealed_geometry_cache import (
    SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
    SealedGeometryCacheMiss,
    cache_manifest_path,
    load_sealed_geometry_cache,
    write_sealed_geometry_cache,
)
from .segy_geometry_receipt import (
    DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE as GEOPATH_MINIMUM_GEOMETRY_CONFIDENCE,
    build_verified_snapshot_segy_geometry_receipts,
)
from .state_store import (
    ConcurrentStateError,
    RecordNotFoundError,
    SQLiteStateStore,
    StateStoreError,
)
from .standard_export import (
    append_output_file_integrity,
    append_standard_manifest_integrity,
    materialize_legacy_bounded_spatial_slice_bundle,
    materialize_standard_spatial_slice_bundle,
    recover_standard_preview_from_slice_bundle,
    write_standard_result_manifest,
)
from .standard_results import (
    build_standard_result_bundle,
    render_standard_result_visualization,
    resolve_standard_result_artifact,
    supports_standard_well_sequence_view,
)
from .system_cache import SystemCacheService
from .task_runtime import TASK_RUNTIME, TaskCancellationRequested
from .visualization_layers import prediction_result_to_artifact_bundle
from .visualization_preview import build_visualization_preview
from .well_sequence_visualization import build_standard_well_sequence_preview
from .registry import normalize_well_name
from .workflow import build_preparation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
_SNAPSHOT_SEMANTIC_CONFIG_PATHS = (
    "curve_knowledge.yaml",
    "faultseg.yaml",
    "fusion.yaml",
    "geopath_tie_v1_chengdu_folds.json",
    "llm.yaml",
    "matching.yaml",
    "preprocessing.yaml",
    "segy_profiles.yaml",
    "surface_seg.yaml",
    "units.yaml",
    "vertical_datum.yaml",
    "well_schema.yaml",
)
_DETERMINISTIC_UNIT_INHERITANCE_CONTRACT_VERSION = (
    "well-seismic.deterministic-unit-inheritance.v1"
)
_DETERMINISTIC_UNIT_INHERITANCE_RULE_ID = "same_well_las_md_endpoint_unique_m_or_ft.v1"
_SERVICE_SOURCE_IDENTITY_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src" / "well_seismic").rglob("*.py")
    )
)
_SERVICE_BUILD_IDENTITY_PATHS = (
    *_SERVICE_SOURCE_IDENTITY_PATHS,
    "scripts/run_faultseg_subprocess.py",
    "scripts/run_wellfuse_facies_3d.py",
    "scripts/run_wellfuse_geobody.py",
    "scripts/run_wellfuse_horizon.py",
    "scripts/run_wellfuse_well_models.py",
    "runtime/wellfuse/src/wellfuse5090/p17_horizon_unknown.py",
    "runtime/wellfuse/src/wellfuse5090/raw_well_downstream.py",
    "runtime/wellfuse/STRUCTURAL_SHA256.json",
    "runtime/wellfuse/manifests/WELL_SHA256.json",
    *(f"configs/{relative}" for relative in _SNAPSHOT_SEMANTIC_CONFIG_PATHS),
)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
RAW_WELL_UPLOAD_ROOT = PROJECT_ROOT / "model_outputs" / "raw_well_uploads"
SOURCE_SNAPSHOT_ROOT = PROJECT_ROOT / "model_outputs" / "source_snapshots"
SEALED_GEOMETRY_CACHE_ROOT = PROJECT_ROOT / ".runtime-cache" / "sealed-segy-geometry"
DEFAULT_PROJECT_ID = "local-default"
LOGGER = logging.getLogger(__name__)
_SERVICE_INSTANCE_ID = uuid.uuid4().hex


_TaskCancellationRequested = TaskCancellationRequested
_CACHE_CLEAR_TASK_STOP_TIMEOUT_SECONDS = 30.0


def _existing_disk_probe(path: Path) -> Path:
    """Return the nearest existing parent without creating an output directory."""

    probe = path.expanduser().resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def _faultseg_prediction_output_base(seismic_path: str | Path) -> Path:
    """Choose a portable Windows disk-backed root for full-survey FaultSeg.

    An explicit FaultSeg root wins, followed by the shared WellFuse artifact
    root.  Without either setting, keep the existing project output root when
    it has the configured full-volume budget; otherwise prefer the SEG-Y drive
    when that drive can hold the complete disk-backed working set.  The chosen
    path is persisted in the effective task request, so reload and integrity
    checks never have to rediscover it heuristically.
    """

    explicit = str(os.getenv("WELL_SEISMIC_FAULTSEG_OUTPUT_ROOT") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    shared = str(os.getenv("WELLFUSE_ARTIFACT_ROOT") or "").strip()
    if shared:
        return (Path(shared).expanduser().resolve() / "model_outputs" / "faultseg_full_volume")

    default = (PROJECT_ROOT / "model_outputs").resolve()
    faultseg_config = _platform_config.get("faultseg", _platform_config)
    required = int(
        faultseg_config.get("maximum_full_volume_working_bytes", 64 * 1024**3)
    )
    try:
        if int(shutil.disk_usage(_existing_disk_probe(default)).free) >= required:
            return default
    except OSError:
        pass

    source = Path(seismic_path).expanduser().resolve()
    source_root = Path(source.anchor) if source.anchor else source.parent
    try:
        if int(shutil.disk_usage(_existing_disk_probe(source_root)).free) >= required:
            return source_root / "WellFuseArtifacts" / "model_outputs" / "faultseg_full_volume"
    except OSError:
        pass
    return default


app = FastAPI(
    title="地层慧眼",
    version="0.3.0",
    description="油气甜点智能识别的地震—测井多模态统一表征大模型平台",
)

if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


_tasks: dict[str, dict[str, Any]] = {}
_transformation_drafts: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_task_reset_lock = threading.Lock()
_cancelled_task_ids: set[str] = set()
_registration_preflight_lock = threading.Lock()
_horizontal_registration_submission_lock = threading.Lock()
_workflow_submission_lock = threading.Lock()
_prediction_submission_lock = threading.Lock()
_standard_result_migration_locks_guard = threading.Lock()
_standard_result_migration_locks: dict[str, threading.Lock] = {}
_snapshot_file_integrity_cache_lock = threading.Lock()
_snapshot_file_integrity_cache: dict[
    tuple[str, int, str], tuple[int, int, int, int, int]
] = {}
_sealed_geometry_cache_locks_guard = threading.Lock()
_sealed_geometry_cache_locks: dict[str, threading.Lock] = {}
_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="well-seismic-inspection"
)
# GPU inference is serialized deliberately.  Running several small models at the
# same time makes the desktop look busy but risks WDDM OOMs and corrupts the
# provenance of throughput measurements.
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="well-seismic-gpu")
_state_store = SQLiteStateStore()
_model_registry = build_default_registry()
_model_registry.load_entry_points()
_interpretation_registry = build_default_interpretation_registry()
_interpretation_registry.load_entry_points()
_fusion_registry = build_default_fusion_registry()
_fusion_registry.load_entry_points()
_platform_config = load_config(CONFIG_DIR, {"inputs": []})
_input_adapters = build_default_input_adapters(_platform_config)
_input_adapters.load_entry_points(_platform_config)
_prediction_runners = build_default_prediction_runners()
_prediction_runners.load_entry_points()
_release_catalog: ReleaseCatalog = build_release_catalog(project_root=PROJECT_ROOT)
_model_registry.apply_release_catalog(_release_catalog)
_system_cache = SystemCacheService.for_project(
    project_root=PROJECT_ROOT,
    memory_caches={
        "segy_visualization": DEFAULT_FAULTSEG_SLICE_CACHE,
        "fault_mask_visualization": DEFAULT_FAULTSEG_MASK_SAMPLE_CACHE,
    },
)
TRANSFORMATION_REGISTRY = PROJECT_ROOT / "输出结果" / "智能转换插件" / "已启用转换.json"


def _standard_result_migration_lock_for(task_id: str) -> threading.Lock:
    """Serialize legacy result migration per task, never across all results."""

    with _standard_result_migration_locks_guard:
        return _standard_result_migration_locks.setdefault(task_id, threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recover_interrupted_runtime_tasks() -> None:
    """Fail orphaned workers explicitly instead of leaving clients locked forever."""

    recovered = _state_store.recover_interrupted_tasks(as_status="failed")
    recovered_ids = {str(task["task_id"]) for task in recovered}
    queued = _state_store.list_tasks(status="queued", limit=10_000)
    interrupted = [
        *recovered,
        *(task for task in queued if task["task_id"] not in recovered_ids),
    ]
    if not interrupted:
        return
    recovered_at = _now()
    for task in interrupted:
        task_id = str(task["task_id"])
        recovery = task.get("recovery")
        previous_status = (
            recovery.get("previous_status")
            if isinstance(recovery, dict)
            else task.get("status")
        )
        _state_store.update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "message": "平台服务曾重启，原后台任务已中断，请重新运行",
                "error": {
                    "type": "ServiceRestartInterruptedTask",
                    "message": "平台服务重启后无法安全续跑原进程内任务，请重新提交",
                },
                "progress_detail": {"phase": "failed", "can_estimate": False},
                "recovery": {
                    "previous_status": previous_status,
                    "recovered_at": recovered_at,
                    "policy": "fail_closed_after_service_restart",
                },
            },
        )
    LOGGER.warning("服务启动时将 %d 个孤立任务标记为失败", len(interrupted))


app.router.add_event_handler("startup", _recover_interrupted_runtime_tasks)


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _service_build_identity() -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in _SERVICE_BUILD_IDENTITY_PATHS:
        path = (PROJECT_ROOT / relative).resolve()
        files[relative] = _sha256_file(path) if path.is_file() else "missing"
    return {
        "contract_version": "well-seismic.service-build.v1",
        "files": files,
        "sha256": canonical_sha256(files),
    }


_SERVICE_BUILD_IDENTITY = _service_build_identity()


def _wellfuse_python_identity() -> dict[str, Any]:
    """Freeze the exact interpreter executable selected by prediction runners."""

    configured = os.getenv("WELLFUSE_PYTHON")
    executable = (
        Path(configured).expanduser().resolve()
        if configured and configured.strip()
        else (PROJECT_ROOT / "runtime" / "python-wellfuse" / "python.exe").resolve()
    )
    exists = executable.is_file()
    size = executable.stat().st_size if exists else None
    digest = _sha256_file(executable) if exists else "missing"
    document = {
        "path": str(executable),
        "size": size,
        "sha256": digest,
    }
    return {
        "contract_version": "well-seismic.python-executable.v1",
        **document,
        "identity_sha256": canonical_sha256(document),
        "status": "verified" if exists and size and len(digest) == 64 else "missing",
    }


_WELLFUSE_PYTHON_IDENTITY = _wellfuse_python_identity()


def _wellfuse_runtime_build_identity() -> dict[str, Any]:
    """Freeze the Python/runtime manifests actually selected for subprocesses."""

    configured_root = os.getenv("WELLFUSE_PROJECT_ROOT")
    root = (
        Path(configured_root).expanduser().resolve()
        if configured_root and configured_root.strip()
        else (PROJECT_ROOT / "runtime" / "wellfuse").resolve()
    )
    candidates: set[Path] = set()
    for directory in (root / "scripts", root / "src" / "wellfuse5090"):
        if directory.is_dir():
            candidates.update(path.resolve() for path in directory.rglob("*.py"))
    manifest_directory = root / "manifests"
    if manifest_directory.is_dir():
        candidates.update(path.resolve() for path in manifest_directory.rglob("*.json"))
    if root.is_dir():
        candidates.update(path.resolve() for path in root.glob("*_SHA256.json"))
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(candidates, key=lambda item: str(item).casefold())
        if path.is_file()
    }
    document = {"root": str(root), "files": files}
    return {
        "contract_version": "well-seismic.wellfuse-runtime-build.v1",
        **document,
        "sha256": canonical_sha256(document),
    }


_WELLFUSE_RUNTIME_BUILD_IDENTITY = _wellfuse_runtime_build_identity()


def _snapshot_file_stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


def _verify_snapshot_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> tuple[int, int, int, int, int]:
    """Verify once per unchanged file identity during this server lifetime.

    A sealed snapshot still owns the full SHA-256.  Reusing a prior successful
    verification is permitted only while size, mtime, ctime, device and file id
    all remain identical; any metadata change forces a fresh full hash.  The
    before/after signature also rejects a file changed during hashing.
    """

    before = _snapshot_file_stat_signature(path)
    if before[0] != expected_size:
        raise ValueError(
            f"source data snapshot asset size changed at the same path: {path}"
        )
    cache_key = (str(path), expected_size, expected_sha256.casefold())
    with _snapshot_file_integrity_cache_lock:
        if _snapshot_file_integrity_cache.get(cache_key) == before:
            return before
    observed_sha256 = _sha256_file(path).casefold()
    after = _snapshot_file_stat_signature(path)
    if after != before:
        raise ValueError(
            f"source data snapshot asset changed while it was verified: {path}"
        )
    if observed_sha256 != expected_sha256.casefold():
        raise ValueError(
            f"source data snapshot asset content changed at the same path: {path}"
        )
    with _snapshot_file_integrity_cache_lock:
        _snapshot_file_integrity_cache[cache_key] = after
    return after


def _snapshot_verified_stat_signatures(
    snapshot_context: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int, int]]:
    """Return the exact file identities verified for this task invocation."""

    signatures: dict[str, tuple[int, int, int, int, int]] = {}
    for raw in snapshot_context.get("snapshot_assets") or []:
        if (
            not isinstance(raw, Mapping)
            or raw.get("integrity_status") != "sha256_verified"
        ):
            continue
        path = Path(str(raw.get("path") or "")).expanduser().resolve()
        values = raw.get("verified_stat_signature")
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            )
        ):
            raise ValueError(
                f"source data snapshot has no verified runtime file identity: {path}"
            )
        key = os.path.normcase(str(path))
        if key in signatures:
            raise ValueError(f"source data snapshot contains duplicate paths: {path}")
        signatures[key] = tuple(int(value) for value in values)  # type: ignore[assignment]
    return signatures


def _assert_snapshot_verified_stat_signatures(
    snapshot_context: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int, int]]:
    """Fail closed on stat-visible drift after full SHA verification.

    The runtime signature covers size, mtime, ctime, device and file id.  It is
    deliberately a cheap TOCTOU guard, not a replacement for the SourceSnapshot
    SHA-256 and not a claim to detect an in-place rewrite that preserves every
    one of those filesystem fields.
    """

    signatures = _snapshot_verified_stat_signatures(snapshot_context)
    for path_key, expected in signatures.items():
        path = Path(path_key)
        try:
            observed = _snapshot_file_stat_signature(path)
        except OSError as exc:
            raise ValueError(
                f"source data snapshot asset disappeared after verification: {path}"
            ) from exc
        if observed != expected:
            raise ValueError(
                "source data snapshot asset changed after verification; refusing "
                f"to combine stale geometry with amplitudes: {path}"
            )
    return signatures


def _declared_output_paths(
    value: object, *, prefix: str = ""
) -> list[tuple[str, Path]]:
    """Return producer-declared output paths without mistaking metadata for paths."""

    paths: list[tuple[str, Path]] = []
    if isinstance(value, dict):
        declared_path = value.get("path")
        if isinstance(declared_path, str) and declared_path.strip():
            paths.append(
                (prefix or "output", Path(declared_path).expanduser().resolve())
            )
            return paths
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_declared_output_paths(item, prefix=label))
    elif isinstance(value, str) and value.strip():
        paths.append((prefix or "output", Path(value).expanduser().resolve()))
    return paths


def _separate_prediction_external_evidence(
    result: dict[str, Any], *, output_root: Path
) -> None:
    """Move allow-listed immutable benchmark references out of producer outputs."""

    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        return
    # ``output_root`` remains part of the call contract for backwards
    # compatibility; membership does not turn model/checkpoint evidence into a
    # prediction output.
    del output_root
    external: dict[str, dict[str, Any]] = {}
    external_output_keys = {
        "benchmark_metrics_json",
        "checkpoint",
        "config_json",
        "metrics_json",
        "raw_prepare_receipt",
    }
    for key, raw_value in list(outputs.items()):
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        if str(key) not in external_output_keys:
            continue
        path = Path(raw_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"prediction external benchmark evidence is missing: {key}={path}"
            )
        external[str(key)] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "relationship": (
                "immutable_external_evaluation_evidence_not_output"
                if str(key) in {"benchmark_metrics_json", "metrics_json"}
                else "immutable_external_runtime_evidence_not_output"
            ),
        }
        outputs.pop(key)
    if external:
        result.setdefault("external_evidence", {}).update(external)


def _seal_prediction_output_integrity(
    result: dict[str, Any], *, producer_task_id: str, output_root: Path
) -> dict[str, Any]:
    """Seal every prediction output before a task may become ``completed``."""

    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("prediction outputs must be a mapping before sealing")
    root = output_root.expanduser().resolve()
    declared = _declared_output_paths(outputs)
    if not declared:
        raise ValueError("prediction produced no sealable output files")
    artifacts: dict[str, dict[str, Any]] = {}
    sealed_file_digests: dict[
        tuple[str, tuple[int, int, int, int, int]], str
    ] = {}
    for label, path in declared:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"prediction output escapes its task directory: {label}={path}"
            ) from exc
        if path.is_dir():
            children: dict[str, dict[str, Any]] = {}
            total_size = 0
            for child in sorted(path.rglob("*"), key=lambda item: str(item).casefold()):
                if not child.is_file():
                    continue
                resolved_child = child.resolve()
                try:
                    resolved_child.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"prediction output directory escapes its task root: {child}"
                    ) from exc
                before = _snapshot_file_stat_signature(resolved_child)
                identity = (str(resolved_child), before)
                digest = sealed_file_digests.get(identity)
                if digest is None:
                    digest = _sha256_file(resolved_child)
                after = _snapshot_file_stat_signature(resolved_child)
                if before != after:
                    raise ValueError(
                        f"prediction output changed while sealing: {label}={child}"
                    )
                sealed_file_digests[identity] = digest
                relative = resolved_child.relative_to(path).as_posix()
                children[relative] = {"size": after[0], "sha256": digest}
                total_size += after[0]
            if not children:
                raise ValueError(
                    f"prediction output directory is empty: {label}={path}"
                )
            artifacts[label] = {
                "kind": "directory",
                "path": str(path),
                "size": total_size,
                "file_count": len(children),
                "sha256": canonical_sha256(children),
                "children": children,
            }
            continue
        if not path.is_file():
            raise FileNotFoundError(f"prediction output is missing: {label}={path}")
        before = _snapshot_file_stat_signature(path)
        if before[0] <= 0:
            raise ValueError(f"prediction output is empty: {label}={path}")
        identity = (str(path), before)
        digest = sealed_file_digests.get(identity)
        if digest is None:
            digest = _sha256_file(path)
        after = _snapshot_file_stat_signature(path)
        if after != before:
            raise ValueError(f"prediction output changed while sealing: {label}={path}")
        sealed_file_digests[identity] = digest
        artifacts[label] = {
            "kind": "file",
            "path": str(path),
            "size": after[0],
            "sha256": digest,
        }
    document: dict[str, Any] = {
        "contract_version": "well-seismic.prediction-output-integrity.v1",
        "producer_task_id": producer_task_id,
        "interpretation_task_id": str(result.get("task_id") or ""),
        "model_id": str(result.get("model_id") or ""),
        "source_snapshot_id": str(result.get("source_snapshot_id") or ""),
        "artifacts": artifacts,
    }
    document["integrity_sha256"] = canonical_sha256(document)
    return document


def _effective_configuration_sha256() -> str:
    records: list[dict[str, Any]] = []
    for relative_path in _SNAPSHOT_SEMANTIC_CONFIG_PATHS:
        path = CONFIG_DIR / relative_path
        record: dict[str, Any] = {"path": relative_path}
        if path.is_file():
            record["sha256"] = _sha256_file(path)
        else:
            # Missing scientific configuration must change the contract rather
            # than silently looking identical to a complete installation.
            record["status"] = "missing"
        records.append(record)
    return canonical_sha256(records)


def _transformation_registry_sha256() -> str:
    if TRANSFORMATION_REGISTRY.is_file():
        return _sha256_file(TRANSFORMATION_REGISTRY)
    return canonical_sha256({"status": "absent"})


def _write_pipeline_sealed_geometry_cache(
    pipeline: WellSeismicPipeline,
    snapshot_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the parsed SEG-Y subset as a regenerable, content-bound cache."""

    snapshot_id = str(snapshot_context.get("source_snapshot_id") or "")
    snapshot_sha256 = str(snapshot_context.get("source_snapshot_fingerprint") or "")
    snapshot_assets = snapshot_context.get("snapshot_assets") or []
    if (
        not snapshot_id
        or len(snapshot_sha256) != 64
        or snapshot_context.get("snapshot_contract_version")
        != SOURCE_SNAPSHOT_CONTRACT_VERSION
    ):
        raise SealedGeometryCacheMiss(
            "only a sealed SourceSnapshot V3 can own a geometry cache"
        )
    return write_sealed_geometry_cache(
        SEALED_GEOMETRY_CACHE_ROOT,
        snapshot_id=snapshot_id,
        source_snapshot_sha256=snapshot_sha256,
        snapshot_assets=[
            dict(item) for item in snapshot_assets if isinstance(item, Mapping)
        ],
        catalog_assets=pipeline.assets,
        seismic=pipeline.seismic,
        effective_config_sha256=_effective_configuration_sha256(),
        transformation_registry_sha256=_transformation_registry_sha256(),
        service_build_sha256=str(_SERVICE_BUILD_IDENTITY["sha256"]),
    )


def _sealed_geometry_cache_identity_kwargs(
    snapshot_context: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "snapshot_id": str(snapshot_context.get("source_snapshot_id") or ""),
        "source_snapshot_sha256": str(
            snapshot_context.get("source_snapshot_fingerprint") or ""
        ),
        "effective_config_sha256": _effective_configuration_sha256(),
        "transformation_registry_sha256": _transformation_registry_sha256(),
        "service_build_sha256": str(_SERVICE_BUILD_IDENTITY["sha256"]),
    }


def _trusted_source_snapshot_geometry_cache_receipt(
    snapshot_context: Mapping[str, Any],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    raw: Any = snapshot_context.get("source_snapshot_segy_geometry_cache_receipt")
    if refresh:
        try:
            source_task = _get_task(
                str(snapshot_context.get("source_snapshot_id") or "")
            )
        except KeyError:
            source_task = None
        if source_task is not None:
            raw = ((source_task.get("result") or {}).get("data_snapshot") or {}).get(
                "segy_geometry_cache"
            )
    return dict(raw) if isinstance(raw, Mapping) else {}


def _persist_source_snapshot_geometry_cache_receipt(
    snapshot_context: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Atomically promote a regenerable cache identity into the control plane."""

    source_task_id = str(snapshot_context.get("source_snapshot_id") or "")
    manifest_sha256 = str(receipt.get("manifest_sha256") or "").casefold()
    if (
        not source_task_id
        or len(manifest_sha256) != 64
        or str(receipt.get("source_snapshot_id") or "") != source_task_id
        or str(receipt.get("source_snapshot_sha256") or "").casefold()
        != str(snapshot_context.get("source_snapshot_fingerprint") or "").casefold()
    ):
        raise ValueError("SEG-Y geometry cache receipt is not bound to this snapshot")

    # SourceSnapshot is immutable, but this cache receipt is explicitly a
    # regenerable control-plane projection.  Write the SQLite record before
    # publishing it to the in-process task cache so a failed persistence never
    # creates a hit that cannot survive restart.
    with _tasks_lock:
        current = _tasks.get(source_task_id)
        if current is None:
            current = _state_store.get_task(source_task_id)
        updated = copy.deepcopy(current)
        if updated.get("status") != "completed":
            raise ValueError("only a completed SourceSnapshot may own a cache receipt")
        result = dict(updated.get("result") or {})
        data_snapshot = dict(result.get("data_snapshot") or {})
        data_snapshot["segy_geometry_cache"] = dict(receipt)
        result["data_snapshot"] = data_snapshot
        updated["result"] = result
        updated["updated_at"] = _now()
        _state_store.update_task(source_task_id, updated)
        _tasks[source_task_id] = updated
    if isinstance(snapshot_context, dict):
        snapshot_context["source_snapshot_segy_geometry_cache_receipt"] = dict(receipt)


def _sealed_geometry_singleflight_lock(
    snapshot_context: Mapping[str, Any],
) -> threading.Lock:
    identity = _sealed_geometry_cache_identity_kwargs(snapshot_context)
    key = str(
        cache_manifest_path(
            SEALED_GEOMETRY_CACHE_ROOT,
            **identity,
        )
    )
    with _sealed_geometry_cache_locks_guard:
        return _sealed_geometry_cache_locks.setdefault(key, threading.Lock())


def _ingest_pipeline_from_sealed_snapshot(
    pipeline: WellSeismicPipeline,
    snapshot_context: Mapping[str, Any],
    *,
    progress: Any = None,
    cache_status: Any = None,
) -> dict[str, Any]:
    """Replay small well assets and reuse immutable full SEG-Y geometry.

    Source asset SHA-256 verification happens before this helper in
    ``_prediction_snapshot_context``.  Cache validation additionally binds the
    numeric arrays to the sealed geometry fingerprint and current parser build.
    Any miss or integrity failure falls back to the authoritative source file.
    """

    identity = _sealed_geometry_cache_identity_kwargs(snapshot_context)
    snapshot_assets = [
        dict(item)
        for item in (snapshot_context.get("snapshot_assets") or [])
        if isinstance(item, Mapping)
    ]
    source_stat_signatures = _assert_snapshot_verified_stat_signatures(snapshot_context)

    def load_cache(
        *,
        refresh_receipt: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
        trusted = _trusted_source_snapshot_geometry_cache_receipt(
            snapshot_context,
            refresh=refresh_receipt,
        )
        try:
            cached, receipt = load_sealed_geometry_cache(
                SEALED_GEOMETRY_CACHE_ROOT,
                snapshot_assets=snapshot_assets,
                catalog_assets=pipeline.assets,
                expected_manifest_sha256=str(trusted.get("manifest_sha256") or "")
                or None,
                **identity,
            )
            return cached, receipt, ""
        except SealedGeometryCacheMiss as exc:
            return {}, None, str(exc)
        except (EOFError, OSError, OverflowError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "封存SEG-Y几何缓存校验失败，将从源文件安全重建：%s",
                exc,
            )
            return {}, None, f"integrity_rejected:{type(exc).__name__}"

    cached_geometry, loaded_receipt, cache_miss_reason = load_cache(
        refresh_receipt=False
    )
    singleflight_lock = _sealed_geometry_singleflight_lock(snapshot_context)
    written_and_persisted_receipt: dict[str, Any] | None = None
    if not cached_geometry:
        singleflight_lock.acquire()
        try:
            # Another worker may have completed the authoritative scan while
            # this worker waited.  Refresh the task receipt, then double-check.
            _assert_snapshot_verified_stat_signatures(snapshot_context)
            cached_geometry, loaded_receipt, second_reason = load_cache(
                refresh_receipt=True
            )
            if not cached_geometry and second_reason:
                cache_miss_reason = second_reason

            if cache_status is not None:
                cache_status(bool(cached_geometry), cache_miss_reason)
            pipeline.ingest(
                progress=progress,
                seismic_geometry_by_path=cached_geometry,
                source_stat_signatures_by_path=source_stat_signatures,
            )
            _assert_snapshot_verified_stat_signatures(snapshot_context)

            misses = int(
                pipeline.ingest_cache_receipt.get("seismic_geometry_misses") or 0
            )
            persisted_receipt: dict[str, Any] | None = None
            if misses > 0:
                try:
                    persisted_receipt = _write_pipeline_sealed_geometry_cache(
                        pipeline,
                        snapshot_context,
                    )
                    _persist_source_snapshot_geometry_cache_receipt(
                        snapshot_context,
                        persisted_receipt,
                    )
                    written_and_persisted_receipt = persisted_receipt
                except Exception as exc:  # regenerable cache cannot invalidate replay
                    LOGGER.warning("无法保存封存SEG-Y几何缓存，派生任务继续：%s", exc)
        finally:
            singleflight_lock.release()
    else:
        if cache_status is not None:
            cache_status(True, cache_miss_reason)
        pipeline.ingest(
            progress=progress,
            seismic_geometry_by_path=cached_geometry,
            source_stat_signatures_by_path=source_stat_signatures,
        )
        _assert_snapshot_verified_stat_signatures(snapshot_context)

    hits = int(pipeline.ingest_cache_receipt.get("seismic_geometry_hits") or 0)
    misses = int(pipeline.ingest_cache_receipt.get("seismic_geometry_misses") or 0)
    state = "hit" if hits > 0 and misses == 0 else "miss"
    persisted_receipt = written_and_persisted_receipt if misses > 0 else None
    if misses > 0:
        persisted = bool(persisted_receipt and persisted_receipt.get("manifest_sha256"))
        if persisted:
            state = "partial_hit_rebuilt" if hits else "miss_rebuilt"
        else:
            state = "partial_hit_not_persisted" if hits else "miss_not_persisted"

    identity_receipt = persisted_receipt or loaded_receipt or {}
    return {
        "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
        "state": state,
        "source_snapshot_id": str(snapshot_context.get("source_snapshot_id") or ""),
        "source_snapshot_sha256": str(
            snapshot_context.get("source_snapshot_fingerprint") or ""
        ),
        "seismic_geometry_hits": hits,
        "seismic_geometry_misses": misses,
        "full_trace_header_scan_performed": misses > 0,
        "cache_miss_reason": cache_miss_reason if misses else None,
        "manifest_sha256": identity_receipt.get("manifest_sha256"),
        "cache_identity_sha256": identity_receipt.get("cache_identity_sha256"),
        "validation": (
            identity_receipt.get("validation")
            or "content_bound_cache_written_after_authoritative_scan"
        ),
        "well_asset_replay_policy": (
            "LAS/DEV are deterministically replayed; sealed SEG-Y full geometry "
            "is reused only after all cache identities pass"
        ),
    }


def _source_snapshot_manifest_from_task(task: dict[str, Any]) -> dict[str, Any] | None:
    result = task.get("result") or {}
    data_snapshot = result.get("data_snapshot") or {}
    raw_path = data_snapshot.get("snapshot_manifest_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"source snapshot manifest is invalid: {path}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("source snapshot manifest must be a JSON object")
    expected_sha = str(data_snapshot.get("snapshot_manifest_sha256") or "")
    if expected_sha and _sha256_file(path) != expected_sha:
        raise ValueError("source snapshot manifest content changed after sealing")
    if str(manifest.get("snapshot_id")) != str(task.get("task_id")):
        raise ValueError("source snapshot manifest id differs from its task lineage")
    return manifest


def _find_reusable_survey_attestation(
    request: InspectionRequest,
    assets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a valid prior human receipt for the exact SEG-Y content contract."""

    if request.survey_attestation is not None:
        return None
    reuse_key = survey_attestation_reuse_key(request, assets)
    if reuse_key is None:
        return None
    try:
        tasks = _state_store.list_tasks(
            project_id=DEFAULT_PROJECT_ID,
            task_type="data_preparation",
            status="completed",
            limit=200,
        )
    except (StateStoreError, TypeError, ValueError) as exc:
        LOGGER.warning("无法检索历史测区声明回执，继续封存当前快照：%s", exc)
        return None
    for task in tasks:
        try:
            manifest = _source_snapshot_manifest_from_task(task)
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        receipt = manifest.get("survey_attestation_receipt")
        hashes = manifest.get("hashes") or {}
        receipt_sha256 = str(hashes.get("survey_attestation_sha256") or "").casefold()
        if not isinstance(receipt, dict) or len(receipt_sha256) != 64:
            continue
        if canonical_sha256(receipt) != receipt_sha256:
            continue
        if receipt.get("source") != "human_user":
            continue
        if str(receipt.get("reuse_key") or "") != reuse_key:
            continue
        return {
            "snapshot_id": str(manifest.get("snapshot_id") or ""),
            "receipt": receipt,
            "receipt_sha256": receipt_sha256,
            "reuse_key": reuse_key,
        }
    return None


def _seal_data_preparation_snapshot(
    task_id: str,
    request: InspectionRequest,
    result: dict[str, Any],
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    parent_snapshot_id: str | None = None,
    system_evidence_decision: dict[str, Any] | None = None,
    runtime_contract_confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the immutable v3 manifest and bind it to the control plane."""

    config_sha256 = _effective_configuration_sha256()
    transformations_sha256 = _transformation_registry_sha256()
    summary = result.get("summary") or {}
    snapshot_assets = [
        item for item in (result.get("assets") or []) if isinstance(item, dict)
    ]
    survey_attestation_reuse_basis = (
        None
        if system_evidence_decision is not None
        else _find_reusable_survey_attestation(request, snapshot_assets)
    )
    preparation = result.get("preparation") or {}
    candidate_group = preparation.get("survey_contract_candidate") or {}
    registration_evidence = None
    if isinstance(candidate_group, dict) and isinstance(
        candidate_group.get("candidates"), list
    ):
        registration_evidence = {
            "candidate_schema_version": candidate_group.get("schema_version"),
            "candidates": list(candidate_group.get("candidates") or []),
            "request_patch": dict(preparation.get("request_patch") or {}),
            "policy": candidate_group.get("policy"),
        }
    manifest = build_source_snapshot_manifest(
        snapshot_id=task_id,
        project_id=project_id,
        created_by_task_id=task_id,
        request=request,
        assets=snapshot_assets,
        effective_config_sha256=config_sha256,
        transformation_registry_sha256=transformations_sha256,
        qc_summary={
            "error_count": int(summary.get("errors") or 0),
            "uncertain_count": int(summary.get("uncertain") or 0),
            "asset_count": int(summary.get("assets") or 0),
            "well_count": int(summary.get("wells") or 0),
        },
        parse_repairs=[
            item
            for collection in (
                result.get("llm_parse_repairs") or [],
                result.get("deterministic_unit_inheritances") or [],
            )
            for item in collection
            if isinstance(item, dict)
        ],
        registration_evidence=registration_evidence,
        survey_attestation_reuse_basis=survey_attestation_reuse_basis,
        parent_snapshot_id=parent_snapshot_id,
        system_evidence_decision=system_evidence_decision,
        runtime_contract_confirmation=runtime_contract_confirmation,
        runtime_contract_review=(
            preparation.get("runtime_contract_review")
            if isinstance(preparation.get("runtime_contract_review"), dict)
            else None
        ),
    )
    if isinstance(manifest.get("runtime_contract_review"), dict):
        preparation["runtime_contract_review"] = copy.deepcopy(
            manifest["runtime_contract_review"]
        )
    output_directory = SOURCE_SNAPSHOT_ROOT / task_id
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "snapshot_manifest.json"
    temporary_manifest_path = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest_path, manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    hashes = dict(manifest["hashes"])
    data_snapshot = result.setdefault("data_snapshot", {})
    data_snapshot.update(
        {
            "contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
            "snapshot_id": task_id,
            "project_id": project_id,
            "state": "sealed",
            "asset_set_sha256": hashes["legacy_asset_set_sha256"],
            "source_content_sha256": hashes["source_content_sha256"],
            "semantics_sha256": hashes["semantics_sha256"],
            "inspection_policy_sha256": hashes["inspection_policy_sha256"],
            "parse_repairs_sha256": hashes["parse_repairs_sha256"],
            "snapshot_sha256": hashes["snapshot_sha256"],
            "snapshot_manifest_path": str(manifest_path.resolve()),
            "snapshot_manifest_sha256": manifest_sha256,
            "semantics": manifest["semantics"],
        }
    )
    attestation_receipt = manifest.get("survey_attestation_receipt")
    if isinstance(manifest.get("registration_evidence"), dict):
        data_snapshot["registration_evidence_sha256"] = hashes[
            "registration_evidence_sha256"
        ]
    if isinstance(manifest.get("runtime_contract_review"), dict):
        data_snapshot["runtime_contract_review_sha256"] = hashes[
            "runtime_contract_review_sha256"
        ]
    if isinstance(attestation_receipt, dict):
        data_snapshot["survey_attestation_receipt"] = dict(attestation_receipt)
        data_snapshot["survey_attestation_sha256"] = hashes["survey_attestation_sha256"]
    system_evidence_receipt = manifest.get("system_evidence_receipt")
    if isinstance(system_evidence_receipt, dict):
        data_snapshot["parent_snapshot_id"] = str(parent_snapshot_id)
        data_snapshot["system_evidence_receipt"] = dict(system_evidence_receipt)
        data_snapshot["system_evidence_receipt_sha256"] = hashes[
            "system_evidence_receipt_sha256"
        ]
    runtime_contract_receipt = manifest.get("runtime_contract_receipt")
    if isinstance(runtime_contract_receipt, dict):
        data_snapshot["parent_snapshot_id"] = str(parent_snapshot_id)
        data_snapshot["runtime_contract_receipt"] = dict(runtime_contract_receipt)
        data_snapshot["runtime_contract_receipt_sha256"] = hashes[
            "runtime_contract_receipt_sha256"
        ]
    _state_store.ensure_project(
        project_id,
        {"name": "本地默认项目", "kind": "desktop_local"},
    )
    _state_store.seal_snapshot(
        project_id,
        task_id,
        {
            "contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
            "state": "sealed",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "hashes": hashes,
            "created_by_task_id": task_id,
            **(
                {
                    "parent_snapshot_id": str(parent_snapshot_id),
                    "system_evidence_receipt_sha256": hashes[
                        "system_evidence_receipt_sha256"
                    ],
                }
                if isinstance(system_evidence_receipt, dict)
                else {}
            ),
            **(
                {
                    "runtime_contract_review_sha256": hashes[
                        "runtime_contract_review_sha256"
                    ]
                }
                if "runtime_contract_review_sha256" in hashes
                else {}
            ),
            **(
                {
                    "parent_snapshot_id": str(parent_snapshot_id),
                    "runtime_contract_receipt_sha256": hashes[
                        "runtime_contract_receipt_sha256"
                    ],
                }
                if isinstance(runtime_contract_receipt, dict)
                else {}
            ),
        },
    )
    _state_store.bind_task_lineage(
        task_id,
        project_id=project_id,
        snapshot_id=task_id,
    )
    bundle = _state_store.create_artifact_bundle(
        {
            "kind": "source_snapshot",
            "contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
            "snapshot_manifest": str(manifest_path.resolve()),
            "snapshot_manifest_sha256": manifest_sha256,
            "snapshot_sha256": hashes["snapshot_sha256"],
            **(
                {"survey_attestation_sha256": hashes["survey_attestation_sha256"]}
                if "survey_attestation_sha256" in hashes
                else {}
            ),
            **(
                {
                    "runtime_contract_review_sha256": hashes[
                        "runtime_contract_review_sha256"
                    ]
                }
                if "runtime_contract_review_sha256" in hashes
                else {}
            ),
            **(
                {
                    "parent_snapshot_id": str(parent_snapshot_id),
                    "system_evidence_receipt_sha256": hashes[
                        "system_evidence_receipt_sha256"
                    ],
                }
                if "system_evidence_receipt_sha256" in hashes
                else {}
            ),
            **(
                {
                    "parent_snapshot_id": str(parent_snapshot_id),
                    "runtime_contract_receipt_sha256": hashes[
                        "runtime_contract_receipt_sha256"
                    ],
                }
                if "runtime_contract_receipt_sha256" in hashes
                else {}
            ),
        },
        bundle_id=f"snapshot-{task_id}",
        project_id=project_id,
        snapshot_id=task_id,
        task_id=task_id,
    )
    data_snapshot["artifact_bundle_id"] = bundle["bundle_id"]
    return manifest


def _set_task(task_id: str, **values: Any) -> None:
    TASK_RUNTIME.check_cancelled(task_id)
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            try:
                task = _state_store.get_task(task_id)
            except RecordNotFoundError as exc:
                raise KeyError(task_id) from exc
            _tasks[task_id] = task
        if task_id in _cancelled_task_ids or task.get("status") == "cancelled":
            # Cache reset makes cancellation an absorbing state.  Every worker
            # reports progress through this function, so a late callback stops
            # the worker instead of reviving a cancelled task or publishing a
            # result after the user has explicitly reset the platform.
            raise _TaskCancellationRequested(task_id)
        task.update(values)
        task["updated_at"] = _now()
        snapshot = dict(task)
        try:
            _state_store.update_task(task_id, snapshot)
        except RecordNotFoundError:
            try:
                _state_store.create_task(snapshot, task_id=task_id)
            except (StateStoreError, TypeError, ValueError) as exc:
                LOGGER.warning("无法持久化任务 %s：%s", task_id, exc)
        except (StateStoreError, TypeError, ValueError) as exc:
            # The runtime task remains available even if a plugin returned a value
            # that cannot be represented in the small JSON control plane.
            LOGGER.warning("无法更新持久化任务 %s：%s", task_id, exc)


def _get_task(task_id: str) -> dict[str, Any]:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is not None:
            return dict(task)
    try:
        task = _state_store.get_task(task_id)
    except RecordNotFoundError as exc:
        raise KeyError(task_id) from exc
    with _tasks_lock:
        _tasks[task_id] = task
    return dict(task)


def _seismic_dimension(geometry: Any) -> tuple[str, str]:
    inline = geometry.inline
    crossline = geometry.crossline
    if inline is not None and crossline is not None:
        unique_inline = int(np.unique(inline).size)
        unique_crossline = int(np.unique(crossline).size)
        if unique_inline > 1 and unique_crossline > 1:
            return (
                "三维地震体",
                f"{unique_inline} 个 Inline × {unique_crossline} 个 Crossline",
            )
        if unique_inline > 1 or unique_crossline > 1:
            return "二维地震测线", "仅一个方向形成有效网格"
    return "地震数据（待确认维度）", "Inline/Crossline 道头不足"


_RUNTIME_CONTRACT_REVIEW_VERSION = "well-seismic.runtime-contract-review.v1"
_RUNTIME_CONTRACT_CONFIRMATION_VERSION = (
    "well-seismic.runtime-contract-confirmation.v1"
)
_RUNTIME_CONTRACT_PROFILE_ID = "CN_GENERAL_RUN_FIRST_V1"


def _runtime_contract_review(
    pipeline: WellSeismicPipeline,
    request: InspectionRequest | None,
    *,
    asset_set_sha256: str,
) -> dict[str, Any]:
    """Describe only missing semantics from the assets that were actually parsed."""

    base: dict[str, Any] = {
        "contract_version": _RUNTIME_CONTRACT_REVIEW_VERSION,
        "required": False,
        "profile_id": _RUNTIME_CONTRACT_PROFILE_ID,
        "fields": [],
        "values": {},
        "time_depth_asset_count": 0,
    }
    if request is None or not any(
        str(asset.role).casefold() == "seismic" for asset in pipeline.assets
    ):
        return base

    stable_suffix = str(asset_set_sha256 or "0" * 12)[:12].upper()
    expected_local_horizontal_crs = f"LOCAL_SURVEY_XY_{stable_suffix}"
    horizontal_crs = str(request.horizontal_crs_id or "").strip()

    def horizontal_crs_allowed(value: str) -> bool:
        if value == expected_local_horizontal_crs:
            return True
        try:
            require_projected_metre_crs(value, field="运行参数水平CRS")
        except CoordinateReferenceError:
            return False
        return True

    horizontal_ready = bool(
        request.coordinate_reference_verified is True
        and horizontal_crs_allowed(horizontal_crs)
    )
    if not horizontal_crs_allowed(horizontal_crs):
        horizontal_crs = expected_local_horizontal_crs
    vertical_crs = str(request.vertical_crs_id or "").strip()
    vertical_ready = bool(
        _registration_crs_identifier_is_concrete(vertical_crs)
        and "MSL" in vertical_crs.upper()
    )
    if not vertical_ready:
        vertical_crs = f"LOCAL_MSL_RUNTIME_{stable_suffix}"

    detected_replacement_velocities_mps: set[float] = set()
    for asset, _reader in getattr(pipeline, "seismic", ()):
        metadata = getattr(pipeline, "time_reference_metadata", {}).get(
            str(asset.path)
        )
        candidate = getattr(metadata, "replacement_velocity_mps", None)
        try:
            numeric_candidate = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_candidate) and numeric_candidate > 0:
            detected_replacement_velocities_mps.add(numeric_candidate)
    detected_replacement_velocity_mps = (
        next(iter(detected_replacement_velocities_mps))
        if len(detected_replacement_velocities_mps) == 1
        else None
    )
    explicit_replacement_velocity_ready = bool(
        request.seismic_replacement_velocity_mps is not None
        and np.isfinite(float(request.seismic_replacement_velocity_mps))
        and float(request.seismic_replacement_velocity_mps) > 0
    )
    values: dict[str, Any] = {
        "horizontal_crs_id": horizontal_crs,
        "horizontal_unit": "m",
        "horizontal_axis_order": (
            request.horizontal_axis_order
            if request.horizontal_axis_order in {"XY", "YX"}
            else "XY"
        ),
        "coordinate_reference_verified": True,
        "vertical_crs_id": vertical_crs,
        "seismic_srd_elevation_m": (
            float(request.seismic_srd_elevation_m)
            if request.seismic_srd_elevation_m is not None
            and np.isfinite(float(request.seismic_srd_elevation_m))
            else 0.0
        ),
        "seismic_time_domain": "TWT",
        "seismic_correction_state": "corrected_to_srd",
        "seismic_replacement_velocity_mps": (
            float(request.seismic_replacement_velocity_mps)
            if explicit_replacement_velocity_ready
            else (
                detected_replacement_velocity_mps
                if detected_replacement_velocity_mps is not None
                else 2000.0
            )
        ),
        "well_coordinate_source_unit": (
            request.well_coordinate_source_unit
            if request.well_coordinate_source_unit in {"m", "ft"}
            else "m"
        ),
        "well_vertical_datum_source_unit": (
            request.well_vertical_datum_source_unit
            if request.well_vertical_datum_source_unit in {"m", "ft"}
            else "m"
        ),
    }
    fields: list[dict[str, Any]] = []

    def add_field(
        key: str,
        label: str,
        *,
        control: str,
        group: str,
        choices: list[tuple[Any, str]] | None = None,
        unit: str | None = None,
        helper: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "key": key,
            "label": label,
            "value": values[key],
            "control": control,
            "group": group,
        }
        if choices:
            item["choices"] = [
                {"value": value, "label": choice_label}
                for value, choice_label in choices
            ]
        if unit:
            item["unit"] = unit
        if helper:
            item["helper"] = helper
        fields.append(item)

    if not horizontal_ready:
        add_field(
            "horizontal_crs_id",
            "水平坐标参考",
            control="text",
            group="空间定位",
            helper="无明确投影时使用当前测区稳定本地坐标系",
        )
    if request.coordinate_reference_verified is not True:
        add_field(
            "coordinate_reference_verified",
            "水平坐标已核对",
            control="select",
            group="空间定位",
            choices=[(True, "已核对")],
        )
    if request.horizontal_unit != "m":
        add_field(
            "horizontal_unit",
            "水平单位",
            control="select",
            group="空间定位",
            choices=[("m", "米")],
        )
    if request.horizontal_axis_order not in {"XY", "YX"}:
        add_field(
            "horizontal_axis_order",
            "坐标轴序",
            control="select",
            group="空间定位",
            choices=[("XY", "X / Y"), ("YX", "Y / X")],
        )
    if not vertical_ready:
        add_field(
            "vertical_crs_id",
            "垂向参考",
            control="text",
            group="垂向与时间",
        )
    if request.seismic_srd_elevation_m is None or not np.isfinite(
        float(request.seismic_srd_elevation_m)
    ):
        add_field(
            "seismic_srd_elevation_m",
            "地震处理基准面 SRD",
            control="number",
            group="垂向与时间",
            unit="m MSL",
        )
    if request.seismic_time_domain != "TWT":
        add_field(
            "seismic_time_domain",
            "地震时间域",
            control="select",
            group="垂向与时间",
            choices=[("TWT", "双程时 TWT")],
        )
    if request.seismic_correction_state != "corrected_to_srd":
        add_field(
            "seismic_correction_state",
            "静校正状态",
            control="select",
            group="垂向与时间",
            choices=[("corrected_to_srd", "已校正至 SRD")],
        )
    if not explicit_replacement_velocity_ready and (
        detected_replacement_velocity_mps is None
    ):
        add_field(
            "seismic_replacement_velocity_mps",
            "替换速度",
            control="number",
            group="垂向与时间",
            unit="m/s",
        )

    has_well_assets = any(
        str(asset.role).casefold()
        in {"well_heads", "well_logs", "trajectories", "well_metadata", "time_depth"}
        for asset in pipeline.assets
    )
    if has_well_assets and request.well_coordinate_source_unit not in {"m", "ft"}:
        add_field(
            "well_coordinate_source_unit",
            "井位源坐标单位",
            control="select",
            group="井数据",
            choices=[("m", "米"), ("ft", "英尺")],
        )
    if has_well_assets and request.well_vertical_datum_source_unit not in {"m", "ft"}:
        add_field(
            "well_vertical_datum_source_unit",
            "井高程源单位",
            control="select",
            group="井数据",
            choices=[("m", "米"), ("ft", "英尺")],
        )

    tables = [
        table
        for entity in pipeline.registry.entities.values()
        for table in entity.time_depth
    ]
    time_depth_sources = {str(table.source).split("#", 1)[0] for table in tables}
    unresolved_detected_domain = False
    for item in pipeline.metadata_detection:
        if not isinstance(item, dict):
            continue
        roles = item.get("识别角色") or []
        if isinstance(roles, str):
            roles = [roles]
        if not any("时深关系" in str(role) for role in roles):
            continue
        source = str(item.get("文件") or "").strip()
        if source:
            time_depth_sources.add(source)
        unresolved_detected_domain = (
            unresolved_detected_domain
            or str(item.get("时深深度域") or "").strip() in {"", "未明确", "unknown"}
        )
    time_depth_asset_count = len(time_depth_sources)

    def resolved_table_value(field: str, allowed: set[str]) -> str | None:
        observed = {
            str(getattr(table, field, "") or "").strip()
            for table in tables
            if str(getattr(table, field, "") or "").strip() in allowed
        }
        return next(iter(observed)) if len(observed) == 1 else None

    if time_depth_asset_count:
        td_specs = (
            (
                "time_depth_default_depth_domain",
                "时深深度域",
                "tvd",
                {"md", "tvd", "tvdss"},
                [("md", "MD"), ("tvd", "TVD"), ("tvdss", "TVDSS")],
                unresolved_detected_domain
                or any(str(table.depth_domain) not in {"md", "tvd", "tvdss"} for table in tables),
            ),
            (
                "time_depth_default_depth_unit",
                "时深深度单位",
                "m",
                {"m", "ft"},
                [("m", "米"), ("ft", "英尺")],
                any(str(table.depth_unit) not in {"m", "ft"} for table in tables)
                or not tables,
            ),
            (
                "time_depth_default_time_unit",
                "时深时间单位",
                "ms",
                {"ms", "s", "us"},
                [("ms", "毫秒"), ("s", "秒"), ("us", "微秒")],
                any(str(table.time_unit) not in {"ms", "s", "us"} for table in tables)
                or not tables,
            ),
            (
                "time_depth_default_depth_datum",
                "时深深度起算面",
                "KB",
                {"KB", "GL", "DF", "RT", "MSL"},
                [(value, value) for value in ("KB", "GL", "DF", "RT", "MSL")],
                any(
                    table.depth_domain in {"md", "tvd"} and not table.depth_datum
                    for table in tables
                )
                or not tables,
            ),
            (
                "time_depth_default_time_reference",
                "时深时间参考",
                "SRD",
                {"SRD", "KB", "GL", "DF", "RT"},
                [(value, value) for value in ("SRD", "KB", "GL", "DF", "RT")],
                any(str(table.time_reference) not in {"SRD", "KB", "GL", "DF", "RT"} for table in tables)
                or not tables,
            ),
            (
                "time_depth_default_time_domain",
                "时深时间域",
                "TWT",
                {"TWT", "OWT"},
                [("TWT", "双程时 TWT"), ("OWT", "单程时 OWT")],
                any(str(table.time_domain) not in {"TWT", "OWT"} for table in tables)
                or not tables,
            ),
            (
                "time_depth_default_correction_state",
                "时深校正状态",
                "corrected_to_srd",
                {"corrected_to_srd", "uncorrected"},
                [
                    ("corrected_to_srd", "已校正至 SRD"),
                    ("uncorrected", "未校正"),
                ],
                any(str(table.correction_state) not in {"corrected_to_srd", "uncorrected"} for table in tables)
                or not tables,
            ),
        )
        for key, label, fallback, allowed, choices, unresolved in td_specs:
            if not unresolved:
                continue
            observed = resolved_table_value(
                key.removeprefix("time_depth_default_"), allowed
            )
            values[key] = observed or fallback
            add_field(
                key,
                label,
                control="select",
                group="时深数据",
                choices=choices,
            )

        tvdss_missing_convention = any(
            table.depth_domain == "tvdss" and not table.depth_convention
            for table in tables
        )
        if tvdss_missing_convention:
            key = "time_depth_default_depth_convention"
            values[key] = "depth_below_msl_positive_down"
            add_field(
                key,
                "TVDSS 正方向",
                control="select",
                group="时深数据",
                choices=[
                    ("depth_below_msl_positive_down", "海平面以下向下为正"),
                    ("elevation_positive_up", "绝对高程向上为正"),
                ],
            )

    # Validate the same object the confirmation endpoint accepts.  This keeps
    # the review JSON and executable API contract from drifting apart.
    values = RuntimeContractValues.model_validate(values).model_dump(
        mode="json", exclude_none=True
    )
    base.update(
        {
            "required": bool(fields),
            "fields": fields,
            "values": values,
            "time_depth_asset_count": time_depth_asset_count,
        }
    )
    return base


def _inspection_result(
    pipeline: WellSeismicPipeline,
    hash_progress: Any = None,
    summary_progress: Any = None,
    *,
    sealed_assets: list[dict[str, Any]] | None = None,
    request: InspectionRequest | None = None,
) -> dict[str, Any]:
    report = pipeline.quality_report(
        progress=hash_progress,
        sealed_assets=sealed_assets,
    )
    if summary_progress is not None:
        summary_progress()
    role_counts = Counter(asset.role for asset in pipeline.assets)
    seismic_items: list[dict[str, Any]] = []
    dimension_counts: Counter[str] = Counter()
    result_rows: list[dict[str, Any]] = []

    for asset, reader in pipeline.seismic:
        geometry = reader.geometry
        if geometry is None:
            continue
        dimension, evidence = _seismic_dimension(geometry)
        inline_count = (
            int(np.unique(geometry.inline).size) if geometry.inline is not None else 0
        )
        crossline_count = (
            int(np.unique(geometry.crossline).size)
            if geometry.crossline is not None
            else 0
        )
        grid_cells = inline_count * crossline_count
        grid_coverage = (
            min(1.0, float(geometry.trace_count / grid_cells)) if grid_cells else 0.0
        )
        dimension_counts[dimension] += 1
        seismic_items.append(
            {
                "name": asset.path.name,
                "path": str(asset.path),
                "dimension": dimension,
                "evidence": evidence,
                "trace_count": geometry.trace_count,
                "samples_per_trace": geometry.samples_per_trace,
                "sample_interval_ms": geometry.sample_interval,
                "shape_zyx": [
                    geometry.samples_per_trace,
                    inline_count,
                    crossline_count,
                ],
                "inline_count": inline_count,
                "crossline_count": crossline_count,
                "grid_coverage": grid_coverage,
                "confidence": geometry.confidence,
                "issues": geometry.issues,
                "model_compatibility": _input_adapters.compatibilities(geometry),
            }
        )

    # Keep every registered seismic asset in the model-neutral inventory, even
    # when a reader cannot reconstruct its geometry. Visualization and model
    # adapters can then explain why an asset is unavailable instead of silently
    # dropping it from the task.
    inspected_paths = {item["path"] for item in seismic_items}
    seismic_errors = {
        item.get("path", ""): item.get("error", "读取失败")
        for item in report["errors"]
        if item.get("role") == "seismic"
    }
    for asset in pipeline.assets:
        asset_path = str(asset.path)
        if asset.role != "seismic" or asset_path in inspected_paths:
            continue
        error = seismic_errors.get(asset_path, "未能重建 Inline/Crossline 几何")
        dimension_counts["地震数据（待确认维度）"] += 1
        seismic_items.append(
            {
                "name": asset.path.name,
                "path": asset_path,
                "dimension": "地震数据（待确认维度）",
                "evidence": error,
                "trace_count": 0,
                "samples_per_trace": 0,
                "sample_interval_ms": 0.0,
                "shape_zyx": [0, 0, 0],
                "inline_count": 0,
                "crossline_count": 0,
                "grid_coverage": 0.0,
                "confidence": 0.0,
                "issues": [error],
                "model_compatibility": _input_adapters.unavailable_compatibilities(
                    f"源文件已登记，但尚未形成模型所需的地震网格：{error}"
                ),
            }
        )

    for dimension in ("二维地震测线", "三维地震体", "地震数据（待确认维度）"):
        count = dimension_counts.get(dimension, 0)
        if count:
            result_rows.append(
                {
                    "type": dimension,
                    "count": count,
                    "evidence": "SEG-Y 二进制头、道头和网格规律",
                    "status": "待确认" if "待确认" in dimension else "可读取",
                }
            )

    log_count = role_counts.get("well_logs", 0)
    if log_count:
        result_rows.append(
            {
                "type": "LAS 测井",
                "count": log_count,
                "evidence": "LAS 版本段、井信息段和曲线定义段",
                "status": "可读取",
            }
        )

    metadata_count = role_counts.get("well_metadata", 0)
    uncertain_metadata = sum(
        1 for item in pipeline.metadata_detection if item.get("状态") == "待确认"
    )
    if metadata_count:
        result_rows.append(
            {
                "type": "井基础信息与井轨迹",
                "count": metadata_count,
                "evidence": "字段映射、数据结构和井名关联",
                "status": (
                    f"{uncertain_metadata} 个待确认" if uncertain_metadata else "已识别"
                ),
            }
        )

    auxiliary_count = role_counts.get("auxiliary", 0)
    survey_count = role_counts.get("survey_geometry", 0)
    interpretation_count = role_counts.get("interpretation", 0)
    if survey_count:
        result_rows.append(
            {
                "type": "测区网格与坐标",
                "count": survey_count,
                "evidence": "独立登记测区网格、线道号与坐标参考文件",
                "status": "待人工核验坐标参考",
            }
        )
    if interpretation_count:
        result_rows.append(
            {
                "type": "解释成果与标签",
                "count": interpretation_count,
                "evidence": "独立登记层位、断层、岩性及其他解释成果",
                "status": "已登记",
            }
        )
    if auxiliary_count:
        result_rows.append(
            {
                "type": "其他辅助数据",
                "count": auxiliary_count,
                "evidence": "登记来源，不参与基础匹配",
                "status": "已登记",
            }
        )

    preparation = build_preparation_report(pipeline)
    visualization_preview = build_visualization_preview(pipeline)
    visualization_preview["seismicInventory"] = seismic_items
    registered_seismic_count = sum(
        1 for asset in pipeline.assets if asset.role == "seismic"
    )
    snapshot_assets_sha256 = snapshot_assets_fingerprint(report["assets"])
    preparation["runtime_contract_review"] = _runtime_contract_review(
        pipeline,
        request,
        asset_set_sha256=snapshot_assets_sha256,
    )
    data_snapshot = {
        "contract_version": "well-seismic.data.v2",
        "snapshot_id": None,
        "asset_set_sha256": snapshot_assets_sha256,
        "asset_count": len(report["assets"]),
        "identity_policy": "full_file_sha256_plus_segy_geometry_fingerprint",
        "semantics": "model_neutral",
        "source_assets": {
            "seismic": registered_seismic_count,
            "well_logs": log_count,
            "well_metadata": metadata_count,
            "survey_geometry": survey_count,
            "interpretation": interpretation_count,
            "auxiliary": auxiliary_count,
        },
        "canonical_data": {
            "seismic_geometry": {
                "axes": ["Z", "INLINE", "CROSSLINE"],
                "registered": registered_seismic_count,
                "readable": sum(1 for item in seismic_items if item["trace_count"] > 0),
                "renderable_3d": len(visualization_preview.get("volumes", [])),
                "renderable_2d": len(visualization_preview.get("lines2d", [])),
            },
            "well_entities": {"count": report["summary"]["wells"]},
            "well_logs": {"count": log_count},
            "vertical_datum": report.get("vertical_datum", {}),
            "survey_geometry": {
                "count": survey_count,
                "coordinate_reference_verified": False,
            },
            "interpretation": {"count": interpretation_count},
        },
        "derived_views": {
            "visualization_preview": {
                "status": (
                    "available"
                    if visualization_preview.get("volumes")
                    or visualization_preview.get("lines2d")
                    or visualization_preview.get("wellLogs")
                    else "unavailable"
                ),
                "model_specific": False,
            },
            "well_seismic_samples": {
                "status": (
                    "available"
                    if preparation["gates"]["can_build_samples"]
                    else "blocked"
                ),
                "optional": True,
                "model_specific": False,
            },
        },
        "downstream_policy": "下游模型必须通过各自输入适配器从该快照派生输入，不得反向修改源数据或通用预处理结果。",
    }

    return {
        "summary": {
            "assets": report["summary"]["assets"],
            "duplicates_skipped": report["summary"]["duplicates_skipped"],
            "wells": report["summary"]["wells"],
            "seismic_files": report["summary"]["seismic_files"],
            "registered_seismic_files": registered_seismic_count,
            "log_files": log_count,
            "metadata_files": metadata_count,
            "survey_files": survey_count,
            "interpretation_files": interpretation_count,
            "auxiliary_files": auxiliary_count,
            "uncertain": uncertain_metadata
            + dimension_counts.get("地震数据（待确认维度）", 0),
            "errors": report["summary"]["errors"],
        },
        "rows": result_rows,
        "seismic": seismic_items,
        "wells": report["wells"],
        "well_entities": report["wells"],
        "assets": report["assets"],
        "metadata_detection": pipeline.metadata_detection,
        "llm_parse_repairs": report.get("llm_parse_repairs", []),
        "deterministic_unit_inheritances": report.get(
            "deterministic_unit_inheritances", []
        ),
        "vertical_datum": report.get("vertical_datum", {}),
        "errors": report["errors"],
        "duplicates": report["duplicates"],
        "inventory": pipeline.automatic_inventory,
        "preparation": preparation,
        "data_snapshot": data_snapshot,
        "visualization_preview": visualization_preview,
    }


def _target_required_modalities(request: InspectionRequest) -> tuple[str, ...]:
    # A task/model left in a restored or legacy form is not an execution
    # contract.  Only an explicit user selection may activate task gates.
    if not request.target_scope_explicit or not request.target_task_id:
        return ()
    try:
        task = _interpretation_registry.get(request.target_task_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    if task.lifecycle == "archived":
        raise ValueError(
            f"目标任务 {request.target_task_id} 已归档，仅支持读取历史结果"
        )
    task_models = [
        item
        for item in _model_registry.list_specs()
        if str(item.metadata.get("prediction_task") or "")
        == request.target_task_id
    ]
    if request.target_model_id:
        model = next(
            (
                item
                for item in _model_registry.list_specs()
                if item.id == request.target_model_id
            ),
            None,
        )
        if model is None:
            raise ValueError(f"未知目标模型：{request.target_model_id}")
        model_task = str(model.metadata.get("prediction_task") or "")
        if model_task != request.target_task_id:
            raise ValueError(
                f"目标模型 {request.target_model_id} 不属于任务 {request.target_task_id}"
            )
        metadata = dict(model.metadata or {})
        if (
            model.runtime_status != "runnable"
            or metadata.get("archived")
            or metadata.get("historical_result_compatibility")
        ):
            raise ValueError(
                f"目标模型 {request.target_model_id} 当前不可运行，不能用于数据准备"
            )
    elif not any(
        model.runtime_status == "runnable"
        and not bool(model.metadata.get("archived"))
        and not bool(model.metadata.get("historical_result_compatibility"))
        for model in task_models
    ):
        raise ValueError(
            f"目标任务 {request.target_task_id} 当前没有可运行模型，不能用于数据准备"
        )
    return tuple(task.required_modalities)


def _target_model_contract(request: InspectionRequest) -> dict[str, Any]:
    """Return one selected runner contract or the strict task-wide union.

    Task-level modality labels are intentionally retained for backwards
    compatibility.  Model-level fields decide whether a preparation issue is
    an actual prerequisite for the selected runner; this avoids manufacturing
    Registration/time-depth requirements for inference contracts that
    explicitly forbid those inputs.  When the task is explicit but the model
    is empty, the caller asked for a reusable snapshot, so the strongest policy
    across that task's registered model flows is returned.
    """

    if not request.target_scope_explicit or not request.target_task_id:
        return {}
    _target_required_modalities(request)
    models = list(_model_registry.list_specs())
    model_by_id = {item.id: item for item in models}
    flows = build_model_data_flow_specs(
        models,
        _input_adapters.capabilities(),
        _prediction_runners.model_ids(),
    )
    if not request.target_model_id:
        task_flows = [
            item for item in flows if item.get("task_id") == request.target_task_id
        ]
        if not task_flows:
            return {}

        registration_rank = {"none": 0, "optional_control": 1, "required": 2}
        prepared_view_rank = {"none": 0, "optional": 1, "preferred": 2, "required": 3}
        registration_policy = max(
            (str(item.get("registration_policy") or "none") for item in task_flows),
            key=lambda value: registration_rank.get(value, -1),
        )
        prepared_view_policy = max(
            (
                str(item.get("prepared_view_policy") or "optional")
                for item in task_flows
            ),
            key=lambda value: prepared_view_rank.get(value, -1),
        )
        task_models = [
            model_by_id[str(item.get("model_id"))]
            for item in task_flows
            if str(item.get("model_id")) in model_by_id
        ]
        time_depth_flags = [
            (model.metadata or {}).get("time_depth_supervision_is_model_input")
            for model in task_models
            if "time_depth_supervision_is_model_input" in (model.metadata or {})
        ]
        if any(value is True for value in time_depth_flags):
            time_depth_policy: bool | None = True
        elif len(time_depth_flags) == len(task_models) and time_depth_flags:
            time_depth_policy = False
        else:
            time_depth_policy = None

        def union_values(field: str) -> list[str]:
            return list(
                dict.fromkeys(
                    str(value)
                    for item in task_flows
                    for value in (item.get(field) or ())
                )
            )

        return {
            "model_id": None,
            "model_ids": [str(item.get("model_id")) for item in task_flows],
            "task_id": request.target_task_id,
            "scope_mode": "strict_task_model_union",
            "required_modalities": union_values("required_modalities"),
            "optional_modalities": union_values("optional_modalities"),
            "source_modes": union_values("source_modes"),
            "target_source_modes": union_values("target_source_modes"),
            "accepted_domains": union_values("accepted_domains"),
            "registration_policy": registration_policy,
            "prepared_view_policy": prepared_view_policy,
            "requires_registration": registration_policy == "required",
            "time_depth_supervision_is_model_input": time_depth_policy,
            "forbidden_inference_parameters": list(
                dict.fromkeys(
                    str(value)
                    for model in task_models
                    for value in (
                        (model.metadata or {}).get("forbidden_inference_parameters")
                        or ()
                    )
                )
            ),
            "degradation_policy": "strict_union_of_task_model_contracts",
            "adapter_registered": all(
                bool(item.get("adapter_registered")) for item in task_flows
            ),
            "runner_registered": all(
                bool(item.get("runner_registered")) for item in task_flows
            ),
            "runnable": all(bool(item.get("runnable")) for item in task_flows),
        }

    model = model_by_id[request.target_model_id]
    flow = next(
        (item for item in flows if item["model_id"] == request.target_model_id),
        {},
    )
    metadata = dict(model.metadata or {})
    requires_registration = bool(metadata.get("requires_registration"))
    fallback_registration_policy = str(
        metadata.get("registration_policy")
        or ("required" if requires_registration else "none")
    )
    if fallback_registration_policy not in {
        "none",
        "optional_control",
        "required",
    }:
        fallback_registration_policy = "required" if requires_registration else "none"
    return {
        "model_id": request.target_model_id,
        "task_id": request.target_task_id,
        "registration_policy": fallback_registration_policy,
        "prepared_view_policy": str(metadata.get("prepared_view_policy") or "optional"),
        **flow,
        "requires_registration": requires_registration,
        "time_depth_supervision_is_model_input": metadata.get(
            "time_depth_supervision_is_model_input"
        ),
        "forbidden_inference_parameters": list(
            metadata.get("forbidden_inference_parameters") or ()
        ),
    }


def inspect_paths(
    request: InspectionRequest,
    progress: Any = None,
    *,
    _pipeline_sink: Any = None,
) -> dict[str, Any]:
    if not any(
        (
            request.seismic_paths,
            request.survey_paths,
            request.log_paths,
            request.well_paths,
            request.time_depth_paths,
            request.interpretation_paths,
            request.auxiliary_paths,
        )
    ):
        raise ValueError("至少需要登记一个有效数据路径")
    original_well_coordinate_source_unit = (
        request.well_coordinate_source_unit
        if request.well_coordinate_source_unit is not None
        else request.horizontal_unit
    )
    coordinate_detection = _autofill_horizontal_coordinate_contract(request)
    if progress:
        progress(
            10,
            "正在校验绝对路径",
            {"phase": "validating", "can_estimate": False},
        )

    target_modalities = _target_required_modalities(request)
    target_model_contract = _target_model_contract(request)

    def _build_effective_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
        return build_explicit_paths_manifest(
            seismic_directory=request.seismic_paths,
            log_directory=request.log_paths,
            metadata_directory=request.well_paths,
            auxiliary_directory=request.auxiliary_paths,
            survey_directory=request.survey_paths,
            interpretation_directory=request.interpretation_paths,
            time_depth_directory=request.time_depth_paths,
            recursive=request.recursive,
            require_seismic=False,
            require_logs=False,
            seismic_srd_elevation_m=request.seismic_srd_elevation_m,
            vertical_crs_id=request.vertical_crs_id,
            horizontal_crs_id=request.horizontal_crs_id,
            well_source_crs_id=request.well_source_crs_id,
            seismic_source_crs_id=request.seismic_source_crs_id,
            horizontal_unit=request.horizontal_unit,
            horizontal_axis_order=request.horizontal_axis_order,
            coordinate_reference_verified=request.coordinate_reference_verified,
            seismic_replacement_velocity_mps=request.seismic_replacement_velocity_mps,
            seismic_time_domain=request.seismic_time_domain,
            seismic_correction_state=request.seismic_correction_state,
            segy_geometry_profile=request.segy_geometry_profile,
            segy_inline_byte=request.segy_inline_byte,
            segy_crossline_byte=request.segy_crossline_byte,
            segy_x_byte=request.segy_x_byte,
            segy_y_byte=request.segy_y_byte,
            segy_coordinate_scalar_byte=request.segy_coordinate_scalar_byte,
            well_coordinate_source_unit=request.well_coordinate_source_unit,
            well_vertical_datum_source_unit=request.well_vertical_datum_source_unit,
            las_twt_source_unit=request.las_twt_source_unit,
            time_depth_default_depth_domain=request.time_depth_default_depth_domain,
            time_depth_default_depth_unit=request.time_depth_default_depth_unit,
            time_depth_default_time_unit=request.time_depth_default_time_unit,
            time_depth_default_depth_datum=request.time_depth_default_depth_datum,
            time_depth_default_depth_convention=request.time_depth_default_depth_convention,
            time_depth_default_time_reference=request.time_depth_default_time_reference,
            time_depth_default_time_domain=request.time_depth_default_time_domain,
            time_depth_default_correction_state=request.time_depth_default_correction_state,
            target_task_id=request.target_task_id,
            target_model_id=request.target_model_id,
            target_scope_explicit=request.target_scope_explicit,
            target_required_modalities=target_modalities,
            target_model_contract=target_model_contract,
        )

    manifest, inventory = _build_effective_manifest()
    if progress:
        progress(
            30,
            "正在建立数据资产目录和检查重复文件",
            {"phase": "cataloging", "can_estimate": False},
        )

    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    _apply_horizontal_coordinate_contract(pipeline, request)
    pipeline.automatic_inventory = inventory
    role_labels = {
        "well_heads": "井位",
        "well_logs": "测井",
        "trajectories": "井轨迹",
        "time_depth": "时深关系",
        "well_metadata": "井元数据",
        "seismic": "地震",
    }
    readable_asset_total = sum(
        1 for asset in pipeline.assets if str(asset.role) in role_labels
    )
    asset_size_by_path = {
        str(Path(asset.path).resolve()): int(Path(asset.path).stat().st_size)
        for asset in pipeline.assets
    }
    reading_started_at = _now()
    if progress:
        progress(
            45,
            f"已登记 {len(pipeline.assets)} 个数据资产，正在读取文件头和井数据",
            {
                "phase": "reading",
                "work_done": 0,
                "work_total": readable_asset_total,
                "unit": "assets",
                "started_at": reading_started_at,
                "can_estimate": False,
            },
        )

    current_asset_key: str | None = None
    current_asset_started_at: str | None = None

    def update_ingest(
        done: int,
        total: int,
        asset: Any | None,
        subwork_done: int = 0,
        subwork_total: int = 0,
    ) -> None:
        nonlocal current_asset_key, current_asset_started_at
        if not progress:
            return
        bounded_total = max(0, int(total))
        bounded_done = min(max(0, int(done)), bounded_total)
        bounded_subwork_total = max(0, int(subwork_total))
        bounded_subwork_done = min(
            max(0, int(subwork_done)),
            bounded_subwork_total,
        )
        fractional_done = float(bounded_done)
        if asset is not None and bounded_subwork_total:
            fractional_done += bounded_subwork_done / bounded_subwork_total
        percent = 45 + (
            round(40 * fractional_done / bounded_total) if bounded_total else 0
        )
        current_item = None
        current_item_size_bytes = None
        if asset is not None:
            asset_path = Path(asset.path)
            current_item = str(Path(asset_path.parent.name) / asset_path.name)
            asset_key = str(Path(asset.path).resolve())
            current_item_size_bytes = asset_size_by_path.get(asset_key)
            if asset_key != current_asset_key:
                current_asset_key = asset_key
                current_asset_started_at = _now()
            current_index = min(bounded_done + 1, bounded_total)
            role_label = role_labels.get(str(asset.role), str(asset.role))
            if bounded_subwork_total:
                message = (
                    f"正在扫描 {role_label}道头 {bounded_subwork_done:,}/"
                    f"{bounded_subwork_total:,}：{current_item}"
                )
            else:
                message = (
                    f"正在读取 {role_label}数据 {current_index}/{bounded_total}："
                    f"{current_item}"
                )
        else:
            message = f"已处理 {bounded_done}/{bounded_total} 个可解析数据资产"
        progress(
            percent,
            message,
            {
                "phase": "reading",
                "work_done": bounded_done,
                "work_total": bounded_total,
                "unit": "assets",
                "current_item": current_item,
                "current_item_size_bytes": current_item_size_bytes,
                "started_at": reading_started_at,
                "current_started_at": (
                    current_asset_started_at if asset is not None else None
                ),
                "subwork_done": bounded_subwork_done,
                "subwork_total": bounded_subwork_total,
                "subunit": "traces" if bounded_subwork_total else None,
                "can_estimate": (
                    bounded_subwork_done > 0
                    and bounded_subwork_done < bounded_subwork_total
                ),
            },
        )

    pipeline.ingest(progress=update_ingest if progress else None)
    coordinate_verification = verify_pipeline_coordinate_contract(
        pipeline,
        target_crs=request.horizontal_crs_id,
    )
    if coordinate_verification.get("verified") is True:
        request.horizontal_crs_id = str(coordinate_verification["target_crs"])
        request.coordinate_reference_verified = True
        request.horizontal_unit = "m"
        request.horizontal_axis_order = "XY"
        raw_unit_derived = (
            coordinate_verification.get("derived_well_coordinate_source_unit") == "m"
        )
        if raw_unit_derived:
            # The first pass retained raw X/Y because their unit was absent.
            # Seal the cross-asset unit inference into asset options, then
            # reparse so this snapshot and every later replay use identical
            # canonical metre coordinates and trajectory units.
            first_pass_verification = dict(coordinate_verification)
            inferred_local_crs = request.horizontal_crs_id
            first_pass_seismic_geometry = {
                str(asset.path.resolve()): reader.geometry
                for asset, reader in pipeline.seismic
                if reader.geometry is not None
            }
            request.well_coordinate_source_unit = "m"
            manifest, inventory = _build_effective_manifest()
            pipeline = WellSeismicPipeline(
                manifest,
                CONFIG_DIR,
                use_llm_fallback=request.use_llm_fallback,
            )
            _apply_horizontal_coordinate_contract(pipeline, request)
            pipeline.automatic_inventory = inventory
            pipeline.ingest(
                seismic_geometry_by_path=first_pass_seismic_geometry,
            )
            replay_verification = verify_pipeline_coordinate_contract(
                pipeline,
                target_crs=inferred_local_crs,
            )
            if not native_grid_replay_is_consistent(
                first_pass_verification,
                replay_verification,
            ):
                raise ValueError("井位原始XY米制推导在规范化重放后不一致，已拒绝封存")
            coordinate_verification = {
                **replay_verification,
                "derived_well_coordinate_source_unit": "m",
                "unit_derivation_receipt": first_pass_verification.get(
                    "unit_derivation_receipt"
                ),
                "unit_derivation_evidence_sha256": first_pass_verification.get(
                    "evidence_sha256"
                ),
                "unit_derivation_provenance": first_pass_verification.get(
                    "provenance"
                ),
                "canonical_replay_verified": True,
            }
        elif request.horizontal_crs_id.startswith("LOCAL_SURVEY_XY_"):
            # Preserve the source-unit option used by the first parse so a
            # later derived task recreates identical sealed asset options.
            request.well_coordinate_source_unit = original_well_coordinate_source_unit
    elif coordinate_verification.get("reason") not in {
        "cross_asset_qc_requires_wells_and_seismic",
    }:
        # A manual checkbox cannot override a detected CRS mismatch.
        request.coordinate_reference_verified = False
    _apply_horizontal_coordinate_contract(pipeline, request)
    hash_total = sum(asset.path.stat().st_size for asset in pipeline.assets)
    hashing_started_at = _now()
    if progress:
        progress(
            90,
            "正在计算已登记资产的完整文件身份哈希",
            {
                "phase": "hashing",
                "work_done": 0,
                "work_total": hash_total,
                "unit": "bytes",
                "started_at": hashing_started_at,
                "can_estimate": False,
            },
        )

    def update_hash(done: int, total: int, path: Path | None) -> None:
        if not progress:
            return
        bounded_total = max(0, int(total))
        bounded_done = min(max(0, int(done)), bounded_total)
        percent = 90 + (round(8 * bounded_done / bounded_total) if bounded_total else 0)
        current_item = (
            str(Path(path.parent.name) / path.name) if path is not None else None
        )
        current_item_size_bytes = (
            asset_size_by_path.get(str(path.resolve())) if path is not None else None
        )
        progress(
            percent,
            (
                f"正在校验文件身份 {bounded_done / 1024 ** 3:.2f}/"
                f"{bounded_total / 1024 ** 3:.2f} GiB"
                + (f"：{current_item}" if current_item else "")
            ),
            {
                "phase": "hashing",
                "work_done": bounded_done,
                "work_total": bounded_total,
                "unit": "bytes",
                "current_item": current_item,
                "current_item_size_bytes": current_item_size_bytes,
                "started_at": hashing_started_at,
                "current_started_at": hashing_started_at,
                "subwork_done": bounded_done,
                "subwork_total": bounded_total,
                "subunit": "bytes",
                "can_estimate": bounded_done > 0 and bounded_done < bounded_total,
            },
        )

    def update_summary() -> None:
        if not progress:
            return
        summary_message = (
            "文件读取与身份校验完成，正在生成质量报告、预览与 Kimi 受控建议"
            if pipeline.decision_resolver.enabled
            else "文件读取与身份校验完成，正在生成质量报告、预览并封存快照"
        )
        progress(
            98,
            summary_message,
            {
                "phase": "summarizing",
                "can_estimate": False,
            },
        )

    result = _inspection_result(
        pipeline,
        hash_progress=update_hash if progress else None,
        summary_progress=update_summary if progress else None,
        request=request,
    )
    result["coordinate_reference"] = {
        "detection": coordinate_detection,
        "verification": coordinate_verification,
        "effective_request": {
            "horizontal_crs_id": request.horizontal_crs_id,
            "well_source_crs_id": request.well_source_crs_id,
            "seismic_source_crs_id": request.seismic_source_crs_id,
            "horizontal_unit": request.horizontal_unit,
            "horizontal_axis_order": request.horizontal_axis_order,
            "coordinate_reference_verified": request.coordinate_reference_verified,
        },
    }
    if _pipeline_sink is not None:
        _pipeline_sink(pipeline)
    return result


def _request_path_contract(
    request: InspectionRequest | dict[str, Any],
) -> dict[str, list[str]]:
    raw = request if isinstance(request, dict) else request.model_dump(mode="json")
    keys = (
        "seismic_paths",
        "survey_paths",
        "log_paths",
        "well_paths",
        "time_depth_paths",
        "interpretation_paths",
        "auxiliary_paths",
    )
    return {
        key: sorted(
            str(Path(value).expanduser().resolve()).casefold()
            for value in (raw.get(key) or [])
            if str(value).strip()
        )
        for key in keys
    }


_PREPARATION_ESTIMATE_REQUEST_FIELDS = (
    "recursive",
    "lightweight",
    "use_llm_fallback",
    "horizontal_crs_id",
    "well_source_crs_id",
    "seismic_source_crs_id",
    "horizontal_unit",
    "horizontal_axis_order",
    "coordinate_reference_verified",
    "seismic_srd_elevation_m",
    "vertical_crs_id",
    "seismic_replacement_velocity_mps",
    "seismic_time_domain",
    "seismic_correction_state",
    "segy_geometry_profile",
    "segy_inline_byte",
    "segy_crossline_byte",
    "segy_x_byte",
    "segy_y_byte",
    "segy_coordinate_scalar_byte",
    "well_coordinate_source_unit",
    "well_vertical_datum_source_unit",
    "las_twt_source_unit",
    "time_depth_default_depth_domain",
    "time_depth_default_depth_unit",
    "time_depth_default_time_unit",
    "time_depth_default_depth_datum",
    "time_depth_default_depth_convention",
    "time_depth_default_time_reference",
    "time_depth_default_time_domain",
    "time_depth_default_correction_state",
    "target_task_id",
    "target_model_id",
    "target_scope_explicit",
)


def _preparation_request_contract_sha256(
    request: InspectionRequest | dict[str, Any],
) -> str:
    raw = request if isinstance(request, dict) else request.model_dump(mode="json")
    return canonical_sha256(
        {
            "paths": _request_path_contract(raw),
            "settings": {
                key: raw.get(key) for key in _PREPARATION_ESTIMATE_REQUEST_FIELDS
            },
        }
    )


def _preparation_input_fingerprint(
    request: InspectionRequest | dict[str, Any],
) -> str:
    """Capture cheap top-level input identity without rescanning large directories."""

    raw = request if isinstance(request, dict) else request.model_dump(mode="json")
    path_keys = (
        "seismic_paths",
        "survey_paths",
        "log_paths",
        "well_paths",
        "time_depth_paths",
        "interpretation_paths",
        "auxiliary_paths",
    )
    entries: list[dict[str, Any]] = []
    for role in path_keys:
        paths = sorted(
            (Path(value).expanduser().resolve() for value in (raw.get(role) or [])),
            key=lambda value: str(value).casefold(),
        )
        for path in paths:
            normalized_path = str(path)
            try:
                stat = path.stat()
            except OSError:
                entries.append(
                    {"role": role, "path": normalized_path, "state": "missing"}
                )
                continue
            entries.append(
                {
                    "role": role,
                    "path": normalized_path,
                    "kind": "directory" if path.is_dir() else "file",
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return canonical_sha256(entries)


def _historical_preparation_estimate(
    request: InspectionRequest,
    *,
    request_contract_sha256: str | None = None,
    input_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Estimate wall time from immutable durations under a matching configuration."""

    requested_contract = (
        request_contract_sha256 or _preparation_request_contract_sha256(request)
    )
    requested_input = input_fingerprint or _preparation_input_fingerprint(request)
    durations: list[float] = []
    try:
        historical_tasks = _state_store.list_tasks(
            project_id=DEFAULT_PROJECT_ID,
            task_type="data_preparation",
            status="completed",
            limit=50,
        )
    except (StateStoreError, TypeError, ValueError) as exc:
        LOGGER.warning("无法读取数据准备历史耗时，继续执行当前任务：%s", exc)
        return None
    for task in historical_tasks:
        if task.get("preparation_contract_sha256") != requested_contract:
            continue
        if task.get("preparation_input_fingerprint") != requested_input:
            continue
        try:
            duration = float(task["preparation_duration_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= duration <= 24 * 60 * 60:
            durations.append(duration)
    if not durations:
        return None
    estimate = float(np.median(np.asarray(durations, dtype=float)))
    return {
        "duration_seconds": round(estimate, 1),
        "samples": len(durations),
        "confidence": "medium" if len(durations) >= 3 else "low",
        "basis": "matching_configuration_history",
    }


def _apply_horizontal_coordinate_contract(
    pipeline: WellSeismicPipeline,
    request: InspectionRequest,
) -> None:
    coordinate_reference = pipeline.config.setdefault("matching", {}).setdefault(
        "coordinate_reference", {}
    )
    coordinate_reference.update(
        {
            "verified": bool(request.coordinate_reference_verified),
            "crs": request.horizontal_crs_id,
            "well_source_crs": request.well_source_crs_id,
            "seismic_source_crs": request.seismic_source_crs_id,
            "horizontal_unit": request.horizontal_unit,
            "axis_order": request.horizontal_axis_order,
        }
    )


def _autofill_horizontal_coordinate_contract(
    request: InspectionRequest,
) -> dict[str, Any]:
    """Fill one canonical CRS contract from explicit file evidence, not guesses."""

    detection = detect_crs_evidence(
        well_paths=request.well_paths,
        seismic_paths=request.seismic_paths,
        recursive=request.recursive,
    )
    target, selection_source = choose_target_crs(
        detection,
        explicit_target=request.horizontal_crs_id,
        explicit_seismic_source=request.seismic_source_crs_id,
    )
    if target:
        request.horizontal_crs_id = target
        request.horizontal_unit = "m"
        if request.horizontal_axis_order == "unknown":
            request.horizontal_axis_order = "XY"
    elif selection_source == ("explicit_target_requires_projected_zone_or_native_grid"):
        # Keep the bare geodetic-family declaration in ``detection`` as
        # evidence, but do not pass it to readers as a pyproj target.
        detection["declared_non_projected_target"] = request.horizontal_crs_id
        request.horizontal_crs_id = None
    detection["selected_target"] = target
    detection["selection_source"] = selection_source
    return detection


def _assert_pipeline_uses_sealed_assets(
    pipeline: WellSeismicPipeline,
    snapshot_context: dict[str, Any],
) -> None:
    """Reject directory rescans that add or drop files after snapshot sealing."""

    if (
        snapshot_context.get("snapshot_contract_version")
        != SOURCE_SNAPSHOT_CONTRACT_VERSION
    ):
        return
    sealed_paths = {
        str(Path(str(item["path"])).expanduser().resolve()).casefold()
        for item in snapshot_context.get("snapshot_assets") or []
        if isinstance(item, dict) and item.get("path")
    }
    observed_paths = {
        str(asset.path.expanduser().resolve()).casefold() for asset in pipeline.assets
    }
    if observed_paths != sealed_paths:
        added = sorted(observed_paths - sealed_paths)
        missing = sorted(sealed_paths - observed_paths)
        raise ValueError(
            "pipeline asset inventory differs from the sealed source snapshot; "
            f"added={added[:5]}, missing={missing[:5]}"
        )
    sealed_by_path = {
        str(Path(str(item["path"])).expanduser().resolve()).casefold(): item
        for item in snapshot_context.get("snapshot_assets") or []
        if isinstance(item, dict) and item.get("path")
    }
    for asset in pipeline.assets:
        sealed = sealed_by_path.get(str(asset.path.expanduser().resolve()).casefold())
        expected_options_sha256 = str((sealed or {}).get("asset_options_sha256") or "")
        if (
            expected_options_sha256
            and canonical_sha256(asset.options) != expected_options_sha256
        ):
            raise ValueError(
                "pipeline asset parse options differ from sealed source snapshot: "
                f"{asset.path}"
            )


def _apply_snapshot_parse_repairs(
    pipeline: WellSeismicPipeline,
    snapshot_context: dict[str, Any],
) -> int:
    """Replay only sealed, validated parser metadata; never call the LLM again."""

    assets = {
        str(asset.path.expanduser().resolve()).casefold(): asset
        for asset in pipeline.assets
    }
    sealed_assets = {
        str(Path(str(item.get("path"))).expanduser().resolve()).casefold(): item
        for item in snapshot_context.get("snapshot_assets") or []
        if isinstance(item, dict) and item.get("path")
    }
    applied = 0
    for raw in snapshot_context.get("source_snapshot_parse_repairs") or []:
        if (
            not isinstance(raw, dict)
            or raw.get("status") != "applied_to_current_snapshot"
        ):
            continue
        contract_version = raw.get("contract_version")
        if contract_version == _DETERMINISTIC_UNIT_INHERITANCE_CONTRACT_VERSION:
            source = Path(str(raw.get("source_path") or "")).expanduser().resolve()
            normalized_source = str(source).casefold()
            asset = assets.get(normalized_source)
            sealed_asset = sealed_assets.get(normalized_source)
            if asset is None or sealed_asset is None:
                raise ValueError(
                    "sealed deterministic unit inheritance points outside the "
                    "snapshot asset set"
                )
            if (
                raw.get("rule_id") != _DETERMINISTIC_UNIT_INHERITANCE_RULE_ID
                or raw.get("field") != "md"
                or raw.get("inherited_unit") not in {"m", "ft"}
                or raw.get("original_preserved") is not True
            ):
                raise ValueError(
                    "sealed deterministic unit inheritance is outside the "
                    "allowlisted rule contract"
                )
            source_sha256 = str(sealed_asset.get("sha256") or "").casefold()
            if (
                len(source_sha256) != 64
                or str(raw.get("source_sha256_before") or "").casefold()
                != source_sha256
                or str(raw.get("source_sha256_after") or "").casefold() != source_sha256
            ):
                raise ValueError(
                    "sealed deterministic unit inheritance source identity changed"
                )
            frozen_evidence = {
                key: value for key, value in raw.items() if key != "evidence_sha256"
            }
            if canonical_sha256(frozen_evidence) != str(
                raw.get("evidence_sha256") or ""
            ):
                raise ValueError(
                    "sealed deterministic unit inheritance evidence changed"
                )
            field_units = dict(asset.options.get("field_units") or {})
            existing_unit = str(field_units.get("md") or "").strip().casefold()
            inherited_unit = str(raw["inherited_unit"])
            if existing_unit not in {"", "unknown", inherited_unit}:
                raise ValueError(
                    "sealed deterministic unit inheritance conflicts with the "
                    "replayed asset options"
                )
            field_units["md"] = inherited_unit
            asset.options["field_units"] = field_units
            asset.options["deterministic_unit_inheritance"] = dict(raw)
            applied += 1
            continue
        if contract_version != REPAIR_CONTRACT_VERSION:
            raise ValueError("sealed parse repair contract is incompatible")
        source = Path(str(raw.get("source_path") or "")).expanduser().resolve()
        asset = assets.get(str(source).casefold())
        if asset is None:
            raise ValueError(
                "sealed parse repair points outside the snapshot asset set"
            )
        expected_source_sha256 = str(raw.get("source_sha256") or "")
        if (
            len(expected_source_sha256) != 64
            or file_sha256(source) != expected_source_sha256
        ):
            raise ValueError("sealed parse repair source content changed")
        patch = raw.get("options_patch")
        if not isinstance(patch, dict) or set(patch) - {
            "delimiter",
            "columns",
            "field_units",
        }:
            raise ValueError("sealed parse repair contains a non-allowlisted option")
        delimiter = patch.get("delimiter")
        if delimiter is not None and delimiter not in {",", "\t", ";", "|"}:
            raise ValueError("sealed parse repair delimiter is not allowlisted")
        columns = patch.get("columns") or {}
        allowed_columns = {
            "well_name",
            "md",
            "tvd",
            "tvdss",
            "inclination",
            "azimuth",
            "x",
            "y",
            "x_offset",
            "y_offset",
        }
        if not isinstance(columns, dict) or set(columns) - allowed_columns:
            raise ValueError("sealed parse repair column mapping is not allowlisted")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 63
            for value in columns.values()
        ):
            raise ValueError("sealed parse repair has an invalid column index")
        field_units = patch.get("field_units") or {}
        allowed_units = {"md", "tvd", "tvdss", "x", "y", "x_offset", "y_offset"}
        if (
            not isinstance(field_units, dict)
            or set(field_units) - allowed_units
            or any(unit not in {"m", "ft"} for unit in field_units.values())
        ):
            raise ValueError("sealed parse repair has a non-allowlisted unit")
        expected_patch_sha256 = repair_fingerprint(
            {
                "contract_version": REPAIR_CONTRACT_VERSION,
                "source_sha256": expected_source_sha256,
                "options_patch": patch,
            }
        )
        if expected_patch_sha256 != str(raw.get("patch_sha256") or ""):
            raise ValueError("sealed parse repair patch identity changed")
        for key, value in patch.items():
            if key in {"columns", "field_units"}:
                current = dict(asset.options.get(key) or {})
                current.update(dict(value))
                asset.options[key] = current
            else:
                asset.options[key] = value
        asset.options["llm_parse_repair"] = {
            key: raw[key]
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
        applied += 1
    return applied


def _validate_source_snapshot(request: PreprocessingRequest) -> dict[str, Any]:
    if not request.source_snapshot_id:
        raise HTTPException(status_code=409, detail="请先完成数据准备，再启动井震标定")
    try:
        task = _get_task(request.source_snapshot_id)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail="引用的数据快照不存在") from exc
    if task.get("task_type") != "data_preparation" or task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="引用的数据准备任务尚未完成")
    if _request_path_contract(request) != _request_path_contract(
        task.get("request") or {}
    ):
        raise HTTPException(
            status_code=409, detail="当前路径与已审计数据快照不一致，请重新执行数据准备"
        )
    try:
        source_manifest = _source_snapshot_manifest_from_task(task)
        if source_manifest is not None:
            validate_snapshot_request_semantics(
                source_manifest,
                request,
                effective_config_sha256=_effective_configuration_sha256(),
                transformation_registry_sha256=_transformation_registry_sha256(),
            )
        _prediction_snapshot_context(request.source_snapshot_id)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"source data snapshot integrity verification failed: {exc}",
        ) from exc
    return task


_TRAJECTORY_FILE_SUFFIXES = frozenset(
    {".dev", ".traj", ".trajectory", ".track", ".well", ".prn"}
)


def _registration_preflight_failure(
    *,
    code: str,
    category: str,
    reason: str,
    horizontal_fallback_allowed: bool,
    requires_new_snapshot: bool,
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return the stable 409 payload consumed by the workflow orchestrator."""

    return RegistrationPreflightFailureDetail(
        code=code,
        category=category,
        horizontal_fallback_allowed=horizontal_fallback_allowed,
        requires_new_snapshot=requires_new_snapshot,
        missing_fields=missing_fields or [],
        reason=reason,
    ).model_dump(mode="json")


def _registration_preflight_failure_from_exception(
    exc: HTTPException,
) -> dict[str, Any]:
    """Never turn a sealed-snapshot failure into permission to downgrade."""

    if isinstance(exc.detail, dict):
        return dict(exc.detail)
    reason = str(exc.detail)
    normalized = reason.casefold()
    if "semantic contract changed" in normalized:
        return _registration_preflight_failure(
            code="source_snapshot_semantic_drift",
            category="source_snapshot_semantic_drift",
            reason=reason,
            horizontal_fallback_allowed=False,
            requires_new_snapshot=True,
        )
    if "integrity verification failed" in normalized:
        return _registration_preflight_failure(
            code="source_snapshot_integrity_verification_failed",
            category="source_snapshot_integrity",
            reason=reason,
            horizontal_fallback_allowed=False,
            requires_new_snapshot=True,
        )
    return _registration_preflight_failure(
        code="source_snapshot_unavailable",
        category="source_snapshot_unavailable",
        reason=reason,
        horizontal_fallback_allowed=False,
        requires_new_snapshot=False,
    )


def _is_source_snapshot_integrity_failure(exc: BaseException) -> bool:
    """Identify sealed-lineage failures that must never authorize a fallback."""

    normalized = str(exc).casefold()
    return any(
        marker in normalized
        for marker in (
            "source snapshot manifest is invalid",
            "source snapshot manifest content changed after sealing",
            "source snapshot manifest id differs from its task lineage",
            "snapshot integrity",
            "snapshot_sha256",
            "sourcesnapshot清单",
        )
    )


def _registration_source_quality_issues(
    request: PreprocessingRequest,
    source_task: dict[str, Any],
) -> list[dict[str, str]]:
    """Detect a trajectory that was supplied but could only yield well heads.

    This gate deliberately reads the sealed preparation receipt rather than
    reparsing mutable source files.  A formal time-domain tie must not silently
    run on head-only geometry when the user provided trajectory stations.
    """

    if not request.well_paths:
        return []
    result = source_task.get("result") or {}
    assets = [item for item in result.get("assets") or [] if isinstance(item, dict)]
    declared = [str(path).strip() for path in request.well_paths if str(path).strip()]

    def _is_declared(path: str) -> bool:
        candidate = Path(path).expanduser()
        candidate_text = str(candidate).casefold()
        for raw in declared:
            root = Path(raw).expanduser()
            root_text = str(root).casefold()
            if candidate_text == root_text or candidate_text.startswith(
                root_text.rstrip("\\/") + os.sep.casefold()
            ):
                return True
        return False

    trajectory_paths = {
        str(item.get("path") or "").strip()
        for item in assets
        if str(item.get("path") or "").strip()
        and Path(str(item.get("path"))).suffix.casefold() in _TRAJECTORY_FILE_SUFFIXES
        and _is_declared(str(item.get("path")))
    }
    trajectory_paths.update(
        raw
        for raw in declared
        if Path(raw).suffix.casefold() in _TRAJECTORY_FILE_SUFFIXES
    )
    if not trajectory_paths:
        return []

    errors = [item for item in result.get("errors") or [] if isinstance(item, dict)]
    detections = [
        item
        for item in result.get("metadata_detection") or []
        if isinstance(item, dict)
    ]
    preparation = result.get("preparation") or {}
    preparation_issues = [
        item for item in preparation.get("issues") or [] if isinstance(item, dict)
    ]
    # The preparation report is the authoritative post-parser count.  A
    # recognized header without any registered station must still be treated as
    # head-only degradation, not a usable trajectory.
    stage_metrics = {
        str(item.get("id") or ""): item.get("metrics") or {}
        for item in preparation.get("stages") or []
        if isinstance(item, dict)
    }
    trajectory_wells_value = stage_metrics.get("well_entity_alignment", {}).get(
        "真实轨迹井"
    )
    try:
        reported_trajectory_wells = int(trajectory_wells_value)
    except (TypeError, ValueError):
        reported_trajectory_wells = None

    def _same_path(left: str, right: str) -> bool:
        return (
            left.casefold() == right.casefold()
            or Path(left).name.casefold() == Path(right).name.casefold()
        )

    issues: list[dict[str, str]] = []
    for path in sorted(trajectory_paths):
        parse_error = next(
            (
                str(item.get("error") or "轨迹解析失败")
                for item in errors
                if _same_path(str(item.get("path") or ""), path)
            ),
            None,
        )
        recognized = any(
            _same_path(str(item.get("文件") or item.get("path") or ""), path)
            and "井轨迹" in str(item.get("识别角色") or item.get("role") or "")
            and "失败" not in str(item.get("状态") or item.get("status") or "")
            for item in detections
        )
        # ``trajectory_degraded`` is the reporting bucket for every audited
        # trajectory transformation, including valid minimum-curvature TVD/XY
        # reconstruction and deterministic unit inheritance.  Those receipts
        # still contain real, full-resolution stations and must not be confused
        # with the head-only fallback this gate is designed to reject.
        #
        # Keep the fail-closed behavior for the one transformation that really
        # fabricates zero offsets because no deviation survey exists.  Parse
        # errors, an unrecognized asset and a zero global station count remain
        # independently blocking below.
        reported_head_only = any(
            _same_path(str(item.get("source") or ""), path)
            and str(item.get("group_key") or "").startswith("trajectory_degraded:")
            and "missing_deviation_survey_no_md_to_tvd_conversion"
            in str(item.get("group_key") or "").casefold()
            for item in preparation_issues
        )
        if parse_error:
            issues.append(
                {
                    "code": "trajectory_parse_failed",
                    "path": path,
                    "reason": f"已提供井轨迹但数据准备解析失败：{path}；{parse_error}",
                }
            )
        elif not recognized or reported_head_only or reported_trajectory_wells == 0:
            issues.append(
                {
                    "code": "trajectory_degraded_to_head_only",
                    "path": path,
                    "reason": f"已提供井轨迹但封存快照未确认轨迹站点，不能以head_only替代：{path}",
                }
            )
    return issues


def _require_registration_source_quality(
    request: PreprocessingRequest,
    source_task: dict[str, Any],
) -> None:
    issues = _registration_source_quality_issues(request, source_task)
    if not issues:
        return
    reason = "；".join(item["reason"] for item in issues)
    raise HTTPException(
        status_code=409,
        detail=_registration_preflight_failure(
            code=issues[0]["code"],
            category="source_quality_unavailable",
            reason=reason,
            horizontal_fallback_allowed=False,
            requires_new_snapshot=True,
            missing_fields=[f"trajectory:{item['path']}" for item in issues],
        ),
    )


def _registration_crs_identifier_is_concrete(value: Any) -> bool:
    """Reject placeholder CRS labels before formal vertical registration."""

    normalized = str(value or "").strip().upper()
    if not normalized or normalized in {"NONE", "N/A", "NA", "-"}:
        return False
    return not any(
        placeholder in normalized for placeholder in ("UNKNOWN", "UNSPECIFIED", "NULL")
    )


def _require_registration_semantic_contract(
    request: PreprocessingRequest,
) -> None:
    """Fail closed before any physical or learned time-depth registration."""

    issues: list[str] = []
    if request.coordinate_reference_verified is not True:
        issues.append("coordinate_reference_verified必须为true")
    if not _registration_crs_identifier_is_concrete(request.horizontal_crs_id):
        issues.append("horizontal_crs_id未声明或仍为UNKNOWN/UNSPECIFIED占位值")
    if request.horizontal_unit != "m":
        issues.append("horizontal_unit必须为m")
    if request.horizontal_axis_order not in {"XY", "YX"}:
        issues.append("horizontal_axis_order必须明确为XY或YX")
    if request.seismic_time_domain != "TWT":
        issues.append("seismic_time_domain必须为TWT")
    if request.seismic_correction_state != "corrected_to_srd":
        issues.append("seismic_correction_state必须为corrected_to_srd")
    if request.seismic_srd_elevation_m is None or not np.isfinite(
        float(request.seismic_srd_elevation_m)
    ):
        issues.append("seismic_srd_elevation_m未明确")
    if not _registration_crs_identifier_is_concrete(request.vertical_crs_id):
        issues.append("vertical_crs_id未声明或仍为UNKNOWN/UNSPECIFIED占位值")
    if issues:
        raise ValueError("井震精细标定语义合同不完整：" + "；".join(issues))


_NATIVE_RELATIVE_REGISTRATION_CONTRACT_VERSION = (
    "well-seismic.native-relative-registration.v1"
)


def _native_relative_registration_contract(
    request: PreprocessingRequest,
    source_task: dict[str, Any],
) -> dict[str, Any]:
    """Return an inference-only contract for a sealed survey without TD data.

    The contract authorizes the existing relative sonic and P13 paths; it does
    not manufacture an SRD, correction state or observed time-depth label.
    Eligibility remains per well at runtime so one incomplete LAS/trajectory
    pair cannot block every otherwise usable well.
    """

    issues: list[str] = []
    if request.time_depth_paths:
        issues.append("native-relative模式仅适用于未提供TD/checkshot/VSP的测区")
    if request.coordinate_reference_verified is not True:
        issues.append("井与SEG-Y水平坐标尚未通过同CRS或原生同网格核验")
    if not _registration_crs_identifier_is_concrete(request.horizontal_crs_id):
        issues.append("水平坐标合同缺少可执行的投影CRS或LOCAL_SURVEY_XY命名空间")
    if request.horizontal_unit != "m" or request.horizontal_axis_order not in {
        "XY",
        "YX",
    }:
        issues.append("水平坐标必须已归一到米并明确轴序")
    result = source_task.get("result") or {}
    manifest = _source_snapshot_manifest_from_task(source_task)
    if not isinstance(manifest, dict):
        issues.append("缺少封存SourceSnapshot清单")
        manifest = {}
    source_assets = [
        dict(item)
        for item in (manifest.get("source_assets") or [])
        if isinstance(item, dict)
    ]
    seismic_assets = [
        item
        for item in source_assets
        if str(item.get("role") or "").casefold() == "seismic"
    ]
    td_assets = [
        item
        for item in source_assets
        if str(item.get("role") or "").casefold() == "time_depth"
    ]
    if len(seismic_assets) != 1:
        issues.append("native-relative自动合同当前要求单一SEG-Y地震体")
    if td_assets:
        issues.append("封存快照包含TD/checkshot/VSP资产")

    preparation = result.get("preparation") or {}
    stages = {
        str(item.get("id") or ""): item.get("metrics") or {}
        for item in (preparation.get("stages") or [])
        if isinstance(item, dict)
    }
    well_metrics = stages.get("well_entity_alignment", {})
    seismic_metrics = stages.get("seismic_geometry", {})
    time_depth_policy = (preparation.get("task_readiness") or {}).get(
        "time_depth_policy"
    ) or {}

    def _count(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    acoustic_wells = _count(time_depth_policy.get("acoustic_candidate_well_count"))
    trajectory_wells = _count(well_metrics.get("真实轨迹井"))
    locatable_wells = _count(well_metrics.get("可定位井"))
    readable_seismic = _count(seismic_metrics.get("有坐标几何"))

    registration_entry_policy = dict(
        (preparation.get("task_readiness") or {}).get("registration_entry_policy") or {}
    )
    per_well_receipts = [
        dict(item)
        for item in (
            registration_entry_policy.get("native_relative_well_receipts") or []
        )
        if isinstance(item, dict)
    ]
    eligible_wells = [
        item for item in per_well_receipts if item.get("eligible") is True
    ]
    isolated_wells = [
        item for item in per_well_receipts if item.get("eligible") is not True
    ]
    if not per_well_receipts:
        issues.append("封存快照缺少逐井native-relative能力收据，请重新执行数据准备")
    if not eligible_wells:
        issues.append(
            "至少需要同一口井同时具备声波曲线、米制井口XY、已解析KB/DF/RT、"
            "完整递增MD-TVD轨迹及地震覆盖"
        )
    if readable_seismic < 1:
        issues.append("SEG-Y缺少可用坐标几何")

    def _same_asset_path(left: Any, right: Any) -> bool:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text:
            return False
        return (
            left_text.casefold() == right_text.casefold()
            or Path(left_text).name.casefold() == Path(right_text).name.casefold()
        )

    seismic_paths = [str(item.get("path") or "") for item in seismic_assets]
    preparation_errors = [
        item for item in (result.get("errors") or []) if isinstance(item, dict)
    ]
    critical_seismic_errors = [
        item
        for item in preparation_errors
        if any(
            _same_asset_path(item.get("path"), seismic_path)
            for seismic_path in seismic_paths
        )
    ]
    if critical_seismic_errors:
        issues.append("封存SEG-Y资产存在解析错误")
    isolated_noncritical_errors = [
        item for item in preparation_errors if item not in critical_seismic_errors
    ]
    if issues:
        raise ValueError("；".join(dict.fromkeys(issues)))

    hashes = manifest.get("hashes") or {}
    snapshot_sha256 = str(hashes.get("snapshot_sha256") or "").casefold()
    if len(snapshot_sha256) != 64:
        raise ValueError("封存快照缺少有效snapshot_sha256")

    time_axis_basis: dict[str, Any] | None = None
    if request.seismic_time_domain == "TWT":
        time_axis_basis = {
            "source": "sealed_source_snapshot_request",
            "rule": "explicit_TWT",
            "confidence": 1.0,
        }
    registration_evidence = manifest.get("registration_evidence") or {}
    candidate_rows = (
        []
        if request.seismic_time_domain == "OWT"
        else registration_evidence.get("candidates") or []
    )
    for candidate in candidate_rows:
        if not isinstance(candidate, dict):
            continue
        if (
            str(candidate.get("field") or "") != "seismic_time_domain"
            or str(candidate.get("value") or "").upper() != "TWT"
        ):
            continue
        candidate_evidence = [
            str(item) for item in (candidate.get("evidence") or []) if str(item).strip()
        ]
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        explicit_verified = bool(
            candidate.get("status") == "verified"
            and confidence >= 0.95
            and candidate.get("requires_human_confirmation") is False
            and candidate.get("auto_applied") is True
        )
        if explicit_verified:
            time_axis_basis = {
                "source": candidate.get("source"),
                "rule": "verified_auto_applied_TWT_evidence",
                "confidence": confidence,
                "evidence": candidate_evidence,
            }
            break
    time_axis_ready = time_axis_basis is not None
    explicit_vertical = (
        str(request.vertical_crs_id or "").strip()
        if _registration_crs_identifier_is_concrete(request.vertical_crs_id)
        else ""
    )
    vertical_namespace = explicit_vertical or (
        f"LOCAL_MSL_SURVEY_{snapshot_sha256[:12].upper()}"
    )
    evidence = {
        "source_snapshot_id": str(source_task.get("task_id") or ""),
        "source_snapshot_sha256": snapshot_sha256,
        "bound_segy_full_content_sha256": seismic_assets[0].get("sha256"),
        "horizontal_crs_id": request.horizontal_crs_id,
        "horizontal_unit": request.horizontal_unit,
        "horizontal_axis_order": request.horizontal_axis_order,
        "acoustic_candidate_well_count": acoustic_wells,
        "measured_trajectory_well_count": trajectory_wells,
        "locatable_well_count": locatable_wells,
        "eligible_well_count": len(eligible_wells),
        "eligible_wells": eligible_wells,
        "isolated_well_count": len(isolated_wells),
        "isolated_noncritical_error_count": len(isolated_noncritical_errors),
        "runtime_eligibility_policy": (
            "sealed_per_well_acoustic_plus_canonical_xy_plus_kb_df_rt_plus_"
            "finite_increasing_md_plus_finite_stationwise_tvd_plus_seismic_coverage"
        ),
        "time_axis_ready": time_axis_ready,
        "time_axis_basis": time_axis_basis,
        "declared_time_domain": request.seismic_time_domain,
    }
    contract = {
        "contract_version": _NATIVE_RELATIVE_REGISTRATION_CONTRACT_VERSION,
        "mode": "native_relative_no_time_depth",
        "fusion_ready_semantics": (
            "quality_gated_MD_to_native_seismic_time_for_downstream_inference"
        ),
        "absolute_reference_ready": False,
        "vertical_crs_id": vertical_namespace,
        "seismic_srd_elevation_m": None,
        "time_domain": "TWT" if time_axis_ready else "native_time_unknown",
        "time_axis_ready": time_axis_ready,
        "time_reference": "native_segy_sample_zero",
        "correction_state": "native_unmodified",
        "time_depth_supervision_is_model_input": False,
        "supervision_eligible": False,
        "training_eligible": False,
        "well_eligibility_scope": "per_well",
        "evidence": evidence,
        "evidence_sha256": canonical_sha256(evidence),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


_REGISTRATION_PREFLIGHT_RULE_VERSION = "well-seismic.registration-preflight.rules.v1"
_REGISTRATION_PREFLIGHT_MIN_CONFIDENCE = SYSTEM_EVIDENCE_MIN_CONFIDENCE
_REGISTRATION_PREFLIGHT_FIELDS = (
    "vertical_crs_id",
    "seismic_srd_elevation_m",
    "seismic_time_domain",
    "seismic_correction_state",
)


def _normalized_registration_candidate_value(
    field: str,
    value: Any,
    *,
    require_formal: bool = False,
) -> Any:
    if field == "vertical_crs_id":
        normalized = str(value or "").strip()
        upper = normalized.upper()
        if not normalized or "MSL" not in upper or "UNSPECIFIED" in upper:
            raise ValueError("垂向CRS候选必须是明确的MSL测区标识")
        return normalized
    if field == "seismic_srd_elevation_m":
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("SRD候选必须是数值") from exc
        if not np.isfinite(normalized):
            raise ValueError("SRD候选必须是有限数值")
        return normalized
    if field == "seismic_time_domain":
        normalized = str(value or "").strip().upper()
        if normalized not in {"TWT", "OWT"}:
            raise ValueError("时间域候选必须为TWT或OWT")
        if require_formal and normalized != "TWT":
            raise ValueError("正式井震标定的时间域候选必须为TWT")
        return normalized
    if field == "seismic_correction_state":
        normalized = str(value or "").strip().lower()
        if normalized not in {"corrected_to_srd", "uncorrected"}:
            raise ValueError("时间校正候选必须为corrected_to_srd或uncorrected")
        if require_formal and normalized != "corrected_to_srd":
            raise ValueError("正式井震标定的时间校正候选必须为corrected_to_srd")
        return normalized
    raise ValueError(f"不支持的井震语义候选字段：{field}")


def _registration_field_is_explicit(field: str, value: Any) -> bool:
    if field == "seismic_srd_elevation_m":
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False
    normalized = str(value or "").strip()
    if field == "vertical_crs_id":
        return bool(normalized) and normalized.upper() != "LOCAL_MSL_UNSPECIFIED"
    return normalized.lower() != "unknown"


def _manifest_seismic_bindings(
    source_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for item in source_manifest.get("source_assets") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").casefold() != "seismic":
            continue
        digest = str(item.get("sha256") or "").strip().casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("封存SEG-Y资产缺少有效的整文件SHA-256")
        bindings.append(
            {
                "asset_id": item.get("id"),
                "name": item.get("name"),
                "size": item.get("size"),
                "full_content_sha256": digest,
            }
        )
    return bindings


def _registration_contract_evidence(
    source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = source_manifest.get("registration_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("父快照未封存可复核的井震语义证据；自动预检将保持水平相对模式")
    if evidence.get("contract_version") != REGISTRATION_EVIDENCE_CONTRACT_VERSION:
        raise ValueError("父快照的井震语义证据版本不受支持")
    hashes = source_manifest.get("hashes") or {}
    expected_sha256 = str(hashes.get("registration_evidence_sha256") or "").casefold()
    actual_sha256 = canonical_sha256(evidence)
    if expected_sha256 != actual_sha256:
        raise ValueError("父快照的井震语义证据身份校验失败")
    bound_assets = evidence.get("bound_segy_assets")
    if not isinstance(bound_assets, list):
        raise ValueError("父快照的井震语义证据缺少SEG-Y内容绑定")
    manifest_bindings = _manifest_seismic_bindings(source_manifest)
    if canonical_sha256(bound_assets) != canonical_sha256(manifest_bindings):
        raise ValueError("父快照的井震语义证据与SEG-Y内容身份不一致")
    if len(manifest_bindings) != 1:
        raise ValueError("自动正式标定当前只支持单一SEG-Y；多体数据将保持水平相对模式")
    raw_candidates = evidence.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("父快照的井震语义候选不可读")
    candidates = [dict(item) for item in raw_candidates if isinstance(item, dict)]
    request_patch = evidence.get("request_patch")
    if not isinstance(request_patch, dict):
        raise ValueError("父快照的井震语义request_patch不可读")
    return candidates, dict(request_patch)


def _sealed_request_registration_candidate(
    field: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "field": field,
        "value": value,
        "confidence": 1.0,
        "status": "verified",
        "inference_source": "explicit_input",
        "requires_human_confirmation": False,
        "auto_applied": True,
        "rule_id": "sealed_request_semantics.v1",
        "source": "sealed_source_snapshot_request",
        "evidence": ["字段已作为父SourceSnapshot的显式语义封存"],
    }


def _bounded_registration_decision(
    source_task: dict[str, Any],
    source_manifest: dict[str, Any],
    request: PreprocessingRequest,
    *,
    parent_snapshot_sha256: str,
) -> tuple[InspectionRequest, dict[str, Any]]:
    candidates, request_patch = _registration_contract_evidence(source_manifest)

    selected: list[dict[str, Any]] = []
    effective_patch: dict[str, Any] = {}
    for field in _REGISTRATION_PREFLIGHT_FIELDS:
        eligible: list[dict[str, Any]] = []
        for raw_candidate in candidates:
            if str(raw_candidate.get("field") or "") != field:
                continue
            inference_source = str(raw_candidate.get("inference_source") or "").strip()
            if inference_source not in {"explicit_input", "rule"}:
                continue
            if str(raw_candidate.get("status") or "").strip() != "verified":
                continue
            if raw_candidate.get("requires_human_confirmation") is not False:
                continue
            if raw_candidate.get("auto_applied") is not True:
                continue
            rule_id = str(raw_candidate.get("rule_id") or "").strip()
            if rule_id not in SYSTEM_EVIDENCE_ALLOWED_RULE_IDS:
                continue
            try:
                confidence = float(raw_candidate.get("confidence"))
            except (TypeError, ValueError):
                continue
            if (
                not np.isfinite(confidence)
                or confidence < _REGISTRATION_PREFLIGHT_MIN_CONFIDENCE
            ):
                continue
            evidence = raw_candidate.get("evidence")
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list) or not any(
                str(item).strip() for item in evidence
            ):
                continue
            try:
                value = _normalized_registration_candidate_value(
                    field, raw_candidate.get("value")
                )
            except ValueError:
                continue
            eligible.append(
                {
                    "field": field,
                    "value": value,
                    "confidence": confidence,
                    "status": "verified",
                    "inference_source": inference_source,
                    "requires_human_confirmation": False,
                    "auto_applied": True,
                    "rule_id": rule_id,
                    "source": str(raw_candidate.get("source") or ""),
                    "evidence": [str(item) for item in evidence if str(item).strip()],
                }
            )
        current_value = getattr(request, field)
        if _registration_field_is_explicit(field, current_value):
            normalized_current = _normalized_registration_candidate_value(
                field, current_value
            )
            conflicting = [
                candidate
                for candidate in eligible
                if candidate["value"] != normalized_current
            ]
            if conflicting:
                raise ValueError(f"{field}的封存显式值与规则候选冲突，禁止自动改写")
            selected_value = normalized_current
            strongest = _sealed_request_registration_candidate(
                field,
                selected_value,
            )
        else:
            values = {
                canonical_sha256(candidate["value"]): candidate["value"]
                for candidate in eligible
            }
            if not values:
                raise ValueError(
                    f"{field}没有可自动采纳的已核验规则证据；"
                    "需人工复核或LLM建议的候选将自动转入水平相对模式"
                )
            if len(values) != 1:
                raise ValueError(f"{field}存在已核验规则候选冲突，禁止自动派生")
            selected_value = next(iter(values.values()))
            strongest = max(
                eligible,
                key=lambda candidate: (
                    float(candidate["confidence"]),
                    candidate["inference_source"] == "explicit_input",
                ),
            )
        _normalized_registration_candidate_value(
            field,
            selected_value,
            require_formal=True,
        )
        if field in request_patch:
            normalized_patch = _normalized_registration_candidate_value(
                field, request_patch[field]
            )
            if normalized_patch != selected_value:
                raise ValueError(f"{field}的request_patch与规则候选冲突，禁止自动派生")
        effective_patch[field] = selected_value
        selected.append(strongest)

    selected.sort(key=lambda candidate: str(candidate["field"]))
    effective_raw = request.model_dump(mode="json")
    effective_raw.update(effective_patch)
    effective_raw.pop("source_snapshot_id", None)
    effective_raw.pop("registration_task_id", None)
    effective_raw.pop("output_directory", None)
    effective_raw["survey_attestation"] = None
    effective_request = InspectionRequest.model_validate(effective_raw)
    registration_request = PreprocessingRequest(
        **effective_request.model_dump(mode="json"),
        source_snapshot_id=str(source_task["task_id"]),
    )
    _require_registration_semantic_contract(registration_request)

    decision_basis = {
        "rule_version": _REGISTRATION_PREFLIGHT_RULE_VERSION,
        "parent_snapshot_id": str(source_task["task_id"]),
        "parent_snapshot_sha256": parent_snapshot_sha256,
        "effective_patch": effective_patch,
        "selected_candidates": selected,
    }
    decision = {
        "decision_type": "bounded_machine_decision",
        "source": "deterministic_rule",
        **decision_basis,
        "decision_basis_sha256": canonical_sha256(decision_basis),
    }
    return effective_request, decision


def _derived_preflight_response(
    *,
    source_snapshot_id: str,
    derived_snapshot_id: str,
) -> RegistrationPreflightResponse:
    try:
        derived_task = _get_task(derived_snapshot_id)
    except KeyError as exc:
        raise ValueError("自动派生快照任务不存在") from exc
    if (
        derived_task.get("task_type") != "data_preparation"
        or derived_task.get("status") != "completed"
        or derived_task.get("parent_task_id") != source_snapshot_id
    ):
        raise ValueError("自动派生快照任务未完成或父快照不一致")
    effective, _snapshot_context = _sealed_horizontal_registration_request(
        HorizontalRegistrationRequest(source_snapshot_id=derived_snapshot_id)
    )
    _require_registration_semantic_contract(effective)
    _require_registration_source_quality(effective, derived_task)
    manifest = _source_snapshot_manifest_from_task(derived_task)
    receipt = (
        manifest.get("system_evidence_receipt") if isinstance(manifest, dict) else None
    )
    if not isinstance(receipt, dict):
        raise ValueError("自动派生快照缺少system_evidence_receipt")
    return RegistrationPreflightResponse(
        source_snapshot_id=source_snapshot_id,
        derived_snapshot_id=derived_snapshot_id,
        resolution="derived_bounded_machine_decision",
        effective_request=effective,
        system_evidence_receipt=receipt,
    )


def _derive_registration_ready_snapshot(
    source_snapshot_id: str,
    source_request: PreprocessingRequest,
) -> RegistrationPreflightResponse:
    source_task = _get_task(source_snapshot_id)
    source_manifest = _source_snapshot_manifest_from_task(source_task)
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("contract_version") != SOURCE_SNAPSHOT_CONTRACT_VERSION
    ):
        raise ValueError("自动井震语义派生仅支持封存的SourceSnapshot V3")
    parent_snapshot_sha256 = str(
        (source_manifest.get("hashes") or {}).get("snapshot_sha256") or ""
    ).casefold()
    if len(parent_snapshot_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in parent_snapshot_sha256
    ):
        raise ValueError("父快照缺少有效snapshot_sha256")
    effective_request, decision = _bounded_registration_decision(
        source_task,
        source_manifest,
        source_request,
        parent_snapshot_sha256=parent_snapshot_sha256,
    )
    decision_basis_sha256 = str(decision["decision_basis_sha256"])
    derived_snapshot_id = decision_basis_sha256[:32]

    with _registration_preflight_lock:
        try:
            existing = _get_task(derived_snapshot_id)
        except KeyError:
            existing = None
        if existing is not None:
            recorded_basis = str(
                (existing.get("bounded_machine_decision") or {}).get(
                    "decision_basis_sha256"
                )
                or ""
            )
            if recorded_basis != decision_basis_sha256:
                raise ValueError("自动派生快照ID已被不同判定占用，禁止覆盖")
            return _derived_preflight_response(
                source_snapshot_id=source_snapshot_id,
                derived_snapshot_id=derived_snapshot_id,
            )

        project_id = str(source_task.get("project_id") or DEFAULT_PROJECT_ID)
        _state_store.ensure_project(
            project_id,
            {"name": "本地默认项目", "kind": "desktop_local"},
        )
        created_at = _now()
        task = _state_store.create_task(
            {
                "progress": 1,
                "message": "正在生成井震标定就绪的不可变派生快照",
                "created_at": created_at,
                "updated_at": created_at,
                "result": None,
                "error": None,
                "parent_task_id": source_snapshot_id,
                "request": effective_request.model_dump(mode="json"),
                "bounded_machine_decision": decision,
            },
            task_id=derived_snapshot_id,
            project_id=project_id,
            task_type="data_preparation",
            status="running",
        )
        with _tasks_lock:
            _tasks[derived_snapshot_id] = task

        try:
            derived_result = copy.deepcopy(source_task.get("result") or {})
            data_snapshot = dict(derived_result.get("data_snapshot") or {})
            for key in (
                "artifact_bundle_id",
                "parent_snapshot_id",
                "snapshot_id",
                "snapshot_manifest_path",
                "snapshot_manifest_sha256",
                "snapshot_sha256",
                "survey_attestation_receipt",
                "survey_attestation_sha256",
                "system_evidence_receipt",
                "system_evidence_receipt_sha256",
            ):
                data_snapshot.pop(key, None)
            derived_result["data_snapshot"] = data_snapshot
            derived_result["bounded_machine_decision"] = {
                **decision,
                "derived_snapshot_id": derived_snapshot_id,
            }
            preparation = derived_result.get("preparation")
            if isinstance(preparation, dict):
                preparation["applied_registration_request_patch"] = dict(
                    decision["effective_patch"]
                )
                preparation["automatic_registration_resolution"] = {
                    "decision_type": "bounded_machine_decision",
                    "rule_version": _REGISTRATION_PREFLIGHT_RULE_VERSION,
                    "parent_snapshot_id": source_snapshot_id,
                    "derived_snapshot_id": derived_snapshot_id,
                }
            _seal_data_preparation_snapshot(
                derived_snapshot_id,
                effective_request,
                derived_result,
                project_id=project_id,
                parent_snapshot_id=source_snapshot_id,
                system_evidence_decision=decision,
            )
            _set_task(
                derived_snapshot_id,
                status="completed",
                progress=100,
                message="井震标定语义已由确定性证据解析并封存",
                result=derived_result,
            )
        except Exception as exc:
            _set_task(
                derived_snapshot_id,
                status="failed",
                progress=100,
                message="井震标定自动语义派生失败",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        return _derived_preflight_response(
            source_snapshot_id=source_snapshot_id,
            derived_snapshot_id=derived_snapshot_id,
        )


def _runtime_contract_replay_assets(
    source_manifest: dict[str, Any],
    effective: PreprocessingRequest,
) -> list[dict[str, Any]]:
    """Recompute only parser-option identities; never ingest source samples."""

    manifest, _inventory = build_explicit_paths_manifest(
        seismic_directory=effective.seismic_paths,
        log_directory=effective.log_paths,
        metadata_directory=effective.well_paths,
        auxiliary_directory=effective.auxiliary_paths,
        survey_directory=effective.survey_paths,
        interpretation_directory=effective.interpretation_paths,
        time_depth_directory=effective.time_depth_paths,
        recursive=effective.recursive,
        require_seismic=False,
        require_logs=False,
        seismic_srd_elevation_m=effective.seismic_srd_elevation_m,
        vertical_crs_id=effective.vertical_crs_id,
        horizontal_crs_id=effective.horizontal_crs_id,
        well_source_crs_id=effective.well_source_crs_id,
        seismic_source_crs_id=effective.seismic_source_crs_id,
        horizontal_unit=effective.horizontal_unit,
        horizontal_axis_order=effective.horizontal_axis_order,
        coordinate_reference_verified=effective.coordinate_reference_verified,
        seismic_replacement_velocity_mps=effective.seismic_replacement_velocity_mps,
        seismic_time_domain=effective.seismic_time_domain,
        seismic_correction_state=effective.seismic_correction_state,
        segy_geometry_profile=effective.segy_geometry_profile,
        segy_inline_byte=effective.segy_inline_byte,
        segy_crossline_byte=effective.segy_crossline_byte,
        segy_x_byte=effective.segy_x_byte,
        segy_y_byte=effective.segy_y_byte,
        segy_coordinate_scalar_byte=effective.segy_coordinate_scalar_byte,
        well_coordinate_source_unit=effective.well_coordinate_source_unit,
        well_vertical_datum_source_unit=effective.well_vertical_datum_source_unit,
        las_twt_source_unit=effective.las_twt_source_unit,
        time_depth_default_depth_domain=effective.time_depth_default_depth_domain,
        time_depth_default_depth_unit=effective.time_depth_default_depth_unit,
        time_depth_default_time_unit=effective.time_depth_default_time_unit,
        time_depth_default_depth_datum=effective.time_depth_default_depth_datum,
        time_depth_default_depth_convention=effective.time_depth_default_depth_convention,
        time_depth_default_time_reference=effective.time_depth_default_time_reference,
        time_depth_default_time_domain=effective.time_depth_default_time_domain,
        time_depth_default_correction_state=effective.time_depth_default_correction_state,
        target_task_id=effective.target_task_id,
        target_model_id=effective.target_model_id,
        target_scope_explicit=effective.target_scope_explicit,
        target_required_modalities=_target_required_modalities(effective),
        target_model_contract=_target_model_contract(effective),
    )
    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=False,
    )
    _apply_horizontal_coordinate_contract(pipeline, effective)
    source_assets = [
        dict(item)
        for item in (source_manifest.get("source_assets") or [])
        if isinstance(item, dict)
    ]
    replay_context = {
        "snapshot_contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
        "snapshot_assets": source_assets,
        "source_snapshot_parse_repairs": [
            dict(item)
            for item in (source_manifest.get("parse_repairs") or [])
            if isinstance(item, dict)
        ],
    }
    _apply_snapshot_parse_repairs(pipeline, replay_context)

    def asset_key(path: Any, role: Any) -> tuple[str, str]:
        return (
            str(Path(str(path)).expanduser().resolve()).casefold(),
            str(role or "").casefold(),
        )

    observed_counts = Counter(
        asset_key(asset.path, asset.role) for asset in pipeline.assets
    )
    sealed_counts = Counter(
        asset_key(item.get("path"), item.get("role"))
        for item in source_assets
        if item.get("path")
    )
    if observed_counts != sealed_counts or len(pipeline.assets) != len(source_assets):
        raise ValueError(
            "runtime contract replay asset inventory differs from the sealed source"
        )
    observed_by_key: dict[tuple[str, str], deque[Any]] = defaultdict(deque)
    for asset in pipeline.assets:
        observed_by_key[asset_key(asset.path, asset.role)].append(asset)
    refreshed: list[dict[str, Any]] = []
    for source in source_assets:
        key = asset_key(source.get("path"), source.get("role"))
        observed = observed_by_key[key].popleft()
        item = dict(source)
        item["asset_options_sha256"] = canonical_sha256(
            observed.options
        )
        refreshed.append(item)
    if any(observed_by_key.values()):
        raise ValueError("runtime contract replay left unmatched sealed assets")
    return refreshed


def _runtime_contract_confirmation_response(
    *,
    source_snapshot_id: str,
    derived_snapshot_id: str,
    decision_basis_sha256: str,
) -> RuntimeContractConfirmationResponse:
    task = _get_task(derived_snapshot_id)
    if (
        task.get("task_type") != "data_preparation"
        or task.get("status") != "completed"
        or task.get("parent_task_id") != source_snapshot_id
    ):
        raise ValueError("运行参数派生快照未完成或父快照不一致")
    stored_decision = task.get("runtime_contract_confirmation") or {}
    if str(stored_decision.get("decision_basis_sha256") or "") != (
        decision_basis_sha256
    ):
        raise ValueError("运行参数派生快照已被不同人工确认占用")
    effective, _context = _sealed_horizontal_registration_request(
        HorizontalRegistrationRequest(source_snapshot_id=derived_snapshot_id)
    )
    _require_registration_semantic_contract(effective)
    manifest = _source_snapshot_manifest_from_task(task)
    receipt = (
        manifest.get("runtime_contract_receipt")
        if isinstance(manifest, dict)
        else None
    )
    if not isinstance(receipt, dict):
        raise ValueError("运行参数派生快照缺少人工确认回执")
    return RuntimeContractConfirmationResponse(
        source_snapshot_id=source_snapshot_id,
        derived_snapshot_id=derived_snapshot_id,
        effective_request=effective,
        runtime_contract_receipt=receipt,
    )


_RUNTIME_RESOLVED_VERTICAL_TITLES = {
    "绝对垂向基准尚未闭合",
    "本地MSL尚未绑定测区垂向CRS",
    "地震处理基准面高程冲突",
    "SRD未知，保留原生地震时间参考",
    "地震处理基准面SRD尚未确认",
}
_RUNTIME_RESOLVED_TIME_TITLES = {
    "地震绝对时间参考未闭合",
    "地震时间参考未统一到SRD",
}


def _apply_confirmed_runtime_contract_to_result(
    result: dict[str, Any],
    effective: PreprocessingRequest,
    decision: dict[str, Any],
) -> None:
    """Refresh the inspection view after sealing human-confirmed semantics.

    Runtime confirmation intentionally replays parser identities without
    reopening a potentially very large SEG-Y.  The source inspection result,
    however, still contains the pre-confirmation SRD/time view.  Project the
    signed contract into that view so the immutable derived snapshot does not
    display resolved fields as ``unknown`` while downstream registration uses
    the confirmed request.
    """

    datum_inventory = result.get("vertical_datum")
    if not isinstance(datum_inventory, dict):
        return

    vertical_crs = {
        "id": str(effective.vertical_crs_id),
        "unit": "m",
        "axis": "elevation_positive_up",
        "status": "declared",
    }
    datum_inventory["vertical_crs"] = vertical_crs
    for item in datum_inventory.get("wells") or []:
        if isinstance(item, dict):
            item["vertical_crs"] = dict(vertical_crs)

    decision_basis_sha256 = str(decision.get("decision_basis_sha256") or "")
    provenance = f"runtime_contract_confirmation:{decision_basis_sha256}"
    srd_elevation_m = float(effective.seismic_srd_elevation_m)
    replacement_velocity_mps = float(effective.seismic_replacement_velocity_mps)
    seismic_items = [
        item
        for item in (datum_inventory.get("seismic") or [])
        if isinstance(item, dict)
    ]
    for item in seismic_items:
        entity_name = str(item.get("entity_name") or "")
        evidence = (
            "human-confirmed runtime contract: "
            f"SRD={srd_elevation_m:g} m MSL, TWT, corrected_to_srd"
        )
        observation = {
            "entity_kind": "seismic",
            "entity_name": entity_name,
            "datum": "SRD",
            "value": srd_elevation_m,
            "unit": "m",
            "relation": "elevation_above_reference",
            "reference": "MSL",
            "source": provenance,
            "evidence": evidence,
            "confidence": 1.0,
            "review_required": False,
            "is_depth_reference": False,
            "absolute_elevation_m": srd_elevation_m,
        }
        observations = [
            dict(value)
            for value in (item.get("observations") or [])
            if isinstance(value, dict)
            and str(value.get("source") or "") != provenance
        ]
        observations.append(observation)
        item.update(
            {
                "datum": "SRD",
                "absolute_elevation_m": srd_elevation_m,
                "canonical_reference": "MSL",
                "positive_direction": "up",
                "source": entity_name,
                "evidence": evidence,
                "confidence": 1.0,
                "ready": True,
                "review_required": False,
                "conflicts": [],
                "observations": observations,
                "vertical_crs": dict(vertical_crs),
                "seismic_reference": {
                    "type": "SRD",
                    "elevation_msl_m": srd_elevation_m,
                    "replacement_velocity_mps": replacement_velocity_mps,
                    "time_domain": "TWT",
                    "time_reference": "SRD",
                    "correction_state": "corrected_to_srd",
                    "provenance": provenance,
                    "evidence": [evidence],
                    "contract_candidates": [],
                    "ready": True,
                },
            }
        )

    well_items = [
        item
        for item in (datum_inventory.get("wells") or [])
        if isinstance(item, dict)
    ]
    ready_wells = sum(bool(item.get("ready")) for item in well_items)
    ready_seismic = sum(bool(item.get("ready")) for item in seismic_items)
    ready_seismic_time = sum(
        bool((item.get("seismic_reference") or {}).get("ready"))
        for item in seismic_items
    )
    vertical_crs_ready = bool(str(effective.vertical_crs_id).strip())
    physical_ready = bool(
        well_items
        and seismic_items
        and vertical_crs_ready
        and ready_wells == len(well_items)
        and ready_seismic == len(seismic_items)
    )
    time_ready = bool(
        seismic_items and ready_seismic_time == len(seismic_items)
    )
    datum_inventory.update(
        {
            "ready_wells": ready_wells,
            "ready_seismic": ready_seismic,
            "ready_seismic_time": ready_seismic_time,
            "vertical_crs_ready": vertical_crs_ready,
        }
    )

    preparation = result.get("preparation")
    if not isinstance(preparation, dict):
        return
    issues = [
        item
        for item in (preparation.get("issues") or [])
        if isinstance(item, dict)
    ]
    filtered_issues: list[dict[str, Any]] = []
    for issue in issues:
        title = str(issue.get("title") or "")
        if physical_ready and title in _RUNTIME_RESOLVED_VERTICAL_TITLES:
            continue
        if time_ready and title in _RUNTIME_RESOLVED_TIME_TITLES:
            continue
        filtered_issues.append(issue)
    preparation["issues"] = filtered_issues

    for stage in preparation.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("id") or "")
        metrics = stage.get("metrics")
        if stage_id == "vertical_datum_normalization" and physical_ready:
            stage["status"] = "就绪"
            if isinstance(metrics, dict):
                metrics.update(
                    {
                        "垂向CRS": str(effective.vertical_crs_id),
                        "已确认井基准": f"{ready_wells}/{len(well_items)}",
                        "已确认地震SRD": f"{ready_seismic}/{len(seismic_items)}",
                    }
                )
        elif stage_id == "seismic_time_reference" and time_ready:
            stage["status"] = "就绪"
            if isinstance(metrics, dict):
                metrics.update(
                    {
                        "目标时间参考": "SRD",
                        "目标时间域": "TWT",
                        "已确认地震时间": (
                            f"{ready_seismic_time}/{len(seismic_items)}"
                        ),
                        "防重复校正": "corrected_to_srd不再校正",
                    }
                )
        stage["issue_count"] = sum(
            1
            for issue in filtered_issues
            if str(issue.get("stage") or "") == stage_id
            and issue.get("attention_required") is True
        )

    attention_issues = [
        item for item in filtered_issues if item.get("attention_required") is True
    ]
    summary = preparation.setdefault("summary", {})
    if isinstance(summary, dict):
        summary.update(
            {
                "blocking": sum(bool(item.get("blocking")) for item in filtered_issues),
                "warnings": sum(
                    item.get("severity") == "警告" for item in attention_issues
                ),
                "information": sum(
                    item.get("attention_required") is not True
                    for item in filtered_issues
                ),
                "attention_required": len(attention_issues),
                "audit_findings": len(filtered_issues) - len(attention_issues),
                "not_required": sum(
                    item.get("required_for_task") is False
                    for item in filtered_issues
                ),
                "autofilled": sum(
                    item.get("resolution_mode") in {"llm_autofill", "rule_autofill"}
                    for item in filtered_issues
                ),
                "survey_input_required": sum(
                    item.get("resolution_mode") == "survey_input"
                    for item in filtered_issues
                ),
            }
        )


def _derive_runtime_contract_snapshot(
    request: RuntimeContractConfirmationRequest,
) -> RuntimeContractConfirmationResponse:
    source_snapshot_id = request.source_snapshot_id
    source_effective, _source_context = _sealed_horizontal_registration_request(
        HorizontalRegistrationRequest(source_snapshot_id=source_snapshot_id)
    )
    source_task = _get_task(source_snapshot_id)
    source_manifest = _source_snapshot_manifest_from_task(source_task)
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("contract_version") != SOURCE_SNAPSHOT_CONTRACT_VERSION
    ):
        raise ValueError("运行参数确认仅支持封存的SourceSnapshot V3")
    review = source_manifest.get("runtime_contract_review") or {}
    runtime_contract_review_sha256 = str(
        (source_manifest.get("hashes") or {}).get(
            "runtime_contract_review_sha256"
        )
        or ""
    ).casefold()
    if review.get("contract_version") != _RUNTIME_CONTRACT_REVIEW_VERSION:
        raise ValueError("源快照未包含可重放的运行参数审核合同，请重新读取数据")
    if (
        len(runtime_contract_review_sha256) != 64
        or canonical_sha256(review) != runtime_contract_review_sha256
    ):
        raise ValueError("源快照运行参数审核合同身份无效")
    if review.get("required") is not True:
        raise ValueError("源快照没有待确认的运行参数")
    time_depth_asset_count = int(review.get("time_depth_asset_count") or 0)
    values = request.values.model_dump(mode="json", exclude_none=True)
    baseline_values = dict(review.get("values") or {})
    review_field_rows = [
        item for item in (review.get("fields") or []) if isinstance(item, dict)
    ]
    reviewed_fields = {
        str(item.get("key") or "")
        for item in review_field_rows
    }
    if (
        not reviewed_fields
        or "" in reviewed_fields
        or len(reviewed_fields) != len(review_field_rows)
    ):
        raise ValueError("源快照运行参数审核字段无效")
    if set(values) != set(baseline_values) or not reviewed_fields.issubset(values):
        raise ValueError("确认值必须完整提交不可变审核合同中的全部字段")
    changed_non_review_fields = sorted(
        key
        for key in set(values) - reviewed_fields
        if values.get(key) != baseline_values.get(key)
    )
    if changed_non_review_fields:
        raise ValueError(
            "非审核字段不得修改：" + ", ".join(changed_non_review_fields)
        )
    submitted_td_fields = {
        key for key in values if key.startswith("time_depth_default_")
    }
    if submitted_td_fields and time_depth_asset_count < 1:
        raise ValueError("源快照没有已解析时深资产，禁止创建时深缺省语义")

    parent_snapshot_sha256 = str(
        (source_manifest.get("hashes") or {}).get("snapshot_sha256") or ""
    ).casefold()
    if len(parent_snapshot_sha256) != 64:
        raise ValueError("父快照缺少有效snapshot_sha256")
    parent_asset_set_sha256 = str(
        (source_manifest.get("hashes") or {}).get("legacy_asset_set_sha256") or ""
    ).casefold()
    if len(parent_asset_set_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in parent_asset_set_sha256
    ):
        raise ValueError("父快照缺少有效资产集合SHA-256")
    expected_local_crs = (
        f"LOCAL_SURVEY_XY_{parent_asset_set_sha256[:12].upper()}"
    )
    horizontal_crs = str(values["horizontal_crs_id"])
    if horizontal_crs != expected_local_crs:
        try:
            require_projected_metre_crs(
                horizontal_crs,
                field="确认的水平CRS",
            )
        except CoordinateReferenceError as exc:
            raise ValueError(
                "水平CRS必须是可解析的米制投影CRS，或当前资产哈希绑定的"
                f"{expected_local_crs}"
            ) from exc
    attestation = request.attestation.model_dump(mode="json")
    user_attestation_sha256 = canonical_sha256(attestation)
    decision_basis_sha256 = canonical_sha256(
        {
            "profile_id": _RUNTIME_CONTRACT_PROFILE_ID,
            "parent_snapshot_id": source_snapshot_id,
            "parent_snapshot_sha256": parent_snapshot_sha256,
            "runtime_contract_review_sha256": runtime_contract_review_sha256,
            "confirmation": request.confirmation,
            "confirmed_values": values,
            "user_attestation_sha256": user_attestation_sha256,
        }
    )
    derived_snapshot_id = decision_basis_sha256[:32]

    raw_effective = source_effective.model_dump(mode="json")
    raw_effective.update(values)
    raw_effective.update(
        {
            "source_snapshot_id": derived_snapshot_id,
            "registration_task_id": None,
            "survey_attestation": attestation,
        }
    )
    effective = PreprocessingRequest.model_validate(raw_effective)
    _require_registration_semantic_contract(effective)
    decision = {
        "contract_version": _RUNTIME_CONTRACT_CONFIRMATION_VERSION,
        "decision_type": "human_runtime_contract_confirmation",
        "source": "human_user",
        "profile_id": _RUNTIME_CONTRACT_PROFILE_ID,
        "parent_snapshot_id": source_snapshot_id,
        "parent_snapshot_sha256": parent_snapshot_sha256,
        "runtime_contract_review_sha256": runtime_contract_review_sha256,
        "confirmed_at": attestation["confirmed_at"],
        "confirmation": request.confirmation,
        "confirmed_values": values,
        "attestation": attestation,
        "user_attestation_sha256": user_attestation_sha256,
        "decision_basis_sha256": decision_basis_sha256,
    }
    project_id = str(source_task.get("project_id") or DEFAULT_PROJECT_ID)
    _state_store.ensure_project(
        project_id,
        {"name": "本地默认项目", "kind": "desktop_local"},
    )

    with _registration_preflight_lock:
        try:
            existing = _get_task(derived_snapshot_id)
        except KeyError:
            existing = None
        if existing is not None:
            stored_decision = existing.get("runtime_contract_confirmation") or {}
            if str(stored_decision.get("decision_basis_sha256") or "") != (
                decision_basis_sha256
            ):
                raise ValueError("运行参数派生任务已被不同人工确认占用")
            if existing.get("status") == "completed":
                return _runtime_contract_confirmation_response(
                    source_snapshot_id=source_snapshot_id,
                    derived_snapshot_id=derived_snapshot_id,
                    decision_basis_sha256=decision_basis_sha256,
                )
            if existing.get("status") == "running":
                try:
                    updated_at = datetime.fromisoformat(
                        str(existing.get("updated_at") or "").replace("Z", "+00:00")
                    )
                    stale = (
                        updated_at.tzinfo is not None
                        and (datetime.now(timezone.utc) - updated_at).total_seconds()
                        >= 300
                    )
                except (TypeError, ValueError):
                    stale = False
                if not stale:
                    raise ValueError("运行参数正在封存，请稍后重试")
            try:
                _state_store.get_snapshot(derived_snapshot_id)
            except RecordNotFoundError:
                pass
            else:
                raise ValueError(
                    "派生快照已封存但任务状态异常，禁止覆盖不可变快照"
                )
            _set_task(
                derived_snapshot_id,
                status="running",
                progress=1,
                message="正在重试封存人工确认的运行参数",
                result=None,
                error=None,
                parent_task_id=source_snapshot_id,
                request=effective.model_dump(mode="json"),
                runtime_contract_confirmation=decision,
                retry_started_at=_now(),
            )
        else:
            created_at = _now()
            task = _state_store.create_task(
                {
                    "progress": 1,
                    "message": "正在封存人工确认的运行参数",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "result": None,
                    "error": None,
                    "parent_task_id": source_snapshot_id,
                    "request": effective.model_dump(mode="json"),
                    "runtime_contract_confirmation": decision,
                },
                task_id=derived_snapshot_id,
                project_id=project_id,
                task_type="data_preparation",
                status="running",
            )
            with _tasks_lock:
                _tasks[derived_snapshot_id] = task

        try:
            derived_result = copy.deepcopy(source_task.get("result") or {})
            derived_result["assets"] = _runtime_contract_replay_assets(
                source_manifest,
                effective,
            )
            _apply_confirmed_runtime_contract_to_result(
                derived_result,
                effective,
                decision,
            )
            data_snapshot = dict(derived_result.get("data_snapshot") or {})
            for key in (
                "artifact_bundle_id",
                "parent_snapshot_id",
                "snapshot_id",
                "snapshot_manifest_path",
                "snapshot_manifest_sha256",
                "snapshot_sha256",
                "survey_attestation_receipt",
                "survey_attestation_sha256",
                "system_evidence_receipt",
                "system_evidence_receipt_sha256",
                "runtime_contract_review_sha256",
                "runtime_contract_receipt",
                "runtime_contract_receipt_sha256",
            ):
                data_snapshot.pop(key, None)
            derived_result["data_snapshot"] = data_snapshot
            preparation = derived_result.setdefault("preparation", {})
            preparation["runtime_contract_review"] = {
                "contract_version": _RUNTIME_CONTRACT_REVIEW_VERSION,
                "required": False,
                "profile_id": _RUNTIME_CONTRACT_PROFILE_ID,
                "fields": [],
                "values": values,
                "time_depth_asset_count": time_depth_asset_count,
                "confirmed_snapshot_id": derived_snapshot_id,
            }
            preparation["runtime_contract_confirmation"] = {
                **decision,
                "derived_snapshot_id": derived_snapshot_id,
            }
            _seal_data_preparation_snapshot(
                derived_snapshot_id,
                effective,
                derived_result,
                project_id=project_id,
                parent_snapshot_id=source_snapshot_id,
                runtime_contract_confirmation=decision,
            )
            _set_task(
                derived_snapshot_id,
                status="completed",
                progress=100,
                message="运行参数已确认并封存",
                result=derived_result,
            )
        except Exception as exc:
            _set_task(
                derived_snapshot_id,
                status="failed",
                progress=100,
                message="运行参数确认封存失败",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        return _runtime_contract_confirmation_response(
            source_snapshot_id=source_snapshot_id,
            derived_snapshot_id=derived_snapshot_id,
            decision_basis_sha256=decision_basis_sha256,
        )


def _sealed_horizontal_registration_request(
    request: HorizontalRegistrationRequest,
) -> tuple[PreprocessingRequest, dict[str, Any]]:
    """Rehydrate exact semantics without synchronously re-hashing a large SEG-Y."""

    try:
        source_task = _get_task(request.source_snapshot_id)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail="引用的数据快照不存在") from exc
    if (
        source_task.get("task_type") != "data_preparation"
        or source_task.get("status") != "completed"
    ):
        raise HTTPException(status_code=409, detail="引用的数据准备任务尚未完成")
    raw = dict(source_task.get("request") or {})
    raw.update(
        {
            "source_snapshot_id": request.source_snapshot_id,
            "output_directory": request.output_directory,
            "registration_task_id": None,
        }
    )
    try:
        effective = PreprocessingRequest.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"封存快照缺少可重放的准备请求：{exc}",
        ) from exc
    try:
        source_manifest = _source_snapshot_manifest_from_task(source_task)
        if source_manifest is not None:
            validate_snapshot_request_semantics(
                source_manifest,
                effective,
                effective_config_sha256=_effective_configuration_sha256(),
                transformation_registry_sha256=_transformation_registry_sha256(),
            )
        data_snapshot = (source_task.get("result") or {}).get("data_snapshot") or {}
        contract_version = str(data_snapshot.get("contract_version") or "")
        if contract_version == SOURCE_SNAPSHOT_CONTRACT_VERSION:
            stored = _state_store.get_snapshot(request.source_snapshot_id)
            if stored.get("state") != "sealed":
                raise ValueError("source snapshot state is not sealed")
        snapshot_context = {
            "project_id": source_task.get("project_id")
            or data_snapshot.get("project_id"),
            "snapshot_contract_version": contract_version,
            "snapshot_assets": [
                dict(item)
                for item in ((source_task.get("result") or {}).get("assets") or [])
                if isinstance(item, dict)
            ],
        }
    except (OSError, RecordNotFoundError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"source data snapshot integrity verification failed: {exc}",
        ) from exc
    return effective, snapshot_context


def _horizontal_registration_idempotency_key(
    request: HorizontalRegistrationRequest,
    effective: PreprocessingRequest,
) -> str:
    """Bind reuse to both the sealed source and its rehydrated request."""

    return canonical_sha256(
        {
            "contract_version": "well-seismic.horizontal-registration-idempotency.v1",
            "source_snapshot_id": request.source_snapshot_id,
            "request": request.model_dump(mode="json"),
            "effective_snapshot_request": effective.model_dump(mode="json"),
        }
    )


def _completed_horizontal_registration_is_complete(task: dict[str, Any]) -> bool:
    """Reuse only an intact product whose files still match sealed hashes."""

    registration = (task.get("result") or {}).get("horizontal_registration") or {}
    output_files = registration.get("output_files") or {}
    integrity = registration.get("output_integrity") or {}
    required_files = (
        "manifest",
        "horizontal_registration_points",
        "horizontal_registration_wells",
        "horizontal_registration_plan_view",
    )
    required_integrity = (
        "horizontal_registration_manifest_sha256",
        "horizontal_registration_points_sha256",
        "horizontal_registration_wells_sha256",
        "horizontal_registration_plan_view_sha256",
        "horizontal_registration_product_sha256",
    )
    if not all(
        isinstance(output_files.get(name), str)
        and Path(str(output_files[name])).is_file()
        for name in required_files
    ) or not all(
        isinstance(integrity.get(name), str) and len(str(integrity[name])) == 64
        for name in required_integrity
    ):
        return False
    try:
        observed = {
            name: file_sha256(Path(str(output_files[name]))) for name in required_files
        }
        expected_by_file = {
            "manifest": "horizontal_registration_manifest_sha256",
            "horizontal_registration_points": "horizontal_registration_points_sha256",
            "horizontal_registration_wells": "horizontal_registration_wells_sha256",
            "horizontal_registration_plan_view": "horizontal_registration_plan_view_sha256",
        }
        if any(
            observed[name].casefold() != str(integrity[expected]).casefold()
            for name, expected in expected_by_file.items()
        ):
            return False
        manifest = json.loads(
            Path(str(output_files["manifest"])).read_text(encoding="utf-8")
        )
        manifest_integrity = manifest.get("output_integrity") or {}
        manifest_hashes = {
            "horizontal_registration_points": "horizontal_registration_points_sha256",
            "horizontal_registration_wells": "horizontal_registration_wells_sha256",
            "horizontal_registration_plan_view": "horizontal_registration_plan_view_sha256",
        }
        if any(
            observed[name].casefold()
            != str(manifest_integrity.get(expected) or "").casefold()
            for name, expected in manifest_hashes.items()
        ):
            return False
        return (
            str(integrity["horizontal_registration_product_sha256"]).casefold()
            == str(
                manifest.get("horizontal_registration_product_sha256") or ""
            ).casefold()
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _reusable_horizontal_registration_task(
    *,
    source_snapshot_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    try:
        candidates = _state_store.list_tasks(
            snapshot_id=source_snapshot_id,
            task_type="horizontal_registration",
            limit=200,
        )
    except (StateStoreError, TypeError, ValueError):
        return None
    for task in candidates:
        if task.get("horizontal_registration_idempotency_key") != idempotency_key:
            continue
        status = str(task.get("status") or "")
        if status in {"queued", "running"}:
            return task
        if status == "completed" and _completed_horizontal_registration_is_complete(
            task
        ):
            return task
    return None


def _workflow_submission_idempotency_key(
    task_type: str,
    request: PreprocessingRequest,
) -> str:
    """Make duplicate clicks deterministic without weakening lineage binding."""

    return canonical_sha256(
        {
            "contract_version": "well-seismic.workflow-submission-idempotency.v1",
            "task_type": task_type,
            "source_snapshot_id": request.source_snapshot_id,
            "registration_task_id": request.registration_task_id,
            "request": request.model_dump(mode="json"),
        }
    )


_LEGACY_LEARNED_REGISTRATION_AUTHORITY_TOKENS = (
    "p13",
    "geopath",
    "wellfuse_align",
)


def _legacy_learned_fusion_track_ids(product: Any) -> list[str]:
    """Identify learned tracks stored in the legacy primary V3 product."""

    learned: list[str] = []
    for identity, track in product.tracks.items():
        if not (
            bool(track.get("fusionReady"))
            and bool(track.get("inferenceEligible", True))
        ):
            continue
        authority_text = " ".join(
            str(track.get(field) or "")
            for field in (
                "registrationSource",
                "sourceAuthority",
                "registrationStatus",
                "registrationAuthority",
                "uncertaintySource",
            )
        ).casefold()
        if any(
            token in authority_text
            for token in _LEGACY_LEARNED_REGISTRATION_AUTHORITY_TOKENS
        ):
            learned.append(str(identity))
    return learned


def _is_human_accepted_geopath_track_set(
    product: Any,
    learned_fusion_well_ids: list[str],
) -> bool:
    return bool(learned_fusion_well_ids) and all(
        str(product.tracks[identity].get("registrationSource") or "")
        == "wellfuse_align_geopath_tie_v1_human_accepted"
        and str(product.tracks[identity].get("sourceAuthority") or "")
        == "learned_geopath_human_accepted"
        and str(product.tracks[identity].get("registrationStatus") or "")
        == "human_accepted_candidate"
        and not bool(product.tracks[identity].get("supervisionEligible"))
        and not bool(product.tracks[identity].get("trainingEligible"))
        for identity in learned_fusion_well_ids
    )


def _completed_registration_is_complete(task: dict[str, Any]) -> bool:
    """Validate the sealed Registration V3 product before reusing a task."""

    registration = (task.get("result") or {}).get("registration") or {}
    output_files = registration.get("output_files") or {}
    integrity = registration.get("output_integrity") or {}
    try:
        manifest_path = Path(str(output_files.get("manifest") or "")).resolve()
        points_path = Path(str(output_files.get("registration_points") or "")).resolve()
        preview_path = Path(
            str(output_files.get("registration_preview") or "")
        ).resolve()
        product = read_registration_product_v3(manifest_path)
        if (
            product.points_path != points_path
            or product.preview_path != preview_path
            or file_sha256(manifest_path)
            != str(integrity.get("registration_manifest_sha256") or "")
            or file_sha256(points_path)
            != str(integrity.get("registration_points_sha256") or "")
            or file_sha256(preview_path)
            != str(integrity.get("registration_preview_sha256") or "")
            or str(product.manifest.get("registration_product_sha256") or "")
            != str(integrity.get("registration_product_sha256") or "")
        ):
            return False
        fusion = registration.get("fusion_consumption_product") or {}
        if fusion.get("state") != "available":
            learned_fusion_well_ids = _legacy_learned_fusion_track_ids(product)
            if not learned_fusion_well_ids:
                return True
            if not _is_human_accepted_geopath_track_set(
                product,
                learned_fusion_well_ids,
            ):
                return False
            registration_task_id = str(task.get("task_id") or "")
            if not registration_task_id:
                return False
            _validate_human_accepted_geopath_lineage(
                task=task,
                registration_task_id=registration_task_id,
                registration=registration,
                manifest=product.manifest,
                product=product,
                learned_fusion_well_ids=learned_fusion_well_ids,
                lineage_stack=frozenset({registration_task_id}),
            )
            return True
        fusion_files = fusion.get("output_files") or {}
        fusion_integrity = fusion.get("output_integrity") or {}
        fusion_manifest = Path(
            str(fusion_files.get("fusion_consumption_manifest") or "")
        ).resolve()
        fusion_points = Path(
            str(fusion_files.get("fusion_consumption_registration_points") or "")
        ).resolve()
        fusion_preview = Path(
            str(fusion_files.get("fusion_consumption_registration_preview") or "")
        ).resolve()
        fusion_product = read_registration_product_v3(fusion_manifest)
        return (
            fusion_product.points_path == fusion_points
            and fusion_product.preview_path == fusion_preview
            and file_sha256(fusion_manifest)
            == str(
                fusion_integrity.get("fusion_consumption_registration_manifest_sha256")
                or ""
            )
            and file_sha256(fusion_points)
            == str(
                fusion_integrity.get("fusion_consumption_registration_points_sha256")
                or ""
            )
            and file_sha256(fusion_preview)
            == str(
                fusion_integrity.get("fusion_consumption_registration_preview_sha256")
                or ""
            )
            and str(fusion_product.manifest.get("registration_product_sha256") or "")
            == str(
                fusion_integrity.get("fusion_consumption_registration_product_sha256")
                or ""
            )
        )
    except (OSError, TypeError, ValueError, KeyError, StateStoreError):
        return False


def _completed_sample_building_is_complete(task: dict[str, Any]) -> bool:
    """PreparedView validation re-hashes all declared derivative artifacts."""

    request = task.get("request") or {}
    prepared = (task.get("result") or {}).get("prepared_view") or {}
    manifest_path = prepared.get("manifest_path")
    if not manifest_path or prepared.get("state") != "ready":
        return False
    try:
        validated = validate_prepared_view_manifest(
            manifest_path,
            expected_view_id=str(task.get("task_id") or ""),
            expected_source_snapshot_id=request.get("source_snapshot_id"),
        )
    except (OSError, TypeError, ValueError):
        return False
    return (
        str(validated.get("manifest_sha256") or "")
        == str(prepared.get("manifest_sha256") or "")
        and str(validated.get("view_sha256") or "")
        == str(prepared.get("view_sha256") or "")
        and str(validated.get("producer_task_id") or "")
        == str(task.get("task_id") or "")
    )


def _reusable_workflow_task(
    *,
    task_type: str,
    source_snapshot_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    try:
        candidates = _state_store.list_tasks(
            snapshot_id=source_snapshot_id,
            task_type=task_type,
            limit=200,
        )
    except (StateStoreError, TypeError, ValueError):
        return None
    complete = (
        _completed_registration_is_complete
        if task_type == "well_tie"
        else _completed_sample_building_is_complete
    )
    for task in candidates:
        if task.get("workflow_submission_idempotency_key") != idempotency_key:
            continue
        status = str(task.get("status") or "")
        if status in {"queued", "running"}:
            return task
        if status == "completed" and complete(task):
            return task
    return None


def _requested_output_directory(request: PreprocessingRequest) -> Path | None:
    raw = str(request.output_directory or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _task_output_directory(task: dict[str, Any]) -> Path | None:
    request = task.get("request") or {}
    raw = str(request.get("output_directory") or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None


def _assert_no_workflow_output_directory_conflict(
    request: PreprocessingRequest,
    *,
    idempotency_key: str,
) -> None:
    requested = _requested_output_directory(request)
    if requested is None:
        return
    try:
        candidates = _state_store.list_tasks(limit=10000)
    except (StateStoreError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="无法核验输出目录占用") from exc
    for task in candidates:
        if task.get("task_type") not in {"well_tie", "sample_building"}:
            continue
        if _task_output_directory(task) != requested:
            continue
        raise HTTPException(
            status_code=409,
            detail=(
                f"输出目录已被{task.get('task_type')}任务{task.get('task_id')}占用；"
                "为避免覆盖封存产品，已拒绝提交"
            ),
        )


def _require_horizontal_registration_contract(
    request: PreprocessingRequest,
    snapshot_context: dict[str, Any],
) -> None:
    """Require only plan-view facts; keep every vertical/TWT fact out of scope."""

    issues: list[str] = []
    if request.coordinate_reference_verified is not True:
        issues.append("coordinate_reference_verified必须为true")
    if not str(request.horizontal_crs_id or "").strip():
        issues.append("horizontal_crs_id未声明")
    if request.horizontal_unit != "m":
        issues.append("horizontal_unit必须为m")
    if request.horizontal_axis_order not in {"XY", "YX"}:
        issues.append("horizontal_axis_order必须明确为XY或YX")
    if request.time_depth_paths:
        issues.append("无时深水平配准快照不得登记time_depth_paths")
    roles = Counter(
        str(item.get("role") or "")
        for item in snapshot_context.get("snapshot_assets") or []
        if isinstance(item, dict)
    )
    if roles.get("time_depth", 0):
        issues.append("封存快照包含time_depth资产")
    if roles.get("seismic", 0) <= 0:
        issues.append("封存快照缺少SEG-Y地震资产")
    if roles.get("well_logs", 0) <= 0:
        issues.append("封存快照缺少LAS测井资产")
    if issues:
        raise ValueError("无时深水平配准合同不完整：" + "；".join(issues))


def horizontal_register_paths(
    request: PreprocessingRequest,
    *,
    task_id: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Generate a plan-view derivative without invoking vertical registration."""

    source_snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
    _require_horizontal_registration_contract(request, source_snapshot_context)
    if progress:
        progress(8, "正在锁定无时深快照与水平坐标合同")
    manifest, inventory = build_explicit_paths_manifest(
        seismic_directory=request.seismic_paths,
        log_directory=request.log_paths,
        metadata_directory=request.well_paths,
        auxiliary_directory=request.auxiliary_paths,
        survey_directory=request.survey_paths,
        interpretation_directory=request.interpretation_paths,
        time_depth_directory=request.time_depth_paths,
        recursive=request.recursive,
        seismic_srd_elevation_m=request.seismic_srd_elevation_m,
        vertical_crs_id=request.vertical_crs_id,
        horizontal_crs_id=request.horizontal_crs_id,
        well_source_crs_id=request.well_source_crs_id,
        seismic_source_crs_id=request.seismic_source_crs_id,
        horizontal_unit=request.horizontal_unit,
        horizontal_axis_order=request.horizontal_axis_order,
        coordinate_reference_verified=request.coordinate_reference_verified,
        seismic_replacement_velocity_mps=request.seismic_replacement_velocity_mps,
        seismic_time_domain=request.seismic_time_domain,
        seismic_correction_state=request.seismic_correction_state,
        segy_geometry_profile=request.segy_geometry_profile,
        segy_inline_byte=request.segy_inline_byte,
        segy_crossline_byte=request.segy_crossline_byte,
        segy_x_byte=request.segy_x_byte,
        segy_y_byte=request.segy_y_byte,
        segy_coordinate_scalar_byte=request.segy_coordinate_scalar_byte,
        well_coordinate_source_unit=request.well_coordinate_source_unit,
        well_vertical_datum_source_unit=request.well_vertical_datum_source_unit,
        las_twt_source_unit=request.las_twt_source_unit,
        time_depth_default_depth_domain=request.time_depth_default_depth_domain,
        time_depth_default_depth_unit=request.time_depth_default_depth_unit,
        time_depth_default_time_unit=request.time_depth_default_time_unit,
        time_depth_default_depth_datum=request.time_depth_default_depth_datum,
        time_depth_default_depth_convention=request.time_depth_default_depth_convention,
        time_depth_default_time_reference=request.time_depth_default_time_reference,
        time_depth_default_time_domain=request.time_depth_default_time_domain,
        time_depth_default_correction_state=request.time_depth_default_correction_state,
        target_task_id=request.target_task_id,
        target_model_id=request.target_model_id,
        target_required_modalities=_target_required_modalities(request),
        target_model_contract=_target_model_contract(request),
    )
    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    _apply_horizontal_coordinate_contract(pipeline, request)
    _apply_snapshot_parse_repairs(pipeline, source_snapshot_context)
    _assert_pipeline_uses_sealed_assets(pipeline, source_snapshot_context)
    pipeline.automatic_inventory = inventory
    if progress:
        progress(25, f"已锁定 {len(pipeline.assets)} 个封存资产，正在校验几何缓存")

    def ingest_progress(
        done: int,
        total: int,
        asset: Any | None,
        subwork_done: int = 0,
        subwork_total: int = 0,
    ) -> None:
        if not progress:
            return
        fractional_done = float(done)
        if asset is not None and subwork_total > 0:
            fractional_done += min(1.0, float(subwork_done) / subwork_total)
        percent = 25 + round(32 * fractional_done / total) if total else 57
        current = Path(asset.path).name if asset is not None else ""
        if subwork_total > 0:
            message = f"正在扫描SEG-Y道头 {subwork_done:,}/{subwork_total:,}" + (
                f"：{current}" if current else ""
            )
        else:
            message = f"正在读取水平配准资产 {min(done + 1, total)}/{total}"
            if current:
                message += f"：{current}"
        progress(percent, message)

    def cache_status(cache_hit: bool, _reason: str) -> None:
        if not progress:
            return
        progress(
            25,
            (
                "正在复用封存SourceSnapshot：资产身份已校验，已加载封存SEG-Y几何；正在快速重放LAS/DEV"
                if cache_hit
                else "封存SourceSnapshot首次建立安全几何缓存：正在扫描SEG-Y道头与完整DEV轨迹"
            ),
        )

    geometry_cache_receipt = _ingest_pipeline_from_sealed_snapshot(
        pipeline,
        source_snapshot_context,
        progress=ingest_progress if progress else None,
        cache_status=cache_status,
    )
    if progress:
        progress(62, "正在计算每个轨迹站点的最近道与网格覆盖")

    def match_progress(done: int, total: int, message: str) -> None:
        if progress:
            fraction = done / total if total else 1.0
            progress(62 + round(25 * fraction), message)

    horizontal = build_horizontal_registration(
        pipeline,
        source_snapshot_id=str(request.source_snapshot_id),
        source_snapshot_fingerprint=source_snapshot_context.get(
            "source_snapshot_fingerprint"
        ),
        horizontal_crs_id=str(request.horizontal_crs_id),
        horizontal_unit=request.horizontal_unit,
        horizontal_axis_order=request.horizontal_axis_order,
        progress=match_progress,
    )
    output_directory = (
        Path(request.output_directory).expanduser().resolve()
        if request.output_directory and request.output_directory.strip()
        else PROJECT_ROOT / "输出结果" / f"水平配准_{task_id[:8]}"
    )
    if progress:
        progress(91, "正在封存水平配准清单、全分辨率点表与平面可视化数据")
    # The producer task id is part of the on-disk manifest, not merely the
    # mutable API envelope, so a completed product cannot be rebound to a
    # different task during reuse.
    horizontal["horizontal_registration_id"] = task_id
    product = write_horizontal_registration_product(output_directory, horizontal)
    output_integrity = {
        "horizontal_registration_manifest_sha256": product.manifest_sha256,
        "horizontal_registration_points_sha256": product.points_sha256,
        "horizontal_registration_wells_sha256": product.wells_sha256,
        "horizontal_registration_plan_view_sha256": product.visualization_sha256,
        "horizontal_registration_product_sha256": product.product_sha256,
    }
    lineage_sha256 = canonical_sha256(
        {
            "horizontal_registration_id": task_id,
            "source_snapshot_id": request.source_snapshot_id,
            "source_snapshot_fingerprint": source_snapshot_context.get(
                "source_snapshot_fingerprint"
            ),
            **output_integrity,
        }
    )
    _assert_snapshot_verified_stat_signatures(source_snapshot_context)
    return {
        "summary": dict(horizontal["summary"]),
        "data_snapshot": {
            "snapshot_id": request.source_snapshot_id,
            "project_id": source_snapshot_context.get("project_id"),
            "snapshot_contract_version": source_snapshot_context.get(
                "snapshot_contract_version"
            ),
            "source_snapshot_fingerprint": source_snapshot_context.get(
                "source_snapshot_fingerprint"
            ),
            "integrity_status": source_snapshot_context.get(
                "snapshot_integrity_status"
            ),
            "segy_geometry_cache": geometry_cache_receipt,
            "relationship": "read_only_derived_view",
        },
        "horizontal_registration": {
            **product.manifest,
            "horizontal_registration_id": task_id,
            "can_build_multimodal_view": False,
            "fusion_ready": False,
            "training_eligible": False,
            "output_directory": str(output_directory),
            "output_integrity": output_integrity,
            "horizontal_registration_lineage_sha256": lineage_sha256,
            "output_files": {
                "manifest": str(product.manifest_path),
                "horizontal_registration_points": str(product.points_path),
                "horizontal_registration_wells": str(product.wells_path),
                "horizontal_registration_plan_view": str(product.visualization_path),
            },
        },
    }


def _filter_formal_tracks_with_unresolved_well_datums(
    pipeline: WellSeismicPipeline,
    tracks: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Fail soft before sealing formal tracks with ambiguous vertical datums.

    A relative sonic candidate may be useful on the SEG-Y sample axis even
    when a well's absolute KB/DF/RT elevation is conflicting.  It must not,
    however, become a formal Registration V3 track because that product writes
    canonical ``z_msl_m`` and ``depth_below_srd_m``.  Horizontal registration
    and source/downstream inputs do not call this filter.
    """

    entities = list(pipeline.registry.entities.values())
    retained: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for identity, track in sorted(
        tracks.items(), key=lambda item: str(item[0]).casefold()
    ):
        identity_candidates = {
            str(value).strip().casefold()
            for value in (
                identity,
                track.get("well_uid"),
                track.get("well_name"),
            )
            if str(value or "").strip()
        }
        matches = []
        for entity in entities:
            entity_candidates = {
                str(value).strip().casefold()
                for value in (
                    entity.well_uid,
                    entity.canonical_name,
                    *entity.aliases,
                )
                if str(value or "").strip()
            }
            if identity_candidates & entity_candidates:
                matches.append(entity)

        datum = None
        datum_error = None
        if len(matches) == 1:
            try:
                datum = pipeline._well_datum(matches[0])
            except (AttributeError, TypeError, ValueError) as exc:
                datum_error = f"{type(exc).__name__}: {exc}"

        if len(matches) == 1 and datum is not None and datum.ready:
            retained[identity] = track
            continue

        entity = matches[0] if len(matches) == 1 else None
        observations = tuple(getattr(datum, "observations", ()) or ())
        skipped.append(
            {
                "well_uid": str(
                    track.get("well_uid")
                    or (None if entity is None else entity.well_uid)
                    or ""
                ),
                "well_name": str(
                    track.get("well_name")
                    or (None if entity is None else entity.canonical_name)
                    or identity
                ),
                "reason": (
                    "formal_well_vertical_datum_unresolved_or_conflicting"
                    if len(matches) == 1
                    else "formal_registration_track_identity_unresolved"
                ),
                "datum": None if datum is None else datum.datum,
                "absolute_elevation_m": (
                    None if datum is None else datum.absolute_elevation_m
                ),
                "datum_ready": bool(datum is not None and datum.ready),
                "datum_error": datum_error,
                "conflicts": list(getattr(datum, "conflicts", ()) or ()),
                "evidence_sources": sorted(
                    {
                        str(getattr(observation, "source", "") or "")
                        for observation in observations
                        if str(getattr(observation, "source", "") or "").strip()
                    }
                ),
                "policy": (
                    "relative sonic evidence is retained in the task audit but "
                    "cannot be sealed as formal z_msl/depth_below_srd geometry"
                ),
            }
        )
    return retained, skipped


def _registration_v3_tracks(
    pipeline: WellSeismicPipeline,
    tracks: list[dict[str, Any]],
    request: PreprocessingRequest,
    *,
    execution_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach canonical datum/time semantics without altering the selected TWT."""

    execution = dict(execution_contract or {})
    absolute_reference_ready = bool(execution.get("absolute_reference_ready", True))
    vertical_crs_id = str(
        execution.get("vertical_crs_id") or request.vertical_crs_id or ""
    )
    seismic_srd_elevation_m = execution.get(
        "seismic_srd_elevation_m",
        request.seismic_srd_elevation_m,
    )
    time_domain = str(execution.get("time_domain") or request.seismic_time_domain)
    time_axis_ready = bool(execution.get("time_axis_ready", time_domain == "TWT"))
    time_reference = str(execution.get("time_reference") or "SRD")
    correction_state = str(
        execution.get("correction_state") or request.seismic_correction_state
    )
    entities = list(pipeline.registry.entities.values())
    by_uid = {str(entity.well_uid).casefold(): entity for entity in entities}
    by_name = {str(entity.canonical_name).casefold(): entity for entity in entities}
    enriched: list[dict[str, Any]] = []
    for raw_track in tracks:
        track = dict(raw_track)
        entity = by_uid.get(str(track.get("well_uid") or "").casefold())
        if entity is None:
            entity = by_name.get(str(track.get("well_name") or "").casefold())
        count = len(track.get("md") or ())
        if count < 2:
            raise ValueError("selected registration track has fewer than two MD rows")
        tvd = np.asarray(track.get("tvd", np.full(count, np.nan)), dtype=float)
        if tvd.shape != (count,):
            raise ValueError("Registration V3 TVD column length differs from MD")
        well_datum = pipeline._well_datum(entity) if entity is not None else None
        reference_elevation = (
            None if well_datum is None else well_datum.absolute_elevation_m
        )
        z_msl = np.asarray(track.get("zMsl", np.full(count, np.nan)), dtype=float)
        if z_msl.shape != (count,):
            raise ValueError("Registration V3 z_msl column length differs from MD")
        if reference_elevation is not None:
            derived_z = float(reference_elevation) - tvd
            z_msl = np.where(np.isfinite(z_msl), z_msl, derived_z)
        depth_srd = np.asarray(
            track.get("depthBelowSrd", np.full(count, np.nan)), dtype=float
        )
        if depth_srd.shape != (count,):
            raise ValueError(
                "Registration V3 depth_below_srd column length differs from MD"
            )
        if seismic_srd_elevation_m is not None:
            derived_depth_srd = float(seismic_srd_elevation_m) - z_msl
            depth_srd = np.where(np.isfinite(depth_srd), depth_srd, derived_depth_srd)
        md = np.asarray(track.get("md"), dtype=float)
        x = np.asarray(track.get("x", np.full(count, np.nan)), dtype=float)
        y = np.asarray(track.get("y", np.full(count, np.nan)), dtype=float)
        twt = np.asarray(track.get("twtMean", np.full(count, np.nan)), dtype=float)
        explicit_valid = track.get("validMask")
        valid_mask = (
            np.asarray(explicit_valid, dtype=bool)
            if explicit_valid is not None
            else np.isfinite(md) & np.isfinite(x) & np.isfinite(y) & np.isfinite(twt)
        )
        if valid_mask.shape != (count,):
            raise ValueError("Registration V3 valid_mask column length differs from MD")
        canonical_vertical_ready = bool(
            np.any(valid_mask)
            and np.all(np.isfinite(tvd[valid_mask]))
            and np.all(np.isfinite(z_msl[valid_mask]))
            and (
                not absolute_reference_ready
                or np.all(np.isfinite(depth_srd[valid_mask]))
            )
        )
        requested_fusion_ready = bool(
            track.get("fusionReady", track.get("trainingEligible", False))
        )
        if requested_fusion_ready and (
            not canonical_vertical_ready or not time_axis_ready
        ):
            diagnostics = dict(track.get("diagnostics") or {})
            diagnostics["registration_v3_fusion_demotion"] = (
                "native seismic time axis is unresolved"
                if not time_axis_ready
                else "canonical TVD/z_msl/depth_below_srd is incomplete"
            )
            track["diagnostics"] = diagnostics
            track["fusionReady"] = False
            track["supervisionEligible"] = False
            track["trainingEligible"] = False
        track.update(
            {
                "zMsl": z_msl.tolist(),
                "depthBelowSrd": depth_srd.tolist(),
                "validMask": valid_mask.tolist(),
                "registrationCoverage": float(np.mean(valid_mask)),
                "wellDepthDatum": (
                    track.get("wellDepthDatum")
                    or (None if well_datum is None else well_datum.datum)
                ),
                "wellReferenceElevationM": (
                    track.get("wellReferenceElevationM", reference_elevation)
                ),
                "horizontalCrsId": request.horizontal_crs_id,
                "horizontalUnit": request.horizontal_unit,
                "horizontalAxisOrder": request.horizontal_axis_order,
                "verticalCrsId": vertical_crs_id,
                "seismicSrdElevationM": seismic_srd_elevation_m,
                "absoluteReferenceReady": absolute_reference_ready,
                "timeDomain": time_domain,
                "timeAxisReady": time_axis_ready,
                "timeReference": time_reference,
                "correctionState": correction_state,
            }
        )
        if not absolute_reference_ready:
            track["supervisionEligible"] = False
            track["trainingEligible"] = False
            diagnostics = dict(track.get("diagnostics") or {})
            diagnostics["native_relative_contract"] = {
                "contract_version": execution.get("contract_version"),
                "contract_sha256": execution.get("contract_sha256"),
                "absolute_reference_ready": False,
                "time_depth_supervision_is_model_input": False,
            }
            track["diagnostics"] = diagnostics
        enriched.append(track)
    return enriched


def calibrate_paths(
    request: PreprocessingRequest,
    *,
    task_id: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Run XY registration and vertical well tie without building fusion samples."""
    if not request.seismic_paths or not request.log_paths:
        raise ValueError("井震标定至少需要地震数据和LAS测井数据")
    try:
        _require_registration_semantic_contract(request)
    except ValueError as formal_exc:
        try:
            source_task = _validate_source_snapshot(request)
            data_snapshot = (source_task.get("result") or {}).get("data_snapshot") or {}
            if str(data_snapshot.get("contract_version") or "") != (
                SOURCE_SNAPSHOT_CONTRACT_VERSION
            ):
                raise ValueError(
                    "native-relative模式仅适用于可验证的封存SourceSnapshot"
                )
            execution_contract = _native_relative_registration_contract(
                request,
                source_task,
            )
        except (
            HTTPException,
            KeyError,
            OSError,
            StateStoreError,
            TypeError,
            ValueError,
        ) as native_exc:
            raise formal_exc from native_exc
    else:
        execution_contract = {
            "mode": "absolute_reference",
            "absolute_reference_ready": True,
            "vertical_crs_id": request.vertical_crs_id,
            "seismic_srd_elevation_m": request.seismic_srd_elevation_m,
            "time_domain": request.seismic_time_domain,
            "time_reference": "SRD",
            "correction_state": request.seismic_correction_state,
            "time_depth_supervision_is_model_input": False,
        }
    source_snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
    if progress:
        progress(8, "正在锁定数据快照与井轨迹几何")
    manifest, inventory = build_explicit_paths_manifest(
        seismic_directory=request.seismic_paths,
        log_directory=request.log_paths,
        metadata_directory=request.well_paths,
        auxiliary_directory=request.auxiliary_paths,
        survey_directory=request.survey_paths,
        interpretation_directory=request.interpretation_paths,
        time_depth_directory=request.time_depth_paths,
        recursive=request.recursive,
        seismic_srd_elevation_m=request.seismic_srd_elevation_m,
        vertical_crs_id=request.vertical_crs_id,
        horizontal_crs_id=request.horizontal_crs_id,
        well_source_crs_id=request.well_source_crs_id,
        seismic_source_crs_id=request.seismic_source_crs_id,
        horizontal_unit=request.horizontal_unit,
        horizontal_axis_order=request.horizontal_axis_order,
        coordinate_reference_verified=request.coordinate_reference_verified,
        seismic_replacement_velocity_mps=request.seismic_replacement_velocity_mps,
        seismic_time_domain=request.seismic_time_domain,
        seismic_correction_state=request.seismic_correction_state,
        segy_geometry_profile=request.segy_geometry_profile,
        segy_inline_byte=request.segy_inline_byte,
        segy_crossline_byte=request.segy_crossline_byte,
        segy_x_byte=request.segy_x_byte,
        segy_y_byte=request.segy_y_byte,
        segy_coordinate_scalar_byte=request.segy_coordinate_scalar_byte,
        well_coordinate_source_unit=request.well_coordinate_source_unit,
        well_vertical_datum_source_unit=request.well_vertical_datum_source_unit,
        las_twt_source_unit=request.las_twt_source_unit,
        time_depth_default_depth_domain=request.time_depth_default_depth_domain,
        time_depth_default_depth_unit=request.time_depth_default_depth_unit,
        time_depth_default_time_unit=request.time_depth_default_time_unit,
        time_depth_default_depth_datum=request.time_depth_default_depth_datum,
        time_depth_default_depth_convention=request.time_depth_default_depth_convention,
        time_depth_default_time_reference=request.time_depth_default_time_reference,
        time_depth_default_time_domain=request.time_depth_default_time_domain,
        time_depth_default_correction_state=request.time_depth_default_correction_state,
        target_task_id=request.target_task_id,
        target_model_id=request.target_model_id,
        target_required_modalities=(
            _interpretation_registry.get(request.target_task_id).required_modalities
            if request.target_task_id
            else ()
        ),
        target_model_contract=_target_model_contract(request),
    )
    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    if execution_contract.get("mode") == "native_relative_no_time_depth":
        pipeline.manifest["vertical_crs"] = {
            "id": execution_contract["vertical_crs_id"],
            "unit": "m",
            "axis": "elevation_positive_up",
            "source": "sealed_native_relative_execution_contract",
        }
    _apply_horizontal_coordinate_contract(pipeline, request)
    _apply_snapshot_parse_repairs(pipeline, source_snapshot_context)
    _assert_pipeline_uses_sealed_assets(pipeline, source_snapshot_context)
    pipeline.automatic_inventory = inventory
    if progress:
        progress(25, f"已锁定 {len(pipeline.assets)} 个封存资产，正在校验几何缓存")

    def cache_status(cache_hit: bool, _reason: str) -> None:
        if not progress:
            return
        progress(
            25,
            (
                "正在复用封存SourceSnapshot：资产身份已校验，已加载封存SEG-Y几何；正在快速重放LAS/DEV"
                if cache_hit
                else "封存SourceSnapshot首次建立安全几何缓存：正在扫描SEG-Y道头并重放LAS/DEV"
            ),
        )

    geometry_cache_receipt = _ingest_pipeline_from_sealed_snapshot(
        pipeline,
        source_snapshot_context,
        cache_status=cache_status,
    )
    output_directory = (
        Path(request.output_directory).expanduser().resolve()
        if request.output_directory and request.output_directory.strip()
        else PROJECT_ROOT / "输出结果" / f"井震标定_{task_id[:8]}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Establish every available physical tie first. P13 is evaluated as an
    # independent learned proposal and can never suppress provided checkshot/
    # VSP or sonic evidence merely because its inference gate passed.
    if progress:
        progress(28, "正在执行checkshot/VSP与声波物理标定")
    alignment_attempts = pipeline.calibrate_wells()
    physical_tracks = dict(pipeline.registration_tracks)

    # The P13 request is still label-free: time-depth/checkshot assets are
    # never serialized into its subprocess.
    p13_runtime = run_p13_registration_candidates(
        pipeline,
        project_root=PROJECT_ROOT,
        output_directory=output_directory,
        cache_directory=(
            PROJECT_ROOT / "model_outputs" / "p13_registration_cache" / task_id
        ),
        source_snapshot_context=source_snapshot_context,
        progress=progress,
    )
    if progress:
        progress(58, "正在按实测时深、声波积分与冻结概率模型的证据优先级仲裁标定结果")
    _arbitrated_tracks, arbitration_decisions = arbitrate_registration_tracks(
        physical_tracks,
        p13_runtime["tracks"],
    )
    # Arbitration remains an audit explaining which candidate would win the
    # legacy single-track view.  The formal physical-primary product itself is
    # built only from observed/physics-derived tracks, never from P13.
    (
        physical_primary_tracks,
        formal_datum_skipped_wells,
    ) = _filter_formal_tracks_with_unresolved_well_datums(
        pipeline,
        physical_tracks,
    )
    pipeline.registration_tracks = physical_primary_tracks
    p13_attempts: list[dict[str, Any]] = []
    for record in p13_runtime["records"]:
        accepted = bool(record.get("raw_candidate_accepted"))
        p13_attempts.append(
            {
                "well_uid": record["well_uid"],
                "well_name": record["well_name"],
                "log_source": record.get("request"),
                "seismic_source": None,
                "status": ("estimated_tie" if accepted else "p13_candidate_rejected"),
                "method": record.get(
                    "registration_source", "wellfuse_align_p13_prediction"
                ),
                "confidence": float(record.get("median_quality") or 0.0),
                "uncertainty_ms": record.get("median_twt_std_ms"),
                "inference_eligible": bool(record.get("inference_ready")),
                "fusion_ready": bool(record.get("fusion_ready")),
                "training_eligible": False,
                "depth_domain": "md",
                "raw_candidate_accepted": accepted,
                "rejection_reason": record.get("rejection_reason"),
                "physics_sanity": record.get("physics_sanity", {}),
                "candidate_manifest": record.get("manifest"),
            }
        )
    for failure in p13_runtime["failures"]:
        if not failure.get("well_uid"):
            continue
        p13_attempts.append(
            {
                "well_uid": failure["well_uid"],
                "well_name": failure.get("well_name"),
                "log_source": failure.get("request"),
                "seismic_source": None,
                "status": "p13_runtime_failed",
                "method": "wellfuse_align_p13_prediction",
                "confidence": 0.0,
                "uncertainty_ms": None,
                "training_eligible": False,
                "depth_domain": "md",
                "error": failure.get("error"),
            }
        )
    alignment_attempts.extend(p13_attempts)
    pipeline.well_ties.extend(p13_attempts)
    tracks = list(physical_primary_tracks.values())
    # One well can legitimately have both LAS and a specialist acoustic file.
    # Keep every attempt in the detailed audit, but report business counts once
    # per well using its strongest legal result.
    strongest_by_well: dict[str, dict[str, Any]] = {}
    usable_statuses = {"provided_tie", "estimated_tie", "vertical_initial"}
    for item in alignment_attempts:
        key = str(
            item.get("well_uid") or item.get("well_name") or item.get("log_source")
        )
        score = (
            str(item.get("status")) in usable_statuses,
            registration_evidence_priority(item),
            bool(item.get("training_eligible")),
            float(item.get("confidence") or 0.0),
        )
        current = strongest_by_well.get(key)
        current_score = (
            (
                str(current.get("status")) in usable_statuses,
                registration_evidence_priority(current),
                bool(current.get("training_eligible")),
                float(current.get("confidence") or 0.0),
            )
            if current is not None
            else (False, -1, False, -1.0)
        )
        if score > current_score:
            strongest_by_well[key] = item
    ties = list(strongest_by_well.values())
    status_counts = Counter(str(item.get("status", "unregistered")) for item in ties)
    method_counts = Counter(str(item.get("method", "none")) for item in ties)
    registered_count = len(tracks)
    inference_ready_count = sum(
        bool(item.get("inferenceEligible", True)) for item in tracks
    )
    fusion_ready_count = sum(
        bool(item.get("fusionReady", item.get("trainingEligible", False)))
        for item in tracks
    )
    candidate_count = registered_count - fusion_ready_count
    blocked_count = max(0, len(ties) - registered_count)
    if registered_count == len(ties) and ties:
        business_status = "usable"
    elif registered_count:
        business_status = "partially_usable"
    else:
        business_status = "blocked"

    tracks = _registration_v3_tracks(
        pipeline,
        tracks,
        request,
        execution_contract=execution_contract,
    )
    # Keep the physical authority product immutable.  A P13 result that passed
    # its model/physics gates becomes a separate feature product only when it
    # can be joined to the exact physical-primary MD samples and canonical
    # vertical geometry.  PreparedView consumes that product explicitly; it
    # never mistakes the learned curve for provided supervision.
    p13_v3_tracks = _registration_v3_tracks(
        pipeline,
        list(p13_runtime["tracks"].values()),
        request,
        execution_contract=execution_contract,
    )
    physical_primary_by_well = {
        str(track.get("well_uid") or track.get("well_name")): track for track in tracks
    }
    p13_by_well = {
        str(track.get("well_uid") or track.get("well_name")): track
        for track in p13_v3_tracks
    }
    fusion_feature_selection = build_p13_fusion_feature_tracks(
        physical_primary_by_well,
        p13_by_well,
    )
    fusion_feature_tracks = list(
        fusion_feature_selection["fusion_feature_tracks"].values()
    )
    # V3 may demote a legacy flag when canonical vertical geometry is
    # incomplete.  The primary count describes the authority product; the
    # feature count is the only count allowed to unlock PreparedView.
    inference_ready_count = sum(
        bool(item.get("inferenceEligible", True)) for item in tracks
    )
    primary_fusion_ready_count = sum(
        bool(item.get("fusionReady", item.get("trainingEligible", False)))
        for item in tracks
    )
    fusion_consumption_by_well = {
        str(track.get("well_uid") or track.get("well_name")): track
        for track in tracks
        if bool(track.get("fusionReady", track.get("trainingEligible", False)))
        and bool(track.get("inferenceEligible", True))
    }
    for feature_track in fusion_feature_tracks:
        identity = str(feature_track.get("well_uid") or feature_track.get("well_name"))
        # A provided/checkshot physical track remains the preferred consumer
        # input.  P13 fills only wells that have no already fusion-ready
        # physical primary.
        fusion_consumption_by_well.setdefault(identity, feature_track)
    fusion_consumption_tracks = list(fusion_consumption_by_well.values())
    fusion_ready_count = len(fusion_consumption_tracks)
    p13_fusion_feature_count = sum(
        "p13" in str(track.get("registrationSource") or "").casefold()
        for track in fusion_consumption_tracks
    )
    fusion_consumption_well_selections = [
        {
            "well_uid": track.get("well_uid"),
            "well_name": track.get("well_name"),
            "source_role": (
                "experimental_p13_feature"
                if "p13" in str(track.get("registrationSource") or "").casefold()
                else "physical_primary"
            ),
            "registration_source": track.get("registrationSource"),
            "physical_primary_identity": str(
                track.get("well_uid") or track.get("well_name")
            ),
        }
        for track in fusion_consumption_tracks
    ]
    candidate_count = max(0, registered_count - fusion_ready_count)
    manifest_fields = {
        "registration_id": task_id,
        "source_snapshot_id": request.source_snapshot_id,
        "source_snapshot_fingerprint": source_snapshot_context.get(
            "source_snapshot_fingerprint"
        ),
        "geometry_authority": "current_snapshot_DEV_or_deviation_survey",
        "geometry_policy": "registration adds TWT by MD and never replaces XYZ/TVD",
        "registration_source_policy": (
            "physical primary remains provided/checkshot/VSP/sonic/physical "
            "initialization; fusion-ready P13 is sealed only in the separate "
            "consumption product and never becomes time-depth supervision"
        ),
        "registration_arbitration": arbitration_decisions,
        "registration_arbitration_role": "legacy_candidate_comparison_audit_only",
        "status_counts": dict(status_counts),
        "method_counts": dict(method_counts),
        "well_count": len(ties),
        "alignment_attempt_count": len(alignment_attempts),
        "registered_well_count": registered_count,
        "inference_ready_well_count": inference_ready_count,
        # This manifest describes the physical-primary CSV and must not claim
        # readiness owned by a different producer file.
        "fusion_ready_well_count": primary_fusion_ready_count,
        "primary_fusion_ready_well_count": primary_fusion_ready_count,
        "fusion_feature_well_count": p13_fusion_feature_count,
        "fusion_consumption_well_count": fusion_ready_count,
        "fusion_consumption_selection_policy": (
            "physical_fusion_ready_else_exact_md_p13_feature_v1"
        ),
        "fusion_consumption_well_selections": fusion_consumption_well_selections,
        "fusion_feature_contract_version": fusion_feature_selection["contract_version"],
        "fusion_feature_decisions": fusion_feature_selection["decisions"],
        "candidate_well_count": candidate_count,
        "blocked_well_count": blocked_count,
        "business_status": business_status,
        "alignment_runtime": "physical_primary_plus_separate_p13_feature_product",
        "p13_runtime_attempted": bool(p13_runtime["attempted"]),
        "p13_checkpoint_executed": bool(p13_runtime["checkpoint_executed"]),
        "p13_runtime_status": p13_runtime["runtime_status"],
        "p13_eligible_well_count": int(p13_runtime["eligible_well_count"]),
        "p13_executed_well_count": int(p13_runtime["executed_well_count"]),
        "p13_raw_candidate_accepted_count": int(p13_runtime["accepted_well_count"]),
        "p13_records": p13_runtime["records"],
        "p13_skipped_wells": p13_runtime["skipped"],
        "formal_datum_skipped_wells": formal_datum_skipped_wells,
        "p13_errors": p13_runtime["failures"],
        "p13_rejection_reasons": sorted(
            {
                str(record["rejection_reason"])
                for record in p13_runtime["records"]
                if record.get("rejection_reason")
            }
        ),
        "time_depth_supervision_is_model_input": False,
        "execution_contract": execution_contract,
        "absolute_reference_ready": bool(
            execution_contract.get("absolute_reference_ready", True)
        ),
        "outputs": {
            "time_depth_las_by_well": {
                str(record["well_uid"]): str(record["time_depth_las"])
                for record in p13_runtime["records"]
                if record.get("time_depth_las")
            },
        },
    }
    registration_semantics = {
        "horizontal_crs_id": request.horizontal_crs_id,
        "horizontal_unit": request.horizontal_unit,
        "horizontal_axis_order": request.horizontal_axis_order,
        "vertical_crs_id": execution_contract.get("vertical_crs_id"),
        "seismic_srd_elevation_m": execution_contract.get("seismic_srd_elevation_m"),
        "absolute_reference_ready": bool(
            execution_contract.get("absolute_reference_ready", True)
        ),
        "time_domain": execution_contract.get("time_domain"),
        "time_axis_ready": bool(execution_contract.get("time_axis_ready", True)),
        "time_reference": execution_contract.get("time_reference"),
        "correction_state": execution_contract.get("correction_state"),
    }
    product = write_registration_product_v3(
        output_directory,
        tracks,
        semantics=registration_semantics,
        manifest_fields=manifest_fields,
        preview_limit=240,
    )
    points_path = product.points_path
    preview_path = product.preview_path
    manifest_path = product.manifest_path
    registration_manifest = product.manifest
    registration_output_integrity = {
        "registration_points_sha256": product.points_sha256,
        "registration_preview_sha256": product.preview_sha256,
        "registration_manifest_sha256": product.manifest_sha256,
        "registration_product_sha256": product.product_sha256,
    }
    consumption_output_files: dict[str, str] = {}
    consumption_output_integrity: dict[str, str] = {}
    fusion_consumption_product: dict[str, Any] = {
        "contract_version": "well-seismic.fusion-consumption.v1",
        "product_role": "fusion_consumption_product",
        "selection_policy": "physical_fusion_ready_else_exact_md_p13_feature_v1",
        "well_count": 0,
        "state": "not_available",
        "p13_feature_well_count": p13_fusion_feature_count,
        "well_selections": fusion_consumption_well_selections,
        "decisions": fusion_feature_selection["decisions"],
        "p13_time_depth_supervision_is_model_input": False,
    }
    if fusion_consumption_tracks:
        consumption_product = write_registration_product_v3(
            output_directory / "fusion_consumption",
            fusion_consumption_tracks,
            semantics=registration_semantics,
            manifest_fields={
                "registration_id": task_id,
                "source_snapshot_id": request.source_snapshot_id,
                "source_snapshot_fingerprint": source_snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                "product_role": "fusion_consumption_product",
                "physical_primary_product_sha256": product.product_sha256,
                "physical_primary_manifest_sha256": product.manifest_sha256,
                "registered_well_count": fusion_ready_count,
                "fusion_ready_well_count": fusion_ready_count,
                "p13_feature_well_count": p13_fusion_feature_count,
                "selection_policy": (
                    "physical_fusion_ready_else_exact_md_p13_feature_v1"
                ),
                "well_selections": fusion_consumption_well_selections,
                "business_status": "fusion_ready_experimental_or_provided",
                "fusion_feature_contract_version": fusion_feature_selection[
                    "contract_version"
                ],
                "fusion_feature_decisions": fusion_feature_selection["decisions"],
                "source_authority_policy": (
                    "P13 supplies TWT features only; physical-primary geometry "
                    "and datum remain authoritative"
                ),
                "p13_time_depth_supervision_is_model_input": False,
            },
            preview_limit=240,
        )
        consumption_output_files = {
            "fusion_consumption_manifest": str(consumption_product.manifest_path),
            "fusion_consumption_registration_points": str(
                consumption_product.points_path
            ),
            "fusion_consumption_registration_preview": str(
                consumption_product.preview_path
            ),
        }
        consumption_output_integrity = {
            # Canonical key mirrors ``output_files.fusion_consumption_manifest``.
            # Keep the older registration-prefixed alias for persisted readers.
            "fusion_consumption_manifest_sha256": consumption_product.manifest_sha256,
            "fusion_consumption_registration_points_sha256": consumption_product.points_sha256,
            "fusion_consumption_registration_preview_sha256": consumption_product.preview_sha256,
            "fusion_consumption_registration_manifest_sha256": consumption_product.manifest_sha256,
            "fusion_consumption_registration_product_sha256": consumption_product.product_sha256,
        }
        fusion_consumption_product = {
            **fusion_consumption_product,
            "state": "available",
            "well_count": fusion_ready_count,
            "physical_primary_product_sha256": product.product_sha256,
            "output_files": dict(consumption_output_files),
            "output_integrity": dict(consumption_output_integrity),
        }
    registration_output_integrity.update(consumption_output_integrity)
    registration_lineage_sha256 = canonical_sha256(
        {
            "registration_id": task_id,
            "source_snapshot_id": request.source_snapshot_id,
            "source_snapshot_fingerprint": registration_manifest.get(
                "source_snapshot_fingerprint"
            ),
            **registration_output_integrity,
        }
    )
    inspection = _inspection_result(
        pipeline,
        sealed_assets=[
            dict(item)
            for item in (source_snapshot_context.get("snapshot_assets") or [])
            if isinstance(item, dict)
        ],
    )
    inspection.setdefault("data_snapshot", {})[
        "segy_geometry_cache"
    ] = geometry_cache_receipt
    _assert_snapshot_verified_stat_signatures(source_snapshot_context)
    return {
        **inspection,
        "registration": {
            **registration_manifest,
            "can_build_multimodal_view": fusion_ready_count > 0,
            "primary_fusion_ready_well_count": primary_fusion_ready_count,
            "fusion_ready_well_count": fusion_ready_count,
            "fusion_feature_well_count": p13_fusion_feature_count,
            "fusion_consumption_well_count": fusion_ready_count,
            "candidate_well_count": candidate_count,
            "fusion_product_status": (
                "experimental_fusion_ready"
                if fusion_ready_count > 0
                else "fine_registration_complete_nonfusion"
            ),
            "fusion_consumption_product": fusion_consumption_product,
            "physical_primary_product": {
                "product_role": "physical_primary",
                "contract_version": registration_manifest.get("contract_version"),
                "well_count": registered_count,
                "output_files": {
                    "manifest": str(manifest_path),
                    "registration_points": str(points_path),
                    "registration_preview": str(preview_path),
                },
                "output_integrity": {
                    key: value
                    for key, value in registration_output_integrity.items()
                    if not key.startswith("fusion_consumption_")
                },
            },
            "active_consumption_product_role": (
                "fusion_consumption_product" if fusion_ready_count > 0 else None
            ),
            "downstream_fusion_ready_well_count": fusion_ready_count,
            "output_integrity": registration_output_integrity,
            "registration_lineage_sha256": registration_lineage_sha256,
            "output_directory": str(output_directory),
            "output_files": {
                "manifest": str(manifest_path),
                "registration_points": str(points_path),
                "registration_preview": str(preview_path),
                **consumption_output_files,
            },
        },
    }


def preprocess_paths(
    request: PreprocessingRequest,
    *,
    task_id: str,
    registration_tracks: dict[str, dict[str, Any]] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    if not request.seismic_paths:
        raise ValueError("至少需要一个地震数据路径")
    if not request.log_paths:
        raise ValueError("至少需要一个测井数据路径")
    if progress:
        progress(8, "正在校验封存SourceSnapshot与Registration V3身份")
    source_snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)

    manifest, inventory = build_explicit_paths_manifest(
        seismic_directory=request.seismic_paths,
        log_directory=request.log_paths,
        metadata_directory=request.well_paths,
        auxiliary_directory=request.auxiliary_paths,
        survey_directory=request.survey_paths,
        interpretation_directory=request.interpretation_paths,
        time_depth_directory=request.time_depth_paths,
        recursive=request.recursive,
        seismic_srd_elevation_m=request.seismic_srd_elevation_m,
        vertical_crs_id=request.vertical_crs_id,
        horizontal_crs_id=request.horizontal_crs_id,
        well_source_crs_id=request.well_source_crs_id,
        seismic_source_crs_id=request.seismic_source_crs_id,
        horizontal_unit=request.horizontal_unit,
        horizontal_axis_order=request.horizontal_axis_order,
        coordinate_reference_verified=request.coordinate_reference_verified,
        seismic_replacement_velocity_mps=request.seismic_replacement_velocity_mps,
        seismic_time_domain=request.seismic_time_domain,
        seismic_correction_state=request.seismic_correction_state,
        segy_geometry_profile=request.segy_geometry_profile,
        segy_inline_byte=request.segy_inline_byte,
        segy_crossline_byte=request.segy_crossline_byte,
        segy_x_byte=request.segy_x_byte,
        segy_y_byte=request.segy_y_byte,
        segy_coordinate_scalar_byte=request.segy_coordinate_scalar_byte,
        well_coordinate_source_unit=request.well_coordinate_source_unit,
        well_vertical_datum_source_unit=request.well_vertical_datum_source_unit,
        las_twt_source_unit=request.las_twt_source_unit,
        time_depth_default_depth_domain=request.time_depth_default_depth_domain,
        time_depth_default_depth_unit=request.time_depth_default_depth_unit,
        time_depth_default_time_unit=request.time_depth_default_time_unit,
        time_depth_default_depth_datum=request.time_depth_default_depth_datum,
        time_depth_default_depth_convention=request.time_depth_default_depth_convention,
        time_depth_default_time_reference=request.time_depth_default_time_reference,
        time_depth_default_time_domain=request.time_depth_default_time_domain,
        time_depth_default_correction_state=request.time_depth_default_correction_state,
        target_task_id=request.target_task_id,
        target_model_id=request.target_model_id,
        target_required_modalities=(
            _interpretation_registry.get(request.target_task_id).required_modalities
            if request.target_task_id
            else ()
        ),
        target_model_contract=_target_model_contract(request),
    )
    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    _apply_horizontal_coordinate_contract(pipeline, request)
    _apply_snapshot_parse_repairs(pipeline, source_snapshot_context)
    _assert_pipeline_uses_sealed_assets(pipeline, source_snapshot_context)
    pipeline.automatic_inventory = inventory
    if progress:
        progress(25, f"已锁定 {len(pipeline.assets)} 个封存资产，正在校验几何缓存")

    def cache_status(cache_hit: bool, _reason: str) -> None:
        if not progress:
            return
        progress(
            25,
            (
                "正在复用封存SourceSnapshot：资产身份已校验，已加载封存SEG-Y几何；正在构建派生视图"
                if cache_hit
                else "封存SourceSnapshot首次建立安全几何缓存：正在扫描SEG-Y道头；完成后构建派生视图"
            ),
        )

    geometry_cache_receipt = _ingest_pipeline_from_sealed_snapshot(
        pipeline,
        source_snapshot_context,
        cache_status=cache_status,
    )
    if progress:
        progress(62, "正在重建井轨迹并构建井震匹配样本")

    samples = pipeline.build_samples(registration_tracks=registration_tracks)
    tie_status_counts = Counter(
        str(item.get("status", "horizontal_only")) for item in pipeline.well_ties
    )
    valid_window_count = sum(1 for item in samples if item.get("seismic_window_valid"))
    training_eligible_count = sum(
        1 for item in samples if item.get("training_eligible")
    )
    vertical_datum_verified_count = sum(
        1 for item in samples if item.get("vertical_datum_verified")
    )
    datum_inventory = pipeline.vertical_datum_inventory()
    vertical_datum_ready = bool(
        datum_inventory.get("wells")
        and datum_inventory.get("seismic")
        and datum_inventory.get("vertical_crs_ready")
        and datum_inventory.get("ready_wells") == len(datum_inventory.get("wells", []))
        and datum_inventory.get("ready_seismic")
        == len(datum_inventory.get("seismic", []))
        and datum_inventory.get("ready_seismic_time")
        == len(datum_inventory.get("seismic", []))
    )
    coordinate_reference_verified = bool(
        pipeline.config.get("matching", {})
        .get("coordinate_reference", {})
        .get("verified", False)
    )
    sealed_assets = [
        dict(item)
        for item in (source_snapshot_context.get("snapshot_assets") or [])
        if isinstance(item, dict)
    ]
    inspection = _inspection_result(
        pipeline,
        sealed_assets=sealed_assets,
    )
    inspection.setdefault("data_snapshot", {})[
        "segy_geometry_cache"
    ] = geometry_cache_receipt
    output_directory = (
        Path(request.output_directory).expanduser().resolve()
        if request.output_directory and request.output_directory.strip()
        else PROJECT_ROOT / "输出结果" / f"前端任务_{task_id[:8]}"
    )
    if progress:
        progress(88, "正在写入中文质量报告、样本索引和多模态样本")
    output_files = pipeline.write_outputs(
        output_directory,
        sealed_assets=sealed_assets,
    )
    _assert_snapshot_verified_stat_signatures(source_snapshot_context)
    return {
        **inspection,
        "matching": {
            "sample_count": len(samples),
            "valid_window_count": valid_window_count,
            "training_eligible_count": training_eligible_count,
            "coordinate_reference_verified": coordinate_reference_verified,
            "vertical_datum_ready": vertical_datum_ready,
            "vertical_datum_verified_count": vertical_datum_verified_count,
            "vertical_alignment_counts": dict(tie_status_counts),
            "registration_task_id": request.registration_task_id,
            "reused_registration_track_count": len(registration_tracks or {}),
            "registration_reuse_policy": (
                "TWT_mean/std/quality are joined by well_uid and MD; the registration "
                "product is inference evidence, never time-depth supervision"
            ),
            "output_directory": str(output_directory),
            "output_files": {name: str(path) for name, path in output_files.items()},
        },
    }


def _run_inspection(task_id: str, request: InspectionRequest) -> None:
    last_progress_write_at = 0.0
    last_progress_phase = ""

    def update(
        progress: int,
        message: str,
        progress_detail: dict[str, Any] | None = None,
    ) -> None:
        nonlocal last_progress_phase, last_progress_write_at
        now = monotonic()
        phase = str((progress_detail or {}).get("phase") or "")
        phase_changed = phase != last_progress_phase
        terminal = progress >= 100 or phase in {"completed", "failed"}
        if not phase_changed and not terminal and now - last_progress_write_at < 0.5:
            return
        values: dict[str, Any] = {
            "status": "running",
            "progress": progress,
            "message": message,
        }
        if progress_detail is not None:
            values["progress_detail"] = progress_detail
        _set_task(task_id, **values)
        last_progress_phase = phase
        last_progress_write_at = now

    started_monotonic = monotonic()
    try:
        _autofill_horizontal_coordinate_contract(request)
        request_contract_sha256 = _preparation_request_contract_sha256(request)
        input_fingerprint = _preparation_input_fingerprint(request)
        preparation_estimate = _historical_preparation_estimate(
            request,
            request_contract_sha256=request_contract_sha256,
            input_fingerprint=input_fingerprint,
        )
        _set_task(
            task_id,
            status="running",
            progress=1,
            message="任务已开始",
            progress_detail={"phase": "submitting", "can_estimate": False},
            preparation_estimate=preparation_estimate,
            preparation_contract_sha256=request_contract_sha256,
            preparation_input_fingerprint=input_fingerprint,
        )
        inspected_pipelines: list[WellSeismicPipeline] = []
        result = inspect_paths(
            request,
            update,
            _pipeline_sink=inspected_pipelines.append,
        )
        # Persist the effective auto-detected contract so refresh/recovery and
        # the sealed snapshot use the same CRS semantics as the completed run.
        _set_task(task_id, request=request.model_dump(mode="json"))
        _seal_data_preparation_snapshot(task_id, request, result)
        _set_task(
            task_id,
            status="running",
            progress=99,
            message="SourceSnapshot已封存，正在保存可跨进程复用的SEG-Y几何缓存",
            progress_detail={"phase": "caching", "can_estimate": False},
        )
        cache_context = {
            "source_snapshot_id": task_id,
            "source_snapshot_fingerprint": (result.get("data_snapshot") or {}).get(
                "snapshot_sha256"
            ),
            "snapshot_contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
            "snapshot_assets": result.get("assets") or [],
        }
        if len(inspected_pipelines) != 1:
            cache_receipt = {
                "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
                "state": "unavailable_regenerable",
                "source_snapshot_id": task_id,
                "source_snapshot_sha256": cache_context["source_snapshot_fingerprint"],
                "seismic_geometry_count": 0,
                "reason": "final_pipeline_not_retained",
            }
        else:
            try:
                cache_receipt = _write_pipeline_sealed_geometry_cache(
                    inspected_pipelines[0],
                    cache_context,
                )
            except Exception as cache_exc:  # cache is regenerable, snapshot is not
                LOGGER.warning(
                    "SourceSnapshot已封存，但无法保存可再生SEG-Y几何缓存：%s",
                    cache_exc,
                )
                cache_receipt = {
                    "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
                    "state": "unavailable_regenerable",
                    "source_snapshot_id": task_id,
                    "source_snapshot_sha256": cache_context[
                        "source_snapshot_fingerprint"
                    ],
                    "seismic_geometry_count": 0,
                    "reason": type(cache_exc).__name__,
                }
        result.setdefault("data_snapshot", {})["segy_geometry_cache"] = cache_receipt
        completed_at = _now()
        try:
            created_at = datetime.fromisoformat(
                str(_get_task(task_id)["created_at"]).replace("Z", "+00:00")
            )
            completed_datetime = datetime.fromisoformat(
                completed_at.replace("Z", "+00:00")
            )
            preparation_duration_seconds = max(
                0.0, (completed_datetime - created_at).total_seconds()
            )
        except (KeyError, TypeError, ValueError):
            preparation_duration_seconds = max(0.0, monotonic() - started_monotonic)
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="路径预检与数据识别完成",
            project_id=DEFAULT_PROJECT_ID,
            snapshot_id=task_id,
            result=result,
            progress_detail={"phase": "completed", "can_estimate": False},
            completed_at=completed_at,
            preparation_duration_seconds=round(preparation_duration_seconds, 3),
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="路径预检失败",
            error={"type": type(exc).__name__, "message": str(exc)},
            progress_detail={"phase": "failed", "can_estimate": False},
        )


def _run_registration(task_id: str, request: PreprocessingRequest) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="井震标定任务已开始")
        result = calibrate_paths(request, task_id=task_id, progress=update)
        result["data_snapshot"]["snapshot_id"] = request.source_snapshot_id
        snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
        registration = result.get("registration") or {}
        bundle = _state_store.create_artifact_bundle(
            {
                "kind": "registration",
                "contract_version": registration.get("contract_version"),
                "registration_id": task_id,
                "source_snapshot_id": request.source_snapshot_id,
                "source_snapshot_fingerprint": snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                "registration_lineage_sha256": registration.get(
                    "registration_lineage_sha256"
                ),
                "fusion_product_status": registration.get("fusion_product_status"),
                "fusion_feature_well_count": registration.get(
                    "fusion_feature_well_count", 0
                ),
                "outputs": dict(registration.get("output_files") or {}),
            },
            bundle_id=f"registration-{task_id}",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
            task_id=task_id,
        )
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message=(
                "井震精细标定与可消费融合特征轨已生成"
                if registration.get("can_build_multimodal_view")
                else "井震精细标定候选已生成；尚未达到融合视图门禁"
            ),
            artifact_bundle_id=bundle["bundle_id"],
            result=result,
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="井震标定失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _run_horizontal_registration(
    task_id: str,
    request: PreprocessingRequest,
) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(
            task_id,
            status="running",
            progress=1,
            message="无时深水平配准任务已开始",
        )
        result = horizontal_register_paths(request, task_id=task_id, progress=update)
        horizontal = result.get("horizontal_registration") or {}
        snapshot_context = result.get("data_snapshot") or {}
        bundle = _state_store.create_artifact_bundle(
            {
                "kind": "horizontal_registration",
                "contract_version": HORIZONTAL_REGISTRATION_CONTRACT_VERSION,
                "horizontal_registration_id": task_id,
                "source_snapshot_id": request.source_snapshot_id,
                "source_snapshot_fingerprint": snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                "horizontal_registration_lineage_sha256": horizontal.get(
                    "horizontal_registration_lineage_sha256"
                ),
                "outputs": dict(horizontal.get("output_files") or {}),
            },
            bundle_id=f"horizontal-registration-{task_id}",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
            task_id=task_id,
        )
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="井轨迹最近道、网格覆盖与水平QC已生成",
            artifact_bundle_id=bundle["bundle_id"],
            result=result,
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="无时深水平配准失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _run_preprocessing(task_id: str, request: PreprocessingRequest) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="预处理任务已开始")
        snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
        registration_tracks: dict[str, dict[str, Any]] = {}
        registration_context: dict[str, Any] = {}
        if request.registration_task_id:
            registration_context = _prediction_registration_context(
                request.registration_task_id,
                source_task_id=request.source_snapshot_id,
                source_snapshot_fingerprint=snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
            )
            points_path = registration_context["registration_points_path"]
            allowed_wells = set(
                registration_context["registration_fusion_ready_well_ids"]
            )
            registration_tracks = {
                well_uid: track
                for well_uid, track in load_registration_points(points_path).items()
                if well_uid in allowed_wells
                and bool(track.get("fusionReady"))
                and bool(track.get("inferenceEligible", True))
            }
            if not registration_tracks:
                raise ValueError(
                    "井震标定成果没有 fusion_ready=true 的可复用逐井MD-TWT轨迹"
                )
        result = preprocess_paths(
            request,
            task_id=task_id,
            registration_tracks=registration_tracks,
            progress=update,
        )
        result["data_snapshot"]["snapshot_id"] = request.source_snapshot_id
        result["registration_task_id"] = request.registration_task_id
        result["source_snapshot_id"] = request.source_snapshot_id
        matching = result.setdefault("matching", {})
        matching_output_directory = str(matching.get("output_directory") or "").strip()
        prepared_root = (
            Path(matching_output_directory).expanduser().resolve()
            if matching_output_directory
            else (PROJECT_ROOT / "model_outputs" / "prepared_views" / task_id)
        )
        prepared_root.mkdir(parents=True, exist_ok=True)
        matching["output_directory"] = str(prepared_root)
        matching.setdefault("output_files", {})
        result["data_snapshot"]["derived_views"]["well_seismic_samples"].update(
            {
                "status": "generated",
                "sample_count": matching.get("sample_count", 0),
                "valid_window_count": matching.get("valid_window_count", 0),
                "training_eligible_count": matching.get("training_eligible_count", 0),
                "output_directory": str(prepared_root),
            }
        )
        output_files = {
            str(name): str(path)
            for name, path in (matching.get("output_files") or {}).items()
            if Path(str(path)).expanduser().resolve().is_file()
        }
        prepared_artifacts = dict(output_files)
        artifact_metadata: dict[str, dict[str, Any]] = {
            name: {
                "role": "derived_sample_export",
                "schema_version": "well-seismic.sample-export.v1",
            }
            for name in output_files
        }
        # A PreparedView must contain the exact reusable inputs, not just a
        # report saying sample construction once happened.  These are sealed
        # references: validation re-hashes every LAS and Registration V3 file.
        for asset in snapshot_context.get("snapshot_assets") or []:
            path = Path(str(asset.get("path") or "")).expanduser().resolve()
            role = str(asset.get("role") or "").casefold()
            if path.suffix.casefold() != ".las" or role not in {
                "well_log",
                "well_logs",
                "log",
                "las",
            }:
                continue
            key = f"canonical_well_las::{asset.get('id') or asset.get('sha256') or path.name}"
            prepared_artifacts[key] = str(path)
            artifact_metadata[key] = {
                "role": "canonical_well_las",
                "schema_version": "well-seismic.canonical-well-input.v1",
                "source_asset_id": asset.get("id"),
            }
        if registration_context:
            for key, role, schema in (
                (
                    "registration_manifest_v3",
                    "registration_manifest_v3",
                    "well-seismic.registration.v3",
                ),
                (
                    "registration_points_v3",
                    "registration_points_v3",
                    "well-seismic.registration.v3",
                ),
            ):
                source_key = (
                    "registration_manifest_path"
                    if key == "registration_manifest_v3"
                    else "registration_points_path"
                )
                prepared_artifacts[key] = str(registration_context[source_key])
                artifact_metadata[key] = {
                    "role": role,
                    "schema_version": schema,
                    "source_registration_task_id": request.registration_task_id,
                    "registration_product_role": registration_context.get(
                        "registration_product_role"
                    ),
                }
            if registration_context.get("registration_product_role") == (
                "fusion_consumption_product"
            ):
                for key, role, source_key in (
                    (
                        "registration_physical_primary_manifest_v3",
                        "registration_physical_primary_manifest_v3",
                        "registration_physical_primary_manifest_path",
                    ),
                    (
                        "registration_physical_primary_points_v3",
                        "registration_physical_primary_points_v3",
                        "registration_physical_primary_points_path",
                    ),
                ):
                    source_path = str(registration_context.get(source_key) or "")
                    if not source_path or not Path(source_path).is_file():
                        raise ValueError(
                            "fusion feature product is missing its physical-primary audit file"
                        )
                    prepared_artifacts[key] = source_path
                    artifact_metadata[key] = {
                        "role": role,
                        "schema_version": "well-seismic.registration.v3",
                        "source_registration_task_id": request.registration_task_id,
                        "consumption_role": "audit_only_physical_authority",
                    }
        prepared_manifest_path = prepared_root / "prepared_view_manifest.json"
        source_snapshot_sha256 = str(
            snapshot_context.get("source_snapshot_fingerprint") or ""
        )
        if not source_snapshot_sha256:
            result["prepared_view"] = {
                "contract_version": "well-seismic.prepared-view.v1",
                "state": "unavailable_legacy_unpinned",
                "view_id": task_id,
                "source_snapshot_id": request.source_snapshot_id,
                "reason": (
                    "legacy source task has no sealed content fingerprint; "
                    "derived outputs remain readable but are not reusable as a "
                    "PreparedView"
                ),
            }
            _set_task(
                task_id,
                status="completed",
                progress=100,
                message="数据预处理完成；旧快照未封存派生视图",
                result=result,
            )
            return
        parents: list[dict[str, Any]] = []
        if request.registration_task_id:
            parents.append(
                {
                    "kind": "registration",
                    "view_id": request.registration_task_id,
                    "view_sha256": registration_context.get(
                        "registration_lineage_sha256"
                    ),
                }
            )
            parents.append(
                {
                    "kind": "registration_consumption_product",
                    "view_id": request.registration_task_id,
                    "view_sha256": registration_context.get(
                        "registration_product_sha256"
                    ),
                }
            )
        prepared_view = write_prepared_view_manifest(
            prepared_manifest_path,
            view_id=task_id,
            kind="multimodal_samples",
            source_snapshot_id=str(request.source_snapshot_id),
            source_snapshot_sha256=source_snapshot_sha256,
            producer_task_id=task_id,
            artifacts=prepared_artifacts,
            artifact_metadata=artifact_metadata,
            parents=parents,
            producer={
                "platform_version": app.version,
                "effective_config_sha256": _effective_configuration_sha256(),
                "transformation_registry_sha256": (_transformation_registry_sha256()),
                "registration_track_count": len(registration_tracks),
                "registration_product_role": registration_context.get(
                    "registration_product_role"
                ),
                "registration_product_sha256": registration_context.get(
                    "registration_product_sha256"
                ),
                "registration_points_sha256": registration_context.get(
                    "registration_points_sha256"
                ),
                "registration_manifest_sha256": registration_context.get(
                    "registration_manifest_sha256"
                ),
                "physical_primary_product_sha256": registration_context.get(
                    "registration_physical_primary_product_sha256"
                ),
                "fusion_ready_well_ids": registration_context.get(
                    "registration_fusion_ready_well_ids", []
                ),
                "reusable_input_contract": "aligned_well_sequence_v1",
            },
            gates={
                "sample_count": result.get("matching", {}).get("sample_count", 0),
                "valid_window_count": result.get("matching", {}).get(
                    "valid_window_count", 0
                ),
                "training_eligible_count": result.get("matching", {}).get(
                    "training_eligible_count", 0
                ),
                "coordinate_reference_verified": result.get("matching", {}).get(
                    "coordinate_reference_verified", False
                ),
                "vertical_datum_ready": result.get("matching", {}).get(
                    "vertical_datum_ready", False
                ),
                "registration_product_role": registration_context.get(
                    "registration_product_role"
                ),
                "registration_fusion_ready_well_ids": registration_context.get(
                    "registration_fusion_ready_well_ids", []
                ),
            },
        )
        result["prepared_view"] = prepared_view
        matching["output_files"]["派生视图清单"] = prepared_view["manifest_path"]
        bundle = _state_store.create_artifact_bundle(
            {
                "kind": "prepared_view",
                "prepared_view": prepared_view,
            },
            bundle_id=f"prepared-{task_id}",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
            task_id=task_id,
        )
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="数据预处理与多模态样本构建完成",
            artifact_bundle_id=bundle["bundle_id"],
            result=result,
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="数据预处理与匹配失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _prediction_snapshot_context(source_task_id: str | None) -> dict[str, Any]:
    """Project the sealed source inventory into model-neutral runner options."""

    if not source_task_id:
        return {}
    source_task = _get_task(source_task_id)
    if source_task.get("status") != "completed":
        raise ValueError("the source data snapshot has not completed")
    result = source_task.get("result") or {}
    data_snapshot = result.get("data_snapshot") or {}
    contract_version = str(data_snapshot.get("contract_version") or "")
    strict_content_addressing = contract_version in {
        "well-seismic.data.v2",
        SOURCE_SNAPSHOT_CONTRACT_VERSION,
    }
    compact_assets: list[dict[str, Any]] = []
    source_assets = [
        item
        for item in (result.get("assets") or [])
        if isinstance(item, dict) and item.get("path")
    ]
    if strict_content_addressing:
        expected_count = int(data_snapshot.get("asset_count") or 0)
        if expected_count != len(source_assets):
            raise ValueError(
                "source data snapshot asset count no longer matches its sealed contract"
            )
        expected_set_sha = str(data_snapshot.get("asset_set_sha256") or "")
        if not expected_set_sha:
            raise ValueError("source data snapshot is missing asset_set_sha256")
        observed_set_sha = snapshot_assets_fingerprint(source_assets)
        if observed_set_sha != expected_set_sha:
            raise ValueError(
                "source data snapshot asset inventory fingerprint mismatch"
            )
    for item in source_assets:
        compact = {
            key: item[key]
            for key in (
                "id",
                "path",
                "role",
                "format",
                "name",
                "size",
                "sha256",
                "geometry_fingerprint",
                "asset_options_sha256",
            )
            if item.get(key) is not None
        }
        if strict_content_addressing:
            path = Path(str(item["path"])).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"source data snapshot asset is missing: {path}")
            expected_size = int(item.get("size") or -1)
            expected_sha = str(item.get("sha256") or "").casefold()
            if len(expected_sha) != 64:
                raise ValueError(
                    f"source data snapshot asset has no valid SHA-256: {path}"
                )
            verified_stat_signature = _verify_snapshot_file_identity(
                path,
                expected_size=expected_size,
                expected_sha256=expected_sha,
            )
            compact["integrity_status"] = "sha256_verified"
            compact["verified_stat_signature"] = list(verified_stat_signature)
        else:
            compact["integrity_status"] = "legacy_unpinned"
        compact_assets.append(compact)
    project_id = source_task.get("project_id") or data_snapshot.get("project_id")
    snapshot_manifest: dict[str, Any] | None = None
    if contract_version == SOURCE_SNAPSHOT_CONTRACT_VERSION:
        snapshot_manifest = _source_snapshot_manifest_from_task(source_task)
        if snapshot_manifest is None:
            raise ValueError("v3 source data snapshot has no sealed manifest")
        hashes = snapshot_manifest.get("hashes") or {}
        if str(hashes.get("snapshot_sha256") or "") != str(
            data_snapshot.get("snapshot_sha256") or ""
        ):
            raise ValueError(
                "source snapshot identity differs from its sealed manifest"
            )
        try:
            stored_snapshot = _state_store.get_snapshot(source_task_id)
        except RecordNotFoundError as exc:
            raise ValueError(
                "v3 source snapshot is missing from the state store"
            ) from exc
        if stored_snapshot.get("state") != "sealed":
            raise ValueError("source snapshot state is not sealed")
        if stored_snapshot.get("manifest_sha256") != data_snapshot.get(
            "snapshot_manifest_sha256"
        ):
            raise ValueError("state-store snapshot manifest identity mismatch")
        project_id = stored_snapshot.get("project_id")
    segy_config = dict(_platform_config.get("segy") or {})
    automatic_segy_geometry_receipts = (
        build_verified_snapshot_segy_geometry_receipts(
            result,
            snapshot_id=source_task_id,
            snapshot_contract_version=contract_version,
            source_snapshot_fingerprint=str(
                data_snapshot.get("snapshot_sha256")
                or data_snapshot.get("asset_set_sha256")
                or ""
            ),
            snapshot_assets=compact_assets,
            minimum_geometry_confidence=max(
                GEOPATH_MINIMUM_GEOMETRY_CONFIDENCE,
                float(segy_config.get("minimum_geometry_confidence", 0.35)),
            ),
        )
        if strict_content_addressing
        else []
    )
    context = {
        "source_snapshot_id": source_task_id,
        "project_id": project_id,
        "snapshot_assets": compact_assets,
        "snapshot_metadata_detection": [
            dict(item)
            for item in (result.get("metadata_detection") or [])
            if isinstance(item, dict)
        ],
        "snapshot_contract_version": contract_version,
        "source_snapshot_fingerprint": data_snapshot.get("snapshot_sha256")
        or data_snapshot.get("asset_set_sha256"),
        "source_snapshot_content_sha256": data_snapshot.get("source_content_sha256"),
        "source_snapshot_semantics_sha256": data_snapshot.get("semantics_sha256"),
        "source_snapshot_semantics": (
            dict(snapshot_manifest.get("semantics") or {})
            if snapshot_manifest is not None
            else dict(data_snapshot.get("semantics") or {})
        ),
        "source_snapshot_segy_geometry_receipts": (automatic_segy_geometry_receipts),
        "source_snapshot_segy_geometry_cache_receipt": (
            dict(data_snapshot.get("segy_geometry_cache") or {})
            if isinstance(data_snapshot.get("segy_geometry_cache"), Mapping)
            else {}
        ),
        "source_snapshot_parse_repairs": (
            list(snapshot_manifest.get("parse_repairs") or [])
            if snapshot_manifest is not None
            else []
        ),
        "source_snapshot_survey_attestation": (
            dict(snapshot_manifest.get("survey_attestation_receipt") or {})
            if snapshot_manifest is not None
            else {}
        ),
        "source_snapshot_parent_id": (
            snapshot_manifest.get("parent_snapshot_id")
            if snapshot_manifest is not None
            else data_snapshot.get("parent_snapshot_id")
        ),
        "source_snapshot_system_evidence_receipt": (
            dict(snapshot_manifest.get("system_evidence_receipt") or {})
            if snapshot_manifest is not None
            else dict(data_snapshot.get("system_evidence_receipt") or {})
        ),
        "source_snapshot_manifest_path": data_snapshot.get("snapshot_manifest_path"),
        "source_snapshot_manifest_sha256": data_snapshot.get(
            "snapshot_manifest_sha256"
        ),
        "snapshot_integrity_status": (
            "sha256_verified" if strict_content_addressing else "legacy_unpinned"
        ),
    }
    _assert_snapshot_verified_stat_signatures(context)
    return context


def _prediction_prepared_view_context(
    prepared_view_task_id: str | None,
    *,
    source_snapshot_id: str | None,
    source_snapshot_sha256: str | None,
    registration_task_id: str | None,
) -> dict[str, Any]:
    """Resolve a sealed derivative without mistaking it for a source snapshot."""

    if not prepared_view_task_id:
        return {}
    task = _get_task(prepared_view_task_id)
    if task.get("task_type") != "sample_building" or task.get("status") != "completed":
        raise ValueError("the requested prepared view has not completed")
    result = task.get("result") or {}
    declared = result.get("prepared_view") or {}
    manifest_path = declared.get("manifest_path")
    if not manifest_path:
        raise ValueError("prepared-view manifest is missing from the task")
    validated = validate_prepared_view_manifest(
        manifest_path,
        expected_view_id=prepared_view_task_id,
        expected_source_snapshot_id=source_snapshot_id,
        expected_source_snapshot_sha256=source_snapshot_sha256,
    )
    declared_manifest_sha256 = str(declared.get("manifest_sha256") or "")
    if not declared_manifest_sha256:
        raise ValueError("prepared-view task is missing its sealed manifest identity")
    if declared_manifest_sha256 != str(validated.get("manifest_sha256") or ""):
        raise ValueError("prepared-view manifest content changed after task completion")
    declared_view_sha256 = str(declared.get("view_sha256") or "")
    if not declared_view_sha256:
        raise ValueError("prepared-view task is missing its sealed view identity")
    if declared_view_sha256 != str(validated.get("view_sha256") or ""):
        raise ValueError("prepared-view identity differs from its completed task")
    if str(validated.get("producer_task_id") or "") != prepared_view_task_id:
        raise ValueError("prepared-view producer task lineage mismatch")
    parents = [
        item for item in (validated.get("parents") or []) if isinstance(item, dict)
    ]
    registration_parents = [
        item
        for item in parents
        if item.get("kind") == "registration" and item.get("view_id")
    ]
    registration_parent_ids = {
        str(item.get("view_id"))
        for item in registration_parents
    }
    for registration_parent in registration_parents:
        parent_id = str(registration_parent.get("view_id"))
        try:
            parent_task = _get_task(parent_id)
        except KeyError as exc:
            raise ValueError(
                "prepared view registration parent task is missing"
            ) from exc
        if (
            parent_task.get("task_type") != "well_tie"
            or parent_task.get("status") != "completed"
        ):
            raise ValueError(
                "prepared view registration parent has not completed"
            )
        parent_result = parent_task.get("result") or {}
        parent_registration = parent_result.get("registration") or {}
        parent_request = parent_task.get("request") or {}
        parent_snapshot_id = str(
            parent_registration.get("source_snapshot_id")
            or parent_task.get("snapshot_id")
            or parent_request.get("source_snapshot_id")
            or ""
        )
        if not parent_snapshot_id or (
            source_snapshot_id and parent_snapshot_id != str(source_snapshot_id)
        ):
            raise ValueError(
                "prepared view registration parent belongs to a different data snapshot"
            )
        parent_snapshot_sha256 = str(
            parent_registration.get("source_snapshot_fingerprint") or ""
        )
        if (
            source_snapshot_sha256
            and parent_snapshot_sha256
            and parent_snapshot_sha256.casefold()
            != str(source_snapshot_sha256).casefold()
        ):
            raise ValueError(
                "prepared view registration parent source fingerprint mismatch"
            )
        expected_lineage_sha256 = str(
            parent_registration.get("registration_lineage_sha256") or ""
        )
        declared_parent_sha256 = str(
            registration_parent.get("view_sha256") or ""
        )
        if expected_lineage_sha256 and (
            declared_parent_sha256.casefold() != expected_lineage_sha256.casefold()
        ):
            raise ValueError(
                "prepared view registration parent lineage hash mismatch"
            )
        fusion_ready_count = max(
            int(parent_registration.get("downstream_fusion_ready_well_count") or 0),
            int(parent_registration.get("fusion_consumption_well_count") or 0),
            int(parent_registration.get("fusion_ready_well_count") or 0),
        )
        if (
            parent_registration.get("can_build_multimodal_view") is False
            or fusion_ready_count <= 0
        ):
            raise ValueError(
                "prepared view registration parent has no fusion-ready wells"
            )
    if registration_task_id and registration_task_id not in registration_parent_ids:
        raise ValueError(
            "prepared view and prediction refer to different registration products"
        )
    if not registration_task_id and registration_parent_ids:
        # A caller may use the well/seismic portions without consuming Align,
        # but the hidden parent must remain visible in provenance.
        registration_relation = "available_in_view_not_requested"
    else:
        registration_relation = "matched" if registration_task_id else "none"
    artifacts = [
        dict(item)
        for item in (validated.get("artifacts") or [])
        if isinstance(item, dict)
    ]
    artifacts_by_role: dict[str, list[dict[str, Any]]] = {}
    for item in artifacts:
        role = str(item.get("role") or "unclassified")
        artifacts_by_role.setdefault(role, []).append(item)
    has_aligned_sequence_inputs = bool(
        artifacts_by_role.get("canonical_well_las")
        and artifacts_by_role.get("registration_manifest_v3")
        and artifacts_by_role.get("registration_points_v3")
    )
    producer = dict(validated.get("producer") or {})
    prepared_registration_product_role = str(
        producer.get("registration_product_role") or ""
    )
    if registration_task_id:
        current_registration = _prediction_registration_context(
            registration_task_id,
            source_task_id=source_snapshot_id,
            source_snapshot_fingerprint=source_snapshot_sha256,
        )
        current_role = str(current_registration.get("registration_product_role") or "")
        if current_role == "fusion_consumption_product":
            if prepared_registration_product_role != current_role:
                raise ValueError(
                    "prepared view does not bind the active fusion consumption product"
                )
            expected_values = {
                "registration_product_sha256": current_registration.get(
                    "registration_product_sha256"
                ),
                "registration_points_sha256": current_registration.get(
                    "registration_points_sha256"
                ),
                "registration_manifest_sha256": current_registration.get(
                    "registration_manifest_sha256"
                ),
                "physical_primary_product_sha256": current_registration.get(
                    "registration_physical_primary_product_sha256"
                ),
            }
            for key, expected in expected_values.items():
                if not expected or str(producer.get(key) or "") != str(expected):
                    raise ValueError(f"prepared view {key} differs from registration")
            consumption_parents = [
                item
                for item in parents
                if item.get("kind") == "registration_consumption_product"
                and str(item.get("view_id") or "") == registration_task_id
            ]
            if len(consumption_parents) != 1 or str(
                consumption_parents[0].get("view_sha256") or ""
            ) != str(current_registration.get("registration_product_sha256") or ""):
                raise ValueError(
                    "prepared view consumption-product parent lineage mismatch"
                )
            manifest_records = artifacts_by_role.get("registration_manifest_v3") or []
            points_records = artifacts_by_role.get("registration_points_v3") or []
            if (
                len(manifest_records) != 1
                or len(points_records) != 1
                or str(manifest_records[0].get("sha256") or "")
                != str(current_registration.get("registration_manifest_sha256") or "")
                or str(points_records[0].get("sha256") or "")
                != str(current_registration.get("registration_points_sha256") or "")
            ):
                raise ValueError(
                    "prepared view artifacts differ from fusion consumption inputs"
                )
            primary_manifest_records = (
                artifacts_by_role.get("registration_physical_primary_manifest_v3") or []
            )
            primary_points_records = (
                artifacts_by_role.get("registration_physical_primary_points_v3") or []
            )
            if (
                len(primary_manifest_records) != 1
                or len(primary_points_records) != 1
                or str(primary_manifest_records[0].get("sha256") or "")
                != str(
                    current_registration.get(
                        "registration_physical_primary_manifest_sha256"
                    )
                    or ""
                )
                or str(primary_points_records[0].get("sha256") or "")
                != str(
                    current_registration.get(
                        "registration_physical_primary_points_sha256"
                    )
                    or ""
                )
            ):
                raise ValueError(
                    "prepared view physical-primary audit artifacts are incomplete"
                )
    return {
        "prepared_view_id": prepared_view_task_id,
        "prepared_view_manifest_path": validated["manifest_path"],
        "prepared_view_manifest_sha256": validated["manifest_sha256"],
        "prepared_view_sha256": validated["view_sha256"],
        "prepared_view_kind": validated["kind"],
        "prepared_view_registration_relation": registration_relation,
        "prepared_view_registration_parent_ids": sorted(registration_parent_ids),
        "prepared_view_available": True,
        "prepared_view_artifacts": artifacts,
        "prepared_view_artifacts_by_role": artifacts_by_role,
        "prepared_view_aligned_sequence_inputs_ready": has_aligned_sequence_inputs,
        "prepared_view_registration_product_role": (
            prepared_registration_product_role or None
        ),
    }


def _latest_compatible_prepared_view_task_id(
    *,
    source_snapshot_id: str,
    source_snapshot_sha256: str | None,
    registration_task_id: str | None,
) -> str | None:
    """Reuse the newest verified fusion view for older source-backed clients."""

    try:
        scoped_candidates = _state_store.list_tasks(
            snapshot_id=source_snapshot_id,
            status="completed",
            task_type="sample_building",
            limit=10_000,
        )
    except (StateStoreError, TypeError, ValueError):
        return None

    def compatible_task_id(candidates: list[dict[str, Any]]) -> str | None:
        for task in candidates:
            task_id = str(task.get("task_id") or "")
            if not task_id:
                continue
            try:
                prepared_view_context = _prediction_prepared_view_context(
                    task_id,
                    source_snapshot_id=source_snapshot_id,
                    source_snapshot_sha256=source_snapshot_sha256,
                    registration_task_id=registration_task_id,
                )
            except (KeyError, OSError, TypeError, ValueError):
                continue
            if not prepared_view_context.get("prepared_view_registration_parent_ids"):
                continue
            return task_id
        return None

    scoped_match = compatible_task_id(scoped_candidates)
    if scoped_match:
        return scoped_match

    # Older completed tasks may predate control-plane snapshot foreign keys.
    # Search them only as a compatibility fallback after the indexed lookup.
    try:
        legacy_candidates = _state_store.list_tasks(
            status="completed",
            task_type="sample_building",
            limit=10_000,
        )
    except (StateStoreError, TypeError, ValueError):
        return None
    scoped_ids = {str(item.get("task_id") or "") for item in scoped_candidates}
    legacy_candidates = [
        item
        for item in legacy_candidates
        if str(item.get("task_id") or "") not in scoped_ids
    ]
    legacy_match = compatible_task_id(legacy_candidates)
    if legacy_match:
        return legacy_match
    return None


def _csv_truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _fusion_ready_registration_well_ids(path: Path) -> list[str]:
    """Audit the sealed CSV by well; aggregate counts are never authoritative."""

    grouped: dict[str, list[bool]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        required = {"well_uid", "fusion_ready"}
        if not required.issubset(columns):
            missing = sorted(required - columns)
            raise ValueError(
                f"registration_points.csv is missing fusion gate columns: {missing}"
            )
        for row in reader:
            well_uid = str(row.get("well_uid") or "").strip()
            if not well_uid:
                continue
            ready = _csv_truth(row.get("fusion_ready"))
            if "inference_eligible" in columns:
                ready = ready and _csv_truth(row.get("inference_eligible"))
            grouped.setdefault(well_uid, []).append(ready)
    return sorted(
        well_uid
        for well_uid, decisions in grouped.items()
        if decisions and all(decisions)
    )


def _validate_fusion_consumption_tracks(
    *,
    consumption_tracks: dict[str, dict[str, Any]],
    physical_tracks: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Revalidate the mixed consumer product against its physical parent."""

    if manifest.get("selection_policy") != (
        "physical_fusion_ready_else_exact_md_p13_feature_v1"
    ):
        raise ValueError("fusion consumption selection policy is missing or invalid")

    physical_by_identity: dict[str, dict[str, Any]] = {}
    for key, track in physical_tracks.items():
        identities = {
            str(key).strip().casefold(),
            str(track.get("well_uid") or "").strip().casefold(),
            str(track.get("well_name") or "").strip().casefold(),
        } - {""}
        for identity in identities:
            existing = physical_by_identity.get(identity)
            if existing is not None and existing is not track:
                raise ValueError("physical-primary registration identity is ambiguous")
            physical_by_identity[identity] = track

    selections = [
        dict(item)
        for item in (manifest.get("well_selections") or [])
        if isinstance(item, dict)
    ]
    if len(selections) != len(consumption_tracks):
        raise ValueError("fusion consumption well selections do not match its tracks")
    selection_by_identity: dict[str, dict[str, Any]] = {}
    for selection in selections:
        identity = (
            str(selection.get("well_uid") or selection.get("well_name") or "")
            .strip()
            .casefold()
        )
        if not identity or identity in selection_by_identity:
            raise ValueError("fusion consumption well selection identity is invalid")
        selection_by_identity[identity] = selection

    p13_count = 0
    for key, consumer in consumption_tracks.items():
        identity = (
            str(consumer.get("well_uid") or consumer.get("well_name") or key)
            .strip()
            .casefold()
        )
        selection = selection_by_identity.get(identity)
        physical = physical_by_identity.get(identity)
        if selection is None or physical is None:
            raise ValueError("fusion consumption track has no unique physical parent")
        declared_parent_identity = (
            str(selection.get("physical_primary_identity") or "").strip().casefold()
        )
        physical_identities = {
            str(physical.get("well_uid") or "").strip().casefold(),
            str(physical.get("well_name") or "").strip().casefold(),
        } - {""}
        if (
            not declared_parent_identity
            or declared_parent_identity not in physical_identities
        ):
            raise ValueError(
                "fusion consumption selection does not bind its physical parent identity"
            )
        consumer_name = str(consumer.get("well_name") or "").strip().casefold()
        physical_name = str(physical.get("well_name") or "").strip().casefold()
        if consumer_name and physical_name and consumer_name != physical_name:
            raise ValueError(
                "fusion consumption track well name differs from its physical parent"
            )
        if not bool(consumer.get("fusionReady")) or not bool(
            consumer.get("inferenceEligible", True)
        ):
            raise ValueError("fusion consumption track is not inference/fusion ready")

        consumer_source = str(consumer.get("registrationSource") or "").casefold()
        is_p13 = "p13" in consumer_source
        expected_role = "experimental_p13_feature" if is_p13 else "physical_primary"
        if selection.get("source_role") != expected_role:
            raise ValueError("fusion consumption source role differs from its track")

        absolute_reference_ready = bool(consumer.get("absoluteReferenceReady", True))
        if absolute_reference_ready != bool(
            physical.get("absoluteReferenceReady", True)
        ):
            raise ValueError(
                "fusion consumption track absolute-reference state differs from its physical parent"
            )
        numeric_fields = (
            "md",
            "tvd",
            "zMsl",
            *(("depthBelowSrd",) if absolute_reference_ready else ()),
            "x",
            "y",
        )
        if is_p13:
            p13_count += 1
            if bool(consumer.get("supervisionEligible")) or bool(
                consumer.get("trainingEligible")
            ):
                raise ValueError("P13 fusion feature became supervision or training")
            valid_mask = np.asarray(consumer.get("validMask") or (), dtype=bool)
            target_md = np.asarray(consumer.get("md") or (), dtype=float)
            source_md = np.asarray(physical.get("md") or (), dtype=float)
            if (
                target_md.size < 2
                or valid_mask.shape != target_md.shape
                or not np.all(valid_mask)
                or source_md.size < target_md.size
                or not np.all(np.isfinite(target_md))
                or not np.all(np.isfinite(source_md))
                or not np.all(np.diff(target_md) > 0.0)
                or not np.all(np.diff(source_md) > 0.0)
            ):
                raise ValueError("P13 fusion feature has an invalid physical MD parent")
            insertion = np.searchsorted(source_md, target_md, side="left")
            right = np.clip(insertion, 0, source_md.size - 1)
            left = np.clip(insertion - 1, 0, source_md.size - 1)
            use_right = np.abs(source_md[right] - target_md) < np.abs(
                source_md[left] - target_md
            )
            indices = np.where(use_right, right, left)
            if np.any(np.abs(source_md[indices] - target_md) > 1e-6):
                raise ValueError(
                    "P13 fusion feature MD is not a physical-primary subset"
                )
            for field in numeric_fields[1:]:
                consumer_values = np.asarray(consumer.get(field) or (), dtype=float)
                physical_values = np.asarray(physical.get(field) or (), dtype=float)
                if (
                    consumer_values.shape != target_md.shape
                    or physical_values.shape != source_md.shape
                    or not np.all(np.isfinite(consumer_values))
                    or not np.all(np.isfinite(physical_values[indices]))
                    or not np.allclose(
                        consumer_values,
                        physical_values[indices],
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    raise ValueError(
                        f"P13 fusion feature {field} differs from physical primary"
                    )
            quality = np.asarray(consumer.get("registrationQuality") or (), dtype=float)
            twt = np.asarray(consumer.get("twtMean") or (), dtype=float)
            twt_std = np.asarray(consumer.get("twtStd") or (), dtype=float)
            if (
                quality.shape != target_md.shape
                or not np.all(np.isfinite(quality))
                or np.any((quality < 0.0) | (quality > 1.0))
                or twt.shape != target_md.shape
                or not np.all(np.isfinite(twt))
                or not np.all(np.diff(twt) > 0.0)
                or twt_std.shape != target_md.shape
                or np.any(np.isinf(twt_std))
                or np.any(twt_std[np.isfinite(twt_std)] < 0.0)
            ):
                raise ValueError(
                    "P13 fusion feature TWT/uncertainty/quality contract is invalid"
                )
        else:
            for field in (*numeric_fields, "twtMean"):
                consumer_values = np.asarray(consumer.get(field) or (), dtype=float)
                physical_values = np.asarray(physical.get(field) or (), dtype=float)
                if consumer_values.shape != physical_values.shape or not np.allclose(
                    consumer_values,
                    physical_values,
                    rtol=0.0,
                    atol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"physical fusion consumer {field} differs from primary"
                    )
            if list(consumer.get("validMask") or ()) != list(
                physical.get("validMask") or ()
            ):
                raise ValueError(
                    "physical fusion consumer valid mask differs from primary"
                )

    if int(manifest.get("p13_feature_well_count") or 0) != p13_count:
        raise ValueError("P13 feature count differs from fusion consumption tracks")


def _validate_human_accepted_geopath_lineage(
    *,
    task: Mapping[str, Any],
    registration_task_id: str,
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    product: Any,
    learned_fusion_well_ids: list[str],
    lineage_stack: frozenset[str],
) -> None:
    """Allow reviewed GeoPath V3 only when its full immutable chain closes."""

    expected_status = "human_accepted_for_experimental_downstream_use"
    if (
        task.get("task_type") != "well_tie"
        or str(manifest.get("candidate_status") or "") != expected_status
        or str(registration.get("candidate_status") or "") != expected_status
    ):
        raise ValueError("learned GeoPath registration has no human-accepted status")
    if str(manifest.get("registration_id") or "") != registration_task_id:
        raise ValueError("human-accepted GeoPath registration id is inconsistent")

    prediction_task_id = str(manifest.get("candidate_prediction_task_id") or "")
    parent_registration_task_id = str(manifest.get("parent_registration_task_id") or "")
    source_snapshot_id = str(manifest.get("source_snapshot_id") or "")
    if (
        not prediction_task_id
        or not parent_registration_task_id
        or not source_snapshot_id
        or prediction_task_id == registration_task_id
        or parent_registration_task_id == registration_task_id
        or str(task.get("parent_task_id") or "") != prediction_task_id
    ):
        raise ValueError("human-accepted GeoPath lineage ids are incomplete or cyclic")

    candidate_manifest_sha256 = str(
        manifest.get("candidate_manifest_sha256") or ""
    ).casefold()
    candidate_product_sha256 = str(
        manifest.get("candidate_product_sha256") or ""
    ).casefold()

    def _is_sha256(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    if not _is_sha256(candidate_manifest_sha256) or not _is_sha256(
        candidate_product_sha256
    ):
        raise ValueError("human-accepted GeoPath candidate hashes are incomplete")

    accepted_well_ids = [
        str(item) for item in (manifest.get("accepted_well_ids") or []) if str(item)
    ]
    if (
        not accepted_well_ids
        or len(set(accepted_well_ids)) != len(accepted_well_ids)
        or set(accepted_well_ids) != set(learned_fusion_well_ids)
    ):
        raise ValueError("human-accepted GeoPath well selection is inconsistent")
    product_fusion_ids = {
        str(identity)
        for identity, track in product.tracks.items()
        if bool(track.get("fusionReady")) and bool(track.get("inferenceEligible", True))
    }
    if product_fusion_ids != set(accepted_well_ids):
        raise ValueError("human-accepted GeoPath product contains an unreviewed well")

    acceptance_request = dict(task.get("candidate_acceptance_request") or {})
    if (
        acceptance_request.get("confirmation") != "ACCEPT_GEOPATH_CANDIDATE"
        or str(
            acceptance_request.get("expected_candidate_manifest_sha256") or ""
        ).casefold()
        != candidate_manifest_sha256
    ):
        raise ValueError("human-accepted GeoPath request is not bound to the candidate")

    result = dict(task.get("result") or {})
    review = dict(result.get("candidate_review") or {})
    current_task_id = str(task.get("task_id") or registration_task_id)
    if (
        str(review.get("candidate_status") or "") != "awaiting_human_review"
        or str(review.get("acceptance_task_id") or "") != current_task_id
        or str(review.get("prediction_task_id") or "") != prediction_task_id
        or str(review.get("parent_registration_task_id") or "")
        != parent_registration_task_id
        or str(review.get("source_snapshot_id") or "") != source_snapshot_id
        or str(review.get("candidate_manifest_sha256") or "").casefold()
        != candidate_manifest_sha256
        or str(review.get("candidate_product_sha256") or "").casefold()
        != candidate_product_sha256
        or set(str(item) for item in (review.get("accepted_well_ids") or []))
        != set(accepted_well_ids)
    ):
        raise ValueError("human-accepted GeoPath review receipt is inconsistent")

    candidate_review = _geopath_candidate_review_payload(
        prediction_task_id,
        expected_manifest_sha256=candidate_manifest_sha256,
    )
    if (
        str(candidate_review.get("candidate_product_sha256") or "").casefold()
        != candidate_product_sha256
        or str(candidate_review.get("source_snapshot_id") or "") != source_snapshot_id
        or str(candidate_review.get("parent_registration_task_id") or "")
        != parent_registration_task_id
        or not set(accepted_well_ids).issubset(
            set(str(item) for item in (candidate_review.get("well_ids") or []))
        )
    ):
        raise ValueError("human-accepted GeoPath candidate lineage changed")
    acceptance_eligibility = {
        str(item.get("well_id") or ""): bool(item.get("acceptance_eligible"))
        for item in (candidate_review.get("wells") or [])
        if isinstance(item, dict)
    }
    if any(
        not acceptance_eligibility.get(identity, False)
        for identity in accepted_well_ids
    ):
        raise ValueError("human-accepted GeoPath includes an ineligible candidate well")

    prediction_task = _get_task(prediction_task_id)
    prediction_result = dict(prediction_task.get("result") or {})
    prediction = dict(prediction_result.get("prediction") or {})
    if (
        prediction_task.get("task_type") != "model_prediction"
        or prediction_task.get("status") != "completed"
        or str(prediction.get("model_id") or "") != "wellfuse_align_geopath_tie_v1"
        or str(prediction_result.get("registration_task_id") or "")
        != parent_registration_task_id
        or str(prediction_result.get("source_task_id") or "") != source_snapshot_id
    ):
        raise ValueError("human-accepted GeoPath prediction lineage is invalid")

    parent_context = _prediction_registration_context(
        parent_registration_task_id,
        source_task_id=source_snapshot_id,
        _lineage_stack=lineage_stack,
    )
    prediction_provenance = dict(prediction.get("provenance") or {})
    if (
        str(prediction_provenance.get("registration_task_id") or "")
        != parent_registration_task_id
        or str(
            prediction_provenance.get("registration_product_sha256") or ""
        ).casefold()
        != str(parent_context.get("registration_product_sha256") or "").casefold()
    ):
        raise ValueError("human-accepted GeoPath is not bound to its parent product")


def _prediction_registration_context(
    registration_task_id: str | None,
    *,
    source_task_id: str | None,
    source_snapshot_fingerprint: str | None = None,
    _lineage_stack: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Resolve one completed registration product for downstream inference."""

    if not registration_task_id:
        return {}
    lineage_stack = frozenset(_lineage_stack or ())
    if registration_task_id in lineage_stack:
        raise ValueError("registration parent lineage contains a cycle")
    lineage_stack = lineage_stack | {registration_task_id}
    task = _get_task(registration_task_id)
    if task.get("status") != "completed":
        raise ValueError("the requested well-seismic registration has not completed")
    result = task.get("result") or {}
    registration = result.get("registration") or {}
    registered = int(registration.get("registered_well_count") or 0)
    registration_source_snapshot = registration.get("source_snapshot_id")
    if (
        source_task_id
        and registration_source_snapshot
        and str(registration_source_snapshot) != str(source_task_id)
    ):
        raise ValueError(
            "registration and prediction refer to different data snapshots"
        )
    outputs = registration.get("outputs") or {}
    output_files = registration.get("output_files") or {}
    fusion_consumption = registration.get("fusion_consumption_product") or {}
    active_consumption_role = str(
        registration.get("active_consumption_product_role") or ""
    )
    if active_consumption_role not in {
        "",
        "physical_primary",
        "fusion_consumption_product",
    }:
        raise ValueError(
            "registration declares an unsupported consumption product role"
        )
    consumption_well_count = int(registration.get("fusion_consumption_well_count") or 0)
    use_consumption_product = (
        active_consumption_role == "fusion_consumption_product"
        or (not active_consumption_role and consumption_well_count > 0)
    )
    if use_consumption_product and (
        fusion_consumption.get("state") != "available"
        or int(fusion_consumption.get("well_count") or 0) != consumption_well_count
    ):
        raise ValueError(
            "registration declares fusion inputs but its sealed consumption product is unavailable"
        )
    chosen_files = (
        fusion_consumption.get("output_files") or output_files
        if use_consumption_product
        else output_files
    )
    chosen_integrity = (
        fusion_consumption.get("output_integrity")
        or registration.get("output_integrity")
        or {}
        if use_consumption_product
        else registration.get("output_integrity") or {}
    )
    points = (
        Path(
            str(
                chosen_files.get("fusion_consumption_registration_points")
                or chosen_files.get("registration_points")
                or (
                    outputs.get("registration_points")
                    if not use_consumption_product
                    else ""
                )
                or ""
            )
        )
        .expanduser()
        .resolve()
    )
    manifest = (
        Path(
            str(
                chosen_files.get("fusion_consumption_manifest")
                or chosen_files.get("manifest")
                or (outputs.get("manifest") if not use_consumption_product else "")
                or ""
            )
        )
        .expanduser()
        .resolve()
    )
    if not points.is_file():
        raise ValueError(
            "registration_points.csv is missing from the registration task"
        )
    if not manifest.is_file():
        raise ValueError("registration manifest is missing from the registration task")
    fusion_ready_well_ids = _fusion_ready_registration_well_ids(points)
    fusion_ready_count = len(fusion_ready_well_ids)
    if fusion_ready_count <= 0:
        raise ValueError(
            "the requested registration contains no per-well fusion_ready=true tracks"
        )
    declared_fusion_ready = int(registration.get("fusion_ready_well_count") or 0)
    if declared_fusion_ready != fusion_ready_count:
        raise ValueError(
            "registration fusion-ready count does not match the sealed per-well CSV"
        )
    if registered < fusion_ready_count:
        raise ValueError(
            "registration registered-well count is smaller than its fusion-ready set"
        )

    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("registration manifest is not valid JSON") from exc
    if not isinstance(manifest_payload, dict):
        raise ValueError("registration manifest must be a JSON object")
    chosen_product = None
    if manifest_payload.get("contract_version") == "well-seismic.registration.v3":
        try:
            chosen_product = read_registration_product_v3(manifest)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "registration product failed its sealed V3 validation"
            ) from exc
        if chosen_product.points_path != points:
            raise ValueError(
                "registration task and manifest refer to different points files"
            )
    elif use_consumption_product:
        raise ValueError("fusion consumption product must use Registration V3")
    if chosen_product is not None and not use_consumption_product:
        legacy_learned_fusion_tracks = _legacy_learned_fusion_track_ids(chosen_product)
        if legacy_learned_fusion_tracks:
            human_accepted_geopath = _is_human_accepted_geopath_track_set(
                chosen_product,
                legacy_learned_fusion_tracks,
            )
            if not human_accepted_geopath:
                raise ValueError(
                    "legacy Registration V3 contains unreviewed learned P13/geopath "
                    "fusion tracks; rerun well-seismic registration"
                )
            _validate_human_accepted_geopath_lineage(
                task=task,
                registration_task_id=registration_task_id,
                registration=registration,
                manifest=manifest_payload,
                product=chosen_product,
                learned_fusion_well_ids=legacy_learned_fusion_tracks,
                lineage_stack=lineage_stack,
            )
    registration_product_sha256 = str(
        (
            chosen_product.manifest if chosen_product is not None else manifest_payload
        ).get("registration_product_sha256")
        or ""
    )
    expected_product_sha256 = str(
        chosen_integrity.get("fusion_consumption_registration_product_sha256")
        or chosen_integrity.get("registration_product_sha256")
        or ""
    )
    if (
        expected_product_sha256
        and registration_product_sha256.casefold() != expected_product_sha256.casefold()
    ):
        raise ValueError(
            "registration product identity differs from the completed task"
        )
    physical_primary_product_sha256 = registration_product_sha256
    physical_primary_points_sha256 = _sha256_file(points)
    physical_primary_manifest_sha256 = _sha256_file(manifest)
    if use_consumption_product:
        assert chosen_product is not None
        if manifest_payload.get("product_role") != "fusion_consumption_product":
            raise ValueError("registration consumption product role is invalid")
        if int(manifest_payload.get("registered_well_count") or 0) != (
            fusion_ready_count
        ):
            raise ValueError("registration consumption manifest well count mismatch")
        p13_consumer_tracks = [
            track
            for track in chosen_product.tracks.values()
            if "p13" in str(track.get("registrationSource") or "").casefold()
        ]
        if int(manifest_payload.get("p13_feature_well_count") or 0) != len(
            p13_consumer_tracks
        ):
            raise ValueError("P13 fusion feature count differs from its sealed product")
        if (
            p13_consumer_tracks
            and manifest_payload.get("fusion_feature_contract_version")
            != FUSION_FEATURE_TRACK_CONTRACT_VERSION
        ):
            raise ValueError(
                "P13 fusion feature contract version is missing or invalid"
            )
        if any(
            bool(track.get("supervisionEligible"))
            or bool(track.get("trainingEligible"))
            for track in p13_consumer_tracks
        ):
            raise ValueError(
                "P13 fusion features cannot become supervision or training"
            )
        primary_manifest = (
            Path(str(output_files.get("manifest") or "")).expanduser().resolve()
        )
        primary_points = (
            Path(str(output_files.get("registration_points") or ""))
            .expanduser()
            .resolve()
        )
        if not primary_manifest.is_file() or not primary_points.is_file():
            raise ValueError(
                "fusion consumption product has no physical-primary audit product"
            )
        try:
            primary_product = read_registration_product_v3(primary_manifest)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "fusion consumption physical-primary product failed sealed V3 validation"
            ) from exc
        if primary_product.points_path != primary_points:
            raise ValueError(
                "physical-primary task and manifest refer to different points files"
            )
        if any(
            "p13" in str(track.get("registrationSource") or "").casefold()
            or "wellfuse_align" in str(track.get("registrationSource") or "").casefold()
            for track in primary_product.tracks.values()
        ):
            raise ValueError("physical-primary registration unexpectedly contains P13")
        _validate_fusion_consumption_tracks(
            consumption_tracks=chosen_product.tracks,
            physical_tracks=primary_product.tracks,
            manifest=manifest_payload,
        )
        if str(manifest_payload.get("physical_primary_product_sha256") or "") != (
            str(primary_product.manifest.get("registration_product_sha256") or "")
        ):
            raise ValueError(
                "fusion consumption does not bind its physical-primary product"
            )
        primary_manifest_sha256 = _sha256_file(primary_manifest)
        primary_points_sha256 = _sha256_file(primary_points)
        physical_primary_product_sha256 = str(
            primary_product.manifest.get("registration_product_sha256") or ""
        )
        physical_primary_points_sha256 = primary_points_sha256
        physical_primary_manifest_sha256 = primary_manifest_sha256
        if str(manifest_payload.get("physical_primary_manifest_sha256") or "") != (
            primary_manifest_sha256
        ):
            raise ValueError(
                "fusion consumption physical-primary manifest binding mismatch"
            )
        registration_integrity = registration.get("output_integrity") or {}
        if (
            str(
                registration_integrity.get("registration_product_sha256") or ""
            ).casefold()
            != physical_primary_product_sha256.casefold()
        ):
            raise ValueError(
                "physical-primary product identity differs from the completed task"
            )
        if (
            str(
                registration_integrity.get("registration_points_sha256") or ""
            ).casefold()
            != primary_points_sha256.casefold()
        ):
            raise ValueError("physical-primary points differ from the completed task")
        if (
            str(
                registration_integrity.get("registration_manifest_sha256") or ""
            ).casefold()
            != primary_manifest_sha256.casefold()
        ):
            raise ValueError(
                "physical-primary manifest differs from the completed task"
            )
    manifest_snapshot = manifest_payload.get("source_snapshot_id")
    if (
        registration_source_snapshot
        and manifest_snapshot
        and str(manifest_snapshot) != str(registration_source_snapshot)
    ):
        raise ValueError("registration manifest source snapshot lineage mismatch")

    points_sha256 = _sha256_file(points)
    manifest_sha256 = _sha256_file(manifest)
    expected_points_sha = str(
        chosen_integrity.get("fusion_consumption_registration_points_sha256")
        or chosen_integrity.get("registration_points_sha256")
        or manifest_payload.get("registration_points_sha256")
        or ""
    ).casefold()
    expected_manifest_sha = str(
        chosen_integrity.get("fusion_consumption_registration_manifest_sha256")
        or chosen_integrity.get("registration_manifest_sha256")
        or ""
    ).casefold()
    if expected_points_sha and points_sha256.casefold() != expected_points_sha:
        raise ValueError("registration_points.csv content changed after registration")
    if expected_manifest_sha and manifest_sha256.casefold() != expected_manifest_sha:
        raise ValueError("registration manifest content changed after registration")

    registration_snapshot_fingerprint = registration.get(
        "source_snapshot_fingerprint"
    ) or manifest_payload.get("source_snapshot_fingerprint")
    if source_snapshot_fingerprint is None and source_task_id:
        source_task = _get_task(source_task_id)
        source_snapshot = (source_task.get("result") or {}).get("data_snapshot") or {}
        source_snapshot_fingerprint = source_snapshot.get(
            "snapshot_sha256"
        ) or source_snapshot.get("asset_set_sha256")
    if (
        registration_snapshot_fingerprint
        and source_snapshot_fingerprint
        and str(registration_snapshot_fingerprint) != str(source_snapshot_fingerprint)
    ):
        raise ValueError(
            "registration and prediction refer to different snapshot content fingerprints"
        )
    integrity_status = (
        "sha256_verified"
        if expected_points_sha and expected_manifest_sha
        else "legacy_partially_pinned"
    )
    return {
        "registration_task_id": registration_task_id,
        "registration_points_path": str(points),
        "registration_manifest_path": str(manifest),
        "registration_source_snapshot_id": registration_source_snapshot,
        "registration_source_snapshot_fingerprint": registration_snapshot_fingerprint,
        "registration_registered_well_count": registered,
        "registration_fusion_ready_well_count": fusion_ready_count,
        "registration_fusion_ready_well_ids": fusion_ready_well_ids,
        "registration_points_sha256": points_sha256,
        "registration_manifest_sha256": manifest_sha256,
        "registration_product_sha256": registration_product_sha256,
        "registration_physical_primary_product_sha256": (
            physical_primary_product_sha256
        ),
        "registration_physical_primary_points_sha256": (physical_primary_points_sha256),
        "registration_physical_primary_manifest_sha256": (
            physical_primary_manifest_sha256
        ),
        "registration_integrity_status": integrity_status,
        "registration_product_role": (
            "fusion_consumption_product"
            if use_consumption_product
            else "physical_or_reviewed_registration"
        ),
        "registration_physical_primary_points_path": (
            str(output_files.get("registration_points") or "")
            if use_consumption_product
            else str(points)
        ),
        "registration_physical_primary_manifest_path": (
            str(output_files.get("manifest") or "")
            if use_consumption_product
            else str(manifest)
        ),
        "registration_lineage_sha256": registration.get("registration_lineage_sha256"),
        "registration_business_status": registration.get("business_status"),
        "registration_availability": "available",
    }


def _attach_result_display_acceptance(
    result: dict[str, Any],
    *,
    layer_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize and persist a fail-closed display decision for one result.

    The producer-owned gate evidence stays under ``display_acceptance``.  The
    platform recomputes ``display_acceptance_decision`` and never trusts a
    producer-declared display status.  Generic layer roles may fill missing
    panel references, but they cannot fill missing gate evidence.
    """

    contract = result.get("display_acceptance")
    if isinstance(contract, dict):
        normalized_contract = copy.deepcopy(contract)
        declared_panels = normalized_contract.get("panels")
        panels = dict(declared_panels) if isinstance(declared_panels, dict) else {}
        layers = layer_bundle.get("layers") if isinstance(layer_bundle, dict) else None
        if isinstance(layers, list):
            derived_panels = comparison_panels_from_layers(
                [item for item in layers if isinstance(item, dict)]
            )
            for panel_name, descriptor in derived_panels.items():
                panels.setdefault(panel_name, descriptor)
        normalized_contract["panels"] = panels
        result["display_acceptance"] = normalized_contract

    decision = evaluate_result_display_acceptance(result).to_dict()
    result["display_acceptance_decision"] = decision
    return decision


def _stored_standard_result_bundle(
    prediction: Mapping[str, Any],
    *,
    execution_task_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a canonical stored bundle without replaying model validators."""

    raw_bundle = prediction.get("standard_result_bundle")
    integrity = prediction.get("output_integrity")
    if not isinstance(raw_bundle, Mapping) or not isinstance(integrity, Mapping):
        return None
    document = copy.deepcopy(dict(raw_bundle))
    claimed = str(document.pop("bundle_sha256", "")).casefold()
    if (
        document.get("contract_version")
        != "well-seismic.standard-result-bundle.v1"
        or len(claimed) != 64
        or canonical_sha256(document).casefold() != claimed
    ):
        return None
    bundle_task_id = str(document.get("execution_task_id") or "")
    if not bundle_task_id or (
        execution_task_id is not None and bundle_task_id != execution_task_id
    ):
        return None
    if str(document.get("model_id") or "") != str(prediction.get("model_id") or ""):
        return None
    if str(document.get("interpretation_task_id") or "") != str(
        prediction.get("task_id") or ""
    ):
        return None
    bundle_integrity = document.get("output_integrity")
    if not isinstance(bundle_integrity, Mapping) or str(
        bundle_integrity.get("sha256") or ""
    ).casefold() != str(integrity.get("integrity_sha256") or "").casefold():
        return None
    document["bundle_sha256"] = claimed
    return document


def _stored_candidate_visualization_decision(
    prediction: Mapping[str, Any],
    *,
    execution_task_id: str | None = None,
) -> dict[str, Any] | None:
    """Reuse the completion-time decision only with its canonical result bundle."""

    bundle = _stored_standard_result_bundle(
        prediction,
        execution_task_id=execution_task_id,
    )
    decision = prediction.get("candidate_visualization_decision")
    if bundle is None or not isinstance(decision, Mapping):
        return None
    model_id = str(prediction.get("model_id") or "")
    if (
        decision.get("contract_version") != CANDIDATE_DISPLAY_CONTRACT_VERSION
        or str(decision.get("model_id") or "") != model_id
        or not isinstance(decision.get("renderable"), bool)
        or not isinstance(decision.get("reason_codes"), list)
    ):
        return None
    visualization = bundle.get("visualization")
    if not isinstance(visualization, Mapping):
        return None
    if decision.get("renderable") is True and not visualization.get(
        "platform_viewer_url"
    ):
        return None
    return copy.deepcopy(dict(decision))


def _stored_surface_horizon_display_contract(
    prediction: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve the completion-time SurfaceSeg receipt without replaying 3-D arrays."""

    if str(prediction.get("model_id") or "") != "seismic_surface_seg":
        return None
    segmentation = prediction.get("segmentation")
    contract = (
        segmentation.get("horizon_display_contract")
        if isinstance(segmentation, Mapping)
        else None
    )
    gate = candidate.get("direction_gate") if isinstance(candidate, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or contract.get("valid") is not True
        or not isinstance(gate, Mapping)
        or gate.get("horizon_display_contract_valid") is not True
    ):
        return None
    bindings = (
        ("raw_horizon_count", "raw_horizon_count"),
        ("display_horizon_count", "display_horizon_count"),
        ("eligible_horizon_ids", "eligible_horizon_ids"),
        ("suppressed_horizon_ids", "suppressed_horizon_ids"),
    )
    if any(gate.get(gate_key) != contract.get(contract_key) for gate_key, contract_key in bindings):
        return None
    return copy.deepcopy(dict(contract))


def _standard_interactive_visualization_models(
    prediction: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return only adapters that the existing workbench gate will admit.

    Every spatial NPY still has the generic bounded-slice viewer.  This helper
    prevents the standard result entry point from redirecting to the legacy
    workbench when that workbench would immediately reject the same candidate.
    """

    model_id = str(prediction.get("model_id") or "")
    if model_id not in set(supported_prediction_visualization_models()):
        return ()
    if model_id == LAYERPULSE_MODEL_ID:
        decision = evaluate_layerpulse_visualization(prediction)
        return (model_id,) if decision.get("renderable") is True else ()
    stored_bundle = _stored_standard_result_bundle(prediction)
    stored_visualization = (
        stored_bundle.get("visualization")
        if isinstance(stored_bundle, Mapping)
        else None
    )
    if isinstance(stored_visualization, Mapping) and stored_visualization.get(
        "platform_viewer_url"
    ):
        return (model_id,)
    display = evaluate_result_display_acceptance(prediction).to_dict()
    candidate = evaluate_candidate_visualization(prediction)
    if display.get("display_status") == "accepted" or candidate.get("renderable"):
        return (model_id,)
    return ()


def _geopath_candidate_review_payload(
    prediction_task_id: str,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve one immutable GeoPath candidate without promoting it."""

    task = _get_task(prediction_task_id)
    if task.get("task_type") != "model_prediction" or task.get("status") != "completed":
        raise ValueError("GeoPath candidate prediction has not completed")
    result = task.get("result") or {}
    prediction = result.get("prediction") or {}
    if str(prediction.get("model_id") or "") != "wellfuse_align_geopath_tie_v1":
        raise ValueError(
            "only a GeoPathTie-V1 prediction can be reviewed as registration"
        )
    outputs = prediction.get("outputs") or {}
    manifest_path = (
        Path(str(outputs.get("registration_manifest") or "")).expanduser().resolve()
    )
    if not manifest_path.is_file():
        raise ValueError("GeoPath candidate Registration V3 manifest is missing")
    manifest_sha256 = _sha256_file(manifest_path)
    if (
        expected_manifest_sha256
        and manifest_sha256.casefold() != str(expected_manifest_sha256).casefold()
    ):
        raise ValueError(
            "GeoPath candidate manifest changed after the review was opened"
        )
    product = read_registration_product_v3(manifest_path)
    if str(product.manifest.get("candidate_status") or "") != "candidate_not_promoted":
        raise ValueError("GeoPath product is not an unpromoted candidate")
    if any(bool(track.get("fusionReady")) for track in product.tracks.values()):
        raise ValueError("GeoPath candidate unexpectedly contains fusion-ready tracks")
    source_snapshot_id = str(
        result.get("source_task_id")
        or prediction.get("source_snapshot_id")
        or product.manifest.get("source_snapshot_id")
        or ""
    )
    parent_registration_task_id = str(
        result.get("registration_task_id")
        or prediction.get("registration_task_id")
        or ""
    )
    if not source_snapshot_id or not parent_registration_task_id:
        raise ValueError(
            "GeoPath candidate is missing snapshot or parent-registration lineage"
        )
    diagnostic_by_well = {
        str(item.get("well_id") or ""): dict(item)
        for item in (product.manifest.get("geopath_well_diagnostics") or [])
        if isinstance(item, dict) and item.get("well_id")
    }
    well_records = []
    for identity in sorted(product.tracks):
        diagnostic = diagnostic_by_well.get(identity, {})
        well_records.append(
            {
                "well_id": identity,
                "geometry": diagnostic.get("geometry"),
                "accepted_fraction": diagnostic.get("accepted_fraction"),
                "aperture_eligible_fraction": diagnostic.get(
                    "aperture_eligible_fraction"
                ),
                "repair_status": diagnostic.get("repair_status"),
                "repair_reason": diagnostic.get("repair_reason"),
                # Missing eligibility is fail-closed: older candidate products
                # must be rerun before they can be promoted.
                "acceptance_eligible": bool(
                    diagnostic.get("acceptance_eligible", False)
                ),
            }
        )
    return {
        "prediction_task_id": prediction_task_id,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": manifest_sha256,
        "candidate_product_sha256": product.manifest.get("registration_product_sha256"),
        "source_snapshot_id": source_snapshot_id,
        "source_snapshot_fingerprint": product.manifest.get(
            "source_snapshot_fingerprint"
        ),
        "parent_registration_task_id": parent_registration_task_id,
        "wells": well_records,
        "well_ids": sorted(product.tracks),
        "candidate_status": "awaiting_human_review",
        "requires_explicit_well_selection": True,
        "uncertainty_calibrated": all(
            bool(track.get("uncertaintyCalibrated"))
            for track in product.tracks.values()
        ),
    }


def _run_geopath_candidate_acceptance(
    task_id: str,
    *,
    prediction_task_id: str,
    acceptance: RegistrationCandidateAcceptanceRequest,
) -> None:
    """Promote only explicitly reviewed wells into a new Registration V3."""

    try:
        _set_task(
            task_id,
            status="running",
            progress=10,
            message="正在核验轨迹感知井震校正候选与人工选择",
        )
        review = _geopath_candidate_review_payload(
            prediction_task_id,
            expected_manifest_sha256=acceptance.expected_candidate_manifest_sha256,
        )
        product = read_registration_product_v3(review["candidate_manifest_path"])
        diagnostic_by_well = {
            str(item.get("well_id") or ""): dict(item)
            for item in (product.manifest.get("geopath_well_diagnostics") or [])
            if isinstance(item, dict) and item.get("well_id")
        }
        aliases: dict[str, str] = {}
        for identity, track in product.tracks.items():
            for alias in (
                identity,
                str(track.get("well_uid") or ""),
                str(track.get("well_name") or ""),
            ):
                if alias.strip():
                    aliases.setdefault(alias.strip().casefold(), identity)
        requested = [
            item.strip() for item in acceptance.accepted_well_ids if item.strip()
        ]
        if len({item.casefold() for item in requested}) != len(requested):
            raise ValueError("accepted_well_ids contains duplicate well identities")
        missing = sorted(item for item in requested if item.casefold() not in aliases)
        if missing:
            raise ValueError(
                "selected wells are absent from the candidate: " + ", ".join(missing)
            )
        selected_identities = [aliases[item.casefold()] for item in requested]
        accepted_tracks: list[dict[str, Any]] = []
        for identity in selected_identities:
            source_track = product.tracks[identity]
            if not bool(source_track.get("inferenceEligible", True)):
                raise ValueError(f"{identity}: candidate is not inference eligible")
            diagnostic = diagnostic_by_well.get(identity)
            if not diagnostic or not bool(diagnostic.get("acceptance_eligible")):
                raise ValueError(
                    f"{identity}: 轨迹感知井震校正候选需要先完成回退或修复审核，"
                    "当前不具备晋级资格"
                )
            updated = copy.deepcopy(source_track)
            updated.update(
                {
                    "registrationSource": "wellfuse_align_geopath_tie_v1_human_accepted",
                    "registrationStatus": "human_accepted_candidate",
                    "sourceAuthority": "learned_geopath_human_accepted",
                    "fusionReady": True,
                    "supervisionEligible": False,
                    "trainingEligible": False,
                }
            )
            updated["diagnostics"] = {
                **dict(updated.get("diagnostics") or {}),
                "candidate_prediction_task_id": prediction_task_id,
                "candidate_manifest_sha256": review["candidate_manifest_sha256"],
                "human_reviewed": True,
                "human_review_does_not_calibrate_uncertainty": True,
            }
            accepted_tracks.append(updated)
        if not accepted_tracks:
            raise ValueError("at least one candidate well must be selected")

        coordinate = product.manifest.get("coordinate_contract") or {}
        vertical = product.manifest.get("vertical_contract") or {}
        timing = product.manifest.get("time_contract") or {}
        output_directory = (
            PROJECT_ROOT / "model_outputs" / "accepted_registrations" / task_id
        )
        written = write_registration_product_v3(
            output_directory,
            accepted_tracks,
            semantics={
                "horizontal_crs_id": coordinate.get("horizontal_crs_id"),
                "horizontal_unit": coordinate.get("horizontal_unit"),
                "horizontal_axis_order": coordinate.get("horizontal_axis_order"),
                "vertical_crs_id": vertical.get("vertical_crs_id"),
                "seismic_srd_elevation_m": vertical.get("seismic_srd_elevation_m"),
                "time_domain": timing.get("time_domain"),
                "time_reference": timing.get("time_reference"),
                "correction_state": timing.get("correction_state"),
            },
            manifest_fields={
                "registration_id": task_id,
                "source_snapshot_id": review["source_snapshot_id"],
                "source_snapshot_fingerprint": review["source_snapshot_fingerprint"],
                "registered_well_count": len(accepted_tracks),
                "fusion_ready_well_count": len(accepted_tracks),
                "business_status": "experimental_human_accepted_candidate",
                "registration_source_policy": (
                    "explicit per-well human acceptance of immutable GeoPathTie-V1 candidate"
                ),
                "parent_registration_task_id": review["parent_registration_task_id"],
                "candidate_prediction_task_id": prediction_task_id,
                "candidate_manifest_sha256": review["candidate_manifest_sha256"],
                "candidate_product_sha256": review["candidate_product_sha256"],
                "accepted_well_ids": selected_identities,
                "review_note": acceptance.review_note.strip(),
                "uncertainty_calibration_status": "not_calibrated",
                "candidate_status": "human_accepted_for_experimental_downstream_use",
            },
        )
        output_integrity = {
            "registration_points_sha256": written.points_sha256,
            "registration_preview_sha256": written.preview_sha256,
            "registration_manifest_sha256": written.manifest_sha256,
            "registration_product_sha256": written.product_sha256,
        }
        registration_lineage_sha256 = canonical_sha256(
            {
                "registration_id": task_id,
                "source_snapshot_id": review["source_snapshot_id"],
                "source_snapshot_fingerprint": review["source_snapshot_fingerprint"],
                "parent_registration_task_id": review["parent_registration_task_id"],
                "candidate_prediction_task_id": prediction_task_id,
                **output_integrity,
            }
        )
        registration = {
            **written.manifest,
            "can_build_multimodal_view": True,
            "output_integrity": output_integrity,
            "registration_lineage_sha256": registration_lineage_sha256,
            "output_directory": str(output_directory.resolve()),
            "output_files": {
                "manifest": str(written.manifest_path),
                "registration_points": str(written.points_path),
                "registration_preview": str(written.preview_path),
            },
        }
        snapshot_context = _prediction_snapshot_context(review["source_snapshot_id"])
        bundle = _state_store.create_artifact_bundle(
            {
                "kind": "registration",
                "contract_version": registration.get("contract_version"),
                "registration_id": task_id,
                "source_snapshot_id": review["source_snapshot_id"],
                "registration_lineage_sha256": registration_lineage_sha256,
                "candidate_review": review,
                "outputs": dict(registration["output_files"]),
            },
            bundle_id=f"registration-{task_id}",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                review["source_snapshot_id"]
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
            task_id=task_id,
        )
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="轨迹感知井震校正候选已按所选井生成实验性标定成果",
            artifact_bundle_id=bundle["bundle_id"],
            result={
                "registration": registration,
                "source_snapshot_id": review["source_snapshot_id"],
                "candidate_review": {
                    **review,
                    "accepted_well_ids": selected_identities,
                    "acceptance_task_id": task_id,
                },
            },
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="轨迹感知井震校正候选审核失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _prediction_model_context_policies(model_spec: Any) -> tuple[str, str]:
    """Return the auxiliary-input policies declared by one prediction model."""

    metadata = dict(getattr(model_spec, "metadata", {}) or {})
    requires_registration = bool(metadata.get("requires_registration"))
    explicit_registration_policy = metadata.get("registration_policy")
    # Legacy plugins without a policy may still inspect an explicitly supplied
    # registration as an optional control.  Only an explicit ``none`` contract
    # authorizes the control plane to discard that context.
    registration_policy = str(
        explicit_registration_policy
        or ("required" if requires_registration else "optional_control")
    )
    if registration_policy not in {"none", "optional_control", "required"}:
        raise ValueError(
            f"unsupported registration_policy for {model_spec.id}: "
            f"{registration_policy}"
        )
    prepared_view_policy = str(
        metadata.get("prepared_view_policy") or "optional"
    )
    if prepared_view_policy not in {"none", "optional", "preferred", "required"}:
        raise ValueError(
            f"unsupported prepared_view_policy for {model_spec.id}: "
            f"{prepared_view_policy}"
        )
    return registration_policy, prepared_view_policy


def _apply_prediction_model_context_policy(
    request: PredictionRequest,
    model_spec: Any,
) -> tuple[PredictionRequest, str, str]:
    """Remove auxiliary inputs that the selected model contract does not consume.

    The submitted client payload is persisted separately by the API.  This
    normalization applies only to the effective execution request, so a pure
    seismic model cannot accidentally validate or attest stale well-registration
    lineage supplied by a generic workbench form.
    """

    registration_policy, prepared_view_policy = (
        _prediction_model_context_policies(model_spec)
    )
    updates: dict[str, Any] = {}
    options = dict(request.options)
    if registration_policy == "none":
        updates["registration_task_id"] = None
        options = {
            key: value
            for key, value in options.items()
            if str(key) != "registration"
            and not str(key).startswith("registration_")
        }
    if prepared_view_policy == "none":
        updates["prepared_view_task_id"] = None
        options = {
            key: value
            for key, value in options.items()
            if str(key) != "prepared_view"
            and not str(key).startswith("prepared_view_")
        }
    if options != request.options:
        updates["options"] = options
    if updates:
        request = request.model_copy(update=updates)
    return request, registration_policy, prepared_view_policy


_SNAPSHOT_ONLY_DOWNSTREAM_WELL_FORBIDDEN_OPTION_KEYS = frozenset(
    {
        "dataset",
        "dataset_id",
        "registered_dataset",
        "well_ids",
        "well_paths",
        "well_las",
        "las_paths",
        "raw_well_path",
        "raw_well_paths",
        "raw_well_root",
        "trajectory_path",
        "trajectory_paths",
        "wellhead_path",
        "wellhead_paths",
        "input_mode",
        "source_mode",
        # These values are server-derived from source_task_id and may not be
        # injected by a public request.
        "source_snapshot_id",
        "snapshot_assets",
        "snapshot_contract_version",
        "project_id",
    }
)


def _forbidden_downstream_well_source_options(options: Mapping[str, Any]) -> list[str]:
    supplied: list[str] = []

    def visit(value: object, path: str = "options") -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key).strip().casefold().replace("-", "_").replace(" ", "_")
                nested_path = f"{path}.{raw_key}"
                if key in _SNAPSHOT_ONLY_DOWNSTREAM_WELL_FORBIDDEN_OPTION_KEYS:
                    supplied.append(nested_path)
                visit(nested, nested_path)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(options)
    return sorted(dict.fromkeys(supplied))


def _validate_snapshot_only_downstream_well_source(
    request: PredictionRequest,
    snapshot_context: Mapping[str, Any],
    *,
    require_current: bool,
) -> None:
    """Fail closed unless a new well-side run consumes the active sealed snapshot."""

    if request.model_id not in SNAPSHOT_ONLY_DOWNSTREAM_WELL_MODEL_IDS:
        return
    explicit_fields: list[str] = []
    if request.seismic_path.strip():
        explicit_fields.append("seismic_path")
    if request.raw_well_paths:
        explicit_fields.append("raw_well_paths")
    if (request.raw_well_root or "").strip():
        explicit_fields.append("raw_well_root")
    if request.trajectory_paths:
        explicit_fields.append("trajectory_paths")
    explicit_fields.extend(_forbidden_downstream_well_source_options(request.options))
    if explicit_fields:
        raise ValueError(
            "后续井任务只接受当前封存 SourceSnapshot 自动派生的井资产；"
            f"不得提交显式或内置数据来源参数：{', '.join(dict.fromkeys(explicit_fields))}"
        )
    source_snapshot_id = str(request.source_task_id or "").strip()
    if not source_snapshot_id:
        raise ValueError("后续井任务必须绑定当前封存 SourceSnapshot")
    if str(snapshot_context.get("source_snapshot_id") or "") != source_snapshot_id:
        raise ValueError("后续井任务的 SourceSnapshot 身份无法核验")
    if snapshot_context.get("snapshot_contract_version") != SOURCE_SNAPSHOT_CONTRACT_VERSION:
        raise ValueError("后续井任务只接受不可变的 SourceSnapshot V3")
    well_assets = [
        item
        for item in (snapshot_context.get("snapshot_assets") or [])
        if isinstance(item, Mapping)
        and str(item.get("role") or "").strip().casefold() in {"well_log", "well_logs"}
        and item.get("path")
    ]
    if not well_assets:
        raise ValueError("当前封存 SourceSnapshot 没有可用测井资产")
    if not require_current:
        return
    project_id = str(snapshot_context.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("当前封存 SourceSnapshot 未绑定项目")
    try:
        project = _state_store.get_project(project_id)
    except RecordNotFoundError as exc:
        raise ValueError("当前封存 SourceSnapshot 所属项目不存在") from exc
    if str(project.get("active_snapshot_id") or "") != source_snapshot_id:
        raise ValueError("所选 SourceSnapshot 不是项目当前活动快照")


def _validate_snapshot_only_property_source(
    request: PredictionRequest,
    snapshot_context: Mapping[str, Any],
    *,
    require_current: bool,
) -> None:
    """Backward-compatible internal alias for property-policy tests/plugins."""

    _validate_snapshot_only_downstream_well_source(
        request,
        snapshot_context,
        require_current=require_current,
    )


class _FaultPredictionProgressThrottle:
    """Bound SQLite writes while preserving per-patch cancellation checks."""

    def __init__(
        self,
        task_id: str,
        *,
        minimum_interval_seconds: float = 0.5,
        minimum_progress_delta: int = 1,
        clock: Any = monotonic,
    ) -> None:
        self.task_id = task_id
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.minimum_progress_delta = int(minimum_progress_delta)
        self.clock = clock
        self.last_persisted_at: float | None = None
        self.last_persisted_progress: int | None = None

    def __call__(self, progress: int, message: str) -> None:
        # This remains outside the persistence gate: the FaultSeg runner calls
        # progress once per patch, so cancellation stays responsive even when
        # thousands of callbacks map to the same integer percentage.
        TASK_RUNTIME.check_cancelled(self.task_id)
        now = float(self.clock())
        current_progress = int(progress)
        progress_changed = (
            self.last_persisted_progress is None
            or abs(current_progress - self.last_persisted_progress)
            >= self.minimum_progress_delta
        )
        interval_elapsed = (
            self.last_persisted_at is None
            or now - self.last_persisted_at >= self.minimum_interval_seconds
        )
        if not progress_changed and not interval_elapsed:
            return
        _set_task(
            self.task_id,
            status="running",
            progress=current_progress,
            message=message,
        )
        self.last_persisted_progress = current_progress
        self.last_persisted_at = now


def _run_prediction(task_id: str, request: PredictionRequest) -> None:
    fault_progress = (
        _FaultPredictionProgressThrottle(task_id)
        if is_fault_volume_model_id(request.model_id)
        else None
    )

    def update(progress: int, message: str) -> None:
        if fault_progress is not None:
            fault_progress(progress, message)
        else:
            _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="模型推理任务已开始")
        if request.model_id in ARCHIVED_WELL_PROPERTY_COMPLETION_MODEL_IDS:
            raise RuntimeError(
                "历史井侧物性模型已从公共平台归档，仅保留既有成果读取"
            )
        registered_spec = next(
            (
                item
                for item in _model_registry.list_specs()
                if item.id == request.model_id
            ),
            None,
        )
        if registered_spec is None:
            raise KeyError(f"模型尚未注册：{request.model_id}")
        if registered_spec.runtime_status != "runnable":
            raise RuntimeError(
                f"模型 {request.model_id} 当前不可运行；已归档模型仅支持读取历史结果"
            )
        if registered_spec.metadata.get("prediction_task") != request.task_id:
            raise ValueError(
                f"模型 {request.model_id} 不属于解释任务 {request.task_id}"
            )
        request, _registration_policy, _prepared_view_policy = (
            _apply_prediction_model_context_policy(request, registered_spec)
        )
        runtime_options = dict(request.options)
        if request.raw_well_paths:
            runtime_options["raw_well_paths"] = list(request.raw_well_paths)
        if request.raw_well_root and request.raw_well_root.strip():
            runtime_options["raw_well_root"] = request.raw_well_root.strip()
        if request.trajectory_paths:
            runtime_options["trajectory_paths"] = list(request.trajectory_paths)
        snapshot_context = _prediction_snapshot_context(request.source_task_id)
        _validate_snapshot_only_downstream_well_source(
            request,
            snapshot_context,
            require_current=True,
        )
        runtime_options.update(snapshot_context)
        registration_context = (
            _prediction_registration_context(
                request.registration_task_id,
                source_task_id=request.source_task_id,
                source_snapshot_fingerprint=snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
            )
            if request.registration_task_id
            else {}
        )
        runtime_options.update(registration_context)
        prepared_view_context = (
            _prediction_prepared_view_context(
                request.prepared_view_task_id,
                source_snapshot_id=request.source_task_id,
                source_snapshot_sha256=snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                registration_task_id=request.registration_task_id,
            )
            if request.prepared_view_task_id
            else {}
        )
        runtime_options.update(prepared_view_context)
        requires_seismic = bool(registered_spec.metadata.get("requires_seismic", True))
        if requires_seismic:
            source = Path(request.seismic_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"地震文件不存在：{source}")
            if source.suffix.lower() not in {".sgy", ".segy"}:
                raise ValueError("当前三维体推理接口接收 .sgy 或 .segy 文件")
            matched_snapshot_asset = next(
                (
                    item
                    for item in snapshot_context.get("snapshot_assets") or []
                    if str(
                        Path(str(item.get("path"))).expanduser().resolve()
                    ).casefold()
                    == str(source).casefold()
                ),
                None,
            )
            if (
                snapshot_context.get("snapshot_contract_version")
                in {
                    "well-seismic.data.v2",
                    SOURCE_SNAPSHOT_CONTRACT_VERSION,
                }
                and matched_snapshot_asset is None
            ):
                raise ValueError(
                    "prediction seismic source is not part of the sealed source snapshot"
                )
            prediction_source_identity = {
                "kind": "seismic_file",
                "path": str(source),
                "sha256": (
                    matched_snapshot_asset.get("sha256")
                    if matched_snapshot_asset is not None
                    else _sha256_file(source)
                ),
                "size": source.stat().st_size,
                "geometry_fingerprint": (
                    matched_snapshot_asset.get("geometry_fingerprint")
                    if matched_snapshot_asset is not None
                    else None
                ),
                "integrity_status": (
                    matched_snapshot_asset.get("integrity_status")
                    if matched_snapshot_asset is not None
                    else "content_hashed_unbound"
                ),
            }
        else:
            source = (
                Path(
                    os.getenv(
                        "WELLFUSE_PROJECT_ROOT", PROJECT_ROOT / "runtime" / "wellfuse"
                    )
                )
                .expanduser()
                .resolve()
            )
            if not (source / "src" / "wellfuse5090").is_dir():
                raise FileNotFoundError(f"WellFuse项目不存在：{source}")
            if runtime_options.get("raw_well_paths") or runtime_options.get(
                "raw_well_root"
            ):
                prediction_source_identity = {
                    "kind": "raw_well_files",
                    "path": str(source),
                    "raw_well_paths": list(runtime_options.get("raw_well_paths") or []),
                    "raw_well_root": runtime_options.get("raw_well_root"),
                    "trajectory_paths": list(
                        runtime_options.get("trajectory_paths") or []
                    ),
                    "wellhead_paths": list(runtime_options.get("wellhead_paths") or []),
                    "integrity_status": "verified_by_raw_well_adapter_and_cli_manifest",
                }
            elif any(
                str(item.get("role", "")).casefold() in {"well_log", "well_logs"}
                for item in (snapshot_context.get("snapshot_assets") or [])
            ):
                prediction_source_identity = {
                    "kind": "sealed_snapshot_wells",
                    "path": str(source),
                    "source_snapshot_id": request.source_task_id,
                    "source_snapshot_fingerprint": snapshot_context.get(
                        "source_snapshot_fingerprint"
                    ),
                    "integrity_status": snapshot_context.get(
                        "snapshot_integrity_status"
                    ),
                }
            else:
                prediction_source_identity = {
                    "kind": "registered_well_dataset",
                    "path": str(source),
                    "dataset": runtime_options.get("dataset"),
                    "well_ids": list(runtime_options.get("well_ids") or []),
                    "integrity_status": "resolved_by_dataset_runner",
                }
        runtime_options["prediction_source_identity"] = prediction_source_identity
        output = (
            Path(request.output_directory).expanduser().resolve()
            if request.output_directory and request.output_directory.strip()
            else PROJECT_ROOT
            / "model_outputs"
            / f"{request.task_id}_{request.model_id}_{task_id[:8]}"
        )
        result = _prediction_runners.run(
            request.model_id,
            ModelInputRequest(
                source=source,
                crop_start=request.crop_start,
                crop_size=request.crop_size,
                options=runtime_options,
            ),
            adapters=_input_adapters,
            config=_platform_config,
            project_root=PROJECT_ROOT,
            output_directory=output,
            device_name=request.device,
            threshold=request.threshold,
            patch_size=request.patch_size,
            overlap=request.overlap,
            options=runtime_options,
            progress=update,
        )
        registration_consumption = attest_prediction_registration_consumption(
            result,
            registration_requested=bool(request.registration_task_id),
            registration_context=runtime_options,
            task_id=request.task_id,
        )
        result["registration_consumption"] = registration_consumption
        result["registration_consumed"] = registration_consumption[
            "registration_consumed"
        ]
        result["registration_usage"] = registration_consumption["status"]
        prepared_view_consumed = bool(
            (result.get("input") or {}).get("prepared_view_consumed")
            or (result.get("provenance") or {}).get("prepared_view_consumed")
        )
        result["prepared_view_consumption"] = {
            "status": (
                "used"
                if request.prepared_view_task_id and prepared_view_consumed
                else (
                    "available_not_used"
                    if request.prepared_view_task_id
                    else "not_requested"
                )
            ),
            "prepared_view_id": request.prepared_view_task_id,
            "prepared_view_consumed": prepared_view_consumed,
            "manifest_sha256": prepared_view_context.get(
                "prepared_view_manifest_sha256"
            ),
            "view_sha256": prepared_view_context.get("prepared_view_sha256"),
        }
        result["task_id"] = request.task_id
        result["task_name"] = _interpretation_registry.get(request.task_id).name
        result.setdefault("provenance", {}).update(
            {
                "source_snapshot_id": request.source_task_id,
                "source_snapshot_fingerprint": snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                "prediction_source_identity": prediction_source_identity,
                "registration_availability": (
                    runtime_options.get("registration_availability")
                    if request.registration_task_id
                    else "not_requested"
                ),
                "registration_usage": registration_consumption["status"],
                "registration_consumed_attested": registration_consumption[
                    "registration_consumed"
                ],
                "registration_consumption_attestation": registration_consumption,
                "prepared_view_id": request.prepared_view_task_id,
                "prepared_view_sha256": prepared_view_context.get(
                    "prepared_view_sha256"
                ),
                "prepared_view_usage": result["prepared_view_consumption"]["status"],
            }
        )
        if request.source_task_id:
            result["source_snapshot_id"] = request.source_task_id
        if request.registration_task_id:
            result["registration_task_id"] = request.registration_task_id
            result.setdefault("provenance", {}).update(
                {
                    "registration_task_id": request.registration_task_id,
                    "registration_manifest": runtime_options.get(
                        "registration_manifest_path"
                    ),
                    "registration_availability": runtime_options.get(
                        "registration_availability"
                    ),
                    "registration_source_snapshot_fingerprint": runtime_options.get(
                        "registration_source_snapshot_fingerprint"
                    ),
                    "registration_points_sha256": runtime_options.get(
                        "registration_points_sha256"
                    ),
                    "registration_manifest_sha256": runtime_options.get(
                        "registration_manifest_sha256"
                    ),
                    "registration_product_sha256": runtime_options.get(
                        "registration_product_sha256"
                    ),
                    "registration_product_role": runtime_options.get(
                        "registration_product_role"
                    ),
                    "registration_fusion_ready_well_ids": runtime_options.get(
                        "registration_fusion_ready_well_ids"
                    ),
                }
            )
        if request.prepared_view_task_id:
            result["prepared_view_task_id"] = request.prepared_view_task_id
        _separate_prediction_external_evidence(result, output_root=output)
        materialize_standard_spatial_slice_bundle(
            result,
            output_root=output,
            execution_task_id=task_id,
        )
        native_output_integrity = _seal_prediction_output_integrity(
            result,
            producer_task_id=task_id,
            output_root=output,
        )
        standard_manifest_path = write_standard_result_manifest(
            result,
            output_root=output,
            native_output_integrity=native_output_integrity,
            execution_task_id=task_id,
        )
        result["output_integrity"] = append_standard_manifest_integrity(
            native_output_integrity,
            manifest_path=standard_manifest_path,
            output_root=output,
        )
        if request.model_id == "wellfuse_align_geopath_tie_v1":
            candidate_manifest = (
                Path(
                    str(
                        (result.get("outputs") or {}).get("registration_manifest") or ""
                    )
                )
                .expanduser()
                .resolve()
            )
            candidate_product = read_registration_product_v3(candidate_manifest)
            diagnostic_by_well = {
                str(item.get("well_id") or ""): dict(item)
                for item in (
                    candidate_product.manifest.get("geopath_well_diagnostics") or []
                )
                if isinstance(item, dict) and item.get("well_id")
            }
            candidate_wells = [
                {
                    "well_id": identity,
                    "geometry": diagnostic_by_well.get(identity, {}).get("geometry"),
                    "accepted_fraction": diagnostic_by_well.get(identity, {}).get(
                        "accepted_fraction"
                    ),
                    "aperture_eligible_fraction": diagnostic_by_well.get(
                        identity, {}
                    ).get("aperture_eligible_fraction"),
                    "repair_status": diagnostic_by_well.get(identity, {}).get(
                        "repair_status"
                    ),
                    "repair_reason": diagnostic_by_well.get(identity, {}).get(
                        "repair_reason"
                    ),
                    "acceptance_eligible": bool(
                        diagnostic_by_well.get(identity, {}).get(
                            "acceptance_eligible", False
                        )
                    ),
                }
                for identity in sorted(candidate_product.tracks)
            ]
            result["candidate_review"] = {
                "status": "awaiting_human_review",
                "candidate_manifest_sha256": _sha256_file(candidate_manifest),
                "candidate_product_sha256": candidate_product.manifest.get(
                    "registration_product_sha256"
                ),
                "well_ids": sorted(candidate_product.tracks),
                "wells": candidate_wells,
                "requires_explicit_well_selection": True,
                "uncertainty_calibrated": all(
                    bool(track.get("uncertaintyCalibrated"))
                    for track in candidate_product.tracks.values()
                ),
                "accept_endpoint": (
                    f"/api/v1/registration/candidates/{quote(task_id, safe='')}/accept"
                ),
                "confirmation": "ACCEPT_GEOPATH_CANDIDATE",
            }
        release_id = (
            str(registered_spec.metadata.get("release_id"))
            if registered_spec and registered_spec.metadata.get("release_id")
            else request.model_id
        )
        layer_bundle: dict[str, Any] | None = None
        try:
            visualization_result: Mapping[str, Any] = result
            if str(result.get("model_id") or "") == LAYERPULSE_MODEL_ID:
                # SEG-Y/CSV files are business downloads for the same eleven
                # native NPY result layers.  Keep them out of the generic
                # viewer adapter so every head appears once, not once per
                # transport format.
                common_exports = result.get("layerpulse_common_exports")
                exported_items = (
                    common_exports.get("exports")
                    if isinstance(common_exports, Mapping)
                    else None
                )
                download_keys = {
                    str(item.get("artifact_key") or "")
                    for item in (exported_items or [])
                    if isinstance(item, Mapping) and item.get("artifact_key")
                }
                if download_keys:
                    visualization_copy = copy.deepcopy(result)
                    visualization_outputs = visualization_copy.get("outputs")
                    if isinstance(visualization_outputs, dict):
                        for download_key in download_keys:
                            visualization_outputs.pop(download_key, None)
                    visualization_result = visualization_copy
            layer_bundle = prediction_result_to_artifact_bundle(
                visualization_result,
                bundle_id=f"prediction-{task_id}",
                release_id=release_id,
                artifact_url_template=(
                    f"/api/v1/tasks/{quote(task_id, safe='')}"
                    "/layer-artifacts/{artifact_id}"
                ),
            ).to_dict()
            result["artifact_bundle"] = layer_bundle
        except (TypeError, ValueError) as exc:
            # A legacy plugin may return only a non-spatial preview.  The old
            # model-specific visualizer remains valid while the missing generic
            # layer mapping is recorded as a compatibility warning.
            result["warnings"] = [
                *(str(item) for item in (result.get("warnings") or [])),
                f"通用成果图层未登记：{exc}",
            ]
        display_acceptance = _attach_result_display_acceptance(
            result,
            layer_bundle=layer_bundle,
        )
        candidate_visualization = evaluate_candidate_visualization(result)
        if str(result.get("model_id") or "") in SUPPORTED_CANDIDATE_MODEL_IDS:
            result["candidate_visualization_decision"] = candidate_visualization
        result["standard_result_bundle"] = build_standard_result_bundle(
            result,
            execution_task_id=task_id,
            interactive_model_ids=_standard_interactive_visualization_models(result),
        )
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message=f"{result['task_name']}推理完成",
            result={
                "prediction": result,
                "source_task_id": request.source_task_id,
                "registration_task_id": request.registration_task_id,
            },
        )
        try:
            bundle = _state_store.create_artifact_bundle(
                {
                    "contract_version": "well-seismic.artifact-bundle.v1",
                    "name": f"{result['task_name']} · {result.get('model_name', request.model_id)}",
                    "model_id": request.model_id,
                    "interpretation_task_id": request.task_id,
                    "source_task_id": request.source_task_id,
                    "registration_task_id": request.registration_task_id,
                    "source_snapshot_fingerprint": snapshot_context.get(
                        "source_snapshot_fingerprint"
                    ),
                    "prediction_source_identity": prediction_source_identity,
                    "registration_lineage_sha256": runtime_options.get(
                        "registration_lineage_sha256"
                    ),
                    "outputs": dict(result.get("outputs") or {}),
                    "layer_bundle": layer_bundle,
                    "standard_result_bundle": copy.deepcopy(
                        result.get("standard_result_bundle")
                    ),
                    "display_acceptance": display_acceptance,
                    "display_acceptance_contract_version": (
                        RESULT_DISPLAY_ACCEPTANCE_CONTRACT_VERSION
                    ),
                    "display_acceptance_contract": copy.deepcopy(
                        result.get("display_acceptance")
                    ),
                    "candidate_visualization": copy.deepcopy(
                        result.get("candidate_visualization_decision")
                    ),
                },
                bundle_id=f"prediction-{task_id}",
                project_id=snapshot_context.get("project_id"),
                snapshot_id=(
                    request.source_task_id
                    if snapshot_context.get("snapshot_contract_version")
                    == SOURCE_SNAPSHOT_CONTRACT_VERSION
                    else None
                ),
                task_id=task_id,
            )
            _set_task(task_id, artifact_bundle_id=bundle["bundle_id"])
        except (StateStoreError, TypeError, ValueError) as exc:
            LOGGER.warning("无法登记推理成果包 %s：%s", task_id, exc)
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="模型推理失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _queue_task(
    task_type: str,
    message: str,
    *,
    project_id: str | None = None,
    snapshot_id: str | None = None,
) -> str:
    with _task_reset_lock:
        task_id = uuid.uuid4().hex
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "progress": 0,
            "message": message,
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
            "project_id": project_id,
            "snapshot_id": snapshot_id,
        }
        try:
            _state_store.create_task(
                task,
                task_id=task_id,
                project_id=project_id,
                snapshot_id=snapshot_id,
            )
        except (StateStoreError, TypeError, ValueError) as exc:
            raise RuntimeError(f"无法持久化新任务 {task_id}：{exc}") from exc
        with _tasks_lock:
            _tasks[task_id] = task
    return task_id


def _submit_task(
    executor: ThreadPoolExecutor,
    task_id: str,
    runner: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Atomically register a queued worker against cache-reset cancellation.

    ``_queue_task`` and this helper both share ``_task_reset_lock``.  If reset
    wins the gap between persistence and submission, the queued tombstone is
    observed here and no stale worker is ever appended to an executor FIFO.
    """

    with _task_reset_lock:
        try:
            task = _get_task(task_id)
        except KeyError:
            return False
        if (
            task_id in _cancelled_task_ids
            or str(task.get("status") or "") != "queued"
        ):
            return False
        TASK_RUNTIME.submit(executor, task_id, runner, *args, **kwargs)
        return True


def _set_submission_metadata(task_id: str, **values: Any) -> bool:
    """Persist queued-task metadata unless reset already won the race."""

    try:
        _set_task(task_id, **values)
    except _TaskCancellationRequested:
        return False
    return True


def _task_created_after_submission(
    task_id: str,
    *,
    submitted: bool,
    queued_message: str,
) -> TaskCreated:
    if submitted:
        return TaskCreated(task_id=task_id, status="queued", message=queued_message)
    try:
        status = str(_get_task(task_id).get("status") or "cancelled")
    except KeyError:
        status = "cancelled"
    return TaskCreated(
        task_id=task_id,
        status=status,
        message="任务在提交前已被“清空缓存并重新开始”操作停止",
    )


def _release_document(*, include_artifacts: bool) -> dict[str, Any]:
    releases: list[dict[str, Any]] = []
    scientific_counts: Counter[str] = Counter()
    runtime_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    artifact_count = 0
    available_artifact_count = 0
    for release in _release_catalog.list():
        item = release.to_dict()
        for field in (
            "name",
            "display_name",
            "description",
            "summary",
            "version",
        ):
            if item.get(field):
                item[field] = public_visualization_text(item[field])
        for field in ("warnings", "limitations", "inputs", "outputs"):
            values = item.get(field)
            if isinstance(values, (list, tuple)):
                item[field] = [public_visualization_text(value) for value in values]
        artifacts = item.get("artifacts") or []
        artifact_count += len(artifacts)
        available_artifact_count += sum(
            bool(artifact.get("exists")) for artifact in artifacts
        )
        scientific_counts[item["scientific_status"]] += 1
        runtime_counts[item["runtime_status"]] += 1
        task_counts[item["task_id"]] += 1
        item["artifact_count"] = len(artifacts)
        item["legacy"] = item.get("source") == "legacy"
        item["available_artifact_count"] = sum(
            bool(artifact.get("exists")) for artifact in artifacts
        )
        if include_artifacts:
            public_artifacts: list[dict[str, Any]] = []
            for artifact in artifacts:
                artifact = dict(artifact)
                if artifact.get("name"):
                    artifact["name"] = public_visualization_text(artifact["name"])
                if artifact.get("description"):
                    artifact["description"] = public_visualization_text(
                        artifact["description"]
                    )
                if artifact.get("exists") and artifact.get("integrity_status") not in {
                    "sha256_mismatch",
                    "size_mismatch",
                    "missing",
                }:
                    artifact["download_url"] = (
                        f"/api/v1/releases/{quote(release.id, safe='')}"
                        f"/artifacts/{quote(str(artifact['id']), safe='')}"
                    )
                public_artifacts.append(artifact)
            item["artifacts"] = public_artifacts
            item["precomputed_artifacts"] = public_artifacts
        else:
            item.pop("artifacts", None)
            item.pop("precomputed_artifacts", None)
        releases.append(item)
    return {
        "schema_version": _release_catalog.schema_version,
        "artifact_root": str(_release_catalog.artifact_root),
        "read_only": True,
        "lifecycle_overlay": _release_catalog.lifecycle_overlay_status,
        "release_count": len(releases),
        "releases": releases,
        "summary": {
            "scientific_status": dict(scientific_counts),
            "runtime_status": dict(runtime_counts),
            "task": dict(task_counts),
            "artifact_count": artifact_count,
            "available_artifact_count": available_artifact_count,
        },
    }


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "地层慧眼",
        "version": app.version,
        "instance_id": _SERVICE_INSTANCE_ID,
        "build_contract_version": _SERVICE_BUILD_IDENTITY["contract_version"],
        "build_sha256": _SERVICE_BUILD_IDENTITY["sha256"],
        "build_files": dict(_SERVICE_BUILD_IDENTITY["files"]),
        "wellfuse_runtime_build_contract_version": (
            _WELLFUSE_RUNTIME_BUILD_IDENTITY["contract_version"]
        ),
        "wellfuse_runtime_root": _WELLFUSE_RUNTIME_BUILD_IDENTITY["root"],
        "wellfuse_runtime_build_sha256": (_WELLFUSE_RUNTIME_BUILD_IDENTITY["sha256"]),
        "wellfuse_runtime_build_files": dict(_WELLFUSE_RUNTIME_BUILD_IDENTITY["files"]),
        "wellfuse_python_contract_version": _WELLFUSE_PYTHON_IDENTITY[
            "contract_version"
        ],
        "wellfuse_python_path": _WELLFUSE_PYTHON_IDENTITY["path"],
        "wellfuse_python_size": _WELLFUSE_PYTHON_IDENTITY["size"],
        "wellfuse_python_sha256": _WELLFUSE_PYTHON_IDENTITY["sha256"],
        "wellfuse_python_identity_sha256": _WELLFUSE_PYTHON_IDENTITY["identity_sha256"],
        "wellfuse_python_status": _WELLFUSE_PYTHON_IDENTITY["status"],
        "runtime_mode": (
            INTERFACES_ONLY_MODE if interface_only_enabled() else "full_runtime"
        ),
        "task_models_enabled": not interface_only_enabled(),
    }


@app.get("/api/v1/system/cache")
def get_system_cache() -> dict[str, Any]:
    """Inspect only caches that are safe to regenerate."""

    return _system_cache.inspect()


def _cancel_active_tasks_for_cache_clear() -> list[dict[str, str]]:
    """Cancel every task that was queued or running when reset began.

    The task lock is held across the compare-and-set and the in-memory mirror
    update.  This gives ``_set_task`` and cache reset one ordering point, so a
    worker cannot publish a late completion over the cancellation tombstone.
    """

    active_tasks: dict[str, dict[str, Any]] = {}
    for status in ("queued", "running"):
        for task in _state_store.list_tasks(status=status, limit=10_000):
            task_id = str(task.get("task_id") or "")
            if task_id:
                active_tasks[task_id] = task

    cancelled: list[dict[str, str]] = []
    cancelled_at = _now()
    for task_id in active_tasks:
        updated: dict[str, Any] | None = None
        previous_status = ""
        with _tasks_lock:
            # The initial list is only discovery.  A queued worker can become
            # running before this lock is acquired, so reload its authoritative
            # status and retry the compare-and-set instead of mistaking that
            # active transition for a terminal task.
            for attempt in range(3):
                try:
                    current = _state_store.get_task(task_id)
                except RecordNotFoundError:
                    _tasks.pop(task_id, None)
                    break
                current_status = str(current.get("status") or "")
                if current_status not in {"queued", "running"}:
                    _tasks[task_id] = current
                    break
                _cancelled_task_ids.add(task_id)
                try:
                    updated = _state_store.transition_task(
                        task_id,
                        "cancelled",
                        expected_status=current_status,
                        updates={
                            "progress": 100,
                            "message": "任务已由清空缓存并重新开始操作停止",
                            "error": {
                                "type": "PlatformResetCancellation",
                                "message": "用户要求停止全部任务并清空可重建缓存",
                            },
                            "progress_detail": {
                                "phase": "cancelled",
                                "can_estimate": False,
                            },
                            "cancelled_at": cancelled_at,
                            "cancellation_reason": (
                                "system_cache_clear_and_restart"
                            ),
                            "previous_status": current_status,
                        },
                    )
                except ConcurrentStateError:
                    if attempt == 2:
                        raise RuntimeError(
                            f"无法稳定停止并发任务 {task_id}"
                        ) from None
                    continue
                previous_status = current_status
                _tasks[task_id] = updated
                break
            if updated is None:
                _cancelled_task_ids.discard(task_id)
                continue
        cancelled.append(
            {
                "task_id": task_id,
                "task_type": str(updated.get("task_type") or ""),
                "previous_status": previous_status,
                "status": "cancelled",
            }
        )
    return sorted(cancelled, key=lambda item: item["task_id"])


@app.post("/api/v1/system/cache/clear")
def clear_system_cache(request: SystemCacheClearRequest) -> dict[str, Any]:
    """Stop active tasks, then clear caches without touching durable data."""

    del request  # Validation of the literal confirms request intent and blocks form POSTs.
    with _task_reset_lock:
        runtime_task_ids = set(TASK_RUNTIME.active_task_ids())
        cancelled_tasks = _cancel_active_tasks_for_cache_clear()
        runtime_task_ids.update(
            str(item.get("task_id") or "") for item in cancelled_tasks
        )
        runtime_task_ids.discard("")
        runtime_stop = TASK_RUNTIME.cancel_and_wait(
            sorted(runtime_task_ids),
            timeout_seconds=_CACHE_CLEAR_TASK_STOP_TIMEOUT_SECONDS,
        )
        if not runtime_stop.quiescent:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "task_stop_incomplete",
                    "message": (
                        "部分执行实体尚未停止，缓存未清理；请稍后重试。"
                    ),
                    "tasks_cancelled": len(cancelled_tasks),
                    "cancelled_tasks": cancelled_tasks,
                    "runtime_stop": runtime_stop.to_dict(),
                },
            )
        result = _system_cache.clear()
    return {
        **result,
        "tasks_cancelled": len(cancelled_tasks),
        "cancelled_tasks": cancelled_tasks,
        "runtime_stop": runtime_stop.to_dict(),
    }


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, Any]:
    document = build_platform_capabilities(
        project_root=PROJECT_ROOT,
        platform_config=_platform_config,
        model_registry=_model_registry,
        interpretation_registry=_interpretation_registry,
        fusion_registry=_fusion_registry,
        input_adapters=_input_adapters,
        prediction_runners=_prediction_runners,
    )
    document["artifact_releases"] = _release_document(include_artifacts=False)
    document["persistence"] = {
        "contract_version": "well-seismic.state-store.v1",
        "backend": "sqlite",
        "history_persistent": True,
        "restart_safe": False,
        "completed_history_restart_safe": True,
        "running_task_restart_safe": False,
        "task_endpoint": "/api/v1/tasks/{task_id}",
        "project_endpoint": "/api/v1/projects",
        "snapshot_endpoint": "/api/v1/snapshots/{snapshot_id}",
        "artifact_release_endpoint": "/api/v1/releases",
        "interrupted_task_policy": "manual_recovery_not_exposed_by_http",
    }
    document["runtime_mode"] = {
        "mode": INTERFACES_ONLY_MODE if interface_only_enabled() else "full_runtime",
        "task_models_enabled": not interface_only_enabled(),
        "weights_attached": not interface_only_enabled(),
        "deferred_extension_root": "models/task-models",
        "message": (
            "当前为无任务权重接口模式；模型 ID、输入/输出合同和任务路由已保留，"
            "请安装外部模型扩展后再启用推理。"
            if interface_only_enabled()
            else "完整任务模型运行时已启用。"
        ),
    }
    return document


@app.get("/api/v1/projects")
def list_projects() -> dict[str, Any]:
    projects = _state_store.list_projects()
    return {"contract_version": "well-seismic.projects.v1", "projects": projects}


@app.put("/api/v1/projects/{project_id}/active-snapshot")
def set_project_active_snapshot(
    project_id: str,
    request: ActiveSnapshotRequest,
) -> dict[str, Any]:
    """Persist the desktop selection after proving snapshot ownership."""

    try:
        project = _state_store.get_project(project_id)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    try:
        snapshot = _state_store.get_snapshot(request.snapshot_id)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="数据快照不存在") from exc

    if str(snapshot.get("project_id")) != str(project.get("project_id")):
        raise HTTPException(status_code=409, detail="数据快照不属于该项目")
    if snapshot.get("state") != "sealed":
        raise HTTPException(
            status_code=409,
            detail="只有不可变封存快照可设为当前快照",
        )

    active_snapshot_id = str(snapshot["snapshot_id"])
    updated_project = _state_store.update_project(
        project_id,
        {"active_snapshot_id": active_snapshot_id},
    )
    return {
        "contract_version": "well-seismic.active-snapshot.v1",
        "active_snapshot_id": active_snapshot_id,
        "project": updated_project,
    }


@app.get("/api/v1/projects/{project_id}/snapshots")
def list_project_snapshots(project_id: str) -> dict[str, Any]:
    try:
        project = _state_store.get_project(project_id)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    snapshots = _state_store.list_snapshots(project_id=project_id)
    for snapshot in snapshots:
        created_by_task_id = str(snapshot.get("created_by_task_id") or "")
        if not created_by_task_id:
            continue
        try:
            task = _state_store.get_task(created_by_task_id)
        except RecordNotFoundError:
            continue
        request = task.get("request")
        seismic_paths = (
            request.get("seismic_paths") if isinstance(request, dict) else None
        )
        if isinstance(seismic_paths, list) and seismic_paths:
            primary_path = Path(str(seismic_paths[0]))
            snapshot["display_name"] = primary_path.name
            snapshot["primary_seismic_path"] = str(primary_path)
    return {
        "contract_version": "well-seismic.snapshots.v1",
        "project": project,
        "snapshots": snapshots,
    }


@app.get("/api/v1/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str) -> dict[str, Any]:
    try:
        snapshot = _state_store.get_snapshot(snapshot_id)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="数据快照不存在") from exc
    return {
        "contract_version": "well-seismic.snapshot-detail.v1",
        "snapshot": snapshot,
        "tasks": _state_store.list_tasks(snapshot_id=snapshot_id),
        "artifact_bundles": _state_store.list_artifact_bundles(snapshot_id=snapshot_id),
    }


@app.get("/api/v1/releases")
def list_releases() -> dict[str, Any]:
    """List only explicitly published, read-only model and result releases."""

    return _release_document(include_artifacts=True)


@app.get("/api/v1/releases/{release_id}")
def get_release(release_id: str) -> dict[str, Any]:
    try:
        release = _release_catalog.get(release_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="发布版本不存在") from exc
    all_releases = _release_document(include_artifacts=True)["releases"]
    return next(item for item in all_releases if item["id"] == release.id)


@app.get("/api/v1/releases/{release_id}/artifacts/{artifact_id}")
def get_release_artifact(release_id: str, artifact_id: str) -> FileResponse:
    """Serve one immutable artifact selected from the catalog allowlist."""

    try:
        release = _release_catalog.get(release_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="发布版本不存在") from exc
    artifact = next(
        (item for item in release.artifacts if item.id == artifact_id), None
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="发布产物不存在")
    path = Path(artifact.path).resolve()
    if not artifact.exists or not path.is_file():
        raise HTTPException(status_code=404, detail="发布产物文件不可用")
    if artifact.integrity_status in {"sha256_mismatch", "size_mismatch", "missing"}:
        raise HTTPException(status_code=409, detail="发布产物完整性校验未通过")
    current_size = path.stat().st_size
    if artifact.size_bytes is not None and current_size != artifact.size_bytes:
        raise HTTPException(status_code=409, detail="发布产物在目录加载后发生变化")
    if artifact.sha256 and _sha256_file(path).casefold() != artifact.sha256.casefold():
        raise HTTPException(
            status_code=409, detail="发布产物 SHA-256 在目录加载后发生变化"
        )
    return FileResponse(
        path,
        media_type=artifact.media_type,
        filename=public_visualization_text(path.name),
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Well-Seismic-Integrity": artifact.integrity_status,
        },
    )


@app.get("/api/v1/llm/status")
def llm_status() -> dict[str, Any]:
    return load_llm_settings(_platform_config).public_status()


@app.get("/api/v1/demo-paths")
def demo_paths() -> dict[str, Any]:
    reference = PROJECT_ROOT.parent / "整理版_地震解释与储层反演数据"
    seismic_2d = reference / "01_综合解释训练数据" / "01_二维地震解释"
    seismic_3d = reference / "01_综合解释训练数据" / "02_三维地震解释"
    logs = reference / "01_综合解释训练数据" / "03_井数据" / "02_LAS测井曲线"
    wells = reference / "01_综合解释训练数据" / "03_井数据"
    available = all(path.exists() for path in (seismic_2d, seismic_3d, logs, wells))
    return {
        "available": available,
        "seismic_paths": [str(seismic_2d), str(seismic_3d)] if available else [],
        "survey_paths": [],
        "log_paths": [str(logs)] if available else [],
        "well_paths": [str(wells)] if available else [],
        "time_depth_paths": [],
        "interpretation_paths": [],
        "auxiliary_paths": [],
    }


@app.post("/api/v1/input-discovery")
def input_discovery(request: InputDiscoveryRequest) -> dict[str, Any]:
    resolver = build_decision_resolver(
        _platform_config,
        requested=request.use_llm_fallback,
    )
    try:
        return discover_input_root(
            request.root_path,
            recursive=request.recursive,
            max_files=request.max_files,
            decision_resolver=resolver,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/data-preparation/tasks", response_model=TaskCreated, status_code=202)
def create_data_preparation_task(request: InspectionRequest) -> TaskCreated:
    if not any(
        (
            request.seismic_paths,
            request.survey_paths,
            request.log_paths,
            request.well_paths,
            request.time_depth_paths,
            request.interpretation_paths,
            request.auxiliary_paths,
        )
    ):
        raise HTTPException(status_code=422, detail="至少需要登记一个有效数据路径")
    try:
        _target_required_modalities(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _state_store.ensure_project(
        DEFAULT_PROJECT_ID,
        {"name": "本地默认项目", "kind": "desktop_local"},
    )
    task_id = _queue_task(
        "data_preparation",
        "数据准备任务已进入本地队列",
        project_id=DEFAULT_PROJECT_ID,
    )
    prepared = _set_submission_metadata(
        task_id,
        request=request.model_dump(mode="json"),
    )
    submitted = prepared and _submit_task(
        _executor,
        task_id,
        _run_inspection,
        request,
    )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="数据准备任务已进入本地队列",
    )


@app.post(
    "/api/v1/data-inspection/tasks",
    response_model=TaskCreated,
    status_code=202,
    deprecated=True,
)
def create_inspection_task(request: InspectionRequest) -> TaskCreated:
    return create_data_preparation_task(request)


@app.post(
    "/api/v1/horizontal-registration/tasks",
    response_model=TaskCreated,
    status_code=202,
)
def create_horizontal_registration_task(
    request: HorizontalRegistrationRequest,
) -> TaskCreated:
    """Create a plan-view well/SEG-Y derivative from one sealed snapshot id."""

    effective, snapshot_context = _sealed_horizontal_registration_request(request)
    try:
        _require_horizontal_registration_contract(effective, snapshot_context)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Horizontal work may intentionally accept a well head without a trajectory,
    # but it must never hide a supplied trajectory that preparation failed to
    # parse.  The receipt is sealed and therefore cannot be repaired in-place.
    try:
        source_task = _get_task(request.source_snapshot_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=409,
            detail=_registration_preflight_failure(
                code="source_snapshot_unavailable",
                category="source_snapshot_unavailable",
                reason="引用的数据快照不存在",
                horizontal_fallback_allowed=False,
                requires_new_snapshot=False,
            ),
        ) from exc
    _require_registration_source_quality(effective, source_task)
    idempotency_key = _horizontal_registration_idempotency_key(request, effective)
    # Queue insertion and lookup share one process lock so double-clicks cannot
    # create two workers before either task reaches SQLite.
    with _horizontal_registration_submission_lock:
        reusable = _reusable_horizontal_registration_task(
            source_snapshot_id=request.source_snapshot_id,
            idempotency_key=idempotency_key,
        )
        if reusable is not None:
            status = str(reusable.get("status") or "queued")
            message = (
                "相同无时深水平配准请求已完成，已复用现有任务"
                if status == "completed"
                else "相同无时深水平配准请求正在执行，已复用现有任务"
            )
            return TaskCreated(
                task_id=str(reusable["task_id"]), status=status, message=message
            )
        task_id = _queue_task(
            "horizontal_registration",
            "无时深水平配准任务已进入本地队列",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
        )
        prepared = _set_submission_metadata(
            task_id,
            parent_task_id=request.source_snapshot_id,
            request=request.model_dump(mode="json"),
            effective_snapshot_request=effective.model_dump(mode="json"),
            horizontal_registration_idempotency_key=idempotency_key,
        )
        submitted = prepared and _submit_task(
            _executor,
            task_id,
            _run_horizontal_registration,
            effective,
        )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="无时深水平配准任务已进入本地队列",
    )


@app.post(
    "/api/v1/registration/runtime-contract",
    response_model=RuntimeContractConfirmationResponse,
)
def confirm_runtime_contract(
    request: RuntimeContractConfirmationRequest,
) -> RuntimeContractConfirmationResponse:
    """Seal editable run-first defaults without reopening source data."""

    try:
        return _derive_runtime_contract_snapshot(request)
    except HTTPException:
        raise
    except (KeyError, OSError, RecordNotFoundError, StateStoreError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/api/v1/registration/preflight",
    response_model=RegistrationPreflightResponse,
)
def create_registration_preflight(
    request: RegistrationPreflightRequest,
) -> RegistrationPreflightResponse:
    """Return a registration-ready immutable snapshot without accepting paths."""

    try:
        effective, _snapshot_context = _sealed_horizontal_registration_request(
            HorizontalRegistrationRequest(
                source_snapshot_id=request.source_snapshot_id,
            )
        )
    except HTTPException as exc:
        if exc.status_code != 409:
            raise
        raise HTTPException(
            status_code=409,
            detail=_registration_preflight_failure_from_exception(exc),
        ) from exc
    try:
        _require_registration_semantic_contract(effective)
    except ValueError:
        pass
    else:
        source_task = _get_task(request.source_snapshot_id)
        _require_registration_source_quality(effective, source_task)
        return RegistrationPreflightResponse(
            source_snapshot_id=request.source_snapshot_id,
            derived_snapshot_id=request.source_snapshot_id,
            resolution="reused_complete_snapshot",
            effective_request=effective,
            system_evidence_receipt=None,
            execution_contract={
                "mode": "absolute_reference",
                "absolute_reference_ready": True,
                "time_depth_supervision_is_model_input": False,
            },
        )

    try:
        return _derive_registration_ready_snapshot(
            request.source_snapshot_id,
            effective,
        )
    except (OSError, StateStoreError, TypeError, ValueError) as formal_exc:
        source_task: dict[str, Any] | None = None
        try:
            source_task = _get_task(request.source_snapshot_id)
            native_contract = _native_relative_registration_contract(
                effective,
                source_task,
            )
        except (KeyError, OSError, TypeError, ValueError) as native_exc:
            reason = (
                f"正式绝对基准合同不可用：{formal_exc}；"
                f"无TD native-relative合同也不可用：{native_exc}"
            )
            source_quality_issues = (
                []
                if source_task is None
                else _registration_source_quality_issues(effective, source_task)
            )
            integrity_failure = any(
                _is_source_snapshot_integrity_failure(exc)
                for exc in (formal_exc, native_exc)
            )
            source_quality_failure = bool(source_quality_issues)
            try:
                _require_horizontal_registration_contract(
                    effective,
                    _snapshot_context,
                )
            except ValueError as horizontal_exc:
                horizontal_contract_failure = True
                reason += f"；geometry fallback合同也不可用：{horizontal_exc}"
            else:
                horizontal_contract_failure = False
            raise HTTPException(
                status_code=409,
                detail=_registration_preflight_failure(
                    code="formal_and_native_relative_contract_unavailable",
                    category=(
                        "source_snapshot_integrity"
                        if integrity_failure
                        else (
                            "source_quality_unavailable"
                            if source_quality_failure
                            else "formal_contract_unavailable"
                        )
                    ),
                    reason=reason,
                    horizontal_fallback_allowed=not (
                        integrity_failure
                        or source_quality_failure
                        or horizontal_contract_failure
                    ),
                    requires_new_snapshot=integrity_failure or source_quality_failure,
                ),
            ) from native_exc
        return RegistrationPreflightResponse(
            source_snapshot_id=request.source_snapshot_id,
            derived_snapshot_id=request.source_snapshot_id,
            resolution="reused_native_relative_snapshot",
            effective_request=effective,
            system_evidence_receipt=None,
            execution_contract=native_contract,
        )


@app.post("/api/v1/registration/tasks", response_model=TaskCreated, status_code=202)
def create_registration_task(request: PreprocessingRequest) -> TaskCreated:
    """Create the mandatory well-tie stage used before time-domain fusion."""
    source_task = _validate_source_snapshot(request)
    if not request.seismic_paths or not request.log_paths:
        raise HTTPException(
            status_code=422, detail="井震标定至少需要地震数据和LAS测井数据"
        )
    execution_contract: dict[str, Any]
    try:
        _require_registration_semantic_contract(request)
    except ValueError as formal_exc:
        try:
            execution_contract = _native_relative_registration_contract(
                request,
                source_task,
            )
        except (OSError, TypeError, ValueError) as native_exc:
            raise HTTPException(
                status_code=409,
                detail=(f"{formal_exc}；无TD native-relative路径不可用：{native_exc}"),
            ) from native_exc
    else:
        execution_contract = {
            "mode": "absolute_reference",
            "absolute_reference_ready": True,
            "time_depth_supervision_is_model_input": False,
        }
    if execution_contract.get("mode") == "absolute_reference":
        _require_registration_source_quality(request, source_task)
    snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
    idempotency_key = _workflow_submission_idempotency_key("well_tie", request)
    with _workflow_submission_lock:
        reusable = _reusable_workflow_task(
            task_type="well_tie",
            source_snapshot_id=str(request.source_snapshot_id),
            idempotency_key=idempotency_key,
        )
        if reusable is not None:
            status = str(reusable.get("status") or "queued")
            return TaskCreated(
                task_id=str(reusable["task_id"]),
                status=status,
                message=(
                    "相同井震精细标定请求已完成，已复用现有任务"
                    if status == "completed"
                    else "相同井震精细标定请求正在执行，已复用现有任务"
                ),
            )
        _assert_no_workflow_output_directory_conflict(
            request,
            idempotency_key=idempotency_key,
        )
        task_id = _queue_task(
            "well_tie",
            "井震标定任务已进入本地队列",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
        )
        prepared = _set_submission_metadata(
            task_id,
            parent_task_id=request.source_snapshot_id,
            request=request.model_dump(mode="json"),
            execution_contract=execution_contract,
            workflow_submission_idempotency_key=idempotency_key,
        )
        # P13 runs the frozen CUDA ensemble.  Keep registration on the same
        # single-GPU queue as interpretation inference to prevent WDDM contention.
        submitted = prepared and _submit_task(
            _gpu_executor,
            task_id,
            _run_registration,
            request,
        )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="井震标定任务已进入本地队列",
    )


@app.post(
    "/api/v1/registration/candidates/{prediction_task_id}/accept",
    response_model=TaskCreated,
    status_code=202,
)
def accept_registration_candidate(
    prediction_task_id: str,
    request: RegistrationCandidateAcceptanceRequest,
) -> TaskCreated:
    """Create a new immutable registration from explicitly selected wells."""

    try:
        review = _geopath_candidate_review_payload(
            prediction_task_id,
            expected_manifest_sha256=request.expected_candidate_manifest_sha256,
        )
        candidate_ids = {str(item).casefold() for item in review.get("well_ids") or []}
        selected_ids = [
            item.strip() for item in request.accepted_well_ids if item.strip()
        ]
        if len({item.casefold() for item in selected_ids}) != len(selected_ids):
            raise ValueError("accepted_well_ids contains duplicate well identities")
        if not selected_ids:
            raise ValueError("at least one candidate well must be selected")
        unknown = sorted(
            item for item in selected_ids if item.casefold() not in candidate_ids
        )
        if unknown:
            raise ValueError(
                "selected wells are absent from the candidate: " + ", ".join(unknown)
            )
        eligibility = {
            str(item.get("well_id") or "").casefold(): bool(
                item.get("acceptance_eligible")
            )
            for item in review.get("wells") or []
            if isinstance(item, dict)
        }
        ineligible = sorted(
            item for item in selected_ids if not eligibility.get(item.casefold(), False)
        )
        if ineligible:
            raise ValueError(
                "所选轨迹感知井震校正候选井不具备晋级资格: "
                + ", ".join(ineligible)
            )
        parent_registration = _get_task(review["parent_registration_task_id"])
        if (
            parent_registration.get("task_type") != "well_tie"
            or parent_registration.get("status") != "completed"
        ):
            raise ValueError("轨迹感知井震校正所依赖的父级标定成果不可用")
        parent_request = parent_registration.get("request") or {}
        snapshot_context = _prediction_snapshot_context(review["source_snapshot_id"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    task_id = _queue_task(
        "well_tie",
        "轨迹感知井震校正候选审核任务已进入本地队列",
        project_id=snapshot_context.get("project_id"),
        snapshot_id=(
            review["source_snapshot_id"]
            if snapshot_context.get("snapshot_contract_version")
            == SOURCE_SNAPSHOT_CONTRACT_VERSION
            else None
        ),
    )
    # Preserve the original registration path contract so the accepted product
    # can build a PreparedView without bypassing snapshot/path consistency.
    prepared = _set_submission_metadata(
        task_id,
        parent_task_id=prediction_task_id,
        request=parent_request,
        candidate_acceptance_request=request.model_dump(mode="json"),
    )
    submitted = prepared and _submit_task(
        _executor,
        task_id,
        _run_geopath_candidate_acceptance,
        prediction_task_id=prediction_task_id,
        acceptance=request,
    )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="轨迹感知井震校正候选审核任务已进入本地队列",
    )


def _validate_registration_task(request: PreprocessingRequest) -> dict[str, Any]:
    _validate_source_snapshot(request)
    if not request.registration_task_id:
        raise HTTPException(
            status_code=409, detail="必须先完成井震标定，才能构建时间域融合视图"
        )
    try:
        task = _get_task(request.registration_task_id)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail="引用的井震标定任务不存在") from exc
    registration = (task.get("result") or {}).get("registration") or {}
    if task.get("task_type") != "well_tie" or task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="引用的井震标定任务尚未完成")
    if registration.get("source_snapshot_id") != request.source_snapshot_id:
        raise HTTPException(status_code=409, detail="井震标定成果不属于当前数据快照")
    if not registration.get("can_build_multimodal_view"):
        raise HTTPException(
            status_code=409, detail="当前标定未产生可用TWT候选，不能构建融合视图"
        )
    if _request_path_contract(request) != _request_path_contract(
        task.get("request") or {}
    ):
        raise HTTPException(status_code=409, detail="当前路径与井震标定任务不一致")
    try:
        snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
        _prediction_registration_context(
            request.registration_task_id,
            source_task_id=request.source_snapshot_id,
            source_snapshot_fingerprint=snapshot_context.get(
                "source_snapshot_fingerprint"
            ),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return task


@app.post("/api/v1/sample-building/tasks", response_model=TaskCreated, status_code=202)
def create_sample_building_task(request: PreprocessingRequest) -> TaskCreated:
    _validate_registration_task(request)
    if not request.seismic_paths:
        raise HTTPException(status_code=422, detail="至少需要一个地震数据路径")
    if not request.log_paths:
        raise HTTPException(status_code=422, detail="至少需要一个测井数据路径")
    snapshot_context = _prediction_snapshot_context(request.source_snapshot_id)
    idempotency_key = _workflow_submission_idempotency_key("sample_building", request)
    with _workflow_submission_lock:
        reusable = _reusable_workflow_task(
            task_type="sample_building",
            source_snapshot_id=str(request.source_snapshot_id),
            idempotency_key=idempotency_key,
        )
        if reusable is not None:
            status = str(reusable.get("status") or "queued")
            return TaskCreated(
                task_id=str(reusable["task_id"]),
                status=status,
                message=(
                    "相同融合视图构建请求已完成，已复用现有任务"
                    if status == "completed"
                    else "相同融合视图构建请求正在执行，已复用现有任务"
                ),
            )
        _assert_no_workflow_output_directory_conflict(
            request,
            idempotency_key=idempotency_key,
        )
        task_id = _queue_task(
            "sample_building",
            "融合视图构建任务已进入本地队列",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_snapshot_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
        )
        prepared = _set_submission_metadata(
            task_id,
            parent_task_id=request.registration_task_id,
            request=request.model_dump(mode="json"),
            workflow_submission_idempotency_key=idempotency_key,
        )
        submitted = prepared and _submit_task(
            _executor,
            task_id,
            _run_preprocessing,
            request,
        )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="融合视图构建任务已进入本地队列",
    )


@app.post(
    "/api/v1/data-preparation/multimodal-view-tasks",
    response_model=TaskCreated,
    status_code=202,
)
def create_multimodal_data_view_task(request: PreprocessingRequest) -> TaskCreated:
    """Build the optional well-seismic view inside the data-preparation layer."""
    return create_sample_building_task(request)


@app.post(
    "/api/v1/preprocessing/tasks",
    response_model=TaskCreated,
    status_code=202,
    deprecated=True,
)
def create_preprocessing_task(request: PreprocessingRequest) -> TaskCreated:
    return create_sample_building_task(request)


@app.post("/api/v1/raw-well-files")
async def upload_raw_well_file(request: Request, filename: str) -> dict[str, Any]:
    """Store one browser-uploaded well or trajectory file for local inference."""

    safe_name = Path(filename).name.strip()
    if not safe_name or safe_name != filename.strip():
        raise HTTPException(status_code=422, detail="文件名不能包含目录")
    if Path(safe_name).suffix.casefold() not in {
        ".las",
        ".csv",
        ".txt",
    }:
        raise HTTPException(
            status_code=422,
            detail="仅支持 LAS、CSV 或 TXT 井数据文件",
        )
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail="上传文件为空")
    if len(content) > 128 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单个井文件不能超过 128 MiB")
    sha256 = hashlib.sha256(content).hexdigest()
    RAW_WELL_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    output = RAW_WELL_UPLOAD_ROOT / f"{sha256[:16]}_{safe_name}"
    if not output.is_file():
        output.write_bytes(content)
    return {
        "name": safe_name,
        "path": str(output.resolve()),
        "size": len(content),
        "sha256": sha256,
    }


@app.post("/api/v1/prediction/tasks", response_model=TaskCreated, status_code=202)
def create_prediction_task(request: PredictionRequest) -> TaskCreated:
    # Preserve the client contract separately from the effective request.  The
    # server may deterministically inject the sole sealed SEG-Y path and its
    # verified header contract below; keeping both forms makes retries and
    # external validation unambiguous without weakening the effective lineage.
    submitted_request = request.model_dump(mode="json")
    if request.model_id in ARCHIVED_WELL_PROPERTY_COMPLETION_MODEL_IDS:
        raise HTTPException(
            status_code=422,
            detail="历史井侧物性模型已归档；请选择当前整井物性预测模型",
        )
    try:
        _interpretation_registry.get(request.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_specs = list(_model_registry.list_specs())
    adapter_ids = {item["model_id"] for item in _input_adapters.capabilities()}
    runner_ids = set(_prediction_runners.model_ids())
    if request.model_id is None:
        candidates = [
            spec
            for spec in model_specs
            if spec.metadata.get("prediction_task") == request.task_id
            and spec.runtime_status == "runnable"
            and spec.metadata.get("public_prediction_enabled") is not False
            and spec.id in adapter_ids
            and spec.id in runner_ids
        ]
        if not candidates:
            raise HTTPException(
                status_code=422,
                detail=f"解释任务 {request.task_id} 没有可运行模型",
            )
        if request.task_id == "well_property":
            raise HTTPException(
                status_code=422,
                detail=(
                    "储层物性包含DEN、POR、LOG_PERM、SW和VSH；" "请明确选择一个物性模型"
                ),
            )
        # Prefer the F3 volume route because it has a complete result/viewer
        # contract.  For mixed well-only/aligned tasks, a request that already
        # carries registration selects the aligned model; otherwise the
        # well-only route avoids manufacturing an unnecessary prerequisite.
        if request.task_id == "facies_3d":
            candidates.sort(key=lambda item: item.id != "wellfuse_facies_3d_f3_fast")
        elif request.registration_task_id:
            candidates.sort(
                key=lambda item: not bool(item.metadata.get("requires_registration"))
            )
        else:
            candidates.sort(
                key=lambda item: bool(item.metadata.get("requires_registration"))
            )
        request = request.model_copy(update={"model_id": candidates[0].id})
    try:
        model_spec = next(spec for spec in model_specs if spec.id == request.model_id)
    except StopIteration as exc:
        raise HTTPException(
            status_code=422, detail=f"模型尚未注册：{request.model_id}"
        ) from exc
    if model_spec.runtime_status != "runnable":
        archived_suffix = (
            "；仅保留历史结果下载与可视化"
            if model_spec.metadata.get("archived")
            else ""
        )
        raise HTTPException(
            status_code=422,
            detail=f"模型 {request.model_id} 当前不可运行{archived_suffix}",
        )
    requires_seismic = bool(model_spec.metadata.get("requires_seismic", True))
    is_direct12b = request.model_id == DIRECT12B_MODEL_ID
    if model_spec.metadata.get("prediction_task") != request.task_id:
        raise HTTPException(
            status_code=422,
            detail=f"模型 {request.model_id} 不属于解释任务 {request.task_id}",
        )
    requested_faultseg_scope = str(
        request.options.get("faultseg_scope") or ""
    ).strip().casefold()
    if (
        request.task_id == "fault"
        and requested_faultseg_scope == FAULTSEG_REPRESENTATIVE_SCOPE
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "历史 128 代表块模式仅供已有结果只读查看；"
                "新断层预测必须使用 center_block_1 或 full_volume"
            ),
        )
    snapshot_context: dict[str, Any] = {}
    if request.source_task_id:
        try:
            snapshot_context = _prediction_snapshot_context(request.source_task_id)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if request.model_id in SNAPSHOT_ONLY_DOWNSTREAM_WELL_MODEL_IDS:
        try:
            _validate_snapshot_only_downstream_well_source(
                request,
                snapshot_context,
                require_current=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Direct-12B owns its raw-to-exact11 materialisation boundary.  Its SEG-Y
    # path lives inside the validated raw bundle, while ``source_task_id`` is
    # retained only as immutable platform lineage.  Do not inject the generic
    # volume-model seismic path (or SEG-Y adapter options) into this request.
    if requires_seismic and not is_direct12b and not request.seismic_path.strip():
        snapshot_seismic_paths = [
            str(item["path"])
            for item in (snapshot_context.get("snapshot_assets") or [])
            if str(item.get("role") or "").strip().casefold() == "seismic"
            and item.get("path")
            and Path(str(item["path"])).suffix.casefold() in {".sgy", ".segy"}
        ]
        if len(snapshot_seismic_paths) == 1:
            request = request.model_copy(
                update={"seismic_path": snapshot_seismic_paths[0]}
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "封存快照中没有唯一可用的三维 SEG-Y 文件；"
                    "仅在存在多个候选时才需要明确选择"
                    if request.source_task_id
                    else "请选择一个三维 SEG-Y 文件"
                ),
            )
    if requires_seismic and not is_direct12b and snapshot_context:
        semantics = snapshot_context.get("source_snapshot_semantics") or {}
        runtime_options = dict(request.options)
        raw_segy_options = runtime_options.get("segy")
        segy_options = (
            dict(raw_segy_options) if isinstance(raw_segy_options, dict) else {}
        )
        for semantic_name, adapter_name in (
            ("segy_geometry_profile", "profile"),
            ("segy_inline_byte", "inline_byte"),
            ("segy_crossline_byte", "crossline_byte"),
            ("segy_x_byte", "x_byte"),
            ("segy_y_byte", "y_byte"),
            ("segy_coordinate_scalar_byte", "coordinate_scalar_byte"),
        ):
            value = semantics.get(semantic_name)
            if value is not None:
                segy_options.setdefault(adapter_name, value)
        if segy_options:
            runtime_options["segy"] = segy_options
        for semantic_name, runner_name in (
            ("segy_inline_byte", "iline_byte"),
            ("segy_crossline_byte", "xline_byte"),
            ("segy_x_byte", "x_byte"),
            ("segy_y_byte", "y_byte"),
            ("segy_coordinate_scalar_byte", "coordinate_scalar_byte"),
        ):
            value = semantics.get(semantic_name)
            if value is not None:
                runtime_options.setdefault(runner_name, value)
        request = request.model_copy(update={"options": runtime_options})
    if is_direct12b:
        try:
            if any(
                (
                    request.seismic_path.strip(),
                    request.raw_well_paths,
                    (request.raw_well_root or "").strip(),
                    request.trajectory_paths,
                    request.prepared_view_task_id,
                    request.registration_task_id,
                )
            ):
                raise ValueError(
                    "多模态井震对齐原始输入只能通过受验raw_bundle或封存11-key manifest提供；"
                    "不接受独立SEG-Y/LAS/轨迹路径、Registration、预处理视图或时深资产"
                )
            validate_direct12b_request_options(request.options, public_request=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        request, registration_policy, prepared_view_policy = (
            _apply_prediction_model_context_policy(request, model_spec)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    requires_registration = registration_policy == "required"
    if requires_registration and not request.registration_task_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"model {request.model_id} requires a completed well-seismic "
                "registration task from the same data snapshot"
            ),
        )
    if request.registration_task_id and not request.source_task_id:
        raise HTTPException(
            status_code=409,
            detail="registration-backed prediction requires its sealed source snapshot id",
        )
    if request.registration_task_id:
        try:
            _prediction_registration_context(
                request.registration_task_id,
                source_task_id=request.source_task_id,
                source_snapshot_fingerprint=snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Workbench predictions are source-backed and must wait for fusion. Older
    # standalone dataset/raw-well runners have no SourceSnapshot lineage to
    # bind, so keep that external compatibility path unchanged.
    if (
        request.task_id != "alignment"
        and request.source_task_id
        and not request.prepared_view_task_id
        and prepared_view_policy != "none"
    ):
        compatible_prepared_view_id = _latest_compatible_prepared_view_task_id(
            source_snapshot_id=request.source_task_id,
            source_snapshot_sha256=snapshot_context.get(
                "source_snapshot_fingerprint"
            ),
            registration_task_id=request.registration_task_id,
        )
        if not compatible_prepared_view_id:
            raise HTTPException(
                status_code=409,
                detail="当前数据快照尚无已完成的井震融合视图，请先完成融合再预测",
            )
        request = request.model_copy(
            update={"prepared_view_task_id": compatible_prepared_view_id}
        )
    if request.prepared_view_task_id:
        if not request.source_task_id:
            raise HTTPException(
                status_code=409,
                detail="prepared-view prediction requires its sealed source snapshot id",
            )
        try:
            prepared_view_context = _prediction_prepared_view_context(
                request.prepared_view_task_id,
                source_snapshot_id=request.source_task_id,
                source_snapshot_sha256=snapshot_context.get(
                    "source_snapshot_fingerprint"
                ),
                registration_task_id=request.registration_task_id,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if (
            request.task_id != "alignment"
            and request.source_task_id
            and not prepared_view_context.get("prepared_view_registration_parent_ids")
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "当前 PreparedView 未绑定已完成的井震标定，"
                    "不能作为正式融合视图解锁预测"
                ),
            )
    if request.model_id not in adapter_ids:
        raise HTTPException(
            status_code=422, detail=f"模型没有已注册的输入适配器：{request.model_id}"
        )
    if request.model_id not in runner_ids:
        raise HTTPException(
            status_code=422, detail=f"模型没有已注册的推理运行器：{request.model_id}"
        )
    effective_request = request.model_dump(mode="json")
    with _prediction_submission_lock:
        for active_status in ("queued", "running"):
            active_tasks = _state_store.list_tasks(
                status=active_status,
                task_type="model_prediction",
                limit=10_000,
            )
            duplicate = next(
                (
                    item
                    for item in active_tasks
                    if item.get("submitted_request") == submitted_request
                    or item.get("request") == effective_request
                ),
                None,
            )
            if duplicate is not None:
                return TaskCreated(
                    task_id=str(duplicate["task_id"]),
                    status=str(duplicate["status"]),
                    message="相同模型推理请求已在队列中，已复用现有任务",
                )
        task_id = _queue_task(
            "model_prediction",
            "模型推理任务已进入本地队列",
            project_id=snapshot_context.get("project_id"),
            snapshot_id=(
                request.source_task_id
                if snapshot_context.get("snapshot_contract_version")
                == SOURCE_SNAPSHOT_CONTRACT_VERSION
                else None
            ),
        )
        if (
            is_fault_volume_model_id(request.model_id)
            and not str(request.output_directory or "").strip()
        ):
            faultseg_output_base = _faultseg_prediction_output_base(
                request.seismic_path
            )
            request = request.model_copy(
                update={
                    "output_directory": str(
                        (
                            faultseg_output_base
                            / f"{request.task_id}_{request.model_id}_{task_id[:8]}"
                        ).resolve()
                    )
                }
            )
            effective_request = request.model_dump(mode="json")
        prepared = _set_submission_metadata(
            task_id,
            parent_task_id=request.source_task_id,
            submitted_request=submitted_request,
            request=effective_request,
        )
        submitted = prepared and _submit_task(
            _gpu_executor,
            task_id,
            _run_prediction,
            request,
        )
    return _task_created_after_submission(
        task_id,
        submitted=submitted,
        queued_message="模型推理任务已进入本地队列",
    )


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    if (
        task.get("task_type") == "model_prediction"
        and task.get("status") == "completed"
    ):
        try:
            task = _ensure_prediction_standard_results(task_id, task)
        except (OSError, TypeError, ValueError) as exc:
            # A historical result can outlive a manually removed output file.
            # Keep the completed task readable, but never publish an unsealed
            # or path-based fallback download.
            task = copy.deepcopy(task)
            task["standard_result_error"] = {
                "code": "legacy_result_could_not_be_sealed",
                "message": str(exc),
            }
    return task


def _legacy_prediction_output_root(
    task: Mapping[str, Any], prediction: Mapping[str, Any]
) -> Path:
    """Infer a narrow trusted root for a historical completed prediction."""

    declared = _declared_output_paths(prediction.get("outputs") or {})
    if not declared:
        raise ValueError("historical prediction has no declared output files")
    parent_paths = [path if path.is_dir() else path.parent for _, path in declared]
    common = Path(os.path.commonpath([str(path) for path in parent_paths])).resolve()
    request = task.get("request")
    request = request if isinstance(request, Mapping) else {}
    requested_root = str(request.get("output_directory") or "").strip()
    trusted_roots = [
        (PROJECT_ROOT / "model_outputs").resolve(),
        (PROJECT_ROOT / "outputs").resolve(),
    ]
    if requested_root:
        trusted_roots.append(Path(requested_root).expanduser().resolve())
    if not any(
        common == trusted or common.is_relative_to(trusted) for trusted in trusted_roots
    ):
        raise ValueError(
            "historical prediction outputs are outside the trusted result roots"
        )
    for _, path in declared:
        try:
            path.relative_to(common)
        except ValueError as exc:
            raise ValueError(
                "historical prediction outputs do not share one root"
            ) from exc
    return common


def _standard_result_manifest_binding_is_current(
    prediction: Mapping[str, Any], *, task_id: str
) -> bool:
    """Verify the small final manifest and its final-integrity membership."""

    try:
        build_standard_result_bundle(
            prediction,
            execution_task_id=task_id,
        )
        outputs = prediction.get("outputs")
        integrity = prediction.get("output_integrity")
        if not isinstance(outputs, Mapping) or not isinstance(integrity, Mapping):
            return False
        path = Path(str(outputs.get("standard_result_manifest_json") or "")).resolve()
        artifacts = integrity.get("artifacts")
        record = (
            artifacts.get("standard_result_manifest_json")
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(record, Mapping) or not path.is_file():
            return False
        before = _snapshot_file_stat_signature(path)
        if before[0] != int(record.get("size") or -1):
            return False
        observed = _sha256_file(path).casefold()
        after = _snapshot_file_stat_signature(path)
        return (
            before == after and observed == str(record.get("sha256") or "").casefold()
        )
    except (OSError, TypeError, ValueError):
        return False


def _integrity_without_standard_manifest(
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    document = copy.deepcopy(dict(integrity))
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("prediction output-integrity artifacts are unavailable")
    artifacts.pop("standard_result_manifest_json", None)
    document.pop("integrity_sha256", None)
    document["integrity_sha256"] = canonical_sha256(document)
    return document


def _verify_sealed_prediction_output_file(
    prediction: Mapping[str, Any],
    integrity: Mapping[str, Any],
    *,
    output_key: str,
) -> None:
    """Recheck one completion-sealed file before interruption recovery reuses it."""

    outputs = prediction.get("outputs")
    artifacts = integrity.get("artifacts")
    if not isinstance(outputs, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("prediction output-integrity bindings are unavailable")
    record = artifacts.get(output_key)
    if not isinstance(record, Mapping) or record.get("kind") != "file":
        raise ValueError(f"sealed standard output has no file record: {output_key}")
    path = Path(str(outputs.get(output_key) or "")).expanduser().resolve()
    sealed_path = Path(str(record.get("path") or "")).expanduser().resolve()
    if path != sealed_path:
        raise ValueError(f"sealed standard output path drifted: {output_key}")
    try:
        _verify_snapshot_file_identity(
            path,
            expected_size=int(record.get("size") or -1),
            expected_sha256=str(record.get("sha256") or ""),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"sealed standard output changed after completion: {output_key}"
        ) from exc


def _ensure_prediction_standard_results(
    task_id: str, task: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach the v1 standard contract, lazily sealing legacy completed tasks."""

    if task.get("task_type") != "model_prediction" or task.get("status") != "completed":
        raise ValueError("standard results require a completed prediction task")
    # The overwhelmingly common read path is already sealed.  Validate its
    # small final manifest before acquiring any migration lock so one legacy
    # multi-gigabyte result cannot stall every unrelated completed task.
    fast_result_wrapper = task.get("result")
    fast_prediction = (
        fast_result_wrapper.get("prediction")
        if isinstance(fast_result_wrapper, Mapping)
        else None
    )
    if (
        isinstance(fast_prediction, Mapping)
        and _stored_standard_result_bundle(
            fast_prediction,
            execution_task_id=task_id,
        )
        is not None
        and _standard_result_manifest_binding_is_current(
            fast_prediction,
            task_id=task_id,
        )
    ):
        return copy.deepcopy(dict(task))

    with _standard_result_migration_lock_for(task_id):
        current = _get_task(task_id)
        result_wrapper = copy.deepcopy(current.get("result") or {})
        prediction = result_wrapper.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError("completed prediction task has no prediction result")
        changed = False
        outputs = prediction.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("completed prediction has no outputs mapping")
        standard_manifest_current = _standard_result_manifest_binding_is_current(
            prediction,
            task_id=task_id,
        )
        if not standard_manifest_current:
            # Historical task records predate both the completion-time sealing
            # gate and the competition-facing standard export.  Rebuild the
            # complete contract from the existing immutable files once.
            _separate_prediction_external_evidence(
                prediction,
                output_root=(PROJECT_ROOT / "model_outputs").resolve(),
            )
            existing_integrity = prediction.get("output_integrity")
            reusable_integrity: dict[str, Any] | None = None
            if isinstance(existing_integrity, Mapping):
                reusable_integrity = _integrity_without_standard_manifest(
                    existing_integrity
                )
            reusable_artifacts = (
                reusable_integrity.get("artifacts")
                if isinstance(reusable_integrity, Mapping)
                else {}
            )
            reusable_artifacts = (
                reusable_artifacts if isinstance(reusable_artifacts, Mapping) else {}
            )
            # Exporter-owned files are trusted only when they reached the same
            # completion seal.  The final manifest is always rebuilt; a sealed
            # slice/preview is preserved byte-for-byte and reverified instead
            # of being silently replaced by the bounded legacy exporter.
            outputs.pop("standard_result_manifest_json", None)
            migration_exporter_output_keys = (
                "standard_slice_bundle_zip",
                "standard_preview_png",
                "fault_mask_slice_summary_csv",
                "fault_mask_audit_json",
            )
            for exporter_output_key in migration_exporter_output_keys:
                if exporter_output_key in reusable_artifacts:
                    _verify_sealed_prediction_output_file(
                        prediction,
                        reusable_integrity,
                        output_key=exporter_output_key,
                    )
                else:
                    outputs.pop(exporter_output_key, None)
            legacy_root = _legacy_prediction_output_root(current, prediction)
            if reusable_integrity is not None:
                validation_prediction = copy.deepcopy(prediction)
                validation_prediction["output_integrity"] = reusable_integrity
                build_standard_result_bundle(
                    validation_prediction,
                    execution_task_id=task_id,
                    interactive_model_ids=_standard_interactive_visualization_models(
                        validation_prediction
                    ),
                )
            if "standard_slice_bundle_zip" in reusable_artifacts:
                spatial_export = prediction.get("standard_spatial_export")
                if not isinstance(spatial_export, Mapping):
                    raise ValueError(
                        "sealed standard slice bundle has no exporter receipt"
                    )
                if "standard_preview_png" not in reusable_artifacts:
                    recover_standard_preview_from_slice_bundle(
                        prediction,
                        output_root=legacy_root,
                        execution_task_id=task_id,
                    )
            else:
                sealed_partial_standard = sorted(
                    key
                    for key in migration_exporter_output_keys
                    if key != "standard_slice_bundle_zip"
                    and key in reusable_artifacts
                )
                complete_attempt: Mapping[str, Any]
                if sealed_partial_standard:
                    complete_attempt = {
                        "status": "not_attempted_partial_sealed_standard_outputs",
                        "reason": (
                            "immutable partial standard exports were preserved: "
                            + ", ".join(sealed_partial_standard)
                        ),
                    }
                else:
                    complete_attempt = materialize_standard_spatial_slice_bundle(
                        prediction,
                        output_root=legacy_root,
                        execution_task_id=task_id,
                    )
                if complete_attempt.get("status") == (
                    "complete_2d_slice_bundle_available"
                ):
                    spatial_export = dict(complete_attempt)
                    spatial_export["migration_mode"] = (
                        "complete_roi_revalidated"
                    )
                    prediction["standard_spatial_export"] = spatial_export
                else:
                    spatial_export = materialize_legacy_bounded_spatial_slice_bundle(
                        prediction,
                        output_root=legacy_root,
                        execution_task_id=task_id,
                    )
                    spatial_export["complete_roi_revalidation"] = {
                        "status": str(complete_attempt.get("status") or "unavailable"),
                        "reason": str(complete_attempt.get("reason") or "ROI could not be proven"),
                    }
                    prediction["standard_spatial_export"] = spatial_export
            if reusable_integrity is None:
                native_output_integrity = _seal_prediction_output_integrity(
                    prediction,
                    producer_task_id=task_id,
                    output_root=legacy_root,
                )
            else:
                native_output_integrity = reusable_integrity
                authoritative_key = str(
                    spatial_export.get("authoritative_output_key") or ""
                )
                authoritative_sha256 = str(
                    spatial_export.get("authoritative_output_sha256") or ""
                ).casefold()
                authoritative_record = (
                    (native_output_integrity.get("artifacts") or {}).get(
                        authoritative_key
                    )
                    if authoritative_key
                    else None
                )
                if authoritative_sha256 and (
                    not isinstance(authoritative_record, Mapping)
                    or authoritative_sha256
                    != str(authoritative_record.get("sha256") or "").casefold()
                ):
                    raise ValueError(
                        "primary spatial result changed after its original completion seal"
                    )
                for exporter_output_key in (
                    "standard_slice_bundle_zip",
                    "standard_preview_png",
                    "fault_mask_slice_summary_csv",
                    "fault_mask_audit_json",
                    "fault_mask_sgy",
                ):
                    output_path = outputs.get(exporter_output_key)
                    if (
                        exporter_output_key not in reusable_artifacts
                        and isinstance(output_path, str)
                        and output_path.strip()
                    ):
                        native_output_integrity = append_output_file_integrity(
                            native_output_integrity,
                            output_key=exporter_output_key,
                            output_path=Path(output_path),
                            output_root=legacy_root,
                        )
            standard_manifest_path = write_standard_result_manifest(
                prediction,
                output_root=legacy_root,
                native_output_integrity=native_output_integrity,
                execution_task_id=task_id,
            )
            prediction["output_integrity"] = append_standard_manifest_integrity(
                native_output_integrity,
                manifest_path=standard_manifest_path,
                output_root=legacy_root,
            )
            changed = True
        standard_bundle = build_standard_result_bundle(
            prediction,
            execution_task_id=task_id,
            interactive_model_ids=_standard_interactive_visualization_models(
                prediction
            ),
        )
        if prediction.get("standard_result_bundle") != standard_bundle:
            prediction["standard_result_bundle"] = standard_bundle
            changed = True
        result_wrapper["prediction"] = prediction
        if changed:
            _set_task(task_id, result=result_wrapper)
            current = _get_task(task_id)
        else:
            current = copy.deepcopy(current)
            current["result"] = result_wrapper
        return current


def _standard_prediction_result(task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    if task.get("task_type") != "model_prediction" or task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="仅已完成的推理任务具有标准结果")
    try:
        task = _ensure_prediction_standard_results(task_id, task)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail=f"标准结果完整性不可用：{exc}"
        ) from exc
    prediction = (task.get("result") or {}).get("prediction")
    if not isinstance(prediction, dict):
        raise HTTPException(status_code=409, detail="推理任务没有标准结果")
    return task, prediction


@app.get("/api/v1/tasks/{task_id}/standard-results/manifest")
def get_standard_result_manifest(task_id: str) -> dict[str, Any]:
    """Return the path-free integrity-bound result manifest."""

    _task, prediction = _standard_prediction_result(task_id)
    stored = _stored_standard_result_bundle(
        prediction,
        execution_task_id=task_id,
    )
    if stored is not None:
        return stored
    return build_standard_result_bundle(
        prediction,
        execution_task_id=task_id,
        interactive_model_ids=_standard_interactive_visualization_models(prediction),
    )


@app.get("/api/v1/tasks/{task_id}/standard-results/artifacts/{artifact_id}")
def get_standard_result_artifact(task_id: str, artifact_id: str) -> FileResponse:
    """Download one sealed result through its opaque standard artifact id."""

    _task, prediction = _standard_prediction_result(task_id)
    try:
        artifact = resolve_standard_result_artifact(
            prediction,
            execution_task_id=task_id,
            artifact_id=artifact_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="标准结果产物不存在") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="标准结果产物已丢失") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail=f"标准结果完整性失败：{exc}"
        ) from exc
    return FileResponse(
        artifact.path,
        filename=artifact.filename,
        media_type=artifact.media_type,
        content_disposition_type="attachment",
        headers={
            "Cache-Control": "private, no-store",
            "ETag": f'"sha256-{artifact.sha256}"',
            "X-Well-Seismic-SHA256": artifact.sha256,
            "X-Well-Seismic-Size": str(artifact.size_bytes),
        },
    )


@app.get("/api/v1/tasks/{task_id}/layerpulse-exports/{output_key}.{file_format}")
def get_layerpulse_common_export(
    task_id: str,
    output_key: str,
    file_format: str,
) -> FileResponse:
    """Download one LayerPulse head in an interoperable, value-stable format.

    New tasks publish these files through the sealed standard-result bundle.
    This endpoint also supports historical completed tasks by deriving one
    result on demand from its still-sealed NPY head and source SEG-Y geometry.
    """

    task, prediction = _standard_prediction_result(task_id)
    if str(prediction.get("model_id") or "") != LAYERPULSE_MODEL_ID:
        raise HTTPException(status_code=404, detail="该任务不是 LayerPulse 结果")
    try:
        spec = resolve_layerpulse_output_spec(output_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="LayerPulse 任务结果不存在") from exc
    resolved_format = str(file_format).strip().casefold()
    if resolved_format not in {"sgy", "csv"}:
        raise HTTPException(status_code=404, detail="不支持的 LayerPulse 下载格式")
    if resolved_format == "csv" and spec.kind != "classification":
        raise HTTPException(status_code=404, detail="连续属性没有分类码表")
    artifact_key = (
        layerpulse_segy_artifact_key(spec)
        if resolved_format == "sgy"
        else layerpulse_class_legend_artifact_key(spec)
    )
    integrity = prediction.get("output_integrity")
    if not isinstance(integrity, Mapping):
        raise HTTPException(status_code=409, detail="LayerPulse 结果尚未完成完整性封存")

    try:
        with _standard_result_migration_lock_for(task_id):
            outputs = prediction.get("outputs")
            artifacts = integrity.get("artifacts")
            existing_is_sealed = (
                isinstance(outputs, Mapping)
                and isinstance(artifacts, Mapping)
                and artifact_key in outputs
                and artifact_key in artifacts
            )
            if existing_is_sealed:
                _verify_sealed_prediction_output_file(
                    prediction,
                    integrity,
                    output_key=artifact_key,
                )
                export_path = Path(str(outputs[artifact_key])).expanduser().resolve()
            else:
                # The native head must remain byte-identical to the completion
                # seal before it can seed a historical common-format export.
                _verify_sealed_prediction_output_file(
                    prediction,
                    integrity,
                    output_key=spec.artifact_key,
                )
                derived_prediction = copy.deepcopy(prediction)
                output_root = _legacy_prediction_output_root(task, prediction)
                materialize_layerpulse_common_exports(
                    derived_prediction,
                    output_root=output_root,
                    only_output_keys={spec.output_key},
                    formats={resolved_format},
                    strict=True,
                )
                derived_outputs = derived_prediction.get("outputs")
                if not isinstance(derived_outputs, Mapping):
                    raise ValueError("LayerPulse common export did not register outputs")
                export_path = Path(
                    str(derived_outputs.get(artifact_key) or "")
                ).expanduser().resolve()
            if not export_path.is_file():
                raise FileNotFoundError("LayerPulse common export file is missing")
            before = _snapshot_file_stat_signature(export_path)
            digest = _sha256_file(export_path).casefold()
            after = _snapshot_file_stat_signature(export_path)
            if before != after or after[0] <= 0:
                raise ValueError("LayerPulse common export changed while opening")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="LayerPulse 下载文件已丢失") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"LayerPulse {resolved_format.upper()} 导出不可用：{exc}",
        ) from exc

    return FileResponse(
        export_path,
        filename=export_path.name,
        media_type=(
            "application/x-segy"
            if resolved_format == "sgy"
            else "text/csv; charset=utf-8"
        ),
        content_disposition_type="attachment",
        headers={
            "Cache-Control": "private, no-store",
            "ETag": f'"sha256-{digest}"',
            "X-Well-Seismic-SHA256": digest,
            "X-Well-Seismic-Size": str(after[0]),
            "X-Well-Seismic-Value-Transform": "none",
        },
    )


@app.get("/api/v1/tasks/{task_id}/standard-results/visualization")
def get_standard_result_visualization(
    task_id: str, artifact_id: str | None = None
) -> Response:
    """Open the 3-D workbench or a bounded LAS/CSV/JSON result viewer."""

    _task, prediction = _standard_prediction_result(task_id)
    bundle = _stored_standard_result_bundle(
        prediction,
        execution_task_id=task_id,
    ) or build_standard_result_bundle(
        prediction,
        execution_task_id=task_id,
        interactive_model_ids=_standard_interactive_visualization_models(prediction),
    )
    platform_viewer_url = bundle["visualization"].get("platform_viewer_url")
    if artifact_id is None and platform_viewer_url:
        return RedirectResponse(str(platform_viewer_url), status_code=307)
    try:
        document = render_standard_result_visualization(
            prediction,
            execution_task_id=task_id,
            artifact_id=artifact_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="标准可视化产物不存在") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="标准可视化产物已丢失") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"标准可视化不可用：{exc}") from exc
    return HTMLResponse(document, headers={"Cache-Control": "private, no-store"})


@app.get("/api/v1/tasks/{task_id}/artifacts/{artifact_name}")
def get_prediction_artifact(task_id: str, artifact_name: str) -> FileResponse:
    """Serve one file explicitly registered by any completed workflow stage."""
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    if task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="仅已完成的任务可以读取产物")
    result = task.get("result") or {}
    outputs: dict[str, Any] = {}
    if task.get("task_type") == "model_prediction":
        prediction = result.get("prediction") or {}
        outputs.update(dict(prediction.get("outputs") or {}))
    elif task.get("task_type") == "well_tie":
        registration = result.get("registration") or {}
        outputs.update(dict(registration.get("outputs") or {}))
        outputs.update(dict(registration.get("output_files") or {}))
    elif task.get("task_type") == "horizontal_registration":
        horizontal = result.get("horizontal_registration") or {}
        outputs.update(dict(horizontal.get("outputs") or {}))
        outputs.update(dict(horizontal.get("output_files") or {}))
    elif task.get("task_type") == "sample_building":
        matching = result.get("matching") or {}
        outputs.update(dict(matching.get("output_files") or {}))
        prepared_view = result.get("prepared_view") or {}
        if prepared_view.get("manifest_path"):
            outputs["prepared_view_manifest"] = prepared_view["manifest_path"]
        for item in prepared_view.get("artifacts") or []:
            if isinstance(item, dict) and item.get("name") and item.get("path"):
                outputs[str(item["name"])] = item["path"]
    elif task.get("task_type") == "data_preparation":
        snapshot = result.get("data_snapshot") or {}
        if snapshot.get("snapshot_manifest_path"):
            outputs["snapshot_manifest"] = snapshot["snapshot_manifest_path"]
    else:
        raise HTTPException(status_code=409, detail="此任务类型没有公开文件产物")
    if artifact_name not in outputs:
        raise HTTPException(status_code=404, detail=f"任务产物不存在：{artifact_name}")
    raw_path = outputs.get(artifact_name)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(status_code=404, detail=f"任务产物不可用：{artifact_name}")
    artifact_path = Path(raw_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"任务产物文件不存在：{artifact_name}"
        )
    return FileResponse(
        artifact_path,
        filename=artifact_path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, no-store"},
    )


@app.get("/api/v1/tasks/{task_id}/layer-artifacts/{artifact_id}")
def get_prediction_layer_artifact(task_id: str, artifact_id: str) -> FileResponse:
    """Resolve an opaque layer artifact id through a completed task manifest."""

    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    prediction = (task.get("result") or {}).get("prediction")
    bundle = prediction.get("artifact_bundle") if isinstance(prediction, dict) else None
    layers = bundle.get("layers") if isinstance(bundle, dict) else None
    if not isinstance(layers, (list, tuple)):
        raise HTTPException(status_code=404, detail="任务没有登记通用成果图层")
    layer = next(
        (
            item
            for item in layers
            if isinstance(item, dict) and item.get("artifact_id") == artifact_id
        ),
        None,
    )
    metadata = layer.get("metadata") if isinstance(layer, dict) else None
    artifact_name = metadata.get("output_key") if isinstance(metadata, dict) else None
    if not isinstance(artifact_name, str) or not artifact_name:
        raise HTTPException(status_code=404, detail="成果图层未绑定安全产物名称")
    return get_prediction_artifact(task_id, artifact_name)


@app.post("/api/v1/tasks/{task_id}/issues/{issue_id}/confirmation")
def confirm_issue(
    task_id: str,
    issue_id: str,
    request: IssueConfirmationRequest,
) -> dict[str, Any]:
    if request.decision not in {"确认采用", "暂不采用"}:
        raise HTTPException(
            status_code=422, detail="decision必须为“确认采用”或“暂不采用”"
        )
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    result = copy.deepcopy(task.get("result") or {})
    preparation = result.get("preparation") or {}
    issue = next(
        (item for item in preparation.get("issues", []) if item.get("id") == issue_id),
        None,
    )
    if issue is None:
        raise HTTPException(status_code=404, detail="当前任务中没有该问题")
    action = request.action.strip() or str(issue.get("recommended_action", ""))
    candidates = set(issue.get("candidate_actions", []))
    if request.decision == "确认采用" and issue.get("blocking"):
        raise HTTPException(
            status_code=409,
            detail=(
                "阻断项不能只改为“已确认”而保持数据合同不变；"
                "请将结构化候选回填到数据合同并重新执行数据准备"
            ),
        )
    if request.decision == "确认采用" and (not action or action not in candidates):
        raise HTTPException(status_code=422, detail="只能确认后端提供的安全候选方案")
    issue["confirmation_status"] = (
        "已确认采用" if request.decision == "确认采用" else "暂不采用"
    )
    issue["confirmed_action"] = action if request.decision == "确认采用" else ""
    issue["confirmed_at"] = _now()
    _set_task(task_id, result=result)
    return dict(issue)


@app.post("/api/v1/tasks/{task_id}/issues/batch-actions")
def batch_issue_actions(
    task_id: str,
    request: BatchIssueActionRequest,
) -> dict[str, Any]:
    """Apply or roll back one auditable batch of safe issue acknowledgements."""

    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    result = copy.deepcopy(task.get("result") or {})
    preparation = result.get("preparation") or {}
    history = preparation.setdefault("decision_batches", [])

    if request.mode == "rollback":
        batch = next(
            (
                item
                for item in reversed(history)
                if item.get("batch_id") == request.batch_id
            ),
            None,
        )
        if batch is None:
            raise HTTPException(status_code=404, detail="没有找到可撤销的一键处理记录")
        by_id = {item.get("id"): item for item in preparation.get("issues", [])}
        for previous in batch.get("previous", []):
            issue = by_id.get(previous.get("id"))
            if issue is not None:
                issue.update(
                    {
                        key: previous.get(key, "")
                        for key in (
                            "confirmation_status",
                            "confirmed_action",
                            "confirmed_at",
                        )
                    }
                )
        batch["rolled_back_at"] = _now()
        _set_task(task_id, result=result)
        return {"preparation": preparation, "batch": batch, "rolled_back": True}

    selected_stages = set(request.stages)
    previous: list[dict[str, Any]] = []
    applied: list[str] = []
    skipped_blocking: list[str] = []
    for issue in preparation.get("issues", []):
        if selected_stages and issue.get("stage") not in selected_stages:
            continue
        if issue.get("confirmation_status") != "待人工确认":
            continue
        if issue.get("blocking") or issue.get("severity") == "错误":
            skipped_blocking.append(str(issue.get("id")))
            continue
        action = str(issue.get("recommended_action") or "").strip()
        if not action or action not in set(issue.get("candidate_actions") or []):
            continue
        previous.append(
            {
                "id": issue.get("id"),
                "confirmation_status": issue.get("confirmation_status", ""),
                "confirmed_action": issue.get("confirmed_action", ""),
                "confirmed_at": issue.get("confirmed_at", ""),
            }
        )
        issue["confirmation_status"] = "已确认采用"
        issue["confirmed_action"] = action
        issue["confirmed_at"] = _now()
        applied.append(str(issue.get("id")))
    batch = {
        "batch_id": uuid.uuid4().hex,
        "mode": "apply_recommended",
        "created_at": _now(),
        "applied_issue_ids": applied,
        "skipped_blocking_issue_ids": skipped_blocking,
        "previous": previous,
    }
    history.append(batch)
    _set_task(task_id, result=result)
    return {
        "preparation": preparation,
        "batch": batch,
        "applied_count": len(applied),
        "skipped_blocking_count": len(skipped_blocking),
        "rolled_back": False,
    }


@app.post("/api/v1/tasks/{task_id}/issues/llm-autofill")
def llm_autofill_issues(task_id: str) -> dict[str, Any]:
    """Apply bounded LLM/rule fills and leave only irreducible survey facts unresolved."""
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    result = copy.deepcopy(task.get("result") or {})
    preparation = result.get("preparation") or {}
    autofilled: list[str] = []
    survey_input: list[str] = []
    for issue in preparation.get("issues", []):
        status = str(issue.get("confirmation_status", ""))
        if issue.get("blocking") or issue.get("severity") == "错误":
            if status in {"待人工确认", "需一次集中补充"}:
                issue["confirmation_status"] = "需一次集中补充"
                issue["resolution_mode"] = "survey_input"
                issue["autofill_validation"] = [
                    "文件证据不足，禁止LLM虚构物理基准、单位或坐标参考"
                ]
                survey_input.append(str(issue.get("id")))
            continue
        if status not in {"待人工确认", "LLM已补全", "系统已自动处理"}:
            continue
        action = str(issue.get("recommended_action") or "").strip()
        candidates = set(issue.get("candidate_actions") or [])
        if status == "待人工确认" and (not action or action not in candidates):
            continue
        source = str(issue.get("recommendation_source") or "规则")
        issue["confirmation_status"] = (
            "LLM已补全" if source == "LLM" else "系统已自动处理"
        )
        issue["resolution_mode"] = (
            "llm_autofill" if source == "LLM" else "rule_autofill"
        )
        issue.setdefault(
            "autofill_patch",
            {
                "operation": "select_bounded_action",
                "action": action,
                "preserve_source": True,
            },
        )
        issue.setdefault(
            "autofill_validation", ["候选值属于后端allowlist", "未修改原始文件"]
        )
        issue["confirmed_action"] = action
        issue["confirmed_at"] = _now()
        autofilled.append(str(issue.get("id")))
    summary = preparation.setdefault("summary", {})
    summary["autofilled"] = sum(
        1
        for issue in preparation.get("issues", [])
        if issue.get("resolution_mode") in {"llm_autofill", "rule_autofill"}
    )
    summary["survey_input_required"] = sum(
        1
        for issue in preparation.get("issues", [])
        if issue.get("resolution_mode") == "survey_input"
    )
    preparation.setdefault("autofill_audit", []).append(
        {
            "created_at": _now(),
            "autofilled_issue_ids": autofilled,
            "survey_input_issue_ids": survey_input,
            "policy": "bounded_llm_fill_then_rule_validation",
        }
    )
    _set_task(task_id, result=result)
    return {
        "preparation": preparation,
        "autofilled_count": len(autofilled),
        "survey_input_required_count": len(survey_input),
    }


@app.post("/api/v1/tasks/{task_id}/issues/{issue_id}/transformation-drafts")
def generate_transformation_draft(task_id: str, issue_id: str) -> dict[str, Any]:
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    preparation = (task.get("result") or {}).get("preparation") or {}
    issue = next(
        (item for item in preparation.get("issues", []) if item.get("id") == issue_id),
        None,
    )
    if issue is None:
        raise HTTPException(status_code=404, detail="当前任务中没有该问题")
    settings = load_llm_settings(_platform_config)
    draft = create_transformation_draft(
        task_id=task_id,
        issue=issue,
        config=_platform_config,
        generator=build_structured_generator(settings),
    )
    with _tasks_lock:
        _transformation_drafts[draft["id"]] = draft
    return draft


@app.post("/api/v1/transformation-drafts/{draft_id}/activation")
def activate_transformation_draft(
    draft_id: str,
    request: TransformationActivationRequest,
) -> dict[str, Any]:
    if request.confirmation != "确认启用":
        raise HTTPException(status_code=422, detail="必须明确提交“确认启用”")
    with _tasks_lock:
        draft = _transformation_drafts.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="转换草案不存在或服务已重启")
    try:
        activate_transformation(draft, TRANSFORMATION_REGISTRY, _platform_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    task_id = str(draft.get("task_id") or "")
    try:
        task = _get_task(task_id)
    except KeyError:
        task = None
    if task is not None:
        result = copy.deepcopy(task.get("result") or {})
        issues = (result.get("preparation") or {}).get("issues", [])
        issue = next(
            (item for item in issues if item.get("id") == draft.get("issue_id")),
            None,
        )
        if issue is not None:
            issue["transformation_draft_id"] = draft_id
            issue["confirmation_status"] = "已启用转换插件"
            issue["confirmed_action"] = draft.get("title", "受控转换适配器")
            issue["confirmed_at"] = draft.get("activated_at", "")
            _set_task(task_id, result=result)
    return dict(draft)


def _assistant_context(task_id: str | None) -> dict[str, Any]:
    if not task_id:
        return {"task": "尚未选择任务"}
    try:
        task = _get_task(task_id)
    except KeyError:
        return {"task": "任务不存在或服务已重启"}
    result = task.get("result") or {}
    preparation = result.get("preparation") or {}
    return {
        "task_id": task_id,
        "task_type": task.get("task_type"),
        "status": task.get("status"),
        "summary": result.get("summary", {}),
        "stages": [
            {
                "id": stage.get("id"),
                "name": stage.get("name"),
                "status": stage.get("status"),
                "issue_count": stage.get("issue_count"),
            }
            for stage in preparation.get("stages", [])
        ],
        "issues": [
            sanitize_llm_payload(
                {
                    "stage": issue.get("stage"),
                    "severity": issue.get("severity"),
                    "title": issue.get("title"),
                    "message": issue.get("message"),
                    "affected_count": issue.get("affected_count", 1),
                    "status": issue.get("confirmation_status"),
                },
                known_paths=issue_local_paths(issue),
            )
            for issue in preparation.get("issues", [])[:12]
        ],
        "gates": preparation.get("gates", {}),
    }


@app.post("/api/v1/assistant/chat")
def assistant_chat(request: AssistantChatRequest) -> dict[str, Any]:
    settings = load_llm_settings(_platform_config)
    generator = build_structured_generator(settings)
    context = _assistant_context(request.task_id)
    allowed_targets = [
        "overview",
        "preparation",
        "visualization",
        "samples",
        "models",
        "prediction",
        "evaluation",
        "settings",
    ]
    if generator is not None:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "actions": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "target": {"type": "string", "enum": allowed_targets},
                        },
                        "required": ["label", "target"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answer", "actions"],
            "additionalProperties": False,
        }
        try:
            response, metadata = generator.generate_json(
                system_prompt=(
                    "你是地层慧眼平台的井震数据工程助手。请只依据提供的任务摘要回答，使用简洁中文；"
                    "不得声称看过未提供的原始SEG-Y或完整LAS，不得编造指标。"
                    "涉及数据修改时说明需要生成受控转换草案并由人工启用。"
                ),
                payload=sanitize_llm_payload(
                    {"question": request.message, "workflow_context": context}
                ),
                schema_name="well_seismic_assistant_response",
                schema=schema,
            )
            return {
                "answer": str(response.get("answer", ""))[:3000],
                "actions": [
                    item
                    for item in response.get("actions", [])
                    if item.get("target") in allowed_targets
                ],
                "source": "LLM",
                **metadata,
            }
        except Exception as exc:
            fallback_error = str(exc)[:300]
    else:
        fallback_error = "LLM未启用或未完成配置"
    issue_count = (
        len(context.get("issues", [])) if isinstance(context.get("issues"), list) else 0
    )
    answer = (
        f"当前任务已记录 {issue_count} 项需要关注的问题。建议先进入“数据准备”，按阶段查看证据；"
        "对无法由知识库确定的映射，可生成受控转换适配器并在自动测试通过后人工启用。"
        if request.task_id
        else "请先执行数据准备。完成后我可以结合当前任务阶段、问题和放行条件给出诊断。"
    )
    return {
        "answer": answer,
        "actions": [{"label": "查看数据准备", "target": "preparation"}],
        "source": "本地工作流助手",
        "provider": "local",
        "model": "workflow-context-v1",
        "request_id": "",
        "warning": fallback_error,
    }


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=503, detail="前端尚未构建，请先运行 npm run build"
        )
    return FileResponse(index, headers={"Cache-Control": "no-store"})


def _dashboard_document(path: Path) -> str:
    """兼容旧的srcdoc打包文件，并在服务端去掉多余iframe外壳。"""
    document = path.read_text(encoding="utf-8-sig")
    marker = 'srcdoc="'
    start = document.find(marker)
    finish = document.rfind('"></iframe>')
    if start >= 0 and finish > start:
        return html.unescape(document[start + len(marker) : finish])
    return document


_EMBEDDED_VISUALIZATION_STYLE = """
<style id="well-seismic-embedded-mode">
html, body {
  min-width: 0 !important;
  color: #20242b !important;
  background: #f3f5f8 !important;
}
body { padding: 0 !important; }
.current-task-banner {
  display: flex;
  min-height: 46px;
  gap: 12px;
  align-items: center;
  padding: 9px 20px;
  color: #4f5966;
  font: 500 13px/1.4 Inter, "Microsoft YaHei UI", sans-serif;
  background: #fff;
  border-bottom: 1px solid #e5e8ee;
}
.current-task-banner strong { color: #1f5fd4; font-size: 14px; }
.current-task-banner span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#future-model-interfaces {
  --font-size-base: 15px;
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
  font-size: 15px !important;
}
#future-model-interfaces .workspace {
  grid-template-columns: 276px minmax(0, 1fr) !important;
  gap: 0 !important;
  align-items: stretch !important;
  background: #f3f5f8 !important;
}
#future-model-interfaces .workspace > .sidebar {
  min-height: 100vh;
  padding: 24px 20px;
  background: #f8f9fb;
  border-right: 1px solid #e5e8ee;
  box-shadow: none;
}
#future-model-interfaces .workspace > .main {
  padding: 20px 22px 32px;
  background: #f3f5f8;
}
#future-model-interfaces .workspace > .sidebar > h2,
#future-model-interfaces #active-panel-label,
#future-model-interfaces .side-nav,
#future-model-interfaces .nav-group-label,
#future-model-interfaces .summary,
#future-model-interfaces [data-side-options]:not([data-side-options="volume"]),
#future-model-interfaces [data-panel-content]:not([data-panel-content="volume"]) {
  display: none !important;
}
#future-model-interfaces [data-side-options="volume"],
#future-model-interfaces [data-panel-content="volume"] {
  display: block !important;
}
#future-model-interfaces .side-options { margin: 0 !important; }
#future-model-interfaces .side-options h3 {
  margin: 0 0 16px;
  color: #252a31;
  font-size: 16px;
  font-weight: 650;
}
#future-model-interfaces .side-options .option-stack { gap: 18px; }
#future-model-interfaces .side-options .option-stack + h3 {
  margin-top: 30px;
  padding-top: 22px;
  border-top: 1px solid #e5e8ee;
}
#future-model-interfaces .form-label,
#future-model-interfaces .form-check-label {
  color: #4e5763;
  font-size: 13px;
}
#future-model-interfaces .form-select {
  min-height: 38px;
  background-color: #fff;
  border-color: #dfe3e9;
  border-radius: 8px;
}
#future-model-interfaces .form-range { accent-color: #2468f2; }
#future-model-interfaces #reset-view {
  min-height: 38px;
  color: #303640;
  background: #fff;
  border-color: #dce0e7;
  border-radius: 8px;
}
#future-model-interfaces [data-panel-content="volume"] {
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#future-model-interfaces #volume-detail {
  min-height: 30px;
  padding: 0 2px 10px;
  color: #7a838f;
  font-size: 12px;
}
#future-model-interfaces .volume3d-stage { margin-top: 0 !important; }
#future-model-interfaces .canvas-label {
  min-height: 32px;
  color: #2b3037;
  font-size: 13px;
  font-weight: 550;
}
#future-model-interfaces .volume3d-canvas {
  min-height: 650px;
  border: 1px solid #e1e5eb;
  border-radius: 10px;
  box-shadow: 0 8px 24px rgb(31 47 68 / 7%);
}
#future-model-interfaces .well-overlay { border-radius: 10px; }
#future-model-interfaces .volume-grid { gap: 12px; }
#future-model-interfaces .volume-grid > div {
  padding: 10px;
  background: #fff;
  border: 1px solid #e5e8ee;
  border-radius: 9px;
}
#future-model-interfaces .seismic-canvas {
  border-color: #e1e5eb;
  border-radius: 6px;
}
@media (min-width: 850px) {
  #future-model-interfaces .volume-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  }
}
@media (max-width: 1100px) {
  #future-model-interfaces .workspace { grid-template-columns: 240px minmax(0, 1fr) !important; }
  #future-model-interfaces .volume3d-canvas { min-height: 480px; }
}
</style>
"""

_EMBEDDED_VISUALIZATION_SCRIPT = """
<script id="well-seismic-embedded-activation">
window.addEventListener("load", () => {
  const root = document.getElementById("future-model-interfaces");
  const volumeButton = root && root.querySelector('[data-panel="volume"]');
  if (volumeButton) volumeButton.click();
});
</script>
"""


def _visualization_unavailable(title: str, message: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<style>body{margin:0;display:grid;min-height:100vh;place-items:center;background:#f3f5f8;"
        "font:500 15px/1.7 Inter,'Microsoft YaHei UI',sans-serif;color:#68717d}.state{max-width:560px;"
        "padding:32px 38px;text-align:center;background:#fff;border:1px solid #e2e6ed;border-radius:14px;"
        "box-shadow:0 14px 40px rgba(40,58,82,.08)}h2{margin:0 0 8px;color:#20252d;font-size:21px}"
        "p{margin:0}</style></head><body><div class='state'>"
        f"<h2>{html.escape(title)}</h2><p>{html.escape(message)}</p></div></body></html>"
    )


def _visualization_failed_diagnostic(
    title: str,
    diagnostics: list[str],
) -> HTMLResponse:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in diagnostics)
    return HTMLResponse(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<style>body{margin:0;display:grid;min-height:100vh;place-items:center;background:#fff7f5;"
        "font:500 15px/1.7 Inter,'Microsoft YaHei UI',sans-serif;color:#704a46}.state{max-width:680px;"
        "padding:32px 38px;background:#fff;border:1px solid #edc9c3;border-radius:14px;"
        "box-shadow:0 14px 40px rgba(82,40,40,.08)}h2{margin:0 0 8px;color:#813f39;font-size:21px}"
        "p{margin:0 0 12px}ul{margin:0;padding-left:20px}li+li{margin-top:6px}</style></head>"
        "<body><div class='state' data-display-status='failed_diagnostic'>"
        f"<h2>{html.escape(title)}</h2><p>结果未通过展示验收，仅开放失败诊断：</p>"
        f"<ul>{items}</ul></div></body></html>",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/v1/visualization/viser-slices", include_in_schema=False)
def move_viser_slices(request: ViserSliceRequest) -> dict[str, Any]:
    if request.x is None and request.y is None and request.z is None:
        raise HTTPException(status_code=422, detail="至少提供一个切片索引")
    try:
        return update_viser_slices(
            request.task_id,
            request.asset_index,
            {"x": request.x, "y": request.y, "z": request.z},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/visualization/viser-layer-mode", include_in_schema=False)
def change_viser_layer_mode(request: ViserLayerModeRequest) -> dict[str, Any]:
    try:
        return update_viser_layer_mode(
            request.task_id,
            request.asset_index,
            request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/cigvis/plotly.min.js", include_in_schema=False)
def cigvis_plotly_bundle() -> Response:
    try:
        source = plotly_javascript(PROJECT_ROOT)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"可视化资源不可用：{public_visualization_text(exc)}",
        ) from exc
    return Response(
        content=source,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _well_binding_identity(item: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return tiered, exact well identifiers without weakening stable IDs."""

    identity = {"uid": set(), "id": set(), "name": set()}
    declared_identity = item.get("wellIdentity")
    source = declared_identity if isinstance(declared_identity, Mapping) else item
    fields = {
        "uid": ("sourceWellUid", "wellUid", "source_well_uid", "well_uid"),
        "id": ("wellId", "well_id"),
        "name": ("name", "wellName", "well_name"),
    }
    for tier, tier_fields in fields.items():
        for field in tier_fields:
            value = str(source.get(field) or "").strip()
            if not value:
                continue
            normalized = normalize_well_name(value)
            if normalized:
                identity[tier].add(normalized)
            if tier == "name":
                base_name = value.split("（", 1)[0].split("(", 1)[0].strip()
                normalized_base = normalize_well_name(base_name)
                if normalized_base:
                    identity[tier].add(normalized_base)
    return identity


def _well_binding_keys(item: Mapping[str, Any]) -> set[str]:
    """Return the union of exact keys; matching itself remains tier-aware."""

    identity = _well_binding_identity(item)
    return set().union(*identity.values())


def _matching_well_indices(
    requested: Mapping[str, Any], wells: list[Mapping[str, Any]]
) -> list[int]:
    """Match UID first and never fall back past an explicit stable identity."""

    requested_identity = _well_binding_identity(requested)
    well_identities = [_well_binding_identity(item) for item in wells]
    for tier in ("uid", "id", "name"):
        requested_keys = requested_identity[tier]
        if not requested_keys:
            continue
        return [
            index
            for index, candidate in enumerate(well_identities)
            if requested_keys.intersection(candidate[tier])
        ]
    return []


def _has_valid_md_trajectory(well: Mapping[str, Any]) -> bool:
    """Require complete finite XYZ+MD arrays and strictly increasing MD."""

    horizontal_alignment = well.get("horizontalAlignment")
    if (
        well.get("displayOnlyGeometry") is True
        or well.get("measuredTrajectory") is False
        or well.get("horizontalGeometryAvailable") is False
        or (
            isinstance(horizontal_alignment, Mapping)
            and horizontal_alignment.get("placementDerived") is True
        )
    ):
        return False
    raw_series = [well.get(field) for field in ("mdM", "x", "y", "z")]
    if not all(isinstance(values, list) for values in raw_series):
        return False
    if any(
        any(isinstance(value, (bool, np.bool_)) for value in values)
        for values in raw_series
    ):
        return False
    md_values, x_values, y_values, z_values = raw_series
    size = len(md_values)
    if size < 2 or any(len(values) != size for values in (x_values, y_values, z_values)):
        return False
    try:
        numeric_series = [np.asarray(values, dtype=np.float64) for values in raw_series]
    except (TypeError, ValueError):
        return False
    if any(values.ndim != 1 for values in numeric_series) or any(
        not np.isfinite(values).all() for values in numeric_series
    ):
        return False
    return bool(np.all(np.diff(numeric_series[0]) > 0.0))


def _well_result_tracks_for_volume(
    sequences: list[Mapping[str, Any]],
    volume: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    vertical_axis = (volume or {}).get("verticalAxis") or {}
    verified_twt_axis = bool(
        isinstance(vertical_axis, Mapping)
        and vertical_axis.get("domain") == "TWT"
        and vertical_axis.get("twtVerified") is True
    )
    wells = [
        item
        for item in ((volume or {}).get("embeddedWells") or [])
        if isinstance(item, Mapping)
    ]
    tracks: list[dict[str, Any]] = []
    for sequence in sequences:
        track = copy.deepcopy(dict(sequence))
        matching_indices = _matching_well_indices(sequence, wells)
        binding: dict[str, Any] = {
            "status": "unmatched",
            "axis": "measured_depth_m",
            "trajectoryIndex": None,
            "trajectoryWellUid": None,
            "trajectoryWellId": None,
            "measuredDepthTrajectoryAvailable": False,
            "reason": "well_identity_not_found",
        }
        twt_placement = "md_only"
        if len(matching_indices) == 1:
            index = matching_indices[0]
            well = wells[index]
            md_available = _has_valid_md_trajectory(well)
            twt_placement = (
                "accepted"
                if md_available
                and verified_twt_axis
                and well.get("alignmentMode") == "time_registered"
                and well.get("provenTwt") is True
                and well.get("formalRegistration") is True
                and well.get("registrationAccepted") is True
                else "md_only"
            )
            binding = {
                "status": "matched",
                "axis": "measured_depth_m",
                "trajectoryIndex": index,
                "trajectoryWellUid": str(well.get("wellUid") or "") or None,
                "trajectoryWellId": str(
                    well.get("wellId") or well.get("name") or ""
                )
                or None,
                "measuredDepthTrajectoryAvailable": md_available,
                "reason": None,
            }
        elif len(matching_indices) > 1:
            binding["status"] = "ambiguous"
            binding["reason"] = "well_identity_is_ambiguous"
        display = dict(track.get("display") or {})
        display.update(
            {
                "attachment": "well_callout",
                "twtPlacement": twt_placement,
            }
        )
        track["display"] = display
        track["binding"] = binding
        tracks.append(track)
    return tracks


def _compose_well_result_preview(
    base_preview: Mapping[str, Any],
    well_sequence_preview: Mapping[str, Any],
    *,
    source_task_id: str | None = None,
) -> dict[str, Any]:
    """Keep the source seismic canvas and add integrity-bound well tracks."""

    preview = copy.deepcopy(dict(base_preview))
    sequences = [
        item
        for item in (well_sequence_preview.get("wellSequences") or [])
        if isinstance(item, Mapping)
    ]
    preview["wellSequences"] = copy.deepcopy(sequences)
    preview["wellSequenceContractVersion"] = str(
        well_sequence_preview.get("contractVersion") or ""
    )
    preview["wellSequenceDisplayPolicy"] = copy.deepcopy(
        well_sequence_preview.get("displayPolicy") or {}
    )
    volumes = [
        item for item in (preview.get("volumes") or []) if isinstance(item, dict)
    ]
    for volume in volumes:
        if source_task_id:
            volume["viewStateKey"] = f"source:{source_task_id}"
        volume["wellResultTracks"] = _well_result_tracks_for_volume(
            sequences, volume
        )
        volume["wellResultTaskId"] = str(
            well_sequence_preview.get("taskId") or ""
        )
    preview["wellResultTracks"] = _well_result_tracks_for_volume(
        sequences, volumes[0] if volumes else None
    )
    return preview


@app.get("/统一数据可视化", include_in_schema=False)
def unified_data_visualization(
    embed: bool = True,
    task_id: str | None = None,
    asset: int | None = None,
    block: str | None = None,
    layerpulse_output: str | None = None,
) -> HTMLResponse:
    if not task_id:
        return _visualization_unavailable(
            "尚未绑定当前任务",
            "请从平台完成数据准备后点击“查看当前任务数据”，系统不会再默认展示历史示例数据。",
        )
    try:
        requested_task = _get_task(task_id)
    except KeyError:
        return _visualization_unavailable(
            "任务数据已失效", "服务可能已重启，请重新执行数据准备。"
        )
    data_task = requested_task
    prediction_result: dict[str, Any] | None = None
    well_sequence_preview: dict[str, Any] | None = None
    display_acceptance: dict[str, Any] = {}
    candidate_visualization: dict[str, Any] = {"renderable": False}
    stored_candidate_visualization: dict[str, Any] | None = None
    if requested_task.get("task_type") == "model_prediction":
        raw_prediction = (requested_task.get("result") or {}).get("prediction")
        if not isinstance(raw_prediction, dict):
            return _visualization_unavailable(
                "预测结果尚未就绪", "请等待推理任务完成后再打开结果工作台。"
            )
        prediction_result = raw_prediction
        if supports_standard_well_sequence_view(prediction_result):
            try:
                requested_task = _ensure_prediction_standard_results(
                    task_id, requested_task
                )
                prediction_result = (requested_task.get("result") or {}).get(
                    "prediction"
                )
                if not isinstance(prediction_result, dict):
                    raise ValueError("推理任务没有标准结果")
                well_sequence_preview = build_standard_well_sequence_preview(
                    prediction_result,
                    execution_task_id=task_id,
                )
                source_task_id = requested_task.get("parent_task_id") or (
                    requested_task.get("result") or {}
                ).get("source_task_id") or prediction_result.get(
                    "source_snapshot_id"
                )
                if source_task_id:
                    try:
                        data_task = _get_task(str(source_task_id))
                    except KeyError:
                        # A sealed well result remains renderable in the same
                        # light viewer when its source snapshot is unavailable,
                        # but no unrelated seismic background is substituted.
                        data_task = requested_task
            except (OSError, TypeError, ValueError) as exc:
                return _visualization_unavailable(
                    "井侧标准成果可视化失败",
                    f"{type(exc).__name__}: {exc}",
                )
        else:
            if str(prediction_result.get("model_id") or "") == LAYERPULSE_MODEL_ID:
                layerpulse_decision = evaluate_layerpulse_visualization(
                    prediction_result
                )
                if layerpulse_decision.get("renderable") is not True:
                    return _visualization_failed_diagnostic(
                        "LayerPulse 结果未通过统一可视化验收",
                        [
                            str(item)
                            for item in layerpulse_decision.get("diagnostics") or []
                        ],
                    )
                display_acceptance = {
                    "contract_version": layerpulse_decision.get("contract_version"),
                    "display_status": "accepted",
                    "quantitative_status": "not_run",
                    "visual_status": "passed",
                    "reason_codes": [],
                    "diagnostics": [],
                    "specialized_gate": "layerpulse_single_forward_artifact_gate",
                }
                candidate_visualization = dict(layerpulse_decision)
            else:
                display_acceptance = evaluate_result_display_acceptance(
                    prediction_result
                ).to_dict()
                stored_candidate_visualization = _stored_candidate_visualization_decision(
                    prediction_result,
                    execution_task_id=task_id,
                )
                candidate_visualization = (
                    stored_candidate_visualization
                    or evaluate_candidate_visualization(prediction_result)
                )
            if display_acceptance["display_status"] == "failed_diagnostic":
                diagnostics = [
                    str(item)
                    for item in display_acceptance.get("diagnostics") or []
                ]
                return _visualization_failed_diagnostic(
                    "预测结果未通过展示验收",
                    diagnostics,
                )
            if (
                display_acceptance["display_status"] != "accepted"
                and not candidate_visualization["renderable"]
            ):
                reason_codes = ", ".join(
                    str(item)
                    for item in display_acceptance.get("reason_codes") or []
                )
                candidate_reason_codes = ", ".join(
                    str(item)
                    for item in candidate_visualization.get("reason_codes") or []
                    if str(item) != "candidate_visualization_not_supported"
                )
                current_status = ", ".join(
                    item for item in (reason_codes, candidate_reason_codes) if item
                )
                return _visualization_unavailable(
                    "预测结果暂不可展示",
                    "定量验收、视觉验收及原始/真值/预测/误差四面板必须完整通过。"
                    + (f" 当前状态：{current_status}" if current_status else ""),
                )
            source_task_id = requested_task.get("parent_task_id") or (
                requested_task.get("result") or {}
            ).get("source_task_id")
            if not source_task_id:
                return _visualization_unavailable(
                    "缺少源数据任务",
                    "该推理结果没有登记source_task_id，无法重建背景地震体。",
                )
            try:
                data_task = _get_task(str(source_task_id))
            except KeyError:
                return _visualization_unavailable(
                    "源数据任务已失效", "请重新执行数据准备，再进入预测解释任务。"
                )
    base_preview = copy.deepcopy(
        (data_task.get("result") or {}).get("visualization_preview") or {}
    )
    preview = (
        _compose_well_result_preview(
            base_preview,
            well_sequence_preview,
            source_task_id=(
                str(data_task.get("task_id") or "")
                if data_task is not requested_task
                else None
            ),
        )
        if well_sequence_preview is not None
        else base_preview
    )
    if prediction_result is not None and well_sequence_preview is None:
        preview["displayAcceptance"] = display_acceptance
        if candidate_visualization["renderable"]:
            preview["candidateVisualization"] = candidate_visualization
        surface_horizon_display_contract = (
            _stored_surface_horizon_display_contract(
                prediction_result,
                candidate_visualization,
            )
            if stored_candidate_visualization is not None
            else None
        )
        try:
            prediction_volume = build_prediction_visualization_payload(
                prediction_result,
                config=_platform_config,
                segy_options={
                    "profile": str(
                        prediction_result.get("input", {}).get(
                            "geometry_profile", "standard_3d"
                        )
                    )
                },
                max_shape_zyx=(
                    (128, 128, 128)
                    if str(prediction_result.get("model_id") or "")
                    == LAYERPULSE_MODEL_ID
                    else (128, 96, 96)
                ),
                layerpulse_output_key=layerpulse_output,
                faultseg_block_id=block,
                verified_surface_horizon_display_contract=(
                    surface_horizon_display_contract
                ),
            )
            prediction_descriptor = prediction_volume["predictionVisualization"]
            prediction_volume["embeddedWells"] = []
            if candidate_visualization["renderable"]:
                prediction_volume["candidateVisualization"] = candidate_visualization
            preview["volumes"] = [prediction_volume, *preview.get("volumes", [])]
            preview["activePrediction"] = {
                "modelId": str(prediction_descriptor["modelId"]),
                "taskId": task_id,
                "preferredLayer": prediction_descriptor.get("preferredLayer"),
            }
        except Exception as exc:
            return _visualization_unavailable(
                "预测结果可视化失败",
                f"{type(exc).__name__}: {exc}",
            )
    if (
        not preview.get("volumes")
        and not preview.get("lines2d")
        and not preview.get("wellSequences")
    ):
        detail = "；".join(str(item) for item in preview.get("issues", [])[:2])
        return _visualization_unavailable(
            "当前任务没有可渲染的地震数据",
            detail
            or "请先在数据准备中读取二维测线或形成可靠Inline/Crossline网格的三维SEG-Y。",
        )
    try:
        selected_asset = asset
        if selected_asset is None:
            selected_asset = 0
            if well_sequence_preview is not None:
                # The catalog is ordered as volumes, 2-D lines, then well
                # sequences.  A well-side prediction should open its actual
                # result rather than silently selecting the source seismic
                # background and making the result appear to be missing.
                selected_asset = len(preview.get("volumes") or []) + len(
                    preview.get("lines2d") or []
                )
        document = render_cigvis_workbench(
            PROJECT_ROOT,
            preview,
            task_id=task_id,
            asset_index=selected_asset,
            embed=embed,
        )
    except Exception as exc:
        return _visualization_unavailable(
            "项目可视化启动失败",
            public_visualization_text(f"{type(exc).__name__}: {exc}"),
        )
    return HTMLResponse(document, headers={"Cache-Control": "no-store"})


@app.get("/三维地震看板", include_in_schema=False, deprecated=True)
def seismic_dashboard(task_id: str | None = None) -> HTMLResponse:
    return unified_data_visualization(task_id=task_id, embed=False)


def run() -> None:
    import uvicorn

    host = os.getenv("WELL_SEISMIC_HOST", "127.0.0.1")
    port = int(os.getenv("WELL_SEISMIC_PORT", "725"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
