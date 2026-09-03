"""Read-only audit for the WellFuse demonstration submission.

The audit intentionally separates four independent facts: catalog metadata,
input-adapter registration, inference-runner registration, and evidence that a
real checkpoint forward actually completed.  In particular, ``runnable`` in a
catalog and a precomputed product are never treated as execution evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .modeling.input_adapters import build_default_input_adapters
from .modeling.registry import build_default_registry
from .prediction import build_default_prediction_runners
from .releases import build_release_catalog

SCHEMA_VERSION = "well-seismic.demo-readiness.v1"
ALIGN_MODEL_ID = "wellfuse_align_p13"

_LIFECYCLE_TASK_BY_MODEL = {
    ALIGN_MODEL_ID: "wellfuse_align",
    "wellfuse_horizon_p17": "chengdu_horizon",
    "wellfuse_facies_3d_p17": "chengdu_facies",
    "wellfuse_channel_p17": "synthetic_channel_geobody",
    "wellfuse_karst_p17": "synthetic_karst_geobody",
}

_DEFAULT_CHECKPOINTS = {
    "wellfuse_facies_3d_p17": (
        (
            "p17/chengdu_facies/volume_3d_v1/seed_20260817/"
            "spatial_fold1/best.pt"
        ),
    ),
    "wellfuse_facies_1d_p17": tuple(
        "p17/chengdu_facies/facies_1d_oof_v1/"
        f"seed_{seed}/spatial_fold1/best.pt"
        for seed in (20260817, 20260829, 20260841)
    ),
    **{
        f"wellfuse_{target}_p18": tuple(
            "p18/chengdu_well_oof_v2/"
            f"{target}/seed_{seed}/spatial_fold1/final_refit.pt"
            for seed in (20260818, 20260841, 20260873)
        )
        for target in ("den", "por", "log_perm", "sw", "vsh")
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None,
    verify_sha256: bool,
    sha_cache: dict[Path, str],
    source: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_file()
    actual: str | None = None
    if exists and verify_sha256:
        if resolved not in sha_cache:
            sha_cache[resolved] = _sha256(resolved)
        actual = sha_cache[resolved]
    if not exists:
        integrity = "missing"
    elif actual is None:
        integrity = "exists_not_hashed"
    elif expected_sha256 and actual != expected_sha256:
        integrity = "sha256_mismatch"
    elif expected_sha256:
        integrity = "sha256_verified"
    else:
        integrity = "sha256_computed_unpinned"
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "integrity_status": integrity,
        "source": source,
    }


def _checkpoint_refs(payload: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = []

    def add(value: Any, sha: Any = None) -> None:
        if isinstance(value, str) and value.casefold().endswith((".pt", ".pth")):
            refs.append((value, str(sha) if sha else None))

    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        add(checkpoint.get("path"), checkpoint.get("sha256"))
    elif isinstance(checkpoint, str):
        add(checkpoint)
    for key in ("checkpoints", "checkpoint_evidence"):
        checkpoints = payload.get(key)
        if isinstance(checkpoints, list):
            for item in checkpoints:
                if isinstance(item, Mapping):
                    add(
                        item.get("path") or item.get("checkpoint"),
                        item.get("sha256"),
                    )
    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        for key in ("members", "member_audit"):
            members = execution.get(key)
            if not isinstance(members, list):
                continue
            for item in members:
                if not isinstance(item, Mapping):
                    continue
                add(item.get("checkpoint") or item.get("path"), item.get("sha256"))
                add(
                    item.get("anchor_checkpoint"),
                    item.get("anchor_checkpoint_sha256"),
                )
    return refs


def classify_smoke_manifest(
    payload: Mapping[str, Any], path: str | Path
) -> dict[str, Any] | None:
    """Return explicit execution evidence, or ``None`` for non-smoke metadata."""

    schema = str(payload.get("schema_version", ""))
    model_id: str | None = None
    executed = False
    unknown_status = "not_validated"
    details: dict[str, Any] = {}

    if schema == "wellfuse.align.unknown-survey-candidate.v1":
        model_id = ALIGN_MODEL_ID
        execution = payload.get("execution", {})
        executed = bool(payload.get("p13_checkpoint_executed")) and bool(
            isinstance(execution, Mapping) and execution.get("member_count")
        )
        unknown_status = (
            "experimental_fusion_ready"
            if payload.get("fusion_ready") is True
            else "executed_but_not_fusion_ready"
        )
        details = {
            "fusion_ready": payload.get("fusion_ready") is True,
            "inference_ready": payload.get("inference_ready") is True,
            "factorized_v3_forward_executed": payload.get(
                "factorized_v3_model_forward_executed"
            )
            is True,
            "time_depth_supervision_is_model_input": payload.get(
                "time_depth_supervision_is_model_input"
            ),
        }
    elif schema == "wellfuse.p17.unknown_horizon_prediction.v1":
        model_id = "wellfuse_horizon_p17"
        inference = payload.get("inference", {})
        executed = bool(
            isinstance(inference, Mapping)
            and inference.get("actual_checkpoint_loaded_and_forward_executed")
            is True
            and int(inference.get("checkpoint_forward_calls", 0)) > 0
        )
        unknown_status = "archived_historical_execution"
        details = {
            "checkpoint_forward_calls": inference.get("checkpoint_forward_calls"),
            "target_surface_labels_opened": payload.get("adaptation", {}).get(
                "target_surface_labels_opened"
            ),
            "historical_only": True,
        }
    elif schema == "wellfuse.p17.chengdu_facies_3d_candidate_inference.v1":
        model_id = "wellfuse_facies_3d_p17"
        window = payload.get("sliding_window", {})
        completed = int(window.get("completed_tiles", 0)) if isinstance(window, Mapping) else 0
        executed = completed > 0 and str(payload.get("status", "")).startswith("completed")
        unknown_status = "experimental_weak_candidate"
        details = {"completed_tiles": completed, "mode": payload.get("mode")}
    elif schema == "wellfuse.unknown_survey_well_runtime.v1":
        candidate = payload.get("model_id")
        if isinstance(candidate, str):
            model_id = candidate
        executed = payload.get("model_executed") is True
        unknown_status = (
            "experimental_checkpoint_executed"
            if executed
            else "baseline_fallback_not_model_execution"
        )
        execution = payload.get("execution", {})
        details = {
            "model_executed": executed,
            "member_count": execution.get("member_count")
            if isinstance(execution, Mapping)
            else None,
            "target_is_model_input": payload.get("target_is_model_input"),
            "time_depth_supervision_opened": payload.get(
                "time_depth_supervision_opened"
            ),
        }
    elif schema == "wellfuse.p17.real_geobody_candidate_full.v1":
        task = str(payload.get("task", "")).casefold()
        if task in {"channel", "karst"}:
            model_id = f"wellfuse_{task}_p17"
        window = payload.get("sliding_window", {})
        executed = bool(
            model_id
            and str(payload.get("status", "")).startswith("completed")
            and isinstance(window, Mapping)
            and int(window.get("tile_count", 0)) > 0
            and payload.get("cuda_used") is True
        )
        unknown_status = "provisional_real_survey_candidate"
        details = {
            "tile_count": window.get("tile_count")
            if isinstance(window, Mapping)
            else None,
            "real_accuracy_metrics_computed": payload.get(
                "evidence_separation", {}
            ).get("real_accuracy_metrics_computed"),
        }
    elif schema == "well-seismic.faultseg-runtime.v1":
        model_id = "faultseg_3d"
        outputs = payload.get("outputs", {})
        executed = bool(
            payload.get("model_executed") is True
            and int(payload.get("checkpoint_forward_calls", 0)) > 0
            and isinstance(outputs, Mapping)
            and Path(str(outputs.get("probability_npy", ""))).is_file()
            and Path(str(outputs.get("mask_npy", ""))).is_file()
        )
        unknown_status = "legacy_engineering_candidate"
        details = {
            "checkpoint_forward_calls": payload.get("checkpoint_forward_calls"),
            "probability": payload.get("probability"),
        }
    elif schema == "well-seismic.surface-seg-runtime.v1":
        model_id = "seismic_surface_seg"
        outputs = payload.get("outputs", {})
        executed = bool(
            payload.get("model_executed") is True
            and int(payload.get("checkpoint_forward_calls", 0)) == 3
            and isinstance(outputs, Mapping)
            and Path(str(outputs.get("mask_npy", ""))).is_file()
            and Path(str(outputs.get("confidence_npy", ""))).is_file()
        )
        unknown_status = "legacy_engineering_candidate"
        details = {
            "checkpoint_forward_calls": payload.get("checkpoint_forward_calls"),
            "segmentation": payload.get("segmentation"),
        }
    if not model_id:
        return None
    resolved_path = Path(path).resolve()
    path_text = str(resolved_path).casefold()
    if "demo_submission_v1" in path_text:
        evidence_scope = "demo_unknown_survey"
    elif "real_geobody_candidates_v1" in path_text:
        evidence_scope = "full_real_survey_candidate"
    elif f"{os.sep}tmp{os.sep}" in path_text:
        evidence_scope = "test_or_synthetic_smoke"
    else:
        evidence_scope = "runtime_artifact"
    return {
        "model_id": model_id,
        "manifest_path": str(resolved_path),
        "manifest_sha256": _sha256(resolved_path),
        "schema_version": schema,
        "evidence_scope": evidence_scope,
        "actual_checkpoint_forward_executed": executed,
        "unknown_survey_status": unknown_status,
        "details": details,
        "checkpoint_refs": [
            {"path": checkpoint, "sha256": sha}
            for checkpoint, sha in _checkpoint_refs(payload)
        ],
    }


def _manifest_paths(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        evidence_names = {
            "manifest.json",
            "faultseg_result.json",
            "surface_seg_result.json",
        }
        if resolved.is_file() and resolved.name.casefold() in evidence_names:
            candidates = (resolved,)
        elif resolved.is_dir():
            candidates = (
                path
                for name in evidence_names
                for path in resolved.rglob(name)
            )
        else:
            continue
        for path in candidates:
            path = path.resolve()
            if path in seen or path.is_symlink():
                continue
            seen.add(path)
            yield path


def _load_lifecycle(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _read_json(path)
    candidates = {
        str(item.get("candidate_id")): item
        for item in document.get("candidates", [])
        if isinstance(item, dict) and item.get("candidate_id")
    }
    selected: dict[str, dict[str, Any]] = {}
    for pointer in document.get("pointers", []):
        if not isinstance(pointer, dict) or pointer.get("pointer_type") != "runtime_default":
            continue
        candidate = candidates.get(str(pointer.get("candidate_id")))
        if candidate:
            selected[str(pointer.get("task_id"))] = candidate
    return document, selected


def _default_smoke_roots(project_root: Path, artifact_root: Path) -> list[Path]:
    return [
        artifact_root / "demo_submission_v1",
        artifact_root / "p13_unknown_survey_runtime",
        artifact_root / "align_x_unknown_survey",
        artifact_root / "p17" / "real_geobody_candidates_v1",
        artifact_root / "tmp",
        project_root / "model_outputs",
    ]


def audit_demo_readiness(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    smoke_roots: Iterable[str | Path] | None = None,
    verify_checkpoint_sha256: bool = True,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    artifacts = Path(artifact_root).expanduser().resolve()
    model_registry = build_default_registry()
    adapter_registry = build_default_input_adapters({})
    runner_registry = build_default_prediction_runners()
    release_catalog = build_release_catalog(
        project_root=project,
        artifact_root=artifacts,
        verify_sha256=verify_checkpoint_sha256,
    )
    model_registry.apply_release_catalog(release_catalog)
    models = {spec.id: spec for spec in model_registry.list_specs()}
    adapter_ids = {
        str(item["model_id"])
        for item in adapter_registry.capabilities()
        if item.get("model_id")
    }
    runner_ids = set(runner_registry.model_ids())
    releases = {
        str(release.model_id): release
        for release in release_catalog.list()
        if getattr(release, "model_id", None)
    }
    lifecycle_path = artifacts / "lifecycle" / "registry.json"
    lifecycle, runtime_defaults = _load_lifecycle(lifecycle_path)

    roots = (
        [Path(item) for item in smoke_roots]
        if smoke_roots is not None
        else _default_smoke_roots(project, artifacts)
    )
    smoke_by_model: dict[str, list[dict[str, Any]]] = {}
    for path in _manifest_paths(roots):
        evidence = classify_smoke_manifest(_read_json(path), path)
        if evidence:
            smoke_by_model.setdefault(evidence["model_id"], []).append(evidence)

    sha_cache: dict[Path, str] = {}
    task_rows: list[dict[str, Any]] = []
    for model_id, spec in models.items():
        if spec.runtime_status == "interface_only":
            continue
        release = releases.get(model_id)
        refs: dict[str, tuple[str | None, str]] = {}
        if release is not None:
            for item in release.artifacts:
                if "checkpoint" not in item.role.casefold() and item.kind != "checkpoint":
                    continue
                refs[item.path] = (item.sha256, f"release:{release.id}")
        lifecycle_task = _LIFECYCLE_TASK_BY_MODEL.get(model_id)
        lifecycle_candidate = runtime_defaults.get(lifecycle_task or "")
        if lifecycle_candidate:
            for item in lifecycle_candidate.get("artifacts", []):
                if isinstance(item, dict) and "checkpoint" in str(
                    item.get("role", "")
                ).casefold():
                    refs[str(item.get("path"))] = (
                        item.get("sha256"),
                        f"lifecycle:{lifecycle_candidate.get('candidate_id')}",
                    )
        for relative in _DEFAULT_CHECKPOINTS.get(model_id, ()):
            refs[str(artifacts / relative)] = (None, "runtime_default")
        evidence = sorted(
            smoke_by_model.get(model_id, []),
            key=lambda item: Path(item["manifest_path"]).stat().st_mtime_ns,
            reverse=True,
        )
        for smoke in evidence:
            for item in smoke["checkpoint_refs"]:
                refs[item["path"]] = (item.get("sha256"), "smoke_manifest")
        checkpoint_rows = [
            _safe_checkpoint(
                path,
                expected_sha256=expected,
                verify_sha256=verify_checkpoint_sha256,
                sha_cache=sha_cache,
                source=source,
            )
            for path, (expected, source) in sorted(refs.items())
            if path and path != "None"
        ]
        checkpoints_ready = bool(checkpoint_rows) and all(
            item["exists"]
            and item["integrity_status"] not in {"missing", "sha256_mismatch"}
            for item in checkpoint_rows
        )
        special_align_runtime = model_id == ALIGN_MODEL_ID
        adapter_present = model_id in adapter_ids or (
            special_align_runtime
            and (project / "src/well_seismic/p13_registration.py").is_file()
        )
        runner_present = model_id in runner_ids or (
            special_align_runtime
            and (artifacts.parent / "scripts/infer_p13_unknown_survey.py").is_file()
        )
        executed = any(
            item["actual_checkpoint_forward_executed"] is True for item in evidence
        )
        latest = evidence[0] if evidence else None
        declared_runnable = spec.runtime_status == "runnable"
        if spec.runtime_status in {"blocked", "precomputed_only"} and not special_align_runtime:
            readiness = spec.runtime_status
        elif executed and special_align_runtime and not latest.get("details", {}).get(
            "fusion_ready", False
        ):
            readiness = "executed_but_not_fusion_ready"
        elif executed and adapter_present and runner_present and checkpoints_ready:
            readiness = "execution_verified_experimental"
        elif adapter_present and runner_present and checkpoints_ready:
            readiness = "static_ready_smoke_missing"
        elif declared_runnable or special_align_runtime:
            readiness = "declared_but_incomplete"
        else:
            readiness = spec.runtime_status
        unknown_status = (
            latest["unknown_survey_status"]
            if latest
            else spec.metadata.get("unknown_survey_runtime_status", "not_executed")
        )
        task_rows.append(
            {
                "model_id": model_id,
                "task_id": spec.metadata.get("prediction_task"),
                "release_id": spec.metadata.get("release_id"),
                "scientific_status": spec.scientific_status,
                "scientific_evidence_class": spec.evidence_class,
                "catalog_runtime_status": spec.runtime_status,
                "release_scientific_status": (
                    release.scientific_status if release is not None else None
                ),
                "release_runtime_status": (
                    release.runtime_status if release is not None else None
                ),
                "lifecycle_runtime_default": (
                    {
                        "candidate_id": lifecycle_candidate.get("candidate_id"),
                        "scientific_status": lifecycle_candidate.get(
                            "scientific_status"
                        ),
                        "runtime_status": lifecycle_candidate.get("runtime_status"),
                    }
                    if lifecycle_candidate
                    else None
                ),
                "unknown_survey_status": unknown_status,
                "readiness": readiness,
                "experimental_unknown_survey": str(unknown_status).startswith(
                    ("experimental", "provisional")
                )
                or unknown_status == "executed_but_not_fusion_ready",
                "requires_registration": bool(
                    spec.metadata.get("requires_registration")
                ),
                "runtime_entrypoint": (
                    "registration_workflow" if special_align_runtime else "prediction_runner"
                ),
                "adapter_present": adapter_present,
                "runner_present": runner_present,
                "checkpoint_set_ready": checkpoints_ready,
                "checkpoints": checkpoint_rows,
                "actual_checkpoint_forward_executed": executed,
                "smoke_evidence": evidence,
                "warnings": list(spec.warnings),
            }
        )

    counts: dict[str, int] = {}
    for row in task_rows:
        counts[row["readiness"]] = counts.get(row["readiness"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only_audit": True,
        "project_root": str(project),
        "artifact_root": str(artifacts),
        "contract": {
            "catalog_registration_is_execution_evidence": False,
            "precomputed_products_are_unknown_survey_execution": False,
            "checkpoint_presence_alone_is_execution_evidence": False,
            "execution_verified_requires": [
                "adapter_or_registration_adapter",
                "runner_or_registration_runtime",
                "checkpoint_integrity",
                "manifest_with_explicit_checkpoint_forward",
            ],
            "scientific_and_unknown_survey_status_are_independent": True,
        },
        "registry": {
            "model_count": len(models),
            "adapter_model_ids": sorted(adapter_ids),
            "runner_model_ids": sorted(runner_ids),
            "lifecycle_registry_path": str(lifecycle_path),
            "lifecycle_registry_sha256": lifecycle.get("registry_sha256"),
            "lifecycle_runtime_default_count": len(runtime_defaults),
        },
        "summary": {"task_count": len(task_rows), "readiness_counts": counts},
        "tasks": task_rows,
    }


def render_report(document: Mapping[str, Any]) -> str:
    lines = [
        "# WellFuse demo submission readiness",
        "",
        f"Generated: `{document.get('generated_at')}`",
        "",
        (
            "This is a read-only engineering audit. Scientific status and unknown-survey "
            "execution status are reported independently. A registered/precomputed model "
            "is never presented as an executed model."
        ),
        "",
        "| Model | Scientific | Unknown survey | Registration | Adapter | Runner | Checkpoint | Forward evidence | Smoke scope | Readiness |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for task in document.get("tasks", []):
        lines.append(
            "| {model_id} | {scientific_status} | {unknown_survey_status} | {reg} | "
            "{adapter} | {runner} | {checkpoint} | {forward} | {scope} | {readiness} |".format(
                **task,
                reg="yes" if task["requires_registration"] else "no",
                adapter="yes" if task["adapter_present"] else "no",
                runner="yes" if task["runner_present"] else "no",
                checkpoint="yes" if task["checkpoint_set_ready"] else "no",
                forward=(
                    "yes" if task["actual_checkpoint_forward_executed"] else "no"
                ),
                scope=(
                    task["smoke_evidence"][0]["evidence_scope"]
                    if task["smoke_evidence"]
                    else "none"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `execution_verified_experimental` means a real checkpoint forward is "
                "recorded, but it does not transfer Chengdu accuracy to an unknown survey."
            ),
            (
                "- `static_ready_smoke_missing` means code and weights are wired but this "
                "audit found no explicit forward manifest."
            ),
            (
                "- `executed_but_not_fusion_ready` means the Align model executed but its "
                "physics/readiness gate rejected downstream fusion."
            ),
            "- `precomputed_only` and `blocked` are never testable online models.",
            "",
        ]
    )
    return "\n".join(lines)


def write_demo_readiness(
    document: Mapping[str, Any], output_directory: str | Path
) -> tuple[Path, Path]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    report = output / "REPORT.md"
    for path, text in (
        (manifest, json.dumps(document, ensure_ascii=False, indent=2) + "\n"),
        (report, render_report(document)),
    ):
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    return manifest, report


__all__ = [
    "SCHEMA_VERSION",
    "audit_demo_readiness",
    "classify_smoke_manifest",
    "render_report",
    "write_demo_readiness",
]
