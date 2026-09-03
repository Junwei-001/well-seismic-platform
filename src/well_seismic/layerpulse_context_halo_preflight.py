"""Explicit CUDA preflight for the fixed LayerPulse V4 context-halo ROI.

The module prepares one 160^3 input around the final clean Chengdu registered
well-anchored 128^3 V4 output ROI, dispatches the existing single-checkpoint
child exactly once, and writes an evidence receipt.  It never edits the
checkpoint or the platform deployment configuration.  Promotion of the
candidate remains a separate reviewed act.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .layerpulse_contract import (
    LAYERPULSE_MODEL_ID,
    LAYERPULSE_REQUEST_SCHEMA,
    LAYERPULSE_TASK_ID,
    load_layerpulse_platform_config,
)
from .layerpulse_runtime import (
    _runtime_paths,
    _subprocess_environment,
    validate_layerpulse_child_result,
)
from .layerpulse_well_bridge import materialize_layerpulse_well_bundle
from .modeling.input_adapters import ModelInputRequest
from .modeling.layerpulse_input_adapter import LayerPulseInputAdapter
from .task_runtime import managed_run

PREFLIGHT_SCHEMA = "well-seismic.layerpulse-context-halo-cuda-preflight.v1"
CONTEXT_WINDOW_SCHEMA = "well-seismic.layerpulse-context-halo-window.v1"
FIXED_ROI_ID = "chengdu_registered_well_anchored_128_v4"
FIXED_OUTPUT_SHAPE_TIX = (128, 128, 128)
FIXED_CROP_START_TIX = (1186, 1655, 307)
FIXED_HALO_TIX = (16, 16, 16)
FIXED_INPUT_SHAPE_TIX = (160, 160, 160)
FIXED_SELECTION_POLICY = "fusion_ready_well_trajectory_anchor"
FIXED_CHECKPOINT_FAMILY = "precision_multitask_expression_fit_v4"
FIXED_ANCHOR_RESULT = (
    Path("model_outputs")
    / "layerpulse_layerpulse_geochronograph_f3x200cf_03a0ef50"
    / "result.json"
)

ChildExecutor = Callable[[list[str], Mapping[str, Any]], Any]


class LayerPulseContextHaloPreflightError(RuntimeError):
    """The fixed preflight contract or its resulting evidence is invalid."""


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayerPulseContextHaloPreflightError(f"{label} must be a mapping")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _mapping(
            json.loads(path.read_text(encoding="utf-8")), label=str(path)
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LayerPulseContextHaloPreflightError(
            f"cannot read JSON {path}: {exc}"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(document), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _anchor_contract(anchor: Mapping[str, Any], *, checkpoint: Path) -> dict[str, Any]:
    input_receipt = _mapping(anchor.get("input"), label="V4 anchor input receipt")
    anchor_checkpoint = _mapping(
        anchor.get("checkpoint"), label="V4 anchor checkpoint receipt"
    )
    crop_start = tuple(int(value) for value in input_receipt.get("crop_start_tix") or ())
    shape = tuple(int(value) for value in input_receipt.get("shape_tix") or ())
    selection = str(input_receipt.get("selection_policy") or "")
    if crop_start != FIXED_CROP_START_TIX:
        raise LayerPulseContextHaloPreflightError("V4 anchor crop start drifted")
    if shape != FIXED_OUTPUT_SHAPE_TIX:
        raise LayerPulseContextHaloPreflightError("V4 anchor output shape drifted")
    if selection != FIXED_SELECTION_POLICY:
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor selection policy drifted"
        )
    if str(anchor_checkpoint.get("family") or "") != FIXED_CHECKPOINT_FAMILY:
        raise LayerPulseContextHaloPreflightError(
            "anchor is not the final V4 checkpoint family"
        )
    if Path(str(anchor_checkpoint.get("path") or "")).resolve() != checkpoint:
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor and deployed unified checkpoint differ"
        )
    if (
        anchor.get("prepared_view_consumed") is not True
        or anchor.get("registration_consumed") is not True
        or anchor.get("well_input_consumed") is not True
    ):
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor did not consume PreparedView, registration and well input"
        )
    source = Path(str(input_receipt.get("source") or "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"V4 anchor source SEG-Y not found: {source}")
    return {
        "source": source,
        "input": input_receipt,
        "well_bridge": _mapping(
            input_receipt.get("well_bridge"), label="V4 anchor well bridge"
        ),
        "provenance": _mapping(
            anchor.get("provenance"), label="V4 anchor provenance"
        ),
    }


def _prepared_view_options(anchor: Mapping[str, Any]) -> dict[str, Any]:
    input_receipt = _mapping(anchor["input"], label="V4 anchor input")
    well_bridge = _mapping(anchor["well_bridge"], label="V4 anchor well bridge")
    provenance = _mapping(anchor["provenance"], label="V4 anchor provenance")
    manifest_path = Path(
        str(well_bridge.get("prepared_view_manifest_path") or "")
    ).expanduser().resolve()
    manifest = _read_json(manifest_path)
    roles = {
        "canonical_well_las",
        "registration_manifest_v3",
        "registration_points_v3",
    }
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    for raw in manifest.get("artifacts") or []:
        if not isinstance(raw, Mapping):
            continue
        role = str(raw.get("role") or "")
        if role in by_role:
            by_role[role].append(dict(raw))
    if any(not by_role[role] for role in roles):
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor PreparedView lacks an aligned-well role"
        )
    registration_parents = [
        str(item.get("view_id"))
        for item in manifest.get("parents") or []
        if isinstance(item, Mapping) and item.get("kind") == "registration"
    ]
    if len(registration_parents) != 1:
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor PreparedView registration parent is ambiguous"
        )
    geometry = _mapping(input_receipt.get("geometry"), label="V4 anchor geometry")
    coordinate_reference = str(
        input_receipt.get("coordinate_reference")
        or geometry.get("coordinate_reference")
        or ""
    )
    if not coordinate_reference:
        raise LayerPulseContextHaloPreflightError(
            "V4 anchor coordinate reference is missing"
        )
    registration = _mapping(
        input_receipt.get("registration_consumption"),
        label="V4 anchor registration consumption",
    )
    return {
        "prepared_view_id": str(well_bridge["prepared_view_id"]),
        "prepared_view_manifest_path": str(manifest_path),
        "prepared_view_manifest_sha256": str(
            well_bridge["prepared_view_manifest_sha256"]
        ),
        "prepared_view_sha256": str(well_bridge["prepared_view_sha256"]),
        "prepared_view_kind": str(well_bridge["prepared_view_kind"]),
        "prepared_view_artifacts_by_role": by_role,
        "prepared_view_registration_relation": "matched",
        "registration_task_id": registration_parents[0],
        "registration_manifest_sha256": str(
            registration["registration_manifest_sha256"]
        ),
        "registration_points_sha256": str(
            registration["registration_points_sha256"]
        ),
        "source_snapshot_id": str(provenance["source_snapshot_id"]),
        "source_snapshot_fingerprint": str(
            provenance["source_snapshot_fingerprint"]
        ),
        "source_snapshot_semantics": {
            "horizontal_crs_id": coordinate_reference,
            "coordinate_reference_verified": True,
        },
    }


def _coordinate_contract(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "well-seismic.layerpulse-context-halo-coordinate.v1",
        "axes": ["TWT", "INLINE", "XLINE"],
        "source_shape_tix": list(provenance["source_shape_tix"]),
        "output_crop_start_tix": list(provenance["crop_start_tix"]),
        "output_shape_tix": list(provenance["crop_size_tix"]),
        "model_input_origin_tix": list(provenance["model_input_origin_tix"]),
        "model_input_shape_tix": list(provenance["model_input_shape_tix"]),
        "output_offset_tix": list(
            provenance["output_offset_in_model_input_tix"]
        ),
        "halo_tix": list(provenance["context_halo_tix"]),
        "padding_before_tix": list(
            provenance["model_input_padding_before_tix"]
        ),
        "padding_after_tix": list(provenance["model_input_padding_after_tix"]),
        "sample_interval_ms": float(provenance["sample_interval_ms"]),
        "coordinate_reference": provenance.get("coordinate_reference"),
        "coordinate_reference_verified": bool(
            provenance.get("coordinate_reference_verified")
        ),
        "output_inline_values": list(provenance["output_inline_values"]),
        "output_crossline_values": list(provenance["output_crossline_values"]),
        "model_inline_values": list(provenance["inline_values"]),
        "model_crossline_values": list(provenance["crossline_values"]),
    }


def _child_request(
    *,
    checkpoint: Path,
    source: Path,
    patch_path: Path,
    valid_mask_path: Path,
    well_bundle_path: Path,
    well_bundle_sha256: str,
    provenance: Mapping[str, Any],
    coordinate_contract: Mapping[str, Any],
    well_anchor: Mapping[str, Any] | None,
    child_output_directory: Path,
) -> dict[str, Any]:
    output_geometry = {
        "sample_interval_ms": provenance.get("sample_interval_ms"),
        "geometry_profile": provenance.get("geometry_profile"),
        "geometry_confidence": provenance.get("geometry_confidence"),
        "inline_values": provenance.get("output_inline_values"),
        "crossline_values": provenance.get("output_crossline_values"),
        "inline_range": provenance.get("output_inline_range"),
        "crossline_range": provenance.get("output_crossline_range"),
        "coordinate_reference": provenance.get("coordinate_reference"),
        "coordinate_reference_verified": bool(
            provenance.get("coordinate_reference_verified")
        ),
        "coordinate_reference_authority": provenance.get(
            "coordinate_reference_authority"
        ),
        "coordinate_contract_sha256": _canonical_sha256(coordinate_contract),
    }
    output_window = {
        "schema_version": CONTEXT_WINDOW_SCHEMA,
        "enabled": True,
        "model_input_shape_tix": list(FIXED_INPUT_SHAPE_TIX),
        "output_offset_tix": list(FIXED_HALO_TIX),
        "output_shape_tix": list(FIXED_OUTPUT_SHAPE_TIX),
        "halo_tix": list(FIXED_HALO_TIX),
        "model_input_origin_tix": list(provenance["model_input_origin_tix"]),
        "source_padding_before_tix": list(
            provenance["model_input_padding_before_tix"]
        ),
        "source_padding_after_tix": list(
            provenance["model_input_padding_after_tix"]
        ),
        "boundary_mode": "constant_zero_with_explicit_valid_mask",
        "output_rule": "central_complete_logits_then_direct_argmax",
        "validation_status": "cuda_preflight_in_progress_not_accepted",
        "single_checkpoint_forward_calls": 1,
    }
    return {
        "schema_version": LAYERPULSE_REQUEST_SCHEMA,
        "task_id": LAYERPULSE_TASK_ID,
        "model_id": LAYERPULSE_MODEL_ID,
        "checkpoint": str(checkpoint),
        "input": {
            "source": str(source),
            "patch_npy": str(patch_path),
            "valid_mask_npy": str(valid_mask_path),
            "axes": ["TWT", "INLINE", "XLINE"],
            "crop_start_tix": list(FIXED_CROP_START_TIX),
            "source_shape_tix": list(provenance["source_shape_tix"]),
            "crop_selection": FIXED_SELECTION_POLICY,
            "selection_policy": FIXED_SELECTION_POLICY,
            "well_anchor": dict(well_anchor) if well_anchor else None,
            "geometry": output_geometry,
            "well_bundle_npz": str(well_bundle_path),
            "well_bundle_sha256": well_bundle_sha256,
            "output_window": output_window,
        },
        "inference": {
            "device": "cuda",
            "scope": "well_anchored_preview_patch",
            "purpose": "context_halo_cuda_memory_preflight_only",
        },
        "output_directory": str(child_output_directory),
    }


def _default_child_executor(command: list[str], kwargs: Mapping[str, Any]) -> Any:
    return managed_run(command, **dict(kwargs))


def _validated_preflight_receipt(
    *,
    child_result: Mapping[str, Any],
    child_output_directory: Path,
    checkpoint: Path,
    coordinate_contract: Mapping[str, Any],
    coordinate_contract_sha256: str,
    well_bundle_path: Path,
    well_bundle_sha256: str,
    patch_path: Path,
    valid_mask_path: Path,
    config_state_before: Mapping[str, Any],
    config_state_after: Mapping[str, Any],
    checkpoint_state_before: Mapping[str, Any],
    checkpoint_state_after: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_layerpulse_child_result(
        child_result,
        output_root=child_output_directory,
        expected_shape_tix=FIXED_OUTPUT_SHAPE_TIX,
        expected_input_shape_tix=FIXED_INPUT_SHAPE_TIX,
        expected_checkpoint=checkpoint,
    )
    inference = _mapping(validated.get("inference"), label="child inference")
    input_receipt = _mapping(validated.get("input"), label="child input")
    provenance = _mapping(validated.get("provenance"), label="child provenance")
    well_input = _mapping(input_receipt.get("well_input"), label="child well input")
    output_window = _mapping(
        input_receipt.get("context_halo"), label="child context-halo receipt"
    )
    if int(validated.get("checkpoint_forward_calls") or 0) != 1:
        raise LayerPulseContextHaloPreflightError("preflight did not use one forward")
    if tuple(validated.get("input_shape_tix") or ()) != FIXED_INPUT_SHAPE_TIX:
        raise LayerPulseContextHaloPreflightError("preflight input is not 160^3")
    if tuple(validated.get("output_shape_tix") or ()) != FIXED_OUTPUT_SHAPE_TIX:
        raise LayerPulseContextHaloPreflightError("preflight output is not 128^3")
    if int(inference.get("peak_allocated_bytes") or 0) <= 0:
        raise LayerPulseContextHaloPreflightError("peak allocated memory is absent")
    if int(inference.get("peak_reserved_bytes") or 0) <= 0:
        raise LayerPulseContextHaloPreflightError("peak reserved memory is absent")
    if float(inference.get("forward_seconds") or 0.0) <= 0.0:
        raise LayerPulseContextHaloPreflightError("forward duration is absent")
    if (
        output_window.get("enabled") is not True
        or tuple(output_window.get("model_input_shape_tix") or ())
        != FIXED_INPUT_SHAPE_TIX
        or tuple(output_window.get("output_shape_tix") or ())
        != FIXED_OUTPUT_SHAPE_TIX
        or tuple(output_window.get("output_offset_tix") or ()) != FIXED_HALO_TIX
        or tuple(output_window.get("halo_tix") or ()) != FIXED_HALO_TIX
        or output_window.get("output_rule")
        != "central_complete_logits_then_direct_argmax"
    ):
        raise LayerPulseContextHaloPreflightError("child context-halo receipt drifted")
    if tuple(input_receipt.get("crop_start_tix") or ()) != FIXED_CROP_START_TIX:
        raise LayerPulseContextHaloPreflightError("child output coordinate origin moved")
    child_geometry = _mapping(input_receipt.get("geometry"), label="child geometry")
    if child_geometry.get("coordinate_contract_sha256") != coordinate_contract_sha256:
        raise LayerPulseContextHaloPreflightError("child coordinate hash drifted")
    if str(well_input.get("bundle_sha256") or "") != well_bundle_sha256:
        raise LayerPulseContextHaloPreflightError("child well-bundle hash drifted")
    if provenance.get("classification_threshold_used") is not False:
        raise LayerPulseContextHaloPreflightError("classification threshold was used")
    if provenance.get("connected_component_cleanup_used") is not False:
        raise LayerPulseContextHaloPreflightError("connected-component cleanup was used")
    if config_state_before != config_state_after:
        raise LayerPulseContextHaloPreflightError("platform config changed during preflight")
    if checkpoint_state_before != checkpoint_state_after:
        raise LayerPulseContextHaloPreflightError("checkpoint changed during preflight")
    task_catalog = validated.get("task_catalog") or []
    checks = _mapping(validated.get("checks"), label="child checks")
    if len(task_catalog) != 11 or checks.get("all_11_tasks_present") is not True:
        raise LayerPulseContextHaloPreflightError("child did not retain all 11 tasks")
    if inference.get("complete_logits_retained") is not True:
        raise LayerPulseContextHaloPreflightError("complete classification logits are absent")
    if checks.get("classification_direct_argmax") is not True:
        raise LayerPulseContextHaloPreflightError("classification is not direct argmax")
    if checks.get("all_outputs_finite") is not True:
        raise LayerPulseContextHaloPreflightError("a child output is non-finite")
    if provenance.get("complete_logits_cropped_before_argmax") is not True:
        raise LayerPulseContextHaloPreflightError(
            "child did not crop complete logits before argmax"
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "pass",
        "purpose": "cuda_memory_and_contract_preflight_only",
        "authorization": {
            "preflight": True,
            "enable_execution": True,
            "dual_authorization_satisfied": True,
        },
        "roi": {
            "id": FIXED_ROI_ID,
            "selection_policy": FIXED_SELECTION_POLICY,
            "output_crop_start_tix": list(FIXED_CROP_START_TIX),
            "output_shape_tix": list(FIXED_OUTPUT_SHAPE_TIX),
            "halo_tix": list(FIXED_HALO_TIX),
            "model_input_shape_tix": list(FIXED_INPUT_SHAPE_TIX),
        },
        "execution": {
            "checkpoint_forward_calls": 1,
            "one_checkpoint": True,
            "task_count": len(task_catalog),
            "peak_allocated_bytes": int(inference["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(inference["peak_reserved_bytes"]),
            "forward_seconds": float(inference["forward_seconds"]),
            "device": inference.get("device"),
            "device_name": inference.get("device_name"),
        },
        "output_contract": {
            "all_11_tasks_present": checks.get("all_11_tasks_present") is True,
            "complete_classification_logits_retained": bool(
                inference.get("complete_logits_retained")
            ),
            "classification_decode": (
                "central_complete_logits_then_direct_argmax_dim1"
            ),
            "classification_direct_argmax": checks.get(
                "classification_direct_argmax"
            )
            is True,
            "classification_threshold_used": False,
            "connected_component_cleanup_used": False,
        },
        "identity": {
            "checkpoint": dict(checkpoint_state_after),
            "coordinate_contract": dict(coordinate_contract),
            "coordinate_contract_sha256": coordinate_contract_sha256,
            "well_bundle_path": str(well_bundle_path),
            "well_bundle_sha256": well_bundle_sha256,
            "model_input_patch_path": str(patch_path),
            "model_input_patch_sha256": _sha256_file(patch_path),
            "model_input_valid_mask_path": str(valid_mask_path),
            "model_input_valid_mask_sha256": _sha256_file(valid_mask_path),
        },
        "mutation_policy": {
            "checkpoint_updated": False,
            "platform_config_updated": False,
            "automatic_activation_performed": False,
            "eligible_for_separate_promotion_review": True,
            "required_next_action": (
                "independent reviewed operation may update config only after "
                "this receipt and output quality are accepted"
            ),
            "config_state": dict(config_state_after),
        },
        "checks": {
            "one_forward": True,
            "input_160_output_128": True,
            "all_outputs_finite": checks.get("all_outputs_finite") is True,
            "coordinate_origin_preserved": tuple(
                input_receipt.get("crop_start_tix") or ()
            )
            == FIXED_CROP_START_TIX,
            "well_bundle_hash_preserved": True,
            "checkpoint_unchanged": True,
            "platform_config_unchanged": True,
        },
        "child_result": validated,
    }


def describe_preflight(project_root: Path) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    config = load_layerpulse_platform_config(project_root)
    inference = _mapping(config.get("inference"), label="LayerPulse inference config")
    candidate = _mapping(
        inference.get("context_halo"), label="LayerPulse context-halo config"
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "mode": "describe_only_no_cuda",
        "roi": {
            "id": FIXED_ROI_ID,
            "anchor_result": str(FIXED_ANCHOR_RESULT),
            "crop_start_tix": list(FIXED_CROP_START_TIX),
            "output_shape_tix": list(FIXED_OUTPUT_SHAPE_TIX),
            "halo_tix": list(FIXED_HALO_TIX),
            "prospective_model_input_shape_tix": list(FIXED_INPUT_SHAPE_TIX),
        },
        "checkpoint": str(_mapping(config["runtime"], label="runtime")["checkpoint"]),
        "authorization": {
            "required_flags": ["--preflight", "--enable-execution"],
            "gpu_execution_started": False,
        },
        "current_candidate_state": candidate,
        "mutation_policy": {
            "checkpoint_update_allowed": False,
            "platform_config_update_allowed": False,
            "automatic_activation_allowed": False,
        },
        "performance_commitment": False,
    }


def run_preflight(
    *,
    project_root: Path,
    output_directory: Path,
    preflight: bool,
    enable_execution: bool,
    child_executor: ChildExecutor | None = None,
) -> dict[str, Any]:
    """Execute the fixed V4-anchor preflight after two authorisations."""

    if not preflight or not enable_execution:
        raise LayerPulseContextHaloPreflightError(
            "CUDA preflight requires both --preflight and --enable-execution"
        )
    project_root = project_root.expanduser().resolve()
    output_root = output_directory.expanduser().resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"preflight output is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise LayerPulseContextHaloPreflightError(
                "preflight output directory must be new or empty"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    receipt_path = output_root / "layerpulse_context_halo_cuda_preflight.json"
    platform_config = load_layerpulse_platform_config(project_root)
    runtime_paths = _runtime_paths(project_root, platform_config)
    config_path = Path(str(platform_config["_config_path"])).resolve(strict=True)
    checkpoint = runtime_paths["checkpoint"]
    config_state_before = {
        **_file_state(config_path),
        "sha256": _sha256_file(config_path),
    }
    checkpoint_state_before = _file_state(checkpoint)
    try:
        anchor_document = _read_json(FIXED_ANCHOR_RESULT.resolve(strict=True))
        anchor = _anchor_contract(anchor_document, checkpoint=checkpoint)
        options = _prepared_view_options(anchor)
        config = load_config(project_root / "configs", {"inputs": []})
        adapter = LayerPulseInputAdapter(config)
        batch = adapter.prepare_with_context(
            ModelInputRequest(
                source=anchor["source"],
                crop_start=FIXED_CROP_START_TIX,
                crop_size=FIXED_OUTPUT_SHAPE_TIX,
                options=options,
            ),
            context_halo_tix=FIXED_HALO_TIX,
        )
        patch = np.asarray(batch.array, dtype=np.float32)
        valid_mask = np.asarray(batch.valid_mask, dtype=np.bool_)
        provenance = _mapping(batch.provenance, label="context adapter provenance")
        if patch.shape != FIXED_INPUT_SHAPE_TIX or valid_mask.shape != patch.shape:
            raise LayerPulseContextHaloPreflightError(
                "context adapter did not materialize 160^3 patch/mask"
            )
        if tuple(provenance.get("crop_start_tix") or ()) != FIXED_CROP_START_TIX:
            raise LayerPulseContextHaloPreflightError("V4 output origin moved")
        if tuple(provenance.get("crop_size_tix") or ()) != FIXED_OUTPUT_SHAPE_TIX:
            raise LayerPulseContextHaloPreflightError("V4 output shape moved")

        staging = output_root / "input"
        child_output = output_root / "child_outputs"
        staging.mkdir(parents=True, exist_ok=True)
        child_output.mkdir(parents=True, exist_ok=True)
        patch_path = staging / "v4_context_160_tix.npy"
        valid_mask_path = staging / "v4_context_160_valid_mask_tix.npy"
        np.save(patch_path, patch, allow_pickle=False)
        np.save(valid_mask_path, valid_mask, allow_pickle=False)
        well_bundle_path, well_receipt = materialize_layerpulse_well_bundle(
            options,
            platform_config=config,
            input_provenance=provenance,
            destination=staging / "v4_context_160_wells_md_trajectory.npz",
        )
        if well_bundle_path is None:
            raise LayerPulseContextHaloPreflightError(
                "V4 context preflight could not materialize its well bundle: "
                + str(well_receipt.get("reason") or well_receipt.get("status"))
            )
        well_bundle_sha256 = _sha256_file(well_bundle_path)
        coordinate_contract = _coordinate_contract(provenance)
        coordinate_sha256 = _canonical_sha256(coordinate_contract)
        anchor_well = _mapping(anchor["input"].get("well_anchor"), label="well anchor")
        anchor_well.pop("time_depth_table_consumed", None)
        request = _child_request(
            checkpoint=checkpoint,
            source=anchor["source"],
            patch_path=patch_path,
            valid_mask_path=valid_mask_path,
            well_bundle_path=well_bundle_path,
            well_bundle_sha256=well_bundle_sha256,
            provenance=provenance,
            coordinate_contract=coordinate_contract,
            well_anchor=anchor_well,
            child_output_directory=child_output,
        )
        request_path = staging / "layerpulse_context_halo_preflight_request.json"
        child_result_path = output_root / "layerpulse_context_halo_child_result.json"
        runtime_log_path = output_root / "layerpulse_context_halo_runtime.log"
        _atomic_json(request_path, request)
        command = [
            str(runtime_paths["python_executable"]),
            str(runtime_paths["script"]),
            "--request",
            str(request_path),
            "--result",
            str(child_result_path),
        ]
        run_kwargs = {
            "cwd": runtime_paths["layerpulse_root"],
            "env": _subprocess_environment(runtime_paths),
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "check": False,
        }
        executor = child_executor or _default_child_executor
        completed = executor(command, run_kwargs)
        stdout = str(getattr(completed, "stdout", ""))
        stderr = str(getattr(completed, "stderr", ""))
        returncode = int(getattr(completed, "returncode", 1))
        runtime_log_path.write_text(
            "\n".join(
                (
                    f"command={json.dumps(command, ensure_ascii=False)}",
                    f"returncode={returncode}",
                    "[stdout]",
                    stdout.rstrip(),
                    "[stderr]",
                    stderr.rstrip(),
                )
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )
        if returncode != 0:
            raise LayerPulseContextHaloPreflightError(
                f"LayerPulse context-halo child failed with exit {returncode}"
            )
        child_result = _read_json(child_result_path.resolve(strict=True))
        config_state_after = {
            **_file_state(config_path),
            "sha256": _sha256_file(config_path),
        }
        checkpoint_state_after = _file_state(checkpoint)
        receipt = _validated_preflight_receipt(
            child_result=child_result,
            child_output_directory=child_output,
            checkpoint=checkpoint,
            coordinate_contract=coordinate_contract,
            coordinate_contract_sha256=coordinate_sha256,
            well_bundle_path=well_bundle_path,
            well_bundle_sha256=well_bundle_sha256,
            patch_path=patch_path,
            valid_mask_path=valid_mask_path,
            config_state_before=config_state_before,
            config_state_after=config_state_after,
            checkpoint_state_before=checkpoint_state_before,
            checkpoint_state_after=checkpoint_state_after,
        )
        receipt["artifacts"] = {
            "receipt_json": str(receipt_path),
            "child_result_json": str(child_result_path),
            "child_request_json": str(request_path),
            "runtime_log": str(runtime_log_path),
        }
        _atomic_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        failure = {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "mutation_policy": {
                "checkpoint_update_requested": False,
                "platform_config_update_requested": False,
                "automatic_activation_performed": False,
            },
            "artifacts": {"receipt_json": str(receipt_path)},
        }
        _atomic_json(receipt_path, failure)
        raise


__all__ = [
    "FIXED_ANCHOR_RESULT",
    "FIXED_CROP_START_TIX",
    "FIXED_HALO_TIX",
    "FIXED_INPUT_SHAPE_TIX",
    "FIXED_OUTPUT_SHAPE_TIX",
    "FIXED_ROI_ID",
    "PREFLIGHT_SCHEMA",
    "LayerPulseContextHaloPreflightError",
    "describe_preflight",
    "run_preflight",
]
