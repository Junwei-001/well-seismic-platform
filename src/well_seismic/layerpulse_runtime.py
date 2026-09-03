"""Managed subprocess runtime for LayerPulse single-checkpoint preview inference."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .layerpulse_contract import (
    LAYERPULSE_CHILD_RESULT_SCHEMA,
    LAYERPULSE_CLASSIFICATION_SPECS,
    LAYERPULSE_MODEL_ID,
    LAYERPULSE_OUTPUT_SPECS,
    LAYERPULSE_REQUEST_SCHEMA,
    LAYERPULSE_REQUIRED_ARTIFACT_KEYS,
    LAYERPULSE_TASK_ID,
    load_layerpulse_platform_config,
    validate_layerpulse_request_options,
)
from .layerpulse_exports import materialize_layerpulse_common_exports
from .layerpulse_well_bridge import materialize_layerpulse_well_bundle
from .modeling.input_adapters import (
    ModelInputAdapterRegistry,
    ModelInputRequest,
)
from .task_runtime import managed_run

Progress = Callable[[int, str], None]

_SEGY_READER_OPTION_KEYS = (
    "profile",
    "inline_byte",
    "crossline_byte",
    "x_byte",
    "y_byte",
    "coordinate_scalar_byte",
)

_TASK_PRESENTATION: dict[str, tuple[str, str, str | None]] = {
    "fault_logits": ("structure", "断层识别", None),
    "connectivity_logits": ("structure", "构造连通性", None),
    "unconformity_logits": ("stratigraphy", "不整合与侵蚀势垒", None),
    "rgt": ("stratigraphy", "RGT / 相对地质时间", "relative"),
    "facies_logits": ("deposition", "F3 六类地震相", None),
    "channel_logits": ("deposition", "河道内部单元", None),
    "karst_logits": ("deposition", "岩溶识别", None),
    "impedance": ("property", "阻抗预测", "model_scale"),
    "porosity": ("property", "孔隙度预测", "model_scale"),
    "well_match": ("well_and_reasoning", "无时深井震匹配", "score"),
    "uncertainty": ("well_and_reasoning", "局部不确定性", "score"),
}

_PREVIEW_SCOPE_BY_CROP_SELECTION = {
    "fusion_ready_well_trajectory_anchor": "well_anchored_preview_patch",
    "explicit_geometry_crop": "explicit_preview_patch",
    "fixed_geometry_center": "fixed_geometry_preview_patch",
}


def _sealed_segy_reader_options(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value[key]
        for key in _SEGY_READER_OPTION_KEYS
        if value.get(key) is not None
    }


def _append_warning(result: dict[str, Any], warning: str) -> None:
    raw = result.get("warnings")
    if isinstance(raw, list):
        warnings = [str(item) for item in raw]
    elif raw in (None, ""):
        warnings = []
    else:
        warnings = [str(raw)]
    warnings.append(warning)
    result["warnings"] = warnings


def _materialize_common_exports_nonfatal(
    result: dict[str, Any], *, output_root: Path
) -> None:
    """Publish interoperable files without turning export into an inference gate."""

    try:
        receipt = materialize_layerpulse_common_exports(
            result,
            output_root=output_root,
            strict=False,
        )
        if isinstance(receipt, Mapping) and receipt.get("status") != "available":
            error_count = len(receipt.get("errors") or [])
            _append_warning(
                result,
                (
                    "LayerPulse SEG-Y 常用格式仅部分可用；"
                    f"{error_count} 项几何导出未完成，已保留原始 NPY 结果。"
                ),
            )
    except Exception as exc:
        _append_warning(result, f"LayerPulse 常用格式导出未完成：{exc}")


def _preview_contract(
    provenance: Mapping[str, Any],
    *,
    explicit_crop_requested: bool,
) -> tuple[str, str]:
    fallback_selection = (
        "explicit_geometry_crop" if explicit_crop_requested else "fixed_geometry_center"
    )
    crop_selection = str(
        provenance.get("crop_selection") or fallback_selection
    ).strip()
    try:
        preview_scope = _PREVIEW_SCOPE_BY_CROP_SELECTION[crop_selection]
    except KeyError as exc:
        raise ValueError(
            f"LayerPulse input adapter returned unsupported crop_selection: {crop_selection}"
        ) from exc
    return crop_selection, preview_scope


def _shape3(
    value: Any,
    *,
    field: str,
    allow_zero: bool = False,
) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"LayerPulse {field} must be a three-integer T/I/X shape")
    shape = tuple(int(item) for item in value)
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in shape):
        raise ValueError(f"LayerPulse {field} entries must be >= {minimum}")
    return shape  # type: ignore[return-value]


def _context_halo_plan(
    inference: Mapping[str, Any],
    *,
    requested_output_shape_tix: tuple[int, int, int],
) -> dict[str, Any]:
    """Resolve the server-owned, fail-closed context-halo candidate.

    The browser/API request never participates in this decision.  A candidate
    cannot become active until the server configuration records an accepted
    CUDA memory preflight; this module itself makes no 160-cube performance or
    memory-fit promise.
    """

    raw = inference.get("context_halo")
    config = dict(raw) if isinstance(raw, Mapping) else {}
    if config.get("enabled") is not True:
        return {
            "enabled": False,
            "halo_tix": (0, 0, 0),
            "output_shape_tix": requested_output_shape_tix,
            "model_input_shape_tix": requested_output_shape_tix,
            "validation_status": str(
                config.get("validation_status") or "not_configured"
            ),
        }
    if config.get("automatic_activation") is not False:
        raise ValueError(
            "LayerPulse context halo must use explicit server activation"
        )
    validation_status = str(config.get("validation_status") or "").strip()
    if validation_status != "accepted_after_cuda_memory_preflight":
        raise ValueError(
            "LayerPulse context halo is not accepted for online inference"
        )
    if config.get("cuda_memory_preflight_passed") is not True:
        raise ValueError(
            "LayerPulse context halo requires an accepted CUDA memory preflight"
        )
    configured_output = _shape3(
        config.get("output_size_tix"), field="context_halo.output_size_tix"
    )
    if configured_output != requested_output_shape_tix:
        raise ValueError(
            "LayerPulse context halo is accepted only for its configured output ROI"
        )
    halo = _shape3(
        config.get("halo_tix"), field="context_halo.halo_tix", allow_zero=True
    )
    if not any(halo):
        raise ValueError("LayerPulse enabled context halo cannot be all zeros")
    patch_multiple = int(inference.get("patch_multiple", 16))
    model_input_shape = tuple(
        output + 2 * margin
        for output, margin in zip(configured_output, halo, strict=True)
    )
    if any(size % patch_multiple for size in model_input_shape):
        raise ValueError(
            "LayerPulse context-halo input shape must satisfy the Backbone multiple"
        )
    return {
        "enabled": True,
        "halo_tix": halo,
        "output_shape_tix": configured_output,
        "model_input_shape_tix": model_input_shape,
        "validation_status": validation_status,
        "boundary_mode": "constant_zero_with_explicit_valid_mask",
        "output_rule": "central_complete_logits_then_direct_argmax",
        "single_checkpoint_forward_calls": 1,
        "performance_commitment": False,
    }


def _child_request_well_anchor(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    """Expose only spatial anchor fields to the no-TD child request.

    The full platform provenance deliberately records
    ``time_depth_table_consumed=False``.  The child request validator rejects
    every key whose name denotes a time-depth input, even when the value is
    false, so forwarding the complete audit receipt would create a false
    positive.  Keep that attestation in the server-owned result provenance and
    pass only the spatial crop-selection evidence across the subprocess
    boundary.
    """

    raw = provenance.get("well_anchor")
    if not isinstance(raw, Mapping):
        return None
    allowed = (
        "well_uid",
        "well_name",
        "inline",
        "crossline",
        "horizontal_grid_span",
        "fusion_ready_point_count",
        "selection_policy",
    )
    return {name: raw[name] for name in allowed if raw.get(name) is not None}


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"LayerPulse {field} must be an object")
    return dict(value)


def _configured_path(
    value: Any,
    *,
    field: str,
    relative_root: Path | None = None,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"LayerPulse {field} is not configured")
    path = Path(text).expanduser()
    if not path.is_absolute():
        if relative_root is None:
            raise ValueError(f"LayerPulse {field} must be absolute")
        path = relative_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LayerPulse {field} not found: {path}")
    return path


def _runtime_paths(
    project_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Path]:
    runtime = _mapping(config.get("runtime"), field="runtime config")

    def configured_value(environment_name: str, field: str) -> Any:
        return os.getenv(environment_name) or runtime.get(field)

    def configured_directory(value: Any, *, field: str) -> Path:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"LayerPulse {field} is not configured")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"LayerPulse {field} not found: {path}")
        return path

    layerpulse_root = configured_directory(
        configured_value("LAYERPULSE_PROJECT_ROOT", "project_root"),
        field="project root",
    )
    if not (layerpulse_root / "src" / "layerpulse").is_dir():
        raise FileNotFoundError(f"LayerPulse source root not found: {layerpulse_root}")
    python_executable = _configured_path(
        configured_value("LAYERPULSE_PYTHON", "python_executable"),
        field="Python executable",
        relative_root=project_root,
    )
    checkpoint = _configured_path(
        configured_value("LAYERPULSE_CHECKPOINT", "checkpoint"),
        field="checkpoint",
        relative_root=project_root,
    )
    delivery_config = _configured_path(
        configured_value("LAYERPULSE_DELIVERY_CONFIG", "delivery_config"),
        field="delivery config",
        relative_root=project_root,
    )
    script = _configured_path(
        runtime.get("platform_script"),
        field="platform inference script",
        relative_root=project_root,
    )
    ncs_source = configured_directory(
        configured_value("LAYERPULSE_NCS_SOURCE_ROOT", "ncs_source_root"),
        field="NCS source root",
    )
    return {
        "layerpulse_root": layerpulse_root,
        "python_executable": python_executable,
        "checkpoint": checkpoint,
        "delivery_config": delivery_config,
        "script": script,
        "ncs_source": ncs_source,
    }


def _subprocess_environment(paths: Mapping[str, Path]) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [
        str(paths["layerpulse_root"] / "src"),
        str(paths["ncs_source"]),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["LAYERPULSE_CHECKPOINT"] = str(paths["checkpoint"])
    return environment


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(
    outputs: Mapping[str, Any],
    key: str,
    *,
    output_root: Path,
) -> Path:
    value = outputs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LayerPulse child result is missing output path: {key}")
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"LayerPulse output escapes its task directory: {key}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"LayerPulse output does not exist: {path}")
    return path


def _validated_volume(path: Path, shape_tix: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))
    if array.shape != shape_tix:
        raise ValueError(f"LayerPulse volume shape {array.shape} differs from {shape_tix}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"LayerPulse output contains non-finite values: {path}")
    return array


def validate_layerpulse_child_result(
    result: Mapping[str, Any],
    *,
    output_root: Path,
    expected_shape_tix: tuple[int, int, int],
    expected_checkpoint: Path,
    expected_input_shape_tix: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Validate the exact 11-task arrays and six complete-logit tensors."""

    document = dict(result)
    if document.get("schema_version") != LAYERPULSE_CHILD_RESULT_SCHEMA:
        raise ValueError("LayerPulse child result schema_version is incompatible")
    if document.get("status") != "pass":
        raise RuntimeError(f"LayerPulse child result did not pass: {document.get('error')}")
    if document.get("task_id") != LAYERPULSE_TASK_ID:
        raise ValueError("LayerPulse child result task_id drifted")
    if document.get("model_id") != LAYERPULSE_MODEL_ID:
        raise ValueError("LayerPulse child result model_id drifted")
    if document.get("model_executed") is not True:
        raise ValueError("LayerPulse child result does not prove model execution")
    if int(document.get("checkpoint_forward_calls") or 0) != 1:
        raise ValueError("LayerPulse preview must execute exactly one checkpoint forward")
    expected_model_input_shape = expected_input_shape_tix or expected_shape_tix
    if (
        tuple(int(item) for item in document.get("input_shape_tix") or ())
        != expected_model_input_shape
    ):
        raise ValueError("LayerPulse child input shape differs from the sealed preview patch")
    reported_output_shape = document.get("output_shape_tix")
    if reported_output_shape is not None and tuple(
        int(item) for item in reported_output_shape
    ) != expected_shape_tix:
        raise ValueError("LayerPulse child output shape differs from the requested ROI")

    checkpoint = _mapping(document.get("checkpoint"), field="checkpoint receipt")
    if Path(str(checkpoint.get("path") or "")).expanduser().resolve() != expected_checkpoint:
        raise ValueError("LayerPulse child loaded a different checkpoint")
    if int(checkpoint.get("parameter_count") or 0) != 174_697_519:
        raise ValueError("LayerPulse checkpoint parameter count drifted")
    if int(checkpoint.get("f_final_channels") or 0) != 96:
        raise ValueError("LayerPulse checkpoint F_final width drifted")
    if int(checkpoint.get("head_count") or 0) != 11:
        raise ValueError("LayerPulse checkpoint head count drifted")
    if checkpoint.get("strict_model_load") is not True:
        raise ValueError("LayerPulse checkpoint was not loaded strictly")
    if checkpoint.get("teacher_required_at_forward") is not False:
        raise ValueError("LayerPulse child unexpectedly requires a teacher at forward")

    outputs = _mapping(document.get("outputs"), field="outputs")
    missing = sorted(LAYERPULSE_REQUIRED_ARTIFACT_KEYS - set(outputs))
    if missing:
        raise ValueError("LayerPulse child outputs are incomplete: " + ", ".join(missing))
    resolved_outputs = {
        key: _artifact_path(outputs, key, output_root=output_root)
        for key in LAYERPULSE_REQUIRED_ARTIFACT_KEYS
    }

    class_arrays: dict[str, np.ndarray] = {}
    for spec in LAYERPULSE_OUTPUT_SPECS:
        volume = _validated_volume(
            resolved_outputs[spec.artifact_key], expected_shape_tix
        )
        if spec.kind == "classification":
            if not np.issubdtype(volume.dtype, np.integer):
                raise ValueError(f"LayerPulse {spec.artifact_key} must be integer class codes")
            if int(volume.min()) < 0 or int(volume.max()) >= spec.channels:
                raise ValueError(f"LayerPulse {spec.artifact_key} class range is invalid")
            class_arrays[spec.output_key] = volume
        elif not np.issubdtype(volume.dtype, np.floating):
            raise ValueError(f"LayerPulse {spec.artifact_key} must be a floating field")

    with np.load(resolved_outputs["complete_logits_npz"], allow_pickle=False) as archive:
        expected_logit_keys = {spec.output_key for spec in LAYERPULSE_CLASSIFICATION_SPECS}
        if set(archive.files) != expected_logit_keys:
            raise ValueError("LayerPulse complete logits container keys drifted")
        for spec in LAYERPULSE_CLASSIFICATION_SPECS:
            logits = np.asarray(archive[spec.output_key])
            if logits.shape != (spec.channels, *expected_shape_tix):
                raise ValueError(f"LayerPulse {spec.output_key} logits shape is invalid")
            if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
                raise ValueError(f"LayerPulse {spec.output_key} logits are not finite floating values")
            direct = np.argmax(logits, axis=0)
            if not np.array_equal(direct, class_arrays[spec.output_key]):
                raise ValueError(
                    f"LayerPulse {spec.output_key} class volume is not direct logits argmax"
                )

    task_catalog = document.get("task_catalog")
    if not isinstance(task_catalog, list) or len(task_catalog) != 11:
        raise ValueError("LayerPulse task catalog must describe exactly 11 tasks")
    catalog_keys = {
        str(item.get("output_key")) for item in task_catalog if isinstance(item, Mapping)
    }
    if catalog_keys != {spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS}:
        raise ValueError("LayerPulse task catalog keys drifted")
    return document


def run_layerpulse_prediction(
    request: ModelInputRequest,
    *,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run one bounded LayerPulse inference without importing it in FastAPI."""

    del threshold, overlap
    runtime_options = {**request.options, **dict(options or {})}
    validate_layerpulse_request_options(runtime_options)
    requested_device = str(device_name or "auto").strip().casefold()
    if requested_device not in {"auto", "cuda"} and not requested_device.startswith("cuda:"):
        raise ValueError("LayerPulse platform inference requires CUDA")
    execution_device = "cuda" if requested_device == "auto" else requested_device

    platform_config = load_layerpulse_platform_config(project_root)
    paths = _runtime_paths(project_root, platform_config)
    inference = _mapping(platform_config.get("inference"), field="inference config")
    default_patch = tuple(int(item) for item in inference.get("default_patch_size_tix", (32, 32, 32)))
    selected_patch = request.crop_size or patch_size or default_patch
    if len(selected_patch) != 3 or any(int(item) < 16 for item in selected_patch):
        raise ValueError("LayerPulse preview patch axes must each be at least 16")
    if any(int(item) % int(inference.get("patch_multiple", 16)) for item in selected_patch):
        raise ValueError("LayerPulse preview patch axes must be divisible by 16")
    selected_output_shape = tuple(int(item) for item in selected_patch)
    context_plan = _context_halo_plan(
        inference,
        requested_output_shape_tix=selected_output_shape,
    )

    if progress:
        progress(
            8,
            (
                "正在解析封存 SEG-Y 几何并物化带上下文的 LayerPulse 预览子体"
                if context_plan["enabled"]
                else "正在解析封存 SEG-Y 几何并物化 LayerPulse 确定性预览子体"
            ),
        )
    adapter = adapters.get(LAYERPULSE_MODEL_ID)
    adapter_request = ModelInputRequest(
        source=request.source,
        crop_start=request.crop_start,
        crop_size=selected_output_shape,
        options=runtime_options,
    )
    if context_plan["enabled"]:
        prepare_with_context = getattr(adapter, "prepare_with_context", None)
        if not callable(prepare_with_context):
            raise TypeError(
                "LayerPulse adapter does not implement the accepted context-halo contract"
            )
        batch = prepare_with_context(
            adapter_request,
            context_halo_tix=context_plan["halo_tix"],
        )
    else:
        batch = adapter.prepare(adapter_request)
    if batch.array is None or batch.valid_mask is None:
        raise ValueError("LayerPulse input adapter did not materialize its preview patch")
    patch = np.asarray(batch.array, dtype=np.float32)
    valid_mask = np.asarray(batch.valid_mask, dtype=np.bool_)
    if patch.ndim != 3 or valid_mask.shape != patch.shape:
        raise ValueError("LayerPulse adapter patch/mask shape contract is invalid")
    if any(int(item) < 16 or int(item) % 16 for item in patch.shape):
        raise ValueError(
            f"LayerPulse resolved preview shape must be divisible by 16 and >=16: {patch.shape}"
        )

    provenance = dict(batch.provenance)
    segy_reader_options = _sealed_segy_reader_options(
        provenance.get("segy_reader_options")
    )
    output_shape = _shape3(
        provenance.get("crop_size_tix") or patch.shape,
        field="adapter crop_size_tix",
    )
    output_offset = _shape3(
        provenance.get("output_offset_in_model_input_tix") or (0, 0, 0),
        field="adapter output_offset_in_model_input_tix",
        allow_zero=True,
    )
    if any(
        offset + count > available
        for offset, count, available in zip(
            output_offset, output_shape, patch.shape, strict=True
        )
    ):
        raise ValueError("LayerPulse output ROI exceeds the materialized model input")
    output_slices = tuple(
        slice(offset, offset + count)
        for offset, count in zip(output_offset, output_shape, strict=True)
    )
    output_patch = np.ascontiguousarray(patch[output_slices], dtype=np.float32)
    output_valid_mask = np.ascontiguousarray(valid_mask[output_slices], dtype=np.bool_)
    if output_patch.shape != output_shape or not np.any(output_valid_mask):
        raise ValueError("LayerPulse resolved output ROI is empty or invalid")
    if context_plan["enabled"] and (
        tuple(int(item) for item in patch.shape)
        != tuple(context_plan["model_input_shape_tix"])
        or output_shape != tuple(context_plan["output_shape_tix"])
        or output_offset != tuple(context_plan["halo_tix"])
    ):
        raise ValueError(
            "LayerPulse adapter geometry differs from the accepted context-halo plan"
        )

    output_root = output_directory.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / "layerpulse_input"
    child_root = output_root / "layerpulse_outputs"
    staging_root.mkdir(parents=True, exist_ok=True)
    child_root.mkdir(parents=True, exist_ok=True)
    patch_path = staging_root / "input_patch_tix.npy"
    valid_mask_path = staging_root / "valid_mask_tix.npy"
    output_patch_path = staging_root / "output_roi_patch_tix.npy"
    output_valid_mask_path = staging_root / "output_roi_valid_mask_tix.npy"
    np.save(patch_path, patch, allow_pickle=False)
    np.save(valid_mask_path, valid_mask, allow_pickle=False)
    np.save(output_patch_path, output_patch, allow_pickle=False)
    np.save(output_valid_mask_path, output_valid_mask, allow_pickle=False)

    crop_selection, preview_scope = _preview_contract(
        provenance,
        explicit_crop_requested=request.crop_start is not None,
    )
    if progress:
        progress(18, "正在将 PreparedView 的 MD、九槽曲线掩码与井轨迹接入 Backbone")
    well_bundle_path, well_bridge_receipt = materialize_layerpulse_well_bundle(
        runtime_options,
        platform_config=config,
        input_provenance=provenance,
        destination=staging_root / "layerpulse_wells_md_trajectory.npz",
    )
    well_bundle_sha256 = (
        _sha256_file(well_bundle_path) if well_bundle_path is not None else None
    )
    request_manifest = {
        "schema_version": LAYERPULSE_REQUEST_SCHEMA,
        "task_id": LAYERPULSE_TASK_ID,
        "model_id": LAYERPULSE_MODEL_ID,
        "checkpoint": str(paths["checkpoint"]),
        "input": {
            "source": str(request.source.expanduser().resolve()),
            "patch_npy": str(patch_path),
            "valid_mask_npy": str(valid_mask_path),
            "axes": ["TWT", "INLINE", "XLINE"],
            "crop_start_tix": list(provenance.get("crop_start_tix") or (0, 0, 0)),
            "source_shape_tix": list(provenance.get("source_shape_tix") or patch.shape),
            "crop_selection": crop_selection,
            "selection_policy": crop_selection,
            "well_anchor": _child_request_well_anchor(provenance),
            "geometry": {
                "sample_interval_ms": provenance.get("sample_interval_ms"),
                "geometry_profile": provenance.get("geometry_profile"),
                "geometry_confidence": provenance.get("geometry_confidence"),
                "segy_reader_options": segy_reader_options,
                "inline_values": (
                    provenance.get("output_inline_values")
                    or provenance.get("inline_values")
                ),
                "crossline_values": (
                    provenance.get("output_crossline_values")
                    or provenance.get("crossline_values")
                ),
                "inline_range": (
                    provenance.get("output_inline_range")
                    or provenance.get("inline_range")
                ),
                "crossline_range": (
                    provenance.get("output_crossline_range")
                    or provenance.get("crossline_range")
                ),
                "coordinate_reference": (
                    provenance.get("coordinate_reference")
                    or "source_seismic_grid_crs_unverified"
                ),
                "coordinate_reference_verified": bool(
                    provenance.get("coordinate_reference_verified")
                ),
                "coordinate_reference_authority": provenance.get(
                    "coordinate_reference_authority"
                ),
            },
        },
        "inference": {"device": execution_device, "scope": preview_scope},
        "output_directory": str(child_root),
    }
    if context_plan["enabled"]:
        request_manifest["input"]["output_window"] = {
            "schema_version": "well-seismic.layerpulse-context-halo-window.v1",
            "enabled": True,
            "model_input_shape_tix": list(patch.shape),
            "output_offset_tix": list(output_offset),
            "output_shape_tix": list(output_shape),
            "halo_tix": list(context_plan["halo_tix"]),
            "model_input_origin_tix": list(
                provenance.get("model_input_origin_tix") or (0, 0, 0)
            ),
            "source_padding_before_tix": list(
                provenance.get("model_input_padding_before_tix") or (0, 0, 0)
            ),
            "source_padding_after_tix": list(
                provenance.get("model_input_padding_after_tix") or (0, 0, 0)
            ),
            "boundary_mode": context_plan["boundary_mode"],
            "output_rule": context_plan["output_rule"],
            "validation_status": context_plan["validation_status"],
            "single_checkpoint_forward_calls": 1,
        }
    if well_bundle_path is not None:
        request_manifest["input"]["well_bundle_npz"] = str(well_bundle_path)
        request_manifest["input"]["well_bundle_sha256"] = well_bundle_sha256
    request_path = staging_root / "layerpulse_inference_request.json"
    child_result_path = output_root / "layerpulse_child_result.json"
    runtime_log = output_root / "layerpulse_runtime.log"
    _write_json(request_path, request_manifest)

    command = [
        str(paths["python_executable"]),
        str(paths["script"]),
        "--request",
        str(request_path),
        "--result",
        str(child_result_path),
    ]
    if progress:
        progress(25, "正在独立 GPU 子进程加载 LayerPulse 单 checkpoint 并一次输出 11 个任务")
    completed = managed_run(
        command,
        cwd=paths["layerpulse_root"],
        env=_subprocess_environment(paths),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    runtime_log.write_text(
        "\n".join(
            (
                f"command={json.dumps(command, ensure_ascii=False)}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
            )
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"LayerPulse subprocess inference failed (exit {completed.returncode}):\n{tail}"
        )
    if not child_result_path.is_file():
        raise FileNotFoundError("LayerPulse subprocess did not create its result JSON")
    raw_result = json.loads(child_result_path.read_text(encoding="utf-8"))
    result = validate_layerpulse_child_result(
        _mapping(raw_result, field="child result"),
        output_root=child_root,
        expected_shape_tix=output_shape,
        expected_checkpoint=paths["checkpoint"],
        expected_input_shape_tix=tuple(int(item) for item in patch.shape),
    )

    result_outputs = _mapping(result.get("outputs"), field="outputs")
    platform_result_path = output_root / "result.json"
    result_outputs.update(
        {
            # Existing visualisation and acceptance consumers receive the
            # coordinate-stable output ROI.  The larger sole-forward input is
            # retained under explicit model_input_* artifact keys.
            "input_patch_npy": str(output_patch_path),
            "input_valid_mask_npy": str(output_valid_mask_path),
            "model_input_patch_npy": str(patch_path),
            "model_input_valid_mask_npy": str(valid_mask_path),
            "request_json": str(request_path),
            "runtime_log": str(runtime_log),
            "child_result_json": str(child_result_path),
            "result_json": str(platform_result_path),
        }
    )
    result["outputs"] = result_outputs
    result_input = _mapping(result.get("input") or {}, field="input")
    child_geometry = result_input.get("geometry")
    result_geometry = (
        dict(child_geometry) if isinstance(child_geometry, Mapping) else {}
    )
    result_geometry.update(dict(request_manifest["input"]["geometry"]))
    result_input.update(
        {
            "source": str(request.source.expanduser().resolve()),
            "patch_npy": str(output_patch_path),
            "valid_mask_npy": str(output_valid_mask_path),
            "shape_tix": list(output_shape),
            "crop_start_tix": list(
                provenance.get("crop_start_tix") or (0, 0, 0)
            ),
            "crop_size_tix": list(output_shape),
            "model_input_patch_npy": str(patch_path),
            "model_input_valid_mask_npy": str(valid_mask_path),
            "model_input_shape_tix": list(patch.shape),
            "output_offset_in_model_input_tix": list(output_offset),
            "context_halo": dict(context_plan),
            "crop_selection": crop_selection,
            "selection_policy": crop_selection,
            "well_anchor": provenance.get("well_anchor"),
            "geometry": result_geometry,
        }
    )
    result["input"] = result_input
    presented_catalog: list[dict[str, Any]] = []
    for raw_entry in result.get("task_catalog") or []:
        entry = dict(raw_entry)
        group, display_name, unit = _TASK_PRESENTATION[str(entry["output_key"])]
        entry.update({"business_group": group, "display_name": display_name})
        if unit is not None:
            entry["unit"] = unit
        presented_catalog.append(entry)
    result["task_catalog"] = presented_catalog
    child_well_input = _mapping(result_input.get("well_input") or {}, field="well input")
    child_well_mode = str(child_well_input.get("mode") or "")
    if child_well_input.get("time_depth_table_consumed") is not False:
        raise ValueError("LayerPulse child must explicitly attest that no time-depth table was consumed")
    if well_bundle_path is None:
        if (
            child_well_mode != "seismic_only_no_td"
            or int(child_well_input.get("well_count") or 0) != 0
            or int(child_well_input.get("valid_station_count") or 0) != 0
        ):
            raise ValueError("LayerPulse child seismic-only receipt differs from dispatch")
        well_input_consumed = False
    else:
        expected_well_count = int(well_bridge_receipt.get("well_count") or 0)
        expected_station_count = int(
            well_bridge_receipt.get("valid_station_count") or 0
        )
        if (
            child_well_mode != "snapshot_wells_md_trajectory_no_td"
            or str(child_well_input.get("bundle_sha256") or "").casefold()
            != well_bundle_sha256
            or int(child_well_input.get("well_count") or 0) != expected_well_count
            or int(child_well_input.get("valid_station_count") or 0)
            != expected_station_count
            or int(child_well_input.get("well_channels") or 0) != 9
            or expected_well_count <= 0
            or expected_station_count <= 0
        ):
            raise ValueError(
                "LayerPulse child well-input receipt differs from the dispatched bundle"
            )
        well_input_consumed = True
    joined_well_ids = [
        str(item.get("well_uid"))
        for item in well_bridge_receipt.get("wells") or []
        if isinstance(item, Mapping) and item.get("well_uid")
    ]
    joined_rows = int(well_bridge_receipt.get("valid_station_count") or 0)
    eligible_in_patch_rows = int(
        well_bridge_receipt.get("eligible_in_patch_station_count")
        or joined_rows
        or 1
    )
    registration_consumption = {
        "status": "consumed" if well_input_consumed else "available_not_used",
        "registration_consumed": well_input_consumed,
        "registration_manifest_sha256": runtime_options.get(
            "registration_manifest_sha256"
        ),
        "registration_points_sha256": runtime_options.get(
            "registration_points_sha256"
        ),
        "joined_well_ids": joined_well_ids,
        "joined_row_count": joined_rows,
        "eligible_in_patch_row_count": eligible_in_patch_rows,
        "registration_total_point_count": int(
            (well_bridge_receipt.get("registration_validation") or {}).get(
                "point_count"
            )
            or 0
        ),
        "join_coverage_fraction": (
            min(1.0, joined_rows / max(eligible_in_patch_rows, 1))
            if well_input_consumed
            else 0.0
        ),
        "feature_channels": [
            "MD",
            "TVD",
            "X",
            "Y",
            "INLINE",
            "XLINE",
            "CURVE_9",
            "CURVE_MISSING_MASK",
        ],
        "time_depth_table_consumed": False,
        "registration_time_fields_consumed": False,
    }
    result_input.update(
        {
            "well_bridge": well_bridge_receipt,
            "well_bundle_npz": (
                str(well_bundle_path) if well_bundle_path is not None else None
            ),
            "well_bundle_sha256": well_bundle_sha256,
            "well_input_consumed": well_input_consumed,
            "prepared_view_consumed": well_input_consumed,
            "registration_consumed": well_input_consumed,
            "registration_consumption": registration_consumption,
        }
    )
    result["input"] = result_input
    result["registration_consumed"] = well_input_consumed
    result["prepared_view_consumed"] = well_input_consumed
    result["well_input_consumed"] = well_input_consumed
    result["full_volume_status"] = "planned_not_executed"
    result_inference = _mapping(result.get("inference") or {}, field="inference")
    result_inference.update(
        {
            "scope": preview_scope,
            "is_complete_volume": False,
            "selection_policy": crop_selection,
            "model_input_shape_tix": list(patch.shape),
            "output_shape_tix": list(output_shape),
            "context_halo_enabled": bool(context_plan["enabled"]),
            "context_halo_validation_status": context_plan["validation_status"],
            "context_halo_performance_commitment": False,
        }
    )
    result["inference"] = result_inference
    result_provenance = _mapping(result.get("provenance") or {}, field="provenance")
    registered_well_assets = _mapping(
        provenance.get("well_assets") or {}, field="provenance.well_assets"
    )
    registered_well_assets.update(
        {
            "forward_mode": (
                "prepared_view_md_trajectory_no_td"
                if well_input_consumed
                else "seismic_only"
            ),
            "well_bundle_materialized": bool(
                well_input_consumed
                and well_bundle_path is not None
                and well_bridge_receipt.get("bundle_materialized") is True
            ),
            "well_input_consumed": well_input_consumed,
            "prepared_view_consumed": well_input_consumed,
            "registration_consumed": well_input_consumed,
            "well_count_in_forward": int(
                well_bridge_receipt.get("well_count") or 0
            ),
            "valid_station_count_in_forward": joined_rows,
            "time_depth_asset_consumed": False,
        }
    )
    result_provenance.update(
        {
            "platform_adapter": type(adapters.get(LAYERPULSE_MODEL_ID)).__name__,
            "source_seismic": str(request.source.expanduser().resolve()),
            "source_snapshot_id": provenance.get("source_snapshot_id"),
            "source_snapshot_fingerprint": provenance.get("source_snapshot_fingerprint"),
            "registered_well_assets": registered_well_assets,
            "crop_selection": crop_selection,
            "preview_scope": preview_scope,
            "well_anchor": provenance.get("well_anchor"),
            "well_input_consumed": well_input_consumed,
            "prepared_view_consumed": well_input_consumed,
            "registration_consumed": well_input_consumed,
            "registration_consumption": registration_consumption,
            "well_bridge": well_bridge_receipt,
            "context_halo": dict(context_plan),
            "model_input_shape_tix": list(patch.shape),
            "output_shape_tix": list(output_shape),
            "output_coordinate_origin_tix": list(
                provenance.get("crop_start_tix") or (0, 0, 0)
            ),
            "classification_postprocess": "complete_logits_central_crop_then_direct_argmax",
            "time_depth_supervision_opened": False,
            "server_owned_config": str(platform_config["_config_path"]),
            "server_owned_delivery_config": str(paths["delivery_config"]),
        }
    )
    result["provenance"] = result_provenance
    _materialize_common_exports_nonfatal(result, output_root=output_root)
    _write_json(platform_result_path, result)
    if progress:
        progress(92, "LayerPulse 11 个任务及完整分类 logits 已通过形状、有限性与 argmax 校验")
    return result


__all__ = ["run_layerpulse_prediction", "validate_layerpulse_child_result"]
