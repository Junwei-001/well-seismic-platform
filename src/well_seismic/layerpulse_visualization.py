"""Bounded CIGVis payloads for the LayerPulse single-checkpoint result.

The deployed LayerPulse runner persists one deterministic seismic input patch,
one valid mask and eleven aligned task volumes.  This adapter only turns those
already-produced arrays into the platform's existing CIGVis payload contract;
it does not run a model, decode logits, threshold a class, or alter a scientific
output.  RGT isochrons are explicitly marked as display derivatives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .faultseg_visualization import AXES_ZYX, _encode_array, _result_document
from .layerpulse_contract import (
    LAYERPULSE_MODEL_ID,
    LAYERPULSE_OUTPUT_SPECS,
    LAYERPULSE_TASK_ID,
    LayerPulseOutputSpec,
)


LAYERPULSE_VISUALIZATION_GATE_CONTRACT = (
    "well-seismic.layerpulse-visualization-gate.v1"
)
LAYERPULSE_CIGVIS_CONTRACT = "well-seismic.layerpulse-cigvis.v1"
_INPUT_AXES = ("TWT", "INLINE", "XLINE")
_MAX_RENDER_VOXELS = 128 * 128 * 128
_RGT_SURFACE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
_RGT_SURFACE_COLORS = ("#2563eb", "#0d9488", "#d97706", "#dc2626")
_PREVIEW_SCOPE_BY_CROP_SELECTION = {
    "fusion_ready_well_trajectory_anchor": "well_anchored_preview_patch",
    "explicit_geometry_crop": "explicit_preview_patch",
    "fixed_geometry_center": "fixed_geometry_preview_patch",
    "floor_center_with_lower_index_tie_break_v1": "fixed_geometry_preview_patch",
}
_PREVIEW_LABEL_BY_SCOPE = {
    "well_anchored_preview_patch": "井轨迹锚定预览子体",
    "explicit_preview_patch": "用户指定预览子体",
    "fixed_geometry_preview_patch": "固定中心预览子体",
}

_DISPLAY_METADATA: dict[str, dict[str, Any]] = {
    "fault_logits": {
        "name": "断层识别",
        "subject": "断层",
        "cmap": "Reds",
        "alpha": 0.78,
    },
    "unconformity_logits": {
        "name": "不整合与侵蚀势垒",
        "subject": "不整合",
        "cmap": "YlOrBr",
        "alpha": 0.76,
    },
    "facies_logits": {
        "name": "F3 六类地震相",
        "subject": "地震相",
        "cmap": "Set3",
        "alpha": 0.70,
    },
    "channel_logits": {
        "name": "河道内部单元",
        "subject": "河道",
        "cmap": "PuBuGn",
        "alpha": 0.74,
    },
    "karst_logits": {
        "name": "岩溶识别",
        "subject": "岩溶",
        "cmap": "copper",
        "alpha": 0.74,
    },
    "rgt": {
        "name": "相对地质时间 RGT",
        "subject": "相对地质时间",
        "cmap": "viridis",
        "alpha": 0.66,
        "unit": "relative_rgt",
    },
    "impedance": {
        "name": "阻抗预测",
        "subject": "阻抗",
        "cmap": "RdBu_r",
        "alpha": 0.64,
        "unit": "checkpoint_fixed_scale",
    },
    "porosity": {
        "name": "孔隙度预测",
        "subject": "孔隙度",
        "cmap": "viridis",
        "alpha": 0.64,
        "unit": "checkpoint_fixed_scale",
    },
    "well_match": {
        "name": "无时深井震匹配场",
        "subject": "井震匹配",
        "cmap": "BrBG",
        "alpha": 0.66,
        "unit": "relative_match_score",
    },
    "connectivity_logits": {
        "name": "构造连通性",
        "subject": "构造连通域",
        "cmap": "Greens",
        "alpha": 0.72,
    },
    "uncertainty": {
        "name": "局部不确定性",
        "subject": "不确定性",
        "cmap": "magma",
        "alpha": 0.62,
        "unit": "checkpoint_output_scale",
    },
}


@dataclass(frozen=True)
class _ValidatedLayerPulseAssets:
    output_root: Path
    shape_tix: tuple[int, int, int]
    input_patch: np.ndarray
    valid_mask: np.ndarray
    task_arrays: dict[str, np.ndarray]
    task_paths: dict[str, Path]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"LayerPulse visualization is missing {label}")
    return value


def _preview_coverage_contract(
    result: Mapping[str, Any],
    input_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    raw_provenance = result.get("provenance")
    provenance = raw_provenance if isinstance(raw_provenance, Mapping) else {}
    raw_inference = result.get("inference")
    inference = raw_inference if isinstance(raw_inference, Mapping) else {}
    crop_selection = str(
        provenance.get("crop_selection")
        or input_metadata.get("crop_selection")
        or input_metadata.get("selection_policy")
        or inference.get("selection_policy")
        or "fixed_geometry_center"
    ).strip()
    expected_scope = _PREVIEW_SCOPE_BY_CROP_SELECTION.get(crop_selection)
    declared_scope = str(inference.get("scope") or "").strip()
    if expected_scope is not None:
        coverage_mode = expected_scope
    elif declared_scope in _PREVIEW_LABEL_BY_SCOPE:
        coverage_mode = declared_scope
    else:
        coverage_mode = "fixed_geometry_preview_patch"
    return coverage_mode, _PREVIEW_LABEL_BY_SCOPE[coverage_mode]


def _shape3(value: Any, *, label: str, allow_zero: bool = False) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"LayerPulse {label} must contain three integers")
    resolved: list[int] = []
    for item in value:
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"LayerPulse {label} must contain three integers") from exc
        if integer != item or (integer < 0 if allow_zero else integer <= 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"LayerPulse {label} entries must be {qualifier} integers")
        resolved.append(integer)
    return tuple(resolved)  # type: ignore[return-value]


def _resolve_declared_path(
    value: Any,
    *,
    base: Path | None,
    label: str,
    suffix: str,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"LayerPulse visualization is missing outputs.{label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base is None:
            raise ValueError(f"LayerPulse outputs.{label} must be absolute")
        path = base / path
    path = path.resolve()
    if path.suffix.casefold() != suffix or not path.is_file():
        raise FileNotFoundError(f"LayerPulse outputs.{label} is not a valid {suffix} file")
    return path


def _output_root(outputs: Mapping[str, Any], *, metadata_base: Path | None) -> Path:
    result_path = _resolve_declared_path(
        outputs.get("result_json"),
        base=metadata_base,
        label="result_json",
        suffix=".json",
    )
    return result_path.parent


def _contained_npy_path(
    outputs: Mapping[str, Any],
    *,
    output_root: Path,
    artifact_key: str,
) -> Path:
    path = _resolve_declared_path(
        outputs.get(artifact_key),
        base=output_root,
        label=artifact_key,
        suffix=".npy",
    )
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"LayerPulse outputs.{artifact_key} escapes the prediction output root"
        ) from exc
    return path


def _load_npy(path: Path, *, label: str) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"LayerPulse {label} cannot be opened as a safe NPY") from exc
    if array.ndim != 3:
        raise ValueError(f"LayerPulse {label} must be a three-dimensional T/I/X array")
    return array


def _validate_task_catalog(result: Mapping[str, Any], shape: tuple[int, int, int]) -> None:
    raw_catalog = result.get("task_catalog")
    if not isinstance(raw_catalog, Sequence) or isinstance(raw_catalog, (str, bytes)):
        raise ValueError("LayerPulse task_catalog must describe all eleven outputs")
    entries = [item for item in raw_catalog if isinstance(item, Mapping)]
    expected_keys = {spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS}
    actual_keys = {str(item.get("output_key") or "") for item in entries}
    if len(entries) != len(LAYERPULSE_OUTPUT_SPECS) or actual_keys != expected_keys:
        raise ValueError("LayerPulse task_catalog does not describe exactly eleven outputs")
    by_key = {str(item.get("output_key")): item for item in entries}
    for spec in LAYERPULSE_OUTPUT_SPECS:
        entry = by_key[spec.output_key]
        if str(entry.get("artifact_key") or "") != spec.artifact_key:
            raise ValueError(f"LayerPulse {spec.output_key} artifact binding drifted")
        if str(entry.get("kind") or "") != spec.kind:
            raise ValueError(f"LayerPulse {spec.output_key} kind drifted")
        if int(entry.get("channels") or 0) != spec.channels:
            raise ValueError(f"LayerPulse {spec.output_key} channel count drifted")
        expected_background = 0 if spec.kind == "classification" else None
        if entry.get("background_index") != expected_background:
            raise ValueError(f"LayerPulse {spec.output_key} background contract drifted")
        if spec.kind == "classification" and entry.get("selection") not in (
            None,
            "direct_argmax_dim1",
        ):
            raise ValueError(f"LayerPulse {spec.output_key} argmax contract drifted")
        if entry.get("finite") is not True:
            raise ValueError(f"LayerPulse {spec.output_key} lacks a finite output receipt")
        reported_shape = _shape3(
            entry.get("output_shape_tix"),
            label=f"task_catalog.{spec.output_key}.output_shape_tix",
        )
        if reported_shape != shape:
            raise ValueError(f"LayerPulse {spec.output_key} catalog shape drifted")


def _validate_execution_contract(result: Mapping[str, Any]) -> None:
    if str(result.get("model_id") or "") != LAYERPULSE_MODEL_ID:
        raise ValueError("LayerPulse visualization model_id is incompatible")
    if str(result.get("task_id") or "") != LAYERPULSE_TASK_ID:
        raise ValueError("LayerPulse visualization task_id is incompatible")
    if result.get("model_executed") is not True:
        raise ValueError("LayerPulse result does not prove model execution")
    if result.get("single_checkpoint") is not True:
        raise ValueError("LayerPulse result does not prove a single checkpoint")
    if int(result.get("single_forward_calls") or 0) != 1 or int(
        result.get("checkpoint_forward_calls") or 0
    ) != 1:
        raise ValueError("LayerPulse result does not prove exactly one checkpoint forward")

    inference = _mapping(result.get("inference"), label="inference")
    checkpoint = _mapping(result.get("checkpoint"), label="checkpoint")
    provenance = _mapping(result.get("provenance"), label="provenance")
    if int(inference.get("forward_calls") or 0) != 1:
        raise ValueError("LayerPulse inference forward count drifted")
    if int(inference.get("task_count") or 0) != len(LAYERPULSE_OUTPUT_SPECS):
        raise ValueError("LayerPulse inference task count drifted")
    if int(checkpoint.get("head_count") or 0) != len(LAYERPULSE_OUTPUT_SPECS):
        raise ValueError("LayerPulse checkpoint head count drifted")
    if checkpoint.get("strict_model_load") is not True:
        raise ValueError("LayerPulse checkpoint was not loaded strictly")
    if provenance.get("one_forward_returns_all_tasks") is not True:
        raise ValueError("LayerPulse result lacks the one-forward multitask receipt")
    if str(provenance.get("head_input_contract") or "") != "shared_F_final_only":
        raise ValueError("LayerPulse shared F_final head-input contract drifted")


def _load_validated_assets(
    result: Mapping[str, Any], *, metadata_base: Path | None
) -> _ValidatedLayerPulseAssets:
    outputs = _mapping(result.get("outputs"), label="outputs")
    input_metadata = _mapping(result.get("input"), label="input")
    axes = tuple(str(axis).strip().upper() for axis in input_metadata.get("axes", ()))
    if axes != _INPUT_AXES:
        raise ValueError(f"LayerPulse input axes must be {_INPUT_AXES}, got {axes}")

    output_root = _output_root(outputs, metadata_base=metadata_base)
    input_patch_path = _contained_npy_path(
        outputs,
        output_root=output_root,
        artifact_key="input_patch_npy",
    )
    valid_mask_path = _contained_npy_path(
        outputs,
        output_root=output_root,
        artifact_key="input_valid_mask_npy",
    )
    input_patch = _load_npy(input_patch_path, label="input_patch_npy")
    valid_mask_raw = _load_npy(valid_mask_path, label="input_valid_mask_npy")
    shape = tuple(int(value) for value in input_patch.shape)
    if int(np.prod(shape, dtype=np.int64)) > _MAX_RENDER_VOXELS:
        raise ValueError("LayerPulse preview exceeds the bounded CIGVis voxel budget")
    if _shape3(input_metadata.get("shape_tix"), label="input.shape_tix") != shape:
        raise ValueError("LayerPulse input patch shape differs from input.shape_tix")
    if not np.issubdtype(input_patch.dtype, np.floating):
        raise ValueError("LayerPulse input patch must be floating point")
    if not np.all(np.isfinite(input_patch)):
        raise ValueError("LayerPulse input patch contains non-finite values")
    if valid_mask_raw.shape != shape:
        raise ValueError("LayerPulse input valid mask shape differs from the input patch")
    if np.issubdtype(valid_mask_raw.dtype, np.bool_):
        valid_mask = np.asarray(valid_mask_raw, dtype=bool)
    elif np.issubdtype(valid_mask_raw.dtype, np.integer) and np.all(
        np.isin(valid_mask_raw, (0, 1))
    ):
        valid_mask = np.asarray(valid_mask_raw, dtype=bool)
    else:
        raise ValueError("LayerPulse input valid mask must contain only boolean values")
    if not np.any(valid_mask):
        raise ValueError("LayerPulse input valid mask contains no renderable voxel")

    task_arrays: dict[str, np.ndarray] = {}
    task_paths: dict[str, Path] = {}
    for spec in LAYERPULSE_OUTPUT_SPECS:
        path = _contained_npy_path(
            outputs,
            output_root=output_root,
            artifact_key=spec.artifact_key,
        )
        array = _load_npy(path, label=spec.artifact_key)
        if tuple(array.shape) != shape:
            raise ValueError(f"LayerPulse {spec.artifact_key} shape differs from the input patch")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"LayerPulse {spec.artifact_key} contains non-finite values")
        if spec.kind == "classification":
            if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
                array.dtype, np.bool_
            ):
                raise ValueError(f"LayerPulse {spec.artifact_key} must contain integer class ids")
            minimum = int(np.min(array))
            maximum = int(np.max(array))
            if minimum < 0 or maximum >= spec.channels:
                raise ValueError(
                    f"LayerPulse {spec.artifact_key} class ids exceed 0..{spec.channels - 1}"
                )
        elif not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"LayerPulse {spec.artifact_key} must be floating point")
        task_arrays[spec.output_key] = array
        task_paths[spec.output_key] = path

    _validate_task_catalog(result, shape)
    return _ValidatedLayerPulseAssets(
        output_root=output_root,
        shape_tix=shape,
        input_patch=input_patch,
        valid_mask=valid_mask,
        task_arrays=task_arrays,
        task_paths=task_paths,
    )


def evaluate_layerpulse_visualization(
    result_or_metadata: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return a fail-closed LayerPulse CIGVis admission decision.

    The evaluator reopens the bounded NPY artifacts and therefore proves the
    actual arrays that the viewer will consume, rather than trusting only a
    producer-declared renderable flag.
    """

    diagnostics: list[str] = []
    reason_codes: list[str] = []
    model_id = ""
    shape: list[int] = []
    output_count = 0
    try:
        result, metadata_base = _result_document(result_or_metadata)
        model_id = str(result.get("model_id") or "")
        try:
            _validate_execution_contract(result)
        except (TypeError, ValueError) as exc:
            diagnostics.append(str(exc))
            message = str(exc).casefold()
            if "task count" in message or "head count" in message:
                reason_codes.append("eleven_task_contract_failed")
            elif "forward" in message:
                reason_codes.append("single_forward_contract_failed")
            elif "single checkpoint" in message or "checkpoint" in message:
                reason_codes.append("single_checkpoint_contract_failed")
            else:
                reason_codes.append("layerpulse_execution_contract_failed")
        if not diagnostics:
            try:
                assets = _load_validated_assets(result, metadata_base=metadata_base)
                shape = list(assets.shape_tix)
                output_count = len(assets.task_arrays)
            except (OSError, TypeError, ValueError) as exc:
                diagnostics.append(str(exc))
                reason_codes.append("layerpulse_output_artifacts_invalid")
    except (OSError, TypeError, ValueError) as exc:
        diagnostics.append(str(exc))
        reason_codes.append("layerpulse_result_document_invalid")

    return {
        "contract_version": LAYERPULSE_VISUALIZATION_GATE_CONTRACT,
        "model_id": model_id,
        "renderable": not diagnostics,
        "diagnostics": diagnostics,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "output_count": output_count,
        "expected_output_count": len(LAYERPULSE_OUTPUT_SPECS),
        "shape_tix": shape,
        "scientific_output_mutated": False,
        "display_derivatives": ["rgt_relative_isochrons"] if not diagnostics else [],
    }


def _seismic_cube(input_patch: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    valid_values = np.asarray(input_patch[valid_mask], dtype=np.float32)
    scale = float(np.percentile(np.abs(valid_values), 99.0))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = float(np.max(np.abs(valid_values)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = 1.0
    normalized = np.zeros(input_patch.shape, dtype=np.float32)
    normalized[valid_mask] = np.clip(
        np.asarray(input_patch[valid_mask], dtype=np.float32) / scale,
        -1.0,
        1.0,
    )
    return np.rint(normalized * 127.0).astype(np.int8), scale


def _classification_volume(
    array: np.ndarray,
    valid_mask: np.ndarray,
    spec: LayerPulseOutputSpec,
    *,
    source_name: str,
) -> dict[str, Any]:
    display = np.zeros(array.shape, dtype=np.uint8)
    foreground_codes = list(range(1, spec.channels))
    for position, class_code in enumerate(foreground_codes):
        display_code = (
            255
            if len(foreground_codes) == 1
            else int(round(32.0 + position * (223.0 / (len(foreground_codes) - 1))))
        )
        display[valid_mask & (array == class_code)] = np.uint8(display_code)
    return {
        **_encode_array(display, "base64-uint8"),
        "source": source_name,
        "valueRange": [0, spec.channels - 1],
        "classCodes": foreground_codes,
        "classNames": list(spec.class_names[1:]),
        "displayCodeRange": [32, 255],
        "invalidDisplayCode": 0,
        "backgroundClassCode": 0,
        "backgroundClassName": spec.class_names[0],
        "backgroundTransparent": True,
        "readOnlyArgmax": True,
        "selection": "direct_argmax_dim1",
        "samplingAggregation": "identity_class_code",
        "interpolation": "nearest",
    }


def _fault_mask_volume(
    array: np.ndarray,
    valid_mask: np.ndarray,
    *,
    source_name: str,
) -> dict[str, Any]:
    """Encode the persisted binary argmax as an exact display mask.

    This is a visualization type conversion only: membership comes directly
    from class code 1 in the sealed argmax artifact.  No probability, threshold
    or connected-component operation participates in this path.
    """

    display = np.zeros(array.shape, dtype=np.uint8)
    display[valid_mask & (array == 1)] = np.uint8(255)
    return {
        **_encode_array(display, "base64-uint8"),
        "source": source_name,
        "valueRange": [0, 1],
        "displayCodeRange": [0, 255],
        "labelValueRange": [0, 1],
        "classCodes": [0, 1],
        "classNames": ["背景", "断层"],
        "invalidDisplayCode": 0,
        "backgroundDisplayCode": 0,
        "transparentDisplayCode": 0,
        "backgroundClassCode": 0,
        "backgroundClassName": "背景",
        "backgroundTransparent": True,
        "readOnlyArgmax": True,
        "selection": "direct_argmax_dim1",
        "producerSelection": "direct_argmax_dim1",
        "producerThresholdUsed": False,
        "connectedComponentCleanupUsed": False,
        "role": "prediction",
        "displayByDefault": True,
        "samplingAggregation": "identity_binary",
        "interpolation": "nearest",
        "displayMembership": "exact_persisted_class_code_1",
        "renderMode3D": "exact_binary_first_surface",
        "isCompleteBinaryMaskDisplay": True,
        "isModelProbability": False,
    }


def _continuous_volume(
    array: np.ndarray,
    valid_mask: np.ndarray,
    *,
    source_name: str,
    unit: str,
) -> dict[str, Any]:
    values = np.asarray(array[valid_mask], dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    display = np.zeros(array.shape, dtype=np.uint8)
    if maximum > minimum:
        normalized = (values - minimum) / (maximum - minimum)
        display[valid_mask] = np.rint(1.0 + normalized * 254.0).astype(np.uint8)
    else:
        display[valid_mask] = np.uint8(128)
    return {
        **_encode_array(display, "base64-uint8"),
        "source": source_name,
        "valueRange": [minimum, maximum],
        "unit": unit,
        "invalidDisplayCode": 0,
        "validDisplayCodeRange": [1, 255],
        "displayTransform": "linear_minmax_to_uint8_1_255",
        "scientificValuesMutated": False,
    }


def _overlay(
    spec: LayerPulseOutputSpec,
    array: np.ndarray,
    valid_mask: np.ndarray,
    *,
    source_name: str,
) -> dict[str, Any]:
    display = _DISPLAY_METADATA[spec.output_key]
    is_fault_mask = spec.output_key == "fault_logits"
    if is_fault_mask:
        volume = _fault_mask_volume(
            array,
            valid_mask,
            source_name=source_name,
        )
        kind = "mask"
        clim = [0.5, 1.0]
    elif spec.kind == "classification":
        volume = _classification_volume(
            array,
            valid_mask,
            spec,
            source_name=source_name,
        )
        kind = "class_code"
        clim = [32.0 / 255.0, 1.0]
    else:
        volume = _continuous_volume(
            array,
            valid_mask,
            source_name=source_name,
            unit=str(display.get("unit") or "checkpoint_output_scale"),
        )
        kind = "continuous"
        # The first valid display code is 1.  Keeping the lower bound here
        # makes invalid code 0 transparent in the existing threshold renderer
        # while preserving the complete valid scalar range by default.
        clim = [1.0 / 255.0, 1.0]
    overlay = {
        "id": f"layerpulse_{spec.output_key}",
        "outputKey": spec.output_key,
        "artifactKey": spec.artifact_key,
        "name": str(display["name"]),
        "subject": str(display["subject"]),
        "onlyLabel": f"仅{display['subject']}结果",
        "kind": kind,
        "volume": volume,
        "clim": clim,
        "cmap": str(display["cmap"]),
        "alpha": float(display["alpha"]),
        "excpt": "min",
        "scientificSource": "persisted_direct_argmax" if spec.kind == "classification" else "persisted_regression_field",
        "readOnly": True,
    }
    if is_fault_mask:
        # The dedicated mask renderer applies NEAREST sampling in both slice
        # and whole-volume views.  ``volume3D`` is the same exact argmax mask,
        # not a probability or an occupancy-derived field.
        overlay.update(
            {
                "volume3D": volume,
                "boundaryColor": "#ffd230",
                "boundaryAlpha": 0.98,
            }
        )
    return overlay


def _json_grid(array: np.ndarray) -> list[list[float | None]]:
    return [
        [round(float(value), 6) if np.isfinite(value) else None for value in row]
        for row in np.asarray(array, dtype=float)
    ]


def _rgt_isochrons(rgt: np.ndarray, valid_mask: np.ndarray) -> list[dict[str, Any]]:
    valid_values = np.asarray(rgt[valid_mask], dtype=np.float32)
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))
    if not maximum > minimum:
        return []
    distances_template = np.empty(rgt.shape, dtype=np.float32)
    surfaces: list[dict[str, Any]] = []
    for fraction, color in zip(
        _RGT_SURFACE_FRACTIONS,
        _RGT_SURFACE_COLORS,
        strict=True,
    ):
        level = minimum + fraction * (maximum - minimum)
        np.subtract(np.asarray(rgt, dtype=np.float32), level, out=distances_template)
        np.abs(distances_template, out=distances_template)
        distances_template[~valid_mask] = np.inf
        has_value = np.any(valid_mask, axis=0)
        nearest = np.argmin(distances_template, axis=0).astype(np.float32)
        grid = np.full(has_value.shape, np.nan, dtype=np.float32)
        grid[has_value] = nearest[has_value]
        inline_positions, crossline_positions = np.nonzero(has_value)
        points = np.column_stack(
            (
                inline_positions,
                crossline_positions,
                grid[has_value],
            )
        )
        surfaces.append(
            {
                "id": f"layerpulse_rgt_isochron_{int(round(fraction * 100)):02d}",
                "name": f"RGT 等时面 {int(round(fraction * 100))}%",
                "kind": "surface",
                "grid": _json_grid(grid),
                "points": np.round(points, 6).tolist(),
                "color": color,
                "alpha": 0.78,
                "status": "display_only",
                "verticalDomain": "RELATIVE_TIME_NO_T0",
                "verticalUnit": "local_sample_index",
                "rgtLevel": round(float(level), 8),
                "validFraction": float(has_value.mean()),
                "displayDerived": {
                    "contractVersion": "well-seismic.layerpulse-rgt-display-derived.v1",
                    "sourceOutputKey": "rgt",
                    "method": "nearest_valid_sample_to_relative_rgt_level",
                    "scientificOutput": False,
                    "modelForwardOutput": False,
                    "t0Used": False,
                },
            }
        )
    return surfaces


def _axis_values(
    input_metadata: Mapping[str, Any],
    shape: tuple[int, int, int],
) -> tuple[list[float], list[int], list[int], float | None]:
    geometry = input_metadata.get("geometry")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    sample_interval = geometry.get("sample_interval_ms")
    try:
        interval_ms = float(sample_interval)
    except (TypeError, ValueError, OverflowError):
        interval_ms = float("nan")
    if not np.isfinite(interval_ms) or interval_ms <= 0:
        interval_ms = float("nan")
        time_values = [float(index) for index in range(shape[0])]
    else:
        time_values = [round(index * interval_ms, 6) for index in range(shape[0])]

    def horizontal_values(key: str, size: int, fallback_start: int) -> list[int]:
        raw = geometry.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == size:
            try:
                values = [int(item) for item in raw]
            except (TypeError, ValueError, OverflowError):
                values = []
            if len(values) == size:
                return values
        return list(range(fallback_start, fallback_start + size))

    crop_start = _shape3(
        input_metadata.get("crop_start_tix")
        or input_metadata.get("crop_start_zyx")
        or (0, 0, 0),
        label="input.crop_start_tix",
        allow_zero=True,
    )
    return (
        time_values,
        horizontal_values("inline_values", shape[1], crop_start[1]),
        horizontal_values("crossline_values", shape[2], crop_start[2]),
        interval_ms if np.isfinite(interval_ms) else None,
    )


def build_layerpulse_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: Any = None,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = (128, 128, 128),
    layerpulse_output_key: str | None = None,
) -> dict[str, Any]:
    """Build one aligned base volume with eleven selectable LayerPulse layers."""

    # These keyword arguments keep this adapter compatible with the common
    # prediction-visualization dispatcher.  LayerPulse uses its sealed input
    # patch and never reopens an arbitrary SEG-Y path in this layer.
    del cache, config, segy_options
    result, metadata_base = _result_document(result_or_metadata)
    decision = evaluate_layerpulse_visualization(result_or_metadata)
    if decision["renderable"] is not True:
        detail = "; ".join(str(item) for item in decision["diagnostics"])
        raise ValueError(f"LayerPulse visualization gate rejected the result: {detail}")
    assets = _load_validated_assets(result, metadata_base=metadata_base)

    maximum_shape = _shape3(max_shape_zyx, label="max_shape_zyx")
    if any(size > limit for size, limit in zip(assets.shape_tix, maximum_shape, strict=True)):
        raise ValueError(
            "LayerPulse visualization patch exceeds the configured bounded CIGVis shape"
        )
    available_keys = [spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS]
    selected_key = str(layerpulse_output_key or available_keys[0])
    if selected_key not in available_keys:
        raise ValueError(f"unsupported LayerPulse visualization output: {selected_key}")

    input_metadata = _mapping(result.get("input"), label="input")
    shape = assets.shape_tix
    crop_start = _shape3(
        input_metadata.get("crop_start_tix")
        or input_metadata.get("crop_start_zyx")
        or (0, 0, 0),
        label="input.crop_start_tix",
        allow_zero=True,
    )
    source_shape = _shape3(
        input_metadata.get("source_shape_tix")
        or input_metadata.get("source_shape_zyx")
        or shape,
        label="input.source_shape_tix",
    )
    if any(
        start + size > available
        for start, size, available in zip(crop_start, shape, source_shape, strict=True)
    ):
        raise ValueError("LayerPulse preview ROI exceeds the declared source shape")
    time_values, inline_values, crossline_values, interval_ms = _axis_values(
        input_metadata,
        shape,
    )
    cube, amplitude_scale = _seismic_cube(assets.input_patch, assets.valid_mask)
    cube_spec = {
        **_encode_array(cube, "base64-int8"),
        "amplitudeScaleP99": amplitude_scale,
        "source": "input_patch_npy",
    }

    overlays = [
        _overlay(
            spec,
            assets.task_arrays[spec.output_key],
            assets.valid_mask,
            source_name=assets.task_paths[spec.output_key].name,
        )
        for spec in LAYERPULSE_OUTPUT_SPECS
    ]
    overlays.sort(key=lambda item: item["outputKey"] != selected_key)
    rgt_surfaces = _rgt_isochrons(assets.task_arrays["rgt"], assets.valid_mask)
    vertical_unit = "ms relative" if interval_ms is not None else "relative sample"
    sample_indices = {
        "z": list(range(crop_start[0], crop_start[0] + shape[0])),
        "inline": list(range(crop_start[1], crop_start[1] + shape[1])),
        "crossline": list(range(crop_start[2], crop_start[2] + shape[2])),
    }
    coverage_mode, coverage_label = _preview_coverage_contract(
        result,
        input_metadata,
    )
    coverage_notice = (
        f"当前结果仅覆盖{coverage_label} "
        f"{shape[0]}×{shape[1]}×{shape[2]}（源体 "
        f"{source_shape[0]}×{source_shape[1]}×{source_shape[2]}），"
        "不代表完整工区推理。"
    )
    payload: dict[str, Any] = {
        "contractVersion": LAYERPULSE_CIGVIS_CONTRACT,
        "assetKind": "volume",
        "name": f"LayerPulse · {_DISPLAY_METADATA[selected_key]['name']}",
        "path": "input_patch_npy",
        "axes": list(AXES_ZYX),
        "sourceAxes": list(_INPUT_AXES),
        "inlineValues": inline_values,
        "crosslineValues": crossline_values,
        "timeValues": time_values,
        "defaultIndices": [size // 2 for size in shape],
        "cube": cube_spec,
        "sampling": {
            "sourceShapeZYX": list(source_shape),
            "cropStartZYX": list(crop_start),
            "cropSizeZYX": list(shape),
            "strideZYX": [1, 1, 1],
            "sampleIndicesZYX": sample_indices,
            "coverageMode": coverage_mode,
            "isCompleteVolume": False,
        },
        "preview": {
            "loadedTraces": int(np.any(assets.valid_mask, axis=0).sum()),
            "requestedTraces": int(shape[1] * shape[2]),
            "sourceTraceCount": int(source_shape[1] * source_shape[2]),
            "validTraceFraction": float(np.any(assets.valid_mask, axis=0).mean()),
            "validVoxelFraction": float(assets.valid_mask.mean()),
            "amplitudeScaleP99": amplitude_scale,
            "cacheHit": False,
            "cacheKey": "sealed_layerpulse_input_patch",
            "coverageMode": coverage_mode,
            "isCompleteVolume": False,
            "coverageNotice": coverage_notice,
        },
        "axisLabels": {
            "inline": "Inline",
            "crossline": "Xline",
            "sample": "相对时间（无 t0）",
        },
        "verticalAxis": {
            "contractVersion": "well-seismic.vertical-axis.v2",
            "domain": "RELATIVE_TIME",
            "label": "相对时间（无 t0）",
            "unit": vertical_unit,
            "reference": "local_preview_top_zero_no_t0",
            "originKnown": False,
            "t0": None,
            "t0Used": False,
            "correctionState": "relative_only",
            "direction": "increasing_downward",
            "top": time_values[0],
            "bottom": time_values[-1],
            "sampleIntervalMs": interval_ms,
            "defaultView": "top_oblique",
        },
        "sliceViewContract": {
            "contractVersion": "well-seismic.layerpulse-slice-view.v1",
            "modelId": LAYERPULSE_MODEL_ID,
            "defaultPlane": "horizon" if selected_key == "rgt" and rgt_surfaces else "i",
            "allowedPlanes": ["z", "i", "x", "horizon"],
            "preferSlice": True,
            "globalConsistent": False,
            "displayMode": "layerpulse_relative_preview",
            "displayNotice": (
                f"{coverage_notice}"
                "相对时间轴从当前预览顶面起算；未读取、推断或假定统一 t0。"
                "RGT 等时面仅为显示派生，不是额外模型 Head。"
            ),
        },
        "overlays": overlays,
        "surfaces": rgt_surfaces,
        "layerPulse": {
            "modelId": LAYERPULSE_MODEL_ID,
            "taskId": LAYERPULSE_TASK_ID,
            "outputCount": len(overlays),
            "selectedOutputKey": selected_key,
            "singleCheckpoint": True,
            "singleForward": True,
            "relativeTimeNoT0": True,
            "scientificOutputsMutated": False,
            "coverage": {
                "mode": coverage_mode,
                "isCompleteVolume": False,
                "sourceShapeTIX": list(source_shape),
                "cropStartTIX": list(crop_start),
                "cropSizeTIX": list(shape),
                "notice": coverage_notice,
            },
            "displayDerived": {
                "rgtIsochrons": True,
                "surfaceCount": len(rgt_surfaces),
                "sourceOutputKey": "rgt",
                "scientificOutput": False,
            },
            "display": {
                "preferredLayer": selected_key,
                "backgroundCmap": "seismic",
                "classificationBackgroundTransparent": True,
            },
        },
        "visualizationAdmission": decision,
    }
    return payload


__all__ = [
    "LAYERPULSE_CIGVIS_CONTRACT",
    "LAYERPULSE_VISUALIZATION_GATE_CONTRACT",
    "build_layerpulse_visualization_payload",
    "evaluate_layerpulse_visualization",
]
