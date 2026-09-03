"""Fail-closed renderability decisions for unvalidated model candidates.

Candidate renderability is deliberately separate from scientific result-display
acceptance.  It permits an honestly labelled engineering result to be inspected
without manufacturing truth, error, or quantitative validation evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .facies_visualization import F3_FACIES_MODEL_ID
from .fault_models import (
    FAULTNET_MODEL_ID,
    FAULTSEG_MODEL_ID,
    FAULT_VOLUME_MODEL_IDS,
    is_fault_volume_model_id,
)
from .horizon_visualization import (
    HORIZON_MODEL_ID,
    evaluate_horizon_candidate_visualization,
)
from .surface_horizon_display_contract import (
    validate_surface_horizon_display_contract,
)
from .surface_seg_visualization import (
    GLOBAL_RECONCILIATION_SCHEMA,
    SURFACE_SEG_MODEL_ID,
    validate_surface_window_inference_receipt,
)

CANDIDATE_DISPLAY_CONTRACT_VERSION = "well-seismic.candidate-display.v1"
FAULTSEG_SCIENTIFIC_STATUS = "full_volume_fault_probability_candidate"
FAULTSEG_CENTER_BLOCK_SCIENTIFIC_STATUS = (
    "center_block_fault_probability_candidate"
)
FAULTSEG_ADAPTIVE_SCIENTIFIC_STATUS = "adaptive_small_volume_fault_candidate"
FAULTSEG_LEGACY_SCIENTIFIC_STATUS = "legacy_engineering_candidate"
SURFACE_SEG_SCIENTIFIC_STATUS = "legacy_engineering_candidate"
_FAULTSEG_REPRESENTATIVE_GRID_SPECS: dict[str, dict[str, Any]] = {
    "representative_grid_36": {
        "contract_version": "well-seismic.faultseg-representative-grid.v1",
        "grid_shape_zyx": (4, 3, 3),
        "block_count": 36,
    },
    "representative_grid_128": {
        "contract_version": "well-seismic.faultseg-representative-grid.v2",
        "grid_shape_zyx": (8, 4, 4),
        "block_count": 128,
    },
}
SUPPORTED_CANDIDATE_MODEL_IDS = frozenset(
    {
        HORIZON_MODEL_ID,
        *FAULT_VOLUME_MODEL_IDS,
        SURFACE_SEG_MODEL_ID,
        F3_FACIES_MODEL_ID,
    }
)


def _prediction_document(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if "prediction" in result and "model_id" not in result:
        nested = result.get("prediction")
        return nested if isinstance(nested, Mapping) else {}
    return result


def _declared_output(outputs: Mapping[str, Any], name: str) -> Any:
    value = outputs.get(name)
    if isinstance(value, (str, Path)) and str(value).strip():
        return value
    return None


def _shape3(value: Any) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    shape: list[int] = []
    for raw in value:
        if isinstance(raw, (bool, np.bool_)):
            return None
        try:
            item = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if item <= 0 or item != raw:
            return None
        shape.append(item)
    return tuple(shape)  # type: ignore[return-value]


def _start3(value: Any) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    start: list[int] = []
    for raw in value:
        if isinstance(raw, (bool, np.bool_)):
            return None
        try:
            item = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if item < 0 or item != raw:
            return None
        start.append(item)
    return tuple(start)  # type: ignore[return-value]


def _axes3(value: Any) -> tuple[str, str, str] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    return tuple(str(item).upper() for item in value)  # type: ignore[return-value]


def _checkpoint_execution_evidenced(prediction: Mapping[str, Any]) -> bool:
    try:
        forward_calls = int(prediction.get("checkpoint_forward_calls", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        isinstance(prediction.get("checkpoint_forward_calls"), bool)
        or forward_calls <= 0
    ):
        return False
    evidence = prediction.get("checkpoint_evidence")
    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or len(evidence) != forward_calls
    ):
        return False
    for raw in evidence:
        if not isinstance(raw, Mapping):
            return False
        digest = str(raw.get("sha256") or "").casefold()
        if (
            not str(raw.get("stage") or "").strip()
            or not str(raw.get("path") or "").strip()
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    return True


def _npy_header(
    outputs: Mapping[str, Any],
    output_name: str,
) -> tuple[tuple[int, ...] | None, np.dtype[Any] | None, str | None]:
    raw = _declared_output(outputs, output_name)
    if raw is None:
        return None, None, f"{output_name.removesuffix('_npy')}_artifact_missing"
    path = Path(str(raw)).expanduser()
    if path.suffix.casefold() != ".npy" or not path.is_file():
        return None, None, f"{output_name.removesuffix('_npy')}_artifact_missing"
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, TypeError, ValueError):
        return None, None, f"{output_name.removesuffix('_npy')}_artifact_unreadable"
    try:
        return tuple(int(item) for item in array.shape), array.dtype, None
    finally:
        del array


def _surface_global_direction_gate(
    segmentation: Mapping[str, Any],
    outputs: Mapping[str, Any],
    expected_shape: tuple[int, int, int] | None,
) -> dict[str, Any]:
    """Admit cross-direction labels only with sealed reconciliation evidence."""

    reasons: list[str] = []
    if segmentation.get("cross_inline_consistent") is not True:
        reasons.append("cross_inline_consistency_not_declared")
    raw_reconciliation = segmentation.get("global_reconciliation")
    has_reconciliation = isinstance(raw_reconciliation, Mapping) and bool(
        raw_reconciliation
    )
    reconciliation = raw_reconciliation
    if not isinstance(reconciliation, Mapping):
        reasons.append("global_reconciliation_missing")
        reconciliation = {}
    else:
        if reconciliation.get("schema_version") != GLOBAL_RECONCILIATION_SCHEMA:
            reasons.append("global_reconciliation_schema_invalid")
        if reconciliation.get("non_crossing_verified") is not True:
            reasons.append("global_non_crossing_not_verified")
        if reconciliation.get("order_graph_acyclic") is not True:
            reasons.append("global_order_graph_not_acyclic")
        if reconciliation.get("global_display_ready") is not True:
            reasons.append("global_display_ready_not_attested")
        if reconciliation.get("output_semantics") != "global_ordered_package_id":
            reasons.append("global_output_semantics_invalid")
        if not str(reconciliation.get("association_scope") or "").strip():
            reasons.append("global_association_scope_missing")
    try:
        package_count = int(reconciliation.get("global_package_count", -1))
        horizon_count = int(reconciliation.get("global_horizon_count", -1))
        dominant_package_fraction = float(
            reconciliation.get("dominant_global_package_fraction")
        )
        maximum_dominant_package_fraction = float(
            reconciliation.get("maximum_dominant_package_fraction")
        )
    except (TypeError, ValueError, OverflowError):
        package_count = horizon_count = -1
        dominant_package_fraction = maximum_dominant_package_fraction = float("nan")
    local_fallback_semantics = bool(
        has_reconciliation
        and reconciliation.get("global_display_ready") is False
        and reconciliation.get("output_semantics") == "local_inline_fallback"
    )
    insufficient_global_horizons = bool(
        has_reconciliation
        and not local_fallback_semantics
        and (package_count < 2 or horizon_count < 1)
    )
    if (
        package_count < 2
        or horizon_count < 1
        or horizon_count != package_count - 1
    ):
        reasons.append("global_reconciliation_counts_invalid")
    if insufficient_global_horizons:
        reasons.append("insufficient_global_horizons")
    if (
        not np.isfinite(dominant_package_fraction)
        or not np.isfinite(maximum_dominant_package_fraction)
        or not np.isclose(
            maximum_dominant_package_fraction, 0.98, rtol=0.0, atol=1e-12
        )
        or not 0.0 <= dominant_package_fraction < maximum_dominant_package_fraction
    ):
        reasons.append("global_dominant_package_gate_invalid")
    try:
        processed_inline_count = int(
            reconciliation.get("processed_inline_count", -1)
        )
        matched_transition_count = int(
            reconciliation.get("matched_transition_count", -1)
        )
    except (TypeError, ValueError, OverflowError):
        processed_inline_count = matched_transition_count = -1
    transition_receipts = reconciliation.get("transition_receipts")
    if (
        expected_shape is None
        or expected_shape[0] < 2
        or processed_inline_count != expected_shape[0]
        or matched_transition_count <= 0
        or not isinstance(transition_receipts, Sequence)
        or isinstance(transition_receipts, (str, bytes))
        or len(transition_receipts) != expected_shape[0]
    ):
        reasons.append("global_transition_evidence_insufficient")

    global_shape, global_dtype, global_error = _npy_header(
        outputs, "global_mask_npy"
    )
    if global_error:
        reasons.append(global_error)
    elif expected_shape is not None and global_shape != expected_shape:
        reasons.append("global_mask_artifact_shape_mismatch")
    if global_dtype is not None and not np.issubdtype(global_dtype, np.integer):
        reasons.append("global_mask_artifact_dtype_invalid")

    horizon_path_raw = _declared_output(outputs, "horizon_surfaces_npz")
    horizon_display_contract: dict[str, Any] = {
        "valid": False,
        "reason_codes": ["horizon_display_contract_missing"],
        "raw_horizon_count": 0,
        "display_horizon_count": 0,
        "eligible_horizon_ids": [],
        "suppressed_horizon_ids": [],
    }
    if horizon_count > 0:
        horizon_path = (
            Path(str(horizon_path_raw)).expanduser()
            if horizon_path_raw is not None
            else None
        )
        if horizon_path is None or horizon_path.suffix.casefold() != ".npz" or not horizon_path.is_file():
            reasons.append("horizon_surfaces_artifact_missing")
        else:
            horizon_display_contract = validate_surface_horizon_display_contract(
                reconciliation,
                horizon_path,
                global_mask_path=_declared_output(outputs, "global_mask_npy"),
                expected_shape_ics=expected_shape,
            )
            reasons.extend(horizon_display_contract["reason_codes"])

    no_display_eligible_horizons = bool(
        horizon_display_contract["valid"]
        and horizon_count > 0
        and horizon_display_contract["display_horizon_count"] == 0
    )
    if no_display_eligible_horizons:
        reasons.append("no_display_eligible_horizons")
    background_only = bool(
        insufficient_global_horizons or no_display_eligible_horizons
    )
    global_consistent = not reasons
    return {
        "global_consistent": global_consistent,
        "background_only": background_only,
        "display_mode": (
            "background_only" if background_only else "interpretation"
        ),
        "label_scope": (
            "global_packages"
            if global_consistent
            else "unavailable"
            if background_only
            else "inline_local"
        ),
        "default_plane": "i",
        "allowed_planes": (
            ["horizon", "z", "i", "x", "interval-i", "interval-x"]
            if global_consistent
            else ["z", "i", "x"]
            if background_only
            else ["i"]
        ),
        "raw_horizon_count": horizon_display_contract["raw_horizon_count"],
        "display_horizon_count": horizon_display_contract[
            "display_horizon_count"
        ],
        "eligible_horizon_ids": horizon_display_contract[
            "eligible_horizon_ids"
        ],
        "suppressed_horizon_ids": horizon_display_contract[
            "suppressed_horizon_ids"
        ],
        "horizon_display_contract_valid": horizon_display_contract["valid"],
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def evaluate_surface_seg_candidate_visualization(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only a complete, shape-consistent SurfaceSeg engineering result.

    This is a renderability decision, not a scientific or quantitative
    acceptance decision.  Global horizon support is nevertheless replayed
    from the raw NPZ and mask so a copied receipt cannot admit a weak surface.
    """

    prediction = _prediction_document(result)
    reasons: list[str] = []
    if prediction.get("model_id") != SURFACE_SEG_MODEL_ID:
        reasons.append("not_surface_seg_model")
    if prediction.get("model_executed") is not True:
        reasons.append("model_execution_not_evidenced")
    if not _checkpoint_execution_evidenced(prediction):
        reasons.append("checkpoint_execution_evidence_missing")

    scientific_status = str(prediction.get("scientific_status") or "")
    if scientific_status != SURFACE_SEG_SCIENTIFIC_STATUS:
        reasons.append("legacy_engineering_candidate_status_missing")

    input_contract = prediction.get("input")
    segmentation = prediction.get("segmentation")
    if not isinstance(input_contract, Mapping):
        reasons.append("input_contract_missing")
        input_contract = {}
    if not isinstance(segmentation, Mapping):
        reasons.append("segmentation_contract_missing")
        segmentation = {}

    expected_axes = ("INLINE", "CROSSLINE", "SAMPLE")
    if _axes3(input_contract.get("axes")) != expected_axes:
        reasons.append("input_axes_invalid")
    if _axes3(segmentation.get("axes")) != expected_axes:
        reasons.append("segmentation_axes_invalid")

    input_shape = _shape3(input_contract.get("shape_ics"))
    segmentation_shape = _shape3(segmentation.get("shape_ics"))
    if input_shape is None:
        reasons.append("input_shape_invalid")
    if segmentation_shape is None:
        reasons.append("segmentation_shape_invalid")
    if (
        input_shape is not None
        and segmentation_shape is not None
        and input_shape != segmentation_shape
    ):
        reasons.append("input_segmentation_shape_mismatch")

    source_shape_zyx = _shape3(input_contract.get("source_shape_zyx"))
    if source_shape_zyx is None:
        reasons.append("source_shape_invalid")
    elif segmentation_shape is not None:
        output_inline, output_crossline, output_sample = segmentation_shape
        source_sample, source_inline, source_crossline = source_shape_zyx
        if (
            output_sample != source_sample
            or output_crossline != source_crossline
            or output_inline > source_inline
        ):
            reasons.append("segmentation_source_shape_mismatch")
        elif output_inline < source_inline:
            inference = prediction.get("inference")
            raw_max_inlines = (
                inference.get("max_inlines") if isinstance(inference, Mapping) else None
            )
            try:
                max_inlines = int(raw_max_inlines)
            except (TypeError, ValueError, OverflowError):
                max_inlines = 0
            if (
                isinstance(raw_max_inlines, bool)
                or max_inlines <= 0
                or max_inlines != raw_max_inlines
                or output_inline != min(max_inlines, source_inline)
            ):
                reasons.append("partial_inline_scope_inconsistent")

    outputs = prediction.get("outputs")
    if not isinstance(outputs, Mapping):
        reasons.append("output_manifest_missing")
        outputs = {}
    mask_shape, mask_dtype, mask_error = _npy_header(outputs, "mask_npy")
    confidence_shape, confidence_dtype, confidence_error = _npy_header(
        outputs, "confidence_npy"
    )
    if mask_error:
        reasons.append(mask_error)
    if confidence_error:
        reasons.append(confidence_error)
    if mask_shape is not None and segmentation_shape is not None:
        if mask_shape != segmentation_shape:
            reasons.append("mask_artifact_shape_mismatch")
    if confidence_shape is not None and segmentation_shape is not None:
        if confidence_shape != segmentation_shape:
            reasons.append("confidence_artifact_shape_mismatch")
    if mask_shape is not None and confidence_shape is not None:
        if mask_shape != confidence_shape:
            reasons.append("mask_confidence_shape_mismatch")
    if mask_dtype is not None and not np.issubdtype(mask_dtype, np.integer):
        reasons.append("mask_artifact_dtype_invalid")
    if confidence_dtype is not None and not np.issubdtype(
        confidence_dtype, np.floating
    ):
        reasons.append("confidence_artifact_dtype_invalid")

    direction_gate = _surface_global_direction_gate(
        segmentation,
        outputs,
        segmentation_shape,
    )
    reconciliation = segmentation.get("global_reconciliation")
    if isinstance(reconciliation, Mapping) and reconciliation:
        local_fallback_semantics = (
            reconciliation.get("global_display_ready") is False
            and reconciliation.get("output_semantics") == "local_inline_fallback"
        )
        local_fallback_available = bool(
            local_fallback_semantics
            and (
                _declared_output(outputs, "local_mask_npy")
                or _declared_output(outputs, "mask_npy")
            )
        )
        if (
            not direction_gate["global_consistent"]
            and not direction_gate.get("background_only")
            and not local_fallback_available
        ):
            reasons.append("local_fallback_artifact_missing")
    abstention = segmentation.get("abstention")
    abstention_reasons: list[str] = []
    if abstention is not None:
        if not isinstance(abstention, Mapping):
            abstention_reasons.append("abstention_contract_invalid")
        else:
            try:
                unknown_label = int(abstention.get("unknown_label"))
                threshold = float(abstention.get("confidence_threshold"))
                valid_count = int(abstention.get("valid_voxel_count"))
                unknown_count = int(abstention.get("unknown_voxel_count"))
                invalid_grid_count = int(
                    abstention.get("invalid_grid_voxel_count", 0)
                )
                unknown_fraction = float(abstention.get("unknown_fraction"))
            except (TypeError, ValueError, OverflowError):
                abstention_reasons.append("abstention_contract_invalid")
            else:
                if (
                    abstention.get("schema_version")
                    != "surface-seg.abstention.v1"
                    or unknown_label != int(segmentation.get("invalid_label", -1))
                    or unknown_label >= 0
                    or not np.isfinite(threshold)
                    or not 0.0 <= threshold <= 1.0
                    or min(valid_count, unknown_count, invalid_grid_count) < 0
                    or unknown_count > valid_count
                    or (
                        segmentation_shape is not None
                        and valid_count + invalid_grid_count
                        != int(np.prod(segmentation_shape))
                    )
                    or not np.isfinite(unknown_fraction)
                    or not np.isclose(
                        unknown_fraction,
                        unknown_count / valid_count if valid_count else 1.0,
                        rtol=1e-6,
                        atol=1e-9,
                    )
                ):
                    abstention_reasons.append("abstention_contract_invalid")
    reasons.extend(abstention_reasons)

    window_reasons: list[str] = []
    inference = prediction.get("inference")
    window = inference.get("window_inference") if isinstance(inference, Mapping) else None
    if window is not None:
        if not isinstance(window, Mapping):
            window_reasons.append("window_inference_contract_invalid")
        else:
            expected_native_shape = (
                (segmentation_shape[2], segmentation_shape[1])
                if segmentation_shape is not None
                else None
            )
            if validate_surface_window_inference_receipt(
                window,
                expected_native_shape=expected_native_shape,
                hard_max_tiles=4,
            ):
                window_reasons.append("window_inference_contract_invalid")
    reasons.extend(window_reasons)

    renderable = not reasons
    return {
        "contract_version": CANDIDATE_DISPLAY_CONTRACT_VERSION,
        "model_id": SURFACE_SEG_MODEL_ID,
        "display_status": "engineering_candidate" if renderable else "unavailable",
        "renderable": renderable,
        "scientific_status": scientific_status or SURFACE_SEG_SCIENTIFIC_STATUS,
        "current_survey_quantitative_acceptance_claimed": False,
        "quantitative_acceptance_claimed": False,
        "truth_metrics_used": False,
        "error_metrics_used": False,
        "direction_gate": direction_gate,
        "unknown_display": {
            "transparent": True,
            "contract_present": isinstance(abstention, Mapping),
            "reason_codes": abstention_reasons,
        },
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def evaluate_faultseg_candidate_visualization(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a full-survey FaultSeg result or a read-only historical result."""

    prediction = _prediction_document(result)
    reasons: list[str] = []
    model_id = str(prediction.get("model_id") or "")
    if not is_fault_volume_model_id(model_id):
        reasons.append("not_faultseg_model")
    if prediction.get("model_executed") is not True:
        reasons.append("model_execution_not_evidenced")
    scientific_status = str(prediction.get("scientific_status") or "")
    inference = prediction.get("inference")
    inference_scope = (
        str(inference.get("scope") or inference.get("faultseg_scope") or "")
        if isinstance(inference, Mapping)
        else ""
    )
    full_volume_mode = inference_scope == "full_volume"
    center_block_mode = (
        model_id == FAULTSEG_MODEL_ID and inference_scope == "center_block_1"
    )
    adaptive_small_volume_mode = bool(
        inference_scope == "debug_crop"
        and isinstance(inference, Mapping)
        and inference.get("adaptive_small_volume_candidate") is True
    )
    representative_grid = prediction.get("representative_grid")
    representative_mode = bool(
        model_id == FAULTSEG_MODEL_ID
        and (
        scientific_status == "representative_sampling_candidate"
        or (
            isinstance(inference, Mapping)
            and str(inference.get("scope") or inference.get("faultseg_scope") or "")
            in _FAULTSEG_REPRESENTATIVE_GRID_SPECS
        )
        or representative_grid is not None
        )
    )
    if representative_mode:
        if scientific_status != "representative_sampling_candidate":
            reasons.append("representative_sampling_candidate_status_missing")
    elif center_block_mode:
        if scientific_status != FAULTSEG_CENTER_BLOCK_SCIENTIFIC_STATUS:
            reasons.append("center_block_fault_probability_status_missing")
    elif adaptive_small_volume_mode:
        if scientific_status != FAULTSEG_ADAPTIVE_SCIENTIFIC_STATUS:
            reasons.append("adaptive_small_volume_fault_candidate_status_missing")
    elif full_volume_mode and scientific_status != FAULTSEG_SCIENTIFIC_STATUS:
        reasons.append("full_volume_fault_probability_status_missing")
    elif not full_volume_mode and scientific_status != FAULTSEG_LEGACY_SCIENTIFIC_STATUS:
        reasons.append("legacy_engineering_candidate_status_missing")

    outputs = prediction.get("outputs")
    if not isinstance(outputs, Mapping):
        reasons.append("output_manifest_missing")
        outputs = {}
    probability_available = False
    primary_display_artifact = "mask_npy"
    block_count = 1
    if representative_mode:
        primary_display_artifact = "representative_grid.blocks[].outputs.mask_npy"
        representative_spec = _FAULTSEG_REPRESENTATIVE_GRID_SPECS.get(
            inference_scope
        )
        if representative_spec is None and isinstance(representative_grid, Mapping):
            representative_spec = next(
                (
                    spec
                    for spec in _FAULTSEG_REPRESENTATIVE_GRID_SPECS.values()
                    if representative_grid.get("contract_version")
                    == spec["contract_version"]
                ),
                None,
            )
        if representative_spec is None:
            reasons.append("representative_grid_contract_invalid")
            representative_spec = {
                "contract_version": "",
                "grid_shape_zyx": (),
                "block_count": 0,
            }
        representative_scope = next(
            (
                scope_name
                for scope_name, spec in _FAULTSEG_REPRESENTATIVE_GRID_SPECS.items()
                if spec == representative_spec
            ),
            "",
        )
        representative_grid_shape = tuple(
            representative_spec["grid_shape_zyx"]
        )
        block_count = int(representative_spec["block_count"])
        if not isinstance(inference, Mapping):
            reasons.append("representative_inference_receipt_missing")
            inference = {}
        try:
            root_forwards = int(prediction.get("checkpoint_forward_calls") or 0)
            inference_forwards = int(inference.get("forward_calls") or 0)
            inference_threshold = float(inference.get("threshold", -1.0))
        except (TypeError, ValueError, OverflowError):
            root_forwards = inference_forwards = 0
            inference_threshold = -1.0
        if root_forwards != block_count or inference_forwards != block_count:
            reasons.append("representative_forward_receipt_invalid")
        if (
            inference.get("scope") != representative_scope
            or inference.get("stitching") is not False
            or list(inference.get("overlap") or []) != [0, 0, 0]
            or inference.get("normalization") != "per_patch_zscore"
            or inference_threshold != 0.518
        ):
            reasons.append("representative_inference_contract_invalid")
        blocks_directory_raw = _declared_output(
            outputs, "representative_blocks_directory"
        )
        receipt_raw = _declared_output(outputs, "representative_grid_receipt_json")
        blocks_directory = (
            Path(str(blocks_directory_raw)).expanduser()
            if blocks_directory_raw is not None
            else None
        )
        receipt_path = (
            Path(str(receipt_raw)).expanduser()
            if receipt_raw is not None
            else None
        )
        if blocks_directory is None or not blocks_directory.is_dir():
            reasons.append("representative_blocks_directory_missing")
        if receipt_path is None or receipt_path.suffix.casefold() != ".json" or not receipt_path.is_file():
            reasons.append("representative_grid_receipt_missing")
        if not isinstance(representative_grid, Mapping):
            reasons.append("representative_grid_contract_missing")
            representative_grid = {}
        try:
            grid_threshold = float(representative_grid.get("threshold", -1.0))
            grid_forwards = int(
                representative_grid.get("forward_calls_total") or 0
            )
            union_coverage_fraction = float(
                representative_grid.get("representative_union_coverage_fraction", -1.0)
            )
        except (TypeError, ValueError, OverflowError):
            grid_threshold = -1.0
            grid_forwards = 0
            union_coverage_fraction = -1.0
        if (
            representative_grid.get("contract_version")
            != representative_spec["contract_version"]
            or representative_grid.get("scope") != "representative_sampling"
            or representative_grid.get("is_full_volume") is not False
            or _shape3(representative_grid.get("grid_shape_zyx"))
            != representative_grid_shape
            or _shape3(representative_grid.get("block_shape_zyx"))
            != (128, 128, 128)
            or representative_grid.get("grid_order")
            != "Z_then_INLINE_then_CROSSLINE"
            or representative_grid.get("unique_source_starts") is not True
            or representative_grid.get("inter_block_stitching") is not False
            or list(representative_grid.get("inference_overlap_zyx") or [])
            != [0, 0, 0]
            or grid_threshold != 0.518
            or grid_forwards != block_count
            or not np.isfinite(union_coverage_fraction)
            or not 0.0 < union_coverage_fraction <= 1.0
        ):
            reasons.append("representative_grid_contract_invalid")
        raw_blocks = representative_grid.get("blocks")
        if (
            not isinstance(raw_blocks, Sequence)
            or isinstance(raw_blocks, (str, bytes))
            or len(raw_blocks) != block_count
        ):
            reasons.append("representative_blocks_invalid")
            raw_blocks = []
        unique_ids: set[str] = set()
        unique_starts: set[tuple[int, int, int]] = set()
        unique_artifacts: set[str] = set()
        for ordinal, block in enumerate(raw_blocks):
            if not isinstance(block, Mapping):
                reasons.append("representative_block_contract_invalid")
                continue
            inline_crossline_plane = (
                representative_grid_shape[1] * representative_grid_shape[2]
            )
            z_index = ordinal // inline_crossline_plane
            inline_index = (
                ordinal % inline_crossline_plane
            ) // representative_grid_shape[2]
            crossline_index = ordinal % representative_grid_shape[2]
            expected_id = f"z{z_index:02d}_i{inline_index:02d}_x{crossline_index:02d}"
            try:
                block_ordinal = int(block.get("ordinal", -1))
                block_grid_index = tuple(int(value) for value in block.get("grid_index_zyx", ()))
                block_start = tuple(int(value) for value in block.get("source_start_zyx", ()))
                block_forwards = int(block.get("forward_calls") or 0)
                block_threshold = float(block.get("threshold", -1.0))
                valid_ratio = float(block.get("valid_trace_ratio", -1.0))
                fault_fraction = float(block.get("fault_fraction", -1.0))
            except (TypeError, ValueError, OverflowError):
                reasons.append("representative_block_contract_invalid")
                continue
            block_id = str(block.get("block_id") or "")
            if (
                block_id != expected_id
                or block_ordinal != ordinal
                or block_grid_index != (z_index, inline_index, crossline_index)
                or len(block_start) != 3
                or _shape3(block.get("shape_zyx")) != (128, 128, 128)
                or block_forwards != 1
                or block_threshold != 0.518
                or block.get("normalization") != "per_patch_zscore"
                or not np.isfinite(valid_ratio)
                or not 0.0 <= valid_ratio <= 1.0
                or not np.isfinite(fault_fraction)
                or not 0.0 <= fault_fraction <= 1.0
            ):
                reasons.append("representative_block_contract_invalid")
            unique_ids.add(block_id)
            unique_starts.add(block_start)
            block_outputs = block.get("outputs")
            if not isinstance(block_outputs, Mapping):
                reasons.append("representative_block_outputs_missing")
                continue
            for output_name, suffix in (
                ("mask_npy", ".npy"),
                ("probability_npy", ".npy"),
                ("metadata_json", ".json"),
            ):
                artifact_raw = _declared_output(block_outputs, output_name)
                artifact = (
                    Path(str(artifact_raw)).expanduser()
                    if artifact_raw is not None
                    else None
                )
                if artifact is None or artifact.suffix.casefold() != suffix or not artifact.is_file():
                    reasons.append(f"representative_block_{output_name}_missing")
                    continue
                artifact_token = str(artifact.resolve())
                if artifact_token in unique_artifacts:
                    reasons.append("representative_block_artifacts_not_independent")
                unique_artifacts.add(artifact_token)
            probability_available = probability_available or (
                _declared_output(block_outputs, "probability_npy") is not None
            )
        if len(unique_ids) != block_count or len(unique_starts) != block_count:
            reasons.append("representative_blocks_not_unique")
        if receipt_path is not None and receipt_path.is_file():
            try:
                import json

                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                reasons.append("representative_grid_receipt_unreadable")
            else:
                receipt_blocks = receipt.get("blocks") if isinstance(receipt, Mapping) else None
                if (
                    not isinstance(receipt, Mapping)
                    or receipt.get("contract_version")
                    != representative_spec["contract_version"]
                    or receipt.get("is_full_volume") is not False
                    or not isinstance(receipt_blocks, Sequence)
                    or len(receipt_blocks) != block_count
                ):
                    reasons.append("representative_grid_receipt_invalid")
    else:
        required = {"mask": _declared_output(outputs, "mask_npy")}
        probability_available = _declared_output(outputs, "probability_npy") is not None
        if center_block_mode:
            required["probability"] = _declared_output(outputs, "probability_npy")
            input_metadata = prediction.get("input")
            if not isinstance(input_metadata, Mapping):
                reasons.append("center_block_input_receipt_missing")
                input_metadata = {}
            source_shape = _shape3(input_metadata.get("source_shape_zyx"))
            result_shape = _shape3(
                input_metadata.get("shape_zyx")
                or input_metadata.get("crop_size_zyx")
            )
            crop_start = _start3(input_metadata.get("crop_start_zyx"))
            expected_start = (
                tuple((available - 128) // 2 for available in source_shape)
                if source_shape is not None
                and all(available >= 128 for available in source_shape)
                else None
            )
            center_receipt = input_metadata.get("center_block")
            if (
                result_shape != (128, 128, 128)
                or crop_start != expected_start
                or not isinstance(center_receipt, Mapping)
                or center_receipt.get("contract_version")
                != "well-seismic.faultseg-center-block.v1"
                or center_receipt.get("scope") != "center_block_1"
                or center_receipt.get("block_id") != "center_block_1"
                or _start3(center_receipt.get("source_start_zyx")) != crop_start
                or _shape3(center_receipt.get("shape_zyx")) != result_shape
                or center_receipt.get("selection_policy")
                != "floor_center_with_lower_index_tie_break_v1"
                or center_receipt.get("boundary_policy")
                != "complete_block_inside_source_no_padding_v1"
            ):
                reasons.append("center_block_coverage_receipt_invalid")
            if not isinstance(inference, Mapping):
                reasons.append("center_block_inference_receipt_missing")
            else:
                try:
                    root_forwards = int(
                        prediction.get("checkpoint_forward_calls") or 0
                    )
                    inference_forwards = int(inference.get("forward_calls") or 0)
                except (TypeError, ValueError, OverflowError):
                    root_forwards = inference_forwards = 0
                if (
                    list(inference.get("patch_size") or []) != [128, 128, 128]
                    or list(inference.get("overlap") or []) != [0, 0, 0]
                    or inference.get("weighted_blending") is not False
                    or inference.get("stitching") is not False
                    or inference.get("full_volume_reconstructed") is not False
                    or root_forwards != 1
                    or inference_forwards != 1
                ):
                    reasons.append("center_block_inference_contract_invalid")
        if full_volume_mode:
            required["probability"] = _declared_output(outputs, "probability_npy")
            input_metadata = prediction.get("input")
            if not isinstance(input_metadata, Mapping):
                reasons.append("full_volume_input_receipt_missing")
                input_metadata = {}
            source_shape = _shape3(input_metadata.get("source_shape_zyx"))
            result_shape = _shape3(
                input_metadata.get("shape_zyx")
                or input_metadata.get("crop_size_zyx")
            )
            crop_start = _start3(input_metadata.get("crop_start_zyx"))
            if (
                input_metadata.get("full_volume_executed") is not True
                or source_shape is None
                or result_shape != source_shape
                or crop_start != (0, 0, 0)
            ):
                reasons.append("full_volume_coverage_receipt_invalid")
            if not isinstance(inference, Mapping):
                reasons.append("full_volume_inference_receipt_missing")
            elif (
                list(inference.get("patch_size") or []) != [128, 128, 128]
                or list(inference.get("overlap") or []) != [64, 64, 64]
                or inference.get("weighted_blending") is not True
                or inference.get("stitching") is not True
                or inference.get("full_volume_reconstructed") is not True
                or (
                    model_id == FAULTNET_MODEL_ID
                    and (
                        inference.get("normalization") != "per_patch_minmax"
                        or inference.get("output_activation") != "identity"
                    )
                )
            ):
                reasons.append("full_volume_inference_contract_invalid")
        reasons.extend(
            f"{name}_artifact_missing"
            for name, value in required.items()
            if value is None
        )

    renderable = not reasons
    return {
        "contract_version": CANDIDATE_DISPLAY_CONTRACT_VERSION,
        "model_id": model_id,
        "display_status": "engineering_candidate" if renderable else "unavailable",
        "renderable": renderable,
        "scientific_status": scientific_status or FAULTSEG_SCIENTIFIC_STATUS,
        "primary_display_artifact": primary_display_artifact,
        "representative_sampling": representative_mode,
        "is_full_volume": (
            False
            if representative_mode or center_block_mode
            else True if full_volume_mode else None
        ),
        "block_count": block_count,
        "probability_role": (
            "full_volume_prediction"
            if full_volume_mode
            else "technical_audit_only"
        ),
        "probability_available": probability_available,
        "quantitative_acceptance_claimed": False,
        "truth_metrics_used": False,
        "error_metrics_used": False,
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def evaluate_f3_facies_candidate_visualization(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow a complete F3 transfer result to render without current-survey claims."""

    prediction = _prediction_document(result)
    reasons: list[str] = []
    if prediction.get("model_id") != F3_FACIES_MODEL_ID:
        reasons.append("not_f3_facies_model")
    if prediction.get("model_executed") is not True:
        reasons.append("model_execution_not_evidenced")
    try:
        forward_calls = int(prediction.get("checkpoint_forward_calls", 0))
    except (TypeError, ValueError, OverflowError):
        forward_calls = 0
    if forward_calls <= 0:
        reasons.append("checkpoint_forward_calls_missing")
    scientific_status = str(prediction.get("scientific_status", ""))
    if scientific_status not in {
        "experimental_transfer_candidate",
        "validated_within_dataset",
    }:
        reasons.append("supported_f3_scientific_status_missing")

    outputs = prediction.get("outputs")
    if not isinstance(outputs, Mapping):
        reasons.append("output_manifest_missing")
        outputs = {}
    required_names = (
        "class_code_npy",
        "valid_trace_mask_npy",
    )
    reasons.extend(
        f"{name}_artifact_missing"
        for name in required_names
        if _declared_output(outputs, name) is None
    )

    renderable = not reasons
    return {
        "contract_version": CANDIDATE_DISPLAY_CONTRACT_VERSION,
        "model_id": F3_FACIES_MODEL_ID,
        "display_status": (
            (
                "experimental_candidate"
                if scientific_status == "experimental_transfer_candidate"
                else scientific_status
            )
            if renderable
            else "unavailable"
        ),
        "renderable": renderable,
        "scientific_status": scientific_status,
        "validated_scope": str(
            prediction.get("validated_scope", "F3_dense_benchmark")
        ),
        "current_survey_quantitative_acceptance_claimed": False,
        "current_roi_quantitative_acceptance_claimed": False,
        "quantitative_acceptance_claimed": False,
        "truth_metrics_used": False,
        "error_metrics_used": False,
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def evaluate_candidate_visualization(result: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch a model-specific candidate decision without widening support."""

    prediction = _prediction_document(result)
    model_id = prediction.get("model_id")
    if model_id == HORIZON_MODEL_ID:
        return evaluate_horizon_candidate_visualization(prediction)
    if is_fault_volume_model_id(model_id):
        return evaluate_faultseg_candidate_visualization(prediction)
    if model_id == SURFACE_SEG_MODEL_ID:
        return evaluate_surface_seg_candidate_visualization(prediction)
    if model_id == F3_FACIES_MODEL_ID:
        return evaluate_f3_facies_candidate_visualization(prediction)
    return {
        "contract_version": CANDIDATE_DISPLAY_CONTRACT_VERSION,
        "model_id": str(model_id or ""),
        "display_status": "unavailable",
        "renderable": False,
        "scientific_status": "candidate_visualization_not_supported",
        "quantitative_acceptance_claimed": False,
        "truth_metrics_used": False,
        "error_metrics_used": False,
        "reason_codes": ["candidate_visualization_not_supported"],
    }


__all__ = [
    "CANDIDATE_DISPLAY_CONTRACT_VERSION",
    "FAULTSEG_MODEL_ID",
    "FAULTSEG_SCIENTIFIC_STATUS",
    "SURFACE_SEG_MODEL_ID",
    "SURFACE_SEG_SCIENTIFIC_STATUS",
    "SUPPORTED_CANDIDATE_MODEL_IDS",
    "evaluate_candidate_visualization",
    "evaluate_f3_facies_candidate_visualization",
    "evaluate_faultseg_candidate_visualization",
    "evaluate_surface_seg_candidate_visualization",
]
