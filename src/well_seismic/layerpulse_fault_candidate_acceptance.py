"""CPU-only historical same-ROI comparison for fault repair checkpoints.

The evaluator consumes already materialised logits.  It never loads a model,
opens CUDA, thresholds a probability, or cleans a connected component.  Every
student mask is re-derived from the complete two-class logits by
``argmax(axis=0)`` and checked against the persisted argmax artifact.

The sealed ``a192`` input is retained only to compare historical candidates on
identical bytes.  It is not the current platform deployment anchor; the active
V4 receipt is owned by the platform runtime and support contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUEST_SCHEMA = "well-seismic.layerpulse-fault-candidate-acceptance-request.v1"
REPORT_SCHEMA = "well-seismic.layerpulse-fault-candidate-acceptance-report.v1"
PLATFORM_ARTIFACT_TYPE = "layerpulse_fault_candidate_same_roi_comparison"

DEFAULT_ANCHOR_INPUT = (
    Path("model_outputs")
    / "layerpulse_layerpulse_geochronograph_f3x200cf_a1922693"
    / "layerpulse_input"
    / "input_patch_tix.npy"
)
DEFAULT_ANCHOR_VALID_MASK = DEFAULT_ANCHOR_INPUT.with_name("valid_mask_tix.npy")
DEFAULT_CURRENT_RESULT = (
    DEFAULT_ANCHOR_INPUT.parents[1] / "layerpulse_child_result.json"
)
DEFAULT_INPUT_SHA256 = (
    "507ab452a409af01eff1cb0993e2a118aea6e301521f5751755fa24b5b177b2b"
)
DEFAULT_SHAPE_TIX = (128, 128, 128)
DEFAULT_CROP_START_TIX = (1186, 1655, 307)
DEFAULT_SELECTION_POLICY = "fusion_ready_well_trajectory_anchor"
REQUIRED_CANDIDATES = ("current", "step300", "step400")
REQUIRED_TEACHERS = ("cig_bench", "b400")

_NO_POSTPROCESSING = {
    "sigmoid": False,
    "softmax": False,
    "threshold": False,
    "connected_component_cleanup": False,
    "morphology": False,
}


class FaultCandidateAcceptanceError(RuntimeError):
    """The offline candidate bundle violates the fixed-ROI contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FaultCandidateAcceptanceError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FaultCandidateAcceptanceError(f"{label} must be a sequence")
    return value


def _shape3(
    value: Any, label: str, *, allow_zero: bool = False
) -> tuple[int, int, int]:
    sequence = _sequence(value, label)
    if len(sequence) != 3:
        raise FaultCandidateAcceptanceError(f"{label} must contain three integers")
    result = tuple(int(item) for item in sequence)
    minimum = 0 if allow_zero else 1
    if any(
        integer != item or integer < minimum for integer, item in zip(result, sequence)
    ):
        raise FaultCandidateAcceptanceError(f"{label} contains invalid integers")
    return result  # type: ignore[return-value]


def _resolve_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FaultCandidateAcceptanceError(f"{label} is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FaultCandidateAcceptanceError(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FaultCandidateAcceptanceError(f"cannot read JSON {path}: {exc}") from exc
    return _mapping(value, str(path))


def _prediction_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model_id" in document and "outputs" in document:
        return document
    for key in ("prediction", "result"):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            if key == "result" and isinstance(nested.get("prediction"), Mapping):
                return _mapping(nested["prediction"], "result.prediction")
            if "model_id" in nested and "outputs" in nested:
                return nested
    raise FaultCandidateAcceptanceError(
        "result JSON has no LayerPulse prediction document"
    )


def _artifact_path(
    outputs: Mapping[str, Any],
    key: str,
    *,
    result_path: Path,
) -> Path:
    return _resolve_path(
        outputs.get(key), base=result_path.parent, label=f"outputs.{key}"
    )


def _load_raw_logits(
    entry: Mapping[str, Any],
    *,
    base: Path,
    result: Mapping[str, Any] | None,
    result_path: Path | None,
) -> tuple[np.ndarray, Path, str]:
    if entry.get("raw_logits_npy") is not None:
        path = _resolve_path(entry["raw_logits_npy"], base=base, label="raw_logits_npy")
        array = np.load(path, allow_pickle=False)
        key = "raw_logits_npy"
    else:
        if result is None or result_path is None:
            raise FaultCandidateAcceptanceError(
                "logit source has neither raw logits nor result JSON"
            )
        outputs = _mapping(result.get("outputs"), "prediction.outputs")
        if outputs.get("fault_logits_npy") is not None:
            path = _artifact_path(outputs, "fault_logits_npy", result_path=result_path)
            array = np.load(path, allow_pickle=False)
            key = "fault_logits"
        else:
            path = _artifact_path(
                outputs, "complete_logits_npz", result_path=result_path
            )
            with np.load(path, allow_pickle=False) as archive:
                if "fault_logits" not in archive.files:
                    raise FaultCandidateAcceptanceError(
                        f"complete logits archive lacks fault_logits: {path}"
                    )
                array = np.asarray(archive["fault_logits"])
            key = "fault_logits"
    array = np.asarray(array)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[0] not in (1, 2):
        raise FaultCandidateAcceptanceError(
            f"raw fault logits must have shape [1|2,T,I,X], got {array.shape}"
        )
    if not bool(np.isfinite(array).all()):
        raise FaultCandidateAcceptanceError(
            f"raw fault logits contain NaN or Inf: {path}"
        )
    return np.ascontiguousarray(array, dtype=np.float32), path, key


def _complete_binary_logits(
    raw: np.ndarray,
    *,
    source_role: str,
    lift: str | None,
) -> np.ndarray:
    if raw.shape[0] == 2:
        if lift not in (None, "none_complete_two_class_raw"):
            raise FaultCandidateAcceptanceError(
                f"{source_role} declares an invalid two-channel logit lift"
            )
        return raw
    if source_role != "teacher" or lift != "background_zero_foreground_raw":
        raise FaultCandidateAcceptanceError(
            "one-channel logits are allowed only for a teacher explicitly lifted as "
            "background_zero_foreground_raw"
        )
    return np.concatenate((np.zeros_like(raw), raw), axis=0)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left64 -= left64.mean()
    right64 -= right64.mean()
    denominator = math.sqrt(float(np.dot(left64, left64) * np.dot(right64, right64)))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None
    value = float(np.dot(left64, right64) / denominator)
    return max(-1.0, min(1.0, value))


def _axis_run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(mask, axis, -1)
    forward = np.zeros(moved.shape, dtype=np.uint16)
    backward = np.zeros(moved.shape, dtype=np.uint16)
    for index in range(moved.shape[-1]):
        previous = forward[..., index - 1] if index else 0
        forward[..., index] = np.where(moved[..., index], previous + 1, 0)
    for index in range(moved.shape[-1] - 1, -1, -1):
        following = backward[..., index + 1] if index + 1 < moved.shape[-1] else 0
        backward[..., index] = np.where(moved[..., index], following + 1, 0)
    lengths = np.where(moved, forward + backward - 1, 0)
    return np.moveaxis(lengths, -1, axis)


def _morphology(mask: np.ndarray) -> dict[str, Any]:
    foreground = int(mask.sum())
    if foreground == 0:
        return {
            "foreground_voxels": 0,
            "mean_six_neighbor_degree": None,
            "isolated_foreground_fraction": None,
            "face_connected_foreground_fraction": None,
            "axis_run_length_mean": [None, None, None],
            "axis_run_length_max": [0, 0, 0],
            "minimum_axis_run_length_mean": None,
            "minimum_axis_run_length_p50": None,
            "minimum_axis_run_length_p90": None,
            "single_voxel_thickness_fraction": None,
            "thin_sheet_voxel_fraction_le_2": None,
        }

    degree = np.zeros(mask.shape, dtype=np.uint8)
    axis_runs: list[np.ndarray] = []
    run_means: list[float] = []
    run_maxima: list[int] = []
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        pair = mask[tuple(lower)] & mask[tuple(upper)]
        degree[tuple(lower)] += pair
        degree[tuple(upper)] += pair

        lengths = _axis_run_lengths(mask, axis)
        values = lengths[mask]
        axis_runs.append(lengths)
        run_means.append(float(values.mean()))
        run_maxima.append(int(values.max()))

    minimum_run = np.minimum(np.minimum(axis_runs[0], axis_runs[1]), axis_runs[2])
    minimum_values = minimum_run[mask].astype(np.float64)
    foreground_degree = degree[mask]
    return {
        "foreground_voxels": foreground,
        "mean_six_neighbor_degree": float(foreground_degree.mean()),
        "isolated_foreground_fraction": float(np.mean(foreground_degree == 0)),
        "face_connected_foreground_fraction": float(np.mean(foreground_degree > 0)),
        "axis_run_length_mean": run_means,
        "axis_run_length_max": run_maxima,
        "minimum_axis_run_length_mean": float(minimum_values.mean()),
        "minimum_axis_run_length_p50": float(np.quantile(minimum_values, 0.5)),
        "minimum_axis_run_length_p90": float(np.quantile(minimum_values, 0.9)),
        "single_voxel_thickness_fraction": float(np.mean(minimum_values <= 1.0)),
        "thin_sheet_voxel_fraction_le_2": float(np.mean(minimum_values <= 2.0)),
    }


def _source_metrics(
    logits: np.ndarray, valid_mask: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels = np.argmax(logits, axis=0).astype(np.uint8)
    foreground = (labels == 1) & valid_mask
    valid_count = int(valid_mask.sum())
    positive_count = int(foreground.sum())
    margin = np.asarray(logits[1] - logits[0], dtype=np.float32)
    metrics = {
        "valid_voxels": valid_count,
        "foreground_voxels": positive_count,
        "foreground_fraction": positive_count / valid_count,
        "margin_mean": float(margin[valid_mask].mean()),
        "margin_p50": float(np.quantile(margin[valid_mask], 0.5)),
        "margin_p90": float(np.quantile(margin[valid_mask], 0.9)),
        "margin_p99": float(np.quantile(margin[valid_mask], 0.99)),
        "morphology_proxies": _morphology(foreground),
    }
    return metrics, labels, margin


def _agreement(
    candidate_labels: np.ndarray,
    candidate_margin: np.ndarray,
    teacher_labels: np.ndarray,
    teacher_margin: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    candidate = (candidate_labels == 1) & valid_mask
    teacher = (teacher_labels == 1) & valid_mask
    intersection = int(np.sum(candidate & teacher))
    union = int(np.sum(candidate | teacher))
    candidate_count = int(candidate.sum())
    teacher_count = int(teacher.sum())
    return {
        "direct_argmax_agreement": float(
            np.mean(candidate_labels[valid_mask] == teacher_labels[valid_mask])
        ),
        "foreground_intersection": intersection,
        "foreground_union": union,
        "foreground_iou": float(intersection / union) if union else 1.0,
        "foreground_dice": (
            float(2 * intersection / (candidate_count + teacher_count))
            if candidate_count + teacher_count
            else 1.0
        ),
        "proxy_precision": (
            float(intersection / candidate_count) if candidate_count else None
        ),
        "proxy_recall": float(intersection / teacher_count) if teacher_count else None,
        "raw_margin_pearson": _pearson(
            candidate_margin[valid_mask], teacher_margin[valid_mask]
        ),
    }


def _input_path_for_entry(
    entry: Mapping[str, Any],
    *,
    base: Path,
    result_path: Path | None,
) -> Path:
    if entry.get("input_patch_npy") is not None:
        return _resolve_path(
            entry["input_patch_npy"], base=base, label="input_patch_npy"
        )
    if result_path is not None:
        inferred = result_path.parent / "layerpulse_input" / "input_patch_tix.npy"
        if inferred.is_file():
            return inferred.resolve()
    raise FaultCandidateAcceptanceError(
        "each candidate/teacher must bind its raw logits to the fixed input_patch_npy"
    )


def _validate_candidate_contract(
    result: Mapping[str, Any],
    *,
    candidate_id: str,
    shape: tuple[int, int, int],
    crop_start: tuple[int, int, int],
    selection_policy: str,
    fixture: bool,
) -> None:
    input_metadata = _mapping(result.get("input"), "prediction.input")
    inference = _mapping(result.get("inference"), "prediction.inference")
    provenance = _mapping(result.get("provenance"), "prediction.provenance")
    if _shape3(input_metadata.get("shape_tix"), "input.shape_tix") != shape:
        raise FaultCandidateAcceptanceError("candidate shape differs from fixed ROI")
    if (
        _shape3(
            input_metadata.get("crop_start_tix"),
            "input.crop_start_tix",
            allow_zero=True,
        )
        != crop_start
    ):
        raise FaultCandidateAcceptanceError(
            "candidate crop start differs from fixed ROI"
        )
    actual_selection = str(
        input_metadata.get("crop_selection")
        or input_metadata.get("selection_policy")
        or provenance.get("crop_selection")
        or ""
    )
    if actual_selection != selection_policy:
        raise FaultCandidateAcceptanceError(
            "candidate is not the fixed well-anchored ROI"
        )
    if (
        result.get("single_checkpoint") is not True
        or int(result.get("single_forward_calls", -1)) != 1
        or inference.get("classification_selection")
        != "complete_logits_direct_argmax_dim1"
        or provenance.get("classification_threshold_used") is not False
        or provenance.get("connected_component_cleanup_used") is not False
    ):
        raise FaultCandidateAcceptanceError(
            "candidate violates single-checkpoint complete-logits direct-argmax contract"
        )
    checkpoint = _mapping(result.get("checkpoint"), "prediction.checkpoint")
    if not fixture:
        if (
            checkpoint.get("strict_model_load") is not True
            or checkpoint.get("teacher_required_at_forward") is not False
            or int(checkpoint.get("parameter_count", -1)) != 174_697_519
            or int(checkpoint.get("f_final_channels", -1)) != 96
            or int(checkpoint.get("head_count", -1)) != 11
        ):
            raise FaultCandidateAcceptanceError(
                "candidate checkpoint architecture/reload receipt differs"
            )
        if candidate_id in {"step300", "step400"}:
            expected_step = 300 if candidate_id == "step300" else 400
            match = re.search(
                r"precision_fault_teacher_repair_step_(\d{6})_v1\.pt$",
                str(checkpoint.get("path") or ""),
                flags=re.IGNORECASE,
            )
            if match is None or int(match.group(1)) != expected_step:
                raise FaultCandidateAcceptanceError(
                    f"candidate {candidate_id} is not the declared repair checkpoint"
                )


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    return float(np.mean(finite)) if finite else None


def describe_request() -> dict[str, Any]:
    """Return the fixed production anchor and a machine-readable request skeleton."""

    return {
        "schema_version": REQUEST_SCHEMA,
        "execution": "cpu_only_saved_logits_no_model_forward",
        "fixed_roi": {
            "input_patch_npy": str(DEFAULT_ANCHOR_INPUT),
            "expected_sha256": DEFAULT_INPUT_SHA256,
            "valid_mask_npy": str(DEFAULT_ANCHOR_VALID_MASK),
            "shape_tix": list(DEFAULT_SHAPE_TIX),
            "crop_start_tix": list(DEFAULT_CROP_START_TIX),
            "selection_policy": DEFAULT_SELECTION_POLICY,
            "current_result_json": str(DEFAULT_CURRENT_RESULT),
        },
        "required_candidate_ids": list(REQUIRED_CANDIDATES),
        "required_teacher_ids": list(REQUIRED_TEACHERS),
        "request_skeleton": {
            "schema_version": REQUEST_SCHEMA,
            "candidates": [
                {"id": "current", "result_json": str(DEFAULT_CURRENT_RESULT)},
                {"id": "step300", "result_json": "<step300 child result.json>"},
                {"id": "step400", "result_json": "<step400 child result.json>"},
            ],
            "teachers": [
                {
                    "id": "cig_bench",
                    "raw_logits_npy": "<CIG-Bench raw [1,T,I,X] logits>",
                    "binary_logit_lift": "background_zero_foreground_raw",
                    "input_patch_npy": str(DEFAULT_ANCHOR_INPUT),
                },
                {
                    "id": "b400",
                    "result_json": "<B400 child result.json>",
                    "binary_logit_lift": "none_complete_two_class_raw",
                },
            ],
        },
        "decoding": "background_inclusive_complete_logits_direct_argmax_axis0",
        "postprocessing": dict(_NO_POSTPROCESSING),
    }


def evaluate_request(
    request: Mapping[str, Any],
    *,
    request_base: Path,
    fixture: bool = False,
) -> dict[str, Any]:
    """Evaluate one complete current/300/400 bundle using NumPy on CPU."""

    if request.get("schema_version") != REQUEST_SCHEMA:
        raise FaultCandidateAcceptanceError("unsupported acceptance request schema")
    roi = request.get("roi")
    roi = _mapping(roi, "roi") if roi is not None else {}
    if fixture:
        anchor_input = _resolve_path(
            roi.get("input_patch_npy"), base=request_base, label="roi.input_patch_npy"
        )
        valid_mask_path = _resolve_path(
            roi.get("valid_mask_npy"), base=request_base, label="roi.valid_mask_npy"
        )
        expected_sha256 = str(roi.get("expected_sha256") or "").lower()
        shape = _shape3(roi.get("shape_tix"), "roi.shape_tix")
        crop_start = _shape3(
            roi.get("crop_start_tix"), "roi.crop_start_tix", allow_zero=True
        )
        selection_policy = str(roi.get("selection_policy") or "")
    else:
        anchor_input = DEFAULT_ANCHOR_INPUT.resolve(strict=True)
        valid_mask_path = DEFAULT_ANCHOR_VALID_MASK.resolve(strict=True)
        expected_sha256 = DEFAULT_INPUT_SHA256
        shape = DEFAULT_SHAPE_TIX
        crop_start = DEFAULT_CROP_START_TIX
        selection_policy = DEFAULT_SELECTION_POLICY
        if roi and any(
            (
                roi.get("input_patch_npy") not in (None, str(DEFAULT_ANCHOR_INPUT)),
                str(roi.get("expected_sha256") or expected_sha256).lower()
                != expected_sha256,
                tuple(roi.get("shape_tix") or shape) != shape,
                tuple(roi.get("crop_start_tix") or crop_start) != crop_start,
                str(roi.get("selection_policy") or selection_policy)
                != selection_policy,
            )
        ):
            raise FaultCandidateAcceptanceError(
                "historical comparison request cannot override the sealed a192 ROI"
            )

    observed_sha256 = _sha256(anchor_input)
    if observed_sha256 != expected_sha256:
        raise FaultCandidateAcceptanceError(
            f"fixed input SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    input_patch = np.load(anchor_input, allow_pickle=False)
    valid_mask = np.asarray(np.load(valid_mask_path, allow_pickle=False), dtype=bool)
    if tuple(input_patch.shape) != shape or tuple(valid_mask.shape) != shape:
        raise FaultCandidateAcceptanceError("fixed input or valid mask shape differs")
    if not bool(np.isfinite(input_patch).all()) or not bool(valid_mask.any()):
        raise FaultCandidateAcceptanceError(
            "fixed input is non-finite or entirely invalid"
        )

    hash_cache: dict[Path, str] = {anchor_input: observed_sha256}

    def require_same_input(
        entry: Mapping[str, Any], result_path: Path | None
    ) -> tuple[Path, str]:
        path = _input_path_for_entry(entry, base=request_base, result_path=result_path)
        digest = hash_cache.setdefault(path, _sha256(path))
        if digest != observed_sha256:
            raise FaultCandidateAcceptanceError(
                "logit source input SHA-256 differs from the sealed historical "
                f"a192 comparison ROI: {path}"
            )
        return path, digest

    raw_candidates = _sequence(request.get("candidates"), "candidates")
    candidate_entries = {
        str(_mapping(item, "candidate").get("id") or ""): _mapping(item, "candidate")
        for item in raw_candidates
    }
    if set(candidate_entries) != set(REQUIRED_CANDIDATES):
        raise FaultCandidateAcceptanceError(
            f"candidate ids must be exactly {REQUIRED_CANDIDATES}"
        )

    candidates: list[dict[str, Any]] = []
    candidate_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_id in REQUIRED_CANDIDATES:
        entry = candidate_entries[candidate_id]
        raw_result_path = entry.get("result_json")
        if candidate_id == "current" and raw_result_path is None and not fixture:
            raw_result_path = DEFAULT_CURRENT_RESULT
        result_path = _resolve_path(
            raw_result_path,
            base=request_base,
            label=f"candidate {candidate_id}.result_json",
        )
        result = _prediction_document(_json(result_path))
        _validate_candidate_contract(
            result,
            candidate_id=candidate_id,
            shape=shape,
            crop_start=crop_start,
            selection_policy=selection_policy,
            fixture=fixture,
        )
        source_input, source_sha256 = require_same_input(entry, result_path)
        raw_logits, logits_path, logits_key = _load_raw_logits(
            entry, base=request_base, result=result, result_path=result_path
        )
        logits = _complete_binary_logits(
            raw_logits,
            source_role="candidate",
            lift=entry.get("binary_logit_lift"),
        )
        if tuple(logits.shape) != (2, *shape):
            raise FaultCandidateAcceptanceError(
                "candidate logits shape differs from fixed ROI"
            )
        metrics, labels, margin = _source_metrics(logits, valid_mask)
        outputs = _mapping(result.get("outputs"), "prediction.outputs")
        candidate_valid_mask_path = _artifact_path(
            outputs, "valid_mask_npy", result_path=result_path
        )
        candidate_valid_mask = np.asarray(
            np.load(candidate_valid_mask_path, allow_pickle=False), dtype=bool
        )
        if not np.array_equal(candidate_valid_mask, valid_mask):
            raise FaultCandidateAcceptanceError(
                f"candidate {candidate_id} valid mask differs from fixed ROI"
            )
        argmax_path = _artifact_path(
            outputs, "fault_argmax_npy", result_path=result_path
        )
        stored_argmax = np.load(argmax_path, allow_pickle=False)
        if tuple(stored_argmax.shape) != shape or not np.array_equal(
            stored_argmax, labels
        ):
            raise FaultCandidateAcceptanceError(
                f"candidate {candidate_id} persisted argmax differs from logits argmax"
            )
        candidates.append(
            {
                "id": candidate_id,
                "role": "stable_anchor"
                if candidate_id == "current"
                else "repair_candidate",
                "result_json": str(result_path),
                "input_patch_npy": str(source_input),
                "input_sha256": source_sha256,
                "raw_logits": {
                    "path": str(logits_path),
                    "key": logits_key,
                    "shape_ctix": list(logits.shape),
                    "finite": True,
                },
                "persisted_argmax_npy": str(argmax_path),
                "valid_mask_npy": str(candidate_valid_mask_path),
                "direct_argmax_exact": True,
                "checkpoint": dict(_mapping(result.get("checkpoint"), "checkpoint")),
                "metrics": metrics,
            }
        )
        candidate_arrays[candidate_id] = (labels, margin)

    raw_teachers = _sequence(request.get("teachers"), "teachers")
    teacher_entries = {
        str(_mapping(item, "teacher").get("id") or ""): _mapping(item, "teacher")
        for item in raw_teachers
    }
    if set(teacher_entries) != set(REQUIRED_TEACHERS):
        raise FaultCandidateAcceptanceError(
            f"teacher ids must be exactly {REQUIRED_TEACHERS}"
        )

    teachers: list[dict[str, Any]] = []
    teacher_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for teacher_id in REQUIRED_TEACHERS:
        entry = teacher_entries[teacher_id]
        result_path = (
            _resolve_path(
                entry["result_json"],
                base=request_base,
                label=f"teacher {teacher_id}.result_json",
            )
            if entry.get("result_json") is not None
            else None
        )
        result = _prediction_document(_json(result_path)) if result_path else None
        source_input, source_sha256 = require_same_input(entry, result_path)
        raw_logits, logits_path, logits_key = _load_raw_logits(
            entry, base=request_base, result=result, result_path=result_path
        )
        expected_teacher_channels = 1 if teacher_id == "cig_bench" else 2
        expected_lift = (
            "background_zero_foreground_raw"
            if teacher_id == "cig_bench"
            else "none_complete_two_class_raw"
        )
        if (
            int(raw_logits.shape[0]) != expected_teacher_channels
            or entry.get("binary_logit_lift") != expected_lift
        ):
            raise FaultCandidateAcceptanceError(
                f"teacher {teacher_id} raw-logit channel/lift contract differs"
            )
        logits = _complete_binary_logits(
            raw_logits,
            source_role="teacher",
            lift=str(entry.get("binary_logit_lift") or "") or None,
        )
        if tuple(logits.shape) != (2, *shape):
            raise FaultCandidateAcceptanceError(
                "teacher logits shape differs from fixed ROI"
            )
        metrics, labels, margin = _source_metrics(logits, valid_mask)
        teachers.append(
            {
                "id": teacher_id,
                "role": "raw_logit_proxy_teacher_not_ground_truth",
                "input_patch_npy": str(source_input),
                "input_sha256": source_sha256,
                "raw_logits": {
                    "path": str(logits_path),
                    "key": logits_key,
                    "stored_channels": int(raw_logits.shape[0]),
                    "complete_channels_after_declared_lift": 2,
                    "shape_ctix": list(logits.shape),
                    "finite": True,
                    "binary_logit_lift": entry.get("binary_logit_lift"),
                },
                "metrics": metrics,
            }
        )
        teacher_arrays[teacher_id] = (labels, margin)

    candidate_by_id = {str(item["id"]): item for item in candidates}
    for candidate_id, candidate in candidate_by_id.items():
        labels, margin = candidate_arrays[candidate_id]
        comparisons = {}
        for teacher_id, (teacher_labels, teacher_margin) in teacher_arrays.items():
            comparisons[teacher_id] = _agreement(
                labels,
                margin,
                teacher_labels,
                teacher_margin,
                valid_mask,
            )
        candidate["teacher_agreement"] = comparisons
        candidate["teacher_aggregate"] = {
            "mean_foreground_iou": _mean(
                [item["foreground_iou"] for item in comparisons.values()]
            ),
            "mean_foreground_dice": _mean(
                [item["foreground_dice"] for item in comparisons.values()]
            ),
            "mean_raw_margin_pearson": _mean(
                [item["raw_margin_pearson"] for item in comparisons.values()]
            ),
        }

    anchor = candidate_by_id["current"]
    anchor_aggregate = _mapping(
        anchor["teacher_aggregate"], "current teacher aggregate"
    )
    anchor_morphology = _mapping(
        _mapping(anchor["metrics"], "current metrics")["morphology_proxies"],
        "current morphology",
    )
    for candidate_id, candidate in candidate_by_id.items():
        aggregate = _mapping(candidate["teacher_aggregate"], "teacher aggregate")
        morphology = _mapping(
            _mapping(candidate["metrics"], "candidate metrics")["morphology_proxies"],
            "candidate morphology",
        )
        candidate["relative_to_current"] = {
            "mean_teacher_dice_delta": float(aggregate["mean_foreground_dice"])
            - float(anchor_aggregate["mean_foreground_dice"]),
            "mean_teacher_iou_delta": float(aggregate["mean_foreground_iou"])
            - float(anchor_aggregate["mean_foreground_iou"]),
            "mean_margin_pearson_delta": (
                float(aggregate["mean_raw_margin_pearson"])
                - float(anchor_aggregate["mean_raw_margin_pearson"])
                if aggregate["mean_raw_margin_pearson"] is not None
                and anchor_aggregate["mean_raw_margin_pearson"] is not None
                else None
            ),
            "thin_sheet_fraction_delta": (
                float(morphology["thin_sheet_voxel_fraction_le_2"])
                - float(anchor_morphology["thin_sheet_voxel_fraction_le_2"])
                if morphology["thin_sheet_voxel_fraction_le_2"] is not None
                and anchor_morphology["thin_sheet_voxel_fraction_le_2"] is not None
                else None
            ),
            "isolated_fraction_delta": (
                float(morphology["isolated_foreground_fraction"])
                - float(anchor_morphology["isolated_foreground_fraction"])
                if morphology["isolated_foreground_fraction"] is not None
                and anchor_morphology["isolated_foreground_fraction"] is not None
                else None
            ),
        }

    ordered = sorted(
        candidates,
        key=lambda item: (
            float(
                _mapping(item["teacher_aggregate"], "aggregate")["mean_foreground_dice"]
            ),
            float(
                _mapping(item["teacher_aggregate"], "aggregate")[
                    "mean_raw_margin_pearson"
                ]
                if _mapping(item["teacher_aggregate"], "aggregate")[
                    "mean_raw_margin_pearson"
                ]
                is not None
                else -1.0
            ),
        ),
        reverse=True,
    )
    candidate_rows = []
    for rank, candidate in enumerate(ordered, start=1):
        candidate["proxy_rank"] = rank
        metrics = _mapping(candidate["metrics"], "metrics")
        morphology = _mapping(metrics["morphology_proxies"], "morphology")
        aggregate = _mapping(candidate["teacher_aggregate"], "aggregate")
        candidate_rows.append(
            {
                "candidate_id": candidate["id"],
                "proxy_rank": rank,
                "foreground_voxels": metrics["foreground_voxels"],
                "foreground_fraction": metrics["foreground_fraction"],
                "minimum_axis_run_length_mean": morphology[
                    "minimum_axis_run_length_mean"
                ],
                "thin_sheet_voxel_fraction_le_2": morphology[
                    "thin_sheet_voxel_fraction_le_2"
                ],
                "isolated_foreground_fraction": morphology[
                    "isolated_foreground_fraction"
                ],
                "mean_teacher_iou": aggregate["mean_foreground_iou"],
                "mean_teacher_dice": aggregate["mean_foreground_dice"],
                "mean_raw_margin_pearson": aggregate["mean_raw_margin_pearson"],
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "platform_artifact": {
            "type": PLATFORM_ARTIFACT_TYPE,
            "title": "LayerPulse 断层候选 · Chengdu 固定井轨迹 128³ ROI",
            "primary_table": candidate_rows,
        },
        "scope": {
            "truth_status": "no_human_fault_labels; teachers_are_proxies_only",
            "execution": "cpu_only_saved_logits_no_model_forward",
            "source": "chengdu_main_3d.segy",
            "shape_tix": list(shape),
            "crop_start_tix": list(crop_start),
            "selection_policy": selection_policy,
            "input_patch_npy": str(anchor_input),
            "expected_input_sha256": expected_sha256,
            "observed_input_sha256": observed_sha256,
            "input_sha256_exact": True,
            "valid_mask_npy": str(valid_mask_path),
            "valid_voxels": int(valid_mask.sum()),
        },
        "contract": {
            "candidate_logits": "complete_background_inclusive_two_class_raw_logits",
            "candidate_selection": "direct_argmax_axis0_equivalent_to_argmax_dim1",
            "teacher_use": "offline_comparison_only_not_final_forward",
            "postprocessing": dict(_NO_POSTPROCESSING),
            "connected_components_measured_or_removed": False,
            "morphology_metrics": (
                "local_axis_run_and_six_face_neighbor_proxies_only; no component labeling"
            ),
        },
        "candidates": candidates,
        "teachers": teachers,
        "selection": {
            "proxy_leader_candidate_id": ordered[0]["id"],
            "ranking_rule": (
                "descending mean raw-teacher foreground Dice, then raw-margin Pearson"
            ),
            "promotion_claim": "proxy_leader_only_not_truth_validated",
            "do_not_promote_if_other_multitask_metrics_regress": True,
        },
    }
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            fieldnames = list(rows[0]) if rows else []
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_request(request_path: Path, output_directory: Path) -> dict[str, Any]:
    """Evaluate and atomically persist the JSON/CSV platform artifacts."""

    request_path = request_path.resolve(strict=True)
    report = evaluate_request(
        _json(request_path),
        request_base=request_path.parent,
        fixture=False,
    )
    output_directory = output_directory.resolve()
    report_path = output_directory / "layerpulse_fault_candidate_comparison.json"
    table_path = output_directory / "layerpulse_fault_candidate_comparison.csv"
    rows = _mapping(report["platform_artifact"], "platform artifact")["primary_table"]
    _atomic_csv(table_path, _sequence(rows, "primary table"))
    persisted = dict(report)
    persisted["artifacts"] = {
        "report_json": str(report_path),
        "comparison_csv": str(table_path),
    }
    _atomic_json(report_path, persisted)
    return persisted


def load_report(path: Path) -> Mapping[str, Any]:
    """Load a platform-readable report and reject an incompatible schema."""

    report = _json(path.resolve(strict=True))
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "pass"
        or _mapping(report.get("platform_artifact"), "platform artifact").get("type")
        != PLATFORM_ARTIFACT_TYPE
    ):
        raise FaultCandidateAcceptanceError("incompatible fault candidate report")
    return report


__all__ = [
    "DEFAULT_ANCHOR_INPUT",
    "DEFAULT_CROP_START_TIX",
    "DEFAULT_INPUT_SHA256",
    "DEFAULT_SELECTION_POLICY",
    "DEFAULT_SHAPE_TIX",
    "REPORT_SCHEMA",
    "REQUEST_SCHEMA",
    "FaultCandidateAcceptanceError",
    "describe_request",
    "evaluate_request",
    "load_report",
    "run_request",
]
