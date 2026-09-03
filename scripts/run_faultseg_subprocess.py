"""Run the bundled FaultSeg checkpoint in an isolated Python process."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_REPRESENTATIVE_SCOPE = "representative_grid_128"
_REPRESENTATIVE_GRID_CONTRACT_VERSION = (
    "well-seismic.faultseg-representative-grid.v2"
)
_REPRESENTATIVE_GRID_SHAPE_ZYX = (8, 4, 4)
_REPRESENTATIVE_BLOCK_SHAPE_ZYX = (128, 128, 128)
_REPRESENTATIVE_BLOCK_COUNT = 128
_REPRESENTATIVE_THRESHOLD = 0.518
_REPRESENTATIVE_OVERLAP = (0, 0, 0)
_CENTER_BLOCK_SCOPE = "center_block_1"
_FULL_VOLUME_SCOPE = "full_volume"
_FULL_VOLUME_PATCH_ZYX = (128, 128, 128)
_FULL_VOLUME_OVERLAP_ZYX = (64, 64, 64)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    return parser.parse_args()


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if hasattr(value, "item"):
        converted = value.item()
        if isinstance(converted, (int, float, str, bool)):
            return converted
    return str(value)


def _load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("FaultSeg subprocess request must be a JSON object")
    return payload


def _shape3(value: Any, *, field: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"FaultSeg {field} must be a three-integer sequence")
    if len(value) != 3:
        raise ValueError(f"FaultSeg {field} must contain exactly three axes")
    shape: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"FaultSeg {field} cannot contain boolean values")
        parsed = int(item)
        if parsed <= 0 or parsed != item:
            raise ValueError(f"FaultSeg {field} must contain positive integers")
        shape.append(parsed)
    return tuple(shape)  # type: ignore[return-value]


def _checkpoint_training_shape(metadata: Mapping[str, Any]) -> tuple[int, int, int] | None:
    args = metadata.get("args")
    if isinstance(args, Mapping):
        raw_shape = args.get("shape")
    else:
        raw_shape = getattr(args, "shape", None)
    if raw_shape is None:
        return None
    return _shape3(raw_shape, field="checkpoint args.shape")


def _validate_training_context(
    request: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    scope = str(request.get("scope") or "").strip().casefold()
    formal = scope in {
        "full_volume",
        "automatic_valid_roi",
        _CENTER_BLOCK_SCOPE,
        _REPRESENTATIVE_SCOPE,
    }
    checkpoint_shape = _checkpoint_training_shape(checkpoint_metadata)
    configured_raw = request.get("training_context_shape")
    configured_shape = (
        _shape3(configured_raw, field="configured training_context_shape")
        if configured_raw is not None
        else None
    )
    requested_patch = _shape3(request["patch_size"], field="requested patch_size")
    authority = str(
        request.get("training_context_authority") or "checkpoint_metadata"
    ).strip().casefold()
    if formal:
        if configured_shape is None:
            raise RuntimeError(
                "FaultSeg formal inference requires configured training_context_shape"
            )
        if authority == "checkpoint_metadata":
            if checkpoint_shape is None:
                raise RuntimeError(
                    "FaultSeg formal inference requires checkpoint metadata args.shape"
                )
            if configured_shape != checkpoint_shape:
                raise RuntimeError(
                    "FaultSeg configured training context drifted from checkpoint "
                    f"args.shape: configured={configured_shape}, checkpoint={checkpoint_shape}"
                )
            if requested_patch != checkpoint_shape:
                raise RuntimeError(
                    "FaultSeg formal primary patch drifted from checkpoint args.shape: "
                    f"patch={requested_patch}, checkpoint={checkpoint_shape}"
                )
        elif authority == "platform_declared_divisible_16":
            if str(request.get("checkpoint_loader") or "") != "torchscript":
                raise RuntimeError(
                    "platform-declared dynamic context is only allowed for TorchScript"
                )
            if any(value % 16 for value in configured_shape):
                raise RuntimeError(
                    "TorchScript platform context must be divisible by 16 on every axis"
                )
            if requested_patch != configured_shape:
                raise RuntimeError(
                    "formal primary patch differs from the configured TorchScript context: "
                    f"patch={requested_patch}, configured={configured_shape}"
                )
        else:
            raise RuntimeError(f"unsupported training context authority: {authority}")
    if authority == "checkpoint_metadata":
        context_validated = bool(
            checkpoint_shape is not None
            and configured_shape is not None
            and checkpoint_shape == configured_shape
            and (not formal or requested_patch == checkpoint_shape)
        )
        primary_patch_matches = bool(
            checkpoint_shape is not None and requested_patch == checkpoint_shape
        )
    else:
        context_validated = bool(
            configured_shape is not None
            and all(value % 16 == 0 for value in configured_shape)
            and (not formal or requested_patch == configured_shape)
        )
        primary_patch_matches = bool(
            configured_shape is not None and requested_patch == configured_shape
        )
    return {
        "checkpoint_training_shape": (
            list(checkpoint_shape) if checkpoint_shape is not None else None
        ),
        "configured_training_context_shape": (
            list(configured_shape) if configured_shape is not None else None
        ),
        "training_context_validated": context_validated,
        "training_context_policy": (
            (
                "formal_checkpoint_authoritative"
                if authority == "checkpoint_metadata"
                else "formal_platform_declared_torchscript_divisible_16"
            )
            if formal
            else "debug_deviation_allowed"
        ),
        "training_context_authority": authority,
        "primary_patch_matches_checkpoint": primary_patch_matches,
        "debug_context_deviation": bool(
            not formal
            and checkpoint_shape is not None
            and requested_patch != checkpoint_shape
        ),
    }


def _bounded_patch(
    requested: tuple[int, int, int],
    shape: tuple[int, int, int],
    *,
    multiple: int = 8,
) -> tuple[int, int, int]:
    bounded = tuple(
        (min(int(size), int(limit)) // multiple) * multiple
        for size, limit in zip(requested, shape, strict=True)
    )
    if any(value < multiple for value in bounded):
        raise ValueError(
            f"FaultSeg volume {shape} cannot contain a valid multiple-of-{multiple} patch"
        )
    return bounded


def _patch_candidates(
    request: dict[str, Any], shape: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    primary = tuple(int(value) for value in request["patch_size"])
    multiple = int(request.get("patch_multiple") or 8)
    if multiple <= 0:
        raise ValueError("patch_multiple must be positive")
    candidates = [_bounded_patch(primary, shape, multiple=multiple)]
    if bool(request.get("allow_patch_fallback", False)):
        for value in request.get("cuda_patch_fallbacks") or []:
            if isinstance(value, (list, tuple)):
                candidate = tuple(int(item) for item in value)
                if len(candidate) != 3:
                    raise ValueError("FaultSeg patch fallback must have three axes")
            else:
                candidate = (int(value),) * 3
            bounded = _bounded_patch(candidate, shape, multiple=multiple)
            if bounded not in candidates:
                candidates.append(bounded)
    return candidates


def _candidate_overlap(
    requested: tuple[int, int, int], patch: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        min(int(value), max(0, int(size) // 2))
        for value, size in zip(requested, patch, strict=True)
    )


def _is_cuda_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return "cuda" in text and "out of memory" in text


def _resolve_normalization_mode(request: Mapping[str, Any]) -> str:
    scope = str(request.get("scope") or "").strip().casefold()
    formal = scope in {
        "full_volume",
        "automatic_valid_roi",
        _CENTER_BLOCK_SCOPE,
        _REPRESENTATIVE_SCOPE,
    }
    default = "per_patch_zscore"
    mode = str(request.get("normalization_mode") or default).strip().casefold()
    aliases = {
        "patch_zscore": "per_patch_zscore",
        "patch_minmax": "per_patch_minmax",
        "shared_roi_zscore": "roi_shared_zscore",
        "formal_roi_zscore": "roi_shared_zscore",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"roi_shared_zscore", "per_patch_zscore", "per_patch_minmax"}:
        raise ValueError(f"unsupported FaultSeg normalization mode: {mode}")
    model_id = str(request.get("model_id") or "faultseg_3d").strip().casefold()
    expected_formal_mode = (
        "per_patch_minmax"
        if model_id == "faultnet_china_field"
        else "per_patch_zscore"
    )
    if formal and mode == "roi_shared_zscore":
        raise RuntimeError(
            "FaultSeg shared-ROI normalization is an uncalibrated experiment "
            "and is not permitted for formal inference"
        )
    if formal and mode != expected_formal_mode:
        raise RuntimeError(
            f"{model_id} formal inference requires {expected_formal_mode}; "
            f"received {mode}"
        )
    if mode == "roi_shared_zscore" and not bool(
        request.get("allow_uncalibrated_normalization_experiment", False)
    ):
        raise RuntimeError(
            "FaultSeg shared-ROI normalization is uncalibrated and requires an "
            "explicit offline-experiment opt-in"
        )
    return mode


def _load_inference_model(
    request: Mapping[str, Any], checkpoint: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    loader = str(request.get("checkpoint_loader") or "state_dict").strip().casefold()
    if loader == "state_dict":
        from src.checkpoint import load_model

        return load_model(checkpoint, device)
    if loader == "torchscript":
        import torch

        # libtorch's Windows path loader can reject an otherwise valid
        # TorchScript file when any parent directory contains Chinese
        # characters.  Passing an already-open binary stream keeps the path
        # handling in Python and is also compatible with ASCII-only paths.
        with checkpoint.open("rb") as stream:
            model = torch.jit.load(stream, map_location=device)
        model.eval()
        return model, {
            "checkpoint_loader": "torchscript",
            "model_id": str(request.get("model_id") or ""),
        }
    raise ValueError(f"unsupported checkpoint loader: {loader}")


def _streaming_zscore_statistics(
    volume: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    block_z: int = 16,
) -> dict[str, Any]:
    """Reduce experimental shared statistics over the declared inference ROI.

    The checkpoint was trained on independently normalized 128^3 items, so
    these statistics are not training-equivalent and are never permitted by
    the formal prediction route.
    """

    count = 0
    mean = 0.0
    m2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    block = max(1, int(block_z))
    for z_start in range(0, volume.shape[0], block):
        z_stop = min(volume.shape[0], z_start + block)
        slab = np.asarray(volume[z_start:z_stop], dtype=np.float32)
        valid = np.isfinite(slab)
        if valid_mask is not None:
            if valid_mask.ndim == 2:
                valid &= valid_mask[None, :, :]
            else:
                valid &= valid_mask[z_start:z_stop]
        values = np.asarray(slab[valid], dtype=np.float64)
        if not values.size:
            continue
        block_count = int(values.size)
        block_mean = float(values.mean())
        centered = values - block_mean
        block_m2 = float(np.dot(centered, centered))
        combined_count = count + block_count
        delta = block_mean - mean
        mean += delta * block_count / combined_count
        m2 += block_m2 + delta * delta * count * block_count / combined_count
        count = combined_count
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    if count < 1:
        raise ValueError("FaultSeg ROI contains no finite valid seismic samples")
    std = float(np.sqrt(max(m2 / count, 0.0)))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("FaultSeg ROI has zero or invalid seismic standard deviation")
    total_voxels = int(np.prod(volume.shape, dtype=np.int64))
    return {
        "algorithm": "streaming_population_mean_std_v1",
        "scope": "declared_inference_roi",
        "finite_valid_voxel_count": count,
        "excluded_voxel_count": total_voxels - count,
        "mean": mean,
        "std": std,
        "minimum": minimum,
        "maximum": maximum,
        "shared_across_patches": True,
        "invalid_fill_normalized": 0.0,
        "training_equivalent": False,
        "calibration_status": "uncalibrated_offline_experiment",
    }


def _finalize_outputs(
    probability: np.ndarray,
    *,
    probability_path: Path,
    mask_path: Path,
    threshold: float,
    valid_mask: np.ndarray | None,
    block_z: int = 16,
) -> dict[str, float | int]:
    """Apply validity, write a uint8 mask and reduce statistics by Z slabs."""

    if not probability_path.is_file():
        np.save(probability_path, np.asarray(probability, dtype=np.float32))
    writable_probability = np.load(
        probability_path, mmap_mode="r+", allow_pickle=False
    )
    mask = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=writable_probability.shape,
    )
    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    count = 0
    positive = 0
    block = max(1, int(block_z))
    for z_start in range(0, writable_probability.shape[0], block):
        z_stop = min(writable_probability.shape[0], z_start + block)
        probability_slab = writable_probability[z_start:z_stop]
        if valid_mask is not None:
            if valid_mask.ndim == 2:
                probability_slab[:, ~valid_mask] = 0.0
            else:
                probability_slab[~valid_mask[z_start:z_stop]] = 0.0
        if not np.isfinite(probability_slab).all():
            raise ValueError("FaultSeg probability contains non-finite values")
        mask_slab = probability_slab >= threshold
        mask[z_start:z_stop] = mask_slab.astype(np.uint8, copy=False)
        minimum = min(minimum, float(probability_slab.min()))
        maximum = max(maximum, float(probability_slab.max()))
        total += float(probability_slab.sum(dtype=np.float64))
        count += int(probability_slab.size)
        positive += int(np.count_nonzero(mask_slab))
    writable_probability.flush()
    mask.flush()
    if isinstance(mask, np.memmap):
        mask._mmap.close()
    if isinstance(writable_probability, np.memmap):
        writable_probability._mmap.close()
    return {
        "min": minimum,
        "max": maximum,
        "mean": total / max(1, count),
        "positive_voxel_count": positive,
        "positive_fraction": positive / max(1, count),
    }


def _representative_grid_blocks(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the fixed public 8x4x4 sampling request fail-closed."""

    if str(request.get("scope") or "").strip().casefold() != _REPRESENTATIVE_SCOPE:
        raise ValueError("FaultSeg representative request uses the wrong scope")
    if (
        str(request.get("representative_grid_contract_version") or "")
        != _REPRESENTATIVE_GRID_CONTRACT_VERSION
    ):
        raise ValueError("FaultSeg representative grid contract version drifted")
    if _shape3(request.get("patch_size"), field="patch_size") != (
        _REPRESENTATIVE_BLOCK_SHAPE_ZYX
    ):
        raise ValueError("FaultSeg representative patch_size is fixed to 128x128x128")
    overlap = tuple(int(value) for value in request.get("overlap") or ())
    if overlap != _REPRESENTATIVE_OVERLAP:
        raise ValueError("FaultSeg representative overlap is fixed to zero")
    if float(request.get("threshold")) != _REPRESENTATIVE_THRESHOLD:
        raise ValueError("FaultSeg representative threshold is fixed to 0.518")
    if bool(request.get("weighted_blending", False)):
        raise ValueError("FaultSeg representative blocks prohibit weighted blending")
    if bool(request.get("allow_patch_fallback", False)):
        raise ValueError("FaultSeg representative blocks prohibit patch fallback")
    if _resolve_normalization_mode(request) != "per_patch_zscore":
        raise ValueError("FaultSeg representative normalization is fixed per patch")
    source_shape = _shape3(
        request.get("source_shape_zyx"), field="source_shape_zyx"
    )
    if _shape3(
        request.get("representative_grid_shape_zyx"),
        field="representative_grid_shape_zyx",
    ) != _REPRESENTATIVE_GRID_SHAPE_ZYX:
        raise ValueError("FaultSeg representative grid shape is fixed to 8x4x4")
    if str(request.get("grid_order") or "") != "Z_then_INLINE_then_CROSSLINE":
        raise ValueError("FaultSeg representative grid order is fixed")
    raw_blocks = request.get("representative_blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != _REPRESENTATIVE_BLOCK_COUNT:
        raise ValueError("FaultSeg representative grid requires exactly 128 blocks")
    blocks: list[dict[str, Any]] = []
    starts: set[tuple[int, int, int]] = set()
    output_paths: set[str] = set()
    ordinal = 0
    for z_index in range(_REPRESENTATIVE_GRID_SHAPE_ZYX[0]):
        for inline_index in range(_REPRESENTATIVE_GRID_SHAPE_ZYX[1]):
            for crossline_index in range(_REPRESENTATIVE_GRID_SHAPE_ZYX[2]):
                raw = raw_blocks[ordinal]
                if not isinstance(raw, Mapping):
                    raise TypeError("FaultSeg representative block must be an object")
                block = dict(raw)
                forbidden_block_parameters = {
                    "patch_size",
                    "overlap",
                    "threshold",
                    "normalization",
                    "weighted_blending",
                }.intersection(block)
                if forbidden_block_parameters:
                    raise ValueError(
                        "FaultSeg representative blocks prohibit per-block parameters"
                    )
                expected_grid_index = [z_index, inline_index, crossline_index]
                expected_id = f"z{z_index:02d}_i{inline_index:02d}_x{crossline_index:02d}"
                if int(block.get("ordinal", -1)) != ordinal:
                    raise ValueError("FaultSeg representative block ordinal drifted")
                if block.get("grid_index_zyx") != expected_grid_index:
                    raise ValueError("FaultSeg representative grid order drifted")
                if str(block.get("block_id") or "") != expected_id:
                    raise ValueError("FaultSeg representative block id drifted")
                if _shape3(block.get("shape_zyx"), field="block shape") != (
                    _REPRESENTATIVE_BLOCK_SHAPE_ZYX
                ):
                    raise ValueError("FaultSeg representative block shape drifted")
                start_raw = block.get("source_start_zyx")
                if not isinstance(start_raw, Sequence) or len(start_raw) != 3:
                    raise ValueError("FaultSeg representative source start is invalid")
                start = tuple(int(value) for value in start_raw)
                if any(value < 0 for value in start) or start in starts:
                    raise ValueError("FaultSeg representative source starts are not unique")
                end_exclusive = tuple(
                    start_value + block_value
                    for start_value, block_value in zip(
                        start, _REPRESENTATIVE_BLOCK_SHAPE_ZYX, strict=True
                    )
                )
                if any(
                    end > available
                    for end, available in zip(
                        end_exclusive, source_shape, strict=True
                    )
                ):
                    raise ValueError(
                        "FaultSeg representative block escapes the source volume"
                    )
                if block.get("source_end_zyx_exclusive") != list(end_exclusive):
                    raise ValueError(
                        "FaultSeg representative source end-exclusive receipt drifted"
                    )
                if block.get("source_end_zyx_inclusive") != [
                    value - 1 for value in end_exclusive
                ]:
                    raise ValueError(
                        "FaultSeg representative source end-inclusive receipt drifted"
                    )
                if not isinstance(block.get("axis_coordinate_ranges"), Mapping):
                    raise ValueError(
                        "FaultSeg representative block omitted axis coordinate ranges"
                    )
                starts.add(start)
                for field in (
                    "input_volume_npy",
                    "valid_mask_npy",
                    "probability_npy",
                    "mask_npy",
                    "metadata_json",
                ):
                    value = str(block.get(field) or "").strip()
                    if not value or value in output_paths:
                        raise ValueError(
                            f"FaultSeg representative block path {field} is missing or reused"
                        )
                    output_paths.add(value)
                blocks.append(block)
                ordinal += 1
    return blocks


def _run_representative_grid(request: dict[str, Any]) -> dict[str, Any]:
    """Run 128 independent checkpoint-native blocks without overlap or stitching."""

    faultseg_root = Path(request["faultseg_root"]).expanduser().resolve()
    checkpoint = Path(request["checkpoint"]).expanduser().resolve()
    result_path = Path(request["result_json"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if str(faultseg_root) not in sys.path:
        sys.path.insert(0, str(faultseg_root))
    from src.checkpoint import load_model
    from src.inference import choose_device, predict_volume

    blocks = _representative_grid_blocks(request)
    device = choose_device(str(request.get("device", "auto")))
    model, checkpoint_metadata = load_model(checkpoint, device)
    training_context_receipt = _validate_training_context(
        request, checkpoint_metadata
    )
    if not training_context_receipt["training_context_validated"]:
        raise RuntimeError("FaultSeg representative training context is not validated")
    normalization_statistics = {
        "algorithm": "per_patch_population_mean_std_v1",
        "scope": "individual_inference_patch",
        "shared_across_patches": False,
        "invalid_fill_normalized": 0.0,
        "training_equivalent": True,
        "calibration_status": "checkpoint_training_preprocessing",
    }
    block_receipts: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        input_path = Path(str(block["input_volume_npy"])).expanduser().resolve()
        valid_mask_path = Path(str(block["valid_mask_npy"])).expanduser().resolve()
        probability_path = Path(str(block["probability_npy"])).expanduser().resolve()
        mask_path = Path(str(block["mask_npy"])).expanduser().resolve()
        metadata_path = Path(str(block["metadata_json"])).expanduser().resolve()
        if not input_path.is_file() or not valid_mask_path.is_file():
            raise FileNotFoundError(
                f"FaultSeg representative staged input is missing for {block['block_id']}"
            )
        volume = np.load(input_path, mmap_mode="r", allow_pickle=False)
        valid_mask = np.load(valid_mask_path, allow_pickle=False).astype(
            bool, copy=False
        )
        if tuple(volume.shape) != _REPRESENTATIVE_BLOCK_SHAPE_ZYX:
            raise ValueError(
                f"FaultSeg representative input {block['block_id']} is not 128x128x128"
            )
        if volume.dtype != np.float32:
            raise TypeError("FaultSeg representative staged inputs must be float32")
        if valid_mask.shape != _REPRESENTATIVE_BLOCK_SHAPE_ZYX[1:]:
            raise ValueError("FaultSeg representative valid mask must be Inline x Xline")
        progress_calls: list[tuple[int, int, tuple[int, int, int]]] = []

        def progress(index: int, total: int, origin: tuple[int, int, int]) -> None:
            progress_calls.append((int(index), int(total), tuple(origin)))
            print(
                json.dumps(
                    {
                        "event": "faultseg_progress",
                        "block_index": block_index,
                        "block_count": _REPRESENTATIVE_BLOCK_COUNT,
                        "block_id": block["block_id"],
                        "index": index,
                        "total": total,
                        "origin": list(origin),
                    }
                ),
                flush=True,
            )

        probability_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        probability = predict_volume(
            model,
            volume,
            device,
            _REPRESENTATIVE_BLOCK_SHAPE_ZYX,
            _REPRESENTATIVE_OVERLAP,
            progress=progress,
            normalize_patches=True,
            weighted_blending=False,
            invalid_value=np.nan,
            probability_path=probability_path,
            scratch_directory=probability_path.parent,
        )
        if progress_calls != [(1, 1, (0, 0, 0))]:
            raise RuntimeError(
                f"FaultSeg representative block {block['block_id']} did not execute "
                f"exactly one checkpoint forward: {progress_calls}"
            )
        if tuple(probability.shape) != _REPRESENTATIVE_BLOCK_SHAPE_ZYX:
            raise ValueError("FaultSeg representative probability shape drifted")
        statistics = _finalize_outputs(
            probability,
            probability_path=probability_path,
            mask_path=mask_path,
            threshold=_REPRESENTATIVE_THRESHOLD,
            valid_mask=valid_mask,
        )
        if isinstance(probability, np.memmap):
            probability._mmap.close()
        if isinstance(volume, np.memmap):
            volume._mmap.close()
        public_block = {
            key: value
            for key, value in block.items()
            if key not in {"input_volume_npy", "valid_mask_npy", "probability_npy", "mask_npy", "metadata_json"}
        }
        valid_voxel_count = int(
            int(block["valid_trace_count"])
            * _REPRESENTATIVE_BLOCK_SHAPE_ZYX[0]
        )
        fault_voxel_count = int(statistics["positive_voxel_count"])
        fault_fraction_of_valid = float(
            fault_voxel_count / max(1, valid_voxel_count)
        )
        public_block.update(
            {
                "valid_trace_ratio": float(block["valid_trace_fraction"]),
                "valid_voxel_count": valid_voxel_count,
                "fault_voxel_count": fault_voxel_count,
                "fault_fraction": fault_fraction_of_valid,
                "fault_fraction_of_valid_voxels": fault_fraction_of_valid,
                "fault_fraction_all_voxels": float(statistics["positive_fraction"]),
                "fault_fraction_denominator": "valid_trace_count_times_block_z",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(request.get("checkpoint_sha256") or ""),
                "checkpoint_epoch": _json_scalar(checkpoint_metadata.get("epoch")),
                "threshold": _REPRESENTATIVE_THRESHOLD,
                "normalization": "per_patch_zscore",
                "normalization_statistics": normalization_statistics,
                "forward_calls": 1,
                "outputs": {
                    "probability_npy": str(probability_path),
                    "mask_npy": str(mask_path),
                    "metadata_json": str(metadata_path),
                },
                "statistics": statistics,
            }
        )
        metadata_path.write_text(
            json.dumps(public_block, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        block_receipts.append(public_block)
    result = {
        "schema_version": "well-seismic.faultseg-subprocess.v1",
        "device": str(device),
        "checkpoint_epoch": _json_scalar(checkpoint_metadata.get("epoch")),
        "patch_count": _REPRESENTATIVE_BLOCK_COUNT,
        "forward_calls": _REPRESENTATIVE_BLOCK_COUNT,
        "selected_patch_size": list(_REPRESENTATIVE_BLOCK_SHAPE_ZYX),
        "selected_overlap": list(_REPRESENTATIVE_OVERLAP),
        "weighted_blending": False,
        "patch_attempts": [
            {
                "patch_size": list(_REPRESENTATIVE_BLOCK_SHAPE_ZYX),
                "overlap": list(_REPRESENTATIVE_OVERLAP),
                "status": "selected_fixed_policy",
            }
        ],
        "inference_context_degraded": False,
        "degradation_reasons": [],
        "normalization": "per_patch_zscore",
        "normalization_statistics": normalization_statistics,
        "threshold": _REPRESENTATIVE_THRESHOLD,
        "threshold_source": request.get("threshold_source"),
        "scope": _REPRESENTATIVE_SCOPE,
        **training_context_receipt,
        "representative_grid": {
            "contract_version": _REPRESENTATIVE_GRID_CONTRACT_VERSION,
            "scope": "representative_sampling",
            "is_full_volume": False,
            "grid_shape_zyx": list(_REPRESENTATIVE_GRID_SHAPE_ZYX),
            "block_shape_zyx": list(_REPRESENTATIVE_BLOCK_SHAPE_ZYX),
            "grid_order": "Z_then_INLINE_then_CROSSLINE",
            "inference_overlap_zyx": list(_REPRESENTATIVE_OVERLAP),
            "inter_block_stitching": False,
            "forward_calls_total": _REPRESENTATIVE_BLOCK_COUNT,
            "blocks": block_receipts,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run(request: dict[str, Any]) -> dict[str, Any]:
    scope = str(request.get("scope") or "").strip().casefold()
    if scope == _REPRESENTATIVE_SCOPE:
        return _run_representative_grid(request)
    if scope == _CENTER_BLOCK_SCOPE:
        requested_patch = _shape3(request["patch_size"], field="requested patch_size")
        requested_overlap = tuple(int(value) for value in request["overlap"])
        if requested_patch != _FULL_VOLUME_PATCH_ZYX:
            raise ValueError("FaultSeg center_block_1 patch_size must be 128x128x128")
        if requested_overlap != (0, 0, 0):
            raise ValueError("FaultSeg center_block_1 overlap must be 0x0x0")
        if bool(request.get("weighted_blending", False)):
            raise ValueError("FaultSeg center_block_1 must execute one unstitched block")
        if bool(request.get("allow_patch_fallback", False)) or (
            request.get("cuda_patch_fallbacks") or []
        ):
            raise ValueError(
                "FaultSeg center_block_1 prohibits CUDA patch fallback; the "
                "128x128x128 checkpoint context must be preserved"
            )
    if scope == _FULL_VOLUME_SCOPE:
        requested_patch = _shape3(request["patch_size"], field="requested patch_size")
        requested_overlap = _shape3(request["overlap"], field="requested overlap")
        if requested_patch != _FULL_VOLUME_PATCH_ZYX:
            raise ValueError("FaultSeg full_volume patch_size must be 128x128x128")
        if requested_overlap != _FULL_VOLUME_OVERLAP_ZYX:
            raise ValueError("FaultSeg full_volume overlap must be 64x64x64")
        if not bool(request.get("weighted_blending", False)):
            raise ValueError("FaultSeg full_volume requires weighted_blending=true")
        if bool(request.get("allow_patch_fallback", False)) or (
            request.get("cuda_patch_fallbacks") or []
        ):
            raise ValueError(
                "FaultSeg full_volume prohibits CUDA patch fallback; the 128x128x128 "
                "checkpoint context must be preserved"
            )
    faultseg_root = Path(request["faultseg_root"]).expanduser().resolve()
    checkpoint = Path(request["checkpoint"]).expanduser().resolve()
    input_path = Path(request["input_volume_npy"]).expanduser().resolve()
    valid_mask_value = request.get("valid_mask_npy")
    probability_path = Path(request["probability_npy"]).expanduser().resolve()
    mask_path = Path(request["mask_npy"]).expanduser().resolve()
    result_path = Path(request["result_json"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if str(faultseg_root) not in sys.path:
        sys.path.insert(0, str(faultseg_root))

    from src.inference import choose_device, predict_volume

    volume = np.load(input_path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"FaultSeg input must be 3D, got {volume.shape}")
    if volume.dtype != np.float32:
        raise TypeError(
            f"FaultSeg staged input must be float32 without conversion, got {volume.dtype}"
        )
    valid_mask = None
    if valid_mask_value:
        valid_mask = np.load(
            Path(valid_mask_value).expanduser().resolve(), allow_pickle=False
        ).astype(bool, copy=False)
        if valid_mask.shape not in {volume.shape, volume.shape[1:]}:
            raise ValueError(
                f"valid mask shape {valid_mask.shape} differs from volume {volume.shape}"
            )

    normalization_mode = _resolve_normalization_mode(request)
    normalization_statistics = (
        _streaming_zscore_statistics(volume, valid_mask)
        if normalization_mode == "roi_shared_zscore"
        else {
            "algorithm": "per_patch_population_mean_std_v1",
            "scope": "individual_inference_patch",
            "shared_across_patches": False,
            "invalid_fill_normalized": 0.0,
            "training_equivalent": True,
            "calibration_status": "checkpoint_training_preprocessing",
        }
    )
    if normalization_mode == "per_patch_minmax":
        normalization_statistics = {
            "algorithm": "per_patch_finite_minmax_v1",
            "scope": "individual_inference_patch",
            "output_range": [0.0, 1.0],
            "shared_across_patches": False,
            "invalid_fill_normalized": 0.5,
            "training_equivalent": True,
            "calibration_status": "official_prediction_preprocessing",
        }

    device = choose_device(str(request.get("device", "auto")))
    model, checkpoint_metadata = _load_inference_model(request, checkpoint, device)
    training_context_receipt = _validate_training_context(
        request, checkpoint_metadata
    )
    output_activation = str(
        request.get("output_activation") or "sigmoid"
    ).strip().casefold()
    progress_state = {"patches": 0, "total": 0}

    def progress(index: int, total: int, origin: tuple[int, int, int]) -> None:
        progress_state["patches"] = index
        progress_state["total"] = total
        print(
            json.dumps(
                {
                    "event": "faultseg_progress",
                    "index": index,
                    "total": total,
                    "origin": list(origin),
                }
            ),
            flush=True,
        )

    probability_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    requested_overlap = tuple(int(value) for value in request["overlap"])
    attempts: list[dict[str, Any]] = []
    selected_patch: tuple[int, int, int] | None = None
    selected_overlap: tuple[int, int, int] | None = None
    probability: np.ndarray | None = None
    candidates = _patch_candidates(request, tuple(int(value) for value in volume.shape))
    for candidate_index, candidate in enumerate(candidates):
        candidate_overlap = _candidate_overlap(requested_overlap, candidate)
        progress_state.update({"patches": 0, "total": 0})
        try:
            probability = predict_volume(
                model,
                volume,
                device,
                candidate,
                candidate_overlap,
                amp_enabled=(
                    str(getattr(device, "type", device)).strip().casefold()
                    == "cuda"
                ),
                progress=progress,
                normalize_patches=True,
                weighted_blending=bool(request.get("weighted_blending", True)),
                invalid_value=np.nan,
                probability_path=probability_path,
                scratch_directory=probability_path.parent,
                normalization_mean=(
                    float(normalization_statistics["mean"])
                    if normalization_mode == "roi_shared_zscore"
                    else None
                ),
                normalization_std=(
                    float(normalization_statistics["std"])
                    if normalization_mode == "roi_shared_zscore"
                    else None
                ),
                patch_normalization=(
                    "minmax"
                    if normalization_mode == "per_patch_minmax"
                    else "zscore"
                ),
                output_activation=output_activation,
            )
        except RuntimeError as exc:
            is_oom = _is_cuda_out_of_memory(exc)
            error_text = f"{type(exc).__name__}: {exc}"
            attempts.append(
                {
                    "patch_size": list(candidate),
                    "overlap": list(candidate_overlap),
                    "status": "cuda_out_of_memory" if is_oom else "failed",
                    "error": error_text,
                }
            )
            if not is_oom or candidate_index == len(candidates) - 1:
                raise
            # Drop failed-forward frame locals before asking CUDA to release
            # its cache.  Otherwise the active exception traceback can retain
            # the input/output tensors and make every smaller retry fail too.
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        selected_patch = candidate
        selected_overlap = candidate_overlap
        attempts.append(
            {
                "patch_size": list(candidate),
                "overlap": list(candidate_overlap),
                "status": "selected",
            }
        )
        break
    if probability is None or selected_patch is None or selected_overlap is None:
        raise RuntimeError("FaultSeg did not select a runnable inference patch")
    if scope == _CENTER_BLOCK_SCOPE and (
        tuple(int(value) for value in volume.shape) != _FULL_VOLUME_PATCH_ZYX
        or tuple(int(value) for value in probability.shape) != _FULL_VOLUME_PATCH_ZYX
        or selected_patch != _FULL_VOLUME_PATCH_ZYX
        or selected_overlap != (0, 0, 0)
        or int(progress_state["patches"]) != 1
        or int(progress_state["total"]) != 1
    ):
        raise ValueError(
            "FaultSeg center_block_1 must complete exactly one 128x128x128 forward"
        )
    statistics = _finalize_outputs(
        probability,
        probability_path=probability_path,
        mask_path=mask_path,
        threshold=float(request["threshold"]),
        valid_mask=valid_mask,
    )
    primary_patch = tuple(int(value) for value in request["patch_size"])
    degraded = selected_patch != primary_patch
    degradation_reasons = []
    if degraded:
        degradation_reasons.append(
            "cuda_out_of_memory_patch_fallback"
            if any(item["status"] == "cuda_out_of_memory" for item in attempts)
            else "source_axis_smaller_than_requested_patch"
        )
    result = {
        "schema_version": "well-seismic.faultseg-subprocess.v1",
        "model_id": str(request.get("model_id") or "faultseg_3d"),
        "device": str(device),
        "checkpoint_loader": str(
            request.get("checkpoint_loader") or "state_dict"
        ),
        "output_activation": output_activation,
        "checkpoint_epoch": _json_scalar(checkpoint_metadata.get("epoch")),
        "patch_count": int(progress_state["patches"]),
        "shape_zyx": list(probability.shape),
        "selected_patch_size": list(selected_patch),
        "selected_overlap": list(selected_overlap),
        "weighted_blending": bool(request.get("weighted_blending", True)),
        "patch_attempts": attempts,
        "inference_context_degraded": degraded,
        "degradation_reasons": degradation_reasons,
        "normalization": normalization_mode,
        "normalization_statistics": normalization_statistics,
        "threshold": float(request["threshold"]),
        "threshold_source": request.get("threshold_source"),
        "scope": request.get("scope"),
        "stitching": scope == _FULL_VOLUME_SCOPE,
        "full_volume_reconstructed": scope == _FULL_VOLUME_SCOPE,
        **training_context_receipt,
        "statistics": statistics,
        "probability_npy": str(probability_path),
        "mask_npy": str(mask_path),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    arguments = _arguments()
    run(_load_request(arguments.request_json.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
