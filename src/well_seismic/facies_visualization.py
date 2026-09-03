"""Bounded CIGVis payloads for the dense F3 facies transfer runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .faultseg_visualization import (
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
    _encode_array,
    _metadata_path,
    _result_document,
    _shape3,
)

F3_FACIES_MODEL_ID = "wellfuse_facies_3d_f3_fast"
_INPUT_AXES = ("TWT", "INLINE", "XLINE")
_CLASS_CODES = tuple(range(6))


def _unit_interval_volume(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"F3 facies {label} contains non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"F3 facies {label} must stay within [0, 1]")
    return np.rint(values * 255.0).astype(np.uint8)


def _volume_spec(
    array: np.ndarray,
    *,
    source: Path,
    value_range: list[float] | list[int],
    **metadata: Any,
) -> dict[str, Any]:
    return {
        **_encode_array(array, "base64-uint8"),
        "valueRange": value_range,
        "source": str(source),
        **metadata,
    }


def build_f3_facies_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
) -> dict[str, Any]:
    """Build an aligned seismic background and bounded facies layers.

    Runtime arrays are stored as ``[INLINE, XLINE, TWT]`` (probability adds a
    leading class axis).  This adapter samples them without materialising the
    complete arrays and returns the platform's ``[Z, INLINE, CROSSLINE]``
    contract.
    """

    result, metadata_base = _result_document(result_or_metadata)
    if result.get("model_id") != F3_FACIES_MODEL_ID:
        raise ValueError(
            f"visualization result must use model_id={F3_FACIES_MODEL_ID}"
        )
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    facies_metadata = result.get("facies")
    inference_metadata = result.get("inference")
    if not all(
        isinstance(value, Mapping)
        for value in (input_metadata, outputs, facies_metadata, inference_metadata)
    ):
        raise ValueError("F3 facies result is missing input/output/facies metadata")
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != _INPUT_AXES:
        raise ValueError(f"F3 facies input axes must be {_INPUT_AXES}, got {axes}")

    crop_start = _shape3(
        input_metadata.get("crop_start_zyx", ()),
        "crop_start_zyx",
        allow_zero=True,
    )
    crop_size = _shape3(input_metadata.get("crop_size_zyx", ()), "crop_size_zyx")
    source_shape = _shape3(
        input_metadata.get("source_shape_zyx", ()), "source_shape_zyx"
    )
    reported_shape = _shape3(
        facies_metadata.get("shape_t_inline_xline", ()),
        "facies.shape_t_inline_xline",
    )
    if reported_shape != crop_size:
        raise ValueError("F3 facies metadata shape does not match its declared ROI")
    if any(
        start + size > available
        for start, size, available in zip(
            crop_start, crop_size, source_shape, strict=True
        )
    ):
        raise ValueError("F3 facies ROI exceeds the source SEG-Y geometry")

    source_value = input_metadata.get("source")
    source = _metadata_path(
        source_value,
        base=metadata_base,
        label="input.source",
        suffix=Path(str(source_value or "")).suffix.lower(),
    )
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError("F3 facies input.source must be SEG-Y")
    artifact_paths = {
        name: _metadata_path(
            outputs.get(name),
            base=metadata_base,
            label=f"outputs.{name}",
            suffix=".npy",
        )
        for name in ("class_code_npy", "valid_trace_mask_npy")
    }
    class_code = np.load(
        artifact_paths["class_code_npy"], mmap_mode="r", allow_pickle=False
    )
    valid_trace_mask = np.load(
        artifact_paths["valid_trace_mask_npy"], mmap_mode="r", allow_pickle=False
    )
    expected_ics = (crop_size[1], crop_size[2], crop_size[0])
    if class_code.shape != expected_ics:
        raise ValueError("F3 facies class_code shape does not match the declared ROI")
    if valid_trace_mask.shape != expected_ics[:2]:
        raise ValueError("F3 facies valid trace mask shape is invalid")

    cache_options = dict(segy_options or {})
    resolved_headers = input_metadata.get("resolved_header_bytes")
    if isinstance(resolved_headers, Mapping):
        if resolved_headers.get("iline") is not None:
            cache_options.setdefault("inline_byte", int(resolved_headers["iline"]))
        if resolved_headers.get("xline") is not None:
            cache_options.setdefault(
                "crossline_byte", int(resolved_headers["xline"])
            )
    background, cache_hit = cache.get_crop(
        source,
        crop_start_zyx=crop_start,
        crop_size_zyx=crop_size,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=cache_options,
        expected_source_shape_zyx=source_shape,
    )
    z_indices, inline_indices, crossline_indices = (
        indices - start
        for indices, start in zip(
            background.sample_indices_zyx, crop_start, strict=True
        )
    )
    selection_ics = np.ix_(inline_indices, crossline_indices, z_indices)
    sampled_valid = np.asarray(
        valid_trace_mask[np.ix_(inline_indices, crossline_indices)], dtype=bool
    )
    valid_zyx = np.broadcast_to(
        sampled_valid[None, :, :], background.cube_int8.shape
    )

    sampled_class = np.asarray(class_code[selection_ics], dtype=np.int16).transpose(
        2, 0, 1
    )
    if not np.all(np.isin(sampled_class[valid_zyx], _CLASS_CODES)):
        raise ValueError("F3 facies class_code contains values outside 0..5")
    class_display = np.zeros(sampled_class.shape, dtype=np.uint8)
    class_display[valid_zyx] = np.rint(
        32.0 + sampled_class[valid_zyx].astype(np.float32) * (223.0 / 5.0)
    ).astype(np.uint8)
    class_spec = _volume_spec(
        class_display,
        source=artifact_paths["class_code_npy"],
        value_range=[0, 5],
        classCodes=list(_CLASS_CODES),
        displayCodeRange=[32, 255],
        invalidDisplayCode=0,
    )

    payload = background.as_payload(cache_hit=cache_hit)
    payload["contractVersion"] = "wellfuse-f3-facies-cigvis-v2"
    scientific_status = str(
        result.get("scientific_status", "experimental_transfer_candidate")
    )
    status_label = (
        "数据集内验证结果"
        if scientific_status == "validated_within_dataset"
        else "迁移候选"
    )
    payload["name"] = (
        f"{source.name} · 六类地震相数据集内结果"
        if scientific_status == "validated_within_dataset"
        else f"{source.name} · 六类地震相迁移候选"
    )
    payload["facies3d"] = {
        "modelId": F3_FACIES_MODEL_ID,
        "scientificStatus": scientific_status,
        "validatedScope": str(result.get("validated_scope", "F3_dense_benchmark")),
        "inferenceMode": str(inference_metadata.get("mode", "roi")),
        "classCode": class_spec,
        "display": {
            "preferredLayer": "class_code",
            "classCmap": "Set3",
            "alpha": 0.68,
        },
        "publicResult": "deterministic_class_volume_only",
    }
    payload["overlays"] = [
        {
            "id": "f3_facies_class_code",
            "name": f"六类地震相（{status_label}）",
            "kind": "class_code",
            "volume": class_spec,
            "clim": [32.0 / 255.0, 1.0],
            "cmap": "Set3",
            "alpha": 0.68,
            "excpt": "min",
        },
    ]
    payload["preview"]["cacheStats"] = cache.stats
    return payload


__all__ = ["F3_FACIES_MODEL_ID", "build_f3_facies_visualization_payload"]
