"""CIGVis payloads for seismic stratigraphic surface segmentation.

The upstream model writes ``mask.npy`` and ``confidence.npy`` in
``[INLINE, CROSSLINE, SAMPLE]`` order.  The platform and its lightweight
visualization cache use ``[Z, INLINE, CROSSLINE]`` instead.  This module keeps
the model outputs memory-mapped, samples only the indices requested by
``SegySliceCache``, and performs the axis conversion on that bounded sample.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .faultseg_visualization import (
    AXES_ZYX,
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
)


AXES_ICS = ("INLINE", "CROSSLINE", "SAMPLE")
SURFACE_SEG_MODEL_ID = "seismic_surface_seg"


def _shape3(value: Sequence[Any], name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three values")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integers, not booleans")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain integers") from exc
        if integer != item or integer <= 0:
            raise ValueError(f"{name} must contain positive integers")
        result.append(integer)
    return tuple(result)  # type: ignore[return-value]


def _result_document(
    result_or_metadata: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(result_or_metadata, Mapping):
        result: Mapping[str, Any] = result_or_metadata
        if "prediction" in result and "model_id" not in result:
            nested = result.get("prediction")
            if not isinstance(nested, Mapping):
                raise ValueError("prediction wrapper must contain a mapping")
            result = nested
        return result, None
    metadata_path = Path(result_or_metadata).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SurfaceSeg metadata does not exist: {metadata_path}")
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("SurfaceSeg metadata JSON must contain an object")
    return document, metadata_path.parent


def _metadata_path(
    value: Any,
    *,
    base: Path | None,
    label: str,
    suffixes: set[str],
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"SurfaceSeg result is missing {label}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise FileNotFoundError(f"invalid {label}: {path}")
    return path


def _encode_uint8(array: np.ndarray, **metadata: Any) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "axes": list(AXES_ZYX),
        "encoding": "base64-uint8",
        "values": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        **metadata,
    }


def _encode_discrete_labels(
    labels_zyx: np.ndarray,
    *,
    invalid_label: int,
    reported_max: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric = np.asarray(labels_zyx)
    if not np.issubdtype(numeric.dtype, np.integer):
        if not np.issubdtype(numeric.dtype, np.number):
            raise ValueError("sampled SurfaceSeg mask must be numeric")
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
            raise ValueError("sampled SurfaceSeg mask must contain integer labels")
    labels = numeric.astype(np.int64, copy=False)
    if np.any((labels < 0) & (labels != invalid_label)):
        raise ValueError(
            f"sampled SurfaceSeg mask contains a negative label other than {invalid_label}"
        )
    valid = labels != invalid_label
    if np.any(labels[valid] > reported_max):
        raise ValueError("sampled SurfaceSeg mask exceeds segmentation.label_range")

    # Reserve code 0 for missing traces.  Valid labels occupy [32, 255], which
    # lets CIGVis use jet + nearest interpolation while ``excpt=min`` makes
    # only the missing-trace sentinel transparent.  Keeping label 0 away from
    # the transparent endpoint also makes the first valid stratum clearly
    # visible in the whole-volume WebGL transfer function.  If there are more
    # than 224 labels the display is intentionally quantized, while the .npy
    # file
    # remains the authoritative int16 result.
    encoded = np.zeros(labels.shape, dtype=np.uint8)
    if np.any(valid):
        if reported_max <= 0:
            encoded[valid] = np.uint8(255)
        else:
            scaled = 32.0 + labels[valid].astype(np.float64) * (223.0 / reported_max)
            encoded[valid] = np.clip(np.rint(scaled), 32, 255).astype(np.uint8)
    return encoded, {
        "invalidLabel": int(invalid_label),
        "labelValueRange": [0, int(reported_max)],
        "displayCodeRange": [32, 255],
        "displayQuantized": bool(reported_max > 223),
    }


def build_surface_seg_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
    overlay_layer: str = "mask",
) -> dict[str, Any]:
    """Build a sparse seismic background with an aligned stratigraphic layer.

    ``overlay_layer`` may be ``"mask"`` (the default discrete instance labels)
    or ``"confidence"`` (the continuous confidence volume).  Both bounded
    arrays are included under ``surfaceSeg`` so a client can switch layers
    without reading the complete SEG-Y background.
    """

    result, metadata_base = _result_document(result_or_metadata)
    if result.get("model_id") != SURFACE_SEG_MODEL_ID:
        raise ValueError(
            f"visualization result must use model_id={SURFACE_SEG_MODEL_ID}"
        )
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    segmentation = result.get("segmentation")
    inference = result.get("inference")
    if not isinstance(input_metadata, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("SurfaceSeg result must contain input and outputs mappings")
    if not isinstance(segmentation, Mapping) or not isinstance(inference, Mapping):
        raise ValueError(
            "SurfaceSeg result must contain segmentation and inference mappings"
        )
    input_axes = tuple(
        str(axis).upper() for axis in input_metadata.get("axes", ())
    )
    segmentation_axes = tuple(
        str(axis).upper()
        for axis in segmentation.get("axes", input_metadata.get("axes", ()))
    )
    if input_axes != AXES_ICS or segmentation_axes != AXES_ICS:
        raise ValueError(
            f"SurfaceSeg axes must be {AXES_ICS}, got input={input_axes}, "
            f"segmentation={segmentation_axes}"
        )

    output_shape_ics = _shape3(
        segmentation.get("shape_ics", input_metadata.get("shape_ics", ())),
        "segmentation.shape_ics",
    )
    input_shape_ics = _shape3(
        input_metadata.get("shape_ics", output_shape_ics),
        "input.shape_ics",
    )
    if input_shape_ics != output_shape_ics:
        raise ValueError(
            "SurfaceSeg input.shape_ics and segmentation.shape_ics must match"
        )
    source_shape_zyx = _shape3(
        input_metadata.get("source_shape_zyx", ()),
        "input.source_shape_zyx",
    )
    output_inline_count, output_crossline_count, output_sample_count = (
        output_shape_ics
    )
    output_shape_zyx = (
        output_sample_count,
        output_inline_count,
        output_crossline_count,
    )
    if (
        output_sample_count != source_shape_zyx[0]
        or output_crossline_count != source_shape_zyx[2]
        or output_inline_count > source_shape_zyx[1]
    ):
        raise ValueError(
            "SurfaceSeg output shape does not align with the SEG-Y "
            f"source shape: ICS={output_shape_ics}, source ZYX={source_shape_zyx}"
        )
    max_inlines_value = inference.get("max_inlines")
    if max_inlines_value is None:
        if output_inline_count != source_shape_zyx[1]:
            raise ValueError(
                "partial Inline output requires inference.max_inlines so the "
                "first-Inline smoke-test alignment is explicit"
            )
        smoke_mode = False
    else:
        try:
            max_inlines = int(max_inlines_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("inference.max_inlines must be a positive integer") from exc
        if (
            isinstance(max_inlines_value, (bool, np.bool_))
            or max_inlines <= 0
            or max_inlines != max_inlines_value
        ):
            raise ValueError("inference.max_inlines must be a positive integer")
        expected_inline_count = min(max_inlines, source_shape_zyx[1])
        if output_inline_count != expected_inline_count:
            raise ValueError(
                "SurfaceSeg smoke output must contain the first "
                f"{expected_inline_count} Inline slices, got {output_inline_count}"
            )
        smoke_mode = output_inline_count < source_shape_zyx[1]

    source = _metadata_path(
        input_metadata.get("source"),
        base=metadata_base,
        label="input.source",
        suffixes={".sgy", ".segy"},
    )
    mask_path = _metadata_path(
        outputs.get("mask_npy"),
        base=metadata_base,
        label="outputs.mask_npy",
        suffixes={".npy"},
    )
    confidence_path = _metadata_path(
        outputs.get("confidence_npy"),
        base=metadata_base,
        label="outputs.confidence_npy",
        suffixes={".npy"},
    )
    mask_ics = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    confidence_ics = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    if mask_ics.ndim != 3 or tuple(mask_ics.shape) != output_shape_ics:
        raise ValueError(
            f"mask array shape {mask_ics.shape} does not match {output_shape_ics}"
        )
    if confidence_ics.ndim != 3 or tuple(confidence_ics.shape) != output_shape_ics:
        raise ValueError(
            "confidence array shape "
            f"{confidence_ics.shape} does not match {output_shape_ics}"
        )
    if not np.issubdtype(mask_ics.dtype, np.number):
        raise ValueError("SurfaceSeg mask array must be numeric")
    if not np.issubdtype(confidence_ics.dtype, np.number):
        raise ValueError("SurfaceSeg confidence array must be numeric")

    label_range_raw = segmentation.get("label_range", ())
    if (
        not isinstance(label_range_raw, Sequence)
        or isinstance(label_range_raw, (str, bytes))
        or len(label_range_raw) != 2
    ):
        raise ValueError("segmentation.label_range must contain [min, max]")
    try:
        reported_min, reported_max = (
            int(label_range_raw[0]),
            int(label_range_raw[1]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("segmentation.label_range must contain integers") from exc
    invalid_label = int(segmentation.get("invalid_label", -1))
    if invalid_label >= 0:
        raise ValueError("segmentation.invalid_label must be negative")
    if reported_min < invalid_label or reported_max < 0 or reported_min > reported_max:
        raise ValueError("invalid segmentation.label_range")

    background, cache_hit = cache.get_crop(
        source,
        crop_start_zyx=(0, 0, 0),
        crop_size_zyx=output_shape_zyx,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=segy_options,
        expected_source_shape_zyx=source_shape_zyx,
    )
    z_indices, inline_indices, crossline_indices = (
        background.sample_indices_zyx
    )
    # NPY is [I, X, S]; select in that native order first and transpose only
    # the bounded sample into the platform's [Z, I, X] contract.
    selection_ics = np.ix_(inline_indices, crossline_indices, z_indices)
    mask_sample_zyx = np.transpose(
        np.asarray(mask_ics[selection_ics]),
        (2, 0, 1),
    )
    confidence_sample_zyx = np.transpose(
        np.asarray(confidence_ics[selection_ics], dtype=np.float32),
        (2, 0, 1),
    )
    if (
        mask_sample_zyx.shape != background.cube_int8.shape
        or confidence_sample_zyx.shape != background.cube_int8.shape
    ):
        raise ValueError(
            "SurfaceSeg overlay sampling is not aligned with the SEG-Y background"
        )
    if (
        not np.all(np.isfinite(confidence_sample_zyx))
        or np.any(confidence_sample_zyx < 0.0)
        or np.any(confidence_sample_zyx > 1.0)
    ):
        raise ValueError("sampled SurfaceSeg confidence contains values outside [0, 1]")

    encoded_labels, label_metadata = _encode_discrete_labels(
        mask_sample_zyx,
        invalid_label=invalid_label,
        reported_max=reported_max,
    )
    confidence_uint8 = np.rint(confidence_sample_zyx * 255.0).astype(np.uint8)
    mask_spec = _encode_uint8(
        encoded_labels,
        valueRange=[0.0, 1.0],
        source=str(mask_path),
        **label_metadata,
    )
    confidence_spec = _encode_uint8(
        confidence_uint8,
        valueRange=[0.0, 1.0],
        source=str(confidence_path),
    )

    normalized_layer = str(overlay_layer).strip().lower()
    if normalized_layer not in {"mask", "confidence"}:
        raise ValueError("overlay_layer must be 'mask' or 'confidence'")
    if normalized_layer == "mask":
        overlay = {
            "id": "surface_seg_labels",
            "name": "地层分割标签",
            "kind": "labels",
            "volume": mask_spec,
            "clim": [0.0, 1.0],
            "cmap": "jet",
            "alpha": 0.68,
            "excpt": "min",
            "interpolation": "nearest",
        }
    else:
        confidence_floor = float(inference.get("query_threshold", 0.35))
        if not np.isfinite(confidence_floor) or not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("inference.query_threshold must stay within [0, 1]")
        overlay = {
            "id": "surface_seg_confidence",
            "name": "地层分割置信度",
            "kind": "confidence",
            "volume": confidence_spec,
            "clim": [confidence_floor, 1.0],
            "cmap": "jet",
            "alpha": 0.62,
            "excpt": "min",
            "interpolation": "nearest",
        }

    payload = background.as_payload(cache_hit=cache_hit)
    payload["contractVersion"] = "surface-seg-cigvis-v1"
    payload["name"] = f"{source.name} · 地层分割"
    payload["surfaceSeg"] = {
        "modelId": SURFACE_SEG_MODEL_ID,
        "sourceAxes": list(AXES_ICS),
        "platformAxes": list(AXES_ZYX),
        "transposeICSToZYX": [2, 0, 1],
        "shapeICS": list(output_shape_ics),
        "shapeZYX": list(output_shape_zyx),
        "processedInlineRange": [0, output_inline_count - 1],
        "smokeMode": smoke_mode,
        "crossInlineConsistent": bool(
            segmentation.get("cross_inline_consistent", False)
        ),
        "mask": mask_spec,
        "confidence": confidence_spec,
        "display": {
            "backgroundCmap": "seismic",
            "preferredLayer": normalized_layer,
            "maskCmap": "jet",
            "maskDiscrete": True,
            "confidenceCmap": "jet",
            "alpha": float(overlay["alpha"]),
            "excludeMinimum": True,
        },
        "cigvis": {
            "method": "add_mask",
            "sourceAxes": list(AXES_ZYX),
            "transposeZYXToLineFirst": [1, 2, 0],
        },
    }
    payload["overlays"] = [overlay]
    payload["preview"]["cacheStats"] = cache.stats
    payload["preview"]["vertical_note"] = (
        "地层标签/置信度由[Inline,Crossline,Sample]转为"
        "[Z,Inline,Crossline]，并与背景体共用同一稀疏采样索引"
    )
    return payload


__all__ = [
    "AXES_ICS",
    "SURFACE_SEG_MODEL_ID",
    "build_surface_seg_visualization_payload",
]
