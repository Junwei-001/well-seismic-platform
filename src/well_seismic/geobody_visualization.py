"""CIGVis adapter for runnable WellFuse P17 Channel/Karst candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .faultseg_visualization import (
    AXES_ZYX,
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
    _encode_array,
    _metadata_path,
    _result_document,
    _shape3,
)


GEOBODY_MODELS = {
    "wellfuse_channel_p17": ("河道候选概率", "YlOrBr"),
    "wellfuse_karst_p17": ("岩溶候选概率", "PuRd"),
}


def build_geobody_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
) -> dict[str, Any]:
    result, metadata_base = _result_document(result_or_metadata)
    model_id = str(result.get("model_id", ""))
    if model_id not in GEOBODY_MODELS:
        raise ValueError(f"unsupported WellFuse geobody visualization model: {model_id}")
    layer_name, color_map = GEOBODY_MODELS[model_id]
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    probability_metadata = result.get("probability")
    inference_metadata = result.get("inference")
    if not all(
        isinstance(value, Mapping)
        for value in (input_metadata, outputs, probability_metadata, inference_metadata)
    ):
        raise ValueError("WellFuse geobody result is missing input/output metadata")
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != AXES_ZYX:
        raise ValueError(f"WellFuse geobody axes must be {AXES_ZYX}, got {axes}")

    crop_start = _shape3(
        input_metadata.get("crop_start_zyx", ()), "crop_start_zyx", allow_zero=True
    )
    crop_size = _shape3(input_metadata.get("crop_size_zyx", ()), "crop_size_zyx")
    source_shape = _shape3(input_metadata.get("source_shape_zyx", ()), "source_shape_zyx")
    probability_shape = _shape3(
        probability_metadata.get("shape_zyx", ()), "probability.shape_zyx"
    )
    if probability_shape != crop_size:
        raise ValueError("WellFuse probability shape does not match the declared crop")

    source_value = input_metadata.get("source")
    source = _metadata_path(
        source_value,
        base=metadata_base,
        label="input.source",
        suffix=Path(str(source_value or "")).suffix.lower(),
    )
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError("WellFuse geobody source must be SEG-Y")
    probability_path = _metadata_path(
        outputs.get("probability_npy"),
        base=metadata_base,
        label="outputs.probability_npy",
        suffix=".npy",
    )
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probability.ndim != 3 or tuple(probability.shape) != crop_size:
        raise ValueError("WellFuse geobody probability array has an invalid shape")
    threshold = float(inference_metadata.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("WellFuse geobody threshold must stay within [0, 1]")

    background, cache_hit = cache.get_crop(
        source,
        crop_start_zyx=crop_start,
        crop_size_zyx=crop_size,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=segy_options,
        expected_source_shape_zyx=source_shape,
    )
    relative_indices = tuple(
        indices - start for indices, start in zip(background.sample_indices_zyx, crop_start)
    )
    sampled = np.asarray(probability[np.ix_(*relative_indices)], dtype=np.float32)
    if sampled.shape != background.cube_int8.shape or not np.all(np.isfinite(sampled)):
        raise ValueError("WellFuse geobody overlay is not aligned with the SEG-Y preview")
    sampled = np.clip(sampled, 0.0, 1.0)
    probability_spec = {
        **_encode_array(np.rint(sampled * 255.0).astype(np.uint8), "base64-uint8"),
        "valueRange": [0.0, 1.0],
        "source": str(probability_path),
    }

    payload = background.as_payload(cache_hit=cache_hit)
    payload["contractVersion"] = "wellfuse-geobody-cigvis-v1"
    payload["name"] = f"{source.name} · {layer_name}"
    payload["geobody"] = {
        "modelId": model_id,
        "threshold": threshold,
        "status": "provisional_real_survey_candidates",
        "probability": probability_spec,
        "display": {
            "preferredLayer": "probability",
            "probabilityCmap": color_map,
            "alpha": 0.64,
        },
    }
    payload["overlays"] = [
        {
            "id": f"{model_id}_probability",
            "name": layer_name,
            "kind": "probability",
            "volume": probability_spec,
            "clim": [threshold, 1.0],
            "cmap": color_map,
            "alpha": 0.64,
            "excpt": "min",
        }
    ]
    payload["preview"]["cacheStats"] = cache.stats
    return payload


__all__ = ["GEOBODY_MODELS", "build_geobody_visualization_payload"]
