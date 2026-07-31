"""Model-dispatched visualization payloads for downstream predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .faultseg_visualization import (
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
    build_faultseg_visualization_payload,
)
from .surface_seg_visualization import (
    SURFACE_SEG_MODEL_ID,
    build_surface_seg_visualization_payload,
)


PredictionBuilder = Callable[..., dict[str, Any]]

_BUILDERS: dict[str, PredictionBuilder] = {
    "faultseg_3d": build_faultseg_visualization_payload,
    SURFACE_SEG_MODEL_ID: build_surface_seg_visualization_payload,
}


def _prediction_document(
    result_or_metadata: Mapping[str, Any] | str | Path,
) -> Mapping[str, Any]:
    if isinstance(result_or_metadata, Mapping):
        result: Mapping[str, Any] = result_or_metadata
        if "prediction" in result and "model_id" not in result:
            nested = result.get("prediction")
            if not isinstance(nested, Mapping):
                raise ValueError("prediction wrapper must contain a mapping")
            return nested
        return result
    path = Path(result_or_metadata).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prediction metadata does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("prediction metadata JSON must contain an object")
    return _prediction_document(document)


def build_prediction_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
    overlay_layer: str | None = None,
) -> dict[str, Any]:
    """Dispatch a prediction to its model-specific CIGVis payload builder."""

    result = _prediction_document(result_or_metadata)
    model_id = str(result.get("model_id", "")).strip()
    try:
        builder = _BUILDERS[model_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_BUILDERS))
        raise ValueError(
            f"model_id={model_id or '<missing>'} has no visualization adapter; "
            f"supported: {supported}"
        ) from exc
    kwargs: dict[str, Any] = {
        "cache": cache,
        "config": config,
        "segy_options": segy_options,
        "max_shape_zyx": max_shape_zyx,
    }
    if model_id == SURFACE_SEG_MODEL_ID and overlay_layer is not None:
        kwargs["overlay_layer"] = overlay_layer
    payload = builder(result_or_metadata, **kwargs)
    model_name = str(result.get("model_name", model_id))
    preferred_layer = (
        payload.get("surfaceSeg", {}).get("display", {}).get("preferredLayer")
        if model_id == SURFACE_SEG_MODEL_ID
        else payload.get("faultSeg", {}).get("display", {}).get("preferredLayer")
    )
    payload["predictionVisualization"] = {
        "modelId": model_id,
        "modelName": model_name,
        "preferredLayer": preferred_layer,
        "overlayCount": len(payload.get("overlays", [])),
    }
    return payload


def supported_prediction_visualization_models() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


__all__ = [
    "build_prediction_visualization_payload",
    "supported_prediction_visualization_models",
]
