"""Model-dispatched visualization payloads for downstream predictions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .facies_visualization import (
    F3_FACIES_MODEL_ID,
    build_f3_facies_visualization_payload,
)
from .fault_models import FAULTNET_MODEL_ID, FAULTSEG_MODEL_ID, FAULT_VOLUME_MODEL_IDS
from .faultseg_visualization import (
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
    build_faultseg_visualization_payload,
)
from .geobody_visualization import GEOBODY_MODELS, build_geobody_visualization_payload
from .horizon_visualization import (
    HORIZON_MODEL_ID,
    build_horizon_visualization_payload,
)
from .layerpulse_contract import LAYERPULSE_MODEL_ID
from .layerpulse_visualization import build_layerpulse_visualization_payload
from .surface_seg_visualization import (
    SURFACE_SEG_MODEL_ID,
    build_surface_seg_visualization_payload,
)

PredictionBuilder = Callable[..., dict[str, Any]]

_BUILDERS: dict[str, PredictionBuilder] = {
    **{
        model_id: build_faultseg_visualization_payload
        for model_id in FAULT_VOLUME_MODEL_IDS
    },
    HORIZON_MODEL_ID: build_horizon_visualization_payload,
    SURFACE_SEG_MODEL_ID: build_surface_seg_visualization_payload,
    F3_FACIES_MODEL_ID: build_f3_facies_visualization_payload,
    LAYERPULSE_MODEL_ID: build_layerpulse_visualization_payload,
    **{model_id: build_geobody_visualization_payload for model_id in GEOBODY_MODELS},
}
_DISPLAY_MODEL_NAMES = {
    FAULTSEG_MODEL_ID: "三维断层识别（SEG-Y地震体→断层二值掩码）",
    FAULTNET_MODEL_ID: "FaultNet 国内实测域断层识别（SEG-Y地震体→断层概率体与掩码）",
    SURFACE_SEG_MODEL_ID: "地层实例分割（SEG-Y地震体→地层标签体与置信度）",
    HORIZON_MODEL_ID: "四层位追踪（时间域SEG-Y→层位面与不确定性）",
    F3_FACIES_MODEL_ID: "六类三维地震相分割（时间域SEG-Y→相体与不确定性）",
    LAYERPULSE_MODEL_ID: "LayerPulse 多模态统一智能解释（共享 Backbone→11 项结果）",
    "wellfuse_channel_p17": "河道地质体识别（SEG-Y地震体→河道概率与几何属性）",
    "wellfuse_karst_p17": "岩溶地质体识别（SEG-Y地震体→岩溶概率与几何属性）",
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
        raise TypeError("prediction metadata JSON must contain an object")
    return _prediction_document(document)


def build_prediction_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
    overlay_layer: str | None = None,
    layerpulse_output_key: str | None = None,
    faultseg_block_id: str | None = None,
    verified_surface_horizon_display_contract: Mapping[str, Any] | None = None,
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
    if model_id == LAYERPULSE_MODEL_ID:
        selected_layerpulse_output = layerpulse_output_key or overlay_layer
        if selected_layerpulse_output is not None:
            kwargs["layerpulse_output_key"] = selected_layerpulse_output
    if (
        model_id == SURFACE_SEG_MODEL_ID
        and verified_surface_horizon_display_contract is not None
    ):
        kwargs["verified_horizon_display_contract"] = (
            verified_surface_horizon_display_contract
        )
    if model_id == FAULTSEG_MODEL_ID and faultseg_block_id is not None:
        kwargs["selected_block_id"] = faultseg_block_id
    payload = builder(result_or_metadata, **kwargs)
    model_name = _DISPLAY_MODEL_NAMES.get(
        model_id, str(result.get("model_name") or "已登记预测模型")
    )
    if model_id == SURFACE_SEG_MODEL_ID:
        preferred_layer = payload.get("surfaceSeg", {}).get("display", {}).get("preferredLayer")
    elif model_id == HORIZON_MODEL_ID:
        preferred_layer = payload.get("horizon", {}).get("display", {}).get(
            "preferredLayer"
        )
    elif model_id in GEOBODY_MODELS:
        preferred_layer = payload.get("geobody", {}).get("display", {}).get("preferredLayer")
    elif model_id == F3_FACIES_MODEL_ID:
        preferred_layer = payload.get("facies3d", {}).get("display", {}).get(
            "preferredLayer"
        )
    elif model_id == LAYERPULSE_MODEL_ID:
        preferred_layer = payload.get("layerPulse", {}).get("display", {}).get(
            "preferredLayer"
        )
    else:
        preferred_layer = payload.get("faultSeg", {}).get("display", {}).get("preferredLayer")
    descriptor = {
        "modelId": model_id,
        "modelName": model_name,
        "preferredLayer": preferred_layer,
        "overlayCount": len(payload.get("overlays", [])),
    }
    if model_id == HORIZON_MODEL_ID:
        descriptor.update(
            {
                "surfaceCount": len(payload.get("surfaces", [])),
                "scientificStatus": payload.get("horizon", {}).get(
                    "scientificStatus"
                ),
            }
        )
    elif model_id == F3_FACIES_MODEL_ID:
        descriptor.update(
            {
                "scientificStatus": payload.get("facies3d", {}).get(
                    "scientificStatus"
                ),
                "validatedScope": payload.get("facies3d", {}).get(
                    "validatedScope"
                ),
                "inferenceMode": payload.get("facies3d", {}).get(
                    "inferenceMode"
                ),
            }
        )
    elif model_id == LAYERPULSE_MODEL_ID:
        descriptor.update(
            {
                "surfaceCount": len(payload.get("surfaces", [])),
                "outputCount": payload.get("layerPulse", {}).get("outputCount"),
                "singleCheckpoint": payload.get("layerPulse", {}).get(
                    "singleCheckpoint"
                ),
                "singleForward": payload.get("layerPulse", {}).get(
                    "singleForward"
                ),
                "relativeTimeNoT0": payload.get("layerPulse", {}).get(
                    "relativeTimeNoT0"
                ),
            }
        )
    payload["predictionVisualization"] = descriptor
    return payload


def supported_prediction_visualization_models() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


__all__ = [
    "build_prediction_visualization_payload",
    "supported_prediction_visualization_models",
]
