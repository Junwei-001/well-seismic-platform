"""Bounded visualization adapter for WellFuse P17 horizon candidates.

The online P17 runner emits four named TWT surfaces for an unknown survey.
Those products are experimental candidates, not held-out predictions with
truth/error evidence.  This module therefore exposes a separate, fail-closed
renderability decision and never manufactures scientific acceptance metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .faultseg_visualization import (
    DEFAULT_FAULTSEG_SLICE_CACHE,
    DEFAULT_MAX_SHAPE_ZYX,
    SegySliceCache,
    _metadata_path,
    _result_document,
    _shape3,
)

HORIZON_MODEL_ID = "wellfuse_horizon_p17"
HORIZON_NAMES = ("Hartha", "Tanuma", "Khasib", "Zubair")
HORIZON_CANDIDATE_DISPLAY_CONTRACT_VERSION = (
    "well-seismic.horizon-candidate-display.v1"
)

_SURFACE_STYLES = (
    ("#168aad", "Blues"),
    ("#43aa8b", "Greens"),
    ("#f9c74f", "YlOrBr"),
    ("#f94144", "Reds"),
)


def _prediction_document(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if "prediction" in result and "model_id" not in result:
        nested = result.get("prediction")
        return nested if isinstance(nested, Mapping) else {}
    return result


def _declared_output(
    outputs: Mapping[str, Any],
    *names: str,
) -> Any:
    for name in names:
        value = outputs.get(name)
        if isinstance(value, (str, Path)) and str(value).strip():
            return value
    return None


def evaluate_horizon_candidate_visualization(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute whether a P17 experimental candidate may be rendered.

    This decision is intentionally separate from result-display acceptance.
    It allows an explicitly labelled candidate to be inspected without
    claiming quantitative validation or inventing truth/error panels.
    """

    prediction = _prediction_document(result)
    reasons: list[str] = []
    if prediction.get("model_id") != HORIZON_MODEL_ID:
        reasons.append("not_p17_horizon_model")
    if prediction.get("model_executed") is not True:
        reasons.append("model_execution_not_evidenced")

    inference = prediction.get("inference")
    if not isinstance(inference, Mapping):
        reasons.append("inference_metadata_missing")
        inference = {}
    if inference.get("result_status") != "experimental_runnable_candidate":
        reasons.append("experimental_candidate_status_missing")
    if inference.get("actual_checkpoint_loaded_and_forward_executed") is not True:
        reasons.append("real_checkpoint_forward_not_evidenced")

    outputs = prediction.get("outputs")
    if not isinstance(outputs, Mapping):
        reasons.append("output_manifest_missing")
        outputs = {}
    required = {
        "horizon_candidates": _declared_output(
            outputs, "horizon_candidates_npz", "candidate_npz"
        ),
        "uncertainty_sigma": _declared_output(
            outputs, "uncertainty_sigma_npy", "uncertainty_npy"
        ),
        "valid_mask": _declared_output(outputs, "valid_mask_npy"),
    }
    reasons.extend(
        f"{name}_artifact_missing" for name, value in required.items() if value is None
    )

    renderable = not reasons
    return {
        "contract_version": HORIZON_CANDIDATE_DISPLAY_CONTRACT_VERSION,
        "display_status": "experimental_candidate" if renderable else "unavailable",
        "renderable": renderable,
        "scientific_status": "experimental_candidate_not_validated_on_current_survey",
        "horizon_names": list(HORIZON_NAMES),
        "quantitative_acceptance_claimed": False,
        "truth_metrics_used": False,
        "error_metrics_used": False,
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_grid(values: np.ndarray) -> list[list[float | None]]:
    return [
        [round(float(value), 6) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def _finite_range(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [round(float(finite.min()), 6), round(float(finite.max()), 6)]


def _shape(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must contain exactly three dimensions")
    return _shape3(value, label)


def _resolved_spatial_roi(
    result: Mapping[str, Any],
    input_metadata: Mapping[str, Any],
    *,
    source_shape: tuple[int, int, int],
    result_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Resolve and cross-check the runner's bounded spatial selection.

    Older P17 results covered the complete SEG-Y grid and did not carry a
    ``spatial_roi`` receipt.  New bounded results carry the same receipt in the
    input and inference sections (and may repeat it under geometry).  Treat the
    receipt as authoritative, require every repeated copy to agree, and convert
    its spatial zero-based indices into the cache's Z/Inline/Xline crop.
    """

    inference = result.get("inference")
    geometry = result.get("geometry")
    candidates = [
        value
        for value in (
            input_metadata.get("spatial_roi"),
            inference.get("spatial_roi") if isinstance(inference, Mapping) else None,
            geometry.get("spatial_roi") if isinstance(geometry, Mapping) else None,
        )
        if value is not None
    ]
    if not candidates:
        if result_shape[1:] != source_shape[1:]:
            raise ValueError(
                "bounded P17 Horizon output is missing its spatial_roi receipt"
            )
        return (0, 0, 0), source_shape

    resolved_records: list[dict[str, int]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"spatial_roi receipt {index} must be a mapping")
        semantics = candidate.get("selection_semantics")
        if semantics != "zero_based_indices_into_sorted_unique_grid_axes":
            raise ValueError(
                f"unsupported P17 Horizon ROI selection semantics: {semantics}"
            )
        resolved = candidate.get("resolved")
        if not isinstance(resolved, Mapping):
            raise ValueError(f"spatial_roi receipt {index} is missing resolved bounds")
        names = (
            "inline_start",
            "inline_count",
            "crossline_start",
            "crossline_count",
        )
        try:
            record = {name: int(resolved[name]) for name in names}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"spatial_roi receipt {index} has invalid resolved bounds"
            ) from exc
        resolved_records.append(record)

        declared_source_shape = candidate.get("source_shape_t_inline_xline")
        if declared_source_shape is not None and _shape(
            declared_source_shape,
            f"spatial_roi[{index}].source_shape_t_inline_xline",
        ) != source_shape:
            raise ValueError("P17 Horizon ROI source shape disagrees with the SEG-Y")
        declared_selected_shape = candidate.get("selected_shape_t_inline_xline")
        expected_selected_shape = (
            source_shape[0],
            record["inline_count"],
            record["crossline_count"],
        )
        if declared_selected_shape is not None and _shape(
            declared_selected_shape,
            f"spatial_roi[{index}].selected_shape_t_inline_xline",
        ) != expected_selected_shape:
            raise ValueError("P17 Horizon ROI selected shape disagrees with its bounds")

    resolved = resolved_records[0]
    if any(record != resolved for record in resolved_records[1:]):
        raise ValueError("P17 Horizon spatial_roi receipts disagree")
    inline_start = resolved["inline_start"]
    inline_count = resolved["inline_count"]
    crossline_start = resolved["crossline_start"]
    crossline_count = resolved["crossline_count"]
    if inline_start < 0 or crossline_start < 0:
        raise ValueError("P17 Horizon ROI starts must be non-negative")
    if inline_count < 1 or crossline_count < 1:
        raise ValueError("P17 Horizon ROI counts must be positive")
    if inline_start + inline_count > source_shape[1] or (
        crossline_start + crossline_count > source_shape[2]
    ):
        raise ValueError("P17 Horizon spatial_roi exceeds the source SEG-Y grid")
    expected_result_shape = (len(HORIZON_NAMES), inline_count, crossline_count)
    if result_shape != expected_result_shape:
        raise ValueError(
            "P17 Horizon output shape disagrees with the resolved spatial_roi"
        )

    expected_bounded = bool(
        inline_start
        or crossline_start
        or inline_count != source_shape[1]
        or crossline_count != source_shape[2]
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("is_bounded") is not None:
            bounded_flag = candidate["is_bounded"]
            if not isinstance(bounded_flag, bool):
                raise TypeError("P17 Horizon spatial_roi bounded flag must be boolean")
            if bounded_flag != expected_bounded:
                raise ValueError("P17 Horizon spatial_roi bounded flag is inconsistent")
    return (
        (0, inline_start, crossline_start),
        (source_shape[0], inline_count, crossline_count),
    )


def _horizon_segy_options(
    result: Mapping[str, Any],
    input_metadata: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Translate P17/native geometry hints into ``SegyReader`` options.

    The P17 runner records native ``iline_byte``/``xline_byte`` options, while
    the platform reader calls the same fields ``inline_byte`` and
    ``crossline_byte``.  Carrying the resolved bytes with the result keeps the
    public visualization builder usable without requiring the caller to also
    supply the platform YAML profile bundle.
    """

    profile = str(input_metadata.get("geometry_profile") or "standard_3d")
    options: dict[str, Any] = {"profile": profile}
    geometry = result.get("geometry")
    geometry_bytes = input_metadata.get("geometry_bytes")
    sources = (
        input_metadata.get("recommended_options"),
        geometry_bytes,
        geometry,
        input_metadata,
    )
    aliases = {
        "inline_byte": (
            "inline_byte",
            "iline_byte",
            "inline_header_byte",
            "iline_header_byte",
        ),
        "crossline_byte": (
            "crossline_byte",
            "xline_byte",
            "crossline_header_byte",
            "xline_header_byte",
        ),
        "x_byte": ("x_byte", "x_header_byte"),
        "y_byte": ("y_byte", "y_header_byte"),
        "coordinate_scalar_byte": ("coordinate_scalar_byte",),
    }
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for target, names in aliases.items():
            for name in names:
                value = source.get(name)
                try:
                    byte = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= byte <= 240:
                    options[target] = byte
                    break

    # A caller may intentionally override any inferred field.  This preserves
    # the existing public API while making the common config=None path robust.
    options.update(dict(explicit or {}))
    return options


def build_horizon_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
) -> dict[str, Any]:
    """Build a bounded seismic preview with four aligned named TWT surfaces."""

    result, metadata_base = _result_document(result_or_metadata)
    decision = evaluate_horizon_candidate_visualization(result)
    if not decision["renderable"]:
        reasons = ", ".join(decision["reason_codes"])
        raise ValueError(f"P17 Horizon candidate is not renderable: {reasons}")

    input_metadata = result.get("input")
    outputs = result.get("outputs")
    if not isinstance(input_metadata, Mapping) or not isinstance(outputs, Mapping):
        raise TypeError("P17 Horizon result must contain input and outputs mappings")
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != ("HORIZON", "INLINE", "XLINE"):
        raise ValueError(
            "P17 Horizon result axes must be HORIZON/INLINE/XLINE, "
            f"got {axes}"
        )

    source_shape = _shape(input_metadata.get("source_shape_zyx"), "source_shape_zyx")
    result_shape = _shape(input_metadata.get("shape_zyx"), "input.shape_zyx")
    if result_shape[0] != len(HORIZON_NAMES):
        raise ValueError(
            "P17 Horizon output grid must contain four named surfaces"
        )
    crop_start, crop_size = _resolved_spatial_roi(
        result,
        input_metadata,
        source_shape=source_shape,
        result_shape=result_shape,
    )

    source_value = input_metadata.get("source")
    source = _metadata_path(
        source_value,
        base=metadata_base,
        label="input.source",
        suffix=Path(str(source_value or "")).suffix.lower(),
    )
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError("P17 Horizon input.source must be SEG-Y")

    candidate_path = _metadata_path(
        _declared_output(outputs, "horizon_candidates_npz", "candidate_npz"),
        base=metadata_base,
        label="outputs.horizon_candidates_npz",
        suffix=".npz",
    )
    uncertainty_path = _metadata_path(
        _declared_output(outputs, "uncertainty_sigma_npy", "uncertainty_npy"),
        base=metadata_base,
        label="outputs.uncertainty_sigma_npy",
        suffix=".npy",
    )
    valid_mask_path = _metadata_path(
        _declared_output(outputs, "valid_mask_npy"),
        base=metadata_base,
        label="outputs.valid_mask_npy",
        suffix=".npy",
    )

    with np.load(candidate_path, allow_pickle=False) as archive:
        required_keys = {
            "horizon_names",
            "prediction_twt_ms",
            "uncertainty_sigma_ms",
            "valid_mask",
            "ilines",
            "xlines",
        }
        missing = sorted(required_keys.difference(archive.files))
        if missing:
            raise ValueError(f"P17 Horizon candidate NPZ is missing keys: {missing}")
        names = tuple(_text(value) for value in archive["horizon_names"].tolist())
        prediction = np.asarray(archive["prediction_twt_ms"], dtype=np.float32)
        archive_sigma = np.asarray(archive["uncertainty_sigma_ms"], dtype=np.float32)
        archive_valid = np.asarray(archive["valid_mask"], dtype=bool)
        ilines = np.asarray(archive["ilines"])
        xlines = np.asarray(archive["xlines"])

    if names != HORIZON_NAMES:
        raise ValueError(f"P17 Horizon names must be {HORIZON_NAMES}, got {names}")
    if prediction.shape != result_shape:
        raise ValueError(
            f"P17 Horizon prediction shape {prediction.shape} does not match {result_shape}"
        )
    if archive_sigma.shape != prediction.shape or archive_valid.shape != prediction.shape:
        raise ValueError("P17 Horizon uncertainty/valid mask shape does not match prediction")
    if ilines.shape != (crop_size[1],) or xlines.shape != (crop_size[2],):
        raise ValueError("P17 Horizon Inline/Xline coordinates do not match its ROI")

    uncertainty = np.load(uncertainty_path, mmap_mode="r", allow_pickle=False)
    valid_mask = np.load(valid_mask_path, mmap_mode="r", allow_pickle=False)
    if uncertainty.shape != prediction.shape or valid_mask.shape != prediction.shape:
        raise ValueError("P17 Horizon standalone uncertainty/valid mask shape is invalid")
    if not np.allclose(
        np.asarray(uncertainty), archive_sigma, rtol=1e-5, atol=1e-5, equal_nan=True
    ):
        raise ValueError("P17 Horizon standalone uncertainty disagrees with candidate NPZ")
    if not np.array_equal(np.asarray(valid_mask, dtype=bool), archive_valid):
        raise ValueError("P17 Horizon standalone valid mask disagrees with candidate NPZ")
    if not np.all(np.isfinite(prediction[archive_valid])) or not np.all(
        np.isfinite(archive_sigma[archive_valid])
    ):
        raise ValueError("P17 Horizon contains non-finite values inside its valid mask")
    common_valid = np.all(archive_valid, axis=0)
    if np.any(common_valid) and np.any(np.diff(prediction[:, common_valid], axis=0) <= 0):
        raise ValueError("P17 Horizon surfaces violate the declared stratigraphic order")

    effective_segy_options = _horizon_segy_options(
        result,
        input_metadata,
        segy_options,
    )
    background, cache_hit = cache.get_crop(
        source,
        crop_start_zyx=crop_start,
        crop_size_zyx=crop_size,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=effective_segy_options,
        expected_source_shape_zyx=source_shape,
    )
    _z_indices, absolute_inline_indices, absolute_crossline_indices = (
        background.sample_indices_zyx
    )
    inline_indices = absolute_inline_indices - crop_start[1]
    crossline_indices = absolute_crossline_indices - crop_start[2]
    if not np.array_equal(ilines[inline_indices], background.inline_values):
        raise ValueError("P17 Horizon Inline coordinates are not aligned with SEG-Y")
    if not np.array_equal(xlines[crossline_indices], background.crossline_values):
        raise ValueError("P17 Horizon Xline coordinates are not aligned with SEG-Y")
    time_values = np.asarray(background.time_values, dtype=float)
    if time_values.size < 2 or not np.all(np.isfinite(time_values)):
        raise ValueError("P17 Horizon visualization requires a finite SEG-Y time axis")
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("P17 Horizon visualization requires an increasing SEG-Y TWT axis")

    sampled_prediction = prediction[:, inline_indices][:, :, crossline_indices]
    sampled_sigma = archive_sigma[:, inline_indices][:, :, crossline_indices]
    sampled_valid = archive_valid[:, inline_indices][:, :, crossline_indices]
    sampled_valid &= background.valid_traces[None, :, :]
    sampled_valid &= sampled_prediction >= time_values[0]
    sampled_valid &= sampled_prediction <= time_values[-1]
    z_coordinates = np.interp(
        sampled_prediction.reshape(-1),
        time_values,
        np.arange(time_values.size, dtype=float),
    ).reshape(sampled_prediction.shape)

    surfaces: list[dict[str, Any]] = []
    for horizon_index, (name, (color, cmap)) in enumerate(
        zip(HORIZON_NAMES, _SURFACE_STYLES, strict=True)
    ):
        visible = sampled_valid[horizon_index]
        if not np.any(visible):
            raise ValueError(f"P17 Horizon {name} has no valid points in the preview")
        grid = np.full(visible.shape, np.nan, dtype=float)
        grid[visible] = z_coordinates[horizon_index][visible]
        inline_position, crossline_position = np.nonzero(visible)
        points = np.column_stack(
            (
                inline_position,
                crossline_position,
                grid[visible],
            )
        )
        sigma_values = sampled_sigma[horizon_index][visible]
        twt_values = sampled_prediction[horizon_index][visible]
        surfaces.append(
            {
                "id": f"p17_horizon_{name.casefold()}",
                "name": name,
                "kind": "surface",
                "grid": _json_grid(grid),
                "points": np.round(points, 6).tolist(),
                "values": np.round(sigma_values.astype(float), 6).tolist(),
                "color": color,
                "cmap": cmap,
                "alpha": 0.82,
                "status": "experimental_candidate",
                "verticalDomain": "TWT",
                "verticalUnit": "ms",
                "twtRangeMs": _finite_range(twt_values),
                "uncertaintySigmaRangeMs": _finite_range(sigma_values),
                "validFraction": float(visible.mean()),
            }
        )

    payload = background.as_payload(cache_hit=cache_hit)
    payload.update(
        {
            "contractVersion": "wellfuse-horizon-cigvis-v1",
            "name": f"{source.name} · 四命名层位实验候选",
            "axisLabels": {
                "inline": "Inline",
                "crossline": "Xline",
                "sample": "双程时 TWT",
            },
            "verticalAxis": {
                "contractVersion": "well-seismic.vertical-axis.v2",
                "domain": "TWT",
                "label": "双程时 TWT（未知工区候选）",
                "unit": "ms",
                "reference": "source_seismic_grid_crs_unverified",
                "correctionState": "unknown",
                "direction": "increasing_downward",
                "top": round(float(time_values[0]), 6),
                "bottom": round(float(time_values[-1]), 6),
                "defaultView": "top_oblique",
                "twtCandidate": True,
                "twtVerified": False,
            },
            "surfaces": surfaces,
            "overlays": [],
            "horizon": {
                "modelId": HORIZON_MODEL_ID,
                "status": "experimental_candidate",
                "scientificStatus": decision["scientific_status"],
                "horizonNames": list(HORIZON_NAMES),
                "surfaceCount": len(surfaces),
                "validFraction": float(sampled_valid.mean()),
                "uncertaintyUnit": "ms TWT",
                "quantitativeAcceptanceClaimed": False,
                "truthMetricsUsed": False,
                "errorMetricsUsed": False,
                "display": {
                    "preferredLayer": "named_horizon_surfaces",
                    "surfaceAlpha": 0.82,
                },
            },
            "candidateVisualization": decision,
        }
    )
    payload["preview"].update(
        {
            "cacheStats": cache.stats,
            "candidateStatus": "experimental_candidate",
            "candidateArtifact": str(candidate_path),
            "uncertaintyArtifact": str(uncertainty_path),
            "validMaskArtifact": str(valid_mask_path),
            "surfaceCount": len(surfaces),
            "sampledSurfacePointCount": int(
                sum(len(surface["points"]) for surface in surfaces)
            ),
        }
    )
    return payload


__all__ = [
    "HORIZON_CANDIDATE_DISPLAY_CONTRACT_VERSION",
    "HORIZON_MODEL_ID",
    "HORIZON_NAMES",
    "build_horizon_visualization_payload",
    "evaluate_horizon_candidate_visualization",
]
