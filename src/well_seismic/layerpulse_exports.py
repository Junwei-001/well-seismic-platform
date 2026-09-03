"""Interoperable, geometry-bound downloads for the eleven LayerPulse heads."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .layerpulse_contract import LAYERPULSE_OUTPUT_SPECS, LayerPulseOutputSpec
from .standard_export import (
    build_geometry_bound_segy_source_context,
    materialize_geometry_bound_volume_segy,
)


LAYERPULSE_COMMON_EXPORT_CONTRACT_VERSION = (
    "well-seismic.layerpulse-common-exports.v1"
)

_CONTINUOUS_VALUE_SEMANTICS = {
    "rgt": "relative",
    "impedance": "model_scale",
    "porosity": "model_scale",
    "well_match": "score",
    "uncertainty": "score",
}


def resolve_layerpulse_output_spec(output_key: str) -> LayerPulseOutputSpec:
    resolved = str(output_key).strip()
    for spec in LAYERPULSE_OUTPUT_SPECS:
        if spec.output_key == resolved:
            return spec
    raise KeyError(f"unknown LayerPulse output key: {resolved}")


def layerpulse_output_stem(spec: LayerPulseOutputSpec) -> str:
    return spec.output_key.removesuffix("_logits")


def layerpulse_segy_artifact_key(spec: LayerPulseOutputSpec) -> str:
    return f"layerpulse_{layerpulse_output_stem(spec)}_download_sgy"


def layerpulse_class_legend_artifact_key(spec: LayerPulseOutputSpec) -> str:
    return f"layerpulse_{layerpulse_output_stem(spec)}_class_legend_csv"


def _shape3(
    value: object,
    *,
    field: str,
    allow_zero: bool = False,
) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"LayerPulse {field} must contain three integers")
    if len(value) != 3:
        raise ValueError(f"LayerPulse {field} must contain three integers")
    try:
        resolved = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"LayerPulse {field} must contain three integers") from exc
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in resolved):
        raise ValueError(f"LayerPulse {field} entries must be >= {minimum}")
    return resolved  # type: ignore[return-value]


def _declared_file(outputs: Mapping[str, Any], output_key: str) -> Path:
    raw = outputs.get(output_key)
    if isinstance(raw, Mapping):
        raw = raw.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"LayerPulse output is not registered: {output_key}")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LayerPulse output is missing: {output_key}")
    return path


def _validate_volume(
    path: Path,
    *,
    spec: LayerPulseOutputSpec,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or tuple(int(item) for item in array.shape) != expected_shape:
        raise ValueError(
            f"LayerPulse {spec.output_key} volume does not match the output ROI"
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"LayerPulse {spec.output_key} volume is not numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"LayerPulse {spec.output_key} volume is not finite")
    if spec.kind == "classification":
        if not np.allclose(array, np.rint(array), atol=0.0, rtol=0.0):
            raise ValueError(
                f"LayerPulse {spec.output_key} class volume is not integer-coded"
            )
        minimum = int(np.min(array))
        maximum = int(np.max(array))
        if minimum < 0 or maximum >= spec.channels:
            raise ValueError(
                f"LayerPulse {spec.output_key} class code exceeds 0..{spec.channels - 1}"
            )
    return array


def _write_class_legend(
    destination: Path,
    *,
    spec: LayerPulseOutputSpec,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".csv.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("output_key", "class_code", "class_name", "value_semantics")
            )
            for class_code, class_name in enumerate(spec.class_names):
                writer.writerow(
                    (spec.output_key, class_code, class_name, "integer_class_code")
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _catalog_by_output_key(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_catalog = result.get("task_catalog")
    if not isinstance(raw_catalog, list):
        raise ValueError("LayerPulse task catalog is unavailable")
    catalog: dict[str, dict[str, Any]] = {}
    for raw in raw_catalog:
        if not isinstance(raw, dict):
            continue
        output_key = str(raw.get("output_key") or "").strip()
        if output_key:
            catalog[output_key] = raw
    expected = {spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS}
    if set(catalog) != expected:
        raise ValueError("LayerPulse task catalog does not describe the exact 11 heads")
    return catalog


def materialize_layerpulse_common_exports(
    result: dict[str, Any],
    *,
    output_root: Path,
    only_output_keys: set[str] | None = None,
    formats: set[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Create common-format downloads without changing model values.

    SEG-Y is produced only when source geometry can be proven and replayed.
    Classification legends remain exportable as CSV even if SEG-Y geometry is
    unavailable.  In non-strict mode an individual export failure is recorded
    and never changes the inference task's completion status.
    """

    outputs = result.get("outputs")
    input_metadata = result.get("input")
    if not isinstance(outputs, dict) or not isinstance(input_metadata, Mapping):
        raise ValueError("LayerPulse result has no mutable outputs/input contract")
    catalog = _catalog_by_output_key(result)
    requested_keys = (
        {str(item) for item in only_output_keys}
        if only_output_keys is not None
        else {spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS}
    )
    unknown = requested_keys - {spec.output_key for spec in LAYERPULSE_OUTPUT_SPECS}
    if unknown:
        raise KeyError(f"unknown LayerPulse output keys: {sorted(unknown)}")
    requested_formats = {str(item).casefold() for item in (formats or {"sgy", "csv"})}
    if not requested_formats or requested_formats - {"sgy", "csv"}:
        raise ValueError("LayerPulse common export formats must be sgy and/or csv")

    source_shape = _shape3(
        input_metadata.get("source_shape_tix"), field="input.source_shape_tix"
    )
    roi_start = _shape3(
        input_metadata.get("crop_start_tix") or (0, 0, 0),
        field="input.crop_start_tix",
        allow_zero=True,
    )
    output_shape = _shape3(
        input_metadata.get("shape_tix") or input_metadata.get("crop_size_tix"),
        field="input.shape_tix",
    )
    if any(
        start + size > available
        for start, size, available in zip(
            roi_start, output_shape, source_shape, strict=True
        )
    ):
        raise ValueError("LayerPulse output ROI exceeds the declared source geometry")

    root = output_root.expanduser().resolve()
    export_root = root / "layerpulse_common_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    try:
        export_root.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("LayerPulse export directory escapes its output root") from exc

    source_context: Mapping[str, Any] | None = None
    context_error: str | None = None
    if "sgy" in requested_formats:
        try:
            source_context = build_geometry_bound_segy_source_context(result)
        except (OSError, TypeError, ValueError) as exc:
            context_error = str(exc)

    pending_outputs: dict[str, str] = {}
    successful: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    valid_mask_key = (
        "input_valid_mask_npy" if "input_valid_mask_npy" in outputs else None
    )
    for spec in LAYERPULSE_OUTPUT_SPECS:
        if spec.output_key not in requested_keys:
            continue
        stem = layerpulse_output_stem(spec)
        volume: np.ndarray | None = None
        if "sgy" in requested_formats:
            try:
                if context_error is not None:
                    raise ValueError(context_error)
                volume = _validate_volume(
                    _declared_file(outputs, spec.artifact_key),
                    spec=spec,
                    expected_shape=output_shape,
                )
                segy_key = layerpulse_segy_artifact_key(spec)
                segy_path = export_root / f"layerpulse_{stem}.sgy"
                value_semantics = (
                    "integer_class_code"
                    if spec.kind == "classification"
                    else _CONTINUOUS_VALUE_SEMANTICS[spec.output_key]
                )
                receipt = materialize_geometry_bound_volume_segy(
                    result,
                    volume,
                    axes=("TWT", "INLINE", "XLINE"),
                    source_shape=source_shape,
                    roi_start=roi_start,
                    destination=segy_path,
                    output_key=segy_key,
                    categorical=spec.kind == "classification",
                    value_semantics=value_semantics,
                    source_context=source_context,
                )
                pending_outputs[segy_key] = str(segy_path)
                successful.append(
                    {
                        "output_key": spec.output_key,
                        "artifact_key": segy_key,
                        "format": "sgy",
                        "role": "primary",
                        "value_semantics": value_semantics,
                        "valid_mask_artifact_key": valid_mask_key,
                        "geometry_receipt": receipt,
                    }
                )
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "output_key": spec.output_key,
                        "format": "sgy",
                        "error": str(exc),
                    }
                )

        if spec.kind == "classification" and "csv" in requested_formats:
            try:
                legend_key = layerpulse_class_legend_artifact_key(spec)
                legend_path = export_root / f"layerpulse_{stem}_class_map.csv"
                _write_class_legend(legend_path, spec=spec)
                pending_outputs[legend_key] = str(legend_path)
                successful.append(
                    {
                        "output_key": spec.output_key,
                        "artifact_key": legend_key,
                        "format": "csv",
                        "role": "class_legend",
                        "value_semantics": "integer_class_code",
                    }
                )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "output_key": spec.output_key,
                        "format": "csv",
                        "error": str(exc),
                    }
                )

    outputs.update(pending_outputs)
    exports_by_head: dict[str, list[dict[str, str]]] = {}
    for item in successful:
        exports_by_head.setdefault(str(item["output_key"]), []).append(
            {
                "artifact_key": str(item["artifact_key"]),
                "format": str(item["format"]),
                "role": str(item["role"]),
            }
        )
    for output_key, descriptors in exports_by_head.items():
        entry = catalog[output_key]
        entry["download_artifact_keys"] = descriptors
        primary = next(
            (item for item in descriptors if item["role"] == "primary"), None
        )
        legend = next(
            (item for item in descriptors if item["role"] == "class_legend"),
            None,
        )
        if primary is not None:
            entry["primary_download_artifact_key"] = primary["artifact_key"]
            entry["primary_download_format"] = primary["format"]
        if legend is not None:
            entry["class_legend_artifact_key"] = legend["artifact_key"]

    requested_count = len(requested_keys) * (1 if "sgy" in requested_formats else 0)
    requested_count += sum(
        1
        for spec in LAYERPULSE_OUTPUT_SPECS
        if spec.output_key in requested_keys
        and spec.kind == "classification"
        and "csv" in requested_formats
    )
    status = (
        "available"
        if len(successful) == requested_count and not errors
        else "partial"
        if successful
        else "unavailable"
    )
    receipt_document = {
        "contract_version": LAYERPULSE_COMMON_EXPORT_CONTRACT_VERSION,
        "status": status,
        "requested_output_keys": sorted(requested_keys),
        "requested_formats": sorted(requested_formats),
        "export_count": len(successful),
        "exports": successful,
        "errors": errors,
        "axis_order": ["TWT", "INLINE", "XLINE"],
        "source_shape_tix": list(source_shape),
        "roi_start_tix": list(roi_start),
        "roi_shape_tix": list(output_shape),
        "value_transform": "none",
        "classification_transport": "integer_class_code_as_ieee_float32_segy",
        "continuous_transport": "finite_model_value_as_ieee_float32_segy",
        "valid_mask_artifact_key": valid_mask_key,
    }
    result["layerpulse_common_exports"] = receipt_document
    if strict and errors:
        details = "; ".join(
            f"{item['output_key']}.{item['format']}: {item['error']}"
            for item in errors
        )
        raise ValueError(f"LayerPulse common export failed: {details}")
    return receipt_document


__all__ = [
    "LAYERPULSE_COMMON_EXPORT_CONTRACT_VERSION",
    "layerpulse_class_legend_artifact_key",
    "layerpulse_output_stem",
    "layerpulse_segy_artifact_key",
    "materialize_layerpulse_common_exports",
    "resolve_layerpulse_output_spec",
]
