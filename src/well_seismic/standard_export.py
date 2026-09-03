"""Competition-facing manifests and reconstructable spatial result exports."""

from __future__ import annotations

import csv
import io
import json
import os
import struct
import zlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .content_identity import canonical_sha256, file_sha256
from .config import load_yaml
from .fault_models import is_fault_volume_model_id
from .io.segy import SegyReader
from .visualization_layers import prediction_result_to_artifact_bundle


STANDARD_EXPORT_MANIFEST_CONTRACT_VERSION = (
    "well-seismic.competition-standard-result.v1"
)
STANDARD_SLICE_BUNDLE_CONTRACT_VERSION = "well-seismic.complete-2d-slice-bundle.v1"
LEGACY_BOUNDED_SLICE_BUNDLE_CONTRACT_VERSION = "well-seismic.bounded-2d-slice-bundle.v1"
FAULT_MASK_AUDIT_CONTRACT_VERSION = "well-seismic.fault-mask-audit.v1"

_PRIMARY_KEY_PRIORITY = (
    "class_code",
    "argmax",
    "label",
    "prediction",
    "mask",
    "probability",
)


def _trusted_json_copy(value: Any, *, description: str) -> Any:
    """Return the exact JSON value that can be hashed and written publicly."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must contain only finite JSON-serializable values"
        ) from exc


def _declared_path(value: object) -> Path | None:
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser().resolve()
    if isinstance(value, Mapping):
        raw = value.get("path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw).expanduser().resolve()
    return None


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _preview_rgb(array: np.ndarray, *, categorical: bool) -> np.ndarray:
    values = np.asarray(array)
    stride = tuple(max(1, int(np.ceil(size / 1024))) for size in values.shape)
    values = values[:: stride[0], :: stride[1]]
    finite = np.isfinite(values)
    if categorical:
        codes = np.zeros(values.shape, dtype=np.int64)
        codes[finite] = np.rint(values[finite]).astype(np.int64)
        palette = np.asarray(
            [
                [8, 17, 31],
                [66, 211, 255],
                [255, 184, 77],
                [156, 245, 122],
                [255, 121, 198],
                [189, 147, 249],
                [248, 113, 113],
                [45, 212, 191],
            ],
            dtype=np.uint8,
        )
        rgb = palette[np.mod(codes, len(palette))]
        rgb[~finite] = 0
        return rgb
    finite_values = values[finite].astype(np.float64, copy=False)
    if finite_values.size:
        low, high = np.percentile(finite_values, (2.0, 98.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite] = np.clip(
        (values[finite].astype(np.float32) - float(low)) / float(high - low),
        0.0,
        1.0,
    )
    # A compact blue-cyan-yellow ramp with no plotting-library dependency.
    red = np.clip(2.2 * normalized - 0.65, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(normalized - 0.58) * 3.0, 0.0, 1.0)
    blue = np.clip(1.25 - 1.35 * normalized, 0.0, 1.0)
    rgb = np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    rgb[~finite] = 0
    return rgb


def _png_bytes(array: np.ndarray, *, categorical: bool) -> bytes:
    rgb = np.ascontiguousarray(_preview_rgb(array, categorical=categorical))
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("standard slice preview must be RGB")
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _fault_mask_png_bytes(array: np.ndarray) -> bytes:
    """Render a binary mask with a red fill and high-contrast yellow edge."""

    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("fault mask preview must be a two-dimensional plane")
    stride = tuple(max(1, int(np.ceil(size / 1024))) for size in values.shape)
    selected = np.asarray(values[:: stride[0], :: stride[1]])
    if not np.all(np.isin(selected, (0, 1, False, True))):
        raise ValueError("fault mask preview found a non-binary value")
    mask = selected != 0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = mask & ~interior
    rgb = np.full((*mask.shape, 3), (238, 241, 244), dtype=np.uint8)
    rgb[interior] = (239, 54, 61)
    rgb[boundary] = (255, 210, 48)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("fault mask preview must be RGB")
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _fault_mask_seismic_overlay_png_bytes(
    mask_array: np.ndarray,
    seismic_array: np.ndarray,
) -> bytes:
    """Render a binary fault mask over its aligned grayscale seismic plane."""

    mask_values = np.asarray(mask_array)
    seismic_values = np.asarray(seismic_array, dtype=np.float32)
    if (
        mask_values.ndim != 2
        or seismic_values.ndim != 2
        or mask_values.shape != seismic_values.shape
    ):
        raise ValueError("fault mask and seismic preview planes must align")
    if not np.all(np.isin(mask_values, (0, 1, False, True))):
        raise ValueError("fault mask preview found a non-binary value")
    stride = tuple(
        max(1, int(np.ceil(size / 1024))) for size in mask_values.shape
    )
    row_starts = np.arange(0, mask_values.shape[0], stride[0], dtype=np.int64)
    column_starts = np.arange(
        0, mask_values.shape[1], stride[1], dtype=np.int64
    )
    active = np.asarray(mask_values != 0, dtype=np.uint8)
    collapsed_rows = np.add.reduceat(active, row_starts, axis=0)
    sampled_mask = (
        np.add.reduceat(collapsed_rows, column_starts, axis=1) > 0
    )
    sampled_seismic = seismic_values[np.ix_(row_starts, column_starts)]
    finite = np.isfinite(sampled_seismic)
    finite_amplitude = np.abs(sampled_seismic[finite]).astype(
        np.float64, copy=False
    )
    scale = (
        float(np.percentile(finite_amplitude, 99.0))
        if finite_amplitude.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = np.zeros(sampled_seismic.shape, dtype=np.float32)
    normalized[finite] = np.clip(sampled_seismic[finite] / scale, -1.0, 1.0)
    gray = np.rint(136.0 + normalized * 106.0).astype(np.uint8)
    rgb = np.repeat(gray[..., np.newaxis], 3, axis=2)
    rgb[~finite] = (232, 236, 240)
    padded = np.pad(sampled_mask, 1, mode="constant", constant_values=False)
    interior = (
        sampled_mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = sampled_mask & ~interior
    if np.any(interior):
        base = rgb[interior].astype(np.float32)
        red = np.asarray((239, 54, 61), dtype=np.float32)
        rgb[interior] = np.rint(base * 0.38 + red * 0.62).astype(np.uint8)
    rgb[boundary] = (255, 205, 40)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("fault seismic overlay preview must be RGB")
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _even_indices(size: int, *, maximum: int = 5) -> list[int]:
    count = min(maximum, size)
    return sorted({int(value) for value in np.rint(np.linspace(0, size - 1, count))})


def _primary_spatial_layer(result: Mapping[str, Any]) -> dict[str, Any] | None:
    outputs = result.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    model_id = str(result.get("model_id") or "").casefold()

    # A producer may already emit a geometry-preserving mask SEG-Y.  The NPY
    # mask remains the authoritative numeric source for the complete slice
    # bundle and exact per-axis census; the native SEG-Y is published alongside
    # it rather than short-circuiting those standard exports.
    if is_fault_volume_model_id(model_id):
        path = _declared_path(outputs.get("mask_npy"))
        input_metadata = result.get("input")
        input_metadata = (
            input_metadata if isinstance(input_metadata, Mapping) else {}
        )
        if path is not None:
            return {
                "priority": 0,
                "output_key": "mask_npy",
                "path": path,
                "kind": "volume",
                "role": "prediction",
                "axis_order": list(
                    input_metadata.get("axes")
                    or ("Z", "INLINE", "CROSSLINE")
                ),
                "geometry": {},
            }

    # These two producer contracts store the quantitative grid inside NPZ.  A
    # generic extension-based layer classifier cannot safely guess the member,
    # so bind the documented member and axes here instead.
    if model_id == "wellfuse_horizon_p17":
        path = _declared_path(outputs.get("horizon_candidates_npz"))
        if path is not None:
            return {
                "priority": 0,
                "output_key": "horizon_candidates_npz",
                "path": path,
                "kind": "surface",
                "role": "prediction",
                "axis_order": ["HORIZON", "INLINE", "XLINE"],
                "geometry": {},
                "array_key": "prediction_twt_ms",
            }
    if model_id == "wellfuse_facies_3d_p17":
        path = _declared_path(outputs.get("sample_candidate_artifact"))
        if path is not None:
            return {
                "priority": 0,
                "output_key": "sample_candidate_artifact",
                "path": path,
                "kind": "volume",
                "role": "prediction",
                "axis_order": ["TWT", "INLINE", "XLINE"],
                "geometry": {},
                "array_key": "argmax_code",
            }
    try:
        bundle = prediction_result_to_artifact_bundle(result)
    except (TypeError, ValueError):
        return None
    candidates: list[dict[str, Any]] = []
    for layer in bundle.layers:
        output_key = str(layer.metadata.get("output_key") or "")
        if (
            layer.kind not in {"volume", "surface"}
            or layer.role != "prediction"
            or "valid_mask" in output_key.casefold()
        ):
            continue
        output_value = outputs.get(output_key)
        path = _declared_path(output_value)
        if path is None:
            continue
        key = output_key.casefold()
        priority = next(
            (
                index
                for index, token in enumerate(_PRIMARY_KEY_PRIORITY)
                if token in key
            ),
            len(_PRIMARY_KEY_PRIORITY),
        )
        candidates.append(
            {
                "priority": priority,
                "output_key": output_key,
                "path": path,
                "kind": layer.kind,
                "role": layer.role,
                "axis_order": list(layer.axis_order),
                "geometry": dict(layer.geometry),
            }
        )
    return (
        min(candidates, key=lambda item: (item["priority"], item["output_key"]))
        if candidates
        else None
    )


def _axis_kind(value: object) -> str:
    name = str(value).strip().upper().replace("-", "_")
    if name in {"INLINE", "ILINE", "I"}:
        return "inline"
    if name in {"CROSSLINE", "XLINE", "XL", "X"}:
        return "crossline"
    if name in {"Z", "T", "TWT", "TIME", "SAMPLE", "DEPTH"}:
        return "sample"
    if name == "HORIZON":
        return "horizon"
    return name.casefold()


def _shape3(value: object) -> tuple[int, int, int] | None:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        try:
            shape = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if all(item >= 0 for item in shape):
            return shape
    return None


def _reorder_zyx(
    values: tuple[int, int, int], axes: Sequence[str]
) -> tuple[int, int, int] | None:
    by_kind = {"sample": values[0], "inline": values[1], "crossline": values[2]}
    reordered: list[int] = []
    for axis in axes:
        kind = _axis_kind(axis)
        if kind not in by_kind:
            return None
        reordered.append(by_kind[kind])
    return tuple(reordered)  # type: ignore[return-value]


def _spatial_roi_contract(
    result: Mapping[str, Any], *, axes: Sequence[str], shape: tuple[int, int, int]
) -> dict[str, Any]:
    """Prove the result extent; fail closed instead of inventing full-survey scope."""

    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    inference = result.get("inference")
    inference = inference if isinstance(inference, Mapping) else {}
    model_id = str(result.get("model_id") or "").casefold()
    source_zyx = _shape3(input_metadata.get("source_shape_zyx"))

    if model_id == "wellfuse_horizon_p17":
        spatial_roi = input_metadata.get("spatial_roi") or inference.get("spatial_roi")
        if not isinstance(spatial_roi, Mapping):
            raise ValueError("horizon result has no auditable spatial_roi receipt")
        resolved = spatial_roi.get("resolved")
        resolved = resolved if isinstance(resolved, Mapping) else spatial_roi
        try:
            inline_start = int(
                resolved["inline_start_index"]
                if "inline_start_index" in resolved
                else resolved["inline_start"]
            )
            crossline_start = int(
                resolved["crossline_start_index"]
                if "crossline_start_index" in resolved
                else resolved["crossline_start"]
            )
            inline_count = int(resolved["inline_count"])
            crossline_count = int(resolved["crossline_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "horizon spatial_roi has no resolved index bounds"
            ) from exc
        expected_by_kind = {
            "horizon": shape[
                axes.index(next(a for a in axes if _axis_kind(a) == "horizon"))
            ],
            "inline": inline_count,
            "crossline": crossline_count,
        }
        if tuple(expected_by_kind[_axis_kind(axis)] for axis in axes) != shape:
            raise ValueError("horizon array shape disagrees with spatial_roi")
        if source_zyx is None:
            source_from_receipt = _shape3(
                resolved.get("source_shape_t_inline_xline")
                or spatial_roi.get("source_shape_t_inline_xline")
            )
            source_zyx = source_from_receipt
        if source_zyx is None:
            raise ValueError("horizon result has no auditable source grid shape")
        starts_by_kind = {
            "horizon": 0,
            "inline": inline_start,
            "crossline": crossline_start,
        }
        source_by_kind = {
            "horizon": expected_by_kind["horizon"],
            "inline": source_zyx[1],
            "crossline": source_zyx[2],
        }
        roi_start = tuple(starts_by_kind[_axis_kind(axis)] for axis in axes)
        source_shape = tuple(source_by_kind[_axis_kind(axis)] for axis in axes)
    else:
        if source_zyx is not None:
            source_shape = _reorder_zyx(source_zyx, axes)
        else:
            source_ics = _shape3(input_metadata.get("source_shape_ics"))
            if source_ics is not None and [_axis_kind(axis) for axis in axes] == [
                "inline",
                "crossline",
                "sample",
            ]:
                source_shape = source_ics
            else:
                source_shape = None
        if source_shape is None:
            raise ValueError("result has no source shape in the declared axis order")

        start_zyx = _shape3(
            input_metadata.get("crop_start_zyx")
            or input_metadata.get("resolved_roi_start_zyx")
        )
        size_zyx = _shape3(
            input_metadata.get("crop_size_zyx")
            or input_metadata.get("resolved_roi_size_zyx")
        )
        if start_zyx is not None and size_zyx is not None:
            roi_start = _reorder_zyx(start_zyx, axes)
            roi_size = _reorder_zyx(size_zyx, axes)
            if roi_start is None or roi_size != shape:
                raise ValueError("array shape disagrees with declared crop receipt")
        elif (
            model_id == "wellfuse_facies_3d_p17"
            and str(inference.get("mode") or "").casefold() == "sample"
        ):
            patch = _shape3(inference.get("patch_size"))
            if patch is None or patch != shape:
                raise ValueError("P17 facies sample shape disagrees with patch_size")
            roi_start = tuple(
                (available - selected) // 2
                for available, selected in zip(source_shape, shape, strict=True)
            )
        elif model_id == "seismic_surface_seg":
            # SurfaceSeg explicitly evaluates the first N complete inline
            # slices and preserves full crossline/sample extent.
            if shape[1:] != source_shape[1:] or shape[0] > source_shape[0]:
                raise ValueError("SurfaceSeg result is not a proven first-inline ROI")
            roi_start = (0, 0, 0)
        elif shape == source_shape:
            roi_start = (0, 0, 0)
        else:
            raise ValueError("partial result has no auditable ROI origin")

    if any(start < 0 for start in roi_start) or any(
        start + size > available
        for start, size, available in zip(roi_start, shape, source_shape, strict=True)
    ):
        raise ValueError("declared ROI exceeds the source grid")
    full_survey = roi_start == (0, 0, 0) and shape == source_shape
    return {
        "coverage": "complete_for_declared_roi",
        "scope": "full_survey" if full_survey else "declared_roi",
        "full_survey": full_survey,
        "source_shape": list(source_shape),
        "source_shape_status": "producer_declared",
        "roi_start": list(roi_start),
        "roi_shape": list(shape),
        "roi_stop_exclusive": [
            start + size for start, size in zip(roi_start, shape, strict=True)
        ],
        "axis_order": list(axes),
    }


def _primary_axis_index(axes: Sequence[str]) -> int:
    # A horizon product is already a stack of named 2-D surfaces.  Cutting it
    # along INLINE produces a misleading HORIZON x XLINE ribbon (for example
    # 4 x 541 pixels) instead of one complete map per interpreted surface.
    # Prefer the semantic HORIZON axis so the standard bundle contains the
    # complete INLINE x XLINE planes that judges and common GIS/NumPy tools
    # expect.  Ordinary seismic volumes continue to prefer INLINE sections.
    for index, axis in enumerate(axes):
        if _axis_kind(axis) == "horizon":
            return index
    for index, axis in enumerate(axes):
        if _axis_kind(axis) == "inline":
            return index
    return 0


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _fault_segy_reader_options(input_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Recover only producer-sealed SEG-Y parser choices from provenance."""

    geometry = input_metadata.get("geometry")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    sealed_options = geometry.get("segy_reader_options") or input_metadata.get(
        "segy_reader_options"
    )
    if isinstance(sealed_options, Mapping):
        allowed = {
            "profile",
            "inline_byte",
            "crossline_byte",
            "x_byte",
            "y_byte",
            "coordinate_scalar_byte",
        }
        options = {
            str(key): value
            for key, value in sealed_options.items()
            if str(key) in allowed and value not in (None, "")
        }
        if options:
            options.setdefault(
                "profile",
                str(
                    geometry.get("geometry_profile")
                    or input_metadata.get("geometry_profile")
                    or "standard_3d"
                ),
            )
            return options
    profile = str(
        geometry.get("geometry_profile")
        or input_metadata.get("geometry_profile")
        or ""
    ).strip()
    options: dict[str, Any] = {"profile": profile or "standard_3d"}
    issues = geometry.get("geometry_issues") or input_metadata.get("geometry_issues")
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return options
    prefixes = {
        "inline_byte=": "inline_byte",
        "crossline_byte=": "crossline_byte",
        "x_byte=": "x_byte",
        "y_byte=": "y_byte",
        "coordinate_scalar_byte=": "coordinate_scalar_byte",
    }
    for raw_issue in issues:
        issue = str(raw_issue)
        for prefix, option_name in prefixes.items():
            if not issue.startswith(prefix):
                continue
            try:
                options[option_name] = int(issue[len(prefix) :].split(":", 1)[0])
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"fault mask SEG-Y provenance has invalid {option_name}"
                ) from None
    return options


def _fault_segy_source_context(result: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect the bound source once for preview overlays and SEG-Y export."""

    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    source_raw = input_metadata.get("source")
    if not isinstance(source_raw, str) or not source_raw.strip():
        raise ValueError("producer result has no bound source SEG-Y path")
    source = Path(source_raw).expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() not in {".sgy", ".segy"}:
        raise ValueError("bound source SEG-Y is unavailable")
    config_path = Path(__file__).resolve().parents[2] / "configs" / "segy_profiles.yaml"
    config = load_yaml(config_path) if config_path.is_file() else {}
    reader = SegyReader(
        source,
        config,
        _fault_segy_reader_options(input_metadata),
    )
    geometry = reader.inspect()
    if geometry.inline is None or geometry.crossline is None:
        raise ValueError("source SEG-Y has no resolved Inline/Crossline geometry")
    inline = np.asarray(geometry.inline, dtype=np.int64)
    crossline = np.asarray(geometry.crossline, dtype=np.int64)
    if (
        inline.ndim != 1
        or crossline.ndim != 1
        or inline.size != crossline.size
    ):
        raise ValueError("source SEG-Y Inline/Crossline headers are incomplete")
    inline_values = np.unique(inline)
    crossline_values = np.unique(crossline)
    trace_lookup: dict[tuple[int, int], int] = {}
    duplicate_pairs: set[tuple[int, int]] = set()
    for trace_index, pair in enumerate(zip(inline, crossline, strict=True)):
        key = (int(pair[0]), int(pair[1]))
        if key in trace_lookup:
            duplicate_pairs.add(key)
        else:
            trace_lookup[key] = trace_index
    return {
        "source": source,
        "reader": reader,
        "geometry": geometry,
        "inline_values": inline_values,
        "crossline_values": crossline_values,
        "trace_lookup": trace_lookup,
        "duplicate_pairs": duplicate_pairs,
    }


def _fault_seismic_plane(
    context: Mapping[str, Any],
    *,
    axes: Sequence[str],
    roi_contract: Mapping[str, Any],
    axis_index: int,
    index: int,
    maximum_horizontal_sample_reads: int = 16_384,
) -> np.ndarray | None:
    """Read one source-backed seismic plane aligned to a local mask plane.

    Vertical Inline/Crossline sections require only one trace row/column.  A
    time/depth slice may require every trace in the ROI, so large horizontal
    planes intentionally fall back to mask-only rendering rather than turning
    a preview into a full-volume read.
    """

    axis_kinds = [_axis_kind(axis) for axis in axes]
    if sorted(axis_kinds) != ["crossline", "inline", "sample"]:
        return None
    if axis_index < 0 or axis_index >= 3:
        return None
    source_shape = tuple(int(value) for value in roi_contract["source_shape"])
    roi_start = tuple(int(value) for value in roi_contract["roi_start"])
    roi_shape = tuple(int(value) for value in roi_contract["roi_shape"])
    by_kind_start = dict(zip(axis_kinds, roi_start, strict=True))
    by_kind_size = dict(zip(axis_kinds, roi_shape, strict=True))
    source_by_kind = dict(zip(axis_kinds, source_shape, strict=True))
    inline_values = np.asarray(context["inline_values"], dtype=np.int64)
    crossline_values = np.asarray(context["crossline_values"], dtype=np.int64)
    geometry = context["geometry"]
    observed_shape = {
        "sample": int(geometry.samples_per_trace),
        "inline": int(inline_values.size),
        "crossline": int(crossline_values.size),
    }
    if observed_shape != source_by_kind:
        return None
    selected_inline = inline_values[
        by_kind_start["inline"] : by_kind_start["inline"]
        + by_kind_size["inline"]
    ]
    selected_crossline = crossline_values[
        by_kind_start["crossline"] : by_kind_start["crossline"]
        + by_kind_size["crossline"]
    ]
    if (
        selected_inline.size != by_kind_size["inline"]
        or selected_crossline.size != by_kind_size["crossline"]
    ):
        return None
    fixed_kind = axis_kinds[axis_index]
    if index < 0 or index >= by_kind_size[fixed_kind]:
        return None
    if (
        fixed_kind == "sample"
        and selected_inline.size * selected_crossline.size
        > int(maximum_horizontal_sample_reads)
    ):
        return None
    z_start = by_kind_start["sample"]
    z_size = by_kind_size["sample"]
    reader = context["reader"]
    trace_lookup = context["trace_lookup"]

    def read_trace(inline_value: int, crossline_value: int, sample_slice: slice) -> np.ndarray | None:
        trace_index = trace_lookup.get((int(inline_value), int(crossline_value)))
        if trace_index is None:
            return None
        values = np.asarray(reader.read_trace(trace_index, sample_slice), dtype=np.float32)
        return values if values.ndim == 1 else None

    if fixed_kind == "inline":
        inline_value = int(selected_inline[index])
        canonical = np.full((z_size, selected_crossline.size), np.nan, dtype=np.float32)
        sample_slice = slice(z_start, z_start + z_size)
        for crossline_offset, crossline_value in enumerate(selected_crossline):
            values = read_trace(inline_value, int(crossline_value), sample_slice)
            if values is not None and values.shape == (z_size,):
                canonical[:, crossline_offset] = values
        canonical_kinds = ["sample", "crossline"]
    elif fixed_kind == "crossline":
        crossline_value = int(selected_crossline[index])
        canonical = np.full((z_size, selected_inline.size), np.nan, dtype=np.float32)
        sample_slice = slice(z_start, z_start + z_size)
        for inline_offset, inline_value in enumerate(selected_inline):
            values = read_trace(int(inline_value), crossline_value, sample_slice)
            if values is not None and values.shape == (z_size,):
                canonical[:, inline_offset] = values
        canonical_kinds = ["sample", "inline"]
    else:
        sample_index = z_start + index
        canonical = np.full(
            (selected_inline.size, selected_crossline.size),
            np.nan,
            dtype=np.float32,
        )
        sample_slice = slice(sample_index, sample_index + 1)
        for inline_offset, inline_value in enumerate(selected_inline):
            for crossline_offset, crossline_value in enumerate(selected_crossline):
                values = read_trace(
                    int(inline_value), int(crossline_value), sample_slice
                )
                if values is not None and values.shape == (1,):
                    canonical[inline_offset, crossline_offset] = values[0]
        canonical_kinds = ["inline", "crossline"]
    plane_kinds = [kind for offset, kind in enumerate(axis_kinds) if offset != axis_index]
    if plane_kinds == canonical_kinds:
        return canonical
    if plane_kinds == canonical_kinds[::-1]:
        return canonical.T
    return None


def _write_geometry_bound_roi_segy(
    result: Mapping[str, Any],
    array: np.ndarray,
    *,
    axes: Sequence[str],
    roi_contract: Mapping[str, Any],
    destination: Path,
    output_key: str,
    categorical: bool,
    value_semantics: str | None = None,
    allowed_class_codes: Sequence[int] | None = None,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one geometry-bound numeric ROI by copying selected source headers.

    Samples are IEEE float32.  Categorical volumes retain exact integer class
    codes; continuous volumes retain their finite model values without
    normalization, clipping, or unit conversion.
    """

    context = (
        dict(source_context)
        if isinstance(source_context, Mapping)
        else _fault_segy_source_context(result)
    )
    source = Path(context["source"])
    axis_kinds = [_axis_kind(axis) for axis in axes]
    if sorted(axis_kinds) != ["crossline", "inline", "sample"]:
        raise ValueError("volume axes are not a unique Z/Inline/Crossline grid")
    permutation = [
        axis_kinds.index("sample"),
        axis_kinds.index("inline"),
        axis_kinds.index("crossline"),
    ]
    volume_zyx = np.transpose(array, permutation)
    if volume_zyx.ndim != 3 or not (
        np.issubdtype(volume_zyx.dtype, np.number)
        or np.issubdtype(volume_zyx.dtype, np.bool_)
    ):
        raise ValueError("SEG-Y export requires a numeric three-dimensional volume")
    resolved_allowed_codes = (
        tuple(int(item) for item in allowed_class_codes)
        if allowed_class_codes is not None
        else None
    )

    source_shape = tuple(int(value) for value in roi_contract["source_shape"])
    roi_start = tuple(int(value) for value in roi_contract["roi_start"])
    roi_shape = tuple(int(value) for value in roi_contract["roi_shape"])
    source_shape_zyx = tuple(source_shape[index] for index in permutation)
    roi_start_zyx = tuple(roi_start[index] for index in permutation)
    roi_shape_zyx = tuple(roi_shape[index] for index in permutation)
    if tuple(volume_zyx.shape) != roi_shape_zyx:
        raise ValueError("SEG-Y export array disagrees with the proven ROI")

    geometry = context["geometry"]
    inline_values = np.asarray(context["inline_values"], dtype=np.int64)
    crossline_values = np.asarray(context["crossline_values"], dtype=np.int64)
    observed_shape = (
        int(geometry.samples_per_trace),
        len(inline_values),
        len(crossline_values),
    )
    if observed_shape != source_shape_zyx:
        raise ValueError("source SEG-Y geometry disagrees with the producer source shape")
    z_start, inline_start, crossline_start = roi_start_zyx
    z_size, inline_size, crossline_size = roi_shape_zyx
    selected_inline = inline_values[inline_start : inline_start + inline_size]
    selected_crossline = crossline_values[
        crossline_start : crossline_start + crossline_size
    ]
    if len(selected_inline) != inline_size or len(selected_crossline) != crossline_size:
        raise ValueError("SEG-Y result ROI exceeds the source trace grid")

    trace_lookup = context["trace_lookup"]
    duplicate_pairs = context["duplicate_pairs"]
    selected_pairs = {
        (int(inline), int(crossline))
        for inline in selected_inline
        for crossline in selected_crossline
    }
    if duplicate_pairs & selected_pairs:
        raise ValueError("selected SEG-Y ROI contains duplicate Inline/Crossline bins")

    time_axis = np.asarray(geometry.time_axis, dtype=np.float64)
    selected_time = time_axis[z_start : z_start + z_size]
    if selected_time.shape != (z_size,) or not np.all(np.isfinite(selected_time)):
        raise ValueError("SEG-Y result ROI has no complete finite time axis")
    interval_ms = float(geometry.sample_interval)
    interval_us = int(round(interval_ms * 1000.0))
    if (
        not np.isfinite(interval_ms)
        or interval_us <= 0
        or interval_us > 65_535
        or not np.isclose(interval_us / 1000.0, interval_ms, atol=1e-9, rtol=0.0)
    ):
        raise ValueError("SEG-Y sample interval cannot be preserved exactly")
    delay_ms = int(round(float(selected_time[0])))
    if not np.isclose(delay_ms, selected_time[0], atol=1e-9, rtol=0.0):
        raise ValueError("SEG-Y ROI time origin is not representable in integer milliseconds")
    if not -32_768 <= delay_ms <= 32_767 or not 1 <= z_size <= 65_535:
        raise ValueError("SEG-Y ROI time origin or sample count exceeds Rev1 fields")

    byte_order = "big" if geometry.endian == ">" else "little"
    float_dtype = np.dtype(">f4" if geometry.endian == ">" else "<f4")
    first_trace_offset = int(np.asarray(geometry.trace_offsets)[0])
    with source.open("rb") as stream:
        prefix = bytearray(stream.read(first_trace_offset))
    if len(prefix) != first_trace_offset or first_trace_offset < 3600:
        raise ValueError("source SEG-Y textual/binary header block is incomplete")

    # Preserve the source textual header and replace only its final four cards
    # with an explicit result-product declaration.  This prevents downstream
    # software from mistaking a model result for the original amplitude cube
    # while the copied binary/trace headers retain the source geometry lineage.
    textual_header = bytes(prefix[:3200])
    ascii_printable = sum(
        byte in {9, 10, 13} or 32 <= byte <= 126 for byte in textual_header
    )
    text_encoding = "ascii" if ascii_printable >= 2400 else "cp500"
    resolved_value_semantics = str(
        value_semantics
        or ("integer_class_code" if categorical else "continuous_model_value")
    ).strip()
    textual_value_semantics = resolved_value_semantics.replace("_", " ").upper()
    cards = (
        "C37 GENERATED BY WELLFUSE LAYERPULSE; SOURCE GEOMETRY LINEAGE PRESERVED",
        f"C38 HEAD={output_key}; IEEE FLOAT32; {textual_value_semantics}; NO VALUE TRANSFORM",
        (
            "C39 ROI T/I/X START="
            f"{','.join(str(item) for item in roi_start)} SHAPE="
            f"{','.join(str(item) for item in roi_shape)}"
        ),
        (
            "C40 CLASS DEFINITIONS IN COMPANION CSV/MANIFEST"
            if categorical
            else "C40 VALUE SEMANTICS IN MANIFEST; DO NOT ASSUME PHYSICAL UNITS"
        ),
    )
    for index, card in enumerate(cards, start=36):
        encoded = card[:80].ljust(80).encode(text_encoding, errors="replace")
        prefix[index * 80 : (index + 1) * 80] = encoded

    def put_unsigned(buffer: bytearray, offset: int, value: int) -> None:
        buffer[offset : offset + 2] = int(value).to_bytes(
            2, byteorder=byte_order, signed=False
        )

    # SEG-Y binary header: interval, samples/trace, IEEE float format code.
    put_unsigned(prefix, 3200 + 16, interval_us)
    put_unsigned(prefix, 3200 + 20, z_size)
    put_unsigned(prefix, 3200 + 24, 5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".sgy.tmp.{os.getpid()}")
    written_traces = 0
    missing_cells = 0
    with source.open("rb") as source_stream, temporary.open("wb") as target:
        target.write(prefix)
        for inline_offset, inline in enumerate(selected_inline):
            for crossline_offset, crossline in enumerate(selected_crossline):
                trace_index = trace_lookup.get((int(inline), int(crossline)))
                if trace_index is None:
                    missing_cells += 1
                    continue
                source_stream.seek(int(geometry.trace_offsets[trace_index]))
                trace_header = bytearray(source_stream.read(240))
                if len(trace_header) != 240:
                    raise ValueError("source SEG-Y trace header is truncated")
                trace_header[108:110] = delay_ms.to_bytes(
                    2, byteorder=byte_order, signed=True
                )
                trace_header[114:116] = z_size.to_bytes(
                    2, byteorder=byte_order, signed=False
                )
                trace_header[116:118] = interval_us.to_bytes(
                    2, byteorder=byte_order, signed=False
                )
                target.write(trace_header)
                trace_values = np.asarray(
                    volume_zyx[:, inline_offset, crossline_offset]
                )
                if not np.all(np.isfinite(trace_values)):
                    raise ValueError("SEG-Y export volume contains non-finite values")
                if categorical and not np.allclose(
                    trace_values,
                    np.rint(trace_values),
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "categorical SEG-Y export contains non-integer class codes"
                    )
                if resolved_allowed_codes is not None and not np.all(
                    np.isin(trace_values, resolved_allowed_codes)
                ):
                    raise ValueError(
                        "categorical SEG-Y export contains an unsupported class code"
                    )
                trace = np.asarray(trace_values, dtype=float_dtype)
                target.write(trace.tobytes(order="C"))
                written_traces += 1
        target.flush()
        os.fsync(target.fileno())
    if written_traces <= 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("SEG-Y ROI has no source-backed traces")
    os.replace(temporary, destination)
    return {
        "status": "roi_volume_segy_available",
        "output_key": output_key,
        "filename": destination.name,
        "sha256": file_sha256(destination),
        "coverage": str(roi_contract["scope"]),
        "source_shape_zyx": list(source_shape_zyx),
        "roi_start_zyx": list(roi_start_zyx),
        "roi_shape_zyx": list(roi_shape_zyx),
        "axis_order": ["Z", "INLINE", "CROSSLINE"],
        "sample_interval_ms": interval_ms,
        "time_origin_ms": float(selected_time[0]),
        "trace_count": written_traces,
        "missing_grid_cell_count": missing_cells,
        "sample_format": "IEEE_FLOAT32",
        "sample_semantics": resolved_value_semantics,
        "value_transform": "none",
        "source_trace_headers_preserved": True,
        "truth_metrics_used": False,
    }


def _write_fault_mask_roi_segy(
    result: Mapping[str, Any],
    array: np.ndarray,
    *,
    axes: Sequence[str],
    roi_contract: Mapping[str, Any],
    destination: Path,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the existing binary fault-mask SEG-Y contract."""

    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("fault mask SEG-Y requires a binary 3-D mask")
    receipt = _write_geometry_bound_roi_segy(
        result,
        array,
        axes=axes,
        roi_contract=roi_contract,
        destination=destination,
        output_key="fault_mask_sgy",
        categorical=True,
        allowed_class_codes=(0, 1),
        source_context=source_context,
    )
    receipt["status"] = "roi_mask_segy_available"
    return receipt


def build_geometry_bound_segy_source_context(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the verified source-header context used by public ROI exports."""

    return _fault_segy_source_context(result)


def materialize_geometry_bound_volume_segy(
    result: Mapping[str, Any],
    array: np.ndarray,
    *,
    axes: Sequence[str],
    source_shape: Sequence[int],
    roi_start: Sequence[int],
    destination: Path,
    output_key: str,
    categorical: bool,
    value_semantics: str | None = None,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a common SEG-Y result for one proven three-dimensional ROI."""

    values = np.asarray(array)
    resolved_axes = [str(item) for item in axes]
    resolved_source_shape = tuple(int(item) for item in source_shape)
    resolved_roi_start = tuple(int(item) for item in roi_start)
    if (
        values.ndim != 3
        or len(resolved_axes) != 3
        or len(resolved_source_shape) != 3
        or len(resolved_roi_start) != 3
        or any(item <= 0 for item in resolved_source_shape)
        or any(item < 0 for item in resolved_roi_start)
    ):
        raise ValueError("SEG-Y export requires valid three-axis source and ROI geometry")
    roi_contract = {
        "scope": (
            "full_survey"
            if resolved_roi_start == (0, 0, 0)
            and tuple(values.shape) == resolved_source_shape
            else "declared_roi"
        ),
        "source_shape": list(resolved_source_shape),
        "roi_start": list(resolved_roi_start),
        "roi_shape": list(values.shape),
        "axis_order": resolved_axes,
    }
    return _write_geometry_bound_roi_segy(
        result,
        values,
        axes=resolved_axes,
        roi_contract=roi_contract,
        destination=destination,
        output_key=output_key,
        categorical=categorical,
        value_semantics=value_semantics,
        source_context=source_context,
    )


def _materialize_fault_mask_segy(
    result: Mapping[str, Any],
    array: np.ndarray,
    *,
    axes: Sequence[str],
    roi_contract: Mapping[str, Any],
    output_root: Path,
    execution_component: str,
    outputs: dict[str, Any],
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    integrity = result.get("output_integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    sealed_artifacts = integrity.get("artifacts")
    sealed_artifacts = (
        sealed_artifacts if isinstance(sealed_artifacts, Mapping) else {}
    )
    existing = _declared_path(outputs.get("fault_mask_sgy"))
    if "fault_mask_sgy" in sealed_artifacts:
        return {
            "status": "sealed_producer_mask_segy_preserved_not_reverified",
            "output_key": "fault_mask_sgy",
            "filename": existing.name if existing is not None else None,
            "standard_export_reverified": False,
            "public_download_ready": False,
            "reason": "sealed producer artifact is immutable; standard exporter did not overwrite it",
            "truth_metrics_used": False,
        }
    # Never publish an unverified producer path under the standard SEG-Y key.
    # A geometry-proven file is regenerated below and then replaces the key;
    # the producer's original file is not deleted.
    outputs.pop("fault_mask_sgy", None)
    destination = output_root / f"fault_mask_{execution_component}.sgy"
    try:
        receipt = _write_fault_mask_roi_segy(
            result,
            array,
            axes=axes,
            roi_contract=roi_contract,
            destination=destination,
            source_context=source_context,
        )
    except (OSError, TypeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        destination.with_suffix(f".sgy.tmp.{os.getpid()}").unlink(missing_ok=True)
        return {
            "status": "not_materialized_geometry_not_proven",
            "reason": str(exc),
            "fallback_outputs": [
                "mask_npy",
                "standard_slice_bundle_zip",
                "fault_mask_slice_summary_csv",
            ],
            "truth_metrics_used": False,
        }
    outputs["fault_mask_sgy"] = str(destination)
    return receipt


def _materialize_fault_mask_audit(
    array: np.ndarray,
    *,
    axes: Sequence[str],
    roi_contract: Mapping[str, Any],
    inference: Mapping[str, Any],
    source_output_key: str,
    source_sha256: str,
    output_root: Path,
    execution_component: str,
    outputs: dict[str, Any],
    segy_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an exact, bounded slice census for the authoritative raw mask.

    This deliberately does not run connected-component filtering or surface
    extraction.  Those operations require domain-dependent connectivity and
    size parameters.  The exported CSV is instead a complete per-axis census
    of the unmodified binary mask and is therefore reproducible without
    treating model probability or thresholded output as validation truth.
    """

    if array.ndim != 3 or len(axes) != 3:
        raise ValueError("fault mask audit requires a three-dimensional grid")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("fault mask audit requires an integer or boolean mask")
    shape = tuple(int(value) for value in array.shape)
    counts = [np.zeros(size, dtype=np.int64) for size in shape]
    plane_size = max(1, shape[1] * shape[2])
    chunk_size = max(1, min(shape[0], (16 * 1024 * 1024) // plane_size))
    foreground_count = 0
    for start in range(0, shape[0], chunk_size):
        stop = min(shape[0], start + chunk_size)
        block = np.asarray(array[start:stop])
        if not np.all(np.isin(block, (0, 1, False, True))):
            raise ValueError("fault mask audit found a non-binary voxel")
        selected = block != 0
        counts[0][start:stop] = selected.sum(axis=(1, 2), dtype=np.int64)
        counts[1] += selected.sum(axis=(0, 2), dtype=np.int64)
        counts[2] += selected.sum(axis=(0, 1), dtype=np.int64)
        foreground_count += int(selected.sum(dtype=np.int64))

    roi_start = tuple(int(value) for value in roi_contract.get("roi_start") or ())
    if len(roi_start) != 3:
        raise ValueError("fault mask audit requires an authoritative ROI origin")
    csv_path = output_root / f"fault_mask_slice_summary_{execution_component}.csv"
    csv_temporary = csv_path.with_suffix(f".csv.tmp.{os.getpid()}")
    with csv_temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "axis",
                "axis_index",
                "local_index",
                "global_index",
                "fault_voxel_count",
                "total_voxel_count",
                "fault_fraction",
            )
        )
        for axis_index, (axis, axis_counts) in enumerate(
            zip(axes, counts, strict=True)
        ):
            total_per_plane = int(np.prod(shape) // shape[axis_index])
            for local_index, count in enumerate(axis_counts):
                writer.writerow(
                    (
                        axis,
                        axis_index,
                        local_index,
                        roi_start[axis_index] + local_index,
                        int(count),
                        total_per_plane,
                        f"{int(count) / total_per_plane:.12g}",
                    )
                )
    os.replace(csv_temporary, csv_path)
    csv_sha256 = file_sha256(csv_path)

    raw_threshold = inference.get("threshold")
    try:
        producer_threshold = (
            float(raw_threshold) if raw_threshold is not None else None
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fault mask producer threshold is invalid") from exc
    if producer_threshold is not None and not (
        np.isfinite(producer_threshold) and 0.0 <= producer_threshold <= 1.0
    ):
        raise ValueError("fault mask producer threshold must stay within [0, 1]")
    total_voxels = int(np.prod(shape))
    audit: dict[str, Any] = {
        "contract_version": FAULT_MASK_AUDIT_CONTRACT_VERSION,
        "source_output_key": source_output_key,
        "source_sha256": source_sha256,
        "shape": list(shape),
        "dtype": str(array.dtype),
        "axis_order": list(axes),
        "roi_start": list(roi_start),
        "foreground_voxel_count": foreground_count,
        "total_voxel_count": total_voxels,
        "foreground_fraction": foreground_count / total_voxels,
        "nonempty_slice_count_by_axis": {
            str(axis): int(np.count_nonzero(axis_counts))
            for axis, axis_counts in zip(axes, counts, strict=True)
        },
        "producer_mask_threshold": producer_threshold,
        "threshold_applied_by_standard_export": False,
        "probability_used_by_standard_export": False,
        "truth_metrics_used": False,
        "connected_component_filter": {
            "applied": False,
            "reason": "no producer-declared connectivity or minimum-component-size contract",
        },
        "surface_extraction": {
            "applied": False,
            "reason": "raw voxel mask is preserved; no geological surface is invented",
        },
        "segy_export": dict(segy_export),
        "slice_summary": {
            "output_key": "fault_mask_slice_summary_csv",
            "filename": csv_path.name,
            "sha256": csv_sha256,
            "row_count": sum(shape),
        },
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    audit_path = output_root / f"fault_mask_audit_{execution_component}.json"
    audit_temporary = audit_path.with_suffix(f".json.tmp.{os.getpid()}")
    audit_temporary.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(audit_temporary, audit_path)
    outputs["fault_mask_slice_summary_csv"] = str(csv_path)
    outputs["fault_mask_audit_json"] = str(audit_path)
    return {
        **audit,
        "audit_output_key": "fault_mask_audit_json",
        "audit_file_sha256": file_sha256(audit_path),
    }


def materialize_legacy_bounded_spatial_slice_bundle(
    result: dict[str, Any], *, output_root: Path, execution_task_id: str
) -> dict[str, Any]:
    """Add a deterministic bounded 2-D slice ZIP when a safe array exists.

    The ZIP is explicitly a visualization/evaluation aid, never a replacement
    for the complete native array.  The full native output remains listed in
    the standard manifest and is the authoritative quantitative result.
    """

    contract_version = LEGACY_BOUNDED_SLICE_BUNDLE_CONTRACT_VERSION
    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("prediction outputs must be a mapping")
    selected = _primary_spatial_layer(result)
    if selected is None:
        receipt = {
            "contract_version": contract_version,
            "status": "not_applicable_non_array_or_non_spatial_result",
            "authoritative_output": "native_outputs",
        }
        result["standard_spatial_export"] = receipt
        return receipt
    source = Path(selected["path"])
    if source.suffix.casefold() in {".sgy", ".segy", ".xyz", ".vtk", ".vtp"}:
        receipt = {
            "contract_version": contract_version,
            "status": "native_standard_spatial_file_available",
            "authoritative_output_key": selected["output_key"],
            "authoritative_format": source.suffix.casefold().lstrip("."),
        }
        result["standard_spatial_export"] = receipt
        return receipt
    if source.suffix.casefold() != ".npy" or not source.is_file():
        receipt = {
            "contract_version": contract_version,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": "primary spatial output is not a directly sliceable NPY array",
        }
        result["standard_spatial_export"] = receipt
        return receipt
    try:
        array = np.load(source, mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError):
        receipt = {
            "contract_version": contract_version,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": "NPY could not be validated as a safe numeric array for bounded slices",
        }
        result["standard_spatial_export"] = receipt
        return receipt
    axes = [str(axis) for axis in selected.get("axis_order") or []]
    derived_argmax = False
    if array.ndim == 4 and len(axes) == 4 and axes[0].casefold() == "class":
        # Only a class-leading probability tensor is reduced.  This implements
        # the competition's multi-class argmax rule without inventing labels.
        if int(np.prod(array.shape[1:])) > 32 * 1024 * 1024:
            receipt = {
                "contract_version": contract_version,
                "status": "native_spatial_output_download_only",
                "authoritative_output_key": selected["output_key"],
                "reason": "class argmax preview would exceed the bounded memory policy",
            }
            result["standard_spatial_export"] = receipt
            return receipt
        array = np.argmax(array, axis=0)
        axes = axes[1:]
        derived_argmax = True
    if array.ndim != 3 or len(axes) != 3:
        receipt = {
            "contract_version": contract_version,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": "array dimensionality or declared axes do not prove a 3-D grid",
        }
        result["standard_spatial_export"] = receipt
        return receipt

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    execution_component = (
        "".join(
            character
            for character in str(execution_task_id)
            if character.isascii() and (character.isalnum() or character in "-_")
        )[:24]
        or "task"
    )
    destination = root / f"standard_2d_slices_{execution_component}.zip"
    temporary = destination.with_suffix(f".zip.tmp.{os.getpid()}")
    categorical = bool(
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
        or any(
            token in str(selected["output_key"]).casefold()
            for token in ("class", "label", "mask", "argmax")
        )
    )
    entries: list[tuple[str, bytes]] = []
    slices: list[dict[str, Any]] = []
    for axis_index, (axis_name, axis_size) in enumerate(
        zip(axes, array.shape, strict=True)
    ):
        for index in _even_indices(int(axis_size)):
            plane = np.asarray(np.take(array, index, axis=axis_index))
            stem = f"{axis_index:02d}_{axis_name}_{index:06d}"
            npy_name = f"slices/{stem}.npy"
            png_name = f"previews/{stem}.png"
            npy_payload = _npy_bytes(plane)
            png_payload = _png_bytes(plane, categorical=categorical)
            entries.extend(((npy_name, npy_payload), (png_name, png_payload)))
            slices.append(
                {
                    "axis": axis_name,
                    "axis_index": axis_index,
                    "index": index,
                    "plane_axes": [
                        axis for offset, axis in enumerate(axes) if offset != axis_index
                    ],
                    "shape": [int(value) for value in plane.shape],
                    "dtype": str(plane.dtype),
                    "npy": npy_name,
                    "npy_sha256": file_sha256_bytes(npy_payload),
                    "png": png_name,
                    "png_sha256": file_sha256_bytes(png_payload),
                }
            )
    source_sha256 = file_sha256(source)
    internal_manifest: dict[str, Any] = {
        "contract_version": contract_version,
        "model_id": str(result.get("model_id") or ""),
        "execution_task_id": execution_task_id,
        "interpretation_task_id": str(result.get("task_id") or ""),
        "source_snapshot_id": str(result.get("source_snapshot_id") or ""),
        "authoritative_output": {
            "output_key": selected["output_key"],
            "filename": source.name,
            "sha256": source_sha256,
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "axis_order": axes,
        },
        "roi": {
            "crop_start": (result.get("input") or {}).get("crop_start_zyx"),
            "crop_size": (result.get("input") or {}).get("crop_size_zyx"),
            "source_shape": (result.get("input") or {}).get("source_shape_zyx"),
        },
        "slice_policy": {
            "mode": "evenly_spaced_bounded",
            "maximum_slices_per_axis": 5,
            "is_complete_volume": False,
            "native_full_result_is_separately_downloadable": True,
        },
        "categorical": categorical,
        "derived_multiclass_argmax": derived_argmax,
        "slices": slices,
    }
    internal_manifest["manifest_sha256"] = canonical_sha256(internal_manifest)
    manifest_payload = (
        json.dumps(internal_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, payload in [("manifest.json", manifest_payload), *sorted(entries)]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    os.replace(temporary, destination)
    outputs["standard_slice_bundle_zip"] = str(destination)
    receipt = {
        "contract_version": contract_version,
        "status": "bounded_2d_slice_bundle_available",
        "migration_mode": "legacy_bounded",
        "coverage": "bounded_visualization_only",
        "authoritative_output_key": selected["output_key"],
        "authoritative_output_sha256": source_sha256,
        "axis_order": axes,
        "shape": [int(value) for value in array.shape],
        "derived_multiclass_argmax": derived_argmax,
        "slice_count": len(slices),
        "slice_bundle_output_key": "standard_slice_bundle_zip",
        "slice_bundle_sha256": file_sha256(destination),
        "is_complete_volume": False,
    }
    result["standard_spatial_export"] = receipt
    return receipt


def materialize_standard_spatial_slice_bundle(
    result: dict[str, Any], *, output_root: Path, execution_task_id: str
) -> dict[str, Any]:
    """Write a complete numeric primary-axis slice package for a proven ROI."""

    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("prediction outputs must be a mapping")
    selected = _primary_spatial_layer(result)
    if selected is None:
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "not_applicable_non_array_or_non_spatial_result",
            "authoritative_output": "native_outputs",
        }
        result["standard_spatial_export"] = receipt
        return receipt

    source = Path(selected["path"])
    suffix = source.suffix.casefold()
    if suffix in {".sgy", ".segy", ".xyz", ".vtk", ".vtp"}:
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "native_standard_spatial_file_available",
            "authoritative_output_key": selected["output_key"],
            "authoritative_format": suffix.lstrip("."),
        }
        result["standard_spatial_export"] = receipt
        return receipt
    if suffix not in {".npy", ".npz"} or not source.is_file():
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": "primary spatial output is not a declared NPY/NPZ numeric array",
        }
        result["standard_spatial_export"] = receipt
        return receipt

    try:
        if suffix == ".npy":
            array = np.load(source, mmap_mode="r", allow_pickle=False)
        else:
            array_key = str(selected.get("array_key") or "")
            if not array_key:
                raise ValueError("NPZ member is not declared by the producer contract")
            with np.load(source, allow_pickle=False) as archive:
                if array_key not in archive.files:
                    raise ValueError(f"declared NPZ member is missing: {array_key}")
                array = np.asarray(archive[array_key])
    except (EOFError, OSError, TypeError, ValueError) as exc:
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": f"numeric array could not be validated: {exc}",
        }
        result["standard_spatial_export"] = receipt
        return receipt

    axes = [str(axis) for axis in selected.get("axis_order") or []]
    derived_argmax = bool(
        array.ndim == 4 and len(axes) == 4 and _axis_kind(axes[0]) == "class"
    )
    if derived_argmax:
        axes = axes[1:]
        output_shape = tuple(int(value) for value in array.shape[1:])
    else:
        output_shape = tuple(int(value) for value in array.shape)
    if (
        len(output_shape) != 3
        or len(axes) != 3
        or any(size <= 0 for size in output_shape)
    ):
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": "array dimensionality or declared axes do not prove a 3-D grid",
        }
        result["standard_spatial_export"] = receipt
        return receipt
    try:
        roi_contract = _spatial_roi_contract(result, axes=axes, shape=output_shape)
    except ValueError as exc:
        receipt = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "status": "native_spatial_output_download_only",
            "authoritative_output_key": selected["output_key"],
            "reason": f"ROI scope could not be proven: {exc}",
        }
        result["standard_spatial_export"] = receipt
        return receipt

    inference = result.get("inference")
    inference = inference if isinstance(inference, Mapping) else {}
    raw_codes = inference.get("class_codes") or inference.get("active_codes")
    class_codes = (
        np.asarray([int(value) for value in raw_codes])
        if isinstance(raw_codes, Sequence) and not isinstance(raw_codes, (str, bytes))
        else None
    )
    if (
        derived_argmax
        and class_codes is not None
        and len(class_codes) != array.shape[0]
    ):
        raise ValueError("class code table length disagrees with CLASS axis")

    categorical = bool(
        derived_argmax
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
        or any(
            token in str(selected["output_key"]).casefold()
            for token in ("class", "label", "mask", "argmax")
        )
    )
    primary_axis_index = _primary_axis_index(axes)
    primary_axis = axes[primary_axis_index]
    primary_axis_size = output_shape[primary_axis_index]
    preview_indices = _even_indices(primary_axis_size, maximum=3)

    def plane_at(index: int) -> np.ndarray:
        if derived_argmax:
            # Take one spatial plane first, then reduce CLASS.  This keeps peak
            # memory bounded by one plane even for a full survey probability cube.
            class_plane = np.asarray(np.take(array, index, axis=primary_axis_index + 1))
            winners = np.argmax(class_plane, axis=0)
            return np.asarray(
                class_codes[winners] if class_codes is not None else winners
            )
        return np.asarray(np.take(array, index, axis=primary_axis_index))

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    execution_component = (
        "".join(
            character
            for character in str(execution_task_id)
            if character.isascii() and (character.isalnum() or character in "-_")
        )[:24]
        or "task"
    )
    destination = root / f"standard_2d_slices_{execution_component}.zip"
    temporary = destination.with_suffix(f".zip.tmp.{os.getpid()}")
    preview_destination = root / f"standard_preview_{execution_component}.png"
    preview_temporary = preview_destination.with_suffix(f".png.tmp.{os.getpid()}")

    representative_index = primary_axis_size // 2
    representative_plane = plane_at(representative_index)
    preview_payload = _png_bytes(representative_plane, categorical=categorical)
    is_fault_mask = (
        is_fault_volume_model_id(result.get("model_id"))
        and "mask" in str(selected["output_key"]).casefold()
        and not derived_argmax
    )
    fault_source_context: Mapping[str, Any] | None = None
    fault_source_context_reason: str | None = None
    seismic_preview_planes: dict[tuple[int, int], np.ndarray | None] = {}
    if is_fault_mask:
        try:
            fault_source_context = _fault_segy_source_context(result)
        except (OSError, TypeError, ValueError) as exc:
            fault_source_context_reason = str(exc)

    def fault_seismic_plane_at(
        requested_axis_index: int, requested_index: int
    ) -> np.ndarray | None:
        key = (int(requested_axis_index), int(requested_index))
        if key in seismic_preview_planes:
            return seismic_preview_planes[key]
        plane: np.ndarray | None = None
        if is_fault_mask and fault_source_context is not None:
            try:
                plane = _fault_seismic_plane(
                    fault_source_context,
                    axes=axes,
                    roi_contract=roi_contract,
                    axis_index=requested_axis_index,
                    index=requested_index,
                )
            except (IndexError, OSError, TypeError, ValueError):
                plane = None
            if plane is not None and not np.any(np.isfinite(plane)):
                plane = None
        seismic_preview_planes[key] = plane
        return plane

    standard_preview_rendering = (
        "categorical_or_continuous_result"
        if not is_fault_mask
        else "binary_mask_red_fill_yellow_boundary"
    )
    if is_fault_mask:
        representative_seismic = fault_seismic_plane_at(
            primary_axis_index, representative_index
        )
        if representative_seismic is not None:
            preview_payload = _fault_mask_seismic_overlay_png_bytes(
                representative_plane,
                representative_seismic,
            )
            standard_preview_rendering = (
                "source_seismic_gray_with_binary_mask_red_fill_yellow_boundary"
            )
        else:
            preview_payload = _fault_mask_png_bytes(representative_plane)
    preview_temporary.write_bytes(preview_payload)
    os.replace(preview_temporary, preview_destination)
    preview_sha256 = file_sha256_bytes(preview_payload)
    outputs["standard_preview_png"] = str(preview_destination)

    source_sha256 = file_sha256(source)
    fault_mask_audit: dict[str, Any] | None = None
    if is_fault_mask:
        fault_mask_segy = _materialize_fault_mask_segy(
            result,
            array,
            axes=axes,
            roi_contract=roi_contract,
            output_root=root,
            execution_component=execution_component,
            outputs=outputs,
            source_context=fault_source_context,
        )
        fault_mask_audit = _materialize_fault_mask_audit(
            array,
            axes=axes,
            roi_contract=roi_contract,
            inference=inference,
            source_output_key=str(selected["output_key"]),
            source_sha256=source_sha256,
            output_root=root,
            execution_component=execution_component,
            outputs=outputs,
            segy_export=fault_mask_segy,
        )
    slices: list[dict[str, Any]] = []
    orthogonal_previews: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for index in range(primary_axis_size):
            plane = plane_at(index)
            stem = f"{primary_axis_index:02d}_{primary_axis}_{index:06d}"
            npy_name = f"slices/{stem}.npy"
            npy_payload = _npy_bytes(plane)
            _zip_write_bytes(archive, npy_name, npy_payload)
            record: dict[str, Any] = {
                "axis": primary_axis,
                "axis_index": primary_axis_index,
                "index": index,
                "plane_axes": [
                    axis
                    for offset, axis in enumerate(axes)
                    if offset != primary_axis_index
                ],
                "shape": [int(value) for value in plane.shape],
                "dtype": str(plane.dtype),
                "npy": npy_name,
                "npy_sha256": file_sha256_bytes(npy_payload),
            }
            if index in preview_indices:
                png_name = f"previews/{stem}.png"
                preview_rendering = "categorical_or_continuous_result"
                if is_fault_mask:
                    seismic_plane = fault_seismic_plane_at(
                        primary_axis_index, index
                    )
                    if seismic_plane is not None:
                        png_payload = _fault_mask_seismic_overlay_png_bytes(
                            plane, seismic_plane
                        )
                        preview_rendering = (
                            "source_seismic_gray_with_binary_mask_red_fill_yellow_boundary"
                        )
                    else:
                        png_payload = _fault_mask_png_bytes(plane)
                        preview_rendering = (
                            "binary_mask_red_fill_yellow_boundary"
                        )
                else:
                    png_payload = _png_bytes(plane, categorical=categorical)
                _zip_write_bytes(archive, png_name, png_payload)
                record["png"] = png_name
                record["png_sha256"] = file_sha256_bytes(png_payload)
                record["rendering"] = preview_rendering
            slices.append(record)

        if is_fault_mask:
            for axis_index, (axis, axis_size) in enumerate(
                zip(axes, output_shape, strict=True)
            ):
                index = int(axis_size // 2)
                plane = np.asarray(np.take(array, index, axis=axis_index))
                stem = f"{axis_index:02d}_{axis}_{index:06d}"
                npy_name = f"orthogonal/{stem}.npy"
                png_name = f"orthogonal/{stem}.png"
                npy_payload = _npy_bytes(plane)
                seismic_plane = fault_seismic_plane_at(axis_index, index)
                if seismic_plane is not None:
                    png_payload = _fault_mask_seismic_overlay_png_bytes(
                        plane, seismic_plane
                    )
                    rendering = (
                        "source_seismic_gray_with_binary_mask_red_fill_yellow_boundary"
                    )
                else:
                    png_payload = _fault_mask_png_bytes(plane)
                    rendering = "binary_mask_red_fill_yellow_boundary"
                _zip_write_bytes(archive, npy_name, npy_payload)
                _zip_write_bytes(archive, png_name, png_payload)
                orthogonal_previews.append(
                    {
                        "axis": axis,
                        "axis_index": axis_index,
                        "index": index,
                        "global_index": int(roi_contract["roi_start"][axis_index])
                        + index,
                        "plane_axes": [
                            plane_axis
                            for offset, plane_axis in enumerate(axes)
                            if offset != axis_index
                        ],
                        "shape": [int(value) for value in plane.shape],
                        "npy": npy_name,
                        "npy_sha256": file_sha256_bytes(npy_payload),
                        "png": png_name,
                        "png_sha256": file_sha256_bytes(png_payload),
                        "rendering": rendering,
                        "quantitative_use": False,
                    }
                )

        internal_manifest: dict[str, Any] = {
            "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
            "model_id": str(result.get("model_id") or ""),
            "execution_task_id": execution_task_id,
            "interpretation_task_id": str(result.get("task_id") or ""),
            "source_snapshot_id": str(result.get("source_snapshot_id") or ""),
            "authoritative_output": {
                "output_key": selected["output_key"],
                "filename": source.name,
                "sha256": source_sha256,
                "native_shape": [int(value) for value in array.shape],
                "delivered_shape": list(output_shape),
                "delivered_dtype": str(representative_plane.dtype),
                "axis_order": axes,
            },
            "coverage": roi_contract,
            "reconstruction": {
                "algorithm": "load slices in ascending index order and stack along slice_axis_index",
                "axis_order": axes,
                "shape": list(output_shape),
                "dtype": str(representative_plane.dtype),
                "slice_axis": primary_axis,
                "slice_axis_index": primary_axis_index,
                "slice_count": primary_axis_size,
                "plane_axes": [
                    axis
                    for offset, axis in enumerate(axes)
                    if offset != primary_axis_index
                ],
                "numeric_format": "NumPy NPY readable with allow_pickle=False",
            },
            "slice_policy": {
                "mode": "complete_primary_axis",
                "coverage": "complete_for_declared_roi",
                "slice_count": primary_axis_size,
                "is_complete_for_declared_roi": True,
                "is_full_survey": bool(roi_contract["full_survey"]),
                "native_result_is_separately_downloadable": True,
            },
            "standard_preview": {
                "output_key": "standard_preview_png",
                "filename": preview_destination.name,
                "sha256": preview_sha256,
                "representative_slice_axis": primary_axis,
                "representative_slice_index": representative_index,
                "quantitative_use": False,
                "rendering": standard_preview_rendering,
                "source_seismic_context": (
                    "available_and_aligned"
                    if standard_preview_rendering.startswith("source_seismic_gray")
                    else "unavailable_mask_only_fallback"
                ),
            },
            "categorical": categorical,
            "derived_multiclass_argmax": derived_argmax,
            "class_codes": class_codes.tolist() if class_codes is not None else None,
            "orthogonal_previews": orthogonal_previews,
            "fault_mask_audit": fault_mask_audit,
            "fault_preview_context": (
                {
                    "status": (
                        "source_seismic_context_available"
                        if fault_source_context is not None
                        else "mask_only_fallback"
                    ),
                    "fallback_reason": fault_source_context_reason,
                    "model_mask_modified": False,
                    "producer_threshold_modified": False,
                }
                if is_fault_mask
                else None
            ),
            "slices": slices,
        }
        internal_manifest["manifest_sha256"] = canonical_sha256(internal_manifest)
        manifest_payload = (
            json.dumps(internal_manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        _zip_write_bytes(archive, "manifest.json", manifest_payload)

    os.replace(temporary, destination)
    outputs["standard_slice_bundle_zip"] = str(destination)
    receipt = {
        "contract_version": STANDARD_SLICE_BUNDLE_CONTRACT_VERSION,
        "status": "complete_2d_slice_bundle_available",
        "authoritative_output_key": selected["output_key"],
        "authoritative_output_sha256": source_sha256,
        "axis_order": axes,
        "shape": list(output_shape),
        "dtype": str(representative_plane.dtype),
        "coverage": "complete_for_declared_roi",
        "scope": roi_contract["scope"],
        "roi": roi_contract,
        "derived_multiclass_argmax": derived_argmax,
        "slice_axis": primary_axis,
        "slice_axis_index": primary_axis_index,
        "slice_count": primary_axis_size,
        "slice_bundle_output_key": "standard_slice_bundle_zip",
        "slice_bundle_sha256": file_sha256(destination),
        "standard_preview_output_key": "standard_preview_png",
        "standard_preview_sha256": preview_sha256,
        "standard_preview_rendering": standard_preview_rendering,
        "standard_preview_source_seismic_context": (
            standard_preview_rendering.startswith("source_seismic_gray")
        ),
        "is_complete_for_declared_roi": True,
        "is_full_survey": bool(roi_contract["full_survey"]),
        "fault_mask_audit": fault_mask_audit,
    }
    result["standard_spatial_export"] = receipt
    return receipt


def recover_standard_preview_from_slice_bundle(
    result: dict[str, Any], *, output_root: Path, execution_task_id: str
) -> Path | None:
    """Recover the exact sealed preview bytes after a pre-seal interruption.

    Complete slice bundles carry the representative PNG and its SHA-256.  This
    recovery never recomputes scientific values: it extracts only those exact
    bytes, validates every binding, and lets the caller add the file to the
    completion integrity manifest.
    """

    outputs = result.get("outputs")
    receipt = result.get("standard_spatial_export")
    if not isinstance(outputs, dict) or not isinstance(receipt, Mapping):
        return None
    if receipt.get("standard_preview_output_key") != "standard_preview_png":
        return None
    slice_path = _declared_path(outputs.get("standard_slice_bundle_zip"))
    if slice_path is None or not slice_path.is_file():
        raise ValueError("sealed standard slice bundle is missing")
    try:
        with zipfile.ZipFile(slice_path) as archive:
            manifest_info = archive.getinfo("manifest.json")
            if manifest_info.file_size > 8 * 1024 * 1024:
                raise ValueError("sealed slice manifest exceeds the recovery limit")
            manifest = json.loads(archive.read(manifest_info))
            if not isinstance(manifest, Mapping):
                raise ValueError("sealed slice manifest is not an object")
            if (
                manifest.get("contract_version")
                != STANDARD_SLICE_BUNDLE_CONTRACT_VERSION
                or str(manifest.get("execution_task_id") or "") != execution_task_id
            ):
                raise ValueError("sealed slice manifest binding is invalid")
            preview = manifest.get("standard_preview")
            if not isinstance(preview, Mapping):
                raise ValueError("sealed slice manifest has no preview binding")
            filename = str(preview.get("filename") or "")
            if not filename or Path(filename).name != filename:
                raise ValueError("sealed slice preview filename is unsafe")
            expected_sha256 = str(preview.get("sha256") or "").casefold()
            if (
                expected_sha256
                != str(receipt.get("standard_preview_sha256") or "").casefold()
            ):
                raise ValueError("sealed slice preview receipt hash disagrees")
            axis = str(preview.get("representative_slice_axis") or "")
            index = int(preview.get("representative_slice_index"))
            slices = manifest.get("slices")
            if not isinstance(slices, Sequence) or isinstance(slices, (str, bytes)):
                raise ValueError("sealed slice manifest has no slice records")
            record = next(
                (
                    item
                    for item in slices
                    if isinstance(item, Mapping)
                    and str(item.get("axis") or "") == axis
                    and int(item.get("index", -1)) == index
                    and item.get("png")
                ),
                None,
            )
            if not isinstance(record, Mapping):
                raise ValueError("sealed slice bundle lacks its representative PNG")
            member = PurePosixPath(str(record.get("png") or ""))
            if (
                member.is_absolute()
                or not member.parts
                or any(part in {"", ".", ".."} for part in member.parts)
            ):
                raise ValueError("sealed slice preview member is unsafe")
            member_info = archive.getinfo(member.as_posix())
            if member_info.file_size > 32 * 1024 * 1024:
                raise ValueError("sealed slice preview exceeds the recovery limit")
            payload = archive.read(member_info)
    except (
        KeyError,
        OSError,
        TypeError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("sealed slice preview could not be recovered") from exc
    observed_sha256 = file_sha256_bytes(payload)
    if (
        observed_sha256 != expected_sha256
        or observed_sha256 != str(record.get("png_sha256") or "").casefold()
    ):
        raise ValueError("sealed slice preview content hash disagrees")
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    outputs["standard_preview_png"] = str(destination)
    return destination


def file_sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _checkpoint_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(*, sha256: object, stage: object = None, path: object = None) -> None:
        digest = str(sha256 or "").strip().casefold()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            return
        record: dict[str, Any] = {"sha256": digest}
        if stage not in (None, ""):
            record["stage"] = str(stage)
        declared = _declared_path(path)
        if declared is not None:
            record["filename"] = declared.name
        existing = next(
            (item for item in records if item.get("sha256") == digest), None
        )
        if existing is None:
            records.append(record)
        else:
            for key, value in record.items():
                if key != "sha256" and key not in existing:
                    existing[key] = value

    def add_path(
        raw: object, *, stage: object = None, explicit_sha256: object = None
    ) -> None:
        path = _declared_path(raw)
        if path is None and isinstance(raw, Mapping):
            path = _declared_path(raw.get("checkpoint_path") or raw.get("checkpoint"))
        declared_digest = str(explicit_sha256 or "").strip().casefold()
        if isinstance(raw, Mapping) and not declared_digest:
            declared_digest = str(raw.get("sha256") or "").strip().casefold()
        if declared_digest and (
            len(declared_digest) != 64
            or any(character not in "0123456789abcdef" for character in declared_digest)
        ):
            raise ValueError("producer checkpoint SHA-256 is invalid")
        if path is not None and path.is_file():
            observed = file_sha256(path).casefold()
            if declared_digest and declared_digest != observed:
                raise ValueError(f"producer checkpoint SHA-256 disagrees: {path.name}")
            add(sha256=declared_digest or observed, stage=stage, path=str(path))
        elif declared_digest:
            add(
                sha256=declared_digest,
                stage=stage,
                path=str(path) if path is not None else None,
            )

    root_checkpoint = result.get("checkpoint")
    if not (
        isinstance(root_checkpoint, Sequence)
        and not isinstance(root_checkpoint, (str, bytes))
    ):
        add_path(
            root_checkpoint,
            explicit_sha256=result.get("checkpoint_sha256"),
        )
    elif result.get("checkpoint_sha256"):
        add(sha256=result.get("checkpoint_sha256"))
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    add_path(
        provenance.get("checkpoint_path") or provenance.get("checkpoint"),
        explicit_sha256=provenance.get("checkpoint_sha256"),
    )
    for key in ("checkpoint_evidence", "checkpoints"):
        raw_records = result.get(key)
        if isinstance(raw_records, Sequence) and not isinstance(
            raw_records, (str, bytes)
        ):
            for raw in raw_records:
                if isinstance(raw, Mapping):
                    add_path(
                        raw,
                        stage=raw.get("stage"),
                        explicit_sha256=raw.get("sha256"),
                    )
                else:
                    add_path(raw, stage=key)

    checkpoint_selection = result.get("checkpoint_selection")
    checkpoint_selection = (
        checkpoint_selection if isinstance(checkpoint_selection, Mapping) else {}
    )
    selection_stage = str(
        checkpoint_selection.get("selection") or "checkpoint_selection"
    )
    selected_paths = checkpoint_selection.get("checkpoints")
    if isinstance(selected_paths, Sequence) and not isinstance(
        selected_paths, (str, bytes)
    ):
        explicit_digests = checkpoint_selection.get("checkpoint_sha256s")
        explicit_digests = (
            list(explicit_digests)
            if isinstance(explicit_digests, Sequence)
            and not isinstance(explicit_digests, (str, bytes))
            else []
        )
        for index, raw in enumerate(selected_paths):
            add_path(
                raw,
                stage=f"{selection_stage}:{index}",
                explicit_sha256=(
                    explicit_digests[index] if index < len(explicit_digests) else None
                ),
            )
    if checkpoint_selection.get("checkpoint") is not None:
        add_path(
            checkpoint_selection.get("checkpoint"),
            stage=selection_stage,
            explicit_sha256=checkpoint_selection.get("checkpoint_sha256"),
        )

    declared_checkpoints = result.get("checkpoint")
    paths: list[object]
    if isinstance(declared_checkpoints, Sequence) and not isinstance(
        declared_checkpoints, (str, bytes)
    ):
        paths = list(declared_checkpoints)
    else:
        paths = [declared_checkpoints]
    for raw_path in paths:
        add_path(raw_path)
    return records


def _decision_contract(
    result: Mapping[str, Any], inference: Mapping[str, Any]
) -> dict[str, Any]:
    output_contract = result.get("output_contract")
    output_contract = (
        output_contract if isinstance(output_contract, Mapping) else None
    )
    task_id = str(result.get("task_id") or "").strip().casefold()
    regression_markers = (
        output_contract.get("decision_type") if output_contract is not None else None,
        output_contract.get("primary_semantics")
        if output_contract is not None
        else None,
        output_contract.get("primary_decision_rule")
        if output_contract is not None
        else None,
    )
    output_contract_declares_regression = any(
        isinstance(marker, str)
        and (
            marker.strip().casefold() == "regression"
            or marker.strip().casefold().endswith("_regression")
        )
        for marker in regression_markers
    )
    if task_id == "well_property" or output_contract_declares_regression:
        primary_output = result.get("primary_output")
        declared_primary_output = (
            output_contract.get("primary_output")
            if output_contract is not None
            else None
        )
        incomplete_reasons: list[str] = []
        if not isinstance(primary_output, str) or not primary_output.strip():
            incomplete_reasons.append("producer supplied no primary_output")
        if output_contract is None:
            incomplete_reasons.append("producer supplied no output_contract mapping")
        else:
            if not isinstance(output_contract.get("contract_version"), str) or not str(
                output_contract.get("contract_version")
            ).strip():
                incomplete_reasons.append("output_contract has no contract_version")
            if (
                not isinstance(declared_primary_output, str)
                or not declared_primary_output.strip()
            ):
                incomplete_reasons.append("output_contract has no primary_output")
            elif (
                isinstance(primary_output, str)
                and primary_output.strip()
                and declared_primary_output != primary_output
            ):
                incomplete_reasons.append(
                    "output_contract primary_output disagrees with producer primary_output"
                )
            if not output_contract_declares_regression:
                incomplete_reasons.append(
                    "output_contract does not declare regression primary semantics"
                )
        formal_scoring_ready = not incomplete_reasons
        formal_scoring_reason = (
            None if formal_scoring_ready else "; ".join(incomplete_reasons)
        )
        return {
            "decision_type": "regression",
            "threshold": None,
            "query_threshold": None,
            "multi_class_rule": None,
            "class_codes": [],
            "class_table": [],
            "background": {
                "code": None,
                "status": "not_applicable_regression",
                "exclude_from_scoring_when_required": False,
            },
            "background_policy": "not_applicable_regression",
            "formal_scoring_ready": formal_scoring_ready,
            "formal_scoring_reason": formal_scoring_reason,
        }

    model_id = str(result.get("model_id") or "").casefold()
    raw_codes = inference.get("class_codes") or inference.get("active_codes")
    codes = (
        [int(value) for value in raw_codes]
        if isinstance(raw_codes, Sequence) and not isinstance(raw_codes, (str, bytes))
        else []
    )
    raw_classes = result.get("classes")
    if raw_classes is None:
        raw_classes = inference.get("classes")
    producer_declared_class_table = bool(
        isinstance(raw_classes, Sequence)
        and not isinstance(raw_classes, (str, bytes))
        and raw_classes
    )
    class_table: list[dict[str, Any]] = []
    if isinstance(raw_classes, Sequence) and not isinstance(raw_classes, (str, bytes)):
        for index, raw_class in enumerate(raw_classes):
            if isinstance(raw_class, Mapping):
                try:
                    code = int(raw_class.get("code", index))
                except (TypeError, ValueError) as exc:
                    raise ValueError("producer class code must be an integer") from exc
                name = str(raw_class.get("name") or f"producer_class_{code}")
                role = str(
                    raw_class.get("role")
                    or (
                        "background"
                        if name.strip().casefold() == "background"
                        else "class"
                    )
                )
            else:
                code = index
                name = str(raw_class)
                role = (
                    "background"
                    if name.strip().casefold() == "background"
                    else "class"
                )
            class_table.append({"code": code, "name": name, "role": role})
        table_codes = [int(item["code"]) for item in class_table]
        if len(table_codes) != len(set(table_codes)):
            raise ValueError("producer class codes must be unique")
        if codes and codes != table_codes:
            raise ValueError(
                "producer class table disagrees with inference class codes"
            )
        codes = table_codes
    segmentation = result.get("segmentation")
    segmentation = segmentation if isinstance(segmentation, Mapping) else {}
    if not codes and model_id == "seismic_surface_seg":
        raw_range = segmentation.get("label_range")
        if (
            isinstance(raw_range, Sequence)
            and not isinstance(raw_range, (str, bytes))
            and len(raw_range) == 2
        ):
            lower, upper = (int(raw_range[0]), int(raw_range[1]))
            if 0 <= lower <= upper and upper - lower <= 511:
                codes = list(range(lower, upper + 1))
    binary_target = {
        "faultseg_3d": "fault",
        "faultnet_china_field": "fault",
        "wellfuse_channel_p17": "channel",
        "wellfuse_karst_p17": "karst",
    }.get(model_id)
    if not codes and binary_target is not None:
        codes = [0, 1]
    binary_contract_applied = bool(
        not class_table and binary_target is not None and codes == [0, 1]
    )
    if binary_contract_applied:
        class_table = [
            {"code": 0, "name": "background", "role": "background"},
            {"code": 1, "name": binary_target, "role": "target"},
        ]
    elif not class_table:
        class_table = [
            {"code": code, "name": f"producer_class_{code}", "role": "class"}
            for code in codes
        ]
    declared_backgrounds = [
        item
        for item in class_table
        if str(item.get("role") or "").casefold() == "background"
    ]
    if len(declared_backgrounds) > 1:
        raise ValueError("producer class table declares multiple background classes")
    if declared_backgrounds:
        background_code = int(declared_backgrounds[0]["code"])
        background = {
            "code": background_code,
            "status": (
                "declared_binary_task_semantics"
                if binary_contract_applied
                else "producer_declared_class_role"
            ),
            "exclude_from_scoring_when_required": True,
        }
        formal_scoring_ready = True
        formal_scoring_reason = None
        background_policy = (
            "registered_binary_task_contract"
            if binary_contract_applied
            else "producer_declared_background_class"
        )
    elif producer_declared_class_table and class_table:
        # ``classes`` is the producer's exhaustive argmax table.  When that
        # table contains no background role, no class may be silently removed
        # from scoring merely because it happens to use code zero.
        background = {
            "code": None,
            "status": "no_background_class",
            "exclude_from_scoring_when_required": False,
        }
        formal_scoring_ready = True
        formal_scoring_reason = None
        background_policy = "producer_declared_complete_class_table_no_background"
    else:
        background = {
            "code": None,
            "status": "unresolved_producer_contract",
            "exclude_from_scoring_when_required": None,
            "reason": (
                "producer supplied class codes without a complete class ontology "
                "or background declaration"
                if codes
                else "producer supplied no class ontology or background declaration"
            ),
        }
        formal_scoring_ready = False
        formal_scoring_reason = str(background["reason"])
        background_policy = "producer_contract_incomplete_scoring_blocked"
    if model_id == "seismic_surface_seg":
        background = {
            "code": None,
            "invalid_code": segmentation.get("invalid_label", -1),
            "status": "no_background_class",
            "exclude_from_scoring_when_required": False,
        }
        formal_scoring_ready = True
        formal_scoring_reason = None
        background_policy = "ordered_surface_instances_no_background_class"
    threshold = (
        inference["threshold"]
        if inference.get("threshold") is not None
        else inference.get("mask_threshold")
    )
    return {
        "decision_type": "classification",
        "threshold": threshold,
        "query_threshold": inference.get("query_threshold"),
        "multi_class_rule": "argmax" if len(class_table) > 2 else None,
        "class_codes": codes,
        "class_table": class_table,
        "background": background,
        "background_policy": background_policy,
        "formal_scoring_ready": formal_scoring_ready,
        "formal_scoring_reason": formal_scoring_reason,
    }


def _manifest_grid_contract(
    result: Mapping[str, Any], input_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the public grid to the proven spatial export when one exists."""

    grid = {
        "axis_order": input_metadata.get("axes") or input_metadata.get("model_order"),
        "roi_start": input_metadata.get("crop_start_zyx")
        or input_metadata.get("resolved_roi_start_zyx"),
        "roi_size": input_metadata.get("crop_size_zyx")
        or input_metadata.get("resolved_roi_size_zyx"),
        "source_shape": input_metadata.get("source_shape_zyx")
        or input_metadata.get("shape_zyx"),
        "coordinate_reference": input_metadata.get("coordinate_reference")
        or input_metadata.get("crs"),
    }
    spatial_export = result.get("standard_spatial_export")
    if not isinstance(spatial_export, Mapping) or not bool(
        spatial_export.get("is_complete_for_declared_roi")
    ):
        return grid

    raw_axes = spatial_export.get("axis_order")
    roi = spatial_export.get("roi")
    if (
        not isinstance(raw_axes, Sequence)
        or isinstance(raw_axes, (str, bytes))
        or len(raw_axes) != 3
        or not isinstance(roi, Mapping)
    ):
        raise ValueError("complete standard spatial export has no authoritative grid")
    axes = [str(axis) for axis in raw_axes]
    roi_axes = roi.get("axis_order")
    if (
        not isinstance(roi_axes, Sequence)
        or isinstance(roi_axes, (str, bytes))
        or [str(axis) for axis in roi_axes] != axes
    ):
        raise ValueError("standard spatial export axes disagree with its ROI")
    roi_start = _shape3(roi.get("roi_start"))
    roi_shape = _shape3(roi.get("roi_shape"))
    source_shape = _shape3(roi.get("source_shape"))
    delivered_shape = _shape3(spatial_export.get("shape"))
    if (
        roi_start is None
        or roi_shape is None
        or source_shape is None
        or delivered_shape != roi_shape
    ):
        raise ValueError("complete standard spatial export has inconsistent ROI shapes")
    grid.update(
        {
            "axis_order": axes,
            "roi_start": list(roi_start),
            "roi_size": list(roi_shape),
            "source_shape": list(source_shape),
        }
    )
    return grid


def write_standard_result_manifest(
    result: dict[str, Any],
    *,
    output_root: Path,
    native_output_integrity: Mapping[str, Any],
    execution_task_id: str,
) -> Path:
    """Write the judge-facing manifest from the already hashed native outputs."""

    integrity_artifacts = native_output_integrity.get("artifacts")
    if not isinstance(integrity_artifacts, Mapping):
        raise ValueError("native output integrity has no artifacts")
    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("prediction outputs must be a mapping")
    public_artifacts: dict[str, Any] = {}
    for key, raw in integrity_artifacts.items():
        if not isinstance(raw, Mapping):
            continue
        record = {
            name: raw.get(name)
            for name in ("kind", "size", "file_count", "sha256", "children")
            if raw.get(name) is not None
        }
        public_artifacts[str(key)] = record
    inference = result.get("inference")
    inference = inference if isinstance(inference, Mapping) else {}
    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_identity = provenance.get("prediction_source_identity")
    source_identity = source_identity if isinstance(source_identity, Mapping) else {}
    checkpoints = _checkpoint_records(result)
    primary_output = result.get("primary_output")
    diagnostic_output = result.get("diagnostic_output")
    output_contract = result.get("output_contract")
    if primary_output is not None and not isinstance(primary_output, str):
        raise ValueError("prediction primary_output must be a string or null")
    if diagnostic_output is not None and not isinstance(diagnostic_output, str):
        raise ValueError("prediction diagnostic_output must be a string or null")
    if output_contract is not None and not isinstance(output_contract, Mapping):
        raise ValueError("prediction output_contract must be a mapping or null")
    manifest: dict[str, Any] = {
        "contract_version": STANDARD_EXPORT_MANIFEST_CONTRACT_VERSION,
        "model": {
            "model_id": str(result.get("model_id") or ""),
            "model_name": str(result.get("model_name") or ""),
            "checkpoints": checkpoints,
        },
        "task": {
            "execution_task_id": execution_task_id,
            "interpretation_task_id": str(result.get("task_id") or ""),
            "source_snapshot_id": str(result.get("source_snapshot_id") or ""),
            "source_snapshot_sha256": provenance.get("source_snapshot_fingerprint"),
        },
        "source": {
            "kind": source_identity.get("kind"),
            "filename": (
                _declared_path(input_metadata.get("source")).name
                if _declared_path(input_metadata.get("source")) is not None
                else None
            ),
            "sha256": source_identity.get("sha256"),
            "size": source_identity.get("size"),
            "geometry_fingerprint": source_identity.get("geometry_fingerprint"),
        },
        "grid": _manifest_grid_contract(result, input_metadata),
        "decision_rule": _decision_contract(result, inference),
        "primary_output": primary_output,
        "diagnostic_output": diagnostic_output,
        "output_contract": output_contract,
        "standard_spatial_export": result.get("standard_spatial_export"),
        "native_output_integrity": {
            "contract_version": native_output_integrity.get("contract_version"),
            "sha256": native_output_integrity.get("integrity_sha256"),
            "artifacts": public_artifacts,
        },
        "scientific_status": result.get("scientific_status") or "candidate",
        "warnings": (
            [str(item) for item in result.get("warnings")]
            if isinstance(result.get("warnings"), Sequence)
            and not isinstance(result.get("warnings"), (str, bytes))
            else ([str(result.get("warnings"))] if result.get("warnings") else [])
        ),
    }
    if "physical_bounds_audit" in result:
        physical_bounds_audit = result.get("physical_bounds_audit")
        if not isinstance(physical_bounds_audit, Mapping):
            raise ValueError("prediction physical_bounds_audit must be a mapping")
        manifest["physical_bounds_audit"] = physical_bounds_audit
    manifest = _trusted_json_copy(
        manifest, description="standard result judge manifest"
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    serialized_manifest = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    execution_component = (
        "".join(
            character
            for character in str(execution_task_id)
            if character.isascii() and (character.isalnum() or character in "-_")
        )[:24]
        or "task"
    )
    path = root / f"standard_result_manifest_{execution_component}.json"
    temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(serialized_manifest, encoding="utf-8")
    os.replace(temporary, path)
    outputs["standard_result_manifest_json"] = str(path)
    return path


def append_output_file_integrity(
    native_output_integrity: Mapping[str, Any],
    *,
    output_key: str,
    output_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal one new output without re-hashing already sealed large arrays."""

    root = output_root.expanduser().resolve()
    path = output_path.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "new standard result output escapes its task directory"
        ) from exc
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (
        before.st_size <= 0
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("new standard result output changed while sealing")
    document = json.loads(json.dumps(dict(native_output_integrity)))
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("native output integrity has no mutable artifacts mapping")
    artifacts[str(output_key)] = {
        "kind": "file",
        "path": str(path),
        "size": after.st_size,
        "sha256": digest,
    }
    document.pop("integrity_sha256", None)
    document["integrity_sha256"] = canonical_sha256(document)
    return document


def append_standard_manifest_integrity(
    native_output_integrity: Mapping[str, Any],
    *,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Extend the sealed native manifest without re-hashing large arrays."""

    return append_output_file_integrity(
        native_output_integrity,
        output_key="standard_result_manifest_json",
        output_path=manifest_path,
        output_root=output_root,
    )


__all__ = [
    "LEGACY_BOUNDED_SLICE_BUNDLE_CONTRACT_VERSION",
    "STANDARD_EXPORT_MANIFEST_CONTRACT_VERSION",
    "STANDARD_SLICE_BUNDLE_CONTRACT_VERSION",
    "append_output_file_integrity",
    "append_standard_manifest_integrity",
    "build_geometry_bound_segy_source_context",
    "materialize_legacy_bounded_spatial_slice_bundle",
    "materialize_geometry_bound_volume_segy",
    "materialize_standard_spatial_slice_bundle",
    "recover_standard_preview_from_slice_bundle",
    "write_standard_result_manifest",
]
