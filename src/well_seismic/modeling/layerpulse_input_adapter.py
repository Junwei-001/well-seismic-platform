"""SEG-Y preview-patch adapter for the LayerPulse platform runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..io.segy import SegyReader
from ..layerpulse_contract import (
    LAYERPULSE_MODEL_ID,
    validate_layerpulse_request_options,
)
from .input_adapters import (
    GeobodySegyInputAdapter,
    ModelInputBatch,
    ModelInputRequest,
)

_SEGY_SUFFIXES = frozenset({".sgy", ".segy"})
_DEFAULT_PATCH_SIZE_TIX = (128, 128, 128)
_SEGY_READER_PROVENANCE_KEYS = (
    "profile",
    "inline_byte",
    "crossline_byte",
    "x_byte",
    "y_byte",
    "coordinate_scalar_byte",
)
_WELL_LOG_ROLES = frozenset({"well_log", "well_logs"})
_WELL_METADATA_ROLES = frozenset({"well_metadata", "well_heads", "well_head"})
_FORBIDDEN_ASSET_MARKERS = (
    "time_depth",
    "time-depth",
    "timedepth",
    "checkshot",
    "check-shot",
    "vsp",
    "velocity_model",
    "velocity-model",
)


def _path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        raw: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        raw = value
    else:
        raise TypeError("LayerPulse path option must be a path or path list")
    paths: list[Path] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LayerPulse declared input asset not found: {path}")
        paths.append(path)
    return list(dict.fromkeys(paths))


def _segy_reader_provenance(options: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only reproducible, non-coordinate-transform reader choices."""

    provenance: dict[str, Any] = {}
    for key in _SEGY_READER_PROVENANCE_KEYS:
        value = options.get(key)
        if value is None:
            continue
        if key == "profile":
            profile = str(value).strip()
            if profile:
                provenance[key] = profile
            continue
        provenance[key] = int(value)
    return provenance


def _snapshot_asset_receipt(options: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory optional wells without making them forward inputs yet."""

    well_logs: list[dict[str, Any]] = []
    well_metadata: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw in options.get("snapshot_assets") or []:
        if not isinstance(raw, Mapping) or not raw.get("path"):
            continue
        role = str(raw.get("role") or "").strip().casefold()
        path = Path(str(raw["path"])).expanduser().resolve()
        marker_text = f"{role} {path.name.casefold()}"
        item = {
            "role": role or None,
            "path": str(path),
            "sha256": str(raw.get("sha256") or "").strip().casefold() or None,
            "exists": path.is_file(),
        }
        if any(marker in marker_text for marker in _FORBIDDEN_ASSET_MARKERS):
            item["reason"] = "time_depth_or_velocity_asset_not_forwarded"
            excluded.append(item)
        elif role in _WELL_LOG_ROLES:
            well_logs.append(item)
        elif role in _WELL_METADATA_ROLES:
            well_metadata.append(item)

    for path in _path_list(options.get("raw_well_paths")):
        well_logs.append(
            {"role": "well_log", "path": str(path), "sha256": None, "exists": True}
        )
    for option_name, role in (
        ("trajectory_paths", "well_metadata"),
        ("wellhead_paths", "well_heads"),
    ):
        for path in _path_list(options.get(option_name)):
            well_metadata.append(
                {"role": role, "path": str(path), "sha256": None, "exists": True}
            )

    def unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            key = str(item["path"]).casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    well_logs = unique(well_logs)
    well_metadata = unique(well_metadata)
    missing_count = sum(not bool(item["exists"]) for item in (*well_logs, *well_metadata))
    return {
        "schema_version": "well-seismic.layerpulse-registered-well-assets.v1",
        "well_logs": well_logs,
        "well_metadata": well_metadata,
        "excluded_forward_assets": excluded,
        "available_asset_count": len(well_logs) + len(well_metadata) - missing_count,
        "missing_asset_count": missing_count,
        "forward_mode": "seismic_only",
        "well_bundle_materialized": False,
        "time_depth_asset_consumed": False,
    }


def _sealed_coordinate_reference(options: Mapping[str, Any]) -> dict[str, Any]:
    semantics = options.get("source_snapshot_semantics")
    if not isinstance(semantics, Mapping):
        return {
            "coordinate_reference": None,
            "coordinate_reference_verified": False,
            "coordinate_reference_authority": None,
        }
    verified = semantics.get("coordinate_reference_verified") is True
    horizontal_crs = str(semantics.get("horizontal_crs_id") or "").strip()
    if not verified or not horizontal_crs:
        return {
            "coordinate_reference": None,
            "coordinate_reference_verified": False,
            "coordinate_reference_authority": None,
        }
    return {
        "coordinate_reference": horizontal_crs,
        "coordinate_reference_verified": True,
        "coordinate_reference_authority": "sealed_source_snapshot_semantics",
    }


def _positive_triple(value: Any, *, field: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"LayerPulse {field} must be a TWT/Inline/Xline triple")
    triple = tuple(int(item) for item in value)
    if any(item <= 0 for item in triple):
        raise ValueError(f"LayerPulse {field} entries must be positive")
    return triple  # type: ignore[return-value]


def _extended_axis_values(
    values: np.ndarray,
    *,
    start: int,
    count: int,
) -> np.ndarray:
    """Return header coordinates for a possibly padded model-input axis.

    In-bounds coordinates are retained byte-for-byte.  Coordinates required by
    a boundary halo are linearly extended only to keep trajectory
    normalisation tied to the model-input geometry; the padded samples remain
    invalid in the separately transported valid mask.
    """

    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or count <= 0:
        raise ValueError("LayerPulse context axis must be non-empty")
    differences = np.diff(axis)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    step = float(np.median(positive)) if positive.size else 1.0
    indices = np.arange(start, start + count, dtype=np.int64)
    output = np.empty(count, dtype=np.float64)
    inside = (indices >= 0) & (indices < axis.size)
    output[inside] = axis[indices[inside]]
    below = indices < 0
    output[below] = axis[0] + indices[below] * step
    above = indices >= axis.size
    output[above] = axis[-1] + (indices[above] - (axis.size - 1)) * step
    if not np.isfinite(output).all() or np.any(np.diff(output) <= 0):
        raise ValueError("LayerPulse context axis coordinates are not monotonic")
    return output


def _xy_to_grid_projection(
    geometry: Any,
    coordinate_reference: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Fit a sealed, diagnostic XY -> Inline/Crossline affine projection.

    Registration V3 deliberately keeps physical XY even when a source product
    cannot authoritatively populate Inline/Crossline.  LayerPulse still needs a
    grid location for trajectory tokens.  The SEG-Y trace headers are the
    authoritative bridge: fit on a deterministic bounded sample, retain fit
    residuals, and expose the projection only when the snapshot CRS is sealed
    and the grid is close to affine.
    """

    if coordinate_reference.get("coordinate_reference_verified") is not True:
        return None
    crs = str(coordinate_reference.get("coordinate_reference") or "").strip()
    if not crs:
        return None
    required = (
        getattr(geometry, "x", None),
        getattr(geometry, "y", None),
        getattr(geometry, "inline", None),
        getattr(geometry, "crossline", None),
    )
    if any(value is None for value in required):
        return None
    x, y, inline, crossline = (
        np.asarray(value, dtype=np.float64).reshape(-1) for value in required
    )
    if not (x.size == y.size == inline.size == crossline.size) or x.size < 4:
        return None
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(inline) & np.isfinite(crossline)
    indices = np.flatnonzero(finite)
    if indices.size < 4:
        return None
    maximum_fit_points = 8192
    if indices.size > maximum_fit_points:
        indices = indices[
            np.linspace(0, indices.size - 1, maximum_fit_points, dtype=np.int64)
        ]
    selected_x = x[indices]
    selected_y = y[indices]
    origin_x = float(selected_x.mean())
    origin_y = float(selected_y.mean())
    design = np.column_stack(
        (selected_x - origin_x, selected_y - origin_y, np.ones(indices.size))
    )
    if int(np.linalg.matrix_rank(design)) < 3:
        return None
    inline_coefficients = np.linalg.lstsq(design, inline[indices], rcond=None)[0]
    crossline_coefficients = np.linalg.lstsq(design, crossline[indices], rcond=None)[0]
    predicted_inline = design @ inline_coefficients
    predicted_crossline = design @ crossline_coefficients
    inline_residual = predicted_inline - inline[indices]
    crossline_residual = predicted_crossline - crossline[indices]
    inline_rmse = float(np.sqrt(np.mean(np.square(inline_residual))))
    crossline_rmse = float(np.sqrt(np.mean(np.square(crossline_residual))))
    inline_p99 = float(np.quantile(np.abs(inline_residual), 0.99))
    crossline_p99 = float(np.quantile(np.abs(crossline_residual), 0.99))
    if (
        not np.isfinite(inline_coefficients).all()
        or not np.isfinite(crossline_coefficients).all()
        or max(inline_rmse, crossline_rmse) > 0.35
        or max(inline_p99, crossline_p99) > 0.75
    ):
        return None
    return {
        "schema_version": "well-seismic.xy-to-grid-affine.v1",
        "source": "sealed_segy_trace_headers_affine_least_squares",
        "horizontal_crs_id": crs,
        "coefficient_order": ["x_minus_origin", "y_minus_origin", "intercept"],
        "origin_xy": [origin_x, origin_y],
        "inline_coefficients": [float(value) for value in inline_coefficients],
        "crossline_coefficients": [float(value) for value in crossline_coefficients],
        "fit_point_count": int(indices.size),
        "inline_rmse_grid_units": inline_rmse,
        "crossline_rmse_grid_units": crossline_rmse,
        "inline_p99_abs_error_grid_units": inline_p99,
        "crossline_p99_abs_error_grid_units": crossline_p99,
    }


def _fusion_ready_well_anchor(
    options: Mapping[str, Any],
    *,
    projection: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Choose a deterministic trajectory anchor without using labels or T-D."""

    if projection is None:
        return None
    # Local imports avoid the input-adapter registry's intentional late import
    # of this module.
    from ..registration_contract import read_registration_points_v3
    from .input_adapters import _prepared_view_aligned_well_inputs

    aligned = _prepared_view_aligned_well_inputs(options)
    if aligned is None:
        return None
    points, _validation = read_registration_points_v3(
        aligned["registration_points_path"]
    )
    origin = np.asarray(projection.get("origin_xy") or (), dtype=np.float64)
    inline_coefficients = np.asarray(
        projection.get("inline_coefficients") or (), dtype=np.float64
    )
    crossline_coefficients = np.asarray(
        projection.get("crossline_coefficients") or (), dtype=np.float64
    )
    if (
        origin.shape != (2,)
        or inline_coefficients.shape != (3,)
        or crossline_coefficients.shape != (3,)
    ):
        return None
    projection_crs = str(projection.get("horizontal_crs_id") or "").casefold()
    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for point in points:
        if not (
            point.valid_mask
            and point.inference_eligible
            and point.fusion_ready
            and point.x is not None
            and point.y is not None
            and np.isfinite(float(point.x))
            and np.isfinite(float(point.y))
            and str(point.horizontal_crs_id or "").casefold() == projection_crs
        ):
            continue
        if (
            point.inline is not None
            and point.crossline is not None
            and np.isfinite(float(point.inline))
            and np.isfinite(float(point.crossline))
        ):
            inline = float(point.inline)
            crossline = float(point.crossline)
        else:
            vector = np.asarray(
                [
                    float(point.x) - origin[0],
                    float(point.y) - origin[1],
                    1.0,
                ],
                dtype=np.float64,
            )
            inline = float(vector @ inline_coefficients)
            crossline = float(vector @ crossline_coefficients)
        if np.isfinite(inline) and np.isfinite(crossline):
            grouped.setdefault(point.well_uid, []).append(
                (inline, crossline, point.well_name)
            )
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for well_uid, rows in grouped.items():
        coordinates = np.asarray([(row[0], row[1]) for row in rows], dtype=np.float64)
        span = float(
            np.hypot(
                np.ptp(coordinates[:, 0]),
                np.ptp(coordinates[:, 1]),
            )
        )
        candidate = {
            "well_uid": well_uid,
            "well_name": rows[0][2],
            "inline": float(np.median(coordinates[:, 0])),
            "crossline": float(np.median(coordinates[:, 1])),
            "horizontal_grid_span": span,
            "fusion_ready_point_count": int(coordinates.shape[0]),
            "selection_policy": (
                "maximum_horizontal_grid_span_then_point_count_then_well_uid_v1"
            ),
            "time_depth_table_consumed": False,
        }
        candidates.append((span, len(rows), well_uid.casefold(), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][3]


class LayerPulseInputAdapter(GeobodySegyInputAdapter):
    """Materialize one deterministic raw-amplitude T/I/X preview patch.

    Full-volume reading remains the future subprocess runner's responsibility.
    Keeping only a bounded patch in the API worker avoids importing PyTorch or
    the LayerPulse package in the web process.
    """

    model_id = LAYERPULSE_MODEL_ID

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.model_id)
        layerpulse = config.get("layerpulse", {})
        self.layerpulse_spec = dict(
            layerpulse.get("inference", {}) if isinstance(layerpulse, Mapping) else {}
        )
        self.default_patch_size_tix = _positive_triple(
            self.layerpulse_spec.get(
                "default_patch_size_tix", _DEFAULT_PATCH_SIZE_TIX
            ),
            field="default_patch_size_tix",
        )

    def capabilities(self) -> dict[str, Any]:
        capability = super().capabilities()
        capability.update(
            {
                "model_id": self.model_id,
                "array_axes": ["TWT", "INLINE", "XLINE"],
                "tensor_axes": ["N", "C", "TWT", "INLINE", "XLINE"],
                "patch_size": list(self.default_patch_size_tix),
                "overlap": [],
                "normalization": "raw_float32_transport; child_runtime_owned",
                "requires_logs": False,
                "optional_logs": True,
                "requires_registration": False,
                "time_depth_table_required": False,
                "input_scope": "deterministic_preview_patch",
                "supports_crop": True,
                "full_volume_status": "planned_not_executed",
                "native_options": ["segy", "iline_byte", "xline_byte"],
            }
        )
        return capability

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = super().compatibility(geometry, options=options)
        source_shape = tuple(
            int(value) for value in result.get("source_shape_zyx", (0, 0, 0))
        )
        has_minimum_shape = len(source_shape) == 3 and all(
            value >= 16 for value in source_shape
        )
        geometry_ready = bool(result.get("ready"))
        default_patch_size = self.default_patch_size_tix
        effective_shape = [
            (min(available, requested) // 16) * 16
            for available, requested in zip(
                source_shape,
                default_patch_size,
                strict=len(source_shape) == 3,
            )
        ] if len(source_shape) == 3 else []
        result.update(
            {
                "ready": geometry_ready and has_minimum_shape,
                "adapter": type(self).__name__,
                "expected_axes": ["TWT", "INLINE", "XLINE"],
                "patch_size": list(default_patch_size),
                "inference_scope": "preview_patch",
                "full_volume_status": "planned_not_executed",
                "time_depth_table_required": False,
                "minimum_shape_tix": [16, 16, 16],
                "effective_preview_shape_tix": effective_shape,
                "preview_shape_clipped": bool(
                    effective_shape and effective_shape != list(default_patch_size)
                ),
            }
        )
        if result["ready"]:
            result["reason"] = (
                "规则三维 SEG-Y 可执行 LayerPulse 单 checkpoint 的确定性预览子体推理"
                if effective_shape == list(default_patch_size)
                else f"规则三维 SEG-Y 可执行 LayerPulse 预览；子体将按工区尺寸裁剪为 {effective_shape}"
            )
        elif geometry_ready:
            result["reason"] = (
                f"LayerPulse 三维 Backbone 要求每轴至少 16 个采样，当前为 {list(source_shape)}"
            )
        else:
            result["reason"] = (
                "LayerPulse 预览推理需要可解析、无重复 Inline/Crossline 的三维 SEG-Y"
            )
        return result

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        return self._prepare(request, context_halo_tix=None)

    def prepare_with_context(
        self,
        request: ModelInputRequest,
        *,
        context_halo_tix: tuple[int, int, int],
    ) -> ModelInputBatch:
        """Materialize one output ROI plus a symmetric, validity-masked halo.

        ``request.crop_start`` and ``request.crop_size`` continue to describe
        the output ROI.  The returned array is larger by twice the halo and is
        the sole model-forward input.  This method is intentionally separate
        from ``prepare`` so a client crop option cannot silently activate the
        unvalidated 160-cube candidate.
        """

        halo = tuple(int(value) for value in context_halo_tix)
        if len(halo) != 3 or any(value < 0 for value in halo):
            raise ValueError("LayerPulse context_halo_tix must be a non-negative triple")
        return self._prepare(request, context_halo_tix=halo)

    def _prepare(
        self,
        request: ModelInputRequest,
        *,
        context_halo_tix: tuple[int, int, int] | None,
    ) -> ModelInputBatch:
        validate_layerpulse_request_options(request.options)
        source = request.source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"LayerPulse SEG-Y source not found: {source}")
        if source.suffix.casefold() not in _SEGY_SUFFIXES:
            raise ValueError("LayerPulse preview input must be a .sgy or .segy file")

        segy_options = dict(request.options.get("segy") or {"profile": "standard_3d"})
        explicit_inline = request.options.get("iline_byte", request.options.get("inline_byte"))
        explicit_crossline = request.options.get(
            "xline_byte", request.options.get("crossline_byte")
        )
        if (explicit_inline is None) != (explicit_crossline is None):
            raise ValueError("LayerPulse inline and crossline bytes must be supplied together")
        if explicit_inline is not None:
            segy_options.update(
                {
                    "inline_byte": int(explicit_inline),
                    "crossline_byte": int(explicit_crossline),
                }
            )

        reader = SegyReader(source, self.config, segy_options)
        geometry = reader.inspect()
        compatibility = self.compatibility(geometry, options=request.options)
        if not compatibility["ready"]:
            raise ValueError(str(compatibility["reason"]))
        if geometry.inline is None or geometry.crossline is None:
            raise ValueError("LayerPulse preview requires resolved Inline/Crossline arrays")

        inline = np.asarray(geometry.inline, dtype=np.int64)
        crossline = np.asarray(geometry.crossline, dtype=np.int64)
        inline_values = np.unique(inline)
        crossline_values = np.unique(crossline)
        source_shape = (
            int(geometry.samples_per_trace),
            int(inline_values.size),
            int(crossline_values.size),
        )
        coordinate_reference = _sealed_coordinate_reference(request.options)
        xy_to_grid_projection = _xy_to_grid_projection(
            geometry,
            coordinate_reference,
        )
        requested_size = _positive_triple(
            request.crop_size or self.default_patch_size_tix,
            field="crop_size",
        )
        bounded_size = tuple(
            min(available, requested)
            for available, requested in zip(source_shape, requested_size, strict=True)
        )
        # The backbone has four stride-two stages. If a requested window is
        # clipped by a smaller survey, keep the largest in-bounds multiple of
        # 16 instead of passing an odd shape to the checkpoint.
        size = tuple((value // 16) * 16 for value in bounded_size)
        if any(value < 16 for value in size):
            raise ValueError(
                f"LayerPulse preview needs at least 16 samples on each axis: {source_shape}"
            )

        well_anchor: dict[str, Any] | None = None
        if request.crop_start is None:
            well_anchor = _fusion_ready_well_anchor(
                request.options,
                projection=xy_to_grid_projection,
            )
            if well_anchor is None:
                start = tuple(
                    (available - count) // 2
                    for available, count in zip(source_shape, size, strict=True)
                )
                selection = "fixed_geometry_center"
            else:
                inline_center = int(
                    np.argmin(
                        np.abs(
                            inline_values.astype(np.float64)
                            - float(well_anchor["inline"])
                        )
                    )
                )
                crossline_center = int(
                    np.argmin(
                        np.abs(
                            crossline_values.astype(np.float64)
                            - float(well_anchor["crossline"])
                        )
                    )
                )
                start = (
                    (source_shape[0] - size[0]) // 2,
                    int(np.clip(inline_center - size[1] // 2, 0, source_shape[1] - size[1])),
                    int(
                        np.clip(
                            crossline_center - size[2] // 2,
                            0,
                            source_shape[2] - size[2],
                        )
                    ),
                )
                selection = "fusion_ready_well_trajectory_anchor"
        else:
            start = tuple(int(value) for value in request.crop_start)
            selection = "explicit_geometry_crop"
        if any(value < 0 for value in start) or any(
            offset + count > available
            for offset, count, available in zip(start, size, source_shape, strict=True)
        ):
            raise ValueError(
                f"LayerPulse preview crop {start}+{size} exceeds source {source_shape}"
            )

        halo = context_halo_tix or (0, 0, 0)
        model_input_shape = tuple(
            count + 2 * margin for count, margin in zip(size, halo, strict=True)
        )
        model_input_origin = tuple(
            offset - margin for offset, margin in zip(start, halo, strict=True)
        )
        source_read_start = tuple(max(0, offset) for offset in model_input_origin)
        source_read_end = tuple(
            min(available, offset + count)
            for offset, count, available in zip(
                model_input_origin,
                model_input_shape,
                source_shape,
                strict=True,
            )
        )
        source_read_size = tuple(
            end - begin
            for begin, end in zip(source_read_start, source_read_end, strict=True)
        )
        padding_before = tuple(
            begin - origin
            for begin, origin in zip(source_read_start, model_input_origin, strict=True)
        )
        padding_after = tuple(
            total - before - read
            for total, before, read in zip(
                model_input_shape,
                padding_before,
                source_read_size,
                strict=True,
            )
        )
        if any(value <= 0 for value in source_read_size):
            raise ValueError("LayerPulse context window does not intersect the SEG-Y volume")

        lookup: dict[tuple[int, int], int] = {}
        for trace_index, pair in enumerate(zip(inline, crossline, strict=True)):
            key = (int(pair[0]), int(pair[1]))
            if key in lookup:
                raise ValueError(f"LayerPulse input contains duplicate grid trace {key}")
            lookup[key] = trace_index

        t_start, inline_start, crossline_start = source_read_start
        t_count, inline_count, crossline_count = source_read_size
        selected_inlines = inline_values[inline_start : inline_start + inline_count]
        selected_crosslines = crossline_values[
            crossline_start : crossline_start + crossline_count
        ]
        output_inlines = inline_values[start[1] : start[1] + size[1]]
        output_crosslines = crossline_values[start[2] : start[2] + size[2]]
        model_inlines = _extended_axis_values(
            inline_values,
            start=model_input_origin[1],
            count=model_input_shape[1],
        )
        model_crosslines = _extended_axis_values(
            crossline_values,
            start=model_input_origin[2],
            count=model_input_shape[2],
        )
        patch = np.zeros(model_input_shape, dtype=np.float32)
        valid_mask = np.zeros(model_input_shape, dtype=np.bool_)
        destination_t_start = padding_before[0]
        destination_inline_start = padding_before[1]
        destination_crossline_start = padding_before[2]
        for inline_index, inline_value in enumerate(selected_inlines):
            destination_inline = destination_inline_start + inline_index
            for crossline_index, crossline_value in enumerate(selected_crosslines):
                destination_crossline = destination_crossline_start + crossline_index
                trace_index = lookup.get((int(inline_value), int(crossline_value)))
                if trace_index is None:
                    continue
                values = np.asarray(
                    reader.read_trace(
                        trace_index,
                        slice(t_start, t_start + t_count),
                    ),
                    dtype=np.float32,
                )
                if values.shape != (t_count,):
                    raise ValueError("LayerPulse SEG-Y trace crop has an unexpected length")
                finite = np.isfinite(values)
                destination_t = slice(destination_t_start, destination_t_start + t_count)
                patch[destination_t, destination_inline, destination_crossline] = np.where(
                    finite, values, 0.0
                )
                valid_mask[
                    destination_t, destination_inline, destination_crossline
                ] = finite
        if not np.any(valid_mask):
            raise ValueError("LayerPulse preview crop contains no finite seismic samples")

        output_slices = tuple(
            slice(margin, margin + count)
            for margin, count in zip(halo, size, strict=True)
        )
        output_valid_mask = valid_mask[output_slices]
        if not np.any(output_valid_mask):
            raise ValueError("LayerPulse output ROI contains no finite seismic samples")

        well_assets = _snapshot_asset_receipt(request.options)
        recommended = compatibility.get("recommended_options")
        return ModelInputBatch(
            model_id=self.model_id,
            array=patch,
            valid_mask=valid_mask,
            axes=("TWT", "INLINE", "XLINE"),
            provenance={
                "source": str(source),
                "source_shape_tix": list(source_shape),
                "crop_start_tix": list(start),
                "crop_size_tix": list(size),
                "crop_selection": selection,
                "well_anchor": well_anchor,
                "sample_interval_ms": float(geometry.sample_interval),
                "geometry_profile": str(geometry.profile),
                "geometry_confidence": float(geometry.confidence),
                "segy_reader_options": _segy_reader_provenance(segy_options),
                "recommended_options": (
                    dict(recommended) if isinstance(recommended, Mapping) else {}
                ),
                # The unsuffixed axes are the model-input axes used to
                # normalise trajectory tokens.  Output axes are kept
                # separately so central cropping cannot shift display/world
                # coordinates.
                "inline_values": [float(value) for value in model_inlines],
                "crossline_values": [float(value) for value in model_crosslines],
                "inline_range": [
                    float(model_inlines[0]),
                    float(model_inlines[-1]),
                ],
                "crossline_range": [
                    float(model_crosslines[0]),
                    float(model_crosslines[-1]),
                ],
                "output_inline_values": [int(value) for value in output_inlines],
                "output_crossline_values": [int(value) for value in output_crosslines],
                "output_inline_range": [
                    int(output_inlines[0]),
                    int(output_inlines[-1]),
                ],
                "output_crossline_range": [
                    int(output_crosslines[0]),
                    int(output_crosslines[-1]),
                ],
                "model_input_shape_tix": list(model_input_shape),
                "model_input_origin_tix": list(model_input_origin),
                "model_input_source_start_tix": list(source_read_start),
                "model_input_source_size_tix": list(source_read_size),
                "model_input_padding_before_tix": list(padding_before),
                "model_input_padding_after_tix": list(padding_after),
                "output_offset_in_model_input_tix": list(halo),
                "context_halo_tix": list(halo),
                "context_halo_enabled": any(halo),
                **coordinate_reference,
                "xy_to_grid_projection": xy_to_grid_projection,
                "valid_sample_fraction": float(valid_mask.mean()),
                "output_valid_sample_fraction": float(output_valid_mask.mean()),
                "materialization": (
                    "bounded_raw_float32_output_roi_with_validity_masked_context_halo"
                    if any(halo)
                    else "bounded_raw_float32_center_preview_patch"
                ),
                "source_snapshot_id": request.options.get("source_snapshot_id"),
                "source_snapshot_fingerprint": request.options.get(
                    "source_snapshot_fingerprint"
                ),
                "well_assets": well_assets,
                "well_input_consumed": False,
                "registration_consumed": False,
                "prepared_view_consumed": False,
                "time_depth_supervision_opened": False,
                "full_volume_status": "planned_not_executed",
            },
        )


__all__ = ["LayerPulseInputAdapter"]
