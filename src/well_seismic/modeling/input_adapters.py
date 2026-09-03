"""Model-specific input adaptation over shared, model-neutral data readers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..content_identity import canonical_sha256, seismic_geometry_identity
from ..fault_models import FAULTNET_MODEL_ID, FAULTSEG_MODEL_ID
from ..platform_mode import interface_only_enabled
from ..faultseg import FaultSegInputSpec, FaultSegVolume, build_faultseg_volume
from ..io.segy import SegyReader
from ..io.tabular import _normalized_column, _rows_with_evidence
from ..model_applicability import (
    observe_seismic_reader,
    select_unlabeled_seismic_roi,
)
from ..segy_geometry_receipt import (
    DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE as GEOPATH_MINIMUM_GEOMETRY_CONFIDENCE,
    validate_snapshot_segy_geometry_receipt,
)

Shape3D = tuple[int, int, int]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelInputRequest:
    source: Path
    crop_start: Shape3D | None = None
    crop_size: Shape3D | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInputBatch:
    model_id: str
    array: np.ndarray | None
    valid_mask: np.ndarray | None
    axes: tuple[str, ...]
    provenance: dict[str, Any]


@runtime_checkable
class ModelInputAdapter(Protocol):
    model_id: str

    def capabilities(self) -> dict[str, Any]: ...

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch: ...


class ModelInputAdapterRegistry:
    entry_point_group = "well_seismic.input_adapters"

    def __init__(self) -> None:
        self._adapters: dict[str, ModelInputAdapter] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(self, adapter: ModelInputAdapter, *, replace: bool = False) -> None:
        if adapter.model_id in self._adapters and not replace:
            raise ValueError(f"input adapter already registered: {adapter.model_id}")
        self._adapters[adapter.model_id] = adapter

    def get(self, model_id: str) -> ModelInputAdapter:
        try:
            return self._adapters[model_id]
        except KeyError as exc:
            raise KeyError(
                f"no input adapter registered for model: {model_id}"
            ) from exc

    def capabilities(self) -> list[dict[str, Any]]:
        return [adapter.capabilities() for adapter in self._adapters.values()]

    def compatibilities(
        self,
        geometry: Any,
        *,
        options_by_model: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Describe whether one inspected seismic asset can enter each adapter."""
        output: dict[str, dict[str, Any]] = {}
        options_by_model = options_by_model or {}
        for model_id, adapter in self._adapters.items():
            checker = getattr(adapter, "compatibility", None)
            if callable(checker):
                output[model_id] = checker(
                    geometry,
                    options=options_by_model.get(model_id, {}),
                )
                continue
            capability = adapter.capabilities()
            output[model_id] = {
                "ready": False,
                "reason": "输入适配器尚未声明数据兼容性检查",
                "adapter": type(adapter).__name__,
                "expected_axes": list(capability.get("array_axes", [])),
                "patch_size": list(capability.get("patch_size", [])),
            }
        return output

    def unavailable_compatibilities(self, reason: str) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for model_id, adapter in self._adapters.items():
            capability = adapter.capabilities()
            output[model_id] = {
                "ready": False,
                "reason": reason,
                "adapter": type(adapter).__name__,
                "expected_axes": list(capability.get("array_axes", [])),
                "patch_size": list(capability.get("patch_size", [])),
            }
        return output

    def load_entry_points(self, config: dict[str, Any]) -> list[str]:
        """Load adapter factories declared as ``factory(config) -> adapter``."""
        if interface_only_enabled():
            return []
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                factory = entry_point.load()
                self.register(factory(config))
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {
                        "plugin": entry_point.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return loaded


_FAULTSEG_DISK_BYTES_PER_VOXEL = 4 + 4 + 4 + 1
FAULTSEG_EXECUTION_SCOPE_CONTRACT_VERSION = (
    "well-seismic.faultseg-execution-scope.v2"
)
FAULTSEG_CENTER_BLOCK_SCOPE = "center_block_1"
FAULTSEG_REPRESENTATIVE_SCOPE = "representative_grid_128"
FAULTSEG_FULL_VOLUME_SCOPE = "full_volume"
FAULTSEG_CENTER_BLOCK_CONTRACT_VERSION = (
    "well-seismic.faultseg-center-block.v1"
)
FAULTSEG_REPRESENTATIVE_GRID_CONTRACT_VERSION = (
    "well-seismic.faultseg-representative-grid.v2"
)
_FAULTSEG_CENTER_BLOCK_SHAPE_ZYX: Shape3D = (128, 128, 128)
_FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX: Shape3D = (128, 128, 128)
_FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX: Shape3D = (8, 4, 4)
_FAULTSEG_REPRESENTATIVE_BLOCK_COUNT = 128
_FAULTSEG_FULL_VOLUME_PATCH_ZYX: Shape3D = (128, 128, 128)
_FAULTSEG_FULL_VOLUME_OVERLAP_ZYX: Shape3D = (64, 64, 64)


def faultseg_execution_scope_contract(
    model_id: str = FAULTSEG_MODEL_ID,
) -> dict[str, Any]:
    """Return the public request contract and duration hints for fault inference."""

    center_block_supported = model_id == FAULTSEG_MODEL_ID
    default_scope = (
        FAULTSEG_CENTER_BLOCK_SCOPE
        if center_block_supported
        else FAULTSEG_FULL_VOLUME_SCOPE
    )
    options: list[dict[str, Any]] = []
    if center_block_supported:
        options.append(
            {
                "value": FAULTSEG_CENTER_BLOCK_SCOPE,
                "label": "工区中心单块",
                "description": "在三轴中心确定性截取 1 个 128³ 完整块进行预测",
                "is_default": True,
                "is_full_volume": False,
                "block_id": FAULTSEG_CENTER_BLOCK_SCOPE,
                "block_count": 1,
                "representative_block_count": None,
                "block_shape_zyx": list(_FAULTSEG_CENTER_BLOCK_SHAPE_ZYX),
                "grid_shape_zyx": None,
                "selection_policy": "floor_center_with_lower_index_tie_break_v1",
                "estimated_duration_class": "single_center_block",
                "long_running": False,
                "requires_long_running_confirmation": False,
                "warning_message": None,
            }
        )
    options.append(
        {
            "value": FAULTSEG_FULL_VOLUME_SCOPE,
            "label": "全区预测",
            "description": "遍历并拼接覆盖整个三维工区的全部 128³ 窗口",
            "is_default": default_scope == FAULTSEG_FULL_VOLUME_SCOPE,
            "is_full_volume": True,
            "representative_block_count": None,
            "block_shape_zyx": list(_FAULTSEG_FULL_VOLUME_PATCH_ZYX),
            "grid_shape_zyx": None,
            "estimated_duration_class": "very_long_full_volume",
            "long_running": True,
            "requires_long_running_confirmation": True,
            "warning_message": "全区预测需遍历并拼接全部 128³ 窗口，耗时较长。",
        }
    )
    allowed_values = [str(option["value"]) for option in options]
    return {
        "contract_version": FAULTSEG_EXECUTION_SCOPE_CONTRACT_VERSION,
        "option_key": "faultseg_scope",
        "default_value": default_scope,
        "allowed_values": allowed_values,
        "options": options,
    }


def faultseg_execution_scope_metadata(
    scope: str,
    *,
    model_id: str = FAULTSEG_MODEL_ID,
) -> dict[str, Any]:
    """Resolve one scope to the exact public metadata returned with a result."""

    normalized = str(scope).strip().casefold()
    contract = faultseg_execution_scope_contract(model_id)
    for option in contract["options"]:
        if option["value"] == normalized:
            return {
                "contract_version": contract["contract_version"],
                "request_option": contract["option_key"],
                **dict(option),
            }
    if normalized == FAULTSEG_REPRESENTATIVE_SCOPE:
        return {
            "contract_version": contract["contract_version"],
            "request_option": contract["option_key"],
            "value": normalized,
            "label": "历史 128 个代表块",
            "description": "历史只读代表块结果；不再用于创建新任务",
            "is_default": False,
            "is_full_volume": False,
            "historical_read_only": True,
            "representative_block_count": _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT,
            "block_shape_zyx": list(_FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX),
            "grid_shape_zyx": list(_FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX),
            "estimated_duration_class": "historical_representative_batch",
            "long_running": False,
            "requires_long_running_confirmation": False,
            "warning_message": None,
        }
    # Historical diagnostic scopes remain explicit but are never advertised as
    # the selectable production default.
    if normalized in {"automatic_valid_roi", "debug_crop"}:
        return {
            "contract_version": contract["contract_version"],
            "request_option": contract["option_key"],
            "value": normalized,
            "label": "诊断范围",
            "description": "历史诊断推理范围",
            "is_default": False,
            "is_full_volume": False,
            "representative_block_count": None,
            "block_shape_zyx": list(_FAULTSEG_FULL_VOLUME_PATCH_ZYX),
            "grid_shape_zyx": None,
            "estimated_duration_class": "diagnostic",
            "long_running": False,
            "requires_long_running_confirmation": False,
            "warning_message": None,
        }
    raise ValueError(f"unsupported FaultSeg scope: {scope}")


def _faultseg_voxel_count(shape: Sequence[int]) -> int:
    return int(np.prod(tuple(int(value) for value in shape), dtype=np.int64))


def _faultseg_working_bytes(shape: Sequence[int], safety_factor: float) -> int:
    if not np.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("FaultSeg minimum_free_space_factor must be at least 1.0")
    return int(
        np.ceil(
            _faultseg_voxel_count(shape)
            * _FAULTSEG_DISK_BYTES_PER_VOXEL
            * float(safety_factor)
        )
    )


def _faultseg_integral_line_values(values: Any, *, field: str) -> np.ndarray:
    """Return compact int64 line numbers or reject ambiguous geometry."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"FaultSeg {field} headers must be one-dimensional")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"FaultSeg {field} headers cannot be boolean")
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"FaultSeg {field} headers must be numeric integers") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"FaultSeg {field} headers contain non-finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(
            f"FaultSeg {field} headers contain non-integer line numbers"
        )
    return rounded.astype(np.int64)


def _faultseg_trace_grid(
    geometry: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a compact [Inline, Xline] -> trace-index grid.

    The previous tuple-key dictionary costs hundreds of megabytes for surveys
    with more than a million traces.  Integer axis indices plus one dense int64
    grid keep the same lookup bounded and make duplicate bins fail closed.
    """

    inline = _faultseg_integral_line_values(geometry.inline, field="Inline")
    crossline = _faultseg_integral_line_values(
        geometry.crossline, field="Crossline"
    )
    trace_count = int(getattr(geometry, "trace_count", inline.size))
    if inline.size != crossline.size or inline.size != trace_count:
        raise ValueError(
            "FaultSeg Inline/Crossline header counts must equal SEG-Y trace_count"
        )
    inline_values = np.unique(inline)
    crossline_values = np.unique(crossline)
    inline_indices = np.searchsorted(inline_values, inline)
    crossline_indices = np.searchsorted(crossline_values, crossline)
    cell_count = int(inline_values.size) * int(crossline_values.size)
    if cell_count < 1:
        raise ValueError("FaultSeg resolved an empty Inline/Crossline grid")
    flat = (
        inline_indices.astype(np.int64, copy=False) * int(crossline_values.size)
        + crossline_indices.astype(np.int64, copy=False)
    )
    ordered = np.sort(flat, kind="stable")
    if ordered.size > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError(
            "FaultSeg rejects duplicate traces in the same Inline/Crossline bin"
        )
    trace_grid = np.full(cell_count, -1, dtype=np.int64)
    trace_grid[flat] = np.arange(trace_count, dtype=np.int64)
    return (
        trace_grid.reshape(inline_values.size, crossline_values.size),
        inline_values,
        crossline_values,
    )


def _materialize_faultseg_npy(
    reader: SegyReader,
    destination: Path,
    *,
    start_zyx: Shape3D,
    size_zyx: Shape3D,
    trace_grid_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """Stream one regular SEG-Y region directly into a disk-backed NPY.

    Only a single trace is decoded in RAM at a time.  Missing grid cells are
    represented by NaN in the staged amplitude volume and by ``False`` in the
    compact two-dimensional validity mask.
    """

    geometry = reader.geometry or reader.inspect()
    if geometry.inline is None or geometry.crossline is None:
        raise ValueError("FaultSeg requires resolved 3D inline/crossline geometry")
    z_start, inline_start, crossline_start = start_zyx
    z_count, inline_count, crossline_count = size_zyx
    trace_grid, inline_all, crossline_all = (
        trace_grid_context
        if trace_grid_context is not None
        else _faultseg_trace_grid(geometry)
    )
    inline_values = inline_all[inline_start : inline_start + inline_count]
    crossline_values = crossline_all[
        crossline_start : crossline_start + crossline_count
    ]
    selected_trace_grid = trace_grid[
        inline_start : inline_start + inline_count,
        crossline_start : crossline_start + crossline_count,
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=size_zyx,
    )
    staged.fill(np.nan)
    valid_traces = np.zeros((inline_count, crossline_count), dtype=bool)
    sample_slice = slice(z_start, z_start + z_count)
    for inline_index in range(inline_count):
        for crossline_index in range(crossline_count):
            trace_index = int(selected_trace_grid[inline_index, crossline_index])
            if trace_index < 0:
                continue
            values = np.asarray(
                reader.read_trace(trace_index, sample_slice), dtype=np.float32
            )
            if values.shape != (z_count,):
                raise ValueError(
                    "SEG-Y trace crop length does not match staged FaultSeg Z dimension"
                )
            staged[:, inline_index, crossline_index] = values
            valid_traces[inline_index, crossline_index] = True
    staged.flush()
    return staged, valid_traces, inline_values, crossline_values


def _faultseg_partition_center_starts(
    length: int,
    *,
    partitions: int,
    block: int,
    axis: str,
) -> tuple[int, ...]:
    """Return deterministic full-block starts at partition centers.

    Integer floor arithmetic makes the policy stable across Python/NumPy
    versions.  Starts are clamped so every block lies completely inside the
    source axis.  A small source, or a source that cannot yield the requested
    number of distinct representative positions, fails closed.
    """

    available = int(length)
    count = int(partitions)
    width = int(block)
    if available < width:
        raise ValueError(
            f"FaultSeg representative grid axis {axis} has {available} samples; "
            f"a complete {width}-sample block is required"
        )
    starts = tuple(
        min(
            available - width,
            max(0, ((2 * index + 1) * available) // (2 * count) - width // 2),
        )
        for index in range(count)
    )
    if len(set(starts)) != count:
        raise ValueError(
            f"FaultSeg representative grid axis {axis} cannot provide {count} "
            f"unique complete {width}-sample blocks from length {available}: {starts}"
        )
    return starts


def _faultseg_coordinate_value(value: Any) -> int | float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("FaultSeg representative grid axis contains non-finite values")
    integer = int(numeric)
    return integer if numeric == integer else numeric


def _faultseg_axis_range(
    *,
    start: int,
    block: int,
    values: np.ndarray | None,
    coordinate_source: str,
    domain: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    end_inclusive = int(start + block - 1)
    receipt: dict[str, Any] = {
        "index_start": int(start),
        "index_end_inclusive": end_inclusive,
        "index_end_exclusive": int(start + block),
        "coordinate_source": coordinate_source,
        "domain": domain,
        "unit": unit,
    }
    if values is not None:
        receipt["coordinate_start"] = _faultseg_coordinate_value(values[start])
        receipt["coordinate_end"] = _faultseg_coordinate_value(values[end_inclusive])
    else:
        receipt["coordinate_start"] = None
        receipt["coordinate_end"] = None
    return receipt


def _faultseg_declared_vertical_metadata(geometry: Any) -> tuple[str | None, str | None]:
    domain = next(
        (
            str(value).strip()
            for value in (
                getattr(geometry, "vertical_domain", None),
                getattr(geometry, "sample_domain", None),
                getattr(geometry, "time_domain", None),
            )
            if value is not None and str(value).strip()
        ),
        None,
    )
    unit = next(
        (
            str(value).strip()
            for value in (
                getattr(geometry, "vertical_unit", None),
                getattr(geometry, "sample_unit", None),
                getattr(geometry, "time_unit", None),
            )
            if value is not None and str(value).strip()
        ),
        None,
    )
    return domain, unit


def _faultseg_center_block_plan(
    geometry: Any,
    shape_zyx: Shape3D,
) -> dict[str, Any]:
    """Plan one deterministic, complete 128-cubed block at survey center."""

    source_shape = tuple(int(value) for value in shape_zyx)
    block_shape = _FAULTSEG_CENTER_BLOCK_SHAPE_ZYX
    if any(
        available < block
        for available, block in zip(source_shape, block_shape, strict=True)
    ):
        raise ValueError(
            "FaultSeg center_block_1 requires every source axis to contain a "
            f"complete {block_shape} block; source is {source_shape}"
        )
    # Floor division gives a stable lower-index tie break whenever an axis has
    # two equally central starts. The complete block is therefore always
    # inside [0, source_length) without a data-dependent search or clamp.
    start = tuple(
        (available - block) // 2
        for available, block in zip(source_shape, block_shape, strict=True)
    )
    end_exclusive = tuple(
        axis_start + axis_size
        for axis_start, axis_size in zip(start, block_shape, strict=True)
    )
    inline_values = np.unique(
        _faultseg_integral_line_values(geometry.inline, field="Inline")
    )
    crossline_values = np.unique(
        _faultseg_integral_line_values(geometry.crossline, field="Crossline")
    )
    if len(inline_values) != source_shape[1] or len(crossline_values) != source_shape[2]:
        raise ValueError("FaultSeg center block header axes differ from source shape")
    raw_vertical = getattr(geometry, "time_axis", None)
    vertical_values: np.ndarray | None = None
    if raw_vertical is not None:
        candidate = np.asarray(raw_vertical)
        if candidate.ndim == 1 and len(candidate) == source_shape[0]:
            numeric = candidate.astype(np.float64)
            if np.isfinite(numeric).all():
                vertical_values = numeric
    vertical_domain, vertical_unit = _faultseg_declared_vertical_metadata(geometry)
    vertical_coordinate_source = (
        "geometry.time_axis" if vertical_values is not None else "not_declared"
    )
    return {
        "contract_version": FAULTSEG_CENTER_BLOCK_CONTRACT_VERSION,
        "scope": FAULTSEG_CENTER_BLOCK_SCOPE,
        "block_id": FAULTSEG_CENTER_BLOCK_SCOPE,
        "block_count": 1,
        "is_full_volume": False,
        "source_shape_zyx": list(source_shape),
        "source_start_zyx": list(start),
        "source_end_zyx_exclusive": list(end_exclusive),
        "source_end_zyx_inclusive": [value - 1 for value in end_exclusive],
        "shape_zyx": list(block_shape),
        "selection_policy": "floor_center_with_lower_index_tie_break_v1",
        "boundary_policy": "complete_block_inside_source_no_padding_v1",
        "axis_coordinate_ranges": {
            "Z": _faultseg_axis_range(
                start=start[0],
                block=block_shape[0],
                values=vertical_values,
                coordinate_source=vertical_coordinate_source,
                domain=vertical_domain,
                unit=vertical_unit,
            ),
            "INLINE": _faultseg_axis_range(
                start=start[1],
                block=block_shape[1],
                values=inline_values,
                coordinate_source="SEG-Y Inline header",
            ),
            "CROSSLINE": _faultseg_axis_range(
                start=start[2],
                block=block_shape[2],
                values=crossline_values,
                coordinate_source="SEG-Y Crossline header",
            ),
        },
        "coverage_fraction": float(
            _faultseg_voxel_count(block_shape) / _faultseg_voxel_count(source_shape)
        ),
    }


def _faultseg_representative_grid_plan(
    geometry: Any,
    shape_zyx: Shape3D,
) -> dict[str, Any]:
    """Plan the sealed 8x4x4 representative grid without reading amplitudes."""

    source_shape = tuple(int(value) for value in shape_zyx)
    block_shape = _FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX
    grid_shape = _FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX
    starts_by_axis = tuple(
        _faultseg_partition_center_starts(
            available,
            partitions=partitions,
            block=block,
            axis=axis,
        )
        for available, partitions, block, axis in zip(
            source_shape,
            grid_shape,
            block_shape,
            ("Z", "INLINE", "CROSSLINE"),
            strict=True,
        )
    )
    inline_values = np.unique(
        _faultseg_integral_line_values(geometry.inline, field="Inline")
    )
    crossline_values = np.unique(
        _faultseg_integral_line_values(geometry.crossline, field="Crossline")
    )
    if len(inline_values) != source_shape[1] or len(crossline_values) != source_shape[2]:
        raise ValueError(
            "FaultSeg representative grid header axes differ from source shape"
        )
    raw_vertical = getattr(geometry, "time_axis", None)
    vertical_values: np.ndarray | None = None
    if raw_vertical is not None:
        candidate = np.asarray(raw_vertical)
        if candidate.ndim == 1 and len(candidate) == source_shape[0]:
            numeric = candidate.astype(np.float64)
            if np.isfinite(numeric).all():
                vertical_values = numeric
    vertical_domain, vertical_unit = _faultseg_declared_vertical_metadata(geometry)
    vertical_coordinate_source = (
        "geometry.time_axis" if vertical_values is not None else "not_declared"
    )
    blocks: list[dict[str, Any]] = []
    ordinal = 0
    for z_grid, z_start in enumerate(starts_by_axis[0]):
        for inline_grid, inline_start in enumerate(starts_by_axis[1]):
            for crossline_grid, crossline_start in enumerate(starts_by_axis[2]):
                start = (z_start, inline_start, crossline_start)
                end_exclusive = tuple(
                    axis_start + axis_size
                    for axis_start, axis_size in zip(start, block_shape, strict=True)
                )
                blocks.append(
                    {
                        "block_id": f"z{z_grid:02d}_i{inline_grid:02d}_x{crossline_grid:02d}",
                        "ordinal": ordinal,
                        "grid_index_zyx": [z_grid, inline_grid, crossline_grid],
                        "source_start_zyx": list(start),
                        "source_end_zyx_exclusive": list(end_exclusive),
                        "source_end_zyx_inclusive": [value - 1 for value in end_exclusive],
                        "shape_zyx": list(block_shape),
                        "axis_coordinate_ranges": {
                            "Z": _faultseg_axis_range(
                                start=z_start,
                                block=block_shape[0],
                                values=vertical_values,
                                coordinate_source=vertical_coordinate_source,
                                domain=vertical_domain,
                                unit=vertical_unit,
                            ),
                            "INLINE": _faultseg_axis_range(
                                start=inline_start,
                                block=block_shape[1],
                                values=inline_values,
                                coordinate_source="SEG-Y Inline header",
                            ),
                            "CROSSLINE": _faultseg_axis_range(
                                start=crossline_start,
                                block=block_shape[2],
                                values=crossline_values,
                                coordinate_source="SEG-Y Crossline header",
                            ),
                        },
                    }
                )
                ordinal += 1
    starts = {tuple(item["source_start_zyx"]) for item in blocks}
    if len(blocks) != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT or len(starts) != len(blocks):
        raise ValueError(
            "FaultSeg representative grid must contain exactly 128 unique source starts"
        )

    def union_length(axis_starts: tuple[int, ...], width: int) -> int:
        merged: list[list[int]] = []
        for start in axis_starts:
            end = start + width
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged)

    union_voxels = int(
        np.prod(
            [
                union_length(axis_starts, width)
                for axis_starts, width in zip(
                    starts_by_axis, block_shape, strict=True
                )
            ],
            dtype=np.int64,
        )
    )
    source_axis_ranges = {
        "Z": _faultseg_axis_range(
            start=0,
            block=source_shape[0],
            values=vertical_values,
            coordinate_source=vertical_coordinate_source,
            domain=vertical_domain,
            unit=vertical_unit,
        ),
        "INLINE": _faultseg_axis_range(
            start=0,
            block=source_shape[1],
            values=inline_values,
            coordinate_source="SEG-Y Inline header",
        ),
        "CROSSLINE": _faultseg_axis_range(
            start=0,
            block=source_shape[2],
            values=crossline_values,
            coordinate_source="SEG-Y Crossline header",
        ),
    }
    return {
        "contract_version": FAULTSEG_REPRESENTATIVE_GRID_CONTRACT_VERSION,
        "scope": "representative_sampling",
        "is_full_volume": False,
        "source_shape_zyx": list(source_shape),
        "source_axis_ranges": source_axis_ranges,
        "sample_interval": (
            float(geometry.sample_interval)
            if getattr(geometry, "sample_interval", None) is not None
            and np.isfinite(float(geometry.sample_interval))
            else None
        ),
        "grid_shape_zyx": list(grid_shape),
        "block_shape_zyx": list(block_shape),
        "block_count": len(blocks),
        "grid_order": "Z_then_INLINE_then_CROSSLINE",
        "start_policy": "partition_center_floor_then_clamp_to_full_source_block_v1",
        "unique_source_starts": True,
        "inference_overlap_zyx": [0, 0, 0],
        "inter_block_stitching": False,
        "sampled_voxel_count_with_overlap": int(
            len(blocks) * _faultseg_voxel_count(block_shape)
        ),
        "representative_union_voxel_count": union_voxels,
        "representative_union_coverage_fraction": float(
            union_voxels / _faultseg_voxel_count(source_shape)
        ),
        "blocks": blocks,
    }


def _faultseg_block_input_evidence(
    array: np.ndarray,
    valid_traces: np.ndarray,
) -> dict[str, Any]:
    total_voxels = int(np.prod(array.shape, dtype=np.int64))
    finite_count = 0
    nonzero_finite_count = 0
    maximum_absolute_amplitude = 0.0
    active_traces = np.zeros(valid_traces.shape, dtype=bool)
    for z_start in range(0, array.shape[0], 16):
        slab = np.asarray(array[z_start : z_start + 16])
        finite = np.isfinite(slab)
        finite_count += int(np.count_nonzero(finite))
        nonzero = finite & (slab != 0.0)
        nonzero_finite_count += int(np.count_nonzero(nonzero))
        if np.any(finite):
            maximum_absolute_amplitude = max(
                maximum_absolute_amplitude,
                float(np.max(np.abs(slab[finite]))),
            )
        active_traces |= np.any(nonzero, axis=0)
    valid_count = int(np.count_nonzero(valid_traces))
    active_valid_count = int(np.count_nonzero(active_traces & valid_traces))
    histogram_bins = 4096
    absolute_p99 = 0.0
    if finite_count and maximum_absolute_amplitude > 0.0:
        histogram = np.zeros(histogram_bins, dtype=np.int64)
        for z_start in range(0, array.shape[0], 16):
            slab = np.asarray(array[z_start : z_start + 16])
            finite_values = np.abs(slab[np.isfinite(slab)]).astype(
                np.float64, copy=False
            )
            if not finite_values.size:
                continue
            indices = np.minimum(
                histogram_bins - 1,
                np.floor(
                    finite_values
                    / maximum_absolute_amplitude
                    * histogram_bins
                ).astype(np.int64),
            )
            histogram += np.bincount(indices, minlength=histogram_bins)
        rank = max(1, int(np.ceil(0.99 * finite_count)))
        bin_index = int(np.searchsorted(np.cumsum(histogram), rank, side="left"))
        absolute_p99 = float(
            maximum_absolute_amplitude * (bin_index + 1) / histogram_bins
        )
    return {
        "total_voxel_count": total_voxels,
        "finite_voxel_count": finite_count,
        "finite_sample_fraction": float(finite_count / max(1, total_voxels)),
        "nonzero_finite_voxel_count": nonzero_finite_count,
        "nonzero_fraction_of_finite": float(
            nonzero_finite_count / max(1, finite_count)
        ),
        "valid_trace_count": valid_count,
        "valid_trace_fraction": float(valid_count / max(1, valid_traces.size)),
        "active_valid_trace_count": active_valid_count,
        "active_trace_fraction_of_valid": float(
            active_valid_count / max(1, valid_count)
        ),
        "input_absolute_amplitude_max": maximum_absolute_amplitude,
        "input_abs_p99": absolute_p99,
        "input_abs_p99_receipt": {
            "algorithm": "absolute_amplitude_fixed_histogram_upper_edge_v1",
            "quantile": 0.99,
            "histogram_bins": histogram_bins,
            "finite_sample_count": finite_count,
        },
    }


class FaultSegInputAdapter:
    model_id = FAULTSEG_MODEL_ID

    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_id: str = FAULTSEG_MODEL_ID,
        config_key: str = "faultseg",
    ) -> None:
        self.config = config
        self.model_id = model_id
        self.config_key = config_key
        self.model_config = dict(config.get(config_key, {}))
        if not self.model_config and model_id == FAULTNET_MODEL_ID:
            # Capability-only callers historically construct the registry with
            # an empty config. Keep that read-only path available while the
            # prediction runner still requires the complete runtime section.
            self.model_config = {
                "patch_size": [128, 128, 128],
                "overlap": [64, 64, 64],
                "patch_multiple": 16,
                "threshold": 0.5,
                "normalization": "per_patch_minmax",
            }
        self.spec = FaultSegInputSpec.from_config({"faultseg": self.model_config})

    def capabilities(self) -> dict[str, Any]:
        scope_contract = faultseg_execution_scope_contract(self.model_id)
        return {
            "model_id": self.model_id,
            "source_formats": ["sgy", "segy"],
            "array_axes": ["Z", "INLINE", "CROSSLINE"],
            "tensor_axes": ["N", "C", "Z", "INLINE", "CROSSLINE"],
            "dtype": "float32",
            "patch_size": list(self.spec.patch_size),
            "overlap": list(self.spec.overlap),
            "normalization": str(
                self.model_config.get("normalization", "per_patch_zscore")
            ),
            "requires_logs": False,
            "input_scope": "selectable_faultseg_scope",
            "supports_crop": False,
            "default_scope": scope_contract["default_value"],
            "supported_scopes": list(scope_contract["allowed_values"]),
            "scope_options": [dict(item) for item in scope_contract["options"]],
            "scope_contract": scope_contract,
            "request_contract": scope_contract,
            "supported_historical_scopes": [
                FAULTSEG_REPRESENTATIVE_SCOPE,
                "automatic_valid_roi",
                "debug_crop",
            ],
            "historical_debug_crop_supports_crop": True,
            "debug_crop_requires_explicit_scope": True,
            "disk_backed_full_volume": True,
            "formal_scope": "full_volume",
            "formal_patch_size_zyx": list(_FAULTSEG_FULL_VOLUME_PATCH_ZYX),
            "formal_overlap_zyx": list(_FAULTSEG_FULL_VOLUME_OVERLAP_ZYX),
            "disk_backed_probability_reconstruction": True,
            "budget_failure_policy": "explicit_error",
            "cuda_patch_fallback": False,
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del options
        scope_contract = faultseg_execution_scope_contract(self.model_id)
        inline_count = (
            int(np.unique(geometry.inline).size) if geometry.inline is not None else 0
        )
        crossline_count = (
            int(np.unique(geometry.crossline).size)
            if geometry.crossline is not None
            else 0
        )
        shape = (int(geometry.samples_per_trace), inline_count, crossline_count)
        is_3d = inline_count > 1 and crossline_count > 1
        full_volume_ready = is_3d and all(
            available >= minimum
            for available, minimum in zip(
                shape, _FAULTSEG_FULL_VOLUME_PATCH_ZYX, strict=True
            )
        )
        center_block_ready = bool(
            self.model_id == FAULTSEG_MODEL_ID and full_volume_ready
        )
        available_scopes = [
            *(
                [FAULTSEG_CENTER_BLOCK_SCOPE]
                if center_block_ready
                else []
            ),
            *([FAULTSEG_FULL_VOLUME_SCOPE] if full_volume_ready else []),
        ]
        ready = bool(available_scopes)
        if center_block_ready:
            reason = "三维网格可在工区三轴中心执行1个完整128³块，也可选择全区预测"
        elif full_volume_ready:
            reason = "三维网格可按128³窗口、64³重叠重建完整工区断层概率体"
        elif not is_3d:
            reason = "FaultSeg 需要同时形成 Inline 和 Crossline 网格的三维 SEG-Y"
        else:
            reason = (
                "FaultSeg 推理要求每个轴至少容纳一个"
                f"{_FAULTSEG_FULL_VOLUME_PATCH_ZYX}训练上下文；当前数据体为{shape}"
            )
        return {
            "ready": ready,
            "reason": reason,
            "adapter": type(self).__name__,
            "expected_axes": ["Z", "INLINE", "CROSSLINE"],
            "patch_size": list(self.spec.patch_size),
            "shape_zyx": list(shape),
            "default_scope": scope_contract["default_value"],
            "default_scope_ready": (
                center_block_ready
                if scope_contract["default_value"] == FAULTSEG_CENTER_BLOCK_SCOPE
                else full_volume_ready
            ),
            "available_scopes": available_scopes,
            "supported_scopes": list(scope_contract["allowed_values"]),
            "scope_options": [dict(item) for item in scope_contract["options"]],
            "scope_contract": scope_contract,
            "request_contract": scope_contract,
            "formal_scope": "full_volume",
            "patch_size_zyx": list(_FAULTSEG_FULL_VOLUME_PATCH_ZYX),
            "overlap_zyx": list(_FAULTSEG_FULL_VOLUME_OVERLAP_ZYX),
            "weighted_blending": True,
            "formal_input_ready": full_volume_ready,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        reader = SegyReader(
            request.source,
            self.config,
            request.options.get("segy", {"profile": "standard_3d"}),
        )
        geometry = reader.inspect()
        if geometry.inline is None or geometry.crossline is None:
            raise ValueError("FaultSeg requires a regular 3D SEG-Y volume")
        shape = (
            geometry.samples_per_trace,
            len(np.unique(geometry.inline)),
            len(np.unique(geometry.crossline)),
        )
        faultseg_config = self.model_config
        requested_scope = str(
            request.options.get("_faultseg_scope_resolved")
            or request.options.get("faultseg_scope")
            or (
                "debug_crop"
                if request.crop_start or request.crop_size
                else faultseg_execution_scope_contract(self.model_id)[
                    "default_value"
                ]
            )
        ).strip().casefold()
        scope_aliases = {
            "auto_roi": "automatic_valid_roi",
            "automatic_roi": "automatic_valid_roi",
            "legacy_crop": "debug_crop",
        }
        requested_scope = scope_aliases.get(requested_scope, requested_scope)
        if requested_scope == FAULTSEG_REPRESENTATIVE_SCOPE:
            raise ValueError(
                "FaultSeg representative_grid_128 is historical read-only scope; "
                "new predictions must use center_block_1 or full_volume"
            )
        if requested_scope not in {
            "full_volume",
            FAULTSEG_CENTER_BLOCK_SCOPE,
            "automatic_valid_roi",
            "debug_crop",
        }:
            raise ValueError(f"unsupported FaultSeg scope: {requested_scope}")
        if requested_scope == FAULTSEG_CENTER_BLOCK_SCOPE and self.model_id != FAULTSEG_MODEL_ID:
            raise ValueError(
                "FaultSeg center_block_1 is only supported by the "
                "registered FaultSeg checkpoint"
            )
        if requested_scope in {
            "full_volume",
            FAULTSEG_CENTER_BLOCK_SCOPE,
        } and (
            request.crop_start is not None or request.crop_size is not None
        ):
            raise ValueError(
                f"FaultSeg {requested_scope} scope cannot include crop_start/crop_size"
            )

        runtime_patch = tuple(
            int(value)
            for value in request.options.get(
                "_faultseg_runtime_patch_size", self.spec.patch_size
            )
        )
        if len(runtime_patch) != 3:
            raise ValueError("FaultSeg runtime patch must be a Z/Inline/Xline triple")
        if requested_scope == "full_volume":
            if self.spec.patch_size != _FAULTSEG_FULL_VOLUME_PATCH_ZYX:
                raise ValueError("FaultSeg full_volume patch_size must be 128x128x128")
            if self.spec.overlap != _FAULTSEG_FULL_VOLUME_OVERLAP_ZYX:
                raise ValueError("FaultSeg full_volume overlap must be 64x64x64")
            if runtime_patch != _FAULTSEG_FULL_VOLUME_PATCH_ZYX:
                raise ValueError(
                    "FaultSeg full_volume patch is fixed to the checkpoint training "
                    f"context {_FAULTSEG_FULL_VOLUME_PATCH_ZYX}; received {runtime_patch}"
                )
        elif requested_scope == FAULTSEG_CENTER_BLOCK_SCOPE:
            if self.spec.patch_size != _FAULTSEG_CENTER_BLOCK_SHAPE_ZYX:
                raise ValueError(
                    "FaultSeg center block patch_size must be 128x128x128"
                )
        formal_minimum = _FAULTSEG_FULL_VOLUME_PATCH_ZYX
        if requested_scope != "debug_crop" and any(
            available < minimum for available, minimum in zip(shape, formal_minimum)
        ):
            raise ValueError(
                "FaultSeg formal inference requires every source/ROI axis to contain "
                f"the checkpoint training context {formal_minimum}; source is {shape}"
            )

        safety_factor = float(faultseg_config.get("minimum_free_space_factor", 1.10))
        hard_limit = int(
            faultseg_config.get(
                "maximum_full_volume_working_bytes", 64 * 1024**3
            )
        )
        if hard_limit <= 0:
            raise ValueError("FaultSeg maximum_full_volume_working_bytes must be positive")
        staging_value = request.options.get("_faultseg_staging_directory")
        staging_directory = (
            Path(str(staging_value)).expanduser().resolve()
            if staging_value not in (None, "")
            else None
        )
        if staging_directory is not None:
            staging_directory.mkdir(parents=True, exist_ok=True)
            disk_free_before = int(shutil.disk_usage(staging_directory).free)
        else:
            disk_free_before = None
        full_working_bytes = _faultseg_working_bytes(shape, safety_factor)
        full_gate_reasons: list[str] = []
        if full_working_bytes > hard_limit:
            full_gate_reasons.append("configured_working_set_limit")
        if disk_free_before is not None and full_working_bytes > disk_free_before:
            full_gate_reasons.append("insufficient_output_volume_free_space")

        if requested_scope == "full_volume" and full_gate_reasons:
            details = ",".join(full_gate_reasons)
            raise RuntimeError(
                "FaultSeg full_volume requires the complete disk-backed working set; "
                f"required_bytes={full_working_bytes}, "
                f"configured_limit_bytes={hard_limit}, "
                f"disk_free_bytes={disk_free_before}, reasons={details}. "
                "Automatic ROI fallback is disabled for formal inference."
            )

        if requested_scope == FAULTSEG_REPRESENTATIVE_SCOPE:
            if runtime_patch != _FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX:
                raise ValueError(
                    "FaultSeg representative grid is fixed to 128x128x128 blocks"
                )
            if staging_directory is None:
                raise RuntimeError(
                    "FaultSeg representative grid requires a disk staging directory"
                )
            plan = _faultseg_representative_grid_plan(geometry, shape)
            grid_working_bytes = int(
                _faultseg_working_bytes(
                    _FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX,
                    safety_factor,
                )
                * _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
            )
            if grid_working_bytes > hard_limit:
                raise RuntimeError(
                    "FaultSeg representative grid exceeds the configured working-set limit"
                )
            if (
                disk_free_before is not None
                and grid_working_bytes > disk_free_before
            ):
                raise RuntimeError(
                    "FaultSeg representative grid exceeds available disk-backed working space"
                )
            trace_grid_context = _faultseg_trace_grid(geometry)
            staged_blocks: list[dict[str, Any]] = []
            input_root = staging_directory / "representative_grid_inputs"
            for planned_block in plan["blocks"]:
                block = dict(planned_block)
                block_directory = input_root / str(block["block_id"])
                input_path = block_directory / "input_volume.npy"
                valid_mask_path = block_directory / "valid_mask.npy"
                start = tuple(int(value) for value in block["source_start_zyx"])
                staged, valid_traces, _, _ = _materialize_faultseg_npy(
                    reader,
                    input_path,
                    start_zyx=start,
                    size_zyx=_FAULTSEG_REPRESENTATIVE_BLOCK_SHAPE_ZYX,
                    trace_grid_context=trace_grid_context,
                )
                evidence = _faultseg_block_input_evidence(staged, valid_traces)
                np.save(valid_mask_path, valid_traces.astype(np.uint8, copy=False))
                staged.flush()
                staged._mmap.close()
                block.update(evidence)
                block["_staged_input_volume_npy"] = str(input_path.resolve())
                block["_staged_valid_mask_npy"] = str(valid_mask_path.resolve())
                staged_blocks.append(block)
            display_scale = max(
                (float(block.get("input_abs_p99") or 0.0) for block in staged_blocks),
                default=0.0,
            )
            if not np.isfinite(display_scale) or display_scale <= 0.0:
                display_scale = 1.0
                display_scale_fallback = "no_positive_finite_amplitude"
            else:
                display_scale_fallback = None
            for block in staged_blocks:
                block["display_amplitude_scale"] = display_scale
            plan["blocks"] = staged_blocks
            plan["source_segy"] = str(reader.path)
            plan["display_amplitude_scale"] = display_scale
            plan["display_amplitude_scale_receipt"] = {
                "algorithm": "maximum_of_128_block_abs_p99_histograms_v2",
                "block_quantile": 0.99,
                "block_count": len(staged_blocks),
                "finite_sample_count": int(
                    sum(
                        int(block.get("finite_voxel_count") or 0)
                        for block in staged_blocks
                    )
                ),
                "fallback": display_scale_fallback,
                "model_input_normalization_modified": False,
            }
            provenance = {
                "source": str(reader.path),
                "model_order": ["Z", "INLINE", "CROSSLINE"],
                "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
                "geometry_profile": geometry.profile,
                "geometry_confidence": geometry.confidence,
                "geometry_issues": list(geometry.issues),
                "source_shape_zyx": list(shape),
                "scope": FAULTSEG_REPRESENTATIVE_SCOPE,
                "scope_reason": "fixed_8x4x4_representative_sampling",
                "full_volume_requested": False,
                "full_volume_executed": False,
                "materialization": "128_independent_disk_backed_npy_blocks",
                "materialized_volume_npy": None,
                "working_set_estimate_bytes": grid_working_bytes,
                "working_set_hard_limit_bytes": hard_limit,
                "disk_free_bytes_before": disk_free_before,
                "representative_grid": plan,
            }
            return ModelInputBatch(
                model_id=self.model_id,
                array=None,
                valid_mask=None,
                axes=("BLOCK", "Z", "INLINE", "CROSSLINE"),
                provenance=provenance,
            )

        roi_selection = None
        center_block_plan: dict[str, Any] | None = None
        if requested_scope == "debug_crop":
            crop_size = request.crop_size or tuple(
                min(available, patch)
                for available, patch in zip(shape, runtime_patch)
            )
            scope = "debug_crop"
            scope_reason = "explicit_diagnostic_crop"
        elif requested_scope == FAULTSEG_CENTER_BLOCK_SCOPE:
            center_block_plan = _faultseg_center_block_plan(geometry, shape)
            crop_start = tuple(
                int(value) for value in center_block_plan["source_start_zyx"]
            )
            crop_size = _FAULTSEG_CENTER_BLOCK_SHAPE_ZYX
            scope = FAULTSEG_CENTER_BLOCK_SCOPE
            scope_reason = "deterministic_three_axis_survey_center_block"
        elif requested_scope == "automatic_valid_roi":
            configured_roi = tuple(
                int(value)
                for value in faultseg_config.get(
                    "automatic_roi_size", (1024, 256, 256)
                )
            )
            if len(configured_roi) != 3:
                raise ValueError(
                    "FaultSeg automatic_roi_size must be a Z/Inline/Xline triple"
                )
            crop_size = tuple(
                min(available, requested)
                for available, requested in zip(shape, configured_roi)
            )
            if any(
                size < minimum
                for size, minimum in zip(crop_size, formal_minimum)
            ):
                raise ValueError(
                    "FaultSeg automatic ROI cannot contain the checkpoint training "
                    f"context {formal_minimum}; candidate ROI is {crop_size}"
                )
            roi_working_bytes = _faultseg_working_bytes(crop_size, safety_factor)
            if roi_working_bytes > hard_limit or (
                disk_free_before is not None
                and roi_working_bytes > disk_free_before
            ):
                raise RuntimeError(
                    "FaultSeg automatic ROI exceeds the configured or available "
                    "disk-backed working-set budget; refusing an unsafe hidden crop"
                )
            roi_selection = select_unlabeled_seismic_roi(
                reader,
                size_zyx=crop_size,
                minimum_valid_trace_fraction=float(
                    faultseg_config.get("minimum_valid_trace_fraction", 0.5)
                ),
                minimum_active_trace_fraction=float(
                    faultseg_config.get("minimum_active_trace_fraction", 0.5)
                ),
                minimum_nonzero_sample_fraction=float(
                    faultseg_config.get("minimum_nonzero_sample_fraction", 0.01)
                ),
                minimum_finite_sample_fraction=float(
                    faultseg_config.get("minimum_finite_sample_fraction", 0.99)
                ),
                maximum_probe_traces=int(
                    faultseg_config.get("maximum_probe_traces", 16)
                ),
                maximum_candidate_probes=int(
                    faultseg_config.get("maximum_candidate_probes", 128)
                ),
            )
            crop_start = roi_selection.start_zyx
            scope = "automatic_valid_roi"
            scope_reason = (
                "explicit_label_free_active_roi"
                if requested_scope == "automatic_valid_roi"
                else "full_volume_working_set_gate:" + ",".join(full_gate_reasons)
            )
        else:
            crop_size = shape
            crop_start = (0, 0, 0)
            scope = "full_volume"
            scope_reason = "complete_regular_segy_grid"

        crop_size = tuple(int(value) for value in crop_size)
        if any(size < 1 for size in crop_size):
            raise ValueError(f"FaultSeg crop/ROI must be non-empty: {crop_size}")
        if any(size > available for size, available in zip(crop_size, shape)):
            raise ValueError(
                f"FaultSeg crop {crop_size} exceeds seismic volume {shape}"
            )
        if requested_scope == "debug_crop":
            if any(size < patch for size, patch in zip(crop_size, runtime_patch)):
                raise ValueError(
                    f"FaultSeg debug crop {crop_size} is smaller than patch {runtime_patch}"
                )
            crop_start = request.crop_start or tuple(
                (available - size) // 2
                for available, size in zip(shape, crop_size)
            )
        if any(
            start < 0 or start + size > available
            for start, size, available in zip(crop_start, crop_size, shape)
        ):
            raise ValueError("FaultSeg crop start/size is outside the seismic volume")

        working_bytes = _faultseg_working_bytes(crop_size, safety_factor)
        if requested_scope == FAULTSEG_CENTER_BLOCK_SCOPE and (
            working_bytes > hard_limit
            or (
                disk_free_before is not None
                and working_bytes > disk_free_before
            )
        ):
            raise RuntimeError(
                "FaultSeg center_block_1 exceeds the configured or available "
                "disk-backed working-set budget"
            )
        in_memory_limit = int(
            faultseg_config.get("api_process_in_memory_bytes", 256 * 1024**2)
        )
        materialized_path: Path | None = None
        if staging_directory is not None:
            materialized_path = staging_directory / "faultseg_input_volume.npy"
            array, valid_traces, inline_values, crossline_values = (
                _materialize_faultseg_npy(
                    reader,
                    materialized_path,
                    start_zyx=crop_start,
                    size_zyx=crop_size,
                )
            )
            materialization = "disk_backed_npy_streamed_from_segy"
        elif _faultseg_voxel_count(crop_size) * 4 <= in_memory_limit:
            z, y, x = crop_start
            dz, dy, dx = crop_size
            volume: FaultSegVolume = build_faultseg_volume(
                reader,
                sample_slice=slice(z, z + dz),
                inline_slice=slice(y, y + dy),
                crossline_slice=slice(x, x + dx),
            )
            array = volume.data
            valid_traces = volume.valid_traces
            inline_values = volume.inline_values
            crossline_values = volume.crossline_values
            materialization = "bounded_in_memory_debug_volume"
        else:
            raise RuntimeError(
                "FaultSeg volume exceeds the API-process in-memory limit and no "
                "disk staging directory was supplied by the prediction runner"
            )
        provenance = {
            "source": str(reader.path),
            "model_order": ["Z", "INLINE", "CROSSLINE"],
            "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
            "geometry_profile": geometry.profile,
            "geometry_confidence": geometry.confidence,
            "geometry_issues": list(geometry.issues),
            "source_shape_zyx": list(shape),
            "crop_start_zyx": list(crop_start),
            "crop_size_zyx": list(crop_size),
            "roi_start_zyx": list(crop_start),
            "roi_size_zyx": list(crop_size),
            "scope": scope,
            "scope_reason": scope_reason,
            "full_volume_requested": requested_scope == "full_volume",
            "full_volume_executed": scope == "full_volume",
            "coverage_fraction": float(
                _faultseg_voxel_count(crop_size) / _faultseg_voxel_count(shape)
            ),
            "working_set_estimate_bytes": working_bytes,
            "full_volume_working_set_estimate_bytes": full_working_bytes,
            "working_set_hard_limit_bytes": hard_limit,
            "disk_free_bytes_before": disk_free_before,
            "materialization": materialization,
            "materialized_volume_npy": (
                str(materialized_path) if materialized_path is not None else None
            ),
            "center_block": center_block_plan,
            "unlabeled_roi_selection": (
                roi_selection.receipt if roi_selection is not None else None
            ),
            "inline_range": [
                int(inline_values[0]),
                int(inline_values[-1]),
            ],
            "crossline_range": [
                int(crossline_values[0]),
                int(crossline_values[-1]),
            ],
            "valid_trace_fraction": float(valid_traces.mean()),
        }
        return ModelInputBatch(
            model_id=self.model_id,
            array=array,
            valid_mask=valid_traces,
            axes=("Z", "INLINE", "CROSSLINE"),
            provenance=provenance,
        )


def _resolved_header_byte(geometry: Any, field: str) -> int | None:
    prefix = f"{field}_byte="
    for issue in getattr(geometry, "issues", ()):
        text = str(issue)
        if not text.startswith(prefix):
            continue
        try:
            return int(text[len(prefix) :].split(":", 1)[0])
        except (TypeError, ValueError):
            return None
    return None


def _positive_integer(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 and parsed == value else None


def _platform_ordered_regular_grid(
    inline: np.ndarray | None,
    crossline: np.ndarray | None,
    inline_count: int,
    crossline_count: int,
) -> bool:
    """Return whether trace order can be reshaped safely as [Inline, Crossline]."""
    if (
        inline is None
        or crossline is None
        or inline_count <= 1
        or crossline_count <= 1
        or inline.size != inline_count * crossline_count
        or crossline.size != inline.size
    ):
        return False
    inline_grid = inline.reshape(inline_count, crossline_count)
    crossline_grid = crossline.reshape(inline_count, crossline_count)
    inline_values = inline_grid[:, 0]
    crossline_values = crossline_grid[0]
    return bool(
        np.all(inline_grid == inline_values[:, None])
        and np.all(crossline_grid == crossline_values[None, :])
        and np.unique(inline_values).size == inline_count
        and np.unique(crossline_values).size == crossline_count
        and np.all(np.diff(inline_values) > 0)
        and np.all(np.diff(crossline_values) > 0)
    )


class SurfaceSegInputAdapter:
    """Preflight the native full-volume SEG-Y input used by seismic_surface_seg.

    The upstream model intentionally owns amplitude scaling and slice
    materialization.  Returning ``array=None`` avoids reading a potentially
    multi-gigabyte cube twice while retaining a stable adapter/provenance gate.
    """

    model_id = "seismic_surface_seg"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = dict(config.get("surface_seg", {}))

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_formats": ["sgy", "segy"],
            "array_axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "tensor_axes": ["N", "C", "SAMPLE", "CROSSLINE"],
            "dtype": "float32",
            "patch_size": [],
            "overlap": [],
            "normalization": "training-range or robust 1%/99% scaling, then 512×512 model normalization",
            "requires_logs": False,
            "input_scope": "full_volume",
            "supports_crop": False,
            "model_image_size": list(self.spec.get("model_image_size", (512, 512))),
            "native_options": [
                "amplitude_mode",
                "query_threshold",
                "mask_threshold",
                "segformer_batch_size",
                "mask2former_batch_size",
                "inline_count",
                "inline_byte",
                "crossline_byte",
                "max_inlines",
                "write_mask_sgy",
            ],
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = options or {}
        inline = (
            None
            if geometry.inline is None
            else np.asarray(geometry.inline, dtype=np.int64)
        )
        crossline = (
            None
            if geometry.crossline is None
            else np.asarray(geometry.crossline, dtype=np.int64)
        )
        inline_count = int(np.unique(inline).size) if inline is not None else 0
        crossline_count = int(np.unique(crossline).size) if crossline is not None else 0
        trace_count = int(geometry.trace_count)
        sample_count = int(geometry.samples_per_trace)
        inline_byte = _resolved_header_byte(geometry, "inline")
        crossline_byte = _resolved_header_byte(geometry, "crossline")
        standard_headers = inline_byte == 189 and crossline_byte == 193
        is_3d = inline_count > 1 and crossline_count > 1
        duplicate_count = 0
        if is_3d and inline is not None and crossline is not None:
            pair_count = int(
                np.unique(np.column_stack((inline, crossline)), axis=0).shape[0]
            )
            duplicate_count = max(0, trace_count - pair_count)
        resolved_header_grid = bool(
            is_3d
            and inline_byte is not None
            and crossline_byte is not None
            and duplicate_count == 0
            and sample_count > 1
        )

        fallback_inline_count = _positive_integer(
            options.get("inline_count", self.spec.get("inline_count"))
        )
        fallback_ready = bool(
            fallback_inline_count
            and trace_count % int(fallback_inline_count) == 0
            and trace_count // int(fallback_inline_count) > 1
        )
        regular_ready = bool(
            is_3d and standard_headers and duplicate_count == 0 and sample_count > 1
        )
        platform_ordered_grid = bool(
            is_3d
            and duplicate_count == 0
            and sample_count > 1
            and trace_count == inline_count * crossline_count
            and _platform_ordered_regular_grid(
                inline,
                crossline,
                inline_count,
                crossline_count,
            )
        )
        ready = (
            regular_ready
            or fallback_ready
            or platform_ordered_grid
            or resolved_header_grid
        )
        recommended_options: dict[str, Any] = {}
        native_inline_count: int | None = None
        if regular_ready:
            reason = "标准 189/193 道头形成无重复的三维后叠加网格，满足地层分割输入要求"
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "standard_headers"
        elif fallback_ready:
            fallback_crossline_count = trace_count // int(fallback_inline_count)
            reason = f"将按显式 inline_count={fallback_inline_count} 重建三维后叠加网格"
            shape_ics = [
                int(fallback_inline_count),
                fallback_crossline_count,
                sample_count,
            ]
            geometry_mode = "explicit_inline_count"
            native_inline_count = int(fallback_inline_count)
        elif resolved_header_grid:
            reason = (
                f"平台已识别三维道头 {inline_byte}/{crossline_byte}，"
                "将直接复用该网格运行地层分割"
            )
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = (
                "standard_headers"
                if standard_headers
                else "platform_resolved_headers"
            )
            recommended_options = {
                "inline_byte": int(inline_byte),
                "crossline_byte": int(crossline_byte),
            }
        elif platform_ordered_grid:
            reason = (
                f"平台已验证 {inline_count} × {crossline_count} 有序完整网格；"
                f"将自动向模型传递 inline_count={inline_count}"
            )
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "platform_inferred_inline_count"
            native_inline_count = inline_count
            recommended_options = {"inline_count": inline_count}
        elif duplicate_count:
            reason = f"检测到 {duplicate_count} 条重复 Inline/Crossline 道，疑似叠前数据，模型不支持"
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "unsupported_duplicate_traces"
        elif is_3d and not standard_headers:
            reason = "平台已识别三维网格，但未能确认可复用的 Inline/Crossline 道头"
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "nonstandard_headers"
        else:
            reason = "地层分割需要规则三维后叠加 SEG-Y；无有效道头时必须显式提供 inline_count"
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "unresolved"
        return {
            "ready": ready,
            "reason": reason,
            "adapter": type(self).__name__,
            "expected_axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "patch_size": [],
            "shape_ics": shape_ics,
            "source_shape_zyx": [shape_ics[2], shape_ics[0], shape_ics[1]],
            "geometry_mode": geometry_mode,
            "duplicate_trace_count": duplicate_count,
            "platform_ordered_grid": platform_ordered_grid,
            "native_inline_count": native_inline_count,
            "recommended_options": recommended_options,
            "supports_crop": False,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        segy_options = dict(
            request.options.get("segy", {"profile": "standard_3d"})
        )
        explicit_inline_byte = request.options.get("inline_byte")
        explicit_crossline_byte = request.options.get("crossline_byte")
        if (explicit_inline_byte is None) != (explicit_crossline_byte is None):
            raise ValueError("SurfaceSeg inline_byte and crossline_byte must be provided together")
        if explicit_inline_byte is not None:
            segy_options.update(
                {
                    "inline_byte": int(explicit_inline_byte),
                    "crossline_byte": int(explicit_crossline_byte),
                }
            )
        reader = SegyReader(
            request.source,
            self.config,
            segy_options,
        )
        geometry = reader.inspect()
        compatibility = self.compatibility(geometry, options=request.options)
        if not compatibility["ready"]:
            raise ValueError(str(compatibility["reason"]))

        shape_ics = tuple(int(value) for value in compatibility["shape_ics"])
        valid_grid = np.ones(shape_ics[:2], dtype=bool)
        if (
            geometry.inline is not None
            and geometry.crossline is not None
            and int(compatibility.get("duplicate_trace_count", 0)) == 0
        ):
            inline = np.asarray(geometry.inline, dtype=np.int64)
            crossline = np.asarray(geometry.crossline, dtype=np.int64)
            inline_values = np.unique(inline)
            crossline_values = np.unique(crossline)
            inline_lookup = {
                int(value): index for index, value in enumerate(inline_values)
            }
            crossline_lookup = {
                int(value): index for index, value in enumerate(crossline_values)
            }
            valid_grid = np.zeros(
                (len(inline_values), len(crossline_values)), dtype=bool
            )
            for inline_value, crossline_value in zip(inline, crossline, strict=True):
                valid_grid[
                    inline_lookup[int(inline_value)],
                    crossline_lookup[int(crossline_value)],
                ] = True

        roi_selection = None
        selection_settings = request.options.get("_unlabeled_roi_selection")
        if isinstance(selection_settings, Mapping):
            source_shape_zyx = (shape_ics[2], shape_ics[0], shape_ics[1])
            defaults = selection_settings.get("default_size_zyx", source_shape_zyx)
            if not isinstance(defaults, (list, tuple)) or len(defaults) != 3:
                raise ValueError("default unlabeled ROI size must be a Z/Inline/Xline triple")
            configured_size = request.crop_size
            option_counts = (
                request.options.get("t_count"),
                request.options.get("inline_count"),
                request.options.get("crossline_count"),
            )
            size_values: list[int] = []
            for axis, (available, option_count, default_count) in enumerate(
                zip(source_shape_zyx, option_counts, defaults, strict=True)
            ):
                if bool(selection_settings.get("full_time")) and axis == 0:
                    count = available
                elif option_count is not None:
                    count = int(option_count)
                elif configured_size is not None:
                    count = int(configured_size[axis])
                elif default_count is None:
                    count = available
                else:
                    count = min(available, int(default_count))
                size_values.append(count)

            explicit_start: list[int | None] = (
                list(request.crop_start)
                if request.crop_start is not None
                else [None, None, None]
            )
            for axis, name in enumerate(
                ("t_start", "inline_start", "crossline_start")
            ):
                if request.options.get(name) is not None:
                    explicit_start[axis] = int(request.options[name])
            if bool(selection_settings.get("full_time")):
                explicit_start[0] = 0
            roi_selection = select_unlabeled_seismic_roi(
                reader,
                size_zyx=tuple(size_values),
                explicit_start_zyx=tuple(explicit_start),
                minimum_valid_trace_fraction=float(
                    selection_settings.get("minimum_valid_trace_fraction", 0.5)
                ),
                minimum_active_trace_fraction=float(
                    selection_settings.get("minimum_active_trace_fraction", 0.5)
                ),
                minimum_nonzero_sample_fraction=float(
                    selection_settings.get(
                        "minimum_nonzero_sample_fraction", 0.01
                    )
                ),
                minimum_finite_sample_fraction=float(
                    selection_settings.get(
                        "minimum_finite_sample_fraction", 0.99
                    )
                ),
                maximum_probe_traces=int(
                    selection_settings.get("maximum_probe_traces", 16)
                ),
                maximum_candidate_probes=int(
                    selection_settings.get("maximum_candidate_probes", 128)
                ),
            )

        if request.options.get("_collect_applicability_profile"):
            if roi_selection is None:
                applicability_observations = observe_seismic_reader(reader)
            else:
                applicability_observations = observe_seismic_reader(
                    reader,
                    trace_indices=roi_selection.trace_indices,
                    inline_count=roi_selection.size_zyx[1],
                    crossline_count=roi_selection.size_zyx[2],
                )
        else:
            applicability_observations = None
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=valid_grid,
            axes=("INLINE", "CROSSLINE", "SAMPLE"),
            provenance={
                "source": str(request.source),
                "shape_ics": list(shape_ics),
                "source_shape_zyx": [shape_ics[2], shape_ics[0], shape_ics[1]],
                "trace_count": int(geometry.trace_count),
                "sample_interval_ms": float(geometry.sample_interval),
                "geometry_profile": str(geometry.profile),
                "geometry_mode": compatibility["geometry_mode"],
                "native_inline_count": compatibility.get("native_inline_count"),
                "recommended_options": compatibility.get("recommended_options", {}),
                "missing_grid_cell_count": int((~valid_grid).sum()),
                "valid_trace_fraction": float(valid_grid.mean()),
                "materialization": "model_native_segy_reader",
                "requested_crop_ignored": bool(request.crop_start or request.crop_size),
                "applicability_observations": applicability_observations,
                "unlabeled_roi_selection": (
                    roi_selection.receipt if roi_selection is not None else None
                ),
                "resolved_roi_start_zyx": (
                    list(roi_selection.start_zyx)
                    if roi_selection is not None
                    else None
                ),
                "resolved_roi_size_zyx": (
                    list(roi_selection.size_zyx)
                    if roi_selection is not None
                    else None
                ),
            },
        )


class GeobodySegyInputAdapter(SurfaceSegInputAdapter):
    """Validate full-volume SEG-Y without materialising it in the API process.

    The frozen P17 Channel/Karst runtime owns its deterministic tiled SEG-Y
    reader.  Reusing the platform's geometry preflight keeps the UI honest
    while avoiding a second multi-gigabyte cube in memory.
    """

    def __init__(self, config: dict[str, Any], model_id: str) -> None:
        super().__init__(config)
        self.model_id = model_id

    def capabilities(self) -> dict[str, Any]:
        capability = super().capabilities()
        capability.update(
            {
                "model_id": self.model_id,
                "array_axes": ["TWT", "INLINE", "XLINE"],
                "tensor_axes": ["N", "C", "TWT", "INLINE", "XLINE"],
                "patch_size": [64, 96, 96],
                "overlap": [32, 48, 48],
                "normalization": "P17 per-tile robust seismic normalization",
                "native_options": [
                    "minimum_voxels",
                    "iline_byte",
                    "xline_byte",
                ],
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
        inline_byte = _resolved_header_byte(geometry, "inline")
        crossline_byte = _resolved_header_byte(geometry, "crossline")
        has_grid_headers = bool(
            inline_byte
            and crossline_byte
            and int(result.get("duplicate_trace_count", 0)) == 0
            and int(result.get("shape_ics", [0, 0, 0])[0]) > 1
            and int(result.get("shape_ics", [0, 0, 0])[1]) > 1
        )
        recommended_options = dict(result.get("recommended_options", {}))
        for inherited_option in ("inline_count", "inline_byte", "crossline_byte"):
            recommended_options.pop(inherited_option, None)
        if has_grid_headers:
            recommended_options.update(
                {"iline_byte": int(inline_byte), "xline_byte": int(crossline_byte)}
            )
            for field, option_name in (
                ("x", "x_byte"),
                ("y", "y_byte"),
                ("coordinate_scalar", "coordinate_scalar_byte"),
            ):
                resolved_byte = _resolved_header_byte(geometry, field)
                if resolved_byte is not None:
                    recommended_options[option_name] = int(resolved_byte)
        result.update(
            {
                "ready": has_grid_headers,
                "adapter": type(self).__name__,
                "expected_axes": ["TWT", "INLINE", "XLINE"],
                "patch_size": [64, 96, 96],
                "recommended_options": recommended_options,
            }
        )
        if has_grid_headers:
            result["reason"] = "规则三维 SEG-Y 满足地质体候选推理输入要求"
        else:
            result["reason"] = "地质体候选模型需要可解析且无重复的 Inline/Crossline 道头"
        return result

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        batch = super().prepare(request)
        batch.model_id = self.model_id
        batch.axes = ("TWT", "INLINE", "XLINE")
        batch.provenance["materialization"] = "wellfuse_p17_native_tiled_segy_reader"
        return batch


class HorizonP17InputAdapter(GeobodySegyInputAdapter):
    """Preflight the label-free inputs of the P17 local horizon-event picker."""

    model_id = "wellfuse_horizon_p17"

    _forbidden_option_fragments = (
        "target_horizon",
        "target_surface",
        "validation_label",
        "test_label",
        "time_depth_table",
        "checkshot",
        "td_table",
    )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.model_id)

    def capabilities(self) -> dict[str, Any]:
        capability = super().capabilities()
        capability.update(
            {
                "model_id": self.model_id,
                "array_axes": ["TWT", "INLINE", "XLINE"],
                "tensor_axes": [
                    "BATCH",
                    "HORIZON",
                    "CHANNEL",
                    "LOCAL_TWT",
                    "INLINE",
                    "XLINE",
                ],
                "patch_size": ["full_local_time_window", 64, 64],
                "overlap": [0, 16, 16],
                "normalization": "per-local-event-window robust MAD/RMS scaling clipped to ±12",
                "requires_logs": False,
                "optional_inputs": [
                    "context_horizons_path (named fixed well-horizon controls)",
                    "registration_points_path (Align lineage; unnamed LAS points stay inert)",
                    "registration_manifest_path",
                ],
                "fallback": "label-free ordered seismic-event prior",
                "scientific_scope": "validated on Chengdu; unknown-survey output is experimental",
                "input_scope": "segy_spatial_roi",
                "supports_crop": True,
                "native_options": [
                    "inline_start",
                    "inline_count",
                    "crossline_start",
                    "crossline_count",
                    "iline_byte",
                    "xline_byte",
                    "x_byte",
                    "y_byte",
                    "coordinate_scalar_byte",
                ],
            }
        )
        return capability

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        geometry_options = dict(options or {})
        # ``inline_count`` is an ROI extent for Horizon P17.  The inherited
        # SurfaceSeg adapter uses the same name for headerless-grid recovery;
        # never let a bounded request fabricate source geometry.
        geometry_options.pop("inline_count", None)
        result = super().compatibility(geometry, options=geometry_options)
        if result["ready"]:
            result["reason"] = (
                "规则三维TWT SEG-Y可进入层位局部事件候选推理；"
                "无命名上下文层位点时将显式使用label-free地震事件先验"
            )
        else:
            result["reason"] = (
                "层位候选模型需要可解析、无重复的Inline/Crossline三维TWT SEG-Y"
            )
        result["adapter"] = type(self).__name__
        result["patch_size"] = ["local_time_window", 64, 64]
        result["unknown_survey_status"] = "experimental"
        return result

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in options.items():
            folded = str(key).casefold()
            if value not in (None, "", [], {}) and any(
                fragment in folded for fragment in cls._forbidden_option_fragments
            ):
                raise ValueError(
                    f"P17 Horizon rejects target/supervision option: {key}"
                )

        path_records: dict[str, Any] = {}
        for key in (
            "context_horizons_path",
            "registration_points_path",
            "registration_manifest_path",
        ):
            raw_path = options.get(key)
            if raw_path in (None, ""):
                continue
            path = Path(str(raw_path)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{key} not found: {path}")
            if key != "registration_manifest_path" and path.suffix.casefold() not in {
                ".csv",
                ".json",
            }:
                raise ValueError(f"{key} must be CSV or JSON: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            path_records[key] = {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }

        inline_points = options.get("registration_points")
        if inline_points not in (None, ""):
            if not isinstance(inline_points, list) or not all(
                isinstance(row, dict) for row in inline_points
            ):
                raise ValueError("registration_points must be a list of objects")
            canonical = json.dumps(
                inline_points,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            path_records["registration_points"] = {
                "row_count": len(inline_points),
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }

        registered_lineage = options.get(
            "registration_source"
        ) == "wellfuse_align_prediction" or bool(options.get("registration_task_id"))
        if (
            options.get("context_horizons_path") not in (None, "")
            and not registered_lineage
        ):
            raise ValueError(
                "context_horizons_path requires registration_source=wellfuse_align_prediction"
            )
        return path_records

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        audited_paths = self._validate_options(request.options)
        profiled_request = ModelInputRequest(
            source=request.source,
            crop_start=request.crop_start,
            crop_size=request.crop_size,
            options={
                **request.options,
                "_collect_applicability_profile": True,
                "_unlabeled_roi_selection": {
                    "full_time": True,
                    "default_size_zyx": [None, None, None],
                    "minimum_valid_trace_fraction": 0.5,
                    "minimum_active_trace_fraction": 0.5,
                    "minimum_nonzero_sample_fraction": 0.01,
                    "maximum_probe_traces": 16,
                },
            },
        )
        batch = super().prepare(profiled_request)
        batch.model_id = self.model_id
        batch.axes = ("TWT", "INLINE", "XLINE")
        batch.provenance.update(
            {
                "materialization": "wellfuse_p17_unknown_horizon_native_tiled_reader",
                "optional_input_audit": audited_paths,
                "registration_source": (
                    request.options.get("registration_source")
                    or (
                        "wellfuse_align_prediction"
                        if request.options.get("registration_task_id")
                        else None
                    )
                ),
                "target_surface_is_model_input": False,
                "time_depth_supervision_is_model_input": False,
                "unknown_survey_status": "experimental",
                "requested_crop_ignored": False,
                "spatial_roi_requested": bool(
                    request.crop_start
                    or request.crop_size
                    or any(
                        request.options.get(name) is not None
                        for name in (
                            "inline_start",
                            "inline_count",
                            "crossline_start",
                            "crossline_count",
                        )
                    )
                ),
            }
        )
        return batch


class Facies3DSegyInputAdapter(GeobodySegyInputAdapter):
    """Preflight a regular SEG-Y for the real P17 weak 3-D facies checkpoint."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, "wellfuse_facies_3d_p17")

    def capabilities(self) -> dict[str, Any]:
        capability = super().capabilities()
        capability.update(
            {
                "model_id": self.model_id,
                "patch_size": [96, 64, 64],
                "overlap": [48, 32, 32],
                "normalization": "P17 robust trace normalization plus label-free global TWT coordinate",
                "requires_logs": False,
                "requires_registration": False,
                "scientific_scope": "weak_supervision_candidate_only",
                "chengdu_default_header_bytes": {"iline": 9, "xline": 21},
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
        result["adapter"] = type(self).__name__
        result["patch_size"] = [96, 64, 64]
        if result["ready"]:
            result["reason"] = "规则三维SEG-Y可执行弱监督三维地震相实验候选推理"
        return result

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        batch = super().prepare(request)
        recommended = batch.provenance.get("recommended_options", {})
        if not isinstance(recommended, Mapping):
            recommended = {}
        batch.provenance.update(
            {
                "materialization": "wellfuse_p17_facies_3d_native_tiled_segy_reader",
                "facies_label_open_count": 0,
                "dense_3d_accuracy_claimed": False,
                "unknown_survey_status": "experimental",
                "facies_3d_header_bytes": {
                    "iline": int(recommended.get("iline_byte", 9)),
                    "xline": int(recommended.get("xline_byte", 21)),
                    "source": (
                        "adapter_detection"
                        if "iline_byte" in recommended
                        and "xline_byte" in recommended
                        else "audited_chengdu_default"
                    ),
                },
                "facies_3d_geometry_contract": {
                    "shape_t_inline_xline": list(
                        map(int, batch.provenance["source_shape_zyx"])
                    ),
                    "trace_count": int(batch.provenance["trace_count"]),
                    "spatial_dimensions_must_exceed_one": True,
                },
            }
        )
        return batch


class F3Facies3DSegyInputAdapter(GeobodySegyInputAdapter):
    """Preflight SEG-Y for the dense F3 2.5-D U-Net transfer runner."""

    model_id = "wellfuse_facies_3d_f3_fast"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.model_id)

    def capabilities(self) -> dict[str, Any]:
        capability = super().capabilities()
        capability.update(
            {
                "model_id": self.model_id,
                "array_axes": ["TWT", "INLINE", "XLINE"],
                "tensor_axes": ["N", "CHANNEL", "XLINE", "TWT"],
                "patch_size": [],
                "overlap": [],
                "normalization": "per-trace robust normalization clipped to [-1,1]",
                "requires_logs": False,
                "requires_registration": False,
                "input_scope": "segy_roi",
                "supports_crop": True,
                "supports_single_trace_roi": True,
                "single_trace_contract": {
                    "inline_count": 1,
                    "crossline_count": 1,
                },
                "native_options": [
                    "t_start",
                    "t_count",
                    "inline_start",
                    "inline_count",
                    "crossline_start",
                    "crossline_count",
                    "iline_byte",
                    "xline_byte",
                    "batch_size",
                ],
                "validated_scope": "F3_dense_benchmark",
                "unknown_survey_status": "experimental_transfer_candidate",
            }
        )
        return capability

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        geometry_options = dict(options or {})
        # Here ``inline_count`` is the ROI size.  SurfaceSeg uses the same
        # option for a headerless full-volume fallback, which would turn a
        # 1x1 trace request into a fake one-inline source geometry.
        geometry_options.pop("inline_count", None)
        result = super().compatibility(geometry, options=geometry_options)
        result.update(
            {
                "adapter": type(self).__name__,
                "expected_axes": ["TWT", "INLINE", "XLINE"],
                "patch_size": [],
                "validated_scope": "F3_dense_benchmark",
                "runtime_scope": "unknown_survey_transfer_candidate",
            }
        )
        if result["ready"]:
            result["reason"] = (
                "规则三维SEG-Y可执行稠密基准模型ROI迁移候选推理；"
                "定量验证范围仍限公开参考工区"
            )
        return result

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        configured_segy = dict(
            request.options.get("segy", {"profile": "standard_3d"})
        )
        explicit_iline = request.options.get(
            "iline_byte", configured_segy.get("inline_byte")
        )
        explicit_xline = request.options.get(
            "xline_byte", configured_segy.get("crossline_byte")
        )
        if explicit_iline is not None or explicit_xline is not None:
            header_candidates = (
                (
                    int(explicit_iline if explicit_iline is not None else 9),
                    int(explicit_xline if explicit_xline is not None else 21),
                ),
            )
        else:
            # Chengdu's audited bytes are tried first, followed by the two
            # geometry layouts observed in the validation assets.  The mixed
            # 189/21 layout is used by Reservoir.  Every candidate must still
            # pass the no-duplicate, two-spatial-dimension geometry gate, so a
            # coincidental byte match cannot silently become a usable volume.
            header_candidates = ((9, 21), (189, 193), (189, 21))

        batch: ModelInputBatch | None = None
        last_error: ValueError | None = None
        resolved_headers: tuple[int, int] | None = None
        for iline_byte, xline_byte in header_candidates:
            segy_options = {
                **configured_segy,
                "inline_byte": iline_byte,
                "crossline_byte": xline_byte,
            }
            prepared_request = ModelInputRequest(
                source=request.source,
                crop_start=request.crop_start,
                crop_size=request.crop_size,
                options={
                    **request.options,
                    "segy": segy_options,
                    "_unlabeled_roi_selection": {
                        "full_time": False,
                        "default_size_zyx": [256, 64, 64],
                        "minimum_valid_trace_fraction": 0.5,
                        "minimum_active_trace_fraction": 0.5,
                        "minimum_nonzero_sample_fraction": 0.01,
                        "maximum_probe_traces": 16,
                    },
                },
            )
            try:
                batch = super().prepare(prepared_request)
            except ValueError as exc:
                last_error = exc
                continue
            resolved_headers = (iline_byte, xline_byte)
            break
        if batch is None or resolved_headers is None:
            assert last_error is not None
            raise last_error
        batch.model_id = self.model_id
        batch.axes = ("TWT", "INLINE", "XLINE")
        batch.provenance.update(
            {
                "materialization": "wellfuse_f3_facies3d_native_segy_roi_reader",
                "validated_scope": "F3_dense_benchmark",
                "runtime_scope": "unknown_survey_transfer_candidate",
                "dense_3d_accuracy_claimed_for_current_survey": False,
                "facies_3d_header_bytes": {
                    "iline": resolved_headers[0],
                    "xline": resolved_headers[1],
                    "source": (
                        "explicit"
                        if explicit_iline is not None or explicit_xline is not None
                        else "deterministic_candidate_probe"
                    ),
                },
            }
        )
        return batch


WELLFUSE_UNKNOWN_WELL_MODEL_IDS = (
    "wellfuse_facies_1d_p17",
    "wellfuse_den_p18",
    "wellfuse_por_p18",
    "wellfuse_log_perm_p18",
    "wellfuse_sw_p18",
    "wellfuse_vsh_p18",
)
ARCHIVED_WELL_PROPERTY_COMPLETION_MODEL_IDS = frozenset(
    WELLFUSE_UNKNOWN_WELL_MODEL_IDS[1:]
)

DATASET_BOUND_FLUID_MODEL_ID = "wellfuse_fluid_interpretation_fast"
DATASET_BOUND_FACIES_1D_MODEL_ID = "wellfuse_facies_1d_chengdu_fast"
DATASET_BOUND_NORTHWEST_PROPERTY_MODEL_IDS = (
    "wellfuse_den_northwest_fast",
    "wellfuse_por_northwest_fast",
    "wellfuse_log_perm_northwest_fast",
    "wellfuse_sw_northwest_fast",
    "wellfuse_vsh_northwest_fast",
)
SNAPSHOT_ONLY_NORTHWEST_PROPERTY_MODEL_IDS = frozenset(
    DATASET_BOUND_NORTHWEST_PROPERTY_MODEL_IDS
)
FRACTURE_DEVELOPMENT_MODEL_ID = "wellfuse_fracture_development_utah_fast"
SNAPSHOT_ONLY_DOWNSTREAM_WELL_MODEL_IDS = frozenset(
    {
        DATASET_BOUND_FLUID_MODEL_ID,
        DATASET_BOUND_FACIES_1D_MODEL_ID,
        WELLFUSE_UNKNOWN_WELL_MODEL_IDS[0],
        FRACTURE_DEVELOPMENT_MODEL_ID,
        *SNAPSHOT_ONLY_NORTHWEST_PROPERTY_MODEL_IDS,
    }
)
DATASET_BOUND_MODEL_IDS = (
    DATASET_BOUND_FLUID_MODEL_ID,
    DATASET_BOUND_FACIES_1D_MODEL_ID,
    *DATASET_BOUND_NORTHWEST_PROPERTY_MODEL_IDS,
)
RAW_WELL_MODEL_IDS = (*DATASET_BOUND_MODEL_IDS, FRACTURE_DEVELOPMENT_MODEL_ID)
NORTHWEST_DATASET_IDS = ("northwest_all", "northwest_oil", "northwest_coal")
FLUID_DATASET_IDS = (*NORTHWEST_DATASET_IDS, "chengdu")
DATASET_BOUND_ALLOWED_DATASETS = {
    DATASET_BOUND_FLUID_MODEL_ID: FLUID_DATASET_IDS,
    DATASET_BOUND_FACIES_1D_MODEL_ID: ("chengdu",),
}
DATASET_BOUND_SELECTIONS = {
    DATASET_BOUND_FLUID_MODEL_ID: "fixed_last",
    DATASET_BOUND_FACIES_1D_MODEL_ID: (
        "fixed_three_seed_probability_plus_training_fold_viterbi"
    ),
    **{
        model_id: "fixed_last"
        for model_id in DATASET_BOUND_NORTHWEST_PROPERTY_MODEL_IDS
    },
}


def _validated_path_list(
    value: Any,
    *,
    option_name: str,
    allowed_extensions: frozenset[str],
) -> list[Path]:
    if value in (None, "", []):
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{option_name} must be a list of local paths")
    paths: list[Path] = []
    for raw in value:
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{option_name} file not found: {path}")
        if path.suffix.casefold() not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise ValueError(f"{option_name} supports only {allowed}: {path}")
        paths.append(path)
    return list(dict.fromkeys(paths))


_TRAJECTORY_NAME_MARKERS = (
    "trajectory",
    "trajectories",
    "deviation",
    "directional_survey",
    "directionalsurvey",
    "well_path",
    "wellpath",
    "轨迹",
    "测斜",
)
_WELLHEAD_NAME_MARKERS = (
    "well_head",
    "well_heads",
    "wellhead",
    "wellheads",
    "well_collar",
    "wellcollar",
    "井位",
    "井口",
)

# These suffixes define the platform's safe binding into the stable raw-well
# CLI.  A preparation parser may recover a WellHead from another format
# (notably Petrel DEV or OpenDtect WELL), but that semantic result does not
# make the source file a safe ``--wellhead-path`` transport.  The CLI processes
# an aggregate candidate set for every LAS; forwarding one compound per-well
# WELL from a mixed TRACK/WELL snapshot would therefore reject unrelated LAS
# files.  Keep DEV/WELL available as trajectory evidence and reserve the
# well-head transport for aggregate tables with row-level identity matching.
_RAW_WELL_TRAJECTORY_SUFFIXES = frozenset(
    {
        ".csv",
        ".txt",
        ".tsv",
        ".prn",
        ".path",
        ".dev",
        ".well",
        ".track",
        ".xlsx",
        ".xlsm",
    }
)
_RAW_WELL_WELLHEAD_SUFFIXES = frozenset(
    {".csv", ".txt", ".tsv", ".xlsx", ".xlsm"}
)

_WELLHEAD_IDENTITY_COLUMNS = frozenset(
    {
        "API",
        "APINUMBER",
        "COMMONWELLNAME",
        "UWI",
        "WELL",
        "WELLID",
        "WELLNAME",
        "WELLNUMBER",
        "井名",
        "井号",
        "井名称",
    }
)
_WELLHEAD_X_COLUMNS = frozenset(
    {
        "X",
        "EASTING",
        "SURFACEX",
        "WELLHEADX",
        "WELLTOPX",
        "XCOORD",
        "XCOORDINATE",
        "XCRD",
        "X坐标",
        "东坐标",
        "井口X",
        "井口X坐标",
    }
)
_WELLHEAD_Y_COLUMNS = frozenset(
    {
        "Y",
        "NORTHING",
        "SURFACEY",
        "WELLHEADY",
        "WELLTOPY",
        "YCOORD",
        "YCOORDINATE",
        "YCRD",
        "Y坐标",
        "北坐标",
        "井口Y",
        "井口Y坐标",
    }
)
_TRAJECTORY_STRUCTURAL_COLUMNS = frozenset(
    {
        "AZIMUTH",
        "DEVIATION",
        "GRIDAZIMUTH",
        "INCLINATION",
        "MD",
        "MEASUREDDEPTH",
        "STATION",
        "STATIONINDEX",
        "TVD",
        "TVDSS",
        "XOFFSET",
        "YOFFSET",
        "方位角",
        "井斜角",
        "测深",
    }
)
_WELLHEAD_SEMANTIC_COLUMNS: dict[str, frozenset[str]] = {
    "x": _WELLHEAD_X_COLUMNS,
    "y": _WELLHEAD_Y_COLUMNS,
    "kb": frozenset(
        {
            "DATUMELEVATION",
            "EKB",
            "KB",
            "KBELEV",
            "KBELEVATION",
            "KBELEVATIONMSL",
            "RTELEVATION",
            "WELLDATUM",
            "KB高程",
            "补心海拔",
            "补心高",
            "补心高程",
        }
    ),
    "ground_elevation": frozenset(
        {
            "GL",
            "GLEV",
            "GLELEVATION",
            "GROUNDELEVATION",
            "GROUNDELEVATIONMSL",
            "GROUNDLEVEL",
            "地面标高",
            "地面海拔",
            "地面高程",
        }
    ),
    "total_depth": frozenset(
        {
            "COMPLETIONDEPTH",
            "TD",
            "TOTALDEPTH",
            "TOTALDEPTHMD",
            "完钻井深",
            "完钻深度",
            "总井深",
            "终孔深度",
        }
    ),
}
_WELLHEAD_CONTRACT_SEMANTIC_COLUMNS: dict[str, frozenset[str]] = {
    "datum_unit": frozenset(
        {"DATUMUNIT", "ELEVATIONUNIT", "VERTICALUNIT", "高程单位"}
    ),
    "horizontal_axis_order": frozenset(
        {"AXISORDER", "HORIZONTALAXISORDER", "坐标轴顺序"}
    ),
    "horizontal_crs_id": frozenset(
        {
            "COORDINATEREFERENCESYSTEM",
            "CRS",
            "CRSID",
            "HORIZONTALCRS",
            "HORIZONTALCRSID",
            "SOURCECRS",
            "坐标参考系",
            "坐标系",
        }
    ),
    "vertical_reference": frozenset(
        {
            "DATUMTYPE",
            "ELEVATIONDATUMTYPE",
            "VERTICALREFERENCE",
            "高程基准类型",
            "垂向基准",
        }
    ),
}
_WELLHEAD_CONTRACT_COLUMNS = frozenset(
    {
        alias
        for aliases in _WELLHEAD_CONTRACT_SEMANTIC_COLUMNS.values()
        for alias in aliases
    }
    | {"GROUNDELEVATIONSTATUS", "SOURCEFILE"}
)


def _single_column_index(
    normalized_header: Sequence[str], aliases: frozenset[str]
) -> int | None:
    indices = [
        index for index, value in enumerate(normalized_header) if value in aliases
    ]
    return indices[0] if len(indices) == 1 else None


def _metadata_identity_key(value: Any) -> str:
    return "".join(
        character
        for character in str(value or "").strip().upper()
        if character.isalnum()
    )


def _normalized_metadata_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return format(float(text), ".12g")
    except ValueError:
        return text.casefold()


def _text_wellhead_structure(path: Path) -> dict[str, Any]:
    """Prove that a text table is one well-head row per well.

    An accepted preparation role alone is insufficient because a trajectory
    table legitimately exposes X/Y/KB at every station.  The raw-well CLI,
    however, accepts only aggregate well-head tables.  ``MD``/``TVD`` alone do
    not prove a station table because aggregate head exports often use those
    names for terminal depths.  Reject trajectory-shaped repeated identities,
    while retaining one-row-per-well aggregate tables.
    """

    try:
        header, rows, _ = _rows_with_evidence(path)
    except Exception as exc:
        return {
            "structural_class": "unreadable_tabular_metadata",
            "wellhead_eligible": False,
            "reason": f"table_parse_failed:{type(exc).__name__}",
        }
    if not header:
        return {
            "structural_class": "headerless_metadata",
            "wellhead_eligible": False,
            "reason": "one_well_one_row_contract_requires_header",
        }
    normalized_header = [_normalized_column(value) for value in header]
    trajectory_columns = sorted(
        {value for value in normalized_header if value in _TRAJECTORY_STRUCTURAL_COLUMNS}
    )
    identity_index = _single_column_index(
        normalized_header, _WELLHEAD_IDENTITY_COLUMNS
    )
    x_index = _single_column_index(normalized_header, _WELLHEAD_X_COLUMNS)
    y_index = _single_column_index(normalized_header, _WELLHEAD_Y_COLUMNS)
    if identity_index is None or x_index is None or y_index is None:
        return {
            "structural_class": "non_wellhead_table",
            "wellhead_eligible": False,
            "reason": "identity_and_unique_xy_columns_required",
        }

    entity_rows: dict[str, list[list[Any]]] = {}
    unidentified_rows = 0
    for row in rows:
        if not any(str(value or "").strip() for value in row):
            continue
        identity = (
            _metadata_identity_key(row[identity_index])
            if identity_index < len(row)
            else ""
        )
        if not identity:
            unidentified_rows += 1
            continue
        entity_rows.setdefault(identity, []).append(row)
    repeated = sorted(
        identity for identity, values in entity_rows.items() if len(values) != 1
    )
    if not entity_rows or unidentified_rows or repeated:
        trajectory_shaped = bool(trajectory_columns and repeated)
        return {
            "structural_class": (
                "trajectory_station_table"
                if trajectory_shaped
                else "repeated_or_unidentified_well_rows"
            ),
            "wellhead_eligible": False,
            "reason": (
                "trajectory_columns_and_repeated_well_rows"
                if trajectory_shaped
                else "one_well_one_row_contract_failed"
            ),
            "entity_count": len(entity_rows),
            "unidentified_row_count": unidentified_rows,
            "repeated_entity_keys": repeated,
            "trajectory_columns": trajectory_columns,
        }

    semantic_indices: dict[str, int] = {}
    for semantic_field, aliases in _WELLHEAD_SEMANTIC_COLUMNS.items():
        index = _single_column_index(normalized_header, aliases)
        if index is not None:
            semantic_indices[semantic_field] = index
    contract_semantic_indices: dict[str, int] = {}
    for semantic_field, aliases in _WELLHEAD_CONTRACT_SEMANTIC_COLUMNS.items():
        index = _single_column_index(normalized_header, aliases)
        if index is not None:
            contract_semantic_indices[f"contract_{semantic_field}"] = index
    entity_signatures: dict[str, dict[str, str]] = {}
    for identity, entity_row in entity_rows.items():
        row = entity_row[0]
        signature = {
            field: _normalized_metadata_value(row[index])
            for field, index in {
                **semantic_indices,
                **contract_semantic_indices,
            }.items()
            if index < len(row) and _normalized_metadata_value(row[index])
        }
        entity_signatures[identity] = signature
    contract_columns = sorted(
        {value for value in normalized_header if value in _WELLHEAD_CONTRACT_COLUMNS}
    )
    return {
        "structural_class": "aggregate_one_well_one_row_head_table",
        "wellhead_eligible": True,
        "reason": "sealed_role_and_one_well_one_row_structure",
        "entity_count": len(entity_rows),
        "entity_keys": sorted(entity_rows),
        "entity_signatures": entity_signatures,
        "semantic_fields": sorted(semantic_indices),
        "contract_semantic_fields": sorted(contract_semantic_indices),
        "contract_columns": contract_columns,
        "trajectory_columns": trajectory_columns,
        # Mandatory identity/X/Y contribute equally.  Explicit datum/CRS/
        # lineage fields outrank a raw duplicate without relying on filenames.
        "authority_score": 3 + len(semantic_indices) + 2 * len(contract_columns),
    }


def _same_wellhead_entities(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_signatures = left.get("entity_signatures") or {}
    right_signatures = right.get("entity_signatures") or {}
    if set(left_signatures) != set(right_signatures) or not left_signatures:
        return False
    for identity in left_signatures:
        left_values = left_signatures[identity]
        right_values = right_signatures[identity]
        common_fields = set(left_values) & set(right_values)
        if not {"x", "y"}.issubset(common_fields):
            return False
        if any(left_values[field] != right_values[field] for field in common_fields):
            return False
    return True


def _select_authoritative_wellhead_paths(
    candidates: Sequence[dict[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Collapse semantically duplicate aggregate tables without path heuristics."""

    if len(candidates) <= 1:
        return [item["path"] for item in candidates], []
    entity_key_sets = [
        set(item["structure"].get("entity_keys") or []) for item in candidates
    ]
    if not all(keys == entity_key_sets[0] for keys in entity_key_sets[1:]):
        return [item["path"] for item in candidates], []
    if not all(
        _same_wellhead_entities(candidates[0]["structure"], item["structure"])
        for item in candidates[1:]
    ):
        paths = [str(item["path"]) for item in candidates]
        raise ValueError(
            "sealed snapshot has well-head tables for the same entities with "
            f"conflicting core values: {paths}"
        )

    highest_score = max(
        int(item["structure"].get("authority_score") or 0) for item in candidates
    )
    leaders = [
        item
        for item in candidates
        if int(item["structure"].get("authority_score") or 0) == highest_score
    ]
    if len(leaders) != 1:
        leader_hashes = {str(item.get("sha256") or "") for item in leaders}
        if len(leader_hashes) != 1 or "" in leader_hashes:
            paths = [str(item["path"]) for item in leaders]
            raise ValueError(
                "sealed snapshot has semantically equivalent well-head tables "
                f"with equal authority but different content: {paths}"
            )
    selected = leaders[0]
    exclusions = [
        {
            "path": str(item["path"]),
            "sha256": item.get("sha256") or None,
            "reason": "semantic_duplicate_lower_contract_authority"
            if item is not selected
            and int(item["structure"].get("authority_score") or 0) < highest_score
            else "byte_identical_semantic_duplicate",
            "selected_path": str(selected["path"]),
            "selected_sha256": selected.get("sha256") or None,
            "authority_score": int(
                item["structure"].get("authority_score") or 0
            ),
            "selected_authority_score": highest_score,
            "entity_keys": item["structure"].get("entity_keys") or [],
            "normalized_entity_signature_sha256": canonical_sha256(
                item["structure"].get("entity_signatures") or {}
            ),
        }
        for item in candidates
        if item is not selected
    ]
    return [selected["path"]], exclusions


def _snapshot_well_metadata_paths(
    options: Mapping[str, Any],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Classify only metadata already accepted by the sealed preparation run.

    The preparation report is the authority for mixed/aggregate workbooks and
    generic CSV names.  Format semantics remain a deterministic fallback for
    unambiguous per-well files (Petrel DEV and OpenDtect track/well).  This
    keeps a sealed snapshot usable by the well-side runners without asking the
    caller to repeat paths, while avoiding content guesses in this adapter.
    """

    trajectories: list[Path] = []
    wellhead_candidates: list[dict[str, Any]] = []
    role_bindings: list[dict[str, Any]] = []
    detected_roles: dict[str, set[str]] = {}
    detected_statuses: dict[str, str] = {}
    for item in options.get("snapshot_metadata_detection") or []:
        if not isinstance(item, Mapping) or not item.get("文件"):
            continue
        path = Path(str(item["文件"])).expanduser().resolve()
        roles = item.get("识别角色") or []
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            continue
        status = str(item.get("状态") or "").strip()
        path_key = str(path).casefold()
        detected_statuses[path_key] = status
        detected_roles[path_key] = {
            str(role).strip().casefold() for role in roles if str(role).strip()
        }
    for asset in options.get("snapshot_assets") or []:
        if not isinstance(asset, Mapping) or not asset.get("path"):
            continue
        if str(asset.get("role") or "").strip().casefold() != "well_metadata":
            continue
        path = Path(str(asset["path"])).expanduser().resolve()
        suffix = path.suffix.casefold()
        if suffix not in _RAW_WELL_TRAJECTORY_SUFFIXES:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"sealed snapshot well metadata not found: {path}")
        stem = path.stem.casefold().replace("-", "_").replace(" ", "_")
        path_key = str(path).casefold()
        sealed_status = detected_statuses.get(path_key, "")
        binding_allowed = not sealed_status or "已识别" in sealed_status
        roles = detected_roles.get(path_key, set()) if binding_allowed else set()
        filename_is_trajectory = any(
            marker in stem for marker in _TRAJECTORY_NAME_MARKERS
        )
        filename_is_wellhead = any(marker in stem for marker in _WELLHEAD_NAME_MARKERS)
        if (
            binding_allowed
            and suffix != ".well"
            and filename_is_trajectory
            and filename_is_wellhead
        ):
            raise ValueError(
                "sealed snapshot well metadata filename is ambiguous between "
                f"trajectory and well-head roles: {path}"
            )
        is_trajectory = bool(
            binding_allowed
            and (
                any("轨迹" in role or "trajectory" in role for role in roles)
                or suffix in {".dev", ".path", ".prn", ".track", ".well"}
                or filename_is_trajectory
            )
        )
        has_wellhead_semantics = bool(
            binding_allowed
            and (
                any(
                    "井位" in role or "wellhead" in role or "well head" in role
                    for role in roles
                )
                or suffix == ".well"
                or filename_is_wellhead
            )
        )
        if not binding_allowed:
            wellhead_structure = {
                "structural_class": "sealed_metadata_not_accepted",
                "wellhead_eligible": False,
                "reason": f"sealed_preparation_status:{sealed_status}",
            }
        elif (
            suffix in {".csv", ".txt", ".tsv"}
            and has_wellhead_semantics
            and not filename_is_trajectory
        ):
            wellhead_structure = _text_wellhead_structure(path)
        elif suffix in {".xlsx", ".xlsm"} and has_wellhead_semantics:
            # For workbooks the preparation parser is the structural
            # authority: it already selected the accepted sheet/table.  Text
            # tables can be rechecked cheaply and deterministically here.
            wellhead_structure = {
                "structural_class": "sealed_parser_accepted_workbook_head_table",
                "wellhead_eligible": True,
                "reason": "sealed_preparation_parser_role",
                "authority_score": 3,
            }
        elif suffix in {".csv", ".txt", ".tsv"} and is_trajectory:
            # Do not materialize a potentially very large station table merely
            # to prove again that an explicit trajectory is not a head table.
            wellhead_structure = {
                "structural_class": "trajectory_role_or_filename_excluded",
                "wellhead_eligible": False,
                "reason": "explicit_trajectory_not_reparsed_as_wellhead",
            }
        else:
            wellhead_structure = {
                "structural_class": "unsupported_wellhead_transport",
                "wellhead_eligible": False,
                "reason": "not_an_aggregate_wellhead_transport",
            }
        is_wellhead = bool(
            has_wellhead_semantics
            and suffix in _RAW_WELL_WELLHEAD_SUFFIXES
            and not filename_is_trajectory
            and wellhead_structure.get("wellhead_eligible")
        )
        if is_trajectory:
            trajectories.append(path)
        if is_wellhead:
            wellhead_candidates.append(
                {
                    "path": path,
                    "sha256": str(asset.get("sha256") or "").casefold(),
                    "structure": wellhead_structure,
                }
            )
        role_bindings.append(
            {
                "path": str(path),
                "sha256": str(asset.get("sha256") or "").casefold() or None,
                "sealed_status": sealed_status or None,
                "sealed_roles": sorted(roles),
                "bound_as_trajectory": bool(is_trajectory),
                "bound_as_wellhead": bool(is_wellhead),
                "wellhead_structural_class": wellhead_structure.get(
                    "structural_class"
                ),
                "wellhead_decision_reason": wellhead_structure.get("reason"),
                "wellhead_entity_count": wellhead_structure.get("entity_count"),
            }
        )
    wellheads, duplicate_exclusions = _select_authoritative_wellhead_paths(
        wellhead_candidates
    )
    receipt = {
        "contract_version": "well-seismic.raw-well-metadata-binding-receipt.v1",
        "authority_order": [
            "sealed_preparation_parser_role",
            "aggregate_one_well_one_row_structure",
            "explicit_contract_completeness",
            "sealed_content_identity",
        ],
        "role_bindings": role_bindings,
        "selected_trajectory_paths": [
            str(path) for path in dict.fromkeys(trajectories)
        ],
        "selected_wellhead_paths": [str(path) for path in wellheads],
        "selected_wellhead_assets": [
            {
                "path": str(item["path"]),
                "sha256": item.get("sha256") or None,
                "authority_score": int(
                    item["structure"].get("authority_score") or 0
                ),
                "entity_keys": item["structure"].get("entity_keys") or [],
                "normalized_entity_signature_sha256": canonical_sha256(
                    item["structure"].get("entity_signatures") or {}
                ),
            }
            for item in wellhead_candidates
            if item["path"] in set(wellheads)
        ],
        "wellhead_duplicate_exclusions": duplicate_exclusions,
    }
    return (
        list(dict.fromkeys(trajectories)),
        wellheads,
        receipt,
    )


def _raw_well_options(options: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_paths = _validated_path_list(
        options.get("raw_well_paths"),
        option_name="raw_well_paths",
        allowed_extensions=frozenset({".las", ".csv", ".txt"}),
    )
    trajectory_paths = _validated_path_list(
        options.get("trajectory_paths"),
        option_name="trajectory_paths",
        allowed_extensions=_RAW_WELL_TRAJECTORY_SUFFIXES,
    )
    wellhead_paths = _validated_path_list(
        options.get("wellhead_paths"),
        option_name="wellhead_paths",
        allowed_extensions=_RAW_WELL_WELLHEAD_SUFFIXES,
    )
    root_value = str(options.get("raw_well_root") or "").strip()
    raw_root = Path(root_value).expanduser().resolve() if root_value else None
    if raw_root is not None and not raw_root.is_dir():
        raise NotADirectoryError(f"raw_well_root directory not found: {raw_root}")
    source_mode = "explicit_raw"
    metadata_role_binding_receipt: dict[str, Any] | None = None
    if not raw_paths and raw_root is None:
        snapshot_paths: list[Path] = []
        for asset in options.get("snapshot_assets") or []:
            if not isinstance(asset, Mapping) or not asset.get("path"):
                continue
            role = str(asset.get("role") or "").strip().casefold()
            path = Path(str(asset["path"])).expanduser().resolve()
            if role not in {"well_log", "well_logs"}:
                continue
            if path.suffix.casefold() not in {".las", ".csv", ".txt"}:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"sealed snapshot well file not found: {path}")
            snapshot_paths.append(path)
        raw_paths = list(dict.fromkeys(snapshot_paths))
        if raw_paths:
            source_mode = "sealed_snapshot"
    if source_mode == "sealed_snapshot" and (
        not trajectory_paths or not wellhead_paths
    ):
        (
            inferred_trajectories,
            inferred_wellheads,
            metadata_role_binding_receipt,
        ) = _snapshot_well_metadata_paths(options)
        if not trajectory_paths:
            trajectory_paths = inferred_trajectories
        if not wellhead_paths:
            wellhead_paths = inferred_wellheads
    if not raw_paths and raw_root is None:
        return None
    return {
        "input_mode": "raw_wells",
        "source_mode": source_mode,
        "raw_well_paths": raw_paths,
        "raw_well_root": raw_root,
        "trajectory_paths": trajectory_paths,
        "wellhead_paths": wellhead_paths,
        "metadata_role_binding_receipt": metadata_role_binding_receipt,
    }


class DatasetBoundWellInputAdapter:
    """Adapt an explicit registered dataset and whole-well selection.

    These fast downstream models execute only against their existing dataset
    repositories.  They intentionally do not consume the request's SEG-Y or a
    well-seismic registration product.
    """

    def __init__(self, model_id: str) -> None:
        if model_id not in RAW_WELL_MODEL_IDS:
            raise ValueError(f"unsupported well-side model: {model_id}")
        self.model_id = model_id
        # Public platform runs for every downstream well task are snapshot-only.
        # The implementation can still read legacy raw/dataset envelopes when
        # replaying old offline fixtures; the API and worker reject those source
        # modes for every newly submitted platform task.
        self.public_snapshot_only = model_id in SNAPSHOT_ONLY_DOWNSTREAM_WELL_MODEL_IDS
        self.snapshot_only = model_id in SNAPSHOT_ONLY_NORTHWEST_PROPERTY_MODEL_IDS
        self.allowed_datasets = DATASET_BOUND_ALLOWED_DATASETS.get(model_id, ())
        self.selection = DATASET_BOUND_SELECTIONS.get(
            model_id, "utah_16b_fixed_depthblock"
        )

    def capabilities(self) -> dict[str, Any]:
        if self.public_snapshot_only:
            return {
                "model_id": self.model_id,
                "source_formats": ["sealed_snapshot"],
                "array_axes": ["MD"],
                "tensor_axes": ["N", "CHANNEL", "MD"],
                "dtype": "float32",
                "patch_size": [],
                "overlap": [],
                "normalization": "fixed property-prediction contract",
                "requires_logs": True,
                "requires_seismic": False,
                "requires_registration": False,
                "input_scope": "sealed_snapshot_whole_wells",
                "supports_crop": False,
                "allowed_datasets": [],
                "required_options_one_of": [["source_snapshot_id"]],
                "supports_raw_wells": False,
                "supports_snapshot_wells": True,
                "supports_dataset_bound": False,
                "source_policy": "sealed_snapshot_only",
                "selection": self.selection,
            }
        return {
            "model_id": self.model_id,
            "source_formats": ["las2", "csv", "txt", "registered_dataset"],
            "array_axes": ["MD"],
            "tensor_axes": ["N", "CHANNEL", "MD"],
            "dtype": "float32",
            "patch_size": [],
            "overlap": [],
            "normalization": "fixed dataset-specific fast downstream contract",
            "requires_logs": True,
            "requires_seismic": False,
            "requires_registration": False,
            "input_scope": "raw_whole_wells_or_registered_dataset",
            "supports_crop": False,
            "allowed_datasets": list(self.allowed_datasets),
            "required_options_one_of": [
                ["raw_well_paths"],
                ["raw_well_root"],
                ["source_snapshot_id"],
                ["dataset", "well_ids"],
            ],
            "supports_raw_wells": True,
            "supports_snapshot_wells": True,
            "supports_dataset_bound": bool(self.allowed_datasets),
            "selection": self.selection,
        }

    def _validated_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        if self.snapshot_only:
            forbidden = {
                "dataset",
                "well_ids",
                "raw_well_paths",
                "raw_well_root",
                "trajectory_paths",
                "wellhead_paths",
            }
            supplied = sorted(forbidden.intersection(options))
            if supplied:
                raise ValueError(
                    f"{self.model_id} accepts only well assets derived from the "
                    "current sealed SourceSnapshot; explicit source options are "
                    f"not allowed: {', '.join(supplied)}"
                )
            if not str(options.get("source_snapshot_id") or "").strip():
                raise ValueError(
                    f"{self.model_id} requires the current sealed SourceSnapshot"
                )
            snapshot = _raw_well_options(options)
            if snapshot is None or snapshot.get("source_mode") != "sealed_snapshot":
                raise ValueError(
                    f"{self.model_id} requires sealed well-log assets from the "
                    "current SourceSnapshot"
                )
            return snapshot
        raw = _raw_well_options(options)
        if raw is not None:
            return raw
        if not self.allowed_datasets:
            raise ValueError(
                f"{self.model_id} requires raw_well_paths or raw_well_root"
            )
        dataset = str(options.get("dataset", "")).strip().casefold()
        if dataset not in self.allowed_datasets:
            allowed = ", ".join(self.allowed_datasets)
            raise ValueError(
                f"dataset must be one of [{allowed}] for {self.model_id}"
            )
        raw_well_ids = options.get("well_ids")
        if not isinstance(raw_well_ids, Sequence) or isinstance(
            raw_well_ids, (str, bytes)
        ):
            raise TypeError("well_ids must be a non-empty list of well ids")
        well_ids = [str(value).strip() for value in raw_well_ids]
        if not well_ids or any(not value for value in well_ids):
            raise ValueError("well_ids must be a non-empty list of well ids")
        return {
            "input_mode": "registered_dataset",
            "dataset": dataset,
            "well_ids": well_ids,
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del geometry
        try:
            resolved = self._validated_options(options or {})
        except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as exc:
            return {
                "ready": False,
                "reason": str(exc),
                "adapter": type(self).__name__,
                "expected_axes": ["MD"],
                "patch_size": [],
                "requires_seismic": False,
                "requires_registration": False,
            }
        if resolved["input_mode"] == "raw_wells":
            raw_paths = resolved["raw_well_paths"]
            raw_root = resolved["raw_well_root"]
            return {
                "ready": True,
                "reason": (
                    f"已选择 {len(raw_paths)} 个原始井文件"
                    + (f"及目录 {raw_root}" if raw_root is not None else "")
                ),
                "adapter": type(self).__name__,
                "expected_axes": ["MD"],
                "patch_size": [],
                "input_mode": "raw_wells",
                "source_mode": resolved["source_mode"],
                "raw_well_count": len(raw_paths),
                "raw_well_root": str(raw_root) if raw_root is not None else None,
                "requires_seismic": False,
                "requires_registration": False,
            }
        dataset = str(resolved["dataset"])
        well_ids = list(resolved["well_ids"])
        return {
            "ready": True,
            "reason": f"{dataset} 已选择 {len(well_ids)} 口整井",
            "adapter": type(self).__name__,
            "expected_axes": ["MD"],
            "patch_size": [],
            "dataset": dataset,
            "well_count": len(well_ids),
            "requires_seismic": False,
            "requires_registration": False,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        resolved = self._validated_options(request.options)
        if resolved["input_mode"] == "raw_wells":
            raw_paths = list(resolved["raw_well_paths"])
            raw_root = resolved["raw_well_root"]
            trajectory_paths = list(resolved["trajectory_paths"])
            wellhead_paths = list(resolved["wellhead_paths"])
            return ModelInputBatch(
                model_id=self.model_id,
                array=None,
                valid_mask=None,
                axes=("MD",),
                provenance={
                    "input_mode": "raw_wells",
                    "source_mode": resolved["source_mode"],
                    "raw_well_paths": [str(path) for path in raw_paths],
                    "raw_well_root": str(raw_root) if raw_root is not None else None,
                    "trajectory_paths": [str(path) for path in trajectory_paths],
                    "wellhead_paths": [str(path) for path in wellhead_paths],
                    "metadata_role_binding_policy": {
                        "contract_version": (
                            "well-seismic.raw-well-metadata-role-binding.v2"
                        ),
                        "trajectory_suffixes": sorted(
                            _RAW_WELL_TRAJECTORY_SUFFIXES
                        ),
                        "wellhead_suffixes": sorted(_RAW_WELL_WELLHEAD_SUFFIXES),
                        "wellhead_scope": (
                            "sealed_role_plus_aggregate_one_well_one_row_only"
                        ),
                        "duplicate_policy": (
                            "same_entities_choose_unique_highest_contract_authority;"
                            "equal_authority_different_content_fail_closed"
                        ),
                    },
                    "metadata_role_binding_receipt": resolved.get(
                        "metadata_role_binding_receipt"
                    ),
                    "raw_well_count": len(raw_paths),
                    "materialization": "wellfuse_raw_well_cli",
                    "scope": "unknown_well_transfer",
                    "selection": self.selection,
                    "seismic_consumed": False,
                    "registration_consumed": False,
                },
            )
        dataset = str(resolved["dataset"])
        well_ids = list(resolved["well_ids"])
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=None,
            axes=("MD",),
            provenance={
                "input_mode": "registered_dataset",
                "dataset": dataset,
                "well_ids": well_ids,
                "well_count": len(well_ids),
                "materialization": "wellfuse_dataset_bound_selected_whole_wells",
                "scope": "within_dataset",
                "selection": self.selection,
                "seismic_consumed": False,
                "registration_consumed": False,
            },
        )


def _well_las_paths_from_options(options: dict[str, Any]) -> list[Path]:
    """Resolve explicit or sealed-snapshot LAS inputs without opening labels."""

    candidates: list[Path] = []
    explicit = options.get("well_las")
    if isinstance(explicit, dict):
        candidates.extend(
            Path(str(value)).expanduser().resolve() for value in explicit.values()
        )
    configured = options.get("las_paths")
    if isinstance(configured, (list, tuple)):
        candidates.extend(
            Path(str(value)).expanduser().resolve() for value in configured
        )
    for asset in options.get("snapshot_assets") or []:
        if not isinstance(asset, dict) or not asset.get("path"):
            continue
        role = str(asset.get("role", "")).casefold()
        path = Path(str(asset["path"])).expanduser().resolve()
        if "well" not in role and path.suffix.casefold() != ".las":
            continue
        candidates.append(path)
    resolved: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            resolved.update(
                path.resolve() for path in candidate.rglob("*.las") if path.is_file()
            )
        elif candidate.is_file() and candidate.suffix.casefold() == ".las":
            resolved.add(candidate)
    return sorted(resolved, key=lambda path: str(path).casefold())


_ALIGNED_PREPARED_VIEW_ROLES = (
    "canonical_well_las",
    "registration_manifest_v3",
    "registration_points_v3",
)


def _prepared_view_aligned_well_inputs(
    options: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve and re-attest the sealed aligned-well PreparedView contract."""

    view_id = str(options.get("prepared_view_id") or "").strip()
    if not view_id:
        if any(
            options.get(key)
            for key in (
                "prepared_view_manifest_path",
                "prepared_view_artifacts_by_role",
                "prepared_view_sha256",
            )
        ):
            raise ValueError("PreparedView identity is incomplete: view_id is missing")
        return None

    from ..prepared_view import validate_prepared_view_manifest
    from ..registration_contract import read_registration_product_v3

    manifest_value = str(options.get("prepared_view_manifest_path") or "").strip()
    manifest_sha256 = str(
        options.get("prepared_view_manifest_sha256") or ""
    ).strip()
    view_sha256 = str(options.get("prepared_view_sha256") or "").strip()
    if not manifest_value or not manifest_sha256 or not view_sha256:
        raise ValueError("PreparedView identity is incomplete: manifest hashes are required")
    source_snapshot_id = str(options.get("source_snapshot_id") or "").strip()
    source_snapshot_sha256 = str(
        options.get("source_snapshot_fingerprint") or ""
    ).strip()
    registration_task_id = str(options.get("registration_task_id") or "").strip()
    if not source_snapshot_id or not source_snapshot_sha256:
        raise ValueError("PreparedView source snapshot lineage is incomplete")
    if not registration_task_id:
        raise ValueError("PreparedView aligned-well input requires a registration task")

    validated = validate_prepared_view_manifest(
        manifest_value,
        expected_view_id=view_id,
        expected_source_snapshot_id=source_snapshot_id,
        expected_source_snapshot_sha256=source_snapshot_sha256,
    )
    if str(validated["manifest_sha256"]).casefold() != manifest_sha256.casefold():
        raise ValueError("PreparedView manifest SHA differs from the dispatched identity")
    if str(validated["view_sha256"]).casefold() != view_sha256.casefold():
        raise ValueError("PreparedView SHA differs from the dispatched identity")
    declared_kind = str(options.get("prepared_view_kind") or "").strip()
    if declared_kind and declared_kind != str(validated.get("kind") or ""):
        raise ValueError("PreparedView kind differs from its sealed manifest")

    registration_parents = {
        str(item.get("view_id"))
        for item in validated.get("parents") or []
        if isinstance(item, Mapping)
        and str(item.get("kind") or "") == "registration"
        and item.get("view_id")
    }
    if registration_parents != {registration_task_id}:
        raise ValueError("PreparedView and prediction registration task lineage mismatch")
    if str(options.get("prepared_view_registration_relation") or "") != "matched":
        raise ValueError("PreparedView registration relation was not attested as matched")

    validated_artifacts = {
        str(item.get("name") or ""): dict(item)
        for item in validated.get("artifacts") or []
        if isinstance(item, Mapping) and item.get("name")
    }
    role_map = options.get("prepared_view_artifacts_by_role")
    if not isinstance(role_map, Mapping):
        raise ValueError("PreparedView artifact role map is missing")

    selected_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in _ALIGNED_PREPARED_VIEW_ROLES:
        raw_items = role_map.get(role)
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raise ValueError(f"PreparedView required artifact role is missing: {role}")
        selected: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"PreparedView role {role} contains an invalid artifact")
            name = str(raw_item.get("name") or "")
            sealed = validated_artifacts.get(name)
            if sealed is None or str(sealed.get("role") or "") != role:
                raise ValueError(
                    f"PreparedView role map does not match sealed artifact: {role}/{name}"
                )
            for key in ("path", "sha256", "size", "schema_version"):
                if str(raw_item.get(key) or "") != str(sealed.get(key) or ""):
                    raise ValueError(
                        "PreparedView role-map artifact identity differs from manifest: "
                        f"{role}/{name}/{key}"
                    )
            selected.append(sealed)
        sealed_names = {
            name
            for name, item in validated_artifacts.items()
            if str(item.get("role") or "") == role
        }
        if {str(item["name"]) for item in selected} != sealed_names or not selected:
            raise ValueError(f"PreparedView role map is incomplete for: {role}")
        selected_by_role[role] = selected

    if len(selected_by_role["registration_manifest_v3"]) != 1:
        raise ValueError("PreparedView must contain exactly one Registration V3 manifest")
    if len(selected_by_role["registration_points_v3"]) != 1:
        raise ValueError("PreparedView must contain exactly one Registration V3 points file")
    for role in ("registration_manifest_v3", "registration_points_v3"):
        item = selected_by_role[role][0]
        if str(item.get("source_registration_task_id") or "") != registration_task_id:
            raise ValueError(
                f"PreparedView {role} was not produced by the requested registration task"
            )

    las_paths = sorted(
        {
            Path(str(item["path"])).expanduser().resolve()
            for item in selected_by_role["canonical_well_las"]
        },
        key=lambda path: str(path).casefold(),
    )
    if any(path.suffix.casefold() != ".las" for path in las_paths):
        raise ValueError("PreparedView canonical_well_las role contains a non-LAS file")
    registration_manifest = Path(
        str(selected_by_role["registration_manifest_v3"][0]["path"])
    ).expanduser().resolve()
    registration_points = Path(
        str(selected_by_role["registration_points_v3"][0]["path"])
    ).expanduser().resolve()
    product = read_registration_product_v3(registration_manifest)
    if product.points_path != registration_points:
        raise ValueError(
            "PreparedView Registration V3 manifest and points do not form one product"
        )
    expected_manifest_sha = str(
        options.get("registration_manifest_sha256") or ""
    ).strip()
    expected_points_sha = str(options.get("registration_points_sha256") or "").strip()
    if not expected_manifest_sha or not expected_points_sha:
        raise ValueError("PreparedView registration hash attestation is incomplete")
    if _file_sha256(registration_manifest).casefold() != expected_manifest_sha.casefold():
        raise ValueError("PreparedView registration manifest differs from registration task")
    if _file_sha256(registration_points).casefold() != expected_points_sha.casefold():
        raise ValueError("PreparedView registration points differ from registration task")

    role_names = {
        role: [str(item["name"]) for item in selected_by_role[role]]
        for role in _ALIGNED_PREPARED_VIEW_ROLES
    }
    artifacts_used = [
        {
            "role": role,
            "name": str(item["name"]),
            "path": str(Path(str(item["path"])).expanduser().resolve()),
            "sha256": str(item["sha256"]),
        }
        for role in _ALIGNED_PREPARED_VIEW_ROLES
        for item in selected_by_role[role]
    ]
    return {
        "las_paths": las_paths,
        "registration_manifest_path": registration_manifest,
        "registration_points_path": registration_points,
        "receipt": {
            "prepared_view_consumed": True,
            "prepared_view_id": view_id,
            "prepared_view_manifest_path": str(
                Path(str(validated["manifest_path"])).resolve()
            ),
            "prepared_view_manifest_sha256": str(validated["manifest_sha256"]),
            "prepared_view_sha256": str(validated["view_sha256"]),
            "prepared_view_kind": str(validated.get("kind") or ""),
            "prepared_view_input_contract": "aligned_well_sequence_v1",
            "prepared_view_roles_used": list(_ALIGNED_PREPARED_VIEW_ROLES),
            "prepared_view_artifact_names_used": role_names,
            "prepared_view_artifacts_used": artifacts_used,
        },
    }


class WellFuseUnknownWellInputAdapter:
    """Validate LAS + immutable registration lineage for P17/P18 1-D runners."""

    def __init__(self, model_id: str) -> None:
        if model_id not in WELLFUSE_UNKNOWN_WELL_MODEL_IDS:
            raise ValueError(f"unsupported WellFuse well-side model: {model_id}")
        self.model_id = model_id

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_formats": ["las2", "registration_points_csv"],
            "array_axes": ["MD"],
            "tensor_axes": ["N", "CHANNEL", "MD"],
            "dtype": "float32",
            "patch_size": [],
            "overlap": [],
            "normalization": "frozen P17/P18 model-specific training contract",
            "requires_logs": True,
            "requires_complete_trajectory": True,
            "requires_registration": True,
            "registration_source": "completed platform registration task",
            "registration_contract_version": "well-seismic.registration.v3",
            "canonical_vertical_coordinate": "z_msl_m",
            "target_is_model_input": False,
            "structural_feature_channels": 0,
            "input_scope": "registered_whole_wells",
            "prepared_view_policy": "preferred",
            "prepared_view_input_contract": "aligned_well_sequence_v1",
            "supports_raw_wells": False,
            "supports_snapshot_wells": True,
            "supports_dataset_bound": False,
            "source_policy": "sealed_snapshot_only",
            "supports_crop": False,
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del geometry
        options = dict(options or {})
        prepared_error: str | None = None
        try:
            prepared = _prepared_view_aligned_well_inputs(options)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            prepared = None
            prepared_error = str(exc)
        if prepared is not None:
            las_paths = list(prepared["las_paths"])
            registration = Path(prepared["registration_points_path"])
            manifest = Path(prepared["registration_manifest_path"])
        else:
            las_paths = _well_las_paths_from_options(options)
            registration = Path(
                str(options.get("registration_points_path", ""))
            ).expanduser()
            manifest = Path(
                str(options.get("registration_manifest_path", ""))
            ).expanduser()
        manifest_contract_version: str | None = None
        if manifest.is_file():
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                manifest_payload = None
            if isinstance(manifest_payload, dict):
                manifest_contract_version = str(
                    manifest_payload.get("contract_version") or ""
                )
        registration_ready = bool(
            registration.is_file()
            and manifest.is_file()
            and manifest_contract_version == "well-seismic.registration.v3"
        )
        ready = bool(las_paths and registration_ready and prepared_error is None)
        if prepared_error is not None:
            reason = prepared_error
        elif ready and prepared is not None:
            reason = "已验签PreparedView中的LAS与Registration V3，可执行井侧推理"
        elif ready:
            reason = "LAS与已完成井震标定成果齐全，可执行实验性未知工区井侧推理"
        elif not las_paths:
            reason = "需要数据快照中的LAS或options.las_paths/well_las"
        else:
            reason = "需要同一标定任务封存的Registration V3 points与manifest"
        return {
            "ready": ready,
            "reason": reason,
            "adapter": type(self).__name__,
            "expected_axes": ["MD"],
            "patch_size": [],
            "las_count": len(las_paths),
            "registration_ready": registration_ready,
            "registration_manifest_ready": manifest.is_file(),
            "registration_contract_version": manifest_contract_version,
            "prepared_view_consumed": bool(prepared is not None and ready),
            "prepared_view_error": prepared_error,
            "supports_crop": False,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        compatibility = self.compatibility(None, options=request.options)
        if not compatibility["ready"]:
            raise ValueError(str(compatibility["reason"]))
        prepared = _prepared_view_aligned_well_inputs(request.options)
        if prepared is not None:
            las_paths = list(prepared["las_paths"])
            registration = Path(prepared["registration_points_path"])
            registration_manifest = Path(prepared["registration_manifest_path"])
            prepared_receipt = dict(prepared["receipt"])
        else:
            las_paths = _well_las_paths_from_options(request.options)
            registration = (
                Path(str(request.options["registration_points_path"]))
                .expanduser()
                .resolve()
            )
            manifest_value = request.options.get("registration_manifest_path")
            registration_manifest = (
                Path(str(manifest_value)).expanduser().resolve()
                if manifest_value
                else None
            )
            prepared_receipt = {
                "prepared_view_consumed": False,
                "prepared_view_id": None,
                "prepared_view_roles_used": [],
            }
        if registration_manifest is not None and not registration_manifest.is_file():
            raise FileNotFoundError(registration_manifest)
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=None,
            axes=("MD",),
            provenance={
                "source_seismic": str(request.source),
                "las_paths": [str(path) for path in las_paths],
                "las_count": len(las_paths),
                "registration_points_path": str(registration),
                "registration_manifest_path": (
                    str(registration_manifest) if registration_manifest else None
                ),
                "registration_task_id": request.options.get("registration_task_id"),
                "source_snapshot_id": request.options.get("source_snapshot_id"),
                **prepared_receipt,
                "materialization": "wellfuse_subprocess_label_free_las_and_registration",
                "target_is_model_input": False,
                "time_depth_supervision_opened": False,
                "structural_feature_channels": 0,
            },
        )


class GeoPathTieInputAdapter:
    """Resolve the sealed SEG-Y and Registration V3 GeoPath contract.

    Registration V3 is the only trajectory authority consumed by the runtime.
    A separately uploaded trajectory is deliberately not accepted as a second,
    potentially divergent, geometry source.
    """

    model_id = "wellfuse_align_geopath_tie_v1"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_formats": ["sgy", "segy"],
            "array_axes": ["MD"],
            "tensor_axes": ["N", "CHANNEL", "MD"],
            "dtype": "float32",
            "patch_size": [],
            "overlap": [],
            "normalization": "frozen GeoPathTie-V1 checkpoint contract",
            "requires_seismic": True,
            "requires_logs": True,
            "requires_complete_trajectory": True,
            "requires_registration": True,
            "registration_source": "completed sealed Registration V3 task",
            "registration_contract_version": "well-seismic.registration.v3",
            "trajectory_authority": "registration_v3_points",
            "requires_sealed_segy_semantics": True,
            "segy_profile_source": (
                "source_snapshot_v3_semantics_or_verified_geometry_receipt"
            ),
            "target_is_model_input": False,
            "supports_snapshot_wells": True,
            "input_scope": "registered_whole_wells",
            "supports_crop": False,
        }

    @staticmethod
    def _trajectory_paths(options: Mapping[str, Any]) -> list[Path]:
        candidates: list[Path] = []
        explicit = options.get("trajectory_paths")
        if isinstance(explicit, (list, tuple)):
            candidates.extend(Path(str(item)).expanduser().resolve() for item in explicit)
        for asset in options.get("snapshot_assets") or []:
            if not isinstance(asset, Mapping) or not asset.get("path"):
                continue
            role = str(asset.get("role", "")).casefold()
            if any(token in role for token in ("trajectory", "deviation", "well_path", "dev")):
                candidates.append(Path(str(asset["path"])).expanduser().resolve())
        return sorted({path for path in candidates if path.is_file()}, key=lambda item: str(item).casefold())

    @staticmethod
    def _sealed_segy_contract(
        options: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, Mapping[str, Any] | None]:
        snapshot_semantics = options.get("source_snapshot_semantics")
        if not isinstance(snapshot_semantics, Mapping):
            raise ValueError("GeoPathTie requires sealed SourceSnapshot V3 semantics")
        expected_semantics_sha256 = str(
            options.get("source_snapshot_semantics_sha256") or ""
        ).casefold()
        if (
            len(expected_semantics_sha256) != 64
            or expected_semantics_sha256 != canonical_sha256(snapshot_semantics)
        ):
            raise ValueError("GeoPathTie source snapshot semantics hash mismatch")
        sealed_fields = (
            "segy_geometry_profile",
            "segy_inline_byte",
            "segy_crossline_byte",
            "segy_x_byte",
            "segy_y_byte",
            "segy_coordinate_scalar_byte",
        )
        populated = [
            field for field in sealed_fields if snapshot_semantics.get(field) is not None
        ]
        if populated and len(populated) != len(sealed_fields):
            raise ValueError(
                "GeoPathTie sealed SEG-Y semantics are partial; automatic evidence "
                "cannot override an incomplete explicit contract"
            )
        if len(populated) == len(sealed_fields):
            return (
                {field: snapshot_semantics[field] for field in sealed_fields},
                "sealed_explicit_snapshot_semantics",
                None,
            )

        seismic_path = Path(str(options.get("seismic_path") or "")).expanduser().resolve()
        matching_receipts = [
            receipt
            for receipt in (options.get("source_snapshot_segy_geometry_receipts") or [])
            if isinstance(receipt, Mapping)
            and receipt.get("source_asset_path")
            and Path(str(receipt["source_asset_path"])).expanduser().resolve()
            == seismic_path
        ]
        if len(matching_receipts) != 1:
            raise ValueError(
                "GeoPathTie requires exactly one verified automatic SEG-Y geometry "
                "receipt for the selected sealed asset"
            )
        receipt = matching_receipts[0]
        contract = validate_snapshot_segy_geometry_receipt(
            receipt,
            source_path=seismic_path,
            source_snapshot_id=str(options.get("source_snapshot_id") or ""),
            source_snapshot_fingerprint=str(
                options.get("source_snapshot_fingerprint") or ""
            ),
            snapshot_contract_version=str(
                options.get("snapshot_contract_version") or ""
            ),
            snapshot_assets=(options.get("snapshot_assets") or []),
        )
        return contract, "sealed_verified_automatic_geometry_receipt", receipt

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(options or {})
        seismic_path = Path(str(options.get("seismic_path") or "")).expanduser()
        points_path = Path(str(options.get("registration_points_path") or "")).expanduser()
        manifest_path = Path(str(options.get("registration_manifest_path") or "")).expanduser()
        las_paths = _well_las_paths_from_options(options)
        sealed_segy_error: str | None = None
        try:
            _contract, sealed_segy_authority, _receipt = self._sealed_segy_contract(
                options
            )
            sealed_segy_ready = bool(
                options.get("snapshot_contract_version")
                == "well-seismic.source-snapshot.v3"
                and options.get("source_snapshot_id")
            )
        except (OSError, TypeError, ValueError) as exc:
            sealed_segy_ready = False
            sealed_segy_authority = None
            sealed_segy_error = str(exc)
        contract_version: str | None = None
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                payload = None
            if isinstance(payload, Mapping):
                contract_version = str(payload.get("contract_version") or "")
        ready = bool(
            seismic_path.is_file()
            and seismic_path.suffix.casefold() in {".sgy", ".segy"}
            and points_path.is_file()
            and manifest_path.is_file()
            and las_paths
            and sealed_segy_ready
            and contract_version == "well-seismic.registration.v3"
        )
        reason = (
            "封存SEG-Y、LAS和Registration V3齐全；轨迹由Registration V3唯一提供"
            if ready
            else (
                "需要同一封存快照中的SEG-Y、逐井LAS、Registration V3，以及"
                "已封存的SEG-Y profile/Inline/Xline/XY/scalar字段"
            )
        )
        return {
            "ready": ready,
            "reason": reason,
            "adapter": type(self).__name__,
            "expected_axes": ["MD"],
            "patch_size": [],
            "trajectory_authority": "registration_v3_points",
            "las_count": len(las_paths),
            "sealed_segy_semantics_ready": sealed_segy_ready,
            "sealed_segy_geometry_ready": sealed_segy_ready,
            "sealed_segy_geometry_authority": sealed_segy_authority,
            "sealed_segy_geometry_error": sealed_segy_error,
            "registration_contract_version": contract_version,
            "registration_ready": bool(points_path.is_file() and manifest_path.is_file()),
            "supports_crop": False,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        options = dict(request.options)
        options["seismic_path"] = str(request.source)
        compatibility = self.compatibility(None, options=options)
        if not compatibility["ready"]:
            raise ValueError(str(compatibility["reason"]))
        from ..registration_contract import read_registration_product_v3

        manifest = Path(str(options["registration_manifest_path"])).expanduser().resolve()
        product = read_registration_product_v3(manifest)
        points = product.points_path
        seismic = request.source.expanduser().resolve()
        snapshot_semantics = dict(options["source_snapshot_semantics"])
        expected_semantics_sha256 = str(
            options.get("source_snapshot_semantics_sha256") or ""
        )
        observed_semantics_sha256 = canonical_sha256(snapshot_semantics)
        if expected_semantics_sha256 != observed_semantics_sha256:
            raise ValueError("GeoPathTie source snapshot semantics hash mismatch")
        sealed_segy_contract, geometry_authority, automatic_receipt = (
            self._sealed_segy_contract(options)
        )
        profile_name = str(sealed_segy_contract["segy_geometry_profile"])
        segy_config = dict(self.config.get("segy") or {})
        profile_definition = dict(
            (segy_config.get("profiles") or {}).get(profile_name) or {}
        )
        if not profile_definition:
            raise ValueError(f"GeoPathTie SEG-Y profile is not configured: {profile_name}")
        sealed_header_bytes = {
            "inline": int(sealed_segy_contract["segy_inline_byte"]),
            "crossline": int(sealed_segy_contract["segy_crossline_byte"]),
            "x": int(sealed_segy_contract["segy_x_byte"]),
            "y": int(sealed_segy_contract["segy_y_byte"]),
        }
        coordinate_scalar_byte = int(
            sealed_segy_contract["segy_coordinate_scalar_byte"]
        )
        profile_definition.update(
            {
                **{name: [value] for name, value in sealed_header_bytes.items()},
                "coordinate_scalar": coordinate_scalar_byte,
            }
        )
        requested_segy = {
            "profile": profile_name,
            **{f"{name}_byte": value for name, value in sealed_header_bytes.items()},
            "coordinate_scalar_byte": coordinate_scalar_byte,
        }
        reader = SegyReader(
            seismic,
            self.config,
            requested_segy,
        )
        geometry = reader.inspect()
        minimum_geometry_confidence = float(
            segy_config.get("minimum_geometry_confidence", 0.35)
        )
        if automatic_receipt is not None:
            minimum_geometry_confidence = max(
                minimum_geometry_confidence,
                GEOPATH_MINIMUM_GEOMETRY_CONFIDENCE,
                float(
                    automatic_receipt.get("minimum_geometry_confidence")
                    or GEOPATH_MINIMUM_GEOMETRY_CONFIDENCE
                ),
            )
        if geometry.confidence < minimum_geometry_confidence:
            raise ValueError(
                "GeoPathTie SEG-Y spatial header confidence is below the configured gate: "
                f"{geometry.confidence:.3f} < {minimum_geometry_confidence:.3f}"
            )
        if geometry.x is None or geometry.y is None:
            raise ValueError("GeoPathTie requires resolved SEG-Y XY coordinates")
        resolved_header_bytes: dict[str, int] = {}
        for issue in geometry.issues:
            if "_byte=" not in issue:
                continue
            field, _, remainder = issue.partition("_byte=")
            raw_byte, _, _ = remainder.partition(":")
            if field in {"inline", "crossline", "x", "y"}:
                resolved_header_bytes[field] = int(raw_byte)
        if set(resolved_header_bytes) != {"inline", "crossline", "x", "y"}:
            raise ValueError("GeoPathTie could not resolve all SEG-Y spatial header bytes")
        if resolved_header_bytes != sealed_header_bytes:
            raise ValueError(
                "GeoPathTie resolved SEG-Y bytes differ from sealed snapshot semantics"
            )
        geometry_identity = seismic_geometry_identity(geometry)
        matched_asset = next(
            (
                asset
                for asset in options.get("snapshot_assets") or []
                if isinstance(asset, Mapping)
                and asset.get("path")
                and Path(str(asset["path"])).expanduser().resolve() == seismic
            ),
            None,
        )
        expected_geometry_fingerprint = (
            str(matched_asset.get("geometry_fingerprint") or "")
            if isinstance(matched_asset, Mapping)
            else ""
        )
        if (
            expected_geometry_fingerprint
            and expected_geometry_fingerprint
            != geometry_identity["geometry_fingerprint"]
        ):
            raise ValueError(
                "GeoPathTie SEG-Y geometry differs from the sealed snapshot interpretation"
            )
        segy_profile_receipt = {
            "contract_version": "well-seismic.segy-profile-receipt.v1",
            "profile_name": profile_name,
            "profile_definition": profile_definition,
            "profile_definition_sha256": canonical_sha256(profile_definition),
            "resolved_header_bytes": resolved_header_bytes,
            "coordinate_scalar_byte": coordinate_scalar_byte,
            "geometry_confidence": float(geometry.confidence),
            "minimum_geometry_confidence": minimum_geometry_confidence,
            "geometry_fingerprint": geometry_identity["geometry_fingerprint"],
            "source_snapshot_id": options.get("source_snapshot_id"),
            "source_snapshot_semantics_sha256": observed_semantics_sha256,
            "geometry_authority": geometry_authority,
            "automatic_geometry_receipt_sha256": (
                automatic_receipt.get("receipt_sha256")
                if automatic_receipt is not None
                else None
            ),
            "source_asset_id": (
                matched_asset.get("id") if isinstance(matched_asset, Mapping) else None
            ),
            "source_asset_sha256": (
                (
                    matched_asset.get("sha256")
                    if isinstance(matched_asset, Mapping)
                    else None
                )
                or _file_sha256(seismic)
            ),
        }
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=None,
            axes=("MD",),
            provenance={
                "source_seismic": str(seismic),
                "seismic_path": str(seismic),
                "seismic_sha256": _file_sha256(seismic),
                "trajectory_authority": "registration_v3_points",
                "ignored_trajectory_paths": [
                    str(path) for path in self._trajectory_paths(options)
                ],
                "las_paths": [str(path) for path in _well_las_paths_from_options(options)],
                "las_count": len(_well_las_paths_from_options(options)),
                "registration_points_path": str(points),
                "registration_manifest_path": str(manifest),
                "registration_task_id": options.get("registration_task_id"),
                "registration_consumed": True,
                "source_snapshot_id": options.get("source_snapshot_id"),
                "registration_contract_version": product.manifest.get("contract_version"),
                "registration_points_sha256": _file_sha256(points),
                "registration_manifest_sha256": _file_sha256(manifest),
                "segy_profile_receipt": segy_profile_receipt,
                "sealed_segy_geometry_authority": geometry_authority,
                "source_snapshot_semantics_sha256": observed_semantics_sha256,
                "materialization": "geopath_tie_v1_sealed_registration_v3",
                "target_is_model_input": False,
                "time_depth_supervision_opened": False,
            },
        )


def build_default_input_adapters(config: dict[str, Any]) -> ModelInputAdapterRegistry:
    from .layerpulse_input_adapter import LayerPulseInputAdapter

    direct12b_disabled = os.getenv("WELLFUSE_DISABLE_DIRECT12B", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not direct12b_disabled:
        from ..direct12b_runtime import Direct12BInputAdapter

    registry = ModelInputAdapterRegistry()
    registry.register(LayerPulseInputAdapter(config))
    registry.register(FaultSegInputAdapter(config))
    registry.register(
        FaultSegInputAdapter(
            config,
            model_id=FAULTNET_MODEL_ID,
            config_key="faultnet",
        )
    )
    registry.register(SurfaceSegInputAdapter(config))
    registry.register(GeobodySegyInputAdapter(config, "wellfuse_channel_p17"))
    registry.register(GeobodySegyInputAdapter(config, "wellfuse_karst_p17"))
    registry.register(HorizonP17InputAdapter(config))
    registry.register(Facies3DSegyInputAdapter(config))
    registry.register(F3Facies3DSegyInputAdapter(config))
    registry.register(GeoPathTieInputAdapter(config))
    if not direct12b_disabled:
        registry.register(Direct12BInputAdapter())
    for model_id in WELLFUSE_UNKNOWN_WELL_MODEL_IDS:
        registry.register(WellFuseUnknownWellInputAdapter(model_id))
    for model_id in DATASET_BOUND_MODEL_IDS:
        registry.register(DatasetBoundWellInputAdapter(model_id))
    registry.register(DatasetBoundWellInputAdapter(FRACTURE_DEVELOPMENT_MODEL_ID))
    return registry
