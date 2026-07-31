"""Model-specific input adaptation over shared, model-neutral data readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..faultseg import FaultSegInputSpec, FaultSegVolume, build_faultseg_volume
from ..io.segy import SegyReader


Shape3D = tuple[int, int, int]


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

    def capabilities(self) -> dict[str, Any]:
        ...

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        ...


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
            raise KeyError(f"no input adapter registered for model: {model_id}") from exc

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
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                factory = entry_point.load()
                self.register(factory(config))
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {"plugin": entry_point.name, "error": f"{type(exc).__name__}: {exc}"}
                )
        return loaded


class FaultSegInputAdapter:
    model_id = "faultseg_3d"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = FaultSegInputSpec.from_config(config)

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_formats": ["sgy", "segy"],
            "array_axes": ["Z", "INLINE", "CROSSLINE"],
            "tensor_axes": ["N", "C", "Z", "INLINE", "CROSSLINE"],
            "dtype": "float32",
            "patch_size": list(self.spec.patch_size),
            "overlap": list(self.spec.overlap),
            "normalization": "per-patch z-score",
            "requires_logs": False,
            "input_scope": "crop_or_full_volume",
            "supports_crop": True,
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del options
        inline_count = int(np.unique(geometry.inline).size) if geometry.inline is not None else 0
        crossline_count = int(np.unique(geometry.crossline).size) if geometry.crossline is not None else 0
        shape = (int(geometry.samples_per_trace), inline_count, crossline_count)
        is_3d = inline_count > 1 and crossline_count > 1
        ready = is_3d and all(size >= patch for size, patch in zip(shape, self.spec.patch_size))
        if ready:
            reason = "三维网格、样点数和空间尺寸满足 FaultSeg 输入要求"
        elif not is_3d:
            reason = "FaultSeg 需要同时形成 Inline 和 Crossline 网格的三维 SEG-Y"
        else:
            reason = f"数据体尺寸小于 FaultSeg 最小 patch {self.spec.patch_size}"
        return {
            "ready": ready,
            "reason": reason,
            "adapter": type(self).__name__,
            "expected_axes": ["Z", "INLINE", "CROSSLINE"],
            "patch_size": list(self.spec.patch_size),
            "shape_zyx": list(shape),
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
        crop_size = request.crop_size or self.spec.patch_size
        if any(size < patch for size, patch in zip(crop_size, self.spec.patch_size)):
            raise ValueError(f"FaultSeg crop {crop_size} is smaller than patch {self.spec.patch_size}")
        if any(size > available for size, available in zip(crop_size, shape)):
            raise ValueError(f"FaultSeg crop {crop_size} exceeds seismic volume {shape}")
        crop_start = request.crop_start or tuple((available - size) // 2 for available, size in zip(shape, crop_size))
        if any(start < 0 or start + size > available for start, size, available in zip(crop_start, crop_size, shape)):
            raise ValueError("FaultSeg crop start/size is outside the seismic volume")

        z, y, x = crop_start
        dz, dy, dx = crop_size
        volume: FaultSegVolume = build_faultseg_volume(
            reader,
            sample_slice=slice(z, z + dz),
            inline_slice=slice(y, y + dy),
            crossline_slice=slice(x, x + dx),
        )
        provenance = {
            **volume.provenance,
            "source_shape_zyx": list(shape),
            "crop_start_zyx": list(crop_start),
            "crop_size_zyx": list(crop_size),
            "inline_range": [int(volume.inline_values[0]), int(volume.inline_values[-1])],
            "crossline_range": [int(volume.crossline_values[0]), int(volume.crossline_values[-1])],
            "valid_trace_fraction": float(volume.valid_traces.mean()),
        }
        return ModelInputBatch(
            model_id=self.model_id,
            array=volume.data,
            valid_mask=volume.valid_traces,
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
            return int(text[len(prefix):].split(":", 1)[0])
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
        inline = None if geometry.inline is None else np.asarray(geometry.inline, dtype=np.int64)
        crossline = None if geometry.crossline is None else np.asarray(geometry.crossline, dtype=np.int64)
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
            pair_count = int(np.unique(np.column_stack((inline, crossline)), axis=0).shape[0])
            duplicate_count = max(0, trace_count - pair_count)

        fallback_inline_count = _positive_integer(
            options.get("inline_count", self.spec.get("inline_count"))
        )
        fallback_ready = bool(
            fallback_inline_count
            and trace_count % int(fallback_inline_count) == 0
            and trace_count // int(fallback_inline_count) > 1
        )
        regular_ready = bool(
            is_3d
            and standard_headers
            and duplicate_count == 0
            and sample_count > 1
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
        ready = regular_ready or fallback_ready or platform_ordered_grid
        recommended_options: dict[str, Any] = {}
        native_inline_count: int | None = None
        if regular_ready:
            reason = "标准 189/193 道头形成无重复的三维后叠加网格，满足地层分割输入要求"
            shape_ics = [inline_count, crossline_count, sample_count]
            geometry_mode = "standard_headers"
        elif fallback_ready:
            fallback_crossline_count = trace_count // int(fallback_inline_count)
            reason = f"将按显式 inline_count={fallback_inline_count} 重建三维后叠加网格"
            shape_ics = [int(fallback_inline_count), fallback_crossline_count, sample_count]
            geometry_mode = "explicit_inline_count"
            native_inline_count = int(fallback_inline_count)
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
            reason = "平台虽识别出三维网格，但模型原生读取器固定使用 189/193 道头；需转换为标准道头或提供 inline_count"
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
        reader = SegyReader(
            request.source,
            self.config,
            request.options.get("segy", {"profile": "standard_3d"}),
        )
        geometry = reader.inspect()
        compatibility = self.compatibility(geometry, options=request.options)
        if not compatibility["ready"]:
            raise ValueError(str(compatibility["reason"]))

        shape_ics = tuple(int(value) for value in compatibility["shape_ics"])
        valid_grid = np.ones(shape_ics[:2], dtype=bool)
        if compatibility["geometry_mode"] == "standard_headers":
            inline = np.asarray(geometry.inline, dtype=np.int64)
            crossline = np.asarray(geometry.crossline, dtype=np.int64)
            inline_values = np.unique(inline)
            crossline_values = np.unique(crossline)
            inline_lookup = {int(value): index for index, value in enumerate(inline_values)}
            crossline_lookup = {int(value): index for index, value in enumerate(crossline_values)}
            valid_grid = np.zeros((len(inline_values), len(crossline_values)), dtype=bool)
            for inline_value, crossline_value in zip(inline, crossline, strict=True):
                valid_grid[
                    inline_lookup[int(inline_value)],
                    crossline_lookup[int(crossline_value)],
                ] = True

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
            },
        )


def build_default_input_adapters(config: dict[str, Any]) -> ModelInputAdapterRegistry:
    registry = ModelInputAdapterRegistry()
    registry.register(FaultSegInputAdapter(config))
    registry.register(SurfaceSegInputAdapter(config))
    return registry
