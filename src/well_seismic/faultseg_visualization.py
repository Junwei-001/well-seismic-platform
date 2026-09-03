"""Bounded, model-aligned visualization payloads for FaultSeg results.

The module deliberately does not load a complete seismic volume.  It reads
only the traces selected by a bounded ``[Z, INLINE, CROSSLINE]`` sampling grid
and uses memory-mapped ``.npy`` outputs for the matching FaultSeg overlay.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .fault_models import is_fault_volume_model_id
from .io.segy import SegyReader


Shape3D = tuple[int, int, int]
AXES_ZYX = ("Z", "INLINE", "CROSSLINE")
DEFAULT_MAX_SHAPE_ZYX: Shape3D = (128, 96, 96)
REPRESENTATIVE_GRID_SPECS: dict[str, dict[str, Any]] = {
    "representative_grid_36": {
        "contract_version": "well-seismic.faultseg-representative-grid.v1",
        "grid_shape_zyx": (4, 3, 3),
        "block_count": 36,
        "display_scale_algorithm": "maximum_of_36_block_abs_p99_histograms_v1",
    },
    "representative_grid_128": {
        "contract_version": "well-seismic.faultseg-representative-grid.v2",
        "grid_shape_zyx": (8, 4, 4),
        "block_count": 128,
        "display_scale_algorithm": "maximum_of_128_block_abs_p99_histograms_v2",
    },
}
REPRESENTATIVE_BLOCK_SHAPE_ZYX: Shape3D = (128, 128, 128)
FAULTSEG_FIXED_THRESHOLD = 0.518

ReaderFactory = Callable[[Path, dict[str, Any], dict[str, Any]], Any]


def _shape3(value: Sequence[Any], name: str, *, allow_zero: bool = False) -> Shape3D:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three Z/Inline/Crossline values")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integers, not booleans")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain integers") from exc
        if integer != item:
            raise ValueError(f"{name} must contain integers")
        if integer < 0 if allow_zero else integer <= 0:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must contain {qualifier} integers")
        result.append(integer)
    return tuple(result)  # type: ignore[return-value]


def _stable_signature(config: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        {"config": config, "options": options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def _encode_array(array: np.ndarray, encoding: str) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(value) for value in contiguous.shape],
        "axes": list(AXES_ZYX),
        "encoding": encoding,
        "values": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def decode_visualization_array(spec: Mapping[str, Any]) -> np.ndarray:
    """Decode an array emitted by :func:`build_faultseg_visualization_payload`."""
    axes = tuple(str(axis).upper() for axis in spec.get("axes", ()))
    if axes != AXES_ZYX:
        raise ValueError(f"visualization axes must be {AXES_ZYX}, got {axes}")
    shape = _shape3(spec.get("shape", ()), "visualization shape")
    encoding = str(spec.get("encoding", ""))
    dtypes: dict[str, np.dtype[Any]] = {
        "base64-int8": np.dtype(np.int8),
        "base64-uint8": np.dtype(np.uint8),
    }
    if encoding not in dtypes:
        raise ValueError(f"unsupported visualization encoding: {encoding}")
    try:
        raw = base64.b64decode(str(spec.get("values", "")), validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 visualization values") from exc
    expected = int(np.prod(shape)) * dtypes[encoding].itemsize
    if len(raw) != expected:
        raise ValueError(f"visualization byte count does not match shape: {len(raw)} != {expected}")
    array = np.frombuffer(raw, dtype=dtypes[encoding]).reshape(shape)
    if encoding == "base64-int8":
        return array.astype(np.float32) / 127.0
    return array.astype(np.float32) / 255.0


def _block_mask_samples(
    mask: np.ndarray,
    *,
    crop_start_zyx: Shape3D,
    crop_size_zyx: Shape3D,
    sample_indices_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build separate 2D-presence and 3D-occupancy mask previews.

    The seismic background remains point-sampled for bounded I/O. Each mask
    2D preview cell reports whether any fault voxel occurs in its source block,
    so a thin fault cannot disappear between sample points. Whole-volume 3D
    rendering receives the fault-voxel fraction of that same block instead.
    Keeping those contracts separate prevents ``block_any`` from inflating a
    sparse binary mask into an almost-solid ray-marched volume. Only one Z
    sampling slab is materialized at a time, preserving the full-volume memmap
    contract.
    """

    relative_starts = tuple(
        np.asarray(indices, dtype=np.int64) - int(crop_start)
        for indices, crop_start in zip(
            sample_indices_zyx, crop_start_zyx, strict=True
        )
    )
    if any(
        starts.ndim != 1
        or starts.size < 1
        or int(starts[0]) != 0
        or np.any(np.diff(starts) <= 0)
        or int(starts[-1]) >= int(size)
        for starts, size in zip(relative_starts, crop_size_zyx, strict=True)
    ):
        raise ValueError("FaultSeg mask sampling grid is outside the declared crop")

    z_starts, inline_starts, crossline_starts = relative_starts
    sampled_any = np.zeros(
        (len(z_starts), len(inline_starts), len(crossline_starts)),
        dtype=bool,
    )
    sampled_fraction = np.zeros(sampled_any.shape, dtype=np.float32)
    source_positive_count = 0
    source_voxel_count = 0
    inline_widths = np.diff(
        np.append(inline_starts, np.int64(crop_size_zyx[1]))
    ).astype(np.int64)
    crossline_widths = np.diff(
        np.append(crossline_starts, np.int64(crop_size_zyx[2]))
    ).astype(np.int64)
    for output_z, z_start in enumerate(z_starts):
        z_stop = (
            int(z_starts[output_z + 1])
            if output_z + 1 < len(z_starts)
            else int(crop_size_zyx[0])
        )
        slab = np.asarray(mask[int(z_start) : z_stop])
        active = slab != 0
        if np.any(active & (slab != 1)):
            raise ValueError("FaultSeg mask must be binary")
        source_positive_count += int(np.count_nonzero(active))
        source_voxel_count += int(active.size)
        collapsed_z = np.sum(active, axis=0, dtype=np.uint32)
        collapsed_inline = np.add.reduceat(
            collapsed_z,
            inline_starts,
            axis=0,
        )
        counts = np.add.reduceat(
            collapsed_inline,
            crossline_starts,
            axis=1,
        )
        block_sizes = (
            int(z_stop - int(z_start))
            * inline_widths[:, np.newaxis]
            * crossline_widths[np.newaxis, :]
        )
        sampled_any[output_z] = counts > 0
        sampled_fraction[output_z] = counts.astype(np.float32) / block_sizes
    source_fraction = (
        float(source_positive_count / source_voxel_count)
        if source_voxel_count
        else 0.0
    )
    preview_presence_fraction = float(np.mean(sampled_any))
    return sampled_any, sampled_fraction, {
        "sourceForegroundFraction": source_fraction,
        "previewPresenceFraction2D": preview_presence_fraction,
        "blockAnyInflationRatio": (
            float(preview_presence_fraction / source_fraction)
            if source_fraction > 0.0
            else 0.0
        ),
        "previewMeanOccupancy3D": float(np.mean(sampled_fraction)),
    }


def _fault_surface_display_threshold(
    occupancy_uint8: np.ndarray,
    *,
    sampling_stride_zyx: Shape3D,
    target_active_fraction: float = 0.01,
) -> tuple[float, dict[str, Any]]:
    """Choose a display-only isosurface level for a thin binary fault sheet.

    A fixed occupancy level such as ``0.50`` is not scale invariant: one
    source-voxel-thick faults occupy roughly ``1 / block_width`` of a preview
    block and therefore disappear as surveys get larger.  Conversely,
    rendering every ``block_any`` hit turns a dense mask into a red wall.  The
    level below keeps the densest one percent of preview blocks while enforcing
    a geometry-derived half-sheet floor.  It never changes the producer mask or
    its checkpoint-calibrated threshold.
    """

    codes = np.asarray(occupancy_uint8, dtype=np.uint8)
    if codes.ndim != 3 or codes.size <= 0:
        raise ValueError("FaultSeg 3D occupancy preview must be a non-empty volume")
    if not 0.0 < float(target_active_fraction) < 1.0:
        raise ValueError("FaultSeg display target fraction must stay within (0, 1)")
    stride = _shape3(sampling_stride_zyx, "sampling_stride_zyx")
    # Half of the occupancy expected from a one-voxel sheet crossing the
    # widest preview block.  This rejects isolated single-voxel hits without
    # requiring a producer-undeclared connected-component filter.
    thin_sheet_floor = 0.5 / float(max(stride))
    thin_sheet_floor_code = max(1, int(math.ceil(thin_sheet_floor * 255.0)))
    flat = codes.reshape(-1)
    quantile = 1.0 - float(target_active_fraction)
    quantile_index = min(
        flat.size - 1,
        max(0, int(math.ceil(quantile * flat.size)) - 1),
    )
    quantile_code = int(np.partition(flat, quantile_index)[quantile_index])
    threshold_code = max(thin_sheet_floor_code, quantile_code, 1)
    threshold = float(threshold_code / 255.0)
    observed_active_fraction = float(np.mean(flat >= threshold_code))
    audit: dict[str, Any] = {
        "selectionPolicy": "preview_p99_with_half_thin_sheet_floor",
        "targetActivePreviewFraction": float(target_active_fraction),
        "observedActivePreviewFraction": observed_active_fraction,
        "occupancyQuantile": quantile,
        "quantileCode": quantile_code,
        "thinSheetOccupancyFloor": float(thin_sheet_floor_code / 255.0),
        "displayThreshold": threshold,
        "displayThresholdCode": threshold_code,
        "sourceMaskModified": False,
        "producerThresholdModified": False,
    }
    return threshold, audit


@dataclass(frozen=True)
class SegySliceKey:
    source: str
    mtime_ns: int
    file_size: int
    crop_start_zyx: Shape3D
    crop_size_zyx: Shape3D
    max_shape_zyx: Shape3D
    reader_signature: str
    display_scale_override: float | None = None

    @property
    def token(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class SparseSegyCrop:
    key: SegySliceKey
    source_shape_zyx: Shape3D
    sampling_stride_zyx: Shape3D
    sample_indices_zyx: tuple[np.ndarray, np.ndarray, np.ndarray]
    inline_values: np.ndarray
    crossline_values: np.ndarray
    time_values: np.ndarray
    cube_int8: np.ndarray
    amplitude_scale_p99: float
    valid_traces: np.ndarray
    source_trace_count: int

    @property
    def nbytes(self) -> int:
        arrays = (
            *self.sample_indices_zyx,
            self.inline_values,
            self.crossline_values,
            self.time_values,
            self.cube_int8,
            self.valid_traces,
        )
        return int(sum(array.nbytes for array in arrays))

    def as_payload(self, *, cache_hit: bool) -> dict[str, Any]:
        z_indices, inline_indices, crossline_indices = self.sample_indices_zyx
        cube = _encode_array(self.cube_int8, "base64-int8")
        cube["amplitudeScaleP99"] = self.amplitude_scale_p99
        return {
            "name": Path(self.key.source).name,
            "path": self.key.source,
            "axes": list(AXES_ZYX),
            "inlineValues": [int(value) for value in self.inline_values],
            "crosslineValues": [int(value) for value in self.crossline_values],
            "timeValues": [round(float(value), 6) for value in self.time_values],
            "defaultIndices": [
                int(len(z_indices) // 2),
                int(len(inline_indices) // 2),
                int(len(crossline_indices) // 2),
            ],
            "cube": cube,
            "sampling": {
                "sourceShapeZYX": list(self.source_shape_zyx),
                "cropStartZYX": list(self.key.crop_start_zyx),
                "cropSizeZYX": list(self.key.crop_size_zyx),
                "strideZYX": list(self.sampling_stride_zyx),
                "sampleIndicesZYX": {
                    "z": z_indices.astype(int).tolist(),
                    "inline": inline_indices.astype(int).tolist(),
                    "crossline": crossline_indices.astype(int).tolist(),
                },
            },
            "preview": {
                "loadedTraces": int(self.valid_traces.sum()),
                "requestedTraces": int(self.valid_traces.size),
                "sourceTraceCount": self.source_trace_count,
                "validTraceFraction": float(self.valid_traces.mean()),
                "amplitudeScaleP99": self.amplitude_scale_p99,
                "cacheHit": cache_hit,
                "cacheKey": self.key.token,
            },
        }


@dataclass
class _InFlightSegySlice:
    """One single-flight crop build shared by callers of the same cache key."""

    event: threading.Event
    generation: int
    crop: SparseSegyCrop | None = None
    cached: bool = False
    error: BaseException | None = None


@dataclass(frozen=True)
class FaultMaskSampleKey:
    """Identity of one full-mask reduction into the bounded display grid."""

    source: str
    mtime_ns: int
    file_size: int
    mask_shape_zyx: Shape3D
    mask_dtype: str
    crop_start_zyx: Shape3D
    crop_size_zyx: Shape3D
    sample_indices_sha256: str

    @property
    def token(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class FaultMaskSamples:
    """Small immutable derivative produced by one authoritative mask scan."""

    key: FaultMaskSampleKey
    block_any: np.ndarray
    block_fraction: np.ndarray
    audit: Mapping[str, Any]

    @property
    def nbytes(self) -> int:
        audit_bytes = len(
            json.dumps(
                dict(self.audit),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return int(self.block_any.nbytes + self.block_fraction.nbytes + audit_bytes)

    def as_result(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        # Callers add renderer-only audit fields, so the small mapping must not
        # be shared even though the cached numeric arrays remain immutable.
        return self.block_any, self.block_fraction, dict(self.audit)


@dataclass
class _InFlightFaultMaskSamples:
    """One full-mask scan shared by concurrent callers of the same key."""

    event: threading.Event
    generation: int
    samples: FaultMaskSamples | None = None
    cached: bool = False
    error: BaseException | None = None


class SegySliceCache:
    """Thread-safe LRU cache for bounded sparse SEG-Y crops."""

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_bytes: int = 64 * 1024 * 1024,
        max_voxels: int = math.prod(REPRESENTATIVE_BLOCK_SHAPE_ZYX),
        reader_factory: ReaderFactory = SegyReader,
    ) -> None:
        if max_entries <= 0 or max_bytes <= 0 or max_voxels <= 0:
            raise ValueError("cache limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.max_voxels = int(max_voxels)
        self.reader_factory = reader_factory
        self._entries: OrderedDict[SegySliceKey, SparseSegyCrop] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._skipped = 0
        self._lock = threading.RLock()
        self._inflight: dict[SegySliceKey, _InFlightSegySlice] = {}
        self._generation = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "skipped": self._skipped,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._generation += 1

    def get_crop(
        self,
        source: str | Path,
        *,
        crop_start_zyx: Sequence[Any],
        crop_size_zyx: Sequence[Any],
        max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
        config: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        expected_source_shape_zyx: Sequence[Any] | None = None,
        display_scale_override: float | None = None,
    ) -> tuple[SparseSegyCrop, bool]:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SEG-Y source does not exist: {path}")
        if path.suffix.lower() not in {".sgy", ".segy"}:
            raise ValueError(f"FaultSeg background source must be SEG-Y: {path}")
        crop_start = _shape3(crop_start_zyx, "crop_start_zyx", allow_zero=True)
        crop_size = _shape3(crop_size_zyx, "crop_size_zyx")
        max_shape = _shape3(max_shape_zyx, "max_shape_zyx")
        expected_shape = (
            _shape3(expected_source_shape_zyx, "source_shape_zyx")
            if expected_source_shape_zyx is not None
            else None
        )
        resolved_display_scale = (
            float(display_scale_override)
            if display_scale_override is not None
            else None
        )
        if resolved_display_scale is not None and (
            not np.isfinite(resolved_display_scale)
            or resolved_display_scale <= 0.0
        ):
            raise ValueError("display_scale_override must be finite and positive")
        sampled_shape = tuple(min(size, math.ceil(size / max(1, math.ceil(size / limit)))) for size, limit in zip(crop_size, max_shape))
        if math.prod(sampled_shape) > self.max_voxels:
            raise ValueError(
                f"sampled crop {sampled_shape} exceeds cache voxel limit {self.max_voxels}"
            )

        reader_config = dict(config or {})
        reader_options = dict(options or {"profile": "standard_3d"})
        stat = path.stat()
        key = SegySliceKey(
            source=str(path),
            mtime_ns=int(stat.st_mtime_ns),
            file_size=int(stat.st_size),
            crop_start_zyx=crop_start,
            crop_size_zyx=crop_size,
            max_shape_zyx=max_shape,
            reader_signature=_stable_signature(reader_config, reader_options),
            display_scale_override=resolved_display_scale,
        )
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached, True
            in_flight = self._inflight.get(key)
            owns_build = in_flight is None
            if in_flight is None:
                in_flight = _InFlightSegySlice(
                    event=threading.Event(),
                    generation=self._generation,
                )
                self._inflight[key] = in_flight
                self._misses += 1

        if not owns_build:
            in_flight.event.wait()
            if in_flight.error is not None:
                raise in_flight.error
            if in_flight.crop is None:
                raise RuntimeError("SEG-Y slice build completed without a crop")
            if in_flight.cached:
                with self._lock:
                    self._hits += 1
                    if key in self._entries:
                        self._entries.move_to_end(key)
            return in_flight.crop, in_flight.cached

        try:
            crop = self._read_sparse_crop(
                key,
                reader_config,
                reader_options,
                expected_shape,
            )
        except BaseException as exc:
            with self._lock:
                in_flight.error = exc
                if self._inflight.get(key) is in_flight:
                    self._inflight.pop(key, None)
                in_flight.event.set()
            raise

        with self._lock:
            cache_is_current = in_flight.generation == self._generation
            if crop.nbytes > self.max_bytes or not cache_is_current:
                if crop.nbytes > self.max_bytes:
                    self._skipped += 1
                cached_build = False
            else:
                self._entries[key] = crop
                self._bytes += crop.nbytes
                while (
                    len(self._entries) > self.max_entries
                    or self._bytes > self.max_bytes
                ):
                    _, removed = self._entries.popitem(last=False)
                    self._bytes -= removed.nbytes
                    self._evictions += 1
                cached_build = True
            in_flight.crop = crop
            in_flight.cached = cached_build
            if self._inflight.get(key) is in_flight:
                self._inflight.pop(key, None)
            in_flight.event.set()
            return crop, False

    def _read_sparse_crop(
        self,
        key: SegySliceKey,
        config: dict[str, Any],
        options: dict[str, Any],
        expected_source_shape_zyx: Shape3D | None,
    ) -> SparseSegyCrop:
        reader = self.reader_factory(Path(key.source), config, options)
        geometry = reader.inspect()
        if geometry.inline is None or geometry.crossline is None:
            raise ValueError("FaultSeg visualization requires resolved 3D Inline/Crossline geometry")
        inline = np.asarray(geometry.inline)
        crossline = np.asarray(geometry.crossline)
        if inline.ndim != 1 or crossline.ndim != 1 or inline.size != geometry.trace_count or crossline.size != geometry.trace_count:
            raise ValueError("SEG-Y Inline/Crossline headers do not match trace_count")
        inline_all = np.unique(inline)
        crossline_all = np.unique(crossline)
        source_shape: Shape3D = (
            int(geometry.samples_per_trace),
            int(inline_all.size),
            int(crossline_all.size),
        )
        if expected_source_shape_zyx is not None and source_shape != expected_source_shape_zyx:
            raise ValueError(
                f"SEG-Y shape {source_shape} does not match FaultSeg source_shape_zyx {expected_source_shape_zyx}"
            )
        if any(start + size > available for start, size, available in zip(key.crop_start_zyx, key.crop_size_zyx, source_shape)):
            raise ValueError(
                f"crop {key.crop_start_zyx}+{key.crop_size_zyx} exceeds SEG-Y shape {source_shape}"
            )

        strides: Shape3D = tuple(
            max(1, math.ceil(size / limit))
            for size, limit in zip(key.crop_size_zyx, key.max_shape_zyx)
        )  # type: ignore[assignment]
        positions = tuple(
            np.arange(start, start + size, stride, dtype=np.int64)
            for start, size, stride in zip(key.crop_start_zyx, key.crop_size_zyx, strides)
        )
        z_indices, inline_indices, crossline_indices = positions
        time_axis = np.asarray(geometry.time_axis, dtype=float)
        if time_axis.ndim != 1 or time_axis.size != source_shape[0]:
            raise ValueError("SEG-Y time axis does not match samples_per_trace")

        lookup: dict[tuple[int, int], int] = {}
        for trace_index, pair in enumerate(zip(inline, crossline)):
            lookup.setdefault((int(pair[0]), int(pair[1])), trace_index)
        selected_inline = inline_all[inline_indices]
        selected_crossline = crossline_all[crossline_indices]
        data = np.full(
            (len(z_indices), len(inline_indices), len(crossline_indices)),
            np.nan,
            dtype=np.float32,
        )
        valid_traces = np.zeros((len(inline_indices), len(crossline_indices)), dtype=bool)
        sample_slice = slice(int(z_indices[0]), int(key.crop_start_zyx[0] + key.crop_size_zyx[0]), strides[0])
        for inline_index, inline_value in enumerate(selected_inline):
            for crossline_index, crossline_value in enumerate(selected_crossline):
                trace_index = lookup.get((int(inline_value), int(crossline_value)))
                if trace_index is None:
                    continue
                values = np.asarray(reader.read_trace(trace_index, sample_slice), dtype=np.float32)
                if values.shape != (len(z_indices),):
                    raise ValueError(
                        "SEG-Y trace slice length does not match sparse FaultSeg background Z dimension"
                    )
                data[:, inline_index, crossline_index] = values
                valid_traces[inline_index, crossline_index] = True
        finite = np.abs(data[np.isfinite(data)])
        if finite.size == 0:
            raise ValueError("requested SEG-Y crop contains no readable traces")
        scale = (
            float(key.display_scale_override)
            if key.display_scale_override is not None
            else float(np.percentile(finite, 99.0))
        )
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        cube_int8 = np.clip(np.nan_to_num(data / scale) * 127.0, -127, 127).astype(np.int8)
        for array in (*positions, selected_inline, selected_crossline, cube_int8, valid_traces):
            array.setflags(write=False)
        selected_time = np.asarray(time_axis[z_indices], dtype=float)
        selected_time.setflags(write=False)
        return SparseSegyCrop(
            key=key,
            source_shape_zyx=source_shape,
            sampling_stride_zyx=strides,
            sample_indices_zyx=positions,  # type: ignore[arg-type]
            inline_values=selected_inline,
            crossline_values=selected_crossline,
            time_values=selected_time,
            cube_int8=cube_int8,
            amplitude_scale_p99=scale,
            valid_traces=valid_traces,
            source_trace_count=int(geometry.trace_count),
        )


class FaultMaskSampleCache:
    """Bounded single-flight LRU for expensive full-mask display reductions.

    FaultSeg masks can be several gigabytes.  Their bounded ``block_any`` and
    occupancy derivatives are only a few megabytes, so retaining those
    immutable derivatives avoids replaying the same full sequential scan on
    every result-page request.  File identity and the exact sampling grid are
    both part of the key; a changed result therefore cannot reuse stale data.
    """

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_bytes: int = 64 * 1024 * 1024,
        sampler: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]
        | None = None,
    ) -> None:
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("cache limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.sampler = sampler
        self._entries: OrderedDict[
            FaultMaskSampleKey, FaultMaskSamples
        ] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._skipped = 0
        self._lock = threading.RLock()
        self._inflight: dict[
            FaultMaskSampleKey, _InFlightFaultMaskSamples
        ] = {}
        self._generation = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "skipped": self._skipped,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            # A scan already in progress may still satisfy its current callers,
            # but it must not repopulate a cache the user explicitly cleared.
            self._generation += 1

    @staticmethod
    def _sample_indices_signature(
        sample_indices_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"well-seismic.fault-mask-sampling-grid.v1\0")
        for axis, values in zip(AXES_ZYX, sample_indices_zyx, strict=True):
            indices = np.ascontiguousarray(values, dtype=np.int64)
            if indices.ndim != 1 or indices.size < 1:
                raise ValueError("FaultSeg sampling indices must be non-empty vectors")
            digest.update(axis.encode("ascii") + b"\0")
            digest.update(int(indices.size).to_bytes(8, "little", signed=False))
            digest.update(indices.tobytes(order="C"))
        return digest.hexdigest()

    def get_samples(
        self,
        mask_path: str | Path,
        mask: np.ndarray,
        *,
        crop_start_zyx: Sequence[Any],
        crop_size_zyx: Sequence[Any],
        sample_indices_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[FaultMaskSamples, bool]:
        path = Path(mask_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".npy":
            raise FileNotFoundError(f"FaultSeg mask does not exist: {path}")
        mask_shape = _shape3(mask.shape, "mask.shape")
        crop_start = _shape3(
            crop_start_zyx,
            "crop_start_zyx",
            allow_zero=True,
        )
        crop_size = _shape3(crop_size_zyx, "crop_size_zyx")
        normalized_indices = tuple(
            np.asarray(values, dtype=np.int64)
            for values in sample_indices_zyx
        )
        if len(normalized_indices) != 3:
            raise ValueError("FaultSeg sampling grid must contain three axes")
        before = path.stat()
        key = FaultMaskSampleKey(
            source=str(path),
            mtime_ns=int(before.st_mtime_ns),
            file_size=int(before.st_size),
            mask_shape_zyx=mask_shape,
            mask_dtype=np.dtype(mask.dtype).str,
            crop_start_zyx=crop_start,
            crop_size_zyx=crop_size,
            sample_indices_sha256=self._sample_indices_signature(
                normalized_indices  # type: ignore[arg-type]
            ),
        )

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached, True
            in_flight = self._inflight.get(key)
            owns_build = in_flight is None
            if in_flight is None:
                in_flight = _InFlightFaultMaskSamples(
                    event=threading.Event(),
                    generation=self._generation,
                )
                self._inflight[key] = in_flight
                self._misses += 1

        if not owns_build:
            in_flight.event.wait()
            if in_flight.error is not None:
                raise in_flight.error
            if in_flight.samples is None:
                raise RuntimeError("FaultSeg mask scan completed without samples")
            if in_flight.cached:
                with self._lock:
                    self._hits += 1
                    if key in self._entries:
                        self._entries.move_to_end(key)
            return in_flight.samples, in_flight.cached

        try:
            sampler = self.sampler or _block_mask_samples
            block_any, block_fraction, audit = sampler(
                mask,
                crop_start_zyx=crop_start,
                crop_size_zyx=crop_size,
                sample_indices_zyx=normalized_indices,
            )
            after = path.stat()
            if (
                int(after.st_mtime_ns) != key.mtime_ns
                or int(after.st_size) != key.file_size
            ):
                raise ValueError("FaultSeg mask changed while building its preview")
            immutable_any = np.array(block_any, dtype=np.bool_, copy=True, order="C")
            immutable_fraction = np.array(
                block_fraction,
                dtype=np.float32,
                copy=True,
                order="C",
            )
            if immutable_any.shape != immutable_fraction.shape:
                raise ValueError("FaultSeg mask preview derivatives disagree in shape")
            immutable_any.setflags(write=False)
            immutable_fraction.setflags(write=False)
            samples = FaultMaskSamples(
                key=key,
                block_any=immutable_any,
                block_fraction=immutable_fraction,
                audit=dict(audit),
            )
        except BaseException as exc:
            with self._lock:
                in_flight.error = exc
                if self._inflight.get(key) is in_flight:
                    self._inflight.pop(key, None)
                in_flight.event.set()
            raise

        with self._lock:
            cache_is_current = in_flight.generation == self._generation
            if samples.nbytes > self.max_bytes or not cache_is_current:
                if samples.nbytes > self.max_bytes:
                    self._skipped += 1
                cached_build = False
            else:
                self._entries[key] = samples
                self._bytes += samples.nbytes
                while (
                    len(self._entries) > self.max_entries
                    or self._bytes > self.max_bytes
                ):
                    _, removed = self._entries.popitem(last=False)
                    self._bytes -= removed.nbytes
                    self._evictions += 1
                cached_build = True
            in_flight.samples = samples
            in_flight.cached = cached_build
            if self._inflight.get(key) is in_flight:
                self._inflight.pop(key, None)
            in_flight.event.set()
            return samples, False


DEFAULT_FAULTSEG_SLICE_CACHE = SegySliceCache()
DEFAULT_FAULTSEG_MASK_SAMPLE_CACHE = FaultMaskSampleCache()


def _result_document(result_or_metadata: Mapping[str, Any] | str | Path) -> tuple[Mapping[str, Any], Path | None]:
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
        raise FileNotFoundError(f"FaultSeg metadata does not exist: {metadata_path}")
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("FaultSeg metadata JSON must contain an object")
    return document, metadata_path.parent


def _metadata_path(value: Any, *, base: Path | None, label: str, suffix: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"FaultSeg result is missing {label}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if path.suffix.lower() != suffix or not path.is_file():
        raise FileNotFoundError(f"invalid {label}: {path}")
    return path


def _finite_fraction(value: Any, label: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite fraction") from exc
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{label} must stay within [0, 1]")
    return fraction


def _representative_grid_requested(result: Mapping[str, Any]) -> bool:
    inference = result.get("inference")
    inference_scope = (
        str(inference.get("scope") or inference.get("faultseg_scope") or "")
        if isinstance(inference, Mapping)
        else ""
    )
    return (
        inference_scope in REPRESENTATIVE_GRID_SPECS
        or "representative_grid" in result
    )


def _representative_block_id(grid_index_zyx: Shape3D) -> str:
    z_index, inline_index, crossline_index = grid_index_zyx
    return f"z{z_index:02d}_i{inline_index:02d}_x{crossline_index:02d}"


def _validate_axis_coordinate_ranges(
    value: Any,
    *,
    start_zyx: Shape3D,
    end_inclusive_zyx: Shape3D,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(AXES_ZYX):
        raise ValueError(f"{label} must declare exact Z/INLINE/CROSSLINE ranges")
    normalized: dict[str, dict[str, Any]] = {}
    for axis, start, end in zip(AXES_ZYX, start_zyx, end_inclusive_zyx, strict=True):
        raw = value.get(axis)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}.{axis} must be an object")
        if int(raw.get("index_start", -1)) != start:
            raise ValueError(f"{label}.{axis}.index_start disagrees with block bounds")
        if int(raw.get("index_end_inclusive", -1)) != end:
            raise ValueError(
                f"{label}.{axis}.index_end_inclusive disagrees with block bounds"
            )
        if "index_end_exclusive" in raw and int(raw["index_end_exclusive"]) != end + 1:
            raise ValueError(
                f"{label}.{axis}.index_end_exclusive disagrees with block bounds"
            )
        normalized[axis] = dict(raw)
    return normalized


def _normalized_representative_grid(
    result: Mapping[str, Any],
    *,
    metadata_base: Path | None,
) -> dict[str, Any]:
    """Validate a sealed representative-grid receipt without opening arrays."""

    if result.get("schema_version") != "well-seismic.faultseg-runtime.v1":
        raise ValueError("representative FaultSeg result has an unsupported schema_version")
    input_metadata = result.get("input")
    inference = result.get("inference")
    grid = result.get("representative_grid")
    outputs = result.get("outputs")
    if not all(
        isinstance(item, Mapping)
        for item in (input_metadata, inference, grid, outputs)
    ):
        raise ValueError(
            "representative FaultSeg result requires input, inference, representative_grid and outputs mappings"
        )
    assert isinstance(input_metadata, Mapping)
    assert isinstance(inference, Mapping)
    assert isinstance(grid, Mapping)
    assert isinstance(outputs, Mapping)
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != AXES_ZYX:
        raise ValueError(f"FaultSeg input axes must be {AXES_ZYX}, got {axes}")
    inference_scope = str(
        inference.get("scope") or inference.get("faultseg_scope") or ""
    )
    representative_spec = REPRESENTATIVE_GRID_SPECS.get(inference_scope)
    if representative_spec is None:
        raise ValueError("representative FaultSeg inference scope is not sealed")
    block_count = int(representative_spec["block_count"])
    expected_grid_shape = tuple(representative_spec["grid_shape_zyx"])
    if int(inference.get("forward_calls") or 0) != block_count:
        raise ValueError(
            f"representative FaultSeg inference must receipt {block_count} forwards"
        )
    if bool(inference.get("stitching", False)):
        raise ValueError("representative FaultSeg blocks must not be stitched")
    if tuple(inference.get("overlap", ())) != (0, 0, 0):
        raise ValueError("representative FaultSeg inference overlap must be [0, 0, 0]")
    if str(inference.get("normalization") or "") != "per_patch_zscore":
        raise ValueError("representative FaultSeg normalization must be per_patch_zscore")
    if float(inference.get("threshold", -1.0)) != FAULTSEG_FIXED_THRESHOLD:
        raise ValueError("representative FaultSeg threshold must be fixed at 0.518")

    if grid.get("contract_version") != representative_spec["contract_version"]:
        raise ValueError("representative FaultSeg grid contract_version is unsupported")
    if grid.get("scope") != "representative_sampling" or grid.get("is_full_volume") is not False:
        raise ValueError("representative FaultSeg grid must explicitly be non-full representative sampling")
    grid_shape = _shape3(grid.get("grid_shape_zyx", ()), "representative_grid.grid_shape_zyx")
    block_shape = _shape3(grid.get("block_shape_zyx", ()), "representative_grid.block_shape_zyx")
    source_shape = _shape3(grid.get("source_shape_zyx", ()), "representative_grid.source_shape_zyx")
    if grid_shape != expected_grid_shape:
        raise ValueError(
            "representative FaultSeg grid_shape_zyx must be "
            f"{list(expected_grid_shape)}"
        )
    if block_shape != REPRESENTATIVE_BLOCK_SHAPE_ZYX:
        raise ValueError("representative FaultSeg block_shape_zyx must be [128, 128, 128]")
    if _shape3(input_metadata.get("source_shape_zyx", ()), "input.source_shape_zyx") != source_shape:
        raise ValueError("representative FaultSeg input/grid source shapes disagree")
    if str(grid.get("grid_order") or "") != "Z_then_INLINE_then_CROSSLINE":
        raise ValueError("representative FaultSeg grid order is unsupported")
    if str(grid.get("start_policy") or "") != "partition_center_floor_then_clamp_to_full_source_block_v1":
        raise ValueError("representative FaultSeg start policy is unsupported")
    if grid.get("unique_source_starts") is not True:
        raise ValueError("representative FaultSeg source starts must be unique")
    if bool(grid.get("inter_block_stitching", False)):
        raise ValueError("representative FaultSeg grid must not be stitched")
    if tuple(grid.get("inference_overlap_zyx", ())) != (0, 0, 0):
        raise ValueError("representative FaultSeg grid inference overlap must be zero")
    if int(grid.get("forward_calls_total") or 0) != block_count:
        raise ValueError(
            f"representative FaultSeg grid must receipt {block_count} forwards"
        )
    if float(grid.get("threshold", -1.0)) != FAULTSEG_FIXED_THRESHOLD:
        raise ValueError("representative FaultSeg grid threshold must be fixed at 0.518")
    if str(grid.get("normalization") or "") != "per_patch_zscore":
        raise ValueError("representative FaultSeg grid normalization must be per_patch_zscore")
    union_coverage_fraction = _finite_fraction(
        grid.get("representative_union_coverage_fraction"),
        "representative_grid.representative_union_coverage_fraction",
    )
    if union_coverage_fraction <= 0.0:
        raise ValueError("representative FaultSeg grid union coverage must be positive")
    try:
        display_amplitude_scale = float(grid.get("display_amplitude_scale"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "representative FaultSeg grid display amplitude scale is missing"
        ) from exc
    if not np.isfinite(display_amplitude_scale) or display_amplitude_scale <= 0.0:
        raise ValueError(
            "representative FaultSeg grid display amplitude scale must be positive"
        )
    display_scale_receipt = grid.get("display_amplitude_scale_receipt")
    if (
        not isinstance(display_scale_receipt, Mapping)
        or str(display_scale_receipt.get("algorithm") or "")
        != representative_spec["display_scale_algorithm"]
        or int(display_scale_receipt.get("block_count") or 0) != block_count
        or float(display_scale_receipt.get("block_quantile") or -1.0) != 0.99
        or int(display_scale_receipt.get("finite_sample_count") or 0) <= 0
        or display_scale_receipt.get("model_input_normalization_modified")
        is not False
    ):
        raise ValueError(
            "representative FaultSeg grid display amplitude scale receipt is invalid"
        )

    source = _metadata_path(
        grid.get("source_segy"),
        base=metadata_base,
        label="representative_grid.source_segy",
        suffix=Path(str(grid.get("source_segy") or "")).suffix.lower(),
    )
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError("representative_grid.source_segy must be SEG-Y")
    receipt_path = _metadata_path(
        outputs.get("representative_grid_receipt_json"),
        base=metadata_base,
        label="outputs.representative_grid_receipt_json",
        suffix=".json",
    )
    blocks_directory = Path(str(outputs.get("representative_blocks_directory") or "")).expanduser()
    if not blocks_directory.is_absolute() and metadata_base is not None:
        blocks_directory = metadata_base / blocks_directory
    blocks_directory = blocks_directory.resolve()
    if not blocks_directory.is_dir():
        raise FileNotFoundError(
            f"invalid outputs.representative_blocks_directory: {blocks_directory}"
        )

    raw_blocks = grid.get("blocks")
    if (
        not isinstance(raw_blocks, Sequence)
        or isinstance(raw_blocks, (str, bytes))
        or len(raw_blocks) != block_count
    ):
        raise ValueError(
            "representative FaultSeg grid must contain exactly "
            f"{block_count} blocks"
        )
    normalized_blocks: list[dict[str, Any]] = []
    starts: set[Shape3D] = set()
    artifact_paths: set[str] = set()
    root_checkpoint = str(grid.get("checkpoint") or result.get("checkpoint") or "")
    root_digest = str(grid.get("checkpoint_sha256") or result.get("checkpoint_sha256") or "")
    if not root_checkpoint or len(root_digest) != 64:
        raise ValueError("representative FaultSeg checkpoint receipt is incomplete")
    for ordinal, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"representative_grid.blocks[{ordinal}] must be an object")
        inline_crossline_plane = grid_shape[1] * grid_shape[2]
        z_index = ordinal // inline_crossline_plane
        inline_index = (ordinal % inline_crossline_plane) // grid_shape[2]
        crossline_index = ordinal % grid_shape[2]
        grid_index = (z_index, inline_index, crossline_index)
        expected_id = _representative_block_id(grid_index)
        if str(raw_block.get("block_id") or "") != expected_id:
            raise ValueError(f"representative block {ordinal} is out of fixed grid order")
        if int(raw_block.get("ordinal", -1)) != ordinal:
            raise ValueError(f"representative block {expected_id} ordinal is invalid")
        if _shape3(raw_block.get("grid_index_zyx", ()), f"{expected_id}.grid_index_zyx", allow_zero=True) != grid_index:
            raise ValueError(f"representative block {expected_id} grid index is invalid")
        start = _shape3(raw_block.get("source_start_zyx", ()), f"{expected_id}.source_start_zyx", allow_zero=True)
        shape = _shape3(raw_block.get("shape_zyx", ()), f"{expected_id}.shape_zyx")
        end_exclusive = _shape3(raw_block.get("source_end_zyx_exclusive", ()), f"{expected_id}.source_end_zyx_exclusive")
        end_inclusive = _shape3(raw_block.get("source_end_zyx_inclusive", ()), f"{expected_id}.source_end_zyx_inclusive", allow_zero=True)
        if shape != block_shape:
            raise ValueError(f"representative block {expected_id} shape must be 128 cubed")
        expected_end = tuple(start_value + size for start_value, size in zip(start, shape, strict=True))
        if end_exclusive != expected_end or end_inclusive != tuple(value - 1 for value in expected_end):
            raise ValueError(f"representative block {expected_id} bounds are inconsistent")
        if any(end > available for end, available in zip(end_exclusive, source_shape, strict=True)):
            raise ValueError(f"representative block {expected_id} exceeds the source survey")
        if start in starts:
            raise ValueError("representative FaultSeg source starts are not unique")
        starts.add(start)
        axis_ranges = _validate_axis_coordinate_ranges(
            raw_block.get("axis_coordinate_ranges"),
            start_zyx=start,
            end_inclusive_zyx=end_inclusive,
            label=f"{expected_id}.axis_coordinate_ranges",
        )
        if float(raw_block.get("threshold", -1.0)) != FAULTSEG_FIXED_THRESHOLD:
            raise ValueError(f"representative block {expected_id} threshold drifted")
        if str(raw_block.get("normalization") or "") != "per_patch_zscore":
            raise ValueError(f"representative block {expected_id} normalization drifted")
        if int(raw_block.get("forward_calls") or 0) != 1:
            raise ValueError(f"representative block {expected_id} must receipt one forward")
        if float(raw_block.get("display_amplitude_scale", -1.0)) != display_amplitude_scale:
            raise ValueError(
                f"representative block {expected_id} display amplitude scale drifted"
            )
        if str(raw_block.get("checkpoint") or "") != root_checkpoint or str(raw_block.get("checkpoint_sha256") or "") != root_digest:
            raise ValueError(f"representative block {expected_id} checkpoint receipt drifted")
        block_outputs = raw_block.get("outputs")
        if not isinstance(block_outputs, Mapping):
            raise ValueError(f"representative block {expected_id} outputs are missing")
        resolved_outputs: dict[str, str] = {}
        for output_name, suffix in (
            ("probability_npy", ".npy"),
            ("mask_npy", ".npy"),
            ("metadata_json", ".json"),
        ):
            artifact = _metadata_path(
                block_outputs.get(output_name),
                base=metadata_base,
                label=f"{expected_id}.outputs.{output_name}",
                suffix=suffix,
            )
            artifact_token = str(artifact)
            if artifact_token in artifact_paths:
                raise ValueError("representative FaultSeg blocks must use independent artifacts")
            artifact_paths.add(artifact_token)
            resolved_outputs[output_name] = artifact_token
        valid_trace_ratio = _finite_fraction(
            raw_block.get("valid_trace_ratio"),
            f"{expected_id}.valid_trace_ratio",
        )
        fault_fraction = _finite_fraction(
            raw_block.get("fault_fraction"),
            f"{expected_id}.fault_fraction",
        )
        valid_fault_fraction = _finite_fraction(
            raw_block.get("fault_fraction_of_valid_voxels", fault_fraction),
            f"{expected_id}.fault_fraction_of_valid_voxels",
        )
        if not np.isclose(fault_fraction, valid_fault_fraction, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"representative block {expected_id} primary fault fraction must use valid voxels"
            )
        all_voxel_fraction_raw = raw_block.get("fault_fraction_all_voxels")
        all_voxel_fraction = (
            _finite_fraction(
                all_voxel_fraction_raw,
                f"{expected_id}.fault_fraction_all_voxels",
            )
            if all_voxel_fraction_raw is not None
            else None
        )
        if "fault_fraction_denominator" in raw_block and raw_block.get(
            "fault_fraction_denominator"
        ) != "valid_trace_count_times_block_z":
            raise ValueError(
                f"representative block {expected_id} fault-fraction denominator is unsupported"
            )
        normalized_blocks.append(
            {
                **dict(raw_block),
                "block_id": expected_id,
                "ordinal": ordinal,
                "grid_index_zyx": list(grid_index),
                "source_start_zyx": list(start),
                "source_end_zyx_exclusive": list(end_exclusive),
                "source_end_zyx_inclusive": list(end_inclusive),
                "shape_zyx": list(shape),
                "axis_coordinate_ranges": axis_ranges,
                "valid_trace_ratio": valid_trace_ratio,
                "fault_fraction": valid_fault_fraction,
                "fault_fraction_of_valid_voxels": valid_fault_fraction,
                "fault_fraction_all_voxels": all_voxel_fraction,
                "fault_fraction_denominator": "valid_trace_count_times_block_z",
                "outputs": resolved_outputs,
            }
        )
    default_block_id = _representative_block_id(
        tuple((axis_size - 1) // 2 for axis_size in grid_shape)
    )
    return {
        **dict(grid),
        "execution_scope": inference_scope,
        "block_count": block_count,
        "source_segy": str(source),
        "source_shape_zyx": list(source_shape),
        "grid_shape_zyx": list(grid_shape),
        "block_shape_zyx": list(block_shape),
        "display_amplitude_scale": display_amplitude_scale,
        "display_amplitude_scale_receipt": dict(display_scale_receipt),
        "representative_union_coverage_fraction": union_coverage_fraction,
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "blocks_directory": str(blocks_directory),
        "blocks": normalized_blocks,
        "default_block_id": default_block_id,
    }


def _public_representative_grid(grid: Mapping[str, Any]) -> dict[str, Any]:
    public_blocks: list[dict[str, Any]] = []
    for block in grid["blocks"]:
        public_blocks.append(
            {
                "blockId": block["block_id"],
                "ordinal": block["ordinal"],
                "gridIndexZYX": block["grid_index_zyx"],
                "sourceStartZYX": block["source_start_zyx"],
                "sourceEndZYXExclusive": block["source_end_zyx_exclusive"],
                "sourceEndZYXInclusive": block["source_end_zyx_inclusive"],
                "shapeZYX": block["shape_zyx"],
                "axisCoordinateRanges": block["axis_coordinate_ranges"],
                "validTraceRatio": block["valid_trace_ratio"],
                "faultFraction": block.get(
                    "fault_fraction_of_valid_voxels", block["fault_fraction"]
                ),
                "faultFractionOfValidVoxels": block.get(
                    "fault_fraction_of_valid_voxels", block["fault_fraction"]
                ),
                "faultFractionAllVoxels": block.get(
                    "fault_fraction_all_voxels"
                ),
                "faultFractionDenominator": block.get(
                    "fault_fraction_denominator",
                    "valid_trace_count_times_block_z",
                ),
                "forwardCalls": block["forward_calls"],
                "checkpointEpoch": block.get("checkpoint_epoch"),
            }
        )
    return {
        "contractVersion": grid["contract_version"],
        "scope": grid["scope"],
        "isFullVolume": False,
        "sourceShapeZYX": grid["source_shape_zyx"],
        "sourceAxisRanges": grid.get("source_axis_ranges"),
        "gridShapeZYX": grid["grid_shape_zyx"],
        "blockShapeZYX": grid["block_shape_zyx"],
        "gridOrder": grid["grid_order"],
        "threshold": FAULTSEG_FIXED_THRESHOLD,
        "normalization": "per_patch_zscore",
        "displayAmplitudeScale": grid["display_amplitude_scale"],
        "displayAmplitudeScaleReceipt": grid[
            "display_amplitude_scale_receipt"
        ],
        "forwardCallsTotal": int(grid["block_count"]),
        "interBlockStitching": False,
        "representativeUnionCoverageFraction": grid.get(
            "representative_union_coverage_fraction"
        ),
        "defaultBlockId": grid["default_block_id"],
        "receiptFileName": Path(str(grid["receipt_path"])).name,
        "receiptSha256": grid["receipt_sha256"],
        "blocks": public_blocks,
    }


def _representative_grid_overview_payload(
    grid: Mapping[str, Any],
    *,
    cache: SegySliceCache,
    config: Mapping[str, Any] | None,
    segy_options: Mapping[str, Any] | None,
    max_shape_zyx: Sequence[Any],
) -> dict[str, Any]:
    """Build one bounded survey preview without opening any block predictions."""

    public_grid = _public_representative_grid(grid)
    source_shape = tuple(int(value) for value in grid["source_shape_zyx"])
    background, cache_hit = cache.get_crop(
        grid["source_segy"],
        crop_start_zyx=(0, 0, 0),
        crop_size_zyx=source_shape,
        max_shape_zyx=max_shape_zyx,
        config=config,
        options=segy_options,
        expected_source_shape_zyx=source_shape,
        display_scale_override=float(grid["display_amplitude_scale"]),
    )
    payload = background.as_payload(cache_hit=cache_hit)
    z_axis = dict((public_grid.get("sourceAxisRanges") or {}).get("Z") or {})
    time_values = list(payload.get("timeValues", []))
    payload.update({
        "contractVersion": "faultseg-cigvis-representative-grid-v1",
        "assetKind": "faultseg_grid",
        "name": f"FaultSeg · {grid['block_count']} 个代表性体块位置总览",
        # The absolute source path is used only by the server-side sparse
        # reader.  It must never enter the browser document.
        "path": Path(str(grid["source_segy"])).name,
        "axes": list(AXES_ZYX),
        "verticalAxis": {
            "label": str(
                z_axis.get("domain") or "原生采样轴（域/单位未核验）"
            ),
            "unit": str(z_axis.get("unit") or ""),
            "top": time_values[0] if time_values else None,
            "bottom": time_values[-1] if time_values else None,
        },
        "faultSegGrid": public_grid,
        "faultSeg": {
            "modelId": "faultseg_3d",
            "threshold": FAULTSEG_FIXED_THRESHOLD,
            "scope": grid["execution_scope"],
            "isFullVolume": False,
            "probabilityAvailable": False,
            "display": {
                "preferredLayer": "mask",
                "backgroundCmap": "gray",
                "maskCmap": "Reds",
                "alpha": 0.58,
                "thresholdAdjustable": False,
            },
        },
        "overlays": [],
        "preview": {
            **dict(payload.get("preview", {})),
            "blockCount": int(grid["block_count"]),
            "loadedBlockCount": 0,
            "largeArraysEmbedded": False,
            "largePredictionArraysEmbedded": False,
            "boundedSeismicPreviewEmbedded": True,
            "boundedSeismicPreviewShapeZYX": list(background.cube_int8.shape),
            "representativeSampling": True,
            "isFullVolume": False,
        },
        "displayNotice": (
            "固定网格代表性抽样："
            f"{grid['grid_shape_zyx'][0]} 个垂向层段 × "
            f"{grid['grid_shape_zyx'][1]}×{grid['grid_shape_zyx'][2]} 横向位置；"
            "体块彼此独立且不拼接，不代表连续完整工区预测。"
        ),
    })
    return payload


def _representative_selected_block_payload(
    grid: Mapping[str, Any],
    *,
    selected_block_id: str,
    cache: SegySliceCache,
    config: Mapping[str, Any] | None,
    segy_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blocks_by_id = {str(block["block_id"]): block for block in grid["blocks"]}
    block = blocks_by_id.get(selected_block_id)
    if block is None:
        raise ValueError(
            f"unknown representative FaultSeg block: {selected_block_id}"
        )
    crop_start = tuple(int(value) for value in block["source_start_zyx"])
    crop_size = tuple(int(value) for value in block["shape_zyx"])
    source_shape = tuple(int(value) for value in grid["source_shape_zyx"])
    mask_path = Path(str(block["outputs"]["mask_npy"]))
    probability_path = Path(str(block["outputs"]["probability_npy"]))
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3 or tuple(mask.shape) != crop_size:
        raise ValueError(
            f"representative block {selected_block_id} mask shape must be 128 cubed"
        )
    if probability.ndim != 3 or tuple(probability.shape) != crop_size:
        raise ValueError(
            f"representative block {selected_block_id} probability shape must be 128 cubed"
        )
    if not (
        np.issubdtype(mask.dtype, np.number)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError("representative FaultSeg mask must be numeric")
    if not np.issubdtype(probability.dtype, np.number):
        raise ValueError("representative FaultSeg probability must be numeric")
    mask_values = np.asarray(mask)
    probability_values = np.asarray(probability, dtype=np.float32)
    if np.any((mask_values != 0) & (mask_values != 1)):
        raise ValueError("representative FaultSeg mask must be exact binary 0/1")
    if (
        not np.all(np.isfinite(probability_values))
        or np.any(probability_values < 0.0)
        or np.any(probability_values > 1.0)
    ):
        raise ValueError("representative FaultSeg probability must stay within [0, 1]")
    expected_mask = probability_values >= FAULTSEG_FIXED_THRESHOLD
    if not np.array_equal(mask_values != 0, expected_mask):
        raise ValueError(
            "representative FaultSeg mask disagrees with fixed threshold 0.518"
        )
    observed_fault_fraction_all = float(np.mean(mask_values != 0))
    fraction_tolerance = 1.0 / float(math.prod(crop_size))
    declared_all_fraction = block.get("fault_fraction_all_voxels")
    if (
        declared_all_fraction is not None
        and abs(observed_fault_fraction_all - float(declared_all_fraction))
        > fraction_tolerance
    ):
        raise ValueError(
            f"representative block {selected_block_id} all-voxel fault fraction receipt drifted"
        )

    background, cache_hit = cache.get_crop(
        grid["source_segy"],
        crop_start_zyx=crop_start,
        crop_size_zyx=crop_size,
        max_shape_zyx=REPRESENTATIVE_BLOCK_SHAPE_ZYX,
        config=config,
        options=segy_options,
        expected_source_shape_zyx=source_shape,
        display_scale_override=float(grid["display_amplitude_scale"]),
    )
    if (
        background.sampling_stride_zyx != (1, 1, 1)
        or tuple(background.cube_int8.shape) != crop_size
    ):
        raise ValueError(
            "representative FaultSeg selected block must load the exact full 128 cubed background"
        )
    observed_valid_trace_ratio = float(background.valid_traces.mean())
    valid_trace_tolerance = 1.0 / float(crop_size[1] * crop_size[2])
    if abs(observed_valid_trace_ratio - float(block["valid_trace_ratio"])) > valid_trace_tolerance:
        raise ValueError(
            f"representative block {selected_block_id} valid_trace_ratio receipt drifted"
        )
    valid_voxels = background.valid_traces[np.newaxis, :, :]
    valid_voxel_count = int(background.valid_traces.sum()) * crop_size[0]
    observed_fault_fraction_valid = float(
        np.count_nonzero((mask_values != 0) & valid_voxels)
        / max(1, valid_voxel_count)
    )
    declared_valid_fraction = float(
        block.get("fault_fraction_of_valid_voxels", block["fault_fraction"])
    )
    valid_fraction_tolerance = 1.0 / float(max(1, valid_voxel_count))
    if (
        abs(observed_fault_fraction_valid - declared_valid_fraction)
        > valid_fraction_tolerance
    ):
        raise ValueError(
            f"representative block {selected_block_id} valid-voxel fault fraction receipt drifted"
        )

    mask_uint8 = (mask_values != 0).astype(np.uint8) * np.uint8(255)
    mask_spec = {
        **_encode_array(mask_uint8, "base64-uint8"),
        "valueRange": [0.0, 1.0],
        "displayCodeRange": [0, 255],
        "labelValueRange": [0, 1],
        "classCodes": [0, 1],
        "backgroundDisplayCode": 0,
        "transparentDisplayCode": 0,
        "sourceArtifactFileName": mask_path.name,
        "role": "prediction",
        "displayByDefault": True,
        "samplingAggregation": "identity_binary",
        "interpolation": "nearest",
        "producerThreshold": FAULTSEG_FIXED_THRESHOLD,
        "producerThresholdAdjustable": False,
        "renderMode3D": "exact_binary_first_surface",
        "isCompleteBinaryMaskDisplay": True,
        "displayThreshold": 0.5,
        "displayThresholdAdjustable": False,
        "displaySlab": {
            "axis": "Z",
            "centerNormalized": 0.5,
            "thicknessNormalized": 1.0,
            "semantics": "complete_selected_128_cubed_block",
        },
    }
    payload = background.as_payload(cache_hit=cache_hit)
    public_grid = _public_representative_grid(grid)
    public_block = next(
        item
        for item in public_grid["blocks"]
        if item["blockId"] == selected_block_id
    )
    payload.update(
        {
            "contractVersion": "faultseg-cigvis-representative-block-v1",
            "assetKind": "volume",
            "name": f"{Path(str(grid['source_segy'])).name} · FaultSeg {selected_block_id}",
            "faultSegGrid": public_grid,
            "selectedRepresentativeBlock": public_block,
            "sliceViewContract": {
                **dict(payload.get("sliceViewContract", {})),
                "preferSlice": True,
                "defaultPlane": "i",
                "allowedPlanes": ["i", "x", "z"],
                "displayMode": "faultseg_exact_binary_mask",
                "displayNotice": (
                    f"当前仅显示独立代表块 {selected_block_id}；二维为精确体素掩码，"
                    "灰度地震背景、红色半透明断层与黄色边界。"
                    f"{grid['block_count']} 块不拼接，不能解释为连续完整体。"
                ),
            },
            "faultSeg": {
                "modelId": "faultseg_3d",
                "threshold": FAULTSEG_FIXED_THRESHOLD,
                "thresholdAdjustable": False,
                "scope": grid["execution_scope"],
                "isFullVolume": False,
                "cropStartZYX": list(crop_start),
                "cropSizeZYX": list(crop_size),
                "probabilityAvailable": True,
                "probabilityEmbedded": False,
                "probabilityArtifactFileName": probability_path.name,
                "mask": mask_spec,
                "mask3D": mask_spec,
                "display": {
                    "backgroundCmap": "gray",
                    "preferredLayer": "mask",
                    "maskCmap": "Reds",
                    "maskClim": [0.5, 1.0],
                    "alpha": 0.58,
                    "boundaryColor": "#ffd230",
                    "thresholdAdjustable": False,
                    "gainPolicy": "shared_fixed_ui_default",
                    "sharedAmplitudeScale": float(
                        grid["display_amplitude_scale"]
                    ),
                },
                "cigvis": {
                    "method": "add_mask",
                    "sourceAxes": list(AXES_ZYX),
                    "transposeZYXToLineFirst": [1, 2, 0],
                },
            },
            "overlays": [
                {
                    "id": "faultseg_mask",
                    "name": "断层二值掩码",
                    "kind": "mask",
                    "volume": mask_spec,
                    "volume3D": mask_spec,
                    "clim": [0.5, 1.0],
                    "cmap": "Reds",
                    "alpha": 0.58,
                    "boundaryColor": "#ffd230",
                    "boundaryAlpha": 0.98,
                    "excpt": "min",
                }
            ],
        }
    )
    payload["path"] = Path(str(grid["source_segy"])).name
    payload["preview"].update(
        {
            "cacheStats": cache.stats,
            "representativeSampling": True,
            "isFullVolume": False,
            "loadedBlockCount": 1,
            "selectedBlockId": selected_block_id,
            "fixedThreshold": FAULTSEG_FIXED_THRESHOLD,
            "thresholdAdjustable": False,
            "observedFaultFractionOfValidVoxels": observed_fault_fraction_valid,
            "observedFaultFractionAllVoxels": observed_fault_fraction_all,
            "inferenceReceipt": {
                "forwardCalls": block["forward_calls"],
                "checkpointEpoch": block.get("checkpoint_epoch"),
                "checkpointSha256": block.get("checkpoint_sha256"),
                "normalization": block["normalization"],
            },
        }
    )
    return payload


def build_faultseg_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    mask_cache: FaultMaskSampleCache = DEFAULT_FAULTSEG_MASK_SAMPLE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
    selected_block_id: str | None = None,
) -> dict[str, Any]:
    """Build a CIGVis-ready sparse background plus aligned FaultSeg layers.

    The returned arrays remain in the platform contract order
    ``[Z, INLINE, CROSSLINE]``.  A CIGVis line-first renderer should transpose
    each of them with ``(1, 2, 0)`` before ``create_slices``/``add_mask``.
    """
    result, metadata_base = _result_document(result_or_metadata)
    result_model_id = str(result.get("model_id") or "")
    if not is_fault_volume_model_id(result_model_id):
        raise ValueError("visualization result must use a registered fault-volume model")
    requested_block = str(selected_block_id or "").strip()
    if _representative_grid_requested(result):
        representative_grid = _normalized_representative_grid(
            result,
            metadata_base=metadata_base,
        )
        if not requested_block:
            return _representative_grid_overview_payload(
                representative_grid,
                cache=cache,
                config=config,
                segy_options=segy_options,
                max_shape_zyx=max_shape_zyx,
            )
        return _representative_selected_block_payload(
            representative_grid,
            selected_block_id=requested_block,
            cache=cache,
            config=config,
            segy_options=segy_options,
        )
    if requested_block:
        raise ValueError("block selection is only valid for a representative grid")
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    probability_metadata = result.get("probability")
    inference_metadata = result.get("inference")
    if not isinstance(input_metadata, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("FaultSeg result must contain input and outputs mappings")
    probability_metadata = (
        probability_metadata if isinstance(probability_metadata, Mapping) else {}
    )
    if not isinstance(inference_metadata, Mapping):
        raise ValueError("FaultSeg result must contain an inference mapping")
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != AXES_ZYX:
        raise ValueError(f"FaultSeg input axes must be {AXES_ZYX}, got {axes}")

    crop_start = _shape3(input_metadata.get("crop_start_zyx", ()), "crop_start_zyx", allow_zero=True)
    crop_size = _shape3(input_metadata.get("crop_size_zyx", ()), "crop_size_zyx")
    input_shape = _shape3(input_metadata.get("shape_zyx", ()), "input.shape_zyx")
    source_shape = _shape3(input_metadata.get("source_shape_zyx", ()), "source_shape_zyx")
    if input_shape != crop_size:
        raise ValueError(
            f"FaultSeg input shape must equal crop_size_zyx: {input_shape}, {crop_size}"
        )
    if any(start + size > available for start, size, available in zip(crop_start, crop_size, source_shape)):
        raise ValueError(f"FaultSeg crop {crop_start}+{crop_size} exceeds source shape {source_shape}")
    inference_scope = str(
        inference_metadata.get("scope")
        or inference_metadata.get("faultseg_scope")
        or ""
    ).strip().casefold()
    center_block_receipt: dict[str, Any] | None = None
    if inference_scope == "center_block_1":
        expected_start = tuple((available - 128) // 2 for available in source_shape)
        raw_center_receipt = input_metadata.get("center_block")
        if (
            crop_size != (128, 128, 128)
            or crop_start != expected_start
            or not isinstance(raw_center_receipt, Mapping)
            or raw_center_receipt.get("contract_version")
            != "well-seismic.faultseg-center-block.v1"
            or raw_center_receipt.get("scope") != "center_block_1"
            or raw_center_receipt.get("block_id") != "center_block_1"
            or _shape3(raw_center_receipt.get("shape_zyx", ()), "center_block.shape_zyx")
            != crop_size
            or _shape3(
                raw_center_receipt.get("source_start_zyx", ()),
                "center_block.source_start_zyx",
                allow_zero=True,
            )
            != crop_start
            or raw_center_receipt.get("selection_policy")
            != "floor_center_with_lower_index_tie_break_v1"
            or raw_center_receipt.get("boundary_policy")
            != "complete_block_inside_source_no_padding_v1"
        ):
            raise ValueError("FaultSeg center_block_1 receipt is invalid")
        if (
            list(inference_metadata.get("patch_size") or []) != [128, 128, 128]
            or list(inference_metadata.get("overlap") or []) != [0, 0, 0]
            or inference_metadata.get("weighted_blending") is not False
            or inference_metadata.get("stitching") is not False
            or inference_metadata.get("full_volume_reconstructed") is not False
            or int(inference_metadata.get("forward_calls") or 0) != 1
            or int(result.get("checkpoint_forward_calls") or 0) != 1
        ):
            raise ValueError("FaultSeg center_block_1 inference contract is invalid")
        center_block_receipt = dict(raw_center_receipt)

    source = _metadata_path(input_metadata.get("source"), base=metadata_base, label="input.source", suffix=Path(str(input_metadata.get("source", ""))).suffix.lower())
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError(f"FaultSeg input.source must be SEG-Y: {source}")
    mask_path = _metadata_path(
        outputs.get("mask_npy"),
        base=metadata_base,
        label="outputs.mask_npy",
        suffix=".npy",
    )
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3 or tuple(mask.shape) != crop_size:
        raise ValueError(f"mask array shape {mask.shape} does not match crop {crop_size}")
    if not (
        np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError("FaultSeg mask array must be numeric")

    probability_path: Path | None = None
    probability: np.ndarray | None = None
    probability_unavailable_reason: str | None = "not_declared"
    raw_probability_path = outputs.get("probability_npy")
    if isinstance(raw_probability_path, (str, Path)) and str(raw_probability_path):
        probability_unavailable_reason = None
        try:
            probability_path = _metadata_path(
                raw_probability_path,
                base=metadata_base,
                label="outputs.probability_npy",
                suffix=".npy",
            )
            probability = np.load(
                probability_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except FileNotFoundError:
            probability_unavailable_reason = "declared_artifact_missing"
        except (EOFError, OSError, TypeError, ValueError):
            probability_unavailable_reason = "declared_artifact_unreadable"
        if probability is not None:
            if probability.ndim != 3 or tuple(probability.shape) != crop_size:
                probability_unavailable_reason = "array_shape_mismatch"
                probability = None
            elif not np.issubdtype(probability.dtype, np.number):
                probability_unavailable_reason = "array_dtype_invalid"
                probability = None
        if probability is not None:
            try:
                probability_shape = _shape3(
                    probability_metadata.get("shape_zyx", probability.shape),
                    "probability.shape_zyx",
                )
                reported_min = float(probability_metadata.get("min", 0.0))
                reported_max = float(probability_metadata.get("max", 1.0))
            except (TypeError, ValueError, OverflowError):
                probability_unavailable_reason = "metadata_invalid"
                probability = None
            else:
                if probability_shape != crop_size:
                    probability_unavailable_reason = "metadata_shape_mismatch"
                    probability = None
                elif (
                    not np.isfinite([reported_min, reported_max]).all()
                    or reported_min < 0.0
                    or reported_max > 1.0
                    or reported_min > reported_max
                ):
                    probability_unavailable_reason = "metadata_range_invalid"
                    probability = None
    threshold = float(inference_metadata.get("threshold", 0.5))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("FaultSeg threshold must stay within [0, 1]")

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
        indices - start
        for indices, start in zip(background.sample_indices_zyx, crop_start)
    )
    selection = np.ix_(*relative_indices)
    probability_sample: np.ndarray | None = None
    if probability is not None:
        try:
            probability_sample = np.asarray(
                probability[selection],
                dtype=np.float32,
            )
        except (IndexError, OSError, TypeError, ValueError):
            probability_unavailable_reason = "sample_unreadable"
            probability_sample = None
        if probability_sample is not None:
            if probability_sample.shape != background.cube_int8.shape:
                probability_unavailable_reason = "sample_shape_mismatch"
                probability_sample = None
            elif (
                not np.all(np.isfinite(probability_sample))
                or np.any(probability_sample < 0.0)
                or np.any(probability_sample > 1.0)
            ):
                probability_unavailable_reason = "sample_range_invalid"
                probability_sample = None
    mask_samples, mask_cache_hit = mask_cache.get_samples(
        mask_path,
        mask,
        crop_start_zyx=crop_start,
        crop_size_zyx=crop_size,
        sample_indices_zyx=background.sample_indices_zyx,
    )
    mask_sample, mask_fraction_sample, mask_display_audit = (
        mask_samples.as_result()
    )
    if mask_sample.shape != background.cube_int8.shape:
        raise ValueError("FaultSeg mask sampling is not aligned with the SEG-Y background")
    mask_uint8 = mask_sample.astype(np.uint8) * np.uint8(255)
    mask_fraction_uint8 = np.rint(mask_fraction_sample * 255.0).astype(np.uint8)
    # A single source voxel must remain represented even when a future survey
    # requires blocks larger than 510 voxels. The value is still the smallest
    # display occupancy code and is never presented as model probability.
    mask_fraction_uint8[mask_sample & (mask_fraction_uint8 == 0)] = np.uint8(1)
    display_surface_threshold, display_surface_audit = (
        _fault_surface_display_threshold(
            mask_fraction_uint8,
            sampling_stride_zyx=background.sampling_stride_zyx,
        )
    )
    mask_display_audit["surfaceDisplay3D"] = display_surface_audit

    payload = background.as_payload(cache_hit=cache_hit)
    payload["contractVersion"] = "faultseg-cigvis-v1"
    # FaultSeg results open on the deterministic orthogonal slice first.  The
    # whole-volume view remains available from the workbench mode switch.
    payload["sliceViewContract"] = {
        **dict(payload.get("sliceViewContract", {})),
        "preferSlice": True,
    }
    payload["name"] = f"{source.name} · FaultSeg"
    probability_spec = None
    if probability_sample is not None and probability_path is not None:
        probability_uint8 = np.rint(probability_sample * 255.0).astype(np.uint8)
        probability_spec = {
            **_encode_array(probability_uint8, "base64-uint8"),
            "valueRange": [0.0, 1.0],
            "source": str(probability_path),
            "role": "technical_audit",
            "displayByDefault": False,
        }
    mask_spec = {
        **_encode_array(mask_uint8, "base64-uint8"),
        "valueRange": [0.0, 1.0],
        # The browser receives display bytes (0/255), while these fields bind
        # them back to the producer's deterministic binary class codes.  Zero
        # is deliberately transparent so the seismic event continuity remains
        # visible beneath the highlighted fault voxels.
        "displayCodeRange": [0, 255],
        "labelValueRange": [0, 1],
        "classCodes": [0, 1],
        "invalidDisplayCode": 0,
        "source": str(mask_path),
        "role": "prediction",
        "displayByDefault": True,
        "samplingAggregation": "block_any",
        "interpolation": "nearest",
        "renderMode3D": "not_applicable_2d_binary_slice",
    }
    mask_3d_spec = {
        **_encode_array(mask_fraction_uint8, "base64-uint8"),
        "valueRange": [0.0, 1.0],
        "sourceLabelValueRange": [0, 1],
        "sourceClassCodes": [0, 1],
        "invalidDisplayCode": 0,
        "source": str(mask_path),
        "role": "display_derivative",
        "samplingAggregation": "block_fault_fraction",
        "interpolation": "trilinear_display_derivative",
        "renderMode3D": "adaptive_block_occupancy_isosurface_first_transition",
        "quantity": "fault_voxel_fraction_per_preview_block",
        "isModelProbability": False,
        "displayThreshold": display_surface_threshold,
        "displayThresholdSemantics": (
            "adaptive_3d_preview_only_not_model_threshold"
        ),
        "displayThresholdSelection": display_surface_audit,
        "displayDerivativeLabel": "binary_mask_block_occupancy_fault_ridge_surface",
        "displayMembership": "inclusive_step",
        "isCompleteBinaryMaskDisplay": False,
        "displaySlab": {
            "axis": "Z",
            "centerNormalized": 0.5,
            "thicknessNormalized": 1.0,
            "semantics": "3d_preview_cutaway_only_not_prediction_crop",
        },
        "surfaceExtraction": {
            "method": "ray_first_trilinear_threshold_crossing_within_display_slab",
            "interpolation": "trilinear_block_occupancy",
            "crossingInterpolation": "linear_between_ray_samples",
            "localizationAccuracy": "at_most_half_preview_voxel_on_dominant_texture_axis",
            "transitionDirection": "background_to_fault",
            "maximumHitsPerRay": 1,
            "alphaAccumulatesWithMaskThickness": False,
            "displaySlabBoundaryTreatedAsSurface": False,
            "volumeBoundaryCapDisplayed": False,
            "rearSurfacesDisplayed": False,
        },
    }
    faultseg_payload = {
        "modelId": result_model_id,
        "scope": inference_scope,
        "isFullVolume": inference_scope == "full_volume",
        "threshold": threshold,
        "cropStartZYX": list(crop_start),
        "cropSizeZYX": list(crop_size),
        "probabilityAvailable": probability_spec is not None,
        "probabilityUnavailableReason": probability_unavailable_reason,
        "mask": mask_spec,
        "mask3D": mask_3d_spec,
        "maskDisplayAudit": mask_display_audit,
        "display": {
            "backgroundCmap": "seismic",
            "preferredLayer": "mask",
            "probabilityCmap": "jet",
            "probabilityClim": [0.0, 1.0],
            "maskCmap": "Reds",
            "maskClim": [0.5, 1.0],
            "alpha": 0.72,
            "excludeMinimum": True,
        },
        "cigvis": {
            "method": "add_mask",
            "sourceAxes": list(AXES_ZYX),
            "transposeZYXToLineFirst": [1, 2, 0],
        },
    }
    if center_block_receipt is not None:
        faultseg_payload["centerBlock"] = center_block_receipt
    if probability_spec is not None:
        faultseg_payload["probability"] = probability_spec
    payload["faultSeg"] = faultseg_payload
    # The public/default renderer-neutral overlay is the deterministic binary
    # prediction.  Probability remains available only inside ``faultSeg`` as a
    # technical audit input; it is intentionally not a selectable/default
    # business-result layer.
    payload["overlays"] = [
        {
            "id": "faultseg_mask",
            "name": "断层二值掩码",
            "kind": "mask",
            "volume": mask_spec,
            "volume3D": mask_3d_spec,
            "clim": [0.5, 1.0],
            "cmap": "Reds",
            "alpha": 0.72,
            "excpt": "min",
        }
    ]
    payload["preview"]["cacheStats"] = cache.stats
    payload["preview"]["maskCacheHit"] = mask_cache_hit
    payload["preview"]["maskCacheKey"] = mask_samples.key.token
    payload["preview"]["maskCacheStats"] = mask_cache.stats
    return payload


__all__ = [
    "AXES_ZYX",
    "DEFAULT_FAULTSEG_MASK_SAMPLE_CACHE",
    "DEFAULT_FAULTSEG_SLICE_CACHE",
    "DEFAULT_MAX_SHAPE_ZYX",
    "FaultMaskSampleCache",
    "FaultMaskSampleKey",
    "FaultMaskSamples",
    "SegySliceCache",
    "SparseSegyCrop",
    "build_faultseg_visualization_payload",
    "decode_visualization_array",
]
