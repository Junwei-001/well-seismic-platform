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

from .io.segy import SegyReader


Shape3D = tuple[int, int, int]
AXES_ZYX = ("Z", "INLINE", "CROSSLINE")
DEFAULT_MAX_SHAPE_ZYX: Shape3D = (128, 96, 96)

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


@dataclass(frozen=True)
class SegySliceKey:
    source: str
    mtime_ns: int
    file_size: int
    crop_start_zyx: Shape3D
    crop_size_zyx: Shape3D
    max_shape_zyx: Shape3D
    reader_signature: str

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


class SegySliceCache:
    """Thread-safe LRU cache for bounded sparse SEG-Y crops."""

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_bytes: int = 64 * 1024 * 1024,
        max_voxels: int = 2_000_000,
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
        )
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return cached, True

            self._misses += 1
            crop = self._read_sparse_crop(
                key,
                reader_config,
                reader_options,
                expected_shape,
            )
            if crop.nbytes > self.max_bytes:
                self._skipped += 1
                return crop, False
            self._entries[key] = crop
            self._bytes += crop.nbytes
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.nbytes
                self._evictions += 1
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
        scale = float(np.percentile(finite, 99.0))
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


DEFAULT_FAULTSEG_SLICE_CACHE = SegySliceCache()


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


def build_faultseg_visualization_payload(
    result_or_metadata: Mapping[str, Any] | str | Path,
    *,
    cache: SegySliceCache = DEFAULT_FAULTSEG_SLICE_CACHE,
    config: Mapping[str, Any] | None = None,
    segy_options: Mapping[str, Any] | None = None,
    max_shape_zyx: Sequence[Any] = DEFAULT_MAX_SHAPE_ZYX,
) -> dict[str, Any]:
    """Build a CIGVis-ready sparse background plus aligned FaultSeg layers.

    The returned arrays remain in the platform contract order
    ``[Z, INLINE, CROSSLINE]``.  A CIGVis line-first renderer should transpose
    each of them with ``(1, 2, 0)`` before ``create_slices``/``add_mask``.
    """
    result, metadata_base = _result_document(result_or_metadata)
    if result.get("model_id") != "faultseg_3d":
        raise ValueError("visualization result must use model_id=faultseg_3d")
    input_metadata = result.get("input")
    outputs = result.get("outputs")
    probability_metadata = result.get("probability")
    inference_metadata = result.get("inference")
    if not isinstance(input_metadata, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("FaultSeg result must contain input and outputs mappings")
    if not isinstance(probability_metadata, Mapping) or not isinstance(inference_metadata, Mapping):
        raise ValueError("FaultSeg result must contain probability and inference mappings")
    axes = tuple(str(axis).upper() for axis in input_metadata.get("axes", ()))
    if axes != AXES_ZYX:
        raise ValueError(f"FaultSeg input axes must be {AXES_ZYX}, got {axes}")

    crop_start = _shape3(input_metadata.get("crop_start_zyx", ()), "crop_start_zyx", allow_zero=True)
    crop_size = _shape3(input_metadata.get("crop_size_zyx", ()), "crop_size_zyx")
    input_shape = _shape3(input_metadata.get("shape_zyx", ()), "input.shape_zyx")
    source_shape = _shape3(input_metadata.get("source_shape_zyx", ()), "source_shape_zyx")
    probability_shape = _shape3(probability_metadata.get("shape_zyx", ()), "probability.shape_zyx")
    if input_shape != crop_size or probability_shape != crop_size:
        raise ValueError(
            f"FaultSeg input/probability shapes must equal crop_size_zyx: {input_shape}, {probability_shape}, {crop_size}"
        )
    if any(start + size > available for start, size, available in zip(crop_start, crop_size, source_shape)):
        raise ValueError(f"FaultSeg crop {crop_start}+{crop_size} exceeds source shape {source_shape}")

    source = _metadata_path(input_metadata.get("source"), base=metadata_base, label="input.source", suffix=Path(str(input_metadata.get("source", ""))).suffix.lower())
    if source.suffix.lower() not in {".sgy", ".segy"}:
        raise ValueError(f"FaultSeg input.source must be SEG-Y: {source}")
    probability_path = _metadata_path(
        outputs.get("probability_npy"),
        base=metadata_base,
        label="outputs.probability_npy",
        suffix=".npy",
    )
    mask_path = _metadata_path(
        outputs.get("mask_npy"),
        base=metadata_base,
        label="outputs.mask_npy",
        suffix=".npy",
    )
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if probability.ndim != 3 or tuple(probability.shape) != crop_size:
        raise ValueError(f"probability array shape {probability.shape} does not match crop {crop_size}")
    if mask.ndim != 3 or tuple(mask.shape) != crop_size:
        raise ValueError(f"mask array shape {mask.shape} does not match crop {crop_size}")
    if not np.issubdtype(probability.dtype, np.number) or not (
        np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError("FaultSeg probability/mask arrays must be numeric")
    reported_min = float(probability_metadata.get("min", 0.0))
    reported_max = float(probability_metadata.get("max", 1.0))
    if not np.isfinite([reported_min, reported_max]).all() or reported_min < 0.0 or reported_max > 1.0 or reported_min > reported_max:
        raise ValueError("FaultSeg probability metadata must stay within [0, 1]")
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
    probability_sample = np.asarray(probability[selection], dtype=np.float32)
    mask_sample = np.asarray(mask[selection])
    if probability_sample.shape != background.cube_int8.shape or mask_sample.shape != background.cube_int8.shape:
        raise ValueError("FaultSeg overlay sampling is not aligned with the SEG-Y background")
    if not np.all(np.isfinite(probability_sample)) or np.any(probability_sample < 0.0) or np.any(probability_sample > 1.0):
        raise ValueError("sampled FaultSeg probability contains invalid values")
    unique_mask = np.unique(mask_sample)
    if not np.all(np.isin(unique_mask, (0, 1, False, True))):
        raise ValueError("sampled FaultSeg mask must be binary")
    probability_uint8 = np.rint(probability_sample * 255.0).astype(np.uint8)
    mask_uint8 = (mask_sample != 0).astype(np.uint8) * np.uint8(255)

    payload = background.as_payload(cache_hit=cache_hit)
    payload["contractVersion"] = "faultseg-cigvis-v1"
    payload["name"] = f"{source.name} · FaultSeg"
    probability_spec = {
        **_encode_array(probability_uint8, "base64-uint8"),
        "valueRange": [0.0, 1.0],
        "source": str(probability_path),
    }
    mask_spec = {
        **_encode_array(mask_uint8, "base64-uint8"),
        "valueRange": [0.0, 1.0],
        "source": str(mask_path),
    }
    payload["faultSeg"] = {
        "modelId": "faultseg_3d",
        "threshold": threshold,
        "cropStartZYX": list(crop_start),
        "cropSizeZYX": list(crop_size),
        "probability": probability_spec,
        "mask": mask_spec,
        "display": {
            "backgroundCmap": "seismic",
            "preferredLayer": "probability",
            "probabilityCmap": "jet",
            "probabilityClim": [0.0, 1.0],
            "maskCmap": "Reds",
            "maskClim": [0.0, 1.0],
            "alpha": 0.62,
            "excludeMinimum": True,
        },
        "cigvis": {
            "method": "add_mask",
            "sourceAxes": list(AXES_ZYX),
            "transposeZYXToLineFirst": [1, 2, 0],
        },
    }
    # This is the renderer-neutral overlay collection already consumed by the
    # CIGVis adapter.  The binary mask remains available under ``faultSeg`` for
    # clients that prefer a hard segmentation instead of the probability map.
    payload["overlays"] = [
        {
            "id": "faultseg_probability",
            "name": "FaultSeg 断层概率",
            "kind": "probability",
            "volume": probability_spec,
            "clim": [threshold, 1.0],
            "cmap": "jet",
            "alpha": 0.62,
            "excpt": "min",
        }
    ]
    payload["preview"]["cacheStats"] = cache.stats
    return payload


__all__ = [
    "AXES_ZYX",
    "DEFAULT_FAULTSEG_SLICE_CACHE",
    "DEFAULT_MAX_SHAPE_ZYX",
    "SegySliceCache",
    "SparseSegyCrop",
    "build_faultseg_visualization_payload",
    "decode_visualization_array",
]
