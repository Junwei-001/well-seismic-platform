"""Sparse-stick fine-tuning utilities for the bundled FaultNet model.

The DQZJ interpretation files contain picked fault sticks, not a dense binary
fault volume.  This module keeps that distinction executable: picked sticks
become a narrow positive tube, distant samples are used as conservative
negatives only on an interpreted inline, and every other voxel stays unknown.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurveyGeometry:
    min_inline: int
    max_inline: int
    min_crossline: int
    max_crossline: int
    inline_increment: int
    crossline_increment: int
    sample_count: int
    sample_interval_ms: float
    delay_ms: float = 0.0

    @property
    def inline_count(self) -> int:
        return (self.max_inline - self.min_inline) // self.inline_increment + 1

    @property
    def crossline_count(self) -> int:
        return (self.max_crossline - self.min_crossline) // self.crossline_increment + 1

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.sample_count, self.inline_count, self.crossline_count)

    def inline_index(self, value: int) -> int:
        delta = int(value) - self.min_inline
        if delta % self.inline_increment:
            raise ValueError(f"inline {value} is off the declared survey grid")
        index = delta // self.inline_increment
        if not 0 <= index < self.inline_count:
            raise ValueError(f"inline {value} is outside the survey")
        return index

    def crossline_index(self, value: int) -> int:
        delta = int(value) - self.min_crossline
        if delta % self.crossline_increment:
            raise ValueError(f"crossline {value} is off the declared survey grid")
        index = delta // self.crossline_increment
        if not 0 <= index < self.crossline_count:
            raise ValueError(f"crossline {value} is outside the survey")
        return index

    def sample_index(self, time_ms: float) -> int:
        index = int(round((float(time_ms) - self.delay_ms) / self.sample_interval_ms))
        if not 0 <= index < self.sample_count:
            raise ValueError(f"time {time_ms} ms is outside the survey")
        return index


@dataclass(frozen=True)
class FaultPick:
    fault_name: str
    inline: int
    crossline: int
    time_ms: float
    connection_flag: int
    source_ordinal: int

    @property
    def stick_id(self) -> tuple[str, int]:
        # Connection flags restart for every fault name.  Some sticks live on
        # an inline and others on a crossline, so line number is not part of
        # the identity.
        return (self.fault_name, self.connection_flag)


@dataclass(frozen=True)
class FaultStick:
    stick_id: tuple[str, int]
    picks: tuple[FaultPick, ...]

    @property
    def minimum_inline(self) -> int:
        return min(pick.inline for pick in self.picks)

    @property
    def maximum_inline(self) -> int:
        return max(pick.inline for pick in self.picks)

    @property
    def center_inline(self) -> int:
        return int(round(float(np.median([pick.inline for pick in self.picks]))))

    @property
    def fault_name(self) -> str:
        return self.stick_id[0]


@dataclass(frozen=True)
class PatchSpec:
    patch_id: str
    split: str
    start_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    center_inline: int
    source_stick_ids: tuple[tuple[str, int], ...]

    @property
    def end_zyx_exclusive(self) -> tuple[int, int, int]:
        return tuple(start + size for start, size in zip(self.start_zyx, self.shape_zyx))

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "split": self.split,
            "start_zyx": list(self.start_zyx),
            "end_zyx_exclusive": list(self.end_zyx_exclusive),
            "shape_zyx": list(self.shape_zyx),
            "center_inline": self.center_inline,
            "source_stick_ids": [list(value) for value in self.source_stick_ids],
        }


def _parse_scalar(value: str) -> int | float | str:
    text = value.strip()
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def parse_survey_file(
    path: str | Path,
    *,
    sample_count: int,
    sample_interval_ms: float,
    delay_ms: float = 0.0,
) -> SurveyGeometry:
    """Parse the GeoEast survey geometry without depending on locale settings."""

    source = Path(path)
    values: dict[str, int | float | str] = {}
    for raw in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0]] = _parse_scalar(parts[1])
    required = ("minLine", "maxLine", "minTrace", "maxTrace", "lineInc", "traceInc")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"survey geometry is missing fields: {missing}")
    geometry = SurveyGeometry(
        min_inline=int(values["minLine"]),
        max_inline=int(values["maxLine"]),
        min_crossline=int(values["minTrace"]),
        max_crossline=int(values["maxTrace"]),
        inline_increment=int(values["lineInc"]),
        crossline_increment=int(values["traceInc"]),
        sample_count=int(sample_count),
        sample_interval_ms=float(sample_interval_ms),
        delay_ms=float(delay_ms),
    )
    if geometry.inline_increment <= 0 or geometry.crossline_increment <= 0:
        raise ValueError("survey increments must be positive")
    if geometry.sample_count <= 0 or geometry.sample_interval_ms <= 0:
        raise ValueError("time geometry must be positive")
    return geometry


def parse_fault_picks(path: str | Path) -> list[FaultPick]:
    """Read GeoEast fault picks and retain the file order within every stick."""

    source = Path(path)
    result: list[FaultPick] = []
    for ordinal, raw in enumerate(
        source.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        parts = raw.split()
        if len(parts) != 9 or parts[0].startswith("#"):
            continue
        try:
            result.append(
                FaultPick(
                    fault_name=str(parts[0]),
                    inline=int(parts[4]),
                    crossline=int(parts[5]),
                    time_ms=float(parts[6]),
                    connection_flag=int(parts[8]),
                    source_ordinal=ordinal,
                )
            )
        except ValueError:
            continue
    if not result:
        raise ValueError(f"no fault picks were parsed from {source}")
    return result


def group_fault_sticks(picks: Sequence[FaultPick]) -> list[FaultStick]:
    grouped: dict[tuple[str, int], list[FaultPick]] = defaultdict(list)
    for pick in picks:
        grouped[pick.stick_id].append(pick)
    result = [
        FaultStick(
            stick_id=stick_id,
            picks=tuple(sorted(rows, key=lambda item: item.source_ordinal)),
        )
        for stick_id, rows in grouped.items()
    ]
    return sorted(
        result,
        key=lambda item: (item.center_inline, item.fault_name, item.stick_id[1]),
    )


def validate_fault_geometry(
    picks: Sequence[FaultPick], geometry: SurveyGeometry
) -> dict[str, Any]:
    errors: list[str] = []
    for pick in picks:
        try:
            geometry.inline_index(pick.inline)
            geometry.crossline_index(pick.crossline)
            geometry.sample_index(pick.time_ms)
        except ValueError as exc:
            errors.append(f"line {pick.source_ordinal}: {exc}")
    if errors:
        preview = "; ".join(errors[:8])
        raise ValueError(f"{len(errors)} fault picks fall outside the SEG-Y geometry: {preview}")
    sticks = group_fault_sticks(picks)
    orientation_counts = {
        "inline": 0,
        "crossline": 0,
        "vertical_same_trace": 0,
        "oblique": 0,
    }
    interpreted_inline_planes: set[int] = set()
    interpreted_crossline_planes: set[int] = set()
    for stick in sticks:
        inline_values = {pick.inline for pick in stick.picks}
        crossline_values = {pick.crossline for pick in stick.picks}
        if len(inline_values) == 1 and len(crossline_values) > 1:
            orientation_counts["inline"] += 1
            interpreted_inline_planes.update(inline_values)
        elif len(crossline_values) == 1 and len(inline_values) > 1:
            orientation_counts["crossline"] += 1
            interpreted_crossline_planes.update(crossline_values)
        elif len(inline_values) == 1 and len(crossline_values) == 1:
            orientation_counts["vertical_same_trace"] += 1
            interpreted_inline_planes.update(inline_values)
        else:
            orientation_counts["oblique"] += 1
    return {
        "pick_count": len(picks),
        "stick_count": len(sticks),
        "fault_names": sorted({pick.fault_name for pick in picks}),
        "annotated_inline_count": len({pick.inline for pick in picks}),
        "interpreted_inline_plane_count": len(interpreted_inline_planes),
        "interpreted_crossline_plane_count": len(interpreted_crossline_planes),
        "stick_orientation_counts": orientation_counts,
        "inline_range": [min(pick.inline for pick in picks), max(pick.inline for pick in picks)],
        "crossline_range": [
            min(pick.crossline for pick in picks),
            max(pick.crossline for pick in picks),
        ],
        "time_range_ms": [min(pick.time_ms for pick in picks), max(pick.time_ms for pick in picks)],
    }


def _clamped_start(center: float, size: int, lower: int, upper_exclusive: int) -> int:
    if size <= 0 or upper_exclusive - lower < size:
        raise ValueError("patch does not fit inside the requested spatial block")
    start = int(round(float(center) - (size - 1) / 2.0))
    return max(lower, min(start, upper_exclusive - size))


def _stable_patch_id(split: str, start_zyx: Sequence[int], shape_zyx: Sequence[int]) -> str:
    token = f"{split}|{','.join(map(str, start_zyx))}|{','.join(map(str, shape_zyx))}"
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()[:12]
    return f"{split}-{digest}"


def build_spatial_patch_specs(
    sticks: Sequence[FaultStick],
    geometry: SurveyGeometry,
    *,
    patch_shape_zyx: Sequence[int],
    split_inline_ranges: Mapping[str, Sequence[int]],
) -> tuple[list[PatchSpec], dict[str, Any]]:
    """Create disjoint, positive-centred patches contained by inline blocks."""

    shape = tuple(int(value) for value in patch_shape_zyx)
    if len(shape) != 3 or any(value <= 0 or value % 16 for value in shape):
        raise ValueError("FaultNet patch dimensions must be positive multiples of 16")
    if any(size > full for size, full in zip(shape, geometry.shape_zyx)):
        raise ValueError("patch is larger than the survey")

    ranges: dict[str, tuple[int, int]] = {}
    occupied: set[int] = set()
    for split, raw_range in split_inline_ranges.items():
        if len(raw_range) != 2:
            raise ValueError(f"split {split!r} must declare [min_inline, max_inline]")
        low, high = (int(raw_range[0]), int(raw_range[1]))
        if low > high:
            raise ValueError(f"split {split!r} has a reversed inline range")
        indices = set(range(low, high + 1, geometry.inline_increment))
        if occupied & indices:
            raise ValueError("spatial split inline ranges overlap")
        occupied |= indices
        ranges[str(split)] = (low, high)

    allowed_stick_ids: dict[str, set[tuple[str, int]]] = {
        split: set() for split in ranges
    }
    skipped_stick_ids: set[tuple[str, int]] = set()
    for stick in sticks:
        containing = [
            name
            for name, (low, high) in ranges.items()
            if low <= stick.minimum_inline and stick.maximum_inline <= high
        ]
        if len(containing) == 1:
            allowed_stick_ids[containing[0]].add(stick.stick_id)
        else:
            skipped_stick_ids.add(stick.stick_id)
    split_names = sorted(allowed_stick_ids)
    for first_index, first_name in enumerate(split_names):
        for second_name in split_names[first_index + 1 :]:
            overlap = allowed_stick_ids[first_name] & allowed_stick_ids[second_name]
            if overlap:
                raise AssertionError(
                    f"complete fault sticks cross {first_name}/{second_name}: {sorted(overlap)[:8]}"
                )

    grouped: dict[tuple[str, tuple[int, int, int]], list[tuple[str, int]]] = defaultdict(list)
    centers: dict[tuple[str, tuple[int, int, int]], int] = {}
    for stick in sticks:
        split = next(
            (
                name
                for name, (low, high) in ranges.items()
                if low <= stick.minimum_inline and stick.maximum_inline <= high
            ),
            None,
        )
        if split is None:
            continue
        low, high = ranges[split]
        block_low = geometry.inline_index(low)
        block_high_exclusive = geometry.inline_index(high) + 1
        inline_span = stick.maximum_inline - stick.minimum_inline
        anchor_count = max(1, int(math.ceil((inline_span + 1) / max(1, shape[1] // 2))))
        inline_anchors = np.linspace(
            stick.minimum_inline,
            stick.maximum_inline,
            num=anchor_count,
        )
        for inline_anchor in inline_anchors:
            local_picks = [
                pick
                for pick in stick.picks
                if abs(float(pick.inline) - float(inline_anchor)) <= shape[1] / 2
            ] or list(stick.picks)
            z_values = [geometry.sample_index(pick.time_ms) for pick in local_picks]
            x_values = [geometry.crossline_index(pick.crossline) for pick in local_picks]
            y_value = geometry.inline_index(int(round(float(inline_anchor))))
            start = (
                _clamped_start(np.median(z_values), shape[0], 0, geometry.sample_count),
                _clamped_start(y_value, shape[1], block_low, block_high_exclusive),
                _clamped_start(np.median(x_values), shape[2], 0, geometry.crossline_count),
            )
            key = (split, start)
            grouped[key].append(stick.stick_id)
            centers[key] = int(round(float(inline_anchor)))

    specs = [
        PatchSpec(
            patch_id=_stable_patch_id(split, start, shape),
            split=split,
            start_zyx=start,
            shape_zyx=shape,
            center_inline=centers[(split, start)],
            source_stick_ids=tuple(sorted(set(stick_ids))),
        )
        for (split, start), stick_ids in grouped.items()
    ]
    specs.sort(key=lambda item: (item.split, item.start_zyx, item.patch_id))
    counts = {name: sum(spec.split == name for spec in specs) for name in ranges}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"at least one spatial split has no usable fault patches: {counts}")
    return specs, {
        "split_inline_ranges": {name: list(value) for name, value in ranges.items()},
        "split_patch_counts": counts,
        "split_stick_counts": {
            name: len(stick_ids) for name, stick_ids in allowed_stick_ids.items()
        },
        "split_stick_ids": {
            name: [list(stick_id) for stick_id in sorted(stick_ids)]
            for name, stick_ids in allowed_stick_ids.items()
        },
        "stick_id_cross_split_intersection_count": 0,
        "skipped_stick_count_in_guard_bands": len(skipped_stick_ids),
        "skipped_stick_ids": [list(stick_id) for stick_id in sorted(skipped_stick_ids)],
        "input_stick_count": len(sticks),
        "deduplicated_patch_count": len(specs),
        "patch_shape_zyx": list(shape),
        "patches_are_contained_by_split": True,
    }


def select_spread_specs(
    specs: Sequence[PatchSpec], *, split: str, maximum: int | None
) -> list[PatchSpec]:
    candidates = [spec for spec in specs if spec.split == split]
    if maximum is None or maximum <= 0 or len(candidates) <= maximum:
        return candidates
    # Evenly spaced deterministic selection covers the entire spatial block
    # and is more auditable than a hidden random subset.
    indices = np.linspace(0, len(candidates) - 1, num=int(maximum), dtype=np.int64)
    return [candidates[int(index)] for index in sorted(set(indices.tolist()))]


def _draw_segment(seed: np.ndarray, first: tuple[int, int, int], second: tuple[int, int, int]) -> None:
    delta = np.asarray(second, dtype=np.int64) - np.asarray(first, dtype=np.int64)
    steps = int(max(abs(int(value)) for value in delta))
    if steps == 0:
        points = np.asarray(first, dtype=np.int64)[None, :]
    else:
        points = np.rint(
            np.linspace(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64), steps + 1)
        ).astype(np.int64)
    valid = np.ones(len(points), dtype=bool)
    for axis, size in enumerate(seed.shape):
        valid &= (points[:, axis] >= 0) & (points[:, axis] < size)
    points = points[valid]
    if len(points):
        seed[points[:, 0], points[:, 1], points[:, 2]] = True


def rasterize_sparse_supervision(
    picks: Sequence[FaultPick],
    geometry: SurveyGeometry,
    spec: PatchSpec,
    *,
    positive_radius_voxels: float,
    reliable_negative_distance_voxels: float,
    reliable_negative_time_margin_voxels: int = 8,
    allowed_stick_ids: set[tuple[str, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Rasterize sticks while keeping non-interpreted planes as unknown."""

    from scipy import ndimage

    if not 0 <= positive_radius_voxels < reliable_negative_distance_voxels:
        raise ValueError("positive radius must be smaller than the negative distance")
    start = np.asarray(spec.start_zyx, dtype=np.int64)
    end = np.asarray(spec.end_zyx_exclusive, dtype=np.int64)
    seed = np.zeros(spec.shape_zyx, dtype=bool)
    in_patch: dict[tuple[str, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for pick in picks:
        if allowed_stick_ids is not None and pick.stick_id not in allowed_stick_ids:
            continue
        point = np.asarray(
            [
                geometry.sample_index(pick.time_ms),
                geometry.inline_index(pick.inline),
                geometry.crossline_index(pick.crossline),
            ],
            dtype=np.int64,
        )
        # Retain one-patch halo points so a segment crossing the patch boundary
        # can still be clipped into the patch instead of being silently lost.
        halo = max(spec.shape_zyx)
        if np.all(point >= start - halo) and np.all(point < end + halo):
            local = tuple((point - start).tolist())
            in_patch[pick.stick_id].append((*local, pick.source_ordinal))

    annotated_local_inlines: set[int] = set()
    annotated_local_crosslines: set[int] = set()
    for rows in in_patch.values():
        ordered = sorted(rows, key=lambda item: item[3])
        points = [(z, y, x) for z, y, x, _ in ordered]
        if not points:
            continue
        unique_inline = {y for _, y, _ in points}
        unique_crossline = {x for _, _, x in points}
        if len(unique_inline) == 1:
            annotated_local_inlines.update(y for y in unique_inline if 0 <= y < seed.shape[1])
        if len(unique_crossline) == 1 and len(unique_inline) > 1:
            annotated_local_crosslines.update(
                x for x in unique_crossline if 0 <= x < seed.shape[2]
            )
        if len(points) == 1:
            _draw_segment(seed, points[0], points[0])
        else:
            for first, second in zip(points[:-1], points[1:]):
                _draw_segment(seed, first, second)
    if not bool(seed.any()):
        raise ValueError(f"patch {spec.patch_id} contains no rasterized fault seed")

    positive = ndimage.distance_transform_edt(~seed) <= float(positive_radius_voxels)
    reliable_negative = np.zeros_like(seed)
    distance = np.full(seed.shape, np.inf, dtype=np.float32)
    for inline_index in sorted(annotated_local_inlines):
        plane_seed = seed[:, inline_index, :]
        if not bool(plane_seed.any()):
            continue
        plane_distance = ndimage.distance_transform_edt(~plane_seed).astype(np.float32)
        plane_negative = plane_distance >= float(reliable_negative_distance_voxels)
        margin = int(reliable_negative_time_margin_voxels)
        if margin > 0 and 2 * margin < plane_negative.shape[0]:
            plane_negative[:margin] = False
            plane_negative[-margin:] = False
        reliable_negative[:, inline_index, :] = plane_negative
        distance[:, inline_index, :] = plane_distance
    for crossline_index in sorted(annotated_local_crosslines):
        plane_seed = seed[:, :, crossline_index]
        if not bool(plane_seed.any()):
            continue
        plane_distance = ndimage.distance_transform_edt(~plane_seed).astype(np.float32)
        plane_negative = plane_distance >= float(reliable_negative_distance_voxels)
        margin = int(reliable_negative_time_margin_voxels)
        if margin > 0 and 2 * margin < plane_negative.shape[0]:
            plane_negative[:margin] = False
            plane_negative[-margin:] = False
        reliable_negative[:, :, crossline_index] |= plane_negative
        distance[:, :, crossline_index] = np.minimum(
            distance[:, :, crossline_index], plane_distance
        )
    reliable_negative &= ~positive
    unknown = ~(positive | reliable_negative)
    if bool((positive & reliable_negative).any()):
        raise AssertionError("positive and reliable-negative masks overlap")
    return {
        "seed_positive": seed,
        "positive": positive,
        "reliable_negative": reliable_negative,
        "unknown": unknown,
        "distance_voxels": distance,
        "annotated_inline_count": np.asarray(len(annotated_local_inlines), dtype=np.int64),
        "annotated_crossline_count": np.asarray(
            len(annotated_local_crosslines), dtype=np.int64
        ),
    }


def normalize_faultnet_patch(volume_zyx: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply the official FaultNet per-cuboid min-max contract."""

    source = np.asarray(volume_zyx, dtype=np.float32)
    finite = np.isfinite(source)
    if not bool(finite.any()):
        raise ValueError("seismic patch contains no finite amplitude")
    low = float(source[finite].min())
    high = float(source[finite].max())
    result = np.full(source.shape, 0.5, dtype=np.float32)
    if high > low:
        result[finite] = (source[finite] - low) / (high - low)
    else:
        result[finite] = 0.5
    return result, {
        "minimum": low,
        "maximum": high,
        "finite_voxels": int(finite.sum()),
        "invalid_fill_normalized": 0.5,
    }


class DenseSegyPatchReader:
    """Read regular line-major SEG-Y patches without loading the full survey."""

    def __init__(self, path: str | Path, geometry: SurveyGeometry) -> None:
        self.path = Path(path)
        self.geometry = geometry
        self._handle: Any | None = None

    def __enter__(self) -> "DenseSegyPatchReader":
        import segyio

        self._handle = segyio.open(str(self.path), "r", strict=False, ignore_geometry=True)
        expected = self.geometry.inline_count * self.geometry.crossline_count
        if int(self._handle.tracecount) != expected:
            self._handle.close()
            self._handle = None
            raise ValueError(
                f"dense SEG-Y trace count mismatch: expected {expected}, "
                f"found {int(self._handle.tracecount)}"
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def read(self, spec: PatchSpec) -> np.ndarray:
        if self._handle is None:
            raise RuntimeError("SEG-Y patch reader is not open")
        z0, y0, x0 = spec.start_zyx
        pz, py, px = spec.shape_zyx
        z1, y1, x1 = spec.end_zyx_exclusive
        if z1 > self.geometry.sample_count or y1 > self.geometry.inline_count:
            raise ValueError("patch exceeds SEG-Y geometry")
        if x1 > self.geometry.crossline_count:
            raise ValueError("patch exceeds SEG-Y geometry")
        output = np.empty(spec.shape_zyx, dtype=np.float32)
        for local_inline, inline_index in enumerate(range(y0, y1)):
            first_trace = inline_index * self.geometry.crossline_count + x0
            trace_block = np.asarray(
                self._handle.trace.raw[first_trace : first_trace + px], dtype=np.float32
            )
            if trace_block.shape != (px, self.geometry.sample_count):
                raise ValueError(
                    f"unexpected SEG-Y trace block shape {trace_block.shape}; "
                    f"expected {(px, self.geometry.sample_count)}"
                )
            output[:, local_inline, :] = trace_block[:, z0:z1].T
        return output


def verify_dqzj_segy_contract(
    path: str | Path,
    geometry: SurveyGeometry,
    *,
    inline_byte: int = 189,
    crossline_byte: int = 21,
) -> dict[str, Any]:
    """Fail early if the unusual DQZJ trace-header mapping has drifted."""

    import segyio

    source = Path(path)
    with segyio.open(str(source), "r", strict=False, ignore_geometry=True) as handle:
        expected_traces = geometry.inline_count * geometry.crossline_count
        if int(handle.tracecount) != expected_traces:
            raise ValueError(
                f"SEG-Y has {int(handle.tracecount)} traces; survey declares {expected_traces}"
            )
        if len(handle.samples) != geometry.sample_count:
            raise ValueError(
                f"SEG-Y has {len(handle.samples)} samples; expected {geometry.sample_count}"
            )
        inline_field = segyio.su.iline if inline_byte == 189 else int(inline_byte)
        crossline_field = segyio.su.cdp if crossline_byte == 21 else int(crossline_byte)
        first = handle.header[0]
        last = handle.header[int(handle.tracecount) - 1]
        observed = {
            "first_inline": int(first[inline_field]),
            "first_crossline": int(first[crossline_field]),
            "last_inline": int(last[inline_field]),
            "last_crossline": int(last[crossline_field]),
        }
        expected = {
            "first_inline": geometry.min_inline,
            "first_crossline": geometry.min_crossline,
            "last_inline": geometry.max_inline,
            "last_crossline": geometry.max_crossline,
        }
        if observed != expected:
            raise ValueError(f"SEG-Y trace-header geometry mismatch: {observed} != {expected}")
        samples = np.asarray(handle.samples, dtype=np.float64)
        if len(samples) > 1:
            observed_interval = float(np.median(np.diff(samples)))
        else:
            observed_interval = float("nan")
        if not math.isclose(
            observed_interval,
            geometry.sample_interval_ms,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"SEG-Y sample interval {observed_interval} ms differs from "
                f"{geometry.sample_interval_ms} ms"
            )
        if not math.isclose(
            float(samples[0]), geometry.delay_ms, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"SEG-Y delay {float(samples[0])} ms differs from {geometry.delay_ms} ms"
            )
        sentinel_count = 0
        for inline_index in range(geometry.inline_count):
            expected_inline = geometry.min_inline + inline_index * geometry.inline_increment
            for crossline_index, expected_crossline in (
                (0, geometry.min_crossline),
                (geometry.crossline_count // 2, geometry.min_crossline + (geometry.crossline_count // 2) * geometry.crossline_increment),
                (geometry.crossline_count - 1, geometry.max_crossline),
            ):
                trace_index = inline_index * geometry.crossline_count + crossline_index
                header = handle.header[trace_index]
                actual_pair = (int(header[inline_field]), int(header[crossline_field]))
                expected_pair = (expected_inline, expected_crossline)
                if actual_pair != expected_pair:
                    raise ValueError(
                        "SEG-Y is not dense line-major at trace "
                        f"{trace_index}: {actual_pair} != {expected_pair}"
                    )
                sentinel_count += 1
        return {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "trace_count": int(handle.tracecount),
            "sample_count": len(handle.samples),
            "sample_interval_ms": observed_interval,
            "delay_ms": float(samples[0]),
            "line_major_header_sentinel_count": sentinel_count,
            "inline_header_byte": int(inline_byte),
            "crossline_header_byte": int(crossline_byte),
            **observed,
        }


def masked_faultnet_loss(
    probability: Any,
    teacher_probability: Any,
    *,
    positive_mask: Any,
    reliable_negative_mask: Any,
    gamma: float = 0.7,
    supervised_bce_weight: float = 0.25,
    reliable_negative_bce_multiplier: float = 1.0,
    teacher_consistency_weight: float = 0.05,
) -> dict[str, Any]:
    """Masked Dice + balanced BCE + unknown-region teacher consistency.

    Unknown voxels are excluded from the binary target.  They participate only
    in a low-weight consistency term against the unmodified upstream model.
    """

    import torch
    from torch.nn import functional as functional

    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    if reliable_negative_bce_multiplier <= 0:
        raise ValueError("reliable-negative BCE multiplier must be positive")
    if probability.shape != teacher_probability.shape:
        raise ValueError("student and teacher output shapes disagree")
    positive = positive_mask.to(dtype=torch.bool, device=probability.device)
    negative = reliable_negative_mask.to(dtype=torch.bool, device=probability.device)
    if positive.shape != probability.shape or negative.shape != probability.shape:
        raise ValueError("supervision masks must match the FaultNet output")
    if bool((positive & negative).any()):
        raise ValueError("positive and reliable-negative masks overlap")
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("each training patch needs positive and reliable-negative voxels")
    supervised = positive | negative
    target = positive.to(probability.dtype)
    eps = torch.finfo(probability.dtype).eps
    clipped = probability.clamp(min=eps, max=1.0 - eps)
    intersection = (clipped * target * supervised).sum()
    denominator = (
        supervised.to(clipped.dtype) * ((1.0 - gamma) * clipped + gamma * target)
    ).sum()
    masked_dice = 1.0 - (intersection + eps) / (denominator + eps)
    positive_bce = -clipped[positive].log().mean()
    negative_bce = -(1.0 - clipped[negative]).log().mean()
    balanced_bce = (
        positive_bce + float(reliable_negative_bce_multiplier) * negative_bce
    ) / (1.0 + float(reliable_negative_bce_multiplier))
    unknown = ~supervised
    if bool(unknown.any()):
        teacher_consistency = functional.mse_loss(
            probability[unknown], teacher_probability.detach()[unknown]
        )
    else:
        teacher_consistency = probability.new_zeros(())
    total = (
        masked_dice
        + float(supervised_bce_weight) * balanced_bce
        + float(teacher_consistency_weight) * teacher_consistency
    )
    return {
        "total": total,
        "masked_dice": masked_dice,
        "balanced_bce": balanced_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "teacher_consistency": teacher_consistency,
        "positive_voxels": positive.sum().detach(),
        "reliable_negative_voxels": negative.sum().detach(),
        "unknown_voxels": unknown.sum().detach(),
    }


def sparse_probability_metrics(
    probability: np.ndarray,
    positive_mask: np.ndarray,
    reliable_negative_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64)
    positive = np.asarray(positive_mask, dtype=bool)
    negative = np.asarray(reliable_negative_mask, dtype=bool)
    if probability.shape != positive.shape or probability.shape != negative.shape:
        raise ValueError("probability and sparse masks must share a shape")
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("sparse metrics require both positive and reliable-negative voxels")
    predicted = probability >= float(threshold)
    true_positive = int((predicted & positive).sum())
    false_negative = int((~predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    true_negative = int((~predicted & negative).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    supervised_scores = np.concatenate((probability[positive], probability[negative]))
    supervised_labels = np.concatenate(
        (
            np.ones(int(positive.sum()), dtype=np.int8),
            np.zeros(int(negative.sum()), dtype=np.int8),
        )
    )
    order = np.argsort(-supervised_scores, kind="stable")
    ordered_labels = supervised_labels[order]
    cumulative_positive = np.cumsum(ordered_labels, dtype=np.float64)
    rank = np.arange(1, len(ordered_labels) + 1, dtype=np.float64)
    precision_curve = cumulative_positive / rank
    average_precision = float(
        precision_curve[ordered_labels.astype(bool)].sum() / max(1, int(positive.sum()))
    )
    positive_mean = float(probability[positive].mean())
    negative_mean = float(probability[negative].mean())
    return {
        "positive_voxels": int(positive.sum()),
        "reliable_negative_voxels": int(negative.sum()),
        "positive_probability_mean": positive_mean,
        "reliable_negative_probability_mean": negative_mean,
        "probability_separation": positive_mean - negative_mean,
        "positive_recall_at_threshold": recall,
        "reliable_negative_false_positive_rate_at_threshold": false_positive
        / max(1, false_positive + true_negative),
        "masked_precision_at_threshold": precision,
        "masked_f1_at_threshold": f1,
        "masked_average_precision": average_precision,
        "threshold": float(threshold),
    }


def aggregate_sparse_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    weighted_fields = (
        "positive_probability_mean",
        "positive_recall_at_threshold",
    )
    negative_weighted_fields = (
        "reliable_negative_probability_mean",
        "reliable_negative_false_positive_rate_at_threshold",
    )
    positive_total = sum(int(row["positive_voxels"]) for row in rows)
    negative_total = sum(int(row["reliable_negative_voxels"]) for row in rows)
    result: dict[str, float | int] = {
        "patch_count": len(rows),
        "positive_voxels": positive_total,
        "reliable_negative_voxels": negative_total,
    }
    for name in weighted_fields:
        result[name] = sum(float(row[name]) * int(row["positive_voxels"]) for row in rows) / max(
            1, positive_total
        )
    for name in negative_weighted_fields:
        result[name] = sum(
            float(row[name]) * int(row["reliable_negative_voxels"]) for row in rows
        ) / max(1, negative_total)
    result["probability_separation"] = float(result["positive_probability_mean"]) - float(
        result["reliable_negative_probability_mean"]
    )
    # Patch-weighted F1/precision are transparent approximations because only
    # summarized confusion metrics, not the raw confusion matrices, are kept.
    for name in (
        "masked_precision_at_threshold",
        "masked_f1_at_threshold",
        "masked_average_precision",
    ):
        result[name] = sum(float(row[name]) for row in rows) / len(rows)
    result["threshold"] = float(rows[0]["threshold"])
    return result


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "DenseSegyPatchReader",
    "FaultPick",
    "FaultStick",
    "PatchSpec",
    "SurveyGeometry",
    "aggregate_sparse_metrics",
    "atomic_write_json",
    "build_spatial_patch_specs",
    "group_fault_sticks",
    "masked_faultnet_loss",
    "normalize_faultnet_patch",
    "parse_fault_picks",
    "parse_survey_file",
    "rasterize_sparse_supervision",
    "select_spread_specs",
    "sha256_file",
    "sparse_probability_metrics",
    "validate_fault_geometry",
    "verify_dqzj_segy_contract",
]
