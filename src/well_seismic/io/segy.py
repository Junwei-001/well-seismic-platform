from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..models import SeismicGeometry


SAMPLE_BYTES = {1: 4, 2: 4, 3: 2, 5: 4, 6: 8, 7: 3, 8: 1, 9: 8, 10: 4, 11: 2, 12: 8, 15: 3, 16: 1}


def _int(data: bytes, byte_1_based: int, size: int, endian: str, signed: bool = True) -> int:
    start = byte_1_based - 1
    return int.from_bytes(data[start:start + size], "big" if endian == ">" else "little", signed=signed)


def _apply_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    scalar = scalar.astype(float)
    factors = np.ones_like(scalar, dtype=float)
    factors[scalar > 0] = scalar[scalar > 0]
    negative = scalar < 0
    factors[negative] = 1.0 / np.abs(scalar[negative])
    return values.astype(float) * factors


def _ibm_to_ieee(raw: bytes, endian: str) -> np.ndarray:
    words = np.frombuffer(raw, dtype=f"{endian}u4").astype(np.uint32)
    sign = np.where(words >> 31, -1.0, 1.0)
    exponent = ((words >> 24) & 0x7F).astype(np.int32) - 64
    fraction = (words & 0x00FFFFFF).astype(np.float64) / float(0x01000000)
    values = sign * fraction * np.power(16.0, exponent)
    values[words == 0] = 0.0
    return values.astype(np.float32)


def _decode_samples(raw: bytes, sample_format: int, endian: str) -> np.ndarray:
    """Decode a complete or partial trace byte window as float32."""
    if sample_format == 1:
        return _ibm_to_ieee(raw, endian)
    if sample_format == 2:
        return np.frombuffer(raw, dtype=f"{endian}i4").astype(np.float32)
    if sample_format == 3:
        return np.frombuffer(raw, dtype=f"{endian}i2").astype(np.float32)
    if sample_format == 5:
        return np.frombuffer(raw, dtype=f"{endian}f4").astype(np.float32)
    if sample_format == 6:
        return np.frombuffer(raw, dtype=f"{endian}f8").astype(np.float32)
    if sample_format == 8:
        return np.frombuffer(raw, dtype="i1").astype(np.float32)
    raise NotImplementedError(f"Trace decoding not implemented for sample format {sample_format}")


class SegyReader:
    def __init__(self, path: str | Path, config: dict[str, Any] | None = None, options: dict[str, Any] | None = None):
        self.path = Path(path)
        self.config = (config or {}).get("segy", config or {})
        self.options = options or {}
        self.geometry: SeismicGeometry | None = None

    def inspect(self) -> SeismicGeometry:
        size = self.path.stat().st_size
        if size < 3600:
            raise ValueError(f"File is too small to be SEG-Y: {self.path}")
        with self.path.open("rb") as handle:
            textual = handle.read(3200)
            binary = handle.read(400)
        endian, score_notes = self._choose_endian(binary, size)
        dt_us = _int(binary, 17, 2, endian, False)
        ns = _int(binary, 21, 2, endian, False)
        fmt = _int(binary, 25, 2, endian, False)
        revision_raw = _int(binary, 301, 2, endian, False)
        revision = ((revision_raw >> 8) & 0xFF) + (revision_raw & 0xFF) / 10.0 if revision_raw else 0.0
        ext_headers = _int(binary, 305, 2, endian, True)
        if ext_headers < 0 or ext_headers > 10000:
            ext_headers = 0
            score_notes.append("invalid_extended_text_header_count_ignored")
        data_start = 3600 + ext_headers * 3200
        bytes_per_sample = SAMPLE_BYTES.get(fmt)
        if bytes_per_sample is None:
            raise ValueError(f"Unsupported SEG-Y sample format code {fmt}: {self.path}")
        trace_size = 240 + ns * bytes_per_sample
        if trace_size <= 240:
            raise ValueError(f"Invalid samples-per-trace {ns}: {self.path}")
        trace_count = max(0, (size - data_start) // trace_size)
        remainder = (size - data_start) % trace_size
        issues = list(score_notes)
        if remainder:
            issues.append(f"fixed_trace_size_remainder:{remainder}")
        if trace_count == 0:
            raise ValueError(f"No traces inferred from SEG-Y: {self.path}")
        offsets = data_start + np.arange(trace_count, dtype=np.int64) * trace_size
        sample_count = min(int(self.config.get("geometry_sample_traces", 2048)), trace_count)
        sample_indices = np.unique(np.linspace(0, trace_count - 1, sample_count, dtype=np.int64))
        headers = self._read_headers(offsets[sample_indices])
        profile_cfg = self.config.get("profiles", {}).get(self.options.get("profile", "standard_3d"), self.config.get("profiles", {}).get("standard_3d", {}))
        selected: dict[str, tuple[int | None, np.ndarray | None, float]] = {}
        for field in ("inline", "crossline", "x", "y"):
            forced = self.options.get(f"{field}_byte")
            candidates = [forced] if forced else profile_cfg.get(field, [])
            selected[field] = self._select_header(headers, candidates, endian, field)
        scalar_byte = int(self.options.get("coordinate_scalar_byte", profile_cfg.get("coordinate_scalar", 71)))
        scalars_sample = np.asarray([_int(h, scalar_byte, 2, endian, True) for h in headers])

        # Read chosen geometry fields for all traces in bounded chunks.
        full_fields: dict[str, np.ndarray | None] = {}
        for field in ("inline", "crossline", "x", "y"):
            byte, _, confidence = selected[field]
            full_fields[field] = self._read_header_field(offsets, byte, endian) if byte else None
            if byte:
                issues.append(f"{field}_byte={byte}:confidence={confidence:.3f}")
        scalars = self._read_header_field(offsets, scalar_byte, endian, size_bytes=2)
        if full_fields["x"] is not None and scalars is not None:
            full_fields["x"] = _apply_scalar(full_fields["x"], scalars)
        if full_fields["y"] is not None and scalars is not None:
            full_fields["y"] = _apply_scalar(full_fields["y"], scalars)
        geometry_confidence = float(np.mean([selected[x][2] for x in ("x", "y")]))
        if selected["inline"][0] and selected["crossline"][0]:
            geometry_confidence = float(np.mean([selected[x][2] for x in ("inline", "crossline", "x", "y")]))
        profile_name = self.options.get("profile", "standard_3d")
        if not selected["x"][0] or not selected["y"][0]:
            issues.append("xy_geometry_not_resolved")
        time_axis = np.arange(ns, dtype=float) * (dt_us / 1000.0)
        delay_byte = int(profile_cfg.get("delay_recording_time", 109))
        try:
            delay = _int(headers[0], delay_byte, 2, endian, True)
            time_axis += delay
        except Exception:
            pass
        self.geometry = SeismicGeometry(
            str(self.path), revision, endian, fmt, float(dt_us) / 1000.0, ns, int(trace_count), time_axis,
            full_fields["inline"], full_fields["crossline"], full_fields["x"], full_fields["y"], offsets,
            scalars, profile_name, geometry_confidence, issues,
        )
        return self.geometry

    def _choose_endian(self, binary: bytes, file_size: int) -> tuple[str, list[str]]:
        candidates = []
        for endian in (">", "<"):
            dt = _int(binary, 17, 2, endian, False)
            ns = _int(binary, 21, 2, endian, False)
            fmt = _int(binary, 25, 2, endian, False)
            score = int(50 <= dt <= 16000) + int(1 <= ns <= 100000) + 2 * int(fmt in SAMPLE_BYTES)
            if fmt in SAMPLE_BYTES and ns > 0:
                trace_size = 240 + ns * SAMPLE_BYTES[fmt]
                score += int((file_size - 3600) // trace_size > 0)
            candidates.append((score, endian, dt, ns, fmt))
        candidates.sort(reverse=True)
        best = candidates[0]
        return best[1], [f"endian_inferred:{'big' if best[1] == '>' else 'little'}:score={best[0]}"]

    def _read_headers(self, offsets: np.ndarray) -> list[bytes]:
        result: list[bytes] = []
        with self.path.open("rb") as handle:
            for offset in offsets:
                handle.seek(int(offset))
                result.append(handle.read(240))
        return result

    @staticmethod
    def _select_header(headers: list[bytes], candidates: list[int | None], endian: str, field: str) -> tuple[int | None, np.ndarray | None, float]:
        best: tuple[int | None, np.ndarray | None, float] = (None, None, 0.0)
        n = max(1, len(headers))
        for byte in candidates:
            if not byte:
                continue
            values = np.asarray([_int(h, int(byte), 4, endian, True) for h in headers], dtype=np.int64)
            nonzero = float(np.mean(values != 0))
            unique_ratio = len(np.unique(values)) / n
            diffs = np.diff(values)
            smooth = float(np.mean(np.abs(diffs) <= max(10, np.percentile(np.abs(diffs), 75) if diffs.size else 10))) if diffs.size else 0.0
            if field in ("x", "y"):
                magnitude = float(np.mean((np.abs(values) >= 100) & (np.abs(values) <= 2_000_000_000)))
                score = 0.35 * nonzero + 0.30 * min(1.0, unique_ratio * 5) + 0.25 * magnitude + 0.10 * smooth
            else:
                plausible = float(np.mean((values >= -10000000) & (values <= 10000000)))
                score = 0.35 * nonzero + 0.35 * min(1.0, unique_ratio * 5) + 0.20 * plausible + 0.10 * smooth
            if score > best[2]:
                best = (int(byte), values, score)
        return best

    def _read_header_field(self, offsets: np.ndarray, byte: int | None, endian: str, size_bytes: int = 4) -> np.ndarray | None:
        if byte is None:
            return None
        if len(offsets) > 1:
            stride = int(offsets[1] - offsets[0])
            if np.all(np.diff(offsets) == stride):
                dtype = np.dtype(f"{endian}i{size_bytes}")
                mapped = np.memmap(self.path, mode="r", dtype="u1")
                view = np.ndarray(
                    shape=(len(offsets),), dtype=dtype, buffer=mapped,
                    offset=int(offsets[0]) + byte - 1, strides=(stride,),
                )
                return np.asarray(view, dtype=np.int64).copy()
        values = np.empty(len(offsets), dtype=np.int64)
        with self.path.open("rb") as handle:
            for i, offset in enumerate(offsets):
                handle.seek(int(offset) + byte - 1)
                values[i] = int.from_bytes(handle.read(size_bytes), "big" if endian == ">" else "little", signed=True)
        return values

    def read_trace(self, trace_index: int, sample_slice: slice | None = None) -> np.ndarray:
        geometry = self.geometry or self.inspect()
        if trace_index < 0 or trace_index >= geometry.trace_count:
            raise IndexError(trace_index)
        bps = SAMPLE_BYTES[geometry.sample_format]
        sample_count = int(geometry.samples_per_trace)
        trace_data_offset = int(geometry.trace_offsets[trace_index]) + 240

        if sample_slice is not None and not isinstance(sample_slice, slice):
            raise TypeError("sample_slice must be a slice or None")

        # Positive slices can be served from a bounded byte window.  For a
        # stepped request we read only the span from the first through the last
        # selected sample, then apply the step in memory.  This avoids reading
        # the prefix/suffix outside the request and, most importantly, avoids
        # materializing a complete trace for cropped 3D volume access.
        if sample_slice is not None:
            start, stop, step = sample_slice.indices(sample_count)
            if step > 0:
                selected_count = len(range(start, stop, step))
                if selected_count == 0:
                    return np.empty(0, dtype=np.float32)
                last_sample = start + (selected_count - 1) * step
                window_count = last_sample - start + 1
                expected_bytes = window_count * bps
                with self.path.open("rb") as handle:
                    handle.seek(trace_data_offset + start * bps)
                    raw = handle.read(expected_bytes)
                if len(raw) != expected_bytes:
                    raise ValueError(
                        f"Truncated SEG-Y trace {trace_index}: expected {expected_bytes} bytes, got {len(raw)}"
                    )
                return _decode_samples(raw, geometry.sample_format, geometry.endian)[::step]

        # Negative-step slices intentionally fall back to the complete trace.
        # Reading backwards sample-by-sample would add substantial seek cost;
        # the fallback preserves NumPy slice semantics safely.
        expected_bytes = sample_count * bps
        with self.path.open("rb") as handle:
            handle.seek(trace_data_offset)
            raw = handle.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError(
                f"Truncated SEG-Y trace {trace_index}: expected {expected_bytes} bytes, got {len(raw)}"
            )
        values = _decode_samples(raw, geometry.sample_format, geometry.endian)
        return values[sample_slice] if sample_slice is not None else values
