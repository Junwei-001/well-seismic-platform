from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..coordinate_reference import canonical_crs_id, transform_xy
from ..models import SeismicGeometry

# Bytes-per-sample for SEG-Y rev 0/1/2 format codes.  This complete table is
# only used to identify endian/header plausibility.  Reading is deliberately
# limited to the formats implemented by ``_decode_samples`` below: knowing a
# sample width is not the same as being able to decode its numeric encoding.
SEG_Y_SAMPLE_BYTES = {
    1: 4,
    2: 4,
    3: 2,
    5: 4,
    6: 8,
    7: 3,
    8: 1,
    9: 8,
    10: 4,
    11: 2,
    12: 8,
    15: 3,
    16: 1,
}
SAMPLE_BYTES = {code: SEG_Y_SAMPLE_BYTES[code] for code in (1, 2, 3, 5, 6, 8)}
TRACE_HEADER_SIZE_BYTES = 240
# SEG-Y trace-sequence / ensemble headers can look statistically perfect when
# scored one field at a time, even though they do not describe a 3-D grid.  The
# values still have to be sequence-like before these byte positions are treated
# as unsafe; this preserves legacy surveys that deliberately reuse a field.
_TRACE_SEQUENCE_LIKE_BYTES = {
    "inline": frozenset({1, 5}),
    "crossline": frozenset({1, 5, 21}),
}


class SourceFileIdentityError(RuntimeError):
    """A sealed SEG-Y path has stat-visible drift after snapshot verification."""


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


def _validate_trace_header_start(field: str, byte: Any, size_bytes: int) -> int:
    """Validate a 1-based field start against the 240-byte trace header."""
    if size_bytes not in (2, 4):
        raise ValueError(
            f"Unsupported SEG-Y trace-header width {size_bytes} for {field!r}"
        )
    if isinstance(byte, bool) or not isinstance(byte, (int, np.integer)):
        raise ValueError(
            f"SEG-Y trace-header field {field!r} must use an integer byte start"
        )
    start = int(byte)
    maximum = TRACE_HEADER_SIZE_BYTES - size_bytes + 1
    if not 1 <= start <= maximum:
        raise ValueError(
            f"SEG-Y trace-header field {field!r} is {size_bytes} bytes and must "
            f"start within 1..{maximum}; got {start}"
        )
    return start


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


def _is_trace_sequence_like(values: np.ndarray) -> bool:
    """Return true for a near-complete monotonically numbered trace sequence."""

    array = np.asarray(values, dtype=np.int64)
    if array.size < 4:
        return False
    unique_ratio = float(np.unique(array).size) / float(array.size)
    if unique_ratio < 0.98:
        return False
    differences = np.diff(array)
    return bool(
        differences.size
        and float(np.mean(np.abs(differences) == 1)) >= 0.98
    )


def _assess_inline_crossline_grid(
    inline: np.ndarray | None,
    crossline: np.ndarray | None,
    *,
    trace_count: int,
    inline_byte: int | None,
    crossline_byte: int | None,
    inline_is_automatic: bool,
    crossline_is_automatic: bool,
    maximum_cell_trace_ratio: float,
) -> dict[str, Any]:
    """Cross-check independently selected headers against a regular 3-D grid."""

    if inline is None or crossline is None:
        return {
            "confidence": 1.0,
            "fatal_reasons": [],
            "issue": "inline_crossline_grid=unresolved",
        }
    inline_values = np.asarray(inline, dtype=np.int64)
    crossline_values = np.asarray(crossline, dtype=np.int64)
    fatal_reasons: list[str] = []
    if inline_values.size != trace_count or crossline_values.size != trace_count:
        fatal_reasons.append("trace_count_mismatch")
    unique_inline = int(np.unique(inline_values).size)
    unique_crossline = int(np.unique(crossline_values).size)
    unique_cells = unique_inline * unique_crossline
    cell_trace_ratio = float(unique_cells) / float(max(1, trace_count))
    # Identical non-constant axes cannot define two independent dimensions.
    if unique_inline > 1 and np.array_equal(inline_values, crossline_values):
        fatal_reasons.append("inline_crossline_same_values")
    maximum_ratio = float(maximum_cell_trace_ratio)
    if not np.isfinite(maximum_ratio) or maximum_ratio < 1.0:
        raise ValueError(
            "maximum_grid_cell_trace_ratio must be finite and at least 1.0"
        )
    if cell_trace_ratio > maximum_ratio:
        fatal_reasons.append("unique_product_exceeds_regular_grid_limit")
    automatic_fields = (
        ("inline", inline_values, inline_byte, inline_is_automatic),
        (
            "crossline",
            crossline_values,
            crossline_byte,
            crossline_is_automatic,
        ),
    )
    for field, values, byte, is_automatic in automatic_fields:
        if (
            is_automatic
            and byte in _TRACE_SEQUENCE_LIKE_BYTES[field]
            and _is_trace_sequence_like(values)
        ):
            fatal_reasons.append(
                f"automatic_{field}_trace_sequence_like_byte_{byte}"
            )
    # A sparse rectangular grid loses confidence in proportion to its empty
    # Cartesian cells.  The hard cap above remains independent of the global
    # minimum-confidence setting, so a manifest cannot silently opt out.
    grid_confidence = min(1.0, 1.0 / max(1.0, cell_trace_ratio))
    return {
        "confidence": grid_confidence,
        "fatal_reasons": fatal_reasons,
        "issue": (
            "inline_crossline_grid="
            f"unique_inline:{unique_inline},"
            f"unique_crossline:{unique_crossline},"
            f"unique_product:{unique_cells},"
            f"trace_count:{trace_count},"
            f"cell_trace_ratio:{cell_trace_ratio:.6f},"
            f"confidence:{grid_confidence:.3f}"
        ),
    }


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
        self.expected_source_stat_signature: tuple[int, int, int, int, int] | None = None

    def bind_expected_source_stat_signature(self, signature: Any) -> None:
        if (
            not isinstance(signature, (list, tuple))
            or len(signature) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in signature
            )
        ):
            raise SourceFileIdentityError(
                f"Invalid verified SEG-Y source identity: {self.path}"
            )
        self.expected_source_stat_signature = tuple(
            int(value) for value in signature
        )  # type: ignore[assignment]
        self._assert_expected_source_identity()

    def _assert_expected_source_identity(self, handle: Any = None) -> None:
        expected = self.expected_source_stat_signature
        if expected is None:
            return
        try:
            observed = _stat_signature(
                os.fstat(handle.fileno()) if handle is not None else self.path.stat()
            )
        except OSError as exc:
            raise SourceFileIdentityError(
                f"Verified SEG-Y source is no longer readable: {self.path}"
            ) from exc
        if observed != expected:
            raise SourceFileIdentityError(
                "Verified SEG-Y source stat identity changed before or during "
                "amplitude/geometry "
                f"read: {self.path}"
            )

    def _read_verified_bytes(self, offset: int, size: int) -> bytes:
        self._assert_expected_source_identity()
        with self.path.open("rb") as handle:
            self._assert_expected_source_identity(handle)
            handle.seek(offset)
            raw = handle.read(size)
            self._assert_expected_source_identity(handle)
        self._assert_expected_source_identity()
        return raw

    def inspect(self, progress: Any = None) -> SeismicGeometry:
        self._assert_expected_source_identity()
        size = self.path.stat().st_size
        if size < 3600:
            raise ValueError(f"File is too small to be SEG-Y: {self.path}")
        with self.path.open("rb") as handle:
            handle.seek(3200)
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
        if fmt not in SEG_Y_SAMPLE_BYTES:
            raise ValueError(
                f"Unknown SEG-Y sample format code {fmt}: {self.path}"
            )
        bytes_per_sample = SAMPLE_BYTES.get(fmt)
        if bytes_per_sample is None:
            supported = ", ".join(str(code) for code in sorted(SAMPLE_BYTES))
            raise ValueError(
                "SEG-Y sample format code "
                f"{fmt} is recognized but decoding is not implemented; "
                f"supported codes are {supported}: {self.path}"
            )
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
        profiles = self.config.get("profiles", {})
        requested_profile = self.options.get("profile")
        profile_name = str(requested_profile or "standard_3d")
        profile_is_explicit = (
            requested_profile is not None
            and self.options.get("profile_source") != "automatic_default"
        )
        custom_explicit_profile = False
        if requested_profile is not None and profile_name not in profiles:
            explicit_geometry_fields = (
                "inline_byte",
                "crossline_byte",
                "x_byte",
                "y_byte",
                "coordinate_scalar_byte",
            )
            if not all(
                self.options.get(field) is not None
                for field in explicit_geometry_fields
            ):
                raise ValueError(
                    f"Unknown explicit SEG-Y geometry profile {profile_name!r}: "
                    f"{self.path}; provide all five trace-header byte fields"
                )
            # A user-defined profile label is provenance, not a hidden parser
            # dependency, when every geometry byte is sealed explicitly.  Use
            # the standard profile only for non-geometry defaults such as the
            # delay-recording-time byte and preserve the declared label.
            custom_explicit_profile = True
        profile_cfg = profiles.get(profile_name, profiles.get("standard_3d", {}))
        selected: dict[str, tuple[int | None, np.ndarray | None, float]] = {}
        for field in ("inline", "crossline", "x", "y"):
            forced = self.options.get(f"{field}_byte")
            candidates = [forced] if forced is not None else profile_cfg.get(field, [])
            selected[field] = self._select_header(headers, candidates, endian, field)
        scalar_byte = _validate_trace_header_start(
            "coordinate_scalar",
            self.options.get(
                "coordinate_scalar_byte",
                profile_cfg.get("coordinate_scalar", 71),
            ),
            2,
        )

        # Read every selected field from one bounded trace-header window per
        # chunk.  The previous implementation traversed a large SEG-Y once per
        # field, which caused five cold-cache passes over the same file.
        field_specs: dict[str, tuple[int, int]] = {}
        for field in ("inline", "crossline", "x", "y"):
            byte = selected[field][0]
            if byte is not None:
                field_specs[field] = (int(byte), 4)
        field_specs["coordinate_scalar"] = (scalar_byte, 2)
        header_values = self._read_header_fields(
            offsets,
            field_specs,
            endian,
            progress=progress,
        )
        full_fields: dict[str, np.ndarray | None] = {}
        for field in ("inline", "crossline", "x", "y"):
            byte, _, confidence = selected[field]
            full_fields[field] = header_values.get(field)
            if byte:
                issues.append(f"{field}_byte={byte}:confidence={confidence:.3f}")
        issues.append(f"coordinate_scalar_byte={scalar_byte}:configured")
        if custom_explicit_profile:
            issues.append(f"custom_explicit_geometry_profile={profile_name}")
        grid_assessment = _assess_inline_crossline_grid(
            full_fields["inline"],
            full_fields["crossline"],
            trace_count=int(trace_count),
            inline_byte=selected["inline"][0],
            crossline_byte=selected["crossline"][0],
            inline_is_automatic=(
                not profile_is_explicit
                and self.options.get("inline_byte") is None
            ),
            crossline_is_automatic=(
                not profile_is_explicit
                and self.options.get("crossline_byte") is None
            ),
            maximum_cell_trace_ratio=float(
                self.config.get("maximum_grid_cell_trace_ratio", 4.0)
            ),
        )
        issues.append(str(grid_assessment["issue"]))
        fatal_grid_reasons = list(grid_assessment["fatal_reasons"])
        if fatal_grid_reasons:
            raise ValueError(
                "Invalid SEG-Y inline/crossline geometry: "
                + ", ".join(fatal_grid_reasons)
                + f"; {grid_assessment['issue']}; "
                + f"inline_byte={selected['inline'][0]}, "
                + f"crossline_byte={selected['crossline'][0]}: {self.path}"
            )
        scalars = header_values.get("coordinate_scalar")
        if full_fields["x"] is not None and scalars is not None:
            full_fields["x"] = _apply_scalar(full_fields["x"], scalars)
        if full_fields["y"] is not None and scalars is not None:
            full_fields["y"] = _apply_scalar(full_fields["y"], scalars)
        source_crs = str(self.options.get("source_crs") or "").strip() or None
        target_crs = str(self.options.get("target_crs") or "").strip() or None
        coordinate_transform: dict[str, Any] = {}
        resolved_horizontal_crs = (
            canonical_crs_id(source_crs, field="SEG-Y源CRS")
            if source_crs
            else None
        )
        if source_crs and target_crs:
            if (full_fields["x"] is None) != (full_fields["y"] is None):
                raise ValueError("SEG-Y X/Y道头必须同时存在才能重投影")
            if full_fields["x"] is not None and full_fields["y"] is not None:
                transformed = transform_xy(
                    full_fields["x"],
                    full_fields["y"],
                    source_crs=source_crs,
                    target_crs=target_crs,
                )
                full_fields["x"] = transformed.x
                full_fields["y"] = transformed.y
                resolved_horizontal_crs = transformed.target_crs
                coordinate_transform = transformed.provenance()
        geometry_confidence = float(np.mean([selected[x][2] for x in ("x", "y")]))
        if selected["inline"][0] and selected["crossline"][0]:
            geometry_confidence = float(np.mean([selected[x][2] for x in ("inline", "crossline", "x", "y")]))
            geometry_confidence = min(
                geometry_confidence,
                float(grid_assessment["confidence"]),
            )
        minimum_confidence = float(
            self.config.get("minimum_geometry_confidence", 0.75)
        )
        explicit_xy_bytes = all(
            self.options.get(f"{field}_byte") is not None for field in ("x", "y")
        )
        inline_crossline_resolved = bool(
            selected["inline"][0] and selected["crossline"][0]
        )
        explicit_inline_crossline_bytes = all(
            self.options.get(f"{field}_byte") is not None
            for field in ("inline", "crossline")
        )
        explicit_geometry_bytes = explicit_xy_bytes and (
            not inline_crossline_resolved or explicit_inline_crossline_bytes
        )
        if (
            geometry_confidence < minimum_confidence
            and not profile_is_explicit
            and not explicit_geometry_bytes
        ):
            raise ValueError(
                "Low-confidence automatic SEG-Y trace-header selection "
                f"({geometry_confidence:.3f} < {minimum_confidence:.3f}); "
                "declare a known options.profile or explicit inline_byte/"
                "crossline_byte/x_byte/y_byte "
                f"before spatial matching: {self.path}"
            )
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
            source_crs=(
                canonical_crs_id(source_crs, field="SEG-Y源CRS")
                if source_crs
                else None
            ),
            horizontal_crs=resolved_horizontal_crs,
            coordinate_transform=coordinate_transform,
        )
        self._assert_expected_source_identity()
        return self.geometry

    def _choose_endian(self, binary: bytes, file_size: int) -> tuple[str, list[str]]:
        candidates = []
        for endian in (">", "<"):
            dt = _int(binary, 17, 2, endian, False)
            ns = _int(binary, 21, 2, endian, False)
            fmt = _int(binary, 25, 2, endian, False)
            score = (
                int(50 <= dt <= 16000)
                + int(1 <= ns <= 100000)
                + 2 * int(fmt in SEG_Y_SAMPLE_BYTES)
            )
            if fmt in SEG_Y_SAMPLE_BYTES and ns > 0:
                trace_size = 240 + ns * SEG_Y_SAMPLE_BYTES[fmt]
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
            if byte is None:
                continue
            byte = _validate_trace_header_start(field, byte, 4)
            values = np.asarray([_int(h, byte, 4, endian, True) for h in headers], dtype=np.int64)
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
        byte = _validate_trace_header_start("trace_header_field", byte, size_bytes)
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

    def _read_header_fields(
        self,
        offsets: np.ndarray,
        fields: dict[str, tuple[int, int]],
        endian: str,
        *,
        progress: Any = None,
    ) -> dict[str, np.ndarray]:
        if not fields:
            return {}
        fields = {
            name: (
                _validate_trace_header_start(name, byte, size_bytes),
                size_bytes,
            )
            for name, (byte, size_bytes) in fields.items()
        }
        count = len(offsets)
        results = {
            name: np.empty(count, dtype=np.int64)
            for name in fields
        }
        if count == 0:
            return results
        if count > 1:
            stride = int(offsets[1] - offsets[0])
            regular = bool(np.all(np.diff(offsets) == stride))
        else:
            stride = 0
            regular = False
        if not regular:
            for name, (byte, size_bytes) in fields.items():
                values = self._read_header_field(
                    offsets,
                    byte,
                    endian,
                    size_bytes=size_bytes,
                )
                if values is not None:
                    results[name][:] = values
            if progress:
                progress(count, count)
            return results

        first_byte = min(byte - 1 for byte, _ in fields.values())
        final_byte = max(
            byte - 1 + size_bytes
            for byte, size_bytes in fields.values()
        )
        window_size = final_byte - first_byte
        configured_chunk = int(self.config.get("geometry_header_chunk_traces", 65_536))
        chunk_size = max(1_024, min(configured_chunk, 262_144))
        mapped = np.memmap(self.path, mode="r", dtype="u1")
        if progress:
            progress(0, count)
        try:
            for start in range(0, count, chunk_size):
                stop = min(count, start + chunk_size)
                header_window = np.ndarray(
                    shape=(stop - start, window_size),
                    dtype="u1",
                    buffer=mapped,
                    offset=int(offsets[start]) + first_byte,
                    strides=(stride, 1),
                )
                block = np.array(header_window, dtype=np.uint8, copy=True, order="C")
                for name, (byte, size_bytes) in fields.items():
                    relative = byte - 1 - first_byte
                    raw = np.ascontiguousarray(
                        block[:, relative:relative + size_bytes]
                    )
                    values = raw.view(np.dtype(f"{endian}i{size_bytes}")).reshape(-1)
                    results[name][start:stop] = values.astype(np.int64, copy=False)
                if progress:
                    progress(stop, count)
        finally:
            del mapped
        return results

    def read_trace(self, trace_index: int, sample_slice: slice | None = None) -> np.ndarray:
        self._assert_expected_source_identity()
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
                    self._assert_expected_source_identity()
                    return np.empty(0, dtype=np.float32)
                last_sample = start + (selected_count - 1) * step
                window_count = last_sample - start + 1
                expected_bytes = window_count * bps
                raw = self._read_verified_bytes(
                    trace_data_offset + start * bps,
                    expected_bytes,
                )
                if len(raw) != expected_bytes:
                    raise ValueError(
                        f"Truncated SEG-Y trace {trace_index}: expected {expected_bytes} bytes, got {len(raw)}"
                    )
                return _decode_samples(raw, geometry.sample_format, geometry.endian)[::step]

        # Negative-step slices intentionally fall back to the complete trace.
        # Reading backwards sample-by-sample would add substantial seek cost;
        # the fallback preserves NumPy slice semantics safely.
        expected_bytes = sample_count * bps
        raw = self._read_verified_bytes(trace_data_offset, expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError(
                f"Truncated SEG-Y trace {trace_index}: expected {expected_bytes} bytes, got {len(raw)}"
            )
        values = _decode_samples(raw, geometry.sample_format, geometry.endian)
        return values[sample_slice] if sample_slice is not None else values
