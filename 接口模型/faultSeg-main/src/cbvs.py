"""Reader for post-stack OpendTect CBVS seismic cubes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class CBVSComponent:
    name: str
    dtype: np.dtype
    start_ms: float
    step_ms: float
    sample_count: int
    byte_offset: int


@dataclass(frozen=True)
class CBVSGeometry:
    inline_start: int
    inline_stop: int
    inline_step: int
    crossline_start: int
    crossline_stop: int
    crossline_step: int
    fully_rectangular: bool


class CBVSVolume:
    """Memory-mapped CBVS reader returning arrays in ``[time, inline, crossline]`` order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as stream:
            prefix = stream.read(16)
            if len(prefix) != 16 or prefix[:3] != b"dGB":
                raise ValueError(f"not a CBVS file: {self.path}")
            self.endian = "<" if prefix[3] else ">"
            self.version = int(prefix[4])
            self.aux_selection = int(prefix[5])
            self.coordinate_policy = int(prefix[6])
            self.header_bytes = struct.unpack(self.endian + "i", prefix[8:12])[0]
            stream.seek(16)
            self.standard_text = self._read_text(stream)
            specifications = []
            for _ in range(self._read_int(stream)):
                name = self._read_text(stream)
                self._read_int(stream)  # OpendTect data type identifier.
                characteristics = stream.read(4)
                dtype = self._decode_dtype(characteristics[0], characteristics[1])
                start, step = struct.unpack(self.endian + "ff", stream.read(8))
                sample_count = self._read_int(stream)
                stream.read(8)  # Reserved linear scaling fields.
                specifications.append((name, dtype, start * 1000.0, step * 1000.0, sample_count))
            geometry = struct.unpack(self.endian + "8i", stream.read(32))
            stream.read(48)  # Inline/crossline-to-XY transform.
            self.sequence_number = self._read_int(stream)
            self.user_text = self._read_text(stream).rstrip()
            data_offset = stream.tell()
        if data_offset != self.header_bytes:
            raise ValueError(
                f"CBVS header length mismatch in {self.path}: {data_offset} != {self.header_bytes}"
            )

        rectangular, traces_per_position, il0, xl0, il1, xl1, dil, dxl = geometry
        if traces_per_position != 1:
            raise ValueError("pre-stack CBVS files are not supported")
        self.geometry = CBVSGeometry(
            il0, il1, dil or 1, xl0, xl1, dxl or 1, bool(rectangular)
        )
        auxiliary_bytes = self._auxiliary_byte_count()
        component_offset = auxiliary_bytes
        components = []
        for name, dtype, start_ms, step_ms, sample_count in specifications:
            components.append(
                CBVSComponent(name, dtype, start_ms, step_ms, sample_count, component_offset)
            )
            component_offset += dtype.itemsize * sample_count
        self.components = tuple(components)
        self.record_bytes = component_offset

        positions = self._read_positions()
        self.trace_count = len(positions)
        expected_size = data_offset + self.trace_count * self.record_bytes + self._trailer_size() + 8
        if expected_size != self.path.stat().st_size:
            raise ValueError(f"CBVS data length mismatch in {self.path}")
        shape = (
            (il1 - il0) // self.geometry.inline_step + 1,
            (xl1 - xl0) // self.geometry.crossline_step + 1,
        )
        self.trace_lookup = np.full(shape, -1, dtype=np.int32)
        for trace_index, (inline, crossline) in enumerate(positions):
            yi = (inline - il0) // self.geometry.inline_step
            xi = (crossline - xl0) // self.geometry.crossline_step
            if 0 <= yi < shape[0] and 0 <= xi < shape[1]:
                self.trace_lookup[yi, xi] = trace_index
        self._raw = np.memmap(
            self.path,
            mode="r",
            dtype=np.uint8,
            offset=data_offset,
            shape=(self.trace_count, self.record_bytes),
        )

    def _read_int(self, stream) -> int:
        raw = stream.read(4)
        if len(raw) != 4:
            raise ValueError(f"truncated CBVS header: {self.path}")
        return int(struct.unpack(self.endian + "i", raw)[0])

    def _read_text(self, stream) -> str:
        length = self._read_int(stream)
        if length < 0:
            raise ValueError(f"negative CBVS text length: {self.path}")
        return stream.read(length).decode(errors="replace")

    def _decode_dtype(self, first: int, second: int) -> np.dtype:
        byte_count = 2 ** (first & 0x07)
        is_integer = bool(first & 0x08)
        is_signed = bool(first & 0x10)
        byte_order = "<" if (second & 0x80 or second & 0x01) else ">"
        if is_integer:
            kind = "i" if is_signed else "u"
        elif byte_count in (4, 8):
            kind = "f"
        else:
            raise ValueError(f"unsupported CBVS float width {byte_count}")
        return np.dtype(f"{byte_order}{kind}{byte_count}")

    def _auxiliary_byte_count(self) -> int:
        count = sum(4 for bit in (1, 4, 8, 16, 32) if self.aux_selection & bit)
        if self.aux_selection & 2 and self.coordinate_policy == 0:
            count += 16
        return count

    def _trailer_size(self) -> int:
        with self.path.open("rb") as stream:
            stream.seek(-8, 2)
            raw = stream.read(8)
        if len(raw) != 8 or raw[5:] != b"BGd":
            raise ValueError(f"missing CBVS trailer marker: {self.path}")
        return int(struct.unpack(self.endian + "i", raw[:4])[0])

    def _read_positions(self) -> list[tuple[int, int]]:
        geometry = self.geometry
        if geometry.fully_rectangular:
            return [
                (inline, crossline)
                for inline in range(
                    geometry.inline_start,
                    geometry.inline_stop + geometry.inline_step,
                    geometry.inline_step,
                )
                for crossline in range(
                    geometry.crossline_start,
                    geometry.crossline_stop + geometry.crossline_step,
                    geometry.crossline_step,
                )
            ]
        trailer_size = self._trailer_size()
        with self.path.open("rb") as stream:
            stream.seek(-8 - trailer_size, 2)
            trailer = stream.read(trailer_size)
        offset = 0
        if self.coordinate_policy == 1:
            coordinate_count = struct.unpack_from(self.endian + "i", trailer, offset)[0]
            offset += 4 + coordinate_count * 16
        inline_count = struct.unpack_from(self.endian + "i", trailer, offset)[0]
        offset += 4
        positions = []
        for _ in range(inline_count):
            inline, segment_count = struct.unpack_from(self.endian + "ii", trailer, offset)
            offset += 8
            for _ in range(segment_count):
                start, stop, step = struct.unpack_from(self.endian + "iii", trailer, offset)
                offset += 12
                positions.extend((inline, crossline) for crossline in range(start, stop + step, step))
        if offset != len(trailer):
            raise ValueError(f"unparsed CBVS trailer bytes in {self.path}")
        return positions

    def component_view(self, component: int | str = 0) -> np.ndarray:
        """Return a zero-copy ``[trace, sample]`` view of one component."""
        if isinstance(component, str):
            matches = [i for i, item in enumerate(self.components) if item.name == component]
            if not matches:
                raise KeyError(component)
            component = matches[0]
        item = self.components[int(component)]
        return np.ndarray(
            (self.trace_count, item.sample_count),
            dtype=item.dtype,
            buffer=self._raw,
            offset=item.byte_offset,
            strides=(self.record_bytes, item.dtype.itemsize),
        )

    def read_crop(
        self,
        sample_slice: slice,
        inline_slice: slice,
        crossline_slice: slice,
        component: int | str = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read a regular crop and return ``(data, valid_trace_mask)``."""
        item_index = component
        if isinstance(component, str):
            item_index = next(i for i, item in enumerate(self.components) if item.name == component)
        item = self.components[int(item_index)]
        samples = np.arange(item.sample_count)[sample_slice]
        inline_indices = np.arange(self.trace_lookup.shape[0])[inline_slice]
        crossline_indices = np.arange(self.trace_lookup.shape[1])[crossline_slice]
        lookup = self.trace_lookup[np.ix_(inline_indices, crossline_indices)]
        valid = lookup >= 0
        result = np.full((len(samples), len(inline_indices), len(crossline_indices)), np.nan, np.float32)
        if valid.any():
            values = self.component_view(int(item_index))[lookup[valid]][:, samples]
            result.reshape(len(samples), -1)[:, valid.ravel()] = np.asarray(values, np.float32).T
            result[np.abs(result) > 1e29] = np.nan
        return result, valid
