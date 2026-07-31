from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from well_seismic.io.segy import SegyReader


class _TrackedBinaryHandle:
    def __init__(self, handle: Any, reads: list[tuple[int, int, int]]) -> None:
        self._handle = handle
        self._reads = reads

    def __enter__(self) -> "_TrackedBinaryHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._handle.close()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        offset = self._handle.tell()
        result = self._handle.read(size)
        self._reads.append((offset, size, len(result)))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _reader_with_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SegyReader, list[tuple[int, int, int]]]:
    path = tmp_path / "partial.sgy"
    values = np.arange(10, dtype=">f4")
    path.write_bytes(bytes(240) + values.tobytes())
    reader = SegyReader(path)
    reader.geometry = SimpleNamespace(
        trace_count=1,
        samples_per_trace=10,
        sample_format=5,
        endian=">",
        trace_offsets=np.asarray([0], dtype=np.int64),
    )
    reads: list[tuple[int, int, int]] = []
    original_open = Path.open

    def tracked_open(target: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(target, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if target == path and mode == "rb":
            return _TrackedBinaryHandle(handle, reads)
        return handle

    monkeypatch.setattr(Path, "open", tracked_open)
    return reader, reads


def test_positive_stepped_slice_reads_only_its_byte_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, reads = _reader_with_tracking(tmp_path, monkeypatch)

    values = reader.read_trace(0, slice(2, 8, 2))

    np.testing.assert_array_equal(values, [2.0, 4.0, 6.0])
    # Selected samples are 2, 4 and 6, so the bounded span is samples 2..6:
    # five float32 values (20 bytes), not the complete 40-byte trace.
    assert reads == [(240 + 2 * 4, 5 * 4, 5 * 4)]


def test_positive_unit_step_and_empty_slice_do_not_read_unrequested_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, reads = _reader_with_tracking(tmp_path, monkeypatch)

    np.testing.assert_array_equal(reader.read_trace(0, slice(-4, None)), [6.0, 7.0, 8.0, 9.0])
    assert reads == [(240 + 6 * 4, 4 * 4, 4 * 4)]
    reads.clear()
    assert reader.read_trace(0, slice(7, 2)).size == 0
    assert reads == []


def test_negative_step_safely_falls_back_to_complete_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, reads = _reader_with_tracking(tmp_path, monkeypatch)

    values = reader.read_trace(0, slice(8, 2, -2))

    np.testing.assert_array_equal(values, [8.0, 6.0, 4.0])
    assert reads == [(240, 10 * 4, 10 * 4)]


def test_zero_step_is_rejected_without_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, reads = _reader_with_tracking(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="slice step cannot be zero"):
        reader.read_trace(0, slice(None, None, 0))
    assert reads == []
