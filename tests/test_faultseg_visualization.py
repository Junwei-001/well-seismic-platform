from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from well_seismic.faultseg_visualization import (
    SegySliceCache,
    build_faultseg_visualization_payload,
    decode_visualization_array,
)


class _FakeReader:
    def __init__(self, path: Path, source_shape: tuple[int, int, int]) -> None:
        self.path = path
        nz, ni, nx = source_shape
        inline, crossline = np.meshgrid(
            np.arange(100, 100 + ni),
            np.arange(200, 200 + nx),
            indexing="ij",
        )
        self.geometry = SimpleNamespace(
            samples_per_trace=nz,
            trace_count=ni * nx,
            inline=inline.ravel(),
            crossline=crossline.ravel(),
            time_axis=np.arange(nz, dtype=float) * 2.0,
        )
        self.reads: list[tuple[int, slice]] = []

    def inspect(self) -> SimpleNamespace:
        return self.geometry

    def read_trace(self, trace_index: int, sample_slice: slice | None = None) -> np.ndarray:
        # A missing slice would mean that the visualization path tried to load
        # a complete trace instead of its bounded Z request.
        assert sample_slice is not None
        self.reads.append((trace_index, sample_slice))
        nx = len(np.unique(self.geometry.crossline))
        inline_index, crossline_index = divmod(trace_index, nx)
        values = (
            np.arange(self.geometry.samples_per_trace, dtype=np.float32) * 100.0
            + inline_index * 10.0
            + crossline_index
        )
        return values[sample_slice]


class _ReaderFactory:
    def __init__(self, source_shape: tuple[int, int, int]) -> None:
        self.source_shape = source_shape
        self.instances: list[_FakeReader] = []

    def __call__(self, path: Path, config: dict, options: dict) -> _FakeReader:
        reader = _FakeReader(path, self.source_shape)
        self.instances.append(reader)
        return reader


def _faultseg_result(
    tmp_path: Path,
    *,
    source_shape: tuple[int, int, int] = (6, 4, 5),
    crop_start: tuple[int, int, int] = (0, 0, 0),
    crop_size: tuple[int, int, int] = (6, 4, 5),
) -> tuple[dict, np.ndarray]:
    source = tmp_path / "synthetic.sgy"
    source.write_bytes(b"sparse-reader-fixture")
    z, inline, crossline = np.indices(crop_size, dtype=np.float32)
    denominator = max(float((z * 20.0 + inline * 5.0 + crossline).max()), 1.0)
    probability = (z * 20.0 + inline * 5.0 + crossline) / denominator
    mask = probability >= 0.5
    probability_path = tmp_path / "faultseg_probability.npy"
    mask_path = tmp_path / "faultseg_mask.npy"
    np.save(probability_path, probability.astype(np.float32))
    np.save(mask_path, mask.astype(np.uint8))
    result = {
        "model_id": "faultseg_3d",
        "input": {
            "shape_zyx": list(crop_size),
            "axes": ["Z", "INLINE", "CROSSLINE"],
            "source": str(source),
            "source_shape_zyx": list(source_shape),
            "crop_start_zyx": list(crop_start),
            "crop_size_zyx": list(crop_size),
        },
        "inference": {"threshold": 0.5},
        "probability": {
            "shape_zyx": list(crop_size),
            "min": float(probability.min()),
            "max": float(probability.max()),
        },
        "outputs": {
            "probability_npy": str(probability_path),
            "mask_npy": str(mask_path),
        },
    }
    return result, probability


def test_payload_sparse_reads_and_faultseg_overlay_use_identical_zyx_grid(tmp_path: Path) -> None:
    result, probability = _faultseg_result(tmp_path)
    factory = _ReaderFactory((6, 4, 5))
    cache = SegySliceCache(reader_factory=factory, max_entries=2, max_bytes=1_000_000)

    payload = build_faultseg_visualization_payload(
        result,
        cache=cache,
        max_shape_zyx=(3, 2, 3),
    )

    assert payload["axes"] == ["Z", "INLINE", "CROSSLINE"]
    assert payload["sampling"]["strideZYX"] == [2, 2, 2]
    assert payload["sampling"]["sampleIndicesZYX"] == {
        "z": [0, 2, 4],
        "inline": [0, 2],
        "crossline": [0, 2, 4],
    }
    assert payload["cube"]["shape"] == [3, 2, 3]
    assert len(factory.instances) == 1
    # Six selected traces are read, rather than the complete 4 x 5 grid.
    assert len(factory.instances[0].reads) == 2 * 3
    assert all(request == slice(0, 6, 2) for _, request in factory.instances[0].reads)

    probability_payload = decode_visualization_array(payload["faultSeg"]["probability"])
    expected = probability[np.ix_([0, 2, 4], [0, 2], [0, 2, 4])]
    np.testing.assert_allclose(probability_payload, expected, atol=1.0 / 255.0)
    mask_payload = decode_visualization_array(payload["faultSeg"]["mask"])
    np.testing.assert_array_equal(mask_payload, (expected >= 0.5).astype(np.float32))
    assert probability_payload.shape == decode_visualization_array(payload["cube"]).shape
    assert payload["faultSeg"]["cigvis"]["transposeZYXToLineFirst"] == [1, 2, 0]
    assert payload["faultSeg"]["display"]["probabilityCmap"] == "jet"
    assert payload["overlays"][0]["volume"] is payload["faultSeg"]["probability"]
    assert payload["overlays"][0]["clim"] == [0.5, 1.0]
    assert payload["overlays"][0]["cmap"] == "jet"

    reads_before = len(factory.instances[0].reads)
    second = build_faultseg_visualization_payload(
        result,
        cache=cache,
        max_shape_zyx=(3, 2, 3),
    )
    assert second["preview"]["cacheHit"] is True
    assert len(factory.instances) == 1
    assert len(factory.instances[0].reads) == reads_before
    assert cache.stats["hits"] == 1


def test_lru_evicts_old_slice_requests_and_mtime_invalidates_entries(tmp_path: Path) -> None:
    result, _ = _faultseg_result(tmp_path)
    source = Path(result["input"]["source"])
    factory = _ReaderFactory((6, 4, 5))
    cache = SegySliceCache(reader_factory=factory, max_entries=2, max_bytes=1_000_000)

    common = {
        "source": source,
        "crop_size_zyx": (2, 2, 2),
        "max_shape_zyx": (2, 2, 2),
        "expected_source_shape_zyx": (6, 4, 5),
    }
    cache.get_crop(crop_start_zyx=(0, 0, 0), **common)  # A
    cache.get_crop(crop_start_zyx=(1, 1, 1), **common)  # B
    cache.get_crop(crop_start_zyx=(0, 0, 0), **common)  # touch A
    cache.get_crop(crop_start_zyx=(2, 1, 1), **common)  # C evicts B
    cache.get_crop(crop_start_zyx=(1, 1, 1), **common)  # B must be re-read
    assert len(factory.instances) == 4
    assert cache.stats["entries"] == 2
    assert cache.stats["hits"] == 1
    assert cache.stats["evictions"] == 2

    old_mtime = source.stat().st_mtime_ns
    os.utime(source, ns=(old_mtime + 10_000_000, old_mtime + 10_000_000))
    cache.get_crop(crop_start_zyx=(0, 0, 0), **common)
    assert len(factory.instances) == 5
    assert cache.stats["misses"] == 5


def test_cache_byte_limit_skips_oversized_entry(tmp_path: Path) -> None:
    result, _ = _faultseg_result(tmp_path)
    factory = _ReaderFactory((6, 4, 5))
    cache = SegySliceCache(reader_factory=factory, max_entries=2, max_bytes=1)

    crop, hit = cache.get_crop(
        result["input"]["source"],
        crop_start_zyx=(0, 0, 0),
        crop_size_zyx=(2, 2, 2),
        max_shape_zyx=(2, 2, 2),
        expected_source_shape_zyx=(6, 4, 5),
    )

    assert crop.nbytes > 1
    assert hit is False
    assert cache.stats["entries"] == 0
    assert cache.stats["skipped"] == 1


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda result, tmp: result["input"].__setitem__("axes", ["INLINE", "CROSSLINE", "Z"]), "axes"),
        (lambda result, tmp: result["input"].__setitem__("source_shape_zyx", [5, 4, 5]), "shape"),
        (lambda result, tmp: result["input"].__setitem__("crop_start_zyx", [1, 0, 0]), "crop"),
        (lambda result, tmp: result["input"].__setitem__("source", str(tmp / "missing.sgy")), "source"),
        (lambda result, tmp: np.save(result["outputs"]["probability_npy"], np.zeros((2, 2, 2), dtype=np.float32)), "probability array shape"),
    ],
)
def test_result_contract_rejects_misaligned_axes_shape_source_and_crop(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    result, _ = _faultseg_result(tmp_path)
    mutate(result, tmp_path)
    cache = SegySliceCache(reader_factory=_ReaderFactory((6, 4, 5)))

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        build_faultseg_visualization_payload(result, cache=cache)


def test_visualization_endpoint_preserves_builder_overlay_contract(monkeypatch) -> None:
    from well_seismic import api

    builder_overlay = {
        "id": "faultseg_probability",
        "name": "FaultSeg 断层概率",
        "kind": "probability",
        "volume": {"encoding": "base64-uint8", "shape": [2, 2, 2], "data": ""},
        "clim": [0.5, 1.0],
        "cmap": "jet",
        "alpha": 0.62,
        "excpt": "min",
        "contractMarker": "owned-by-builder",
    }
    builder_payload = {
        "faultSeg": {
            "threshold": 0.5,
            "display": {"probabilityCmap": "jet", "alpha": 0.62},
        },
        "overlays": [builder_overlay],
        "embeddedWells": [{"name": "should-be-cleared-for-prediction"}],
        "preview": {},
    }
    tasks = {
        "source-task": {
            "task_type": "data_preparation",
            "result": {"visualization_preview": {"volumes": [], "lines2d": []}},
        },
        "prediction-task": {
            "task_type": "model_prediction",
            "parent_task_id": "source-task",
            "result": {
                "prediction": {"input": {"geometry_profile": "standard_3d"}},
            },
        },
    }
    captured: dict = {}

    monkeypatch.setattr(api, "_tasks", tasks)
    builder_payload["predictionVisualization"] = {
        "modelId": "faultseg_3d",
        "modelName": "FaultSeg",
        "preferredLayer": "probability",
        "overlayCount": 1,
    }
    monkeypatch.setattr(
        api,
        "build_prediction_visualization_payload",
        lambda *_args, **_kwargs: builder_payload,
    )

    def fake_render(_project_root, preview, **kwargs):
        captured["preview"] = preview
        captured["kwargs"] = kwargs
        return "<html><body>ok</body></html>"

    monkeypatch.setattr(api, "render_cigvis_workbench", fake_render)

    response = api.unified_data_visualization(task_id="prediction-task")

    assert response.status_code == 200
    rendered_volume = captured["preview"]["volumes"][0]
    assert rendered_volume["overlays"] == [builder_overlay]
    assert rendered_volume["overlays"][0]["cmap"] == "jet"
    assert rendered_volume["overlays"][0]["contractMarker"] == "owned-by-builder"
    assert rendered_volume["embeddedWells"] == []
