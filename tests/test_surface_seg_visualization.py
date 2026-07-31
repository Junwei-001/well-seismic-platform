from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from well_seismic.faultseg_visualization import (
    SegySliceCache,
    decode_visualization_array,
)
from well_seismic.prediction_visualization import (
    build_prediction_visualization_payload,
)
from well_seismic.surface_seg_visualization import (
    build_surface_seg_visualization_payload,
)


class _FakeReader:
    def __init__(self, path: Path, source_shape_zyx: tuple[int, int, int]) -> None:
        self.path = path
        nz, ni, nx = source_shape_zyx
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

    def read_trace(
        self,
        trace_index: int,
        sample_slice: slice | None = None,
    ) -> np.ndarray:
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
    def __init__(self, source_shape_zyx: tuple[int, int, int]) -> None:
        self.source_shape_zyx = source_shape_zyx
        self.instances: list[_FakeReader] = []

    def __call__(self, path: Path, config: dict, options: dict) -> _FakeReader:
        reader = _FakeReader(path, self.source_shape_zyx)
        self.instances.append(reader)
        return reader


def _surface_result(
    tmp_path: Path,
    *,
    source_shape_zyx: tuple[int, int, int] = (6, 4, 5),
    output_inline_count: int = 2,
    max_inlines: int | None = 2,
) -> tuple[dict, np.ndarray, np.ndarray]:
    source = tmp_path / "surface.sgy"
    source.write_bytes(b"sparse-reader-fixture")
    nz, _, nx = source_shape_zyx
    inline, crossline, sample = np.indices(
        (output_inline_count, nx, nz),
        dtype=np.int16,
    )
    # Include label 0 deliberately: it is a valid stratum, not the
    # missing-trace sentinel (-1).
    labels = ((inline + crossline + sample) % 4).astype(np.int16)
    labels[0, -1, :] = -1
    confidence = (
        0.2
        + 0.7
        * (
            inline.astype(np.float32) * 10.0
            + crossline.astype(np.float32) * 2.0
            + sample.astype(np.float32)
        )
        / max(
            float(
                (
                    inline.astype(np.float32) * 10.0
                    + crossline.astype(np.float32) * 2.0
                    + sample.astype(np.float32)
                ).max()
            ),
            1.0,
        )
    ).astype(np.float16)
    mask_path = tmp_path / "mask.npy"
    confidence_path = tmp_path / "confidence.npy"
    np.save(mask_path, labels)
    np.save(confidence_path, confidence)
    result = {
        "model_id": "seismic_surface_seg",
        "model_name": "Seismic Surface Seg 地层分割",
        "input": {
            "source": str(source),
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "shape_ics": list(labels.shape),
            "source_shape_zyx": list(source_shape_zyx),
        },
        "inference": {
            "max_inlines": max_inlines,
            "query_threshold": 0.35,
        },
        "segmentation": {
            "shape_ics": list(labels.shape),
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "label_range": [-1, 3],
            "invalid_label": -1,
            "cross_inline_consistent": False,
        },
        "outputs": {
            "mask_npy": str(mask_path),
            "confidence_npy": str(confidence_path),
        },
    }
    return result, labels, confidence


def test_surface_payload_transposes_ics_to_zyx_on_same_sparse_grid(
    tmp_path: Path,
) -> None:
    result, labels, _ = _surface_result(tmp_path)
    factory = _ReaderFactory((6, 4, 5))
    cache = SegySliceCache(
        reader_factory=factory,
        max_entries=2,
        max_bytes=1_000_000,
    )

    payload = build_surface_seg_visualization_payload(
        result,
        cache=cache,
        max_shape_zyx=(3, 2, 3),
    )

    assert payload["axes"] == ["Z", "INLINE", "CROSSLINE"]
    assert payload["sampling"]["strideZYX"] == [2, 1, 2]
    assert payload["sampling"]["sampleIndicesZYX"] == {
        "z": [0, 2, 4],
        "inline": [0, 1],
        "crossline": [0, 2, 4],
    }
    assert payload["cube"]["shape"] == [3, 2, 3]
    assert len(factory.instances) == 1
    # Only two sampled Inlines x three sampled Crosslines are read.
    assert len(factory.instances[0].reads) == 6
    assert all(request == slice(0, 6, 2) for _, request in factory.instances[0].reads)

    surface = payload["surfaceSeg"]
    assert surface["sourceAxes"] == ["INLINE", "CROSSLINE", "SAMPLE"]
    assert surface["platformAxes"] == ["Z", "INLINE", "CROSSLINE"]
    assert surface["transposeICSToZYX"] == [2, 0, 1]
    assert surface["smokeMode"] is True
    assert surface["processedInlineRange"] == [0, 1]
    assert surface["display"]["maskDiscrete"] is True
    assert payload["overlays"][0]["kind"] == "labels"
    assert payload["overlays"][0]["cmap"] == "jet"

    encoded = decode_visualization_array(surface["mask"])
    expected_labels = np.transpose(
        labels[np.ix_([0, 1], [0, 2, 4], [0, 2, 4])],
        (2, 0, 1),
    )
    assert encoded.shape == expected_labels.shape
    # Invalid -1 maps to transparent code zero, while valid label 0 maps to a
    # positive display code and therefore remains visible with excpt=min.
    np.testing.assert_array_equal(encoded[expected_labels == -1], 0.0)
    assert np.all(encoded[expected_labels == 0] >= 32.0 / 255.0)
    assert payload["overlays"][0]["excpt"] == "min"
    assert encoded.shape == decode_visualization_array(payload["cube"]).shape


def test_surface_confidence_is_an_optional_continuous_overlay(
    tmp_path: Path,
) -> None:
    result, _, confidence = _surface_result(tmp_path)
    cache = SegySliceCache(reader_factory=_ReaderFactory((6, 4, 5)))

    payload = build_surface_seg_visualization_payload(
        result,
        cache=cache,
        max_shape_zyx=(6, 2, 5),
        overlay_layer="confidence",
    )

    assert payload["surfaceSeg"]["display"]["preferredLayer"] == "confidence"
    assert payload["overlays"][0]["kind"] == "confidence"
    assert payload["overlays"][0]["clim"] == [0.35, 1.0]
    decoded = decode_visualization_array(payload["surfaceSeg"]["confidence"])
    expected = np.transpose(confidence[:, :, :].astype(np.float32), (2, 0, 1))
    np.testing.assert_allclose(decoded, expected, atol=1.0 / 255.0)


def test_dispatcher_selects_surface_builder_and_reports_common_descriptor(
    tmp_path: Path,
) -> None:
    result, _, _ = _surface_result(tmp_path)
    cache = SegySliceCache(reader_factory=_ReaderFactory((6, 4, 5)))

    payload = build_prediction_visualization_payload(
        result,
        cache=cache,
        max_shape_zyx=(3, 2, 3),
    )

    assert payload["predictionVisualization"] == {
        "modelId": "seismic_surface_seg",
        "modelName": "Seismic Surface Seg 地层分割",
        "preferredLayer": "mask",
        "overlayCount": 1,
    }
    with pytest.raises(ValueError, match="no visualization adapter"):
        build_prediction_visualization_payload({"model_id": "unknown"})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda result, tmp: result["input"].__setitem__(
                "axes", ["Z", "INLINE", "CROSSLINE"]
            ),
            "axes",
        ),
        (
            lambda result, tmp: result["input"].__setitem__(
                "source_shape_zyx", [6, 4, 4]
            ),
            "does not align",
        ),
        (
            lambda result, tmp: result["inference"].__setitem__(
                "max_inlines", None
            ),
            "partial Inline",
        ),
        (
            lambda result, tmp: np.save(
                result["outputs"]["mask_npy"],
                np.zeros((1, 1, 1), dtype=np.int16),
            ),
            "mask array shape",
        ),
        (
            lambda result, tmp: np.save(
                result["outputs"]["confidence_npy"],
                np.full((2, 5, 6), 1.5, dtype=np.float32),
            ),
            "outside",
        ),
        (
            lambda result, tmp: result["input"].__setitem__(
                "source", str(tmp / "missing.sgy")
            ),
            "input.source",
        ),
    ],
)
def test_surface_payload_rejects_invalid_axes_shape_smoke_and_outputs(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    result, _, _ = _surface_result(tmp_path)
    mutate(result, tmp_path)
    cache = SegySliceCache(reader_factory=_ReaderFactory((6, 4, 5)))

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        build_surface_seg_visualization_payload(result, cache=cache)
