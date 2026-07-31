from types import SimpleNamespace

import numpy as np

from well_seismic.models import WellLog
from well_seismic.registry import WellRegistry
from well_seismic.visualization_preview import _build_line_preview, _well_log_payloads, build_visualization_preview


def test_empty_conventional_curve_is_not_counted_as_available() -> None:
    registry = WellRegistry()
    registry.add_log(
        WellLog(
            well_name="TEST-1",
            depth=np.array([1000.0, 1001.0, 1002.0]),
            curves={
                "GR": np.array([np.nan, np.nan, np.nan]),
                "DT": np.array([220.0, 225.0, 230.0]),
            },
            masks={
                "GR": np.array([False, False, False]),
                "DT": np.array([True, True, True]),
            },
            curve_info={},
            header={},
            source="TEST-1.las",
            version="2.0",
        )
    )

    payload = _well_log_payloads(SimpleNamespace(registry=registry), max_points=3)

    assert len(payload) == 1
    assert [curve["id"] for curve in payload[0]["curves"]] == ["DT"]
    assert payload[0]["coverage"] == "1/9"


def test_two_dimensional_segy_is_renderable_without_inline_crossline_headers() -> None:
    geometry = SimpleNamespace(
        trace_count=5,
        samples_per_trace=6,
        time_axis=np.arange(6, dtype=float) * 2.0,
        inline=None,
        crossline=None,
        x=None,
        y=None,
    )
    reader = SimpleNamespace(
        geometry=geometry,
        read_trace=lambda index: np.arange(6, dtype=np.float32) + index,
    )
    asset = SimpleNamespace(path=__import__("pathlib").Path("line-a.sgy"))

    payload = _build_line_preview(
        asset,
        reader,
        geometry,
        max_trace_samples=5,
        max_time_samples=6,
    )

    assert payload is not None
    assert payload["lineAxis"] == "Trace"
    assert payload["image"]["shape"] == [6, 5]
    assert payload["preview"]["source_trace_count"] == 5

    pipeline = SimpleNamespace(
        seismic=[(asset, reader)],
        registry=SimpleNamespace(entities={}),
    )
    preview = build_visualization_preview(pipeline)
    assert len(preview["lines2d"]) == 1
    assert preview["volumes"] == []
