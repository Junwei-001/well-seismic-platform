from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from well_seismic import cigvis_adapter


def _volume_payload() -> dict:
    cube = np.arange(3 * 4 * 5, dtype=np.int8).reshape(3, 4, 5)
    return {
        "name": "preview.sgy",
        "path": "preview.sgy",
        "cube": {
            "encoding": "base64-int8",
            "shape": [3, 4, 5],
            "values": base64.b64encode(cube.tobytes()).decode("ascii"),
        },
        "defaultIndices": [1, 2, 3],
        "embeddedWells": [],
    }


def _volume_payload_with_overlay(
    *,
    kind: str = "probability",
    name: str = "FaultSeg probability",
) -> dict:
    payload = _volume_payload()
    probability = np.zeros((3, 4, 5), dtype=np.uint8)
    probability[1, 2, 3] = 255
    payload["overlays"] = [
        {
            "name": name,
            "kind": kind,
            "volume": {
                "encoding": "base64-uint8",
                "shape": [3, 4, 5],
                "values": base64.b64encode(probability.tobytes()).decode("ascii"),
            },
            "clim": [0.5, 1.0],
            "cmap": "jet",
            "alpha": 0.62,
            "excpt": "min",
        }
    ]
    return payload


def test_platform_zyx_cube_is_transposed_to_cigvis_inline_crossline_z() -> None:
    payload = _volume_payload()
    decoded = cigvis_adapter._decode_array(payload["cube"], 3)
    converted = cigvis_adapter._volume_array(payload)

    assert decoded.shape == (3, 4, 5)
    assert converted.shape == (4, 5, 3)
    assert converted[2, 3, 1] == decoded[1, 2, 3]
    assert cigvis_adapter._slice_positions(payload, converted.shape) == {
        "x": [2],
        "y": [3],
        "z": [1],
    }


def test_workbench_prefers_viser_if_scene_service_is_available(monkeypatch) -> None:
    monkeypatch.setattr(cigvis_adapter, "_ensure_viser_scene", lambda *_args, **_kwargs: 18080)
    document = cigvis_adapter.render_cigvis_workbench(
        Path.cwd(),
        {"volumes": [_volume_payload()], "lines2d": []},
        task_id="task-1234",
    )

    assert 'id="cigvis-viser-frame"' in document
    assert 'hostname+":18080/"' in document
    assert "WebGL2 整体三维 + CIGVis Viser 切片" in document
    assert 'data-view="whole"' in document
    assert 'data-view="slice" hidden' in document
    assert 'data-view-target="whole"' in document
    assert "整体 3D" in document
    assert "切片查看" in document
    assert "地震振幅 · seismic" in document
    assert "预测结果 · jet" in document
    assert "预测概率 · jet" not in document
    assert "vec3 seismic(float x)" in document
    assert "event.preventDefault()" in document
    assert "state.zoom+=" not in document
    assert "仅预测概率体" not in document
    assert "移动切片" in document
    assert "/api/v1/visualization/viser-slices" in document
    assert "assets-collapsed" in document
    assert "task-123" in document


def test_probability_volume_can_be_viewed_without_seismic_background(monkeypatch) -> None:
    monkeypatch.setattr(cigvis_adapter, "_ensure_viser_scene", lambda *_args, **_kwargs: 18080)
    document = cigvis_adapter.render_cigvis_workbench(
        Path.cwd(),
        {"volumes": [_volume_payload_with_overlay()], "lines2d": []},
        task_id="task-overlay",
    )

    assert "地震 + 预测" in document
    assert "仅预测概率体" in document
    assert 'data-layer-mode="prediction"' in document
    assert "/api/v1/visualization/viser-layer-mode" in document
    assert "u_predictionOnly" in document
    assert "setWholeVolumeLayerMode" in document
    assert "预测概率阈值" in document
    assert "FaultSeg probability · jet" in document


def test_label_overlay_uses_surface_segmentation_terms(monkeypatch) -> None:
    monkeypatch.setattr(cigvis_adapter, "_ensure_viser_scene", lambda *_args, **_kwargs: 18080)
    document = cigvis_adapter.render_cigvis_workbench(
        Path.cwd(),
        {
            "volumes": [
                _volume_payload_with_overlay(
                    kind="labels",
                    name="地层分割标签",
                )
            ],
            "lines2d": [],
        },
        task_id="task-labels",
    )

    assert "地层分割标签 · jet" in document
    assert "地层分割标签叠加" in document
    assert "仅地层标签体" in document
    assert "1 个预测图层" in document
    assert '"show_threshold":false' in document
    assert "FaultSeg断层叠加" not in document
    assert "个FaultSeg结果" not in document
    assert "预测概率阈值" not in document


def test_faultseg_uint8_overlay_decodes_to_probability_volume() -> None:
    probability = np.array([[[0, 127], [255, 64]]], dtype=np.uint8)
    decoded = cigvis_adapter._decode_overlay(
        {
            "encoding": "base64-uint8",
            "shape": [1, 2, 2],
            "values": base64.b64encode(probability.tobytes()).decode("ascii"),
        }
    )

    assert decoded.shape == (1, 2, 2)
    np.testing.assert_allclose(decoded, probability.astype(np.float32) / 255.0)


def test_local_cigvis_positions_singleton_slice_without_dividing_by_zero() -> None:
    cigvis_adapter._load_cigvis(Path.cwd())
    from cigvis.visernodes.volume_slice import VolumeSlice

    node = VolumeSlice(np.zeros((1, 4, 5), dtype=np.float32), axis="x", pos=0)
    assert np.all(np.isfinite(node.position))
    assert node.position[0] == node.vol_shape[0] * node.scale[0] / 2


def test_server_side_slice_control_updates_active_scene(monkeypatch) -> None:
    class Handle:
        def __init__(self, value: int) -> None:
            self.value = value

    class Server:
        _gui_slice_handles = {
            "x": [Handle(1)],
            "y": [Handle(2)],
            "z": [Handle(3)],
        }

    monkeypatch.setattr(cigvis_adapter, "_VISER_SERVER", Server())
    monkeypatch.setattr(cigvis_adapter, "_VISER_SCENE_KEY", ("task-1234", 0, "digest"))
    monkeypatch.setattr(cigvis_adapter, "_VISER_SCENE_SHAPE", (4, 5, 3))

    result = cigvis_adapter.update_viser_slices(
        "task-1234",
        0,
        {"x": 3, "y": 4, "z": 2},
    )

    assert result["positions"] == {"x": 3, "y": 4, "z": 2}
    assert result["shape"] == [4, 5, 3]


def test_server_side_layer_mode_updates_active_scene(monkeypatch) -> None:
    class Server:
        def __init__(self) -> None:
            self.background_values: list[bool] = []

        def set_slice_background_visible(self, visible: bool) -> int:
            self.background_values.append(visible)
            return 3

    server = Server()
    monkeypatch.setattr(cigvis_adapter, "_VISER_SERVER", server)
    monkeypatch.setattr(cigvis_adapter, "_VISER_SCENE_KEY", ("task-overlay", 0, "digest"))

    prediction = cigvis_adapter.update_viser_layer_mode("task-overlay", 0, "prediction")
    combined = cigvis_adapter.update_viser_layer_mode("task-overlay", 0, "combined")

    assert server.background_values == [False, True]
    assert prediction["background_visible"] is False
    assert prediction["mode"] == "prediction"
    assert combined["background_visible"] is True


def test_workbench_falls_back_to_plotly_when_viser_fails(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("viser unavailable")

    monkeypatch.setattr(cigvis_adapter, "_ensure_viser_scene", fail)
    monkeypatch.setattr(cigvis_adapter, "_render_volume_plotly", lambda *_args, **_kwargs: "<div>plotly</div>")
    monkeypatch.setattr(cigvis_adapter, "_stop_viser_service", lambda: None)

    document = cigvis_adapter.render_cigvis_workbench(
        Path.cwd(),
        {"volumes": [_volume_payload()], "lines2d": []},
        task_id="task-5678",
    )

    assert "<div>plotly</div>" in document
    assert "WebGL2 整体三维 + Plotly 切片回退" in document
    assert "viser unavailable" in document
