from __future__ import annotations

import atexit
import base64
import hashlib
import html
import importlib
import io
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np


_CIGVIS_LOCK = threading.RLock()
_CIGVIS_MODULE: Any | None = None
_PLOTLY_MODULE: Any | None = None
_VISER_MODULE: Any | None = None
_VISER_SERVER: Any | None = None
_VISER_SCENE_KEY: tuple[str, int, str] | None = None
_VISER_SCENE_SHAPE: tuple[int, int, int] | None = None
_VISER_ERROR = ""


def _cigvis_root(project_root: Path) -> Path:
    return project_root / "接口模型" / "cigvis-main" / "cigvis-main"


def _local_version(root: Path) -> str:
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
        return match.group(1) if match else "local"
    except Exception:
        return "local"


def _load_cigvis(project_root: Path) -> tuple[Any, Any]:
    global _CIGVIS_MODULE, _PLOTLY_MODULE
    with _CIGVIS_LOCK:
        if _CIGVIS_MODULE is not None and _PLOTLY_MODULE is not None:
            return _CIGVIS_MODULE, _PLOTLY_MODULE

        root = _cigvis_root(project_root).resolve()
        if not (root / "cigvis" / "__init__.py").is_file():
            raise RuntimeError(f"未找到本地 cigvis 源码：{root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        os.environ.setdefault("MPLBACKEND", "Agg")

        cigvis = importlib.import_module("cigvis")
        module_path = Path(cigvis.__file__ or "").resolve()
        if root not in module_path.parents:
            raise RuntimeError(f"当前加载的 cigvis 不是指定本地仓库：{module_path}")
        plotlyplot = importlib.import_module("cigvis.plotlyplot")
        cigvis.set_order(True)
        cigvis.set_axis_reversed(False, True, True)
        _CIGVIS_MODULE = cigvis
        _PLOTLY_MODULE = plotlyplot
        return cigvis, plotlyplot


def _load_viser(project_root: Path) -> Any:
    global _VISER_MODULE
    with _CIGVIS_LOCK:
        if _VISER_MODULE is not None:
            return _VISER_MODULE
        _load_cigvis(project_root)
        importlib.import_module("viser")
        _VISER_MODULE = importlib.import_module("cigvis.viserplot")
        return _VISER_MODULE


def cigvis_status(project_root: Path) -> dict[str, Any]:
    root = _cigvis_root(project_root).resolve()
    try:
        _load_cigvis(project_root)
        viser_error = ""
        try:
            _load_viser(project_root)
            viser_available = True
        except Exception as exc:
            viser_available = False
            viser_error = f"{type(exc).__name__}: {exc}"
        return {
            "available": True,
            "version": _local_version(root),
            "root": str(root),
            "backend": "viser" if viser_available else "plotly+matplotlib",
            "preferred_backend": "viser",
            "viser_available": viser_available,
            "fallback_backend": "plotly+matplotlib",
            "web_engine": "CIGVis Viser" if viser_available else "CIGVis Plotly",
            "viser_error": viser_error,
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "version": _local_version(root),
            "root": str(root),
            "backend": "unavailable",
            "preferred_backend": "viser",
            "viser_available": False,
            "fallback_backend": "plotly+matplotlib",
            "web_engine": "CIGVis",
            "viser_error": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def plotly_javascript(project_root: Path) -> str:
    _load_cigvis(project_root)
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def _decode_array(spec: dict[str, Any], ndim: int) -> np.ndarray:
    if spec.get("encoding") != "base64-int8":
        raise ValueError("CIGVis适配器仅接受base64-int8轻量预览")
    shape = tuple(int(value) for value in spec.get("shape", ()))
    if len(shape) != ndim or any(value <= 0 for value in shape):
        raise ValueError(f"无效的{ndim}维预览形状：{shape}")
    raw = base64.b64decode(str(spec.get("values", "")), validate=True)
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError(f"预览体字节数与形状不一致：{len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float32) / 127.0


def _decode_overlay(spec: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in spec.get("shape", ()))
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"无效的预测叠加体形状：{shape}")
    encoding = str(spec.get("encoding", ""))
    dtype_scale = {
        "base64-uint8": (np.uint8, 255.0),
        "base64-int8": (np.int8, 127.0),
        "base64-float32": (np.dtype("<f4"), 1.0),
    }
    if encoding not in dtype_scale:
        raise ValueError(f"不支持的叠加体编码：{encoding}")
    dtype, scale = dtype_scale[encoding]
    raw = base64.b64decode(str(spec.get("values", "")), validate=True)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f"叠加体字节数与形状不一致：{len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32) / scale


def _tick_axis(values: list[Any], title: str, max_ticks: int = 6) -> dict[str, Any]:
    clean = [value for value in values if value is not None]
    axis: dict[str, Any] = {
        "title": {"text": title, "font": {"size": 12, "color": "#425466"}},
        "gridcolor": "rgba(111,135,157,.18)",
        "zerolinecolor": "rgba(111,135,157,.28)",
        "backgroundcolor": "#f8fafc",
        "showbackground": True,
        "tickfont": {"size": 10, "color": "#526779"},
    }
    if clean:
        indices = np.unique(np.linspace(0, len(clean) - 1, min(len(clean), max_ticks), dtype=int))
        axis["tickmode"] = "array"
        axis["tickvals"] = [int(index) for index in indices]
        axis["ticktext"] = [f"{float(clean[index]):g}" for index in indices]
    return axis


def _volume_array(volume_payload: dict[str, Any]) -> np.ndarray:
    # Platform preview contract is [Z, Inline, Crossline]; CIGVis line-first
    # contract is [Inline, Crossline, Z].
    return np.transpose(_decode_array(volume_payload["cube"], 3), (1, 2, 0))


def _slice_positions(volume_payload: dict[str, Any], shape: tuple[int, ...]) -> dict[str, list[int]]:
    default = [int(value) for value in volume_payload.get("defaultIndices", [])]
    if len(default) != 3:
        default = [shape[2] // 2, shape[0] // 2, shape[1] // 2]
    return {
        "x": [min(max(default[1], 0), shape[0] - 1)],
        "y": [min(max(default[2], 0), shape[1] - 1)],
        "z": [min(max(default[0], 0), shape[2] - 1)],
    }


def _well_log_arrays(volume_payload: dict[str, Any], shape: tuple[int, ...]) -> tuple[list[np.ndarray], list[str]]:
    logs: list[np.ndarray] = []
    names: list[str] = []
    for well in volume_payload.get("embeddedWells", []):
        inline = np.asarray(well.get("y", []), dtype=float) * max(shape[0] - 1, 1)
        crossline = np.asarray(well.get("x", []), dtype=float) * max(shape[1] - 1, 1)
        z = np.asarray(well.get("z", []), dtype=float) * max(shape[2] - 1, 1)
        if inline.size >= 2 and inline.size == crossline.size == z.size:
            logs.append(np.column_stack((inline, crossline, z)))
            names.append(str(well.get("name", "井轨迹")))
    return logs, names


def _render_volume_plotly(project_root: Path, volume_payload: dict[str, Any], task_id: str) -> str:
    cigvis, plotlyplot = _load_cigvis(project_root)
    with _CIGVIS_LOCK:
        cigvis.set_order(True)
        cigvis.set_axis_reversed(False, True, True)
        volume = _volume_array(volume_payload)
        pos = _slice_positions(volume_payload, volume.shape)
        nodes = plotlyplot.create_slices(
            volume,
            pos=pos,
            clim=[-1.0, 1.0],
            cmap="seismic",
            interpolation="nearest",
        )
        for overlay in volume_payload.get("overlays", []):
            overlay_spec = overlay.get("volume", overlay)
            overlay_volume = np.transpose(_decode_overlay(overlay_spec), (1, 2, 0))
            if overlay_volume.shape != volume.shape:
                raise ValueError(f"预测叠加体与背景体不对齐：{overlay_volume.shape} != {volume.shape}")
            nodes = plotlyplot.add_mask(
                nodes,
                overlay_volume,
                clim=overlay.get("clim", [0.5, 1.0]),
                cmap=overlay.get("cmap", "jet"),
                alpha=float(overlay.get("alpha", 0.62)),
                excpt=overlay.get("excpt", "min"),
                interpolation="nearest",
            )
        well_logs, well_names = _well_log_arrays(volume_payload, volume.shape)
        if well_logs:
            nodes += plotlyplot.create_line_logs(well_logs, cmap="viridis", line_width=7)

        fig = plotlyplot.plot3D(nodes, show=False, size=(900, 1200))
        base_count = 3
        for index, trace in enumerate(fig.data):
            if index < base_count:
                trace.name = ("Inline切片", "Crossline切片", "时间切片")[index]
                trace.showlegend = False
            else:
                name_index = index - base_count
                trace.name = well_names[name_index] if name_index < len(well_names) else "井轨迹"
                trace.showlegend = True
        fig.update_layout(
            autosize=True,
            height=None,
            width=None,
            template="plotly_white",
            paper_bgcolor="#f7f9fc",
            plot_bgcolor="#f7f9fc",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            legend={
                "x": 0.012,
                "y": 0.99,
                "bgcolor": "rgba(255,255,255,.88)",
                "bordercolor": "rgba(99,120,143,.25)",
                "borderwidth": 1,
                "font": {"size": 10, "color": "#33475b"},
            },
            scene={
                "xaxis": _tick_axis(volume_payload.get("inlineValues", []), "Inline"),
                "yaxis": _tick_axis(volume_payload.get("crosslineValues", []), "Crossline"),
                "zaxis": _tick_axis(volume_payload.get("timeValues", []), "TWT / ms"),
                "aspectmode": "manual",
                "aspectratio": {
                    "x": 1.0,
                    "y": max(0.55, min(1.35, volume.shape[1] / max(volume.shape[0], 1))),
                    "z": 1.35,
                },
                "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.18}},
                "bgcolor": "#f7f9fc",
            },
            uirevision=f"cigvis-{task_id}",
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs="/cigvis/plotly.min.js",
            config={
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": False,
                "toImageButtonOptions": {"format": "png", "filename": f"cigvis_{task_id[:8]}", "scale": 2},
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
            div_id="cigvis-plot",
        )


def _stop_viser_service() -> None:
    global _VISER_SERVER, _VISER_SCENE_KEY, _VISER_SCENE_SHAPE
    with _CIGVIS_LOCK:
        server = _VISER_SERVER
        _VISER_SERVER = None
        _VISER_SCENE_KEY = None
        _VISER_SCENE_SHAPE = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass


atexit.register(_stop_viser_service)


def update_viser_slices(
    task_id: str,
    asset_index: int,
    positions: dict[str, int | None],
) -> dict[str, Any]:
    """Move the active CIGVis slices through the server-side GUI handles."""
    with _CIGVIS_LOCK:
        if _VISER_SERVER is None or _VISER_SCENE_KEY is None or _VISER_SCENE_SHAPE is None:
            raise RuntimeError("当前没有活动的CIGVis Viser三维场景")
        if _VISER_SCENE_KEY[:2] != (task_id, int(asset_index)):
            raise RuntimeError("切片控制请求与当前CIGVis场景不匹配，请刷新工作台")
        handles_by_axis = getattr(_VISER_SERVER, "_gui_slice_handles", {})
        shape_by_axis = dict(zip(("x", "y", "z"), _VISER_SCENE_SHAPE))
        for axis in ("x", "y", "z"):
            value = positions.get(axis)
            if value is None:
                continue
            value = int(value)
            if value < 0 or value >= shape_by_axis[axis]:
                raise ValueError(f"{axis}切片索引超出范围：{value}")
            for handle in handles_by_axis.get(axis, []):
                handle.value = value
        current = {
            axis: int(handles[0].value)
            for axis, handles in handles_by_axis.items()
            if handles
        }
        return {"task_id": task_id, "asset": int(asset_index), "positions": current, "shape": list(_VISER_SCENE_SHAPE)}


def update_viser_layer_mode(
    task_id: str,
    asset_index: int,
    mode: str,
) -> dict[str, Any]:
    """Switch CIGVis slices between combined and prediction-only rendering."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"combined", "prediction"}:
        raise ValueError("图层模式必须是 combined 或 prediction")
    with _CIGVIS_LOCK:
        if _VISER_SERVER is None or _VISER_SCENE_KEY is None:
            raise RuntimeError("当前没有活动的CIGVis Viser三维场景")
        if _VISER_SCENE_KEY[:2] != (task_id, int(asset_index)):
            raise RuntimeError("图层控制请求与当前CIGVis场景不匹配，请刷新工作台")
        setter = getattr(_VISER_SERVER, "set_slice_background_visible", None)
        if not callable(setter):
            raise RuntimeError("当前CIGVis版本不支持独立控制背景图层")
        try:
            slice_count = int(setter(normalized_mode == "combined"))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return {
            "task_id": task_id,
            "asset": int(asset_index),
            "mode": normalized_mode,
            "background_visible": normalized_mode == "combined",
            "slice_count": slice_count,
        }


def _lock_viser_camera(
    server: Any,
    look_at: tuple[float, float, float],
    distance: float | None = None,
) -> None:
    """Keep Viser centered while still allowing orbit rotation."""
    if getattr(server, "_well_seismic_camera_lock", False):
        return
    server._well_seismic_camera_lock = True
    fixed_target = np.asarray(look_at, dtype=float)
    configured_distance = float(distance) if distance is not None else None

    @server.on_client_connect
    def _register_camera_lock(client: Any) -> None:
        initial_position = np.asarray(client.camera.position, dtype=float)
        initial_target = np.asarray(client.camera.look_at, dtype=float)
        initial_distance = float(np.linalg.norm(initial_position - initial_target))
        fixed_distance = configured_distance or (initial_distance if initial_distance > 1e-6 else 3.0)

        @client.camera.on_update
        def _keep_camera_centered(_: Any) -> None:
            position = np.asarray(client.camera.position, dtype=float)
            current_target = np.asarray(client.camera.look_at, dtype=float)
            direction = position - current_target
            length = float(np.linalg.norm(direction))
            if not np.isfinite(length) or length < 1e-6:
                direction = np.asarray((1.0, -1.0, -0.72), dtype=float)
                length = float(np.linalg.norm(direction))
            desired_position = fixed_target + direction / length * fixed_distance
            if (
                np.linalg.norm(current_target - fixed_target) > 1e-5
                or abs(length - fixed_distance) > 1e-5
            ):
                client.camera.look_at = tuple(float(value) for value in fixed_target)
                client.camera.position = tuple(float(value) for value in desired_position)


def _ensure_viser_scene(
    project_root: Path,
    volume_payload: dict[str, Any],
    task_id: str,
    asset_index: int,
) -> int:
    global _VISER_SERVER, _VISER_SCENE_KEY, _VISER_SCENE_SHAPE, _VISER_ERROR
    cube = volume_payload.get("cube", {})
    digest = hashlib.sha1(str(cube.get("values", "")).encode("ascii", errors="ignore")).hexdigest()[:16]
    scene_key = (task_id, int(asset_index), digest)
    with _CIGVIS_LOCK:
        viserplot = _load_viser(project_root)
        if _VISER_SERVER is not None and _VISER_SCENE_KEY != scene_key:
            _stop_viser_service()
        if _VISER_SERVER is None:
            _VISER_SERVER = viserplot.create_server(port=8080, label="well-seismic-cigvis", verbose=False)
            try:
                _VISER_SERVER.gui.configure_theme(
                    control_layout="collapsible",
                    control_width="small",
                    dark_mode=False,
                    show_logo=False,
                    show_share_button=False,
                    brand_color=(22, 119, 232),
                )
            except Exception:
                pass
        if _VISER_SCENE_KEY != scene_key:
            if hasattr(_VISER_SERVER, "configure_ui"):
                _VISER_SERVER.configure_ui(
                    language="zh-CN",
                    axis_labels={"x": "Inline（主测线）", "y": "Crossline（联络测线）", "z": "时间 / 深度"},
                    labels={"slices_folder": "高级切片控制"},
                )
            volume = _volume_array(volume_payload)
            original_shape = tuple(int(value) for value in volume.shape)
            nodes = viserplot.create_slices(
                volume,
                pos=_slice_positions(volume_payload, original_shape),
                clim=[-1.0, 1.0],
                cmap="seismic",
                intersection_lines=False,
            )
            for overlay in volume_payload.get("overlays", []):
                overlay_spec = overlay.get("volume", overlay)
                overlay_volume = np.transpose(_decode_overlay(overlay_spec), (1, 2, 0))
                if overlay_volume.shape != original_shape:
                    raise ValueError(f"预测叠加体与背景体不对齐：{overlay_volume.shape} != {original_shape}")
                nodes = viserplot.add_mask(
                    nodes,
                    overlay_volume,
                    clim=overlay.get("clim", [0.5, 1.0]),
                    cmap=overlay.get("cmap", "jet"),
                    alpha=float(overlay.get("alpha", 0.62)),
                    excpt=overlay.get("excpt", "min"),
                )
            well_logs, _ = _well_log_arrays(volume_payload, original_shape)
            if well_logs:
                nodes += viserplot.create_well_logs(
                    well_logs,
                    logs_type="line",
                    cmap="viridis",
                    width=3,
                )
            y_scale = max(0.55, min(1.35, volume.shape[1] / max(volume.shape[0], 1)))
            axis_scales = np.asarray((1.0, y_scale, 1.25), dtype=float)
            initial_scale = 1.5 / max(volume.shape)
            extents = np.asarray(volume.shape, dtype=float) * initial_scale * axis_scales
            center = extents / 2.0
            viserplot.plot3D(
                nodes,
                server=_VISER_SERVER,
                run_app=False,
                axis_scales=axis_scales.tolist(),
                fov=28,
                look_at=center.tolist(),
            )
            _lock_viser_camera(
                _VISER_SERVER,
                tuple(float(value) for value in center),
            )
            _VISER_SCENE_KEY = scene_key
            _VISER_SCENE_SHAPE = original_shape
        _VISER_ERROR = ""
        return int(_VISER_SERVER.get_port())


def _viser_fragment(port: int) -> str:
    return (
        '<div class="cigvis-viser-view">'
        '<div class="viser-loading" id="viser-loading">正在连接 CIGVis Viser…</div>'
        '<iframe id="cigvis-viser-frame" title="CIGVis Viser 三维地震场景" '
        'allow="fullscreen; clipboard-write" allowfullscreen></iframe></div>'
        '<script>'
        '(function(){const frame=document.getElementById("cigvis-viser-frame");'
        'const loading=document.getElementById("viser-loading");'
        f'frame.src=window.location.protocol+"//"+window.location.hostname+":{int(port)}/";'
        'frame.addEventListener("load",()=>{if(loading)loading.hidden=true;});})();'
        '</script>'
    )


def _overlay_ui_metadata(overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Return model-neutral labels for one downstream visualization layer."""
    item = overlay or {}
    kind = (
        "prediction"
        if overlay is None
        else str(item.get("kind", "probability")).strip().lower()
    )
    if kind == "labels":
        subject = "地层标签"
        default_name = "地层分割标签"
        only_label = "仅地层标签体"
        threshold_label = ""
        show_threshold = False
    elif kind == "confidence":
        subject = "预测置信度"
        default_name = "预测置信度"
        only_label = "仅预测置信度体"
        threshold_label = "置信度阈值"
        show_threshold = True
    elif kind == "probability":
        subject = "预测概率"
        default_name = "预测概率"
        only_label = "仅预测概率体"
        threshold_label = "预测概率阈值"
        show_threshold = True
    else:
        subject = "预测结果"
        default_name = "预测结果"
        only_label = "仅预测结果"
        threshold_label = "结果显示下限"
        show_threshold = True
    return {
        "kind": kind,
        "subject": subject,
        "display_name": str(item.get("name", default_name)),
        "only_label": only_label,
        "threshold_label": threshold_label,
        "show_threshold": show_threshold,
    }


def _webgl_volume_fragment(volume_payload: dict[str, Any]) -> str:
    """Build a centered WebGL2 ray-marched whole-volume view.

    The browser receives the same lightweight [Z, Inline, Crossline] preview
    used by CIGVis.  Seismic amplitudes are rendered with the seismic
    diverging colormap and the first
    downstream probability volume is blended with a jet transfer function.
    """
    cube = dict(volume_payload.get("cube", {}))
    shape = [int(value) for value in cube.get("shape", [])]
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"无效的整体三维预览形状：{shape}")

    overlay_payload: dict[str, Any] | None = None
    overlay_ui = _overlay_ui_metadata(None)
    overlays = list(volume_payload.get("overlays", []))
    if overlays:
        overlay = overlays[0]
        overlay_ui = _overlay_ui_metadata(overlay)
        overlay_spec = dict(overlay.get("volume", overlay))
        overlay_shape = [int(value) for value in overlay_spec.get("shape", [])]
        if overlay_shape != shape:
            raise ValueError(f"预测叠加体与背景体不对齐：{overlay_shape} != {shape}")
        overlay_payload = {
            "volume": overlay_spec,
            "clim": list(overlay.get("clim", [0.5, 1.0])),
            "alpha": float(overlay.get("alpha", 0.62)),
            "cmap": "jet",
            "name": str(overlay.get("name", "下游预测结果")),
            "kind": str(overlay.get("kind", "probability")),
            "ui": overlay_ui,
        }

    payload = {
        "cube": cube,
        "overlay": overlay_payload,
        "wells": list(volume_payload.get("embeddedWells", [])),
        "name": str(volume_payload.get("name", "三维地震体")),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''
<div class="volume-render-view" id="volume-render-view">
  <canvas id="volume-render-canvas" aria-label="整体三维地震体"></canvas>
  <canvas id="volume-wire-canvas" aria-hidden="true"></canvas>
  <div class="volume-render-tools">
    <label class="base-volume-tool"><span>振幅增益</span><input id="volume-gain" type="range" min="0.5" max="3" value="1.45" step="0.05"></label>
    <label class="base-volume-tool"><span>体透明度</span><input id="volume-opacity" type="range" min="0.03" max="0.28" value="0.12" step="0.01"></label>
    <label class="base-volume-tool"><span>振幅阈值</span><input id="volume-threshold" type="range" min="0.02" max="0.55" value="0.14" step="0.01"></label>
    <label class="overlay-tool" id="volume-overlay-tool"><span id="volume-overlay-threshold-label">__OVERLAY_THRESHOLD_LABEL__</span><input id="volume-overlay-threshold" type="range" min="0" max="1" value="0.5" step="0.01"></label>
    <button id="volume-reset" type="button">重置居中视角</button>
  </div>
  <div class="volume-legend"><span id="volume-seismic-legend"><i class="seismic-ramp"></i>地震振幅 · seismic</span><span id="volume-overlay-legend"><i class="jet-ramp"></i><b id="volume-overlay-legend-label">__OVERLAY_LEGEND_LABEL__</b></span></div>
  <div class="volume-render-status" id="volume-render-status">正在初始化整体三维体渲染…</div>
</div>
<script>
(function(){
  const payload=__PAYLOAD__;
  const root=document.getElementById('volume-render-view');
  const canvas=document.getElementById('volume-render-canvas');
  const wireCanvas=document.getElementById('volume-wire-canvas');
  const status=document.getElementById('volume-render-status');
  const gain=document.getElementById('volume-gain');
  const opacity=document.getElementById('volume-opacity');
  const threshold=document.getElementById('volume-threshold');
  const overlayThreshold=document.getElementById('volume-overlay-threshold');
  const overlayThresholdLabel=document.getElementById('volume-overlay-threshold-label');
  const overlayTool=document.getElementById('volume-overlay-tool');
  const baseTools=root.querySelectorAll('.base-volume-tool');
  const seismicLegend=document.getElementById('volume-seismic-legend');
  const overlayLegend=document.getElementById('volume-overlay-legend');
  const overlayLegendLabel=document.getElementById('volume-overlay-legend-label');
  const gl=canvas.getContext('webgl2',{alpha:true,antialias:true,premultipliedAlpha:false});
  const state={yaw:-0.66,pitch:-0.38,zoom:1.72,predictionOnly:false,program:null,volumeTexture:null,overlayTexture:null,position:-1};
  const shape=payload.cube.shape.map(Number);
  const maxShape=Math.max(...shape);
  // Texture coordinates are Crossline, Inline, Z.
  const boxScale=[Math.max(.46,shape[2]/maxShape),Math.max(.46,shape[1]/maxShape),Math.max(.46,shape[0]/maxShape)];
  const overlayUi=payload.overlay?.ui||{};
  const overlayName=String(overlayUi.display_name||payload.overlay?.name||'预测结果');
  if(!payload.overlay){overlayTool.hidden=true;overlayLegend.hidden=true;}
  else{
    if(payload.overlay.clim&&payload.overlay.clim.length)overlayThreshold.value=String(payload.overlay.clim[0]);
    overlayTool.hidden=overlayUi.show_threshold===false;
    if(overlayThresholdLabel)overlayThresholdLabel.textContent=String(overlayUi.threshold_label||'结果显示下限');
    if(overlayLegendLabel)overlayLegendLabel.textContent=overlayName+' · jet';
  }

  function decodeBytes(spec){
    const binary=atob(spec.values||'');
    const bytes=new Uint8Array(binary.length);
    for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
    return bytes;
  }
  function decodeSeismic(spec){
    const raw=decodeBytes(spec), signed=new Int8Array(raw.buffer), output=new Uint8Array(raw.length);
    for(let i=0;i<signed.length;i++)output[i]=signed[i]+128;
    return output;
  }
  function decodeProbability(spec){
    const raw=decodeBytes(spec), output=new Uint8Array(Number(spec.shape[0])*Number(spec.shape[1])*Number(spec.shape[2]));
    if(spec.encoding==='base64-uint8'){output.set(raw);return output;}
    if(spec.encoding==='base64-int8'){
      const signed=new Int8Array(raw.buffer);
      for(let i=0;i<output.length;i++)output[i]=Math.round(Math.max(0,Math.min(1,signed[i]/127))*255);
      return output;
    }
    if(spec.encoding==='base64-float32'){
      const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
      for(let i=0;i<output.length;i++)output[i]=Math.round(Math.max(0,Math.min(1,view.getFloat32(i*4,true)))*255);
      return output;
    }
    throw new Error('不支持的预测叠加体编码：'+spec.encoding);
  }
  function compile(type,source){
    const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader)||'着色器编译失败');
    return shader;
  }
  function createTexture(values){
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,texture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,shape[2],shape[1],shape[0],0,gl.RED,gl.UNSIGNED_BYTE,values);
    return texture;
  }
  function initialize(){
    if(!gl){status.textContent='当前浏览器不支持 WebGL2，请切换到切片查看';root.classList.add('unsupported');return;}
    const vertex=`#version 300 es
      in vec2 a_position;out vec2 v_uv;
      void main(){v_uv=a_position;gl_Position=vec4(a_position,0.0,1.0);}`;
    const fragment=`#version 300 es
      precision highp float;precision highp sampler3D;
      in vec2 v_uv;out vec4 outColor;
      uniform sampler3D u_volume;uniform sampler3D u_overlay;
      uniform float u_yaw;uniform float u_pitch;uniform float u_zoom;uniform float u_gain;uniform float u_opacity;uniform float u_threshold;uniform float u_overlayThreshold;uniform float u_overlayOpacity;uniform float u_aspect;uniform float u_hasOverlay;uniform float u_predictionOnly;uniform vec3 u_boxScale;
      mat3 rotateY(float a){float c=cos(a),s=sin(a);return mat3(c,0.,-s,0.,1.,0.,s,0.,c);}
      mat3 rotateX(float a){float c=cos(a),s=sin(a);return mat3(1.,0.,0.,0.,c,s,0.,-s,c);}
      bool hitBox(vec3 origin,vec3 direction,out float nearValue,out float farValue){vec3 halfBox=.5*u_boxScale;vec3 inv=1./direction;vec3 first=(-halfBox-origin)*inv;vec3 second=(halfBox-origin)*inv;vec3 lower=min(first,second);vec3 upper=max(first,second);nearValue=max(max(lower.x,lower.y),lower.z);farValue=min(min(upper.x,upper.y),upper.z);return farValue>max(nearValue,0.);}
      vec3 seismic(float x){float value=clamp(x,-1.,1.);vec3 neutral=vec3(.96);vec3 negative=vec3(.02,.12,.82);vec3 positive=vec3(.84,.03,.02);return value<0.?mix(neutral,negative,-value):mix(neutral,positive,value);}
      vec3 jet(float x){return clamp(1.5-abs(4.*x-vec3(3.,2.,1.)),0.,1.);}
      void main(){
        vec2 screen=v_uv;screen.x*=u_aspect;
        vec3 cameraOrigin=vec3(0.,0.,u_zoom);vec3 cameraDirection=normalize(vec3(screen,-1.9));mat3 rotation=rotateY(u_yaw)*rotateX(u_pitch);
        vec3 rayOrigin=transpose(rotation)*cameraOrigin;vec3 rayDirection=transpose(rotation)*cameraDirection;float nearValue;float farValue;
        if(!hitBox(rayOrigin,rayDirection,nearValue,farValue))discard;
        float start=max(nearValue,0.);float stepLength=(farValue-start)/192.;vec4 accumulated=vec4(0.);
        for(int index=0;index<192;index++){
          float distanceValue=start+(float(index)+.5)*stepLength;vec3 local=rayOrigin+rayDirection*distanceValue;vec3 tex=local/u_boxScale+.5;
          float amplitude=(texture(u_volume,tex).r*255.-128.)/127.;float scaledAmplitude=clamp(amplitude*u_gain,-1.,1.);float magnitude=abs(scaledAmplitude);
          float baseAlpha=(1.-u_predictionOnly)*smoothstep(u_threshold,1.,magnitude)*u_opacity;vec3 baseColor=seismic(scaledAmplitude);
          float probability=texture(u_overlay,tex).r;float maskAlpha=u_hasOverlay*smoothstep(u_overlayThreshold,min(1.,u_overlayThreshold+.18),probability)*u_overlayOpacity;
          float sampleAlpha=max(baseAlpha,maskAlpha);vec3 sampleColor=mix(baseColor,jet(probability),clamp(maskAlpha/max(sampleAlpha,.0001),0.,1.));
          accumulated.rgb+=(1.-accumulated.a)*sampleAlpha*sampleColor;accumulated.a+=(1.-accumulated.a)*sampleAlpha;if(accumulated.a>.985)break;
        }
        if(accumulated.a<.01)discard;outColor=accumulated;
      }`;
    const program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program)||'着色器链接失败');
    state.program=program;state.position=gl.getAttribLocation(program,'a_position');
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);gl.enableVertexAttribArray(state.position);gl.vertexAttribPointer(state.position,2,gl.FLOAT,false,0,0);
    state.volumeTexture=createTexture(decodeSeismic(payload.cube));
    state.overlayTexture=createTexture(payload.overlay?decodeProbability(payload.overlay.volume):new Uint8Array(shape[0]*shape[1]*shape[2]));
    updateLayerStatus();
    render();
  }
  function updateLayerStatus(){
    if(!payload.overlay){status.textContent='整体三维 · seismic 地震背景';return;}
    status.textContent=state.predictionOnly?'整体三维 · 仅显示 '+overlayName:'整体三维 · seismic 地震背景 + jet '+overlayName;
    if(seismicLegend)seismicLegend.hidden=state.predictionOnly;
  }
  function uniform(name){return gl.getUniformLocation(state.program,name);}
  function render(){
    if(!gl||!state.program||!state.volumeTexture)return;
    gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(state.program);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,state.volumeTexture);gl.uniform1i(uniform('u_volume'),0);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,state.overlayTexture);gl.uniform1i(uniform('u_overlay'),1);
    gl.uniform1f(uniform('u_yaw'),state.yaw);gl.uniform1f(uniform('u_pitch'),state.pitch);gl.uniform1f(uniform('u_zoom'),state.zoom);
    gl.uniform1f(uniform('u_gain'),Number(gain.value));gl.uniform1f(uniform('u_opacity'),Number(opacity.value));gl.uniform1f(uniform('u_threshold'),Number(threshold.value));
    gl.uniform1f(uniform('u_overlayThreshold'),Number(overlayThreshold.value));gl.uniform1f(uniform('u_overlayOpacity'),payload.overlay?Math.max(.04,Number(payload.overlay.alpha||.62)*(state.predictionOnly ? .55 : .26)):0);
    gl.uniform1f(uniform('u_hasOverlay'),payload.overlay?1:0);gl.uniform1f(uniform('u_predictionOnly'),state.predictionOnly?1:0);gl.uniform1f(uniform('u_aspect'),canvas.width/Math.max(canvas.height,1));gl.uniform3fv(uniform('u_boxScale'),boxScale);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);drawGuides();
  }
  function project(point){
    const cx=Math.cos(state.pitch),sx=Math.sin(state.pitch),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw);
    const afterX=[point[0],cx*point[1]-sx*point[2],sx*point[1]+cx*point[2]];
    const camera=[cy*afterX[0]+sy*afterX[2],afterX[1],-sy*afterX[0]+cy*afterX[2]];const dz=camera[2]-state.zoom;if(dz>=-.05)return null;
    const aspect=canvas.clientWidth/Math.max(canvas.clientHeight,1);const x=(-1.9*camera[0]/dz)/aspect;const y=-1.9*camera[1]/dz;
    return [(x+1)*.5*canvas.clientWidth,(1-y)*.5*canvas.clientHeight];
  }
  function drawGuides(){
    const context=wireCanvas.getContext('2d');context.clearRect(0,0,wireCanvas.clientWidth,wireCanvas.clientHeight);context.lineWidth=1;context.strokeStyle='rgba(51,72,92,.48)';
    const corners=[];for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)corners.push(project([(x-.5)*boxScale[0],(y-.5)*boxScale[1],(z-.5)*boxScale[2]]));
    const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];context.beginPath();edges.forEach(([a,b])=>{if(corners[a]&&corners[b]){context.moveTo(...corners[a]);context.lineTo(...corners[b]);}});context.stroke();
    context.strokeStyle='rgba(0,105,92,.9)';context.fillStyle='#00695c';context.lineWidth=1.4;
    (payload.wells||[]).forEach(well=>{const points=(well.x||[]).map((x,i)=>project([(x-.5)*boxScale[0],((well.y||[])[i]-.5)*boxScale[1],((well.z||[])[i]-.5)*boxScale[2]])).filter(Boolean);if(points.length<2)return;context.beginPath();points.forEach((p,i)=>i?context.lineTo(...p):context.moveTo(...p));context.stroke();context.font='11px sans-serif';context.fillText(well.name||'井',points[0][0]+5,points[0][1]-5);});
  }
  function resize(){
    const ratio=Math.min(window.devicePixelRatio||1,2),width=Math.max(1,Math.round(canvas.clientWidth*ratio)),height=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;wireCanvas.width=width;wireCanvas.height=height;wireCanvas.getContext('2d').setTransform(ratio,0,0,ratio,0,0);}render();
  }
  let dragging=false,lastX=0,lastY=0;
  canvas.addEventListener('pointerdown',event=>{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{if(!dragging)return;state.yaw+=(event.clientX-lastX)*.007;state.pitch=Math.max(-1.25,Math.min(1.25,state.pitch+(event.clientY-lastY)*.007));lastX=event.clientX;lastY=event.clientY;render();});
  canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);
  // Deliberately consume the wheel: scale and target stay fixed to prevent drift.
  canvas.addEventListener('wheel',event=>event.preventDefault(),{passive:false});
  [gain,opacity,threshold,overlayThreshold].forEach(control=>control.addEventListener('input',render));
  function reset(){state.yaw=-.66;state.pitch=-.38;state.zoom=1.72;render();}
  document.getElementById('volume-reset').addEventListener('click',reset);window.resetWholeVolumeView=reset;
  window.setWholeVolumeLayerMode=(mode)=>{state.predictionOnly=Boolean(payload.overlay)&&mode==='prediction';baseTools.forEach(tool=>tool.hidden=state.predictionOnly);updateLayerStatus();render();};
  if(window.ResizeObserver)new ResizeObserver(resize).observe(root);window.addEventListener('resize',resize);
  try{initialize();resize();}catch(error){status.textContent='整体三维初始化失败：'+(error instanceof Error?error.message:String(error));root.classList.add('unsupported');}
})();
</script>
'''
    return (
        template.replace("__PAYLOAD__", payload_json)
        .replace(
            "__OVERLAY_THRESHOLD_LABEL__",
            html.escape(str(overlay_ui["threshold_label"])),
        )
        .replace(
            "__OVERLAY_LEGEND_LABEL__",
            html.escape(f'{overlay_ui["display_name"]} · jet'),
        )
    )


def _render_volume(
    project_root: Path,
    volume_payload: dict[str, Any],
    task_id: str,
    asset_index: int,
) -> tuple[str, str, str]:
    global _VISER_ERROR
    whole_fragment = _webgl_volume_fragment(volume_payload)
    try:
        port = _ensure_viser_scene(project_root, volume_payload, task_id, asset_index)
        slice_fragment = _viser_fragment(port)
        return (
            f'<section class="view-panel whole-volume-panel" data-view="whole">{whole_fragment}</section>'
            f'<section class="view-panel slice-volume-panel" data-view="slice" hidden>{slice_fragment}</section>',
            "WebGL2 整体三维 + CIGVis Viser 切片",
            "",
        )
    except Exception as exc:
        _VISER_ERROR = f"{type(exc).__name__}: {exc}"
        _stop_viser_service()
        slice_fragment = _render_volume_plotly(project_root, volume_payload, task_id)
        return (
            f'<section class="view-panel whole-volume-panel" data-view="whole">{whole_fragment}</section>'
            f'<section class="view-panel slice-volume-panel" data-view="slice" hidden>{slice_fragment}</section>',
            "WebGL2 整体三维 + Plotly 切片回退",
            _VISER_ERROR,
        )


def _render_line(project_root: Path, line_payload: dict[str, Any]) -> str:
    cigvis, _ = _load_cigvis(project_root)
    with _CIGVIS_LOCK:
        cigvis.set_order(True)
        import matplotlib.pyplot as plt

        image = _decode_array(line_payload["image"], 2)
        fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor="#f7f9fc")
        ax.set_facecolor("#ffffff")
        time_values = [value for value in line_payload.get("timeValues", []) if value is not None]
        trace_values = [value for value in line_payload.get("traceValues", []) if value is not None]
        xsample = None
        ysample = None
        if len(trace_values) >= 2:
            xsample = [float(trace_values[0]), float(trace_values[1]) - float(trace_values[0])]
        if len(time_values) >= 2:
            ysample = [float(time_values[0]), float(time_values[1]) - float(time_values[0])]
        # plot2d transposes line-first inputs; image.T preserves the platform's
        # [time, trace] view after CIGVis applies its own orientation contract.
        cigvis.plot2d(
            image.T,
            cmap="seismic",
            clim=[-1.0, 1.0],
            interpolation="nearest",
            aspect="auto",
            title="",
            xlabel=str(line_payload.get("lineAxis", "Trace")),
            ylabel="TWT / ms",
            xsample=xsample,
            ysample=ysample,
            cbar="Normalized amplitude",
            show=False,
            ax=ax,
        )
        ax.tick_params(colors="#526779", labelsize=9)
        ax.xaxis.label.set_color("#33475b")
        ax.yaxis.label.set_color("#33475b")
        ax.title.set_color("#24384d")
        for spine in ax.spines.values():
            spine.set_color("#c8d3df")
        for colorbar_axis in fig.axes[1:]:
            colorbar_axis.tick_params(colors="#526779", labelsize=8)
            colorbar_axis.xaxis.label.set_color("#425466")
            colorbar_axis.yaxis.label.set_color("#425466")
            for spine in colorbar_axis.spines.values():
                spine.set_color("#c8d3df")
        fig.tight_layout(pad=1.2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div class="cigvis-line-view">'
        f'<img alt="{html.escape(str(line_payload.get("name", "二维地震剖面")))}" '
        f'src="data:image/png;base64,{encoded}"></div>'
    )


def _asset_catalog(preview: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        *(("volume", item) for item in preview.get("volumes", [])),
        *(("line", item) for item in preview.get("lines2d", [])),
    ]


def render_cigvis_workbench(
    project_root: Path,
    preview: dict[str, Any],
    *,
    task_id: str,
    asset_index: int = 0,
    embed: bool = True,
) -> str:
    status = cigvis_status(project_root)
    if not status["available"]:
        raise RuntimeError(status["error"] or "CIGVis不可用")
    assets = _asset_catalog(preview)
    if not assets:
        raise ValueError("当前任务没有可交给CIGVis渲染的二维或三维地震数据")
    asset_index = min(max(int(asset_index), 0), len(assets) - 1)
    kind, selected = assets[asset_index]
    if kind == "volume":
        fragment, active_engine, engine_warning = _render_volume(
            project_root,
            selected,
            task_id,
            asset_index,
        )
    else:
        fragment = _render_line(project_root, selected)
        active_engine = "Matplotlib 二维剖面"
        engine_warning = ""

    links: list[str] = []
    for index, (asset_kind, item) in enumerate(assets):
        query = urlencode({"embed": 1 if embed else 0, "task_id": task_id, "asset": index})
        label = "3D" if asset_kind == "volume" else "2D"
        active = " active" if index == asset_index else ""
        links.append(
            f'<a class="asset-link{active}" href="/统一数据可视化?{query}">'
            f'<span>{label}</span><div><strong>{html.escape(str(item.get("name", "未命名地震")))}</strong>'
            f'<small>{"整体三维 / 动态切片" if asset_kind == "volume" else "二维地震剖面"}</small></div></a>'
        )

    standalone_query = urlencode({"embed": 0, "task_id": task_id, "asset": asset_index})
    selected_shape = selected.get("cube", selected.get("image", {})).get("shape", [])
    selected_overlays = list(selected.get("overlays", [])) if kind == "volume" else []
    overlay_count = len(selected_overlays)
    primary_overlay = selected_overlays[0] if selected_overlays else {}
    overlay_ui = _overlay_ui_metadata(primary_overlay if selected_overlays else None)
    overlay_name = str(overlay_ui["display_name"])
    overlay_subject = str(overlay_ui["subject"])
    prediction_only_label = str(overlay_ui["only_label"])
    kind_label = (f"{overlay_name}叠加" if overlay_count else "三维地震体") if kind == "volume" else "二维地震测线"
    has_slice_console = kind == "volume" and "CIGVis Viser" in active_engine and len(selected_shape) == 3
    mode_switch = ""
    if kind == "volume":
        mode_switch = (
            '<nav class="view-mode-switch" aria-label="三维查看模式">'
            '<button class="view-mode-button active" type="button" data-view-target="whole">'
            '<b>整体 3D</b><small>默认</small></button>'
            '<button class="view-mode-button" type="button" data-view-target="slice">'
            '<b>切片查看</b><small>CIGVis</small></button></nav>'
        )
    layer_switch = ""
    if overlay_count:
        layer_switch = (
            '<nav class="layer-mode-switch" aria-label="预测图层显示">'
            '<button class="layer-mode-button active" type="button" data-layer-mode="combined" aria-pressed="true">'
            '<b>地震 + 预测</b></button>'
            '<button class="layer-mode-button" type="button" data-layer-mode="prediction" aria-pressed="false">'
            f'<b>{html.escape(prediction_only_label)}</b></button></nav>'
        )
    slice_console = ""
    slice_values_json = "{}"
    if has_slice_console:
        shape_xyz = (int(selected_shape[1]), int(selected_shape[2]), int(selected_shape[0]))
        positions = _slice_positions(selected, shape_xyz)
        axis_values = {
            "x": list(selected.get("inlineValues", [])),
            "y": list(selected.get("crosslineValues", [])),
            "z": list(selected.get("timeValues", [])),
        }
        slice_values_json = json.dumps(axis_values, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        controls: list[str] = []
        for axis, label in (("x", "Inline"), ("y", "Crossline"), ("z", "时间 / 深度")):
            position = positions[axis][0]
            values = axis_values[axis]
            display_value = values[position] if position < len(values) and values[position] is not None else position
            controls.append(
                f'<label class="slice-control" data-axis="{axis}"><span><b>{label}</b>'
                f'<output>{html.escape(str(display_value))}</output></span><input type="range" min="0" '
                f'max="{shape_xyz[("x", "y", "z").index(axis)] - 1}" value="{position}" step="1"></label>'
            )
        slice_console = (
            '<section class="slice-console"><div class="slice-console-title"><strong>移动切片</strong>'
            '<small id="slice-status">拖动滑块即可更新三维场景</small></div>'
            f'{"".join(controls)}</section>'
        )
    workbench_class = "workbench"
    if embed:
        workbench_class += " embedded assets-collapsed"
    if has_slice_console:
        workbench_class += " with-slice-console whole-active"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIGVis 地震解释工作台</title>
<style>
:root{{--bg:#edf2f7;--panel:#fff;--panel2:#f7f9fc;--line:#d7e0e9;--text:#203246;--muted:#6f8295;--blue:#1677e8;--green:#15966d}}
*{{box-sizing:border-box}}html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:var(--bg);font-family:Inter,"Microsoft YaHei UI",sans-serif;color:var(--text)}}
.workbench{{display:grid;width:100%;height:100vh;grid-template-columns:220px minmax(0,1fr);grid-template-rows:52px minmax(0,1fr);background:linear-gradient(145deg,#f8fafc,#eef3f8)}}
.workbench.with-slice-console{{grid-template-rows:52px minmax(0,1fr) 0}}.workbench.with-slice-console.slice-active{{grid-template-rows:52px minmax(0,1fr) 86px}}.workbench.assets-collapsed{{grid-template-columns:0 minmax(0,1fr)}}.workbench.assets-collapsed .sidebar{{padding:0;border:0;opacity:0;pointer-events:none}}
.topbar{{display:flex;grid-column:1/-1;gap:14px;align-items:center;padding:0 14px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);box-shadow:0 2px 10px rgba(35,58,81,.05)}}
.brand{{display:flex;gap:9px;align-items:center;min-width:220px}}.brand b{{display:grid;width:29px;height:29px;place-items:center;color:#fff;font-size:10px;background:linear-gradient(145deg,#2687ef,#1263c5);border-radius:7px}}.brand div{{display:grid}}.brand strong{{font-size:13px}}.brand small{{color:var(--muted);font-size:9px;letter-spacing:.08em}}
.dataset-title{{display:flex;min-width:0;gap:9px;align-items:center;flex:1}}.dataset-title>span{{padding:3px 7px;color:{'#087b67' if kind == 'volume' else '#1267bb'};font-size:9px;font-weight:750;background:{'#e6f4ef' if kind == 'volume' else '#e8f2fc'};border-radius:4px}}.dataset-title strong{{overflow:hidden;font-size:13px;text-overflow:ellipsis;white-space:nowrap}}
.view-mode-switch{{display:flex;flex:0 0 auto;gap:4px;padding:3px;background:#e8eef5;border:1px solid #d3deea;border-radius:8px}}.view-mode-switch .view-mode-button{{display:flex;min-height:34px;gap:5px;align-items:center;padding:0 13px;color:#536b82;background:transparent;border-color:transparent;border-radius:6px}}.view-mode-button b{{font-size:11px}}.view-mode-button small{{padding:1px 4px;color:#7f91a3;font-size:8px;background:rgba(255,255,255,.65);border-radius:3px}}.view-mode-switch .view-mode-button.active{{color:#fff;background:#1677e8;border-color:#1677e8;box-shadow:0 4px 10px rgba(22,119,232,.22)}}.view-mode-switch .view-mode-button.active small{{color:#145fae;background:#fff}}
.layer-mode-switch{{display:flex;flex:0 0 auto;gap:3px;padding:3px;background:#eef2f6;border:1px solid #d7e0e9;border-radius:8px}}.layer-mode-switch .layer-mode-button{{min-height:30px;padding:0 9px;color:#60758a;font-size:9px;background:transparent;border-color:transparent;border-radius:6px}}.layer-mode-switch .layer-mode-button.active{{color:#fff;background:#16866f;border-color:#16866f;box-shadow:0 4px 10px rgba(22,134,111,.2)}}.layer-mode-switch .layer-mode-button:disabled{{cursor:wait;opacity:.62}}
.top-actions{{display:flex;gap:6px}}button,.top-actions a{{display:inline-flex;min-height:31px;align-items:center;padding:0 10px;color:#41586f;font-size:10px;font-weight:650;text-decoration:none;background:#f7f9fc;border:1px solid #d2dce7;border-radius:6px;cursor:pointer}}button:hover,.top-actions a:hover{{border-color:#78a9df;color:#0f66c7;background:#eef6ff}}
.sidebar{{grid-row:2/-1;min-height:0;overflow:auto;padding:12px;background:#fff;border-right:1px solid var(--line);transition:padding .18s,opacity .18s;scrollbar-width:thin;scrollbar-color:#b9c7d6 transparent}}.sidebar::-webkit-scrollbar{{width:6px}}.sidebar::-webkit-scrollbar-track{{background:transparent}}.sidebar::-webkit-scrollbar-thumb{{background:#b9c7d6;border-radius:8px}}.section-label{{display:block;margin:3px 5px 8px;color:#7b8d9f;font-size:9px;font-weight:750;letter-spacing:.12em}}.asset-list{{display:grid;gap:4px}}.asset-link{{display:grid;grid-template-columns:29px minmax(0,1fr);gap:8px;align-items:center;padding:8px;color:inherit;text-decoration:none;border:1px solid transparent;border-radius:7px}}.asset-link:hover{{background:#f3f7fb}}.asset-link.active{{background:#edf5fe;border-color:#a8c9ec}}.asset-link>span{{display:grid;width:28px;height:28px;place-items:center;color:#637b93;font-size:9px;font-weight:800;background:#edf2f7;border-radius:6px}}.asset-link.active>span{{color:#fff;background:#1677e8}}.asset-link div{{display:grid;min-width:0}}.asset-link strong{{overflow:hidden;font-size:11px;text-overflow:ellipsis;white-space:nowrap}}.asset-link small{{color:#8191a2;font-size:9px}}
.meta-card{{display:grid;gap:8px;margin-top:13px;padding:10px;background:#f7f9fc;border:1px solid #dce4ed;border-radius:8px}}.meta-card div{{display:flex;justify-content:space-between;gap:8px}}.meta-card span{{color:#7b8d9f;font-size:9px}}.meta-card strong{{color:#40566d;font-size:9px;text-align:right}}.engine{{display:flex;gap:7px;align-items:flex-start;margin-top:11px;padding:9px;color:#708499;font-size:9px;line-height:1.45;border-top:1px solid #dce4ed}}.engine i{{flex:0 0 auto;width:7px;height:7px;margin-top:3px;background:var(--green);border-radius:50%;box-shadow:0 0 0 3px rgba(21,150,109,.1)}}.engine.warning i{{background:#e6a23c;box-shadow:0 0 0 3px rgba(230,162,60,.1)}}
.viewport{{grid-column:2;grid-row:2;position:relative;min-width:0;min-height:0;overflow:hidden;background:#f5f7fa}}.plot-shell,.view-panel{{position:absolute;inset:0}}.view-panel[hidden]{{display:none!important}}#cigvis-plot,.plotly-graph-div{{width:100%!important;height:100%!important}}.cigvis-viser-view,.cigvis-viser-view iframe{{display:block;width:100%;height:100%;border:0}}.cigvis-viser-view{{position:relative;background:#f5f7fa}}.viser-loading{{position:absolute;inset:0;z-index:2;display:grid;place-items:center;color:#60758a;font-size:12px;background:radial-gradient(circle,#fff,#eef3f8 72%)}}.viser-loading[hidden]{{display:none}}.cigvis-line-view{{display:grid;width:100%;height:100%;place-items:center;padding:12px;background:#f4f7fa}}.cigvis-line-view img{{display:block;max-width:100%;max-height:100%;object-fit:contain;border:1px solid #cdd8e3;border-radius:7px;box-shadow:0 12px 34px rgba(30,55,80,.1)}}
.volume-render-view{{position:absolute;inset:0;overflow:hidden;background:radial-gradient(circle at 50% 46%,#fff 0,#f1f4f7 54%,#e2e8ee 100%)}}#volume-render-canvas,#volume-wire-canvas{{position:absolute;inset:0;display:block;width:100%;height:100%}}#volume-render-canvas{{cursor:grab;touch-action:none}}#volume-render-canvas:active{{cursor:grabbing}}#volume-wire-canvas{{pointer-events:none}}.volume-render-tools{{position:absolute;top:14px;left:14px;z-index:3;display:grid;width:182px;gap:8px;padding:11px;background:rgba(255,255,255,.91);border:1px solid #ccd7e2;border-radius:8px;box-shadow:0 8px 24px rgba(31,55,79,.1);backdrop-filter:blur(8px)}}.volume-render-tools label{{display:grid;gap:4px}}.volume-render-tools [hidden],.volume-legend [hidden]{{display:none!important}}.volume-render-tools label>span{{color:#5f7286;font-size:9px;font-weight:650}}.volume-render-tools input{{width:100%;height:4px;accent-color:#1677e8}}.volume-render-tools button{{justify-content:center;min-height:28px;margin-top:2px}}.volume-legend{{position:absolute;right:14px;top:14px;z-index:3;display:grid;gap:6px;padding:9px 10px;color:#546a7f;font-size:9px;background:rgba(255,255,255,.91);border:1px solid #ccd7e2;border-radius:7px}}.volume-legend span{{display:flex;gap:7px;align-items:center}}.volume-legend i{{display:block;width:64px;height:7px;border:1px solid rgba(45,63,80,.2);border-radius:2px}}.seismic-ramp{{background:linear-gradient(90deg,#0517b8,#f7f7f7 50%,#c20b0b)}}.jet-ramp{{background:linear-gradient(90deg,#000080,#006cff,#7dff7a,#ffb500,#800000)}}.volume-render-status{{position:absolute;right:14px;bottom:12px;z-index:3;padding:5px 8px;color:#536b80;font-size:9px;background:rgba(255,255,255,.86);border:1px solid #d1dbe5;border-radius:5px}}.volume-render-view.unsupported::after{{position:absolute;inset:38% 22%;display:grid;place-items:center;content:'请切换到“切片查看”';color:#61768b;font-size:13px;background:#fff;border:1px dashed #b8c6d4;border-radius:8px}}
.hint{{position:absolute;right:10px;bottom:9px;z-index:5;padding:5px 8px;color:#5f7488;font-size:9px;background:rgba(255,255,255,.88);border:1px solid #d5dfe9;border-radius:5px;pointer-events:none}}
.slice-console{{grid-column:2;grid-row:3;display:grid;visibility:hidden;grid-template-columns:138px repeat(3,minmax(140px,1fr));gap:16px;align-items:center;overflow:hidden;padding:0 18px;background:#fff;border-top:0 solid var(--line);opacity:0;box-shadow:0 -3px 12px rgba(35,58,81,.04);transition:opacity .16s}}.workbench.slice-active .slice-console{{visibility:visible;padding:10px 18px;border-top-width:1px;opacity:1}}.slice-console-title{{display:grid}}.slice-console-title strong{{font-size:12px}}.slice-console-title small{{color:#8091a2;font-size:9px}}.slice-control{{display:grid;gap:6px}}.slice-control span{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.slice-control b{{font-size:10px}}.slice-control output{{color:#146ac5;font-size:10px;font-weight:750}}.slice-control input{{width:100%;height:5px;accent-color:#1677e8;cursor:ew-resize}}
@media(max-width:1120px){{.dataset-title{{display:none}}.view-mode-switch .view-mode-button{{padding:0 9px}}.view-mode-button small{{display:none}}.layer-mode-switch .layer-mode-button{{padding:0 7px}}}}
@media(max-width:820px){{.workbench,.workbench.assets-collapsed{{grid-template-columns:1fr;grid-template-rows:50px 92px minmax(0,1fr)}}.workbench.with-slice-console,.workbench.with-slice-console.assets-collapsed{{grid-template-rows:50px 92px minmax(0,1fr) 0}}.workbench.with-slice-console.slice-active,.workbench.with-slice-console.assets-collapsed.slice-active{{grid-template-rows:50px 92px minmax(0,1fr) 126px}}.topbar{{padding:0 8px}}.brand{{min-width:auto}}.brand div{{display:none}}.sidebar,.workbench.assets-collapsed .sidebar{{display:block;grid-column:1;grid-row:2;padding:7px;overflow-x:auto;overflow-y:hidden;opacity:1;pointer-events:auto;border-right:0;border-bottom:1px solid var(--line)}}.section-label,.meta-card,.engine{{display:none}}.asset-list{{display:flex}}.asset-link{{min-width:175px}}.viewport{{grid-column:1;grid-row:3}}.slice-console{{grid-column:1;grid-row:4;grid-template-columns:repeat(3,minmax(90px,1fr));gap:8px;padding:0 10px}}.workbench.slice-active .slice-console{{padding:8px 10px}}.slice-console-title{{display:none}}.volume-render-tools{{width:160px}}}}
</style></head>
<body><div class="{workbench_class}" id="workbench"><header class="topbar"><div class="brand"><b>CIG</b><div><strong>CIGVis 地震解释工作台</strong><small>GEOPHYSICAL VISUALIZATION</small></div></div><div class="dataset-title"><span>{kind_label}</span><strong>{html.escape(str(selected.get('name', '未命名地震')))}</strong></div>{mode_switch}{layer_switch}<div class="top-actions"><button id="asset-toggle" type="button">数据集</button><button id="fit-view" type="button">重置视角</button><button id="fullscreen" type="button">全屏</button><a href="/统一数据可视化?{standalone_query}" target="_blank" rel="noopener">独立窗口</a></div></header>
<aside class="sidebar"><span class="section-label">当前任务地震资产</span><nav class="asset-list">{''.join(links)}</nav><section class="meta-card"><div><span>渲染类型</span><strong>{kind_label}</strong></div><div><span>数据形状</span><strong>{' × '.join(str(value) for value in selected_shape)}</strong></div>{f'<div><span>叠加图层</span><strong>{overlay_count} 个预测图层</strong></div>' if overlay_count else ''}<div><span>任务快照</span><strong>{html.escape(task_id[:8])}</strong></div><div><span>源文件</span><strong>{html.escape(Path(str(selected.get('path', ''))).name)}</strong></div></section><div class="engine{' warning' if engine_warning else ''}"><i></i><span>CIGVis {html.escape(str(status['version']))} · {html.escape(active_engine)}{'<br>Viser 启动失败，已自动回退：' + html.escape(engine_warning) if engine_warning else ''}</span></div></aside>
<main class="viewport"><div class="plot-shell">{fragment}</div><div class="hint" id="view-hint">{'拖动旋转 · 地震体已居中 · 缩放和平移已锁定' if kind == 'volume' else 'CIGVis 二维剖面 · 等比例适配窗口'}</div></main>{slice_console}</div>
<script>
const workbench=document.getElementById('workbench');
const plot=document.getElementById('cigvis-plot');
const viserFrame=document.getElementById('cigvis-viser-frame');
const resizePlot=()=>{{if(plot&&window.Plotly)window.Plotly.Plots.resize(plot)}};
document.getElementById('asset-toggle').addEventListener('click',()=>workbench.classList.toggle('assets-collapsed'));
let activeView={json.dumps('whole' if kind == 'volume' else 'line')};
let activeLayerMode='combined';
const overlaySubject={json.dumps(overlay_subject, ensure_ascii=False)};
const viewHint=document.getElementById('view-hint');
const layerModeButtons=Array.from(document.querySelectorAll('.layer-mode-button'));
const updateViewHint=()=>{{
  if(!viewHint)return;
  if(activeLayerMode==='prediction'){{viewHint.textContent=activeView==='slice'?'仅显示'+overlaySubject+'切片 · 使用下方滑块移动切片':'仅显示'+overlaySubject+'体 · 拖动旋转 · 缩放和平移已锁定';return;}}
  viewHint.textContent=activeView==='slice'?'使用下方滑块移动 Inline / Crossline / 时间切片 · 相机中心和距离已锁定':'拖动旋转 · 地震体已居中 · 缩放和平移已锁定';
}};
const applyLayerMode=(mode)=>{{
  activeLayerMode=mode;
  layerModeButtons.forEach(button=>{{const active=button.dataset.layerMode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}});
  window.setWholeVolumeLayerMode?.(mode);
  updateViewHint();
}};
const setLayerMode=async(mode)=>{{
  if(!['combined','prediction'].includes(mode)||mode===activeLayerMode)return;
  layerModeButtons.forEach(button=>button.disabled=true);
  try{{
    if({json.dumps(has_slice_console)}){{
      const response=await fetch('/api/v1/visualization/viser-layer-mode',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{task_id:{json.dumps(task_id)},asset_index:{asset_index},mode}})}});
      if(!response.ok)throw new Error((await response.json()).detail||'图层切换失败');
    }}
    applyLayerMode(mode);
  }}catch(error){{if(viewHint)viewHint.textContent=error instanceof Error?error.message:'图层切换失败';}}
  finally{{layerModeButtons.forEach(button=>button.disabled=false);}}
}};
const setView=(view)=>{{
  activeView=view;
  document.querySelectorAll('.view-panel').forEach(panel=>panel.hidden=panel.dataset.view!==view);
  document.querySelectorAll('.view-mode-button').forEach(button=>button.classList.toggle('active',button.dataset.viewTarget===view));
  workbench.classList.toggle('slice-active',view==='slice');workbench.classList.toggle('whole-active',view!=='slice');
  updateViewHint();
  requestAnimationFrame(()=>{{if(view==='slice')resizePlot();else{{window.resetWholeVolumeView?.();window.dispatchEvent(new Event('resize'));}}}});
}};
document.querySelectorAll('.view-mode-button').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.viewTarget)));
layerModeButtons.forEach(button=>button.addEventListener('click',()=>setLayerMode(button.dataset.layerMode)));
document.getElementById('fit-view').addEventListener('click',()=>{{
  if(activeView==='whole'){{window.resetWholeVolumeView?.();return;}}
  resizePlot();if(plot&&window.Plotly)window.Plotly.relayout(plot,{{'scene.camera.eye':{{x:1.55,y:-1.65,z:1.18}},'scene.camera.center':{{x:0,y:0,z:0}}}});
  if(viserFrame){{const src=viserFrame.getAttribute('src');if(src)viserFrame.setAttribute('src',src)}}
}});
document.getElementById('fullscreen').addEventListener('click',()=>{{if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();else document.exitFullscreen?.()}});
const sliceValues={slice_values_json};
const sliceTimers={{}};
document.querySelectorAll('.slice-control').forEach(control=>{{
  const axis=control.dataset.axis,input=control.querySelector('input'),output=control.querySelector('output');
  input.addEventListener('input',()=>{{
    const index=Number(input.value),values=sliceValues[axis]||[];
    output.value=values[index]??index;
    clearTimeout(sliceTimers[axis]);
    const status=document.getElementById('slice-status');if(status)status.textContent='正在更新切片…';
    sliceTimers[axis]=setTimeout(async()=>{{
      try{{
        const body={{task_id:{json.dumps(task_id)},asset_index:{asset_index}}};body[axis]=index;
        const response=await fetch('/api/v1/visualization/viser-slices',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
        if(!response.ok)throw new Error((await response.json()).detail||'切片更新失败');
        if(status)status.textContent='切片已更新';
      }}catch(error){{if(status)status.textContent=error instanceof Error?error.message:'切片更新失败'}}
    }},70);
  }});
}});
if(window.ResizeObserver)new ResizeObserver(resizePlot).observe(document.querySelector('.viewport'));window.addEventListener('load',resizePlot);
</script></body></html>"""
