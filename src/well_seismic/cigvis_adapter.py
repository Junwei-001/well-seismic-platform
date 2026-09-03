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
from typing import Any, Sequence
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

_CROSSLINE_DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS = 1
_MAXIMUM_WELL_RESULT_TRACKS = 128
_MAXIMUM_WELL_RESULT_INTERVALS = 20000
_MAXIMUM_WELL_RESULT_CURVE_POINTS = 1200

_WELL_RESULT_SUBJECTS = {
    "fluid_interpretation": "流体解释",
    "facies_1d": "一维地震相",
    "fracture_development": "裂缝相对发育",
}
_WELL_RESULT_PALETTES = {
    "fluid_interpretation": (
        "#64748b",
        "#2695d3",
        "#e8aa28",
        "#e75f6d",
        "#8b68c8",
    ),
    "fracture_development": ("#75a9c5", "#e3a72f", "#d74735"),
    "facies_1d": (
        "#335caa",
        "#2f8f83",
        "#e0a93b",
        "#d66a4e",
        "#7756b3",
        "#35a661",
        "#bf568b",
        "#718096",
    ),
}

# Only these four public well-side prediction tasks have a sealed result
# contract that can be drawn as a conventional MD log board.  Keeping the
# expected result kind in the profile prevents a similarly-shaped result from
# another task (or a mixed task bundle) from silently taking over the fixed
# left-hand layout.
_WELL_LOG_LAYOUT_PROFILES = {
    "well_property": {
        "kind": "property_curve",
        "title": "储层物性测井图版",
        "track_heading": "连续物性曲线（真实结果）",
    },
    "fluid_interpretation": {
        "kind": "categorical_intervals",
        "title": "流体解释测井图版",
        "track_heading": "流体解释分类（确定性）",
    },
    "facies_1d": {
        "kind": "categorical_intervals",
        "title": "一维地震相测井图版",
        "track_heading": "地震相分类（确定性）",
    },
    "fracture_development": {
        "kind": "categorical_intervals",
        "title": "裂缝发育测井图版",
        "track_heading": "相对发育等级（确定性）",
    },
}

_PUBLIC_VISUALIZATION_NAME_REPLACEMENTS = (
    (re.compile(r"cig[\s_-]*bench", re.IGNORECASE), "公开地质体基准"),
    (re.compile(r"cigvis", re.IGNORECASE), "平台可视化"),
    (re.compile(r"(?<![A-Za-z0-9_])cig(?![A-Za-z0-9_])", re.IGNORECASE), "平台可视化"),
    (
        re.compile(
            r"(?<![A-Za-z0-9])(?:seismic[\s_-]*foundation[\s_-]*model|"
            r"sfm(?:[\s_-]*base(?:[\s_-]*224)?|"
            r"[\s_-]*(?:tokens?|view[\s_-]*mask))?)(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
        "空间特征融合模块",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9])moment(?:[\s_-]*1[\s_-]*small|"
            r"[\s_-]*(?:tokens?|patch[\s_-]*mask))?(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
        "测井序列特征模块",
    ),
    (re.compile(r"(?<![A-Za-z0-9_])ncs(?![A-Za-z0-9_])", re.IGNORECASE), "全局编码组件"),
    (re.compile(r"vit(?:25d|3d)", re.IGNORECASE), "地震特征编码分支"),
    (re.compile(r"viser", re.IGNORECASE), "平台三维引擎"),
    (re.compile(r"(?:plotly|matplotlib)", re.IGNORECASE), "平台二维引擎"),
    (re.compile(r"fault[\s_-]*(?:seg|net)", re.IGNORECASE), "断层识别"),
    (re.compile(r"(?:segformer|mask2former)", re.IGNORECASE), "地层分割"),
    (
        re.compile(
            r"(?:seismic|legacy)?[\s_-]*surface[\s_-]*seg", re.IGNORECASE
        ),
        "有序地层分割",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9])u[\s_-]*net(?![A-Za-z0-9])", re.IGNORECASE),
        "地震相分割网络",
    ),
)


def _public_visualization_text(value: object, fallback: str = "") -> str:
    """Remove implementation and open-source names from display-only copy."""

    text = str(value or fallback)
    for pattern, replacement in _PUBLIC_VISUALIZATION_NAME_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def public_visualization_text(value: object, fallback: str = "") -> str:
    """Return the shared project-facing label for visualization errors and UI."""

    return _public_visualization_text(value, fallback)


def _cigvis_root(project_root: Path) -> Path:
    return project_root / "接口模型" / "cigvis-main" / "cigvis-main"


def _local_version(root: Path) -> str:
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE
        )
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
        cigvis.set_axis_reversed(False, False, True)
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
        except Exception:
            viser_available = False
            viser_error = "平台三维可视化兼容组件未就绪"
        return {
            "available": True,
            "version": _local_version(root),
            "backend": "platform-3d" if viser_available else "platform-2d",
            "preferred_backend": "platform-3d",
            "viser_available": viser_available,
            "fallback_backend": "platform-2d",
            "web_engine": "平台三维可视化引擎",
            "viser_error": viser_error,
            "error": "",
        }
    except Exception:
        return {
            "available": False,
            "version": _local_version(root),
            "backend": "unavailable",
            "preferred_backend": "platform-3d",
            "viser_available": False,
            "fallback_backend": "platform-2d",
            "web_engine": "平台三维可视化引擎",
            "viser_error": "",
            "error": "本地可视化组件不可用",
        }


def plotly_javascript(project_root: Path) -> str:
    _load_cigvis(project_root)
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def _decode_array(spec: dict[str, Any], ndim: int) -> np.ndarray:
    if spec.get("encoding") != "base64-int8":
        raise ValueError("三维可视化组件仅接受base64-int8轻量预览")
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
        "tickfont": {"size": 12, "color": "#526779"},
    }
    if clean:
        indices = np.unique(
            np.linspace(0, len(clean) - 1, min(len(clean), max_ticks), dtype=int)
        )
        axis["tickmode"] = "array"
        axis["tickvals"] = [int(index) for index in indices]
        axis["ticktext"] = [f"{float(clean[index]):g}" for index in indices]
    return axis


def _axis_labels(volume_payload: dict[str, Any]) -> dict[str, str]:
    labels = volume_payload.get("axisLabels")
    if not isinstance(labels, dict):
        labels = {}
    return {
        "inline": str(labels.get("inline") or "Inline"),
        "crossline": str(labels.get("crossline") or "Crossline"),
        "sample": str(labels.get("sample") or "时间 / 深度"),
    }


def _volume_array(volume_payload: dict[str, Any]) -> np.ndarray:
    # Platform preview contract is [Z, Inline, Crossline]; CIGVis line-first
    # contract is [Inline, Crossline, Z].
    return np.transpose(_decode_array(volume_payload["cube"], 3), (1, 2, 0))


def _slice_positions(
    volume_payload: dict[str, Any], shape: tuple[int, ...]
) -> dict[str, list[int]]:
    default = [int(value) for value in volume_payload.get("defaultIndices", [])]
    if len(default) != 3:
        default = [shape[2] // 2, shape[0] // 2, shape[1] // 2]
    return {
        "x": [min(max(default[1], 0), shape[0] - 1)],
        "y": [min(max(default[2], 0), shape[1] - 1)],
        "z": [min(max(default[0], 0), shape[2] - 1)],
    }


def _well_log_arrays(
    volume_payload: dict[str, Any], shape: tuple[int, ...]
) -> tuple[list[np.ndarray], list[str]]:
    logs: list[np.ndarray] = []
    names: list[str] = []
    for well in volume_payload.get("embeddedWells", []):
        inline = np.asarray(well.get("y", []), dtype=float) * max(shape[0] - 1, 1)
        crossline = np.asarray(well.get("x", []), dtype=float) * max(shape[1] - 1, 1)
        z = np.asarray(well.get("z", []), dtype=float) * max(shape[2] - 1, 1)
        if inline.size >= 2 and inline.size == crossline.size == z.size:
            logs.append(np.column_stack((inline, crossline, z)))
            geometry_label = str(well.get("geometryLabel", "井型未判定"))
            names.append(f"{well.get('name', '井轨迹')} · {geometry_label}")
    return logs, names


def _scene_surface_arrays(volume_payload: dict[str, Any]) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for item in volume_payload.get("surfaces", []):
        points = np.asarray(item.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
            continue
        values = np.asarray(item.get("values", []), dtype=float).reshape(-1)
        if values.size == points.shape[0]:
            points = np.column_stack((points[:, :3], values))
        surfaces.append({**item, "array": points})
    return surfaces


def _render_volume_plotly(
    project_root: Path, volume_payload: dict[str, Any], task_id: str
) -> str:
    cigvis, plotlyplot = _load_cigvis(project_root)
    with _CIGVIS_LOCK:
        cigvis.set_order(True)
        cigvis.set_axis_reversed(False, False, True)
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
                raise ValueError(
                    f"预测叠加体与背景体不对齐：{overlay_volume.shape} != {volume.shape}"
                )
            nodes = plotlyplot.add_mask(
                nodes,
                overlay_volume,
                clim=overlay.get("clim", [0.5, 1.0]),
                cmap=overlay.get("cmap", "jet"),
                alpha=float(overlay.get("alpha", 0.62)),
                excpt=overlay.get("excpt", "min"),
                interpolation="nearest",
            )
        scene_surfaces = _scene_surface_arrays(volume_payload)
        for surface in scene_surfaces:
            if surface.get("kind") == "surface" and surface.get("grid"):
                grid = np.asarray(surface["grid"], dtype=float)
                nodes += plotlyplot.create_surfaces(
                    grid,
                    value_type="depth",
                    clim=[0, max(volume.shape[2] - 1, 1)],
                    cmap=str(surface.get("cmap") or "jet"),
                    show_cbar=True,
                    opacity=float(surface.get("alpha") or 0.72),
                    name=_public_visualization_text(surface.get("name"), "解释面"),
                )
            else:
                color: Any = str(surface.get("color") or "#ef4444")
                values = (
                    surface["array"][:, 3] if surface["array"].shape[1] >= 4 else None
                )
                if values is not None:
                    color = values
                nodes += plotlyplot.create_points(
                    surface["array"][:, :3],
                    color=color,
                    size=4,
                    sym="circle",
                )
        well_logs, well_names = _well_log_arrays(volume_payload, volume.shape)
        if well_logs:
            nodes += plotlyplot.create_line_logs(
                well_logs, cmap="viridis", line_width=7
            )

        fig = plotlyplot.plot3D(nodes, show=False, size=(900, 1200))
        base_count = 3
        extra_trace_names = [
            _public_visualization_text(surface.get("name"), "解释面")
            for surface in scene_surfaces
        ] + well_names
        for index, trace in enumerate(fig.data):
            if index < base_count:
                trace.name = ("Inline切片", "Crossline切片", "时间切片")[index]
                trace.showlegend = False
            else:
                name_index = index - base_count
                trace.name = (
                    extra_trace_names[name_index]
                    if name_index < len(extra_trace_names)
                    else str(getattr(trace, "name", "") or "解释成果")
                )
                trace.showlegend = True
        vertical_axis = dict(volume_payload.get("verticalAxis", {}))
        vertical_title = f"{vertical_axis.get('label', '采样轴（域未核验）')} / {vertical_axis.get('unit', '')}".rstrip(
            " / "
        )
        horizontal_extent = dict(volume_payload.get("horizontalExtent", {}))
        inline_span = float(horizontal_extent.get("inlineSpanM", 0.0) or 0.0)
        crossline_span = float(horizontal_extent.get("crosslineSpanM", 0.0) or 0.0)
        horizontal_max = max(inline_span, crossline_span)
        inline_ratio = inline_span / horizontal_max if horizontal_max > 0 else 1.0
        crossline_ratio = (
            crossline_span / horizontal_max
            if horizontal_max > 0
            else volume.shape[1] / max(volume.shape[0], 1)
        )
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
                "font": {"size": 12, "color": "#33475b"},
            },
            scene={
                "xaxis": _tick_axis(
                    volume_payload.get("inlineValues", []),
                    _axis_labels(volume_payload)["inline"],
                ),
                "yaxis": _tick_axis(
                    volume_payload.get("crosslineValues", []),
                    _axis_labels(volume_payload)["crossline"],
                ),
                "zaxis": _tick_axis(
                    volume_payload.get("timeValues", []), vertical_title
                ),
                "aspectmode": "manual",
                "aspectratio": {
                    "x": max(0.18, inline_ratio),
                    "y": max(0.18, crossline_ratio),
                    "z": 0.64,
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
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"cigvis_{task_id[:8]}",
                    "scale": 2,
                },
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
        if (
            _VISER_SERVER is None
            or _VISER_SCENE_KEY is None
            or _VISER_SCENE_SHAPE is None
        ):
            raise RuntimeError("当前没有活动的三维可视化场景")
        if _VISER_SCENE_KEY[:2] != (task_id, int(asset_index)):
            raise RuntimeError("切片控制请求与当前三维场景不匹配，请刷新工作台")
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
        return {
            "task_id": task_id,
            "asset": int(asset_index),
            "positions": current,
            "shape": list(_VISER_SCENE_SHAPE),
        }


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
            raise RuntimeError("当前没有活动的三维可视化场景")
        if _VISER_SCENE_KEY[:2] != (task_id, int(asset_index)):
            raise RuntimeError("图层控制请求与当前三维场景不匹配，请刷新工作台")
        setter = getattr(_VISER_SERVER, "set_slice_background_visible", None)
        if not callable(setter):
            raise RuntimeError("当前三维可视化组件不支持独立控制背景图层")
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
        fixed_distance = configured_distance or (
            initial_distance if initial_distance > 1e-6 else 3.0
        )

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
                client.camera.position = tuple(
                    float(value) for value in desired_position
                )


def _ensure_viser_scene(
    project_root: Path,
    volume_payload: dict[str, Any],
    task_id: str,
    asset_index: int,
) -> int:
    global _VISER_SERVER, _VISER_SCENE_KEY, _VISER_SCENE_SHAPE, _VISER_ERROR
    cube = volume_payload.get("cube", {})
    digest = hashlib.sha1(
        str(cube.get("values", "")).encode("ascii", errors="ignore")
    ).hexdigest()[:16]
    scene_key = (task_id, int(asset_index), digest)
    with _CIGVIS_LOCK:
        viserplot = _load_viser(project_root)
        if _VISER_SERVER is not None and _VISER_SCENE_KEY != scene_key:
            _stop_viser_service()
        if _VISER_SERVER is None:
            _VISER_SERVER = viserplot.create_server(
                port=8080, label="well-seismic-cigvis", verbose=False
            )
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
                    axis_labels={
                        "x": _axis_labels(volume_payload)["inline"],
                        "y": _axis_labels(volume_payload)["crossline"],
                        "z": _axis_labels(volume_payload)["sample"],
                    },
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
                    raise ValueError(
                        f"预测叠加体与背景体不对齐：{overlay_volume.shape} != {original_shape}"
                    )
                nodes = viserplot.add_mask(
                    nodes,
                    overlay_volume,
                    clim=overlay.get("clim", [0.5, 1.0]),
                    cmap=overlay.get("cmap", "jet"),
                    alpha=float(overlay.get("alpha", 0.62)),
                    excpt=overlay.get("excpt", "min"),
                )
            for surface in _scene_surface_arrays(volume_payload):
                values = (
                    surface["array"][:, 3] if surface["array"].shape[1] >= 4 else None
                )
                if surface.get("kind") == "surface" and surface.get("grid"):
                    nodes += viserplot.create_surfaces(
                        np.asarray(surface["grid"], dtype=float),
                        value_type="depth",
                        clim=[0, max(original_shape[2] - 1, 1)],
                        cmap=str(surface.get("cmap") or "jet"),
                        alpha=float(surface.get("alpha") or 0.72),
                    )
                else:
                    nodes += viserplot.create_points(
                        surface["array"][:, :3],
                        r=3,
                        color=(
                            None if values is not None else surface.get("color", "red")
                        ),
                        values=values,
                        cmap=str(surface.get("cmap") or "jet"),
                    )
            well_logs, _ = _well_log_arrays(volume_payload, original_shape)
            if well_logs:
                nodes += viserplot.create_well_logs(
                    well_logs,
                    logs_type="line",
                    cmap="viridis",
                    width=3,
                )
            horizontal_extent = dict(volume_payload.get("horizontalExtent", {}))
            inline_span = float(horizontal_extent.get("inlineSpanM", 0.0) or 0.0)
            crossline_span = float(horizontal_extent.get("crosslineSpanM", 0.0) or 0.0)
            horizontal_max = max(inline_span, crossline_span)
            desired_extents = np.asarray(
                (
                    (
                        max(0.18, inline_span / horizontal_max)
                        if horizontal_max > 0
                        else 1.0
                    ),
                    (
                        max(0.18, crossline_span / horizontal_max)
                        if horizontal_max > 0
                        else max(
                            0.55, min(1.35, volume.shape[1] / max(volume.shape[0], 1))
                        )
                    ),
                    0.64,
                ),
                dtype=float,
            )
            shape_array = np.asarray(volume.shape, dtype=float)
            axis_scales = desired_extents * float(max(volume.shape)) / shape_array
            extents = 1.5 * desired_extents
            center = extents / 2.0
            camera_position = center + np.asarray((1.25, -1.35, -1.1), dtype=float)
            viserplot.plot3D(
                nodes,
                server=_VISER_SERVER,
                run_app=False,
                axis_scales=axis_scales.tolist(),
                fov=28,
                look_at=center.tolist(),
                position=camera_position.tolist(),
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
        '<div class="viser-loading" id="viser-loading">正在连接三维可视化场景…</div>'
        '<iframe id="cigvis-viser-frame" title="三维地震成果场景" '
        'allow="fullscreen; clipboard-write" allowfullscreen></iframe></div>'
        "<script>"
        '(function(){const frame=document.getElementById("cigvis-viser-frame");'
        'const loading=document.getElementById("viser-loading");'
        f'frame.src=window.location.protocol+"//"+window.location.hostname+":{int(port)}/";'
        'frame.addEventListener("load",()=>{if(loading)loading.hidden=true;});})();'
        "</script>"
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
        label_scope = str(item.get("labelScope", "")).casefold()
        if label_scope == "global_packages":
            subject = "全局层位"
            default_name = "全局层间编号"
            only_label = "仅全局层位结果"
        elif label_scope == "inline_local":
            subject = "Inline 局部分层"
            default_name = "局部分层（仅当前 Inline）"
            only_label = "仅 Inline 局部分层"
        else:
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
    elif kind == "class_code":
        subject = "地震相类别"
        default_name = "地震相确定类别"
        only_label = "仅地震相类别"
        threshold_label = ""
        show_threshold = False
    elif kind == "mask":
        subject = "断层掩码"
        default_name = "断层二值掩码"
        only_label = "仅断层掩码"
        threshold_label = ""
        show_threshold = False
    elif kind in {"continuous", "scalar", "regression"}:
        subject = "连续预测场"
        default_name = "连续预测场"
        only_label = "仅连续预测场"
        threshold_label = ""
        show_threshold = False
    else:
        subject = "预测结果"
        default_name = "预测结果"
        only_label = "仅预测结果"
        threshold_label = "结果显示下限"
        show_threshold = True
    subject = _public_visualization_text(item.get("subject"), subject)
    only_label = _public_visualization_text(
        item.get("onlyLabel", item.get("only_label")), only_label
    )
    threshold_label = _public_visualization_text(
        item.get("thresholdLabel", item.get("threshold_label")), threshold_label
    )
    if "showThreshold" in item or "show_threshold" in item:
        show_threshold = bool(item.get("showThreshold", item.get("show_threshold")))
    return {
        "kind": kind,
        "subject": subject,
        "display_name": _public_visualization_text(
            item.get("displayName", item.get("name")), default_name
        ),
        "only_label": only_label,
        "threshold_label": threshold_label,
        "show_threshold": show_threshold,
        "label_scope": str(item.get("labelScope", "")),
        "global_consistent": item.get("globalConsistent") is True,
    }


def _well_log_layout_profile(raw_tracks: object) -> dict[str, str] | None:
    """Return the fixed-board profile for one pure, supported well-side task.

    A board may contain multiple wells, but every track must belong to the
    same public task and must use that task's sealed result kind.  Mixed task
    bundles and lookalike/unknown result shapes deliberately keep the generic
    result rail.
    """

    if not isinstance(raw_tracks, list) or not raw_tracks:
        return None
    task_ids = {
        str(track.get("taskId") or track.get("task_id") or "")
        .strip()
        .casefold()
        for track in raw_tracks
        if isinstance(track, dict)
    }
    if len(task_ids) != 1 or len(raw_tracks) != sum(
        isinstance(track, dict) for track in raw_tracks
    ):
        return None
    task_id = next(iter(task_ids))
    profile = _WELL_LOG_LAYOUT_PROFILES.get(task_id)
    if profile is None:
        return None
    expected_kind = str(profile["kind"])
    accepted_kinds = (
        {"categorical_intervals", "classification_intervals"}
        if expected_kind == "categorical_intervals"
        else {expected_kind}
    )
    if not all(
        str(track.get("kind") or "").strip().casefold() in accepted_kinds
        for track in raw_tracks
    ):
        return None
    return {"task_id": task_id, **profile}


def _uses_facies_log_layout(raw_tracks: object) -> bool:
    """Compatibility predicate for callers that only need fixed-board state."""

    return _well_log_layout_profile(raw_tracks) is not None


def _webgl_volume_fragment(
    volume_payload: dict[str, Any], *, task_id: str = ""
) -> str:
    """Build a centered WebGL2 ray-marched whole-volume view.

    The browser receives the same lightweight [Z, Inline, Crossline] preview
    used by CIGVis.  Seismic amplitudes are rendered with the seismic
    diverging colormap and the first prediction overlay is blended using its
    declared display contract. Fault masks remain binary and nearest-neighbour
    sampled in 2D. Their optional 3D derivative is block occupancy, rendered
    with trilinear display interpolation and an explicitly non-model threshold.
    """
    cube = dict(volume_payload.get("cube", {}))
    shape = [int(value) for value in cube.get("shape", [])]
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"无效的整体三维预览形状：{shape}")
    default_indices = [size // 2 for size in shape]
    declared_default_indices = volume_payload.get("defaultIndices")
    if (
        isinstance(declared_default_indices, Sequence)
        and not isinstance(declared_default_indices, (str, bytes, bytearray))
        and len(declared_default_indices) == 3
    ):
        try:
            parsed_default_indices = [int(value) for value in declared_default_indices]
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            default_indices = [
                min(max(value, 0), size - 1)
                for value, size in zip(parsed_default_indices, shape, strict=True)
            ]

    overlay_payloads: list[dict[str, Any]] = []
    overlay_ui = _overlay_ui_metadata(None)
    overlays = list(volume_payload.get("overlays", []))
    for overlay_index, overlay in enumerate(overlays):
        if not isinstance(overlay, dict):
            raise ValueError(f"第 {overlay_index + 1} 个预测叠加层不是对象")
        overlay_spec = dict(overlay.get("volume", overlay))
        overlay_shape = [int(value) for value in overlay_spec.get("shape", [])]
        if overlay_shape != shape:
            raise ValueError(f"预测叠加体与背景体不对齐：{overlay_shape} != {shape}")
        overlay_3d_source = overlay.get("volume3D", overlay_spec)
        if not isinstance(overlay_3d_source, dict):
            raise ValueError(f"第 {overlay_index + 1} 个三维叠加体不是对象")
        overlay_3d_spec = dict(overlay_3d_source)
        overlay_3d_shape = [int(value) for value in overlay_3d_spec.get("shape", [])]
        if overlay_3d_shape != shape:
            raise ValueError(
                f"三维预测叠加体与背景体不对齐：{overlay_3d_shape} != {shape}"
            )
        item_ui = _overlay_ui_metadata(overlay)
        overlay_payloads.append(
            {
                "volume": overlay_spec,
                "volume3D": overlay_3d_spec,
                "clim": list(overlay.get("clim", [0.5, 1.0])),
                "alpha": float(overlay.get("alpha", 0.62)),
                "cmap": str(overlay.get("cmap", "jet")),
                "name": _public_visualization_text(
                    overlay.get("name"), "下游预测结果"
                ),
                "kind": str(overlay.get("kind", "probability")),
                "labelScope": str(overlay.get("labelScope", "")),
                "globalConsistent": overlay.get("globalConsistent") is True,
                "renderStyle": str(overlay.get("renderStyle", "")),
                "boundaryColor": str(overlay.get("boundaryColor", "#f59e0b")),
                "boundaryAlpha": float(overlay.get("boundaryAlpha", 0.98)),
                "ui": item_ui,
            }
        )
    overlay_payload = overlay_payloads[0] if overlay_payloads else None
    if overlay_payload is not None:
        overlay_ui = dict(overlay_payload["ui"])
    overlay_is_occupancy_mask = bool(
        overlay_payload is not None
        and str(overlay_payload.get("kind", "")).casefold() == "mask"
        and str(
            dict(overlay_payload.get("volume3D", {})).get(
                "samplingAggregation", ""
            )
        ).casefold()
        == "block_fault_fraction"
    )

    surface_items = [
        {
            **item,
            "name": _public_visualization_text(item.get("name"), "解释面"),
        }
        for item in volume_payload.get("surfaces", [])
        if isinstance(item, dict)
    ]
    candidate_visualization = dict(volume_payload.get("candidateVisualization", {}))
    surface_legend = "".join(
        '<span class="surface-legend">'
        f'<b style="color:{html.escape(str(item.get("color") or "#ef4444"))}">●</b>'
        f'{html.escape(_public_visualization_text(item.get("name"), "解释面"))}</span>'
        for item in surface_items
    )
    candidate_legend = ""
    if candidate_visualization.get("renderable"):
        candidate_label = (
            "工程候选"
            if candidate_visualization.get("display_status") == "engineering_candidate"
            else "实验候选"
        )
        candidate_legend = (
            '<span class="candidate-surface-status"><b>'
            f"{html.escape(candidate_label)}</b>"
            "未作为当前工区科学验收结果</span>"
        )

    slice_view_contract = dict(volume_payload.get("sliceViewContract", {}))
    if "displayNotice" in slice_view_contract:
        slice_view_contract["displayNotice"] = _public_visualization_text(
            slice_view_contract.get("displayNotice")
        )
    public_fault_grid = dict(volume_payload.get("faultSegGrid", {}))
    if public_fault_grid:
        public_fault_grid["receiptFileName"] = "断层识别推理回执"
        public_fault_grid.pop("receiptPath", None)
    well_result_tracks = list(volume_payload.get("wellResultTracks", []))
    well_log_profile = _well_log_layout_profile(well_result_tracks)
    facies_log_mode = well_log_profile is not None
    well_log_title = (
        str(well_log_profile["title"]) if well_log_profile else "井侧预测结果"
    )
    well_log_track_heading = (
        str(well_log_profile["track_heading"])
        if well_log_profile
        else "井侧预测结果"
    )
    payload = {
        "cube": cube,
        "overlay": overlay_payload,
        "overlays": overlay_payloads,
        # Keep source ordering intact: result bindings use the original
        # embeddedWells index.  The browser applies a presentation-only
        # fallback for incomplete/legacy geometry without ever promoting it to
        # an accepted MD->TWT placement.
        "wells": [
            item
            for item in volume_payload.get("embeddedWells", [])
            if isinstance(item, dict)
        ],
        "wellResultTracks": well_result_tracks,
        # A pure, contract-backed well-side result receives a conventional
        # fixed log board at the left of the same viewer. Mixed task bundles
        # deliberately retain the generic well-result rail.
        "faciesLogMode": facies_log_mode,
        "wellLogTaskId": str(well_log_profile["task_id"]) if well_log_profile else "",
        "wellLogTitle": well_log_title if well_log_profile else "",
        "viewStateKey": str(volume_payload.get("viewStateKey") or task_id).strip()[:256],
        "name": _public_visualization_text(volume_payload.get("name"), "三维地震体"),
        "verticalAxis": dict(volume_payload.get("verticalAxis", {})),
        "horizontalExtent": dict(volume_payload.get("horizontalExtent", {})),
        "axisLabels": _axis_labels(volume_payload),
        "defaultIndices": default_indices,
        "timeValues": list(volume_payload.get("timeValues", [])),
        "inlineValues": list(volume_payload.get("inlineValues", [])),
        "crosslineValues": list(volume_payload.get("crosslineValues", [])),
        "surfaces": [
            {key: value for key, value in item.items()} for item in surface_items
        ],
        "sliceViewContract": slice_view_contract,
        "candidateVisualization": candidate_visualization,
        "gridOverviewMode": str(volume_payload.get("assetKind") or "")
        == "faultseg_grid",
        "faultSegGrid": public_fault_grid,
    }
    payload_json = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    grid_overview = payload["gridOverviewMode"] is True
    grid_shape = [
        int(value)
        for value in payload["faultSegGrid"].get("gridShapeZYX", [])
    ]
    grid_block_count = len(payload["faultSegGrid"].get("blocks", []))
    grid_layer_count = grid_shape[0] if len(grid_shape) == 3 else 0
    grid_legend = (
        '<span class="fault-grid-layer-legend"><i></i>'
        f'<b>{grid_layer_count} 个垂向层色</b>'
        '<small>仅表示抽样位置，不表示断层概率</small></span>'
        if grid_overview
        else ""
    )
    well_result_log_head = (
        '<div class="well-result-log-track-head" aria-hidden="true">'
        f'<span>MD / m</span><strong>{html.escape(well_log_track_heading)}</strong></div>'
        if facies_log_mode
        else ""
    )
    well_log_task_attr = (
        f' data-log-task="{html.escape(str(well_log_profile["task_id"]))}"'
        if well_log_profile
        else ""
    )
    well_result_hud = (
        '<svg class="well-result-leader" id="well-result-leader" aria-hidden="true" hidden>'
        '<path id="well-result-leader-path"></path><circle id="well-result-anchor" r="4"></circle></svg>'
        f'<aside class="well-result-hud" id="well-result-hud" aria-label="'
        f'{html.escape(well_log_title)}"{well_log_task_attr} hidden>'
        f'<header><div><span id="well-result-kicker">'
        f'{html.escape(well_log_title)}'
        '</span><strong id="well-result-title"></strong></div>'
        '<nav aria-label="结果面板操作"><button id="well-result-prev" type="button" aria-label="上一口结果井">‹</button>'
        '<button id="well-result-next" type="button" aria-label="下一口结果井">›</button>'
        '<button id="well-result-size" type="button" aria-label="放大结果面板" aria-pressed="false">⤢</button>'
        '<button id="well-result-collapse" type="button" aria-label="折叠结果面板" aria-expanded="true">−</button></nav></header>'
        '<div class="well-result-body" id="well-result-body">'
        '<p class="well-result-link-state" id="well-result-link-state"></p>'
        f'{well_result_log_head}'
        '<div class="well-result-track-layout"><div class="well-result-axis" id="well-result-axis" aria-hidden="true">'
        '<span id="well-result-md-top"></span><span id="well-result-md-middle"></span>'
        '<span id="well-result-md-bottom"></span></div>'
        '<div class="well-result-classification-track" id="well-result-classification-track" '
        'role="listbox" aria-label="确定性 MD 分类层段"></div>'
        '<div class="well-result-property-track" id="well-result-property-track" hidden>'
        '<svg id="well-result-property-svg" viewBox="0 0 200 600" '
        'preserveAspectRatio="none" role="img" aria-label="储层物性连续 MD 曲线">'
        '<polygon id="well-result-property-band" hidden></polygon>'
        '<polyline id="well-result-property-curve"></polyline>'
        '<line id="well-result-property-cursor-line" x1="0" x2="200" hidden></line>'
        '<circle id="well-result-property-cursor" r="5" hidden></circle></svg>'
        '<div class="well-result-property-scale"><span id="well-result-property-min"></span>'
        '<strong id="well-result-property-label"></strong>'
        '<span id="well-result-property-max"></span></div>'
        '<input id="well-result-property-probe" type="range" min="0" max="1" '
        'value="0" step="1" aria-label="选择储层物性曲线 MD 样点"></div></div>'
        '<div class="well-result-legend" id="well-result-legend"></div>'
        '<output class="well-result-detail" id="well-result-detail" aria-live="polite"></output>'
        '</div></aside>'
        if payload["wellResultTracks"]
        else ""
    )
    template = r"""
<div class="volume-render-view__GRID_ROOT_CLASS__" id="volume-render-view"__GRID_ROOT_ATTRS__>
  <canvas id="volume-render-canvas" aria-label="__CANVAS_ARIA_LABEL__"__CANVAS_KEYBOARD_ATTRS__></canvas>
  <canvas id="volume-wire-canvas" aria-hidden="true"></canvas>
  <div class="fault-grid-3d-tooltip" id="fault-grid-3d-tooltip" hidden></div>
  __WELL_RESULT_HUD__
  <div class="volume-render-tools">
    <label class="base-volume-tool"><span>振幅增益</span><input id="volume-gain" type="range" min="0.5" max="3" value="1.45" step="0.05"></label>
    <label class="base-volume-tool"><span>体透明度</span><input id="volume-opacity" type="range" min="0.03" max="0.28" value="0.12" step="0.01"></label>
    <label class="base-volume-tool"><span>振幅阈值</span><input id="volume-threshold" type="range" min="0.02" max="0.55" value="0.14" step="0.01"></label>
    <label class="overlay-tool" id="volume-overlay-tool"><span id="volume-overlay-threshold-label">__OVERLAY_THRESHOLD_LABEL__</span><input id="volume-overlay-threshold" type="range" min="0.00392156862745098" max="1" value="0.5" step="0.01"></label>
    <label class="mask-slab-tool" id="volume-mask-slab-axis-tool"><span>三维显示窗方向（仅裁剪显示）</span><select id="volume-mask-slab-axis"><option value="2">时间 / 深度</option><option value="1">Inline</option><option value="0">Crossline</option></select></label>
    <label class="mask-slab-tool" id="volume-mask-slab-center-tool"><span>三维显示窗中心（仅裁剪显示）</span><input id="volume-mask-slab-center" type="range" min="0" max="1" value="0.5" step="0.01"></label>
    <label class="mask-slab-tool" id="volume-mask-slab-thickness-tool"><span>三维显示窗厚度（仅裁剪显示）</span><input id="volume-mask-slab-thickness" type="range" min="0.04" max="1" value="0.25" step="0.01"></label>
    <button id="volume-reset" type="button">恢复上方斜视</button>
  </div>
  <div class="volume-legend"><span class="vertical-axis-legend"><b>垂向</b><em id="volume-vertical-axis">时间轴向下增加</em></span>__CANDIDATE_LEGEND__<span id="volume-seismic-legend"><i class="seismic-ramp"></i>地震振幅 · seismic</span><span id="volume-overlay-legend"><i id="volume-overlay-ramp" class="jet-ramp"></i><b id="volume-overlay-legend-label">__OVERLAY_LEGEND_LABEL__</b></span>__GRID_LEGEND____SURFACE_LEGEND__<span class="well-type-legend"><b style="color:#1677e8">●</b>直井</span><span class="well-type-legend"><b style="color:#e6a23c">●</b>斜井</span><span class="well-type-legend"><b style="color:#d23b63">●</b>水平井</span><span class="well-type-legend"><b style="color:#d97706">┄</b>时间轨迹</span><span class="well-type-legend"><b style="color:#39718c">┄</b>井轨迹参考</span></div>
  <div class="volume-render-status" id="volume-render-status" role="status" aria-live="polite">正在初始化整体三维体渲染…</div>
</div>
<script>
(function(){
  const payload=__PAYLOAD__;
  // The orthogonal 2D viewer consumes the exact same bounded payload as the
  // whole-volume renderer.  Keeping one in-page object prevents duplicated
  // base64 cubes and guarantees both views use identical axes and samples.
  window.__wellSeismicVolumePayload=payload;
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
  const maskSlabAxis=document.getElementById('volume-mask-slab-axis');
  const maskSlabCenter=document.getElementById('volume-mask-slab-center');
  const maskSlabThickness=document.getElementById('volume-mask-slab-thickness');
  const maskSlabTools=root.querySelectorAll('.mask-slab-tool');
  const baseTools=root.querySelectorAll('.base-volume-tool');
  const seismicLegend=document.getElementById('volume-seismic-legend');
  const overlayLegend=document.getElementById('volume-overlay-legend');
  const surfaceLegends=root.querySelectorAll('.surface-legend');
  const wellTypeLegends=root.querySelectorAll('.well-type-legend');
  const overlayLegendLabel=document.getElementById('volume-overlay-legend-label');
  const overlayRamp=document.getElementById('volume-overlay-ramp');
  const gridTooltip=document.getElementById('fault-grid-3d-tooltip');
  const wellResultHud=document.getElementById('well-result-hud');
  const wellResultLeader=document.getElementById('well-result-leader');
  const wellResultLeaderPath=document.getElementById('well-result-leader-path');
  const wellResultAnchor=document.getElementById('well-result-anchor');
  const wellResultTitle=document.getElementById('well-result-title');
  const wellResultAxis=document.getElementById('well-result-axis');
  const wellResultLinkState=document.getElementById('well-result-link-state');
  const wellResultTrackElement=document.getElementById('well-result-classification-track');
  const wellResultPropertyTrack=document.getElementById('well-result-property-track');
  const wellResultPropertySvg=document.getElementById('well-result-property-svg');
  const wellResultPropertyBand=document.getElementById('well-result-property-band');
  const wellResultPropertyCurve=document.getElementById('well-result-property-curve');
  const wellResultPropertyCursorLine=document.getElementById('well-result-property-cursor-line');
  const wellResultPropertyCursor=document.getElementById('well-result-property-cursor');
  const wellResultPropertyMinimum=document.getElementById('well-result-property-min');
  const wellResultPropertyLabel=document.getElementById('well-result-property-label');
  const wellResultPropertyMaximum=document.getElementById('well-result-property-max');
  const wellResultPropertyProbe=document.getElementById('well-result-property-probe');
  const wellResultLegend=document.getElementById('well-result-legend');
  const wellResultDetail=document.getElementById('well-result-detail');
  const wellResultMdTop=document.getElementById('well-result-md-top');
  const wellResultMdMiddle=document.getElementById('well-result-md-middle');
  const wellResultMdBottom=document.getElementById('well-result-md-bottom');
  const wellResultPrev=document.getElementById('well-result-prev');
  const wellResultNext=document.getElementById('well-result-next');
  const wellResultSize=document.getElementById('well-result-size');
  const wellResultCollapse=document.getElementById('well-result-collapse');
  const wellResultHeader=wellResultHud?.querySelector('header')||null;
  const gl=canvas.getContext('webgl2',{alpha:true,antialias:true,premultipliedAlpha:false});
  const gridOverview=payload.gridOverviewMode===true;
  const faciesLogMode=payload.faciesLogMode===true;
  const faultGrid=gridOverview&&payload.faultSegGrid&&typeof payload.faultSegGrid==='object'?payload.faultSegGrid:null;
  const gridBlocks=faultGrid&&Array.isArray(faultGrid.blocks)?[...faultGrid.blocks].sort((a,b)=>Number(a.ordinal)-Number(b.ordinal)):[];
  const gridShape=faultGrid&&Array.isArray(faultGrid.gridShapeZYX)?faultGrid.gridShapeZYX.map(Number):[0,0,0];
  const gridBlockCount=gridShape.length===3?gridShape[0]*gridShape[1]*gridShape[2]:0;
  const gridRowWidth=Math.max(1,gridShape[2]||1),gridPlaneSize=Math.max(1,(gridShape[1]||1)*gridRowWidth);
  const defaultGridIndex=Math.max(0,gridBlocks.findIndex(block=>block.blockId===faultGrid?.defaultBlockId));
  const state={yaw:-0.66,pitch:-0.38,zoom:1.72,predictionOnly:false,program:null,volumeTexture:null,overlayTexture:null,position:-1,selectedBlockIndex:defaultGridIndex,hoveredBlockIndex:-1,wellResultHudTop:82,wellResultHudExpanded:false,wellResultHudCollapsed:false};
  const viewStateStorageKey=payload.viewStateKey?'well-seismic:cigvis-view:'+String(payload.viewStateKey):'';
  function restoreViewState(){
    if(!viewStateStorageKey)return;
    try{const saved=JSON.parse(sessionStorage.getItem(viewStateStorageKey)||'null');if(!saved||typeof saved!=='object')return;const yaw=Number(saved.yaw),pitch=Number(saved.pitch),zoom=Number(saved.zoom),wellResultHudTop=Number(saved.wellResultHudTop);if(Number.isFinite(yaw)&&Math.abs(yaw)<10000)state.yaw=yaw;if(Number.isFinite(pitch))state.pitch=Math.max(-1.25,Math.min(1.25,pitch));if(Number.isFinite(zoom)&&zoom>=.72&&zoom<=6)state.zoom=zoom;if(Number.isFinite(wellResultHudTop))state.wellResultHudTop=wellResultHudTop;state.wellResultHudExpanded=saved.wellResultHudExpanded===true;state.wellResultHudCollapsed=saved.wellResultHudCollapsed===true;}catch(_error){}
  }
  function saveViewState(){
    if(!viewStateStorageKey)return;
    try{sessionStorage.setItem(viewStateStorageKey,JSON.stringify({yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,wellResultHudTop:state.wellResultHudTop,wellResultHudExpanded:state.wellResultHudExpanded,wellResultHudCollapsed:state.wellResultHudCollapsed}));}catch(_error){}
  }
  restoreViewState();
  const shape=payload.cube.shape.map(Number);
  const sourceShape=faultGrid&&Array.isArray(faultGrid.sourceShapeZYX)?faultGrid.sourceShapeZYX.map(Number):shape;
  const gridLayerColors=[[22,119,232],[14,165,164],[245,158,11],[168,85,247]];
  let projectedGridBlocks=[];
  const wellResultTracks=Array.isArray(payload.wellResultTracks)?payload.wellResultTracks:[];
  let selectedWellResultTrackIndex=0,activeWellResultIntervalIndex=-1,hoveredWellResultIntervalIndex=-1,activeWellPropertyIndex=0,hoveredProjectedWellIndex=-1;
  let projectedWellRecords=[];
  const verticalAxis=payload.verticalAxis||{};
  const axisLabels=payload.axisLabels||{};
  const verticalAxisLabel=document.getElementById('volume-vertical-axis');
  if(verticalAxisLabel){const top=verticalAxis.top??'—',bottom=verticalAxis.bottom??'—',unit=verticalAxis.unit||'',label=verticalAxis.label||'采样轴（域未核验）';verticalAxisLabel.textContent=`${label} 向下 · 顶 ${top} ${unit} / 底 ${bottom} ${unit}`;}
  const maxShape=Math.max(...shape);
  // Texture coordinates are Crossline, Inline, Z.
  const horizontalExtent=payload.horizontalExtent||{};
  const inlineSpan=Number(horizontalExtent.inlineSpanM||0),crosslineSpan=Number(horizontalExtent.crosslineSpanM||0),maxHorizontal=Math.max(inlineSpan,crosslineSpan);
  const sourceHorizontalMaximum=Math.max(Number(sourceShape[1]||1),Number(sourceShape[2]||1));
  const boxScale=maxHorizontal>0?[Math.max(.18,crosslineSpan/maxHorizontal),Math.max(.18,inlineSpan/maxHorizontal),.64]:gridOverview?[Math.max(.18,sourceShape[2]/sourceHorizontalMaximum),Math.max(.18,sourceShape[1]/sourceHorizontalMaximum),.64]:[Math.max(.46,shape[2]/maxShape),Math.max(.46,shape[1]/maxShape),Math.max(.46,shape[0]/maxShape)];
  const overlayUi=payload.overlay?.ui||{};
  const overlayKind=String(payload.overlay?.kind||'').toLowerCase();
  const overlayIsMask=overlayKind==='mask';
  const overlayIsCategorical=overlayIsMask||overlayKind==='class_code'||overlayKind==='labels';
  const overlayVolume3D=payload.overlay?.volume3D||payload.overlay?.volume||null;
  const maskAggregation=String(overlayVolume3D?.samplingAggregation||'').toLowerCase();
  const maskUsesOccupancy=overlayIsMask&&maskAggregation==='block_fault_fraction';
  const minimumNonzeroOccupancy=1/255;
  const overlayName=String(overlayUi.display_name||payload.overlay?.name||'预测结果');
  const displaySlab=overlayVolume3D?.displaySlab||{};
  const slabAxisCode={Z:'2',TWT:'2',TIME:'2',DEPTH:'2',INLINE:'1',I:'1',CROSSLINE:'0',XLINE:'0',X:'0'};
  const declaredSlabAxis=String(displaySlab.axis||'Z').toUpperCase();
  if(Object.prototype.hasOwnProperty.call(slabAxisCode,declaredSlabAxis))maskSlabAxis.value=slabAxisCode[declaredSlabAxis];
  if(Number.isFinite(Number(displaySlab.centerNormalized)))maskSlabCenter.value=String(Math.max(0,Math.min(1,Number(displaySlab.centerNormalized))));
  if(Number.isFinite(Number(displaySlab.thicknessNormalized)))maskSlabThickness.value=String(Math.max(.04,Math.min(1,Number(displaySlab.thicknessNormalized))));
  maskSlabTools.forEach(tool=>tool.hidden=!maskUsesOccupancy);
  if(!payload.overlay){overlayTool.hidden=true;overlayLegend.hidden=true;}
  else{
    if(maskUsesOccupancy&&Number.isFinite(Number(overlayVolume3D?.displayThreshold)))overlayThreshold.value=String(Math.max(minimumNonzeroOccupancy,Number(overlayVolume3D.displayThreshold)));
    else if(payload.overlay.clim&&payload.overlay.clim.length)overlayThreshold.value=String(payload.overlay.clim[0]);
    overlayTool.hidden=overlayUi.show_threshold===false&&!maskUsesOccupancy;
    if(overlayThresholdLabel)overlayThresholdLabel.textContent=maskUsesOccupancy?'三维断层脊面强度下限（仅显示派生）':String(overlayUi.threshold_label||'结果显示下限');
    if(overlayLegendLabel)overlayLegendLabel.textContent=overlayIsMask?(maskUsesOccupancy?overlayName+' · 区块断层脊面（由二值掩码的区块占据率派生，不是概率）':overlayName+' · 二值 0/1（NEAREST）'):overlayName+' · jet';
    if(overlayRamp&&overlayIsMask){overlayRamp.classList.remove('jet-ramp');overlayRamp.classList.add('fault-mask-ramp');}
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
    if(spec.encoding==='base64-uint8'){
      const aggregation=String(spec.samplingAggregation||'').toLowerCase();
      if(overlayIsMask&&aggregation!=='block_fault_fraction'){for(let i=0;i<raw.length;i++){if(raw[i]!==0&&raw[i]!==255)throw new Error('断层二值掩码只能包含显示码 0/255');}}
      output.set(raw);return output;
    }
    if(overlayIsMask)throw new Error('断层二值掩码三维渲染只接受 base64-uint8 离散体');
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
  function createTexture(values,useNearest=false){
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,texture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
    const filter=useNearest?gl.NEAREST:gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,filter);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,filter);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,shape[2],shape[1],shape[0],0,gl.RED,gl.UNSIGNED_BYTE,values);
    return texture;
  }
  function openGridFallback(message){
    if(!gridOverview)return false;status.textContent=message;root.classList.add('unsupported');
    const open=()=>{const fallback=document.getElementById('fault-grid-fallback');if(fallback)fallback.open=true;};
    if(document.readyState==='loading')window.addEventListener('load',open,{once:true});else open();return true;
  }
  function initialize(){
    if(!gl){if(!openGridFallback(`当前浏览器不支持 WebGL2，已展开下方二维${gridBlockCount}块列表`)){status.textContent='当前浏览器不支持 WebGL2，请切换到切片查看';root.classList.add('unsupported');}return;}
    const vertex=`#version 300 es
      in vec2 a_position;out vec2 v_uv;
      void main(){v_uv=a_position;gl_Position=vec4(a_position,0.0,1.0);}`;
    const fragment=`#version 300 es
      precision highp float;precision highp sampler3D;
      in vec2 v_uv;out vec4 outColor;
      uniform sampler3D u_volume;uniform sampler3D u_overlay;
      uniform float u_yaw;uniform float u_pitch;uniform float u_zoom;uniform float u_gain;uniform float u_opacity;uniform float u_threshold;uniform float u_overlayThreshold;uniform float u_overlayOpacity;uniform float u_aspect;uniform float u_hasOverlay;uniform float u_predictionOnly;uniform float u_overlayIsMask;uniform float u_maskUsesOccupancy;uniform float u_maskSlabAxis;uniform float u_maskSlabCenter;uniform float u_maskSlabHalfThickness;uniform vec3 u_boxScale;uniform vec3 u_textureShape;
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
        float start=max(nearValue,0.);float rayLength=farValue-start;
        vec3 voxelTravel=abs(rayDirection/u_boxScale)*u_textureShape*rayLength;
        int stepCount=int(clamp(ceil(max(max(voxelTravel.x,voxelTravel.y),voxelTravel.z)*2.),64.,384.));
        float stepLength=rayLength/float(stepCount);vec4 accumulated=vec4(0.);float previousMask=0.;float previousOverlayValue=0.;float hasPrevious=0.;vec3 previousTex=vec3(0.);float maskHit=0.;float maskDepth=1.;vec3 maskTex=vec3(0.);
        for(int index=0;index<384;index++){
          if(index>=stepCount)break;
          float distanceValue=start+(float(index)+.5)*stepLength;vec3 local=rayOrigin+rayDirection*distanceValue;vec3 tex=local/u_boxScale+.5;tex.z=1.-tex.z;
          float amplitude=(texture(u_volume,tex).r*255.-128.)/127.;float scaledAmplitude=clamp(amplitude*u_gain,-1.,1.);float magnitude=abs(scaledAmplitude);
          float baseAlpha=(1.-u_predictionOnly)*smoothstep(u_threshold,1.,magnitude)*u_opacity;vec3 baseColor=seismic(scaledAmplitude);
          float overlayValue=texture(u_overlay,tex).r;
          float occupancyThreshold=max(1./255.,u_overlayThreshold);
          float rawOverlayMembership=u_overlayIsMask>.5?(u_maskUsesOccupancy>.5?(overlayValue>0.?step(occupancyThreshold,overlayValue):0.):step(.5,overlayValue)):0.;
          float occupancyShade=clamp((overlayValue-occupancyThreshold)/max(.02,1.-occupancyThreshold),0.,1.);
          float rawOverlayStrength=u_overlayIsMask>.5?(u_maskUsesOccupancy>.5?rawOverlayMembership*(.28+.72*occupancyShade):rawOverlayMembership):smoothstep(u_overlayThreshold,min(1.,u_overlayThreshold+.18),overlayValue);
          float slabCoordinate=u_maskSlabAxis<.5?tex.x:(u_maskSlabAxis<1.5?tex.y:tex.z);float slabVisible=u_maskUsesOccupancy>.5?step(abs(slabCoordinate-u_maskSlabCenter),u_maskSlabHalfThickness):1.;float overlayStrength=rawOverlayStrength*slabVisible;
          if(u_hasOverlay>.5&&u_overlayIsMask>.5&&maskHit<.5&&hasPrevious>.5&&rawOverlayMembership>.5&&previousMask<=0.){
            float activeOverlayThreshold=u_maskUsesOccupancy>.5?occupancyThreshold:u_overlayThreshold;float crossing=hasPrevious>.5?clamp((activeOverlayThreshold-previousOverlayValue)/max(overlayValue-previousOverlayValue,.000001),0.,1.):1.;
            vec3 crossingTex=hasPrevious>.5?mix(previousTex,tex,crossing):tex;float crossingSlabCoordinate=u_maskSlabAxis<.5?crossingTex.x:(u_maskSlabAxis<1.5?crossingTex.y:crossingTex.z);float crossingInSlab=u_maskUsesOccupancy>.5?step(abs(crossingSlabCoordinate-u_maskSlabCenter),u_maskSlabHalfThickness):1.;
            if(crossingInSlab>.5){maskHit=1.;maskDepth=(max(float(index)-1.,0.)+crossing)/float(max(stepCount-1,1));maskTex=crossingTex;}
          }
          if(u_overlayIsMask>.5){previousMask=rawOverlayMembership;previousOverlayValue=overlayValue;previousTex=tex;hasPrevious=1.;}
          float maskAlpha=u_overlayIsMask>.5?0.:u_hasOverlay*overlayStrength*u_overlayOpacity;
          vec3 overlayColor=u_overlayIsMask>.5?vec3(.94,.08,.11):jet(overlayValue);
          float sampleAlpha=max(baseAlpha,maskAlpha);vec3 sampleColor=mix(baseColor,overlayColor,clamp(maskAlpha/max(sampleAlpha,.0001),0.,1.));
          accumulated.rgb+=(1.-accumulated.a)*sampleAlpha*sampleColor;accumulated.a+=(1.-accumulated.a)*sampleAlpha;if(accumulated.a>.985&&!(u_hasOverlay>.5&&u_overlayIsMask>.5&&maskHit<.5))break;
        }
        if(u_hasOverlay>.5&&u_overlayIsMask>.5&&maskHit>.5){
          vec3 texel=1./u_textureShape;
          vec3 gradient=vec3(
            texture(u_overlay,clamp(maskTex+vec3(texel.x,0.,0.),0.,1.)).r-texture(u_overlay,clamp(maskTex-vec3(texel.x,0.,0.),0.,1.)).r,
            texture(u_overlay,clamp(maskTex+vec3(0.,texel.y,0.),0.,1.)).r-texture(u_overlay,clamp(maskTex-vec3(0.,texel.y,0.),0.,1.)).r,
            texture(u_overlay,clamp(maskTex+vec3(0.,0.,texel.z),0.,1.)).r-texture(u_overlay,clamp(maskTex-vec3(0.,0.,texel.z),0.,1.)).r
          );
          vec3 normal=length(gradient)>.0001?normalize(gradient):normalize(-rayDirection);
          float lighting=.32+.68*abs(dot(normal,normalize(vec3(.38,.52,.76))));
          lighting*=.72+.28*(1.-maskDepth);
          float projectedAlpha=clamp(u_overlayOpacity,0.,.86);
          vec3 projectedColor=mix(vec3(.48,.012,.024),vec3(1.,.20,.08),lighting);
          accumulated.rgb=projectedColor*projectedAlpha+accumulated.rgb*(1.-projectedAlpha);
          accumulated.a=projectedAlpha+accumulated.a*(1.-projectedAlpha);
        }
        if(accumulated.a<.01)discard;outColor=vec4(accumulated.rgb/max(accumulated.a,.0001),accumulated.a);
      }`;
    const program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program)||'着色器链接失败');
    state.program=program;state.position=gl.getAttribLocation(program,'a_position');
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);gl.enableVertexAttribArray(state.position);gl.vertexAttribPointer(state.position,2,gl.FLOAT,false,0,0);
    state.volumeTexture=createTexture(decodeSeismic(payload.cube),false);
    state.overlayTexture=createTexture(overlayVolume3D?decodeProbability(overlayVolume3D):new Uint8Array(shape[0]*shape[1]*shape[2]),overlayIsCategorical&&!maskUsesOccupancy);
    updateLayerStatus();
    render();
  }
  function updateLayerStatus(){
    surfaceLegends.forEach(legend=>legend.hidden=state.predictionOnly);
    wellTypeLegends.forEach(legend=>legend.hidden=state.predictionOnly);
    if(!payload.overlay){status.textContent=gridOverview?`完整工区有界地震预览 · ${gridBlockCount}个半透明代表块可点击 · 层色仅表示垂向位置`:'整体三维 · seismic 地震背景';return;}
    const slabCenter=Number(maskSlabCenter.value),slabThickness=Number(maskSlabThickness.value),slabStart=Math.max(0,slabCenter-slabThickness*.5),slabEnd=Math.min(1,slabCenter+slabThickness*.5);
    const slabAxis=Number(maskSlabAxis.value),slabAxisName=slabAxis<.5?'Crossline':slabAxis<1.5?'Inline':'时间/深度';
    const axisTopRaw=verticalAxis.top,axisBottomRaw=verticalAxis.bottom,axisTop=Number(axisTopRaw),axisBottom=Number(axisBottomRaw),axisUnit=String(verticalAxis.unit||'');
    const hasAxisRange=axisTopRaw!==null&&axisTopRaw!==undefined&&String(axisTopRaw).trim()!==''&&axisBottomRaw!==null&&axisBottomRaw!==undefined&&String(axisBottomRaw).trim()!==''&&Number.isFinite(axisTop)&&Number.isFinite(axisBottom);
    const horizontalValues=slabAxis<.5?(payload.crosslineValues||[]):(payload.inlineValues||[]);
    const startIndex=Math.max(0,Math.min(horizontalValues.length-1,Math.round(slabStart*Math.max(0,horizontalValues.length-1)))),endIndex=Math.max(0,Math.min(horizontalValues.length-1,Math.round(slabEnd*Math.max(0,horizontalValues.length-1))));
    const horizontalRange=horizontalValues.length?`${horizontalValues[startIndex]}–${horizontalValues[endIndex]}`:`${(slabStart*100).toFixed(0)}%–${(slabEnd*100).toFixed(0)}%`;
    const slabRange=slabAxis>1.5&&hasAxisRange?`${(axisTop+(axisBottom-axisTop)*slabStart).toFixed(1)}–${(axisTop+(axisBottom-axisTop)*slabEnd).toFixed(1)} ${axisUnit}`:horizontalRange;
    const renderContract=overlayIsMask?(maskUsesOccupancy?` · 区块断层脊面（显示强度 ≥ ${Number(overlayThreshold.value).toFixed(2)}） · ${slabAxisName} ${slabRange} · 原始二值掩码及模型阈值未改 · 每条视线仅取首个面`:' · 二值 NEAREST'):' · jet 连续结果';
    status.textContent=(state.predictionOnly?'整体三维 · 仅显示 '+overlayName:'整体三维 · seismic 地震背景 + '+overlayName)+renderContract;
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
    const overlayOpacity=overlayIsMask?(maskUsesOccupancy?(state.predictionOnly ? .74 : .42):(state.predictionOnly ? .72 : .48)):Math.max(.04,Number(payload.overlay?.alpha||.62)*(state.predictionOnly ? .55 : .26));
    const overlayThresholdValue=maskUsesOccupancy?Math.max(minimumNonzeroOccupancy,Number(overlayThreshold.value)):Number(overlayThreshold.value);
    gl.uniform1f(uniform('u_overlayThreshold'),overlayThresholdValue);gl.uniform1f(uniform('u_overlayOpacity'),payload.overlay?overlayOpacity:0);
    gl.uniform1f(uniform('u_hasOverlay'),payload.overlay?1:0);gl.uniform1f(uniform('u_predictionOnly'),state.predictionOnly?1:0);gl.uniform1f(uniform('u_overlayIsMask'),overlayIsMask?1:0);gl.uniform1f(uniform('u_maskUsesOccupancy'),maskUsesOccupancy?1:0);gl.uniform1f(uniform('u_maskSlabAxis'),Number(maskSlabAxis.value));gl.uniform1f(uniform('u_maskSlabCenter'),Number(maskSlabCenter.value));gl.uniform1f(uniform('u_maskSlabHalfThickness'),Number(maskSlabThickness.value)*.5);gl.uniform1f(uniform('u_aspect'),canvas.width/Math.max(canvas.height,1));gl.uniform3fv(uniform('u_boxScale'),boxScale);gl.uniform3fv(uniform('u_textureShape'),[shape[2],shape[1],shape[0]]);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);drawGuides();
  }
  function projectWithDepth(point){
    const cx=Math.cos(state.pitch),sx=Math.sin(state.pitch),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw);
    const afterX=[point[0],cx*point[1]-sx*point[2],sx*point[1]+cx*point[2]];
    const camera=[cy*afterX[0]+sy*afterX[2],afterX[1],-sy*afterX[0]+cy*afterX[2]];const dz=camera[2]-state.zoom;if(dz>=-.05)return null;
    const aspect=canvas.clientWidth/Math.max(canvas.clientHeight,1);const x=(-1.9*camera[0]/dz)/aspect;const y=-1.9*camera[1]/dz;
    return {x:(x+1)*.5*canvas.clientWidth,y:(1-y)*.5*canvas.clientHeight,depth:camera[2]};
  }
  function project(point){const value=projectWithDepth(point);return value?[value.x,value.y]:null;}
  function displayZ(normalizedTwt){return (.5-normalizedTwt)*boxScale[2];}
  function normalizeWellIdentity(value){
    return String(value??'').normalize('NFKC').trim().replace(/[（(](?:直井预览|vertical preview)[）)]$/i,'').replace(/[\s._-]+/g,'').toLocaleLowerCase();
  }
  function wellIdentityKeys(well){
    return new Set([well?.wellId,well?.wellUid,well?.well_id,well?.well_uid,well?.name].map(normalizeWellIdentity).filter(Boolean));
  }
  function trackWellKey(track){return normalizeWellIdentity(track?.wellId||track?.well_id||track?.wellUid||track?.well_uid);}
  function matchingWellIndexes(track){
    const key=trackWellKey(track);if(!key)return [];
    return (payload.wells||[]).map((well,index)=>wellIdentityKeys(well).has(key)?index:-1).filter(index=>index>=0);
  }
  function bindingMatchesWell(binding,well){
    if(!binding||!well)return false;
    const expectedUid=normalizeWellIdentity(binding.trajectoryWellUid),actualUid=normalizeWellIdentity(well.wellUid||well.well_uid);
    if(expectedUid&&expectedUid!==actualUid)return false;
    const expectedId=normalizeWellIdentity(binding.trajectoryWellId),actualId=normalizeWellIdentity(well.wellId||well.well_id||well.name);
    return !expectedId||expectedId===actualId;
  }
  function linkedWellIndex(track){
    const binding=track?.binding;if(binding&&typeof binding==='object'){
      const index=Number(binding.trajectoryIndex);
      const well=Number.isInteger(index)&&index>=0&&index<(payload.wells||[]).length?(payload.wells||[])[index]:null;
      return binding.status==='matched'&&bindingMatchesWell(binding,well)?index:-1;
    }
    const matches=matchingWellIndexes(track);return matches.length===1?matches[0]:-1;
  }
  function currentWellResultTrack(){return wellResultTracks[selectedWellResultTrackIndex]||null;}
  function resultWellMd(well){const values=well?.mdM||well?.md||well?.mdValues||well?.measuredDepthM||[];return Array.isArray(values)?values.map(Number):[];}
  function wellVisualGeometry(well){
    const x=Array.isArray(well?.x)?well.x.map(Number):[],y=Array.isArray(well?.y)?well.y.map(Number):[],rawZ=Array.isArray(well?.z)?well.z.map(Number):[],md=resultWellMd(well),count=Math.min(x.length,y.length),stations=[];
    for(let index=0;index<count;index++){if(Number.isFinite(x[index])&&Number.isFinite(y[index]))stations.push({index,x:x[index],y:y[index],z:Number.isFinite(rawZ[index])?rawZ[index]:null});}
    if(!stations.length)return {stations:[],displayMode:'unavailable',label:'无法定位'};
    const mode=String(well?.alignmentMode||''),verified=mode==='time_registered'&&well?.provenTwt===true&&verticalAxis?.domain==='TWT'&&verticalAxis?.twtVerified===true&&well?.formalRegistration===true&&well?.registrationAccepted===true,timeCandidate=mode==='time_registration_candidate',depthReference=mode==='depth_normalized_preview'||well?.verticalDisplayMode==='relative_tvd_preview';
    const finiteZ=stations.filter(station=>Number.isFinite(station.z)),zValues=finiteZ.map(station=>Number(station.z)),zSpread=zValues.length?Math.max(...zValues)-Math.min(...zValues):0,xValues=stations.map(station=>station.x),yValues=stations.map(station=>station.y),planSpread=Math.max(Math.max(...xValues)-Math.min(...xValues),Math.max(...yValues)-Math.min(...yValues));
    if(verified&&finiteZ.length===stations.length)return {stations,displayMode:'verified_twt',label:'TWT已核验'};
    if(timeCandidate&&finiteZ.length===stations.length)return {stations,displayMode:'time_candidate',label:'时间轨迹'};
    if(depthReference&&finiteZ.length===stations.length)return {stations,displayMode:'depth_reference',label:'井轨迹参考'};
    if(finiteZ.length===stations.length&&zSpread>1e-6)return {stations,displayMode:'unverified_z_reference',label:'井轨迹参考'};
    if(stations.length===1)return {stations:[{...stations[0],z:Number.isFinite(stations[0].z)?stations[0].z:0}],displayMode:'wellhead',label:'仅井口定位'};
    if(planSpread>1e-6)return {stations:stations.map(station=>({...station,z:Number.isFinite(station.z)?station.z:0})),displayMode:'xy_plan',label:'仅XY平面'};
    const stationMd=stations.map(station=>md[station.index]),finiteMd=stationMd.every(Number.isFinite),minimum=finiteMd?Math.min(...stationMd):0,maximum=finiteMd?Math.max(...stationMd):stations.length-1,span=maximum-minimum;
    const referenceStations=stations.map((station,index)=>({...station,z:.05+.9*((finiteMd&&span>1e-9?(stationMd[index]-minimum)/span:index/Math.max(stations.length-1,1)))}));
    return {stations:referenceStations,displayMode:'md_reference',label:'井轨迹参考'};
  }
  function verifiedMdToTwt(well,track=currentWellResultTrack()){
    const md=resultWellMd(well),x=(well?.x||[]).map(Number),y=(well?.y||[]).map(Number),z=(well?.z||[]).map(Number);
    const binding=track?.binding,display=track?.display||{},bindingAllows=!binding||(binding.status==='matched'&&binding.measuredDepthTrajectoryAvailable===true),placementAllows=!display.twtPlacement||display.twtPlacement==='accepted';
    const sameLength=md.length>=2&&md.length===x.length&&md.length===y.length&&md.length===z.length;
    const finite=sameLength&&[md,x,y,z].every(values=>values.every(Number.isFinite));
    const increasing=finite&&md.slice(1).every((value,index)=>value>md[index]);
    return bindingAllows&&placementAllows&&well?.alignmentMode==='time_registered'&&well?.provenTwt===true&&verticalAxis?.domain==='TWT'&&verticalAxis?.twtVerified===true&&well?.formalRegistration===true&&well?.registrationAccepted===true&&increasing;
  }
  function formatMdNumber(value){const numeric=Number(value);return Number.isFinite(numeric)?numeric.toLocaleString('zh-CN',{maximumFractionDigits:1}):'—';}
  function formatMd(value){return formatMdNumber(value)+(Number.isFinite(Number(value))?' m':'');}
  function renderWellResultDepthAxis(minimum,maximum){
    if(!wellResultAxis)return;
    if(!faciesLogMode){wellResultMdTop.textContent=formatMd(minimum);wellResultMdMiddle.textContent=formatMd((minimum+maximum)/2);wellResultMdBottom.textContent=formatMd(maximum);return;}
    wellResultAxis.replaceChildren();
    for(let index=0;index<=10;index++){
      const tick=document.createElement('span'),fraction=index/10;
      tick.textContent=formatMdNumber(minimum+(maximum-minimum)*fraction);
      tick.style.top=`${fraction*100}%`;tick.dataset.tick=String(index);
      wellResultAxis.append(tick);
    }
  }
  function activeWellResultInterval(){
    const track=currentWellResultTrack(),index=hoveredWellResultIntervalIndex>=0?hoveredWellResultIntervalIndex:activeWellResultIntervalIndex;
    return track&&index>=0?track.intervals?.[index]||null:null;
  }
  function updateWellResultButtons(){
    if(!wellResultTrackElement)return;
    wellResultTrackElement.querySelectorAll('.well-result-segment').forEach(button=>{
      const index=Number(button.dataset.intervalIndex),active=index===activeWellResultIntervalIndex,hovered=index===hoveredWellResultIntervalIndex;
      button.classList.toggle('active',active);button.classList.toggle('hovered',hovered);button.setAttribute('aria-selected',String(active));
    });
  }
  function describeWellResultInterval(interval){
    if(!interval)return '悬停分类段查看真实 MD 范围';
    const secondary=interval.secondaryLabel?' / '+interval.secondaryLabel:'';
    return `${interval.label}${secondary} · MD ${formatMd(interval.topMdM)}–${formatMd(interval.bottomMdM)}`;
  }
  function formatWellPropertyValue(value,unit=''){
    const numeric=Number(value);if(!Number.isFinite(numeric))return '—';
    const formatted=numeric.toLocaleString('zh-CN',{maximumSignificantDigits:6});return unit?`${formatted} ${unit}`:formatted;
  }
  function nearestWellPropertyIndex(mdValues,target){
    if(!mdValues.length)return -1;let low=0,high=mdValues.length-1;
    while(low<high){const middle=Math.floor((low+high)/2);if(Number(mdValues[middle])<target)low=middle+1;else high=middle;}
    if(low>0&&Math.abs(Number(mdValues[low-1])-target)<=Math.abs(Number(mdValues[low])-target))return low-1;return low;
  }
  function describeWellPropertyPoint(track,index){
    const axis=track?.verticalAxis||{},curve=track?.curve||{},mdValues=axis.values||[],values=curve.primaryValues||[],unit=String(curve.unit||''),target=String(curve.label||curve.target||'物性');
    if(index<0||index>=mdValues.length||index>=values.length)return `${target} · 横向按本井显示范围线性归一化，物理值未改变`;
    const lower=Array.isArray(curve.lowerValues)?Number(curve.lowerValues[index]):NaN,upper=Array.isArray(curve.upperValues)?Number(curve.upperValues[index]):NaN,interval=Number.isFinite(lower)&&Number.isFinite(upper)?` · 区间 ${formatWellPropertyValue(lower,unit)}–${formatWellPropertyValue(upper,unit)}`:'';
    return `MD ${formatMd(mdValues[index])} · ${target} ${formatWellPropertyValue(values[index],unit)}${interval}`;
  }
  function updateWellPropertyProbe(track,index){
    const axis=track?.verticalAxis||{},curve=track?.curve||{},mdValues=axis.values||[],values=curve.primaryValues||[],normalization=curve.displayNormalization||{},minimum=Number(normalization.minimum),maximum=Number(normalization.maximum),span=Math.max(maximum-minimum,1e-12),mdMinimum=Number(axis.minimum),mdMaximum=Number(axis.maximum),mdSpan=Math.max(mdMaximum-mdMinimum,1e-12);
    if(!mdValues.length||mdValues.length!==values.length)return;
    activeWellPropertyIndex=Math.max(0,Math.min(mdValues.length-1,Number(index)||0));const value=Number(values[activeWellPropertyIndex]),md=Number(mdValues[activeWellPropertyIndex]),x=Math.max(0,Math.min(200,(value-minimum)/span*200)),y=Math.max(0,Math.min(600,(md-mdMinimum)/mdSpan*600));
    if(wellResultPropertyProbe){wellResultPropertyProbe.value=String(activeWellPropertyIndex);wellResultPropertyProbe.setAttribute('aria-valuetext',describeWellPropertyPoint(track,activeWellPropertyIndex));}
    if(wellResultPropertyCursorLine){wellResultPropertyCursorLine.hidden=false;wellResultPropertyCursorLine.setAttribute('y1',y.toFixed(2));wellResultPropertyCursorLine.setAttribute('y2',y.toFixed(2));}
    if(wellResultPropertyCursor){wellResultPropertyCursor.hidden=false;wellResultPropertyCursor.setAttribute('cx',x.toFixed(2));wellResultPropertyCursor.setAttribute('cy',y.toFixed(2));}
    if(wellResultDetail)wellResultDetail.value=describeWellPropertyPoint(track,activeWellPropertyIndex);
  }
  function renderWellResultProperty(track){
    if(!wellResultPropertyTrack||!wellResultPropertySvg||!wellResultPropertyCurve||!wellResultPropertyProbe)return;
    const axis=track.verticalAxis||{},curve=track.curve||{},mdValues=Array.isArray(axis.values)?axis.values.map(Number):[],values=Array.isArray(curve.primaryValues)?curve.primaryValues.map(Number):[],normalization=curve.displayNormalization||{},minimum=Number(normalization.minimum),maximum=Number(normalization.maximum),span=Math.max(maximum-minimum,1e-12),mdMinimum=Number(axis.minimum),mdMaximum=Number(axis.maximum),mdSpan=Math.max(mdMaximum-mdMinimum,1e-12);
    const point=(value,md)=>`${Math.max(0,Math.min(200,(Number(value)-minimum)/span*200)).toFixed(2)},${Math.max(0,Math.min(600,(Number(md)-mdMinimum)/mdSpan*600)).toFixed(2)}`;
    wellResultPropertyCurve.setAttribute('points',mdValues.map((md,index)=>point(values[index],md)).join(' '));
    const lower=Array.isArray(curve.lowerValues)?curve.lowerValues.map(Number):[],upper=Array.isArray(curve.upperValues)?curve.upperValues.map(Number):[],hasBand=lower.length===mdValues.length&&upper.length===mdValues.length;
    if(wellResultPropertyBand){wellResultPropertyBand.hidden=!hasBand;wellResultPropertyBand.setAttribute('points',hasBand?[...mdValues.map((md,index)=>point(lower[index],md)),...mdValues.map((md,index)=>point(upper[upper.length-1-index],mdValues[mdValues.length-1-index]))].join(' '):'');}
    const unit=String(curve.unit||''),label=String(curve.label||curve.target||'物性');
    if(wellResultPropertyMinimum)wellResultPropertyMinimum.textContent=formatWellPropertyValue(Number(curve.valueMinimum),unit);
    if(wellResultPropertyMaximum)wellResultPropertyMaximum.textContent=formatWellPropertyValue(Number(curve.valueMaximum),unit);
    if(wellResultPropertyLabel)wellResultPropertyLabel.textContent=unit?`${label} / ${unit}`:label;
    wellResultPropertySvg.setAttribute('aria-label',`${label} 储层物性连续 MD 曲线；横向仅按本井范围归一化显示`);
    wellResultPropertyProbe.max=String(Math.max(0,mdValues.length-1));wellResultPropertyProbe.value=String(Math.max(0,Math.min(mdValues.length-1,activeWellPropertyIndex)));
    wellResultPropertyProbe.oninput=()=>updateWellPropertyProbe(track,Number(wellResultPropertyProbe.value));
    wellResultPropertySvg.onpointermove=event=>{const bounds=wellResultPropertySvg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientY-bounds.top)/Math.max(bounds.height,1))),targetMd=mdMinimum+fraction*mdSpan;updateWellPropertyProbe(track,nearestWellPropertyIndex(mdValues,targetMd));};
    updateWellPropertyProbe(track,activeWellPropertyIndex);
  }
  function linkedProjectedWellRecord(track){
    const index=linkedWellIndex(track);return index<0?null:projectedWellRecords.find(record=>record.wellIndex===index)||null;
  }
  function applyWellResultHudState(){
    if(!wellResultHud)return;
    if(faciesLogMode){state.wellResultHudExpanded=false;state.wellResultHudCollapsed=false;}
    wellResultHud.classList.toggle('expanded',state.wellResultHudExpanded);wellResultHud.classList.toggle('collapsed',state.wellResultHudCollapsed);
    root.classList.toggle('well-result-expanded',state.wellResultHudExpanded);
    if(wellResultSize){wellResultSize.hidden=faciesLogMode;wellResultSize.textContent=state.wellResultHudExpanded?'⤡':'⤢';wellResultSize.setAttribute('aria-label',state.wellResultHudExpanded?'还原结果面板':'放大结果面板');wellResultSize.setAttribute('aria-pressed',String(state.wellResultHudExpanded));}
    if(wellResultCollapse){wellResultCollapse.hidden=faciesLogMode;wellResultCollapse.textContent=state.wellResultHudCollapsed?'+':'−';wellResultCollapse.setAttribute('aria-label',state.wellResultHudCollapsed?'展开结果面板':'折叠结果面板');wellResultCollapse.setAttribute('aria-expanded',String(!state.wellResultHudCollapsed));}
  }
  function positionWellResultHud(){
    if(!wellResultHud||!wellResultTracks.length)return;
    if(state.predictionOnly){wellResultHud.hidden=true;if(wellResultLeader)wellResultLeader.hidden=true;return;}
    applyWellResultHudState();wellResultHud.hidden=false;
    const track=currentWellResultTrack(),record=track?linkedProjectedWellRecord(track):null;
    const rootWidth=root.clientWidth,rootHeight=root.clientHeight,hudWidth=wellResultHud.offsetWidth||340,hudHeight=wellResultHud.offsetHeight||520,compact=rootWidth<=(faciesLogMode?520:900);
    const minimumTop=compact?Math.max(10,canvas.clientHeight+10):14,maximumTop=Math.max(minimumTop,rootHeight-hudHeight-14);
    const left=compact?Math.max(14,(rootWidth-hudWidth)/2):faciesLogMode?14:Math.max(14,rootWidth-hudWidth-14),top=!compact&&faciesLogMode?14:Math.max(minimumTop,Math.min(maximumTop,state.wellResultHudTop));
    const anchor=record?.points?.[0]||null;
    if(!compact&&faciesLogMode){wellResultHud.style.left='14px';wellResultHud.style.right='auto';}else{wellResultHud.style.left=compact?`${left}px`:'auto';wellResultHud.style.right=compact?'auto':'14px';}wellResultHud.style.top=`${top}px`;
    if(!wellResultLeader||!wellResultLeaderPath||!wellResultAnchor||!anchor){if(wellResultLeader)wellResultLeader.hidden=true;return;}
    const anchorX=anchor[0]+canvas.offsetLeft,anchorY=anchor[1]+canvas.offsetTop;
    const targetX=compact?Math.max(left+28,Math.min(left+hudWidth-28,anchorX)):faciesLogMode?left+hudWidth:left,targetY=compact?top:Math.max(top+36,Math.min(top+hudHeight-30,anchorY));
    const elbowX=anchorX+(targetX-anchorX)*.55;
    wellResultLeader.hidden=false;wellResultLeader.setAttribute('viewBox',`0 0 ${Math.max(rootWidth,1)} ${Math.max(rootHeight,1)}`);
    wellResultLeaderPath.setAttribute('d',compact?`M ${anchorX.toFixed(1)} ${anchorY.toFixed(1)} L ${anchorX.toFixed(1)} ${(top-12).toFixed(1)} L ${targetX.toFixed(1)} ${targetY.toFixed(1)}`:`M ${anchorX.toFixed(1)} ${anchorY.toFixed(1)} L ${elbowX.toFixed(1)} ${targetY.toFixed(1)} L ${targetX.toFixed(1)} ${targetY.toFixed(1)}`);
    wellResultAnchor.setAttribute('cx',anchorX.toFixed(1));wellResultAnchor.setAttribute('cy',anchorY.toFixed(1));
  }
  function renderWellResultHud(){
    if(!wellResultHud||!wellResultTrackElement||!wellResultTracks.length)return;
    applyWellResultHudState();
    selectedWellResultTrackIndex=Math.max(0,Math.min(wellResultTracks.length-1,selectedWellResultTrackIndex));
    const track=currentWellResultTrack(),axis=track.verticalAxis||{},minimum=Number(axis.minimum),maximum=Number(axis.maximum),span=Math.max(maximum-minimum,1e-9),wellIndex=linkedWellIndex(track),well=wellIndex>=0?(payload.wells||[])[wellIndex]:null,projectable=verifiedMdToTwt(well),propertyTrack=track.kind==='property_curve';
    wellResultHud.hidden=false;wellResultHud.classList.toggle('md-only',!projectable);wellResultHud.classList.toggle('property-result',propertyTrack);wellResultHud.classList.toggle('facies-result',faciesLogMode);
    wellResultTitle.textContent=`${track.wellId} · ${track.subject||'井侧分类'}`;
    let linkStateText='';
    if(wellIndex<0)linkStateText='未匹配到唯一井轨迹 · MD-only';
    else if(projectable)linkStateText=propertyTrack?'对应井已核验 MD → TWT · 物性曲线可联动井轨迹':'对应井已核验 MD → TWT · 分类段可联动井轨迹';
    wellResultLinkState.textContent=linkStateText;wellResultLinkState.hidden=!linkStateText;
    renderWellResultDepthAxis(minimum,maximum);
    wellResultTrackElement.hidden=propertyTrack;if(wellResultPropertyTrack)wellResultPropertyTrack.hidden=!propertyTrack;if(wellResultLegend)wellResultLegend.hidden=propertyTrack;wellResultTrackElement.replaceChildren();
    if(propertyTrack){renderWellResultProperty(track);}else{
      (track.intervals||[]).forEach((interval,index)=>{
        const segment=document.createElement('button'),top=(Number(interval.topMdM)-minimum)/span*100,height=Math.max(.3,(Number(interval.bottomMdM)-Number(interval.topMdM))/span*100),description=describeWellResultInterval(interval);
        segment.type='button';segment.className='well-result-segment';segment.dataset.intervalIndex=String(index);segment.style.top=`${Math.max(0,Math.min(100,top))}%`;segment.style.height=`${Math.max(.3,Math.min(100-top,height))}%`;segment.style.background=String(interval.color||'#718096');segment.setAttribute('role','option');segment.setAttribute('aria-label',description);segment.title=description;
        const enter=()=>{hoveredWellResultIntervalIndex=index;if(wellResultDetail)wellResultDetail.value=description;updateWellResultButtons();drawGuides();};
        const leave=()=>{hoveredWellResultIntervalIndex=-1;if(wellResultDetail)wellResultDetail.value=describeWellResultInterval(activeWellResultInterval());updateWellResultButtons();drawGuides();};
        segment.addEventListener('pointerenter',enter);segment.addEventListener('pointerleave',leave);segment.addEventListener('focus',enter);segment.addEventListener('blur',leave);
        segment.addEventListener('click',event=>{event.stopPropagation();activeWellResultIntervalIndex=activeWellResultIntervalIndex===index?-1:index;if(wellResultDetail)wellResultDetail.value=describeWellResultInterval(activeWellResultInterval());updateWellResultButtons();drawGuides();});
        wellResultTrackElement.append(segment);
      });
      if(wellResultLegend){wellResultLegend.replaceChildren();const seen=new Set();for(const interval of track.intervals||[]){const identity=`${interval.code}:${interval.label}`;if(seen.has(identity)||seen.size>=8)continue;seen.add(identity);const item=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=String(interval.color||'#718096');item.append(swatch,document.createTextNode(String(interval.label)));wellResultLegend.append(item);}}
      if(wellResultDetail)wellResultDetail.value=describeWellResultInterval(activeWellResultInterval());
    }
    if(wellResultPrev&&wellResultNext){const multiple=wellResultTracks.length>1;wellResultPrev.hidden=!multiple;wellResultNext.hidden=!multiple;}
    updateWellResultButtons();positionWellResultHud();requestAnimationFrame(positionWellResultHud);
  }
  function selectWellResultTrack(index){
    if(!wellResultTracks.length)return;selectedWellResultTrackIndex=(index+wellResultTracks.length)%wellResultTracks.length;activeWellResultIntervalIndex=-1;hoveredWellResultIntervalIndex=-1;activeWellPropertyIndex=0;renderWellResultHud();drawGuides();
  }
  function distanceToSegment(px,py,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;if(length<=1e-9)return Math.hypot(px-a[0],py-a[1]);const t=Math.max(0,Math.min(1,((px-a[0])*dx+(py-a[1])*dy)/length));return Math.hypot(px-(a[0]+t*dx),py-(a[1]+t*dy));}
  function pickProjectedWell(clientX,clientY){
    const bounds=canvas.getBoundingClientRect(),x=clientX-bounds.left,y=clientY-bounds.top;
    return projectedWellRecords.map(record=>({record,distance:record.points.length===1?Math.hypot(x-record.points[0][0],y-record.points[0][1]):record.points.slice(1).reduce((minimum,point,index)=>Math.min(minimum,distanceToSegment(x,y,record.points[index],point)),Infinity)})).filter(item=>item.distance<=10).sort((a,b)=>a.distance-b.distance)[0]?.record||null;
  }
  function selectProjectedWell(record){
    const keys=wellIdentityKeys(record?.well),current=currentWellResultTrack(),currentKey=trackWellKey(current);
    const index=wellResultTracks.findIndex(track=>keys.has(trackWellKey(track)));
    if(index>=0&&(!currentKey||!keys.has(currentKey)||index!==selectedWellResultTrackIndex))selectWellResultTrack(index);
  }
  function gridBlockCorners(block){
    const start=(block.sourceStartZYX||[]).map(Number),end=(block.sourceEndZYXExclusive||[]).map(Number);
    if(start.length!==3||end.length!==3||sourceShape.length!==3)return [];
    const xBounds=[start[2],end[2]].map(value=>(value/sourceShape[2]-.5)*boxScale[0]);
    const yBounds=[start[1],end[1]].map(value=>(value/sourceShape[1]-.5)*boxScale[1]);
    const zBounds=[start[0],end[0]].map(value=>displayZ(value/sourceShape[0]));
    const corners=[];for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)corners.push(projectWithDepth([xBounds[x],yBounds[y],zBounds[z]]));
    return corners.every(Boolean)?corners:[];
  }
  function convexHull(points){
    const sorted=[...points].sort((a,b)=>a.x-b.x||a.y-b.y);if(sorted.length<=2)return sorted;
    const cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x),lower=[],upper=[];
    sorted.forEach(point=>{while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),point)<=0)lower.pop();lower.push(point);});
    [...sorted].reverse().forEach(point=>{while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),point)<=0)upper.pop();upper.push(point);});
    lower.pop();upper.pop();return lower.concat(upper);
  }
  function tracePolygon(context,points){context.beginPath();points.forEach((point,index)=>index?context.lineTo(point.x,point.y):context.moveTo(point.x,point.y));context.closePath();}
  function gridRange(block,axis){const value=block.axisCoordinateRanges?.[axis]||{},start=value.coordinate_start??value.index_start??'—',end=value.coordinate_end??value.index_end_inclusive??'—';return `${start}–${end}`;}
  function blockSummary(block){return `${block.blockId} · Z ${gridRange(block,'Z')} · I ${gridRange(block,'INLINE')} · X ${gridRange(block,'CROSSLINE')} · 有效道 ${(Number(block.validTraceRatio)*100).toFixed(1)}% · 断层 ${(Number(block.faultFraction)*100).toFixed(2)}%`;}
  function drawFaultGrid(context){
    if(!gridOverview||gridBlocks.length!==gridBlockCount){projectedGridBlocks=[];return;}
    const faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]],edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    const records=gridBlocks.map((block,index)=>{const corners=gridBlockCorners(block);if(!corners.length)return null;return {block,index,corners,polygon:convexHull(corners),depth:corners.reduce((sum,point)=>sum+point.depth,0)/corners.length};}).filter(Boolean).sort((a,b)=>a.depth-b.depth);
    records.forEach(record=>{
      const layer=Math.max(0,Number(record.block.gridIndexZYX?.[0]||0)),color=gridLayerColors[layer%gridLayerColors.length],active=record.index===state.hoveredBlockIndex||record.index===state.selectedBlockIndex;
      const orderedFaces=faces.map(face=>({points:face.map(index=>record.corners[index]),depth:face.reduce((sum,index)=>sum+record.corners[index].depth,0)/face.length})).sort((a,b)=>a.depth-b.depth);
      context.fillStyle=`rgba(${color.join(',')},${active ? .24 : .095})`;orderedFaces.forEach(face=>{tracePolygon(context,face.points);context.fill();});
      context.strokeStyle=active?`rgba(${color.join(',')},.98)`:`rgba(${color.join(',')},.62)`;context.lineWidth=active?2.2:1;context.beginPath();edges.forEach(([a,b])=>{context.moveTo(record.corners[a].x,record.corners[a].y);context.lineTo(record.corners[b].x,record.corners[b].y);});context.stroke();
      if(active){const center=record.corners.reduce((sum,point)=>[sum[0]+point.x,sum[1]+point.y],[0,0]).map(value=>value/record.corners.length);context.font='700 12px sans-serif';context.lineJoin='round';context.strokeStyle='rgba(255,255,255,.98)';context.lineWidth=4;context.strokeText(record.block.blockId,center[0]+5,center[1]-5);context.fillStyle=`rgb(${color.join(',')})`;context.fillText(record.block.blockId,center[0]+5,center[1]-5);}
    });
    projectedGridBlocks=records;
  }
  function pointInPolygon(x,y,polygon){let inside=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j],crosses=(a.y>y)!==(b.y>y)&&x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x;if(crosses)inside=!inside;}return inside;}
  function pickGridBlock(clientX,clientY){
    if(!gridOverview)return null;const bounds=canvas.getBoundingClientRect(),x=clientX-bounds.left,y=clientY-bounds.top;
    return projectedGridBlocks.filter(record=>pointInPolygon(x,y,record.polygon)).sort((a,b)=>b.depth-a.depth)[0]||null;
  }
  function showGridBlock(record,event){
    state.hoveredBlockIndex=record?record.index:-1;canvas.style.cursor=record?'pointer':'grab';
    if(gridTooltip){gridTooltip.hidden=!record;if(record){gridTooltip.textContent=blockSummary(record.block);if(event){const bounds=root.getBoundingClientRect();gridTooltip.style.left=`${Math.min(bounds.width-290,Math.max(12,event.clientX-bounds.left+12))}px`;gridTooltip.style.top=`${Math.min(bounds.height-62,Math.max(12,event.clientY-bounds.top+12))}px`;}}}
    drawGuides();
  }
  function activateGridBlock(block){const target=String(block?.selectionHref||'');if(gridOverview&&target.startsWith('/统一数据可视化?'))window.location.assign(target);}
  function drawGuides(){
    const context=wireCanvas.getContext('2d');context.clearRect(0,0,wireCanvas.clientWidth,wireCanvas.clientHeight);context.lineWidth=1;context.strokeStyle='rgba(51,72,92,.48)';
    const corners=[];for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)corners.push(project([(x-.5)*boxScale[0],(y-.5)*boxScale[1],(z-.5)*boxScale[2]]));
    drawFaultGrid(context);
    const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];context.beginPath();edges.forEach(([a,b])=>{if(corners[a]&&corners[b]){context.moveTo(...corners[a]);context.lineTo(...corners[b]);}});context.stroke();
    const topCorner=project([-.5*boxScale[0],-.5*boxScale[1],.5*boxScale[2]]),bottomCorner=project([-.5*boxScale[0],-.5*boxScale[1],-.5*boxScale[2]]),axisUnit=verticalAxis.unit||'';context.font='12px sans-serif';context.fillStyle='#536b80';if(topCorner)context.fillText(`地震顶面 · ${verticalAxis.top??'—'} ${axisUnit}`,topCorner[0]+5,topCorner[1]-6);if(bottomCorner)context.fillText(`地震底面 · ${verticalAxis.bottom??'—'} ${axisUnit}`,bottomCorner[0]+5,bottomCorner[1]+13);
    const wellColors={vertical:'#1677e8',deviated:'#e6a23c',horizontal:'#d23b63'};
    projectedWellRecords=[];
    if(!state.predictionOnly){
    (payload.surfaces||[]).forEach(surface=>{const points=(surface.points||[]).map(point=>project([((Number(point[1])/(shape[2]-1||1))-.5)*boxScale[0],((Number(point[0])/(shape[1]-1||1))-.5)*boxScale[1],displayZ(Number(point[2])/(shape[0]-1||1))])).filter(Boolean);if(!points.length)return;context.fillStyle=surface.color||'#ef4444';const radius=surface.kind==='points'?2.7:1.7;points.forEach(point=>{context.beginPath();context.arc(point[0],point[1],radius,0,Math.PI*2);context.fill();});});
    const selectedTrack=currentWellResultTrack(),selectedLinkedWellIndex=selectedTrack?linkedWellIndex(selectedTrack):-1,selectedInterval=activeWellResultInterval();
    (payload.wells||[]).forEach((well,wellIndex)=>{
      const visual=wellVisualGeometry(well),pointRecords=visual.stations.map(station=>({sourceIndex:station.index,point:project([(station.x-.5)*boxScale[0],(station.y-.5)*boxScale[1],displayZ(station.z)])})).filter(record=>record.point),points=pointRecords.map(record=>record.point);
      if(!points.length)return;
      projectedWellRecords.push({well,wellIndex,points,displayMode:visual.displayMode});
      const accepted=visual.displayMode==='verified_twt',timeLocated=visual.displayMode==='time_candidate',referenceLocated=['depth_reference','unverified_z_reference','md_reference'].includes(visual.displayMode);
      const selected=wellIndex===selectedLinkedWellIndex,emphasized=selected||wellIndex===hoveredProjectedWellIndex;
      const color=accepted?(wellColors[well.geometryType]||'#00695c'):timeLocated?'#d97706':referenceLocated?'#39718c':'#697785';
      const pathWidth=((accepted||timeLocated)?(well.geometryType==='horizontal'?4.2:3.2):referenceLocated?3:2.6)+(emphasized?1.4:0);
      const dash=accepted?[]:timeLocated?[9,4]:referenceLocated?[4,4]:[6,4];
      const tracePath=()=>{context.beginPath();points.forEach((p,i)=>i?context.lineTo(...p):context.moveTo(...p));};
      // A neutral halo keeps every real trajectory legible over red/blue
      // seismic amplitudes without changing its scientific-state colour.
      if(points.length>1){context.lineJoin='round';context.lineCap='round';context.setLineDash(dash);
        context.strokeStyle='rgba(255,255,255,.94)';context.lineWidth=pathWidth+3.2;tracePath();context.stroke();
        context.strokeStyle=color;context.lineWidth=pathWidth;tracePath();context.stroke();context.setLineDash([]);}
      if(selected&&selectedInterval&&verifiedMdToTwt(well)){
        const md=resultWellMd(well),intervalPoints=pointRecords.filter(record=>md[record.sourceIndex]>=Number(selectedInterval.topMdM)-1e-6&&md[record.sourceIndex]<=Number(selectedInterval.bottomMdM)+1e-6).map(record=>record.point);
        if(intervalPoints.length>=2){const intervalPath=()=>{context.beginPath();intervalPoints.forEach((point,index)=>index?context.lineTo(...point):context.moveTo(...point));};context.strokeStyle='rgba(255,255,255,.97)';context.lineWidth=8;intervalPath();context.stroke();context.strokeStyle=String(selectedInterval.color||'#16866f');context.lineWidth=5;intervalPath();context.stroke();}
      }
      (points.length===1?[points[0]]:[points[0],points[points.length-1]]).forEach((point,index)=>{context.beginPath();context.fillStyle=index===0?'#ffffff':color;context.strokeStyle=color;context.lineWidth=2;context.arc(point[0],point[1],points.length===1?5.4:index===0?4.6:3.8,0,Math.PI*2);context.fill();context.stroke();});
      const planGuides=Array.isArray(well.planGuides)&&well.planGuides.length?well.planGuides:(well.topGuide?[well.topGuide]:[]);
      if(timeLocated)planGuides.forEach(guide=>{if(!Array.isArray(guide.x)||!Array.isArray(guide.y)||!Array.isArray(guide.z))return;const guidePoints=guide.x.map((x,i)=>project([(x-.5)*boxScale[0],((guide.y||[])[i]-.5)*boxScale[1],displayZ((guide.z||[])[i])])).filter(Boolean);if(guidePoints.length>1){context.strokeStyle='rgba(52,93,132,.55)';context.setLineDash([3,4]);context.beginPath();guidePoints.forEach((point,i)=>i?context.lineTo(...point):context.moveTo(...point));context.stroke();context.setLineDash([]);}});
      if(timeLocated&&Array.isArray(well.zLow)&&well.zLow.length===well.z.length&&Array.isArray(well.zHigh)&&well.zHigh.length===well.z.length){context.strokeStyle='rgba(22,119,232,.24)';const stride=Math.max(1,Math.floor(well.z.length/8));for(let i=0;i<well.z.length;i+=stride){const low=project([((well.x||[])[i]-.5)*boxScale[0],((well.y||[])[i]-.5)*boxScale[1],displayZ(well.zLow[i])]),high=project([((well.x||[])[i]-.5)*boxScale[0],((well.y||[])[i]-.5)*boxScale[1],displayZ(well.zHigh[i])]);if(low&&high){context.beginPath();context.moveTo(...low);context.lineTo(...high);context.stroke();}}}
      const derivedDisplay=well.displayOnlyGeometry===true||well.visualizationOnly===true||well.horizontalAlignment?.placementDerived===true;
      const wellName=well.name||'井',geometryLabel=well.geometryLabel||'';
      const label=derivedDisplay?wellName:wellName+(geometryLabel?' · '+geometryLabel:'')+(accepted?' · TWT':'');
      const labelX=points[0][0]+8,labelY=points[0][1]-8;
      context.font=`${emphasized?'700':'600'} 12px sans-serif`;context.lineJoin='round';context.strokeStyle='rgba(255,255,255,.98)';context.lineWidth=4.5;context.strokeText(label,labelX,labelY);context.fillStyle=color;context.fillText(label,labelX,labelY);
    });
    }
    positionWellResultHud();
  }
  function resize(){
    const ratio=Math.min(window.devicePixelRatio||1,2),width=Math.max(1,Math.round(canvas.clientWidth*ratio)),height=Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;wireCanvas.width=width;wireCanvas.height=height;wireCanvas.getContext('2d').setTransform(ratio,0,0,ratio,0,0);}render();
  }
  let dragging=false,lastX=0,lastY=0,pointerStartX=0,pointerStartY=0,dragDistance=0;
  canvas.addEventListener('pointerdown',event=>{dragging=true;lastX=pointerStartX=event.clientX;lastY=pointerStartY=event.clientY;dragDistance=0;canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{if(!dragging){if(gridOverview){showGridBlock(pickGridBlock(event.clientX,event.clientY),event);return;}const record=pickProjectedWell(event.clientX,event.clientY),nextIndex=record?record.wellIndex:-1;if(nextIndex!==hoveredProjectedWellIndex){hoveredProjectedWellIndex=nextIndex;drawGuides();}canvas.style.cursor=record?'pointer':'grab';return;}const dx=event.clientX-lastX,dy=event.clientY-lastY;dragDistance+=Math.hypot(dx,dy);state.yaw+=dx*.007;state.pitch=Math.max(-1.25,Math.min(1.25,state.pitch+dy*.007));lastX=event.clientX;lastY=event.clientY;render();});
  canvas.addEventListener('pointerup',event=>{const clicked=dragDistance<=6&&Math.hypot(event.clientX-pointerStartX,event.clientY-pointerStartY)<=6,record=clicked&&gridOverview?pickGridBlock(event.clientX,event.clientY):null,wellRecord=clicked&&!gridOverview?pickProjectedWell(event.clientX,event.clientY):null;dragging=false;saveViewState();if(record){state.selectedBlockIndex=record.index;showGridBlock(record,event);activateGridBlock(record.block);}else if(wellRecord)selectProjectedWell(wellRecord);});
  canvas.addEventListener('pointercancel',()=>{dragging=false;saveViewState();});
  canvas.addEventListener('pointerleave',()=>{if(gridOverview&&!dragging)showGridBlock(null,null);else if(!dragging&&hoveredProjectedWellIndex>=0){hoveredProjectedWellIndex=-1;canvas.style.cursor='grab';drawGuides();}});
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();openGridFallback(`WebGL 上下文已丢失，已展开下方二维${gridBlockCount}块列表`);});
  canvas.addEventListener('keydown',event=>{if(!gridOverview||!gridBlocks.length)return;let next=state.selectedBlockIndex,handled=true;if(event.key==='ArrowRight')next++;else if(event.key==='ArrowLeft')next--;else if(event.key==='ArrowDown')next+=gridRowWidth;else if(event.key==='ArrowUp')next-=gridRowWidth;else if(event.key==='PageDown')next+=gridPlaneSize;else if(event.key==='PageUp')next-=gridPlaneSize;else if(event.key==='Home')next=0;else if(event.key==='End')next=gridBlocks.length-1;else if(event.key==='Enter'||event.key===' '){activateGridBlock(gridBlocks[state.selectedBlockIndex]);}else handled=false;if(!handled)return;event.preventDefault();state.selectedBlockIndex=Math.max(0,Math.min(gridBlocks.length-1,next));status.textContent='已选择 '+blockSummary(gridBlocks[state.selectedBlockIndex])+' · Enter / Space 打开完整128³块';drawGuides();});
  canvas.addEventListener('wheel',event=>{event.preventDefault();const unit=event.deltaMode===1?16:event.deltaMode===2?Math.max(canvas.clientHeight,1):1,delta=Math.max(-240,Math.min(240,event.deltaY*unit));state.zoom=Math.max(.72,Math.min(6,state.zoom*Math.exp(delta*.0012)));saveViewState();render();},{passive:false});
  [gain,opacity,threshold].forEach(control=>control.addEventListener('input',render));
  overlayThreshold.addEventListener('input',()=>{updateLayerStatus();render();});
  [maskSlabAxis,maskSlabCenter,maskSlabThickness].forEach(control=>control.addEventListener('input',()=>{updateLayerStatus();render();}));
  if(wellResultPrev)wellResultPrev.addEventListener('click',()=>selectWellResultTrack(selectedWellResultTrackIndex-1));
  if(wellResultNext)wellResultNext.addEventListener('click',()=>selectWellResultTrack(selectedWellResultTrackIndex+1));
  if(wellResultSize)wellResultSize.addEventListener('click',event=>{event.stopPropagation();state.wellResultHudExpanded=!state.wellResultHudExpanded;applyWellResultHudState();saveViewState();requestAnimationFrame(()=>{resize();positionWellResultHud();});});
  if(wellResultCollapse)wellResultCollapse.addEventListener('click',event=>{event.stopPropagation();state.wellResultHudCollapsed=!state.wellResultHudCollapsed;applyWellResultHudState();saveViewState();requestAnimationFrame(()=>{resize();positionWellResultHud();});});
  let draggingWellResultHud=false,wellResultHudDragOffsetY=0,wellResultHudPointerId=-1;
  function finishWellResultHudDrag(){if(!draggingWellResultHud)return;draggingWellResultHud=false;wellResultHud?.classList.remove('dragging');saveViewState();}
  if(wellResultHeader){
    wellResultHeader.addEventListener('pointerdown',event=>{if(faciesLogMode||event.target?.closest?.('button'))return;event.preventDefault();const rootBounds=root.getBoundingClientRect(),hudBounds=wellResultHud.getBoundingClientRect();draggingWellResultHud=true;wellResultHudPointerId=event.pointerId;wellResultHudDragOffsetY=event.clientY-hudBounds.top;state.wellResultHudTop=hudBounds.top-rootBounds.top;wellResultHud.classList.add('dragging');wellResultHeader.setPointerCapture(event.pointerId);});
    wellResultHeader.addEventListener('pointermove',event=>{if(!draggingWellResultHud||event.pointerId!==wellResultHudPointerId)return;const rootBounds=root.getBoundingClientRect();state.wellResultHudTop=event.clientY-rootBounds.top-wellResultHudDragOffsetY;positionWellResultHud();});
    wellResultHeader.addEventListener('pointerup',event=>{if(event.pointerId!==wellResultHudPointerId)return;finishWellResultHudDrag();});
    wellResultHeader.addEventListener('pointercancel',finishWellResultHudDrag);
  }
  function reset(){state.yaw=-.66;state.pitch=-.38;state.zoom=1.72;saveViewState();render();}
  document.getElementById('volume-reset').addEventListener('click',reset);window.resetWholeVolumeView=reset;
  window.setWholeVolumeLayerMode=(mode)=>{state.predictionOnly=Boolean(payload.overlay)&&mode==='prediction';baseTools.forEach(tool=>tool.hidden=state.predictionOnly);updateLayerStatus();render();};
  if(window.ResizeObserver)new ResizeObserver(resize).observe(root);window.addEventListener('resize',resize);
  renderWellResultHud();
  try{initialize();resize();}catch(error){const message='整体三维初始化失败：'+(error instanceof Error?error.message:String(error));if(!openGridFallback(message+`；已展开下方二维${gridBlockCount}块列表`)){status.textContent=message;root.classList.add('unsupported');}}
})();
</script>
"""
    return (
        template.replace("__PAYLOAD__", payload_json)
        .replace(
            "__GRID_ROOT_CLASS__",
            (" fault-grid-overview-volume" if grid_overview else "")
            + (" has-well-results" if payload["wellResultTracks"] else "")
            + (" facies-log-layout" if facies_log_mode else ""),
        )
        .replace(
            "__GRID_ROOT_ATTRS__",
            f' data-testid="fault-grid-3d-overview" data-block-count="{grid_block_count}"'
            if grid_overview
            else "",
        )
        .replace(
            "__CANVAS_ARIA_LABEL__",
            f"完整工区三维地震预览与{grid_block_count}个可选择代表块"
            if grid_overview
            else "整体三维地震体",
        )
        .replace(
            "__CANVAS_KEYBOARD_ATTRS__",
            ' tabindex="0" data-testid="fault-grid-3d-canvas" '
            'aria-describedby="fault-grid-3d-help"'
            if grid_overview
            else ' tabindex="0"' if payload["wellResultTracks"] else "",
        )
        .replace(
            "__OVERLAY_THRESHOLD_LABEL__",
            html.escape(str(overlay_ui["threshold_label"])),
        )
        .replace(
            "__OVERLAY_LEGEND_LABEL__",
            html.escape(
                f'{overlay_ui["display_name"]} · '
                + (
                    "区块占据率三线性阈值面（显示派生）"
                    if overlay_is_occupancy_mask
                    else "二值 0/1（NEAREST）"
                    if overlay_payload is not None
                    and str(overlay_payload.get("kind", "")).casefold() == "mask"
                    else "jet"
                )
            ),
        )
        .replace("__SURFACE_LEGEND__", surface_legend)
        .replace("__CANDIDATE_LEGEND__", candidate_legend)
        .replace("__GRID_LEGEND__", grid_legend)
        .replace("__WELL_RESULT_HUD__", well_result_hud)
    )


def _crossline_profile_single_gap_connection(
    profiles: Sequence[Sequence[Any]],
    *,
    surface_index: int,
    missing_index: int,
) -> bool:
    """Return whether one missing Crossline profile point is safe to bridge.

    This mirrors the browser's display-only polyline rule.  It never fills a
    grid cell: the target must be one interior missing point bracketed by
    immediate finite neighbours, and the linearly interpolated values of all
    rendered horizons must remain strictly ordered at that position.
    """

    if not profiles or not 0 <= surface_index < len(profiles):
        return False
    width = len(profiles[surface_index])
    if missing_index <= 0 or missing_index >= width - 1:
        return False
    if any(len(profile) != width for profile in profiles):
        return False

    def finite(value: Any) -> float | None:
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    if finite(profiles[surface_index][missing_index]) is not None:
        return False
    ordered_values: list[float] = []
    for profile in profiles:
        current = finite(profile[missing_index])
        if current is None:
            left = finite(profile[missing_index - 1])
            right = finite(profile[missing_index + 1])
            if left is None or right is None:
                return False
            current = (left + right) / 2.0
        ordered_values.append(current)
    return all(
        lower > upper
        for upper, lower in zip(ordered_values, ordered_values[1:])
    )


def _orthogonal_display_span_zyx(
    volume_payload: dict[str, Any], preview_shape_zyx: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Return the source-sample grid span represented by a bounded preview.

    Sparse previews may use a different stride on each axis.  Their array
    shape therefore cannot be used as the displayed seismic aspect ratio.
    Prefer the exact crop span, otherwise reconstruct it from the sealed
    sample indices and clamp the final half-open cell to the source shape.
    Payloads without sparse-sampling metadata retain their existing pixel
    geometry.  This is an array-grid aspect contract, not a claim that seismic
    sample indices, Inline numbers, and Crossline numbers share physical units.
    """

    def positive_triplet(value: Any) -> tuple[int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        if any(isinstance(item, (bool, np.bool_)) for item in value):
            return None
        try:
            numeric = tuple(float(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        if any(not np.isfinite(item) or not item.is_integer() for item in numeric):
            return None
        try:
            triplet = tuple(int(item) for item in numeric)
        except (TypeError, ValueError, OverflowError):
            return None
        if any(item <= 0 for item in triplet):
            return None
        return triplet  # type: ignore[return-value]

    sampling = volume_payload.get("sampling")
    if not isinstance(sampling, dict):
        return preview_shape_zyx
    crop_size = positive_triplet(sampling.get("cropSizeZYX"))
    if crop_size is not None:
        return crop_size

    raw_indices = sampling.get("sampleIndicesZYX")
    if not isinstance(raw_indices, dict):
        return preview_shape_zyx
    source_shape = positive_triplet(sampling.get("sourceShapeZYX"))
    axis_names = ("z", "inline", "crossline")
    spans: list[int] = []
    for axis, (axis_name, preview_count) in enumerate(
        zip(axis_names, preview_shape_zyx, strict=True)
    ):
        values = raw_indices.get(axis_name)
        if not isinstance(values, list) or len(values) != preview_count:
            return preview_shape_zyx
        try:
            positions = np.asarray(values, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            return preview_shape_zyx
        if (
            positions.ndim != 1
            or positions.size == 0
            or np.any(positions < 0)
            or (positions.size > 1 and np.any(np.diff(positions) <= 0))
        ):
            return preview_shape_zyx
        if positions.size > 1:
            step = max(1, int(round(float(np.median(np.diff(positions))))))
        else:
            step = 1
        span = int(positions[-1] - positions[0]) + step
        if source_shape is not None:
            available = source_shape[axis] - int(positions[0])
            if available <= 0:
                return preview_shape_zyx
            span = min(span, available)
        spans.append(max(1, span))
    return tuple(spans)  # type: ignore[return-value]


def _orthogonal_slice_fragment(volume_payload: dict[str, Any]) -> str:
    """Render robust 2D orthogonal slices from the bounded browser payload.

    No additional source-file read or server-side scene state is involved.  A
    horizon product opens as a true Inline/Xline plan map; ordinary volumes
    open on the middle time/depth slice.  Categorical overlays use a discrete
    palette instead of the misleading continuous jet ramp.
    """
    cube = dict(volume_payload.get("cube", {}))
    shape = [int(value) for value in cube.get("shape", [])]
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"无效的二维切片预览形状：{shape}")
    nz, ni, nx = shape
    for axis_name, expected in (
        ("timeValues", nz),
        ("inlineValues", ni),
        ("crosslineValues", nx),
    ):
        values = list(volume_payload.get(axis_name, []))
        if values and len(values) != expected:
            raise ValueError(
                f"二维切片轴 {axis_name} 长度 {len(values)} 与数据形状 {expected} 不一致"
            )
    has_surface_grid = False
    surface_grid_count = 0
    derived_surface_grids: list[tuple[int, list[Any]]] = []
    for surface_index, surface in enumerate(volume_payload.get("surfaces", [])):
        if not isinstance(surface, dict) or "grid" not in surface:
            continue
        has_surface_grid = True
        surface_grid_count += 1
        grid = surface.get("grid")
        if not isinstance(grid, list) or len(grid) != ni:
            raise ValueError(
                f"第 {surface_index + 1} 个层位二维网格 Inline 维度不等于 {ni}"
            )
        for row in grid:
            if not isinstance(row, list) or len(row) != nx:
                raise ValueError(
                    f"第 {surface_index + 1} 个层位二维网格 Crossline 维度不等于 {nx}"
                )
            if any(
                value is not None and not np.isfinite(float(value)) for value in row
            ):
                raise ValueError(f"第 {surface_index + 1} 个层位二维网格含非有限值")
        has_cross_inline_grid = "crossInlineDisplayGrid" in surface
        has_short_gap_bridge = "shortGapBridgeMaxMissingPreviewCells" in surface
        has_cross_inline_correction = (
            "crossInlineMaxCorrectionPreviewCells" in surface
        )
        if len(
            {
                has_cross_inline_grid,
                has_short_gap_bridge,
                has_cross_inline_correction,
            }
        ) != 1:
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 显示修复合同不完整"
            )
        if not has_cross_inline_grid:
            continue
        cross_inline_grid = surface.get("crossInlineDisplayGrid")
        if not isinstance(cross_inline_grid, list) or len(cross_inline_grid) != ni:
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 显示网格 Inline 维度不等于 {ni}"
            )
        for row in cross_inline_grid:
            if not isinstance(row, list) or len(row) != nx:
                raise ValueError(
                    f"第 {surface_index + 1} 个层位 Crossline 显示网格 Crossline 维度不等于 {nx}"
                )
            if any(
                value is not None and not np.isfinite(float(value)) for value in row
            ):
                raise ValueError(
                    f"第 {surface_index + 1} 个层位 Crossline 显示网格含非有限值"
                )
        raw_bridge_limit = surface.get("shortGapBridgeMaxMissingPreviewCells")
        if isinstance(raw_bridge_limit, (bool, np.bool_)):
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 单缺口显示桥接上限必须为 1"
            )
        try:
            bridge_limit = int(raw_bridge_limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 单缺口显示桥接上限必须为 1"
            ) from exc
        if (
            bridge_limit
            != _CROSSLINE_DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS
            or bridge_limit != raw_bridge_limit
        ):
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 单缺口显示桥接上限必须为 1"
            )
        raw_correction = surface.get("crossInlineMaxCorrectionPreviewCells")
        if isinstance(raw_correction, (bool, np.bool_)):
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 最大显示校正必须为有限非负数"
            )
        try:
            max_correction = float(raw_correction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 最大显示校正必须为有限非负数"
            ) from exc
        if not np.isfinite(max_correction) or max_correction < 0.0:
            raise ValueError(
                f"第 {surface_index + 1} 个层位 Crossline 最大显示校正必须为有限非负数"
            )
        for raw_row, display_row in zip(grid, cross_inline_grid, strict=True):
            for raw_value, display_value in zip(
                raw_row, display_row, strict=True
            ):
                raw_missing = raw_value is None
                display_missing = display_value is None
                if raw_missing != display_missing:
                    raise ValueError(
                        f"第 {surface_index + 1} 个层位 Crossline 显示修复改变了原始有限支持"
                    )
                if raw_missing:
                    continue
                if abs(float(display_value) - float(raw_value)) > max_correction + 1e-9:
                    raise ValueError(
                        f"第 {surface_index + 1} 个层位 Crossline 显示校正超过声明上限"
                    )
        derived_surface_grids.append((surface_index, cross_inline_grid))
    if derived_surface_grids and len(derived_surface_grids) != surface_grid_count:
        raise ValueError(
            "Crossline 显示修复合同必须由所有可显示层位同时提供，禁止混合原始/派生层位"
        )
    for (upper_index, upper), (lower_index, lower) in zip(
        derived_surface_grids, derived_surface_grids[1:]
    ):
        for upper_row, lower_row in zip(upper, lower, strict=True):
            for upper_value, lower_value in zip(
                upper_row, lower_row, strict=True
            ):
                if upper_value is None or lower_value is None:
                    continue
                if float(lower_value) <= float(upper_value):
                    raise ValueError(
                        "Crossline 显示修复后的层位发生交叉或相接："
                        f"第 {upper_index + 1}/{lower_index + 1} 个层位"
                    )
    time_values = list(volume_payload.get("timeValues", []))
    if has_surface_grid and len(time_values) != nz:
        raise ValueError("层位二维平面缺少与Z采样严格对应的时间轴")
    if has_surface_grid and (
        not np.all(np.isfinite(np.asarray(time_values, dtype=float)))
        or np.any(np.diff(np.asarray(time_values, dtype=float)) <= 0)
    ):
        raise ValueError("层位二维平面的时间轴必须有限且严格递增")

    source_span_zyx = _orthogonal_display_span_zyx(
        volume_payload, (nz, ni, nx)
    )

    slice_contract = volume_payload.get("sliceViewContract")
    surface_seg_global = (
        isinstance(slice_contract, dict)
        and slice_contract.get("globalConsistent") is True
        and str(slice_contract.get("modelId") or "") == "seismic_surface_seg"
    )
    if surface_seg_global:
        toolbar_fragment = r"""
  <header class="orthogonal-toolbar orthogonal-toolbar--surface-global">
    <nav aria-label="二维层位解释视图" id="orthogonal-plane-buttons">
      <button type="button" data-plane="i" data-mode-label="Inline 层位解释">Inline 层位解释</button>
      <details class="orthogonal-qc-menu" id="orthogonal-qc-views">
        <summary><span id="orthogonal-qc-summary-label">QC 与派生视图</span></summary>
        <div class="orthogonal-qc-panel">
          <div class="orthogonal-qc-group" role="group" aria-labelledby="orthogonal-qc-continuity-label">
            <span class="orthogonal-qc-group-label" id="orthogonal-qc-continuity-label">连续性 QC</span>
            <button type="button" data-plane="x" data-mode-label="Crossline 剖面">Crossline 剖面</button>
          </div>
          <div class="orthogonal-qc-group" role="group" aria-labelledby="orthogonal-qc-interval-label">
            <span class="orthogonal-qc-group-label" id="orthogonal-qc-interval-label">层间派生</span>
            <button type="button" data-plane="interval-i" data-mode-label="Inline 层间分区" data-derivation="ordered-horizon-intervals" hidden>Inline 层间分区</button>
            <button type="button" data-plane="interval-x" data-mode-label="Crossline 层间分区" data-derivation="ordered-horizon-intervals" hidden>Crossline 层间分区</button>
          </div>
          <div class="orthogonal-qc-group" role="group" aria-labelledby="orthogonal-qc-advanced-label">
            <span class="orthogonal-qc-group-label" id="orthogonal-qc-advanced-label">高级 QC</span>
            <button type="button" data-plane="horizon" data-mode-label="单层位时间/深度面 QC" hidden>单层位时间/深度面 QC</button>
            <button type="button" data-plane="z" data-mode-label="水平标签切片 QC">水平标签切片 QC</button>
          </div>
        </div>
      </details>
    </nav>
    <label id="orthogonal-overlay-choice" hidden><span>结果图层</span><select></select></label>
    <label id="orthogonal-surface-choice" hidden><span>命名层位</span><select></select></label>
  </header>
"""
    else:
        toolbar_fragment = r"""
  <header class="orthogonal-toolbar">
    <nav aria-label="二维切片方向" id="orthogonal-plane-buttons">
      <button type="button" data-plane="i">Inline 剖面</button>
      <button type="button" data-plane="x">Crossline 剖面</button>
      <button type="button" data-plane="interval-i" data-derivation="ordered-horizon-intervals" hidden>Inline 层间分区</button>
      <button type="button" data-plane="interval-x" data-derivation="ordered-horizon-intervals" hidden>Crossline 层间分区</button>
      <button type="button" data-plane="horizon" hidden>单层位俯视图（QC）</button>
      <button type="button" data-plane="z">水平标签切片（QC）</button>
    </nav>
    <label id="orthogonal-overlay-choice" hidden><span>结果图层</span><select></select></label>
    <label id="orthogonal-surface-choice" hidden><span>命名层位</span><select></select></label>
  </header>
"""

    fragment = r"""
<div class="orthogonal-slice-view" id="orthogonal-slice-view">
__ORTHOGONAL_TOOLBAR__
  <div class="orthogonal-stage">
    <canvas id="orthogonal-slice-canvas" aria-label="平台二维正交切片"></canvas>
    <aside class="orthogonal-legend" id="orthogonal-legend"></aside>
    <div class="orthogonal-empty" id="orthogonal-empty" hidden>当前平面没有可显示的有效样点</div>
  </div>
  <footer class="orthogonal-controls">
    <label id="orthogonal-index-control"><span id="orthogonal-index-label">切片</span><input id="orthogonal-index" type="range" min="0" max="1" value="0" step="1"><output id="orthogonal-index-value">0</output></label>
    <label id="orthogonal-gain-control"><span>振幅增益</span><input id="orthogonal-gain" type="range" min="0.5" max="3" value="1.45" step="0.05"><output>1.45</output></label>
    <label id="orthogonal-threshold-control"><span>结果下限</span><input id="orthogonal-threshold" type="range" min="0" max="1" value="0.5" step="0.01"><output>0.50</output></label>
    <div class="orthogonal-status" id="orthogonal-status">正在准备二维切片…</div>
  </footer>
</div>
<style>
.orthogonal-slice-view{position:absolute;inset:0;display:grid;grid-template-rows:auto minmax(0,1fr) auto;background:#eef3f7;color:#263b4f}
.orthogonal-toolbar{display:flex;min-height:52px;gap:12px;align-items:center;padding:8px 14px;background:#fff;border-bottom:1px solid #d4dee8;box-shadow:0 2px 9px rgba(31,55,79,.05)}
.orthogonal-toolbar nav{display:flex;gap:5px}.orthogonal-toolbar button{min-height:36px;padding:0 14px;font-size:12px}.orthogonal-toolbar button.active{color:#fff;background:#1677e8;border-color:#1677e8;box-shadow:0 4px 12px rgba(22,119,232,.2)}.orthogonal-toolbar button:disabled{cursor:not-allowed;color:#97a5b3;background:#f2f5f8;border-color:#dce4eb;opacity:.72}
.orthogonal-toolbar--surface-global{position:relative;z-index:10}.orthogonal-toolbar--surface-global nav{align-items:center}.orthogonal-qc-menu{position:relative}.orthogonal-qc-menu>summary{display:flex;min-height:34px;align-items:center;gap:8px;padding:0 13px;list-style:none;color:#49647d;font-size:12px;font-weight:700;white-space:nowrap;cursor:pointer;background:#f7fafc;border:1px solid #cbd8e5;border-radius:6px;user-select:none}.orthogonal-qc-menu>summary::-webkit-details-marker{display:none}.orthogonal-qc-menu>summary::after{content:'▾';font-size:12px;transition:transform .16s ease}.orthogonal-qc-menu[open]>summary::after{transform:rotate(180deg)}.orthogonal-qc-menu>summary:focus-visible{outline:2px solid #1677e8;outline-offset:2px}.orthogonal-qc-menu.has-active-view>summary{color:#1268c4;background:#eef6ff;border-color:#8dbceb}.orthogonal-qc-panel{position:absolute;top:calc(100% + 8px);left:0;z-index:20;display:grid;width:min(340px,calc(100vw - 28px));gap:10px;padding:11px;background:#fff;border:1px solid #cad7e4;border-radius:9px;box-shadow:0 14px 34px rgba(24,52,78,.2)}.orthogonal-qc-group{display:grid;grid-template-columns:1fr 1fr;gap:6px}.orthogonal-qc-group-label{grid-column:1/-1;color:#71869a;font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.orthogonal-qc-group+.orthogonal-qc-group{padding-top:9px;border-top:1px solid #e3eaf0}.orthogonal-qc-group button{width:100%;min-height:34px;padding:5px 9px;text-align:left;line-height:1.25}
.orthogonal-toolbar label{display:flex;gap:7px;align-items:center;margin-left:auto;color:#61768a;font-size:12px}.orthogonal-toolbar label+label{margin-left:0}.orthogonal-toolbar select{min-width:150px;height:34px;padding:0 8px;color:#2d455c;background:#f8fafc;border:1px solid #cbd7e3;border-radius:6px}
.orthogonal-stage{position:relative;min-width:0;min-height:0;overflow:hidden;background:var(--scene-bg)}#orthogonal-slice-canvas{position:absolute;inset:14px 116px 14px 14px;display:block;width:calc(100% - 130px);height:calc(100% - 28px);image-rendering:auto;background:var(--scene-flat);border:1px solid #b9c9d6;border-radius:7px;box-shadow:0 10px 28px rgba(34,57,78,.09)}
.orthogonal-legend{position:absolute;top:14px;right:14px;display:grid;width:88px;gap:6px;padding:9px;color:#526a80;font-size:12px;background:rgba(255,255,255,.94);border:1px solid #cbd7e2;border-radius:7px}.orthogonal-legend strong{font-size:12px;color:#30485e}.orthogonal-ramp{height:116px;border:1px solid #aebdca;border-radius:3px}.orthogonal-class-row{display:grid;grid-template-columns:12px minmax(0,1fr);gap:5px;align-items:center}.orthogonal-class-row i{width:11px;height:11px;border:1px solid rgba(0,0,0,.14);border-radius:2px}
.orthogonal-interval-note{padding-top:6px;border-top:1px solid #d7e0e8;color:#8a4b08;line-height:1.35}.orthogonal-interval-row{display:grid;grid-template-columns:12px minmax(0,1fr);gap:5px;align-items:center}.orthogonal-interval-row i{width:11px;height:11px;border:1px solid rgba(0,0,0,.14);border-radius:2px}.orthogonal-interval-row b{font-size:12px;line-height:1.25}
.orthogonal-empty{position:absolute;inset:42% 28%;display:grid;place-items:center;color:#61768b;font-size:12px;background:#fff;border:1px dashed #b8c6d4;border-radius:7px}
.orthogonal-slice-view [hidden]{display:none!important}
.orthogonal-controls{display:grid;grid-template-columns:minmax(240px,2fr) minmax(150px,1fr) minmax(150px,1fr) minmax(190px,1fr);gap:18px;align-items:center;min-height:78px;padding:10px 16px;background:#fff;border-top:1px solid #d4dee8}.orthogonal-controls label{display:grid;grid-template-columns:auto minmax(80px,1fr) 44px;gap:8px;align-items:center}.orthogonal-controls span,.orthogonal-controls output{font-size:12px}.orthogonal-controls output{color:#146ac5;font-weight:750;text-align:right}.orthogonal-controls input{width:100%;accent-color:#1677e8}.orthogonal-status{color:#60758a;font-size:12px;line-height:1.45}
@media(max-width:900px){.orthogonal-toolbar{flex-wrap:wrap}.orthogonal-toolbar label{margin-left:0}.orthogonal-controls{grid-template-columns:1fr 1fr}.orthogonal-status{grid-column:1/-1}#orthogonal-slice-canvas{inset:10px 96px 10px 10px;width:calc(100% - 106px);height:calc(100% - 20px)}.orthogonal-legend{top:10px;right:10px;width:78px}}
@media(max-width:900px){.orthogonal-toolbar--surface-global nav{width:100%;flex-wrap:nowrap}.orthogonal-qc-panel{max-height:min(58vh,360px);overflow:auto}}
@media(max-width:520px){.orthogonal-toolbar--surface-global nav{justify-content:space-between}.orthogonal-toolbar--surface-global nav>button{flex:1 1 auto;padding-inline:9px}.orthogonal-qc-menu{flex:0 1 auto;min-width:0}.orthogonal-qc-menu>summary{max-width:52vw;padding-inline:9px;overflow:hidden;text-overflow:ellipsis}.orthogonal-qc-panel{right:0;left:auto}.orthogonal-qc-group{grid-template-columns:1fr}}
</style>
<script>
(function(){
  const payload=window.__wellSeismicVolumePayload;
  const root=document.getElementById('orthogonal-slice-view'),canvas=document.getElementById('orthogonal-slice-canvas'),context=canvas.getContext('2d');
  const status=document.getElementById('orthogonal-status'),empty=document.getElementById('orthogonal-empty'),legend=document.getElementById('orthogonal-legend');
  const indexControl=document.getElementById('orthogonal-index-control'),indexInput=document.getElementById('orthogonal-index'),indexLabel=document.getElementById('orthogonal-index-label'),indexValue=document.getElementById('orthogonal-index-value');
  const gainInput=document.getElementById('orthogonal-gain'),gainOutput=gainInput.parentElement.querySelector('output');
  const thresholdControl=document.getElementById('orthogonal-threshold-control'),thresholdInput=document.getElementById('orthogonal-threshold'),thresholdOutput=thresholdInput.parentElement.querySelector('output');
  const surfaceChoice=document.getElementById('orthogonal-surface-choice'),surfaceSelect=surfaceChoice.querySelector('select');
  const overlayChoice=document.getElementById('orthogonal-overlay-choice'),overlaySelect=overlayChoice.querySelector('select');
  const layerModeSwitch=document.querySelector('.layer-mode-switch'),gainControl=gainInput.parentElement;
  const qcDetails=document.getElementById('orthogonal-qc-views'),qcSummaryLabel=document.getElementById('orthogonal-qc-summary-label');
  if(!payload||!payload.cube){status.textContent='二维切片载荷缺失';empty.hidden=false;return;}
  let shape,nz,ni,nx,surfaces,overlays,axisValues,cubeSigned,overlayDatas=[],overlayData=null;
  const sourceSpanZYX=__SOURCE_SPAN_ZYX__;
  const crosslineShortGapBridgeMaxMissingPreviewCells=__CROSSLINE_SHORT_GAP_LIMIT__;
  const axisLabels=payload.axisLabels||{},verticalAxis=payload.verticalAxis||{},sliceContract=payload.sliceViewContract||{},displayNotice=String((payload.sliceViewContract||{}).displayNotice||'');
  function planeNotice(){return displayNotice;}
  function updatePlaneControls(){const horizon=state?.plane==='horizon';surfaceChoice.hidden=!horizon||!surfaces.length;overlayChoice.hidden=horizon||!overlays.length;gainControl.hidden=horizon;if(layerModeSwitch)layerModeSwitch.hidden=horizon;}
  const scopedPlanes=Array.isArray(sliceContract.allowedPlanes)&&sliceContract.allowedPlanes.length?new Set(sliceContract.allowedPlanes.map(String)):null;
  const globalConsistent=sliceContract.globalConsistent===true,backgroundOnly=sliceContract.displayMode==='background_only',faultsegExactBlock=sliceContract.displayMode==='faultseg_exact_binary_mask',surfaceSegGlobal=globalConsistent&&String(sliceContract.modelId||'')==='seismic_surface_seg';
  const zPlaneButton=document.querySelector('[data-plane="z"]');if(faultsegExactBlock&&zPlaneButton)zPlaneButton.textContent='时间 / 深度切片';
  function bytes(spec){if(!spec||typeof spec.values!=='string')throw new Error('切片载荷缺少base64 values');const binary=atob(spec.values),result=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)result[i]=binary.charCodeAt(i);return result;}
  function checkedShape(spec,label){if(!Array.isArray(spec?.shape)||spec.shape.length!==3)throw new Error(label+'缺少三维shape');const current=spec.shape.map(Number);if(current.some(value=>!Number.isInteger(value)||value<=0))throw new Error(label+'含无效shape');return current;}
  function decodeOverlay(spec){const currentShape=checkedShape(spec,'结果叠加体');if(currentShape.some((value,index)=>value!==shape[index]))throw new Error('结果叠加体与背景体shape不一致');const raw=bytes(spec),count=currentShape[0]*currentShape[1]*currentShape[2],expected=spec.encoding==='base64-float32'?count*4:count;if(raw.byteLength!==expected)throw new Error(`结果叠加体字节数${raw.byteLength}与合同${expected}不一致`);const values=new Float32Array(count),display=new Uint8Array(count);if(spec.encoding==='base64-uint8'){display.set(raw);for(let i=0;i<count;i++)values[i]=display[i]/255;return {values,display};}if(spec.encoding==='base64-int8'){const signed=new Int8Array(raw.buffer,raw.byteOffset,raw.byteLength);for(let i=0;i<count;i++){values[i]=Math.max(0,Math.min(1,signed[i]/127));display[i]=Math.round(values[i]*255);}return {values,display};}if(spec.encoding==='base64-float32'){const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);for(let i=0;i<count;i++){const value=view.getFloat32(i*4,true);if(!Number.isFinite(value))throw new Error(`结果叠加体第${i}个值非有限`);values[i]=Math.max(0,Math.min(1,value));display[i]=Math.round(values[i]*255);}return {values,display};}throw new Error('不支持的结果体编码：'+spec.encoding);}
  try{
    shape=checkedShape(payload.cube,'地震背景体');[nz,ni,nx]=shape;
    if(payload.cube.encoding!=='base64-int8')throw new Error('地震背景体必须使用base64-int8编码');
    const cubeRaw=bytes(payload.cube),cubeCount=nz*ni*nx;if(cubeRaw.byteLength!==cubeCount)throw new Error(`地震背景体字节数${cubeRaw.byteLength}与合同${cubeCount}不一致`);cubeSigned=new Int8Array(cubeRaw.buffer,cubeRaw.byteOffset,cubeRaw.byteLength);
    surfaces=Array.isArray(payload.surfaces)?payload.surfaces.filter(item=>Array.isArray(item?.grid)):[];
    overlays=Array.isArray(payload.overlays)?payload.overlays.filter(item=>item?.volume):[];
    axisValues={z:Array.isArray(payload.timeValues)?payload.timeValues:[],i:Array.isArray(payload.inlineValues)?payload.inlineValues:[],x:Array.isArray(payload.crosslineValues)?payload.crosslineValues:[]};
    overlayDatas=overlays.map(item=>decodeOverlay(item.volume));overlayData=overlayDatas[0]||null;
  }catch(error){status.textContent='二维切片载荷校验失败：'+(error instanceof Error?error.message:String(error));empty.hidden=false;return;}
  const fallbackPlane=surfaces.length?'horizon':'z',requestedPlane=String(sliceContract.defaultPlane||fallbackPlane),contractPlane=!scopedPlanes||scopedPlanes.has(requestedPlane)?requestedPlane:(scopedPlanes.has('i')?'i':fallbackPlane),defaultPlane=surfaceSegGlobal&&(!scopedPlanes||scopedPlanes.has('i'))?'i':contractPlane;
  const requestedDefaultIndices=Array.isArray(payload.defaultIndices)&&payload.defaultIndices.length===3?payload.defaultIndices:[];
  function initialSliceIndex(value,maximum){const numeric=Number(value);return Number.isFinite(numeric)?Math.max(0,Math.min(maximum,Math.round(numeric))):Math.floor((maximum+1)/2);}
  const defaultSliceIndices={z:initialSliceIndex(requestedDefaultIndices[0],nz-1),i:initialSliceIndex(requestedDefaultIndices[1],ni-1),x:initialSliceIndex(requestedDefaultIndices[2],nx-1)};
  const state={plane:defaultPlane,index:{...defaultSliceIndices},surface:0,overlay:0,layerMode:'combined'};
  const planeButtons=[...document.querySelectorAll('[data-plane]')];
  function updatePlaneMenu(){if(!qcDetails||!qcSummaryLabel)return;const activeButton=document.querySelector(`[data-plane="${state.plane}"]`),qcActive=Boolean(activeButton&&qcDetails.contains(activeButton)),modeLabel=String(activeButton?.dataset.modeLabel||activeButton?.textContent||'').trim();qcDetails.classList.toggle('has-active-view',qcActive);qcSummaryLabel.textContent=qcActive&&modeLabel?'QC 与派生视图 · '+modeLabel:'QC 与派生视图';qcDetails.title=qcActive&&modeLabel?'当前视图：'+modeLabel:'';}
  function selectPlane(plane,closeMenu=true,renderNow=true){const button=planeButtons.find(item=>item.dataset.plane===plane);if(!button||button.disabled)return false;state.plane=plane;for(const item of planeButtons){const active=item===button;item.classList.toggle('active',active);if(active)item.setAttribute('aria-current','page');else item.removeAttribute('aria-current');}if(qcDetails&&closeMenu)qcDetails.open=false;configureIndex();if(renderNow)draw();return true;}
  const flat=(z,i,x)=>(z*ni+i)*nx+x;
  function mix(a,b,t){return Math.round(a+(b-a)*t)}
  function seismicColor(value){const v=Math.max(-1,Math.min(1,value*Number(gainInput.value)));return v<0?[mix(248,8,-v),mix(248,39,-v),mix(248,190,-v)]:[mix(248,196,v),mix(248,18,v),mix(248,12,v)];}
  function seismicGray(value){const v=Math.max(-1,Math.min(1,value*Number(gainInput.value))),level=mix(34,238,(v+1)/2);return [level,level,level];}
  const categorical=[[27,83,171],[225,91,53],[34,155,91],[142,68,173],[238,173,31],[15,151,167],[215,48,112],[113,117,122]];
  function hslRgb(h,s,l){const c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs((h/60)%2-1)),m=l-c/2;let rgb=h<60?[c,x,0]:h<120?[x,c,0]:h<180?[0,c,x]:h<240?[0,x,c]:h<300?[x,0,c]:[c,0,x];return rgb.map(value=>Math.round((value+m)*255));}
  function categoricalColor(index){return index<categorical.length?categorical[index]:hslRgb((index*137.508)%360,.58,.46);}
  function hexColor(value,fallback){const match=/^#([0-9a-f]{6})$/i.exec(String(value||''));return match?[parseInt(match[1].slice(0,2),16),parseInt(match[1].slice(2,4),16),parseInt(match[1].slice(4,6),16)]:fallback;}
  function turbo(t){const x=Math.max(0,Math.min(1,t));const stops=[[48,18,59],[38,94,171],[30,170,137],[149,214,64],[253,174,40],[180,4,38]],p=x*(stops.length-1),a=Math.min(stops.length-2,Math.floor(p)),f=p-a;return stops[a].map((v,k)=>mix(v,stops[a+1][k],f));}
  function classDescriptor(raw,spec){const display=Array.isArray(spec.displayCodeRange)?spec.displayCodeRange.map(Number):[0,255],fraction=Math.max(0,Math.min(1,(raw-display[0])/Math.max(1,display[1]-display[0]))),codes=Array.isArray(spec.classCodes)?spec.classCodes:[];if(codes.length){const position=Math.max(0,Math.min(codes.length-1,Math.round(fraction*(codes.length-1))));return {code:codes[position],paletteIndex:position};}const labels=Array.isArray(spec.labelValueRange)?spec.labelValueRange.map(Number):[0,1],code=Math.round(labels[0]+fraction*(labels[1]-labels[0]));return {code,paletteIndex:Math.max(0,code-labels[0])};}
  function displayedClassCodes(spec){const codes=Array.isArray(spec.classCodes)?spec.classCodes:[];if(codes.length)return codes.map((code,index)=>({code,paletteIndex:index}));const range=Array.isArray(spec.labelValueRange)?spec.labelValueRange.map(Number):[0,1],count=Math.max(1,Math.round(range[1]-range[0]+1));if(count<=8)return Array.from({length:count},(_,index)=>({code:range[0]+index,paletteIndex:index}));const positions=[0,.2,.4,.6,.8,1].map(fraction=>Math.round(range[0]+fraction*(range[1]-range[0])));return [...new Set(positions)].map(code=>({code,paletteIndex:Math.max(0,code-range[0])}));}
  function maskBoundary(index,invalid){const z=Math.floor(index/(ni*nx)),rest=index-z*ni*nx,i=Math.floor(rest/nx),x=rest-i*nx,neighbors=[[z-1,i,x],[z+1,i,x],[z,i-1,x],[z,i+1,x],[z,i,x-1],[z,i,x+1]];return neighbors.some(([zz,ii,xx])=>zz<0||zz>=nz||ii<0||ii>=ni||xx<0||xx>=nx||overlayData.display[flat(zz,ii,xx)]===invalid);}
  function labelBoundary(index,invalid){const z=Math.floor(index/(ni*nx)),rest=index-z*ni*nx,i=Math.floor(rest/nx),x=rest-i*nx,current=overlayData.display[index],plane=state.plane;let neighbors;if(plane==='z')neighbors=[[z,i-1,x],[z,i+1,x],[z,i,x-1],[z,i,x+1]];else if(plane==='x'||plane==='interval-x')neighbors=[[z-1,i,x],[z+1,i,x],[z,i-1,x],[z,i+1,x]];else neighbors=[[z-1,i,x],[z+1,i,x],[z,i,x-1],[z,i,x+1]];return neighbors.some(([zz,ii,xx])=>{if(zz<0||zz>=nz||ii<0||ii>=ni||xx<0||xx>=nx)return false;const adjacent=overlayData.display[flat(zz,ii,xx)];return adjacent!==invalid&&adjacent!==current;});}
  function overlayColor(index){const item=overlays[state.overlay]||{},kind=String(item.kind||'probability').toLowerCase(),raw=overlayData.display[index],value=overlayData.values[index],invalid=Number(item.volume.invalidDisplayCode??0);if(kind==='mask'){if(raw===invalid)return {rgb:[0,0,0],alpha:0,code:null};const boundary=maskBoundary(index,invalid);return {rgb:boundary?[255,210,48]:[239,54,61],alpha:boundary ? 0.92 : Number(item.alpha||.58),code:1};}if(kind==='class_code'||kind==='labels'){if(raw===invalid)return {rgb:[0,0,0],alpha:0,code:null};const descriptor=classDescriptor(raw,item.volume);if(kind==='labels'&&labelBoundary(index,invalid))return {rgb:hexColor(item.boundaryColor,[245,158,11]),alpha:Number(item.boundaryAlpha||.98),code:descriptor.code,boundary:true};return {rgb:categoricalColor(descriptor.paletteIndex),alpha:kind==='labels'?Math.min(.24,Number(item.alpha||.20)):Number(item.alpha||.72),code:descriptor.code,boundary:false};}if(value<Number(thresholdInput.value))return {rgb:[0,0,0],alpha:0,code:null};return {rgb:turbo(value),alpha:Number(item.alpha||.62),code:null};}
  function axisValue(axis,index){const values=axisValues[axis]||[];return values[index]??index;}
  function axisValueAt(axis,position){const values=axisValues[axis]||[];if(!values.length)return position;const lower=Math.max(0,Math.min(values.length-1,Math.floor(position))),upper=Math.max(0,Math.min(values.length-1,Math.ceil(position))),fraction=Math.max(0,Math.min(1,position-lower)),a=Number(values[lower]),b=Number(values[upper]);if(!Number.isFinite(a)||!Number.isFinite(b))throw new Error('切片坐标轴含非有限值');return a+(b-a)*fraction;}
  function formatValue(value){return typeof value==='number'&&Number.isFinite(value)?Number(value.toFixed(3)):String(value??'—');}
  function setLegend(kind,minValue,maxValue,forceContinuous=false){legend.replaceChildren();const title=document.createElement('strong');title.textContent=kind;legend.append(title);if(!forceContinuous&&kind.includes('掩码')){for(const [color,text] of [['rgb(255,210,48)','断层边界'],['rgb(239,54,61)','断层掩码']]){const row=document.createElement('span');row.className='orthogonal-class-row';const swatch=document.createElement('i');swatch.style.background=color;const label=document.createElement('b');label.textContent=text;row.append(swatch,label);legend.append(row);}return;}if(!forceContinuous&&(kind.includes('类别')||kind.includes('分层')||kind.includes('层位')||kind.includes('层间'))){const item=overlays[state.overlay]||{},scope=String(item.labelScope||'').toLowerCase(),prefix=scope==='global_packages'?'层间 G':scope==='inline_local'?'局部层 ':'类别 ';for(const descriptor of displayedClassCodes(item.volume||{})){const row=document.createElement('span');row.className='orthogonal-class-row';const swatch=document.createElement('i');swatch.style.background=`rgb(${categoricalColor(descriptor.paletteIndex).join(',')})`;const label=document.createElement('b');label.textContent=prefix+descriptor.code;row.append(swatch,label);legend.append(row);}if(item.volume?.unknownTransparent===true){const row=document.createElement('span');row.className='orthogonal-class-row';const swatch=document.createElement('i');swatch.style.background='repeating-linear-gradient(135deg,#fff 0 3px,#dce4eb 3px 6px)';const label=document.createElement('b');label.textContent='未判定（透明）';row.append(swatch,label);legend.append(row);}return;}const ramp=document.createElement('i');ramp.className='orthogonal-ramp';ramp.style.background=kind.includes('灰度')?'linear-gradient(#eeeeee,#888888 50%,#222222)':kind.includes('振幅')?'linear-gradient(#c4120c,#f8f8f8 50%,#0827be)':'linear-gradient(#b40426,#fdae31,#1fa77a,#2660aa,#30123b)';legend.append(ramp);const top=document.createElement('span');top.textContent=formatValue(maxValue);const bottom=document.createElement('span');bottom.textContent=formatValue(minValue);legend.append(top,bottom);}
  const intervalPalette=[[47,93,160],[46,139,112],[230,185,61],[221,111,48],[166,74,143],[94,83,155],[26,145,166],[171,96,53]];
  function intervalColor(index,count){return index<intervalPalette.length?intervalPalette[index]:turbo((index+.5)/Math.max(1,count));}
  function surfaceGridForPlane(surface,plane){if((plane==='x'||plane==='interval-x')&&Array.isArray(surface.crossInlineDisplayGrid))return surface.crossInlineDisplayGrid;return surface.grid||[];}
  function finiteProfileValue(raw){if(raw===null||raw===undefined||raw==='')return null;const value=Number(raw);return Number.isFinite(value)?value:null;}
  function profileSingleGapConnection(surfaceIndex,plane,index,horizontal){if(plane!=='x'||crosslineShortGapBridgeMaxMissingPreviewCells!==1||horizontal<=0||horizontal>=ni-1)return false;const targetGrid=surfaceGridForPlane(surfaces[surfaceIndex],plane);if(finiteProfileValue(targetGrid[horizontal]?.[index])!==null)return false;const orderedValues=[];for(const surface of surfaces){const grid=surfaceGridForPlane(surface,plane);let value=finiteProfileValue(grid[horizontal]?.[index]);if(value===null){const left=finiteProfileValue(grid[horizontal-1]?.[index]),right=finiteProfileValue(grid[horizontal+1]?.[index]);if(left===null||right===null)return false;value=(left+right)/2;}orderedValues.push(value);}for(let surface=1;surface<orderedValues.length;surface++)if(orderedValues[surface]<=orderedValues[surface-1])return false;return true;}
  function orderedBoundaries(inlineIndex,crosslineIndex,plane){const rawValues=surfaces.map(surface=>surfaceGridForPlane(surface,plane)?.[inlineIndex]?.[crosslineIndex]);if(rawValues.some(value=>value===null||value===undefined||value===''||!Number.isFinite(Number(value))))return null;const values=rawValues.map(Number);for(let index=1;index<values.length;index++)if(values[index]<=values[index-1])return null;return values;}
  function intervalDescriptor(z,inlineIndex,crosslineIndex,plane){const boundaries=orderedBoundaries(inlineIndex,crosslineIndex,plane);if(!boundaries)return null;let region=0;while(region<boundaries.length&&z>=boundaries[region])region++;return {region,color:intervalColor(region,boundaries.length+1)};}
  function intervalLabel(index){if(index===0)return `${surfaces[0]?.name||'首层位'} 之上`;if(index===surfaces.length)return `${surfaces.at(-1)?.name||'末层位'} 之下`;return `${surfaces[index-1]?.name||`层位 ${index}`} — ${surfaces[index]?.name||`层位 ${index+1}`}`;}
  function setIntervalLegend(){legend.replaceChildren();const title=document.createElement('strong');title.textContent='层间分区';legend.append(title);for(let index=0;index<=surfaces.length;index++){const row=document.createElement('span');row.className='orthogonal-interval-row';const swatch=document.createElement('i');swatch.style.background=`rgb(${intervalColor(index,surfaces.length+1).join(',')})`;const label=document.createElement('b');label.textContent=intervalLabel(index);row.append(swatch,label);legend.append(row);}const note=document.createElement('small');note.className='orthogonal-interval-note';note.textContent='由候选层位派生，非地层分类/沉积相预测';legend.append(note);}
  function renderPixels(width,height,sampler){const image=new ImageData(width,height);let valid=0,min=Infinity,max=-Infinity,intervalValid=0,intervalInvalid=0;for(let y=0;y<height;y++)for(let x=0;x<width;x++){const result=sampler(x,y),offset=(y*width+x)*4;if(!result){image.data[offset]=0;image.data[offset+1]=0;image.data[offset+2]=0;image.data[offset+3]=0;continue;}valid++;if(result.intervalValid===true)intervalValid++;if(result.intervalValid===false)intervalInvalid++;min=Math.min(min,result.value);max=Math.max(max,result.value);image.data[offset]=result.rgb[0];image.data[offset+1]=result.rgb[1];image.data[offset+2]=result.rgb[2];image.data[offset+3]=255;}return {image,valid,min,max,intervalValid,intervalInvalid};}
  function suppressCrosslineVoxelLabels(kind){return surfaceSegGlobal&&state.plane==='x'&&state.layerMode==='combined'&&kind==='labels';}
  function composite(index){const amplitude=cubeSigned[index]/127,current=overlays[state.overlay]||{},kind=String(current.kind||'').toLowerCase(),base=backgroundOnly||kind==='mask'||kind==='labels'?seismicGray(amplitude):seismicColor(amplitude);if(!overlayData)return {rgb:base,value:amplitude};if(suppressCrosslineVoxelLabels(kind))return {rgb:seismicGray(amplitude),value:amplitude};const overlay=overlayColor(index);if(state.layerMode==='prediction'){return overlay.alpha?{rgb:overlay.rgb,value:overlayData.values[index]}:null;}return {rgb:overlay.alpha?base.map((v,k)=>mix(v,overlay.rgb[k],overlay.alpha)):base,value:amplitude};}
  function drawProfileSurfaces(plane,index,dx,dy,dw,dh,ratio){const inlinePlane=plane==='i'||plane==='interval-i',horizontalCount=inlinePlane?nx:ni;for(let surfaceIndex=0;surfaceIndex<surfaces.length;surfaceIndex++){const surface=surfaces[surfaceIndex],grid=surfaceGridForPlane(surface,plane);context.beginPath();let drawing=false;for(let horizontal=0;horizontal<horizontalCount;horizontal++){const raw=inlinePlane?grid[index]?.[horizontal]:grid[horizontal]?.[index],z=Number(raw);if(raw===null||raw===undefined||raw===''||!Number.isFinite(z)){if(drawing&&profileSingleGapConnection(surfaceIndex,plane,index,horizontal))continue;drawing=false;continue;}const px=dx+(horizontal/Math.max(1,horizontalCount-1))*dw,py=dy+(z/Math.max(1,nz-1))*dh;if(drawing)context.lineTo(px,py);else context.moveTo(px,py);drawing=true;}context.shadowBlur=0;context.strokeStyle='rgba(255,255,255,.96)';context.lineWidth=5.4*ratio;context.stroke();context.strokeStyle=surface.color||categorical[surfaceIndex%categorical.length].map(value=>value.toString(16).padStart(2,'0')).join('').replace(/^/,'#');context.lineWidth=2.8*ratio;context.stroke();}}
  function planeDisplayExtent(plane){if(plane==='horizon'||plane==='z')return [sourceSpanZYX[2],sourceSpanZYX[1]];if(plane==='i'||plane==='interval-i')return [sourceSpanZYX[2],sourceSpanZYX[0]];return [sourceSpanZYX[1],sourceSpanZYX[0]];}
  function draw(){
    try{
      let width=nx,height=ni,result,title,legendName,categoricalMode=false,profileSurface=null,boundaryProfile=null;
      if(state.plane==='horizon'){
        const surface=surfaces[state.surface],grid=surface.grid||[];width=nx;height=ni;let zMin=Infinity,zMax=-Infinity;
        for(const row of grid)for(const value of row)if(Number.isFinite(value)){zMin=Math.min(zMin,Number(value));zMax=Math.max(zMax,Number(value));}
        if(!Number.isFinite(zMin)){zMin=0;zMax=1;}
        result=renderPixels(width,height,(x,y)=>{const z=Number(grid[y]?.[x]);if(!Number.isFinite(z))return null;const t=(z-zMin)/Math.max(1e-9,zMax-zMin);return {rgb:turbo(t),value:axisValueAt('z',Math.max(0,Math.min(nz-1,z)))};});
        title=(surface.name||'全局层位')+' · Inline/Xline 二维平面';legendName=(verticalAxis.label||'层位时间')+' / '+(verticalAxis.unit||'');setLegend(legendName,result.min,result.max,true);indexControl.hidden=true;thresholdControl.hidden=true;
      }else{
        indexControl.hidden=false;const axis=state.plane.endsWith('-i')?'i':state.plane.endsWith('-x')?'x':state.plane,index=state.index[axis];
        if(state.plane==='z'){
          width=nx;height=ni;result=renderPixels(width,height,(x,y)=>composite(flat(index,y,x)));title=`${verticalAxis.label||'采样轴'} ${formatValue(axisValue('z',index))} ${verticalAxis.unit||''} · Inline/Xline 平面`;
        }else if(state.plane==='i'){
          width=nx;height=nz;result=renderPixels(width,height,(x,y)=>composite(flat(y,index,x)));title=`Inline ${formatValue(axisValue('i',index))} · Crossline/${verticalAxis.label||'采样轴'} 剖面`;if(globalConsistent&&surfaces.length)boundaryProfile={plane:'i',index};
        }else if(state.plane==='x'){
          width=ni;height=nz;result=renderPixels(width,height,(x,y)=>composite(flat(y,x,index)));title=`Crossline ${formatValue(axisValue('x',index))} · Inline/${verticalAxis.label||'采样轴'} 剖面`;if(globalConsistent&&surfaces.length)boundaryProfile={plane:'x',index};
        }else if(state.plane==='interval-i'){
          width=nx;height=nz;result=renderPixels(width,height,(x,y)=>{const amplitude=cubeSigned[flat(y,index,x)]/127,base=seismicGray(amplitude),interval=intervalDescriptor(y,index,x,state.plane);return {rgb:interval?base.map((value,k)=>mix(value,interval.color[k],.20)):base,value:amplitude,intervalValid:Boolean(interval)};});title=`Inline ${formatValue(axisValue('i',index))} · 全局层间分区剖面`;profileSurface={plane:state.plane,index};
        }else{
          width=ni;height=nz;result=renderPixels(width,height,(x,y)=>{const amplitude=cubeSigned[flat(y,x,index)]/127,base=seismicGray(amplitude),interval=intervalDescriptor(y,x,index,state.plane);return {rgb:interval?base.map((value,k)=>mix(value,interval.color[k],.20)):base,value:amplitude,intervalValid:Boolean(interval)};});title=`Crossline ${formatValue(axisValue('x',index))} · 全局层间分区剖面`;profileSurface={plane:state.plane,index};
        }
        if(profileSurface){categoricalMode=true;thresholdControl.hidden=true;setIntervalLegend();}
        else{const current=overlays[state.overlay]||{},currentKind=String(current.kind||'').toLowerCase(),crosslineLineOnly=suppressCrosslineVoxelLabels(currentKind),legendTitle=crosslineLineOnly?'灰度地震振幅':backgroundOnly?'灰度地震振幅':currentKind==='mask'?'断层掩码':currentKind==='labels'?(current.name||'局部分层'):currentKind==='class_code'?'预测类别':overlayData&&state.layerMode==='prediction'?'预测结果':'地震振幅';categoricalMode=!crosslineLineOnly&&['class_code','labels','mask'].includes(currentKind);thresholdControl.hidden=!overlayData||categoricalMode||crosslineLineOnly;setLegend(legendTitle,result.min,result.max);}
      }
      empty.hidden=result.valid>0;
      const ratio=Math.min(window.devicePixelRatio||1,2),bounds=canvas.getBoundingClientRect(),cw=Math.max(1,Math.round(bounds.width*ratio)),ch=Math.max(1,Math.round(bounds.height*ratio));
      if(canvas.width!==cw||canvas.height!==ch){canvas.width=cw;canvas.height=ch;}
      context.clearRect(0,0,cw,ch);context.fillStyle='#eef3f7';context.fillRect(0,0,cw,ch);
      const buffer=document.createElement('canvas');buffer.width=width;buffer.height=height;buffer.getContext('2d').putImageData(result.image,0,0);
      const [displayWidth,displayHeight]=planeDisplayExtent(state.plane),scale=Math.min((cw-24*ratio)/displayWidth,(ch-42*ratio)/displayHeight),dw=Math.max(1,displayWidth*scale),dh=Math.max(1,displayHeight*scale),dx=(cw-dw)/2,dy=(ch-dh)/2+8*ratio;
      context.imageSmoothingEnabled=!categoricalMode;context.imageSmoothingQuality='high';context.drawImage(buffer,dx,dy,dw,dh);
      if(profileSurface)drawProfileSurfaces(profileSurface.plane,profileSurface.index,dx,dy,dw,dh,ratio);
      if(boundaryProfile)drawProfileSurfaces(boundaryProfile.plane,boundaryProfile.index,dx,dy,dw,dh,ratio);
      context.strokeStyle='#60758a';context.lineWidth=ratio;context.strokeRect(dx,dy,dw,dh);context.fillStyle='#334b61';context.font=`${12*ratio}px sans-serif`;context.fillText(title,12*ratio,15*ratio);
      const activeNotice=surfaceSegGlobal?'':planeNotice(),noticeSuffix=activeNotice?` · ${activeNotice}`:'';status.textContent=(profileSurface?`${title} · ${result.intervalValid.toLocaleString()} 个分区像素 · ${result.intervalInvalid.toLocaleString()} 个缺失/交叉位置保持透明 · 全局层位派生`:boundaryProfile?`${title} · ${result.valid.toLocaleString()} 个有效像素 · 已叠加全局层位边界`:`${title} · ${result.valid.toLocaleString()} 个有效像素 · 平台原生二维渲染`)+noticeSuffix;
    }
    catch(error){status.textContent='二维切片渲染失败：'+(error instanceof Error?error.message:String(error));empty.hidden=false;}}
  function configureIndex(){const plane=state.plane;updatePlaneControls();updatePlaneMenu();if(plane==='horizon')return;const axis=plane.endsWith('-i')?'i':plane.endsWith('-x')?'x':plane,maxima={z:nz-1,i:ni-1,x:nx-1},labels={z:verticalAxis.label||'时间 / 深度',i:axisLabels.inline||'Inline',x:axisLabels.crossline||'Crossline'};indexInput.max=String(maxima[axis]);indexInput.value=String(state.index[axis]);indexLabel.textContent=labels[axis];indexValue.textContent=formatValue(axisValue(axis,state.index[axis]));}
  surfaces.forEach((surface,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=surface.name||`层位 ${index+1}`;surfaceSelect.append(option);});const horizonButton=document.querySelector('[data-plane="horizon"]');horizonButton.hidden=!surfaces.length||Boolean(scopedPlanes&&!scopedPlanes.has('horizon'));
  const disabledReasons=sliceContract.disabledPlaneReasons||{};
  planeButtons.forEach(button=>{const plane=String(button.dataset.plane),allowed=!scopedPlanes||scopedPlanes.has(plane);button.disabled=!allowed;if(!allowed)button.title=String(disabledReasons[plane]||'当前结果不具备该方向的科学显示合同');});
  document.querySelectorAll('[data-derivation="ordered-horizon-intervals"]').forEach(button=>button.hidden=surfaces.length<2||Boolean(scopedPlanes&&!scopedPlanes.has(String(button.dataset.plane))));
  overlays.forEach((item,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=item.name||`结果图层 ${index+1}`;overlaySelect.append(option);});overlayChoice.hidden=!overlays.length;
  if(overlays.length&&Array.isArray(overlays[0].clim))thresholdInput.value=String(overlays[0].clim[0]??.5);
  planeButtons.forEach(button=>button.addEventListener('click',()=>selectPlane(String(button.dataset.plane))));
  indexInput.addEventListener('input',()=>{const axis=state.plane.endsWith('-i')?'i':state.plane.endsWith('-x')?'x':state.plane;state.index[axis]=Number(indexInput.value);indexValue.textContent=formatValue(axisValue(axis,state.index[axis]));draw();});gainInput.addEventListener('input',()=>{gainOutput.value=Number(gainInput.value).toFixed(2);draw();});thresholdInput.addEventListener('input',()=>{thresholdOutput.value=Number(thresholdInput.value).toFixed(2);draw();});surfaceSelect.addEventListener('change',()=>{state.surface=Number(surfaceSelect.value);draw();});overlaySelect.addEventListener('change',()=>{state.overlay=Number(overlaySelect.value);overlayData=overlayDatas[state.overlay]||null;thresholdInput.value=String(overlays[state.overlay].clim?.[0]??.5);thresholdOutput.value=Number(thresholdInput.value).toFixed(2);draw();});
  window.setOrthogonalSliceLayerMode=mode=>{state.layerMode=mode==='prediction'?'prediction':'combined';updatePlaneControls();draw();};window.resizeOrthogonalSliceView=draw;window.resetOrthogonalSliceView=()=>{state.index={...defaultSliceIndices};configureIndex();draw();};
  selectPlane(state.plane,false,false);if(window.ResizeObserver)new ResizeObserver(()=>draw()).observe(root);requestAnimationFrame(draw);
})();
</script>
"""
    return fragment.replace("__ORTHOGONAL_TOOLBAR__", toolbar_fragment).replace(
        "__SOURCE_SPAN_ZYX__",
        json.dumps(list(source_span_zyx), separators=(",", ":")),
    ).replace(
        "__CROSSLINE_SHORT_GAP_LIMIT__",
        str(_CROSSLINE_DISPLAY_SHORT_GAP_BRIDGE_MAX_MISSING_PREVIEW_CELLS),
    )


def _render_volume(
    project_root: Path,
    volume_payload: dict[str, Any],
    task_id: str,
    asset_index: int,
) -> tuple[str, str, str]:
    whole_fragment = _webgl_volume_fragment(volume_payload, task_id=task_id)
    # The Viser iframe could report a healthy WebGL context while rendering an
    # entirely blank canvas.  The primary slice workflow therefore stays in
    # this document and renders the already sealed, bounded preview payload on
    # a plain Canvas2D.  This is deterministic, needs no sidecar process, and
    # also gives horizon products a useful plan-view presentation.
    slice_fragment = _orthogonal_slice_fragment(volume_payload)
    overlays = list(volume_payload.get("overlays", []))
    primary_kind = (
        str(overlays[0].get("kind", "")).strip().casefold() if overlays else ""
    )
    # Fault masks have a dedicated categorical WebGL path (NEAREST texture +
    # binary shader), so they can open in whole-volume mode without pretending
    # to be a continuous probability field. Other categorical volumes remain
    # on the conservative 2D-only path until they receive an equivalent 3D
    # palette contract.
    prefer_slice = (
        bool(volume_payload.get("surfaces"))
        or primary_kind in {"class_code", "labels"}
        or dict(volume_payload.get("sliceViewContract", {})).get("preferSlice")
        is True
    )
    # In the fixed well-side layout the log board lives in the whole panel.
    # Keep that panel mounted while a right-hand orthogonal slice is selected;
    # the workbench overlays only the seismic side of the panel.
    whole_hidden = (
        " hidden"
        if prefer_slice
        and not _uses_facies_log_layout(volume_payload.get("wellResultTracks"))
        else ""
    )
    slice_hidden = "" if prefer_slice else " hidden"
    return (
        f'<section class="view-panel whole-volume-panel" data-view="whole"{whole_hidden}>{whole_fragment}</section>'
        f'<section class="view-panel slice-volume-panel" data-view="slice"{slice_hidden}>{slice_fragment}</section>',
        "WebGL2 整体三维 + 平台二维正交切片",
        "",
    )


def _render_line(project_root: Path, line_payload: dict[str, Any]) -> str:
    cigvis, _ = _load_cigvis(project_root)
    with _CIGVIS_LOCK:
        cigvis.set_order(True)
        import matplotlib.pyplot as plt

        image = _decode_array(line_payload["image"], 2)
        fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor="#e6edf3")
        ax.set_facecolor("#eef3f7")
        time_values = [
            value for value in line_payload.get("timeValues", []) if value is not None
        ]
        trace_values = [
            value for value in line_payload.get("traceValues", []) if value is not None
        ]
        xsample = None
        ysample = None
        if len(trace_values) >= 2:
            xsample = [
                float(trace_values[0]),
                float(trace_values[1]) - float(trace_values[0]),
            ]
        if len(time_values) >= 2:
            ysample = [
                float(time_values[0]),
                float(time_values[1]) - float(time_values[0]),
            ]
        # plot2d transposes line-first inputs; image.T preserves the platform's
        # [time, trace] view after CIGVis applies its own orientation contract.
        vertical_axis = dict(line_payload.get("verticalAxis", {}))
        vertical_title = f"{vertical_axis.get('label', '采样轴（域未核验）')} / {vertical_axis.get('unit', '')}".rstrip(
            " / "
        )
        cigvis.plot2d(
            image.T,
            cmap="seismic",
            clim=[-1.0, 1.0],
            interpolation="nearest",
            aspect="auto",
            title="",
            xlabel=str(line_payload.get("lineAxis", "Trace")),
            ylabel=vertical_title,
            xsample=xsample,
            ysample=ysample,
            cbar="Normalized amplitude",
            show=False,
            ax=ax,
        )
        ax.tick_params(colors="#526779", labelsize=12)
        ax.xaxis.label.set_color("#33475b")
        ax.yaxis.label.set_color("#33475b")
        ax.title.set_color("#24384d")
        for spine in ax.spines.values():
            spine.set_color("#c8d3df")
        for colorbar_axis in fig.axes[1:]:
            colorbar_axis.set_facecolor("#eef3f7")
            colorbar_axis.tick_params(colors="#526779", labelsize=12)
            colorbar_axis.xaxis.label.set_color("#425466")
            colorbar_axis.yaxis.label.set_color("#425466")
            for spine in colorbar_axis.spines.values():
                spine.set_color("#c8d3df")
        fig.tight_layout(pad=1.2)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div class="cigvis-line-view">'
        f'<img alt="{html.escape(_public_visualization_text(line_payload.get("name"), "二维地震剖面"))}" '
        f'src="data:image/png;base64,{encoded}"></div>'
    )


def _well_sequence_fragment(sequence: dict[str, Any]) -> str:
    """Render one deterministic measured-depth product inside the CIG shell."""

    width, height = 1080, 720
    plot_top, plot_bottom = 72.0, 48.0
    plot_height = height - plot_top - plot_bottom
    axis = sequence.get("verticalAxis")
    if not isinstance(axis, dict) or axis.get("kind") != "measured_depth":
        raise ValueError("井序列缺少真实MD轴")
    kind = str(sequence.get("kind") or "")
    scientific_status = str(sequence.get("scientificStatus") or "unknown")
    scientifically_validated = scientific_status.casefold().startswith("validated")
    scientific_badges = (
        '<b class="well-badge safe">数据域内已验证</b>'
        if scientifically_validated
        else '<b class="well-badge candidate">实验候选 · 当前工区未定量验收</b>'
    ) + (
        '<small class="well-status-code">scientific_status: '
        f'{html.escape(scientific_status)}</small>'
    )
    fragments: list[str] = [
        '<div class="cigvis-well-view"><section class="well-chart-card">',
        '<div class="well-chart-heading"><div><span>MEASURED DEPTH RESULT</span>',
        f'<h2>{html.escape(_public_visualization_text(sequence.get("name"), "井序列成果"))}</h2></div>',
    ]
    if kind == "property_curve":
        curve = sequence.get("curve")
        if not isinstance(curve, dict):
            raise ValueError("储层物性井序列缺少主曲线")
        md = np.asarray(axis.get("values", []), dtype=float)
        primary = np.asarray(curve.get("primaryValues", []), dtype=float)
        if (
            md.ndim != 1
            or md.size < 2
            or primary.shape != md.shape
            or not np.isfinite(md).all()
            or not np.isfinite(primary).all()
            or not np.all(np.diff(md) > 0)
        ):
            raise ValueError("储层物性主曲线与真实MD轴不一致")
        lower = np.asarray(curve.get("lowerValues", []), dtype=float)
        upper = np.asarray(curve.get("upperValues", []), dtype=float)
        has_uncertainty = (
            lower.shape == md.shape
            and upper.shape == md.shape
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
        )
        domain_values = (
            np.concatenate((lower, upper, primary)) if has_uncertainty else primary
        )
        x_min = float(np.min(domain_values))
        x_max = float(np.max(domain_values))
        if x_max <= x_min:
            padding = max(abs(x_min) * 0.05, 0.5)
            x_min -= padding
            x_max += padding
        else:
            padding = (x_max - x_min) * 0.08
            x_min -= padding
            x_max += padding
        md_min, md_max = float(md[0]), float(md[-1])
        plot_left, plot_right = 158.0, 92.0
        plot_width = width - plot_left - plot_right

        def x(value: float) -> float:
            return plot_left + (value - x_min) / (x_max - x_min) * plot_width

        def y(value: float) -> float:
            return plot_top + (value - md_min) / (md_max - md_min) * plot_height

        target = html.escape(str(curve.get("label") or curve.get("target") or "物性"))
        unit = html.escape(str(curve.get("unit") or ""))
        uncertainty_badge = (
            '<b class="well-badge uncertainty">主值 ± 预测标准差</b>'
            if has_uncertainty
            else '<b class="well-badge">物理主值</b>'
        )
        fragments.extend(
            (
                f'<div class="well-heading-badges">{scientific_badges}'
                f'{uncertainty_badge}<b class="well-badge safe">非概率主图</b>'
                '</div></div>',
                f'<svg viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{target} 储层物性真实MD曲线">',
                f'<rect class="well-plot-bg" x="{plot_left}" y="{plot_top}" '
                f'width="{plot_width}" height="{plot_height}" rx="8"/>',
            )
        )
        for tick in np.linspace(0.0, 1.0, 6):
            tick_x = plot_left + tick * plot_width
            value = x_min + tick * (x_max - x_min)
            fragments.append(
                f'<line class="well-grid" x1="{tick_x:.2f}" y1="{plot_top}" '
                f'x2="{tick_x:.2f}" y2="{plot_top + plot_height}"/>'
                f'<text class="well-axis-text" x="{tick_x:.2f}" y="{height - 18}" '
                f'text-anchor="middle">{value:.4g}</text>'
            )
        for tick in np.linspace(0.0, 1.0, 7):
            tick_y = plot_top + tick * plot_height
            value = md_min + tick * (md_max - md_min)
            fragments.append(
                f'<line class="well-grid" x1="{plot_left}" y1="{tick_y:.2f}" '
                f'x2="{plot_left + plot_width}" y2="{tick_y:.2f}"/>'
                f'<text class="well-axis-text" x="{plot_left - 14}" y="{tick_y + 4:.2f}" '
                f'text-anchor="end">{value:.5g}</text>'
            )
        if has_uncertainty:
            band_points = [
                *(f"{x(value):.2f},{y(depth):.2f}" for depth, value in zip(md, lower, strict=True)),
                *(
                    f"{x(value):.2f},{y(depth):.2f}"
                    for depth, value in zip(md[::-1], upper[::-1], strict=True)
                ),
            ]
            fragments.append(
                f'<polygon class="well-uncertainty-band" points="{" ".join(band_points)}"/>'
            )
        curve_points = " ".join(
            f"{x(value):.2f},{y(depth):.2f}"
            for depth, value in zip(md, primary, strict=True)
        )
        fragments.extend(
            (
                f'<polyline class="well-primary-curve" points="{curve_points}"/>',
                f'<text class="well-axis-title" x="{plot_left + plot_width / 2:.2f}" '
                f'y="{height - 2}" text-anchor="middle">{target}'
                f'{" / " + unit if unit else ""}</text>',
                f'<text class="well-axis-title" transform="translate(24 '
                f'{plot_top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">'
                '真实测量深度 MD / m（向下增大）</text>',
                '</svg>',
            )
        )
    elif kind == "categorical_intervals":
        raw_intervals = sequence.get("intervals")
        if not isinstance(raw_intervals, list) or not raw_intervals:
            raise ValueError("确定性井序列缺少MD层段")
        top_md = float(axis.get("minimum"))
        bottom_md = float(axis.get("maximum"))
        if not np.isfinite([top_md, bottom_md]).all() or bottom_md <= top_md:
            raise ValueError("确定性井序列MD范围无效")
        is_fracture = sequence.get("taskId") == "fracture_development"
        subject = "裂缝相对发育层段" if is_fracture else "一维地震相层段"
        fragments.extend(
            (
                f'<div class="well-heading-badges">{scientific_badges}'
                f'<b class="well-badge">{subject}</b>'
                '<b class="well-badge safe">确定性 · 无概率</b></div></div>',
                f'<svg viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{subject}真实MD层段">',
                f'<rect class="well-plot-bg" x="188" y="{plot_top}" width="420" '
                f'height="{plot_height}" rx="8"/>',
            )
        )
        fracture_palette = {0: "#75a9c5", 1: "#e3a72f", 2: "#d74735"}
        facies_palette = (
            "#335caa",
            "#2f8f83",
            "#e0a93b",
            "#d66a4e",
            "#7756b3",
            "#35a661",
            "#bf568b",
            "#718096",
        )
        legend: dict[tuple[int, str], str] = {}
        for raw in raw_intervals:
            if not isinstance(raw, dict):
                raise ValueError("确定性井序列层段记录无效")
            top = float(raw.get("topMdM"))
            bottom = float(raw.get("bottomMdM"))
            code = int(raw.get("code"))
            label = str(raw.get("label") or "")
            label_zh = str(raw.get("labelZh") or label)
            if not np.isfinite([top, bottom]).all() or bottom < top or not label:
                raise ValueError("确定性井序列层段边界或标签无效")
            y0 = plot_top + (top - top_md) / (bottom_md - top_md) * plot_height
            y1 = plot_top + (bottom - top_md) / (bottom_md - top_md) * plot_height
            block_height = max(y1 - y0, 1.2)
            color = (
                fracture_palette.get(code, "#94a3b8")
                if is_fracture
                else facies_palette[abs(code) % len(facies_palette)]
            )
            display_label = f"{label_zh} / {label}" if label_zh != label else label
            legend[(code, display_label)] = color
            fragments.append(
                f'<rect x="189" y="{y0:.2f}" width="418" height="{block_height:.2f}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            if block_height >= 18:
                fragments.append(
                    f'<text class="well-interval-label" x="630" y="{y0 + 14:.2f}">'
                    f'{html.escape(display_label)} · MD {top:.5g}–{bottom:.5g} m</text>'
                )
        for tick in np.linspace(0.0, 1.0, 7):
            tick_y = plot_top + tick * plot_height
            value = top_md + tick * (bottom_md - top_md)
            fragments.append(
                f'<line class="well-tick" x1="178" y1="{tick_y:.2f}" x2="188" '
                f'y2="{tick_y:.2f}"/><text class="well-axis-text" x="166" '
                f'y="{tick_y + 4:.2f}" text-anchor="end">{value:.5g}</text>'
            )
        fragments.append(
            f'<text class="well-axis-title" transform="translate(24 '
            f'{plot_top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">'
            '真实测量深度 MD / m（向下增大）</text></svg>'
        )
        fragments.append('<div class="well-track-legend">')
        for (code, label), color in sorted(legend.items()):
            fragments.append(
                f'<span><i style="background:{color}"></i>{html.escape(label)} '
                f'<small>代码 {code}</small></span>'
            )
        fragments.append("</div>")
    else:
        raise ValueError("三维可视化组件不支持该井序列类型")
    fragments.append(
        '<p class="well-display-note">MD 直接来自封存标准成果；主图不显示概率。'
        '切换左侧井名可查看同一任务的其他井。</p></section></div>'
    )
    return "".join(fragments)


_FAULTSEG_GRID_SPECS: dict[str, dict[str, Any]] = {
    "well-seismic.faultseg-representative-grid.v1": {
        "grid_shape_zyx": (4, 3, 3),
        "block_count": 36,
        "display_scale_algorithm": "maximum_of_36_block_abs_p99_histograms_v1",
    },
    "well-seismic.faultseg-representative-grid.v2": {
        "grid_shape_zyx": (8, 4, 4),
        "block_count": 128,
        "display_scale_algorithm": "maximum_of_128_block_abs_p99_histograms_v2",
    },
}
_FAULTSEG_BLOCK_SHAPE_ZYX = (128, 128, 128)
_FAULTSEG_GRID_ORDER = "Z_then_INLINE_then_CROSSLINE"
_FAULTSEG_FIXED_THRESHOLD = 0.518


def _integer_triplet(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label}必须是三个整数")
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是三个整数") from exc
    if any(item < 0 for item in values):
        raise ValueError(f"{label}不能包含负数")
    return values  # type: ignore[return-value]


def _fraction(value: Any, label: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是有限比例") from exc
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{label}必须位于[0, 1]")
    return fraction


def _faultseg_axis_ranges(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_ranges = block.get("axisCoordinateRanges")
    if not isinstance(raw_ranges, dict) or set(raw_ranges) != {
        "Z",
        "INLINE",
        "CROSSLINE",
    }:
        raise ValueError("断层识别代表块必须声明Z/INLINE/CROSSLINE坐标范围")
    ranges: dict[str, dict[str, Any]] = {}
    for axis in ("Z", "INLINE", "CROSSLINE"):
        raw = raw_ranges.get(axis)
        if not isinstance(raw, dict):
            raise ValueError(f"断层识别代表块{axis}坐标范围无效")
        try:
            index_start = int(raw["index_start"])
            index_end = int(raw["index_end_inclusive"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"断层识别代表块{axis}索引范围无效") from exc
        if index_start < 0 or index_end < index_start:
            raise ValueError(f"断层识别代表块{axis}索引范围无效")
        ranges[axis] = {
            **raw,
            "index_start": index_start,
            "index_end_inclusive": index_end,
        }
    return ranges


def _validated_faultseg_grid(payload: dict[str, Any]) -> dict[str, Any]:
    raw_grid = payload.get("faultSegGrid")
    if not isinstance(raw_grid, dict):
        raise ValueError("断层识别代表块位置图缺少有效合同")
    grid_spec = _FAULTSEG_GRID_SPECS.get(str(raw_grid.get("contractVersion") or ""))
    if grid_spec is None:
        raise ValueError("断层识别代表块位置图合同版本不受支持")
    expected_grid_shape = tuple(grid_spec["grid_shape_zyx"])
    block_count = int(grid_spec["block_count"])
    if raw_grid.get("scope") != "representative_sampling":
        raise ValueError("断层识别位置图必须明确为固定代表性抽样")
    if raw_grid.get("isFullVolume") is not False:
        raise ValueError("断层识别代表块位置图不得声明为全体积")
    grid_shape = _integer_triplet(raw_grid.get("gridShapeZYX"), "gridShapeZYX")
    if grid_shape != expected_grid_shape:
        raise ValueError(f"断层识别代表块位置图必须为{'×'.join(map(str, expected_grid_shape))}")
    if _integer_triplet(raw_grid.get("blockShapeZYX"), "blockShapeZYX") != _FAULTSEG_BLOCK_SHAPE_ZYX:
        raise ValueError("断层识别代表块必须为128×128×128")
    source_shape = _integer_triplet(raw_grid.get("sourceShapeZYX"), "sourceShapeZYX")
    if any(value < 128 for value in source_shape):
        raise ValueError("断层识别源体积小于128³代表块")
    if raw_grid.get("gridOrder") != _FAULTSEG_GRID_ORDER:
        raise ValueError("断层识别代表块位置图顺序不受支持")
    try:
        threshold = float(raw_grid.get("threshold"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("断层识别代表块阈值回执无效") from exc
    if threshold != _FAULTSEG_FIXED_THRESHOLD:
        raise ValueError("断层识别代表块必须使用固定共享阈值0.518")
    if str(raw_grid.get("normalization") or "") != "per_patch_zscore":
        raise ValueError("断层识别代表块必须使用共享per_patch_zscore归一化")
    if int(raw_grid.get("forwardCallsTotal") or 0) != block_count:
        raise ValueError(
            f"断层识别代表块位置图必须回执{block_count}次独立前向"
        )
    if raw_grid.get("interBlockStitching") is not False:
        raise ValueError("断层识别代表块不得声明块间拼接")
    union_coverage_fraction = _fraction(
        raw_grid.get("representativeUnionCoverageFraction"),
        "断层识别代表块工区并集覆盖比例",
    )
    if union_coverage_fraction <= 0.0:
        raise ValueError("断层识别代表块工区并集覆盖比例必须为正")
    receipt_name = str(raw_grid.get("receiptFileName") or "").strip()
    if not receipt_name:
        receipt_path = str(raw_grid.get("receiptPath") or "").strip()
        receipt_name = Path(receipt_path).name if receipt_path else ""
    if not receipt_name:
        raise ValueError("断层识别代表块位置图缺少推理回执")
    receipt_sha256 = str(raw_grid.get("receiptSha256") or "").strip().lower()
    if receipt_sha256 and (
        len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise ValueError("断层识别代表块位置图推理回执摘要无效")
    try:
        display_amplitude_scale = float(raw_grid.get("displayAmplitudeScale"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("断层识别代表块缺少共享地震增益回执") from exc
    display_scale_receipt = raw_grid.get("displayAmplitudeScaleReceipt")
    if (
        not np.isfinite(display_amplitude_scale)
        or display_amplitude_scale <= 0.0
        or not isinstance(display_scale_receipt, dict)
        or display_scale_receipt.get("algorithm")
        != grid_spec["display_scale_algorithm"]
        or int(display_scale_receipt.get("block_count") or 0) != block_count
        or display_scale_receipt.get("model_input_normalization_modified")
        is not False
    ):
        raise ValueError("断层识别代表块共享地震增益回执无效")

    raw_blocks = raw_grid.get("blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != block_count:
        raise ValueError(f"断层识别代表块位置图必须包含{block_count}个独立块")
    blocks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_positions: set[tuple[int, int, int]] = set()
    for ordinal, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise ValueError("断层识别代表块位置图包含无效块")
        inline_crossline_plane = grid_shape[1] * grid_shape[2]
        expected_position = (
            ordinal // inline_crossline_plane,
            (ordinal % inline_crossline_plane) // grid_shape[2],
            ordinal % grid_shape[2],
        )
        position = _integer_triplet(
            raw_block.get("gridIndexZYX"),
            f"断层识别代表块{ordinal} gridIndexZYX",
        )
        expected_id = (
            f"z{expected_position[0]:02d}_i{expected_position[1]:02d}_x{expected_position[2]:02d}"
        )
        block_id = str(raw_block.get("blockId") or "")
        try:
            block_ordinal = int(raw_block.get("ordinal"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"断层识别代表块{expected_id}序号无效") from exc
        if position != expected_position or block_ordinal != ordinal or block_id != expected_id:
            raise ValueError("断层识别代表块位置图未按固定Z/Inline/Crossline顺序排列")
        if block_id in seen_ids or position in seen_positions:
            raise ValueError("断层识别代表块位置图包含重复块")
        seen_ids.add(block_id)
        seen_positions.add(position)
        block_shape = _integer_triplet(
            raw_block.get("shapeZYX", raw_grid.get("blockShapeZYX")),
            f"断层识别代表块{block_id} shapeZYX",
        )
        if block_shape != _FAULTSEG_BLOCK_SHAPE_ZYX:
            raise ValueError(f"断层识别代表块{block_id}形状必须为128×128×128")
        start = _integer_triplet(
            raw_block.get("sourceStartZYX"),
            f"断层识别代表块{block_id} sourceStartZYX",
        )
        end_exclusive = _integer_triplet(
            raw_block.get("sourceEndZYXExclusive"),
            f"断层识别代表块{block_id} sourceEndZYXExclusive",
        )
        end_inclusive = _integer_triplet(
            raw_block.get("sourceEndZYXInclusive"),
            f"断层识别代表块{block_id} sourceEndZYXInclusive",
        )
        expected_end_exclusive = tuple(
            start_value + size
            for start_value, size in zip(start, block_shape, strict=True)
        )
        if (
            end_exclusive != expected_end_exclusive
            or end_inclusive != tuple(value - 1 for value in end_exclusive)
            or any(
                end > available
                for end, available in zip(end_exclusive, source_shape, strict=True)
            )
        ):
            raise ValueError(f"断层识别代表块{block_id}源坐标边界无效")
        ranges = _faultseg_axis_ranges(raw_block)
        for axis, axis_start, axis_end in zip(
            ("Z", "INLINE", "CROSSLINE"),
            start,
            end_inclusive,
            strict=True,
        ):
            if (
                ranges[axis]["index_start"] != axis_start
                or ranges[axis]["index_end_inclusive"] != axis_end
            ):
                raise ValueError(f"断层识别代表块{block_id}{axis}坐标回执不一致")
        if int(raw_block.get("forwardCalls") or 0) != 1:
            raise ValueError(f"断层识别代表块{block_id}必须回执一次独立前向")
        valid_trace_ratio = _fraction(
            raw_block.get("validTraceRatio"), f"断层识别代表块{block_id}有效道比例"
        )
        fault_fraction = _fraction(
            raw_block.get("faultFraction"), f"断层识别代表块{block_id}断层比例"
        )
        blocks.append(
            {
                **raw_block,
                "blockId": block_id,
                "ordinal": ordinal,
                "gridIndexZYX": list(position),
                "axisCoordinateRanges": ranges,
                "validTraceRatio": valid_trace_ratio,
                "faultFraction": fault_fraction,
            }
        )
    default_block_id = str(raw_grid.get("defaultBlockId") or "")
    expected_default_block_id = (
        f"z{(grid_shape[0] - 1) // 2:02d}_"
        f"i{(grid_shape[1] - 1) // 2:02d}_"
        f"x{(grid_shape[2] - 1) // 2:02d}"
    )
    if default_block_id != expected_default_block_id or default_block_id not in seen_ids:
        raise ValueError(
            "断层识别代表块位置图默认块必须是中心代表块"
            f"{expected_default_block_id}"
        )
    return {
        **raw_grid,
        "blockCount": block_count,
        "threshold": threshold,
        "displayAmplitudeScale": display_amplitude_scale,
        "representativeUnionCoverageFraction": union_coverage_fraction,
        "receiptFileName": receipt_name,
        "defaultBlockId": default_block_id,
        "blocks": blocks,
    }


def _range_text(block: dict[str, Any], axis: str) -> str:
    axis_range = dict(block["axisCoordinateRanges"][axis])
    coordinate_start = axis_range.get("coordinate_start")
    coordinate_end = axis_range.get("coordinate_end")
    if coordinate_start is not None and coordinate_end is not None:
        try:
            return f"{float(coordinate_start):g}–{float(coordinate_end):g}"
        except (TypeError, ValueError, OverflowError):
            pass
    return (
        f"{int(axis_range['index_start'])}–"
        f"{int(axis_range['index_end_inclusive'])}"
    )


def _faultseg_grid_fragment(
    payload: dict[str, Any],
    *,
    task_id: str,
    asset_index: int,
    embed: bool,
) -> str:
    grid = _validated_faultseg_grid(payload)
    block_hrefs = {
        str(block["blockId"]): "/统一数据可视化?"
        + urlencode(
            {
                "task_id": task_id,
                "asset": asset_index,
                "block": str(block["blockId"]),
                "embed": 1 if embed else 0,
            }
        )
        for block in grid["blocks"]
    }
    webgl_grid = {
        **grid,
        "blocks": [
            {**block, "selectionHref": block_hrefs[str(block["blockId"])]}
            for block in grid["blocks"]
        ],
    }
    webgl_payload = {**payload, "faultSegGrid": webgl_grid}
    webgl_fragment = _webgl_volume_fragment(webgl_payload, task_id=task_id)
    grid_shape = tuple(int(value) for value in grid["gridShapeZYX"])
    block_count = int(grid["blockCount"])
    by_depth: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(grid_shape[0])
    }
    for block in grid["blocks"]:
        by_depth[int(block["gridIndexZYX"][0])].append(block)

    depth_fragments: list[str] = []
    for depth_index in range(grid_shape[0]):
        depth_blocks = by_depth[depth_index]
        z_range = _range_text(depth_blocks[0], "Z")
        cards: list[str] = []
        for block in depth_blocks:
            block_id = str(block["blockId"])
            default_class = (
                " default" if block_id == grid["defaultBlockId"] else ""
            )
            default_badge = (
                '<em title="平台默认展示块">默认</em>'
                if block_id == grid["defaultBlockId"]
                else ""
            )
            cards.append(
                f'<a class="fault-grid-block{default_class}" data-block-id="{html.escape(block_id)}" '
                f'href="{html.escape(block_hrefs[block_id], quote=True)}" '
                f'title="打开{html.escape(block_id)}的128³独立块">'
                f'<span><b>{html.escape(block_id)}</b>{default_badge}</span>'
                f'<small>I {_range_text(block, "INLINE")} · X {_range_text(block, "CROSSLINE")}</small>'
                f'<small>有效道 {float(block["validTraceRatio"]):.1%} · 断层 {float(block["faultFraction"]):.2%}</small>'
                "</a>"
            )
        depth_fragments.append(
            f'<section class="fault-grid-depth" data-depth-index="{depth_index}">'
            '<header><span>垂向层</span>'
            f'<b>Z{depth_index + 1} / {grid_shape[0]}</b><small>Z {html.escape(z_range)}</small></header>'
            f'<div class="fault-grid-matrix" style="grid-template-columns:repeat({grid_shape[2]},minmax(0,1fr));grid-template-rows:repeat({grid_shape[1]},minmax(0,1fr))">'
            f'{"".join(cards)}</div></section>'
        )

    receipt_name = "断层识别推理回执"
    source_shape = "×".join(str(value) for value in grid["sourceShapeZYX"])
    coverage = float(grid["representativeUnionCoverageFraction"])
    return (
        '<section class="fault-grid-view" aria-label="断层识别固定代表块位置图">'
        '<header class="fault-grid-heading"><div><span>断层识别代表块</span>'
        f'<h2>完整工区三维地震预览 · {block_count} 块可拾取位置</h2></div>'
        f'<div class="fault-grid-facts"><b>{grid_shape[0]} × {grid_shape[1]} × {grid_shape[2]}</b>'
        f'<small>完整工区 {html.escape(source_shape)} · 并集覆盖 {coverage:.2%}</small>'
        '<small>每块 128³ · 阈值固定 0.518</small></div></header>'
        '<p class="fault-grid-notice"><strong>固定代表性抽样，不是全体积。</strong>'
        f'{block_count} 块独立推理且不拼接；部分抽样窗可发生几何重叠，其余区域未覆盖，'
        '不构成连续全工区预测。点击三维块仅载入对应完整 128³ 结果。</p>'
        f'<div class="fault-grid-3d-stage">{webgl_fragment}'
        '<p class="fault-grid-3d-help" id="fault-grid-3d-help">拖动旋转工区；悬停查看回执；'
        '单击块进入二维 Inline。键盘方向键选择，Enter / Space 打开。</p></div>'
        '<details class="fault-grid-fallback" id="fault-grid-fallback">'
        f'<summary>二维 {grid_shape[0]} 层 × {grid_shape[1]}×{grid_shape[2]} 位置列表（WebGL 不可用兜底）</summary>'
        f'<div class="fault-grid-depths" style="grid-template-rows:repeat({grid_shape[0]},minmax(108px,1fr))">'
        f'{"".join(depth_fragments)}</div></details>'
        f'<footer class="fault-grid-receipt"><span>代表块并集覆盖完整工区 {coverage:.2%} · '
        '共享显示合同：统一增益 / 配色 · 固定阈值 0.518</span>'
        f'<span>推理回执：{html.escape(receipt_name)} · {block_count} 次独立前向</span></footer></section>'
    )


def _well_result_interval_color(task_id: str, code: int, raw_color: object) -> str:
    color = str(raw_color or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return color.casefold()
    palette = _WELL_RESULT_PALETTES.get(task_id, _WELL_RESULT_PALETTES["facies_1d"])
    return palette[abs(int(code)) % len(palette)]


def _well_result_binding_and_placement(
    raw_track: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Return the fail-closed trajectory binding shared by all well tracks."""

    raw_binding = raw_track.get("binding")
    binding: dict[str, Any] | None = None
    if isinstance(raw_binding, dict):
        status = str(raw_binding.get("status") or "unmatched")
        if status not in {"matched", "unmatched", "ambiguous"}:
            status = "unmatched"
        trajectory_index: int | None = None
        raw_index = raw_binding.get("trajectoryIndex")
        if (
            isinstance(raw_index, (int, np.integer))
            and not isinstance(raw_index, (bool, np.bool_))
            and int(raw_index) >= 0
        ):
            trajectory_index = int(raw_index)
        if status == "matched" and trajectory_index is None:
            status = "unmatched"
        binding = {
            "status": status,
            "axis": "measured_depth_m",
            "trajectoryIndex": trajectory_index,
            "trajectoryWellUid": str(raw_binding.get("trajectoryWellUid") or ""),
            "trajectoryWellId": str(raw_binding.get("trajectoryWellId") or ""),
            "measuredDepthTrajectoryAvailable": raw_binding.get(
                "measuredDepthTrajectoryAvailable"
            )
            is True,
            "reason": str(raw_binding.get("reason") or ""),
        }
    raw_display = raw_track.get("display")
    raw_display = raw_display if isinstance(raw_display, dict) else {}
    twt_placement = str(raw_display.get("twtPlacement") or "md_only")
    if twt_placement not in {"accepted", "md_only"}:
        twt_placement = "md_only"
    if binding is not None and binding["status"] != "matched":
        twt_placement = "md_only"
    return binding, twt_placement


def _well_property_result_track(raw_track: dict[str, Any]) -> dict[str, Any] | None:
    """Bound one continuous physical-property curve for the linked HUD.

    The sealed MD samples and physical values are preserved verbatim.  Only the
    browser's horizontal plotting coordinate is linearly normalized per well;
    that display transform is declared in the payload and never presented as a
    model or data normalization.
    """

    raw_axis = raw_track.get("verticalAxis")
    raw_curve = raw_track.get("curve")
    if not isinstance(raw_axis, dict) or not isinstance(raw_curve, dict):
        return None
    if str(raw_axis.get("kind") or "") != "measured_depth":
        return None
    try:
        md = np.asarray(raw_axis.get("values"), dtype=np.float64)
        primary = np.asarray(raw_curve.get("primaryValues"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        md.ndim != 1
        or primary.ndim != 1
        or md.size < 2
        or md.size > _MAXIMUM_WELL_RESULT_CURVE_POINTS
        or primary.shape != md.shape
        or not np.isfinite(md).all()
        or not np.isfinite(primary).all()
        or not np.all(np.diff(md) > 0.0)
    ):
        return None

    well_id = str(
        raw_track.get("wellId")
        or raw_track.get("well_id")
        or raw_track.get("wellUid")
        or raw_track.get("well_uid")
        or ""
    ).strip()
    if not well_id:
        return None
    task_id = str(
        raw_track.get("taskId")
        or raw_track.get("task_id")
        or raw_track.get("interpretationTaskId")
        or "well_property"
    ).strip()
    target = str(raw_curve.get("target") or raw_curve.get("label") or "").strip()
    label = str(raw_curve.get("label") or target or "储层物性").strip()
    unit = str(raw_curve.get("unit") or "").strip()
    if not target or len(target) > 64 or len(label) > 96 or len(unit) > 32:
        return None

    raw_lower = raw_curve.get("lowerValues")
    raw_upper = raw_curve.get("upperValues")
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    if raw_lower is not None or raw_upper is not None:
        if raw_lower is None or raw_upper is None:
            return None
        try:
            lower = np.asarray(raw_lower, dtype=np.float64)
            upper = np.asarray(raw_upper, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            lower.shape != md.shape
            or upper.shape != md.shape
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower > primary)
            or np.any(primary > upper)
        ):
            return None

    domain_values = (
        np.concatenate((lower, primary, upper))
        if lower is not None and upper is not None
        else primary
    )
    value_minimum = float(np.min(domain_values))
    value_maximum = float(np.max(domain_values))
    display_minimum = value_minimum
    display_maximum = value_maximum
    if display_maximum <= display_minimum:
        padding = max(abs(display_minimum) * 0.05, 0.5)
        display_minimum -= padding
        display_maximum += padding

    binding, twt_placement = _well_result_binding_and_placement(raw_track)
    curve: dict[str, Any] = {
        "target": target,
        "label": label,
        "unit": unit,
        "primaryValues": primary.tolist(),
        "primarySemantics": str(
            raw_curve.get("primarySemantics") or "model_regression"
        )[:128],
        "valueMinimum": value_minimum,
        "valueMaximum": value_maximum,
        "displayNormalization": {
            "method": "per_well_linear_display_only",
            "minimum": display_minimum,
            "maximum": display_maximum,
            "rawValuesPreserved": True,
            "appliedToModelOutput": False,
        },
    }
    if lower is not None and upper is not None:
        curve["lowerValues"] = lower.tolist()
        curve["upperValues"] = upper.tolist()

    subject = f"{label} 储层物性" if "储层物性" not in label else label
    return {
        "kind": "property_curve",
        "name": str(raw_track.get("name") or f"{well_id} · {subject}").strip(),
        "wellId": well_id,
        "taskId": task_id,
        "subject": subject,
        "scientificStatus": str(raw_track.get("scientificStatus") or "unknown"),
        "verticalAxis": {
            "kind": "measured_depth",
            "label": "MD",
            "unit": "m",
            "minimum": float(md[0]),
            "maximum": float(md[-1]),
            "values": md.tolist(),
            "source": "sealed_result_samples",
        },
        "curve": curve,
        "sampleCount": int(md.size),
        "display": {
            "mainPlot": "continuous_physical_property_curve",
            "probabilityDisplayed": False,
            "attachment": "well_callout",
            "twtPlacement": twt_placement,
            "horizontalNormalization": "per_well_linear_display_only",
        },
        **({"binding": binding} if binding is not None else {}),
    }


def _well_result_track(raw_track: object) -> dict[str, Any] | None:
    """Bound one deterministic MD track before it enters the embedded viewer.

    Categorical tracks receive deterministic interval boundaries and display
    labels only.  Property tracks retain their sealed physical values and use
    an explicitly display-only horizontal transform.
    """

    if not isinstance(raw_track, dict):
        return None
    kind = str(raw_track.get("kind") or "categorical_intervals").strip()
    if kind == "property_curve":
        return _well_property_result_track(raw_track)
    raw_intervals = raw_track.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        return None
    if len(raw_intervals) > _MAXIMUM_WELL_RESULT_INTERVALS:
        return None
    if kind not in {"categorical_intervals", "classification_intervals"}:
        return None
    task_id = str(
        raw_track.get("taskId")
        or raw_track.get("task_id")
        or raw_track.get("interpretationTaskId")
        or ""
    ).strip()
    subject = str(raw_track.get("subject") or _WELL_RESULT_SUBJECTS.get(task_id) or "井侧分类").strip()
    well_id = str(
        raw_track.get("wellId")
        or raw_track.get("well_id")
        or raw_track.get("wellUid")
        or raw_track.get("well_uid")
        or ""
    ).strip()
    if not well_id:
        return None

    intervals: list[dict[str, Any]] = []
    for raw_interval in raw_intervals:
        if not isinstance(raw_interval, dict):
            return None
        try:
            top = float(
                raw_interval.get("topMdM", raw_interval.get("top_md_m"))
            )
            bottom = float(
                raw_interval.get("bottomMdM", raw_interval.get("bottom_md_m"))
            )
            numeric_code = float(
                raw_interval.get("code", raw_interval.get("class_code", 0))
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(numeric_code) or not numeric_code.is_integer():
            return None
        code = int(numeric_code)
        label = str(
            raw_interval.get("labelZh")
            or raw_interval.get("label_zh")
            or raw_interval.get("label")
            or raw_interval.get("class_name")
            or ""
        ).strip()
        secondary_label = str(raw_interval.get("label") or "").strip()
        if (
            not np.isfinite(top)
            or not np.isfinite(bottom)
            or bottom <= top
            or not label
        ):
            return None
        intervals.append(
            {
                "topMdM": top,
                "bottomMdM": bottom,
                "code": code,
                "label": label,
                "secondaryLabel": (
                    secondary_label if secondary_label and secondary_label != label else ""
                ),
                "color": _well_result_interval_color(
                    task_id, code, raw_interval.get("color")
                ),
            }
        )
    intervals.sort(key=lambda item: (item["topMdM"], item["bottomMdM"]))
    if any(
        current["topMdM"] < previous["bottomMdM"] - 1e-6
        for previous, current in zip(intervals, intervals[1:], strict=False)
    ):
        return None
    minimum = min(item["topMdM"] for item in intervals)
    maximum = max(item["bottomMdM"] for item in intervals)
    if maximum <= minimum:
        return None
    binding, twt_placement = _well_result_binding_and_placement(raw_track)
    return {
        "kind": "categorical_intervals",
        "name": str(raw_track.get("name") or f"{well_id} · {subject}").strip(),
        "wellId": well_id,
        "taskId": task_id,
        "subject": subject,
        "scientificStatus": str(raw_track.get("scientificStatus") or "unknown"),
        "verticalAxis": {
            "kind": "measured_depth",
            "label": "MD",
            "unit": "m",
            "minimum": minimum,
            "maximum": maximum,
            "source": "sealed_interval_boundaries",
        },
        "intervals": intervals,
        "intervalCount": len(intervals),
        "display": {
            "mainPlot": "deterministic_md_intervals",
            "probabilityDisplayed": False,
            "scoreDisplayed": False,
            "attachment": "well_callout",
            "twtPlacement": twt_placement,
        },
        **({"binding": binding} if binding is not None else {}),
    }


def _well_result_tracks(
    preview: dict[str, Any], volume_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates: list[object] = []
    for source in (
        volume_payload.get("wellResultTracks"),
        preview.get("wellResultTracks"),
        preview.get("wellSequences"),
    ):
        if isinstance(source, list):
            candidates.extend(source)
    tracks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float]] = set()
    for raw_track in candidates[: _MAXIMUM_WELL_RESULT_TRACKS * 3]:
        track = _well_result_track(raw_track)
        if track is None:
            continue
        axis = track["verticalAxis"]
        identity = (
            str(track["wellId"]).strip().casefold(),
            str(track["taskId"]).strip().casefold(),
            float(axis["minimum"]),
            float(axis["maximum"]),
        )
        if identity in seen:
            continue
        seen.add(identity)
        tracks.append(track)
        if len(tracks) >= _MAXIMUM_WELL_RESULT_TRACKS:
            break
    return tracks


def _asset_catalog(preview: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        *(
            (
                "fault_grid"
                if str(item.get("assetKind") or "") == "faultseg_grid"
                else "volume",
                item,
            )
            for item in preview.get("volumes", [])
        ),
        *(("line", item) for item in preview.get("lines2d", [])),
        *(("well_sequence", item) for item in preview.get("wellSequences", [])),
    ]


def render_cigvis_workbench(
    project_root: Path,
    preview: dict[str, Any],
    *,
    task_id: str,
    asset_index: int = 0,
    embed: bool = True,
) -> str:
    assets = _asset_catalog(preview)
    if not assets:
        raise ValueError("当前任务没有可渲染的二维或三维地震数据")
    asset_index = min(max(int(asset_index), 0), len(assets) - 1)
    kind, selected = assets[asset_index]
    status = cigvis_status(project_root)
    if not status["available"]:
        # Volume, representative-grid and well-sequence views are rendered by
        # the self-contained browser workbench.  Keep them available on a
        # clean Windows runtime even when the optional local Python CIGVis
        # scientific stack is unavailable; 2-D line rendering still requires
        # that stack and therefore remains fail-closed.
        if kind == "line":
            raise RuntimeError(status["error"] or "二维地震可视化组件不可用")
        status = {
            **status,
            "available": True,
            "version": _local_version(_cigvis_root(project_root)),
            "backend": "platform_web",
            "web_engine": "Platform WebGL2",
        }
    if kind == "volume":
        selected = {
            **selected,
            "wellResultTracks": _well_result_tracks(preview, selected),
        }
    linked_well_log_profile = (
        _well_log_layout_profile(selected.get("wellResultTracks"))
        if kind == "volume"
        else None
    )
    linked_facies_layout = linked_well_log_profile is not None
    linked_well_log_title = (
        str(linked_well_log_profile["title"])
        if linked_well_log_profile
        else "井侧预测图版"
    )
    representative_grid = (
        _validated_faultseg_grid(selected)
        if isinstance(selected.get("faultSegGrid"), dict)
        else {}
    )
    representative_block_count = int(
        representative_grid.get("blockCount")
        or len(representative_grid.get("blocks", []))
    )
    if kind == "fault_grid":
        fragment = _faultseg_grid_fragment(
            selected,
            task_id=task_id,
            asset_index=asset_index,
            embed=embed,
        )
        active_engine = "平台断层成果可视化"
        engine_warning = ""
    elif kind == "volume":
        fragment, active_engine, engine_warning = _render_volume(
            project_root,
            selected,
            task_id,
            asset_index,
        )
    elif kind == "line":
        fragment = _render_line(project_root, selected)
        active_engine = "平台二维地震成果可视化"
        engine_warning = ""
    else:
        fragment = _well_sequence_fragment(selected)
        active_engine = "平台井序列成果可视化"
        engine_warning = ""

    links: list[str] = []
    for index, (asset_kind, item) in enumerate(assets):
        query_items: dict[str, Any] = {
            "embed": 1 if embed else 0,
            "task_id": task_id,
            "asset": index,
        }
        item_selection = item.get("selectedRepresentativeBlock")
        item_block_id = (
            str(item_selection.get("blockId") or "")
            if isinstance(item_selection, dict)
            else str(item_selection or "")
        )
        if asset_kind == "volume" and item_block_id:
            query_items["block"] = item_block_id
        query = urlencode(query_items)
        label = (
            "格"
            if asset_kind == "fault_grid"
            else "3D"
            if asset_kind == "volume"
            else "井"
            if asset_kind == "well_sequence"
            else "2D"
        )
        active = " active" if index == asset_index else ""
        asset_description = (
            f"{representative_block_count}块固定代表性位置图"
            if asset_kind == "fault_grid"
            else "整体三维 / 动态切片"
            if asset_kind == "volume"
            else "真实MD确定性成果"
            if asset_kind == "well_sequence"
            else "二维地震剖面"
        )
        links.append(
            f'<a class="asset-link{active}" href="/统一数据可视化?{query}">'
            f'<span>{label}</span><div><strong>{html.escape(_public_visualization_text(item.get("name"), "未命名地震"))}</strong>'
            f'<small>{asset_description}</small></div></a>'
        )

    raw_selected_block = selected.get("selectedRepresentativeBlock")
    selected_block_id = (
        str(raw_selected_block.get("blockId") or "")
        if isinstance(raw_selected_block, dict)
        else str(raw_selected_block or "")
    )
    selected_block: dict[str, Any] = {}
    if selected_block_id:
        if not representative_grid:
            raise ValueError("断层识别选中块缺少代表性位置图合同")
        selected_block = next(
            (
                block
                for block in representative_grid["blocks"]
                if block["blockId"] == selected_block_id
            ),
            {},
        )
        if not selected_block:
            raise ValueError(
                "断层识别选中块不属于当前"
                f"{representative_block_count}块位置图"
            )
    standalone_items: dict[str, Any] = {
        "embed": 0,
        "task_id": task_id,
        "asset": asset_index,
    }
    if selected_block_id:
        standalone_items["block"] = selected_block_id
    standalone_query = urlencode(standalone_items)
    selected_shape = selected.get("cube", selected.get("image", {})).get("shape", [])
    if kind == "fault_grid":
        selected_shape = representative_grid["gridShapeZYX"]
    if kind == "well_sequence":
        selected_shape = [
            int(selected.get("sampleCount") or selected.get("intervalCount") or 0)
        ]
    selected_overlays = list(selected.get("overlays", [])) if kind == "volume" else []
    selected_surfaces = list(selected.get("surfaces", [])) if kind == "volume" else []
    selected_candidate = (
        dict(selected.get("candidateVisualization", {})) if kind == "volume" else {}
    )
    overlay_count = len(selected_overlays)
    surface_count = len(selected_surfaces)
    primary_overlay = selected_overlays[0] if selected_overlays else {}
    primary_overlay_kind = str(primary_overlay.get("kind", "")).strip().casefold()
    primary_overlay_3d = dict(primary_overlay.get("volume3D", {}))
    fault_occupancy_3d = (
        primary_overlay_kind == "mask"
        and str(primary_overlay_3d.get("samplingAggregation", "")).casefold()
        == "block_fault_fraction"
    )
    default_layer_mode = "combined"
    categorical_slice_only = primary_overlay_kind in {"class_code", "labels"}
    default_volume_view = (
        "slice"
        if (
            selected_surfaces
            or categorical_slice_only
            or dict(selected.get("sliceViewContract", {})).get("preferSlice") is True
        )
        else "whole"
    )
    overlay_ui = _overlay_ui_metadata(primary_overlay if selected_overlays else None)
    overlay_name = str(overlay_ui["display_name"])
    overlay_subject = str(overlay_ui["subject"])
    prediction_only_label = str(overlay_ui["only_label"])
    candidate_label = (
        "工程候选"
        if selected_candidate.get("display_status") == "engineering_candidate"
        else "实验候选"
    )
    horizon_candidate_meta = (
        "<div><span>候选层位</span><strong>"
        + html.escape(
            " / ".join(
                _public_visualization_text(item.get("name"), "解释面")
                for item in selected_surfaces
            )
        )
        + "</strong></div>"
        if surface_count and selected_candidate.get("renderable")
        else ""
    )
    candidate_status_meta = (
        "<div><span>科学状态</span><strong>"
        f"{html.escape(candidate_label)} · 当前工区未验收</strong></div>"
        if selected_candidate.get("renderable")
        else ""
    )
    kind_label = (
        "断层识别代表块位置图"
        if kind == "fault_grid"
        else (
            f"{overlay_name}叠加"
            if overlay_count
            else f"{surface_count} 个命名层位候选面" if surface_count else "三维地震体"
        )
        if kind == "volume"
        else "井侧确定性成果"
        if kind == "well_sequence"
        else "二维地震测线"
    )
    mode_switch = ""
    if kind == "volume":
        categorical_kind_label = "类别体"
        whole_disabled = (
            f' disabled title="{categorical_kind_label}禁止连续插值；请使用二维离散切片"'
            if categorical_slice_only
            else ""
        )
        whole_badge = (
            f"{categorical_kind_label}禁用"
            if categorical_slice_only
            else "连续等值面"
            if fault_occupancy_3d
            else "二值 NEAREST"
            if primary_overlay_kind == "mask"
            else "默认"
        )
        mode_switch = (
            '<nav class="view-mode-switch" aria-label="三维查看模式">'
            f'<button class="view-mode-button{" active" if default_volume_view == "whole" else ""}" type="button" data-view-target="whole">'
            f"<b>整体 3D</b><small>{whole_badge}</small></button>"
            f'<button class="view-mode-button{" active" if default_volume_view == "slice" else ""}" type="button" data-view-target="slice">'
            "<b>二维切片</b><small>稳定</small></button></nav>"
        )
        if whole_disabled:
            mode_switch = mode_switch.replace(
                'type="button" data-view-target="whole"',
                f'type="button" data-view-target="whole"{whole_disabled}',
                1,
            )
    layer_switch = ""
    if overlay_count:
        layer_switch = (
            '<nav class="layer-mode-switch" aria-label="预测图层显示">'
            f'<button class="layer-mode-button{" active" if default_layer_mode == "combined" else ""}" type="button" data-layer-mode="combined" aria-pressed="{str(default_layer_mode == "combined").lower()}">'
            "<b>地震 + 预测</b></button>"
            f'<button class="layer-mode-button{" active" if default_layer_mode == "prediction" else ""}" type="button" data-layer-mode="prediction" aria-pressed="{str(default_layer_mode == "prediction").lower()}">'
            f"<b>{html.escape(prediction_only_label)}</b></button></nav>"
        )
    overview_action = ""
    representative_meta = ""
    source_label = (
        "SHA-256 完整性封存"
        if kind == "well_sequence"
        else _public_visualization_text(Path(str(selected.get("path", ""))).name)
    )
    shape_label = "位置格网" if kind == "fault_grid" else (
        "成果数量" if kind == "well_sequence" else "数据形状"
    )
    if representative_grid:
        receipt_name = "断层识别推理回执"
        source_label = receipt_name
        representative_meta = (
            '<div><span>抽样合同</span><strong>'
            f'固定{representative_block_count}块 · 非全体积</strong></div>'
            f'<div><span>工区并集覆盖</span><strong>{float(representative_grid["representativeUnionCoverageFraction"]):.2%}</strong></div>'
            '<div><span>共享阈值</span><strong>0.518（不可调）</strong></div>'
            f'<div><span>推理回执</span><strong>{html.escape(receipt_name)} · '
            f'{representative_block_count}次独立前向</strong></div>'
        )
    if selected_block:
        overview_items: dict[str, Any] = {
            "task_id": task_id,
            "asset": asset_index,
            "embed": 1 if embed else 0,
        }
        overview_query = urlencode(overview_items)
        overview_action = (
            f'<a class="grid-overview-link" href="/统一数据可视化?{overview_query}">'
            f"返回{representative_block_count}块位置图</a>"
        )
        representative_meta += (
            f'<div><span>当前代表块</span><strong>{html.escape(selected_block_id)} · 128³</strong></div>'
            f'<div><span>Z 坐标范围</span><strong>{html.escape(_range_text(selected_block, "Z"))}</strong></div>'
            f'<div><span>Inline 范围</span><strong>{html.escape(_range_text(selected_block, "INLINE"))}</strong></div>'
            f'<div><span>Crossline 范围</span><strong>{html.escape(_range_text(selected_block, "CROSSLINE"))}</strong></div>'
            f'<div><span>有效道比例</span><strong>{float(selected_block["validTraceRatio"]):.2%}</strong></div>'
            f'<div><span>断层比例</span><strong>{float(selected_block["faultFraction"]):.2%}</strong></div>'
        )
        fragment += (
            '<div class="representative-block-banner"><strong>固定代表性抽样 · 非全体积</strong>'
            f'<span>{html.escape(selected_block_id)} · 当前仅载入这个完整128³块 · '
            '独立推理且未拼接（抽样窗可局部重叠） · 共享阈值0.518</span></div>'
        )
    slice_console = ""
    workbench_class = "workbench"
    if linked_facies_layout:
        workbench_class += " facies-linked-workbench"
    if embed:
        workbench_class += " embedded assets-collapsed"
    selected_display_name = _public_visualization_text(
        selected.get("name"), "未命名地震"
    )
    engine_display_name = _public_visualization_text(
        active_engine, "平台可视化引擎"
    )
    engine_fallback_note = (
        "<br>三维渲染不可用，已自动切换兼容视图" if engine_warning else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>地震成果可视化</title>
<style>
:root{{--bg:#edf2f7;--panel:#fff;--panel2:#f7f9fc;--line:#d7e0e9;--text:#203246;--muted:#6f8295;--blue:#1677e8;--green:#15966d;--scene-bg:radial-gradient(circle at 48% 42%,#f8fbfd 0%,#eef3f7 58%,#e3eaf0 100%);--scene-flat:#eef3f7;--plate-bg:rgba(255,255,255,.96)}}
*{{box-sizing:border-box}}html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:var(--scene-bg);font-family:Inter,"Microsoft YaHei UI",sans-serif;color:var(--text)}}
.workbench{{display:grid;width:100%;height:100vh;grid-template-columns:220px minmax(0,1fr);grid-template-rows:52px minmax(0,1fr);background:var(--scene-bg)}}
.workbench.with-slice-console{{grid-template-rows:52px minmax(0,1fr) 0}}.workbench.with-slice-console.slice-active{{grid-template-rows:52px minmax(0,1fr) 86px}}.workbench.assets-collapsed{{grid-template-columns:0 minmax(0,1fr)}}.workbench.assets-collapsed .sidebar{{padding:0;border:0;opacity:0;pointer-events:none}}
.topbar{{display:flex;grid-column:1/-1;gap:14px;align-items:center;padding:0 14px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);box-shadow:0 2px 10px rgba(35,58,81,.05)}}
.dataset-title{{display:flex;min-width:0;gap:9px;align-items:center;flex:1}}.dataset-title>span{{padding:3px 7px;color:{'#087b67' if kind == 'volume' else '#1267bb'};font-size:12px;font-weight:750;background:{'#e6f4ef' if kind == 'volume' else '#e8f2fc'};border-radius:4px}}.dataset-title strong{{overflow:hidden;font-size:13px;text-overflow:ellipsis;white-space:nowrap}}
.view-mode-switch{{display:flex;flex:0 0 auto;gap:4px;padding:3px;background:#e8eef5;border:1px solid #d3deea;border-radius:8px}}.view-mode-switch .view-mode-button{{display:flex;min-height:34px;gap:5px;align-items:center;padding:0 13px;color:#536b82;background:transparent;border-color:transparent;border-radius:6px}}.view-mode-button b{{font-size:12px}}.view-mode-button small{{padding:1px 4px;color:#7f91a3;font-size:12px;background:rgba(255,255,255,.65);border-radius:3px}}.view-mode-switch .view-mode-button.active{{color:#fff;background:#1677e8;border-color:#1677e8;box-shadow:0 4px 10px rgba(22,119,232,.22)}}.view-mode-switch .view-mode-button.active small{{color:#145fae;background:#fff}}.view-mode-switch .view-mode-button:disabled{{cursor:not-allowed;opacity:.55}}
.layer-mode-switch{{display:flex;flex:0 0 auto;gap:3px;padding:3px;background:#eef2f6;border:1px solid #d7e0e9;border-radius:8px}}.layer-mode-switch[hidden]{{display:none!important}}.layer-mode-switch .layer-mode-button{{min-height:30px;padding:0 9px;color:#60758a;font-size:12px;background:transparent;border-color:transparent;border-radius:6px}}.layer-mode-switch .layer-mode-button.active{{color:#fff;background:#16866f;border-color:#16866f;box-shadow:0 4px 10px rgba(22,134,111,.2)}}.layer-mode-switch .layer-mode-button:disabled{{cursor:wait;opacity:.62}}
.top-actions{{display:flex;gap:6px}}button,.top-actions a{{display:inline-flex;min-height:31px;align-items:center;padding:0 10px;color:#41586f;font-size:12px;font-weight:650;text-decoration:none;background:#f7f9fc;border:1px solid #d2dce7;border-radius:6px;cursor:pointer}}button:hover,.top-actions a:hover{{border-color:#78a9df;color:#0f66c7;background:#eef6ff}}
.sidebar{{grid-row:2/-1;min-height:0;overflow:auto;padding:12px;background:#fff;border-right:1px solid var(--line);transition:padding .18s,opacity .18s;scrollbar-width:thin;scrollbar-color:#b9c7d6 transparent}}.sidebar::-webkit-scrollbar{{width:6px}}.sidebar::-webkit-scrollbar-track{{background:transparent}}.sidebar::-webkit-scrollbar-thumb{{background:#b9c7d6;border-radius:8px}}.section-label{{display:block;margin:3px 5px 8px;color:#7b8d9f;font-size:12px;font-weight:750;letter-spacing:.12em}}.asset-list{{display:grid;gap:4px}}.asset-link{{display:grid;grid-template-columns:29px minmax(0,1fr);gap:8px;align-items:center;padding:8px;color:inherit;text-decoration:none;border:1px solid transparent;border-radius:7px}}.asset-link:hover{{background:#f3f7fb}}.asset-link.active{{background:#edf5fe;border-color:#a8c9ec}}.asset-link>span{{display:grid;width:28px;height:28px;place-items:center;color:#637b93;font-size:12px;font-weight:800;background:#edf2f7;border-radius:6px}}.asset-link.active>span{{color:#fff;background:#1677e8}}.asset-link div{{display:grid;min-width:0}}.asset-link strong{{overflow:hidden;font-size:12px;text-overflow:ellipsis;white-space:nowrap}}.asset-link small{{color:#8191a2;font-size:12px}}
.meta-card{{display:grid;gap:8px;margin-top:13px;padding:10px;background:#f7f9fc;border:1px solid #dce4ed;border-radius:8px}}.meta-card div{{display:flex;justify-content:space-between;gap:8px}}.meta-card span{{color:#7b8d9f;font-size:12px}}.meta-card strong{{color:#40566d;font-size:12px;text-align:right}}.engine{{display:flex;gap:7px;align-items:flex-start;margin-top:11px;padding:9px;color:#708499;font-size:12px;line-height:1.45;border-top:1px solid #dce4ed}}.engine i{{flex:0 0 auto;width:7px;height:7px;margin-top:3px;background:var(--green);border-radius:50%;box-shadow:0 0 0 3px rgba(21,150,109,.1)}}.engine.warning i{{background:#e6a23c;box-shadow:0 0 0 3px rgba(230,162,60,.1)}}
.viewport{{grid-column:2;grid-row:2;position:relative;min-width:0;min-height:0;overflow:hidden;background:var(--scene-bg)}}.plot-shell,.view-panel{{position:absolute;inset:0}}.view-panel[hidden]{{display:none!important}}#cigvis-plot,.plotly-graph-div{{width:100%!important;height:100%!important}}.cigvis-viser-view,.cigvis-viser-view iframe{{display:block;width:100%;height:100%;border:0}}.cigvis-viser-view{{position:relative;background:var(--scene-bg)}}.viser-loading{{position:absolute;inset:0;z-index:2;display:grid;place-items:center;color:#60758a;font-size:12px;background:var(--scene-bg)}}.viser-loading[hidden]{{display:none}}.cigvis-line-view{{display:grid;width:100%;height:100%;place-items:center;padding:12px;background:var(--scene-bg)}}.cigvis-line-view img{{display:block;max-width:100%;max-height:100%;object-fit:contain;border:1px solid #c1cfda;border-radius:7px;box-shadow:0 12px 34px rgba(30,55,80,.1)}}
.cigvis-well-view{{width:100%;height:100%;overflow:auto;padding:22px;background:var(--scene-bg)}}.well-chart-card{{max-width:1180px;min-height:100%;margin:auto;padding:20px 24px;background:var(--plate-bg);border:1px solid #c7d5df;border-radius:14px;box-shadow:0 16px 42px rgba(31,55,79,.1)}}.well-chart-heading{{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:10px}}.well-chart-heading span{{color:#16866f;font-size:12px;font-weight:800;letter-spacing:.15em}}.well-chart-heading h2{{margin:3px 0 0;font-size:18px}}.well-heading-badges{{display:flex;max-width:62%;flex-wrap:wrap;justify-content:flex-end;gap:7px}}.well-badge{{padding:5px 8px;color:#315f82;font-size:12px;background:#e8f3fb;border:1px solid #b9d6eb;border-radius:999px}}.well-badge.uncertainty{{color:#775410;background:#fff5d8;border-color:#ecd18a}}.well-badge.safe{{color:#08725d;background:#e6f5ef;border-color:#a9d9c9}}.well-badge.candidate{{color:#9a5209;background:#fff1dd;border-color:#e6be81}}.well-status-code{{flex-basis:100%;color:#8b99a7;font-size:12px;text-align:right}}.well-chart-card svg{{display:block;width:100%;max-height:calc(100vh - 195px);min-height:430px}}.well-plot-bg{{fill:#f8fbfd;stroke:#bfd0df;stroke-width:1}}.well-grid{{stroke:#dbe5ed;stroke-width:1}}.well-tick{{stroke:#71869a;stroke-width:1}}.well-axis-text{{fill:#6b7f92;font-size:12px}}.well-axis-title{{fill:#40566d;font-size:12px;font-weight:650}}.well-uncertainty-band{{fill:#e8b43e;fill-opacity:.24;stroke:none}}.well-primary-curve{{fill:none;stroke:#116fbd;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}}.well-interval-label{{fill:#344b60;font-size:12px;font-weight:650}}.well-track-legend{{display:flex;flex-wrap:wrap;gap:8px 16px;margin:4px 0 10px}}.well-track-legend span{{display:inline-flex;gap:7px;align-items:center;color:#435a70;font-size:12px}}.well-track-legend i{{width:13px;height:13px;border-radius:3px}}.well-track-legend small{{color:#8494a4}}.well-display-note{{margin:8px 0 0;color:#718599;font-size:12px}}
.fault-grid-view{{position:absolute;inset:0;display:grid;grid-template-rows:auto auto minmax(0,1fr) auto auto;gap:8px;overflow:hidden;padding:12px 14px;background:linear-gradient(145deg,#f7fafc,#eaf0f6)}}.fault-grid-heading{{display:flex;justify-content:space-between;gap:18px;align-items:end}}.fault-grid-heading span{{color:#1677e8;font-size:12px;font-weight:850;letter-spacing:.16em}}.fault-grid-heading h2{{margin:2px 0 0;font-size:17px}}.fault-grid-facts{{display:grid;text-align:right}}.fault-grid-facts b{{color:#1267bb;font-size:16px}}.fault-grid-facts small{{color:#72869a;font-size:12px}}.fault-grid-notice{{margin:0;padding:7px 10px;color:#76510d;font-size:12px;line-height:1.45;background:#fff6dc;border:1px solid #e7c875;border-radius:7px}}.fault-grid-notice strong{{color:#9b5705}}.fault-grid-3d-stage{{position:relative;min-width:0;min-height:0;overflow:hidden;background:#edf2f6;border:1px solid #cbd8e3;border-radius:10px;box-shadow:0 10px 28px rgba(31,55,79,.09)}}.fault-grid-3d-help{{position:absolute;left:12px;bottom:10px;z-index:4;max-width:58%;margin:0;padding:5px 8px;color:#405b72;font-size:12px;background:rgba(255,255,255,.9);border:1px solid #cad8e4;border-radius:5px;pointer-events:none}}.fault-grid-fallback{{max-height:38vh;overflow:auto;color:#536b80;background:rgba(255,255,255,.74);border:1px solid #cfdae5;border-radius:7px}}.fault-grid-fallback>summary{{padding:6px 9px;color:#176dbc;font-size:12px;font-weight:750;cursor:pointer;list-style-position:inside}}.fault-grid-fallback[open]>summary{{border-bottom:1px solid #d7e1ea}}.fault-grid-fallback .fault-grid-depths{{padding:7px}}.fault-grid-depths{{display:grid;grid-template-rows:repeat(4,minmax(108px,1fr));gap:7px;min-height:0}}.fault-grid-depth{{display:grid;grid-template-columns:94px minmax(0,1fr);gap:8px;min-height:0;padding:6px;background:rgba(255,255,255,.95);border:1px solid #d5e0ea;border-radius:8px;box-shadow:0 4px 12px rgba(31,55,79,.04)}}.fault-grid-depth>header{{display:grid;align-content:center;justify-items:center;padding:5px;color:#536b82;background:#edf3f8;border-radius:6px}}.fault-grid-depth>header span{{font-size:12px}}.fault-grid-depth>header b{{color:#186dbd;font-size:13px}}.fault-grid-depth>header small{{font-size:12px;text-align:center}}.fault-grid-matrix{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(3,minmax(0,1fr));gap:4px;min-width:0;min-height:0}}.fault-grid-block{{display:grid;min-width:0;min-height:0;align-content:center;gap:1px;padding:3px 6px;color:#40566d;text-decoration:none;background:#f8fafc;border:1px solid #d9e2eb;border-radius:5px;transition:border-color .13s,background .13s,transform .13s}}.fault-grid-block:hover{{z-index:1;color:#0f66c7;background:#eef6ff;border-color:#68a3df;transform:translateY(-1px)}}.fault-grid-block.default{{background:#eaf4ff;border-color:#77ace0;box-shadow:inset 3px 0 #1677e8}}.fault-grid-block>span{{display:flex;min-width:0;gap:5px;align-items:center}}.fault-grid-block b{{overflow:hidden;font-size:12px;text-overflow:ellipsis;white-space:nowrap}}.fault-grid-block em{{padding:1px 3px;color:#fff;font-size:12px;font-style:normal;background:#1677e8;border-radius:3px}}.fault-grid-block small{{overflow:hidden;color:#708499;font-size:12px;text-overflow:ellipsis;white-space:nowrap}}.fault-grid-receipt{{display:flex;justify-content:space-between;gap:14px;color:#6f8295;font-size:12px}}.fault-grid-receipt span:last-child{{text-align:right}}.fault-grid-overview-volume .volume-render-tools label,.fault-grid-overview-volume .volume-render-tools .overlay-tool,.fault-grid-overview-volume .mask-slab-tool,.fault-grid-overview-volume .well-type-legend{{display:none!important}}.fault-grid-overview-volume .volume-render-tools{{width:auto;padding:7px}}.fault-grid-overview-volume .volume-render-tools button{{margin:0}}.fault-grid-layer-legend{{display:grid!important;grid-template-columns:58px auto;column-gap:7px}}.fault-grid-layer-legend>i{{grid-row:1/3;width:58px!important;height:10px!important;background:linear-gradient(90deg,#1677e8 0 25%,#0ea5a4 25% 50%,#f59e0b 50% 75%,#a855f7 75%)}}.fault-grid-layer-legend small{{color:#8a6418;font-size:12px}}.fault-grid-3d-tooltip{{position:absolute;z-index:6;width:278px;padding:7px 9px;color:#29445c;font-size:12px;line-height:1.5;background:rgba(255,255,255,.96);border:1px solid #76aee0;border-radius:6px;box-shadow:0 8px 22px rgba(31,55,79,.17);pointer-events:none}}.fault-grid-overview-volume.unsupported::after{{content:'WebGL2 不可用，请使用下方二维代表块列表'}}.grid-overview-link{{display:flex;justify-content:center;margin-top:9px;padding:7px;color:#1267bb;font-size:12px;font-weight:750;text-decoration:none;background:#eaf4ff;border:1px solid #9fc6ec;border-radius:6px}}.grid-overview-link:hover{{background:#dceeff}}.representative-block-banner{{position:absolute;top:8px;left:50%;z-index:7;display:grid;min-width:340px;max-width:56%;padding:6px 10px;text-align:center;background:rgba(255,248,222,.94);border:1px solid #e5c46a;border-radius:7px;box-shadow:0 5px 16px rgba(72,56,18,.12);transform:translateX(-50%);pointer-events:none}}.representative-block-banner strong{{color:#955408;font-size:12px}}.representative-block-banner span{{color:#765d2c;font-size:12px}}
.volume-render-view{{--well-result-rail-width:clamp(320px,30vw,380px);position:absolute;inset:0;overflow:hidden;background:var(--scene-bg)}}#volume-render-canvas,#volume-wire-canvas{{position:absolute;inset:0;display:block;width:100%;height:100%}}#volume-render-canvas{{cursor:grab;touch-action:none}}#volume-render-canvas:active{{cursor:grabbing}}#volume-wire-canvas{{pointer-events:none}}.volume-render-view.has-well-results::before{{position:absolute;top:0;right:0;bottom:0;z-index:2;width:var(--well-result-rail-width);content:'';background:linear-gradient(180deg,rgba(247,250,252,.98),rgba(235,242,247,.98));border-left:1px solid #c8d6e1;box-shadow:-16px 0 36px rgba(31,55,79,.06);pointer-events:none}}.volume-render-view.has-well-results.well-result-expanded{{--well-result-rail-width:clamp(420px,36vw,520px)}}.volume-render-view.has-well-results #volume-render-canvas,.volume-render-view.has-well-results #volume-wire-canvas{{right:var(--well-result-rail-width);width:calc(100% - var(--well-result-rail-width))}}.volume-render-tools{{position:absolute;top:14px;left:14px;z-index:3;display:grid;width:182px;gap:8px;padding:11px;background:rgba(255,255,255,.91);border:1px solid #ccd7e2;border-radius:8px;box-shadow:0 8px 24px rgba(31,55,79,.1);backdrop-filter:blur(8px)}}.volume-render-tools label{{display:grid;gap:4px}}.volume-render-tools [hidden],.volume-legend [hidden]{{display:none!important}}.volume-render-tools label>span{{color:#5f7286;font-size:12px;font-weight:650}}.volume-render-tools input{{width:100%;height:4px;accent-color:#1677e8}}.volume-render-tools select{{width:100%;height:28px;padding:0 6px;color:#40566d;background:#f8fafc;border:1px solid #cbd7e2;border-radius:5px}}.volume-render-tools button{{justify-content:center;min-height:28px;margin-top:2px}}.volume-legend{{position:absolute;right:14px;top:14px;z-index:3;display:grid;gap:6px;padding:9px 10px;color:#546a7f;font-size:12px;background:rgba(255,255,255,.91);border:1px solid #ccd7e2;border-radius:7px}}.volume-render-view.has-well-results .volume-legend{{right:calc(var(--well-result-rail-width) + 14px)}}.volume-legend span{{display:flex;gap:7px;align-items:center}}.volume-legend .vertical-axis-legend{{display:grid;gap:1px;padding-bottom:5px;border-bottom:1px solid #dbe3eb}}.vertical-axis-legend b{{color:#334b61;font-size:12px}}.vertical-axis-legend em{{color:#1677e8;font-size:12px;font-style:normal}}.volume-legend i{{display:block;width:64px;height:7px;border:1px solid rgba(45,63,80,.2);border-radius:2px}}.seismic-ramp{{background:linear-gradient(90deg,#0517b8,#f7f7f7 50%,#c20b0b)}}.jet-ramp{{background:linear-gradient(90deg,#000080,#006cff,#7dff7a,#ffb500,#800000)}}.fault-mask-ramp{{background:linear-gradient(90deg,transparent 0 46%,#ef141c 46% 100%)}}.volume-render-status{{position:absolute;right:14px;bottom:12px;z-index:3;padding:5px 8px;color:#536b80;font-size:12px;background:rgba(255,255,255,.86);border:1px solid #d1dbe5;border-radius:5px}}.volume-render-view.has-well-results .volume-render-status{{right:calc(var(--well-result-rail-width) + 14px)}}.volume-render-view.unsupported::after{{position:absolute;inset:38% 22%;display:grid;place-items:center;content:'请切换到“切片查看”';color:#61768b;font-size:13px;background:var(--plate-bg);border:1px dashed #b8c6d4;border-radius:8px}}
.well-result-leader{{position:absolute;inset:0;z-index:4;width:100%;height:100%;overflow:visible;pointer-events:none}}.well-result-leader path{{fill:none;stroke:rgba(46,82,108,.62);stroke-width:1.5;stroke-dasharray:4 3}}.well-result-leader circle{{fill:#fff;stroke:#16866f;stroke-width:2.2}}.well-result-hud{{position:absolute;z-index:5;display:flex;width:calc(var(--well-result-rail-width) - 28px);max-height:calc(100% - 28px);flex-direction:column;padding:14px;background:rgba(255,255,255,.96);border:1px solid #bfcfdb;border-radius:14px;box-shadow:0 18px 42px rgba(31,55,79,.16);backdrop-filter:blur(14px) saturate(140%);transition:width .18s,box-shadow .18s}}.well-result-hud[hidden]{{display:none}}.well-result-hud.dragging{{box-shadow:0 22px 52px rgba(22,91,124,.24);transition:none}}.well-result-hud>header{{display:flex;flex:0 0 auto;gap:10px;align-items:center;justify-content:space-between;padding-bottom:11px;border-bottom:1px solid #d8e3eb;cursor:ns-resize;touch-action:none;user-select:none}}.well-result-hud>header>div{{display:grid;min-width:0;gap:2px}}.well-result-hud>header span{{color:#16866f;font-size:12px;font-weight:850;letter-spacing:.13em}}.well-result-hud>header strong{{overflow:hidden;color:#213f56;font-size:14px;text-overflow:ellipsis;white-space:nowrap}}.well-result-hud>header nav{{display:flex;flex:0 0 auto;gap:4px}}.well-result-hud>header button{{display:grid;width:29px;min-height:29px;padding:0;place-items:center;color:#365970;font-size:15px;background:#f3f7fa;border:1px solid #ccd9e3;border-radius:7px}}.well-result-hud>header button:hover{{color:#0d6e61;background:#eaf6f2;border-color:#8dc7b7}}.well-result-hud>header button[hidden]{{display:none}}.well-result-body{{min-height:0;overflow:auto;padding:0 2px 1px;scrollbar-width:thin;scrollbar-color:#b9c8d4 transparent}}.well-result-hud.collapsed{{width:calc(var(--well-result-rail-width) - 28px)}}.well-result-hud.collapsed .well-result-body{{display:none}}.well-result-hud.collapsed>header{{padding-bottom:0;border-bottom:0}}.well-result-link-state{{margin:11px 0;color:#16866f;font-size:12px;font-weight:650;line-height:1.5}}.well-result-hud.md-only .well-result-link-state{{color:#946014}}.well-result-track-layout{{display:grid;height:clamp(300px,50vh,520px);grid-template-columns:68px 58px;gap:12px;justify-content:center}}.well-result-hud.expanded .well-result-track-layout{{height:clamp(360px,58vh,620px);grid-template-columns:78px 88px}}.well-result-hud.property-result .well-result-track-layout{{grid-template-columns:68px minmax(150px,1fr);justify-content:stretch}}.well-result-hud.property-result.expanded .well-result-track-layout{{grid-template-columns:78px minmax(220px,1fr)}}.well-result-axis{{display:flex;flex-direction:column;justify-content:space-between;color:#61788c;font-size:12px;text-align:right}}.well-result-axis span:first-child::before{{content:'MD ↓';display:block;margin-bottom:3px;color:#31526a;font-size:12px;font-weight:750}}.well-result-classification-track{{position:relative;overflow:hidden;background:#edf2f5;border:1px solid #adc0ce;border-radius:7px;box-shadow:inset 0 0 0 2px rgba(255,255,255,.62)}}.well-result-property-track{{display:grid;min-width:0;grid-template-rows:minmax(0,1fr) auto auto;gap:7px;padding:7px;background:linear-gradient(90deg,#f8fbfd,#edf5f8);border:1px solid #adc0ce;border-radius:7px;box-shadow:inset 0 0 0 2px rgba(255,255,255,.62)}}.well-result-property-track[hidden]{{display:none!important}}.well-result-property-track svg{{display:block;width:100%;height:100%;min-height:0;background:repeating-linear-gradient(90deg,transparent 0 calc(25% - 1px),rgba(104,137,160,.18) 25%);border-bottom:1px solid #cbd9e2;touch-action:none}}#well-result-property-band{{fill:rgba(33,142,165,.16);stroke:none}}#well-result-property-curve{{fill:none;stroke:#147b91;stroke-width:3;vector-effect:non-scaling-stroke}}#well-result-property-cursor-line{{stroke:rgba(31,75,102,.55);stroke-width:1.5;stroke-dasharray:4 3;vector-effect:non-scaling-stroke}}#well-result-property-cursor{{fill:#fff;stroke:#126e84;stroke-width:3;vector-effect:non-scaling-stroke}}.well-result-property-scale{{display:grid;grid-template-columns:1fr auto 1fr;gap:6px;align-items:center;color:#63798c;font-size:12px}}.well-result-property-scale strong{{color:#31566d;font-size:12px;text-align:center}}.well-result-property-scale span:last-child{{text-align:right}}#well-result-property-probe{{width:100%;height:4px;margin:2px 0;accent-color:#147b91}}.well-result-segment{{position:absolute;right:0;left:0;display:block;width:100%;min-height:2px;padding:0;border:0;border-radius:0;box-shadow:inset 0 -1px rgba(255,255,255,.86);cursor:pointer}}.well-result-segment:hover,.well-result-segment.hovered,.well-result-segment:focus-visible{{z-index:2;outline:3px solid #1b5678;outline-offset:-3px;filter:saturate(1.14) brightness(1.03)}}.well-result-segment.active{{z-index:3;outline:3px solid #163f59;outline-offset:-3px;box-shadow:0 0 0 4px rgba(22,63,89,.14)}}.well-result-legend{{display:flex;max-height:74px;gap:8px 13px;align-items:center;flex-wrap:wrap;overflow:auto;margin-top:12px}}.well-result-legend span{{display:inline-flex;gap:6px;align-items:center;color:#405d73;font-size:12px}}.well-result-legend i{{width:11px;height:11px;border-radius:3px}}.well-result-detail{{display:block;min-height:34px;margin-top:10px;padding-top:9px;color:#405e75;font-size:12px;line-height:1.55;border-top:1px solid #dce6ed}}.well-result-hud.md-only .well-result-detail::before{{content:'井侧结果 · ';color:#9a671c;font-weight:720}}
.volume-render-view.facies-log-layout{{--well-result-rail-width:clamp(340px,31vw,430px)}}.volume-render-view.facies-log-layout.has-well-results::before{{right:auto;left:0;background:#eef3f6;border-right:1px solid #9eafbc;border-left:0;box-shadow:16px 0 36px rgba(31,55,79,.07)}}.volume-render-view.facies-log-layout.has-well-results #volume-render-canvas,.volume-render-view.facies-log-layout.has-well-results #volume-wire-canvas{{right:0;left:var(--well-result-rail-width);width:calc(100% - var(--well-result-rail-width))}}.volume-render-view.facies-log-layout .volume-render-tools{{left:calc(var(--well-result-rail-width) + 14px)}}.volume-render-view.facies-log-layout.has-well-results .volume-legend{{right:14px}}.volume-render-view.facies-log-layout.has-well-results .volume-render-status{{right:14px}}.facies-log-layout .well-result-hud{{height:calc(100% - 28px);max-height:none;padding:12px;background:#fff;border-color:#9eafbc;border-radius:5px;box-shadow:0 8px 22px rgba(31,55,79,.11);backdrop-filter:none}}.facies-log-layout .well-result-hud>header{{padding:1px 2px 10px;cursor:default}}.facies-log-layout .well-result-hud>header span{{color:#31526a;letter-spacing:.08em}}.facies-log-layout .well-result-body{{display:flex;min-height:0;flex:1;flex-direction:column;overflow:auto}}.well-result-log-track-head{{display:grid;grid-template-columns:88px minmax(112px,1fr);gap:8px;margin:10px 0 5px;color:#38566b;font-size:12px;text-align:center}}.well-result-log-track-head span,.well-result-log-track-head strong{{padding:5px 4px;background:#edf2f5;border:1px solid #9eafbc}}.facies-log-layout .well-result-track-layout{{height:clamp(340px,56vh,640px);flex:0 0 auto;grid-template-columns:88px minmax(112px,1fr);gap:8px;justify-content:stretch;background:repeating-linear-gradient(to bottom,rgba(90,115,133,.2) 0 1px,transparent 1px 10%)}}.facies-log-layout .well-result-axis{{position:relative;display:block;color:#405f74;font-variant-numeric:tabular-nums}}.facies-log-layout .well-result-axis span{{position:absolute;right:4px;transform:translateY(-50%)}}.facies-log-layout .well-result-axis span[data-tick="0"]{{transform:none}}.facies-log-layout .well-result-axis span[data-tick="10"]{{transform:translateY(-100%)}}.facies-log-layout .well-result-axis span:first-child::before{{content:none}}.facies-log-layout .well-result-classification-track{{overflow:hidden;background:#f7f9fa;border-color:#778e9f;border-radius:0;box-shadow:none}}.facies-log-layout .well-result-classification-track::after{{position:absolute;inset:0;z-index:4;content:'';background:repeating-linear-gradient(to bottom,rgba(255,255,255,.72) 0 1px,transparent 1px 10%),repeating-linear-gradient(to right,rgba(42,68,86,.17) 0 1px,transparent 1px 25%);pointer-events:none}}.facies-log-layout .well-result-property-track{{padding:0;background:#f7f9fa;border-color:#778e9f;border-radius:0;box-shadow:none}}.facies-log-layout .well-result-property-track svg{{background:repeating-linear-gradient(to bottom,rgba(255,255,255,.72) 0 1px,transparent 1px 10%),repeating-linear-gradient(to right,rgba(42,68,86,.17) 0 1px,transparent 1px 25%)}}.facies-log-layout .well-result-property-scale{{padding:0 6px 3px}}.facies-log-layout #well-result-property-probe{{width:calc(100% - 12px);margin:0 6px 6px}}.facies-log-layout .well-result-segment{{z-index:1}}.facies-log-layout .well-result-segment:hover,.facies-log-layout .well-result-segment.hovered,.facies-log-layout .well-result-segment:focus-visible,.facies-log-layout .well-result-segment.active{{z-index:3}}.facies-log-layout .well-result-legend{{flex:0 0 auto;max-height:86px;padding-top:10px;border-top:1px solid #c7d3dc}}.facies-log-layout .well-result-detail{{position:relative;z-index:5;flex:0 0 auto;margin-top:8px;padding:8px;color:#405e75;background:#edf2f5;border:1px solid #c7d3dc}}.facies-log-layout .well-result-hud.property-result .well-result-detail{{order:2;margin:0 0 7px}}.facies-log-layout .well-result-hud.property-result .well-result-track-layout{{order:3}}.facies-log-layout .well-result-hud.property-result .well-result-legend{{order:4}}
.facies-log-layout .well-result-hud.property-result .well-result-body{{display:grid;grid-template-rows:auto auto minmax(0,1fr) auto auto;overflow:hidden}}.facies-log-layout .well-result-hud.property-result .well-result-link-state{{grid-row:1}}.facies-log-layout .well-result-hud.property-result .well-result-log-track-head{{grid-row:2}}.facies-log-layout .well-result-hud.property-result .well-result-track-layout{{height:auto;min-height:0;grid-row:3;overflow:hidden;flex:none;order:initial}}.facies-log-layout .well-result-hud.property-result .well-result-property-track{{height:100%;min-height:0;overflow:hidden}}.facies-log-layout .well-result-hud.property-result .well-result-legend{{grid-row:4;order:initial;margin-top:0}}.facies-log-layout .well-result-hud.property-result .well-result-detail{{position:static;z-index:auto;grid-row:5;order:initial;margin:8px 0 0}}
.workbench.facies-linked-workbench{{--facies-log-rail-width:clamp(340px,31vw,430px)}}.facies-linked-workbench .slice-volume-panel:not([hidden]){{left:var(--facies-log-rail-width);z-index:4;border-left:1px solid #9eafbc}}
.candidate-surface-status{{display:grid!important;padding:6px;color:#8a4b08;background:#fff4df;border:1px solid #f0c783;border-radius:5px}}.candidate-surface-status b{{color:#a45100;font-size:12px}}.surface-legend b{{font-size:15px;line-height:1}}
.hint{{position:absolute;right:10px;bottom:9px;z-index:5;padding:5px 8px;color:#5f7488;font-size:12px;background:rgba(255,255,255,.88);border:1px solid #d5dfe9;border-radius:5px;pointer-events:none}}
.slice-console{{grid-column:2;grid-row:3;display:grid;visibility:hidden;grid-template-columns:138px repeat(3,minmax(140px,1fr));gap:16px;align-items:center;overflow:hidden;padding:0 18px;background:#fff;border-top:0 solid var(--line);opacity:0;box-shadow:0 -3px 12px rgba(35,58,81,.04);transition:opacity .16s}}.workbench.slice-active .slice-console{{visibility:visible;padding:10px 18px;border-top-width:1px;opacity:1}}.slice-console-title{{display:grid}}.slice-console-title strong{{font-size:12px}}.slice-console-title small{{color:#8091a2;font-size:12px}}.slice-control{{display:grid;gap:6px}}.slice-control span{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.slice-control b{{font-size:12px}}.slice-control output{{color:#146ac5;font-size:12px;font-weight:750}}.slice-control input{{width:100%;height:5px;accent-color:#1677e8;cursor:ew-resize}}
@media(max-width:900px){{.volume-render-view.facies-log-layout.has-well-results::before{{top:auto;right:0;left:0;width:auto;height:var(--well-result-rail-height);border-top:1px solid #c8d6e1;border-right:0}}.volume-render-view.facies-log-layout.has-well-results #volume-render-canvas,.volume-render-view.facies-log-layout.has-well-results #volume-wire-canvas{{right:0;bottom:var(--well-result-rail-height);left:0;width:100%;height:calc(100% - var(--well-result-rail-height))}}.volume-render-view.facies-log-layout .volume-render-tools{{left:10px}}.volume-render-view.facies-log-layout .well-result-hud{{height:auto;max-height:calc(var(--well-result-rail-height) - 20px)}}.facies-log-layout .well-result-track-layout{{height:clamp(155px,23vh,240px);grid-template-columns:88px minmax(112px,1fr)}}.facies-log-layout .well-result-hud>header{{cursor:default}}}}
@media(max-width:900px){{.volume-render-view.has-well-results{{--well-result-rail-height:clamp(270px,42vh,370px)}}.volume-render-view.has-well-results.well-result-expanded{{--well-result-rail-height:clamp(330px,52vh,460px)}}.volume-render-view.has-well-results::before{{top:auto;left:0;width:auto;height:var(--well-result-rail-height);border-top:1px solid #c8d6e1;border-left:0;box-shadow:0 -16px 36px rgba(31,55,79,.06)}}.volume-render-view.has-well-results #volume-render-canvas,.volume-render-view.has-well-results #volume-wire-canvas{{right:0;bottom:var(--well-result-rail-height);width:100%;height:calc(100% - var(--well-result-rail-height))}}.volume-render-view.has-well-results .volume-legend{{right:10px}}.volume-render-view.has-well-results .volume-render-status{{right:10px;bottom:calc(var(--well-result-rail-height) + 10px)}}.well-result-hud,.well-result-hud.collapsed{{width:calc(100% - 28px);max-height:calc(var(--well-result-rail-height) - 20px)}}.well-result-track-layout,.well-result-hud.expanded .well-result-track-layout{{height:clamp(155px,23vh,240px);grid-template-columns:68px 64px}}.well-result-hud>header{{cursor:ns-resize}}}}
@media(max-width:900px){{.workbench.facies-linked-workbench{{--facies-log-rail-height:clamp(270px,42vh,370px)}}.facies-linked-workbench .slice-volume-panel:not([hidden]){{right:0;bottom:var(--facies-log-rail-height);left:0;border-bottom:1px solid #9eafbc;border-left:0}}}}
@media(max-width:1120px){{.dataset-title{{display:none}}.view-mode-switch .view-mode-button{{padding:0 9px}}.view-mode-button small{{display:none}}.layer-mode-switch .layer-mode-button{{padding:0 7px}}}}
@media(max-width:820px){{.workbench,.workbench.assets-collapsed{{grid-template-columns:1fr;grid-template-rows:50px 92px minmax(0,1fr)}}.workbench.with-slice-console,.workbench.with-slice-console.assets-collapsed{{grid-template-rows:50px 92px minmax(0,1fr) 0}}.workbench.with-slice-console.slice-active,.workbench.with-slice-console.assets-collapsed.slice-active{{grid-template-rows:50px 92px minmax(0,1fr) 126px}}.topbar{{padding:0 8px}}.sidebar,.workbench.assets-collapsed .sidebar{{display:block;grid-column:1;grid-row:2;padding:7px;overflow-x:auto;overflow-y:hidden;opacity:1;pointer-events:auto;border-right:0;border-bottom:1px solid var(--line)}}.section-label,.meta-card,.engine{{display:none}}.asset-list{{display:flex}}.asset-link{{min-width:175px}}.viewport{{grid-column:1;grid-row:3}}.slice-console{{grid-column:1;grid-row:4;grid-template-columns:repeat(3,minmax(90px,1fr));gap:8px;padding:0 10px}}.workbench.slice-active .slice-console{{padding:8px 10px}}.slice-console-title{{display:none}}.volume-render-tools{{width:160px}}.fault-grid-view{{overflow:hidden;padding:9px}}.fault-grid-heading h2{{font-size:13px}}.fault-grid-facts{{display:none}}.fault-grid-3d-help{{max-width:70%;font-size:12px}}.fault-grid-depths{{grid-template-rows:repeat(4,150px)}}.fault-grid-depth{{grid-template-columns:68px minmax(0,1fr)}}.fault-grid-block{{padding:2px 4px}}.fault-grid-block small{{font-size:12px}}.representative-block-banner{{top:5px;min-width:260px;max-width:72%}}}}
@media(min-width:521px) and (max-width:900px){{.volume-render-view.facies-log-layout{{--well-result-rail-width:clamp(232px,40vw,280px)}}.volume-render-view.facies-log-layout.has-well-results::before{{top:0;right:auto;bottom:0;left:0;width:var(--well-result-rail-width);height:auto;border-top:0;border-right:1px solid #9eafbc;border-left:0;box-shadow:16px 0 36px rgba(31,55,79,.07)}}.volume-render-view.facies-log-layout.has-well-results #volume-render-canvas,.volume-render-view.facies-log-layout.has-well-results #volume-wire-canvas{{top:0;right:0;bottom:0;left:var(--well-result-rail-width);width:calc(100% - var(--well-result-rail-width));height:100%}}.volume-render-view.facies-log-layout .well-result-hud,.volume-render-view.facies-log-layout .well-result-hud.collapsed{{width:calc(var(--well-result-rail-width) - 20px);height:calc(100% - 20px);max-height:none}}.volume-render-view.facies-log-layout .volume-render-tools{{left:calc(var(--well-result-rail-width) + 10px)}}.volume-render-view.facies-log-layout.has-well-results .volume-render-status{{right:10px;bottom:10px}}.facies-log-layout .well-result-track-layout,.facies-log-layout .well-result-hud.expanded .well-result-track-layout{{height:clamp(300px,52vh,600px);grid-template-columns:76px minmax(112px,1fr)}}.workbench.facies-linked-workbench{{--facies-log-rail-width:clamp(232px,40vw,280px)}}.facies-linked-workbench .slice-volume-panel:not([hidden]){{top:0;right:0;bottom:0;left:var(--facies-log-rail-width);border-bottom:0;border-left:1px solid #9eafbc}}}}
@media(min-width:521px){{.facies-log-layout .well-result-hud:not(.property-result) .well-result-track-layout{{height:auto;min-height:0;flex:1 1 auto}}}}
</style></head>
<body><div class="{workbench_class}" id="workbench"><header class="topbar"><div class="dataset-title"><span>{kind_label}</span><strong>{html.escape(selected_display_name)}</strong></div>{mode_switch}{layer_switch}<div class="top-actions">{overview_action}<button id="asset-toggle" type="button">{'井列表' if kind == 'well_sequence' else '数据集'}</button><button id="fit-view" type="button">{'完整范围' if kind in {'well_sequence', 'fault_grid'} else '恢复视图'}</button><button id="fullscreen" type="button">全屏</button><a href="/统一数据可视化?{standalone_query}" target="_blank" rel="noopener">独立窗口</a></div></header>
<aside class="sidebar"><span class="section-label">{'当前任务井序列成果' if kind == 'well_sequence' else '当前任务地震资产'}</span><nav class="asset-list">{''.join(links)}</nav>{overview_action}<section class="meta-card"><div><span>渲染类型</span><strong>{kind_label}</strong></div><div><span>{shape_label}</span><strong>{' × '.join(str(value) for value in selected_shape)}</strong></div>{f'<div><span>叠加图层</span><strong>{overlay_count} 个预测图层</strong></div>' if overlay_count else ''}{horizon_candidate_meta}{candidate_status_meta}{representative_meta}<div><span>任务快照</span><strong>{html.escape(task_id[:8])}</strong></div><div><span>成果来源</span><strong>{html.escape(source_label)}</strong></div></section><div class="engine{' warning' if engine_warning else ''}"><i></i><span>{html.escape(engine_display_name)}{engine_fallback_note}</span></div></aside>
<main class="viewport"><div class="plot-shell">{fragment}</div><div class="hint" id="view-hint"{' hidden' if kind == 'fault_grid' else ''}>{f'固定{representative_block_count}块位置图 · 点击后仅加载当前完整128³块' if kind == 'fault_grid' else '鼠标拖动旋转 · 滚轮缩放 · 井轨迹随地震体联动' if kind == 'volume' else '真实MD轴 · 确定性成果 · 左侧切换井' if kind == 'well_sequence' else '二维地震剖面 · 垂向采样从上到下增大'}</div></main>{slice_console}</div>
<script>
const workbench=document.getElementById('workbench');
const plot=document.getElementById('cigvis-plot');
const resizePlot=()=>{{if(plot&&window.Plotly)window.Plotly.Plots.resize(plot)}};
document.getElementById('asset-toggle').addEventListener('click',()=>workbench.classList.toggle('assets-collapsed'));
let activeView={json.dumps(default_volume_view if kind == 'volume' else 'grid' if kind == 'fault_grid' else 'line')};
let activeLayerMode={json.dumps(default_layer_mode)};
const assetKind={json.dumps(kind)};
const overlaySubject={json.dumps(overlay_subject, ensure_ascii=False)};
const faultsegExactBlock={json.dumps(dict(selected.get("sliceViewContract", {})).get("displayMode") == "faultseg_exact_binary_mask")};
const faciesLinkedLayout=workbench.classList.contains('facies-linked-workbench');
const viewHint=document.getElementById('view-hint');
const layerModeButtons=Array.from(document.querySelectorAll('.layer-mode-button'));
const updateViewHint=()=>{{
  if(!viewHint)return;
  if(assetKind==='fault_grid'){{viewHint.textContent='固定{representative_block_count}块位置图 · 点击后仅加载当前完整128³块';return;}}
  if(assetKind==='well_sequence'){{viewHint.textContent='真实MD轴 · 确定性成果 · 左侧切换井';return;}}
  if(faciesLinkedLayout){{viewHint.textContent=activeView==='slice'?{json.dumps(f'左侧{linked_well_log_title} · 右侧正交地震切片', ensure_ascii=False)}:{json.dumps(f'左侧{linked_well_log_title} · 右侧三维地震与井轨迹联动', ensure_ascii=False)};return;}}
  if(faultsegExactBlock){{viewHint.textContent=activeView==='slice'?'默认 Inline 地震 + 断层掩码 · 支持 Crossline 和时间 / 深度切片':'当前完整128³代表块 · 三维窗口厚度100% · 不与其他块拼接';return;}}
  if(activeLayerMode==='prediction'){{viewHint.textContent=activeView==='slice'?'仅显示'+overlaySubject+'二维结果 · 可切换平面和图层':'仅显示'+overlaySubject+'体 · 拖动旋转 · 滚轮缩放';return;}}
  viewHint.textContent=activeView==='slice'?'默认 Inline 清晰分层 · 单层位俯视图用于时间 / 深度查看':'鼠标拖动旋转 · 滚轮缩放 · 井轨迹随地震体联动';
}};
const applyLayerMode=(mode)=>{{
  activeLayerMode=mode;
  layerModeButtons.forEach(button=>{{const active=button.dataset.layerMode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}});
  window.setWholeVolumeLayerMode?.(mode);
  window.setOrthogonalSliceLayerMode?.(mode);
  updateViewHint();
}};
const setLayerMode=(mode)=>{{
  if(!['combined','prediction'].includes(mode)||mode===activeLayerMode)return;
  applyLayerMode(mode);
}};
const setView=(view)=>{{
  activeView=view;
  document.querySelectorAll('.view-panel').forEach(panel=>{{panel.hidden=panel.dataset.view!==view&&!(faciesLinkedLayout&&panel.dataset.view==='whole');}});
  document.querySelectorAll('.view-mode-button').forEach(button=>button.classList.toggle('active',button.dataset.viewTarget===view));
  workbench.classList.toggle('slice-active',view==='slice');workbench.classList.toggle('whole-active',view!=='slice');
  updateViewHint();
  requestAnimationFrame(()=>{{if(view==='slice'){{resizePlot();window.resizeOrthogonalSliceView?.();}}else{{window.resetWholeVolumeView?.();window.dispatchEvent(new Event('resize'));}}}});
}};
document.querySelectorAll('.view-mode-button').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.viewTarget)));
layerModeButtons.forEach(button=>button.addEventListener('click',()=>setLayerMode(button.dataset.layerMode)));
document.getElementById('fit-view').addEventListener('click',()=>{{
  if(assetKind==='fault_grid'){{window.resetWholeVolumeView?.();return;}}
  if(assetKind==='well_sequence'){{document.querySelector('.cigvis-well-view')?.scrollTo({{top:0,behavior:'smooth'}});return;}}
  if(activeView==='whole'){{window.resetWholeVolumeView?.();return;}}
  window.resetOrthogonalSliceView?.();resizePlot();if(plot&&window.Plotly)window.Plotly.relayout(plot,{{'scene.camera.eye':{{x:1.55,y:-1.65,z:1.18}},'scene.camera.center':{{x:0,y:0,z:0}}}});
}});
document.getElementById('fullscreen').addEventListener('click',()=>{{if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();else document.exitFullscreen?.()}});
updateViewHint();
if(layerModeButtons.length)applyLayerMode(activeLayerMode);
if(activeView==='slice')requestAnimationFrame(()=>window.resizeOrthogonalSliceView?.());
if(window.ResizeObserver)new ResizeObserver(resizePlot).observe(document.querySelector('.viewport'));window.addEventListener('load',resizePlot);
</script></body></html>"""
