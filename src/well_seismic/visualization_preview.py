from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np


NINE_CURVES = (
    ("SP", "自然电位 SP", "#d97706", False),
    ("GR", "自然伽马 GR", "#16a34a", False),
    ("CAL", "井径 CAL", "#64748b", False),
    ("DT", "声波时差 AC", "#2563eb", False),
    ("NPHI", "中子 CNL", "#7c3aed", False),
    ("RHOB", "密度 DEN", "#dc2626", False),
    ("MSFL", "微球聚焦 MSFL", "#0891b2", True),
    ("RS", "浅侧向 LLS", "#ea580c", True),
    ("RT", "深侧向 LLD", "#be123c", True),
)


def _sample_positions(size: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, max(0, size - 1), min(size, count), dtype=np.int64))


def _finite_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values if np.isfinite(value)]


def _nullable_list(values: np.ndarray) -> list[float | None]:
    return [round(float(value), 5) if np.isfinite(value) else None for value in values]


def _well_log_payloads(pipeline: Any, max_points: int = 720) -> list[dict[str, Any]]:
    """Expose real, standardized conventional-nine curves as a lightweight preview."""
    result: list[dict[str, Any]] = []
    for entity in pipeline.registry.entities.values():
        for log_index, log in enumerate(entity.logs):
            depth = np.asarray(log.depth, dtype=float)
            if not depth.size:
                continue
            indices = _sample_positions(depth.size, max_points)
            sampled_depth = depth[indices]
            curves: list[dict[str, Any]] = []
            for curve_id, label, color, logarithmic in NINE_CURVES:
                if curve_id not in log.curves:
                    continue
                values = np.asarray(log.curves[curve_id], dtype=float)[indices]
                mask_source = np.asarray(log.masks.get(curve_id, np.isfinite(log.curves[curve_id])), dtype=bool)[indices]
                values = np.where(mask_source, values, np.nan)
                valid_count = int(np.sum(np.isfinite(values)))
                if valid_count == 0:
                    continue
                info = log.curve_info.get(curve_id)
                curves.append({
                    "id": curve_id,
                    "label": label,
                    "unit": info.standard_unit if info is not None else "",
                    "color": color,
                    "scale": "log" if logarithmic else "linear",
                    "values": _nullable_list(values),
                    "validCount": valid_count,
                })
            if not curves:
                continue
            source_name = Path(log.source).name
            display_name = entity.canonical_name if len(entity.logs) == 1 else f"{entity.canonical_name} · {source_name}"
            result.append({
                "id": f"{entity.well_uid}:{log_index}",
                "name": display_name,
                "wellName": entity.canonical_name,
                "source": log.source,
                "version": log.version,
                "depthUnit": "m",
                "depth": _nullable_list(sampled_depth),
                "curves": curves,
                "coverage": f"{len(curves)}/9",
            })
    return result


def _trajectory_payloads(pipeline: Any, max_points: int = 160) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for entity in pipeline.registry.entities.values():
        head = entity.preferred_head
        trajectory = entity.preferred_trajectory
        if trajectory is not None:
            md = np.asarray(trajectory.md, dtype=float)
            tvd = np.asarray(trajectory.tvd, dtype=float)
            if trajectory.x is not None and trajectory.y is not None:
                x = np.asarray(trajectory.x, dtype=float)
                y = np.asarray(trajectory.y, dtype=float)
            elif head is not None and head.x is not None and head.y is not None:
                x = float(head.x) + np.asarray(trajectory.x_offset, dtype=float)
                y = float(head.y) + np.asarray(trajectory.y_offset, dtype=float)
            else:
                continue
            count = min(len(md), len(tvd), len(x), len(y))
            valid = np.isfinite(md[:count]) & np.isfinite(tvd[:count]) & np.isfinite(x[:count]) & np.isfinite(y[:count])
            indices = np.flatnonzero(valid)
            if indices.size < 2:
                continue
            indices = indices[_sample_positions(indices.size, max_points)]
            trajectories.append({
                "name": entity.canonical_name,
                "confidence": round(float(trajectory.confidence), 4),
                "x": _finite_list(x[indices]),
                "y": _finite_list(y[indices]),
                "tvd": _finite_list(tvd[indices]),
                "md": _finite_list(md[indices]),
                "geometryMethod": "真实井轨迹",
            })
            continue

        if head is None or head.x is None or head.y is None:
            continue
        log_depths = [np.asarray(log.depth, dtype=float) for log in entity.logs if len(log.depth)]
        finite_parts = [depth[np.isfinite(depth)] for depth in log_depths if np.any(np.isfinite(depth))]
        finite_depths = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=float)
        total_depth = float(np.max(finite_depths)) if finite_depths.size else float(head.total_depth_md or 1.0)
        trajectories.append({
            "name": f"{entity.canonical_name}（直井预览）",
            "confidence": round(min(float(head.confidence), 0.5), 4),
            "x": [float(head.x), float(head.x)],
            "y": [float(head.y), float(head.y)],
            "tvd": [0.0, total_depth],
            "md": [0.0, total_depth],
            "geometryMethod": "缺少真实轨迹，仅按井口直井预览",
        })
    return trajectories


def _embedded_wells(
    geometry: Any,
    inline_values: np.ndarray,
    crossline_values: np.ndarray,
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if geometry.x is None or geometry.y is None or geometry.inline is None or geometry.crossline is None:
        return []
    x = np.asarray(geometry.x, dtype=float)
    y = np.asarray(geometry.y, dtype=float)
    inline = np.asarray(geometry.inline, dtype=float)
    crossline = np.asarray(geometry.crossline, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(inline) & np.isfinite(crossline)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 3:
        return []
    fit_indices = valid_indices[_sample_positions(valid_indices.size, 5000)]
    design = np.column_stack((x[fit_indices], y[fit_indices], np.ones(fit_indices.size)))
    try:
        inline_coefficients = np.linalg.lstsq(design, inline[fit_indices], rcond=None)[0]
        crossline_coefficients = np.linalg.lstsq(design, crossline[fit_indices], rcond=None)[0]
    except np.linalg.LinAlgError:
        return []

    inline_min, inline_max = float(inline_values[0]), float(inline_values[-1])
    crossline_min, crossline_max = float(crossline_values[0]), float(crossline_values[-1])
    inline_span = max(abs(inline_max - inline_min), 1.0)
    crossline_span = max(abs(crossline_max - crossline_min), 1.0)
    result: list[dict[str, Any]] = []
    for trajectory in trajectories:
        tx = np.asarray(trajectory["x"], dtype=float)
        ty = np.asarray(trajectory["y"], dtype=float)
        tvd = np.asarray(trajectory["tvd"], dtype=float)
        if tx.size < 2 or tx.size != ty.size or tx.size != tvd.size:
            continue
        points = np.column_stack((tx, ty, np.ones(tx.size)))
        mapped_inline = points @ inline_coefficients
        mapped_crossline = points @ crossline_coefficients
        normalized_x = (mapped_crossline - crossline_min) / crossline_span
        normalized_y = (mapped_inline - inline_min) / inline_span
        if not np.any((normalized_x >= -0.15) & (normalized_x <= 1.15) & (normalized_y >= -0.15) & (normalized_y <= 1.15)):
            continue
        tvd_min = float(np.nanmin(tvd))
        tvd_span = max(float(np.nanmax(tvd) - tvd_min), 1.0)
        normalized_z = (tvd - tvd_min) / tvd_span
        head_distance = float(np.min(np.hypot(x[fit_indices] - tx[0], y[fit_indices] - ty[0])))
        result.append({
            "name": trajectory["name"],
            "x": np.clip(normalized_x, 0.0, 1.0).round(5).tolist(),
            "y": np.clip(normalized_y, 0.0, 1.0).round(5).tolist(),
            "z": np.clip(normalized_z, 0.0, 1.0).round(5).tolist(),
            "geometryMethod": f"{trajectory['geometryMethod']}；垂向为TVD比例预览，未冒充时深标定",
            "geometryConfidence": round(float(trajectory["confidence"]) * 0.65, 4),
            "distance": round(head_distance, 3),
        })
    return result


def _build_line_preview(
    asset: Any,
    reader: Any,
    geometry: Any,
    *,
    max_trace_samples: int,
    max_time_samples: int,
) -> dict[str, Any] | None:
    trace_indices = _sample_positions(int(geometry.trace_count), max_trace_samples)
    time_indices = _sample_positions(int(geometry.samples_per_trace), max_time_samples)
    if trace_indices.size < 2 or time_indices.size < 2:
        return None

    image = np.zeros((time_indices.size, trace_indices.size), dtype=np.float32)
    for column, trace_index in enumerate(trace_indices):
        trace = np.asarray(reader.read_trace(int(trace_index)), dtype=np.float32)
        image[:, column] = trace[time_indices]

    finite = np.abs(image[np.isfinite(image)])
    scale = float(np.percentile(finite, 99.0)) if finite.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    encoded = np.clip(np.nan_to_num(image / scale) * 127.0, -127, 127).astype(np.int8)

    inline = np.asarray(geometry.inline, dtype=float) if geometry.inline is not None else np.asarray([])
    crossline = np.asarray(geometry.crossline, dtype=float) if geometry.crossline is not None else np.asarray([])
    inline_count = int(np.unique(inline[np.isfinite(inline)]).size) if inline.size else 0
    crossline_count = int(np.unique(crossline[np.isfinite(crossline)]).size) if crossline.size else 0
    if inline_count >= crossline_count and inline.size:
        line_axis = "Inline"
        trace_values = inline[trace_indices]
    elif crossline.size:
        line_axis = "Crossline"
        trace_values = crossline[trace_indices]
    else:
        line_axis = "Trace"
        trace_values = trace_indices.astype(float)

    distance_values: np.ndarray | None = None
    if geometry.x is not None and geometry.y is not None:
        x = np.asarray(geometry.x, dtype=float)[trace_indices]
        y = np.asarray(geometry.y, dtype=float)[trace_indices]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            steps = np.hypot(np.diff(x), np.diff(y))
            distance_values = np.r_[0.0, np.cumsum(steps)]

    time_values = np.asarray(geometry.time_axis, dtype=float)[time_indices]
    return {
        "name": asset.path.name,
        "path": str(Path(asset.path)),
        "lineAxis": line_axis,
        "traceValues": _nullable_list(trace_values),
        "distanceValues": _nullable_list(distance_values) if distance_values is not None else [],
        "timeValues": _nullable_list(time_values),
        "image": {
            "shape": [int(value) for value in encoded.shape],
            "encoding": "base64-int8",
            "values": base64.b64encode(encoded.tobytes(order="C")).decode("ascii"),
        },
        "preview": {
            "sampled_traces": int(trace_indices.size),
            "source_trace_count": int(geometry.trace_count),
            "amplitude_scale_p99": scale,
        },
    }


def build_visualization_preview(
    pipeline: Any,
    *,
    max_volumes: int = 3,
    max_lines: int = 12,
    max_time_samples: int = 72,
    max_inline_samples: int = 24,
    max_crossline_samples: int = 32,
    max_line_time_samples: int = 320,
    max_line_trace_samples: int = 480,
) -> dict[str, Any]:
    """从当前任务稀疏读取真实SEG-Y，生成模型无关的二维/三维预览。"""
    trajectories = _trajectory_payloads(pipeline)
    volumes: list[dict[str, Any]] = []
    lines2d: list[dict[str, Any]] = []
    issues: list[str] = []
    for asset, reader in pipeline.seismic:
        geometry = reader.geometry
        if geometry is None:
            continue
        unique_inline = (
            np.unique(np.asarray(geometry.inline, dtype=np.int64))
            if geometry.inline is not None
            else np.asarray([], dtype=np.int64)
        )
        unique_crossline = (
            np.unique(np.asarray(geometry.crossline, dtype=np.int64))
            if geometry.crossline is not None
            else np.asarray([], dtype=np.int64)
        )
        if unique_inline.size < 2 or unique_crossline.size < 2:
            if len(lines2d) >= max_lines:
                continue
            try:
                line_preview = _build_line_preview(
                    asset,
                    reader,
                    geometry,
                    max_trace_samples=max_line_trace_samples,
                    max_time_samples=max_line_time_samples,
                )
            except Exception as exc:
                issues.append(f"{asset.path.name}轻量二维预览失败：{exc}")
                continue
            if line_preview is not None:
                lines2d.append(line_preview)
            continue
        if len(volumes) >= max_volumes:
            continue
        inline_values = unique_inline[_sample_positions(unique_inline.size, max_inline_samples)]
        crossline_values = unique_crossline[_sample_positions(unique_crossline.size, max_crossline_samples)]
        time_indices = _sample_positions(int(geometry.samples_per_trace), max_time_samples)
        pair_to_trace: dict[tuple[int, int], int] = {}
        for trace_index, (inline_value, crossline_value) in enumerate(zip(geometry.inline, geometry.crossline)):
            pair_to_trace.setdefault((int(inline_value), int(crossline_value)), trace_index)

        cube = np.zeros((time_indices.size, inline_values.size, crossline_values.size), dtype=np.float32)
        loaded_traces = 0
        try:
            for inline_index, inline_value in enumerate(inline_values):
                for crossline_index, crossline_value in enumerate(crossline_values):
                    trace_index = pair_to_trace.get((int(inline_value), int(crossline_value)))
                    if trace_index is None:
                        continue
                    trace = np.asarray(reader.read_trace(trace_index), dtype=np.float32)
                    cube[:, inline_index, crossline_index] = trace[time_indices]
                    loaded_traces += 1
        except Exception as exc:
            issues.append(f"{asset.path.name}轻量三维预览失败：{exc}")
            continue
        if loaded_traces == 0:
            issues.append(f"{asset.path.name}未形成可读取的Inline/Crossline稀疏网格")
            continue
        finite = np.abs(cube[np.isfinite(cube)])
        scale = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        encoded_cube = np.clip(np.nan_to_num(cube / scale) * 127.0, -127, 127).astype(np.int8)
        time_values = np.asarray(geometry.time_axis, dtype=float)[time_indices]
        volumes.append({
            "name": asset.path.name,
            "path": str(Path(asset.path)),
            "inline": int(inline_values[len(inline_values) // 2]),
            "crossline": int(crossline_values[len(crossline_values) // 2]),
            "time": float(time_values[len(time_values) // 2]),
            "timeMax": float(time_values[-1]),
            "inlineRange": [int(inline_values[0]), int(inline_values[-1])],
            "crosslineRange": [int(crossline_values[0]), int(crossline_values[-1])],
            "inlineValues": [int(value) for value in inline_values],
            "crosslineValues": [int(value) for value in crossline_values],
            "timeValues": [round(float(value), 3) for value in time_values],
            "defaultIndices": [int(time_values.size // 2), int(inline_values.size // 2), int(crossline_values.size // 2)],
            "cube": {
                "shape": [int(value) for value in encoded_cube.shape],
                "encoding": "base64-int8",
                "values": base64.b64encode(encoded_cube.tobytes(order="C")).decode("ascii"),
            },
            "embeddedWells": _embedded_wells(geometry, inline_values, crossline_values, trajectories),
            "preview": {
                "loaded_traces": loaded_traces,
                "source_trace_count": int(geometry.trace_count),
                "amplitude_scale_p99": scale,
                "vertical_note": "井轨迹垂向仅作TVD比例预览；无时深关系时不进入正式垂向融合",
            },
        })
    return {
        "volumes": volumes,
        "lines2d": lines2d,
        "trajectories": trajectories,
        "wellLogs": _well_log_payloads(pipeline),
        "issues": issues,
        "source": "当前任务真实数据的稀疏降采样预览",
    }
