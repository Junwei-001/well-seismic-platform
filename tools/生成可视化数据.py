from __future__ import annotations

import json
import sys
from base64 import b64encode
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from well_seismic.config import load_yaml
from well_seismic.io.segy import SegyReader
from well_seismic.pipeline import WellSeismicPipeline


INPUT = PROJECT / "比赛输入示例_自动识别"
CONFIG = PROJECT / "configs"
OUTPUT = PROJECT / "可视化界面" / "可视化数据.json"


def quantize_cube(cube: np.ndarray) -> dict:
    cube = np.nan_to_num(np.asarray(cube, dtype=np.float32))
    finite = np.abs(cube[np.isfinite(cube)])
    scale = float(np.percentile(finite, 98.5)) if finite.size else 1.0
    scale = scale or 1.0
    values = np.rint(np.clip(cube / scale, -1, 1) * 127).astype(np.int8)
    return {
        "shape": [int(value) for value in values.shape],
        "encoding": "base64-int8",
        "values": b64encode(values.tobytes(order="C")).decode("ascii"),
    }
def embedded_wells(pipeline: WellSeismicPipeline, geom, il_select: np.ndarray, xl_select: np.ndarray) -> list[dict]:
    if geom.x is None or geom.y is None or geom.inline is None or geom.crossline is None:
        return []
    valid_traces = np.isfinite(geom.x) & np.isfinite(geom.y)
    trace_indices = np.flatnonzero(valid_traces)
    tree = cKDTree(np.column_stack([geom.x[valid_traces], geom.y[valid_traces]]))
    result = []
    for entity in pipeline.registry.entities.values():
        trajectory = entity.preferred_trajectory
        head = entity.preferred_head
        geometry_method = "实测/设计井轨迹"
        geometry_confidence = float(trajectory.confidence) if trajectory is not None else 0.35
        if trajectory is not None:
            if trajectory.x is not None and trajectory.y is not None:
                x, y = np.asarray(trajectory.x), np.asarray(trajectory.y)
            elif head is not None and head.x is not None and head.y is not None:
                x = float(head.x) + np.asarray(trajectory.x_offset)
                y = float(head.y) + np.asarray(trajectory.y_offset)
            else:
                continue
            tvd = np.asarray(trajectory.tvd)
        elif head is not None and head.x is not None and head.y is not None and entity.logs:
            log = max(entity.logs, key=lambda item: len(item.depth))
            tvd = np.asarray(log.depth, dtype=float)
            x = np.full_like(tvd, float(head.x))
            y = np.full_like(tvd, float(head.y))
            geometry_method = "井口直井回退"
        else:
            continue
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(tvd)
        x, y, tvd = x[finite], y[finite], tvd[finite]
        if not len(tvd):
            continue
        selected = np.unique(np.linspace(0, len(tvd) - 1, min(120, len(tvd)), dtype=int))
        x, y, tvd = x[selected], y[selected], tvd[selected]
        distance, local = tree.query(np.column_stack([x, y]), k=1)
        if float(np.nanmin(distance)) > 1000.0:
            continue
        nearest = trace_indices[np.asarray(local, dtype=int)]
        inline = np.asarray(geom.inline)[nearest]
        crossline = np.asarray(geom.crossline)[nearest]
        transform = pipeline._depth_time_transform(entity)
        seismic_time = np.asarray(transform.depth_to_time(tvd), dtype=float)
        valid_time = np.isfinite(seismic_time) & (seismic_time >= 0) & (seismic_time <= float(geom.time_axis[-1]))
        if valid_time.sum() >= 2:
            x, y, tvd, inline, crossline, distance, seismic_time = (
                values[valid_time] for values in (x, y, tvd, inline, crossline, distance, seismic_time)
            )
        else:
            seismic_time = (tvd - np.nanmin(tvd)) / max(float(np.nanmax(tvd) - np.nanmin(tvd)), 1.0) * float(geom.time_axis[-1])
        result.append({
            "name": entity.canonical_name,
            "x": np.round(np.clip(np.interp(crossline, xl_select, np.linspace(0, 1, len(xl_select))), 0, 1), 5).tolist(),
            "y": np.round(np.clip(np.interp(inline, il_select, np.linspace(0, 1, len(il_select))), 0, 1), 5).tolist(),
            "z": np.round(np.clip(seismic_time / float(geom.time_axis[-1]), 0, 1), 5).tolist(),
            "geometryMethod": geometry_method,
            "geometryConfidence": round(geometry_confidence, 2),
            "distance": round(float(np.nanmedian(distance)), 2),
        })
    return result


def volume_data(name: str, path: Path, pipeline: WellSeismicPipeline) -> dict:
    cfg = load_yaml(CONFIG / "segy_profiles.yaml")
    reader = SegyReader(path, cfg, {"profile": "standard_3d"})
    geom = reader.inspect()
    if geom.inline is None or geom.crossline is None:
        raise ValueError(f"无法获得三维网格：{path}")
    inlines = np.unique(geom.inline)
    crosslines = np.unique(geom.crossline)
    inline_mid = int(inlines[len(inlines) // 2])
    crossline_mid = int(crosslines[len(crosslines) // 2])
    sample_indices = np.unique(np.linspace(0, geom.samples_per_trace - 1, min(96, geom.samples_per_trace), dtype=int))
    il_select = inlines[np.unique(np.linspace(0, len(inlines) - 1, min(40, len(inlines)), dtype=int))]
    xl_select = crosslines[np.unique(np.linspace(0, len(crosslines) - 1, min(52, len(crosslines)), dtype=int))]
    pair_to_index = {(int(il), int(xl)): i for i, (il, xl) in enumerate(zip(geom.inline, geom.crossline))}
    cube = np.zeros((len(sample_indices), len(il_select), len(xl_select)), dtype=np.float32)
    for i, il in enumerate(il_select):
        for j, xl in enumerate(xl_select):
            trace_index = pair_to_index.get((int(il), int(xl)))
            if trace_index is not None:
                cube[:, i, j] = reader.read_trace(trace_index)[sample_indices]
    inline_index = len(il_select) // 2
    crossline_index = len(xl_select) // 2
    time_index = min(range(len(sample_indices)), key=lambda index: abs(float(geom.time_axis[sample_indices[index]]) - min(1500.0, float(geom.time_axis[-1]) * 0.65)))
    return {
        "name": name, "path": str(path),
        "inline": int(il_select[inline_index]), "crossline": int(xl_select[crossline_index]),
        "time": round(float(geom.time_axis[sample_indices[time_index]]), 1), "timeMax": round(float(geom.time_axis[-1]), 1),
        "inlineRange": [int(inlines.min()), int(inlines.max())], "crosslineRange": [int(crosslines.min()), int(crosslines.max())],
        "inlineValues": [int(value) for value in il_select],
        "crosslineValues": [int(value) for value in xl_select],
        "timeValues": [round(float(geom.time_axis[index]), 1) for index in sample_indices],
        "defaultIndices": [int(time_index), int(inline_index), int(crossline_index)],
        "cube": quantize_cube(cube),
        "embeddedWells": embedded_wells(pipeline, geom, il_select, xl_select),
    }


def trajectory_data(pipeline: WellSeismicPipeline) -> list[dict]:
    output = []
    for entity in pipeline.registry.entities.values():
        trajectory = entity.preferred_trajectory
        head = entity.preferred_head
        if trajectory is None:
            continue
        if trajectory.x is not None and trajectory.y is not None:
            x, y = trajectory.x, trajectory.y
        elif head is not None and head.x is not None and head.y is not None:
            x, y = head.x + trajectory.x_offset, head.y + trajectory.y_offset
        else:
            continue
        count = min(220, len(trajectory.md))
        selected = np.unique(np.linspace(0, len(trajectory.md) - 1, count, dtype=int))
        output.append({
            "name": entity.canonical_name, "confidence": round(float(trajectory.confidence), 2),
            "x": np.round(x[selected], 2).tolist(), "y": np.round(y[selected], 2).tolist(),
            "tvd": np.round(trajectory.tvd[selected], 2).tolist(), "md": np.round(trajectory.md[selected], 2).tolist(),
        })
    return output


def main() -> None:
    pipeline = WellSeismicPipeline.from_input_root(INPUT, CONFIG).ingest()
    volumes = [
        volume_data("综合解释三维地震", INPUT / "01_地震数据" / "三维地震" / "综合解释训练数据" / "mig.sgy", pipeline),
        volume_data("环青三维地震", INPUT / "01_地震数据" / "三维地震" / "环青反演数据" / "sesmic3D.sgy", pipeline),
    ]
    payload = {"volumes": volumes, "trajectories": trajectory_data(pipeline)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print({"volumes": len(volumes), "trajectories": len(payload["trajectories"]), "bytes": OUTPUT.stat().st_size, "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
