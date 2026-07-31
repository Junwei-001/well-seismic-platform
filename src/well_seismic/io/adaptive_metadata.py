from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .tabular import _rows, read_time_depth, read_trajectory, read_well_heads
from ..models import Trajectory, WellHead


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


@dataclass
class MetadataResult:
    heads: list[WellHead] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    time_depth: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    time_depth_domain: str | None = None
    detected_roles: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    status: str = "待确认"
    accepted: bool = False
    decision_source: str = "rule"


def _map_header(header: list[str], aliases: dict[str, list[str]]) -> dict[str, int]:
    available = {_key(name): i for i, name in enumerate(header)}
    columns: dict[str, int] = {}
    for field, names in aliases.items():
        for name in names:
            if _key(name) in available:
                columns[field] = available[_key(name)]
                break
    return columns


def _time_depth_domain(columns: dict[str, int]) -> str | None:
    """Infer a time-depth table domain only when the header maps it explicitly."""

    depth_column = columns.get("depth")
    if depth_column is None:
        return None
    for domain in ("tvdss", "tvd", "md"):
        if columns.get(domain) == depth_column:
            return domain
    return None


def _numeric_matrix(rows: list[list[str]], start: int) -> np.ndarray | None:
    try:
        width = min(len(row) for row in rows)
        return np.asarray([[float(v) for v in row[start:width]] for row in rows], dtype=float)
    except (ValueError, IndexError):
        return None


def _monotonic_by_well(values: np.ndarray, names: list[str]) -> list[bool]:
    result = []
    for column in range(values.shape[1]):
        valid = True
        for name in set(names):
            series = values[np.asarray([item == name for item in names]), column]
            if series.size > 1 and not np.all(np.diff(series) >= 0):
                valid = False
                break
        result.append(valid)
    return result


def read_adaptive_metadata(path: str | Path, field_aliases: dict[str, list[str]]) -> MetadataResult:
    """Read one metadata file that may contain one or several semantic roles."""
    path = Path(path)
    header, rows = _rows(path)
    result = MetadataResult()
    if not rows:
        result.evidence.append("文件中没有可解析的数据行")
        return result

    columns = _map_header(header, field_aliases) if header else {}
    if header:
        result.evidence.append("检测到表头：" + ",".join(header))
        if "well_name" not in columns and rows and not _is_number(rows[0][0]):
            columns["well_name"] = 0
        if {"x", "y"}.issubset(columns):
            raw_heads = read_well_heads(path, {"columns": columns})
            unique_heads: dict[str, WellHead] = {}
            for head in raw_heads:
                unique_heads.setdefault(_key(head.well_name), head)
            result.heads = list(unique_heads.values())
            result.detected_roles.append("井位与海拔")
        trajectory_fields = {"tvd", "tvdss", "inclination", "azimuth", "x_offset", "y_offset"}
        if "md" in columns and trajectory_fields.intersection(columns):
            result.trajectories = read_trajectory(path, {"columns": columns, "well_name": path.stem})
            result.detected_roles.append("井轨迹")
        if {"depth", "time"}.issubset(columns):
            result.time_depth = read_time_depth(path, {"columns": columns, "well_name": path.stem})
            result.time_depth_domain = _time_depth_domain(columns)
            result.detected_roles.append("时深关系")
            if result.time_depth_domain:
                result.evidence.append(f"时深表深度域由表头识别为{result.time_depth_domain.upper()}")
            else:
                result.evidence.append("时深表表头未明确MD/TVD/TVDSS深度域")
        if result.detected_roles:
            result.confidence = 0.95
            result.status = "已识别"
            result.accepted = True
            return result
        result.confidence = 1.0
        result.status = "不参与"
        result.evidence.append("表头不含井位、海拔或井轨迹所需字段，已作为其他井相关资料保留但不进入基础匹配")
        return result

    # Headerless fallback uses structure and value dimensions; filename only adds weak evidence.
    first_is_name = not _is_number(rows[0][0])
    numeric = _numeric_matrix(rows, 1 if first_is_name else 0)
    if numeric is None or numeric.shape[1] < 2:
        result.evidence.append("无表头且数值列不足，不能可靠识别")
        return result
    names = [row[0] for row in rows] if first_is_name else [path.stem] * len(rows)
    unique_names = len(set(names))
    group_sizes = {name: names.count(name) for name in set(names)}
    max_group_rows = max(group_sizes.values())
    row_count = len(rows)
    medians = np.nanmedian(np.abs(numeric), axis=0)
    monotonic = _monotonic_by_well(numeric, names)
    filename_hint = bool(re.search(r"(^|[_\-])(TD|TIME.?DEPTH)([_\-.]|$)", path.stem, re.I))

    # Few rows + first two very large coordinates is a multi-well well-head table.
    if row_count <= 1000 and numeric.shape[1] >= 2 and medians[0] > 10000 and medians[1] > 10000:
        options = {"columns": {"well_name": 0 if first_is_name else None, "x": 1 if first_is_name else 0, "y": 2 if first_is_name else 1}}
        if numeric.shape[1] >= 3:
            options["columns"]["kb"] = 3 if first_is_name else 2
        if numeric.shape[1] >= 4:
            options["columns"]["total_depth_md"] = 4 if first_is_name else 3
        result.heads = read_well_heads(path, options)
        result.detected_roles.append("井位与海拔")
        result.evidence.append("无表头：少量记录、井名列和两列投影坐标量级")

    # Many rows + monotonic first numeric column is normally a trajectory survey.
    if row_count >= 20 and max_group_rows >= 2 and monotonic[0] and not filename_hint:
        trajectory_columns: dict[str, int] = {"md": 1 if first_is_name else 0}
        if first_is_name:
            trajectory_columns["well_name"] = 0
        if numeric.shape[1] >= 3:
            trajectory_columns["x_offset"] = 2 if first_is_name else 1
            trajectory_columns["y_offset"] = 3 if first_is_name else 2
        result.trajectories = read_trajectory(path, {"columns": trajectory_columns, "well_name": path.stem})
        result.detected_roles.append("井轨迹候选")
        result.evidence.append("无表头：首数值列单调且记录数较多")

    unresolved_time_depth_owner = False
    if numeric.shape[1] >= 2 and max_group_rows >= 2 and monotonic[0] and monotonic[1] and (filename_hint or row_count < 5000):
        td_columns = {"depth": 1 if first_is_name else 0, "time": 2 if first_is_name else 1}
        if first_is_name:
            td_columns["well_name"] = 0
            result.time_depth = read_time_depth(path, {"columns": td_columns})
            result.detected_roles.append("时深关系候选")
        else:
            inferred_well = re.sub(r"(?i)([_\-]?(P[_\-]?)?TD|[_\-]?TIME.?DEPTH)$", "", path.stem).strip("_- ")
            if inferred_well:
                result.time_depth = read_time_depth(path, {"columns": td_columns, "well_name": inferred_well})
                result.detected_roles.append("时深关系候选")
                result.evidence.append(f"井名仅由文件名弱推断：{inferred_well}")
            else:
                unresolved_time_depth_owner = True
                result.detected_roles.append("未归属井的时深关系候选")
                result.evidence.append("时深表缺少井名列，文件名也不能确定所属井")
        result.evidence.append("无表头：前两数值列单调；文件名提示仅作为辅助证据" if filename_hint else "无表头：前两数值列单调，需通过物理合理性复核")

    if result.detected_roles:
        # Ambiguous headerless files may legitimately provide multiple roles, but remain reviewable.
        result.confidence = 0.75 if filename_hint or len(result.detected_roles) == 1 else 0.55
        if unresolved_time_depth_owner:
            result.confidence = min(result.confidence, 0.55)
        result.status = "已识别" if result.confidence >= 0.7 else "待确认"
        result.accepted = result.confidence >= 0.7
    else:
        result.evidence.append("未达到任何元数据角色的识别阈值")
    return result


def apply_llm_metadata_decision(result: MetadataResult, choice: str, confidence: float) -> MetadataResult:
    """Apply only a role already backed by parsed candidate payload; never invent missing values."""
    if choice == "井位与海拔" and result.heads:
        result.trajectories = []
        result.time_depth = {}
    elif choice == "井轨迹" and result.trajectories:
        result.heads = []
        result.time_depth = {}
    elif choice == "时深关系" and result.time_depth and not any(
        "未归属井" in role for role in result.detected_roles
    ):
        result.heads = []
        result.trajectories = []
    else:
        return result
    result.detected_roles = [choice]
    result.confidence = float(confidence)
    result.status = "LLM已确认"
    result.accepted = True
    result.decision_source = "llm"
    result.evidence.append(f"LLM仅在候选集合内确认主角色：{choice}")
    return result


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
