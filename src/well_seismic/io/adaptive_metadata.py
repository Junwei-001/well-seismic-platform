from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np

from ..content_identity import canonical_sha256, file_sha256
from ..models import Trajectory, WellHead
from ..trajectory import minimum_curvature
from ..vertical_datum import length_to_metres
from .tabular import (
    ColumnMappingConflict,
    _map_columns_from_aliases,
    _normalize_length_unit,
    _normalized_column,
    _rows_with_evidence,
    read_petrel_dev_header,
    read_time_depth,
    read_trajectory,
    read_well_heads,
)
from .opendtect_well import read_opendtect_well_track


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


@dataclass
class MetadataResult:
    heads: list[WellHead] = dataclass_field(default_factory=list)
    trajectories: list[Trajectory] = dataclass_field(default_factory=list)
    time_depth: dict[str, tuple[np.ndarray, np.ndarray]] = dataclass_field(default_factory=dict)
    time_depth_domain: str | None = None
    detected_roles: list[str] = dataclass_field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = dataclass_field(default_factory=list)
    status: str = "待确认"
    accepted: bool = False
    decision_source: str = "rule"
    deterministic_trajectory_schema: dict[str, object] | None = None


def _map_header(
    header: list[str],
    aliases: dict[str, list[str]],
    *,
    rows: list[list[object]] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, int]:
    raw_well_aliases = aliases.get("well_name", [])
    well_aliases = (
        [raw_well_aliases]
        if isinstance(raw_well_aliases, str)
        else list(raw_well_aliases)
    )
    normalized_aliases = {
        _normalized_column(value) for value in well_aliases
    }
    candidate_indices = [
        index
        for index, value in enumerate(header)
        if _normalized_column(value) in normalized_aliases
    ]
    if len(candidate_indices) <= 1 or rows is None:
        return _map_columns_from_aliases(header, aliases)

    reduced_aliases = {
        field: names for field, names in aliases.items() if field != "well_name"
    }
    columns = _map_columns_from_aliases(header, reduced_aliases)
    populated_rows = [row for row in rows if any(str(value or "").strip() for value in row)]

    def values(index: int) -> list[str]:
        return [
            _normalized_column(str(row[index]))
            for row in populated_rows
            if index < len(row) and str(row[index] or "").strip()
        ]

    metrics = {
        index: (len(values(index)), len(set(values(index))))
        for index in candidate_indices
    }
    complete = [
        index for index in candidate_indices if metrics[index][0] == len(populated_rows)
    ]
    selected: int | None = complete[0] if len(complete) == 1 else None

    if selected is None:
        candidate_values = {index: values(index) for index in candidate_indices}
        ratios = {
            index: (
                metrics[index][1] / metrics[index][0]
                if metrics[index][0]
                else 0.0
            )
            for index in candidate_indices
        }
        high_identity = [index for index in candidate_indices if ratios[index] >= 0.8]
        low_identity = [index for index in candidate_indices if ratios[index] <= 0.2]
        if len(high_identity) == 1 and len(low_identity) == len(candidate_indices) - 1:
            selected = high_identity[0]
        elif len(complete) > 1:
            reference = complete[0]
            if all(candidate_values[index] == candidate_values[reference] for index in complete[1:]):
                selected = reference

    if selected is None:
        labels = ",".join(
            f"{index}:{header[index]}" for index in candidate_indices
        )
        raise ColumnMappingConflict(
            [f"field_matches_multiple_columns:well_name={labels}"]
        )
    columns["well_name"] = selected
    if evidence is not None:
        coverage, distinct = metrics[selected]
        evidence.append(
            "井标识列自动选择："
            f"{header[selected]}（非空{coverage}/{len(populated_rows)}，唯一值{distinct}）"
        )
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


def _infer_headerless_md_offsets_angles(
    path: Path,
    rows: list[list[str]],
    *,
    first_is_name: bool,
    options: dict[str, object],
) -> tuple[list[Trajectory], dict[str, object]] | None:
    """Recognise ``MD DX DY AZI INC`` only from a closed physics check.

    A five-number headerless row is not, by itself, a column declaration.  We
    therefore test both possible meanings of its two angle columns and accept
    one only when minimum-curvature integration reproduces the supplied DX/DY
    path to survey-export precision.  The counterfactual must fail and the
    observed path must contain material horizontal displacement; vertical or
    otherwise degenerate surveys deliberately stay on the conservative
    headerless fallback.

    Values are parsed through :func:`read_trajectory` first, so its strict
    per-field length-unit contract remains authoritative.  In particular this
    rule cannot be used to make an unknown MD/offset unit silently become
    metres.
    """

    expected_width = 6 if first_is_name else 5
    if len(rows) < 20 or any(len(row) != expected_width for row in rows):
        return None

    md_column = 1 if first_is_name else 0
    x_offset_column = md_column + 1
    y_offset_column = md_column + 2
    fourth_column = md_column + 3
    fifth_column = md_column + 4
    base_columns: dict[str, int] = {
        "md": md_column,
        "x_offset": x_offset_column,
        "y_offset": y_offset_column,
    }
    if first_is_name:
        base_columns["well_name"] = 0

    candidate_specs = (
        ("MD_DX_DY_AZI_INC", fourth_column, fifth_column),
        ("MD_DX_DY_INC_AZI", fifth_column, fourth_column),
    )
    candidate_receipts: list[dict[str, object]] = []
    accepted: list[tuple[list[Trajectory], dict[str, int], dict[str, object]]] = []

    for label, azimuth_column, inclination_column in candidate_specs:
        columns = {
            **base_columns,
            "azimuth": azimuth_column,
            "inclination": inclination_column,
        }
        receipt: dict[str, object] = {
            "mapping": label,
            "columns": dict(columns),
            "accepted": False,
            "well_checks": [],
        }
        try:
            trajectories = read_trajectory(
                path,
                {
                    **options,
                    "columns": columns,
                    "well_name": path.stem,
                },
            )
        except Exception as exc:
            receipt["rejection_reason"] = (
                f"parse_failed:{type(exc).__name__}:{str(exc)[:240]}"
            )
            candidate_receipts.append(receipt)
            continue

        valid_candidate = bool(trajectories)
        well_checks: list[dict[str, object]] = []
        for trajectory in trajectories:
            md = np.asarray(trajectory.md, dtype=float)
            x_offset = np.asarray(trajectory.x_offset, dtype=float)
            y_offset = np.asarray(trajectory.y_offset, dtype=float)
            inclination = np.asarray(trajectory.inclination, dtype=float)
            azimuth = np.asarray(trajectory.azimuth, dtype=float)
            if (
                md.size < 20
                or not (
                    md.shape
                    == x_offset.shape
                    == y_offset.shape
                    == inclination.shape
                    == azimuth.shape
                )
                or not all(
                    np.isfinite(values).all()
                    for values in (md, x_offset, y_offset, inclination, azimuth)
                )
                or np.any(np.diff(md) < 0.0)
                or float(np.min(inclination)) < 0.0
                or float(np.max(inclination)) > 180.0
                or float(np.min(azimuth)) < 0.0
                or float(np.max(azimuth)) > 360.0
            ):
                valid_candidate = False
                well_checks.append(
                    {
                        "well_name": trajectory.well_name,
                        "accepted": False,
                        "reason": "invalid_station_or_angle_domain",
                    }
                )
                continue

            _, predicted_east, predicted_north = minimum_curvature(
                md,
                inclination,
                azimuth,
            )
            observed = np.column_stack(
                (x_offset - x_offset[0], y_offset - y_offset[0])
            )
            predicted = np.column_stack((predicted_east, predicted_north))
            vector_error = np.linalg.norm(observed - predicted, axis=1)
            horizontal_extent_m = float(
                np.max(np.linalg.norm(observed, axis=1))
            )
            md_span_m = float(md[-1] - md[0])
            maximum_vector_error_m = float(np.max(vector_error))
            rms_vector_error_m = float(np.sqrt(np.mean(vector_error**2)))
            # Four-decimal DX/DY plus three-decimal survey angles accumulate a
            # few millimetres over multi-kilometre paths.  Both an absolute
            # export-precision bound and a relative path-fit bound must pass.
            maximum_error_tolerance_m = max(0.01, 5e-6 * md_span_m)
            relative_path_error = maximum_vector_error_m / max(
                horizontal_extent_m, 1.0
            )
            material_horizontal_path = horizontal_extent_m >= max(
                1.0, 1e-4 * md_span_m
            )
            passed = bool(
                md_span_m > 0.0
                and material_horizontal_path
                and maximum_vector_error_m <= maximum_error_tolerance_m
                and relative_path_error <= 1e-3
            )
            well_checks.append(
                {
                    "well_name": trajectory.well_name,
                    "station_count": int(md.size),
                    "md_span_m": md_span_m,
                    "horizontal_extent_m": horizontal_extent_m,
                    "maximum_vector_error_m": maximum_vector_error_m,
                    "rms_vector_error_m": rms_vector_error_m,
                    "maximum_error_tolerance_m": maximum_error_tolerance_m,
                    "relative_path_error": relative_path_error,
                    "material_horizontal_path": material_horizontal_path,
                    "inclination_range_deg": [
                        float(np.min(inclination)),
                        float(np.max(inclination)),
                    ],
                    "azimuth_range_deg": [
                        float(np.min(azimuth)),
                        float(np.max(azimuth)),
                    ],
                    "accepted": passed,
                }
            )
            valid_candidate = valid_candidate and passed

        receipt["well_checks"] = well_checks
        receipt["accepted"] = valid_candidate
        candidate_receipts.append(receipt)
        if valid_candidate:
            accepted.append((trajectories, columns, receipt))

    if len(accepted) != 1:
        return None

    source_sha_before = file_sha256(path)
    trajectories, columns, accepted_receipt = accepted[0]
    source_sha_after = file_sha256(path)
    if source_sha_after != source_sha_before:
        return None

    evidence: dict[str, object] = {
        "contract_version": "well-seismic.headerless-trajectory-physics.v1",
        "status": "verified_for_current_snapshot",
        "rule_id": "unique_minimum_curvature_md_dx_dy_angle_mapping.v1",
        "source_path": str(path.resolve()),
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "original_preserved": True,
        "row_contract": "optional_well_name_plus_exactly_five_numeric_columns",
        "candidate_count": len(candidate_receipts),
        "accepted_candidate_count": 1,
        "accepted_mapping": accepted_receipt["mapping"],
        "accepted_columns": dict(columns),
        "candidates": candidate_receipts,
        "acceptance_policy": {
            "minimum_station_count_per_well": 20,
            "inclination_range_deg": [0.0, 180.0],
            "azimuth_range_deg": [0.0, 360.0],
            "minimum_horizontal_extent_m": "max(1.0, 1e-4 * md_span_m)",
            "maximum_vector_error_m": "max(0.01, 5e-6 * md_span_m)",
            "maximum_relative_path_error": 1e-3,
            "counterfactual_must_fail": True,
        },
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    evidence_sha256 = str(evidence["evidence_sha256"])
    for trajectory in trajectories:
        trajectory.confidence = min(float(trajectory.confidence), 0.98)
        trajectory.issues.append(
            "headerless_angle_columns_verified_by_minimum_curvature:"
            + evidence_sha256
        )
    return trajectories, evidence


def read_adaptive_metadata(
    path: str | Path,
    field_aliases: dict[str, list[str]],
    options: dict[str, object] | None = None,
) -> MetadataResult:
    """Read one metadata file that may contain one or several semantic roles."""
    path = Path(path)
    trajectory_options = dict(options or {})
    if path.suffix.casefold() in {".well", ".track"}:
        parsed = read_opendtect_well_track(path, trajectory_options)
        return MetadataResult(
            heads=[parsed.head],
            trajectories=[parsed.trajectory],
            # A well track is spatial MD/TVD geometry only.  In particular,
            # these four columns must never be reinterpreted as checkshot/TWT.
            time_depth={},
            time_depth_domain=None,
            detected_roles=["井位与海拔", "井轨迹"],
            confidence=parsed.trajectory.confidence,
            evidence=list(parsed.evidence),
            status="已识别",
            accepted=True,
            decision_source="opendtect_file_contract",
        )
    if path.suffix.casefold() == ".dev":
        metadata = read_petrel_dev_header(path)
        trajectories = read_trajectory(
            path,
            {**trajectory_options, "field_aliases": field_aliases},
        )
        if not trajectories:
            return MetadataResult(evidence=["DEV文件未找到可解析的MD/轨迹数据表"])
        trajectory = trajectories[0]
        result = MetadataResult(
            trajectories=trajectories,
            detected_roles=["井位与海拔", "井轨迹"],
            confidence=0.99,
            evidence=["识别为Petrel/通用DEV轨迹表", "MD和轨迹列通过字段别名及数值表头识别"],
            status="已识别",
            accepted=True,
        )
        # Trajectory X/Y have already passed the strict per-field unit contract
        # and are canonical metres, so prefer them over raw DEV header values.
        x = None
        y = None
        if trajectory.x is not None and trajectory.x.size:
            x = float(trajectory.x[0])
        if trajectory.y is not None and trajectory.y.size:
            y = float(trajectory.y[0])
        coordinate_issues: list[str] = []
        for field in ("x", "y"):
            current_value = x if field == "x" else y
            if current_value is not None or metadata.get(field) is None:
                continue
            declared = (
                metadata.get(f"{field}_unit")
                or trajectory_options.get(f"{field}_unit")
                or trajectory_options.get("coordinate_unit")
                or trajectory_options.get("horizontal_unit")
                or trajectory_options.get("length_unit")
            )
            unit = _normalize_length_unit(declared)
            if unit is None:
                coordinate_issues.append(
                    f"horizontal_coordinate_unit_unknown:{field}"
                )
                continue
            value_m = float(length_to_metres(float(metadata[field]), unit))
            if field == "x":
                x = value_m
            else:
                y = value_m
        if x is not None and y is not None:
            result.heads = [WellHead(
                well_name=str(metadata.get("well_name") or trajectory.well_name),
                x=float(x),
                y=float(y),
                kb=None if metadata.get("kb") is None else float(metadata["kb"]),
                crs=trajectory.horizontal_crs or metadata.get("crs"),
                source=str(path),
                confidence=0.99,
                vertical_datum_unit=metadata.get("kb_unit"),
                horizontal_unit="m",
                coordinate_issues=coordinate_issues,
                source_x=(
                    None if metadata.get("x") is None else float(metadata["x"])
                ),
                source_y=(
                    None if metadata.get("y") is None else float(metadata["y"])
                ),
                source_crs=trajectory.source_crs or metadata.get("crs"),
                coordinate_transform=dict(trajectory.coordinate_transform),
            )]
        elif coordinate_issues:
            result.evidence.extend(coordinate_issues)
        return result
    header, rows, row_evidence = _rows_with_evidence(path)
    result = MetadataResult(evidence=list(row_evidence))
    if not rows:
        result.evidence.append("文件中没有可解析的数据行")
        return result

    if header:
        result.evidence.append("检测到表头：" + ",".join(header))
        try:
            columns = _map_header(
                header,
                field_aliases,
                rows=rows,
                evidence=result.evidence,
            )
        except ColumnMappingConflict as exc:
            result.status = "待确认"
            result.accepted = False
            result.evidence.append(
                "字段自动映射存在歧义，仅此文件需要确认："
                + "；".join(exc.conflicts)
            )
            return result
        if "well_name" not in columns and rows and not _is_number(rows[0][0]):
            columns["well_name"] = 0
        if {"x", "y"}.issubset(columns):
            raw_heads = read_well_heads(
                path,
                {**trajectory_options, "columns": columns},
            )
            unique_heads: dict[str, WellHead] = {}
            for head in raw_heads:
                unique_heads.setdefault(_key(head.well_name), head)
            result.heads = list(unique_heads.values())
            result.detected_roles.append("井位与海拔")
        trajectory_fields = {"tvd", "tvdss", "inclination", "azimuth", "x_offset", "y_offset"}
        if "md" in columns and trajectory_fields.intersection(columns):
            result.trajectories = read_trajectory(
                path,
                {
                    **trajectory_options,
                    "columns": columns,
                    "well_name": path.stem,
                },
            )
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

    columns: dict[str, int] = {}

    # Headerless fallback uses structure and value dimensions; filename only adds weak evidence.
    first_is_name = not _is_number(rows[0][0])
    numeric = _numeric_matrix(rows, 1 if first_is_name else 0)
    if numeric is None or numeric.shape[1] < 2:
        result.evidence.append("无表头且数值列不足，不能可靠识别")
        return result
    names = [row[0] for row in rows] if first_is_name else [path.stem] * len(rows)
    group_sizes = {name: names.count(name) for name in set(names)}
    max_group_rows = max(group_sizes.values())
    row_count = len(rows)
    medians = np.nanmedian(np.abs(numeric), axis=0)
    monotonic = _monotonic_by_well(numeric, names)
    filename_hint = bool(re.search(r"(^|[_\-])(TD|TIME.?DEPTH)([_\-.]|$)", path.stem, re.I))

    # Few rows + first two very large coordinates is a multi-well well-head table.
    if row_count <= 1000 and numeric.shape[1] >= 2 and medians[0] > 10000 and medians[1] > 10000:
        head_options = {
            **trajectory_options,
            "columns": {
                "well_name": 0 if first_is_name else None,
                "x": 1 if first_is_name else 0,
                "y": 2 if first_is_name else 1,
            },
        }
        if numeric.shape[1] >= 3:
            head_options["columns"]["kb"] = 3 if first_is_name else 2
        if numeric.shape[1] >= 4:
            head_options["columns"]["total_depth_md"] = 4 if first_is_name else 3
        result.heads = read_well_heads(path, head_options)
        result.detected_roles.append("井位与海拔")
        result.evidence.append("无表头：少量记录、井名列和两列投影坐标量级")

    # Exact five-number trajectory exports can disclose their two angle columns
    # through an independent minimum-curvature closure against supplied DX/DY.
    # Do this before the conservative MD/DX/DY fallback; unknown physical units
    # still fail inside the strict trajectory reader and remain unresolved.
    physics_inference = None
    if row_count >= 20 and max_group_rows >= 2 and monotonic[0] and not filename_hint:
        physics_inference = _infer_headerless_md_offsets_angles(
            path,
            rows,
            first_is_name=first_is_name,
            options=trajectory_options,
        )
    if physics_inference is not None:
        trajectories, physics_evidence = physics_inference
        result.trajectories = trajectories
        result.detected_roles.append("井轨迹")
        result.confidence = 0.98
        result.status = "已识别"
        result.accepted = True
        result.decision_source = "deterministic_minimum_curvature_closure"
        result.deterministic_trajectory_schema = physics_evidence
        accepted_checks = next(
            item["well_checks"]
            for item in physics_evidence["candidates"]
            if item["accepted"]
        )
        result.evidence.extend(
            [
                "无表头五数值列通过最小曲率闭合唯一识别为MD/DX/DY/AZI/INC",
                "交换AZI/INC的反事实映射未通过；未依据文件名或数值量级猜测角度列",
                "源文件只读；物理映射证据摘要"
                + str(physics_evidence["evidence_sha256"]),
                *[
                    "轨迹物理闭合："
                    f"{check['well_name']}，站点{check['station_count']}，"
                    f"最大二维残差{check['maximum_vector_error_m']:.6g}m，"
                    f"容限{check['maximum_error_tolerance_m']:.6g}m，"
                    f"相对路径误差{check['relative_path_error']:.6g}"
                    for check in accepted_checks
                ],
            ]
        )
        return result

    # Many rows + monotonic first numeric column is normally a trajectory survey.
    if row_count >= 20 and max_group_rows >= 2 and monotonic[0] and not filename_hint:
        trajectory_columns: dict[str, int] = {"md": 1 if first_is_name else 0}
        if first_is_name:
            trajectory_columns["well_name"] = 0
        if numeric.shape[1] >= 3:
            trajectory_columns["x_offset"] = 2 if first_is_name else 1
            trajectory_columns["y_offset"] = 3 if first_is_name else 2
        result.trajectories = read_trajectory(
            path,
            {
                **trajectory_options,
                "columns": trajectory_columns,
                "well_name": path.stem,
            },
        )
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
