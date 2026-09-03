from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from ..alignment import build_spatial_aligner

STAGE_DEFINITIONS = (
    ("asset_registration", "文件登记与解析", "识别文件角色、版本、来源并隔离重复文件"),
    ("log_preprocessing", "测井曲线清洗", "统一曲线名与单位，处理空值、深度顺序和异常值"),
    ("well_entity_alignment", "井实体与轨迹对齐", "合并井位、海拔、轨迹和LAS并检查空间基准"),
    ("seismic_geometry", "地震几何重建", "解析SEG-Y道头、采样信息、坐标及二维/三维几何"),
    ("vertical_datum_normalization", "垂向基准统一", "识别MSL、SRD、KB、GL等语义并统一为向上为正的MSL绝对高程"),
    ("seismic_time_reference", "地震时间基准统一", "核验TWT/OWT、SRD时间参考和静校正状态，防止重复datum校正"),
    ("spatial_alignment", "井震空间对齐", "将井轨迹定位到地震道并计算距离与置信度"),
    ("vertical_alignment", "时间域井震标定", "构建可追溯的TVD/MD至TWT关系并评估不确定度"),
    ("sample_building", "多模态样本构建", "输出地震窗口、测井特征、掩码、置信度和来源记录"),
)

CONVENTIONAL_NINE_CURVES = (
    ("SP", "SP"),
    ("GR", "GR"),
    ("CAL", "CALI"),
    ("DT", "AC"),
    ("NPHI", "CNL"),
    ("RHOB", "DEN"),
    ("MSFL", "MSFL"),
    ("RS", "LLS"),
    ("RT", "LLD"),
)

STAGE_SAFE_ACTIONS = {
    "asset_registration": (
        "核对绝对路径、文件权限和扩展名后重新读取",
        "隔离失败文件，先继续处理其余已识别资产",
        "由LLM根据文件头和字段证据补全解析方案，再由格式规则校验",
    ),
    "log_preprocessing": (
        "优先按常规测井九线知识库复核曲线名和单位映射",
        "保留原始曲线与掩码，并将该曲线标为低置信度",
        "调整空值、异常值或重复深度聚合策略后重新处理",
    ),
    "well_entity_alignment": (
        "复核井名映射，并关联正确的井位、海拔和轨迹记录",
        "缺少真实轨迹时仅保留井口水平预览，不把MD自动当作TVD",
        "将受影响井排除出高可信井震融合样本",
    ),
    "seismic_geometry": (
        "指定SEG-Y道头字节位置、字节序或厂商配置后重新解析",
        "仅保留振幅预览，不让低置信度几何进入正式空间匹配",
        "核对坐标单位、坐标标量和Inline/Crossline字段",
    ),
    "vertical_datum_normalization": (
        "核对地震处理基准面SRD及其相对MSL的绝对高程",
        "核对井深起算面KB/DF/RT；GL只作地面高程证据，不自动替代KB",
        "复核.dev、LAS头与时深文件#KB，同类基准超出容差时人工选择可信来源",
    ),
    "seismic_time_reference": (
        "确认SEG-Y时间域为TWT，并声明corrected_to_srd或uncorrected",
        "checkshot/VSP及时深表分别声明time_reference、time_domain与correction_state",
        "uncorrected数据只用替换速度执行近地表datum静校正；unknown保持阻断",
    ),
    "spatial_alignment": (
        "先修复井位或地震坐标问题，再执行井震空间对齐",
        "调整最大匹配距离并重新计算距离与置信度",
        "满足放行条件后进入井震对齐与样本构建",
    ),
    "vertical_alignment": (
        "优先使用DT/VP积分并结合RHOB生成合成地震候选标定",
        "复核井深基准、地震零时刻和井顶替代速度",
        "自动候选需人工确认或达到项目门槛后才能进入训练集",
    ),
    "sample_building": (
        "仅输出掩码有效且来源完整的样本",
        "进入样本构建模块执行独立任务",
        "暂缓构建并保留当前数据准备报告",
    ),
}


def _issue(
    *,
    stage: str,
    severity: str,
    title: str,
    message: str,
    source: str = "",
    sources: list[str] | None = None,
    action: str = "",
    blocking: bool = False,
    group_key: str = "",
    affected_entity: str = "",
    contract_candidates: list[dict[str, Any]] | None = None,
    confirmation_group: str = "",
) -> dict[str, Any]:
    all_sources = list(dict.fromkeys(str(item) for item in (sources or ([source] if source else [])) if str(item)))
    digest = hashlib.sha1(
        "\x1f".join((stage, severity, title, message, "|".join(all_sources))).encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"{stage}:{digest}",
        "stage": stage,
        "severity": severity,
        "title": title,
        "message": message,
        "source": source,
        "sources": all_sources,
        "affected_count": len(all_sources),
        "affected_entities": [affected_entity] if affected_entity else [],
        "group_key": group_key,
        "action": action,
        "blocking": blocking,
        "candidate_actions": [],
        "recommended_action": action,
        "recommendation_source": "规则",
        "recommendation_confidence": None,
        "recommendation_reason": "依据内置质量规则生成",
        "confirmation_status": "无需确认" if severity == "提示" else "待人工确认",
        "confirmed_action": "",
        "confirmed_at": "",
        "contract_candidates": list(contract_candidates or []),
        "confirmation_group": confirmation_group,
        "llm_role": "advisory_only" if contract_candidates else "bounded_action_selection",
    }


def _aggregate_issue_groups(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated per-well findings into one actionable root cause."""

    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in issues:
        group_key = str(issue.get("group_key") or "")
        if not group_key:
            output.append(issue)
            continue
        key = (str(issue.get("stage")), group_key)
        if key not in grouped:
            merged = dict(issue)
            merged["sources"] = list(issue.get("sources") or [])
            merged["affected_entities"] = list(issue.get("affected_entities") or [])
            grouped[key] = merged
            output.append(merged)
            continue
        merged = grouped[key]
        merged["sources"] = list(dict.fromkeys([*merged["sources"], *(issue.get("sources") or [])]))
        merged["affected_entities"] = list(dict.fromkeys([
            *merged["affected_entities"], *(issue.get("affected_entities") or [])
        ]))
        merged["blocking"] = bool(merged.get("blocking") or issue.get("blocking"))
    for issue in output:
        entities = list(issue.get("affected_entities") or [])
        sources = list(issue.get("sources") or [])
        issue["affected_count"] = len(entities) if entities else len(sources)
        if entities and len(entities) > 1:
            sample = "、".join(entities[:8])
            suffix = f" 等{len(entities)}口井" if len(entities) > 8 else ""
            issue["message"] = f"影响井：{sample}{suffix}。{issue['message']}"
        if issue.get("group_key"):
            digest = hashlib.sha1(
                f"{issue['stage']}\x1f{issue['group_key']}".encode()
            ).hexdigest()[:10]
            issue["id"] = f"{issue['stage']}:{digest}"
    return output


def _attach_safe_recommendations(pipeline: Any, issues: list[dict[str, Any]]) -> None:
    """Let the LLM choose only bounded actions; deterministic validators remain authoritative."""
    resolver = getattr(pipeline, "decision_resolver", None)
    max_llm_recommendations = max(
        0,
        int(pipeline.config.get("llm", {}).get("max_issue_recommendations", 6)),
    )
    llm_attempts = 0
    for issue in issues:
        if issue.get("scope_disposition") == "inventory_only":
            issue["candidate_actions"] = []
            issue["recommended_action"] = (
                "已记录资产问题；选择任务和模型后再按其输入合同判断是否需要处理"
            )
            issue["recommendation_reason"] = (
                "当前为通用资产盘点，尚未启用任何任务或模型门禁"
            )
            continue
        if issue.get("required_for_task") is False:
            issue["candidate_actions"] = []
            issue["recommended_action"] = "当前目标任务不使用该数据，不需要处理"
            issue["recommendation_reason"] = "依据所选模型的输入合同自动跳过"
            continue
        candidates = list(dict.fromkeys(
            item
            for item in (issue.get("action", ""), *STAGE_SAFE_ACTIONS.get(issue["stage"], ()))
            if item
        ))
        issue["candidate_actions"] = candidates
        if candidates and not issue.get("recommended_action"):
            issue["recommended_action"] = candidates[0]
        if issue["severity"] == "提示" or resolver is None or llm_attempts >= max_llm_recommendations:
            continue
        decision = resolver.resolve_issue_action(issue, candidates)
        if decision is None:
            continue
        llm_attempts += 1
        if decision.accepted:
            issue["recommended_action"] = decision.choice
            issue["recommendation_source"] = "LLM"
            issue["recommendation_confidence"] = round(float(decision.confidence), 4)
            issue["recommendation_reason"] = decision.reason or "从安全候选方案中选择"
        else:
            issue["recommendation_reason"] = "智能研判置信度不足，已保留规则建议"


def _finalize_issue_resolution(issues: list[dict[str, Any]]) -> None:
    """Apply auditable autofill policies and collapse unknowable items to one survey input step."""
    for issue in issues:
        issue["resolution_mode"] = "none"
        issue["autofill_patch"] = {}
        issue["autofill_validation"] = []
        if issue.get("required_for_task") is False or issue.get("severity") == "提示":
            continue

        stage = str(issue.get("stage", ""))
        group_key = str(issue.get("group_key", ""))
        if stage == "log_preprocessing" and not issue.get("blocking"):
            if "duplicate_depth" in group_key:
                patch = {"operation": "aggregate_duplicate_depth", "reducer": "median", "preserve_source": True}
            elif "curve_conflict" in group_key:
                patch = {
                    "operation": "keep_selected_curve_with_provenance",
                    "alternatives": "retained_as_source_curves",
                    "downstream_mask": "standard_curve_valid_mask",
                }
            else:
                patch = {
                    "operation": "preserve_raw_curve",
                    "unit_policy": "keep_source_unit",
                    "confidence": "low_until_convertible",
                    "downstream_mask": "exclude_from_standardized_model_input",
                }
            issue["resolution_mode"] = "llm_autofill" if issue.get("recommendation_source") == "LLM" else "rule_autofill"
            issue["autofill_patch"] = patch
            issue["autofill_validation"] = [
                "原始曲线和值未改写",
                "标准九线只读取显式有效mask",
                "来源、单位和处理步骤写入审计记录",
            ]
            issue["confirmation_status"] = "LLM已补全" if issue["resolution_mode"] == "llm_autofill" else "系统已自动处理"
            issue["confirmed_action"] = str(issue.get("recommended_action") or "")
            continue

        if (
            not issue.get("blocking")
            and issue.get("recommendation_source") == "LLM"
            and float(issue.get("recommendation_confidence") or 0.0) >= 0.8
        ):
            issue["resolution_mode"] = "llm_autofill"
            issue["autofill_patch"] = {
                "operation": "select_bounded_action",
                "action": issue.get("recommended_action"),
            }
            issue["autofill_validation"] = ["候选值属于后端allowlist", "未修改原始文件"]
            issue["confirmation_status"] = "LLM已补全"
            issue["confirmed_action"] = str(issue.get("recommended_action") or "")
            continue

        issue["resolution_mode"] = "survey_input"
        issue["confirmation_status"] = "需一次集中补充"
        issue["autofill_validation"] = ["现有文件证据不足，禁止LLM虚构物理基准或坐标参考"]


def _apply_attention_policy(issues: list[dict[str, Any]]) -> None:
    """Separate user action from retained audit evidence.

    Preparation deliberately keeps every parser, provenance and scientific
    boundary finding.  Surfacing all of those records as warnings makes the
    actionable failures indistinguishable from evidence that has already been
    handled or is irrelevant to the selected model.  The backend remains the
    single authority: only unresolved findings required by the current task
    enter the primary attention queue.
    """

    pending_statuses = {"待人工确认", "需一次集中补充"}
    for issue in issues:
        required = issue.get("required_for_task", True) is not False
        severity = str(issue.get("severity") or "提示")
        unresolved = str(issue.get("confirmation_status") or "") in pending_statuses
        needs_survey_input = issue.get("resolution_mode") == "survey_input"
        attention_required = bool(
            required
            and (
                issue.get("blocking") is True
                or (
                    unresolved
                    and needs_survey_input
                    and severity in {"错误", "警告"}
                )
            )
        )
        issue["attention_required"] = attention_required
        issue["display_bucket"] = "must_attention" if attention_required else "audit"
        issue["display_severity"] = severity if attention_required else "提示"


def _stage_status(
    stage_id: str,
    issues: list[dict[str, Any]],
    ready: bool,
    executed: bool = True,
    required: bool = True,
    inventory_only: bool = False,
) -> str:
    if not required:
        if inventory_only:
            return "就绪" if ready else "未就绪"
        return "本任务不需要"
    relevant = [item for item in issues if item["stage"] == stage_id]
    if any(item["blocking"] for item in relevant):
        return "阻断"
    if any(item.get("attention_required") for item in relevant):
        return "需确认"
    if not executed:
        return "待执行" if ready else "未就绪"
    return "就绪" if ready else "未就绪"


def _has_acoustic_time_candidate(log: Any) -> bool:
    """Return whether a log has an explicit curve family usable for a sonic tie.

    This is only an input-capability hint for the preparation report.  The
    physical well-tie implementation remains authoritative and still applies
    finite-value, unit, coverage and correlation quality gates before it emits
    a candidate.
    """

    curves = {str(name).upper() for name in getattr(log, "curves", {})}
    return bool(curves & {"DT", "VP"}) or {"PIMP", "RHOB"}.issubset(curves)


def _native_relative_well_capability_receipts(
    pipeline: Any,
    entities: list[Any],
) -> list[dict[str, Any]]:
    """Audit the same-well prerequisites used by native-time registration.

    This is a data-capability receipt, independent of whether a downstream
    task has already been selected.  It does not claim that P13 or the sonic
    physics gates passed; those remain runtime decisions.
    """

    matching = pipeline.config.get("matching", {})
    coordinate_reference = dict(matching.get("coordinate_reference") or {})
    coordinate_contract_ready = bool(
        coordinate_reference.get("verified") is True
        and str(coordinate_reference.get("crs") or "").strip()
        and str(coordinate_reference.get("horizontal_unit") or "").casefold()
        == "m"
        and str(coordinate_reference.get("axis_order") or "").upper()
        in {"XY", "YX"}
    )
    max_horizontal_distance = float(
        matching.get("max_horizontal_distance", 500.0)
    )
    try:
        sources = pipeline._selected_seismic_sources(matching)
        aligner = build_spatial_aligner(matching).fit(sources)
    except (AttributeError, TypeError, ValueError):
        aligner = None

    receipts: list[dict[str, Any]] = []
    for entity in entities:
        reasons: list[str] = []
        acoustic_ready = any(
            _has_acoustic_time_candidate(log) for log in entity.logs
        )
        if not acoustic_ready:
            reasons.append("missing_dt_vp_or_pimp_rhob")

        head = entity.preferred_head
        head_xy_ready = bool(
            head is not None
            and head.x is not None
            and head.y is not None
            and np.isfinite(float(head.x))
            and np.isfinite(float(head.y))
            and str(getattr(head, "horizontal_unit", "unknown")).casefold()
            == "m"
        )
        if not head_xy_ready:
            reasons.append("missing_canonical_metre_wellhead_xy")

        try:
            well_datum = pipeline._well_datum(entity)
        except (AttributeError, TypeError, ValueError):
            well_datum = None
        datum_kind = str(getattr(well_datum, "datum", "") or "").upper()
        datum_elevation = getattr(well_datum, "absolute_elevation_m", None)
        datum_ready = bool(
            well_datum is not None
            and getattr(well_datum, "ready", False)
            and datum_kind in {"KB", "DF", "RT"}
            and datum_elevation is not None
            and np.isfinite(float(datum_elevation))
            and not getattr(well_datum, "conflicts", ())
        )
        if not datum_ready:
            reasons.append("unresolved_kb_df_rt_msl_reference")

        trajectory = entity.preferred_trajectory
        trajectory_station_count = 0
        trajectory_ready = False
        trajectory_x: np.ndarray | None = None
        trajectory_y: np.ndarray | None = None
        if trajectory is not None:
            md = np.asarray(trajectory.md, dtype=float)
            tvd = np.asarray(trajectory.tvd, dtype=float)
            trajectory_station_count = int(md.size)
            if trajectory.x is not None and trajectory.y is not None:
                trajectory_x = np.asarray(trajectory.x, dtype=float)
                trajectory_y = np.asarray(trajectory.y, dtype=float)
            elif head_xy_ready:
                trajectory_x = float(head.x) + np.asarray(
                    trajectory.x_offset, dtype=float
                )
                trajectory_y = float(head.y) + np.asarray(
                    trajectory.y_offset, dtype=float
                )
            vertical_semantics = dict(
                getattr(trajectory, "vertical_semantics", {}) or {}
            )
            issues = {
                str(item).casefold()
                for item in (getattr(trajectory, "issues", ()) or ())
            }
            generated_head_only = any(
                "missing_deviation_survey" in item for item in issues
            ) and (trajectory.x is None or trajectory.y is None)
            trajectory_ready = bool(
                md.size >= 2
                and tvd.shape == md.shape
                and trajectory_x is not None
                and trajectory_y is not None
                and trajectory_x.shape == md.shape
                and trajectory_y.shape == md.shape
                and np.all(np.isfinite(md))
                and np.all(np.isfinite(tvd))
                and np.all(np.isfinite(trajectory_x))
                and np.all(np.isfinite(trajectory_y))
                and np.all(np.diff(md) > 0.0)
                and vertical_semantics.get("registration_eligible") is not False
                and not generated_head_only
            )
        if not trajectory_ready:
            reasons.append("incomplete_measured_md_tvd_trajectory")

        seismic_coverage_ready = False
        nearest_trace_distance_m: float | None = None
        trajectory_max_trace_distance_m: float | None = None
        if (
            coordinate_contract_ready
            and aligner is not None
            and head_xy_ready
            and trajectory_ready
            and trajectory_x is not None
            and trajectory_y is not None
        ):
            reference = aligner.match(float(head.x), float(head.y))
            if (
                reference is not None
                and float(reference.distance) <= max_horizontal_distance
            ):
                distances: list[float] = []
                for station_x, station_y in zip(trajectory_x, trajectory_y):
                    station_reference = aligner.match(
                        float(station_x),
                        float(station_y),
                        asset=reference.asset,
                    )
                    if station_reference is None:
                        distances = []
                        break
                    distances.append(float(station_reference.distance))
                if distances and max(distances) <= max_horizontal_distance:
                    seismic_coverage_ready = True
                    nearest_trace_distance_m = float(reference.distance)
                    trajectory_max_trace_distance_m = float(max(distances))
        if not coordinate_contract_ready:
            reasons.append("horizontal_coordinate_contract_unverified")
        elif not seismic_coverage_ready:
            reasons.append("well_or_trajectory_outside_seismic_coverage")

        receipts.append(
            {
                "well_uid": str(entity.well_uid),
                "well_name": str(entity.canonical_name),
                "acoustic_ready": acoustic_ready,
                "head_xy_ready": head_xy_ready,
                "datum_ready": datum_ready,
                "well_depth_datum": datum_kind or None,
                "trajectory_ready": trajectory_ready,
                "trajectory_station_count": trajectory_station_count,
                "seismic_coverage_ready": seismic_coverage_ready,
                "nearest_trace_distance_m": nearest_trace_distance_m,
                "trajectory_max_trace_distance_m": trajectory_max_trace_distance_m,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    return receipts


def _survey_contract_candidate(
    datum_inventory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a reviewable next-request patch without mutating this snapshot."""

    allowed_fields = {
        "vertical_crs_id",
        "seismic_srd_elevation_m",
        "seismic_time_domain",
        "seismic_correction_state",
        "seismic_time_reference",
    }
    candidates = [
        dict(item)
        for item in datum_inventory.get("contract_candidates", [])
        if isinstance(item, dict) and str(item.get("field")) in allowed_fields
    ]
    rank = {"verified": 3, "candidate": 2, "review_required": 1}
    request_fields = {
        "vertical_crs_id",
        "seismic_srd_elevation_m",
        "seismic_time_domain",
        "seismic_correction_state",
    }
    request_patch: dict[str, Any] = {}
    for field in request_fields:
        usable = [
            item
            for item in candidates
            if item.get("field") == field
            and str(item.get("status")) in rank
            and item.get("value") not in {None, "", "unknown"}
        ]
        if not usable:
            continue
        best_rank = max(rank[str(item["status"])] for item in usable)
        best = [item for item in usable if rank[str(item["status"])] == best_rank]
        values = {
            repr(item.get("value")): item.get("value")
            for item in best
        }
        if len(values) == 1:
            request_patch[field] = next(iter(values.values()))

    seismic = list(datum_inventory.get("seismic", []))
    unresolved: list[str] = []
    if not datum_inventory.get("vertical_crs_ready"):
        unresolved.append("vertical_crs_id")
    if seismic and any(not item.get("ready") for item in seismic):
        unresolved.append("seismic_srd_elevation_m")
    if seismic and any(
        item.get("seismic_reference", {}).get("time_domain") != "TWT"
        for item in seismic
    ):
        unresolved.append("seismic_time_domain")
    if seismic and any(
        item.get("seismic_reference", {}).get("correction_state")
        != "corrected_to_srd"
        for item in seismic
    ):
        unresolved.append("seismic_correction_state")
    confirmation_required = any(
        bool(item.get("requires_human_confirmation"))
        and item.get("field") in request_patch
        and item.get("value") == request_patch.get(str(item.get("field")))
        for item in candidates
    )
    contract = {
        "schema_version": "well-seismic.survey-contract-candidate.v1",
        "confirmation_group": "seismic_vertical_contract",
        "candidates": candidates,
        "confirmation_required": confirmation_required,
        "unresolved_fields": list(dict.fromkeys(unresolved)),
        "policy": (
            "规则明确声明可直接复用；PSTM/Final Datum组合证据仅形成待确认候选；"
            "LLM建议不写入有效合同，确认后必须重新执行数据准备"
        ),
    }
    return contract, request_patch


def build_preparation_report(pipeline: Any) -> dict[str, Any]:
    """把底层读取证据转换为可操作的预处理阶段、问题和放行条件。"""
    issues: list[dict[str, Any]] = []
    assets = pipeline.assets
    entities = list(pipeline.registry.entities.values())
    logs = [log for entity in entities for log in entity.logs]
    seismic_geometries = [
        (asset, reader.geometry)
        for asset, reader in pipeline.seismic
        if reader.geometry is not None
    ]
    datum_inventory = pipeline.vertical_datum_inventory()
    survey_contract_candidate, survey_contract_request_patch = (
        _survey_contract_candidate(datum_inventory)
    )
    contract_candidates_by_field: dict[str, list[dict[str, Any]]] = {}
    for candidate in survey_contract_candidate["candidates"]:
        contract_candidates_by_field.setdefault(
            str(candidate.get("field")), []
        ).append(candidate)
    well_datum_items = list(datum_inventory.get("wells", []))
    seismic_datum_items = list(datum_inventory.get("seismic", []))
    physical_datum_ready = bool(
        well_datum_items
        and seismic_datum_items
        and datum_inventory.get("vertical_crs_ready")
        and all(item.get("ready") for item in well_datum_items)
        and all(item.get("ready") for item in seismic_datum_items)
    )
    seismic_time_ready = bool(
        seismic_datum_items
        and datum_inventory.get("ready_seismic_time") == len(seismic_datum_items)
    )
    datum_ready = physical_datum_ready and seismic_time_ready
    target_task = dict(pipeline.manifest.get("target_task") or {})
    target_task_id = str(target_task.get("task_id") or "").strip()
    # Manifests created before scope_explicit existed are treated as scoped
    # only when they already contain a task id.  New API manifests always
    # carry the flag, so a stale/default task cannot silently activate gates.
    scope_explicit_value = target_task.get("scope_explicit")
    task_scoped = bool(target_task_id) if scope_explicit_value is None else bool(
        target_task_id and scope_explicit_value
    )
    inventory_only = not task_scoped
    required_modalities = tuple(str(item) for item in target_task.get("required_modalities") or ())
    target_model_contract = dict(target_task.get("model_contract") or {})
    model_modalities = tuple(
        str(item)
        for item in target_model_contract.get("required_modalities") or ()
    )
    modality_text = " ".join(model_modalities or required_modalities).casefold()
    registration_policy = str(
        target_model_contract.get("registration_policy") or ""
    )
    prepared_view_policy = str(
        target_model_contract.get("prepared_view_policy") or "optional"
    )
    requires_registration_inputs = (
        registration_policy == "required" or prepared_view_policy == "required"
    )
    needs_seismic = (
        requires_registration_inputs
        or any(token in modality_text for token in ("地震", "seismic", "seg-y", "registration"))
    )
    needs_logs = (
        requires_registration_inputs
        or any(token in modality_text for token in ("las", "测井", "井曲线", "well_log", "registration"))
    )
    needs_trajectory = (
        requires_registration_inputs
        or any(token in modality_text for token in ("轨迹", "align", "井震", "trajectory", "registration"))
    )
    needs_multimodal_alignment = requires_registration_inputs or (
        needs_seismic and (needs_logs or needs_trajectory)
    )
    provided_time_depth_wells = sum(1 for entity in entities if entity.time_depth)
    acoustic_candidate_wells = sum(
        1
        for entity in entities
        if any(_has_acoustic_time_candidate(log) for log in entity.logs)
    )
    native_relative_well_receipts = _native_relative_well_capability_receipts(
        pipeline,
        entities,
    )
    native_relative_candidate_wells = sum(
        bool(item.get("eligible")) for item in native_relative_well_receipts
    )
    time_depth_explicitly_required = any(
        token in modality_text
        for token in ("时深", "time-depth", "time_depth", "checkshot", "vsp")
    )
    model_forbids_time_depth_supervision = (
        target_model_contract.get("time_depth_supervision_is_model_input") is False
        and registration_policy != "required"
        and prepared_view_policy != "required"
    )
    requires_vertical_products = (
        requires_registration_inputs
        or (needs_multimodal_alignment and not model_forbids_time_depth_supervision)
    )
    # Competition data deliberately omits checkshot/VSP/time-depth tables.  A
    # well with DT/AC, a real MD->TVD trajectory and a located head can still
    # enter the existing relative-sonic + seismic-QC + P13 route.  Missing
    # absolute SRD/correction metadata remains visible, but it is not an input
    # gate for a snapshot-bound native-time mapping.
    native_relative_data_capability = bool(
        provided_time_depth_wells == 0
        and not time_depth_explicitly_required
        and native_relative_candidate_wells > 0
        and seismic_geometries
    )
    native_relative_registration_candidate = bool(
        native_relative_data_capability
        and (not task_scoped or needs_multimodal_alignment)
    )
    requires_absolute_vertical_products = bool(
        requires_vertical_products and not native_relative_registration_candidate
    )
    required_by_stage = {
        "asset_registration": task_scoped,
        "log_preprocessing": needs_logs,
        "well_entity_alignment": needs_logs or needs_trajectory,
        "seismic_geometry": needs_seismic,
        "vertical_datum_normalization": requires_absolute_vertical_products,
        "seismic_time_reference": requires_absolute_vertical_products,
        "spatial_alignment": needs_multimodal_alignment,
        "vertical_alignment": requires_vertical_products,
        "sample_building": requires_vertical_products,
    }

    for error in pipeline.errors:
        issues.append(_issue(
            stage="asset_registration",
            severity="错误",
            title="文件读取失败",
            message=error.get("error", "未知读取错误"),
            source=error.get("path", ""),
            action="由LLM读取文件头与字段证据生成解析补丁，并在隔离副本上通过格式校验",
            blocking=True,
        ))
    parse_repairs = list(getattr(pipeline, "llm_parse_repairs", []))
    for repair in parse_repairs:
        if repair.get("status") != "applied_to_current_snapshot":
            continue
        patch_sha256 = str(repair.get("patch_sha256") or "")
        issues.append(_issue(
            stage="asset_registration",
            severity="提示",
            title="LLM结构化解析补丁已验证",
            message=(
                "原始文件保持只读；分隔符、列映射或单位补丁已在内存重解析，"
                "并通过格式、轨迹物理和独立单位佐证门。"
            ),
            source=str(repair.get("source_path") or ""),
            action="采用已封存的结构化解析补丁处理当前数据快照",
            group_key=f"llm_parse_repair:{patch_sha256}",
        ))
    if not assets:
        issues.append(_issue(
            stage="asset_registration",
            severity="错误",
            title="没有发现可处理资产",
            message="登记路径存在，但没有匹配到支持的文件。",
            action="检查路径、递归选项和文件扩展名",
            blocking=True,
        ))

    grouped_log_issues: dict[str, dict[str, list[str]]] = {}
    for log in logs:
        for raw in log.issues:
            if ":unit_conversion_unavailable:" in raw:
                _, conversion = raw.split(":unit_conversion_unavailable:", 1)
                root = f"unit_conversion_unavailable:{conversion}"
            elif raw.startswith("curve_conflict:"):
                parts = raw.split(":", 2)
                root = ":".join(parts[:2])
            elif "duplicate_depth" in raw:
                root = "duplicate_depth"
            else:
                root = raw
            group = grouped_log_issues.setdefault(root, {"sources": [], "examples": []})
            group["sources"].append(log.source)
            group["examples"].append(raw)
    for root, group in grouped_log_issues.items():
        sources = list(dict.fromkeys(group["sources"]))
        examples = list(dict.fromkeys(group["examples"]))
        duplicate = "duplicate_depth" in root
        example_text = "；".join(examples[:4])
        if len(examples) > 4:
            example_text += f"；另有{len(examples) - 4}种曲线命名"
        issues.append(_issue(
            stage="log_preprocessing",
            severity="警告",
            title="测井曲线需要复核",
            message=example_text,
            source=sources[0] if sources else "",
            sources=sources,
            action="由LLM补全曲线语义；无法安全换算时保留原值、单位与mask" if not duplicate else "自动按中位数聚合重复深度并保留来源记录",
            group_key=f"log:{root}",
        ))
    if not logs:
        issues.append(_issue(
            stage="log_preprocessing",
            severity="提示",
            title="当前没有LAS测井",
            message="仍可进行地震单模态可视化和基线模型实验，但不能构建井震多模态样本。",
            action="如需井震融合，请登记LAS路径",
        ))

    uncertain_metadata = []
    for item in pipeline.metadata_detection:
        roles = item.get("识别角色", [])
        concerns_core_metadata = any(
            "井位" in str(role) or "井轨迹" in str(role)
            for role in roles
        )
        if item.get("状态") == "待确认" and concerns_core_metadata:
            uncertain_metadata.append(item)
    for item in uncertain_metadata:
        issues.append(_issue(
            stage="well_entity_alignment",
            severity="警告",
            title="井相关文件角色不确定",
            message="；".join(str(value) for value in item.get("证据", [])) or "缺少可靠字段证据",
            source=str(item.get("文件", "")),
            action="人工确认字段含义和井名归属后再参与正式匹配",
        ))
    uncertain_time_depth_domain = [
        item for item in pipeline.metadata_detection
        if any("时深" in str(role) for role in item.get("识别角色", []))
        and item.get("时深深度域") == "未明确"
    ]
    if uncertain_time_depth_domain:
        issues.append(_issue(
            stage="vertical_alignment",
            severity="警告",
            title="时深表缺少明确深度域",
            message=(
                f"检测到 {len(uncertain_time_depth_domain)} 个时深关系候选未说明MD、TVD或TVDSS，"
                "系统已禁止其进入垂向标定，避免把错误深度域套到地震时间。"
            ),
            sources=[str(item.get("文件", "")) for item in uncertain_time_depth_domain],
            action="在表头或输入配置中明确depth_domain，并核对深度与时间单位",
        ))
    if (
        task_scoped
        and required_by_stage["vertical_alignment"]
        and provided_time_depth_wells == 0
        and not time_depth_explicitly_required
    ):
        if acoustic_candidate_wells:
            message = (
                "所选任务的输入合同不要求时深表、checkshot或VSP。"
                f"检测到 {acoustic_candidate_wells} 口井具备DT/VP或阻抗速度候选；"
                "系统可生成带不确定性的声波井震标定候选，但在独立质量门通过前"
                "不会将其标为训练或高置信融合样本。"
            )
            action = "继续候选标定；保留来源、不确定性和质量门结果"
        else:
            message = (
                "所选任务的输入合同不要求时深表、checkshot或VSP。当前也没有"
                "可审计的声波速度候选，因此通用物理预处理只保留井位/轨迹的"
                "水平空间定位；具备无时深推理合同的模型仍可在独立适配器内生成"
                "候选TWT。系统不会伪造TWT，依赖正式时间域融合或训练的后续步骤"
                "必须等待候选通过独立质量门。"
            )
            action = "继续井位与轨迹空间核验；如需时间域融合再提供可靠垂向证据"
        issues.append(_issue(
            stage="vertical_alignment",
            severity="提示",
            title="未提供实测时深控制，不阻断登记与候选推理",
            message=message,
            action=action,
            group_key="optional_time_depth_control_absent",
        ))
    missing_trajectory_entities = []
    for entity in entities:
        if entity.logs and entity.preferred_head is None:
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="错误",
                title="部分LAS井缺少井位",
                message="LAS已关联，但没有可用井口坐标，无法进行井震空间匹配。",
                action="补充井位文件或在井名映射中关联正确记录",
                # Competition readiness is per well.  Keep this well masked
                # when another well already satisfies the native-relative
                # contract; do not discard the whole survey.
                blocking=not native_relative_registration_candidate,
                group_key="missing_well_head",
                affected_entity=entity.canonical_name,
            ))
        elif entity.logs and entity.preferred_head and (
            entity.preferred_head.x is None or entity.preferred_head.y is None
        ):
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="错误",
                title="部分井位坐标不完整",
                message="井口记录缺少X或Y坐标。",
                source=entity.preferred_head.source,
                action="检查坐标字段映射",
                blocking=not native_relative_registration_candidate,
                group_key="incomplete_well_head_coordinates",
                affected_entity=entity.canonical_name,
            ))
        if entity.logs and entity.preferred_trajectory is None:
            missing_trajectory_entities.append(entity)
        elif entity.preferred_trajectory is not None:
            for message in entity.preferred_trajectory.issues:
                issues.append(_issue(
                    stage="well_entity_alignment",
                    severity="警告",
                    title="部分轨迹采用降级或重建",
                    message=message,
                    source=entity.preferred_trajectory.source,
                    action="复核井斜、方位和坐标字段；真实轨迹缺失时保留低置信度标记",
                    group_key=f"trajectory_degraded:{message}",
                    affected_entity=entity.canonical_name,
                ))
        for conflict in entity.conflicts:
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="警告",
                title="部分井位记录冲突",
                message=conflict,
                action="人工选择可信井位来源",
                group_key="well_head_conflicts",
                affected_entity=entity.canonical_name,
            ))
    if missing_trajectory_entities:
        names = "、".join(entity.canonical_name for entity in missing_trajectory_entities[:8])
        if len(missing_trajectory_entities) > 8:
            names += f" 等{len(missing_trajectory_entities)}口井"
        issues.append(_issue(
            stage="well_entity_alignment",
            severity="警告",
            title="部分LAS井缺少真实井轨迹",
            message=f"{names} 当前仅保留井口水平位置；MD不能自动当作TVD，因此不会生成z_msl_m或时间域井震窗口。",
            action="补充真实井轨迹，或提供明确的md_offset_to_trajectory_m/人工深度tie",
        ))

    geometry_with_coordinates = 0
    geometry_inference_records = 0
    low_confidence_geometry = 0
    geometry_confidence_threshold = float(
        pipeline.config.get("segy", {}).get("minimum_geometry_confidence", 0.75)
    )
    for asset, geometry in seismic_geometries:
        if geometry.x is not None and geometry.y is not None:
            geometry_with_coordinates += 1
        else:
            issues.append(_issue(
                stage="seismic_geometry",
                severity="错误",
                title="SEG-Y缺少可用平面坐标",
                message="已读取振幅和采样信息，但无法与井位进行空间匹配。",
                source=str(asset.path),
                action="由LLM依据二进制头、道头统计和空间连续性补全坐标字节候选，规则往返验证后采用",
                blocking=True,
            ))
        geometry_inference_records += len(geometry.issues)
        if geometry.confidence < geometry_confidence_threshold:
            low_confidence_geometry += 1
            issues.append(_issue(
                stage="seismic_geometry",
                severity="警告",
                title="地震几何置信度偏低",
                message=(
                    f"综合置信度 {geometry.confidence:.3f}，低于"
                    f" {geometry_confidence_threshold:.2f}；自动字节序和道头候选仍保留在来源记录中。"
                ),
                source=str(asset.path),
                action="由LLM生成受限SEG-Y字段映射，并通过道头范围与网格连续性校验",
            ))
    if not seismic_geometries:
        issues.append(_issue(
            stage="seismic_geometry",
            severity="提示",
            title="当前没有可用地震数据",
            message="仍可清洗LAS和整理井数据，但不能可视化地震或构建井震样本。",
            action="如需地震任务，请登记SEG-Y路径",
        ))

    if not datum_inventory.get("vertical_crs_ready"):
        issues.append(_issue(
            stage="vertical_datum_normalization",
            severity=(
                "提示" if native_relative_registration_candidate else "错误"
            ),
            title=(
                "绝对垂向基准尚未闭合"
                if native_relative_registration_candidate
                else "本地MSL尚未绑定测区垂向CRS"
            ),
            message=(
                "平台将以封存快照身份隔离当前测区；这不阻断本快照内的DT/AC精细标定，"
                "但成果不能宣称已具备跨工区绝对垂向基准。"
                if native_relative_registration_candidate
                else "当前为LOCAL_MSL_UNSPECIFIED，禁止默认认为不同工区的MSL完全相同。"
            ),
            action=(
                "继续本快照内的相对精细标定；仅在绝对地图/跨工区使用前补充正式基准"
                if native_relative_registration_candidate
                else "填写类似LOCAL_MSL_CHENGDU的测区唯一垂向CRS标识；有正式垂向CRS时填写正式ID"
            ),
            blocking=not native_relative_registration_candidate,
            contract_candidates=contract_candidates_by_field.get(
                "vertical_crs_id", []
            ),
            confirmation_group="seismic_vertical_contract",
        ))

    for item in well_datum_items:
        if item.get("conflicts"):
            issues.append(_issue(
                stage="vertical_datum_normalization",
                severity="错误",
                title="部分井垂向基准冲突",
                message="；".join(str(value) for value in item.get("conflicts", [])),
                sources=[str(obs.get("source", "")) for obs in item.get("observations", [])],
                action="核对.dev、井位表、LAS头和时深文件#KB，人工确认可信绝对高程",
                blocking=True,
                group_key="well_vertical_datum_conflict",
                affected_entity=str(item.get("entity_name", "井")),
            ))
        elif not item.get("ready"):
            observations = item.get("observations", [])
            issues.append(_issue(
                stage="vertical_datum_normalization",
                severity="错误",
                title="部分井缺少已确认的井深基准",
                message=(
                    "未能确定KB/DF/RT相对MSL的绝对高程。"
                    "GL是独立地面高程，不能在未说明测井深度起算面时自动替代KB。"
                ),
                sources=[str(obs.get("source", "")) for obs in observations],
                action="在井位、.dev、LAS或时深文件中明确KB/DF/RT及m/ft单位和MSL参考",
                blocking=True,
                group_key="well_vertical_datum_missing",
                affected_entity=str(item.get("entity_name", "井")),
            ))

    for item in seismic_datum_items:
        if item.get("conflicts"):
            issues.append(_issue(
                stage="vertical_datum_normalization",
                severity="错误",
                title="地震处理基准面高程冲突",
                message="；".join(str(value) for value in item.get("conflicts", [])),
                sources=[str(obs.get("source", "")) for obs in item.get("observations", [])],
                action="以处理报告或SEG-Y文本头为准，确认唯一SRD绝对高程",
                blocking=True,
                contract_candidates=contract_candidates_by_field.get(
                    "seismic_srd_elevation_m", []
                ),
                confirmation_group="seismic_vertical_contract",
            ))
        elif not item.get("ready"):
            issues.append(_issue(
                stage="vertical_datum_normalization",
                severity=(
                    "提示" if native_relative_registration_candidate else "错误"
                ),
                title=(
                    "SRD未知，保留原生地震时间参考"
                    if native_relative_registration_candidate
                    else "地震处理基准面SRD尚未确认"
                ),
                message=(
                    "不会默认SRD=0，也不会重复施加datum校正；DT/AC积分与冻结概率标定仍可在"
                    "当前SEG-Y原生时间轴上精细标定，absolute_reference_ready保持false。"
                    if native_relative_registration_candidate
                    else "缺少SRD相对MSL的绝对高程，系统已禁止进入时间域井震对齐。"
                ),
                source=str(item.get("entity_name", "")),
                action=(
                    "继续原生时间轴标定；仅在需要绝对基准成果时补充处理报告"
                    if native_relative_registration_candidate
                    else "在SEG-Y文本头或资产配置中提供SRD/processing datum的m MSL绝对高程"
                ),
                blocking=not native_relative_registration_candidate,
                contract_candidates=contract_candidates_by_field.get(
                    "seismic_srd_elevation_m", []
                ),
                confirmation_group="seismic_vertical_contract",
            ))

        time_contract = item.get("seismic_reference", {})
        if not time_contract.get("ready"):
            issues.append(_issue(
                stage="seismic_time_reference",
                severity=(
                    "提示" if native_relative_registration_candidate else "错误"
                ),
                title=(
                    "地震绝对时间参考未闭合"
                    if native_relative_registration_candidate
                    else "地震时间参考未统一到SRD"
                ),
                message=(
                    f"time_domain={time_contract.get('time_domain', 'unknown')}，"
                    f"time_reference={time_contract.get('time_reference', 'unknown')}，"
                    f"correction_state={time_contract.get('correction_state', 'unknown')}。"
                    + (
                        "系统保持原生时间样点并运行DT/AC与冻结概率标定质量门；未知状态不会触发第二次静校正，"
                        "但绝对基准成果保持未就绪。"
                        if native_relative_registration_candidate
                        else "当前实现只接受明确声明为TWT且corrected_to_srd的SEG-Y进入井震标定。"
                    )
                ),
                source=str(item.get("entity_name", "")),
                action=(
                    "继续原生时间轴精细标定，并保留时间域假设与质量门回执"
                    if native_relative_registration_candidate
                    else "核对SEG-Y文本头或处理报告；若尚未校正，先完成逐道静校正后再登记为corrected_to_srd"
                ),
                blocking=not native_relative_registration_candidate,
                contract_candidates=[
                    *contract_candidates_by_field.get("seismic_time_domain", []),
                    *contract_candidates_by_field.get("seismic_time_reference", []),
                    *contract_candidates_by_field.get(
                        "seismic_correction_state", []
                    ),
                ],
                confirmation_group="seismic_vertical_contract",
            ))

    for entity in entities:
        for table in entity.time_depth:
            reasons = []
            if table.depth_unit == "unknown" or table.time_unit == "unknown":
                reasons.append("深度或时间单位未确认")
            if table.time_domain not in {"TWT", "OWT"}:
                reasons.append("time_domain未声明TWT/OWT")
            if table.correction_state == "unknown":
                reasons.append("correction_state未知")
            if table.correction_state == "corrected_to_srd" and table.time_reference != "SRD":
                reasons.append("已校正表未声明time_reference=SRD")
            if table.correction_state == "uncorrected" and (
                table.time_reference not in {"KB", "GL", "DF", "RT"}
                or table.replacement_velocity_mps is None
            ):
                reasons.append("未校正表缺少参考面或替换速度")
            if table.depth_domain in {"md", "tvd"} and not table.depth_datum:
                reasons.append("深度起算面未声明")
            if table.depth_domain == "tvdss" and not table.depth_convention:
                reasons.append("TVDSS符号约定未声明")
            if reasons:
                issues.append(_issue(
                    stage="seismic_time_reference",
                    severity="错误",
                    title=f"{entity.canonical_name}时深/checkshot时间合同不完整",
                    message="；".join(reasons),
                    source=table.source,
                    action="补充depth_datum、time_reference、time_domain和correction_state；unknown不得自动处理",
                    blocking=True,
                ))

    matchable_wells = sum(
        1
        for entity in entities
        if entity.logs
        and entity.preferred_head is not None
        and entity.preferred_head.x is not None
        and entity.preferred_head.y is not None
    )
    spatially_locatable_wells = sum(
        1
        for entity in entities
        if entity.preferred_head is not None
        and entity.preferred_head.x is not None
        and entity.preferred_head.y is not None
        and (entity.logs or entity.preferred_trajectory is not None)
    )
    can_align = matchable_wells > 0 and geometry_with_coordinates > 0
    can_preview_spatial_alignment = (
        spatially_locatable_wells > 0 and geometry_with_coordinates > 0
    )
    if not can_preview_spatial_alignment:
        issues.append(_issue(
            stage="spatial_alignment",
            severity="提示",
            title="井震空间对齐尚未就绪",
            message=(
                f"可定位井 {spatially_locatable_wells} 口，可定位地震资产 "
                f"{geometry_with_coordinates} 个。"
            ),
            action="先解决井位和地震坐标问题",
        ))
    elif not pipeline.samples:
        if matchable_wells:
            spatial_message = "井和地震几何已经具备匹配条件。"
            spatial_action = "进入“样本构建”启动空间对齐任务"
        else:
            spatial_message = (
                "井位、真实轨迹和地震几何已具备水平定位条件；当前没有LAS，"
                "因此只开放轨迹叠加和空间质检，不生成按测井深度采样的井震样本。"
            )
            spatial_action = "执行轨迹空间叠加与距离质检"
        issues.append(_issue(
            stage="spatial_alignment",
            severity="提示",
            title="空间对齐等待执行",
            message=spatial_message,
            action=spatial_action,
        ))

    coordinate_reference = pipeline.config.get("matching", {}).get("coordinate_reference", {})
    coordinate_reference_verified = bool(coordinate_reference.get("verified", False))
    native_relative_input_ready = bool(
        native_relative_registration_candidate
        and coordinate_reference_verified
        and geometry_with_coordinates > 0
        and native_relative_candidate_wells > 0
    )
    if can_preview_spatial_alignment and not coordinate_reference_verified:
        issues.append(_issue(
            stage="spatial_alignment",
            severity="警告",
            title="井震坐标参考尚未核验",
            message="当前可生成空间匹配候选，但无法证明井位与SEG-Y XY处于同一CRS和长度单位，因此不会放行多模态训练。",
            action="核对井位与SEG-Y的CRS、坐标单位和坐标缩放后，将matching.coordinate_reference.verified设为true",
            blocking=needs_multimodal_alignment,
        ))

    if not pipeline.samples:
        issues.append(_issue(
            stage="sample_building",
            severity="提示",
            title="多模态样本尚未生成",
            message="数据准备只验证是否可构建，正式样本在独立任务中生成。",
            action="进入“样本构建”执行",
        ))
    elif not any(sample.get("seismic_window_valid") for sample in pipeline.samples):
        issues.append(_issue(
            stage="vertical_alignment",
            severity="警告",
            title="尚无有效时间域井震窗口",
            message="水平位置已匹配，但缺少可用时深关系或声波积分结果未覆盖地震时间轴。",
            action="核对DT/VP、井深基准、替代速度和地震时间零点",
        ))
    elif coordinate_reference_verified and not any(sample.get("training_eligible") for sample in pipeline.samples):
        issues.append(_issue(
            stage="vertical_alignment",
            severity="提示",
            title="井震标定仍为候选状态",
            message="已生成时间域窗口，但自动估计结果默认不直接进入多模态训练。",
            action="在井震标定工作台复核后批准，或配置经盲测验证的放行阈值",
        ))

    issues = _aggregate_issue_groups(issues)
    for issue in issues:
        required_for_task = (
            False
            if inventory_only
            else required_by_stage.get(str(issue.get("stage")), True)
        )
        issue["required_for_task"] = required_for_task
        issue["original_blocking"] = bool(issue.get("blocking"))
        if not required_for_task:
            issue["blocking"] = False
            if inventory_only:
                issue["scope_disposition"] = "inventory_only"
                issue["confirmation_status"] = "无需确认"
            else:
                issue["confirmation_status"] = "本任务不需要"

    _attach_safe_recommendations(pipeline, issues)
    _finalize_issue_resolution(issues)
    repairs_by_group = {
        f"llm_parse_repair:{repair.get('patch_sha256')}": repair
        for repair in parse_repairs
        if repair.get("status") == "applied_to_current_snapshot"
    }
    for issue in issues:
        repair = repairs_by_group.get(str(issue.get("group_key") or ""))
        if repair is None:
            continue
        issue["resolution_mode"] = "llm_autofill"
        issue["recommendation_source"] = "LLM"
        issue["recommendation_confidence"] = repair.get("confidence")
        issue["confirmation_status"] = "LLM已补全并复检"
        issue["confirmed_action"] = issue.get("recommended_action")
        issue["autofill_patch"] = {
            "operation": "apply_structured_parse_metadata_to_snapshot",
            "options_patch": repair.get("options_patch") or {},
            "patch_sha256": repair.get("patch_sha256"),
        }
        issue["autofill_validation"] = [
            "原始文件SHA-256前后一致",
            "隔离内存重解析通过",
            "轨迹格式与物理门通过",
            "单位具有manifest逐字段声明或同井LAS覆盖之一的独立佐证",
            *(str(item) for item in repair.get("corroboration") or []),
        ]
        issue["repair_provenance"] = {
            key: repair.get(key)
            for key in (
                "contract_version", "source_sha256", "patch_sha256",
                "provider", "model", "request_id",
            )
        }
    _apply_attention_policy(issues)
    llm_records = list(getattr(pipeline.decision_resolver, "records", []))
    llm_accepted = sum(1 for item in llm_records if item.get("是否采纳"))

    available_curve_ids = {name for log in logs for name in log.curves}
    available_nine = [
        display for canonical, display in CONVENTIONAL_NINE_CURVES
        if canonical in available_curve_ids
    ]
    missing_nine = [
        display for canonical, display in CONVENTIONAL_NINE_CURVES
        if canonical not in available_curve_ids
    ]
    trajectory_wells = sum(
        1
        for entity in entities
        if entity.preferred_trajectory is not None
    )
    tie_statuses = Counter(str(item.get("status", "horizontal_only")) for item in getattr(pipeline, "well_ties", []))
    valid_window_samples = sum(1 for sample in pipeline.samples if sample.get("seismic_window_valid"))
    training_samples = sum(1 for sample in pipeline.samples if sample.get("training_eligible"))
    stage_metrics = {
        "asset_registration": {
            "数据资产": len(assets),
            "跳过重复": len(pipeline.duplicates),
            "读取错误": len(pipeline.errors),
            "LLM判断": len(llm_records),
            "LLM已采纳": llm_accepted,
            "LLM解析补丁候选": len(parse_repairs),
            "LLM解析补丁已应用": sum(
                1 for item in parse_repairs
                if item.get("status") == "applied_to_current_snapshot"
            ),
        },
        "log_preprocessing": {
            "LAS记录": len(logs),
            "标准曲线类型": len({name for log in logs for name in log.curves}),
            "曲线样点": sum(len(log.depth) for log in logs),
            "常规九线覆盖": f"{len(available_nine)}/9",
            "已识别九线": "、".join(available_nine) or "无",
            "当前未提供": "、".join(missing_nine) or "无",
            "自动处理记录": sum(len(log.processing_steps) for log in logs),
        },
        "well_entity_alignment": {
            "井实体": len(entities),
            "可定位井": matchable_wells,
            "LAS井": sum(1 for entity in entities if entity.logs),
            "真实轨迹井": trajectory_wells,
            "缺少MD到TVD映射井": len(missing_trajectory_entities),
        },
        "seismic_geometry": {
            "地震资产": len(seismic_geometries),
            "有坐标几何": geometry_with_coordinates,
            "自动推断记录": geometry_inference_records,
            "低置信度几何": low_confidence_geometry,
        },
        "vertical_datum_normalization": {
            "垂向CRS": str(datum_inventory.get("vertical_crs", {}).get("id", "LOCAL_MSL_UNSPECIFIED")),
            "物理字段": "z_msl_m / depth_below_msl_m",
            "已确认井基准": f"{datum_inventory.get('ready_wells', 0)}/{len(well_datum_items)}",
            "已确认地震SRD": f"{datum_inventory.get('ready_seismic', 0)}/{len(seismic_datum_items)}",
            "基准冲突": int(datum_inventory.get("conflicts", 0)),
            "正方向": "z_msl_m向上为正",
        },
        "seismic_time_reference": {
            "目标时间参考": (
                "原生SEG-Y时间轴（绝对成果另需SRD）"
                if native_relative_registration_candidate
                else "SRD"
            ),
            "目标时间域": "TWT",
            "已确认地震时间": f"{datum_inventory.get('ready_seismic_time', 0)}/{len(seismic_datum_items)}",
            "防重复校正": "corrected_to_srd不再校正",
            "unknown处理": (
                "不重复校正；仅阻断绝对基准声明"
                if native_relative_registration_candidate
                else "阻断"
            ),
        },
        "spatial_alignment": {
            "可匹配井": matchable_wells,
            "可水平定位井": spatially_locatable_wells,
            "匹配方法": str(pipeline.config.get("matching", {}).get("method", "nearest_trace")),
            "邻域道数": int(pipeline.config.get("matching", {}).get("neighbor_traces", 1)),
            "坐标参考已核验": "是" if coordinate_reference_verified else "否",
            "已匹配样本": len(pipeline.samples),
        },
        "vertical_alignment": {
            "井-测井标定记录": len(getattr(pipeline, "well_ties", [])),
            "深度域未明确的时深候选": len(uncertain_time_depth_domain),
            "实测时深关系": tie_statuses.get("provided_tie", 0),
            "含实测时深控制井": provided_time_depth_wells,
            "具备声波候选输入井": acoustic_candidate_wells,
            "具备无TD精细标定合同井": native_relative_candidate_wells,
            "声波积分候选": tie_statuses.get("estimated_tie", 0),
            "仅初始关系": tie_statuses.get("vertical_initial", 0),
            "仅水平定位": tie_statuses.get("horizontal_only", 0),
            "基准未统一阻断": tie_statuses.get("datum_unresolved", 0),
            "时间参考未统一阻断": tie_statuses.get("time_reference_unresolved", 0),
            "有效时间窗样本": valid_window_samples,
            "可训练样本": training_samples,
        },
        "sample_building": {
            "样本数量": len(pipeline.samples),
            "带时间窗样本": valid_window_samples,
            "可训练样本": training_samples,
        },
    }

    ready_by_stage = {
        "asset_registration": bool(assets) and not pipeline.errors,
        "log_preprocessing": bool(logs),
        "well_entity_alignment": bool(entities),
        "seismic_geometry": bool(seismic_geometries),
        "vertical_datum_normalization": physical_datum_ready,
        "seismic_time_reference": seismic_time_ready,
        "spatial_alignment": can_preview_spatial_alignment,
        "vertical_alignment": bool(valid_window_samples),
        "sample_building": bool(pipeline.samples),
    }
    stages = []
    for stage_id, name, description in STAGE_DEFINITIONS:
        stages.append({
            "id": stage_id,
            "name": name,
            "description": description,
            "status": _stage_status(
                stage_id,
                issues,
                ready_by_stage[stage_id],
                executed=stage_id not in {"spatial_alignment", "vertical_alignment", "sample_building"} or bool(pipeline.samples),
                required=required_by_stage[stage_id],
                inventory_only=inventory_only,
            ),
            "metrics": stage_metrics[stage_id],
            "issue_count": sum(
                1
                for issue in issues
                if issue["stage"] == stage_id
                and issue.get("attention_required") is True
            ),
        })

    optional_issues = [issue for issue in issues if not issue.get("required_for_task", True)]
    attention_issues = [issue for issue in issues if issue.get("attention_required") is True]
    attention_severity_counts = Counter(issue["severity"] for issue in attention_issues)
    can_visualize = bool(seismic_geometries) or bool(entities)
    can_build_samples = can_align and (datum_ready or native_relative_input_ready)
    multimodal_input_ready = (
        native_relative_input_ready
        if native_relative_registration_candidate
        else (
            can_build_samples
            if requires_vertical_products
            else can_preview_spatial_alignment and coordinate_reference_verified
        )
    )
    selected_task_ready = inventory_only or (
        (not needs_seismic or bool(seismic_geometries))
        and (not needs_logs or bool(logs))
        and (not needs_trajectory or trajectory_wells > 0)
        and (not needs_multimodal_alignment or multimodal_input_ready)
        and not any(issue["blocking"] for issue in issues)
    )
    return {
        "stages": stages,
        "issues": issues,
        "survey_contract_candidate": survey_contract_candidate,
        "request_patch": survey_contract_request_patch,
        "summary": {
            "blocking": sum(1 for item in issues if item["blocking"]),
            "warnings": attention_severity_counts["警告"],
            "information": sum(
                1 for item in issues if item.get("attention_required") is not True
            ),
            "attention_required": len(attention_issues),
            "audit_findings": len(issues) - len(attention_issues),
            "not_required": len(optional_issues),
            "autofilled": sum(
                1 for item in issues if item.get("resolution_mode") in {"llm_autofill", "rule_autofill"}
            ),
            "survey_input_required": sum(
                1 for item in issues if item.get("resolution_mode") == "survey_input"
            ),
        },
        "gates": {
            "can_visualize": can_visualize,
            "can_preview_spatial_alignment": can_preview_spatial_alignment,
            "can_build_samples": can_build_samples,
            "can_train_seismic_baseline": bool(seismic_geometries),
            "can_train_multimodal": bool(training_samples),
            "can_run_high_confidence_fusion": bool(training_samples) and trajectory_wells > 0,
            "can_enter_task_selection": inventory_only,
            "can_run_prediction": selected_task_ready if task_scoped else False,
            "can_run_selected_task": selected_task_ready,
        },
        "task_readiness": {
            "scope_mode": "task_model" if task_scoped else "inventory_only",
            "task_id": target_task_id if task_scoped else None,
            "model_id": target_task.get("model_id") if task_scoped else None,
            "required_modalities": list(required_modalities) if task_scoped else [],
            "model_contract": target_model_contract if task_scoped else {},
            "required_stages": [stage for stage, required in required_by_stage.items() if required],
            "not_required_stages": [stage for stage, required in required_by_stage.items() if not required],
            "ready": selected_task_ready,
            "time_depth_policy": {
                "provided_control_required": time_depth_explicitly_required,
                "provided_control_well_count": provided_time_depth_wells,
                "acoustic_candidate_well_count": acoustic_candidate_wells,
                "missing_provided_control_blocks_current_task": time_depth_explicitly_required,
                "model_forbids_time_depth_supervision": model_forbids_time_depth_supervision,
                "horizontal_only_never_implies_twt": True,
                # Backward-compatible field: downstream sample training still
                # requires an accepted alignment, but that alignment need not
                # be authoritative registration supervision.
                "training_requires_accepted_vertical_alignment": True,
            },
            "registration_entry_policy": {
                "mode": (
                    "native_relative_no_time_depth"
                    if native_relative_registration_candidate
                    else "absolute_reference"
                ),
                "native_relative_data_capability": (
                    native_relative_data_capability
                ),
                "native_relative_candidate_well_count": (
                    native_relative_candidate_wells
                ),
                "native_relative_well_receipts": (
                    native_relative_well_receipts
                ),
                "native_relative_registration_ready": native_relative_input_ready,
                "absolute_reference_required_for_entry": (
                    requires_absolute_vertical_products
                ),
                "aligned_features_require_quality_gate": True,
                "registration_supervision_requires_authoritative_control": True,
                "no_time_depth_registration_supervision_eligible": False,
            },
        },
    }
