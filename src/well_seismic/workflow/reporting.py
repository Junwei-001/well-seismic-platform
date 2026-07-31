from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


STAGE_DEFINITIONS = (
    ("asset_registration", "文件登记与解析", "识别文件角色、版本、来源并隔离重复文件"),
    ("log_preprocessing", "测井曲线清洗", "统一曲线名与单位，处理空值、深度顺序和异常值"),
    ("well_entity_alignment", "井实体与轨迹对齐", "合并井位、海拔、轨迹和LAS并检查空间基准"),
    ("seismic_geometry", "地震几何重建", "解析SEG-Y道头、采样信息、坐标及二维/三维几何"),
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
        "在配置中心选择对应格式或厂商解析配置",
    ),
    "log_preprocessing": (
        "优先按常规测井九线知识库复核曲线名和单位映射",
        "保留原始曲线与掩码，并将该曲线标为低置信度",
        "调整空值、异常值或重复深度聚合策略后重新处理",
    ),
    "well_entity_alignment": (
        "复核井名映射，并关联正确的井位、海拔和轨迹记录",
        "缺少真实轨迹时仅保留直井低置信度预览",
        "将受影响井排除出高可信井震融合样本",
    ),
    "seismic_geometry": (
        "指定SEG-Y道头字节位置、字节序或厂商配置后重新解析",
        "仅保留振幅预览，不让低置信度几何进入正式空间匹配",
        "核对坐标单位、坐标标量和Inline/Crossline字段",
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
    }


def _attach_safe_recommendations(pipeline: Any, issues: list[dict[str, Any]]) -> None:
    """让LLM只在安全候选项中排序；任何建议都必须由用户确认。"""
    resolver = getattr(pipeline, "decision_resolver", None)
    max_llm_recommendations = max(
        0,
        int(pipeline.config.get("llm", {}).get("max_issue_recommendations", 6)),
    )
    llm_attempts = 0
    for issue in issues:
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


def _stage_status(stage_id: str, issues: list[dict[str, Any]], ready: bool, executed: bool = True) -> str:
    relevant = [item for item in issues if item["stage"] == stage_id]
    if any(item["blocking"] for item in relevant):
        return "阻断"
    if any(item["severity"] == "警告" for item in relevant):
        return "需确认"
    if not executed:
        return "待执行" if ready else "未就绪"
    return "就绪" if ready else "未就绪"


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

    for error in pipeline.errors:
        issues.append(_issue(
            stage="asset_registration",
            severity="错误",
            title="文件读取失败",
            message=error.get("error", "未知读取错误"),
            source=error.get("path", ""),
            action="检查文件格式、访问权限或在配置中心选择解析配置",
            blocking=True,
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

    grouped_log_issues: dict[str, list[str]] = {}
    for log in logs:
        for raw in log.issues:
            grouped_log_issues.setdefault(raw, []).append(log.source)
    for raw, sources in grouped_log_issues.items():
        duplicate = "duplicate_depth" in raw
        issues.append(_issue(
            stage="log_preprocessing",
            severity="警告",
            title="测井曲线需要复核",
            message=raw,
            source=sources[0] if sources else "",
            sources=sources,
            action="在配置中心调整曲线映射、单位或清洗策略" if not duplicate else "确认重复深度聚合策略",
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
    missing_trajectory_entities = []
    for entity in entities:
        if entity.logs and entity.preferred_head is None:
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="错误",
                title=f"{entity.canonical_name}缺少井位",
                message="LAS已关联，但没有可用井口坐标，无法进行井震空间匹配。",
                action="补充井位文件或在井名映射中关联正确记录",
                blocking=True,
            ))
        elif entity.logs and entity.preferred_head and (
            entity.preferred_head.x is None or entity.preferred_head.y is None
        ):
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="错误",
                title=f"{entity.canonical_name}井位坐标不完整",
                message="井口记录缺少X或Y坐标。",
                source=entity.preferred_head.source,
                action="检查坐标字段映射",
                blocking=True,
            ))
        if entity.logs and entity.preferred_trajectory is None:
            missing_trajectory_entities.append(entity)
        elif entity.preferred_trajectory is not None:
            for message in entity.preferred_trajectory.issues:
                issues.append(_issue(
                    stage="well_entity_alignment",
                    severity="警告",
                    title=f"{entity.canonical_name}轨迹采用降级或重建",
                    message=message,
                    source=entity.preferred_trajectory.source,
                    action="复核井斜、方位和坐标字段；真实轨迹缺失时保留低置信度标记",
                ))
        for conflict in entity.conflicts:
            issues.append(_issue(
                stage="well_entity_alignment",
                severity="警告",
                title=f"{entity.canonical_name}井位记录冲突",
                message=conflict,
                action="人工选择可信井位来源",
            ))
    if missing_trajectory_entities:
        names = "、".join(entity.canonical_name for entity in missing_trajectory_entities[:8])
        if len(missing_trajectory_entities) > 8:
            names += f" 等{len(missing_trajectory_entities)}口井"
        issues.append(_issue(
            stage="well_entity_alignment",
            severity="警告",
            title="部分LAS井缺少真实井轨迹",
            message=f"{names} 当前仅能按井口直井位置降级采样；该结果带低置信度标记。",
            action="补充井轨迹或建立井名映射；降级样本可预览，但默认不作为高可信井震融合数据",
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
                action="在配置中心指定道头坐标字节位置或选择厂商配置",
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
                action="在配置中心选择厂商SEG-Y配置或明确指定道头字节位置",
            ))
    if not seismic_geometries:
        issues.append(_issue(
            stage="seismic_geometry",
            severity="提示",
            title="当前没有可用地震数据",
            message="仍可清洗LAS和整理井数据，但不能可视化地震或构建井震样本。",
            action="如需地震任务，请登记SEG-Y路径",
        ))

    matchable_wells = sum(
        1
        for entity in entities
        if entity.logs
        and entity.preferred_head is not None
        and entity.preferred_head.x is not None
        and entity.preferred_head.y is not None
    )
    can_align = matchable_wells > 0 and geometry_with_coordinates > 0
    if not can_align:
        issues.append(_issue(
            stage="spatial_alignment",
            severity="提示",
            title="井震空间对齐尚未就绪",
            message=f"可定位井 {matchable_wells} 口，可定位地震资产 {geometry_with_coordinates} 个。",
            action="先解决井位和地震坐标问题",
        ))
    elif not pipeline.samples:
        issues.append(_issue(
            stage="spatial_alignment",
            severity="提示",
            title="空间对齐等待执行",
            message="井和地震几何已经具备匹配条件。",
            action="进入“样本构建”启动空间对齐任务",
        ))

    coordinate_reference = pipeline.config.get("matching", {}).get("coordinate_reference", {})
    coordinate_reference_verified = bool(coordinate_reference.get("verified", False))
    if can_align and not coordinate_reference_verified:
        issues.append(_issue(
            stage="spatial_alignment",
            severity="警告",
            title="井震坐标参考尚未核验",
            message="当前可生成空间匹配候选，但无法证明井位与SEG-Y XY处于同一CRS和长度单位，因此不会放行多模态训练。",
            action="核对井位与SEG-Y的CRS、坐标单位和坐标缩放后，将matching.coordinate_reference.verified设为true",
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

    _attach_safe_recommendations(pipeline, issues)
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
        if entity.logs and entity.preferred_trajectory is not None
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
            "直井降级井": len(missing_trajectory_entities),
        },
        "seismic_geometry": {
            "地震资产": len(seismic_geometries),
            "有坐标几何": geometry_with_coordinates,
            "自动推断记录": geometry_inference_records,
            "低置信度几何": low_confidence_geometry,
        },
        "spatial_alignment": {
            "可匹配井": matchable_wells,
            "匹配方法": str(pipeline.config.get("matching", {}).get("method", "nearest_trace")),
            "邻域道数": int(pipeline.config.get("matching", {}).get("neighbor_traces", 1)),
            "坐标参考已核验": "是" if coordinate_reference_verified else "否",
            "已匹配样本": len(pipeline.samples),
        },
        "vertical_alignment": {
            "井-测井标定记录": len(getattr(pipeline, "well_ties", [])),
            "深度域未明确的时深候选": len(uncertain_time_depth_domain),
            "实测时深关系": tie_statuses.get("provided_tie", 0),
            "声波积分候选": tie_statuses.get("estimated_tie", 0),
            "仅初始关系": tie_statuses.get("vertical_initial", 0),
            "仅水平定位": tie_statuses.get("horizontal_only", 0),
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
        "spatial_alignment": can_align,
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
            ),
            "metrics": stage_metrics[stage_id],
            "issue_count": sum(
                1
                for issue in issues
                if issue["stage"] == stage_id and issue["severity"] != "提示"
            ),
        })

    severity_counts = Counter(issue["severity"] for issue in issues)
    can_visualize = bool(seismic_geometries) or bool(entities)
    can_build_samples = can_align
    return {
        "stages": stages,
        "issues": issues,
        "summary": {
            "blocking": sum(1 for item in issues if item["blocking"]),
            "warnings": severity_counts["警告"],
            "information": severity_counts["提示"],
        },
        "gates": {
            "can_visualize": can_visualize,
            "can_build_samples": can_build_samples,
            "can_train_seismic_baseline": bool(seismic_geometries),
            "can_train_multimodal": bool(training_samples),
            "can_run_high_confidence_fusion": bool(training_samples) and trajectory_wells > 0,
            "can_run_prediction": False,
        },
    }
