"""Compose the public platform capability document from independent registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cigvis_adapter import cigvis_status, public_visualization_text
from .data_flow import build_model_data_flow_specs
from .llm import load_llm_settings


def _is_historical_model(spec: Any) -> bool:
    metadata = dict(getattr(spec, "metadata", {}) or {})
    return bool(
        metadata.get("archived")
        or metadata.get("historical_result_compatibility")
    )


def _public_model_capability(model: dict[str, Any]) -> dict[str, Any]:
    """Project display projection; runtime IDs and routing contracts stay stable."""

    public = dict(model)
    public.pop("implementation", None)
    for field in ("name", "category", "status", "description", "version"):
        if public.get(field):
            public[field] = public_visualization_text(public[field])
    for field in ("inputs", "outputs", "warnings"):
        values = public.get(field)
        if isinstance(values, (list, tuple)):
            public[field] = [public_visualization_text(value) for value in values]
    metadata = public.get("metadata")
    if isinstance(metadata, dict):
        public["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key not in {"upstream", "checkpoint", "checkpoint_path", "source_path"}
        }
    return public


def build_platform_capabilities(
    *,
    project_root: Path,
    platform_config: dict[str, Any],
    model_registry: Any,
    interpretation_registry: Any,
    fusion_registry: Any,
    input_adapters: Any,
    prediction_runners: Any,
) -> dict[str, Any]:
    model_specs = list(model_registry.list_specs())
    historical_model_ids = {
        str(spec.id) for spec in model_specs if _is_historical_model(spec)
    }
    public_model_specs = [
        spec for spec in model_specs if str(spec.id) not in historical_model_ids
    ]
    model_capabilities = dict(model_registry.capabilities())
    model_capabilities["models"] = [
        _public_model_capability(model)
        for model in (model_capabilities.get("models") or [])
        if str(model.get("id") or "") not in historical_model_ids
    ]
    adapter_capabilities = list(input_adapters.capabilities())
    public_adapter_capabilities = [
        adapter
        for adapter in adapter_capabilities
        if str(adapter.get("model_id") or "") not in historical_model_ids
    ]
    runner_model_ids = [
        str(model_id)
        for model_id in prediction_runners.model_ids()
        if str(model_id) not in historical_model_ids
    ]
    interpretation_tasks = interpretation_registry.capabilities(
        public_model_specs,
        runner_model_ids,
    )
    prediction_menu_tasks = [
        task
        for task in interpretation_tasks
        if task.get("show_in_prediction_menu") is True
    ]
    model_data_flows = build_model_data_flow_specs(
        public_model_specs,
        public_adapter_capabilities,
        runner_model_ids,
    )
    llm_status = load_llm_settings(platform_config).public_status()
    return {
        "workflow": [
            {"id": "overview", "name": "项目总览", "section": "project", "purpose": "汇总进度、放行条件与阻塞问题"},
            {"id": "preparation", "name": "数据准备", "section": "workflow", "purpose": "形成模型无关数据快照，并按需构建井震多模态数据视图"},
            {"id": "visualization", "name": "数据可视化", "section": "workflow", "purpose": "从通用快照显示完整地震资产、三维体、井轨迹和测井图层"},
            {"id": "prediction", "name": "预测解释", "section": "workflow", "purpose": "按任务进入独立页面，在任务内选择兼容模型"},
            {"id": "evaluation", "name": "评估导出", "section": "workflow", "purpose": "比较指标、版本、置信度并导出结果"},
            {"id": "models", "name": "模型中心", "section": "system", "purpose": "登记模型、输入适配器、运行器和版本"},
            {"id": "settings", "name": "配置中心", "section": "system", "purpose": "管理曲线、单位、SEG-Y和井字段知识映射"},
        ],
        "data_snapshot_contract": {
            "version": "well-seismic.source-snapshot.v3",
            "semantics": "model_neutral",
            "layers": ["source_assets", "canonical_data", "derived_views", "model_adapters", "task_outputs"],
            "identity": (
                "source_content_plus_crs_vertical_time_and_transform_semantics"
            ),
            "mutation_policy": "same_path_content_replacement_fails_closed",
            "policy": "通用预处理不感知具体模型；模型差异由输入适配器处理。",
        },
        "model_data_flows": model_data_flows,
        "data_flow_contract": {
            "version": "well-seismic.model-data-flow.v1",
            "scope": "model_level",
            "source_snapshot": "well-seismic.source-snapshot.v3 target",
            "prepared_view": "well-seismic.prepared-view.v1 target",
            "registration": "well-seismic.registration.v3 target",
            "input_attestation": "well-seismic.prediction-input-attestation.v1",
            "policy": (
                "任务定义地质语义，具体模型声明所需模态、标定策略、"
                "合法降级和实际消费证明。"
            ),
        },
        "visualization": {
            "contract_version": "2.0",
            "scene_endpoint": "/统一数据可视化?embed=1",
            "release_catalog_endpoint": "/api/v1/releases",
            "engine": cigvis_status(project_root),
            "layer_protocol": {
                "schema_version": "well-seismic.visualization-layer.v1",
                "required_fields": ["id", "name", "kind", "role", "source"],
                "kinds": ["volume", "surface", "points", "trajectory", "well_curve", "table", "report"],
                "roles": ["source", "baseline", "prediction", "observation", "uncertainty", "valid_mask", "quality", "evidence"],
                "spatial_metadata": ["axis_order", "crs", "origin", "spacing", "unit", "datum"],
                "policy": "图层可独立开关；科学状态、运行状态和可视化能力互不替代。",
            },
            "layers": [
                {"id": "seismic_volume", "name": "三维地震振幅体", "status": "平台可视化"},
                {"id": "seismic_slices", "name": "Inline/Crossline/时间切片", "status": "三维动态交互"},
                {"id": "seismic_line", "name": "二维地震剖面", "status": "平台可视化"},
                {"id": "well_trajectory", "name": "井轨迹叠加", "status": "平台可视化"},
                {"id": "well_logs", "name": "测井曲线", "status": "数据接口"},
                {"id": "spatial_matches", "name": "井震匹配点", "status": "接口"},
                {"id": "model_prediction", "name": "概率体与分割结果", "status": "接口"},
                {"id": "horizon_surface", "name": "命名层位面与有序地层场", "status": "冻结成果接口"},
                {"id": "facies_class", "name": "离散地震相类别与分类切片", "status": "冻结成果接口"},
                {"id": "geobody_candidate", "name": "特殊地质体候选解释", "status": "冻结成果接口"},
                {"id": "well_prediction", "name": "井曲线预测、区间与有效掩码", "status": "冻结成果接口"},
            ],
            "extension_points": [
                "VisualizationLayerProvider",
                "VolumeRendererAdapter",
                "TrajectoryOverlayProvider",
                "PredictionOverlayProvider",
            ],
        },
        **model_capabilities,
        "model_input_adapters": public_adapter_capabilities,
        "prediction_runner_model_ids": runner_model_ids,
        # This is the public prediction-menu contract, not a dump of every
        # registered research, prerequisite or archived interpretation task.
        # Those lower-level registrations remain available to the model,
        # runner and release registries without becoming user-facing entries.
        "prediction_tasks": prediction_menu_tasks,
        "interpretation_task_contract": {
            "entry_point_group": interpretation_registry.entry_point_group,
            "plugin_load_errors": list(interpretation_registry.plugin_load_errors),
            "model_binding": 'ModelSpec.metadata["prediction_task"]',
            "runner_binding": "PredictionRunnerRegistry.register(model_id, runner)",
        },
        "fusion_strategies": fusion_registry.capabilities(),
        "runtime_plugin_contract": {
            "input_adapter_entry_point_group": input_adapters.entry_point_group,
            "prediction_runner_entry_point_group": prediction_runners.entry_point_group,
            "fusion_strategy_entry_point_group": fusion_registry.entry_point_group,
            "plugin_load_errors": [
                *input_adapters.plugin_load_errors,
                *prediction_runners.plugin_load_errors,
                *fusion_registry.plugin_load_errors,
            ],
        },
        "configuration_libraries": [
            {"id": "curve_knowledge", "name": "曲线知识映射库", "file": "configs/curve_knowledge.yaml"},
            {"id": "units", "name": "单位知识与换算库", "file": "configs/units.yaml"},
            {"id": "well_schema", "name": "井字段与井名映射", "file": "configs/well_schema.yaml"},
            {"id": "vertical_datum", "name": "MSL/SRD/KB/GL垂向基准规则", "file": "configs/vertical_datum.yaml"},
            {"id": "segy_profiles", "name": "SEG-Y版本与道头配置", "file": "configs/segy_profiles.yaml"},
            {"id": "preprocessing", "name": "清洗与重采样策略", "file": "configs/preprocessing.yaml"},
            {"id": "matching", "name": "井震空间对齐策略", "file": "configs/matching.yaml"},
            {"id": "fusion", "name": "井震融合策略", "file": "configs/fusion.yaml"},
            {"id": "fault_models", "name": "断层识别输入与推理策略", "file": "configs/faultseg.yaml"},
            {"id": "surface_seg", "name": "地层分割输入与推理策略", "file": "configs/surface_seg.yaml"},
            {"id": "llm", "name": "LLM受控判断策略", "file": "configs/llm.yaml"},
        ],
        "llm": {
            **llm_status,
            "trigger_policy": "规则优先；未决问题可由LLM生成受控转换适配器，自动测试后人工启用",
            "guardrails": [
                "转换适配器只允许单位、曲线、井名、字段、空值及基准/时间语义白名单操作",
                "LLM输出必须通过结构、数值范围和样例执行测试",
                "非常规单位可由LLM在白名单中识别；换算系数、符号、容差及坐标转换由确定性代码执行",
                "物理空间使用z_msl_m；时间空间单独声明SRD、TWT/OWT和correction_state",
                "corrected_to_srd不重复校正，uncorrected仅执行近地表datum静校正，unknown阻断",
                "不得生成或执行任意脚本，不凭空补造坐标、基准高程、深度和时深关系",
                "所有草案、测试、人工启用和来源均写入审计",
            ],
            "transformation_endpoint": "/api/v1/tasks/{task_id}/issues/{issue_id}/transformation-drafts",
            "assistant_endpoint": "/api/v1/assistant/chat",
            "required_environment": [
                "WELL_SEISMIC_LLM_ENABLED",
                "WELL_SEISMIC_LLM_MODEL",
                "KIMI_API_KEY",
            ],
        },
    }
