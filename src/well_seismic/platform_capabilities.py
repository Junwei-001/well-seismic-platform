"""Compose the public platform capability document from independent registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cigvis_adapter import cigvis_status
from .llm import load_llm_settings


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
    model_capabilities = model_registry.capabilities()
    interpretation_tasks = interpretation_registry.capabilities(
        model_registry.list_specs(),
        prediction_runners.model_ids(),
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
            "version": "1.0",
            "semantics": "model_neutral",
            "layers": ["source_assets", "canonical_data", "derived_views", "model_adapters", "task_outputs"],
            "policy": "通用预处理不感知具体模型；模型差异由输入适配器处理。",
        },
        "visualization": {
            "contract_version": "2.0",
            "scene_endpoint": "/统一数据可视化?embed=1",
            "engine": cigvis_status(project_root),
            "layers": [
                {"id": "seismic_volume", "name": "三维地震振幅体", "status": "CIGVis"},
                {"id": "seismic_slices", "name": "Inline/Crossline/时间切片", "status": "Viser动态交互"},
                {"id": "seismic_line", "name": "二维地震剖面", "status": "CIGVis"},
                {"id": "well_trajectory", "name": "井轨迹叠加", "status": "CIGVis"},
                {"id": "well_logs", "name": "测井曲线", "status": "数据接口"},
                {"id": "spatial_matches", "name": "井震匹配点", "status": "接口"},
                {"id": "model_prediction", "name": "概率体与分割结果", "status": "接口"},
            ],
            "extension_points": [
                "VisualizationLayerProvider",
                "VolumeRendererAdapter",
                "TrajectoryOverlayProvider",
                "PredictionOverlayProvider",
            ],
        },
        **model_capabilities,
        "model_input_adapters": input_adapters.capabilities(),
        "prediction_runner_model_ids": prediction_runners.model_ids(),
        "prediction_tasks": interpretation_tasks,
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
            {"id": "segy_profiles", "name": "SEG-Y版本与道头配置", "file": "configs/segy_profiles.yaml"},
            {"id": "preprocessing", "name": "清洗与重采样策略", "file": "configs/preprocessing.yaml"},
            {"id": "matching", "name": "井震空间对齐策略", "file": "configs/matching.yaml"},
            {"id": "fusion", "name": "井震融合策略", "file": "configs/fusion.yaml"},
            {"id": "faultseg", "name": "FaultSeg输入与推理策略", "file": "configs/faultseg.yaml"},
            {"id": "surface_seg", "name": "地层分割输入与推理策略", "file": "configs/surface_seg.yaml"},
            {"id": "llm", "name": "LLM受控判断策略", "file": "configs/llm.yaml"},
        ],
        "llm": {
            **llm_status,
            "trigger_policy": "规则优先；未决问题可由LLM生成受控转换适配器，自动测试后人工启用",
            "guardrails": [
                "转换适配器只允许单位、曲线、井名、字段和空值五类白名单操作",
                "LLM输出必须通过结构、数值范围和样例执行测试",
                "不得生成或执行任意脚本，不推断坐标、深度和时深关系",
                "所有草案、测试、人工启用和来源均写入审计",
            ],
            "transformation_endpoint": "/api/v1/tasks/{task_id}/issues/{issue_id}/transformation-drafts",
            "assistant_endpoint": "/api/v1/assistant/chat",
            "required_environment": [
                "WELL_SEISMIC_LLM_ENABLED",
                "WELL_SEISMIC_LLM_MODEL",
                "GLM_API_KEY",
            ],
        },
    }
