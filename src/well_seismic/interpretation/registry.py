from __future__ import annotations

from importlib.metadata import entry_points
from typing import Iterable

from ..modeling.contracts import ModelSpec
from .contracts import InterpretationTaskSpec


class InterpretationTaskRegistry:
    """Registry for geological task semantics, separate from model runtime code."""

    entry_point_group = "well_seismic.interpretation_tasks"

    def __init__(self) -> None:
        self._specs: dict[str, InterpretationTaskSpec] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(self, spec: InterpretationTaskSpec, *, replace: bool = False) -> None:
        if spec.id in self._specs and not replace:
            raise ValueError(f"解释任务ID已注册：{spec.id}")
        self._specs[spec.id] = spec

    def get(self, task_id: str) -> InterpretationTaskSpec:
        try:
            return self._specs[task_id]
        except KeyError as exc:
            raise KeyError(f"未知解释任务：{task_id}") from exc

    def list_specs(self) -> list[InterpretationTaskSpec]:
        return sorted(self._specs.values(), key=lambda item: (item.order, item.id))

    def load_entry_points(self) -> list[str]:
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                register = entry_point.load()
                register(self)
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {"plugin": entry_point.name, "error": f"{type(exc).__name__}: {exc}"}
                )
        return loaded

    def capabilities(
        self,
        model_specs: Iterable[ModelSpec],
        runner_model_ids: Iterable[str],
    ) -> list[dict[str, object]]:
        models = list(model_specs)
        runners = set(runner_model_ids)
        output: list[dict[str, object]] = []
        for task in self.list_specs():
            task_models = [
                model for model in models if model.metadata.get("prediction_task") == task.id
            ]
            runnable_model_ids = [model.id for model in task_models if model.id in runners]
            item = task.to_dict()
            item.update(
                {
                    "model_ids": [model.id for model in task_models],
                    "runnable_model_ids": runnable_model_ids,
                    "available": bool(runnable_model_ids),
                    "status": "可运行" if runnable_model_ids else "等待模型插件",
                    # Compatibility fields consumed by older frontends.
                    "model_id": runnable_model_ids[0] if runnable_model_ids else None,
                    "output": " / ".join(task.outputs),
                }
            )
            output.append(item)
        return output


def build_default_interpretation_registry() -> InterpretationTaskRegistry:
    registry = InterpretationTaskRegistry()
    for spec in (
        InterpretationTaskSpec(
            id="fault",
            name="断层识别",
            short_name="断层分割",
            description="识别断层概率体、断层面与二值分割结果。",
            outputs=("断层概率体", "断层分割体"),
            required_modalities=("三维地震",),
            evaluation_metrics=("Dice", "IoU", "连通性"),
            order=10,
        ),
        InterpretationTaskSpec(
            id="strata",
            name="地层分割",
            short_name="地层分割",
            description="对三维后叠加地震逐 Inline 识别有序地层实例，并输出标签体与置信度体。",
            outputs=("地层实例标签体", "分割置信度体", "标签 SEG-Y"),
            required_modalities=("三维后叠加地震",),
            evaluation_metrics=("mIoU", "边界误差", "地层顺序一致性"),
            order=15,
        ),
        InterpretationTaskSpec(
            id="horizon",
            name="层位追踪",
            short_name="层位追踪",
            description="追踪关键层位面并输出层位概率与拾取置信度。",
            outputs=("层位面", "层位概率", "拾取置信度"),
            required_modalities=("二维或三维地震",),
            evaluation_metrics=("MAE", "连续性", "覆盖率"),
            order=20,
        ),
        InterpretationTaskSpec(
            id="facies",
            name="沉积相预测",
            short_name="沉积相预测",
            description="融合地震响应与井旁标定，输出沉积相类别和概率。",
            outputs=("沉积相分类", "沉积相概率体"),
            required_modalities=("地震", "可选测井与井震样本"),
            evaluation_metrics=("F1", "mIoU", "井旁一致性"),
            order=30,
        ),
        InterpretationTaskSpec(
            id="fracture",
            name="裂缝识别",
            short_name="裂缝识别",
            description="识别裂缝响应、优势方位与空间连通性。",
            outputs=("裂缝概率体", "裂缝分割体", "优势方位"),
            required_modalities=("三维地震", "可选井资料"),
            evaluation_metrics=("F1", "连通性", "方位误差"),
            order=40,
        ),
        InterpretationTaskSpec(
            id="reservoir",
            name="有利储层",
            short_name="有利储层",
            description="联合岩性、物性和地震表征预测有利储层分布。",
            outputs=("储层概率体", "品质表征", "不确定性"),
            required_modalities=("地震", "测井", "井震对齐"),
            evaluation_metrics=("AUC", "F1", "井间泛化"),
            order=50,
        ),
        InterpretationTaskSpec(
            id="target",
            name="有利目标",
            short_name="有利目标",
            description="综合断层、层位、相带和储层结果圈定候选甜点目标。",
            outputs=("目标区概率", "候选连通体", "排序依据"),
            required_modalities=("上游解释成果", "井震融合特征"),
            evaluation_metrics=("Precision@K", "目标覆盖率", "风险校准"),
            order=60,
        ),
    ):
        registry.register(spec)
    return registry
