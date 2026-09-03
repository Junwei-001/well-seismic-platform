from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import entry_points

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
            runnable_model_ids = [
                model.id
                for model in task_models
                if model.runtime_status == "runnable" and model.id in runners
            ]
            runnable_models = [
                model
                for model in task_models
                if model.runtime_status == "runnable" and model.id in runners
            ]
            requires_registration = any(
                bool(model.metadata.get("requires_registration"))
                for model in runnable_models
            )
            runtime_statuses = sorted({model.runtime_status for model in task_models})
            scientific_statuses = sorted({model.scientific_status for model in task_models})
            if task.lifecycle == "archived":
                task_status = "已归档"
            elif runnable_model_ids:
                task_status = "可运行"
            elif "adapter_required" in runtime_statuses:
                task_status = "运行适配待完成"
            elif "precomputed_only" in runtime_statuses or task.lifecycle == "evidence_only":
                task_status = "已有冻结成果"
            elif runtime_statuses and set(runtime_statuses) == {"blocked"}:
                task_status = "运行已阻断"
            else:
                task_status = "等待模型插件"
            item = task.to_dict()
            item.update(
                {
                    "model_ids": [model.id for model in task_models],
                    "runnable_model_ids": runnable_model_ids,
                    "model_runtime_statuses": runtime_statuses,
                    "model_scientific_statuses": scientific_statuses,
                    "available": bool(runnable_model_ids),
                    "requires_registration": requires_registration,
                    "prerequisite_task_ids": (
                        ["alignment"] if requires_registration else []
                    ),
                    "active": task.lifecycle != "archived",
                    "status": task_status,
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
            id="layerpulse",
            name="LayerPulse 智能解释",
            short_name="LayerPulse",
            description=(
                "单一共享 Backbone 一次输出构造、地层、地震相、地质体、属性、"
                "井震匹配、连通性与不确定性共 11 项结果。"
            ),
            outputs=("6 个完整多类 logits/直接 argmax", "5 个连续场", "统一预览图集与运行回执"),
            required_modalities=("三维后叠加地震", "可选登记井曲线与完整轨迹（无时深表）"),
            evaluation_metrics=("多任务完整性", "有限性与类别范围", "固定几何预览可视化"),
            order=1,
            group="foundation_model",
            # The data-registration chooser exposes this as the parallel
            # LayerPulse route; do not mix it into the legacy seven-task menu.
            show_in_prediction_menu=False,
        ),
        InterpretationTaskSpec(
            id="alignment",
            name="井震概率注册",
            short_name="WellFuse-Align",
            description="沿完整井轨迹输出每个LAS点的TWT均值、标准差、质量与基准面记录。",
            outputs=("TWT_mean/TWT_std", "注册CSV/LAS", "质量与来源清单"),
            required_modalities=("LAS", "完整井轨迹", "三维地震"),
            evaluation_metrics=("整井MAE", "覆盖率", "不确定性校准"),
            order=5,
            group="registration",
            lifecycle="evidence_only",
        ),
        InterpretationTaskSpec(
            id="fault",
            name="断层识别",
            short_name="断层识别",
            description="从三维地震体输出确定性断层掩码、正交切片与逐切片统计；概率仅保留为技术审计。",
            outputs=("断层二值掩码", "断层正交切片", "逐切片统计"),
            required_modalities=("三维地震",),
            evaluation_metrics=("Dice", "IoU", "连通性"),
            order=10,
            group="structure",
            show_in_prediction_menu=True,
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
            group="structure",
        ),
        InterpretationTaskSpec(
            id="horizon",
            name="层位识别",
            short_name="层位识别",
            description=(
                "对规则三维后叠加地震逐 Inline 识别有序地层实例，输出完整标签体、"
                "置信度体和可直接读取的标签 SEG-Y；实例标签不冒充跨 Inline 命名层位。"
            ),
            outputs=("地层实例标签体", "分割置信度体", "标签 SEG-Y"),
            required_modalities=("规则三维后叠加地震",),
            evaluation_metrics=("mIoU", "边界误差", "地层顺序一致性"),
            order=20,
            group="structure",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="horizon_legacy",
            name="历史层位面成果",
            short_name="历史层位面成果",
            description="保留既有四层位候选面的下载与可视化解析，不再接受新推理任务。",
            outputs=("历史候选层位面", "历史TWT与有效掩码", "历史不确定性"),
            required_modalities=("历史完成任务",),
            evaluation_metrics=("历史证据复核",),
            order=21,
            group="archive",
            lifecycle="archived",
        ),
        InterpretationTaskSpec(
            id="facies_1d",
            name="一维地震相分类",
            short_name="一维地震相分类",
            description=(
                "基于当前封存快照井资产与已完成融合视图，沿井深输出Viterbi确定性"
                "沉积相序列及连续相层段。"
            ),
            outputs=("逐MD确定相类别CSV", "连续相层段CSV"),
            required_modalities=("当前封存快照井曲线", "当前PreparedView"),
            evaluation_metrics=("整井Macro-F1", "V/D/H宏平均", "Bootstrap"),
            order=50,
            group="facies",
            lifecycle="evidence_only",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="facies",
            name="沉积相预测",
            short_name="沉积相预测",
            description="融合地震响应与井旁标定，输出离散沉积相类别与分类切片。",
            outputs=("离散沉积相分类", "分类切片包"),
            required_modalities=("地震", "可选测井与井震样本"),
            evaluation_metrics=("F1", "mIoU", "井旁一致性"),
            order=30,
            group="facies",
            lifecycle="archived",
        ),
        InterpretationTaskSpec(
            id="facies_3d",
            name="三维地震相分割",
            short_name="三维地震相分割",
            description=(
                "生成三维离散地震相候选体及分类切片；模型级元数据严格区分"
                "公开稠密基准工区验证和未知工区迁移候选，概率仅供内部判别。"
            ),
            outputs=("离散相类别候选体", "分类切片包", "有效掩码"),
            required_modalities=("三维地震", "可选井侧Facies"),
            evaluation_metrics=("mIoU", "Macro-F1", "井旁一致性", "稳定性"),
            order=60,
            group="facies",
            lifecycle="evidence_only",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="channel",
            name="河道地质体",
            short_name="Channel",
            description="输出河道候选概率、边界、距离、骨架、连通性与不确定性。",
            outputs=("河道概率体", "边界/骨架", "不确定性与有效掩码"),
            required_modalities=("三维地震",),
            evaluation_metrics=("连通性", "拓扑保持", "候选稳定性"),
            order=35,
            group="geobody",
            lifecycle="evidence_only",
        ),
        InterpretationTaskSpec(
            id="karst",
            name="岩溶地质体",
            short_name="Karst",
            description="输出岩溶候选概率、边界、距离、连通体与不确定性。",
            outputs=("岩溶概率体", "边界/连通体", "不确定性与有效掩码"),
            required_modalities=("三维地震",),
            evaluation_metrics=("连通性", "拓扑保持", "候选稳定性"),
            order=36,
            group="geobody",
            lifecycle="evidence_only",
        ),
        InterpretationTaskSpec(
            id="fracture_development",
            name="井侧裂缝发育排序",
            short_name="裂缝发育排序",
            description=(
                "沿整井MD输出低/中/高相对发育连续深度段；当前不等同于地震体素裂缝分割。"
            ),
            outputs=("连续MD发育层段CSV", "低/中/高相对发育级别"),
            required_modalities=("当前封存快照井曲线", "当前PreparedView"),
            evaluation_metrics=("Spearman", "Top-Q PR-AUC lift", "区段F1"),
            order=70,
            group="well",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="fracture",
            name="空间裂缝识别（未开放）",
            short_name="空间裂缝（未开放）",
            description="识别裂缝响应、优势方位与空间连通性。",
            outputs=("裂缝概率体", "裂缝分割体", "优势方位"),
            required_modalities=("三维地震", "可选井资料"),
            evaluation_metrics=("F1", "连通性", "方位误差"),
            order=40,
            group="archived",
            lifecycle="archived",
        ),
        InterpretationTaskSpec(
            id="well_property",
            name="储层物性预测",
            short_name="储层物性预测",
            description=(
                "基于当前封存快照中的井曲线预测DEN、POR、LOG_PERM、SW和VSH，"
                "输出可追溯的整井物性曲线。"
            ),
            outputs=("整井预测CSV/LAS/NPZ", "预测区间", "不确定性与有效掩码"),
            required_modalities=("封存快照井曲线",),
            evaluation_metrics=("整井MAE", "RMSE", "R²"),
            order=30,
            group="well",
            lifecycle="evidence_only",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="fluid_interpretation",
            name="流体解释",
            short_name="流体解释",
            description=(
                "读取当前封存快照井资产，将 Dry、Water、Oil、Gas、Mixed 五类判别"
                "整理为连续MD流体层段；概率只在进程内参与判别，不落盘公开。"
            ),
            outputs=("连续流体层段 CSV",),
            required_modalities=("当前封存快照标准九线测井", "当前PreparedView"),
            evaluation_metrics=("整井Macro-F1", "Accuracy", "含烃F1"),
            order=40,
            group="fluid",
            show_in_prediction_menu=True,
        ),
        InterpretationTaskSpec(
            id="hydrocarbon_evidence",
            name="含烃证据",
            short_name="Hydrocarbon",
            description="输出研究用途的含烃概率、阈值来源和几何适用性警告。",
            outputs=("含烃概率", "预测区间", "校准与证据报告"),
            required_modalities=("井曲线", "完整井轨迹", "Align预测"),
            evaluation_metrics=("Whole-well F1", "AUROC/AUPRC", "ECE"),
            order=47,
            group="fluid",
            lifecycle="evidence_only",
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
            group="composite",
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
            group="composite",
        ),
    ):
        registry.register(spec)
    return registry
