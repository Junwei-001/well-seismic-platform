from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable

from .contracts import FusionPlugin, ModelPlugin, ModelSpec


Factory = Callable[[], Any]


class ModelRegistry:
    """模型元数据与实现解耦；外部包可通过入口点或显式注册接入。"""

    entry_point_group = "well_seismic.plugins"

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._factories: dict[str, Factory] = {}
        self._protocols: dict[str, str] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(
        self,
        spec: ModelSpec,
        factory: Factory | None = None,
        *,
        protocol: str = "model",
        replace: bool = False,
    ) -> None:
        if protocol not in {"model", "fusion"}:
            raise ValueError(f"未知插件协议：{protocol}")
        if spec.id in self._specs and not replace:
            raise ValueError(f"模型ID已注册：{spec.id}")
        self._specs[spec.id] = spec
        self._protocols[spec.id] = protocol
        if factory is not None:
            self._factories[spec.id] = factory

    def list_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def create(self, model_id: str) -> Any:
        if model_id not in self._specs:
            raise KeyError(f"未知模型：{model_id}")
        if model_id not in self._factories:
            raise RuntimeError(f"模型仅预留接口，尚未安装实现：{model_id}")
        plugin = self._factories[model_id]()
        protocol = self._protocols[model_id]
        if protocol == "model" and not isinstance(plugin, ModelPlugin):
            raise TypeError(f"插件不符合ModelPlugin协议：{model_id}")
        if protocol == "fusion" and not isinstance(plugin, FusionPlugin):
            raise TypeError(f"插件不符合FusionPlugin协议：{model_id}")
        return plugin

    def load_entry_points(self) -> list[str]:
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                register = entry_point.load()
                register(self)
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append({
                    "plugin": entry_point.name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return loaded

    def capabilities(self) -> dict[str, Any]:
        models = [spec.to_dict() for spec in self.list_specs()]
        return {
            "plugin_contract": {
                "entry_point_group": self.entry_point_group,
                "required_methods": ["fit", "predict", "save", "load"],
                "fusion_required_methods": [
                    "fit",
                    "transform",
                    "fit_transform",
                    "state_dict",
                    "load_state_dict",
                ],
                "stable_inputs": [
                    "地震窗口或三维子体",
                    "标准测井曲线",
                    "井轨迹与空间坐标",
                    "掩码、置信度和来源记录",
                ],
            },
            "plugin_load_errors": list(self.plugin_load_errors),
            "models": models,
        }


def build_default_registry() -> ModelRegistry:
    from ..fusion import ConfidenceGatedFusion

    registry = ModelRegistry()
    specs = (
        ModelSpec(
            id="faultseg_3d",
            name="FaultSeg 三维断层分割",
            category="地震分割",
            status="预处理已适配",
            description="主地震分割模型；系统输入遵循 FaultSeg 的三维体块、轴序、归一化和滑窗契约。",
            inputs=("float32 三维地震体 [Z, Inline, Crossline]",),
            outputs=("断层概率体", "断层二值掩码"),
            implementation="接口模型/faultSeg-main",
            metadata={
                "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
                "channels": 1,
                "config": "configs/faultseg.yaml",
                "training_required": False,
                "prediction_task": "fault",
            },
        ),
        ModelSpec(
            id="seismic_surface_seg",
            name="Seismic Surface Seg 地层分割",
            category="地震分割",
            status="推理已接入",
            description="SegFormer Base、SegFormer Refine 与 Mask2Former 组成的三阶段有序地层实例分割模型。",
            inputs=("规则三维后叠加 SEG-Y [Inline, Crossline, Sample]",),
            outputs=("地层实例标签体", "分割置信度体", "标签 SEG-Y"),
            implementation="接口模型/seismic_surface_seg",
            metadata={
                "input_order": ["INLINE", "CROSSLINE", "SAMPLE"],
                "slice_tensor_order": ["N", "C", "SAMPLE", "CROSSLINE"],
                "channels": 3,
                "config": "configs/surface_seg.yaml",
                "training_required": False,
                "prediction_task": "strata",
                "inference_scope": "full_volume",
            },
        ),
        ModelSpec(
            id="seismic_baseline",
            name="地震单模态Baseline",
            category="基础模型",
            status="接口预留",
            description="建立不依赖测井的二维、2.5D或三维地震分割基线。",
            inputs=("地震切片或三维子体", "任务标签"),
            outputs=("地质目标概率图", "分割结果"),
        ),
        ModelSpec(
            id="well_log_encoder",
            name="测井编码器",
            category="特征编码",
            status="接口预留",
            description="编码标准LAS曲线、缺失掩码和深度位置。",
            inputs=("标准测井曲线", "深度轴", "曲线掩码"),
            outputs=("测井特征序列",),
        ),
        ModelSpec(
            id="trajectory_encoder",
            name="井轨迹与位置编码器",
            category="特征编码",
            status="接口预留",
            description="表达XYZ、MD、TVD及井到地震道距离。",
            inputs=("井轨迹", "空间坐标", "对齐置信度"),
            outputs=("空间位置特征",),
        ),
        ModelSpec(
            id="well_seismic_alignment",
            name="可学习井震对齐",
            category="井震融合",
            status="接口预留",
            description="在最近道基线基础上学习局部空间或垂向偏移。",
            inputs=("地震邻域", "测井特征", "轨迹位置"),
            outputs=("对齐特征", "偏移量", "对齐不确定性"),
        ),
        ModelSpec(
            id="confidence_gated_fusion",
            name="置信度门控融合",
            category="井震融合",
            status="内置基线",
            description="使用水平与垂向置信度控制井震特征融合。",
            inputs=("地震特征", "测井特征", "掩码与置信度"),
            outputs=("融合特征",),
            implementation="well_seismic.fusion.ConfidenceGatedFusion",
        ),
        ModelSpec(
            id="learnable_fusion",
            name="可学习井震融合",
            category="井震融合",
            status="接口预留",
            description="可接入门控网络、交叉注意力或多模态Transformer。",
            inputs=("地震编码", "测井编码", "位置编码"),
            outputs=("多模态融合特征",),
        ),
    )
    for spec in specs:
        if spec.id == "confidence_gated_fusion":
            registry.register(spec, ConfidenceGatedFusion, protocol="fusion")
        else:
            registry.register(spec, protocol="fusion" if spec.category == "井震融合" else "model")
    return registry
