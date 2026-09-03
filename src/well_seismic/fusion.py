from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Iterable

import numpy as np

from .platform_mode import interface_only_enabled


class WellSeismicFusion(ABC):
    def fit(self, samples: Iterable[dict[str, Any]], labels: Any = None) -> "WellSeismicFusion":
        return self

    @abstractmethod
    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fit_transform(self, samples: Iterable[dict[str, Any]], labels: Any = None) -> list[dict[str, Any]]:
        rows = list(samples)
        return self.fit(rows, labels).transform(rows)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class IdentityFusion(WellSeismicFusion):
    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(samples)


class ConcatenateFusion(WellSeismicFusion):
    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for sample in samples:
            item = dict(sample)
            well = np.asarray(list(sample.get("well_features", {}).values()), dtype=float)
            seismic = np.asarray(sample.get("seismic_window") or [], dtype=float)
            item["fused_features"] = np.concatenate([well, seismic]).tolist()
            output.append(item)
        return output


class WeightedFusion(WellSeismicFusion):
    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for sample in samples:
            item = dict(sample)
            weight = float(sample.get("horizontal_confidence", 0)) * float(sample.get("vertical_confidence", 0))
            item["fusion_weight"] = weight
            if sample.get("seismic_window") is not None:
                item["weighted_seismic"] = (np.asarray(sample["seismic_window"], dtype=float) * weight).tolist()
            output.append(item)
        return output


class ConfidenceGatedFusion(WellSeismicFusion):
    """可训练模型之前的稳健基线：标准化后按井震对齐置信度门控地震特征。"""

    algorithm_name = "confidence_gated"
    algorithm_version = 1

    def __init__(self, curve_order: list[str] | None = None, epsilon: float = 1e-6):
        self.curve_order = list(curve_order or [])
        self.epsilon = float(epsilon)
        self.well_stats: dict[str, tuple[float, float]] = {}
        self.seismic_stats = (0.0, 1.0)

    @staticmethod
    def _robust_stats(values: list[float], epsilon: float) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if not len(array):
            return 0.0, 1.0
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)) * 1.4826)
        return median, max(mad, epsilon)

    def fit(self, samples: Iterable[dict[str, Any]], labels: Any = None) -> "ConfidenceGatedFusion":
        rows = list(samples)
        if not self.curve_order:
            self.curve_order = sorted({name for row in rows for name in row.get("well_features", {})})
        for name in self.curve_order:
            self.well_stats[name] = self._robust_stats(
                [row.get("well_features", {}).get(name, np.nan) for row in rows], self.epsilon
            )
        seismic_values = [
            value
            for row in rows
            for value in (row.get("seismic_window") or [])
            if np.isfinite(value)
        ]
        self.seismic_stats = self._robust_stats(seismic_values, self.epsilon)
        return self

    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        seismic_center, seismic_scale = self.seismic_stats
        for sample in samples:
            item = dict(sample)
            features = sample.get("well_features", {})
            masks = sample.get("well_mask", {})
            well_values, well_mask = [], []
            for name in self.curve_order:
                value = features.get(name)
                available = bool(masks.get(name, value is not None)) and value is not None and np.isfinite(value)
                center, scale = self.well_stats.get(name, (0.0, 1.0))
                well_values.append((float(value) - center) / scale if available else 0.0)
                well_mask.append(1.0 if available else 0.0)
            seismic = np.asarray(sample.get("seismic_window") or [], dtype=float)
            seismic_mask = np.isfinite(seismic).astype(float)
            seismic_values = np.nan_to_num((seismic - seismic_center) / seismic_scale)
            horizontal = float(np.clip(sample.get("horizontal_confidence", 0.0), 0.0, 1.0))
            vertical = float(np.clip(sample.get("vertical_confidence", 0.0), 0.0, 1.0))
            gate = horizontal * vertical if len(seismic) else 0.0
            item["fused_features"] = (
                well_values + (seismic_values * gate).tolist() + well_mask + seismic_mask.tolist() + [gate]
            )
            item["fusion_weight"] = gate
            item["fusion_metadata"] = {
                "algorithm": self.algorithm_name,
                "version": self.algorithm_version,
                "curve_order": list(self.curve_order),
                "horizontal_confidence": horizontal,
                "vertical_confidence": vertical,
            }
            output.append(item)
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "version": self.algorithm_version,
            "curve_order": list(self.curve_order),
            "epsilon": self.epsilon,
            "well_stats": {name: list(values) for name, values in self.well_stats.items()},
            "seismic_stats": list(self.seismic_stats),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.curve_order = list(state["curve_order"])
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.well_stats = {name: tuple(map(float, values)) for name, values in state["well_stats"].items()}
        self.seismic_stats = tuple(map(float, state["seismic_stats"]))


class LearnableFusionAdapter(WellSeismicFusion):
    """Protocol adapter: inject a callable/model without forcing PyTorch as a dependency."""
    def __init__(self, model: Any):
        self.model = model

    def transform(self, samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.model(list(samples))

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict() if hasattr(self.model, "state_dict") else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(state)


@dataclass(frozen=True)
class FusionStrategySpec:
    id: str
    name: str
    stage: str
    status: str
    description: str
    inputs: tuple[str, ...]
    output: str
    training_required: bool = False
    recommended_for: str = ""
    contract_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FusionFactory = Callable[[dict[str, Any], Any], WellSeismicFusion]


class FusionRegistry:
    """井震融合策略注册中心；元数据、工厂和下游任务相互解耦。"""

    entry_point_group = "well_seismic.fusion_strategies"

    def __init__(self) -> None:
        self._specs: dict[str, FusionStrategySpec] = {}
        self._factories: dict[str, FusionFactory] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(
        self,
        spec: FusionStrategySpec,
        factory: FusionFactory,
        *,
        replace: bool = False,
    ) -> None:
        if spec.id in self._specs and not replace:
            raise ValueError(f"井震融合策略ID已注册：{spec.id}")
        self._specs[spec.id] = spec
        self._factories[spec.id] = factory

    def create(
        self,
        strategy_id: str,
        *,
        options: dict[str, Any] | None = None,
        model: Any = None,
    ) -> WellSeismicFusion:
        if strategy_id not in self._factories:
            raise ValueError(f"未知井震融合算法：{strategy_id}")
        return self._factories[strategy_id](dict(options or {}), model)

    def capabilities(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]

    def load_entry_points(self) -> list[str]:
        """Load callbacks declared as ``register(fusion_registry)``."""
        if interface_only_enabled():
            return []
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


def _build_learnable_fusion(options: dict[str, Any], model: Any) -> WellSeismicFusion:
    del options
    if model is None:
        raise ValueError("learnable融合算法需要传入model")
    return LearnableFusionAdapter(model)


def build_default_fusion_registry() -> FusionRegistry:
    registry = FusionRegistry()
    registry.register(
        FusionStrategySpec(
            id="identity",
            name="原样透传",
            stage="消融基线",
            status="内置",
            description="不改变标准样本，用于核验数据契约与消融实验。",
            inputs=("标准井震样本",),
            output="原始样本",
            recommended_for="接口联调与消融",
        ),
        lambda options, model: IdentityFusion(),
    )
    registry.register(
        FusionStrategySpec(
            id="concatenate",
            name="直接拼接",
            stage="特征级融合",
            status="内置",
            description="按稳定曲线顺序拼接测井特征与地震窗口。",
            inputs=("测井特征", "地震窗口"),
            output="拼接特征",
            recommended_for="快速单模态/多模态对照",
        ),
        lambda options, model: ConcatenateFusion(),
    )
    registry.register(
        FusionStrategySpec(
            id="weighted",
            name="置信度加权",
            stage="特征级融合",
            status="内置",
            description="使用水平与垂向对齐置信度加权地震响应。",
            inputs=("地震窗口", "水平置信度", "垂向置信度"),
            output="加权地震特征",
            recommended_for="对齐质量敏感的稳健基线",
        ),
        lambda options, model: WeightedFusion(),
    )
    registry.register(
        FusionStrategySpec(
            id="confidence_gated",
            name="置信度门控融合",
            stage="统一表征",
            status="当前默认",
            description="稳健标准化后联合缺失掩码和井震对齐置信度形成统一特征。",
            inputs=("标准测井曲线", "地震窗口", "缺失掩码", "对齐置信度"),
            output="可追溯融合特征",
            recommended_for="当前训练与推理基线",
        ),
        lambda options, model: ConfidenceGatedFusion(**options),
    )
    registry.register(
        FusionStrategySpec(
            id="learnable",
            name="可学习融合适配器",
            stage="统一表征",
            status="待接入模型",
            description="为门控网络、交叉注意力和多模态 Transformer 提供框架无关适配层。",
            inputs=("地震编码", "测井编码", "位置编码", "质量掩码"),
            output="任务共享或任务专属融合特征",
            training_required=True,
            recommended_for="新增井震融合方案",
        ),
        _build_learnable_fusion,
    )
    return registry


def build_fusion(
    config: dict[str, Any] | None = None,
    model: Any = None,
    *,
    registry: FusionRegistry | None = None,
) -> WellSeismicFusion:
    """配置驱动的融合算法工厂；新增策略只需注册规格与工厂。"""
    options = dict(config or {})
    name = str(options.pop("algorithm", "confidence_gated")).lower()
    return (registry or build_default_fusion_registry()).create(
        name,
        options=options,
        model=model,
    )
