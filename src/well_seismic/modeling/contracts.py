from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    category: str
    status: str
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    version: str = "接口1.0"
    configurable: bool = True
    implementation: str | None = None
    scientific_status: str = "unassessed"
    runtime_status: str = "interface_only"
    evidence_class: str = "none"
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ModelPlugin(Protocol):
    """小模型的最小稳定协议，不强制依赖PyTorch。"""

    spec: ModelSpec

    def fit(self, dataset: Any, *, validation_data: Any = None, config: dict[str, Any] | None = None) -> Any:
        ...

    def predict(self, batch: Any, *, config: dict[str, Any] | None = None) -> Any:
        ...

    def save(self, directory: str | Path) -> None:
        ...

    def load(self, directory: str | Path) -> None:
        ...


@runtime_checkable
class FusionPlugin(Protocol):
    """井震融合组件协议，与下游预测模型协议相互独立。"""

    def fit(self, samples: Any, labels: Any = None) -> Any:
        ...

    def transform(self, samples: Any) -> list[dict[str, Any]]:
        ...

    def fit_transform(self, samples: Any, labels: Any = None) -> list[dict[str, Any]]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...
