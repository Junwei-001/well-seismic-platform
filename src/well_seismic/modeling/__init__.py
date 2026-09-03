"""井震融合、训练模型和下游任务的插件化接口。"""

from .contracts import FusionPlugin, ModelPlugin, ModelSpec
from .input_adapters import (
    HorizonP17InputAdapter,
    ModelInputAdapterRegistry,
    ModelInputBatch,
    ModelInputRequest,
    SurfaceSegInputAdapter,
    build_default_input_adapters,
)
from .registry import ModelRegistry, build_default_registry

__all__ = [
    "FusionPlugin",
    "HorizonP17InputAdapter",
    "ModelInputAdapterRegistry",
    "ModelInputBatch",
    "ModelInputRequest",
    "ModelPlugin",
    "ModelRegistry",
    "ModelSpec",
    "SurfaceSegInputAdapter",
    "build_default_input_adapters",
    "build_default_registry",
]
