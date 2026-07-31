"""井震融合、训练模型和下游任务的插件化接口。"""

from .contracts import FusionPlugin, ModelPlugin, ModelSpec
from .registry import ModelRegistry, build_default_registry
from .input_adapters import (
    SurfaceSegInputAdapter,
    ModelInputAdapterRegistry,
    ModelInputBatch,
    ModelInputRequest,
    build_default_input_adapters,
)

__all__ = [
    "FusionPlugin",
    "ModelPlugin",
    "ModelSpec",
    "ModelRegistry",
    "SurfaceSegInputAdapter",
    "ModelInputAdapterRegistry",
    "ModelInputBatch",
    "ModelInputRequest",
    "build_default_input_adapters",
    "build_default_registry",
]
