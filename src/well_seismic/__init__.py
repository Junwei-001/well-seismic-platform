"""Manifest-driven well/seismic preprocessing."""

from .pipeline import WellSeismicPipeline
from .datasets import JsonlMultimodalDataset
from .faultseg import FaultSegInputSpec, FaultSegVolume, build_faultseg_volume
from .fusion import (
    ConfidenceGatedFusion,
    FusionRegistry,
    FusionStrategySpec,
    LearnableFusionAdapter,
    build_default_fusion_registry,
    build_fusion,
)

__all__ = [
    "WellSeismicPipeline",
    "JsonlMultimodalDataset",
    "ConfidenceGatedFusion",
    "FusionRegistry",
    "FusionStrategySpec",
    "LearnableFusionAdapter",
    "build_default_fusion_registry",
    "build_fusion",
]
__version__ = "0.1.0"
