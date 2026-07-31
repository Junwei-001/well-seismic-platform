"""Checkpoint helpers shared by all training and inference entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import FaultSegNet


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FaultSegNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
        metadata = checkpoint
        bottleneck = int(checkpoint.get("bottleneck_channels", 128))
    else:
        state = checkpoint
        metadata = {}
        # Infer the variant from conv4a's output dimension.
        bottleneck = int(state["conv4a.weight"].shape[0])
    model = FaultSegNet(bottleneck_channels=bottleneck)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, metadata
