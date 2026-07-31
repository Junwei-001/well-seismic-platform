"""Portable inference-profile loading and automatic input-domain detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "inference_profiles.json"


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    checkpoint: Path
    threshold: float
    threshold_grid: tuple[float, ...]
    description: str


def detect_profile(path: str | Path) -> str:
    name = Path(path).name.lower().replace("_", "-").replace(" ", "-")
    if any(token in name for token in ("fault-enhancement", "fault-enhanced", "fef")):
        return "fault-enhanced"
    return "field-raw"


def load_profile(
    requested: str,
    input_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> InferenceProfile:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    name = detect_profile(input_path) if requested == "auto" else requested
    if name not in config["profiles"]:
        available = ", ".join(sorted(config["profiles"]))
        raise KeyError(f"unknown inference profile {name!r}; choose from {available}")
    values = config["profiles"][name]
    checkpoint = Path(values["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = config_path.parent.parent / checkpoint
    return InferenceProfile(
        name=name,
        checkpoint=checkpoint,
        threshold=float(values["threshold"]),
        threshold_grid=tuple(float(value) for value in values["threshold_grid"]),
        description=str(values.get("description", "")),
    )
