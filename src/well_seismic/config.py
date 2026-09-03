from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_dir: str | Path, manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    config_dir = Path(config_dir)
    cfg: dict[str, Any] = {}
    for name in (
        "curve_knowledge.yaml",
        "units.yaml",
        "segy_profiles.yaml",
        "well_schema.yaml",
        "vertical_datum.yaml",
        "preprocessing.yaml",
        "matching.yaml",
        "fusion.yaml",
        "faultseg.yaml",
        "surface_seg.yaml",
        "llm.yaml",
    ):
        path = config_dir / name
        if path.exists():
            cfg = deep_merge(cfg, load_yaml(path))
    if isinstance(manifest, dict):
        cfg["manifest"] = manifest
        cfg["manifest_path"] = str((config_dir / "运行时自动清单.yaml").resolve())
    else:
        cfg["manifest"] = load_yaml(manifest)
        cfg["manifest_path"] = str(Path(manifest).resolve())
    return cfg
