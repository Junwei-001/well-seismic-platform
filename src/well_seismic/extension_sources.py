"""Hash-bound input sources declared by external task-model packages.

Well-side models normally consume a sealed SourceSnapshot (or, historically, a
WellFuse checkout).  An extension package may instead declare one registered
dataset that ships inside its own ``models/task-models`` payload::

    ModelSpec.metadata["extension_source"] = {
        "kind": "registered_dataset",
        "path": "models/task-models/<model_id>/<version>/payload/<file>",
        "sha256": "<64 hex>",
    }

The declaration is fail-closed: the path must stay inside
``models/task-models``, exist, and match the declared SHA-256 at every run.
``WELL_SEISMIC_MODEL_PROJECT_ROOT`` selects the deployment root used by model
registration and input resolution; invalid configured roots fail immediately.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXTENSION_SOURCE_CONTRACT_VERSION = "well-seismic.extension-source.v1"
_SUPPORTED_KINDS = frozenset({"registered_dataset"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def declared_extension_source(spec: Any) -> Mapping[str, Any] | None:
    declared = (getattr(spec, "metadata", {}) or {}).get("extension_source")
    if declared is None:
        return None
    if not isinstance(declared, Mapping) or declared.get("kind") not in _SUPPORTED_KINDS:
        raise ValueError(f"模型 {getattr(spec, 'id', '?')} 的 extension_source 数据源声明不受支持")
    return declared


def resolve_extension_source(
    spec: Any, *, project_root: Path
) -> tuple[Path, dict[str, Any]] | None:
    """Return ``(source_path, prediction_source_identity)`` or ``None`` if undeclared."""

    declared = declared_extension_source(spec)
    if declared is None:
        return None
    configured_root = os.environ.get("WELL_SEISMIC_MODEL_PROJECT_ROOT")
    if configured_root is not None:
        selected = Path(configured_root.strip())
        if not configured_root.strip() or not selected.is_absolute():
            raise ValueError("WELL_SEISMIC_MODEL_PROJECT_ROOT 必须配置为项目目录的绝对路径")
        project_root = selected.resolve()
        if not (project_root / "models/task-models").is_dir():
            raise ValueError("WELL_SEISMIC_MODEL_PROJECT_ROOT 指定的项目缺少 models/task-models 目录")
    root = (project_root / "models" / "task-models").resolve()
    path = (project_root / str(declared.get("path") or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("extension_source 数据源必须位于 models/task-models 目录内") from exc
    if not path.is_file():
        raise FileNotFoundError(f"extension_source 数据源文件不存在：{path}")
    expected = str(declared.get("sha256") or "").strip().casefold()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("extension_source 的 SHA-256 必须为 64 位小写十六进制字符")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"extension_source 数据源 SHA-256 校验不匹配：{observed} != {expected}"
        )
    return path, {
        "kind": "extension_registered_dataset",
        "contract_version": EXTENSION_SOURCE_CONTRACT_VERSION,
        "path": str(path),
        "sha256": observed,
        "size": path.stat().st_size,
        "integrity_status": "sha256_verified_against_model_metadata",
    }


__all__ = [
    "EXTENSION_SOURCE_CONTRACT_VERSION",
    "declared_extension_source",
    "resolve_extension_source",
]
