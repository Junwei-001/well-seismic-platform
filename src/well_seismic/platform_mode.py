"""Runtime mode switches for the standalone platform shell.

The extracted platform deliberately ships without task-model checkpoints.  A
single environment switch keeps that boundary explicit: model/task contracts
remain visible to the UI and API, while external model plugins and prediction
workers stay disabled until a later extension package is installed.

The default remains the historical full-runtime behaviour so this module does
not change the source platform when it is copied back into another checkout.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any


MODEL_MODE_ENV = "WELLFUSE_MODEL_MODE"
DISABLE_TASK_MODELS_ENV = "WELLFUSE_DISABLE_TASK_MODELS"
INTERFACES_ONLY_MODE = "interfaces_only"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_INTERFACE_ONLY_VALUES = frozenset(
    {
        "interface_only",
        "interfaces_only",
        "no_weights",
        "skeleton",
        "shell",
        "disabled",
    }
)
_PACKAGE_MARKER = (
    Path(__file__).resolve().parents[2] / "models" / "INTERFACE_ONLY"
)


def _is_true(value: object) -> bool:
    return str(value or "").strip().casefold() in _TRUE_VALUES


def interface_only_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the process must expose contracts without model runtimes."""

    values = os.environ if environ is None else environ
    mode = str(values.get(MODEL_MODE_ENV, "")).strip().casefold()
    return (
        mode in _INTERFACE_ONLY_VALUES
        or _is_true(values.get(DISABLE_TASK_MODELS_ENV))
        or _PACKAGE_MARKER.is_file()
    )


def annotate_interface_only_spec(spec: Any) -> Any:
    """Mark a ``ModelSpec`` as waiting for an external adapter/checkpoint.

    Failed/blocked research records remain blocked.  A pure fusion baseline
    can still be used for data-contract checks; all other runnable or
    precomputed model declarations become explicit extension points.
    """

    runtime_status = str(getattr(spec, "runtime_status", "") or "")
    model_id = str(getattr(spec, "id", "") or "")
    if runtime_status in {"blocked", "failed", "precomputed_only", "unavailable"}:
        return spec
    # The confidence-gated fusion implementation is deterministic and has no
    # task checkpoint.  Keep it available for contract-level smoke tests.
    if model_id == "confidence_gated_fusion":
        return spec
    metadata = dict(getattr(spec, "metadata", {}) or {})
    metadata.update(
        {
            "interface_only": True,
            "weights_attached": False,
            "public_prediction_enabled": False,
            "runtime_mode": INTERFACES_ONLY_MODE,
            "deferred_extension_root": "models/task-models",
        }
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *tuple(getattr(spec, "warnings", ()) or ()),
                "独立接口版未携带任务模型权重；请通过外部插件/适配器接入后再运行。",
            )
        )
    )
    return replace(
        spec,
        status="接口保留·等待外部模型适配",
        runtime_status="adapter_required",
        metadata=metadata,
        warnings=warnings,
    )


def interface_only_release(release: Any) -> Any:
    """Project a release declaration onto the no-weight runtime boundary."""

    runtime_status = str(getattr(release, "runtime_status", "") or "")
    if runtime_status in {"blocked", "failed", "precomputed_only", "unavailable"}:
        return release
    metadata = dict(getattr(release, "metadata", {}) or {})
    metadata.update(
        {
            "interface_only": True,
            "weights_attached": False,
            "public_prediction_enabled": False,
            "runtime_mode": INTERFACES_ONLY_MODE,
            "deferred_extension_root": "models/task-models",
        }
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *tuple(getattr(release, "warnings", ()) or ()),
                "独立接口版仅保留发布合同；任务权重/运行时需由外部扩展提供。",
            )
        )
    )
    return replace(
        release,
        runtime_status="adapter_required",
        metadata=metadata,
        warnings=warnings,
    )


__all__ = [
    "DISABLE_TASK_MODELS_ENV",
    "INTERFACES_ONLY_MODE",
    "MODEL_MODE_ENV",
    "annotate_interface_only_spec",
    "interface_only_enabled",
    "interface_only_release",
]
