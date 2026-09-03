"""Stable platform contract for the LayerPulse single-checkpoint runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .modeling.contracts import ModelSpec

LAYERPULSE_TASK_ID = "layerpulse"
LAYERPULSE_MODEL_ID = "layerpulse_geochronograph_f3x200cf"
LAYERPULSE_CONFIG_RELATIVE_PATH = Path("configs/layerpulse.yaml")
LAYERPULSE_CONFIG_SCHEMA = "well-seismic.layerpulse-platform-config.v1"
LAYERPULSE_REQUEST_SCHEMA = "well-seismic.layerpulse-inference-request.v1"
LAYERPULSE_CHILD_RESULT_SCHEMA = "well-seismic.layerpulse-child-result.v1"


@dataclass(frozen=True)
class LayerPulseOutputSpec:
    output_key: str
    artifact_key: str
    kind: str
    channels: int
    class_names: tuple[str, ...] = ()

    @property
    def logits_key(self) -> str | None:
        return self.output_key if self.kind == "classification" else None


LAYERPULSE_OUTPUT_SPECS = (
    LayerPulseOutputSpec(
        "fault_logits", "fault_argmax_npy", "classification", 2, ("background", "fault")
    ),
    LayerPulseOutputSpec(
        "unconformity_logits",
        "unconformity_argmax_npy",
        "classification",
        2,
        ("background", "unconformity"),
    ),
    LayerPulseOutputSpec(
        "facies_logits",
        "facies_argmax_npy",
        "classification",
        7,
        (
            "background",
            "upper_ns",
            "middle_ns",
            "lower_ns",
            "rijnland_chalk",
            "scruff",
            "zechstein",
        ),
    ),
    LayerPulseOutputSpec(
        "channel_logits",
        "channel_argmax_npy",
        "classification",
        5,
        ("background", "channel_1", "channel_2", "channel_3", "channel_4"),
    ),
    LayerPulseOutputSpec(
        "karst_logits", "karst_argmax_npy", "classification", 2, ("background", "karst")
    ),
    LayerPulseOutputSpec("rgt", "rgt_npy", "regression", 1),
    LayerPulseOutputSpec("impedance", "impedance_npy", "regression", 1),
    LayerPulseOutputSpec("porosity", "porosity_npy", "regression", 1),
    LayerPulseOutputSpec("well_match", "well_match_npy", "regression", 1),
    LayerPulseOutputSpec(
        "connectivity_logits",
        "connectivity_argmax_npy",
        "classification",
        2,
        ("background", "connected"),
    ),
    LayerPulseOutputSpec("uncertainty", "uncertainty_npy", "regression", 1),
)

LAYERPULSE_CLASSIFICATION_SPECS = tuple(
    spec for spec in LAYERPULSE_OUTPUT_SPECS if spec.kind == "classification"
)
LAYERPULSE_REGRESSION_SPECS = tuple(
    spec for spec in LAYERPULSE_OUTPUT_SPECS if spec.kind == "regression"
)
LAYERPULSE_REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        *(spec.artifact_key for spec in LAYERPULSE_OUTPUT_SPECS),
        "complete_logits_npz",
        "manifest_json",
        "receipt_json",
    }
)

# The API is allowed to control inference geometry and execution device. It is
# not allowed to select executable code, a checkpoint, or a training/evaluation
# input that changes the deployed forward contract.
_SERVER_OWNED_OPTION_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "config",
        "config_path",
        "delivery_config",
        "delivery_config_path",
        "layerpulse_root",
        "layerpulse_project_root",
        "python",
        "python_executable",
        "runtime_python",
        "script",
        "script_path",
        "platform_script",
    }
)
_FORBIDDEN_FORWARD_OPTION_KEYS = frozenset(
    {
        "td_teacher",
        "time_depth",
        "time_depth_table",
        "time_depth_path",
        "checkshot",
        "checkshot_path",
        "vsp",
        "vsp_path",
        "velocity_model",
        "velocity_model_path",
        "t0",
    }
)

# These exact top-level containers are replaced by the API with content read
# from a sealed SourceSnapshot/PreparedView before the model adapter runs.  A
# prepared view may inventory a checkshot or time-depth asset for lineage even
# though LayerPulse never forwards that asset to its model.  Keep this an exact
# allow-list (rather than accepting ``prepared_view_*`` prefixes) so a client
# cannot hide forward controls in a look-alike namespace.
_SERVER_INJECTED_CONTEXT_OPTION_KEYS = frozenset(
    {
        "snapshot_assets",
        "snapshot_metadata_detection",
        "source_snapshot_semantics",
        "source_snapshot_segy_geometry_receipts",
        "prediction_source_identity",
        "prepared_view_artifacts",
        "prepared_view_artifacts_by_role",
    }
)


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {}, ())


def _walk_option_keys(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().casefold()
            qualified = f"{prefix}.{key}" if prefix else key
            found.append((qualified, item))
            found.extend(_walk_option_keys(item, qualified))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_option_keys(item, f"{prefix}[{index}]"))
    return found


def validate_layerpulse_request_options(options: Mapping[str, Any] | None) -> None:
    """Reject external attempts to alter the fixed deployment/forward contract."""

    violations: list[str] = []
    # Sealed lineage is injected by the server and may legitimately describe
    # excluded time-depth/checkshot assets. Those records are inventory only
    # and are never forwarded to the model, so only request-control namespaces
    # participate in override validation.
    control_options = {
        key: value
        for key, value in (options or {}).items()
        if str(key).strip().casefold()
        not in _SERVER_INJECTED_CONTEXT_OPTION_KEYS
    }
    for qualified, value in _walk_option_keys(control_options):
        leaf = qualified.rsplit(".", 1)[-1].split("[", 1)[0]
        if leaf.startswith("_layerpulse_"):
            violations.append(qualified)
        elif leaf in _SERVER_OWNED_OPTION_KEYS and _nonempty(value):
            violations.append(qualified)
        elif leaf in _FORBIDDEN_FORWARD_OPTION_KEYS and _nonempty(value):
            violations.append(qualified)
    if violations:
        raise ValueError(
            "LayerPulse request cannot override server deployment or provide "
            "time-depth supervision: " + ", ".join(sorted(set(violations)))
        )


def load_layerpulse_platform_config(project_root: Path) -> dict[str, Any]:
    """Load and validate the one server-owned LayerPulse deployment config."""

    config_path = (project_root / LAYERPULSE_CONFIG_RELATIVE_PATH).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LayerPulse platform config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    layerpulse = payload.get("layerpulse")
    if not isinstance(layerpulse, Mapping):
        raise TypeError("configs/layerpulse.yaml must contain a layerpulse mapping")
    config = dict(layerpulse)
    if config.get("schema_version") != LAYERPULSE_CONFIG_SCHEMA:
        raise ValueError("LayerPulse platform config schema_version is incompatible")
    if config.get("task_id") != LAYERPULSE_TASK_ID:
        raise ValueError("LayerPulse platform config task_id drifted")
    if config.get("model_id") != LAYERPULSE_MODEL_ID:
        raise ValueError("LayerPulse platform config model_id drifted")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise TypeError("LayerPulse platform runtime config is missing")
    if runtime.get("request_schema") != LAYERPULSE_REQUEST_SCHEMA:
        raise ValueError("LayerPulse request schema drifted")
    if runtime.get("result_schema") != LAYERPULSE_CHILD_RESULT_SCHEMA:
        raise ValueError("LayerPulse child result schema drifted")
    config["_config_path"] = str(config_path)
    return config


def layerpulse_model_spec() -> ModelSpec:
    """Return the public registry declaration without importing the model."""

    disabled_for_portable_package = os.getenv(
        "WELLFUSE_DISABLE_LAYERPULSE", ""
    ).strip().casefold() in {"1", "true", "yes", "on"}

    return ModelSpec(
        id=LAYERPULSE_MODEL_ID,
        name="LayerPulse 智能解释（单 checkpoint 多任务基础模型）",
        category="地震—测井多模态基础模型",
        status="预览子体推理已接入",
        description=(
            "以相对地质时间和构造连通传播为核心，一次共享 Backbone forward "
            "同时输出断层、不整合、地震相、河道、岩溶、RGT、阻抗、孔隙度、"
            "井震匹配、连通性与不确定性。"
        ),
        inputs=(
            "三维后叠加 SEG-Y",
            "同一 SourceSnapshot 的 Registration V3、PreparedView、登记井曲线与完整 MD 轨迹（无时深表）",
        ),
        outputs=("11任务确定性预览子体", "完整分类 logits 与直接 argmax", "连续属性场与预览图集"),
        version="F3X200CF",
        implementation="server-owned-config:configs/layerpulse.yaml",
        scientific_status="candidate",
        runtime_status=(
            "unavailable" if disabled_for_portable_package else "runnable"
        ),
        evidence_class="single_checkpoint_multitask_candidate",
        warnings=(
            "平台接入执行确定性预览子体（默认 128×128×128；可用融合井轨迹优先作为空间锚点）；全体积滑窗仍为 planned_not_executed。",
            "在线推理必须绑定同一数据快照的 Registration V3 与 PreparedView；时深表、checkshot、VSP、速度模型和 t0 只可作为资产清单证据，不进入最终 forward。",
            "未知工区结果属于迁移候选，不冒充目标工区定量验证。",
        ),
        metadata={
            "prediction_task": LAYERPULSE_TASK_ID,
            "config": LAYERPULSE_CONFIG_RELATIVE_PATH.as_posix(),
            "requires_seismic": True,
            "requires_registration": True,
            "requires_complete_trajectory": True,
            "registration_policy": "required",
            "prepared_view_policy": "required",
            "prepared_view_consumed": True,
            "prepared_view_input_contract": "prepared-view-md-trajectory-no-td.v1",
            "supports_raw_wells": False,
            "supports_snapshot_wells": True,
            "snapshot_well_depth_domain": "MD_M",
            "missing_curve_policy": "masked",
            "horizontal_well_policy": "ordered_md_trajectory_attention",
            "time_depth_supervision_is_model_input": False,
            "teacher_required_at_forward": False,
            "parameter_count": 174_697_519,
            "f_final_channels": 96,
            "head_count": 11,
            "head_input": "shared_F_final_only",
            "classification_selection": "complete_logits_direct_argmax_dim1",
            "inference_scope": "preview_patch",
            "default_patch_size_tix": [128, 128, 128],
            "full_volume_status": "planned_not_executed",
            "public_prediction_enabled": not disabled_for_portable_package,
            "portable_runtime_status": (
                "external_assets_required"
                if disabled_for_portable_package
                else "configured_by_server"
            ),
        },
    )


__all__ = [
    "LAYERPULSE_CHILD_RESULT_SCHEMA",
    "LAYERPULSE_CLASSIFICATION_SPECS",
    "LAYERPULSE_CONFIG_RELATIVE_PATH",
    "LAYERPULSE_CONFIG_SCHEMA",
    "LAYERPULSE_MODEL_ID",
    "LAYERPULSE_OUTPUT_SPECS",
    "LAYERPULSE_REGRESSION_SPECS",
    "LAYERPULSE_REQUEST_SCHEMA",
    "LAYERPULSE_REQUIRED_ARTIFACT_KEYS",
    "LAYERPULSE_TASK_ID",
    "LayerPulseOutputSpec",
    "layerpulse_model_spec",
    "load_layerpulse_platform_config",
    "validate_layerpulse_request_options",
]
