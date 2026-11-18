"""Entry-point objects: ``ModelSpec`` registration and input adapters.

Every entry point is named after its ``model_id`` (the platform requires this
for prediction runners; the other groups follow the same rule for symmetry).
Registration verifies the payload hashes and the isolated interpreter first, so
a tampered or missing payload surfaces as a ``plugin_load_errors`` entry and the
model never becomes selectable.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from well_seismic.modeling.contracts import ModelSpec
from well_seismic.modeling.input_adapters import ModelInputBatch, ModelInputRequest

from .catalog import (
    FAULT_AXES,
    FAULT_MODEL_ID,
    MODEL_VERSIONS,
    PROPERTY_MODEL_IDS,
    RECONSTRUCTION_MODEL_ID,
    validate_fault_request,
    validate_property_request,
    validate_reconstruction_request,
)
from .manifest import PLATFORM_ROOT, model_dir_for, read_manifest, verify_payload

MODEL_PYTHON_ENV = "WELL_SEISMIC_SIXTRACK_PYTHON"


def model_python(env_name: str = MODEL_PYTHON_ENV) -> Path:
    """Absolute path of the isolated model interpreter; fail closed when absent."""

    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is not set; the six-track models cannot run")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"{env_name} must be an absolute executable path: {raw}")
    return path


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PLATFORM_ROOT).as_posix()


def build_spec(model_id: str) -> ModelSpec:
    from .catalog import LITHOFACIES_MODEL_ID, SWEETSPOT_MODEL_ID
    if model_id == SWEETSPOT_MODEL_ID:
        from .sweetspot_plugin import build_spec as build_sweetspot_spec
        return build_sweetspot_spec()
    if model_id == LITHOFACIES_MODEL_ID:
        from .lithofacies_plugin import build_spec as build_lithofacies_spec
        return build_lithofacies_spec()
    model_dir = model_dir_for(model_id, MODEL_VERSIONS[model_id])
    manifest = read_manifest(model_dir)
    if manifest["model_id"] != model_id or manifest["version"] != MODEL_VERSIONS[model_id]:
        raise RuntimeError(f"manifest identity drift in {model_dir}")
    payload = {entry["role"]: entry for entry in manifest["payload"]}
    common = {
        "prediction_task": manifest["task_id"],
        "registration_policy": "none",
        "prepared_view_policy": "none",
        "public_prediction_enabled": True,
        "weights_attached": True,
        "extension_package": "well-seismic-sixtrack-pilot",
        "payload_manifest": _project_relative(model_dir / "manifest.json"),
        "payload_version": manifest["version"],
        "payload_sha256": {role: entry["sha256"] for role, entry in payload.items()},
        "evidence": {
            key: {
                name: value
                for name, value in record.items()
                if name in {"path", "sha256", "status", "stable_champion", "operational_reference"}
            }
            if isinstance(record, Mapping)
            else record
            for key, record in manifest["evidence"].items()
        },
        "license": manifest["license"],
        "applicability": manifest["applicability"],
        "claim_boundary": manifest["claim_boundary"],
        "advanced_options": manifest.get("advanced_options", []),
        "execution": "isolated_subprocess_cpu",
        "submission_channel": manifest["submission_channel"],
    }
    if model_id == FAULT_MODEL_ID:
        return ModelSpec(
            id=model_id,
            name=manifest["display_name"],
            category="地震断层·六赛道扩展",
            status="轻量基线·仅开发集验证",
            description=(
                "P78 断层稳定冠军确认未产生新冠军（SwinUNETR 阈值稳定门未过），保留的 operational "
                "reference：逐 inline 切片 5 个局部特征 + SGD logistic。按 P78 参考做法直接作用于 "
                "SEG-Y 原始振幅（训练时用的是 z-score 后的 ST0202），输出断层概率体与阈值掩码。"
                "精度很低（测试 F1≈0.016），只作可运行基线；需显式给出体块。"
                "授权工作台先上传并封存 SEG-Y，再提交封存编号；本地独立 API 使用显式路径。"
            ),
            inputs=("三维叠后 SEG-Y，原始振幅 float32 [Z, INLINE, CROSSLINE]",),
            outputs=("断层概率体 (npy)", "阈值断层掩码 (npy)", "网格坐标 (json)"),
            version=manifest["version"],
            implementation="wellseismic_sixtrack_pilot.worker.fault_logistic (isolated CPU subprocess)",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="p78_verified_retain_incumbent_development_only",
            warnings=(
                "轻量基线/仅开发集验证：audited_v2 测试 precision≈0.008、F1≈0.016，掩码正类比例偏高，不能当作生产断层解释。",
                "训练用 z-score 后的 ST0202R08；推理按 P78 参考直接用 ST10010 原始振幅，已在训练尺度之外（P30 盒掩码正类约 79%）。",
                "授权工作台须先上传并封存 SEG-Y，再用 input_bundle_id 和体块选项提交；未封存输入会被拒绝。独立本地 API 的显式 seismic_path 不能传给授权桥接入口。",
            ),
            metadata={
                **common,
                "requires_seismic": True,
                "competition_track_id": "fault",
                "input_mode": "segy_volume",
                "supports_new_area": False,
                "axis_order": list(FAULT_AXES),
                "default_threshold": 0.4505760073661804,
            },
        )
    if model_id == RECONSTRUCTION_MODEL_ID:
        index = payload["development_fold_index"]
        return ModelSpec(
            id=model_id, name=manifest["display_name"], category="三维重建·六赛道扩展",
            status="P21开发集验证·档案绑定",
            description="固定三核条件重建：选择原始开发折，使用512个训练支撑标签、已准备特征及缓存PyKrige基线输出2048个查询预测。",
            inputs=("options.fold_id：原始开发折0..4",),
            outputs=("查询点预测与三核成员NPZ", "输入输出哈希与开发折溯源回执"),
            version=manifest["version"],
            implementation="wellseismic_sixtrack_pilot.worker.reconstruction_request (isolated CPU subprocess)",
            scientific_status="conditional", runtime_status="runnable",
            evidence_class="p21_conditional_development_exact_numerical_parity",
            warnings=(manifest["claim_boundary"], "许可审查待完成，未统一发布。"),
            metadata={**common, "requires_seismic": False, "input_mode": "dataset_bound",
                      "supports_raw_wells": False, "supports_new_area": False,
                      "extension_source": {"kind": "registered_dataset", "path": _project_relative(model_dir / index["path"]), "sha256": index["sha256"]}},
        )
    target = PROPERTY_MODEL_IDS[model_id]
    archive = payload["development_archive"]
    return ModelSpec(
        id=model_id,
        name=manifest["display_name"],
        category="井侧物性·六赛道扩展",
        status="P67稳定冠军·仅开发集验证·档案绑定",
        description=(
            f"P67 确认的物性稳定 development 冠军：冻结 TabICL（in-context，不训练权重）预测 {target}。"
            "档案绑定模式：只能预测已封装开发档案中的井族，上下文取其余三个开发井族的 192 行，"
            "标准化只在上下文上拟合；15/9-F-15 冻结测试井永不进入。"
        ),
        inputs=("档案绑定：options.query_family（15/9-19、15/9-F-1、15/9-F-11、15/9-F-12 之一）",),
        outputs=(f"{target} 逐样本预测 CSV/NPZ（含 CPI 参考标签与误差指标）",),
        version=manifest["version"],
        implementation="wellseismic_sixtrack_pilot.worker.property_tabicl (isolated CPU subprocess)",
        scientific_status="conditional",
        runtime_status="runnable",
        evidence_class="p67_three_seed_logo4_oof_development_only",
        warnings=(
            "仅开发集 LOGO4 验证，frozen test 未读取；参考标签是 CPI 解释值而非岩心。",
            "CPU 推理，与 P67 的 GPU(AMP) 结果有小幅数值漂移。",
            "当前只支持档案绑定输入；对新井预测需要“全量特征构建”进阶能力（未启用，见 advanced_options）。",
            "授权工作台与本地独立 API 均可提交已登记的 options.query_family；授权桥接入口不接受用户路径或未登记的井族。",
        ),
        metadata={
            **common,
            "requires_seismic": False,
            "competition_track_id": "reservoir_property",
            "supports_raw_wells": False,
            "supports_new_area": False,
            "target_curve": target,
            "unit": manifest["unit"],
            "input_mode": "dataset_bound",
            "extension_source": {
                "kind": "registered_dataset",
                "path": _project_relative(model_dir / archive["path"]),
                "sha256": archive["sha256"],
            },
        },
    )


def _registrar(model_id: str) -> Callable[[Any], None]:
    def register(registry: Any) -> None:
        model_python()
        verify_payload(model_dir_for(model_id, MODEL_VERSIONS[model_id]))
        registry.register(build_spec(model_id))

    register.__name__ = f"register_{model_id}"
    return register


class _DeferredAdapter:
    """The tensors are built inside the isolated worker; the adapter only validates."""

    model_id: str
    validator: Callable[[Mapping[str, Any]], None]

    def capabilities(self) -> dict[str, Any]:
        raise NotImplementedError

    def validate_prediction_request(self, request: Mapping[str, Any]) -> None:
        self.validator(request)

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=None,
            axes=tuple(self.capabilities().get("array_axes", ())),
            provenance={"prepared_by": "isolated worker", "source": str(request.source)},
        )


class FaultInputAdapter(_DeferredAdapter):
    model_id = FAULT_MODEL_ID
    validator = staticmethod(validate_fault_request)

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter": type(self).__name__,
            "source_formats": ["segy", "sgy"],
            "array_axes": list(FAULT_AXES),
            "dtype": "float32",
            "normalization": "none (raw amplitude)",
            "patch_size": [],
            "requires_seismic": True,
            "requires_logs": False,
            "box_options": ["inline_range", "crossline_range", "sample_range", "crop_start/crop_size"],
            "device": "cpu",
        }


class PropertyInputAdapter(_DeferredAdapter):
    validator = staticmethod(validate_property_request)

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter": type(self).__name__,
            "source_formats": ["registered_dataset"],
            "array_axes": ["SAMPLE", "FEATURE"],
            "feature_count": 153,
            "patch_size": [],
            "requires_seismic": False,
            "requires_logs": True,
            "supports_dataset_bound": True,
            "supports_raw_wells": False,
            "supports_snapshot_wells": False,
            "target": PROPERTY_MODEL_IDS[self.model_id],
            "device": "cpu",
        }


class ReconstructionInputAdapter(_DeferredAdapter):
    model_id = RECONSTRUCTION_MODEL_ID
    validator = staticmethod(validate_reconstruction_request)

    def capabilities(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "adapter": type(self).__name__,
                "source_formats": ["registered_dataset"], "array_axes": ["QUERY"],
                "requires_seismic": False, "requires_logs": False, "patch_size": [],
                "supports_dataset_bound": True, "supports_raw_wells": False,
                "supports_new_area": False, "options": {"fold_id": [0, 1, 2, 3, 4]}, "device": "cpu"}


def _adapter_factory(model_id: str) -> Callable[[Any], _DeferredAdapter]:
    def factory(config: Any) -> _DeferredAdapter:
        del config
        if model_id == RECONSTRUCTION_MODEL_ID:
            return ReconstructionInputAdapter()
        return FaultInputAdapter() if model_id == FAULT_MODEL_ID else PropertyInputAdapter(model_id)

    factory.__name__ = f"adapter_{model_id}"
    return factory


register_sixtrack_fault_local_logistic_audited_v2 = _registrar(FAULT_MODEL_ID)
register_sixtrack_property_tabicl_phif = _registrar("sixtrack_property_tabicl_phif")
register_sixtrack_property_tabicl_klogh = _registrar("sixtrack_property_tabicl_klogh")
register_sixtrack_property_tabicl_sw = _registrar("sixtrack_property_tabicl_sw")

adapter_sixtrack_fault_local_logistic_audited_v2 = _adapter_factory(FAULT_MODEL_ID)
adapter_sixtrack_property_tabicl_phif = _adapter_factory("sixtrack_property_tabicl_phif")
adapter_sixtrack_property_tabicl_klogh = _adapter_factory("sixtrack_property_tabicl_klogh")
adapter_sixtrack_property_tabicl_sw = _adapter_factory("sixtrack_property_tabicl_sw")
register_sixtrack_reconstruction_p21 = _registrar(RECONSTRUCTION_MODEL_ID)
adapter_sixtrack_reconstruction_p21 = _adapter_factory(RECONSTRUCTION_MODEL_ID)


def register_reconstruction_task(registry: Any) -> None:
    from well_seismic.interpretation.contracts import InterpretationTaskSpec
    registry.register(InterpretationTaskSpec(
        id="volume_reconstruction", name="三维物性重建", short_name="三维重建",
        description="在明确的井约束和地震特征下预测三维物性；当前P21只支持原始开发档案折。",
        outputs=("查询点物性预测", "条件重建来源回执"),
        required_modalities=("已登记P21开发折",), evaluation_metrics=("开发集RMSE", "数值复现一致性"),
        order=70, group="reconstruction", evidence_policy="conditional_development",
    ))
