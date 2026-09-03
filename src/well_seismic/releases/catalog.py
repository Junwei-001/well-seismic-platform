"""Explicit, fail-safe discovery of frozen WellFuse and legacy releases.

No recursive filesystem discovery is performed here.  Every exposed artifact
comes from the declarations below, except Align per-well products where only
immediate well directories and three safe filenames are inspected.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .contracts import ArtifactRelease, ModelRelease, ReleaseArtifact
from ..platform_mode import interface_only_enabled, interface_only_release
from .lifecycle_overlay import LifecycleOverlayError, LifecycleRegistryOverlay

_SAFE_ALIGN_FILES = {
    "manifest.json": ("manifest", "report", "application/json"),
    "time_depth.csv": ("registration_table", "well_curve", "text/csv"),
    "time_depth.las": ("registration_las", "well_curve", "application/x-las"),
}
_DISALLOWED_PARTS = {
    "labels",
    "label",
    "cache",
    "pytest",
    ".pytest_cache",
    "tmp",
    "temp",
}
_HASH_LIMIT_BYTES = 64 * 1024 * 1024
_EXPLICIT_ARTIFACT_INTEGRITY: dict[str, dict[str, Any]] = {
    (
        "fracture_depthblock_real/v2_mdcal_5fold_100ep/"
        "utah_forge__16B78-32/best_model.pt"
    ): {
        "size_bytes": 905973,
        "sha256": "24f1c907101468a0f6d0934b211e69dca8c1ac2777a716d33f7c4b20f2a2d7c1",
    },
    (
        "fracture_depthblock_real/v2_mdcal_5fold_100ep/"
        "utah_forge__16B78-32/model_selection.json"
    ): {
        "size_bytes": 1542,
        "sha256": "0ec962a71fa64a4df2a2b3f501be9cd530cc5e77bc58aa428c50be5e309fae97",
    },
}
_EXPLICIT_WELLFUSE_PROJECT_INTEGRITY: dict[str, dict[str, Any]] = {
    "artifacts/p16_1/align/p13_ensemble_manifest.json": {
        "size_bytes": 11823,
        "sha256": "2696e8adde89c01ad38caec9d08a6b7d7c34dd61f4327f87d2971a84d1b2096d",
    },
    (
        "artifacts/wellfuse_12b_direct/f5_final_fit_v1/"
        "production_anchor_cpu_preflight.json"
    ): {
        "size_bytes": 15403,
        "sha256": "0266923c49753e2292240108b4cbf7df7c7af4051b91923f6c02364f22d94167",
    },
    (
        "artifacts/wellfuse_12b_direct/f4_task_oof_v2/facies_1d/"
        "three_seed_oof/completion.json"
    ): {
        "size_bytes": 28898,
        "sha256": "b361dc2c30ba63f8dfab08cb133a431009eb63d54fb0f1a575c5c11248d77322",
    },
    (
        "artifacts/wellfuse_12b_direct/f4_task_oof_v2/den/"
        "three_seed_oof/completion.json"
    ): {
        "size_bytes": 93318,
        "sha256": "38d8c1122159e78c2f6f37f953a502da0590888f500120c58e0f0dac6b99cdf6",
    },
}
_EXPLICIT_DIRECT12B_PUBLIC_DATA_INTEGRITY: dict[str, dict[str, Any]] = {
    (
        "11_direct12b_cache/raw_prepare_smoke/"
        "penobscot_canada_b41_v1/raw_prepare_receipt_v1.json"
    ): {
        "size_bytes": 7747,
        "sha256": "9b61eec3ea2c6ae06502ce59b0399c0bf24d2c2a87a6fa393283518778085ba8",
    },
}

# This table binds public compatibility ids to immutable lifecycle candidate
# ids.  It contains no status or promotion decision: those are read solely
# from artifacts/lifecycle/registry.json.  Models absent from this table remain
# adapter/capability fallbacks and cannot inherit a scientific claim from the
# static catalog.
_LIFECYCLE_RELEASE_BINDINGS: dict[str, tuple[str, str]] = {
    "wellfuse_align_p13_chengdu": ("wellfuse_align", "p13-scientific-ensemble-v1"),
    "wellfuse_p17_horizon": ("chengdu_horizon", "p17-horizon-event-ensemble-v1"),
    "wellfuse_p17_facies_3d": (
        "chengdu_facies",
        "p17-facies-3d-existing-candidate-v1",
    ),
    "wellfuse_p17_fault_failed": (
        "chengdu_fault",
        "p17-fault-pu-ensemble-failed-v1",
    ),
    "wellfuse_p17_channel": (
        "synthetic_channel_geobody",
        "p17-channel-topology-candidate-v1",
    ),
    "wellfuse_p17_karst": (
        "synthetic_karst_geobody",
        "p17-karst-topology-candidate-v1",
    ),
}


@dataclass(frozen=True)
class _AssetSpec:
    id: str
    name: str
    role: str
    kind: str
    root: str
    relative_path: str
    layer: str | None = None
    media_type: str | None = None
    unit: str | None = None
    axis_order: str | None = None
    uncertainty_definition: str | None = None


@dataclass(frozen=True)
class _ReleaseSpec:
    id: str
    name: str
    version: str
    task_id: str
    description: str
    scientific_status: str
    runtime_status: str
    evidence_class: str
    scope: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    assets: tuple[_AssetSpec, ...] = ()
    model_id: str | None = None
    runner_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _asset(
    path: str,
    *,
    root: str,
    role: str,
    layer: str | None = None,
    kind: str = "file",
    name: str | None = None,
    unit: str | None = None,
    axis_order: str | None = None,
    uncertainty_definition: str | None = None,
) -> _AssetSpec:
    safe_id = path.replace("\\", "/").replace("/", ".").replace(" ", "_").lower()
    return _AssetSpec(
        id=safe_id,
        name=name or Path(path).name,
        role=role,
        kind=kind,
        root=root,
        relative_path=path.replace("\\", "/"),
        layer=layer,
        unit=unit,
        axis_order=axis_order,
        uncertainty_definition=uncertainty_definition,
    )


def _well_release(
    curve: str,
    *,
    scientific_status: str,
    warning: str = "",
) -> _ReleaseSpec:
    warnings = (warning,) if warning else ()
    return _ReleaseSpec(
        id=f"wellfuse_p18_{curve.lower()}",
        name={
            "DEN": "历史封存井曲线→DEN只读成果",
            "POR": "历史封存井曲线→POR只读成果",
            "LOG_PERM": "历史封存井曲线→LOG_PERM只读成果",
            "SW": "历史封存井曲线→SW只读成果",
            "VSH": "历史封存井曲线→VSH只读成果",
        }[curve],
        version="p18-oof-v2",
        task_id="well_property",
        description=f"成都整井空间外折 {curve} 预测证据与冻结成果。",
        scientific_status=scientific_status,
        runtime_status="runnable",
        evidence_class="three_seed_six_spatial_whole_well_oof",
        scope=("chengdu", "vertical", "deviated", "horizontal"),
        warnings=warnings
        + ("当前发布是 OOF 科学证据，不可任选单个外折 checkpoint 作为新井部署模型。",),
        assets=(
            _asset(
                "SCIENTIFIC_DECISION.json",
                root="p18",
                role="scientific_decision",
                layer="report",
            ),
            _asset(
                f"chengdu_well_oof_v2/{curve.lower()}/three_seed_oof/completion.json",
                root="p18",
                role="oof_completion",
                layer="report",
            ),
        ),
        model_id=f"wellfuse_{curve.lower()}_p18",
        runner_id=f"wellfuse_{curve.lower()}_p18",
        metadata={
            "target_curve": curve,
            "public_prediction_enabled": False,
            "public_archived": True,
            "unknown_survey_runtime_status": "experimental_runnable",
            "baseline_fallback_discloses_model_executed_false": True,
            "requires_registration": True,
        },
    )


def _fast_fluid_release() -> _ReleaseSpec:
    return _ReleaseSpec(
        id="wellfuse_fluid_interpretation_fast",
        name="五类流体解释（登记九线LAS→连续确定流体层段）",
        version="fixed-last-northwest12-chengdu10-v1",
        task_id="fluid_interpretation",
        description=(
            "登记数据集内的五类整井流体解释：西北使用 well_only，"
            "成都使用 seismic_fusion fixed-last 最终权重；唯一业务成果为"
            "连续MD确定流体层段CSV，概率只在进程内参与判别。"
        ),
        scientific_status="validated",
        runtime_status="runnable",
        evidence_class="fixed_last_three_fold_complete_well_oof",
        scope=(
            "within_dataset",
            "northwest_all",
            "northwest_oil",
            "northwest_coal",
            "chengdu",
        ),
        warnings=(
            "仅支持已登记的西北和成都数据资产；未验证未知工区或跨工区泛化。",
            "展示指标来自整井外折；在线CSV由fixed-last最终模型生成，不是所选井的OOF预测。",
            "成都默认seismic_fusion，西北仅well_only；两套指标不可视为同一数据域。",
        ),
        assets=(
            _asset(
                "fluid_interpretation/northwest_all/well_only_fixed_last/metrics.json",
                root="artifacts",
                role="northwest_oof_metrics",
                layer="report",
            ),
            _asset(
                "fluid_interpretation/northwest_all/well_only_fixed_last/best.pt",
                root="artifacts",
                role="northwest_final_checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "fluid_interpretation/chengdu/seismic_fusion_fixed_last/metrics.json",
                root="artifacts",
                role="chengdu_oof_metrics",
                layer="report",
            ),
            _asset(
                "fluid_interpretation/chengdu/seismic_fusion_fixed_last/best.pt",
                root="artifacts",
                role="chengdu_final_checkpoint",
                kind="checkpoint",
            ),
        ),
        model_id="wellfuse_fluid_interpretation_fast",
        runner_id="wellfuse_fluid_interpretation_fast",
        metadata={
            "scope": "within_dataset",
            "selection": "fixed_last",
            "datasets": [
                "northwest_all",
                "northwest_oil",
                "northwest_coal",
                "chengdu",
            ],
            "features_by_dataset": {
                "northwest_all": "well_only",
                "northwest_oil": "well_only",
                "northwest_coal": "well_only",
                "chengdu": "seismic_fusion",
            },
            "validation_protocol": "three_outer_fold_complete_well_oof",
            "runtime_prediction_semantics": "fixed_last_final_refit",
            "cross_survey_validated": False,
            "requires_registration": False,
            "validation": {
                "northwest_all": {
                    "epochs": 12,
                    "material_macro_f1": 0.6973426524958883,
                    "hydrocarbon_f1": 0.6837761786655693,
                    "majority_baseline_macro_f1": 0.13623844420591452,
                },
                "chengdu": {
                    "epochs": 10,
                    "material_macro_f1": 0.7361471907285347,
                    "hydrocarbon_f1": 0.8750332005312086,
                    "majority_baseline_macro_f1": 0.282393807006881,
                },
            },
        },
    )


def _fracture_development_release() -> _ReleaseSpec:
    root = "fracture_depthblock_real/v2_mdcal_5fold_100ep/utah_forge__16B78-32"
    return _ReleaseSpec(
        id="wellfuse_fracture_development_utah_fast",
        name="井侧裂缝发育排序（LAS/CSV测井→连续发育层段）",
        version="utah16b-depthblock-v2",
        task_id="fracture_development",
        description=(
            "Utah FORGE 16B常规测井裂缝相对发育排序模型，支持赛事原始井文件推理。"
        ),
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="single_well_purged_depth_block_candidate",
        scope=("utah_forge_16b", "raw_well_transfer_candidate"),
        warnings=(
            "单井连续深度块验证不是跨区块外部验证。",
            "输出为本井内低/中/高相对发育连续深度段，不是精确裂缝点或绝对事件密度。",
        ),
        assets=(
            _asset(
                f"{root}/best_model.pt",
                root="artifacts",
                role="checkpoint",
                kind="checkpoint",
            ),
            _asset(
                f"{root}/model_selection.json",
                root="artifacts",
                role="validation_metrics",
                layer="report",
            ),
        ),
        model_id="wellfuse_fracture_development_utah_fast",
        runner_id="wellfuse_fracture_development_utah_fast",
        metadata={
            "requires_seismic": False,
            "requires_registration": False,
            "supports_raw_wells": True,
            "source_expert": "utah_forge_16b",
            "validation_protocol": "five_purged_contiguous_depth_blocks",
            "macro_fold_spearman": 0.274311,
            "macro_fold_top_quartile_pr_auc_lift": 0.107949,
        },
    )


def _fast_facies_1d_release() -> _ReleaseSpec:
    return _ReleaseSpec(
        id="wellfuse_facies_1d_chengdu_fast",
        name="登记井侧沉积相分类（九线LAS→确定相序列与层段）",
        version="legacy-viterbi-three-seed-v1",
        task_id="facies_1d",
        description=(
            "成都46井登记合同内的井侧沉积相序列：冻结三Seed集成判别，"
            "并使用训练折转移矩阵执行一阶Viterbi解码；公开成果仅含确定性连续层段。"
        ),
        scientific_status="validated",
        runtime_status="runnable",
        evidence_class="three_seed_six_spatial_whole_well_oof_viterbi",
        scope=("within_dataset", "chengdu", "46_wells"),
        warnings=(
            "仅验证成都登记数据集内的46口井，不代表未知井或跨工区泛化。",
            "输入是九线测井、完整井轨迹与Align特征，不消费地震振幅分支。",
            "在线整井输出由冻结部署ensemble生成，不等同于所选井的OOF预测。",
        ),
        assets=(
            _asset(
                "facies1d_fast/legacy_viterbi_v1/facies1d_fast_v1.pt",
                root="artifacts",
                role="deployment_checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "facies1d_fast/legacy_viterbi_v1/config.json",
                root="artifacts",
                role="runtime_config",
                layer="report",
            ),
            _asset(
                "facies1d_fast/legacy_viterbi_v1/metrics.json",
                root="artifacts",
                role="oof_metrics",
                layer="report",
            ),
        ),
        model_id="wellfuse_facies_1d_chengdu_fast",
        runner_id="wellfuse_facies_1d_chengdu_fast",
        metadata={
            "scope": "within_dataset",
            "datasets": ["chengdu"],
            "selection": (
                "fixed_three_seed_probability_plus_training_fold_viterbi"
            ),
            "validation_protocol": "three_seed_six_spatial_whole_well_oof",
            "cross_survey_validated": False,
            "requires_seismic": False,
            "requires_registration": False,
            "input_semantics": "nine_logs_trajectory_and_align_without_seismic_amplitude",
            "validation": {
                "well_count": 46,
                "whole_well_decoded_macro_f1": 0.5813523934720161,
                "whole_well_raw_macro_f1": 0.5757918645906513,
            },
        },
    )


def _fast_facies_3d_f3_release() -> _ReleaseSpec:
    return _ReleaseSpec(
        id="wellfuse_facies_3d_f3_fast",
        name="六类三维地震相分割（时间域SEG-Y→离散相体与切片）",
        version="f3-dense-25d-unet-v1",
        task_id="facies_3d",
        description=(
            "公开稠密标签训练的2.5D U-Net；支持三维SEG-Y自动有效ROI和1x1"
            "代表性单道推理，公开输出六类离散相体与分类切片。"
        ),
        scientific_status="conditional",
        runtime_status="runnable",
        evidence_class="real_dense_f3_frozen_test_once",
        scope=("F3_dense_benchmark", "unknown_survey_transfer_candidate"),
        warnings=(
            "定量指标只适用于该公开基准的冻结测试；当前工区输出是跨工区迁移候选。",
            "类别0至5沿用训练数据本体，不自动映射为其他工区井侧相代码。",
            "1x1空间ROI是单条真实地震道的沿TWT分类，不是井侧测井相模型。",
        ),
        assets=(
            _asset(
                "facies3d_fast/f3_dense_25d_unet_tune/f3_facies3d_fast_best.pt",
                root="artifacts",
                role="f3_dense_checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "facies3d_fast/f3_dense_25d_unet_tune/metrics_test_once.json",
                root="artifacts",
                role="f3_frozen_test_metrics",
                layer="report",
            ),
        ),
        model_id="wellfuse_facies_3d_f3_fast",
        runner_id="wellfuse_facies_3d_f3_fast",
        metadata={
            "validated_scope": "F3_dense_benchmark",
            "unknown_survey_runtime_status": "experimental_transfer_candidate",
            "supports_single_trace_roi": True,
            "single_trace_contract": "inline_count=1,crossline_count=1",
            "requires_seismic": True,
            "requires_registration": False,
            "class_codes": [0, 1, 2, 3, 4, 5],
            "validation": {
                "metric_name": "mIoU",
                "miou": 0.6246008984838184,
                "macro_f1": 0.7402113283428177,
                "applies_to_current_survey": False,
            },
        },
    )


def _fast_property_release(
    target: str,
    *,
    epochs: int,
    unit: str,
    mae: float,
    baseline_mae: float,
    r2: float,
) -> _ReleaseSpec:
    run_name = f"northwest_all_fixed_last_{epochs}ep"
    target_root = f"fast_downstream/northwest_property/{run_name}/{target}"
    return _ReleaseSpec(
        id=f"wellfuse_{target.lower()}_northwest_fast",
        name={
            "DEN": "井侧密度预测（登记九线LAS→DEN整井曲线）",
            "POR": "井侧孔隙度预测（登记九线LAS→POR整井曲线）",
            "LOG_PERM": "井侧渗透率预测（登记九线LAS→LOG_PERM整井曲线）",
            "SW": "井侧含水饱和度预测（登记九线LAS→SW整井曲线）",
            "VSH": "井侧泥质含量预测（登记九线LAS→VSH整井曲线）",
        }[target],
        version=f"fixed-last-{epochs}ep-3fold-v1",
        task_id="well_property",
        description=(
            f"登记西北数据集内的 {target} 整井预测，使用三折 fixed-last checkpoint ensemble。"
        ),
        scientific_status="validated",
        runtime_status="runnable",
        evidence_class="fixed_last_three_fold_complete_well_oof",
        scope=(
            "within_dataset",
            "northwest_all",
            "northwest_oil",
            "northwest_coal",
        ),
        warnings=(
            "仅支持已登记的西北数据资产；未验证成都、未知工区或跨工区泛化。",
            "OOF指标来自northwest_all三折整井外折；油区/煤岩气子集推理不是独立外部验证。",
            "在线CSV是三折fixed-last checkpoint ensemble，不等于所选井的OOF预测。",
        ),
        assets=(
            _asset(
                f"fast_downstream/northwest_property/{run_name}/metrics.json",
                root="artifacts",
                role="oof_metrics",
                layer="report",
            ),
            *tuple(
                _asset(
                    f"{target_root}/fold_{fold}/best.pt",
                    root="artifacts",
                    role=f"fold_{fold}_checkpoint",
                    kind="checkpoint",
                )
                for fold in range(3)
            ),
        ),
        model_id=f"wellfuse_{target.lower()}_northwest_fast",
        runner_id=f"wellfuse_{target.lower()}_northwest_fast",
        metadata={
            "scope": "within_dataset",
            "selection": "fixed_last",
            "datasets": ["northwest_all", "northwest_oil", "northwest_coal"],
            "target_curve": target,
            "unit": unit,
            "epochs": epochs,
            "fold_count": 3,
            "validation_protocol": "three_outer_fold_complete_well_oof",
            "runtime_prediction_semantics": "three_fold_fixed_last_ensemble",
            "cross_survey_validated": False,
            "requires_registration": False,
            "validation": {
                "metric_name": "mae",
                "mae": mae,
                "baseline_mae": baseline_mae,
                "r2": r2,
            },
        },
    )


_RELEASE_SPECS: tuple[_ReleaseSpec, ...] = (
    _ReleaseSpec(
        id="legacy_faultseg_3d",
        name="全工区三维断层识别（SEG-Y→断层概率体与掩码）",
        version="full-volume-v1",
        task_id="fault",
        description=(
            "FaultSeg3D 三维 U-Net；以 128³ 子体积遍历原始三维地震体，"
            "在 64³ 重叠区进行概率加权融合，并封存完整工区概率体与断层掩码。"
        ),
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="legacy_external_checkpoint",
        scope=("generic_3d_seismic",),
        warnings=("工程可运行不等同于成都真实断层科学验证通过。",),
        assets=(
            _asset(
                "models/manifest.json",
                root="project",
                role="weight_manifest",
                layer="report",
            ),
            _asset(
                "models/wellfuse/structural/fault/faultseg-best.pt",
                root="project",
                role="checkpoint",
                kind="checkpoint",
            ),
        ),
        model_id="faultseg_3d",
        runner_id="faultseg_3d",
        metadata={
            "legacy": False,
            "inference_scope": "full_volume",
            "patch_size_zyx": [128, 128, 128],
            "overlap_zyx": [64, 64, 64],
            "weighted_probability_blending": True,
        },
    ),
    _ReleaseSpec(
        id="faultnet_china_field",
        name="FaultNet 开源断层识别（国内数据微调候选）",
        version="upstream-torchscript-gamma-0.5-0.7-v1",
        task_id="fault",
        description=(
            "中国石油大学（华东）FaultNet 官方 TorchScript 权重；平台保持"
            "逐窗口 min-max、概率直出与 0.5 掩码阈值，并复用全工区重叠加权重建。"
        ),
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="external_china_field_checkpoint",
        scope=("generic_3d_seismic", "china_field_pretraining_transfer"),
        warnings=(
            "目标工区微调与空间独立验证尚未完成。",
            "上游仓库未逐个披露三个 Gamma 权重各自的精确训练样本清单。",
        ),
        assets=(
            _asset(
                "models/manifest.json",
                root="project",
                role="weight_manifest",
                layer="report",
            ),
            _asset(
                "runtime/wellfuse/third_party/faultnet/LICENSE",
                root="project",
                role="license",
                layer="report",
            ),
            *tuple(
                _asset(
                    f"models/wellfuse/structural/faultnet/FaultNet_Gamma{gamma}.pt",
                    root="project",
                    role=f"checkpoint_gamma_{gamma}",
                    kind="checkpoint",
                )
                for gamma in ("0.5", "0.6", "0.7")
            ),
        ),
        model_id="faultnet_china_field",
        runner_id="faultnet_china_field",
        metadata={
            "upstream": "https://github.com/douyimin/FaultNet",
            "license": "Apache-2.0",
            "checkpoint_format": "torchscript",
            "default_gamma": "0.7",
            "normalization": "per_patch_minmax",
            "output_activation": "identity_probability",
            "threshold": 0.5,
            "inference_scope": "full_volume",
            "patch_size_zyx": [128, 128, 128],
            "overlap_zyx": [64, 64, 64],
            "weighted_probability_blending": True,
        },
    ),
    _ReleaseSpec(
        id="legacy_surface_seg",
        name="地层实例分割（SEG-Y地震体→地层标签体与置信度）",
        version="legacy-v1",
        task_id="horizon",
        description="原平台 SegFormer/Mask2Former 有序地层实例分割模型。",
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="legacy_external_checkpoint",
        scope=("regular_3d_seismic", "inline_slice_instance_segmentation"),
        warnings=("实例分割输出不应被表述为经过命名层位验证的层位追踪结果。",),
        assets=(
            _asset(
                "models/manifest.json",
                root="project",
                role="weight_manifest",
                layer="report",
            ),
            _asset(
                "接口模型/seismic_surface_seg/models/segformer-base/best.pt",
                root="project",
                role="checkpoint_base",
                kind="checkpoint",
            ),
            _asset(
                "接口模型/seismic_surface_seg/models/segformer-refine/best.pt",
                root="project",
                role="checkpoint_refine",
                kind="checkpoint",
            ),
            _asset(
                "接口模型/seismic_surface_seg/models/mask2former/best.pt",
                root="project",
                role="checkpoint_mask2former",
                kind="checkpoint",
            ),
        ),
        model_id="seismic_surface_seg",
        runner_id="seismic_surface_seg",
        metadata={"legacy": True},
    ),
    _ReleaseSpec(
        id="wellfuse_align_p13_chengdu",
        name="自动精细井震标定（SEG-Y+LAS+完整轨迹→TWT与不确定性）",
        version="p16.1-p13-ensemble-v1",
        task_id="alignment",
        description="118 井沿 LAS MD 采样网格导出的 TWT_mean 与 ensemble spread 冻结成果。",
        scientific_status="validated",
        runtime_status="runnable",
        evidence_class="three_seed_six_spatial_fold_ensemble",
        scope=("chengdu", "118_wells", "vertical", "deviated", "horizontal"),
        warnings=(
            "未知工区会实际执行冻结概率集成与物理候选链；物理或多峰歧义门拒绝时显式回退，不伪报精细标定成功。",
            "在线结果属于实验性推理证据，只能经 fusion_ready 门进入下游，永远不作为时深监督。",
            "TWT_std 未经过未知工区覆盖率/ECE 校准，不应表述为已校准置信区间。",
        ),
        assets=(
            _asset(
                "p13_ensemble_manifest.json",
                root="p16_align",
                role="ensemble_manifest",
                layer="report",
            ),
        ),
        model_id="wellfuse_align_p13",
        metadata={
            "registration_source": "wellfuse_align_prediction",
            "runtime_entrypoint": "registration_task",
            "unknown_survey_runtime_status": "experimental_runnable",
            "protected_fallback_chain": "factorized_v3->scientific_p13_or_physics_valid_v2",
            "complete_trajectory_required": True,
            "time_depth_supervision_is_model_input": False,
            "las_value_component_status": "rejected_at_g2_negative_component_gate",
            "uncertainty_calibrated_for_unknown_surveys": False,
        },
    ),
    _ReleaseSpec(
        id="wellfuse_geoalign_12b_direct_v1",
        name="多模态井震对齐（SEG-Y+LAS+完整轨迹→TWT与联合表征）",
        version="direct12b-v1-f6-final-runtime-ready",
        task_id="alignment",
        description=(
            "11,666,744,866参数4-bit主体与82,833,030参数的18成员冻结概率标定保护集成"
            "组成11,749,577,896参数部署体。F6最终运行契约已备，当前活动指针仍"
            "保持D1，等待F5 final-fit后原子切换。"
        ),
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="direct12b_d1_unlabeled_and_production_inference",
        scope=(
            "sealed_direct12b_11key",
            "raw_bundle_prepare_then_exact11",
            "chengdu_118",
            "f3_foundation",
            "teapot_single_well_runtime",
            "penobscot_b41_cpu_raw_prepare",
        ),
        warnings=(
            "当前D1指针只代表12B工程链路已跑通；F5 final-fit完成前不切换正式指针。",
            "最终推理必须从正式清单加载3组随机种子×6个数据折共18成员冻结保护集成。",
            "生产请求接受封存11-key manifest，或将time-domain SEG-Y、测井和实测轨迹raw bundle先转换为exact11；两种模式均不接受时深表、checkshot、VSP与监督参数。",
            "井侧沉积相任务存在多模态残差净增；密度补全任务的多模态残差门为0，成果仅归因于保护专家。",
        ),
        assets=(
            _asset(
                "models/wellfuse/direct12b/release_pointer_v1.json",
                root="project",
                role="active_adapter_pointer",
                layer="report",
            ),
            _asset(
                "models/wellfuse/direct12b/f6_final_integration_candidate_v1.json",
                root="project",
                role="f6_final_integration_candidate",
                layer="report",
            ),
            _asset(
                "artifacts/p16_1/align/p13_ensemble_manifest.json",
                root="wellfuse_project",
                role="production_anchor_scientific_manifest",
                layer="report",
            ),
            _asset(
                (
                    "artifacts/wellfuse_12b_direct/f5_final_fit_v1/"
                    "production_anchor_cpu_preflight.json"
                ),
                root="wellfuse_project",
                role="production_anchor_cpu_preflight",
                layer="report",
            ),
            _asset(
                (
                    "artifacts/wellfuse_12b_direct/f4_task_oof_v2/facies_1d/"
                    "three_seed_oof/completion.json"
                ),
                root="wellfuse_project",
                role="task_adapter_evidence_facies_v2",
                layer="report",
            ),
            _asset(
                (
                    "artifacts/wellfuse_12b_direct/f4_task_oof_v2/den/"
                    "three_seed_oof/completion.json"
                ),
                root="wellfuse_project",
                role="task_adapter_evidence_den_v2",
                layer="report",
            ),
            _asset(
                (
                    "11_direct12b_cache/raw_prepare_smoke/"
                    "penobscot_canada_b41_v1/raw_prepare_receipt_v1.json"
                ),
                root="direct12b_public_data",
                role="raw_prepare_cpu_smoke_receipt",
                layer="report",
            ),
        ),
        model_id="WellFuse-GeoAlign-12B-Direct-v1",
        runner_id="WellFuse-GeoAlign-12B-Direct-v1",
        metadata={
            "logical_parameter_count": 11_666_744_866,
            "core_logical_parameter_count": 11_666_744_866,
            "production_anchor_member_count": 18,
            "production_anchor_logical_parameter_count": 82_833_030,
            "deployment_logical_parameter_count": 11_749_577_896,
            "parameter_breakdown": {
                "core": {
                    "label": "量化多模态主体",
                    "logical_parameter_count": 11_666_744_866,
                },
                "production_anchor": {
                    "label": "冻结概率标定保护集成",
                    "member_count": 18,
                    "logical_parameter_count": 82_833_030,
                },
                "deployment": {
                    "label": "deployment loaded total",
                    "logical_parameter_count": 11_749_577_896,
                },
            },
            "quantization": "nf4_4bit",
            "input_contract": (
                "sealed_exact11_manifest_or_strict_no_td_raw_bundle_prepare"
            ),
            "input_modes": [
                "sealed_external_manifest",
                "raw_bundle_prepare_then_exact11",
            ],
            "forward_input_key_count": 11,
            "effect_status": "provisional_non_final",
            "final_runtime_contract_ready": True,
            "final_pointer_switch_pending_f5_final_fit": True,
            "production_anchor": {
                "source": "artifacts/p16_1/align/p13_ensemble_manifest.json",
                "schema_version": "wellfuse.align.p13-ensemble.v1",
                "sha256": "2696e8adde89c01ad38caec9d08a6b7d7c34dd61f4327f87d2971a84d1b2096d",
                "member_count": 18,
                "seed_count": 3,
                "fold_count": 6,
                "frozen": True,
                "time_depth_supervision_is_model_input": False,
            },
            "task_adapter_evidence": [
                {
                    "task": "facies_1d",
                    "version": "v2",
                    "status": "completed",
                    "schema_version": "wellfuse.direct12b_facies_three_seed_oof.v2",
                    "well_count": 46,
                    "seed_count": 3,
                    "fold_count": 6,
                    "whole_well_macro_f1": 0.5857756499149214,
                    "effect_attribution": {
                        "protected_expert": "三组随机种子井侧沉积相保护专家",
                        "direct12b_residual_contribution": "positive_net_increment",
                        "macro_f1_gain_over_protected_expert": 0.009983785324270023,
                        "balanced_accuracy_gain_over_protected_expert": 0.009117665639904127,
                        "nonzero_residual_cells": 16,
                        "protected_update0_cells": 2,
                    },
                },
                {
                    "task": "den",
                    "version": "v2",
                    "status": "completed",
                    "schema_version": "wellfuse.direct12b_den_protected_temporal_oof.v2",
                    "well_count": 104,
                    "seed_count": 3,
                    "fold_count": 6,
                    "macro_mae_g_cm3": 0.0457767773678691,
                    "effect_attribution": {
                        "protected_expert": "密度曲线补全保护专家",
                        "direct12b_residual_gate": 0.0,
                        "direct12b_residual_contribution": "zero_by_gate",
                        "metric_credit": "protected_expert_only",
                    },
                },
            ],
            "raw_prepare_evidence": {
                "survey": "penobscot_canada_raw_b41",
                "well": "b-41",
                "mode": "exact11",
                "foundation_device": "cpu",
                "gpu_used": False,
                "labels_or_time_depth_opened": False,
                "receipt_sha256": (
                    "9b61eec3ea2c6ae06502ce59b0399c0b"
                    "f24d2c2a87a6fa393283518778085ba8"
                ),
                "single_survey_strictly_below_1tb": True,
            },
            "pointer_reload_policy": "read_each_inference_request",
            "atomic_switch_supported": True,
            "requires_seismic": False,
            "raw_bundle_requires_time_domain_seismic": True,
            "requires_registration": False,
            "supports_raw_wells": True,
            "raw_prepare_default_device": "cpu",
            "raw_prepare_receipt_required": True,
            "time_depth_supervision_is_model_input": False,
            "forbidden_inference_parameters": ["TD", "checkshot", "VSP"],
        },
    ),
    _ReleaseSpec(
        id="wellfuse_align_geopath_tie_v1",
        name="轨迹感知井震校正（井震数据+完整轨迹→候选时深轨）",
        version="geopath-tie-v1",
        task_id="alignment",
        description="统一垂井、斜井和水平井的封存Registration V3候选精细标定运行时。",
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="chengdu_oof_double_seed_and_f3_regression",
        scope=("chengdu", "sealed_snapshot", "vertical", "deviated", "horizontal"),
        warnings=(
            "独立实验候选不改变默认自动精细井震标定的证据仲裁链。",
            "候选输出默认fusion_ready=false，必须经产品接受后进入下游。",
        ),
        assets=(
            _asset(
                "models/wellfuse/geopath_tie_v1/checkpoints/geopath_full/final_all.pt",
                root="project",
                role="checkpoint_geopath_full",
                layer="checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "model_outputs/geopath_tie_v1/promotion_summary.json",
                root="project",
                role="promotion_summary",
                layer="report",
            ),
            _asset(
                "model_outputs/geopath_tie_v1/f3_regression_raw_seed20260820/metrics.json",
                root="project",
                role="f3_regression_metrics",
                layer="report",
            ),
            _asset(
                "model_outputs/geopath_tie_v1/FINAL_REPORT.md",
                root="project",
                role="scientific_report",
                layer="report",
            ),
        ),
        model_id="wellfuse_align_geopath_tie_v1",
        runner_id="wellfuse_align_geopath_tie_v1",
        metadata={
            "registration_source": "wellfuse_align_geopath_tie_v1_candidate",
            "runtime_entrypoint": "prediction_task",
            "requires_registration": True,
            "requires_complete_trajectory": True,
            "candidate_only_by_default": True,
            "output_contract": "well-seismic.registration.v3",
        },
    ),
    _ReleaseSpec(
        id="wellfuse_p17_horizon",
        name="历史四层位追踪（既有成果→只读下载与可视化）",
        version="p17-ensemble-3seed-v1",
        task_id="horizon_legacy",
        description="已归档的四层位及有序地层场成果，仅保留既有结果复核。",
        scientific_status="validated",
        runtime_status="precomputed_only",
        evidence_class="real_dense_surface_fixed_validation_and_post_regression",
        scope=("chengdu", "Hartha", "Tanuma", "Khasib", "Zubair"),
        warnings=(
            "成都科学证据与未知工区实验runner状态必须分开解读。",
            "在线推理优先使用固定全局上下文井IDW；缺失时只能使用显式label-free事件先验。",
            "未知工区候选未验证精度，且任何目标层位面都禁止进入推理输入。",
        ),
        assets=(
            _asset(
                "final_report/REPORT.md",
                root="p17",
                role="scientific_report",
                layer="report",
            ),
            _asset(
                "horizon_event/ensemble_3seed_v1/post_regression/prediction_manifest.json",
                root="p17",
                role="prediction_manifest",
                layer="report",
            ),
            _asset(
                "horizon_event/ensemble_3seed_v1/post_regression/four_horizon_ensemble.npz",
                root="p17",
                role="horizon_predictions",
                layer="surface",
                unit="ms TWT",
                axis_order="compact_trace,horizon",
            ),
            _asset(
                "horizon_event/ensemble_3seed_v1/post_regression/ordered_relative_stratigraphic_field.npy",
                root="p17",
                role="ordered_stratigraphic_field",
                layer="volume",
                unit="relative stratigraphic coordinate",
                axis_order="time,compact_trace_ordinal",
            ),
            *tuple(
                _asset(
                    f"horizon_event/ensemble_3seed_v1/post_regression/petrel_xyz/{index:02d}_{name}.xyz",
                    root="p17",
                    role=f"petrel_surface_{name.lower()}",
                    layer="surface",
                    unit="ms TWT",
                )
                for index, name in enumerate(
                    ("Hartha", "Tanuma", "Khasib", "Zubair"), start=1
                )
            ),
        ),
        model_id="wellfuse_horizon_p17",
        metadata={
            "protected_prior": "context_well_idw_or_label_free_seismic_event",
            "horizon_count": 4,
            "unknown_survey_runtime_status": "archived",
            "target_surface_is_model_input": False,
            "archived": True,
            "historical_result_compatibility": True,
        },
    ),
    _ReleaseSpec(
        id="wellfuse_p17_facies_1d",
        name="井侧沉积相分类（九线LAS+完整轨迹→确定相序列与层段）",
        version="p17-oof-ensemble-v1",
        task_id="facies_1d",
        description="46 井三 Seed 六空间整井外折地震相预测成果。",
        scientific_status="validated",
        runtime_status="runnable",
        evidence_class="three_seed_six_spatial_whole_well_oof",
        scope=("chengdu", "46_wells"),
        warnings=("当前是 OOF 成果；新井部署需冻结最终 ensemble/refit。",),
        assets=(
            _asset(
                "chengdu_facies/facies_1d_ensemble_v1/manifest.json",
                root="p17",
                role="ensemble_manifest",
                layer="report",
            ),
            _asset(
                "chengdu_facies/facies_1d_ensemble_v1/per_well_metrics.csv",
                root="p17",
                role="per_well_metrics",
                layer="table",
            ),
        ),
        model_id="wellfuse_facies_1d_p17",
        runner_id="wellfuse_facies_1d_p17",
        metadata={
            "unknown_survey_runtime_status": "experimental_runnable",
            "facies_3d_is_not_substituted": True,
            "requires_registration": True,
        },
    ),
    _ReleaseSpec(
        id="wellfuse_p17_facies_3d",
        name="三维地震相分割（SEG-Y地震体→离散候选相体）",
        version="p17-candidate-v1",
        task_id="facies_3d",
        description="成都三维地震相候选解释成果。",
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="weak_candidate_gate",
        scope=("chengdu",),
        warnings=("仅弱候选，不得表述为真实成都三维相定量精度。",),
        assets=(
            _asset(
                "chengdu_facies/final_products_v1/runner_manifest.json",
                root="p17",
                role="candidate_manifest",
                layer="report",
            ),
        ),
        model_id="wellfuse_facies_3d_p17",
        runner_id="wellfuse_facies_3d_p17",
        metadata={
            "unknown_survey_runtime_status": "experimental_runnable",
            "default_inference_mode": "sample",
            "dense_3d_accuracy_claimed": False,
        },
    ),
    _ReleaseSpec(
        id="wellfuse_p17_fault_failed",
        name="稀疏断层研究档案（地震体→失败审计证据）",
        version="p17-final",
        task_id="fault",
        description="真实稀疏断层训练的冻结失败证据，仅供研究审计。",
        scientific_status="failed",
        runtime_status="blocked",
        evidence_class="real_sparse_interpretation_failed",
        scope=("chengdu",),
        warnings=(
            "输出发生退化，不是可交付断层解释；请使用 legacy FaultSeg 作工程基线。",
        ),
        assets=(
            _asset(
                "final_report/REPORT.md",
                root="p17",
                role="scientific_report",
                layer="report",
            ),
            _asset(
                "final_evidence_audit_v1/evidence_audit.json",
                root="p17",
                role="failure_decision",
                layer="report",
            ),
        ),
        model_id="wellfuse_fault_p17_failed",
    ),
    _ReleaseSpec(
        id="wellfuse_p17_channel",
        name="河道地质体识别（SEG-Y地震体→河道概率与几何属性）",
        version="p17-provisional-v1",
        task_id="channel",
        description="河道拓扑训练及真实工区候选解释。",
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="synthetic_dense_plus_real_unlabelled_candidate",
        scope=("cig_geobody", "real_unlabelled_candidates"),
        warnings=("真实工区未完成定量标签裁决；在线输出仅作为候选解释。",),
        assets=(
            _asset(
                "channel_topology/scientific_decision.json",
                root="p17",
                role="scientific_decision",
                layer="report",
            ),
            _asset(
                "channel_topology/rounds/seed_20260817/best.pt",
                root="p17",
                role="checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "real_geobody_candidates_v1/runner_manifest.json",
                root="p17",
                role="candidate_manifest",
                layer="report",
            ),
        ),
        model_id="wellfuse_channel_p17",
        runner_id="wellfuse_channel_p17",
    ),
    _ReleaseSpec(
        id="wellfuse_p17_karst",
        name="岩溶地质体识别（SEG-Y地震体→岩溶概率与几何属性）",
        version="p17-provisional-v1",
        task_id="karst",
        description="岩溶拓扑训练及真实工区候选解释。",
        scientific_status="candidate",
        runtime_status="runnable",
        evidence_class="synthetic_dense_plus_real_unlabelled_candidate",
        scope=("cig_geobody", "real_unlabelled_candidates"),
        warnings=("真实工区未完成定量标签裁决；在线输出仅作为候选解释。",),
        assets=(
            _asset(
                "karst_topology/scientific_decision.json",
                root="p17",
                role="scientific_decision",
                layer="report",
            ),
            _asset(
                "karst_topology/rounds/seed_20260817/best.pt",
                root="p17",
                role="checkpoint",
                kind="checkpoint",
            ),
            _asset(
                "real_geobody_candidates_v1/runner_manifest.json",
                root="p17",
                role="candidate_manifest",
                layer="report",
            ),
        ),
        model_id="wellfuse_karst_p17",
        runner_id="wellfuse_karst_p17",
    ),
    _fast_fluid_release(),
    _fracture_development_release(),
    _fast_facies_1d_release(),
    _fast_facies_3d_f3_release(),
    _fast_property_release(
        "DEN",
        epochs=8,
        unit="g/cm3",
        mae=0.03819043681056162,
        baseline_mae=0.09547154621040133,
        r2=0.8286167634037275,
    ),
    _fast_property_release(
        "POR",
        epochs=4,
        unit="fraction",
        mae=0.012150709559287515,
        baseline_mae=0.040881497608030934,
        r2=0.8633162540379803,
    ),
    _fast_property_release(
        "LOG_PERM",
        epochs=8,
        unit="log10(mD)",
        mae=0.2740407870201879,
        baseline_mae=0.7473427704135497,
        r2=0.7855383570831367,
    ),
    _fast_property_release(
        "SW",
        epochs=4,
        unit="fraction",
        mae=0.08670953086111602,
        baseline_mae=0.1395990729926923,
        r2=0.6145622367613592,
    ),
    _fast_property_release(
        "VSH",
        epochs=8,
        unit="fraction",
        mae=0.039367577585021296,
        baseline_mae=0.07518425095090489,
        r2=0.7054012948002524,
    ),
    _well_release("DEN", scientific_status="validated"),
    _well_release(
        "LOG_PERM",
        scientific_status="validated",
        warning="斜井 spatial_fold1/fold6 存在局部负增益。",
    ),
    _well_release("SW", scientific_status="validated"),
    _well_release(
        "POR",
        scientific_status="conditional",
        warning="斜井绝对增益为正，但逐井相对宏指标为负。",
    ),
    _well_release(
        "VSH",
        scientific_status="conditional",
        warning="水平井逐井相对增益为负，且不确定性低估。",
    ),
    _ReleaseSpec(
        id="wellfuse_p18_hydrocarbon",
        name="含烃指示证据（井曲线+完整轨迹→含烃概率与不确定性）",
        version="p18-fluid-oof-v2",
        task_id="hydrocarbon_evidence",
        description="真实区间标签上的含烃概率证据。",
        scientific_status="conditional",
        runtime_status="precomputed_only",
        evidence_class="three_seed_six_spatial_whole_well_fold_oof",
        scope=("chengdu", "63_label_wells"),
        warnings=("垂直井 F1 明显较弱；不得扩展解释为已验证水、气或八分类模型。",),
        assets=(
            _asset(
                "SCIENTIFIC_DECISION.json",
                root="p18",
                role="scientific_decision",
                layer="report",
            ),
            _asset(
                "chengdu_fluid/oof_v2/three_seed_oof_ensemble/manifest.json",
                root="p18",
                role="ensemble_manifest",
                layer="report",
            ),
            _asset(
                "chengdu_fluid/oof_v2/three_seed_oof_ensemble/per_well_metrics.csv",
                root="p18",
                role="per_well_metrics",
                layer="table",
            ),
        ),
        model_id="wellfuse_hydrocarbon_p18",
    ),
    _ReleaseSpec(
        id="wellfuse_p18_fluid_8class_failed",
        name="八类流体研究档案（井曲线→失败审计证据）",
        version="p18-final",
        task_id="hydrocarbon_evidence",
        description="辅助八分类任务的冻结失败结论。",
        scientific_status="failed",
        runtime_status="blocked",
        evidence_class="sparse_interval_labels_failed",
        scope=("chengdu",),
        warnings=("supported-class macro F1 仅 0.2825。",),
        assets=(
            _asset(
                "SCIENTIFIC_DECISION.json",
                root="p18",
                role="failure_decision",
                layer="report",
            ),
        ),
        model_id="wellfuse_fluid_8class_p18_failed",
    ),
    _ReleaseSpec(
        id="wellfuse_p18_water_gas_failed",
        name="水气判别研究档案（井曲线→失败审计证据）",
        version="p18-final",
        task_id="hydrocarbon_evidence",
        description="水与气试油关联的冻结失败结论。",
        scientific_status="failed",
        runtime_status="blocked",
        evidence_class="oil_test_association_failed",
        scope=("chengdu",),
        warnings=("Water 接近随机，Gas 低于随机；不得作为预测能力开放。",),
        assets=(
            _asset(
                "SCIENTIFIC_DECISION.json",
                root="p18",
                role="failure_decision",
                layer="report",
            ),
        ),
        model_id="wellfuse_water_gas_p18_failed",
    ),
)


class ReleaseCatalog:
    """In-memory view over immutable, explicitly declared releases."""

    schema_version = "well-seismic.release-catalog.v1"

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        artifact_root: str | Path | None = None,
        verify_sha256: bool = True,
        lifecycle_registry_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(
            project_root or Path(__file__).resolve().parents[3]
        ).resolve()
        bundled_wellfuse_root = self.project_root / "runtime" / "wellfuse"
        configured_project_root = self._optional_path(
            os.getenv("WELLFUSE_PROJECT_ROOT")
        )
        self.wellfuse_project_root = (
            configured_project_root or bundled_wellfuse_root.resolve()
        )

        # Resolution is deliberately declarative.  In particular, a configured
        # but missing directory remains selected so a bad deployment fails
        # closed instead of silently borrowing artifacts from another checkout.
        argument_artifact_root = self._optional_path(artifact_root)
        configured_artifact_root = self._optional_path(
            os.getenv("WELLFUSE_ARTIFACT_ROOT")
        )
        if argument_artifact_root is not None:
            self.artifact_root = argument_artifact_root
            self.artifact_root_source = "argument"
        elif configured_artifact_root is not None:
            self.artifact_root = configured_artifact_root
            self.artifact_root_source = "WELLFUSE_ARTIFACT_ROOT"
        elif configured_project_root is not None:
            self.artifact_root = (configured_project_root / "artifacts").resolve()
            self.artifact_root_source = "WELLFUSE_PROJECT_ROOT/artifacts"
        else:
            self.artifact_root = (bundled_wellfuse_root / "artifacts").resolve()
            self.artifact_root_source = "bundled_runtime"
        configured_public_data_root = os.getenv("WELLFUSE_PUBLIC_DATA_ROOT")
        self.direct12b_public_data_root = (
            Path(configured_public_data_root).expanduser().resolve()
            if configured_public_data_root and configured_public_data_root.strip()
            else (self.project_root / "data" / "external").resolve()
        )
        self.verify_sha256 = bool(verify_sha256)
        self._inventory_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._legacy_manifest = self._read_json(
            self.project_root / "models" / "manifest.json"
        )
        self._lifecycle_overlay: LifecycleRegistryOverlay | None = None
        self._lifecycle_overlay_status: dict[str, Any]
        configured_registry = lifecycle_registry_path or os.getenv(
            "WELLFUSE_LIFECYCLE_REGISTRY"
        )
        registry_path = Path(
            configured_registry or self.artifact_root / "lifecycle" / "registry.json"
        ).resolve()
        if not registry_path.is_file() and configured_registry is None:
            self._lifecycle_overlay_status = {
                "schema_version": "well-seismic.lifecycle-overlay.v1",
                "enabled": False,
                "read_only": True,
                "fail_closed": True,
                "status": "not_configured",
                "release_count": 0,
            }
        else:
            try:
                self._lifecycle_overlay = LifecycleRegistryOverlay(
                    registry_path,
                    artifact_root=self.artifact_root,
                )
            except LifecycleOverlayError as exc:
                self._lifecycle_overlay_status = {
                    "schema_version": "well-seismic.lifecycle-overlay.v1",
                    "enabled": False,
                    "read_only": True,
                    "fail_closed": True,
                    "status": "rejected",
                    "release_count": 0,
                    "error": str(exc),
                }
            else:
                self._lifecycle_overlay_status = {
                    **self._lifecycle_overlay.capabilities(),
                    "status": "verified",
                }
        releases = [
            self._apply_lifecycle_authority(
                self._apply_direct12b_pointer_state(release)
            )
            for release in self._build_releases()
        ]
        if self._lifecycle_overlay is not None:
            bound = set(_LIFECYCLE_RELEASE_BINDINGS.values())
            releases.extend(
                release
                for release in self._lifecycle_overlay.list_releases()
                if (
                    release.task_id,
                    str(
                        dict(release.metadata.get("lifecycle", {})).get("candidate_id")
                        or ""
                    ),
                )
                not in bound
            )
        if interface_only_enabled():
            # Keep release IDs and artifact contracts visible to the unchanged
            # frontend, but never advertise a task checkpoint as runnable.
            releases = [interface_only_release(release) for release in releases]
        if len({release.id for release in releases}) != len(releases):
            raise ValueError("Release ids collide after lifecycle overlay")
        self._releases = {release.id: release for release in releases}

    def _apply_direct12b_pointer_state(
        self, release: ArtifactRelease
    ) -> ArtifactRelease:
        if release.id != "wellfuse_geoalign_12b_direct_v1":
            return release
        from ..direct12b_runtime import direct12b_pointer_public_state

        pointer_state = direct12b_pointer_public_state(self.project_root)
        pointer_valid = bool(pointer_state.get("valid"))
        final_active = bool(pointer_state.get("final_release_active"))
        warnings = release.warnings
        if final_active:
            warnings = tuple(
                warning
                for warning in warnings
                if not warning.startswith("当前D1指针")
            ) + (
                "正式拟合指针已激活；每次推理均重新验证18成员冻结概率标定保护集成与无时深监督边界。",
            )
        return replace(
            release,
            runtime_status=(release.runtime_status if pointer_valid else "unavailable"),
            version=(
                "direct12b-v1-f5-final-active"
                if final_active
                else release.version
            ),
            description=(
                (
                    "11,666,744,866参数4-bit主体与82,833,030参数的18成员冻结概率标定"
                    "保护集成组成11,749,577,896参数正式部署体；正式拟合指针已激活。"
                )
                if final_active
                else release.description
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                **release.metadata,
                "effect_status": pointer_state.get(
                    "effect_status", "pointer_invalid"
                ),
                "active_pointer_state": pointer_state,
                "final_pointer_switch_pending_f5_final_fit": not final_active,
            },
        )

    def _apply_lifecycle_authority(self, release: ArtifactRelease) -> ArtifactRelease:
        """Overlay lifecycle truth while preserving stable public release ids.

        The static declarations remain useful as capability descriptions and
        adapter fallbacks.  They are not allowed to promote scientific state.
        A corrupt configured snapshot blocks WellFuse runtime entries, while
        the two independent legacy checkpoints remain available.
        """

        if release.source == "legacy":
            return release
        if bool(release.metadata.get("archived")):
            # Product retirement is stronger than a historical lifecycle
            # pointer. Keep immutable evidence downloadable, but never let an
            # old runtime-default pointer re-enable new inference.
            return replace(
                release,
                runtime_status="precomputed_only",
                metadata={
                    **release.metadata,
                    "status_authority": {
                        "source": "product_archived",
                        "automatic_scientific_promotion": False,
                        "scientific_incumbent": False,
                        "runtime_default": False,
                    },
                },
            )
        overlay_status = str(self._lifecycle_overlay_status.get("status", ""))
        authority = {
            "registry_sha256": self._lifecycle_overlay_status.get("registry_sha256"),
            "automatic_scientific_promotion": False,
            "scientific_incumbent": False,
            "runtime_default": False,
        }
        if overlay_status == "rejected":
            return replace(
                release,
                scientific_status="unassessed",
                runtime_status="blocked",
                evidence_class="lifecycle_registry_rejected",
                warnings=release.warnings
                + ("生命周期快照校验失败；WellFuse运行时按fail-closed策略停用。",),
                metadata={
                    **release.metadata,
                    "status_authority": {**authority, "source": "rejected_lifecycle_snapshot"},
                },
                source="wellfuse_static_fallback",
            )
        if self._lifecycle_overlay is None:
            # Compatibility for installations that have never configured a
            # lifecycle snapshot.  Status provenance is explicit so it cannot
            # be mistaken for a pointer-backed promotion.
            return replace(
                release,
                metadata={
                    **release.metadata,
                    "status_authority": {**authority, "source": "static_legacy_fallback"},
                },
                source="wellfuse_static_fallback",
            )

        binding = _LIFECYCLE_RELEASE_BINDINGS.get(release.id)
        candidate = None
        if binding is not None:
            candidate = self._lifecycle_overlay.resolve_candidate(
                task_id=binding[0], candidate_id=binding[1]
            )
        if candidate is None:
            # P18 and other runnable adapters that have not yet been entered in
            # the lifecycle registry remain explicit candidates.  Their
            # adapter may run, but no static validated/conditional claim is
            # allowed to survive the verified registry boundary.
            fallback_runtime = release.runtime_status
            if fallback_runtime not in {"runnable", "adapter_required", "precomputed_only"}:
                fallback_runtime = "blocked"
            return replace(
                release,
                scientific_status="candidate" if fallback_runtime == "runnable" else "unassessed",
                runtime_status=fallback_runtime,
                evidence_class="static_capability_adapter_fallback",
                warnings=release.warnings
                + (
                    "该模型尚无生命周期候选绑定；仅保留适配器/能力回退，不是科学晋级。",
                ),
                metadata={
                    **release.metadata,
                    "status_authority": {
                        **authority,
                        "source": "verified_registry_unbound_adapter_fallback",
                        "runtime_fallback": fallback_runtime == "runnable",
                    },
                },
                source="wellfuse_static_fallback",
            )

        lifecycle = dict(candidate.metadata.get("lifecycle", {}))
        scientific_incumbent = bool(lifecycle.get("scientific_incumbent"))
        runtime_default = bool(lifecycle.get("runtime_default"))
        runtime_status = candidate.runtime_status
        if runtime_status == "runnable" and not runtime_default:
            runtime_status = "adapter_required"
        verified_artifacts = tuple(
            replace(artifact, id=f"lifecycle-{artifact.id}")
            for artifact in candidate.artifacts
        )
        existing_ids = {artifact.id for artifact in release.artifacts}
        verified_artifacts = tuple(
            artifact for artifact in verified_artifacts if artifact.id not in existing_ids
        )
        return replace(
            release,
            version=candidate.version,
            scientific_status=candidate.scientific_status,
            runtime_status=runtime_status,
            evidence_class=candidate.evidence_class,
            warnings=tuple(dict.fromkeys((*release.warnings, *candidate.warnings))),
            artifacts=(*release.artifacts, *verified_artifacts),
            metadata={
                **release.metadata,
                "status_authority": {
                    **authority,
                    "source": "verified_lifecycle_registry",
                    "candidate_id": binding[1],
                    "candidate_sha256": lifecycle.get("candidate_sha256"),
                    "scientific_incumbent": scientific_incumbent,
                    "runtime_default": runtime_default,
                    "pointer_generations": lifecycle.get("pointer_generations", {}),
                },
            },
            source="wellfuse_lifecycle",
        )

    def list(self) -> list[ArtifactRelease]:
        """Return releases in stable declaration order."""

        return list(self._releases.values())

    def list_releases(self) -> list[ArtifactRelease]:
        return self.list()

    @property
    def lifecycle_overlay_status(self) -> dict[str, Any]:
        """Return a copy so API clients cannot mutate the catalog state."""

        return dict(self._lifecycle_overlay_status)

    def get(self, release_id: str) -> ArtifactRelease:
        try:
            return self._releases[release_id]
        except KeyError as exc:
            raise KeyError(f"Unknown release: {release_id}") from exc

    def capabilities(self) -> dict[str, Any]:
        releases = [release.to_dict() for release in self.list()]
        return {
            "schema_version": self.schema_version,
            "artifact_root": str(self.artifact_root),
            "artifact_root_source": self.artifact_root_source,
            "read_only": True,
            "release_count": len(releases),
            "releases": releases,
            "statuses": {
                "scientific": [
                    "unassessed",
                    "candidate",
                    "selected_for_refit",
                    "validated",
                    "conditional",
                    "failed",
                    "rejected",
                ],
                "runtime": [
                    "runnable",
                    "adapter_required",
                    "precomputed_only",
                    "blocked",
                    "unavailable",
                ],
            },
            "lifecycle_overlay": self.lifecycle_overlay_status,
        }

    def _build_releases(self) -> Iterable[ArtifactRelease]:
        for spec in _RELEASE_SPECS:
            artifacts = tuple(self._resolve_asset(asset) for asset in spec.assets)
            if spec.id == "wellfuse_align_p13_chengdu":
                artifacts += tuple(self._align_products())
            kwargs = {
                "id": spec.id,
                "name": spec.name,
                "version": spec.version,
                "task_id": spec.task_id,
                "description": spec.description,
                "scientific_status": spec.scientific_status,
                "runtime_status": spec.runtime_status,
                "evidence_class": spec.evidence_class,
                "scope": spec.scope,
                "warnings": spec.warnings,
                "artifacts": artifacts,
                "metadata": spec.metadata,
                "source": "legacy" if spec.id.startswith("legacy_") else "wellfuse",
            }
            if spec.model_id:
                yield ModelRelease(
                    model_id=spec.model_id, runner_id=spec.runner_id, **kwargs
                )
            else:
                yield ArtifactRelease(**kwargs)

    def _root(self, name: str) -> Path:
        roots = {
            "project": self.project_root,
            "artifacts": self.artifact_root,
            "p16_align": self.artifact_root / "p16_1" / "align",
            "p17": self.artifact_root / "p17",
            "p18": self.artifact_root / "p18",
            "wellfuse_project": self.wellfuse_project_root,
            "direct12b_public_data": self.direct12b_public_data_root,
        }
        return roots[name]

    def _resolve_asset(self, spec: _AssetSpec) -> ReleaseArtifact:
        root = self._root(spec.root).resolve()
        relative = Path(spec.relative_path)
        self._validate_relative(relative)
        path = (root / relative).resolve()
        if not self._is_within(path, root):
            raise ValueError(f"Release artifact escapes its root: {spec.relative_path}")
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else None
        expected = self._expected_integrity(spec.root, spec.relative_path)
        expected_sha = expected.get("sha256")
        expected_size = expected.get("size_bytes", expected.get("size"))
        integrity = "untracked" if exists else "missing"
        if exists and expected_size is not None and size_bytes != int(expected_size):
            integrity = "size_mismatch"
        elif exists and expected_sha:
            if (
                self.verify_sha256
                and size_bytes is not None
                and size_bytes <= _HASH_LIMIT_BYTES
            ):
                integrity = (
                    "sha256_verified"
                    if self._sha256(path) == expected_sha
                    else "sha256_mismatch"
                )
            else:
                integrity = "inventory_declared"
        elif exists and expected.get("declared"):
            integrity = "manifest_declared"
        elif exists and expected_size is not None:
            integrity = "size_verified"
        media_type = (
            spec.media_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        return ReleaseArtifact(
            id=spec.id,
            name=spec.name,
            role=spec.role,
            kind=spec.kind,
            path=str(path),
            relative_path=spec.relative_path,
            exists=exists,
            media_type=media_type,
            layer=spec.layer,
            sha256=expected_sha,
            size_bytes=size_bytes,
            integrity_status=integrity,
            unit=spec.unit,
            axis_order=spec.axis_order,
            uncertainty_definition=spec.uncertainty_definition,
        )

    def _align_products(self) -> Iterable[ReleaseArtifact]:
        root = self._root("p16_align").resolve()
        products = root / "chengdu_predictions"
        if not products.is_dir() or not self._is_within(products.resolve(), root):
            return
        # Deliberately shallow: immediate well directory + three fixed basenames.
        for well_dir in sorted(
            products.iterdir(), key=lambda item: item.name.casefold()
        ):
            if not well_dir.is_dir() or well_dir.is_symlink():
                continue
            for filename, (role, layer, media_type) in _SAFE_ALIGN_FILES.items():
                relative = Path("chengdu_predictions") / well_dir.name / filename
                spec = _AssetSpec(
                    id=f"well.{well_dir.name}.{role}".lower(),
                    name=f"{well_dir.name} {filename}",
                    role=role,
                    kind="file",
                    root="p16_align",
                    relative_path=relative.as_posix(),
                    layer=layer,
                    media_type=media_type,
                    unit="ms TWT" if filename != "manifest.json" else None,
                    uncertainty_definition=(
                        "ensemble aleatoric plus epistemic standard deviation"
                        if filename != "manifest.json"
                        else None
                    ),
                )
                yield self._resolve_asset(spec)

    def _expected_integrity(self, root_name: str, relative_path: str) -> dict[str, Any]:
        normalized = relative_path.replace("\\", "/")
        if root_name == "artifacts":
            return _EXPLICIT_ARTIFACT_INTEGRITY.get(normalized, {})
        if root_name == "wellfuse_project":
            return _EXPLICIT_WELLFUSE_PROJECT_INTEGRITY.get(normalized, {})
        if root_name == "direct12b_public_data":
            return _EXPLICIT_DIRECT12B_PUBLIC_DATA_INTEGRITY.get(normalized, {})
        if root_name in {"p17", "p18"}:
            return self._inventory(root_name).get(normalized, {})
        if root_name == "project":
            for item in self._legacy_manifest.get("models", []):
                if str(item.get("path", "")).replace("\\", "/") == normalized:
                    return item
        if root_name == "p16_align":
            if normalized == "p13_ensemble_manifest.json":
                manifest = self._read_json(self._root(root_name) / normalized)
                valid = (
                    manifest.get("schema_version") == "wellfuse.align.p13-ensemble.v1"
                    and manifest.get("member_count") == len(manifest.get("members", []))
                    and manifest.get("registration_source")
                    == "wellfuse_align_prediction"
                )
                return {"declared": valid}
            parts = Path(normalized).parts
            if len(parts) == 3 and parts[0] == "chengdu_predictions":
                well_manifest = self._read_json(
                    self._root(root_name) / parts[0] / parts[1] / "manifest.json"
                )
                if parts[2] == "manifest.json":
                    return {
                        "declared": well_manifest.get("schema_version")
                        == "wellfuse.align.registration.v1"
                    }
                return {"declared": parts[2] in well_manifest.get("files", [])}
        return {}

    def _inventory(self, name: str) -> dict[str, dict[str, Any]]:
        if name not in self._inventory_cache:
            document = self._read_json(self._root(name) / "FROZEN_INVENTORY.json")
            self._inventory_cache[name] = {
                str(item.get("path", "")).replace("\\", "/"): item
                for item in document.get("files", [])
                if isinstance(item, dict) and item.get("path")
            }
        return self._inventory_cache[name]

    @staticmethod
    def _optional_path(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _validate_relative(path: Path) -> None:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe release artifact path: {path}")
        lowered = [part.casefold() for part in path.parts]
        if any(
            part == disallowed or part.startswith(f"{disallowed}_")
            for part in lowered
            for disallowed in _DISALLOWED_PARTS
        ):
            raise ValueError(f"Disallowed release artifact path: {path}")


def build_release_catalog(
    *,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    verify_sha256: bool = True,
    lifecycle_registry_path: str | Path | None = None,
) -> ReleaseCatalog:
    return ReleaseCatalog(
        project_root=project_root,
        artifact_root=artifact_root,
        verify_sha256=verify_sha256,
        lifecycle_registry_path=lifecycle_registry_path,
    )
