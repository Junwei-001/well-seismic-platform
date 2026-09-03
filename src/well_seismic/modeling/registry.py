from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ..fault_models import FAULTNET_MODEL_ID
from ..platform_mode import annotate_interface_only_spec, interface_only_enabled
from .contracts import FusionPlugin, ModelPlugin, ModelSpec

Factory = Callable[[], Any]


class ModelRegistry:
    """模型元数据与实现解耦；外部包可通过入口点或显式注册接入。"""

    entry_point_group = "well_seismic.plugins"

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._factories: dict[str, Factory] = {}
        self._protocols: dict[str, str] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(
        self,
        spec: ModelSpec,
        factory: Factory | None = None,
        *,
        protocol: str = "model",
        replace: bool = False,
    ) -> None:
        if protocol not in {"model", "fusion"}:
            raise ValueError(f"未知插件协议：{protocol}")
        if spec.id in self._specs and not replace:
            raise ValueError(f"模型ID已注册：{spec.id}")
        self._specs[spec.id] = spec
        self._protocols[spec.id] = protocol
        if factory is not None:
            self._factories[spec.id] = factory

    def list_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def create(self, model_id: str) -> Any:
        if model_id not in self._specs:
            raise KeyError(f"未知模型：{model_id}")
        if model_id not in self._factories:
            raise RuntimeError(f"模型仅预留接口，尚未安装实现：{model_id}")
        plugin = self._factories[model_id]()
        protocol = self._protocols[model_id]
        if protocol == "model" and not isinstance(plugin, ModelPlugin):
            raise TypeError(f"插件不符合ModelPlugin协议：{model_id}")
        if protocol == "fusion" and not isinstance(plugin, FusionPlugin):
            raise TypeError(f"插件不符合FusionPlugin协议：{model_id}")
        return plugin

    def load_entry_points(self) -> list[str]:
        # The extracted shell must not import an extension that can pull model
        # weights implicitly.  A later task-model package can be enabled by
        # starting the full runtime without this mode switch.
        if interface_only_enabled():
            return []
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                register = entry_point.load()
                register(self)
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {
                        "plugin": entry_point.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return loaded

    def capabilities(self) -> dict[str, Any]:
        models = [spec.to_dict() for spec in self.list_specs()]
        return {
            "plugin_contract": {
                "entry_point_group": self.entry_point_group,
                "required_methods": ["fit", "predict", "save", "load"],
                "fusion_required_methods": [
                    "fit",
                    "transform",
                    "fit_transform",
                    "state_dict",
                    "load_state_dict",
                ],
                "stable_inputs": [
                    "地震窗口或三维子体",
                    "标准测井曲线",
                    "井轨迹与空间坐标",
                    "掩码、置信度和来源记录",
                ],
            },
            "plugin_load_errors": list(self.plugin_load_errors),
            "models": models,
        }

    def apply_release_catalog(self, catalog: Any) -> None:
        """Project pointer-backed lifecycle state onto static model specs.

        ModelSpec remains the adapter/capability declaration.  Scientific and
        runtime state is copied from ReleaseCatalog, which in turn resolves the
        checksum-verified lifecycle snapshot.  Missing cross-references fail
        closed instead of silently retaining a stale hard-coded promotion.
        """

        for model_id, spec in tuple(self._specs.items()):
            release_id = spec.metadata.get("release_id")
            if not release_id:
                continue
            try:
                release = catalog.get(str(release_id))
            except KeyError:
                self._specs[model_id] = replace(
                    spec,
                    scientific_status="unassessed",
                    runtime_status="blocked",
                    evidence_class="release_cross_reference_missing",
                    warnings=spec.warnings
                    + ("发布目录缺少交叉引用；模型按fail-closed策略停用。",),
                    metadata={
                        **spec.metadata,
                        "status_authority": {
                            "source": "release_cross_reference_missing",
                            "automatic_scientific_promotion": False,
                        },
                    },
                )
                continue
            self._specs[model_id] = replace(
                spec,
                version=release.version,
                scientific_status=release.scientific_status,
                runtime_status=release.runtime_status,
                evidence_class=release.evidence_class,
                warnings=tuple(dict.fromkeys((*spec.warnings, *release.warnings))),
                metadata={
                    **spec.metadata,
                    "status_authority": dict(
                        release.metadata.get("status_authority", {})
                    ),
                },
            )
        if interface_only_enabled():
            self.mark_interface_only()

    def mark_interface_only(self) -> None:
        """Keep model IDs visible while removing runnable claims."""

        self._specs = {
            model_id: annotate_interface_only_spec(spec)
            for model_id, spec in self._specs.items()
        }


def build_default_registry() -> ModelRegistry:
    from ..fusion import ConfidenceGatedFusion
    from ..layerpulse_contract import layerpulse_model_spec
    from ..direct12b_runtime import (
        DIRECT12B_CORE_LOGICAL_PARAMETERS,
        DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS,
        DIRECT12B_MODEL_ID,
        DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS,
        DIRECT12B_PRODUCTION_ANCHOR_MEMBERS,
        DIRECT12B_RELEASE_ID,
        direct12b_pointer_public_state,
    )

    registry = ModelRegistry()
    direct12b_pointer_state = direct12b_pointer_public_state(
        Path(__file__).resolve().parents[3]
    )
    direct12b_final_active = bool(
        direct12b_pointer_state.get("final_release_active")
    )
    specs = (
        layerpulse_model_spec(),
        ModelSpec(
            id="faultseg_3d",
            name="三维地震体→慧眼三维断层识别结果",
            category="地震分割",
            status="默认工区中心单块；可选全区预测",
            description=(
                "项目三维卷积分割网络；默认在工区三轴中心执行1个"
                "独立128³完整块，也可选择以64³重叠执行全区滑窗并重建"
                "连续的全工区三维断层概率体。"
            ),
            inputs=("float32 三维地震体 [Z, Inline, Crossline]",),
            outputs=("中心单块或全区断层结果", "确定性断层掩码", "正交切片与三维联动视图"),
            implementation="runtime/wellfuse/third_party/faultseg",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="external_checkpoint",
            warnings=(
                "与稀疏断层研究中的科学失败结果分开；当前模型属于外部实验基线。",
            ),
            metadata={
                "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
                "channels": 1,
                "config": "configs/faultseg.yaml",
                "training_required": False,
                "prediction_task": "fault",
                "release_id": "legacy_faultseg_3d",
                "requires_registration": False,
                "registration_policy": "none",
                "prepared_view_policy": "none",
                "prepared_view_consumed": False,
                "inference_scope": "selectable",
                "default_scope": "center_block_1",
                "supported_scopes": ["center_block_1", "full_volume"],
                "historical_read_only_scopes": ["representative_grid_128"],
                "full_volume_long_running": True,
                "patch_size_zyx": [128, 128, 128],
                "overlap_zyx": [64, 64, 64],
                "weighted_probability_blending": True,
            },
        ),
        ModelSpec(
            id=FAULTNET_MODEL_ID,
            name="SEG-Y地震体→慧眼区域增强断层结果",
            category="地震分割",
            status="预训练推理已接入",
            description=(
                "项目区域增强断层网络；使用已登记的封装推理权重与逐窗口 "
                "min-max 合同，对 128³ 重叠子体积推理并以 "
                "64³ 重叠区概率加权融合重建完整工区。"
            ),
            inputs=("float32 三维地震体 [Z, Inline, Crossline]",),
            outputs=("全工区三维断层概率体", "确定性断层掩码", "全工区正交切片与三维联动视图"),
            implementation="runtime/wellfuse/third_party/faultnet",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="external_china_field_checkpoint",
            warnings=(
                "国内实测数据参与方法训练不等同于目标工区验证；微调后仍需按空间分区独立验收。",
                "上游仓库未逐个披露三个 Gamma 权重各自的精确训练样本清单。",
            ),
            metadata={
                "tensor_order": ["N", "C", "Z", "INLINE", "CROSSLINE"],
                "channels": 1,
                "config": "configs/faultseg.yaml#faultnet",
                "training_required": False,
                "fine_tuning_supported": True,
                "prediction_task": "fault",
                "release_id": "faultnet_china_field",
                "requires_registration": False,
                "registration_policy": "none",
                "prepared_view_policy": "none",
                "prepared_view_consumed": False,
                "inference_scope": "full_volume",
                "patch_size_zyx": [128, 128, 128],
                "overlap_zyx": [64, 64, 64],
                "weighted_probability_blending": True,
                "normalization": "per_patch_minmax",
                "output_activation": "identity_probability",
                "checkpoint_variants": ["Gamma0.5", "Gamma0.6", "Gamma0.7"],
                "default_checkpoint_variant": "Gamma0.7",
                "license": "Apache-2.0",
                "upstream": "https://github.com/douyimin/FaultNet",
            },
        ),
        ModelSpec(
            id="seismic_surface_seg",
            name="地层实例分割（SEG-Y地震体→地层标签体与置信度）",
            category="地震分割",
            status="推理已接入",
            description="基础分割、细化分割与实例分割组成的三阶段有序地层实例分割模型。",
            inputs=("规则三维后叠加 SEG-Y [Inline, Crossline, Sample]",),
            outputs=("地层实例标签体", "分割置信度体", "标签 SEG-Y"),
            implementation="接口模型/seismic_surface_seg",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="external_checkpoint",
            warnings=("逐Inline实例标签不等同于命名层位面或跨Inline连续层位。",),
            metadata={
                "input_order": ["INLINE", "CROSSLINE", "SAMPLE"],
                "slice_tensor_order": ["N", "C", "SAMPLE", "CROSSLINE"],
                "channels": 3,
                "config": "configs/surface_seg.yaml",
                "training_required": False,
                "prediction_task": "horizon",
                "registration_policy": "none",
                "prepared_view_policy": "none",
                "inference_scope": "full_volume",
                "release_id": "legacy_surface_seg",
            },
        ),
        ModelSpec(
            id="wellfuse_align_p13",
            name="自动精细井震标定（SEG-Y+LAS+完整轨迹→TWT与不确定性）",
            category="概率地下注册",
            status="冻结成果与未知工区实验运行时已接入",
            description="冻结概率集成模型使用地震、声波和完整轨迹，在 LAS 的 MD 网格输出TWT均值、离散度与物理一致性门；未知工区结果明确标记为实验性。",
            inputs=("LAS的MD采样网格", "完整MD/TVDSS/X/Y/Z轨迹", "三维地震特征"),
            outputs=("TWT_mean/ensemble_spread", "CSV/LAS", "质量与来源清单"),
            implementation="wellfuse5090 ensemble runtime",
            scientific_status="validated",
            runtime_status="runnable",
            evidence_class="real_well_oof_and_frozen_predictions",
            metadata={
                "prediction_task": "alignment",
                "release_id": "wellfuse_align_p13_chengdu",
                "registration_source": "wellfuse_align_prediction",
                "runtime_entrypoint": "registration_task",
                "unknown_survey_runtime_status": "experimental_runnable",
                "protected_fallback_chain": "factorized_v3->scientific_p13_or_physics_valid_v2",
                "las_value_component_status": "rejected_at_g2_negative_component_gate",
                "uncertainty_calibrated_for_unknown_surveys": False,
            },
        ),
        ModelSpec(
            id=DIRECT12B_MODEL_ID,
            name="多模态井震对齐（SEG-Y+LAS+完整轨迹→TWT与联合表征）",
            category="井震融合基础模型",
            status=(
                "正式部署指针已激活·18成员冻结概率标定保护集成"
                if direct12b_final_active
                else "最终接入已就绪·当前仍为临时部署指针"
            ),
            description=(
                "量化井震多模态主体与18成员冻结概率标定保护集成组成"
                "11.75B部署体；最终指针切换后通过封存11-key输入预测严格单调"
                "TWT(MD)、对齐概率、不确定性与共享井震表征；也可将严格无TD的"
                "raw bundle先转为同exact11契约。"
            ),
            inputs=(
                "封存十一张量井震输入NPZ清单",
                "可选raw bundle：time-domain SEG-Y + LAS/CSV/TXT + 实测轨迹",
                "常规测井及缺失掩码",
                "原生时间域地震token",
                "井口与完整轨迹token",
            ),
            outputs=(
                "严格单调TWT(MD)",
                "对齐概率与不确定性",
                "井震联合表征",
                "生产推理receipt",
            ),
            version=(
                "direct12b-v1-f5-final-active"
                if direct12b_final_active
                else "direct12b-v1-f6-final-runtime-ready"
            ),
            implementation="well_seismic.direct12b_runtime",
            scientific_status="candidate",
            runtime_status=(
                "runnable"
                if bool(direct12b_pointer_state.get("valid"))
                else "unavailable"
            ),
            evidence_class="direct12b_d1_unlabeled_and_production_inference",
            warnings=(
                (
                    "正式部署指针已激活，运行时必须逐请求验证18成员冻结概率标定保护集成。"
                    if direct12b_final_active
                    else "当前活动指针仍指向无标签临时权重；正式拟合完成前不会提前切换。"
                ),
                "最终推理必须加载正式清单绑定的3组随机种子×6个数据折共18成员冻结保护集成。",
                "测试与推理不接受时深表、checkshot、VSP或监督参数。",
            ),
            metadata={
                "prediction_task": "alignment",
                "release_id": DIRECT12B_RELEASE_ID,
                "runtime_entrypoint": "prediction_task",
                "input_mode": "sealed_manifest_or_raw_bundle_prepare",
                "input_modes": [
                    "sealed_external_manifest",
                    "raw_bundle_prepare_then_exact11",
                ],
                "requires_seismic": False,
                "raw_bundle_requires_time_domain_seismic": True,
                "requires_registration": False,
                "supports_raw_wells": True,
                "raw_prepare_default_device": "cpu",
                "raw_prepare_receipt_required": True,
                "logical_parameter_count": DIRECT12B_CORE_LOGICAL_PARAMETERS,
                "core_logical_parameter_count": DIRECT12B_CORE_LOGICAL_PARAMETERS,
                "production_anchor_member_count": (
                    DIRECT12B_PRODUCTION_ANCHOR_MEMBERS
                ),
                "production_anchor_logical_parameter_count": (
                    DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS
                ),
                "deployment_logical_parameter_count": (
                    DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS
                ),
                "parameter_breakdown": {
                    "core": {
                        "label": "量化多模态主体",
                        "logical_parameter_count": DIRECT12B_CORE_LOGICAL_PARAMETERS,
                    },
                    "production_anchor": {
                        "label": "冻结概率标定保护集成（18成员）",
                        "member_count": DIRECT12B_PRODUCTION_ANCHOR_MEMBERS,
                        "logical_parameter_count": (
                            DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS
                        ),
                    },
                    "deployment": {
                        "label": "deployment loaded total",
                        "logical_parameter_count": (
                            DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS
                        ),
                    },
                },
                "quantization": "nf4_4bit",
                "active_adapter_pointer": (
                    "models/wellfuse/direct12b/release_pointer_v1.json"
                ),
                "effect_status": direct12b_pointer_state.get(
                    "effect_status", "pointer_invalid"
                ),
                "active_pointer_state": direct12b_pointer_state,
                "final_runtime_contract_ready": True,
                "public_prediction_enabled": bool(
                    direct12b_pointer_state.get("valid")
                ),
                "final_pointer_switch_pending_f5_final_fit": (
                    not direct12b_final_active
                ),
                "production_anchor_source": (
                    "artifacts/p16_1/align/p13_ensemble_manifest.json"
                ),
                "time_depth_supervision_is_model_input": False,
                "forbidden_inference_parameters": ["TD", "checkshot", "VSP"],
            },
        ),
        ModelSpec(
            id="wellfuse_align_geopath_tie_v1",
            name="轨迹感知井震校正（井震数据+完整轨迹→候选时深轨）",
            category="概率地下注册",
            status="固定checkpoint候选运行时已接入",
            description=(
                "统一垂井、斜井和水平井的轨迹感知井震精细标定候选；"
                "在同一封存快照和Registration V3先验上输出独立候选产品。"
            ),
            inputs=(
                "封存source snapshot",
                "Registration V3 points/manifest",
                "完整MD/TVD/XYZ轨迹",
                "封存LAS（DT/密度及曲线mask）",
                "三维SEG-Y",
            ),
            outputs=(
                "候选Registration V3",
                "逐井TWT与路径不确定度",
                "registration consumption receipt",
            ),
            implementation="well_seismic.alignment.geopath_runtime",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="chengdu_oof_double_seed_and_f3_regression",
            warnings=(
                "该模型是独立实验候选，不改变默认自动精细井震标定的证据仲裁链。",
                "在线候选默认fusion_ready=false，须经产品侧接受后才能进入下游。",
                "未知工区的Registration V3不等同于已完成跨工区精度验证。",
            ),
            metadata={
                "prediction_task": "alignment",
                "release_id": "wellfuse_align_geopath_tie_v1",
                "registration_source": "wellfuse_align_geopath_tie_v1_candidate",
                "runtime_entrypoint": "prediction_task",
                "requires_seismic": True,
                "requires_registration": True,
                "requires_complete_trajectory": True,
                "requires_logs": True,
                "registration_policy": "required",
                "unknown_survey_runtime_status": "experimental_candidate",
                "checkpoint_policy": "fixed_final_epoch_geopath_full",
                "output_contract": "well-seismic.registration.v3",
                "candidate_only_by_default": True,
            },
        ),
        ModelSpec(
            id="wellfuse_horizon_p17",
            name="历史四层位追踪（既有成果→只读下载与可视化）",
            category="历史层位面成果",
            status="已归档·仅兼容既有完成任务",
            description=(
                "井插值或label-free地震事件先验与局部地震残差组合的四层位专家；"
                "未知工区输出始终标记为实验候选。"
            ),
            inputs=(
                "三维TWT SEG-Y",
                "可选固定上下文井层位控制点",
                "可选Align预测lineage",
            ),
            outputs=(
                "Hartha/Tanuma/Khasib/Zubair候选层位面",
                "Petrel XYZ",
                "三Seed近似不确定性与有效mask",
            ),
            implementation="wellfuse5090.p17_horizon_unknown",
            scientific_status="validated",
            runtime_status="precomputed_only",
            evidence_class="real_dense_horizon_surfaces_fixed_spatial_blocks",
            warnings=(
                "科学通过范围仍仅为成都固定空间块；未知工区在线输出不是已验证精度。",
                "无命名上下文层位点时使用label-free地震事件先验，绝不伪造井层位标签。",
                "模型是显式先验附近±96 ms的局部修正器，不是绝对TWT生成器。",
            ),
            metadata={
                "prediction_task": "horizon_legacy",
                "release_id": "wellfuse_p17_horizon",
                "unknown_survey_runtime_status": "archived",
                "registration_source": "wellfuse_align_prediction",
                "target_surface_is_model_input": False,
                "archived": True,
                "historical_result_compatibility": True,
            },
        ),
        ModelSpec(
            id="wellfuse_facies_1d_p17",
            name="井侧沉积相分类（九线LAS+完整轨迹→确定相序列与层段）",
            category="井侧沉积相",
            status="科学通过·OOF成果",
            description="三Seed六空间整井折的井侧沉积相预测。",
            inputs=("九线LAS与缺失掩码", "完整井轨迹", "Align预测"),
            outputs=("逐MD确定相类别CSV", "连续相层段CSV"),
            implementation="wellfuse5090.p17_facies_training",
            scientific_status="validated",
            runtime_status="runnable",
            evidence_class="real_whole_well_oof_sparse_intervals",
            metadata={
                "prediction_task": "facies_1d",
                "release_id": "wellfuse_p17_facies_1d",
                "unknown_survey_runtime_status": "experimental_runnable",
                "facies_3d_is_not_substituted": True,
                "requires_registration": True,
                "prepared_view_policy": "preferred",
                "prepared_view_consumed": True,
                "prepared_view_input_contract": "aligned_well_sequence_v1",
            },
        ),
        ModelSpec(
            id="wellfuse_facies_3d_p17",
            name="三维地震相分割（SEG-Y地震体→离散候选相体）",
            category="三维地震相候选",
            status="弱监督候选",
            description="成都真实弱监督三维地震相候选模型；不等同于定量三维真值验证。",
            inputs=("三维地震", "可选井侧相上下文"),
            outputs=("离散候选相体", "分类切片与有效掩码"),
            implementation="wellfuse5090.p17_facies_products",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="real_weakly_supervised_candidate_volume",
            warnings=(
                "现有名为full的冻结产物空间尺寸仅1×1，禁止作为成都三维相体展示。",
            ),
            metadata={
                "prediction_task": "facies_3d",
                "release_id": "wellfuse_p17_facies_3d",
                "unknown_survey_runtime_status": "experimental_runnable",
                "requires_registration": False,
                "default_inference_mode": "sample",
            },
        ),
        ModelSpec(
            id="wellfuse_facies_1d_chengdu_fast",
            name="井侧沉积相分类（封存快照井资产→确定相序列与层段）",
            category="井侧沉积相",
            status="成都数据集内已验证·可运行",
            description=(
                "冻结三Seed内部ensemble与一阶Viterbi解码器，在成都46井登记合同内"
                "输出完整MD确定相序列与连续相层段。"
            ),
            inputs=("当前封存 SourceSnapshot", "九线测井与缺失掩码", "快照轨迹/Align"),
            outputs=("逐MD确定相类别CSV", "连续相层段CSV"),
            implementation="wellfuse5090.facies1d_fast",
            scientific_status="validated",
            runtime_status="runnable",
            evidence_class="three_seed_six_spatial_whole_well_oof_viterbi",
            warnings=(
                "仅验证成都登记数据集内的46口井，不代表未知井或跨工区泛化。",
                "在线整井输出由冻结部署ensemble生成，不等同于所选井的OOF预测。",
                "当前冻结模型消费测井、轨迹与Align特征，不消费地震振幅分支。",
            ),
            metadata={
                "prediction_task": "facies_1d",
                "release_id": "wellfuse_facies_1d_chengdu_fast",
                "datasets": ["chengdu"],
                "requires_seismic": False,
                "requires_registration": False,
                "supports_raw_wells": False,
                "supports_snapshot_wells": True,
                "source_policy": "sealed_snapshot_only",
                "scope": "within_dataset",
                "selection": "fixed_three_seed_probability_plus_training_fold_viterbi",
                "active_codes": [0, 1, 2, 3, 5, 6],
            },
        ),
        ModelSpec(
            id="wellfuse_facies_3d_f3_fast",
            name="六类三维地震相分割（时间域SEG-Y→离散相体与切片）",
            category="三维地震相",
            status="公开稠密基准条件通过·跨工区迁移候选",
            description=(
                "公开稠密标签训练的多剖面卷积分割网络；对当前三维SEG-Y自动选择有效ROI，"
                "公开输出六类离散相体与分类切片。"
            ),
            inputs=("规则三维时间域SEG-Y",),
            outputs=("六类离散相体", "分类切片包", "有效掩码"),
            implementation="wellfuse5090.facies3d_fast.F3Facies25DUNet",
            scientific_status="conditional",
            runtime_status="runnable",
            evidence_class="real_dense_f3_frozen_test_once",
            warnings=(
                "定量验证范围仅为公开稠密基准；当前SEG-Y推理属于跨工区迁移候选。",
                "类别编号0至5沿用训练数据本体，不自动映射为其他工区井侧相代码。",
            ),
            metadata={
                "prediction_task": "facies_3d",
                "release_id": "wellfuse_facies_3d_f3_fast",
                "requires_seismic": True,
                "requires_registration": False,
                "validated_scope": "F3_dense_benchmark",
                "unknown_survey_runtime_status": "experimental_transfer_candidate",
                "source_formats": ["sgy", "segy"],
                "class_codes": [0, 1, 2, 3, 4, 5],
            },
        ),
        ModelSpec(
            id="wellfuse_channel_p17",
            name="河道地质体识别（SEG-Y地震体→河道概率与几何属性）",
            category="特殊地质体",
            status="候选模型·在线推理已接入",
            description="基于合成密集监督训练并在真实工区生成河道候选解释。",
            inputs=("三维地震",),
            outputs=("概率/边界/距离/骨架", "不确定性与有效掩码"),
            implementation="wellfuse5090.p17_geobody_inference",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="synthetic_dense_with_real_candidate_inference",
            warnings=("真实工区未进行定量标签验收；在线输出必须标记为候选解释。",),
            metadata={
                "prediction_task": "channel",
                "release_id": "wellfuse_p17_channel",
            },
        ),
        ModelSpec(
            id="wellfuse_karst_p17",
            name="岩溶地质体识别（SEG-Y地震体→岩溶概率与几何属性）",
            category="特殊地质体",
            status="候选模型·在线推理已接入",
            description="基于合成密集监督训练并在真实工区生成岩溶候选解释。",
            inputs=("三维地震",),
            outputs=("概率/边界/距离/连通体", "不确定性与有效掩码"),
            implementation="wellfuse5090.p17_geobody_inference",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="synthetic_dense_with_real_candidate_inference",
            warnings=("真实工区未进行定量标签验收；在线输出必须标记为候选解释。",),
            metadata={"prediction_task": "karst", "release_id": "wellfuse_p17_karst"},
        ),
        ModelSpec(
            id="wellfuse_fracture_development_utah_fast",
            name="井侧裂缝发育排序（封存快照井资产→连续发育层段）",
            category="井侧裂缝发育解释",
            status="候选模型·原始井推理已接入",
            description=(
                "以常规测井、缺失掩码和MD形成低/中/高相对发育连续深度段；"
                "当前冻结专家来自Utah FORGE 16B。"
            ),
            inputs=("当前封存 SourceSnapshot", "快照测井资产", "可选快照井轨迹"),
            outputs=("连续MD发育层段CSV", "低/中/高相对发育级别"),
            implementation="wellfuse5090.fracture_depthblock_training",
            scientific_status="candidate",
            runtime_status="runnable",
            evidence_class="single_well_purged_depth_block_candidate",
            warnings=(
                "验证范围主要为Utah 16B单井连续深度块，不代表跨区块泛化。",
                "输出是相对发育排序，不是精确裂缝点或绝对裂缝密度。",
            ),
            metadata={
                "prediction_task": "fracture_development",
                "release_id": "wellfuse_fracture_development_utah_fast",
                "requires_seismic": False,
                "requires_registration": False,
                "supports_raw_wells": False,
                "supports_snapshot_wells": True,
                "source_policy": "sealed_snapshot_only",
                "scope": "unknown_well_transfer_candidate",
                "source_expert": "utah_forge_16b",
                "output_semantics": "relative_fracture_development_intervals",
            },
        ),
        ModelSpec(
            id="wellfuse_fault_p17_failed",
            name="稀疏断层研究档案（地震体→失败审计证据）",
            category="断层研究档案",
            status="科学失败·禁止运行",
            description="成都稀疏断层棒PU训练结果退化，仅保留失败证据。",
            inputs=("三维地震",),
            outputs=("失败审计与决策记录",),
            implementation="wellfuse5090.p17_fault_pu",
            configurable=False,
            scientific_status="failed",
            runtime_status="blocked",
            evidence_class="real_sparse_fault_sticks_positive_unlabelled",
            warnings=("禁止替换或冒充现有可运行三维断层识别基线。",),
            metadata={
                "prediction_task": "fault",
                "release_id": "wellfuse_p17_fault_failed",
            },
        ),
        *(
            ModelSpec(
                id=f"wellfuse_{curve.lower()}_p18",
                name={
                    "DEN": "历史封存井曲线→DEN只读成果",
                    "POR": "历史封存井曲线→POR只读成果",
                    "LOG_PERM": "历史封存井曲线→LOG_PERM只读成果",
                    "SW": "历史封存井曲线→SW只读成果",
                    "VSH": "历史封存井曲线→VSH只读成果",
                }[curve],
                category="历史井侧物性成果",
                status=status,
                description=f"参考工区三组随机种子、六个空间整井折的{curve}冻结成果；公共平台不提供新运行。",
                inputs=(
                    "九线LAS与缺失掩码（目标及直接别名剔除）",
                    "完整井轨迹",
                    "Align预测",
                ),
                outputs=(f"{curve}预测", "不确定性/分位数", "LAS/NPZ"),
                implementation="wellfuse5090.p18_well_training",
                scientific_status=scientific_status,
                runtime_status="runnable",
                evidence_class="three_seed_six_spatial_whole_well_fold_oof",
                warnings=warnings,
                metadata={
                    "prediction_task": "well_property",
                    "public_prediction_enabled": False,
                    "public_archived": True,
                    "target_curve": curve,
                    "release_id": f"wellfuse_p18_{curve.lower()}",
                    "unknown_survey_runtime_status": "experimental_runnable",
                    "baseline_fallback_discloses_model_executed_false": True,
                    # The aligned P18 runner consumes LAS plus the sealed
                    # Registration V3 product.  SEG-Y was consumed upstream by
                    # registration and is not reopened by this subprocess.
                    "requires_seismic": False,
                    "requires_registration": True,
                    "prepared_view_policy": "preferred",
                    "prepared_view_consumed": True,
                    "prepared_view_input_contract": "aligned_well_sequence_v1",
                },
            )
            for curve, status, scientific_status, warnings in (
                ("DEN", "科学通过·OOF成果", "validated", ()),
                (
                    "POR",
                    "条件通过·斜井相对指标警告",
                    "conditional",
                    ("斜井逐井相对宏指标为负。",),
                ),
                (
                    "LOG_PERM",
                    "科学通过·局部斜井折警告",
                    "validated",
                    ("空间fold1/fold6斜井子集为负。",),
                ),
                ("SW", "科学通过·OOF成果", "validated", ()),
                (
                    "VSH",
                    "绝对MAE通过·相对增益混合",
                    "conditional",
                    ("水平井相对增益为负且不确定性失准。",),
                ),
            )
        ),
        ModelSpec(
            id="wellfuse_fluid_interpretation_fast",
            name="五类流体解释（封存快照井资产→连续确定流体层段）",
            category="流体解释",
            status="数据集内已验证·可运行",
            description=(
                "使用 fixed-last 权重解释当前封存快照中的井资产，判别 "
                "Dry、Water、Oil、Gas、Mixed，并通过可审计的最小层厚规则"
                "生成连续MD层段；概率只在进程内参与判别，不落盘公开。"
            ),
            inputs=("当前封存 SourceSnapshot", "标准九线测井", "快照井身份"),
            outputs=("连续流体层段CSV",),
            implementation="wellfuse5090.fluid_interpretation_inference",
            scientific_status="validated",
            runtime_status="runnable",
            evidence_class="fixed_last_three_fold_complete_well_oof",
            warnings=("验证范围仅限各自登记数据集，不代表跨工区泛化。",),
            metadata={
                "prediction_task": "fluid_interpretation",
                "datasets": [
                    "northwest_all",
                    "northwest_oil",
                    "northwest_coal",
                    "chengdu",
                ],
                "requires_seismic": False,
                "requires_registration": False,
                "supports_raw_wells": False,
                "supports_snapshot_wells": True,
                "source_policy": "sealed_snapshot_only",
                "scope": "within_dataset",
                "selection": "fixed_last",
            },
        ),
        *(
            ModelSpec(
                id=f"wellfuse_{target.lower()}_northwest_fast",
                name={
                    "DEN": "井侧密度预测（封存快照井资产→DEN整井曲线）",
                    "POR": "井侧孔隙度预测（封存快照井资产→POR整井曲线）",
                    "LOG_PERM": "井侧渗透率预测（封存快照井资产→LOG_PERM整井曲线）",
                    "SW": "井侧含水饱和度预测（封存快照井资产→SW整井曲线）",
                    "VSH": "井侧泥质含量预测（封存快照井资产→VSH整井曲线）",
                }[target],
                category="西北储层物性",
                status="数据集内已验证·可运行",
                description=(
                    f"使用固定权重对当前封存快照中的井资产预测 {target}。"
                ),
                inputs=("当前封存 SourceSnapshot", "标准九线测井"),
                outputs=(f"{target}整井预测", "有效掩码", "整井CSV/NPZ"),
                implementation="wellfuse5090.northwest_property_inference",
                scientific_status="validated",
                runtime_status="runnable",
                evidence_class="fixed_last_three_fold_complete_well_oof",
                warnings=("验证范围仅限登记西北数据集，不代表跨工区泛化。",),
                metadata={
                    "prediction_task": "well_property",
                    "target_curve": target,
                    "requires_seismic": False,
                    "requires_registration": False,
                    "supports_raw_wells": False,
                    "supports_snapshot_wells": True,
                    "source_policy": "sealed_snapshot_only",
                    "scope": "sealed_snapshot_whole_wells",
                    "selection": "fixed_last",
                },
            )
            for target in ("DEN", "POR", "LOG_PERM", "SW", "VSH")
        ),
        ModelSpec(
            id="wellfuse_hydrocarbon_p18",
            name="含烃指示证据（井曲线+完整轨迹→含烃概率与不确定性）",
            category="流体证据",
            status="条件通过·垂直井较弱",
            description="成都整井OOF含烃概率研究模型。",
            inputs=("井曲线", "完整井轨迹", "Align预测"),
            outputs=("含烃概率", "校准与不确定性", "证据报告"),
            implementation="wellfuse5090.p18_fluid_training",
            scientific_status="conditional",
            runtime_status="precomputed_only",
            evidence_class="three_seed_six_spatial_whole_well_fold_oof",
            warnings=("垂直井F1显著较弱；不支持八分类、水或气的独立解释。",),
            metadata={
                "prediction_task": "hydrocarbon_evidence",
                "release_id": "wellfuse_p18_hydrocarbon",
            },
        ),
        ModelSpec(
            id="wellfuse_fluid_8class_p18_failed",
            name="八类流体研究档案（井曲线→失败审计证据）",
            category="流体研究档案",
            status="科学失败·禁止运行",
            description="成都稀疏区间标签八分类任务未通过，仅保留冻结决策证据。",
            inputs=("井曲线", "完整井轨迹", "Align预测"),
            outputs=("失败审计与科学决策记录",),
            implementation="wellfuse5090.p18_fluid_training",
            configurable=False,
            scientific_status="failed",
            runtime_status="blocked",
            evidence_class="sparse_interval_labels_failed",
            warnings=(
                "supported-class macro F1仅0.2825，禁止作为流体八分类能力开放。",
            ),
            metadata={
                "prediction_task": "hydrocarbon_evidence",
                "release_id": "wellfuse_p18_fluid_8class_failed",
            },
        ),
        ModelSpec(
            id="wellfuse_water_gas_p18_failed",
            name="水气判别研究档案（井曲线→失败审计证据）",
            category="流体研究档案",
            status="科学失败·禁止运行",
            description="水与气试油关联未达到有效预测能力，仅保留冻结决策证据。",
            inputs=("井曲线", "完整井轨迹", "Align预测"),
            outputs=("失败审计与科学决策记录",),
            implementation="wellfuse5090.p18_fluid_training",
            configurable=False,
            scientific_status="failed",
            runtime_status="blocked",
            evidence_class="oil_test_association_failed",
            warnings=("Water接近随机且Gas低于随机，禁止作为预测能力开放。",),
            metadata={
                "prediction_task": "hydrocarbon_evidence",
                "release_id": "wellfuse_p18_water_gas_failed",
            },
        ),
        ModelSpec(
            id="seismic_baseline",
            name="地震单模态Baseline",
            category="基础模型",
            status="接口预留",
            description="建立不依赖测井的二维、2.5D或三维地震分割基线。",
            inputs=("地震切片或三维子体", "任务标签"),
            outputs=("地质目标概率图", "分割结果"),
        ),
        ModelSpec(
            id="well_log_encoder",
            name="测井编码器",
            category="特征编码",
            status="接口预留",
            description="编码标准LAS曲线、缺失掩码和深度位置。",
            inputs=("标准测井曲线", "深度轴", "曲线掩码"),
            outputs=("测井特征序列",),
        ),
        ModelSpec(
            id="trajectory_encoder",
            name="井轨迹与位置编码器",
            category="特征编码",
            status="接口预留",
            description="表达XYZ、MD、TVD及井到地震道距离。",
            inputs=("井轨迹", "空间坐标", "对齐置信度"),
            outputs=("空间位置特征",),
        ),
        ModelSpec(
            id="well_seismic_alignment",
            name="可学习井震对齐",
            category="井震融合",
            status="接口预留",
            description="在最近道基线基础上学习局部空间或垂向偏移。",
            inputs=("地震邻域", "测井特征", "轨迹位置"),
            outputs=("对齐特征", "偏移量", "对齐不确定性"),
        ),
        ModelSpec(
            id="confidence_gated_fusion",
            name="置信度门控融合",
            category="井震融合",
            status="内置基线",
            description="使用水平与垂向置信度控制井震特征融合。",
            inputs=("地震特征", "测井特征", "掩码与置信度"),
            outputs=("融合特征",),
            implementation="well_seismic.fusion.ConfidenceGatedFusion",
        ),
        ModelSpec(
            id="learnable_fusion",
            name="可学习井震融合",
            category="井震融合",
            status="接口预留",
            description="可接入门控网络、交叉注意力或多模态序列网络。",
            inputs=("地震编码", "测井编码", "位置编码"),
            outputs=("多模态融合特征",),
        ),
    )
    direct12b_disabled = os.getenv("WELLFUSE_DISABLE_DIRECT12B", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for spec in specs:
        if direct12b_disabled and spec.id == DIRECT12B_MODEL_ID:
            continue
        if interface_only_enabled():
            spec = annotate_interface_only_spec(spec)
        if spec.id == "confidence_gated_fusion":
            registry.register(spec, ConfidenceGatedFusion, protocol="fusion")
        else:
            registry.register(
                spec, protocol="fusion" if spec.category == "井震融合" else "model"
            )
    return registry
