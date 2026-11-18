"""Registered original-development P7 T3 model and its fail-closed source adapter."""
from .catalog import (SWEETSPOT_MODEL_ID, MODEL_VERSIONS, _require_cpu, _reject_fields,
                      _reject_unknown_options, _RAW_SOURCE_FIELDS, _LINEAGE_FIELDS)
from .manifest import model_dir_for, read_manifest, verify_payload, TASK_MODELS_ROOT
from .plugin import _DeferredAdapter, model_python


def validate_request(request):
    _require_cpu(request)
    _reject_fields(request, ("seismic_path", *_RAW_SOURCE_FIELDS, *_LINEAGE_FIELDS),
                   "dataset-bound P7 T3 does not accept raw data or snapshot lineage")
    if any(request.get(key) is not None for key in ("crop_start", "crop_size", "patch_size", "overlap", "threshold")):
        raise ValueError("crop/patch/threshold fields do not apply to dataset-bound P7 T3")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, frozenset({"fold_id"}))
    if type(options.get("fold_id")) is not int or options["fold_id"] not in range(4):
        raise ValueError("options.fold_id must be an integer 0..3")



def build_spec():
    from well_seismic.modeling.contracts import ModelSpec
    directory = model_dir_for(SWEETSPOT_MODEL_ID, MODEL_VERSIONS[SWEETSPOT_MODEL_ID])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    if (manifest["model_id"] != SWEETSPOT_MODEL_ID or manifest["version"] != MODEL_VERSIONS[SWEETSPOT_MODEL_ID]
            or manifest["task_id"] != "sweetspot_productivity"):
        raise ValueError("P7 T3 manifest/model identity mismatch")
    index = next(entry for entry in manifest["payload"] if entry["role"] == "development_fold_index")
    return ModelSpec(id=SWEETSPOT_MODEL_ID, name=manifest["display_name"], category="甜点T3·六赛道扩展",
        status="P7 T3原开发折·未最终refit", description="冻结Chronos与原训练选择历史均值融合；观测索引生产预测。",
        inputs=("原开发折0..3的七通道30条因果观测历史",), outputs=("T3融合预测CSV/NPZ", "输入权重输出溯源回执"),
        version=manifest["version"], implementation="wellseismic_sixtrack_pilot.worker.sweetspot_request",
        scientific_status="conditional", runtime_status="runnable", evidence_class="p7_observation_development_cpu_replay",
        warnings=(manifest["claim_boundary"],), metadata={
            "competition_track_id": "sweetspot", "extension_package": "well-seismic-sixtrack-pilot",
            "prediction_task": "sweetspot_productivity", "show_in_prediction_menu": False,
            "source_lineage_policy": "none", "prepared_view_policy": "none",
            "public_prediction_enabled": True, "weights_attached": True, "requires_seismic": False,
            "input_mode": "dataset_bound", "supports_raw_wells": False, "supports_new_area": False,
            "claim_boundary": manifest["claim_boundary"], "payload_version": manifest["version"],
            "payload_sha256": {entry["role"]: entry["sha256"] for entry in manifest["payload"]},
            "evidence": manifest["evidence"], "applicability": manifest["applicability"],
            "execution": "isolated_subprocess_cpu", "submission_channel": manifest["submission_channel"],
            "payload_manifest": str((directory / "manifest.json").relative_to(TASK_MODELS_ROOT.parents[1])),
            "license": manifest["license"], "extension_source": {"kind": "registered_dataset",
            "path": str(payload["development_fold_index"].relative_to(TASK_MODELS_ROOT.parents[1])), "sha256": index["sha256"]}})


def register(registry):
    model_python("WELL_SEISMIC_CHRONOS_PYTHON")
    registry.register(build_spec())


class SweetspotInputAdapter(_DeferredAdapter):
    model_id = SWEETSPOT_MODEL_ID
    validator = staticmethod(validate_request)

    def capabilities(self):
        return {"model_id": self.model_id, "source_formats": ["registered_dataset"], "array_axes": ["QUERY"],
            "requires_seismic": False, "requires_logs": False, "patch_size": [], "supports_dataset_bound": True,
            "supports_raw_wells": False, "supports_new_area": False, "device": "cpu"}


def adapter(config):
    return SweetspotInputAdapter()


def register_task(registry):
    from well_seismic.interpretation.contracts import InterpretationTaskSpec
    registry.register(InterpretationTaskSpec(id="sweetspot_productivity", name="甜点T3生产预测", short_name="甜点T3",
        description="P7原开发历史生产预测；观测索引，不保证连续日历天。", outputs=("T3融合预测",),
        required_modalities=("已登记因果生产历史",), evaluation_metrics=("原开发折MAE",),
        order=72, group="sweetspot", evidence_policy="conditional_development"))
