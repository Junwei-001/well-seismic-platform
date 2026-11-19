"""Registered original-development P76 model and its fail-closed source adapter."""
from .catalog import (LITHOFACIES_MODEL_ID, MODEL_VERSIONS, _require_cpu, _reject_fields,
                      _reject_unknown_options, _RAW_SOURCE_FIELDS, _LINEAGE_FIELDS)
from .manifest import model_dir_for, read_manifest, verify_payload, TASK_MODELS_ROOT
from .plugin import _DeferredAdapter, model_python


def validate_request(request):
    _require_cpu(request)
    _reject_fields(request, ("seismic_path", *_RAW_SOURCE_FIELDS, *_LINEAGE_FIELDS),
                   "dataset-bound P76 does not accept raw data or snapshot lineage")
    if any(request.get(key) is not None for key in ("crop_start", "crop_size", "patch_size", "overlap", "threshold")):
        raise ValueError("crop/patch/threshold fields do not apply to dataset-bound P76")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, frozenset({"fold_id", "repeat_id"}))
    if type(options.get("fold_id")) is not int or options["fold_id"] not in range(3):
        raise ValueError("options.fold_id must be an integer 0..2")
    if type(options.get("repeat_id")) is not int or options["repeat_id"] not in (3, 4, 5):
        raise ValueError("options.repeat_id must be a confirmation integer 3..5")


def build_spec():
    from well_seismic.modeling.contracts import ModelSpec
    directory = model_dir_for(LITHOFACIES_MODEL_ID, MODEL_VERSIONS[LITHOFACIES_MODEL_ID])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    if (manifest["model_id"] != LITHOFACIES_MODEL_ID or manifest["version"] != MODEL_VERSIONS[LITHOFACIES_MODEL_ID]
            or manifest["task_id"] != "lithofacies_classification"):
        raise ValueError("P76 manifest/model identity mismatch")
    index = next(entry for entry in manifest["payload"] if entry["role"] == "development_fold_index")
    return ModelSpec(id=LITHOFACIES_MODEL_ID, name=manifest["display_name"], category="岩相·六赛道扩展",
        status="P76开发冠军·未最终refit", description="固定0.25e2+0.75e8概率集成，仅运行原开发折九类岩相窗口。",
        inputs=("原开发折0..2，确认重复3..5",), outputs=("九类代码与概率CSV/NPZ", "输入权重输出溯源回执"),
        version=manifest["version"], implementation="wellseismic_sixtrack_pilot.worker.lithofacies_request",
        scientific_status="conditional", runtime_status="runnable", evidence_class="p76_development_cpu_replay",
        warnings=(manifest["claim_boundary"],), metadata={
            "competition_track_id": "lithofacies", "extension_package": "well-seismic-sixtrack-pilot",
            "prediction_task": "lithofacies_classification", "show_in_prediction_menu": False,
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
    model_python()
    registry.register(build_spec())


class LithofaciesInputAdapter(_DeferredAdapter):
    model_id = LITHOFACIES_MODEL_ID
    validator = staticmethod(validate_request)

    def capabilities(self):
        return {"model_id": self.model_id, "source_formats": ["registered_dataset"], "array_axes": ["QUERY"],
            "requires_seismic": False, "requires_logs": False, "patch_size": [], "supports_dataset_bound": True,
            "supports_raw_wells": False, "supports_new_area": False, "device": "cpu"}


def adapter(config):
    return LithofaciesInputAdapter()


def register_task(registry):
    from well_seismic.interpretation.contracts import InterpretationTaskSpec
    registry.register(InterpretationTaskSpec(id="lithofacies_classification", name="岩相窗口分类", short_name="岩相识别",
        description="原开发窗口固定九类分类；未提供新井预处理或类别名称映射。", outputs=("九类代码与概率",),
        required_modalities=("已登记开发窗口",), evaluation_metrics=("开发集九类Macro-F1",),
        order=71, group="lithofacies", evidence_policy="conditional_development"))
