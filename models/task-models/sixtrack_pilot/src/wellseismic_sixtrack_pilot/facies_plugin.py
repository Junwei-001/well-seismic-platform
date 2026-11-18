"""Independent F3/Penobscot original-development model registrations."""
import os
from pathlib import Path
import subprocess

from .catalog import MODEL_VERSIONS, _require_cpu, _reject_fields, _reject_unknown_options, _RAW_SOURCE_FIELDS, _LINEAGE_FIELDS
from .facies_contract import MODEL_IDS, TASK_CLASSES, source_lock
from .facies_runner import FACIES_PYTHON_ENV
from .manifest import model_dir_for, read_manifest, verify_payload, TASK_MODELS_ROOT
from .plugin import _DeferredAdapter, model_python

TASK_ID = "seismic_facies_slices"
SAM2_SOURCE_ENV = "WELL_SEISMIC_SAM2_SOURCE"
DATASET_BY_MODEL = {model: dataset for dataset, model in MODEL_IDS.items()}


def sam2_source():
    value = os.environ.get(SAM2_SOURCE_ENV, "")
    path = Path(value).expanduser()
    if not value or not path.is_absolute() or not (path / "sam2/build_sam.py").is_file():
        raise RuntimeError(f"{SAM2_SOURCE_ENV} must name an absolute locked SAM2 source root")
    path = path.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=path, text=True).strip()
    if revision != source_lock()["sam2_source_revision"] or Path(top).resolve() != path:
        raise RuntimeError("P81 SAM2 source revision/root mismatch")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=path).returncode:
            raise RuntimeError("P81 SAM2 source has tracked modifications")
    return path


def validate_request(request):
    _require_cpu(request)
    _reject_fields(request, ("seismic_path", *_RAW_SOURCE_FIELDS, *_LINEAGE_FIELDS),
                   "dataset-bound P81 slices do not accept raw data or snapshot lineage")
    if any(request.get(key) is not None for key in ("crop_start", "crop_size", "patch_size", "overlap", "threshold")):
        raise ValueError("crop/patch/threshold fields do not apply to native P81 slices")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, frozenset({"fold_id", "repeat_id"}))
    for key, limit in (("fold_id", 5), ("repeat_id", 3)):
        if type(options.get(key)) is not int or options[key] not in range(limit):
            raise ValueError(f"options.{key} must be an original integer 0..{limit - 1}")


def build_spec(model_id):
    from well_seismic.modeling.contracts import ModelSpec
    directory = model_dir_for(model_id, MODEL_VERSIONS[model_id])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    dataset = DATASET_BY_MODEL[model_id]
    if (manifest["model_id"] != model_id or manifest["task_id"] != TASK_ID
            or manifest["version"] != MODEL_VERSIONS[model_id] or manifest["dataset_task_id"] != dataset):
        raise ValueError("P81 manifest/model/dataset identity mismatch")
    index = next(item for item in manifest["payload"] if item["role"] == "development_cell_index")
    return ModelSpec(id=model_id, name=manifest["display_name"], category="地震相·六赛道扩展",
        status=manifest["research_policy"], description="原开发折128×128切片；完整训练权重与独立类别空间。",
        inputs=("原开发折0..4与原重复0..2",), outputs=("逐像素类别码、概率和logits", "切片身份和运行溯源"),
        version=manifest["version"], implementation="wellseismic_sixtrack_pilot.worker.facies_request",
        scientific_status="conditional", runtime_status="runnable", evidence_class="p81_dataset_bound_cpu_replay",
        warnings=(manifest["claim_boundary"],), metadata={
            "competition_track_id": "seismic_facies", "dataset_task_id": dataset,
            "num_classes": TASK_CLASSES[dataset], "label_version": manifest["label_version"],
            "research_policy": manifest["research_policy"], "stable_champion": dataset == "facies_f3",
            "extension_package": "well-seismic-sixtrack-pilot", "prediction_task": TASK_ID,
            "show_in_prediction_menu": False, "source_lineage_policy": "none", "prepared_view_policy": "none",
            "public_prediction_enabled": True, "weights_attached": True, "requires_seismic": False,
            "input_mode": "dataset_bound", "supports_raw_wells": False, "supports_new_area": False,
            "claim_boundary": manifest["claim_boundary"], "payload_version": manifest["version"],
            "payload_sha256": {e["role"]: e["sha256"] for e in manifest["payload"]},
            "evidence": manifest["evidence"], "applicability": manifest["applicability"],
            "execution": "isolated_subprocess_cpu", "submission_channel": manifest["submission_channel"],
            "payload_manifest": str((directory / "manifest.json").relative_to(TASK_MODELS_ROOT.parents[1])),
            "license": manifest["license"], "sam2_source_revision": source_lock()["sam2_source_revision"],
            "extension_source": {"kind": "registered_dataset",
                "path": str(payload["development_cell_index"].relative_to(TASK_MODELS_ROOT.parents[1])), "sha256": index["sha256"]}})


def _register(registry, dataset):
    model_python(FACIES_PYTHON_ENV)
    sam2_source()
    registry.register(build_spec(MODEL_IDS[dataset]))


def register_f3(registry):
    _register(registry, "facies_f3")


def register_penobscot(registry):
    _register(registry, "facies_penobscot")


class FaciesInputAdapter(_DeferredAdapter):
    validator = staticmethod(validate_request)

    def __init__(self, dataset):
        self.model_id = MODEL_IDS[dataset]

    def capabilities(self):
        return {"model_id": self.model_id, "source_formats": ["registered_dataset"],
            "array_axes": ["QUERY", "CHANNEL", "IMAGE_Y", "IMAGE_X"], "requires_seismic": False,
            "requires_logs": False, "patch_size": [], "supports_dataset_bound": True,
            "supports_raw_wells": False, "supports_new_area": False, "device": "cpu"}


def adapter_f3(config):
    return FaciesInputAdapter("facies_f3")


def adapter_penobscot(config):
    return FaciesInputAdapter("facies_penobscot")


def register_task(registry):
    from well_seismic.interpretation.contracts import InterpretationTaskSpec
    registry.register(InterpretationTaskSpec(id=TASK_ID, name="地震相开发切片分割", short_name="地震相",
        description="F3/Penobscot独立类别空间，原开发折二维切片分割。", outputs=("逐像素类别码与概率",),
        required_modalities=("已登记原开发切片",), evaluation_metrics=("全部配置类别开发mIoU",),
        order=73, group="seismic_facies", evidence_policy="conditional_development"))
