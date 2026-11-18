"""Internal CPU managed tasks for hash-bound original P81 development slices."""
from pathlib import Path

from .facies_contract import MODEL_IDS, validate_options, validate_binding
from .manifest import PACKAGE_DIRECTORY, sha256_file
from .runner import _invoke_worker, _public_options

FACIES_PYTHON_ENV = "WELL_SEISMIC_FACIES_PYTHON"


def run_facies_prepared(request, *, state_path, recovery_receipt_path, sam2_source,
                        output_directory, device_name="cpu", options=None, progress=None):
    public = _public_options(options if options is not None else request.options)
    validate_options(public)
    if str(device_name).strip().casefold() not in ("", "auto", "cpu"):
        raise ValueError("P81 managed tasks use CPU only")
    if request.crop_start is not None or request.crop_size is not None:
        raise ValueError("P81 native slice tasks do not accept volume crop fields")
    source, output = Path(request.source).resolve(), Path(output_directory).resolve()
    state, receipt_path, upstream = Path(state_path).resolve(), Path(recovery_receipt_path).resolve(), Path(sam2_source).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing P81 results are never overwritten")
    binding = validate_binding(source, state, receipt_path, public)
    if progress:
        progress(10, "地震相：隔离进程执行原开发切片分割")
    worker = _invoke_worker("wellseismic_sixtrack_pilot.worker.facies_request", {
        "input_path": str(source), "state_path": str(state), "recovery_receipt_path": str(receipt_path),
        "sam2_source": str(upstream), "options": public, "output_directory": str(output / "predictions")},
        output, python_env=FACIES_PYTHON_ENV)
    for key, path in worker["outputs"].items():
        if sha256_file(Path(path)) != worker["output_sha256"][key]:
            raise RuntimeError("P81 output hash mismatch")
    if worker["query_rows"] != binding["query_rows"] or worker["num_classes"] != binding["num_classes"]:
        raise RuntimeError("P81 worker returned the wrong query or label space identity")
    if progress:
        progress(90, "地震相：已核对切片类别、样本身份和结果哈希")
    return {"model_id": MODEL_IDS[public["task_id"]], "model_name": "地震相 P81（原开发切片）",
        "model_version": "p81-development-native-slices", "device": "cpu", "scientific_status": "conditional",
        "input": {"source": str(source), "sha256": public["input_sha256"], "mode": "prepared_development_slices"},
        "outputs": {**worker["outputs"], "run_receipt_json": str(output / "worker_result.json")},
        "primary_output": "predictions_npz", "output_contract": {"contract_version": "sixtrack-facies-prepared.v1",
            "primary_output": "predictions_npz", "decision_type": "multiclass_segmentation", "query_rows": 16,
            "num_classes": worker["num_classes"], "output_axes": ["QUERY", "CLASS", "IMAGE_Y", "IMAGE_X"]},
        "warnings": ["仅原开发折128×128切片；类别空间按数据集独立。新区预处理、最终refit和授权用户工作流未完成。"],
        "provenance": {**worker, "execution": "isolated_subprocess_cpu_managed",
            "worker_source_sha256": sha256_file(PACKAGE_DIRECTORY / "worker/facies_p81.py")},
        "registration_consumed": False}


def _run_registered(model_id, request, *, output_directory, device_name="cpu", options=None,
                    threshold=None, patch_size=None, overlap=None, progress=None, **kwargs):
    import json
    from well_seismic.modeling.input_adapters import ModelInputRequest
    from .facies_plugin import DATASET_BY_MODEL, validate_request, sam2_source
    from .catalog import MODEL_VERSIONS
    from .manifest import model_dir_for, read_manifest, verify_payload, atomic_json
    public = _public_options(options if options is not None else request.options)
    validate_request({**kwargs, "options": public, "device": device_name,
        "crop_start": request.crop_start, "crop_size": request.crop_size,
        "threshold": threshold, "patch_size": patch_size, "overlap": overlap})
    directory = model_dir_for(model_id, MODEL_VERSIONS[model_id])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    dataset = DATASET_BY_MODEL[model_id]
    if (manifest["model_id"] != model_id or manifest["dataset_task_id"] != dataset
            or manifest["version"] != MODEL_VERSIONS[model_id]
            or manifest["task_id"] != "seismic_facies_slices"):
        raise ValueError("P81 manifest/model/dataset identity mismatch")
    if Path(request.source).resolve() != payload["development_cell_index"]:
        raise ValueError("P81 source must be the registered original cell index")
    index = json.loads(payload["development_cell_index"].read_text())
    if (index["schema_version"] != "well-seismic.p81-development-cells.v1"
            or index["dataset_task_id"] != dataset or index["query_targets_present"] is not False):
        raise ValueError("P81 dataset index identity mismatch")
    cell = next(c for c in index["cells"] if (c["fold_id"], c["repeat_id"]) == (public["fold_id"], public["repeat_id"]))
    entries = {e["role"]: e for e in manifest["payload"]}
    internal = {**public, "task_id": dataset,
        "input_sha256": entries[cell["query_role"]]["sha256"],
        "state_sha256": entries[cell["state_role"]]["sha256"],
        "recovery_receipt_sha256": entries[cell["recovery_role"]]["sha256"]}
    result = run_facies_prepared(ModelInputRequest(payload[cell["query_role"]], options=internal),
        state_path=payload[cell["state_role"]], recovery_receipt_path=payload[cell["recovery_role"]],
        sam2_source=sam2_source(), output_directory=output_directory, device_name=device_name, progress=progress)
    if result["provenance"]["seed"] != cell["seed"]:
        raise RuntimeError("P81 worker and registered seed mismatch")
    import hashlib
    slice_index = json.loads(Path(result["outputs"]["slice_index_json"]).read_text())
    roster_hash = hashlib.sha256(json.dumps(slice_index["sample_ids"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if roster_hash != cell["query_sample_ids_sha256"]:
        raise RuntimeError("P81 output query roster does not match the original registered cell")
    binding = {"model_id": model_id, "payload_version": manifest["version"], "dataset_task_id": dataset,
        **public, "seed": cell["seed"], "query_rows": result["provenance"]["query_rows"],
        "num_classes": result["provenance"]["num_classes"], "label_version": result["provenance"]["label_version"],
        "query_sample_ids_sha256": cell["query_sample_ids_sha256"],
        "payload_manifest_sha256": sha256_file(directory / "manifest.json"),
        "state_sha256": internal["state_sha256"], "recovery_receipt_sha256": internal["recovery_receipt_sha256"]}
    receipt_path = Path(result["outputs"]["run_receipt_json"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(platform_registered=True, platform_model_binding=binding)
    atomic_json(receipt_path, receipt)
    result.update(model_version=manifest["version"], model_name=manifest["display_name"], warnings=[manifest["claim_boundary"]])
    result["input"].update(mode="dataset_bound", development_fold_id=public["fold_id"], repeat_id=public["repeat_id"])
    result["provenance"].update(platform_registered=True, platform_model_binding=binding)
    return result


def run_registered_f3(request, **kwargs):
    return _run_registered(MODEL_IDS["facies_f3"], request, **kwargs)


def run_registered_penobscot(request, **kwargs):
    return _run_registered(MODEL_IDS["facies_penobscot"], request, **kwargs)
