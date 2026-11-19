"""Internal P76 managed runner; public registration and source authorization remain separate."""
from pathlib import Path

from .manifest import PACKAGE_DIRECTORY, sha256_file, model_dir_for, read_manifest, verify_payload, atomic_json
from .runner import _invoke_worker, _public_options


def run_lithofacies_prepared(request, *, checkpoint_path: Path, output_directory: Path,
                             device_name="cpu", options=None, progress=None):
    """Caller authorizes both paths; content and fixed scientific options fail closed."""
    options = _public_options(options if options is not None else request.options)
    if set(options) != {"input_sha256", "fold_id", "repeat_id"}:
        raise ValueError("P76 requires only input_sha256/fold_id/repeat_id")
    checksum, fold, repeat = options["input_sha256"], options["fold_id"], options["repeat_id"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("P76 input_sha256 must be 64 lowercase hex characters")
    if type(fold) is not int or fold not in range(3) or type(repeat) is not int or repeat not in (3, 4, 5):
        raise ValueError("P76 requires original development fold 0..2 and confirmation repeat 3..5")
    if str(device_name).casefold().strip() not in {"auto", "cpu", ""}:
        raise ValueError("P76 managed worker runs on CPU only")
    if request.crop_start is not None or request.crop_size is not None:
        raise ValueError("P76 aligned windows do not accept volume crop fields")
    source, output = Path(request.source).resolve(), Path(output_directory).resolve()
    if output.exists(): raise ValueError("Use a new output directory; existing results are never overwritten")
    if sha256_file(source) != checksum: raise ValueError("P76 input hash mismatch")
    if progress: progress(10, "岩相 P76：隔离进程执行冻结概率集成")
    worker = _invoke_worker("wellseismic_sixtrack_pilot.worker.lithofacies_request", {
        "input_path": str(source), "input_sha256": checksum,
        "checkpoint_path": str(Path(checkpoint_path).resolve()), "fold_id": fold, "repeat_id": repeat,
        "output_directory": str(output / "predictions"),
    }, output)
    for key, path in worker["outputs"].items():
        if sha256_file(Path(path)) != worker["output_sha256"][key]:
            raise RuntimeError("P76 output hash mismatch")
    if progress: progress(90, "岩相 P76：推理完成，已核对结果与概率表身份")
    return {"model_id": "sixtrack_lithofacies_p76", "model_name": "岩相 P76（开发折固定概率集成）",
        "model_version": "p76-development-e2e8", "device": "cpu", "scientific_status": "conditional",
        "input": {"source": str(source), "sha256": checksum, "mode": "prepared_windows"},
        "outputs": {**worker["outputs"], "run_receipt_json": str(output / "worker_result.json")},
        "prepared_input_contract": worker["prepared_input_contract"],
        "primary_output": "predictions_csv", "output_contract": {
            "contract_version": "sixtrack-lithofacies-prepared.v1", "primary_output": "predictions_csv",
            "decision_type": "classification", "query_rows": worker["query_rows"], "class_codes": list(range(9))},
        "warnings": ["仅原开发折验证；最终 refit、新井预处理、类别名称溯源和多用户网页尚未完成。"],
        "provenance": {**worker, "execution": "isolated_subprocess_cpu_managed",
            "worker_source_sha256": sha256_file(PACKAGE_DIRECTORY / "worker/lithofacies_p76.py")},
        "registration_consumed": False}


def run_registered(request, *, output_directory, device_name="cpu", options=None,
                   threshold=None, patch_size=None, overlap=None, progress=None, **kwargs):
    import json
    from well_seismic.modeling.input_adapters import ModelInputRequest
    from .catalog import LITHOFACIES_MODEL_ID, MODEL_VERSIONS
    from .lithofacies_plugin import validate_request
    public = _public_options(options if options is not None else request.options)
    validate_request({**kwargs, "options": public, "device": device_name, "crop_start": request.crop_start,
        "crop_size": request.crop_size, "threshold": threshold, "patch_size": patch_size, "overlap": overlap})
    directory = model_dir_for(LITHOFACIES_MODEL_ID, MODEL_VERSIONS[LITHOFACIES_MODEL_ID])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    if Path(request.source).resolve() != payload["development_fold_index"]:
        raise ValueError("P76 source must be the registered development fold index")
    index = json.loads(payload["development_fold_index"].read_text())
    if index["schema_version"] != "well-seismic.p76-development-folds.v1" or index["query_targets_present"] is not False:
        raise ValueError("P76 development index contract mismatch")
    fold = next(row for row in index["folds"] if row["fold_id"] == public["fold_id"])
    entry = next(row for row in manifest["payload"] if row["role"] == fold["payload_role"])
    result = run_lithofacies_prepared(ModelInputRequest(payload[fold["payload_role"]], options={
        "input_sha256": entry["sha256"], **public}), checkpoint_path=payload["tabicl_classifier"],
        output_directory=output_directory, device_name=device_name, progress=progress)
    binding = {"model_id": LITHOFACIES_MODEL_ID, "payload_version": manifest["version"],
        "fold_id": public["fold_id"], "repeat_id": public["repeat_id"], "partition_hash": fold["partition_hash"],
        "payload_manifest_sha256": sha256_file(directory / "manifest.json"),
        "query_rows": fold["query_rows"], "support_rows": fold["train_rows"]}
    receipt_path = Path(result["outputs"]["run_receipt_json"])
    receipt = json.loads(receipt_path.read_text());receipt.update(platform_registered=True,platform_model_binding=binding)
    atomic_json(receipt_path, receipt)
    result.update(model_version=manifest["version"], model_name=manifest["display_name"], warnings=[manifest["claim_boundary"]])
    result["input"].update(mode="dataset_bound", development_fold_id=public["fold_id"])
    result["provenance"].update(platform_registered=True, platform_model_binding=binding)
    return result
