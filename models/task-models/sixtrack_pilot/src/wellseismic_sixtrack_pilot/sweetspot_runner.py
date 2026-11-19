"""Internal P7 managed query runner; public authorization/registration are separate."""
from pathlib import Path

from .manifest import PACKAGE_DIRECTORY, sha256_file
from .runner import _invoke_worker, _public_options

CHRONOS_PYTHON_ENV = "WELL_SEISMIC_CHRONOS_PYTHON"


def run_sweetspot_prepared(request, *, snapshot_path: Path, output_directory: Path,
                          device_name="cpu", options=None, progress=None):
    options = _public_options(options if options is not None else request.options)
    if set(options) != {"input_sha256", "fold_id"}:
        raise ValueError("P7 requires only input_sha256/fold_id")
    checksum, fold = options["input_sha256"], options["fold_id"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("P7 input_sha256 must be 64 lowercase hex characters")
    if type(fold) is not int or fold not in range(4):
        raise ValueError("P7 requires original development fold 0..3")
    if str(device_name).casefold().strip() not in {"auto", "cpu", ""}:
        raise ValueError("P7 managed worker runs on CPU only")
    if request.crop_start is not None or request.crop_size is not None:
        raise ValueError("P7 history queries do not accept volume crop fields")
    source, output = Path(request.source).resolve(), Path(output_directory).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing results are never overwritten")
    if sha256_file(source) != checksum:
        raise ValueError("P7 input hash mismatch")
    if progress:
        progress(10, "甜点 T3：隔离进程执行冻结 Chronos 与历史均值融合")
    worker = _invoke_worker("wellseismic_sixtrack_pilot.worker.sweetspot_request", {
        "input_path": str(source), "input_sha256": checksum, "snapshot_path": str(Path(snapshot_path).resolve()),
        "fold_id": fold, "output_directory": str(output / "predictions")}, output, python_env=CHRONOS_PYTHON_ENV)
    for key, path in worker["outputs"].items():
        if sha256_file(Path(path)) != worker["output_sha256"][key]:
            raise RuntimeError("P7 output hash mismatch")
    if progress:
        progress(90, "甜点 T3：已核对融合预测与中文表身份")
    return {"model_id": "sixtrack_sweetspot_p7_t3", "model_name": "甜点 T3 P7（观测索引生产预测）",
        "model_version": "p7-development-train-blend", "device": "cpu", "scientific_status": "conditional",
        "input": {"source": str(source), "sha256": checksum, "mode": "prepared_production_history"},
        "outputs": {**worker["outputs"], "run_receipt_json": str(output / "worker_result.json")},
        "primary_output": "predictions_csv", "output_contract": {
            "contract_version": "sixtrack-sweetspot-prepared.v1", "primary_output": "predictions_csv",
            "decision_type": "regression", "query_rows": worker["query_rows"],
            "time_semantics": worker["time_semantics"]},
        "warnings": ["仅 T3 原开发折观测索引预测；不是连续30天日历预测或空间甜点图。最终 refit、新井授权和其他甜点目标未完成。"],
        "provenance": {**worker, "execution": "isolated_subprocess_cpu_managed",
            "worker_source_sha256": sha256_file(PACKAGE_DIRECTORY / "worker/sweetspot_p7.py")},
        "registration_consumed": False}


def run_registered(request, *, output_directory, device_name="cpu", options=None,
                   threshold=None, patch_size=None, overlap=None, progress=None, **kwargs):
    import json
    from well_seismic.modeling.input_adapters import ModelInputRequest
    from .catalog import SWEETSPOT_MODEL_ID, MODEL_VERSIONS
    from .sweetspot_plugin import validate_request
    from .manifest import model_dir_for, read_manifest, verify_payload, atomic_json
    public = _public_options(options if options is not None else request.options)
    validate_request({**kwargs, "options": public, "device": device_name, "crop_start": request.crop_start,
        "crop_size": request.crop_size, "threshold": threshold, "patch_size": patch_size, "overlap": overlap})
    directory = model_dir_for(SWEETSPOT_MODEL_ID, MODEL_VERSIONS[SWEETSPOT_MODEL_ID])
    manifest, payload = read_manifest(directory), verify_payload(directory)
    if Path(request.source).resolve() != payload["development_fold_index"]:
        raise ValueError("P7 source must be the registered development fold index")
    index = json.loads(payload["development_fold_index"].read_text())
    if index["schema_version"] != "well-seismic.p7-development-folds.v1" or index["query_targets_present"] is not False:
        raise ValueError("P7 development index contract mismatch")
    fold = next(row for row in index["folds"] if row["fold_id"] == public["fold_id"])
    entry = next(row for row in manifest["payload"] if row["role"] == fold["payload_role"])
    result = run_sweetspot_prepared(ModelInputRequest(payload[fold["payload_role"]], options={
        "input_sha256": entry["sha256"], **public}), snapshot_path=payload["chronos_weights"].parent,
        output_directory=output_directory, device_name=device_name, progress=progress)
    if (result["provenance"]["chronos_weight"] != fold["chronos_weight"]
            or result["provenance"]["query_rows"] != fold["query_rows"]):
        raise RuntimeError("P7 worker result and registered fold identity mismatch")
    binding = {"model_id": SWEETSPOT_MODEL_ID, "payload_version": manifest["version"],
        "fold_id": public["fold_id"], "query_sample_ids_sha256": fold["validation_sample_ids_sha256"],
        "payload_manifest_sha256": sha256_file(directory / "manifest.json"),
        "query_rows": fold["query_rows"], "chronos_weight": fold["chronos_weight"]}
    receipt_path = Path(result["outputs"]["run_receipt_json"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(platform_registered=True, platform_model_binding=binding)
    atomic_json(receipt_path, receipt)
    result.update(model_version=manifest["version"], model_name=manifest["display_name"], warnings=[manifest["claim_boundary"]])
    result["input"].update(mode="dataset_bound", development_fold_id=public["fold_id"])
    result["provenance"].update(platform_registered=True, platform_model_binding=binding)
    return result
