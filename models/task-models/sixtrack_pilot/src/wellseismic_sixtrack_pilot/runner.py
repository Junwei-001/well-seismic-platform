"""Prediction runners: hand the request to the isolated worker, then shape the result.

The worker runs with the dedicated model interpreter, ``CUDA_VISIBLE_DEVICES=""``
(CPU only, no GPU is ever touched), user site-packages disabled and HF offline.
It is started through the platform's cancellation-aware ``managed_run`` so the
platform's task cancellation also stops the child process.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from well_seismic.modeling.input_adapters import ModelInputRequest
from well_seismic.task_runtime import managed_run

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
from .manifest import PACKAGE_DIRECTORY, atomic_json, model_dir_for, read_manifest, sha256_file, verify_payload
from .plugin import model_python

WORKER_TIMEOUT_SECONDS = 1800
WORKER_THREADS = "8"
_PLATFORM_INJECTED_OPTIONS = frozenset({"prediction_source_identity"})

_WORKER_BOOTSTRAP = r'''
import hashlib, importlib.util, json, pathlib, runpy, sys
expected = json.loads(sys.argv[1])
identity_path = pathlib.Path(sys.argv[2])
module = sys.argv[3]
spec = importlib.util.find_spec("wellseismic_sixtrack_pilot")
if spec is None or spec.origin is None:
    raise RuntimeError("Six-track worker package is not installed")
root = pathlib.Path(spec.origin).resolve().parent
actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json")}
if actual != expected:
    raise RuntimeError("Six-track worker package identity mismatch; prediction was not started")
identity = {"schema_version": "well-seismic.worker-package-identity.v1",
    "status": "verified_before_worker_import", "package_directory": str(root),
    "file_count": len(actual), "files": actual,
    "content_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()}
with identity_path.open("x") as stream:
    json.dump(identity, stream, sort_keys=True)
    stream.write("\n")
sys.argv = [module, *sys.argv[4:]]
runpy.run_module(module, run_name="__main__")
'''


def package_file_bindings(directory: Path) -> dict[str, str]:
    """Bind package code and packaged data, not interpreter libraries or weights."""
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in (".py", ".json")}


def _public_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in dict(options or {}).items() if key not in _PLATFORM_INJECTED_OPTIONS}


def _worker_env() -> dict[str, str]:
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot")
        if not system_root or not Path(system_root).is_dir():
            raise RuntimeError("Windows SystemRoot is unavailable for the isolated model worker")
        system_path = os.pathsep.join((str(Path(system_root) / "System32"), system_root))
    else:
        system_root = None
        system_path = "/usr/local/bin:/usr/bin:/bin"
    env = {
        "PATH": system_path,
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": WORKER_THREADS,
        "MKL_NUM_THREADS": WORKER_THREADS,
        "OPENBLAS_NUM_THREADS": WORKER_THREADS,
        "HF_HUB_OFFLINE": "1",
    }
    if system_root is not None:
        env["SystemRoot"] = system_root
        env["WINDIR"] = system_root
    # Source development can share this package's src. An installed platform
    # wheel requires the same wheel in the worker interpreter; never expose
    # the platform's entire site-packages as a worker dependency directory.
    if PACKAGE_DIRECTORY.parent.name == "src":
        env["PYTHONPATH"] = str(PACKAGE_DIRECTORY.parent)
    return env


def _invoke_worker(module: str, payload: dict[str, Any], output_directory: Path,
                   *, python_env: str | None = None) -> dict[str, Any]:
    interpreter = model_python(python_env) if python_env else model_python()
    output_directory.mkdir(parents=True, exist_ok=True)
    request_path = output_directory / "worker_request.json"
    result_path = output_directory / "worker_result.json"
    identity_path = output_directory / "worker_package_identity.json"
    expected_package = package_file_bindings(PACKAGE_DIRECTORY)
    atomic_json(request_path, payload)
    completed = managed_run(
        [str(interpreter), "-B", "-c", _WORKER_BOOTSTRAP,
         json.dumps(expected_package, sort_keys=True), str(identity_path), module,
         "--request", str(request_path), "--result", str(result_path)],
        env=_worker_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if not result_path.is_file():
        tail = (completed.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError(f"{module} exited {completed.returncode} without a result: {' | '.join(tail)}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 or result.get("status") != "completed":
        raise RuntimeError(
            f"{module} failed: {result.get('error_type', 'error')}: {result.get('error', completed.returncode)}"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("files") != expected_package or identity.get("status") != "verified_before_worker_import":
        raise RuntimeError("Worker package identity receipt does not match this invocation")
    result["verified_worker_package"] = identity
    return result


def _manifest_context(model_id: str) -> tuple[Path, dict[str, Any]]:
    model_dir = model_dir_for(model_id, MODEL_VERSIONS[model_id])
    return model_dir, read_manifest(model_dir)


def _checkpoint_records(model_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": str(model_dir / entry["path"]), "sha256": entry["sha256"], "stage": entry["role"]}
        for entry in manifest["payload"]
    ]


def _provenance(model_dir: Path, manifest: Mapping[str, Any], worker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "extension_package": "well-seismic-sixtrack-pilot",
        "payload_manifest": str(model_dir / "manifest.json"),
        "payload_version": manifest["version"],
        "evidence": manifest["evidence"],
        "claim_boundary": manifest["claim_boundary"],
        "worker_runtime": worker.get("runtime"),
        "verified_worker_package": worker.get("verified_worker_package"),
    }


def _progress(callback: Callable[[int, str], None] | None, value: int, message: str) -> None:
    if callback is not None:
        callback(value, message)


def run_fault_prediction(
    request: ModelInputRequest,
    *,
    output_directory: Path,
    threshold: float | None = None,
    device_name: str = "auto",
    options: Mapping[str, Any] | None = None,
    progress: Callable[[int, str], None] | None = None,
    **_: Any,
) -> dict[str, Any]:
    public_options = _public_options(options if options is not None else request.options)
    validate_fault_request(
        {
            "device": device_name,
            "options": public_options,
            "crop_start": request.crop_start,
            "crop_size": request.crop_size,
        }
    )
    model_dir, manifest = _manifest_context(FAULT_MODEL_ID)
    seismic_path = Path(request.source).resolve()
    _progress(progress, 10, "六赛道断层基线：隔离进程读取 SEG-Y 并推理")
    worker = _invoke_worker(
        "wellseismic_sixtrack_pilot.worker.fault_logistic",
        {
            "model_dir": str(model_dir),
            "output_directory": str(Path(output_directory).resolve()),
            "seismic_path": str(seismic_path),
            "threshold": threshold,
            "crop_start": list(request.crop_start) if request.crop_start is not None else None,
            "crop_size": list(request.crop_size) if request.crop_size is not None else None,
            "options": {key: value for key, value in public_options.items() if key != "axis_order"},
        },
        Path(output_directory),
    )
    _progress(progress, 90, "六赛道断层基线：推理完成，正在封存结果")
    grid, summary = worker["grid"], worker["summary"]
    outputs = dict(worker["outputs"])
    outputs["run_receipt_json"] = str(Path(output_directory) / "worker_result.json")
    return {
        "model_id": FAULT_MODEL_ID,
        "model_name": manifest["display_name"],
        "model_version": manifest["version"],
        "device": "cpu",
        "input": {
            "source": str(seismic_path),
            "axes": list(FAULT_AXES),
            "crop_start_zyx": grid["crop_start_zyx"],
            "crop_size_zyx": grid["shape_zyx"],
            "shape_zyx": grid["shape_zyx"],
            "source_shape_zyx": [
                grid["survey_samples_per_trace"],
                grid["survey_inline_range"][1] - grid["survey_inline_range"][0] + 1,
                grid["survey_crossline_range"][1] - grid["survey_crossline_range"][0] + 1,
            ],
            "inline_range": grid["inline_range"],
            "crossline_range": grid["crossline_range"],
            "sample_range": grid["sample_range"],
            "amplitude": "raw float32, no normalisation",
        },
        "inference": {
            "threshold": summary["threshold"],
            "threshold_source": summary["threshold_source"],
            "execution": "isolated_subprocess_cpu",
            "class_codes": [0, 1],
        },
        "classes": [
            {"code": 0, "name": "background", "role": "background"},
            {"code": 1, "name": "fault", "role": "class"},
        ],
        "summary": summary,
        "outputs": outputs,
        "primary_output": "mask_npy",
        "diagnostic_output": "probability_npy",
        "output_contract": {
            "contract_version": "sixtrack-fault-volume.v1",
            "primary_output": "mask_npy",
            "decision_type": "binary_threshold",
            "axis_order": list(FAULT_AXES),
            "probability_output": "probability_npy",
        },
        "checkpoints": _checkpoint_records(model_dir, manifest),
        "scientific_status": "candidate",
        "warnings": [
            "轻量基线/仅开发集验证：audited_v2 测试 F1≈0.016，掩码正类比例偏高；概率体仅供技术审计。",
            f"本次掩码正类比例 {summary['mask_positive_fraction']:.3f}。",
        ],
        "provenance": _provenance(model_dir, manifest, worker),
        "registration_consumed": False,
    }


def _property_tie_warning(contract: Mapping[str, Any]) -> str:
    tie = contract.get("well_tie")
    if isinstance(tie, Mapping):
        rows, outside = tie.get("query_rows"), tie.get("extrapolated_query_rows")
        if (tie.get("source_archive_sha256") ==
                "9b2e00389446802f7306a2798c3d2f216251eb7744b6c7ad61e1f3e242bcb347"
                and tie.get("method") == "official_surface_picks_piecewise_linear_md"
                and tie.get("independent_accuracy_verified") is False
                and type(rows) is int and type(outside) is int
                and type(contract.get("rows")) is int and rows == contract["rows"]
                and 0 < rows <= 256 and 0 <= outside <= rows):
            return (f"井震位置由官方层位拾取点分段线性换算；当前 {rows} 个查询样点中 "
                    f"{outside} 个超出拾取深度范围，使用端点外推。标定精度尚未独立验收。")
    return "查询样点的井震外推范围未核实；井震标定精度尚未独立验收。"


def run_property_prediction(
    request: ModelInputRequest,
    *,
    model_id: str,
    output_directory: Path,
    device_name: str = "auto",
    options: Mapping[str, Any] | None = None,
    progress: Callable[[int, str], None] | None = None,
    **_: Any,
) -> dict[str, Any]:
    public_options = _public_options(options if options is not None else request.options)
    validate_property_request({"device": device_name, "options": public_options})
    target = PROPERTY_MODEL_IDS[model_id]
    model_dir, manifest = _manifest_context(model_id)
    _progress(progress, 10, f"六赛道物性 {target}：隔离进程运行冻结 TabICL")
    worker = _invoke_worker(
        "wellseismic_sixtrack_pilot.worker.property_tabicl",
        {
            "model_dir": str(model_dir),
            "output_directory": str(Path(output_directory).resolve()),
            "target": target,
            "options": public_options,
        },
        Path(output_directory),
    )
    _progress(progress, 90, f"六赛道物性 {target}：推理完成，正在封存结果")
    outputs = dict(worker["outputs"])
    outputs["run_receipt_json"] = str(Path(output_directory) / "worker_result.json")
    return {
        "model_id": model_id,
        "model_name": manifest["display_name"],
        "model_version": manifest["version"],
        "device": "cpu",
        "input": {
            "source": str(Path(request.source).resolve()),
            "mode": "dataset_bound",
            "query_family": public_options["query_family"],
            "fold": worker["fold"],
        },
        "inference": {
            "target": target,
            "unit": worker["unit"],
            "seed": worker["seed"],
            "random_state": worker["random_state"],
            "n_estimators": worker["runtime"]["n_estimators"],
            "execution": "isolated_subprocess_cpu",
        },
        "metrics": worker["metrics"],
        "outputs": outputs,
        "primary_output": "predictions_csv",
        "prepared_input_contract": worker["prepared_input_contract"],
        "output_contract": {
            "contract_version": "sixtrack-property-prediction.v1",
            "primary_output": "predictions_csv",
            "decision_type": "regression",
            "target": target,
            "unit": worker["unit"],
        },
        "checkpoints": _checkpoint_records(model_dir, manifest),
        "scientific_status": "conditional",
        "warnings": [
            "仅开发集 LOGO4 验证；参考标签为 CPI 解释值。",
            "档案绑定模式：新井预测需要未启用的“全量特征构建”进阶能力。",
            _property_tie_warning(worker["prepared_input_contract"]),
        ],
        "provenance": _provenance(model_dir, manifest, worker)
        | {"prediction_model_domain_sha256": worker["prediction_model_domain_sha256"]},
        "registration_consumed": False,
    }


def _property_runner(model_id: str) -> Callable[..., dict[str, Any]]:
    def runner(request: ModelInputRequest, **kwargs: Any) -> dict[str, Any]:
        return run_property_prediction(request, model_id=model_id, **kwargs)

    runner.__name__ = f"run_{model_id}"
    return runner


def run_reconstruction_prepared(
    request: ModelInputRequest,
    *,
    output_directory: Path,
    device_name: str = "auto",
    options: Mapping[str, Any] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Use the real managed worker contract; no public model registration yet.

    The caller must provide an authorized prepared NPZ and its pinned hash.
    New-area preprocessing and a platform payload manifest remain separate work.
    """
    public_options = _public_options(options if options is not None else request.options)
    if str(device_name).strip().casefold() not in {"auto", "cpu", ""}:
        raise ValueError("P21 prepared inference runs on CPU only")
    if request.crop_start is not None or request.crop_size is not None:
        raise ValueError("P21 prepared input does not accept volume crop fields")
    checksum = public_options.get("input_sha256")
    if (
        set(public_options) != {"input_sha256"}
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("P21 requires only options.input_sha256 (64 lowercase hex characters)")
    source = Path(request.source).resolve()
    if sha256_file(source) != checksum:
        raise ValueError("P21 prepared input hash mismatch")
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise ValueError("Use a new output directory; existing results are never overwritten")
    _progress(progress, 10, "三维重建 P21：隔离进程处理已准备输入")
    worker = _invoke_worker(
        "wellseismic_sixtrack_pilot.worker.reconstruction_request",
        {
            "input_path": str(source), "input_sha256": checksum,
            "output_directory": str(output_directory / "predictions"),
        },
        output_directory,
    )
    prediction_path = Path(worker["outputs"]["predictions_npz"])
    if sha256_file(prediction_path) != worker["output_sha256"]:
        raise RuntimeError("P21 output hash mismatch")
    _progress(progress, 90, "三维重建 P21：推理完成，已核对输出哈希")
    return {
        "model_id": "sixtrack_reconstruction_p21",
        "model_name": "三维重建 P21（已准备输入·开发集验证）",
        "model_version": "p21-fixed-three-kernel",
        "device": "cpu",
        "input": {"source": str(source), "mode": "prepared_input", "sha256": checksum},
        "outputs": {**worker["outputs"], "run_receipt_json": str(output_directory / "worker_result.json")},
        "primary_output": "predictions_npz",
        "output_contract": {
            "contract_version": "sixtrack-reconstruction-prepared.v1",
            "primary_output": "predictions_npz", "decision_type": "regression",
            "axes": ["QUERY"], "query_rows": worker["query_rows"],
        },
        "scientific_status": "conditional",
        "warnings": ["仅验证原始开发集条件重建；新区域特征准备和实际 PyKrige 基线计算尚未接入。"],
        "provenance": {
            "prepared_input_sha256": checksum, "output_sha256": worker["output_sha256"],
            "execution": "isolated_subprocess_cpu_managed",
            "query_targets_consumed": False, "cached_pykrige_baseline_consumed": True,
            "platform_registered": False, "new_area_input_supported": False,
            "worker_source_sha256": sha256_file(PACKAGE_DIRECTORY / "worker/reconstruction_p21.py"),
            "verified_worker_package": worker.get("verified_worker_package"),
        },
        "registration_consumed": False,
    }


def run_reconstruction_prediction(
    request: ModelInputRequest, *, output_directory: Path,
    device_name: str = "auto", options: Mapping[str, Any] | None = None,
    threshold: float | None = None, patch_size: Any = None, overlap: Any = None,
    progress: Callable[[int, str], None] | None = None, **_: Any,
) -> dict[str, Any]:
    """Registered dataset-bound path; select one verified original development fold."""
    public_options = _public_options(options if options is not None else request.options)
    validate_reconstruction_request({"device": device_name, "options": public_options,
        "crop_start": request.crop_start, "crop_size": request.crop_size,
        "threshold": threshold, "patch_size": patch_size, "overlap": overlap})
    model_dir, manifest = _manifest_context(RECONSTRUCTION_MODEL_ID)
    payload = verify_payload(model_dir)
    index_path = payload["development_fold_index"]
    if Path(request.source).resolve() != index_path:
        raise ValueError("P21 source must be the registered development fold index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["schema_version"] != "well-seismic.p21-development-folds.v1" or index["query_targets_present"] is not False:
        raise ValueError("P21 development index contract mismatch")
    fold_id = public_options["fold_id"]
    fold = next(item for item in index["folds"] if item["fold_id"] == fold_id)
    entries = {entry["role"]: entry for entry in manifest["payload"]}
    selected = entries[fold["payload_role"]]
    result = run_reconstruction_prepared(
        ModelInputRequest(payload[fold["payload_role"]], options={"input_sha256": selected["sha256"]}),
        output_directory=output_directory, device_name=device_name, progress=progress,
    )
    binding = {
        "model_id": RECONSTRUCTION_MODEL_ID, "payload_version": manifest["version"],
        "fold_id": fold_id, "query_rows": fold["query_rows"], "support_rows": fold["train_rows"],
        "train_indices_sha256": fold["train_indices_sha256"],
        "query_indices_sha256": fold["query_indices_sha256"],
        "payload_manifest_sha256": sha256_file(model_dir / "manifest.json"),
        "development_fold_index_sha256": entries["development_fold_index"]["sha256"],
    }
    receipt_path = Path(result["outputs"]["run_receipt_json"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # The numerical worker does not register models; the registered runner adds
    # the model binding to the receipt exposed by the API.
    receipt.update(platform_registered=True, platform_model_binding=binding)
    atomic_json(receipt_path, receipt)
    result.update(model_name=manifest["display_name"], model_version=manifest["version"],
                  checkpoints=_checkpoint_records(model_dir, manifest), warnings=[manifest["claim_boundary"]])
    result["input"].update(mode="dataset_bound", development_fold_id=fold_id,
                           development_fold_index=str(index_path))
    result["provenance"].update(platform_registered=True, platform_model_binding=binding,
                                evidence=manifest["evidence"], claim_boundary=manifest["claim_boundary"])
    return result


sixtrack_fault_local_logistic_audited_v2 = run_fault_prediction
sixtrack_property_tabicl_phif = _property_runner("sixtrack_property_tabicl_phif")
sixtrack_property_tabicl_klogh = _property_runner("sixtrack_property_tabicl_klogh")
sixtrack_property_tabicl_sw = _property_runner("sixtrack_property_tabicl_sw")
sixtrack_reconstruction_p21 = run_reconstruction_prediction
