"""Structured P81 task subprocess; only sealed development inputs are accepted."""
import argparse
from pathlib import Path

import numpy as np

from ..facies_contract import validate_binding, SOURCE_LOCK_SHA256
from ..manifest import atomic_json, sha256_file, load_request, fail
from .facies_p81 import predict


def run(request):
    if set(request) != {"input_path", "state_path", "recovery_receipt_path", "sam2_source", "options", "output_directory"}:
        raise ValueError("P81 worker request fields mismatch")
    source, state, receipt_path = (Path(request[key]) for key in ("input_path", "state_path", "recovery_receipt_path"))
    options, output = request["options"], Path(request["output_directory"])
    if output.exists():
        raise ValueError("P81 refuses to overwrite prediction outputs")
    binding = validate_binding(source, state, receipt_path, options)

    def progress(stage, details):
        atomic_json(output.parent / "worker_stage.json", {"status": "running", "stage": stage,
            "task_id": options["task_id"], "fold_id": options["fold_id"], "repeat_id": options["repeat_id"],
            "input_sha256": options["input_sha256"], **details})

    result = predict(source, state, options["state_sha256"], options["task_id"], Path(request["sam2_source"]), progress=progress)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path, index_path = output / "facies_predictions.npz", output / "facies_slice_index.json"
    np.savez_compressed(prediction_path, **result)
    atomic_json(index_path, {"schema_version": "well-seismic.p81-slice-index.v1", "task_id": options["task_id"],
        "label_version": binding["label_version"], "num_classes": binding["num_classes"],
        "class_codes": list(range(binding["num_classes"])), "image_shape": [128, 128],
        "spatial_reference": "archive_slice_pixel_indices_only", "sample_ids": result["sample_ids"].tolist(),
        "fold_id": options["fold_id"], "repeat_id": options["repeat_id"], "seed": binding["seed"]})
    outputs = {"predictions_npz": str(prediction_path), "slice_index_json": str(index_path)}
    receipt = {"schema_version": "well-seismic.p81-managed-worker.v1", "status": "completed", "device": "cpu",
        **options, "seed": binding["seed"], "label_version": binding["label_version"], "num_classes": binding["num_classes"],
        "query_rows": len(result["sample_ids"]), "outputs": outputs,
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
        "source_lock_sha256": SOURCE_LOCK_SHA256, "registered": False, "query_labels_consumed": False,
        "frozen_test_accessed": False, "training_performed": False, "research_default_changed": False,
        "fold_train_preprocessor": binding["preprocessor"], "fold_split_hash": binding["fold_split_hash"]}
    atomic_json(output / "worker_result.json", receipt)
    atomic_json(output.parent / "worker_stage.json", {"status": "completed", "stage": "outputs_saved",
        "task_id": options["task_id"], "fold_id": options["fold_id"], "repeat_id": options["repeat_id"]})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(load_request(args.request))
    except Exception as exc:
        raise SystemExit(fail(args.result, exc))
    atomic_json(args.result, result)
