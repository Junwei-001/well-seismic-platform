"""Structured P7 worker protocol; no parent-process Chronos dependency."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..manifest import atomic_json, fail, load_request, sha256_file
from .sweetspot_p7 import run as predict


def run(request):
    if set(request) != {"input_path", "input_sha256", "snapshot_path", "fold_id", "output_directory"}:
        raise ValueError("P7 worker request fields mismatch")
    source, output = Path(request["input_path"]), Path(request["output_directory"])
    def progress(stage):
        atomic_json(output.parent / "worker_stage.json", {"status": "running", "stage": stage,
            "fold_id": request["fold_id"], "input_sha256": request["input_sha256"]})
    predict(source, request["input_sha256"], Path(request["snapshot_path"]), request["fold_id"], output, progress)
    receipt = json.loads((output / "worker_result.json").read_text())
    npz, table = output / "sweetspot_t3_predictions.npz", output / "sweetspot_t3_predictions.csv"
    with np.load(npz, allow_pickle=False) as arrays, table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["查询行", "样本标识", "P7融合预测", "Chronos观测索引均值", "历史油量均值"])
        for row, values in enumerate(zip(arrays["sample_ids"], arrays["prediction"],
                                         arrays["foundation_prediction"], arrays["history_mean"]), start=1):
            writer.writerow([row, str(values[0]), *[float(x) for x in values[1:]]])
    receipt.update(outputs={"predictions_npz": str(npz), "predictions_csv": str(table)},
                   output_sha256={"predictions_npz": sha256_file(npz), "predictions_csv": sha256_file(table)})
    atomic_json(output / "worker_result.json", receipt)
    atomic_json(output.parent / "worker_stage.json", {"status": "completed", "stage": "outputs_saved",
        "fold_id": request["fold_id"], "input_sha256": request["input_sha256"]})
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
