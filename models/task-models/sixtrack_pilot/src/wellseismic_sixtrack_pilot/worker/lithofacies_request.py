"""Structured isolated worker protocol for the P76 development ensemble."""
import argparse
import csv
from pathlib import Path

import numpy as np

from ..manifest import atomic_json, fail, load_request, sha256_file
from .lithofacies_p76 import INPUT_KEYS, predict


def export_query_inputs(inputs, path):
    """Preserve validation window order without exporting support or target labels."""
    if set(inputs) != INPUT_KEYS:
        raise ValueError("P76 query export requires the original label-free query input fields")
    well, seismic = np.asarray(inputs["query_well"]), np.asarray(inputs["query_seismic"])
    if (well.ndim != 3 or not 1 <= len(well) <= 360 or well.shape[1:] != (26, 33)
            or seismic.shape != (len(well), 3, 3, 33)):
        raise ValueError("P76 prepared query window shapes are invalid")
    if not np.isfinite(well).all() or not np.isfinite(seismic).all():
        raise ValueError("P76 prepared query windows must be finite")
    masks = well[:, 13:]
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("P76 prepared query visibility channels must be binary")
    path = Path(path)
    if path.exists():
        raise ValueError("P76 query input output must not already exist")
    np.savez_compressed(path, query_row_indices=np.arange(len(well), dtype=np.uint32),
        well_log_values=well[:, :13], well_log_valid=masks.astype(bool), seismic_patch=seismic)
    return {"contract_version": "sixtrack-lithofacies-query-inputs.v1",
        "output_key": "prepared_query_inputs_npz", "query_rows": len(well),
        "query_targets_exported": False, "support_labels_exported": False,
        "original_full_traces": False, "well_log_channel_names": None,
        "value_space": "original_fold_train_prepared_windows",
        "array_axes": {"well_log_values": ["QUERY", "CHANNEL", "WINDOW_SAMPLE"],
            "well_log_valid": ["QUERY", "CHANNEL", "WINDOW_SAMPLE"],
            "seismic_patch": ["QUERY", "LOCAL_Y", "LOCAL_X", "WINDOW_SAMPLE"]}}


def run(request):
    if set(request) != {"input_path", "input_sha256", "checkpoint_path", "fold_id", "repeat_id", "output_directory"}:
        raise ValueError("P76 worker request fields mismatch")
    source, output = Path(request["input_path"]), Path(request["output_directory"])
    if output.exists(): raise ValueError("Use a new output directory")
    if sha256_file(source) != request["input_sha256"]: raise ValueError("P76 input hash mismatch")
    with np.load(source, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    def progress(component):
        atomic_json(output.parent / "worker_stage.json", {
            "latest_completed_component": component, "input_sha256": request["input_sha256"],
            "fold_id": request["fold_id"], "repeat_id": request["repeat_id"], "status": "running"})
    values, receipt = predict(inputs, Path(request["checkpoint_path"]),
        fold=request["fold_id"], repeat=request["repeat_id"], progress=progress)
    output.mkdir(parents=True, exist_ok=False)
    predictions, table = output / "lithofacies_predictions.npz", output / "lithofacies_predictions.csv"
    np.savez_compressed(predictions, **values)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["查询行", "岩相代码", *[f"类别{code}概率" for code in range(9)]])
        for row, (code, probability) in enumerate(zip(values["prediction"], np.exp(values["logits"])), start=1):
            writer.writerow([row, int(code), *[float(value) for value in probability]])
    query_inputs = output / "lithofacies_query_inputs.npz"
    input_contract = export_query_inputs(inputs, query_inputs)
    receipt.update(status="completed", input_sha256=request["input_sha256"],
        fold_id=request["fold_id"], repeat_id=request["repeat_id"],
        output_sha256={"predictions_npz": sha256_file(predictions), "predictions_csv": sha256_file(table), "prepared_query_inputs_npz": sha256_file(query_inputs)},
        outputs={"predictions_npz": str(predictions), "predictions_csv": str(table),
                 "prepared_query_inputs_npz": str(query_inputs)}, prepared_input_contract=input_contract)
    atomic_json(output / "receipt.json", receipt)
    atomic_json(output.parent / "worker_stage.json", {
        "latest_completed_component": "e8", "input_sha256": request["input_sha256"],
        "fold_id": request["fold_id"], "repeat_id": request["repeat_id"], "status": "completed"})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try: result = run(load_request(args.request))
    except Exception as exc: return fail(args.result, exc)
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__": raise SystemExit(main())
