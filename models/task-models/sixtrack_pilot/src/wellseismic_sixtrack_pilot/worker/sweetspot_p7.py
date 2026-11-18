"""Frozen P7 T3 query inference on observation-index histories, never labels."""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np

from ..manifest import atomic_json, sha256_file

WEIGHTS_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CONFIG_SHA256 = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
FOLD_WEIGHTS = (0.7000000000000001, 0.63, 0.64, 0.65)
COVARIATES = ("bore_gas_vol", "bore_wat_vol", "on_stream_hrs", "avg_downhole_pressure",
              "avg_choke_size_p", "avg_whp_p")
FIELDS = frozenset({"query_sequence", "history_timestamps", "sample_ids"})


def load_queries(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError("P7 input must contain only query histories, timestamps and identities; labels are forbidden")
        sequence = archive["query_sequence"]
        time = archive["history_timestamps"]
        ids = archive["sample_ids"]
    if (sequence.ndim != 3 or sequence.shape[1:] != (7, 30)
            or not 1 <= len(sequence) <= 512 or sequence.dtype.kind not in "fi"):
        raise ValueError("P7 requires 1..512 real histories shaped [N,7,30]")
    sequence = sequence.astype(np.float32)
    if np.isinf(sequence).any() or np.isnan(sequence[:, 0, :]).all(axis=1).any():
        raise ValueError("P7 oil history must be observed and histories cannot contain infinity")
    if (time.shape != (len(sequence), 30) or time.dtype.kind != "U"
            or ids.shape != (len(sequence),) or ids.dtype.kind != "U"
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("P7 history timestamps and unique sample identities must align")
    for sample_id, timestamps in zip(ids, time):
        parts = str(sample_id).split(":", 2)
        if len(parts) != 3 or parts[0] != "productivity" or not parts[1]:
            raise ValueError("P7 sample identity must identify a productivity cutoff")
        days = timestamps.astype("datetime64[D]")
        cutoff = np.datetime64(parts[2], "D")
        if (np.isnat(days).any() or np.isnat(cutoff) or not (days < cutoff).all()
                or not (np.diff(days) > np.timedelta64(0, "D")).all()
                or days[-1] - days[0] > np.timedelta64(45, "D")):
            raise ValueError("P7 histories must be increasing pre-cutoff observations spanning at most 45 days")
    return sequence, time, ids


def forecast(pipeline, sequence):
    inputs = [{"target": row[0].copy(), "past_covariates": {
        name: row[index].copy() for index, name in enumerate(COVARIATES, start=1)}} for row in sequence]
    matches = np.flatnonzero(np.isclose(np.asarray(pipeline.quantiles, float), 0.5))
    if len(matches) != 1:
        raise ValueError("P7 requires exactly one median quantile")
    outputs = pipeline.predict(inputs, prediction_length=30, batch_size=196, cross_learning=False)
    if len(outputs) != len(inputs):
        raise ValueError("P7 forecast item count mismatch")
    daily = []
    for output in outputs:
        values = output.detach().float().cpu().numpy()
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 30:
            raise ValueError("P7 forecast dimensions mismatch")
        daily.append(np.maximum(np.asarray(values[0, int(matches[0])], np.float64), 0.0))
    daily = np.asarray(daily)
    if not np.isfinite(daily).all():
        raise ValueError("P7 returned nonfinite forecasts")
    return daily, daily.mean(axis=1)


def run(input_path: Path, input_sha256: str, snapshot: Path, fold_id: int, output: Path, progress=None):
    if type(fold_id) is not int or fold_id not in range(4):
        raise ValueError("P7 original fold must be an integer 0..3")
    if output.exists():
        raise ValueError("Use a new output directory")
    if sha256_file(input_path) != input_sha256:
        raise ValueError("P7 input hash mismatch")
    sequence, time, ids = load_queries(input_path)
    for filename, expected in (("model.safetensors", WEIGHTS_SHA256), ("config.json", CONFIG_SHA256)):
        if sha256_file(snapshot / filename) != expected:
            raise ValueError(f"P7 pretrained source hash mismatch: {filename}")
    if importlib.metadata.version("chronos-forecasting") != "2.2.0":
        raise ValueError("P7 requires chronos-forecasting 2.2.0")
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(1)
    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot.resolve()), device_map="cpu",
                                              torch_dtype=torch.float32, local_files_only=True)
    observed_forward = False
    def completed_forward(module, inputs, result):
        nonlocal observed_forward
        if not observed_forward:
            observed_forward = True
            if progress:
                progress("pretrained_forward_completed")
    handle = pipeline.model.register_forward_hook(completed_forward)
    try:
        steps, foundation = forecast(pipeline, sequence)
    finally:
        handle.remove()
    if progress:
        progress("forecast_completed")
    # Original history control uses float64 raw history, before model float32 conversion.
    with np.load(input_path, allow_pickle=False) as archive:
        history = np.nanmean(np.asarray(archive["query_sequence"], np.float64)[:, 0], axis=1)
    weight = FOLD_WEIGHTS[fold_id]
    prediction = weight * foundation + (1.0 - weight) * history
    output.mkdir(parents=True, exist_ok=False)
    path = output / "sweetspot_t3_predictions.npz"
    np.savez_compressed(path, prediction=prediction, foundation_prediction=foundation,
                        history_mean=history, forecast_observation_steps=steps, sample_ids=ids)
    atomic_json(output / "worker_result.json", {"status": "completed", "fold_id": fold_id,
        "query_rows": len(ids), "chronos_weight": weight, "input_sha256": input_sha256,
        "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256,
        "output_sha256": sha256_file(path), "source_sha256": sha256_file(Path(__file__)),
        "chronos_forecasting_version": "2.2.0", "execution": "cpu_float32_one_thread",
        "query_labels_consumed": False, "future_covariates_consumed": False,
        "blend_weight_refitted": False, "cross_learning": False, "registered": False,
        "deployed": False, "time_semantics": "30 observation-index forecast steps, not guaranteed calendar days",
        "claim_boundary": "Original P7 development queries only; fixed archived train-selected blend weight. No final refit, new-well authorization, calendar forecasting or sweetspot spatial ranking."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.input_sha256, args.snapshot, args.fold, args.output_directory)
