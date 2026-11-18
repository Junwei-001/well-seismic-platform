"""P21 fixed three-kernel inference on explicitly prepared fold inputs.

This is not yet a registered platform model. It requires the original fold's
coordinate/seismic/PCA transforms and PyKrige prediction, all prepared without
query targets. It does not build these inputs from new wells or seismic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from ..manifest import atomic_json, sha256_file

INPUT_KEYS = frozenset({
    "train_coordinate", "train_seismic", "train_latent", "train_target",
    "query_coordinate", "query_seismic", "query_latent", "pykrige_prediction",
})
SEISMIC_WEIGHTS = (0.0, 0.1, 0.2)
NEIGHBOURS = 64
DISTANCE_POWER = 1.5
BLEND_WEIGHT = 0.75


def predict_prepared(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and three component predictions; never consume query labels."""
    if set(inputs) != INPUT_KEYS:
        raise ValueError("P21 requires exactly the prepared input fields; query labels are forbidden")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in inputs.items()}
    baseline = arrays["pykrige_prediction"]
    if baseline.ndim != 1 or not 1 <= len(baseline) <= 2048:
        raise ValueError("P21 requires 1..2048 aligned query rows")
    for prefix, count in (("train", 512), ("query", len(baseline))):
        for suffix, columns in (("coordinate", 3), ("seismic", 3), ("latent", 16)):
            if arrays[f"{prefix}_{suffix}"].shape != (count, columns):
                raise ValueError(f"invalid {prefix}_{suffix} shape")
    if arrays["train_target"].shape != (512,):
        raise ValueError("P21 uses exactly the original 512-label support budget")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("P21 prepared inputs must be finite")
    components = []
    # Keep the research arithmetic/order: scale Z, assemble the metric, find
    # 64 neighbours, IDW power 1.5, blend each member with PyKrige, then mean.
    with threadpool_limits(limits=1):
        for seismic_weight in SEISMIC_WEIGHTS:
            metrics = []
            for prefix in ("train", "query"):
                coordinate = arrays[f"{prefix}_coordinate"].copy()
                coordinate[:, 2] *= 4.0
                metrics.append(np.column_stack([
                    coordinate, seismic_weight * arrays[f"{prefix}_seismic"],
                    0.1 * arrays[f"{prefix}_latent"],
                ]))
            model = NearestNeighbors(n_neighbors=NEIGHBOURS, n_jobs=1).fit(metrics[0])
            distance, rows = model.kneighbors(metrics[1])
            weights = 1.0 / np.maximum(distance, 1e-8) ** DISTANCE_POWER
            kernel = np.sum(weights * arrays["train_target"][rows], axis=1) / np.sum(weights, axis=1)
            components.append((1.0 - BLEND_WEIGHT) * baseline + BLEND_WEIGHT * kernel)
    members = np.stack(components)
    prediction = np.mean(members, axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("P21 generated non-finite predictions")
    return prediction, members


def run_prepared(input_path: Path, input_sha256: str, output_directory: Path) -> dict:
    """Run a hash-bound prepared input and return its durable receipt."""
    observed = sha256_file(input_path)
    if observed != input_sha256:
        raise ValueError("P21 prepared input hash mismatch")
    if output_directory.exists():
        raise ValueError("Use a new output directory; existing results are never overwritten")
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    prediction, members = predict_prepared(inputs)
    output_directory.mkdir(parents=True)
    output = output_directory / "p21_predictions.npz"
    np.savez_compressed(output, prediction=prediction, component_predictions=members)
    points = output_directory / "p21_query_points.npz"
    np.savez_compressed(points, query_coordinate=inputs["query_coordinate"], prediction=prediction)
    receipt = {
        "status": "completed", "prepared_input_sha256": observed,
        "output_sha256": sha256_file(output), "query_rows": len(prediction),
        "execution": "cpu_prepared_input_only", "query_targets_consumed": False,
        "new_area_input_supported": False, "platform_registered": False,
        "cached_pykrige_baseline_consumed": True,
        "query_points_sha256": sha256_file(points),
        "outputs": {"predictions_npz": str(output), "prediction_points_npz": str(points)},
    }
    atomic_json(output_directory / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_prepared(args.input, args.input_sha256, args.output_directory)


if __name__ == "__main__":
    main()
