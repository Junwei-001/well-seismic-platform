"""P21 train-only metric preparation and real PyKrige baseline computation.

Inputs are already aligned, target-free seismic/geometry/GFM feature rows.
Kriging coordinates are explicit: the research baseline uses the fold-wide
preprocessing scale, not the P21 metric's 512-support coordinate scale.
This module does not extract GFM features or establish new-area provenance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pykrige
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..manifest import atomic_json, sha256_file

FEATURE_INPUT_KEYS = frozenset({
    "train_raw_features", "query_raw_features", "train_foundation", "query_foundation",
    "train_target", "train_kriging_coordinate", "query_kriging_coordinate",
})


def prepare_inputs(inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Port P18 metrics and P5 baseline; query labels are never accepted."""
    if set(inputs) != FEATURE_INPUT_KEYS:
        raise ValueError("P21 preparation requires exactly the feature fields; query labels are forbidden")
    values = {key: np.asarray(array, dtype=np.float64) for key, array in inputs.items()}
    query = values["query_raw_features"]
    if query.ndim != 2 or query.shape[1] != 6 or not 1 <= len(query) <= 2048:
        raise ValueError("P21 preparation requires 1..2048 query rows with six raw features")
    if values["train_raw_features"].shape != (512, 6) or values["train_target"].shape != (512,):
        raise ValueError("P21 preparation requires exactly 512 support rows")
    foundation = values["train_foundation"]
    if foundation.ndim != 2 or foundation.shape[0] != 512 or foundation.shape[1] < 16:
        raise ValueError("P21 requires at least 16 aligned foundation features")
    if values["query_foundation"].shape != (len(query), foundation.shape[1]):
        raise ValueError("P21 query foundation features are misaligned")
    for prefix, count in (("train", 512), ("query", len(query))):
        if values[f"{prefix}_kriging_coordinate"].shape != (count, 3):
            raise ValueError("P21 kriging coordinates must be aligned three-column matrices")
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("P21 preparation inputs must be finite")
    if pykrige.__version__ != "1.7.3":
        raise ValueError("P21 baseline requires research-locked PyKrige 1.7.3")
    with threadpool_limits(limits=1):
        # Exact P18 order; no query rows or support targets enter scaling/PCA.
        scaler = StandardScaler().fit(foundation)
        train_scaled = scaler.transform(foundation)
        query_scaled = scaler.transform(values["query_foundation"])
        pca = PCA(n_components=16, random_state=2693).fit(train_scaled)
        train_latent, query_latent = pca.transform(train_scaled), pca.transform(query_scaled)
        latent_scale = np.std(train_latent, axis=0)
        latent_scale[latent_scale < 1e-8] = 1.0
        coordinate_scale = np.std(values["train_raw_features"][:, 3:6], axis=0)
        coordinate_scale[coordinate_scale < 1e-8] = 1.0
        seismic_scale = np.std(values["train_raw_features"][:, 0:3], axis=0)
        seismic_scale[seismic_scale < 1e-8] = 1.0
        prepared = {"train_latent": train_latent / latent_scale,
                    "query_latent": query_latent / latent_scale, "train_target": values["train_target"]}
        for prefix in ("train", "query"):
            raw = values[f"{prefix}_raw_features"]
            prepared[f"{prefix}_coordinate"] = raw[:, 3:6] / coordinate_scale
            prepared[f"{prefix}_seismic"] = raw[:, 0:3] / seismic_scale
        xyz, qxyz = values["train_kriging_coordinate"], values["query_kriging_coordinate"]
        model = OrdinaryKriging3D(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], values["train_target"],
            variogram_model="linear", nlags=4, verbose=False, enable_plotting=False,
        )
        prediction, variance = model.execute("points", qxyz[:, 0], qxyz[:, 1], qxyz[:, 2])
        # P5 Stage-3 _save_prediction archives the baseline in float32. P21
        # subsequently reads it as float64; retain that scientific contract.
        prepared["pykrige_prediction"] = np.asarray(prediction, dtype=np.float32).astype(np.float64)
        variance = np.asarray(variance, dtype=np.float64)
    if not all(np.isfinite(array).all() for array in [*prepared.values(), variance]):
        raise FloatingPointError("P21 preparation generated non-finite outputs")
    audit = {
        "fit_rows": 512, "query_rows": len(query), "pca_components": 16,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "all_metric_transforms_fit_on_support_only": True,
        "target_used_for_metric_fit": False, "query_targets_consumed": False,
        "baseline_computed": True, "cached_pykrige_baseline_consumed": False,
        "pykrige_version": pykrige.__version__, "variogram_model": "linear", "nlags": 4,
        "kriging_coordinate_source": "explicit original fold-preprocessing coordinates",
        "baseline_archive_precision": "float32 widened to float64, matching P5-to-P21 handoff",
        "new_area_feature_extraction_supported": False, "platform_registered": False,
    }
    return prepared, variance, audit


def run(input_path: Path, input_sha256: str, output_directory: Path) -> dict:
    if sha256_file(input_path) != input_sha256:
        raise ValueError("P21 feature input hash mismatch")
    if output_directory.exists():
        raise ValueError("Use a new output directory; existing results are never overwritten")
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    prepared, variance, audit = prepare_inputs(inputs)
    output_directory.mkdir(parents=True)
    output = output_directory / "prepared.npz"
    variance_path = output_directory / "pykrige_variance.npy"
    np.savez_compressed(output, **prepared)
    np.save(variance_path, variance, allow_pickle=False)
    audit.update(status="completed", feature_input_sha256=input_sha256,
                 prepared_output_sha256=sha256_file(output), variance_sha256=sha256_file(variance_path))
    atomic_json(output_directory / "receipt.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.input_sha256, args.output_directory)


if __name__ == "__main__":
    main()
