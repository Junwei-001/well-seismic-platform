"""Managed live P21 preparation: target-free seismic, local GFM, PyKrige, inference.

Internal development workflow; callers must authorize paths before invocation.
No public model registration or new-area scientific validity is implied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..manifest import atomic_json, fail, load_request, sha256_file
from .reconstruction_gfm import extract_features, load_encoder
from .reconstruction_prepare import prepare_inputs
from .reconstruction_p21 import run_prepared

REQUEST_KEYS = frozenset({"volume_path", "volume_sha256", "support_path", "support_sha256",
                          "source_root", "snapshot", "output_directory"})
SUPPORT_KEYS = frozenset({"train_indices_kji", "query_indices_kji", "train_target",
                         "train_kriging_coordinate", "query_kriging_coordinate"})


def assemble_inputs(volume: dict, support: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Validate addresses before any expensive forward; derive raw rows from volume."""
    if set(volume) != {"seismic", "active", "normalized_coordinates"} or set(support) != SUPPORT_KEYS:
        raise ValueError("Live P21 requires exact target-free volume and support fields; cached features/query targets forbidden")
    active, seismic, coordinates = volume["active"], volume["seismic"], volume["normalized_coordinates"]
    if active.ndim != 3 or active.dtype != np.bool_ or seismic.shape != (3, *active.shape) or coordinates.shape != seismic.shape:
        raise ValueError("Live P21 volume fields must align in KJI order")
    target = support["train_target"]
    if target.shape != (512,) or not np.isfinite(target).all():
        raise ValueError("Live P21 requires 512 finite training labels")
    inputs, requested = {"train_target": target}, []
    for prefix in ("train", "query"):
        indices = support[f"{prefix}_indices_kji"]
        if indices.ndim != 2 or indices.shape[1] != 3 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Live P21 indices must be integer N by 3 KJI")
        if (prefix == "train" and len(indices) != 512) or (prefix == "query" and not 1 <= len(indices) <= 2048):
            raise ValueError("Live P21 support/query budgets exceeded")
        if np.any(indices < 0) or np.any(indices >= np.asarray(active.shape)) or not active[tuple(indices.T)].all():
            raise ValueError("Live P21 index is outside active volume")
        if len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("Live P21 duplicate cell indices")
        xyz = support[f"{prefix}_kriging_coordinate"]
        if xyz.shape != (len(indices), 3) or not np.isfinite(xyz).all():
            raise ValueError("Live P21 explicit fold kriging coordinates are invalid")
        raw = np.column_stack([seismic[:, *indices.T].T, coordinates[:, *indices.T].T])
        if not np.isfinite(raw).all():
            raise ValueError("Live P21 addressed raw features must be finite")
        inputs[f"{prefix}_raw_features"] = raw.astype(np.float64)
        inputs[f"{prefix}_kriging_coordinate"] = xyz
        requested.append(indices)
    if set(map(tuple, requested[0])) & set(map(tuple, requested[1])):
        raise ValueError("Live P21 training and query cells must be disjoint")
    unique, inverse = np.unique(np.vstack(requested), axis=0, return_inverse=True)
    return inputs, unique, inverse


def run(request: dict) -> dict:
    if set(request) != REQUEST_KEYS:
        raise ValueError("Live P21 request fields mismatch")
    output = Path(request["output_directory"])
    if output.exists():
        raise ValueError("Use a new output directory")
    archives = {}
    for role in ("volume", "support"):
        path = Path(request[f"{role}_path"])
        if sha256_file(path) != request[f"{role}_sha256"]:
            raise ValueError(f"Live P21 {role} hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            archives[role] = {key: archive[key] for key in archive.files}
    inputs, indices, inverse = assemble_inputs(archives["volume"], archives["support"])
    network, assets = load_encoder(Path(request["source_root"]), Path(request["snapshot"]))
    features = extract_features(archives["volume"]["seismic"], archives["volume"]["active"], indices, network)
    inputs["train_foundation"] = features[inverse[:512]]
    inputs["query_foundation"] = features[inverse[512:]]
    prepared, variance, preparation = prepare_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    prepared_path, variance_path = output / "prepared.npz", output / "pykrige_variance.npy"
    np.savez_compressed(prepared_path, **prepared)
    np.save(variance_path, variance, allow_pickle=False)
    inference = run_prepared(prepared_path, sha256_file(prepared_path), output / "predictions")
    receipt = {"status": "completed", "execution": "live_local_pretrained_gfm_cpu",
               "query_rows": len(inputs["query_raw_features"]), "encoded_unique_rows": len(indices),
               "encoded_fixed_i_slices": len(np.unique(indices[:, 2])), "assets": assets,
               "volume_sha256": request["volume_sha256"], "support_sha256": request["support_sha256"],
               "preparation": preparation, "inference": inference,
               "query_targets_consumed": False, "cached_foundation_features_consumed": False,
               "cached_pykrige_baseline_consumed": False, "platform_registered": False,
               "new_area_verified": False, "deployed": False,
               "output_sha256": inference["output_sha256"],
               "outputs": {**inference["outputs"], "prepared_npz": str(prepared_path),
                           "pykrige_variance_npy": str(variance_path)},
               "variance_sha256": sha256_file(variance_path)}
    atomic_json(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(load_request(args.request))
    except Exception as exc:
        return fail(args.result, exc)
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
