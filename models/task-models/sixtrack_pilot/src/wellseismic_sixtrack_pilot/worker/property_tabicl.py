"""Isolated worker: P67 frozen TabICL for one reservoir-property target.

Runs inside the dedicated model interpreter.  Dataset-bound mode only: the query
rows and the in-context rows both come from the packaged, hash-verified
``development_logo4.npz``; no raw SEG-Y/LAS path is accepted.

For ``query_family`` F the worker selects the LOGO4 fold whose validation group
is F.  The fold's 192 training rows (64 depth-spread rows from each of the three
other development families) are the TabICL context and the only rows used to fit
the input standardisation; the fold's 81 rows of F are predicted.  That is the
exact P67 cell for ``(seed, fold)``, so one run doubles as a parity replay.
The frozen-test family 15/9-F-15 is absent from the archive and rejected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..manifest import atomic_json, fail, load_request, verify_payload

TARGETS = ("PHIF", "KLOGH", "SW")
UNITS = {"PHIF": "fraction", "KLOGH": "mD", "SW": "fraction"}
FROZEN_TEST_FAMILY = "15/9-F-15"
N_ESTIMATORS = 2
CHECKPOINT_VERSION = "tabicl-regressor-v2-20260212.ckpt"
# Bound to upstream per-row metadata by p71-property-upstream-recovery-20261013.
CHANNEL_SOURCE_SHA256 = "9b2e00389446802f7306a2798c3d2f216251eb7744b6c7ad61e1f3e242bcb347"


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


# --- verbatim from reservoir_p5_stage2.py (sha256 00291a9c…) -----------------
def _fit_stats(seismic: np.ndarray, logs: np.ndarray) -> dict[str, np.ndarray]:
    seismic_flat = seismic.reshape(len(seismic), -1)
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(logs), -1)
    mean, std = np.zeros(values.shape[1]), np.ones(values.shape[1])
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if observed.size:
            mean[column], std[column] = observed.mean(), observed.std() + 1e-8
    return {
        "seismic_mean": seismic_flat.mean(axis=0),
        "seismic_std": seismic_flat.std(axis=0) + 1e-8,
        "log_mean": mean,
        "log_std": std,
    }


def _transform(
    seismic: np.ndarray, logs: np.ndarray, stats: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seismic_flat = seismic.reshape(len(seismic), -1)
    seismic_flat = (seismic_flat - stats["seismic_mean"]) / stats["seismic_std"]
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).astype(float).reshape(len(logs), -1)
    values = ((values - stats["log_mean"]) / stats["log_std"]) * masks
    tabular = np.concatenate([seismic_flat, values, masks], axis=1)
    normalized_logs = np.concatenate(
        [values.reshape(len(logs), 9, 4), masks.reshape(len(logs), 9, 4)], axis=2
    )
    arrays = (seismic_flat.reshape(seismic.shape), normalized_logs, tabular)
    if tabular.shape[1] != 153 or not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("fold preprocessing produced invalid inputs")
    return arrays


# --- verbatim semantics of p5_contract.model_to_physical (sha256 066dc813…) --
def model_to_physical(target: str, values: Any, *, prediction: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if target == "KLOGH":
        if prediction:
            array = np.maximum(array, 0.0)
        return np.expm1(array)
    if target in {"PHIF", "SW"}:
        return np.clip(array, 0.0, 1.0) if prediction else array.copy()
    raise KeyError(target)


def load_query_fold(archive_path: Path, query_family: str) -> dict[str, Any]:
    """Mirror ``reservoir_p5_stage3.load_fold`` for the fold validating ``query_family``."""

    if query_family == FROZEN_TEST_FAMILY:
        raise ValueError("15/9-F-15 is the frozen test family and is never predicted")
    archive_bytes = Path(archive_path).read_bytes()
    source_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["split_manifest_json"]))
        stored_hash = manifest.pop("split_hash")
        if _hash_payload(manifest) != stored_hash:
            raise RuntimeError("development archive split manifest hash mismatch")
        if manifest["test_firewall"]["test_access"]:
            raise RuntimeError("development archive violates the frozen-test firewall")
        fold = next(
            (item for item in manifest["folds"] if item["validation_groups"] == [query_family]),
            None,
        )
        if fold is None:
            families = sorted(group for item in manifest["folds"] for group in item["validation_groups"])
            raise ValueError(f"query_family must be one of {families}")
        prefix = f"fold_{fold['fold_id']}"
        data = {
            "split": archive[f"{prefix}_split"].astype(str),
            "seismic": np.asarray(archive[f"{prefix}_seismic_patch"], dtype=float),
            "logs": np.asarray(archive[f"{prefix}_well_log_sequence"], dtype=float),
            "labels": np.asarray(archive[f"{prefix}_labels_model_domain"], dtype=float),
            "masks": np.asarray(archive[f"{prefix}_target_masks"], dtype=bool),
            "sample_ids": archive[f"{prefix}_sample_ids"].astype(str),
            "families": archive[f"{prefix}_family_ids"].astype(str),
            "wells": archive[f"{prefix}_well_ids"].astype(str),
            "depths": np.asarray(archive[f"{prefix}_depths_m"], dtype=float),
        }
    if FROZEN_TEST_FAMILY in set(data["families"].tolist()):
        raise RuntimeError("development archive contains the frozen-test family")
    train = np.flatnonzero(data["split"] == "train")
    query = np.flatnonzero(data["split"] == "validation")
    if set(data["families"][train]) & set(data["families"][query]):
        raise RuntimeError("context and query families overlap")
    stats = _fit_stats(data["seismic"][train], data["logs"][train])
    _, _, tabular = _transform(data["seismic"], data["logs"], stats)
    return {
        "source_archive_sha256": source_sha256,
        "fold_id": int(fold["fold_id"]),
        "split_hash": stored_hash,
        "context_groups": list(fold["train_groups"]),
        "query_groups": list(fold["validation_groups"]),
        "tabular": tabular,
        # Preserve prepared archive values before context-fitted normalization.
        # These are sampled windows, not complete original LAS/SEG-Y traces.
        "prepared_seismic": data["seismic"],
        "prepared_logs": data["logs"],
        "train_indices": train,
        "query_indices": query,
        **{key: data[key] for key in ("labels", "masks", "sample_ids", "families", "wells", "depths")},
    }


def export_query_inputs(fold: Mapping[str, Any], query: np.ndarray, path: Path) -> dict[str, Any]:
    """Export only the selected validation inputs, with no targets or context labels."""
    query = np.asarray(query)
    if query.ndim != 1 or not np.issubdtype(query.dtype, np.integer):
        raise ValueError("query indices must be a one-dimensional integer array")
    if not len(query) or len(np.unique(query)) != len(query) or not np.isin(
        query, fold["query_indices"]
    ).all():
        raise ValueError("query inputs must contain unique validation rows only")
    seismic = np.asarray(fold["prepared_seismic"])[query]
    logs = np.asarray(fold["prepared_logs"])[query]
    if seismic.shape != (len(query), 3, 3, 9) or logs.shape != (len(query), 9, 8):
        raise ValueError("prepared property input window shapes do not match the model contract")
    if not np.isfinite(seismic).all() or not np.isfinite(logs).all():
        raise ValueError("prepared property input windows contain nonfinite values")
    masks = logs[:, :, 4:8]
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("prepared property input masks must be binary")
    units = None
    if fold.get("source_archive_sha256") == CHANNEL_SOURCE_SHA256:
        unit_lock = json.loads((Path(__file__).parents[1] / "source_locks" /
                                "property_input_units.json").read_text())
        if (unit_lock["archive_sha256"] != CHANNEL_SOURCE_SHA256 or
                hashlib.sha256(_canonical(unit_lock["units_by_well"])).hexdigest() !=
                unit_lock["units_sha256"]):
            raise ValueError("property input unit source lock mismatch")
        units = unit_lock["units_by_well"]
    well_tie = None
    if fold.get("source_archive_sha256") == CHANNEL_SOURCE_SHA256:
        tie_lock = json.loads((Path(__file__).parents[1] / "source_locks" /
                               "property_well_tie.json").read_text())
        if tie_lock["archive_sha256"] != CHANNEL_SOURCE_SHA256:
            raise ValueError("property well tie source lock mismatch")
        outside = set(tie_lock["extrapolated_query_sample_ids"])
        well_tie = {
            "source_archive_sha256": CHANNEL_SOURCE_SHA256,
            "method": "official_surface_picks_piecewise_linear_md",
            "query_rows": int(len(query)),
            "extrapolated_query_rows": sum(str(s) in outside for s in fold["sample_ids"][query]),
            "independent_accuracy_verified": False,
        }
    np.savez(path, sample_ids=fold["sample_ids"][query], well_ids=fold["wells"][query],
        mother_well_families=fold["families"][query], depth_m=fold["depths"][query],
        seismic_patch=seismic, well_log_values=logs[:, :, :4], well_log_valid=masks.astype(bool))
    return {
        "contract_version": "sixtrack-property-query-inputs.v1",
        "output_key": "prepared_query_inputs_npz",
        "rows": int(len(query)),
        "split_hash": fold["split_hash"],
        "fold_id": fold["fold_id"],
        "normalization": "prepared_archive_before_context_fitted_standardization",
        "seismic_axis_order": ["sample", "inline_offset", "crossline_offset", "window_position"],
        "well_log_axis_order": ["sample", "window_position", "channel_index"],
        "well_log_channel_indices": [0, 1, 2, 3],
        "well_log_channel_names": ["GR", "RT", "NPHI", "RHOB"]
        if fold.get("source_archive_sha256") == CHANNEL_SOURCE_SHA256 else None,
        "channel_name_source_sha256": fold.get("source_archive_sha256")
        if fold.get("source_archive_sha256") == CHANNEL_SOURCE_SHA256 else None,
        "well_log_units_by_well": units,
        "well_tie": well_tie,
        "signal_units": "not_declared_by_development_archive",
        "depth_unit": "m",
        "query_targets_exported": False,
        "original_full_traces": False,
    }


def run(request_path: Path) -> dict[str, Any]:
    request = load_request(request_path)
    target = str(request["target"])
    if target not in TARGETS:
        raise ValueError(f"unknown target {target}")
    target_index = TARGETS.index(target)
    options = request.get("options") or {}
    query_family = str(options.get("query_family") or "").strip()
    seed = int(options.get("seed", 2693))
    payload = verify_payload(request["model_dir"])
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    fold = load_query_fold(payload["development_archive"], query_family)
    query = fold["query_indices"]
    wanted_wells = [str(item) for item in options.get("query_well_ids") or []]
    if wanted_wells:
        unknown = sorted(set(wanted_wells) - set(fold["wells"][query].tolist()))
        if unknown:
            raise ValueError(f"query_well_ids not in {query_family}: {unknown}")
        query = query[np.isin(fold["wells"][query], wanted_wells)]
    context = fold["train_indices"]
    context = context[fold["masks"][context, target_index]]

    import torch
    import tabicl

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    estimator = tabicl.TabICLRegressor(
        n_estimators=N_ESTIMATORS,
        batch_size=1,
        model_path=str(payload["tabicl_checkpoint"]),
        allow_auto_download=False,
        checkpoint_version=CHECKPOINT_VERSION,
        device="cpu",
        offload_mode="auto",
        random_state=seed + target_index,
        verbose=False,
    )
    x = np.asarray(fold["tabular"], dtype=np.float64)
    estimator.fit(x[context], fold["labels"][context, target_index])
    raw = np.asarray(estimator.predict(x[query]), dtype=np.float64).reshape(-1)
    if raw.shape != (len(query),) or not np.isfinite(raw).all():
        raise FloatingPointError("TabICL produced a non-finite or misshaped prediction")
    physical = model_to_physical(target, raw, prediction=True)
    truth_mask = fold["masks"][query, target_index]
    truth_model = fold["labels"][query, target_index]
    truth_physical = model_to_physical(target, truth_model, prediction=False)

    scored = truth_mask
    residual = physical[scored] - truth_physical[scored]
    denominator = float(np.sum((truth_physical[scored] - truth_physical[scored].mean()) ** 2))
    metrics = {
        "rows_scored": int(scored.sum()),
        "physical_mae": float(np.mean(np.abs(residual))),
        "physical_rmse": float(np.sqrt(np.mean(residual**2))),
        "physical_r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None,
        "model_domain_rmse": float(np.sqrt(np.mean((raw[scored] - truth_model[scored]) ** 2))),
        "reference_label": "CPI interpretation label (property-phif-klogh-sw-cpi-v1), not core",
    }

    csv_path = output_dir / f"{target.lower()}_predictions.csv"
    npz_path = output_dir / f"{target.lower()}_predictions.npz"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "well_id", "mother_well_family", "depth_m", "target", "unit",
            "prediction", "prediction_model_domain", "reference_label", "reference_label_valid",
        ])
        for row, index in enumerate(query.tolist()):
            writer.writerow([
                fold["sample_ids"][index], fold["wells"][index], fold["families"][index],
                f"{fold['depths'][index]:.4f}", target, UNITS[target],
                repr(float(physical[row])), repr(float(raw[row])),
                repr(float(truth_physical[row])) if truth_mask[row] else "",
                int(bool(truth_mask[row])),
            ])
    np.savez(
        npz_path,
        sample_ids=fold["sample_ids"][query],
        well_ids=fold["wells"][query],
        depth_m=fold["depths"][query],
        prediction=physical,
        prediction_model_domain=raw,
        reference_label=truth_physical,
        reference_label_valid=truth_mask,
    )
    inputs_path = output_dir / f"{target.lower()}_query_inputs.npz"
    input_contract = export_query_inputs(fold, query, inputs_path)
    return {
        "status": "completed",
        "outputs": {"predictions_csv": str(csv_path), "predictions_npz": str(npz_path),
                    "prepared_query_inputs_npz": str(inputs_path)},
        "prepared_input_contract": input_contract,
        "target": target,
        "unit": UNITS[target],
        "fold": {
            "fold_id": fold["fold_id"],
            "split_hash": fold["split_hash"],
            "context_groups": fold["context_groups"],
            "query_groups": fold["query_groups"],
            "context_rows": int(len(context)),
            "query_rows": int(len(query)),
            "query_well_ids": sorted(set(fold["wells"][query].tolist())),
            "preprocessing_fit": "context_rows_only",
        },
        "metrics": metrics,
        "seed": seed,
        "random_state": seed + target_index,
        "prediction_model_domain_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "tabicl": getattr(tabicl, "__version__", None),
            "device": "cpu",
            "n_estimators": N_ESTIMATORS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.request)
    except Exception as exc:  # reported to the platform as a structured failure
        return fail(args.result, exc)
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
