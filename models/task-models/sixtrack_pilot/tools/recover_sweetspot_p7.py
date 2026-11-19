"""Recover only original P7 query histories; never construct query targets."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellseismic_sixtrack_pilot.manifest import sha256_file, atomic_json

SPLIT = "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/split_manifest.json"
SUMMARY = "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json"
LOADER = "_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_stage2_data.py"
LOCKS = {SPLIT: "c44277ffc1f6fb6b5dd740952e921c732d45193c7cb3b5c3dfc061e79025c62a",
         SUMMARY: "bb27792bcc4d34fb622ebaa5eea908b7ae4c6a08a5fa1ff19fda4d3dc97b7d3b",
         LOADER: "8583ed8b5b07ebdb5aafbb1ba549d81f04f1c42c3a4f852c8c56d79051866b0a"}


def recover(root: Path, output: Path):
    if output.exists():
        raise ValueError("Use a new output directory")
    for relative, expected in LOCKS.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"P7 recovery source hash mismatch: {relative}")
    split = json.loads((root / SPLIT).read_text())
    summary = json.loads((root / SUMMARY).read_text())
    if summary["decision"]["selected_method"] != "F1_chronos2_train_blend":
        raise ValueError("P7 selected method drift")
    sys.path.insert(0, str(root.resolve()))
    data = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_data")
    # The authorized workbook loader parses development raw measurements.
    # We do not call the label-building or _take APIs; only pre-cutoff rows
    # enter exported model inputs. Later raw measurements are not exported.
    frame, provenance = data._load_development_production(root, set(split["development_groups"]))
    wells = {str(group): well.sort_values("DATEPRD").reset_index(drop=True)
             for group, well in frame.groupby("WELL_BORE_CODE", sort=True)}
    output.mkdir(parents=True, exist_ok=False)
    folds = []
    for fold in split["folds"]:
        fold_id = int(fold["fold_id"])
        ids = data._seeded_subset(fold["validation_sample_ids"], 512, target_id="T3",
                                 lane="validation" if fold_id == 0 else f"fold-{fold_id}-validation")
        sequences, timestamps = [], []
        for sample_id in ids:
            _, group, cutoff = sample_id.split(":", 2)
            well = wells[group]
            matches = np.flatnonzero((well["DATEPRD"] == pd.Timestamp(cutoff)).to_numpy())
            if len(matches) != 1 or matches[0] < 30:
                raise ValueError("P7 cutoff identity is ambiguous or lacks history")
            history = well.iloc[matches[0] - 30:matches[0]]
            if not (history["DATEPRD"] < pd.Timestamp(cutoff)).all():
                raise ValueError("P7 history crosses its prediction cutoff")
            sequences.append(data._production_sequence(history))
            timestamps.append(history["DATEPRD"].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"))
        path = output / f"fold_{fold_id}_query.npz"
        np.savez_compressed(path, query_sequence=np.asarray(sequences),
                            history_timestamps=np.asarray(timestamps), sample_ids=np.asarray(ids))
        weight = next(row for row in summary["blend_weights"] if row["fold_id"] == fold_id)
        folds.append({"fold_id": fold_id, "query_rows": len(ids), "input_sha256": sha256_file(path),
                      "chronos_weight": weight["chronos_weight"],
                      "validation_sample_ids_sha256": data.canonical_sha256(list(ids)),
                      "time_semantics": "30 observation records, not necessarily 30 consecutive calendar days"})
    atomic_json(output / "recovery.json", {"status": "completed", "folds": folds,
        "source_sha256": LOCKS, "production_source": provenance,
        "query_targets_constructed": False, "query_targets_exported": False,
        "future_measurements_exported": False, "frozen_test_accessed": False,
        "registered": False, "deployed": False, "source_files_modified": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    recover(args.competition_root, args.output_directory)
