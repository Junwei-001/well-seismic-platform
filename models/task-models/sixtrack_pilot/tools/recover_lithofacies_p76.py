"""Read-only recovery of hash-pinned P76 development windows, without query labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellseismic_sixtrack_pilot.manifest import atomic_json, sha256_file

ARCHIVE_SHA256 = "9c2d27b0bf44ce96cf6da743949ab6db866cdf9661b840b58c2df34c7cee0113"
ARCHIVE_RELATIVE = "_pipelines/02_task_datasets/lithofacies/_outputs/p68_lithofacies_incumbent_confirmation/input/development_train_logo3.npz"


def recover(competition_root: Path, output: Path):
    if output.exists(): raise ValueError("Use a new output directory")
    source = competition_root / ARCHIVE_RELATIVE
    if sha256_file(source) != ARCHIVE_SHA256: raise ValueError("P76 source archive hash mismatch")
    folds, payloads = [], []
    with np.load(source, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        if (manifest["schema_version"] != "lithofacies-p68-train-only-logo3/v1"
                or manifest["frozen_test_accessed"] is not False
                or manifest["guard_features_or_labels_accessed"] is not False):
            raise ValueError("P76 development source violates the holdout firewall")
        for fold_id in range(3):
            row = next(row for row in manifest["folds"] if row["fold_id"] == fold_id)
            if row["preprocessor_fit_scope"] != "fold_train_families_only" or row["guard_features_or_labels_accessed"] is not False:
                raise ValueError("P76 fold preprocessing scope drift")
            fields = {key: archive[f"f{fold_id}_{key}"] for key in ("train_well", "train_seismic", "train_labels")}
            fields.update({f"query_{kind}": archive[f"f{fold_id}_validation_{kind}"] for kind in ("well", "seismic")})
            if len(fields["train_well"]) != row["train_samples"] or len(fields["query_well"]) != row["validation_samples"]:
                raise ValueError("P76 development sample roster drift")
            folds.append(row); payloads.append(fields)
    output.mkdir(parents=True, exist_ok=False)
    for fold_id, (row, fields) in enumerate(zip(folds, payloads)):
        path = output / f"fold_{fold_id}_input.npz"
        np.savez_compressed(path, **fields)
        row = dict(row, input_sha256=sha256_file(path), fields=list(fields))
        atomic_json(output / f"fold_{fold_id}_input.json", row)
    atomic_json(output / "recovery.json", {
        "status": "completed", "source_archive_sha256": ARCHIVE_SHA256, "split_hash": manifest["split_hash"],
        "query_labels_loaded": False, "guard_or_frozen_test_loaded": False,
        "folds": folds, "source_files_modified": False, "platform_registered": False, "deployed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    recover(args.competition_root, args.output_directory)


if __name__ == "__main__": main()
