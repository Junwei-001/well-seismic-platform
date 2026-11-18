#!/usr/bin/env python3
"""Recover original development seismic volume without dense property labels.

Reuse the hash-bound original GRID and weak-time geometry parser. Seismic
sampling runs through the portable worker. Optional original point-feature
inputs provide a direct five-fold parity check, without query targets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from wellseismic_sixtrack_pilot.manifest import atomic_json, sha256_file
from wellseismic_sixtrack_pilot.worker.reconstruction_seismic import sample_attributes


def run(args) -> dict:
    output = args.output_directory.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    science = args.competition_root / "_pipelines/02_task_datasets/reconstruction"
    source = science / "build_dataset.py"
    if sha256_file(source) != "8b5d2e33b729136e71bef518fa155310d3601e9e194480798e3def288310d2ff":
        raise ValueError("Original geometry source hash mismatch")
    sys.path[:0] = [str(science.resolve()), str(args.competition_root.resolve())]
    geometry = importlib.import_module("build_dataset")
    geometry.WELL_TIE = args.well_tie.resolve()
    with ZipFile(args.eclipse_zip) as archive:
        shape, centres, active = geometry._parse_eclipse_grid(archive)
        with archive.open(geometry.ECLIPSE_GRID_MEMBER) as stream:
            grid_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if grid_hash != "634fc63884fc18c1b98b45d6e8c579bb968cccba540d3480042e1538d4edb781" or shape != (63, 100, 108):
        raise ValueError("Original GRID identity mismatch")
    reference = SimpleNamespace(active=active.reshape(shape), **{
        name: centres[:, column].reshape(shape) for column, name in enumerate(("x", "y", "z"))
    })
    normalized = geometry._normalise_coordinates(reference)[:3]
    mask = reference.active[:, :, :72].copy()
    if int(mask.sum()) != 134222:
        raise ValueError("Original development active-cell count mismatch")
    physical = [getattr(reference, key)[:, :, :72][mask].astype(np.float64) for key in ("x", "y", "z")]
    with np.load(args.seismic_index, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    with threadpool_limits(limits=1):
        inline, crossline, in_bounds = geometry._xy_to_il_xl(physical[0], physical[1], index)
        if not in_bounds.all():
            raise ValueError("Original development cells lie outside seismic survey")
        twt, tie_audit = geometry._estimate_twt_from_weak_ties(*physical, index)
        attributes, seismic_audit = sample_attributes(args.segy, inline, crossline, twt, index)
    seismic = np.zeros((3, *mask.shape), dtype=np.float32)
    seismic[:, mask] = attributes.T
    coordinates = np.stack([value[:, :, :72] for value in normalized]).astype(np.float32)
    parity = []
    if args.feature_input_root is not None:
        receipt = json.loads((args.feature_input_root / "parity.json").read_text())
        tree = cKDTree(coordinates[:, mask].T.astype(np.float64))
        for entry in receipt["feature_inputs"]:
            path = args.feature_input_root / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("Original point-feature input hash mismatch")
            with np.load(path, allow_pickle=False) as archive:
                for prefix in ("train", "query"):
                    raw = archive[f"{prefix}_raw_features"]
                    distance, rows = tree.query(raw[:, 3:6])
                    if float(np.max(distance)) > 1e-8:
                        raise ValueError("Original feature coordinates do not match recovered geometry")
                    expected = raw[:, :3].astype(np.float32)
                    np.testing.assert_array_equal(attributes[rows], expected)
                    parity.append({"fold_id": entry["fold_id"], "role": prefix, "rows": len(raw),
                                   "seismic_attributes_bitwise_equal": True,
                                   "maximum_coordinate_match_distance": float(np.max(distance))})
    output.mkdir(parents=True)
    volume = output / "development_seismic.npz"
    np.savez_compressed(volume, seismic=seismic, active=mask, normalized_coordinates=coordinates)
    result = {
        "status": "completed", "scope": "original development seismic/geometry recovery only",
        "volume_shape_kji": list(mask.shape), "seismic_shape": list(seismic.shape),
        "active_cells": int(mask.sum()), "grid_member_sha256": grid_hash,
        "volume_sha256": sha256_file(volume), "point_parity": parity,
        "seismic_sampling": seismic_audit, "weak_time_alignment": tie_audit,
        "input_sha256": {"segy": sha256_file(args.segy), "seismic_index": sha256_file(args.seismic_index),
                         "well_tie": sha256_file(args.well_tie), "geometry_source": sha256_file(source)},
        "well_tie_fields_read": ["depth_m", "twt_est_ms", "inline", "crossline"],
        "raw_poro_or_init_labels_read": False, "query_labels_read": False,
        "frozen_test_labels_read": False, "train_h5_recreated": False,
        "pretrained_gfm_executed": False, "new_area_verified": False, "deployed": False,
    }
    atomic_json(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("competition-root", "eclipse-zip", "segy", "seismic-index", "well-tie", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--feature-input-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
