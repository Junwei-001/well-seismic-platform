#!/usr/bin/env python3
"""Instantly regenerate a fault mask/QC from saved probabilities without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from src.data import parse_shape
from src.thresholds import probability_summary, select_threshold
from src.visualization import save_orthogonal_preview, save_threshold_sweep
from src.volumes import read_volume


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(","))
    if not values or any(not 0 <= value <= 1 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in [0,1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probability", type=Path, help="float32 probability TIFF")
    parser.add_argument("mask", type=Path, help="output uint8 mask TIFF")
    parser.add_argument("--threshold", default="0.5", help="fixed, otsu, or quantile:0.98")
    parser.add_argument("--seismic", type=Path, help="optional TIFF/CBVS/DAT/SGY input for overlay")
    parser.add_argument("--seismic-format", default="auto", choices=("auto", "tiff", "cbvs", "dat", "sgy"))
    parser.add_argument("--shape", type=parse_shape, help="storage shape for DAT seismic")
    parser.add_argument("--component", default="0")
    parser.add_argument("--threshold-grid", type=parse_grid, default=(0.3, 0.5, 0.7, 0.9))
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--threshold-sweep", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probability = np.asarray(tifffile.imread(args.probability), dtype=np.float32)
    if probability.ndim != 3:
        raise ValueError(f"probability must be 3D, got {probability.shape}")
    selection = select_threshold(args.threshold, probability, 0.5)
    mask = probability >= selection.value
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.mask, mask.astype(np.uint8), bigtiff=True, metadata={"axes": "ZYX"})
    outputs: dict[str, str | None] = {"mask": str(args.mask.resolve()), "preview": None, "threshold_sweep": None}
    if args.seismic:
        component: int | str = int(args.component) if args.component.isdigit() else args.component
        seismic = read_volume(args.seismic, format=args.seismic_format, shape=args.shape, component=component).data
        if seismic.shape != probability.shape:
            raise ValueError(f"seismic shape {seismic.shape} differs from probability {probability.shape}")
        display = np.nan_to_num(seismic, nan=0.0)
        preview = args.preview or args.mask.with_name(args.mask.stem + "_preview.png")
        sweep = args.threshold_sweep or args.mask.with_name(args.mask.stem + "_thresholds.png")
        save_orthogonal_preview(display, probability, preview, selection.value)
        save_threshold_sweep(display, probability, sweep, tuple(sorted(set((*args.threshold_grid, selection.value)))))
        outputs.update({"preview": str(preview.resolve()), "threshold_sweep": str(sweep.resolve())})
    result = {
        "probability": str(args.probability.resolve()),
        "threshold": {"value": selection.value, "method": selection.method},
        "summary": probability_summary(probability, selection.value),
        "outputs": outputs,
    }
    args.mask.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
