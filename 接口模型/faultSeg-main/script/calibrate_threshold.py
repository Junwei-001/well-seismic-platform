#!/usr/bin/env python3
"""Calibrate a probability threshold on paired labeled validation volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checkpoint import load_model
from src.data import FaultVolumeDataset, discover_ids, parse_shape
from src.inference import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/faultseg-best.pt"))
    parser.add_argument("--seis", type=Path, default=Path("data/validation/seis"))
    parser.add_argument("--fault", type=Path, default=Path("data/validation/fault"))
    parser.add_argument("--shape", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--metric", choices=("dice", "iou", "youden"), default="dice")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/calibration/threshold.json"))
    parser.add_argument("--plot", type=Path, default=Path("output/calibration/threshold_curve.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 20:
        raise ValueError("--bins must be at least 20")
    ids = discover_ids(args.seis, args.fault)
    if not ids:
        raise FileNotFoundError("no paired validation volumes were found")
    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    dataset = FaultVolumeDataset(args.seis, args.fault, ids, args.shape)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    positive_histogram = np.zeros(args.bins, dtype=np.int64)
    negative_histogram = np.zeros(args.bins, dtype=np.int64)
    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.inference_mode():
        for index, (seismic, fault) in enumerate(loader, start=1):
            seismic = seismic.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                probability = torch.sigmoid(model(seismic))[0, 0].float().cpu().numpy()
            truth = fault[0, 0].numpy() >= 0.5
            bin_index = np.minimum((probability * args.bins).astype(np.int32), args.bins - 1)
            positive_histogram += np.bincount(bin_index[truth], minlength=args.bins)
            negative_histogram += np.bincount(bin_index[~truth], minlength=args.bins)
            print(f"volume {index}/{len(dataset)}", flush=True)

    true_positive = np.cumsum(positive_histogram[::-1], dtype=np.float64)[::-1]
    false_positive = np.cumsum(negative_histogram[::-1], dtype=np.float64)[::-1]
    positive_total = float(positive_histogram.sum())
    negative_total = float(negative_histogram.sum())
    false_negative = positive_total - true_positive
    true_negative = negative_total - false_positive
    eps = 1e-12
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (positive_total + eps)
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative + eps)
    iou = true_positive / (true_positive + false_positive + false_negative + eps)
    specificity = true_negative / (negative_total + eps)
    youden = recall + specificity - 1.0
    curves = {"dice": dice, "iou": iou, "youden": youden}
    best_index = int(np.argmax(curves[args.metric]))
    thresholds = np.arange(args.bins, dtype=np.float64) / args.bins
    result = {
        "checkpoint": str(args.checkpoint),
        "validation_volumes": len(dataset),
        "optimized_metric": args.metric,
        "threshold": float(thresholds[best_index]),
        "metrics": {
            "precision": float(precision[best_index]),
            "recall": float(recall[best_index]),
            "dice": float(dice[best_index]),
            "iou": float(iou[best_index]),
            "specificity": float(specificity[best_index]),
            "youden": float(youden[best_index]),
        },
        "histogram_bins": args.bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(thresholds, dice, label="Dice")
    axis.plot(thresholds, iou, label="IoU")
    axis.plot(thresholds, precision, label="Precision", alpha=0.8)
    axis.plot(thresholds, recall, label="Recall", alpha=0.8)
    axis.axvline(thresholds[best_index], color="black", linestyle="--", label=f"selected={thresholds[best_index]:.3f}")
    axis.set(xlabel="Probability threshold", ylabel="Metric", xlim=(0, 1), ylim=(0, 1), title="FaultSeg3D threshold calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=170)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.plot}")


if __name__ == "__main__":
    main()
