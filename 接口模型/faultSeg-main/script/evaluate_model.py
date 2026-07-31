#!/usr/bin/env python3
"""One-click FaultSeg3D evaluation, threshold selection, and visual QC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checkpoint import load_model
from src.data import FaultVolumeDataset, discover_ids, parse_shape
from src.evaluation import METRIC_NAMES, ProbabilityHistogram
from src.inference import choose_device


def parse_threshold(text: str) -> str | float:
    if text.lower() == "auto":
        return "auto"
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be auto or a number in [0, 1]")
    return value


def parse_threshold_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("threshold grid must be comma-separated values in [0, 1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/faultseg-best.pt"))
    parser.add_argument("--seis", type=Path, default=Path("data/validation/seis"))
    parser.add_argument("--fault", type=Path, default=Path("data/validation/fault"))
    parser.add_argument("--shape", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation/latest"))
    parser.add_argument("--threshold", type=parse_threshold, default="auto", help="auto or fixed probability, e.g. 0.5")
    parser.add_argument("--optimize", choices=("dice", "iou", "youden"), default="dice")
    parser.add_argument("--bins", type=int, default=1000, help="threshold resolution for automatic selection")
    parser.add_argument("--threshold-grid", type=parse_threshold_grid, default=(0.3, 0.5, 0.7, 0.9))
    parser.add_argument("--visual-volumes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _write_curves(path: Path, curves: dict[str, np.ndarray]) -> None:
    fields = ("threshold",) + METRIC_NAMES
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(curves["threshold"])):
            writer.writerow({name: float(curves[name][index]) for name in fields})


def _write_per_volume(
    path: Path,
    volume_ids: Sequence[int],
    histograms: Sequence[ProbabilityHistogram],
    threshold: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for volume_id, histogram in zip(volume_ids, histograms):
        row: dict[str, float | int] = {"volume_id": volume_id, "threshold": threshold}
        row.update(histogram.metrics_at(threshold))
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _save_curves(path: Path, curves: dict[str, np.ndarray], selected: float) -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for metric in ("dice", "iou", "precision", "recall"):
        axes[0].plot(curves["threshold"], curves[metric], label=metric.capitalize())
    axes[0].axvline(selected, color="black", linestyle="--", label=f"selected={selected:.3f}")
    axes[0].set(xlabel="Probability threshold", ylabel="Metric", xlim=(0, 1), ylim=(0, 1), title="Threshold calibration")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(curves["recall"], curves["precision"], color="tab:purple")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Precision–recall curve")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _save_qualitative(
    path: Path,
    samples: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    selected: float,
    threshold_grid: Sequence[float],
) -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    columns = 4 + len(threshold_grid)
    figure, axes = plt.subplots(len(samples), columns, figsize=(3.1 * columns, 3.2 * len(samples)), squeeze=False, constrained_layout=True)
    for row, (volume_id, seismic, truth, probability) in enumerate(samples):
        slice_index = int(np.argmax(truth.sum(axis=(1, 2))))
        seis = seismic[slice_index]
        label = truth[slice_index]
        prob = probability[slice_index]
        low, high = np.percentile(seis[np.isfinite(seis)], (1, 99))
        axes[row, 0].imshow(seis, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 0].set_title(f"Volume {volume_id} · Z={slice_index}\nSeismic")
        axes[row, 1].imshow(label, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(prob, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[row, 2].set_title("Probability")
        axes[row, 3].imshow(seis, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 3].imshow(np.ma.masked_where(prob < selected, prob), cmap="autumn", vmin=selected, vmax=1, alpha=0.72, aspect="auto")
        axes[row, 3].contour(label, levels=[0.5], colors=["cyan"], linewidths=0.6)
        axes[row, 3].set_title(f"Selected p ≥ {selected:.3f}\ncyan = truth")
        for offset, threshold in enumerate(threshold_grid, start=4):
            axes[row, offset].imshow(prob >= threshold, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[row, offset].set_title(f"Mask p ≥ {threshold:g}\npositive={(prob >= threshold).mean():.2%}")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle("FaultSeg3D validation: threshold sensitivity", fontsize=16)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.visual_volumes < 1:
        raise ValueError("--visual-volumes must be at least 1")
    ids = discover_ids(args.seis, args.fault)
    if not ids:
        raise FileNotFoundError("no paired evaluation volumes were found")
    device = choose_device(args.device)
    model, checkpoint_metadata = load_model(args.checkpoint, device)
    dataset = FaultVolumeDataset(args.seis, args.fault, ids, args.shape)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    aggregate = ProbabilityHistogram(args.bins)
    per_volume: list[ProbabilityHistogram] = []
    samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.inference_mode():
        for index, (seismic, fault) in enumerate(loader):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                probability = torch.sigmoid(model(seismic.to(device, non_blocking=True)))[0, 0].float().cpu().numpy()
            seismic_array = seismic[0, 0].numpy()
            truth = fault[0, 0].numpy() >= 0.5
            histogram = ProbabilityHistogram(args.bins)
            histogram.update(probability, truth)
            aggregate.merge(histogram)
            per_volume.append(histogram)
            if len(samples) < min(args.visual_volumes, len(dataset)):
                samples.append((ids[index], seismic_array, truth, probability))
            print(f"volume {index + 1}/{len(dataset)}", flush=True)

    if args.threshold == "auto":
        selected, selected_metrics = aggregate.best(args.optimize)
        threshold_mode = "auto"
    else:
        selected = float(args.threshold)
        selected_metrics = aggregate.metrics_at(selected)
        threshold_mode = "fixed"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    curves = aggregate.curves()
    _write_curves(output / "threshold_metrics.csv", curves)
    rows = _write_per_volume(output / "per_volume_metrics.csv", ids, per_volume, selected)
    _save_curves(output / "threshold_curves.png", curves, selected)
    visual_thresholds = tuple(dict.fromkeys((*args.threshold_grid, selected)))
    _save_qualitative(output / "qualitative_comparison.png", samples, selected, visual_thresholds)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_seismic": str(args.seis.resolve()),
        "validation_fault": str(args.fault.resolve()),
        "volumes": len(dataset),
        "shape": list(args.shape),
        "device": str(device),
        "amp": amp_enabled,
        "threshold_mode": threshold_mode,
        "optimized_metric": args.optimize if threshold_mode == "auto" else None,
        "threshold": selected,
        "threshold_resolution": 1.0 / args.bins,
        "metrics": selected_metrics,
        "per_volume_dice_mean": float(np.mean([row["dice"] for row in rows])),
        "per_volume_dice_std": float(np.std([row["dice"] for row in rows])),
        "checkpoint_metadata": {key: checkpoint_metadata[key] for key in ("epoch", "best_metric", "bottleneck_channels") if key in checkpoint_metadata},
        "artifacts": {
            "threshold_metrics": "threshold_metrics.csv",
            "per_volume_metrics": "per_volume_metrics.csv",
            "threshold_curves": "threshold_curves.png",
            "qualitative_comparison": "qualitative_comparison.png",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"evaluation artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
