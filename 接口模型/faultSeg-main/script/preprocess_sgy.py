#!/usr/bin/env python3
"""Apply the standalone F3-style fault-enhancement preprocessing to SEG-Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

from src.filters import fault_enhancement_filter
from src.volumes import read_volume, write_sgy_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .sgy/.segy")
    parser.add_argument("output", type=Path, help="enhanced .sgy or 3D .tif")
    parser.add_argument("--inline-byte", type=int, default=189)
    parser.add_argument("--crossline-byte", type=int, default=193)
    parser.add_argument("--half-window", type=int, default=7, help="7 samples = ±28 ms at 4 ms")
    parser.add_argument("--similarity-gate", type=float, default=0.85)
    parser.add_argument("--similarity-output", type=Path)
    parser.add_argument("--preview", type=Path)
    return parser.parse_args()


def _save_preview(path: Path, original: np.ndarray, enhanced: np.ndarray, similarity: np.ndarray, gate: float) -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    indices = tuple(size // 2 for size in original.shape)
    figure, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row, (axis, index) in enumerate(zip(range(3), indices)):
        source = np.take(original, index, axis=axis)
        result = np.take(enhanced, index, axis=axis)
        coherence = np.take(similarity, index, axis=axis)
        values = source[np.isfinite(source)]
        low, high = np.percentile(values, (1, 99))
        axes[row, 0].imshow(source, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 0].set_title("Original")
        axes[row, 1].imshow(result, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[row, 1].set_title("Fault-enhanced")
        axes[row, 2].imshow(coherence, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
        axes[row, 2].contour(coherence < gate, levels=[0.5], colors="red", linewidths=0.4)
        axes[row, 2].set_title(f"Similarity · red < {gate:g}")
        for item in axes[row]:
            item.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source = read_volume(
        args.input, format="sgy", inline_byte=args.inline_byte,
        crossline_byte=args.crossline_byte,
    )
    original = source.data
    enhanced, similarity = fault_enhancement_filter(
        original, half_window=args.half_window,
        similarity_gate=args.similarity_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".sgy", ".segy"}:
        source.data = enhanced
        write_sgy_like(args.input, args.output, source)
        output_format = "sgy"
    elif args.output.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(args.output, enhanced.astype(np.float32), bigtiff=True, metadata={"axes": "ZYX"})
        output_format = "tiff"
    else:
        raise ValueError("output must end in .sgy, .segy, .tif, or .tiff")
    similarity_path = args.similarity_output or args.output.with_name(args.output.stem + "_similarity.tif")
    tifffile.imwrite(similarity_path, similarity.astype(np.float32), bigtiff=True, metadata={"axes": "ZYX"})
    preview = args.preview or args.output.with_name(args.output.stem + "_filter_qc.png")
    _save_preview(preview, original, enhanced, similarity, args.similarity_gate)
    metadata = {
        "source": str(args.input.resolve()), "output": str(args.output.resolve()),
        "output_format": output_format, "shape_zyx": list(enhanced.shape),
        "filter": {
            "name": "F3 fault-enhancement reconstruction",
            "mode": "non-steered-approximation",
            "half_window_samples": args.half_window,
            "similarity_gate": args.similarity_gate,
            "sequence": ["normalized similarity", "3x3 median", "minimum-similarity position", "similarity gate"],
            "limitation": "Standalone SEG-Y contains no dip-steering cube; output is not sample-identical to OpendTect FEF.",
        },
        "similarity": str(similarity_path.resolve()), "preview": str(preview.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
