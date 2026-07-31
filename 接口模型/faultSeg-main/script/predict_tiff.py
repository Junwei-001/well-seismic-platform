#!/usr/bin/env python3
"""Run tiled FaultSeg3D inference on a 3D TIFF stack and create a preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom

from src.checkpoint import load_model
from src.data import normalize_seismic, parse_shape
from src.inference import choose_device, predict_volume, print_progress
from src.profiles import DEFAULT_CONFIG, load_profile
from src.thresholds import probability_summary, select_threshold
from src.visualization import save_orthogonal_preview, save_threshold_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="3D TIFF seismic stack")
    parser.add_argument("output", type=Path, help="output float32 probability TIFF")
    parser.add_argument("--profile", default="auto", help="auto, synthetic, field-raw, or fault-enhanced")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="override the profile checkpoint")
    parser.add_argument("--patch-size", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--overlap", type=parse_shape, default=(32, 32, 32))
    parser.add_argument(
        "--normalization", choices=("patch", "volume"), default="patch",
        help="z-score each patch (default) or the complete volume",
    )
    parser.add_argument(
        "--vertical-scale", type=float, default=1.0,
        help="resample the Z/time axis before inference, then restore it",
    )
    parser.add_argument("--preview", type=Path, help="orthogonal preview PNG")
    parser.add_argument("--threshold", default="profile", help="profile, otsu, quantile:0.98, or a value")
    parser.add_argument("--mask-output", type=Path, help="uint8 binary-mask TIFF")
    parser.add_argument("--threshold-sweep", type=Path, help="threshold comparison PNG")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-threshold-sweep", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile, args.input, args.config)
    checkpoint = args.checkpoint or profile.checkpoint
    device = choose_device(args.device)
    model, _ = load_model(checkpoint, device)
    source = tifffile.imread(args.input)
    if source.ndim != 3:
        raise ValueError(f"expected a 3D TIFF stack, got shape {source.shape}")
    if args.vertical_scale <= 0:
        raise ValueError("--vertical-scale must be positive")
    seismic = np.asarray(source, dtype=np.float32)
    if args.vertical_scale != 1.0:
        seismic = zoom(seismic, (args.vertical_scale, 1.0, 1.0), order=1)
    if args.normalization == "volume":
        seismic = normalize_seismic(seismic)
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(
        f"input_shape={source.shape} inference_shape={seismic.shape} "
        f"dtype={source.dtype} device={device} amp={amp_enabled}",
        flush=True,
    )
    probability = predict_volume(
        model,
        seismic,
        device,
        args.patch_size,
        args.overlap,
        amp_enabled,
        normalize_patches=args.normalization == "patch",
        weighted_blending=True,
        invalid_value=0.0 if args.normalization == "patch" else None,
        progress=print_progress,
    )
    if probability.shape[0] != source.shape[0]:
        probability = zoom(
            probability,
            (source.shape[0] / probability.shape[0], 1.0, 1.0),
            order=1,
        )
        probability = probability[: source.shape[0]]
    invalid_traces = np.all(source == 0, axis=0)
    probability[:, invalid_traces] = 0.0
    valid_probability = probability[:, ~invalid_traces]
    selection = select_threshold(args.threshold, valid_probability, profile.threshold)
    mask = (probability >= selection.value) & ~invalid_traces[None, :, :]
    positive_fraction = float(mask[:, ~invalid_traces].mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        args.output,
        np.asarray(probability, dtype=np.float32),
        bigtiff=True,
        metadata={"axes": "ZYX"},
    )
    preview = args.preview or args.output.with_name(args.output.stem + "_preview.png")
    save_orthogonal_preview(source, probability, preview, selection.value)
    mask_output = args.mask_output or args.output.with_name(args.output.stem + "_mask.tif")
    if not args.no_mask:
        tifffile.imwrite(mask_output, mask.astype(np.uint8), bigtiff=True, metadata={"axes": "ZYX"})
    sweep_output = args.threshold_sweep or args.output.with_name(args.output.stem + "_thresholds.png")
    if not args.no_threshold_sweep:
        thresholds = tuple(sorted(set((*profile.threshold_grid, selection.value))))
        save_threshold_sweep(source, probability, sweep_output, thresholds)
    metadata = {
        "source": str(args.input),
        "profile": profile.name,
        "profile_description": profile.description,
        "checkpoint": str(checkpoint),
        "shape_zyx": list(probability.shape),
        "threshold": {"value": selection.value, "method": selection.method},
        "probability": probability_summary(valid_probability, selection.value),
        "outputs": {
            "probability": str(args.output),
            "mask": None if args.no_mask else str(mask_output),
            "preview": str(preview),
            "threshold_sweep": None if args.no_threshold_sweep else str(sweep_output),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote probability={args.output} preview={preview} profile={profile.name} "
        f"threshold={selection.value:.6g} ({selection.method}) positive={positive_fraction:.3%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
