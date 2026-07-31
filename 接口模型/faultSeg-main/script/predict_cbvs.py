#!/usr/bin/env python3
"""Run tiled FaultSeg3D inference directly on an OpendTect CBVS cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from src.cbvs import CBVSVolume
from src.checkpoint import load_model
from src.data import parse_shape
from src.inference import choose_device, predict_volume, print_progress
from src.profiles import DEFAULT_CONFIG, load_profile
from src.thresholds import probability_summary, select_threshold
from src.visualization import save_orthogonal_preview, save_threshold_sweep


def parse_slice(text: str) -> slice:
    values = text.split(":")
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range must be START:STOP (zero-based, STOP exclusive)")
    start, stop = (int(value) if value else None for value in values)
    if start is not None and start < 0 or stop is not None and stop <= 0:
        raise argparse.ArgumentTypeError("range bounds must be non-negative")
    if start is not None and stop is not None and stop <= start:
        raise argparse.ArgumentTypeError("range STOP must exceed START")
    return slice(start, stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="output float32 probability TIFF")
    parser.add_argument("--profile", default="auto", help="auto, field-raw, fault-enhanced, or synthetic")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="override the profile checkpoint")
    parser.add_argument("--component", default="0", help="component index or exact name")
    parser.add_argument("--samples", type=parse_slice, default=slice(None), help="sample START:STOP")
    parser.add_argument("--inlines", type=parse_slice, default=slice(None), help="inline-index START:STOP")
    parser.add_argument("--crosslines", type=parse_slice, default=slice(None), help="crossline-index START:STOP")
    parser.add_argument("--patch-size", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--overlap", type=parse_shape, default=(32, 32, 32))
    parser.add_argument("--preview", type=Path)
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
    source = CBVSVolume(args.input)
    component: int | str = int(args.component) if args.component.isdigit() else args.component
    seismic, valid_traces = source.read_crop(
        args.samples, args.inlines, args.crosslines, component
    )
    if any(size < patch for size, patch in zip(seismic.shape, args.patch_size)):
        raise ValueError(f"crop {seismic.shape} is smaller than patch size {args.patch_size}")
    device = choose_device(args.device)
    model, _ = load_model(checkpoint, device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(
        f"profile={profile.name} checkpoint={checkpoint.name} component={component!r} "
        f"crop_shape={seismic.shape} valid_traces={valid_traces.mean():.3%} "
        f"device={device} amp={amp_enabled}",
        flush=True,
    )
    probability = predict_volume(
        model,
        seismic,
        device,
        args.patch_size,
        args.overlap,
        amp_enabled,
        normalize_patches=True,
        weighted_blending=True,
        progress=print_progress,
    )
    probability[:, ~valid_traces] = 0.0
    valid_probability = probability[:, valid_traces]
    selection = select_threshold(args.threshold, valid_probability, profile.threshold)
    mask = (probability >= selection.value) & valid_traces[None, :, :]
    positive_fraction = float(mask[:, valid_traces].mean())
    display_seismic = np.nan_to_num(seismic, nan=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, probability.astype(np.float32), bigtiff=True, metadata={"axes": "ZYX"})
    preview = args.preview or args.output.with_name(args.output.stem + "_preview.png")
    save_orthogonal_preview(display_seismic, probability, preview, selection.value)
    mask_output = args.mask_output or args.output.with_name(args.output.stem + "_mask.tif")
    if not args.no_mask:
        tifffile.imwrite(mask_output, mask.astype(np.uint8), bigtiff=True, metadata={"axes": "ZYX"})
    sweep_output = args.threshold_sweep or args.output.with_name(args.output.stem + "_thresholds.png")
    if not args.no_threshold_sweep:
        thresholds = tuple(sorted(set((*profile.threshold_grid, selection.value))))
        save_threshold_sweep(display_seismic, probability, sweep_output, thresholds)
    item = source.components[component] if isinstance(component, int) else next(
        value for value in source.components if value.name == component
    )
    metadata = {
        "source": str(args.input),
        "profile": profile.name,
        "profile_description": profile.description,
        "checkpoint": str(checkpoint),
        "component": item.name,
        "shape_zyx": list(seismic.shape),
        "sample_start_ms": item.start_ms + (args.samples.start or 0) * item.step_ms,
        "sample_step_ms": item.step_ms,
        "inline_index_range": [args.inlines.start, args.inlines.stop],
        "crossline_index_range": [args.crosslines.start, args.crosslines.stop],
        "valid_trace_fraction": float(valid_traces.mean()),
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
        f"wrote probability={args.output} preview={preview} threshold={selection.value:.6g} "
        f"({selection.method}) positive={positive_fraction:.3%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
