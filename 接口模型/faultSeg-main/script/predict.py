#!/usr/bin/env python3
"""Run FaultSeg3D inference on a raw float32 seismic volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.checkpoint import load_model
from src.data import normalize_seismic, parse_shape, read_raw, to_model_order
from src.inference import choose_device, predict_volume, print_progress
from src.thresholds import validate_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input float32 .dat volume")
    parser.add_argument("output", type=Path, help="output float32 fault probabilities")
    parser.add_argument("--checkpoint", type=Path, default=Path("model/faultseg-best.pt"))
    parser.add_argument(
        "--shape", type=parse_shape, default=(128, 128, 128),
        help="input storage shape before transpose (default: 128,128,128)",
    )
    parser.add_argument(
        "--patch-size", type=parse_shape,
        help="model-order patch size; omit to process the complete volume",
    )
    parser.add_argument(
        "--overlap", type=parse_shape, default=(32, 32, 32),
        help="patch overlap when --patch-size is used",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-output", type=Path, help="optional uint8 binary-mask .dat")
    parser.add_argument(
        "--model-order-output", action="store_true",
        help="do not transpose predictions back to the input storage order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = validate_threshold(args.threshold)
    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    seismic = normalize_seismic(to_model_order(read_raw(args.input, args.shape)))
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(
        f"device={device} model_shape={seismic.shape} "
        f"bottleneck={model.bottleneck_channels} amp={amp_enabled}"
    )
    prediction = predict_volume(
        model,
        seismic,
        device,
        args.patch_size,
        args.overlap,
        amp_enabled,
        progress=print_progress,
    )
    output = prediction if args.model_order_output else prediction.transpose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(output, dtype=np.float32).tofile(args.output)
    if args.mask_output:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(output >= threshold, dtype=np.uint8).tofile(args.mask_output)
    print(
        f"wrote {args.output} shape={output.shape} "
        f"min={prediction.min():.6g} max={prediction.max():.6g} "
        f"mean={prediction.mean():.6g}"
    )
    if args.mask_output:
        print(f"wrote {args.mask_output} threshold={threshold:g}")


if __name__ == "__main__":
    main()
