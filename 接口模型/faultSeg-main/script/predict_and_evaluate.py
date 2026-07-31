#!/usr/bin/env python3
"""One-click TIFF/CBVS/DAT/SGY prediction with optional labeled evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import parse_shape
from src.inference import print_progress
from src.pipeline import PredictionRequest, run_prediction
from src.profiles import DEFAULT_CONFIG


def _component(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--format", choices=("auto", "tiff", "cbvs", "dat", "sgy"), default="auto")
    parser.add_argument("--shape", type=parse_shape, help="required storage shape for DAT")
    parser.add_argument("--component", type=_component, default=0)
    parser.add_argument("--label", type=Path, help="optional 3D ground-truth volume")
    parser.add_argument("--label-format", choices=("auto", "tiff", "cbvs", "dat", "sgy"), default="auto")
    parser.add_argument("--label-shape", type=parse_shape)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", default="profile", help="profile, best/auto (with label), otsu, quantile:0.98, or 0..1")
    parser.add_argument("--optimize", choices=("dice", "iou", "youden"), default="dice")
    parser.add_argument("--patch-size", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--overlap", type=parse_shape, default=(32, 32, 32))
    parser.add_argument("--preprocess", choices=("none", "fault-enhancement"), default="none")
    parser.add_argument("--similarity-gate", type=float, default=0.85)
    parser.add_argument("--similarity-half-window", type=int, default=7)
    parser.add_argument("--inline-byte", type=int, default=189, help="SEG-Y inline header byte")
    parser.add_argument("--crossline-byte", type=int, default=193, help="SEG-Y crossline header byte")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = PredictionRequest(
        input_path=args.input, output_dir=args.output_dir, input_format=args.format,
        shape=args.shape, label_path=args.label, label_format=args.label_format,
        label_shape=args.label_shape, profile=args.profile, config_path=args.config,
        checkpoint=args.checkpoint, component=args.component, threshold=args.threshold,
        optimize=args.optimize, patch_size=args.patch_size, overlap=args.overlap,
        preprocessing=args.preprocess, similarity_gate=args.similarity_gate,
        similarity_half_window=args.similarity_half_window,
        inline_byte=args.inline_byte, crossline_byte=args.crossline_byte,
        device=args.device, amp=not args.no_amp,
    )
    result = run_prediction(request, progress=print_progress)
    print(json.dumps(result, indent=2))
    print(f"prediction artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
