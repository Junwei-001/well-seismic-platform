"""Thin subprocess boundary for frozen WellFuse P17 geobody inference.

This script is intentionally kept outside the web process so the platform can
use its small Python 3.11 service environment while inference runs in the
CUDA-enabled WellFuse environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _triple(value: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected T,INLINE,XLINE")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wellfuse-root", type=Path, required=True)
    parser.add_argument("--task", choices=("channel", "karst"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--segy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--survey-name", default="platform_input")
    parser.add_argument("--patch", type=_triple, default=(64, 96, 96))
    parser.add_argument("--overlap", type=_triple, default=(32, 48, 48))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-voxels", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--iline-byte", type=int, default=189)
    parser.add_argument("--xline-byte", type=int, default=193)
    args = parser.parse_args()

    source_root = args.wellfuse_root.resolve()
    source_package = source_root / "src"
    if not source_package.is_dir():
        raise FileNotFoundError(f"WellFuse source package not found: {source_package}")
    sys.path.insert(0, str(source_package))

    from wellfuse5090.p17_geobody_inference import run_full_candidate_inference

    result = run_full_candidate_inference(
        task=args.task,
        checkpoint_path=args.checkpoint.resolve(),
        segy_path=args.segy.resolve(),
        output_root=args.output.resolve(),
        survey_name=args.survey_name,
        patch_shape=args.patch,
        overlap=args.overlap,
        threshold=args.threshold,
        minimum_voxels=args.minimum_voxels,
        device=args.device,
        iline_byte=args.iline_byte,
        xline_byte=args.xline_byte,
    )
    print("WELLFUSE_GEOBODY_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
