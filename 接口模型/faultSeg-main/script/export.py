#!/usr/bin/env python3
"""Export a trained FaultSeg3D checkpoint as a deployable TorchScript model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.checkpoint import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/faultseg-best.pt"))
    parser.add_argument("--output", type=Path, default=Path("model/faultseg-best.ts"))
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint, torch.device("cpu"))
    scripted = torch.jit.script(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, args.output)
    print(f"exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
