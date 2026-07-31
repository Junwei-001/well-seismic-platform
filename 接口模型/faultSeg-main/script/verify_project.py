#!/usr/bin/env python3
"""Verify that a copied FaultSeg3D project has its runtime files and models."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from src.profiles import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dependencies = {}
    failures = []
    for name in ("torch", "numpy", "matplotlib", "scipy", "tifffile", "segyio"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = getattr(module, "__version__", "available")
        except Exception as error:  # pragma: no cover - environment-dependent diagnostic
            dependencies[name] = None
            failures.append(f"dependency {name}: {error}")
    profiles = {}
    try:
        config = json.loads(args.config.read_text())
        for name, values in config["profiles"].items():
            checkpoint = Path(values["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = args.config.resolve().parent.parent / checkpoint
            profiles[name] = {
                "checkpoint": str(checkpoint),
                "exists": checkpoint.is_file(),
                "threshold": values["threshold"],
            }
            if not checkpoint.is_file():
                failures.append(f"profile {name}: missing {checkpoint}")
    except Exception as error:
        failures.append(f"profile config: {error}")
    report = {
        "project_root": str(root),
        "python": sys.version.split()[0],
        "dependencies": dependencies,
        "profiles": profiles,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
