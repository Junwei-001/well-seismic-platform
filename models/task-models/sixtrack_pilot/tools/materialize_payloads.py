#!/usr/bin/env python3
"""Copy the hash-bound six-track payloads into ``<model_id>/<version>/payload/``.

This is the recipe for the git-ignored payload files.  Sources are only read;
every source is hashed before copying and every copy is hashed again.  Identical
payloads shared by several model ids (the TabICL checkpoint, the development
archive) are hard-linked to the first verified copy instead of duplicated.

    python tools/materialize_payloads.py \
        --competition-root ~/projects/师弟-军伟的比赛-2693e5 \
        --tabicl-checkpoint ~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/<rev>/tabicl-regressor-v2-20260212.ckpt
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wellseismic_sixtrack_pilot.manifest import (  # noqa: E402
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    TASK_MODELS_ROOT,
    read_manifest,
    sha256_file,
    verify_payload,
)


def source_path(entry: dict, args: argparse.Namespace) -> Path:
    source = entry["source"]
    if source.get("kind") == "facies_recovered":
        if args.facies_recovered_root is None:
            raise SystemExit("--facies-recovered-root is required for P81 slice/query/state payloads")
        return (args.facies_recovered_root.expanduser() / source["path"]).resolve()
    if source.get("kind") == "sweetspot_histories":
        if args.sweetspot_input_root is None:
            raise SystemExit("--sweetspot-input-root is required for P7 payloads")
        return (args.sweetspot_input_root.expanduser() / source["path"]).resolve()
    if source.get("kind") == "huggingface_chronos":
        if args.chronos_snapshot is None:
            raise SystemExit("--chronos-snapshot is required for P7 weights/config")
        return (args.chronos_snapshot.expanduser() / source["path"]).resolve()
    if source.get("kind") == "lithofacies_windows":
        if args.lithofacies_input_root is None:
            raise SystemExit("--lithofacies-input-root is required for P76 payloads")
        return (args.lithofacies_input_root.expanduser() / source["path"]).resolve()
    if source.get("kind") == "huggingface_lithofacies":
        if args.lithofacies_checkpoint is None:
            raise SystemExit("--lithofacies-checkpoint is required for P76 weights")
        return args.lithofacies_checkpoint.expanduser().resolve()
    if source.get("kind") == "reconstruction_prepared":
        if args.reconstruction_prepared_root is None:
            raise SystemExit("--reconstruction-prepared-root is required for P21 payloads")
        return (args.reconstruction_prepared_root.expanduser() / source["path"]).resolve()
    if source.get("kind") == "platform_recipe":
        return (TASK_MODELS_ROOT.parents[1] / source["path"]).resolve()
    if source.get("kind") == "huggingface":
        if args.tabicl_checkpoint is None:
            raise SystemExit("--tabicl-checkpoint is required for the TabICL payload")
        return args.tabicl_checkpoint.expanduser().resolve()
    return (args.competition_root.expanduser() / source["path"]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--tabicl-checkpoint", type=Path)
    parser.add_argument("--reconstruction-prepared-root", type=Path)
    parser.add_argument("--lithofacies-input-root", type=Path)
    parser.add_argument("--lithofacies-checkpoint", type=Path)
    parser.add_argument("--sweetspot-input-root", type=Path)
    parser.add_argument("--chronos-snapshot", type=Path)
    parser.add_argument("--facies-recovered-root", type=Path)
    args = parser.parse_args()

    materialized: dict[str, Path] = {}
    model_dirs = sorted(path.parent for path in TASK_MODELS_ROOT.glob(f"*/*/{MANIFEST_NAME}"))
    for model_dir in model_dirs:
        manifest = read_manifest(model_dir)
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            continue
        for entry in manifest["payload"]:
            expected = str(entry["sha256"]).lower()
            target = model_dir / entry["path"]
            if target.is_file() and sha256_file(target) == expected:
                materialized.setdefault(expected, target)
                continue
            if expected in materialized:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                os.link(materialized[expected], target)
            else:
                source = source_path(entry, args)
                if sha256_file(source) != expected:
                    raise SystemExit(f"source hash drift for {manifest['model_id']}:{entry['role']} at {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                shutil.copyfile(source, target)
                target.chmod(0o444)
                materialized[expected] = target
            print(f"{manifest['model_id']}: {entry['role']} -> {target.relative_to(TASK_MODELS_ROOT)}")
        verify_payload(model_dir)
        print(f"verified {model_dir.relative_to(TASK_MODELS_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
