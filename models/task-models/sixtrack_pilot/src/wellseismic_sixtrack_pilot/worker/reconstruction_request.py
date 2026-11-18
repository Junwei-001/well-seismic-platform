"""Platform worker request protocol for P21 prepared inputs; not registered yet."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..manifest import atomic_json, fail, load_request
from .reconstruction_p21 import run_prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = load_request(args.request)
        if set(request) != {"input_path", "input_sha256", "output_directory"}:
            raise ValueError("P21 worker request requires exactly input_path/input_sha256/output_directory")
        result = run_prepared(
            Path(request["input_path"]), request["input_sha256"], Path(request["output_directory"])
        )
    except Exception as exc:
        return fail(args.result, exc)
    atomic_json(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
