"""Thin process boundary for P17/P18 unknown-survey well inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wellfuse-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    source = (args.wellfuse_root / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"WellFuse source directory not found: {source}")
    sys.path.insert(0, str(source))
    from wellfuse5090.unknown_well_runtime import run_unknown_well_request

    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_unknown_well_request(request)
    print("WELLFUSE_WELL_RESULT=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
