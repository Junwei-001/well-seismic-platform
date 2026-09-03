"""Thin subprocess boundary for P17 unknown-survey horizon candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pair(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    if len(result) != 2:
        raise ValueError("expected an inline,xline pair")
    return result


def _optional_integer(value: object) -> int | None:
    return None if value is None else int(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wellfuse-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.wellfuse_root.resolve()
    source_package = source_root / "src"
    if not source_package.is_dir():
        raise FileNotFoundError(f"WellFuse source package not found: {source_package}")
    sys.path.insert(0, str(source_package))

    request_path = args.request_json.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("horizon request JSON must contain an object")

    from wellfuse5090.p17_horizon_unknown import run_unknown_horizon_inference

    result = run_unknown_horizon_inference(
        segy_path=Path(request["segy_path"]).resolve(),
        checkpoint_paths=[Path(path).resolve() for path in request["checkpoint_paths"]],
        output_root=Path(request["output_root"]).resolve(),
        survey_name=str(request.get("survey_name", "platform_input")),
        context_horizons_path=request.get("context_horizons_path"),
        registration_points_path=request.get("registration_points_path"),
        registration_manifest_path=request.get("registration_manifest_path"),
        registration_points=request.get("registration_points"),
        registration_source=request.get("registration_source"),
        device_name=str(request.get("device", "cuda")),
        tile_size=_pair(request.get("tile_size"), (64, 64)),
        overlap=_pair(request.get("overlap"), (16, 16)),
        iline_byte=int(request.get("iline_byte", 189)),
        xline_byte=int(request.get("xline_byte", 193)),
        x_byte=int(request.get("x_byte", 181)),
        y_byte=int(request.get("y_byte", 185)),
        coordinate_scalar_byte=int(request.get("coordinate_scalar_byte", 71)),
        inline_start=_optional_integer(request.get("inline_start")),
        inline_count=_optional_integer(request.get("inline_count")),
        crossline_start=_optional_integer(request.get("crossline_start")),
        crossline_count=_optional_integer(request.get("crossline_count")),
    )
    print(
        "WELLFUSE_HORIZON_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
