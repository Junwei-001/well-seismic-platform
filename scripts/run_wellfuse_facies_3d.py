"""Execute the frozen P17 Facies-3D candidate in the WellFuse runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


def _validate_geometry(
    seismic: Mapping[str, object],
    *,
    expected_trace_count: int,
    expected_sample_count: int,
) -> dict[str, object]:
    shape_value = seismic.get("shape_t_inline_xline")
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 3:
        raise RuntimeError("Facies-3D manifest has no TWT/inline/xline shape")
    shape = tuple(int(value) for value in shape_value)
    if shape[0] != expected_sample_count or shape[1] <= 1 or shape[2] <= 1:
        raise RuntimeError(
            "Facies-3D rejected collapsed or mismatched seismic geometry: "
            f"shape={list(shape)}, expected_samples={expected_sample_count}"
        )
    valid_fraction = float(seismic.get("valid_trace_fraction", float("nan")))
    tolerance = max(1, (expected_trace_count + 999) // 1000)
    if not math.isfinite(valid_fraction) or not 0.0 < valid_fraction <= 1.0:
        raise RuntimeError(
            f"Facies-3D rejected invalid valid_trace_fraction={valid_fraction}"
        )
    inferred_trace_count = int(round(valid_fraction * shape[1] * shape[2]))
    if abs(
        inferred_trace_count - expected_trace_count
    ) > tolerance:
        raise RuntimeError(
            "Facies-3D rejected a grid inconsistent with the inspected SEG-Y: "
            f"inferred={inferred_trace_count}, expected={expected_trace_count}, "
            f"tolerance={tolerance}"
        )
    return {
        "shape_t_inline_xline": list(shape),
        "expected_trace_count": expected_trace_count,
        "inferred_trace_count": inferred_trace_count,
        "trace_count_tolerance": tolerance,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wellfuse-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    source = (args.wellfuse_root / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"WellFuse source directory not found: {source}")
    sys.path.insert(0, str(source))
    from wellfuse5090.p17_facies_products import run_facies_3d_candidate_inference

    request = json.loads(args.request.read_text(encoding="utf-8"))
    requested_device = str(request.get("requested_device", "auto")).casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported Facies-3D device: {requested_device}")
    manifest = run_facies_3d_candidate_inference(
        checkpoint_path=request["checkpoint_path"],
        segy_path=request["segy_path"],
        output_root=request["output_directory"],
        mode=request["mode"],
        patch_shape=request["patch_shape"],
        overlap=request["overlap"],
        iline_byte=int(request.get("iline_byte", 9)),
        xline_byte=int(request.get("xline_byte", 21)),
    )
    geometry_validation = _validate_geometry(
        manifest["seismic"],
        expected_trace_count=int(request["expected_trace_count"]),
        expected_sample_count=int(request["expected_sample_count"]),
    )
    result = {
        "model_id": "wellfuse_facies_3d_p17",
        "model_executed": True,
        "execution_status": "experimental_weak_supervision_candidate",
        "mode": manifest["mode"],
        "manifest": str((Path(request["output_directory"]) / "manifest.json").resolve()),
        "checkpoint": manifest["checkpoint"],
        "seismic": manifest["seismic"],
        "outputs": manifest["outputs"],
        "completed_tiles": int(manifest["sliding_window"]["completed_tiles"]),
        "scientific_status": manifest["scientific_status"],
        "requested_device": requested_device,
        "device": "cpu",
        "geometry_validation": geometry_validation,
    }
    print("WELLFUSE_FACIES3D_RESULT=" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
