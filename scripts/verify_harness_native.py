"""Reproduce native Harness + DVC validation in a new isolated source snapshot.

Uses synthetic data and the current environment; never copies .env, credentials,
existing task state, outputs, or evidence. No neural weights are fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def generate_inputs(destination):
    import numpy as np
    import segyio

    from well_seismic.alignment.well_tie import acoustic_reflectivity, ricker_wavelet

    input_dir = Path(destination)
    input_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(601, dtype=float)
    vp = np.full(len(depth), 2500.0)
    rho = 2.3 + 0.1 * np.sin(depth / 24.0) + 0.18 * (depth > 250)
    las = "~Version\nVERS. 2.0 : CWLS\nWRAP. NO : unwrapped\n~Well\nSTRT.M 0 : Start\nSTOP.M 600 : Stop\nSTEP.M 1 : Step\nNULL. -999.25 : Null\nWELL. SYNTHETIC-HARNESS : synthetic, not field data\n~Curve\nDEPT.M : Depth\nDT.US/M : Sonic\nRHOB.G/CM3 : Bulk density\nGR.API : Gamma ray\n~ASCII\n"
    las += (
        "\n".join(
            f"{d:.1f} 400 {r:.8f} {60 + 5 * np.sin(d / 40):.8f}"
            for d, r in zip(depth, rho)
        )
        + "\n"
    )
    time_axis = np.arange(701, dtype=float)
    trace = np.convolve(
        np.interp(
            time_axis,
            (depth[:-1] + depth[1:]) * 0.4,
            acoustic_reflectivity(vp, rho),
            left=0,
            right=0,
        ),
        ricker_wavelet(30, 1),
        "same",
    ).astype(np.float32)
    seismic = input_dir / "synthetic.sgy"
    spec = segyio.spec()
    spec.sorting, spec.format = 2, 5
    spec.samples, spec.ilines, spec.xlines = time_axis, [1, 2, 3], [1, 2, 3]
    with segyio.create(str(seismic), spec) as stream:
        stream.text[0] = segyio.tools.create_text_header(
            {
                1: "SYNTHETIC HARNESS ACCEPTANCE DATA - NOT FIELD DATA",
                2: "EPSG:32650; XY METRES; TWT MS; SRD 0 M MSL; CORRECTED TO SRD",
            }
        )
        stream.bin[segyio.BinField.Interval] = 1000
        for index in range(9):
            il, xl = divmod(index, 3)
            stream.header[index] = {
                segyio.TraceField.INLINE_3D: il + 1,
                segyio.TraceField.CROSSLINE_3D: xl + 1,
                segyio.TraceField.CDP_X: 500000 + il * 25,
                segyio.TraceField.CDP_Y: 4400000 + xl * 25,
                segyio.TraceField.SourceX: 500000 + il * 25,
                segyio.TraceField.SourceY: 4400000 + xl * 25,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CoordinateUnits: 1,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 1000,
                segyio.TraceField.DelayRecordingTime: 0,
            }
            stream.trace[index] = trace
    table = (
        "WELL,MD,TWT\n"
        + "\n".join(f"SYNTHETIC-HARNESS,{d},{d * 0.8}" for d in range(0, 601, 50))
        + "\n"
    )
    survey = (
        "WELL,MD(m),TVD(m),X(m),Y(m)\n"
        + "\n".join(
            f"SYNTHETIC-HARNESS,{d},{d},500025,4400025" for d in range(0, 601, 50)
        )
        + "\n"
    )
    files = [
        ("log", "synthetic.las", las.encode()),
        ("seismic", seismic.name, seismic.read_bytes()),
        (
            "well",
            "wellhead.csv",
            b"WELL,X,Y,KB,DATUM_UNIT,TD\nSYNTHETIC-HARNESS,500025,4400025,0,m,600\n",
        ),
        ("trajectory", "trajectory.csv", survey.encode()),
        ("time_depth", "timedepth.csv", table.encode()),
    ]
    for _role, name, content in files:
        (input_dir / name).write_bytes(content)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generate-only", type=Path)
    args = parser.parse_args()
    if args.generate_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        files = generate_inputs(args.generate_only)
        emit(
            {
                "data_kind": "synthetic",
                "files": [str(args.generate_only / name) for _, name, _ in files],
            }
        )
        return
    if args.isolated_root is None:
        source = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix="hy-native-"))
        for name in ("src", "configs", "interfaces"):
            shutil.copytree(
                source / name,
                root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        (root / "models").mkdir()
        for name in ("manifest.json", "INTERFACE_ONLY"):
            shutil.copy2(source / "models" / name, root / "models" / name)
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(("WELL_SEISMIC_GAIA", "WELLFUSE_", "GAIA_", "MINIMAX_")):
                env.pop(key)
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            WELL_SEISMIC_STATE_DB=str(root / "state.sqlite3"),
            WELLFUSE_MODEL_MODE="interfaces_only",
            WELL_SEISMIC_LLM_ENABLED="false",
        )
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--isolated-root",
            str(root),
        ]
        if args.output:
            command += ["--output", str(args.output.resolve())]
        emit(
            {"phase": "isolated_snapshot", "root": str(root), "data_kind": "synthetic"}
        )
        raise SystemExit(subprocess.call(command, cwd=root, env=env))

    root = args.isolated_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from well_seismic import api
    from well_seismic.gaia_ui import public_value

    assert api.PROJECT_ROOT == root
    assert (
        api._state_store.path.resolve().is_relative_to(root)
        if hasattr(api._state_store, "path")
        else True
    )
    harness = api._huiyan_harness
    owner = "c" * 64  # isolated visitor identity, not a real user account
    results = {
        "data_kind": "synthetic",
        "scope": "isolated native business + real DVC; no trained-model claim",
        "steps": [],
    }

    def action(operation, arguments):
        receipt = harness.execute(owner, operation, arguments, str(uuid.uuid4()))
        deadline = time.monotonic() + 180
        while receipt["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.2)
            receipt = harness.get_run(owner, receipt["execution_id"])
        results["steps"].append(receipt)
        emit(
            {
                "operation": operation,
                "status": receipt["status"],
                "task_id": receipt.get("task_id"),
                "error": receipt.get("error"),
            }
        )
        return receipt

    pid = action("project_create", {"name": "Synthetic native Harness verification"})[
        "project_id"
    ]
    files = generate_inputs(root / "synthetic_inputs")
    ids = []
    for role, name, content in files:
        uploaded = harness.upload(owner, pid, role, name, content)
        ids.append(uploaded["asset"]["id"])
        results["steps"].append({"operation": "asset_upload", "role": role, **uploaded})
        emit(
            {
                "operation": "asset_upload",
                "role": role,
                "versioned": bool(uploaded.get("revision_id")),
            }
        )
    action(
        "curve_preview",
        {"project_id": pid, "asset_id": ids[0], "curve": "RHOB", "limit": 8},
    )
    action(
        "curve_mapping",
        {
            "project_id": pid,
            "asset_id": ids[0],
            "source_curve": "RHOB",
            "target_curve": "DEN",
        },
    )
    semantics = {
        "horizontal_crs_id": "EPSG:32650",
        "horizontal_unit": "m",
        "horizontal_axis_order": "XY",
        "coordinate_reference_verified": True,
        "well_source_crs_id": "EPSG:32650",
        "seismic_source_crs_id": "EPSG:32650",
        "vertical_crs_id": "MSL",
        "seismic_srd_elevation_m": 0,
        "seismic_time_domain": "TWT",
        "seismic_correction_state": "corrected_to_srd",
        "seismic_replacement_velocity_mps": 2500,
        "well_coordinate_source_unit": "m",
        "well_vertical_datum_source_unit": "m",
        "segy_geometry_profile": "standard_3d",
        "segy_inline_byte": 189,
        "segy_crossline_byte": 193,
        "segy_x_byte": 181,
        "segy_y_byte": 185,
        "segy_coordinate_scalar_byte": 71,
        "time_depth_default_depth_domain": "md",
        "time_depth_default_depth_unit": "m",
        "time_depth_default_time_unit": "ms",
        "time_depth_default_depth_datum": "KB",
        "time_depth_default_time_reference": "SRD",
        "time_depth_default_time_domain": "TWT",
        "time_depth_default_correction_state": "corrected_to_srd",
    }
    prepared = action(
        "data_prepare", {"project_id": pid, "asset_ids": ids, "semantics": semantics}
    )
    if prepared.get("task_id"):
        task_id = prepared["task_id"]
        action("visualization", {"project_id": pid, "task_id": task_id})
        registered = action("registration", {"project_id": pid, "task_id": task_id})
        if registered["status"] == "completed":
            action(
                "sample_build",
                {
                    "project_id": pid,
                    "task_id": task_id,
                    "registration_task_id": registered["task_id"],
                },
            )
        action(
            "prediction",
            {
                "project_id": pid,
                "task_id": task_id,
                "interpretation_task": "well_property",
                "model_id": "wellfuse_den_northwest_fast",
            },
        )
        action("dvc_archive", {"project_id": pid, "task_id": task_id})
    versions = action("dvc_list", {"project_id": pid})
    items = versions.get("result", {}).get("versions", [])
    if len(items) >= 2:
        action(
            "dvc_compare",
            {
                "project_id": pid,
                "from_revision_id": items[-1]["revision_id"],
                "to_revision_id": items[0]["revision_id"],
            },
        )
        restored = action(
            "dvc_restore", {"project_id": pid, "revision_id": items[0]["revision_id"]}
        )
        recovered = restored.get("result", {}).get("restored_assets", [])
        if recovered:
            action(
                "data_prepare",
                {
                    "project_id": pid,
                    "asset_ids": [a["id"] for a in recovered],
                    "semantics": semantics,
                },
            )
    report = public_value(results)
    output = args.output or root / "native-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit(
        {
            "report": str(output),
            "isolated_root": str(root),
            "completed_steps": sum(
                s.get("status") == "completed" for s in results["steps"]
            ),
            "blocked_steps": sum(
                s.get("status") == "blocked" for s in results["steps"]
            ),
        }
    )


if __name__ == "__main__":
    main()
