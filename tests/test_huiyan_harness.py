"""Boundary tests plus actual LAS parsing/mapping and native physics algorithm.

The queue double isolates permission/idempotency tests. It is not evidence that
neural inference ran; the absent-model test asserts a blocked receipt instead.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from well_seismic.alignment.well_tie import (
    acoustic_reflectivity,
    build_sonic_time_domain_alignment,
    ricker_wavelet,
)
from well_seismic.api_models import TaskCreated
from well_seismic.config import load_config
from well_seismic.huiyan_harness import HuiyanHarness, build_router

ROOT = Path(__file__).resolve().parents[1]
OWNER, OTHER = "a" * 64, "b" * 64


class StoreDouble:
    def archive(self, subject, project, files, metadata, parent_revision_id=None):
        assert all(Path(path).is_file() for path in files.values())
        return {
            "revision_id": "dv_" + uuid.uuid4().hex,
            "files": list(files),
            "metadata": metadata,
        }


@pytest.fixture
def harness(tmp_path):
    tasks = {}

    def submit(request):
        task_id = uuid.uuid4().hex
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "request": request.model_dump(),
            "result": {},
        }
        return TaskCreated(task_id=task_id, status="queued", message="queued")

    def get_task(task_id):
        return tasks[task_id]

    h = HuiyanHarness(
        tmp_path / "harness",
        config=load_config(ROOT / "configs", {}),
        native={"data_prepare": submit, "task": get_task, "capabilities": dict},
        dvc_store=StoreDouble(),
    )
    h.test_tasks = tasks
    return h


def project_asset(harness):
    created = harness.execute(
        OWNER, "project_create", {"name": "synthetic visitor project"}
    )
    pid = created["project_id"]
    # New independent fixture bytes. Do not rewrite the historical sample whose
    # G/C3 unit is rejected by the native knowledge base.
    content = (
        (ROOT / "tests/fixtures/gaia_synthetic.las")
        .read_bytes()
        .replace(b"G/C3", b"G/CM3")
    )
    upload = harness.upload(OWNER, pid, "log", "synthetic.las", content)
    return pid, upload["asset"]["id"]


def test_native_las_mapping_is_derived_and_values_unchanged(harness):
    pid, aid = project_asset(harness)
    original = harness.asset(OWNER, pid, aid)
    before = Path(original["path"]).read_bytes()
    response = harness.execute(
        OWNER,
        "curve_mapping",
        {
            "project_id": pid,
            "asset_id": aid,
            "source_curve": "RHOB",
            "target_curve": "DEN",
        },
    )
    assert response["status"] == "completed"
    assert response["result"]["mapping"]["values_unchanged"] is True
    mapped = harness.asset(OWNER, pid, response["result"]["asset"]["id"])
    assert mapped["parent_id"] == aid
    assert b"DEN.G/CM3" in Path(mapped["path"]).read_bytes()
    assert Path(original["path"]).read_bytes() == before
    preview = harness.execute(
        OWNER,
        "curve_preview",
        {"project_id": pid, "asset_id": mapped["id"], "curve": "DEN"},
    )
    assert preview["result"]["curves"]["RHOB"]["values"] == [2.3, 2.32, 2.35, 2.33]
    assert str(harness.root) not in json.dumps(preview)


def test_cross_owner_asset_task_execution_are_rejected(harness):
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    assert run["status"] == "running"
    for operation in (
        lambda: harness.asset(OTHER, pid, aid),
        lambda: harness.own_task(OTHER, pid, run["task_id"]),
        lambda: harness.get_run(OTHER, run["execution_id"]),
    ):
        with pytest.raises(HTTPException) as error:
            operation()
        assert error.value.status_code == 404


def test_idempotency_digest_and_native_failure(harness):
    pid, aid = project_asset(harness)
    oid = str(uuid.uuid4())
    args = {"project_id": pid, "asset_ids": [aid]}
    first = harness.execute(OWNER, "data_prepare", args, oid)
    assert (
        harness.execute(OWNER, "data_prepare", args, oid)["execution_id"]
        == first["execution_id"]
    )
    assert len(harness.test_tasks) == 1
    expected = hashlib.sha256(
        json.dumps(
            {"operation": "data_prepare", "arguments": args},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert first["request_sha256"] == expected
    with pytest.raises(HTTPException) as error:
        harness.execute(OWNER, "data_prepare", {"project_id": pid}, oid)
    assert error.value.status_code == 409
    harness.test_tasks[first["task_id"]]["status"] = "failed"
    assert harness.get_run(OWNER, first["execution_id"])["status"] == "failed"


def test_completed_native_result_versioned_and_restart_replay(harness):
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    harness.test_tasks[run["task_id"]]["status"] = "completed"
    done = harness.get_run(OWNER, run["execution_id"])
    assert done["status"] == "completed" and done["revision_id"]
    restarted = HuiyanHarness(
        harness.root,
        config=harness.config,
        native=harness.native,
        dvc_store=StoreDouble(),
    )
    assert restarted.get_run(OWNER, run["execution_id"]) == done


def test_absent_model_is_blocked_never_placeholder_prediction(harness):
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    result = harness.execute(
        OWNER,
        "prediction",
        {
            "project_id": pid,
            "task_id": run["task_id"],
            "interpretation_task": "well_property",
            "model_id": "wellfuse_den_northwest_fast",
        },
    )
    assert result["status"] == "blocked"
    assert result["error"]["code"] == "model_assets_missing"
    assert len(harness.test_tasks) == 1


def test_paths_extra_identity_and_corrupt_assets_rejected(harness):
    pid, aid = project_asset(harness)
    with pytest.raises(HTTPException):
        harness.execute(OWNER, "data_prepare", {"project_id": pid, "path": "C:/secret"})
    with pytest.raises(HTTPException):
        harness.execute(
            OWNER,
            "data_prepare",
            {"project_id": pid, "semantics": {"log_paths": ["C:/secret"]}},
        )
    asset = harness.asset(OWNER, pid, aid)
    Path(asset["path"]).write_bytes(b"tampered")
    response = harness.execute(
        OWNER, "curve_preview", {"project_id": pid, "asset_id": aid}
    )
    assert response["status"] == "blocked"
    assert response["error"]["code"] == "asset_integrity_failed"


def test_unknown_density_unit_cannot_be_fixed_by_renaming(harness):
    pid = harness.execute(
        OWNER, "project_create", {"name": "bad unit negative control"}
    )["project_id"]
    uploaded = harness.upload(
        OWNER,
        pid,
        "log",
        "unrecognized-density-unit.las",
        (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes(),
    )
    response = harness.execute(
        OWNER,
        "curve_mapping",
        {
            "project_id": pid,
            "asset_id": uploaded["asset"]["id"],
            "source_curve": "RHOB",
            "target_curve": "DEN",
        },
    )
    assert response["status"] == "blocked"
    assert response["error"]["code"] == "curve_values_unavailable"


def test_real_native_physics_well_tie():
    # Explicitly synthetic signal; this executes the existing physics path,
    # without installing weights or claiming P13/LayerPulse inference.
    depth = np.arange(601, dtype=float)
    velocity = np.full(depth.size, 2500.0)
    density = 2.3 + 0.1 * np.sin(depth / 24.0) + 0.18 * (depth > 250)
    time = np.arange(650, dtype=float)
    reflection = acoustic_reflectivity(velocity, density)
    reflection_time = (depth[:-1] + depth[1:]) / 2 * 0.8
    trace = np.convolve(
        np.interp(time, reflection_time, reflection, left=0, right=0),
        ricker_wavelet(30, 1),
        "same",
    )
    result = build_sonic_time_domain_alignment(
        depth,
        {"VP": velocity, "RHOB": density},
        trace,
        time,
        {"datum_is_explicit": True},
    )
    assert result is not None
    assert result.method and result.confidence > 0
    assert not result.training_eligible


def test_http_auth_and_owner_scope(harness, monkeypatch):
    monkeypatch.setattr(
        "well_seismic.gaia_tasks.service_token", lambda: "test-service-token-" * 3
    )
    app = FastAPI()
    app.include_router(build_router(harness))
    client = TestClient(app)
    assert client.get("/api/v1/gaia/harness/actions").status_code == 401
    headers = {
        "Authorization": "Bearer " + "test-service-token-" * 3,
        "X-Gaia-Subject": OWNER,
    }
    response = client.post(
        "/api/v1/gaia/harness/actions",
        headers=headers,
        json={"operation": "project_create", "arguments": {"name": "visitor"}},
    )
    assert response.status_code == 200
    execution_id = response.json()["execution_id"]
    headers["X-Gaia-Subject"] = OTHER
    assert (
        client.get(
            "/api/v1/gaia/harness/runs/" + execution_id, headers=headers
        ).status_code
        == 404
    )


def test_harness_viewer_query_and_navigation_are_owned(harness, monkeypatch):
    monkeypatch.setattr(
        "well_seismic.gaia_tasks.service_token", lambda: "test-service-token-" * 3
    )
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    task_id = run["task_id"]
    observed = []

    def render(**kwargs):
        observed.append(kwargs)
        return HTMLResponse(
            '<a href="/统一数据可视化?task_id=' + task_id + '&amp;asset=3">next</a>'
        )

    harness.native["visualization"] = render
    app = FastAPI()
    app.include_router(build_router(harness))
    client = TestClient(app)
    headers = {
        "Authorization": "Bearer " + "test-service-token-" * 3,
        "X-Gaia-Subject": OWNER,
    }
    path = f"/api/v1/gaia/harness/tasks/{task_id}/visualization"
    response = client.get(path + "?asset=2&block=z00_i01_x02", headers=headers)
    assert response.status_code == 200
    assert 'href="?asset=3"' in response.text
    assert observed == [
        {"task_id": task_id, "embed": True, "asset": 2, "block": "z00_i01_x02"}
    ]
    assert client.get(path + "?task_id=" + task_id, headers=headers).status_code == 422
    assert (
        client.get(path, headers={**headers, "X-Gaia-Subject": OTHER}).status_code
        == 404
    )
    assert len(observed) == 1


def test_p21_visualization_receipt_requires_real_interactive_content(harness):
    pid = harness.execute(OWNER, "project_create", {"name": "P21 dev fold"})[
        "project_id"
    ]
    task_id = uuid.uuid4().hex
    harness.test_tasks[task_id] = {
        "task_id": task_id,
        "status": "completed",
        "result": {"prediction": {"model_id": "sixtrack_reconstruction_p21"}},
    }
    harness.bind_task(OWNER, pid, task_id)
    harness.native["visualization"] = lambda **kwargs: HTMLResponse(
        '<main id="workbench"><canvas id="point-canvas"></canvas>'
        '<select id="point-view"></select></main>'
    )
    args = {"project_id": pid, "task_id": task_id}
    result = harness.execute(OWNER, "visualization", args)
    assert result["status"] == "completed"
    assert result["result"]["viewer_available"] is True
    assert result["result"]["viewer_kind"] == "p21_query_points"
    assert "盖亚" in result["result"]["viewer_entry"]

    harness.native["visualization"] = lambda **kwargs: HTMLResponse(
        '<main id="workbench">No plotted points</main>'
    )
    rejected = harness.execute(OWNER, "visualization", args)
    assert rejected["status"] == "blocked"
    assert rejected["error"]["code"] == "visualization_unavailable"
    with pytest.raises(HTTPException) as other:
        harness.own_task(OTHER, pid, task_id)
    assert other.value.status_code == 404


def test_archive_deduplicates_same_declared_file_without_losing_receipt(harness):
    artifact = harness.root / "outputs" / "prediction.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"sealed native output")
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "completed",
        "result": {"prediction": {
            "outputs": {"prediction": str(artifact)},
            "output_files": {"prediction": str(artifact)},
        }},
    }
    files, names = harness._task_archive_files(task_id, task)
    assert "results/task.json" in files
    assert len(files) == 2
    assert list(names.values()) == ["prediction.npz"]


def test_blocked_upload_dvc_does_not_block_health_or_catalog(harness, monkeypatch):
    monkeypatch.setattr(
        "well_seismic.gaia_tasks.service_token", lambda: "test-service-token-" * 3
    )
    pid = harness.execute(OWNER, "project_create", {"name": "concurrent upload"})[
        "project_id"
    ]
    entered, release = threading.Event(), threading.Event()
    original = harness.dvc.archive

    def blocked_archive(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test must release the simulated DVC operation"
        return original(*args, **kwargs)

    monkeypatch.setattr(harness.dvc, "archive", blocked_archive)
    app = FastAPI()
    app.include_router(build_router(harness))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    headers = {
        "Authorization": "Bearer " + "test-service-token-" * 3,
        "X-Gaia-Subject": OWNER,
    }
    # TestClient's context uses one ASGI event loop for concurrent requests.
    # The barrier models a slow CLI without sleeping or invoking actual DVC.
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as callers:
        upload = callers.submit(
            client.post,
            f"/api/v1/gaia/harness/projects/{pid}/assets/well",
            params={"filename": "well.csv"},
            content=b"WELL,X,Y\nDEMO,0,0\n",
            headers=headers,
        )
        try:
            assert entered.wait(2), "upload did not reach its DVC archive"
            ping = callers.submit(client.get, "/health")
            catalog = callers.submit(
                client.get, "/api/v1/gaia/harness/actions", headers=headers
            )
            assert ping.result(timeout=2).json() == {"status": "ok"}
            assert catalog.result(timeout=2).status_code == 200
            assert not upload.done(), (
                "health/catalog must finish while upload is still blocked"
            )
        finally:
            release.set()
        assert upload.result(timeout=2).status_code == 201


def test_schema_constraints_and_operation_type(harness):
    pid, aid = project_asset(harness)
    for operation, args in (
        ([], {}),
        ("curve_preview", {"project_id": pid, "asset_id": aid, "limit": 501}),
        ("data_prepare", {"project_id": pid, "asset_ids": [aid, aid]}),
        (
            "data_prepare",
            {"project_id": pid, "semantics": {"horizontal_unit": "furlong"}},
        ),
    ):
        with pytest.raises(HTTPException) as error:
            harness.execute(OWNER, operation, args)
        assert error.value.status_code == 422


def test_trajectory_and_well_head_both_reach_native_well_paths(harness):
    pid, aid = project_asset(harness)
    head = harness.upload(OWNER, pid, "well", "head.csv", b"WELL,X,Y\nTEST,1,2\n")[
        "asset"
    ]["id"]
    track = harness.upload(
        OWNER, pid, "trajectory", "track.csv", b"WELL,MD(m),TVD(m)\nTEST,0,0\n"
    )["asset"]["id"]
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid, head, track]}
    )
    request = harness.test_tasks[run["task_id"]]["request"]
    assert len(request["well_paths"]) == 2
    assert request["survey_paths"] == []


def test_native_completed_but_business_blocked_is_not_success(harness):
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    task = harness.test_tasks[run["task_id"]]
    task.update(
        status="completed",
        result={
            "registration": {"business_status": "blocked", "registered_well_count": 0}
        },
    )
    receipt = harness.get_run(OWNER, run["execution_id"])
    assert receipt["status"] == "blocked"
    assert receipt["error"]["code"] == "native_business_blocked"


def test_upload_reply_loss_retry_is_content_idempotent(harness):
    pid = harness.execute(OWNER, "project_create", {"name": "upload retry"})[
        "project_id"
    ]
    content = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    first = harness.upload(OWNER, pid, "log", "source.las", content)
    restarted = HuiyanHarness(
        harness.root,
        config=harness.config,
        native=harness.native,
        dvc_store=StoreDouble(),
    )
    again = restarted.upload(OWNER, pid, "log", "source.las", content)
    assert first == again
    assert len(restarted.assets(OWNER, pid)) == 1
    changed = restarted.upload(
        OWNER, pid, "log", "source.las", content.replace(b"2.30", b"2.31")
    )
    assert changed["asset"]["id"] != first["asset"]["id"]
    with pytest.raises(HTTPException) as error:
        restarted.upload(OTHER, pid, "log", "source.las", content)
    assert error.value.status_code == 404


def test_restore_preserves_all_sealed_asset_roles(harness, tmp_path):
    pid, _ = project_asset(harness)
    records = []
    files = {}
    for role, suffix in (
        ("log", ".las"),
        ("seismic", ".sgy"),
        ("well", ".csv"),
        ("trajectory", ".csv"),
        ("time_depth", ".csv"),
    ):
        record = {
            "id": "ha_" + uuid.uuid4().hex,
            "project_id": pid,
            "role": role,
            "name": "original" + suffix,
            "sha256": "0" * 64,
            "parent_id": None,
        }
        path = tmp_path / (record["id"] + suffix)
        path.write_bytes(b"verified DVC resolver double")
        files[harness.logical_asset_name(record)] = path
        records.append(record)

    class RestoredStore:
        def restore_owned(self, subject, project, revision):
            assert (subject, project) == (OWNER, pid)
            return {
                "restore_id": "dr_" + "1" * 32,
                "revision_id": revision,
                "project_id": project,
            }

        def get(self, subject, project, revision):
            return {"metadata": {"asset_records": records}}

        def resolve_restored_files(self, subject, project, restore_id):
            return files

    harness.dvc = RestoredStore()
    result = harness.execute(
        OWNER, "dvc_restore", {"project_id": pid, "revision_id": "dv_" + "2" * 32}
    )
    assert result["status"] == "completed"
    restored = result["result"]["restored_assets"]
    assert {a["role"] for a in restored} == {r["role"] for r in records}
    assert all(
        harness.asset(OWNER, pid, a["id"])["project_id"] == pid for a in restored
    )
