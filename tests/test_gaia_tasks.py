import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from well_seismic.gaia_tasks import build_router, result_report


@pytest.fixture
def client(tmp_path, monkeypatch):
    from well_seismic import api
    from well_seismic.state_store import SQLiteStateStore

    token = tmp_path / "token"
    token.write_text("unit-test-only-service-token-32-characters")
    monkeypatch.setenv("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", str(token))
    monkeypatch.setattr(
        api, "_state_store", SQLiteStateStore(tmp_path / "tasks.sqlite3")
    )
    monkeypatch.setattr(api, "_tasks", {})
    monkeypatch.setattr(api, "SOURCE_SNAPSHOT_ROOT", tmp_path / "snapshots")
    app = FastAPI()
    app.include_router(
        build_router(
            tmp_path / "inputs", api.create_data_preparation_task, api.get_task
        )
    )
    with TestClient(app) as test_client:
        yield test_client


HEADERS = {
    "Authorization": "Bearer unit-test-only-service-token-32-characters",
    "X-Gaia-Subject": "a" * 64,
}
RUN = "run_1234567890abcdef"
DATA = Path(__file__).with_name("fixtures").joinpath("gaia_synthetic.las").read_bytes()


def test_real_preparation_roundtrip_and_idempotency(client):
    url = f"/api/v1/gaia/runs/{RUN}"
    first = client.post(url, content=DATA, headers=HEADERS)
    assert first.status_code == 202, first.text
    for _ in range(100):
        result = client.get(url, headers=HEADERS).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert result["status"] == "completed", result
    assert result["report"]["summary"]["wells"] == 1
    assert result["report"]["wells"][0]["logs"][0]["samples"] == 4
    assert result["input_sha256"] == hashlib.sha256(DATA).hexdigest()
    again = client.post(url, content=DATA, headers=HEADERS).json()
    assert again["task_id"] == first.json()["task_id"]
    assert client.post(url, content=DATA + b"\n", headers=HEADERS).status_code == 409
    other = {**HEADERS, "X-Gaia-Subject": "b" * 64}
    assert client.get(url, headers=other).status_code == 404
    assert client.post(url, content=DATA, headers=other).status_code == 404
    assert "storage_path" not in str(result) and "input.las" not in str(result)


def test_credential_and_input_gates(client, monkeypatch):
    url = f"/api/v1/gaia/runs/{RUN}"
    assert client.post(url, content=DATA).status_code == 401
    assert (
        client.post(
            url, content=DATA, headers={**HEADERS, "Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert client.post(url, content=b"", headers=HEADERS).status_code == 422
    assert client.post(url, content=b"not a LAS", headers=HEADERS).status_code == 422
    assert (
        client.post("/api/v1/gaia/runs/bad", content=DATA, headers=HEADERS).status_code
        == 422
    )
    monkeypatch.setenv("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", "missing-token")
    assert client.get("/api/v1/gaia/capabilities", headers=HEADERS).status_code == 503


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr("well_seismic.gaia_tasks.MAX_BYTES", 16)
    assert (
        client.post(
            f"/api/v1/gaia/runs/{RUN}", content=DATA, headers=HEADERS
        ).status_code
        == 413
    )


def test_changed_sealed_input_is_not_reported_as_success():
    with pytest.raises(ValueError, match="Sealed input"):
        result_report(
            {"result": {"assets": [{"sha256": "different"}]}},
            {"input_sha256": "original"},
        )


@pytest.mark.parametrize("kind", ["legacy", "raw"])
def test_other_upload_routes_keep_event_loop_responsive(tmp_path, monkeypatch, kind):
    from well_seismic import api
    from well_seismic.api_models import TaskCreated

    entered, release = threading.Event(), threading.Event()
    app = FastAPI()

    def blocked_work(*args):
        entered.set()
        assert release.wait(5), "test must release the upload worker"
        if kind == "legacy":
            return TaskCreated(task_id="c" * 32, status="queued", message="queued")
        return {"name": "input.las", "size": len(DATA)}

    if kind == "legacy":
        monkeypatch.setattr(
            "well_seismic.gaia_tasks.service_token",
            lambda: "unit-test-only-service-token-32-characters",
        )
        app.include_router(
            build_router(
                tmp_path / "inputs",
                blocked_work,
                lambda task_id: {"task_id": task_id, "status": "queued"},
            )
        )
        path, expected = f"/api/v1/gaia/runs/{RUN}", 202
    else:
        monkeypatch.setattr(api, "_store_raw_well_file", blocked_work)
        app.add_api_route("/raw", api.upload_raw_well_file, methods=["POST"])
        path, expected = "/raw?filename=input.las", 200

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    with TestClient(app) as caller, ThreadPoolExecutor(max_workers=2) as requests:
        upload = requests.submit(caller.post, path, content=DATA, headers=HEADERS)
        try:
            assert entered.wait(2)
            ping = requests.submit(caller.get, "/health")
            assert ping.result(timeout=2).json() == {"status": "ok"}
            assert not upload.done()
        finally:
            release.set()
        assert upload.result(timeout=2).status_code == expected
