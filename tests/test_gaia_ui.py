import json
import sqlite3
from urllib.parse import urljoin
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.testclient import TestClient

from well_seismic.gaia_ui import build_router, owned_visualization_html, public_value

TOKEN = "unit-only-ui-service-token-32-characters"
TASK = "a" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}", "X-Gaia-Subject": "a" * 64}
OTHER = {**HEADERS, "X-Gaia-Subject": "b" * 64}
PREFIX = "/api/v1/gaia/ui/"


def test_public_value_does_not_treat_https_as_a_windows_drive():
    assert (
        public_value("https://example.invalid/other") == "https://example.invalid/other"
    )
    assert public_value("D:/private/input.las") == "input.las"
    assert public_value("C:\\private\\model.bin") == "model.bin"


def test_only_successful_native_viewer_emits_bound_ready_signal():
    native = '<html><body><div id="workbench">real viewer</div></body></html>'
    result = owned_visualization_html(native, TASK)
    assert '"type": "huiyan:visualization-ready"' in result
    assert '"taskId": "' + TASK + '"' in result
    assert "window.location.origin" in result
    assert "visualization-ready" not in owned_visualization_html(
        native, TASK, status_code=500
    )
    assert "visualization-ready" not in owned_visualization_html(
        "<html><body>不可用</body></html>", TASK
    )
    assert "visualization-ready" not in owned_visualization_html(native, "foreign/path")


@pytest.fixture
def client(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text(TOKEN)
    monkeypatch.setenv("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", str(token))
    root = tmp_path / "data"
    root.mkdir()
    with sqlite3.connect(root / "receipts.sqlite3") as db:
        db.execute("CREATE TABLE receipts (run_id TEXT, subject TEXT, task_id TEXT)")
        db.execute(
            "INSERT INTO receipts VALUES (?, ?, ?)",
            ("run_1234567890abcdef", "a" * 64, TASK),
        )
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="/assets/app.js"></script>')
    (dist / "assets/app.js").write_text("window.workbench = true;")
    (dist / "secret.txt").write_text("not-public")
    manifest_file = tmp_path / "snapshot.json"
    manifest_file.write_text('{"source":"D:/private/input.las","samples":4}')

    def assistant(payload):
        return {
            "answer": "test reply",
            "conversation_id": str(payload.conversation_id or uuid4()),
        }

    def visualization(task_id, **kwargs):
        return HTMLResponse(
            '<a href="/统一数据可视化?task_id='
            + task_id
            + '&amp;asset=2&amp;embed=0">next</a>'
            '<a href="/统一数据可视化?task_id=' + "b" * 32 + '&asset=1">foreign</a>'
            '<a href="https://example.invalid/other">unchanged</a>'
            '<script>const target="/统一数据可视化?task_id='
            + task_id
            + '&block=z00_i01_x02";'
            'if(target.startsWith("/统一数据可视化?"))window.location.assign(target);</script>'
            + json.dumps(kwargs, sort_keys=True)
        )

    app = FastAPI()
    app.include_router(
        build_router(
            root,
            dist,
            get_task=lambda task: {
                "task_id": task,
                "request": {"log_paths": ["D:/private/input.las"]},
                "result": {"samples": 4},
            },
            capabilities=lambda: {
                "assistant": {"available": True},
                "path": "/home/private/model.bin",
            },
            health=lambda: {
                "status": "ok",
                "version": "test",
                "wellfuse_python_path": "D:/private/python.exe",
            },
            assistant=assistant,
            manifest=lambda task: FileResponse(manifest_file),
            visualization=visualization,
        )
    )
    with TestClient(app) as test:
        yield test


def test_page_assets_and_health_require_service_auth(client):
    for path in ("", "assets/app.js", "api/v1/health", "api/v1/capabilities"):
        assert client.get(PREFIX + path).status_code == 401
        assert client.get(PREFIX + path, headers=HEADERS).status_code == 200
    assert 'src="./assets/app.js"' in client.get(PREFIX, headers=HEADERS).text
    assert "private" not in client.get(PREFIX + "api/v1/health", headers=HEADERS).text
    assert (
        "private"
        not in client.get(PREFIX + "api/v1/capabilities", headers=HEADERS).text
    )


def test_relative_assets_stay_under_the_authenticated_gateway(client):
    html = client.get(PREFIX, headers=HEADERS).text
    asset = html.split('src="')[1].split('"')[0]
    for prefix in ("/api/plug/huiyan/ui/", "/api/workflow-runs/huiyan/ui/"):
        assert urljoin("https://gaia.example" + prefix + "?job_ref=example", asset) == (
            "https://gaia.example" + prefix + "assets/app.js"
        )


def test_native_result_is_owned_and_paths_are_withheld(client):
    path = PREFIX + f"api/v1/tasks/{TASK}"
    assert client.get(path, headers=OTHER).status_code == 404
    response = client.get(path, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["request"]["log_paths"] == ["input.las"]
    assert response.json()["result"]["samples"] == 4
    assert (
        client.get(PREFIX + "api/v1/tasks/" + "c" * 32, headers=HEADERS).status_code
        == 404
    )


def test_embedded_chat_rejects_other_users_task_and_conversation(client):
    path = PREFIX + "api/v1/assistant/chat"
    assert (
        client.post(
            path, headers=OTHER, json={"message": "hello", "task_id": TASK}
        ).status_code
        == 404
    )
    first = client.post(
        path, headers=HEADERS, json={"message": "hello", "task_id": TASK}
    )
    assert first.status_code == 200
    session = first.json()["conversation_id"]
    assert (
        client.post(
            path, headers=OTHER, json={"message": "follow", "conversation_id": session}
        ).status_code
        == 404
    )
    reply = client.post(
        path, headers=HEADERS, json={"message": "follow", "conversation_id": session}
    )
    assert reply.status_code == 200 and reply.json()["conversation_id"] == session


def test_snapshot_download_is_scoped_and_redacted(client):
    path = PREFIX + f"api/v1/tasks/{TASK}/artifacts/snapshot_manifest"
    assert client.get(path, headers=OTHER).status_code == 404
    reply = client.get(path, headers=HEADERS)
    assert reply.status_code == 200
    assert reply.json() == {"source": "input.las", "samples": 4}
    assert "attachment" in reply.headers["content-disposition"]


def test_workbench_does_not_publish_desktop_control_or_arbitrary_files(client):
    for path in (
        "assets/%2e%2e/secret.txt",
        "assets/../secret.txt",
        "api/v1/projects",
        "api/v1/system/cache",
        "api/v1/demo-paths",
    ):
        assert client.get(PREFIX + path, headers=HEADERS).status_code == 404
    assert (
        client.post(
            PREFIX + "api/v1/data-preparation/tasks",
            headers=HEADERS,
            json={"log_paths": ["D:/private/other.las"]},
        ).status_code
        == 404
    )


def test_owned_visualization_navigation_preserves_proxy_and_query(client):
    path = PREFIX + f"api/v1/tasks/{TASK}/visualization"
    response = client.get(
        path + "?asset=2&block=z00_i01_x02&layerpulse_output=fault_logits",
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert 'href="?asset=2"' in response.text
    assert 'href="#unavailable"' in response.text
    assert "https://example.invalid/other" in response.text
    assert 'target="?block=z00_i01_x02"' in response.text
    assert 'target.startsWith("?")' in response.text
    assert '"asset": 2' in response.text
    assert "/统一数据可视化" not in response.text
    assert response.headers["cache-control"] == "private, no-store"


def test_visualization_rejects_other_owner_and_query_override(client):
    path = PREFIX + f"api/v1/tasks/{TASK}/visualization"
    assert client.get(path, headers=OTHER).status_code == 404
    assert client.get(path).status_code == 401
    for query in (
        "task_id=" + "b" * 32,
        "asset=-1",
        "asset=1&asset=2",
        "block=../secret",
        "layerpulse_output=",
        "endpoint=https://evil.invalid",
    ):
        assert client.get(path + "?" + query, headers=HEADERS).status_code == 422
