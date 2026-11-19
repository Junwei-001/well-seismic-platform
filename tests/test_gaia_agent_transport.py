"""ASGI transport, authentication and durable agent rejections; model/tool mocks."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from well_seismic.gaia_minimax import GaiaMiniMaxSettings
from well_seismic.gaia_ui import build_router


SETTINGS = GaiaMiniMaxSettings("unit-provider", "MiniMax-M2.7", "unit-only-key")
TOKEN = "unit-only-concurrent-transport-token-32-characters"
OWNER, PEER = "a" * 64, "b" * 64
HEADERS = {"Authorization": f"Bearer {TOKEN}", "X-Gaia-Subject": OWNER}
PATH = "/api/v1/gaia/ui/api/v1/assistant/chat"


def answer(content):
    return {"model": SETTINGS.model, "content": content, "stop_reason": "end_turn"}


def body(request_id, conversation_id=None):
    return {
        "message": "请列出我的项目", "request_id": str(request_id),
        "conversation_id": conversation_id,
    }


@pytest.fixture
def transport(tmp_path, monkeypatch):
    from well_seismic import api

    token_file = tmp_path / "service-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setenv("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", str(token_file))
    root = tmp_path / "ui"
    h = Mock()
    h.root = tmp_path / "harness"
    h.assistant_context.return_value = {"binding": "unselected"}
    h.catalog.return_value = {
        "actions": [{
            "name": "project_list", "description": "list owned projects",
            "parameters": {"type": "object", "properties": {}},
        }]
    }
    h.execute.return_value = {
        "execution_id": "hx_" + "3" * 32, "operation": "project_list",
        "status": "completed", "result": {"projects": []},
    }
    monkeypatch.setattr(api, "_huiyan_harness", h)
    monkeypatch.setattr(api, "_assistant_context", lambda _task: {})
    monkeypatch.setattr(GaiaMiniMaxSettings, "from_environment", lambda: SETTINGS)
    app = FastAPI()
    app.include_router(build_router(
        root, tmp_path / "dist", get_task=Mock(), capabilities=Mock(), health=Mock(),
        assistant=Mock(side_effect=AssertionError("owned route required")),
        assistant_owned=api.assistant_chat_authenticated,
    ))
    with TestClient(app) as client:
        yield client, h, root


def prime(client):
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        return_value=answer([{"type": "text", "text": "ready"}]),
    ):
        reply = client.post(PATH, headers=HEADERS, json=body(uuid4()))
    assert reply.status_code == 200
    return reply.json()["conversation_id"]


def test_busy_terminal_response_survives_real_authenticated_chat_transport(transport):
    client, h, _root = transport
    conversation = prime(client)
    h.reset_mock()
    first_id, second_id = uuid4(), uuid4()
    entered, release = Event(), Event()

    def provider(_settings, _payload):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
            return answer([{
                "type": "tool_use", "name": "project_list", "id": "owned-list",
                "input": {},
            }])
        return answer([{"type": "text", "text": "完成"}])

    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=provider) as request:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first_future = pool.submit(
                client.post, PATH, headers=HEADERS, json=body(first_id, conversation),
            )
            try:
                assert entered.wait(5)
                active = client.post(PATH, headers=HEADERS, json=body(first_id, conversation))
                assert active.status_code == 200
                assert active.json()["request_state"] == "in_progress"
                rejected = client.post(PATH, headers=HEADERS, json=body(second_id, conversation))
                assert rejected.status_code == 200
                result = rejected.json()
                assert result["request_state"] == "finished"
                assert result["error"]["code"] == "conversation_busy"
                assert result["executions"] == []
                assert result["conversation_id"] == conversation
                assert result["request_id"] == str(second_id)
                assert result["provider"] == "unavailable"
                assert TOKEN not in rejected.text and SETTINGS.api_key not in rejected.text
                h.execute.assert_not_called()
                assert request.call_count == 1
                repeated = client.post(PATH, headers=HEADERS, json=body(second_id, conversation))
                assert repeated.status_code == 200 and repeated.json() == result
            finally:
                release.set()
                first = first_future.result(timeout=10)
        assert first.status_code == 200
        assert first.json()["request_state"] == "finished"
        assert len(first.json()["executions"]) == 1
        repeated = client.post(PATH, headers=HEADERS, json=body(second_id, conversation))
        assert repeated.status_code == 200 and repeated.json() == result
        assert request.call_count == 2 and h.execute.call_count == 1
        assert h.catalog.call_count == 1
    with sqlite3.connect(h.root / "assistant_requests.sqlite3") as db:
        assert db.execute(
            "SELECT state FROM requests WHERE subject=? AND request_id=?",
            (OWNER, str(second_id)),
        ).fetchone() == ("finished",)


def test_foreign_conversation_rejected_by_route_before_claim_or_provider(transport):
    client, h, _root = transport
    conversation = prime(client)
    h.reset_mock()
    request_id = uuid4()
    peer_headers = {**HEADERS, "X-Gaia-Subject": PEER}
    with patch("well_seismic.gaia_agent.request_minimax_message") as provider:
        first = client.post(PATH, headers=peer_headers, json=body(request_id, conversation))
        second = client.post(PATH, headers=peer_headers, json=body(request_id, conversation))
        assert first.status_code == second.status_code == 404
        assert first.json() == second.json()
        assert conversation not in first.text and OWNER not in first.text
        assert "executions" not in first.json()
        provider.assert_not_called()
    h.assistant_context.assert_not_called()
    h.catalog.assert_not_called()
    h.execute.assert_not_called()
    with sqlite3.connect(h.root / "assistant_requests.sqlite3") as db:
        assert db.execute(
            "SELECT COUNT(*) FROM requests WHERE subject=? AND request_id=?",
            (PEER, str(request_id)),
        ).fetchone() == (0,)


def test_agent_owner_guard_serializes_terminal_rejection_for_inconsistent_ui_registry(transport):
    client, h, root = transport
    conversation = prime(client)
    h.reset_mock()
    # Simulate inconsistent persisted UI metadata in this isolated fixture.
    # The independent agent ownership guard must still deny, never execute.
    with sqlite3.connect(root / "ui_sessions.sqlite3") as db:
        db.execute("UPDATE sessions SET subject=? WHERE id=?", (PEER, conversation))
    peer_headers = {**HEADERS, "X-Gaia-Subject": PEER}
    request_id = uuid4()
    with patch("well_seismic.gaia_agent.request_minimax_message") as provider:
        first = client.post(PATH, headers=peer_headers, json=body(request_id, conversation))
        assert first.status_code == 200
        result = first.json()
        assert result["request_state"] == "finished"
        assert result["error"]["code"] == "conversation_unavailable"
        assert result["conversation_id"] is None and result["executions"] == []
        assert conversation not in first.text and OWNER not in first.text
        second = client.post(PATH, headers=peer_headers, json=body(request_id, conversation))
        assert second.status_code == 200 and second.json() == result
        provider.assert_not_called()
    h.catalog.assert_not_called()
    h.execute.assert_not_called()
    with sqlite3.connect(h.root / "assistant_requests.sqlite3") as db:
        assert db.execute(
            "SELECT state FROM requests WHERE subject=? AND request_id=?",
            (PEER, str(request_id)),
        ).fetchone() == ("finished",)
