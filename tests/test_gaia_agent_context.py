"""Owned task context and safe error classification for natural sidebar actions."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from well_seismic.api_models import AssistantChatRequest
from well_seismic.gaia_agent import chat_with_huiyan_agent
from well_seismic.gaia_minimax import GaiaMiniMaxSettings
from well_seismic.huiyan_harness import HuiyanHarness

OWNER, OTHER = "a" * 64, "b" * 64
TASK = "c" * 32
SETTINGS = GaiaMiniMaxSettings("unit-provider", "MiniMax-M2.7", "unit-only-key")


@pytest.fixture
def harness(tmp_path):
    return HuiyanHarness(
        tmp_path / "data" / "huiyan_harness", config={}, native={},
        legacy_root=tmp_path / "data" / "gaia_tasks",
    )


def owned_project(harness):
    project = harness.execute(OWNER, "project_create", {"name": "current task project"})
    harness.bind_task(OWNER, project["project_id"], TASK)
    # The current task must not use the latest entry in project_list.
    harness.execute(OWNER, "project_create", {"name": "newer unrelated project"})
    return project["project_id"]


def test_selected_task_context_resolves_exact_owned_project_without_writes(harness):
    project = owned_project(harness)
    ledger = harness.root / "harness.sqlite3"
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    result = harness.assistant_context(OWNER, TASK)
    assert result == {
        "binding": "bound", "task_id": TASK,
        "project_id": project, "project_name": "current task project",
    }
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    assert "subject" not in result and str(harness.root) not in json.dumps(result)


@pytest.mark.parametrize("subject, task", [(OTHER, TASK), (OWNER, "f" * 32), (OWNER, "../task")])
def test_unowned_or_unknown_task_context_is_denied(harness, subject, task):
    owned_project(harness)
    with pytest.raises(HTTPException) as error:
        harness.assistant_context(subject, task)
    assert error.value.status_code == 404


def test_legacy_owned_task_is_explicitly_unbound_without_creating_project(harness):
    harness.legacy_root.mkdir(parents=True)
    with sqlite3.connect(harness.legacy_root / "receipts.sqlite3") as db:
        db.execute("CREATE TABLE receipts(run_id TEXT,task_id TEXT,subject TEXT)")
        db.execute("INSERT INTO receipts VALUES(?,?,?)", ("legacy", TASK, OWNER))
    context = harness.assistant_context(OWNER, TASK)
    assert context["binding"] == "unbound" and context["task_id"] == TASK
    assert "project_id" not in context
    assert not (harness.root / "harness.sqlite3").exists()
    with pytest.raises(HTTPException) as error:
        harness.assistant_context(OTHER, TASK)
    assert error.value.status_code == 404


def test_unselected_context_does_not_create_storage(harness):
    assert harness.assistant_context(OWNER, None) == {"binding": "unselected"}
    assert not harness.root.exists()


def test_inconsistent_project_owner_is_not_accepted_as_legacy(harness):
    project = owned_project(harness)
    with harness.db() as db:
        db.execute("UPDATE projects SET subject=? WHERE id=?", (OTHER, project))
    with pytest.raises(HTTPException) as error:
        harness.assistant_context(OWNER, TASK)
    assert error.value.status_code == 404


def test_authenticated_api_keeps_server_binding_first_and_sanitized(harness):
    from well_seismic import api

    project = owned_project(harness)
    with (
        patch.object(api, "_huiyan_harness", harness),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=SETTINGS),
        patch.object(api, "_assistant_context", return_value={
            "task_id": TASK, "path": "D:/private/well.las", "summary": "x" * 17000,
        }),
        patch("well_seismic.gaia_agent.chat_with_huiyan_agent", return_value={"answer": "ok"}) as agent,
    ):
        api.assistant_chat_authenticated(AssistantChatRequest(message="rename", task_id=TASK), subject=OWNER)
    context = agent.call_args.kwargs["context"]
    assert list(context)[0] == "harness"
    assert context["harness"]["project_id"] == project
    assert project in json.dumps(context, ensure_ascii=False)[:16000]
    assert "D:/private" not in json.dumps(context)


def test_authenticated_api_rejects_other_owner_before_native_context_or_model(harness):
    from well_seismic import api

    owned_project(harness)
    with (
        patch.object(api, "_huiyan_harness", harness),
        patch.object(GaiaMiniMaxSettings, "from_environment", return_value=SETTINGS),
        patch.object(api, "_assistant_context") as native_context,
        patch("well_seismic.gaia_agent.chat_with_huiyan_agent") as agent,
        pytest.raises(HTTPException) as error,
    ):
        api.assistant_chat_authenticated(AssistantChatRequest(message="rename", task_id=TASK), subject=OTHER)
    assert error.value.status_code == 404
    native_context.assert_not_called()
    agent.assert_not_called()


@pytest.mark.parametrize("status, code", [
    (401, "not_found"), (403, "not_found"), (404, "not_found"),
    (409, "request_conflict"), (413, "input_limit"),
    (422, "invalid_arguments"), (503, "service_unavailable"), (500, "execution_error"),
])
def test_expected_http_errors_keep_safe_distinct_codes(tmp_path, status, code):
    h = Mock()
    h.root = tmp_path
    h.catalog.return_value = {"actions": [{"name": "project_list", "description": "list", "parameters": {"type": "object"}}]}
    h.execute.side_effect = HTTPException(status, "Bearer private-value at D:/private/file")
    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=[
        {"model": SETTINGS.model, "content": [{"type": "tool_use", "id": "call1", "name": "project_list", "input": {}}], "stop_reason": "tool_use"},
        {"model": SETTINGS.model, "content": [{"type": "text", "text": "All done"}], "stop_reason": "end_turn"},
    ]) as provider:
        result = chat_with_huiyan_agent(SETTINGS, subject=OWNER, question="list", context={}, conversation_id=None, harness=h)
    receipt = result["executions"][0]
    assert receipt["status"] == "failed"
    assert receipt["error"]["code"] == code
    assert code in provider.call_args.args[1]["messages"][-1]["content"][0]["content"]
    assert "private-value" not in json.dumps(result) and "D:/private" not in json.dumps(result)
    assert "All done" not in result["answer"]
