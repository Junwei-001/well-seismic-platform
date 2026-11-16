"""Status queries close the owned native execution, including its DVC receipt."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from uuid import uuid4

import pytest
from fastapi import HTTPException

from well_seismic.dvc_store import DvcStoreError
from well_seismic.gaia_agent import _bounded_receipt
from test_huiyan_harness import OWNER, OTHER, harness, project_asset


def start(harness, monkeypatch):
    project, asset = project_asset(harness)
    run = harness.execute(OWNER, "data_prepare", {"project_id": project, "asset_ids": [asset]})
    archives = []
    original = harness.dvc.archive
    def archive(*args, **kwargs):
        archives.append(args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(harness.dvc, "archive", archive)
    return project, run, archives


def persisted(harness, run):
    with harness.db() as db:
        return json.loads(db.execute("SELECT receipt FROM executions WHERE id=?", (run["execution_id"],)).fetchone()[0])


def status(harness, project, task, operation_id=None):
    return harness.execute(OWNER, "task_status", {"project_id": project, "task_id": task}, operation_id)


def test_status_finalizes_once_and_returns_same_original_revision_on_repeat(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    harness.test_tasks[run["task_id"]].update(status="completed", progress=100)
    operation_id = str(uuid4())
    answer = status(harness, project, run["task_id"], operation_id)
    done = persisted(harness, run)
    assert done["status"] == "completed" and done.get("revision_id")
    assert answer["status"] == "completed" and answer["revision_id"] == done["revision_id"]
    assert answer["result"]["original_executions"][0]["execution_id"] == run["execution_id"]
    assert status(harness, project, run["task_id"], operation_id) == answer
    again = status(harness, project, run["task_id"])
    assert again["revision_id"] == done["revision_id"]
    assert persisted(harness, run) == done
    assert len(archives) == 1 and len(harness.test_tasks) == 1


def test_status_archival_failure_remains_blocked_without_hidden_retry(harness, monkeypatch):
    project, run, _ = start(harness, monkeypatch)
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(1)
        raise DvcStoreError("dvc_unavailable", "private failure detail")
    monkeypatch.setattr(harness.dvc, "archive", unavailable)
    harness.test_tasks[run["task_id"]].update(status="completed", result={"large": "synthetic" * 2000})
    answer = status(harness, project, run["task_id"])
    original = persisted(harness, run)
    assert original["status"] == answer["status"] == "blocked"
    assert original["error"]["code"] == answer["error"]["code"] == "versioning_incomplete"
    assert not original.get("revision_id") and not answer.get("revision_id")
    short = _bounded_receipt(answer)
    assert short["status"] == "blocked" and short["error"]["code"] == "versioning_incomplete"
    assert short["result"]["original_executions"][0]["status"] == "blocked"
    assert "private failure detail" not in json.dumps(answer)
    assert status(harness, project, run["task_id"])["status"] == "blocked"
    assert len(attempts) == 1 and persisted(harness, run) == original


@pytest.mark.parametrize("native_status", ["failed", "cancelled", "interrupted"])
def test_status_preserves_native_failure(harness, monkeypatch, native_status):
    project, run, archives = start(harness, monkeypatch)
    harness.test_tasks[run["task_id"]].update(status=native_status, message="synthetic failure")
    answer = status(harness, project, run["task_id"])
    original = persisted(harness, run)
    assert original["status"] == "failed"
    assert original["error"]["code"] == "native_task_failed"
    assert answer["status"] == "blocked" and answer["error"]["code"] == "native_task_failed"
    assert not archives and not original.get("revision_id")


def test_status_preserves_business_gate_and_archived_diagnostic(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    harness.test_tasks[run["task_id"]].update(status="completed", result={
        "registration": {"business_status": "blocked", "status_counts": {"usable": 0}},
        "large": "synthetic" * 2000,
    })
    answer = status(harness, project, run["task_id"])
    original = persisted(harness, run)
    assert original["status"] == answer["status"] == "blocked"
    assert answer["error"]["code"] == "native_business_blocked"
    assert original.get("revision_id") and answer["result"]["revision_id"] == original["revision_id"]
    short = _bounded_receipt(answer)
    assert short["result"]["revision_id"] == original["revision_id"]
    assert short["result"]["original_executions"][0]["status"] == "blocked"
    assert len(archives) == 1


def test_status_cannot_finalize_other_owner_or_wrong_owned_project(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    another = harness.execute(OWNER, "project_create", {"name": "unrelated"})["project_id"]
    harness.test_tasks[run["task_id"]]["status"] = "completed"
    for subject, pid in [(OTHER, project), (OWNER, another)]:
        with pytest.raises(HTTPException) as caught:
            harness.execute(subject, "task_status", {"project_id": pid, "task_id": run["task_id"]})
        assert caught.value.status_code == 404
    assert persisted(harness, run)["status"] == "running" and not archives


def test_parallel_ui_refresh_and_ai_status_only_archive_once(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    harness.test_tasks[run["task_id"]]["status"] = "completed"
    archive = harness.dvc.archive
    entered, release = threading.Event(), threading.Event()
    def slow_archive(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return archive(*args, **kwargs)
    monkeypatch.setattr(harness.dvc, "archive", slow_archive)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(harness.get_run, OWNER, run["execution_id"])
        assert entered.wait(3)
        second = pool.submit(harness.get_run, OWNER, run["execution_id"])
        query = pool.submit(status, harness, project, run["task_id"])
        release.set()
        first, second, query = first.result(5), second.result(5), query.result(5)
    assert len(archives) == 1
    assert first == second == persisted(harness, run)
    assert query["revision_id"] == first["revision_id"]


def test_status_does_not_invent_native_execution_for_imported_task(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    imported = uuid4().hex
    harness.test_tasks[imported] = {"task_id": imported, "status": "completed", "result": {}}
    harness.bind_task(OWNER, project, imported)
    answer = status(harness, project, imported)
    assert answer["status"] == "completed"
    assert answer["result"]["original_executions"] == []
    assert not answer.get("revision_id") and not archives
    assert persisted(harness, run)["status"] == "running"


def test_status_keeps_queued_native_pending_without_self_refresh(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    answer = status(harness, project, run["task_id"])
    assert answer["status"] == "completed"
    assert answer["result"]["original_executions"][0]["status"] == "running"
    assert persisted(harness, run)["status"] == "running" and not archives
    assert len(answer["result"]["original_executions"]) == 1


def test_task_completing_between_reads_returns_consistent_task_and_execution(harness, monkeypatch):
    project, run, archives = start(harness, monkeypatch)
    calls = []
    def transitioning(task_id):
        assert task_id == run["task_id"]
        calls.append(1)
        snapshot = json.loads(json.dumps(harness.test_tasks[task_id]))
        snapshot.update(status="running" if len(calls) == 1 else "completed", progress=100)
        return snapshot
    monkeypatch.setitem(harness.native, "task", transitioning)
    answer = status(harness, project, run["task_id"])
    assert answer["result"]["task"]["status"] == "completed"
    assert answer["result"]["original_executions"][0]["status"] == "completed"
    assert answer["revision_id"] == persisted(harness, run)["revision_id"]
    assert len(archives) == 1 and len(calls) >= 3


@pytest.mark.parametrize("outcome", ["completed", "archive_blocked", "native_failed", "business_blocked"])
def test_agent_final_response_refreshes_original_execution_after_status(harness, monkeypatch, outcome):
    from well_seismic.gaia_agent import chat_with_huiyan_agent
    from well_seismic.gaia_minimax import GaiaMiniMaxSettings

    project, asset = project_asset(harness)
    attempts = []
    archive = harness.dvc.archive
    def record_archive(*args, **kwargs):
        attempts.append(1)
        if outcome == "archive_blocked":
            raise DvcStoreError("dvc_unavailable", "private-error")
        return archive(*args, **kwargs)
    monkeypatch.setattr(harness.dvc, "archive", record_archive)
    rounds = []
    def provider(settings, payload):
        rounds.append(1)
        if len(rounds) == 1:
            name, args = "data_prepare", {"project_id": project, "asset_ids": [asset]}
        elif len(rounds) == 2:
            task_id, task = next(iter(harness.test_tasks.items()))
            task.update(status="failed" if outcome == "native_failed" else "completed", progress=100)
            if outcome == "business_blocked":
                task["result"] = {"registration": {"business_status": "blocked"}}
            name, args = "task_status", {"project_id": project, "task_id": task_id}
        else:
            return {"model": "unit", "content": [{"type": "text", "text": "已完成所有操作。"}], "stop_reason": "end_turn"}
        return {"model": "unit", "content": [{"type": "tool_use", "id": "call" + str(len(rounds)), "name": name, "input": args}], "stop_reason": "tool_use"}
    monkeypatch.setattr("well_seismic.gaia_agent.request_minimax_message", provider)
    request_id = uuid4()
    response = chat_with_huiyan_agent(
        GaiaMiniMaxSettings("unit", "unit", "unit-key"), subject=OWNER,
        question="请检查当前项目并确认完成及归档结果。", context={},
        conversation_id=None, harness=harness, request_id=request_id,
    )
    original = response["executions"][0]
    expected = {"completed": "completed", "archive_blocked": "blocked", "native_failed": "failed", "business_blocked": "blocked"}[outcome]
    assert original["status"] == expected
    assert original == _bounded_receipt(persisted(harness, original))
    assert len(harness.test_tasks) == 1
    assert len(attempts) == (0 if outcome == "native_failed" else 1)
    if outcome == "completed":
        assert original["revision_id"] == response["executions"][1]["revision_id"]
        assert "未完成项" not in response["answer"]
    else:
        assert "已完成所有操作。" not in response["answer"]
        assert original.get("error")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(harness.root / "assistant_requests.sqlite3")) as db:
        stored = json.loads(db.execute("SELECT response FROM requests WHERE subject=? AND request_id=?", (OWNER, str(request_id))).fetchone()[0])
    assert stored["executions"] == response["executions"]


@pytest.mark.parametrize("outcome", ["completed", "blocked"])
def test_late_native_completion_updates_response_and_next_turn_history(harness, monkeypatch, outcome):
    from uuid import UUID
    from well_seismic.gaia_agent import chat_with_huiyan_agent
    from well_seismic.gaia_minimax import GaiaMiniMaxSettings

    project, asset = project_asset(harness)
    if outcome == "blocked":
        def unavailable(*args, **kwargs):
            raise DvcStoreError("dvc_unavailable", "private-error")
        monkeypatch.setattr(harness.dvc, "archive", unavailable)
    requests = []
    old_prose = "供应商最后一轮旧说法：所有归档已完成。"
    def provider(settings, payload):
        requests.append(json.loads(json.dumps(payload)))
        if len(requests) == 1:
            return {"model": "unit", "content": [{"type": "tool_use", "id": "prepare", "name": "data_prepare", "input": {"project_id": project, "asset_ids": [asset]}}], "stop_reason": "tool_use"}
        if len(requests) == 2:
            # There is no task_status tool in this turn. Only final refresh can
            # discover the terminal state after the provider receives running.
            prior_result = json.loads(payload["messages"][-1]["content"][0]["content"])
            assert prior_result["status"] == "running"
            next(iter(harness.test_tasks.values())).update(status="completed", progress=100)
            return {"model": "unit", "content": [{"type": "text", "text": old_prose}], "stop_reason": "end_turn"}
        return {"model": "unit", "content": [{"type": "text", "text": "已根据实际回执核对。"}], "stop_reason": "end_turn"}
    monkeypatch.setattr("well_seismic.gaia_agent.request_minimax_message", provider)
    settings = GaiaMiniMaxSettings("unit", "unit", "unit-key")
    result = chat_with_huiyan_agent(settings, subject=OWNER, question="请检查项目数据。", context={}, conversation_id=None, harness=harness)
    assert result["executions"][0]["status"] == outcome
    assert old_prose not in result["answer"]
    chat_with_huiyan_agent(settings, subject=OWNER, question="现在状态是什么？", context={}, conversation_id=UUID(result["conversation_id"]), harness=harness)
    previous_assistant = [message for message in requests[-1]["messages"][:-1] if message["role"] == "assistant"][-1]
    assert previous_assistant["content"] == result["answer"]
    assert old_prose not in json.dumps(requests[-1]["messages"], ensure_ascii=False)
    assert len(harness.test_tasks) == 1
