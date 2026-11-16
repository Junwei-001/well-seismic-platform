"""Concurrent assistant requests retain truthful, durable terminal rejections."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest

from well_seismic.assistant_requests import AssistantRequestStore
from well_seismic.gaia_agent import chat_with_huiyan_agent
from well_seismic.gaia_minimax import GaiaMiniMaxError, GaiaMiniMaxSettings


SETTINGS = GaiaMiniMaxSettings("unit-provider", "MiniMax-M2.7", "unit-only-key")
OWNER = "a" * 64
QUESTION = "请列出我的项目"


def harness(tmp_path):
    value = Mock()
    value.root = tmp_path
    value.catalog.return_value = {
        "actions": [{
            "name": "project_list", "description": "list owned projects",
            "parameters": {"type": "object", "properties": {}},
        }]
    }
    value.execute.return_value = {
        "execution_id": "hx_" + "1" * 32, "operation": "project_list",
        "status": "completed", "result": {"projects": []},
    }
    return value


def message(content):
    return {"model": SETTINGS.model, "content": content, "stop_reason": "end_turn"}


def ask(h, request_id, conversation_id=None, *, subject=OWNER):
    return chat_with_huiyan_agent(
        SETTINGS, subject=subject, question=QUESTION, context={},
        conversation_id=conversation_id, harness=h, request_id=request_id,
    )


def payload(conversation_id):
    return {
        "question": QUESTION, "task_id": None,
        "conversation_id": str(conversation_id) if conversation_id else None,
    }


def prime_conversation(h):
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        return_value=message([{"type": "text", "text": "ready"}]),
    ):
        conversation = UUID(ask(h, uuid4())["conversation_id"])
    h.reset_mock()
    return conversation


def test_competing_uuid_is_terminal_and_replayed_after_lock_released(tmp_path):
    h = harness(tmp_path)
    conversation = prime_conversation(h)
    first_id, rejected_id = uuid4(), uuid4()
    entered, release = Event(), Event()

    def provider(_settings, request):
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "test failed to release the first provider call"
            return message([{
                "type": "tool_use", "id": "owned-list", "name": "project_list",
                "input": {},
            }])
        return message([{"type": "text", "text": "完成"}])

    with patch("well_seismic.gaia_agent.request_minimax_message", side_effect=provider) as request:
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(ask, h, first_id, conversation)
            try:
                assert entered.wait(5)
                # A duplicate of the active UUID must never start another run.
                duplicate_active = ask(h, first_id, conversation)
                assert duplicate_active["request_state"] == "in_progress"
                try:
                    rejection = ask(h, rejected_id, conversation)
                except GaiaMiniMaxError:
                    # Preserve the observed pre-fix ledger failure in the assertion.
                    rejection = None
                ledger = AssistantRequestStore(h.root / "assistant_requests.sqlite3")
                rejected_record = ledger.claim(OWNER, rejected_id, payload(conversation))
                assert rejected_record["state"] == "finished"
                assert rejection == rejected_record["response"]
                assert rejection["error"]["code"] == "conversation_busy"
                assert rejection["request_state"] == "finished"
                assert rejection["executions"] == []
                assert rejection["conversation_id"] == str(conversation)
                h.execute.assert_not_called()
                assert request.call_count == 1
                assert ask(h, rejected_id, conversation) == rejection
            finally:
                release.set()
                first = running.result(timeout=10)
        assert first["request_state"] == "finished"
        assert len(first["executions"]) == 1
        assert ask(h, first_id, conversation) == first
        # Neither a later retry nor loss of the in-memory session reruns a rejection.
        with patch("well_seismic.gaia_agent._session", side_effect=AssertionError("must replay ledger")):
            assert ask(h, rejected_id, conversation) == rejection
        assert h.execute.call_count == 1
        assert request.call_count == 2
        assert h.catalog.call_count == 1


def test_foreign_conversation_rejection_is_terminal_and_never_reexecuted(tmp_path):
    h = harness(tmp_path)
    conversation = prime_conversation(h)
    request_id = uuid4()
    with patch("well_seismic.gaia_agent.request_minimax_message") as request:
        result = ask(h, request_id, conversation, subject="b" * 64)
        assert result["request_state"] == "finished"
        assert result["error"]["code"] == "conversation_unavailable"
        assert result["executions"] == []
        assert result["conversation_id"] is None
        assert ask(h, request_id, conversation, subject="b" * 64) == result
    record = AssistantRequestStore(h.root / "assistant_requests.sqlite3").claim(
        "b" * 64, request_id, payload(conversation),
    )
    assert record == {"state": "finished", "response": result}
    h.catalog.assert_not_called()
    h.execute.assert_not_called()
    request.assert_not_called()


def test_unexpected_failure_after_operation_keeps_uncertain_request_claim(tmp_path):
    h = harness(tmp_path)
    request_id = uuid4()
    with patch(
        "well_seismic.gaia_agent.request_minimax_message",
        side_effect=[
            message([{"type": "tool_use", "id": "list", "name": "project_list", "input": {}}]),
            RuntimeError("unexpected provider termination"),
        ],
    ) as request:
        with pytest.raises(RuntimeError, match="unexpected provider termination"):
            ask(h, request_id)
        replay = ask(h, request_id)
        assert replay["request_state"] == "in_progress"
        assert replay["executions"] == [h.execute.return_value]
        assert request.call_count == 2
    h.execute.assert_called_once()


def test_uncertain_failure_before_receipt_is_not_automatically_reclaimed(tmp_path):
    h = harness(tmp_path)
    request_id = uuid4()
    # Only explicitly classified pre-execution rejections may be finalized.
    h.catalog.side_effect = RuntimeError("unknown startup failure")
    with pytest.raises(RuntimeError, match="unknown startup failure"):
        ask(h, request_id)
    replay = ask(h, request_id)
    assert replay["request_state"] == "in_progress"
    assert replay["executions"] == []
    assert h.catalog.call_count == 1
    h.execute.assert_not_called()
