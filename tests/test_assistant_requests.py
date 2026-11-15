"""Durable assistant idempotency boundary tests; no provider calls or credentials."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from well_seismic.assistant_requests import (
    MAX_BODY_BYTES,
    AssistantRequestError,
    AssistantRequestStore,
)

OWNER = "a" * 64
OTHER = "b" * 64
PAYLOAD = {
    "question": "执行数据检查",
    "context": {"task_id": "test"},
    "conversation_id": None,
}


def code(expected, call):
    with pytest.raises(AssistantRequestError) as caught:
        call()
    assert caught.value.code == expected


@pytest.fixture
def ledger(tmp_path):
    return AssistantRequestStore(tmp_path / "assistant_requests.sqlite3")


def test_progress_and_finish_survive_restart_without_reexecution(ledger):
    key = str(uuid4())
    assert ledger.claim(OWNER, key, PAYLOAD) == {"state": "fresh", "response": None}
    progress = {
        "request_id": key,
        "executions": [{"execution_id": "hx_unit", "status": "completed"}],
    }
    ledger.save_progress(OWNER, key, progress)
    restarted = AssistantRequestStore(ledger.path)
    assert restarted.claim(OWNER, key, PAYLOAD) == {
        "state": "in_progress",
        "response": progress,
    }
    final = dict(progress, answer="数据检查已完成。")
    restarted.finish(OWNER, key, final)
    assert ledger.claim(OWNER, key, PAYLOAD) == {"state": "finished", "response": final}
    ledger.finish(OWNER, key, final)
    code("request_conflict", lambda: ledger.save_progress(OWNER, key, progress))
    code("request_conflict", lambda: ledger.finish(OWNER, key, {"answer": "changed"}))


def test_unfinished_empty_claim_is_not_automatically_reclaimed(ledger):
    key = str(uuid4())
    ledger.claim(OWNER, key, PAYLOAD)
    assert AssistantRequestStore(ledger.path).claim(OWNER, key, PAYLOAD) == {
        "state": "in_progress",
        "response": None,
    }


@pytest.mark.parametrize("round_number", range(8))
def test_duplicate_parallel_claim_has_exactly_one_winner(ledger, round_number):
    # Each parametrized run starts with a genuinely cold database. Every worker
    # creates an independent store/connection, without a process-local lock.
    key = str(uuid4())
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _: AssistantRequestStore(ledger.path).claim(OWNER, key, PAYLOAD),
                range(32),
            )
        )
    assert sum(item["state"] == "fresh" for item in results) == 1
    assert sum(item["state"] == "in_progress" for item in results) == 31


def test_owner_isolation_and_payload_conflict(ledger):
    key = str(uuid4())
    ledger.claim(OWNER, key, PAYLOAD)
    ledger.finish(OWNER, key, {"answer": "owner-private"})
    code(
        "request_not_found", lambda: ledger.finish(OTHER, key, {"answer": "overwrite"})
    )
    assert ledger.claim(OTHER, key, PAYLOAD) == {"state": "fresh", "response": None}
    code(
        "request_conflict",
        lambda: ledger.claim(OWNER, key, dict(PAYLOAD, question="different")),
    )
    assert ledger.claim(OWNER, key, PAYLOAD)["response"] == {"answer": "owner-private"}


def test_only_payload_hash_is_persisted(ledger):
    key = str(uuid4())
    ledger.claim(OWNER, key, dict(PAYLOAD, question="payload-unique-never-persist"))
    with ledger._db() as db:
        dump = "\n".join(db.iterdump())
    assert "payload-unique-never-persist" not in dump
    assert "task_id" not in dump


@pytest.mark.parametrize(
    "response",
    [
        {"api_key": "unit-only"},
        {"nested": [{"type": "thinking", "text": "private"}]},
        {"Authorization": "unit-only"},
        {"thinking": "private"},
        {"value": float("nan")},
    ],
)
def test_private_or_invalid_response_is_rejected(ledger, response):
    key = str(uuid4())
    ledger.claim(OWNER, key, PAYLOAD)
    code("invalid_response", lambda: ledger.save_progress(OWNER, key, response))
    assert ledger.claim(OWNER, key, PAYLOAD)["response"] is None


def test_body_size_and_identity_validation(ledger):
    key = str(uuid4())
    code("invalid_owner", lambda: ledger.claim("visitor", key, PAYLOAD))
    code("invalid_request", lambda: ledger.claim(OWNER, "../request", PAYLOAD))
    code(
        "body_limit",
        lambda: ledger.claim(OWNER, key, {"question": "x" * MAX_BODY_BYTES}),
    )
    ledger.claim(OWNER, key, PAYLOAD)
    code(
        "body_limit",
        lambda: ledger.finish(OWNER, key, {"answer": "中" * (MAX_BODY_BYTES // 2)}),
    )
    code(
        "invalid_response",
        lambda: ledger.finish(OWNER, key, {"request_id": str(uuid4())}),
    )
    assert ledger.claim(OWNER, key, PAYLOAD)["state"] == "in_progress"


def test_structured_public_metadata_type_is_not_treated_as_private(ledger):
    key = str(uuid4())
    ledger.claim(OWNER, key, PAYLOAD)
    response = {"result": {"type": ["curve", "numeric"]}}
    ledger.finish(OWNER, key, response)
    assert ledger.claim(OWNER, key, PAYLOAD)["response"] == response
