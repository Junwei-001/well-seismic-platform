"""Durable owner-scoped idempotency for assistant requests.

The payload is hashed, never persisted. Only caller-sanitized public progress and
final responses are stored; thinking blocks and credential fields are rejected.
An unfinished request is never implicitly reclaimed after a restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

MAX_BODY_BYTES = 256 * 1024
_OWNER = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_KEYS = {
    "thinking",
    "redacted_thinking",
    "api_key",
    "authorization",
    "password",
    "access_token",
    "refresh_token",
    "client_secret",
    "raw_request",
    "raw_response",
}


class AssistantRequestError(RuntimeError):
    """Stable, sanitized ledger error suitable for an authenticated endpoint."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fail(code: str, message: str):
    raise AssistantRequestError(code, message)


def _identity(subject: str, request_id: str | UUID) -> tuple[str, str]:
    if not isinstance(subject, str) or not _OWNER.fullmatch(subject):
        _fail("invalid_owner", "An authenticated owner is required.")
    try:
        key = str(UUID(str(request_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AssistantRequestError(
            "invalid_request", "A UUID request identifier is required."
        ) from exc
    return subject, key


def _encoded(value: Any, *, response=False) -> str:
    if not isinstance(value, dict):
        _fail(
            "invalid_response" if response else "invalid_request",
            "Request bodies must be JSON objects.",
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise AssistantRequestError(
            "invalid_response" if response else "invalid_request",
            "Request bodies must be finite JSON.",
        ) from exc
    if len(encoded.encode("utf-8")) > MAX_BODY_BYTES:
        _fail("body_limit", "The assistant request or response exceeds the size limit.")
    if response:

        def check(item, depth=0):
            if depth > 24:
                _fail("invalid_response", "Response nesting exceeds the limit.")
            if isinstance(item, dict):
                if any(str(key).lower() in _PRIVATE_KEYS for key in item) or (
                    isinstance(item.get("type"), str)
                    and item["type"] in {"thinking", "redacted_thinking"}
                ):
                    _fail(
                        "invalid_response",
                        "Private provider or credential fields cannot be persisted.",
                    )
                for child in item.values():
                    check(child, depth + 1)
            elif isinstance(item, list):
                for child in item:
                    check(child, depth + 1)

        check(value)
    return encoded


def _safe_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for item in [*reversed(absolute.parents), absolute]:
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            _fail(
                "invalid_configuration",
                "The ledger path cannot contain links or reparse points.",
            )
    return absolute


class AssistantRequestStore:
    """Exactly one claimant per authenticated owner and stable client request ID."""

    def __init__(self, db_path: Path):
        self.path = _safe_path(Path(db_path))

    @contextmanager
    def _db(self):
        connection = None
        try:
            _safe_path(self.path)
            for suffix in ("-wal", "-shm", "-journal"):
                _safe_path(Path(str(self.path) + suffix))
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA synchronous=FULL")
            # Claim the write lock before checking/creating the schema. Switching
            # journal modes per connection races on a fresh database, and a
            # deferred read-to-write upgrade can fail without a busy wait.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""CREATE TABLE IF NOT EXISTS requests (
                subject TEXT NOT NULL, request_id TEXT NOT NULL, payload_digest TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('in_progress','finished')),
                response TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(subject, request_id))""")
            yield connection
            connection.commit()
        except (sqlite3.Error, OSError) as exc:
            if connection is not None:
                connection.rollback()
            raise AssistantRequestError(
                "storage_failed", "The assistant request ledger is unavailable."
            ) from exc
        except Exception:
            if connection is not None:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()

    def claim(self, subject: str, request_id: str | UUID, payload: dict) -> dict:
        """Return fresh, a stored final response, or the last unfinished progress."""
        subject, key = _identity(subject, request_id)
        digest = hashlib.sha256(_encoded(payload).encode("utf-8")).hexdigest()
        now = datetime.now(UTC).isoformat()
        with self._db() as db:
            row = db.execute(
                "SELECT payload_digest,state,response FROM requests WHERE subject=? AND request_id=?",
                (subject, key),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    _fail(
                        "request_conflict",
                        "This request identifier was already used for different input.",
                    )
                response = json.loads(row[2]) if row[2] is not None else None
                return {"state": row[1], "response": response}
            db.execute(
                "INSERT INTO requests VALUES(?,?,?,'in_progress',NULL,?,?)",
                (subject, key, digest, now, now),
            )
            return {"state": "fresh", "response": None}

    def _save(
        self, subject: str, request_id: str | UUID, response: dict, *, finish: bool
    ):
        subject, key = _identity(subject, request_id)
        encoded = _encoded(response, response=True)
        if response.get("request_id") not in (None, key):
            _fail(
                "invalid_response",
                "Response request identity does not match its claim.",
            )
        with self._db() as db:
            row = db.execute(
                "SELECT state,response FROM requests WHERE subject=? AND request_id=?",
                (subject, key),
            ).fetchone()
            if row is None:
                _fail("request_not_found", "The assistant request was not found.")
            if row[0] == "finished":
                if finish and row[1] == encoded:
                    return
                _fail("request_conflict", "A completed response cannot be overwritten.")
            db.execute(
                "UPDATE requests SET state=?,response=?,updated_at=? WHERE subject=? AND request_id=?",
                (
                    "finished" if finish else "in_progress",
                    encoded,
                    datetime.now(UTC).isoformat(),
                    subject,
                    key,
                ),
            )

    def save_progress(
        self, subject: str, request_id: str | UUID, response: dict
    ) -> None:
        self._save(subject, request_id, response, finish=False)

    def finish(self, subject: str, request_id: str | UUID, response: dict) -> None:
        self._save(subject, request_id, response, finish=True)
