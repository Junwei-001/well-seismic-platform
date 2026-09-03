"""Durable local control-plane state for the desktop platform.

The platform stores large seismic and model products on the filesystem.  This
module deliberately stores only their small control-plane manifests in SQLite:
projects, immutable-input snapshots, asynchronous tasks, and artifact bundles.

The implementation uses only the Python standard library.  Each operation gets
its own SQLite connection, while an in-process re-entrant lock makes compound
read/modify/write operations safe for the API's worker threads.  WAL mode also
allows a second process (for example a GPU worker) to read state concurrently.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"queued", "running", "completed", "failed", "cancelled"}
)
TERMINAL_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "cancelled"}
)

_SCHEMA_VERSION: Final[int] = 1
_PROJECT_RESERVED = frozenset({"project_id", "created_at", "updated_at"})
_SNAPSHOT_RESERVED = frozenset(
    {"snapshot_id", "project_id", "created_at", "updated_at"}
)
_TASK_RESERVED = frozenset(
    {
        "task_id",
        "project_id",
        "snapshot_id",
        "task_type",
        "status",
        "created_at",
        "updated_at",
    }
)
_BUNDLE_RESERVED = frozenset(
    {
        "bundle_id",
        "project_id",
        "snapshot_id",
        "task_id",
        "created_at",
        "updated_at",
    }
)


class StateStoreError(RuntimeError):
    """Base exception raised by the persistent control-plane store."""


class RecordNotFoundError(KeyError, StateStoreError):
    """Raised when an entity id does not exist."""


class ConcurrentStateError(StateStoreError):
    """Raised when a compare-and-set task transition loses a race."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state_db_path() -> Path:
    """Return the configured database path without creating it.

    ``WELL_SEISMIC_STATE_DB`` has highest priority.  Otherwise the database is
    placed beside generated model products, never under the source-data tree.
    """

    configured = os.getenv("WELL_SEISMIC_STATE_DB", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "model_outputs" / "platform_state.sqlite3").resolve()


def _json_dump(payload: Mapping[str, Any] | None) -> str:
    value = dict(payload or {})
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("state payload must contain only finite JSON values") from exc


def _json_load(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise StateStoreError("stored payload is not a JSON object")
    return value


def _payload_without(payload: Mapping[str, Any], reserved: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in reserved}


def _validate_identifier(value: str, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _validate_limit(limit: int | None, offset: int) -> tuple[int | None, int]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must not be negative")
    return limit, offset


class SQLiteStateStore:
    """SQLite-backed repository for local platform control-plane records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else default_state_db_path()
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS state_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_project
                    ON snapshots(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    snapshot_id TEXT,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
                    ),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_status_updated
                    ON tasks(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_project_updated
                    ON tasks(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS artifact_bundles (
                    bundle_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    snapshot_id TEXT,
                    task_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id),
                    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id),
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_bundles_task_updated
                    ON artifact_bundles(task_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_bundles_project_updated
                    ON artifact_bundles(project_id, updated_at DESC);
                INSERT INTO state_meta(key, value) VALUES ('schema_version', '1')
                    ON CONFLICT(key) DO NOTHING;
                COMMIT;
                """
            )
            row = connection.execute(
                "SELECT value FROM state_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None or int(row["value"]) != _SCHEMA_VERSION:
                raise StateStoreError(
                    f"unsupported state schema version: {None if row is None else row['value']}"
                )

    @staticmethod
    def _row_project(row: sqlite3.Row) -> dict[str, Any]:
        result = _json_load(row["payload_json"])
        result.update(
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        return result

    @staticmethod
    def _row_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        result = _json_load(row["payload_json"])
        result.update(
            snapshot_id=row["snapshot_id"],
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        return result

    @staticmethod
    def _row_task(row: sqlite3.Row) -> dict[str, Any]:
        result = _json_load(row["payload_json"])
        result.update(
            task_id=row["task_id"],
            project_id=row["project_id"],
            snapshot_id=row["snapshot_id"],
            task_type=row["task_type"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        return result

    @staticmethod
    def _row_bundle(row: sqlite3.Row) -> dict[str, Any]:
        result = _json_load(row["payload_json"])
        result.update(
            bundle_id=row["bundle_id"],
            project_id=row["project_id"],
            snapshot_id=row["snapshot_id"],
            task_id=row["task_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        return result

    def create_project(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        raw = dict(payload or {})
        identifier = _validate_identifier(
            project_id or str(raw.get("project_id") or uuid.uuid4().hex), "project_id"
        )
        now = _utc_now()
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO projects VALUES (?, ?, ?, ?)",
                    (identifier, _json_dump(_payload_without(raw, _PROJECT_RESERVED)), now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise StateStoreError(f"project already exists: {identifier}") from exc
        return self.get_project(identifier)

    def ensure_project(
        self,
        project_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an existing project or create it once.

        This is intentionally a get-or-create operation rather than an upsert:
        a later caller cannot silently replace control-plane metadata chosen by
        the creator.  Explicit changes continue to use :meth:`update_project`.
        """

        identifier = _validate_identifier(project_id, "project_id")
        raw = dict(payload or {})
        payload_json = _json_dump(_payload_without(raw, _PROJECT_RESERVED))
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO projects(project_id, payload_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(project_id) DO NOTHING""",
                (identifier, payload_json, now, now),
            )
        return self.get_project(identifier)

    def get_project(self, project_id: str) -> dict[str, Any]:
        identifier = _validate_identifier(project_id, "project_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(identifier)
        return self._row_project(row)

    def update_project(
        self, project_id: str, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        identifier = _validate_identifier(project_id, "project_id")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM projects WHERE project_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RecordNotFoundError(identifier)
            payload = _json_load(row["payload_json"])
            payload.update(_payload_without(dict(updates), _PROJECT_RESERVED))
            connection.execute(
                "UPDATE projects SET payload_json = ?, updated_at = ? WHERE project_id = ?",
                (_json_dump(payload), _utc_now(), identifier),
            )
            connection.commit()
        return self.get_project(identifier)

    def list_projects(self, *, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        limit, offset = _validate_limit(limit, offset)
        sql = "SELECT * FROM projects ORDER BY updated_at DESC, project_id"
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            parameters = (limit, offset)
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            parameters = (offset,)
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._row_project(row) for row in rows]

    def create_snapshot(
        self,
        project_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        project = _validate_identifier(project_id, "project_id")
        raw = dict(payload or {})
        identifier = _validate_identifier(
            snapshot_id or str(raw.get("snapshot_id") or uuid.uuid4().hex), "snapshot_id"
        )
        now = _utc_now()
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
                    (
                        identifier,
                        project,
                        _json_dump(_payload_without(raw, _SNAPSHOT_RESERVED)),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateStoreError(
                    f"cannot create snapshot {identifier} for project {project}"
                ) from exc
        return self.get_snapshot(identifier)

    def seal_snapshot(
        self,
        project_id: str,
        snapshot_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create one immutable sealed snapshot, idempotently.

        The only legal transition is absence to ``sealed``.  Repeating the
        exact same seal operation returns the original row without changing
        its timestamps.  Reusing an id for another project or payload is a
        lineage conflict and is rejected.
        """

        project = _validate_identifier(project_id, "project_id")
        identifier = _validate_identifier(snapshot_id, "snapshot_id")
        raw = _payload_without(dict(payload or {}), _SNAPSHOT_RESERVED)
        declared_state = raw.get("state")
        if declared_state not in {None, "sealed"}:
            raise ValueError("seal_snapshot payload state must be 'sealed'")
        raw["state"] = "sealed"
        payload_json = _json_dump(raw)
        now = _utc_now()

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_id, payload_json FROM snapshots WHERE snapshot_id = ?",
                (identifier,),
            ).fetchone()
            if row is not None:
                existing_project = str(row["project_id"])
                existing_payload = _json_dump(_json_load(row["payload_json"]))
                if existing_project != project:
                    connection.rollback()
                    raise StateStoreError(
                        f"snapshot {identifier} is already sealed for project "
                        f"{existing_project}, not {project}"
                    )
                if existing_payload != payload_json:
                    connection.rollback()
                    raise StateStoreError(
                        f"snapshot {identifier} is already sealed with different payload"
                    )
                connection.commit()
            else:
                try:
                    connection.execute(
                        "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
                        (identifier, project, payload_json, now, now),
                    )
                except sqlite3.IntegrityError as exc:
                    connection.rollback()
                    raise StateStoreError(
                        f"cannot seal snapshot {identifier} for project {project}"
                    ) from exc
                connection.commit()
        return self.get_snapshot(identifier)

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        identifier = _validate_identifier(snapshot_id, "snapshot_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(identifier)
        return self._row_snapshot(row)

    def update_snapshot(
        self, snapshot_id: str, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            current = self.get_snapshot(snapshot_id)
            if current.get("state") == "sealed":
                raise StateStoreError(f"sealed snapshot is immutable: {snapshot_id}")
            return self._update_json_record(
                table="snapshots",
                id_column="snapshot_id",
                identifier=snapshot_id,
                updates=updates,
                reserved=_SNAPSHOT_RESERVED,
                reader=self.get_snapshot,
            )

    def list_snapshots(
        self,
        *,
        project_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit, offset = _validate_limit(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            parameters.append(_validate_identifier(project_id, "project_id"))
        rows = self._list_rows("snapshots", clauses, parameters, limit, offset)
        return [self._row_snapshot(row) for row in rows]

    def create_task(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        task_id: str | None = None,
        project_id: str | None = None,
        snapshot_id: str | None = None,
        task_type: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        raw = dict(payload or {})
        identifier = _validate_identifier(
            task_id or str(raw.get("task_id") or uuid.uuid4().hex), "task_id"
        )
        project = project_id if project_id is not None else raw.get("project_id")
        snapshot = snapshot_id if snapshot_id is not None else raw.get("snapshot_id")
        kind = _validate_identifier(task_type or str(raw.get("task_type") or "task"), "task_type")
        state = str(status or raw.get("status") or "queued")
        self._validate_task_status(state)
        normalized_project = (
            _validate_identifier(str(project), "project_id") if project else None
        )
        normalized_snapshot = (
            _validate_identifier(str(snapshot), "snapshot_id") if snapshot else None
        )
        now = _utc_now()
        with self._lock, self._connect() as connection:
            if normalized_snapshot is not None:
                snapshot_row = connection.execute(
                    "SELECT project_id FROM snapshots WHERE snapshot_id = ?",
                    (normalized_snapshot,),
                ).fetchone()
                if snapshot_row is None:
                    raise StateStoreError(
                        f"task {identifier} refers to missing snapshot: "
                        f"{normalized_snapshot}"
                    )
                snapshot_project = str(snapshot_row["project_id"])
                if normalized_project is None:
                    normalized_project = snapshot_project
                elif normalized_project != snapshot_project:
                    raise StateStoreError(
                        f"task {identifier} project {normalized_project} does not own "
                        f"snapshot {normalized_snapshot} (project {snapshot_project})"
                    )
            try:
                connection.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        normalized_project,
                        normalized_snapshot,
                        kind,
                        state,
                        _json_dump(_payload_without(raw, _TASK_RESERVED)),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateStoreError(f"cannot create task: {identifier}") from exc
        return self.get_task(identifier)

    def bind_task_lineage(
        self,
        task_id: str,
        *,
        project_id: str,
        snapshot_id: str,
    ) -> dict[str, Any]:
        """Bind an existing task to one sealed snapshot exactly once.

        Data preparation cannot know its immutable snapshot payload until the
        inspection worker finishes.  This narrow operation fills the reserved
        foreign keys after sealing, while refusing any re-parenting.
        """

        task = _validate_identifier(task_id, "task_id")
        project = _validate_identifier(project_id, "project_id")
        snapshot = _validate_identifier(snapshot_id, "snapshot_id")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            snapshot_row = connection.execute(
                "SELECT project_id FROM snapshots WHERE snapshot_id = ?",
                (snapshot,),
            ).fetchone()
            if snapshot_row is None:
                connection.rollback()
                raise StateStoreError(f"cannot bind task to missing snapshot: {snapshot}")
            snapshot_project = str(snapshot_row["project_id"])
            if snapshot_project != project:
                connection.rollback()
                raise StateStoreError(
                    f"snapshot {snapshot} belongs to {snapshot_project}, not {project}"
                )
            task_row = connection.execute(
                "SELECT project_id, snapshot_id FROM tasks WHERE task_id = ?",
                (task,),
            ).fetchone()
            if task_row is None:
                connection.rollback()
                raise RecordNotFoundError(task)
            current_project = task_row["project_id"]
            current_snapshot = task_row["snapshot_id"]
            if current_project not in {None, project} or current_snapshot not in {
                None,
                snapshot,
            }:
                connection.rollback()
                raise StateStoreError(f"task {task} lineage is already bound")
            if current_project == project and current_snapshot == snapshot:
                connection.commit()
                return self.get_task(task)
            connection.execute(
                """UPDATE tasks SET project_id = ?, snapshot_id = ?, updated_at = ?
                   WHERE task_id = ?""",
                (project, snapshot, _utc_now(), task),
            )
            connection.commit()
        return self.get_task(task)

    def get_task(self, task_id: str) -> dict[str, Any]:
        identifier = _validate_identifier(task_id, "task_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(identifier)
        return self._row_task(row)

    def update_task(self, task_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        identifier = _validate_identifier(task_id, "task_id")
        changes = dict(updates)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RecordNotFoundError(identifier)
            status = str(changes.get("status", row["status"]))
            self._validate_task_status(status)
            task_type = _validate_identifier(
                str(changes.get("task_type", row["task_type"])), "task_type"
            )
            payload = _json_load(row["payload_json"])
            payload.update(_payload_without(changes, _TASK_RESERVED))
            connection.execute(
                """UPDATE tasks
                   SET task_type = ?, status = ?, payload_json = ?, updated_at = ?
                   WHERE task_id = ?""",
                (task_type, status, _json_dump(payload), _utc_now(), identifier),
            )
            connection.commit()
        return self.get_task(identifier)

    def replace_task(self, task_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
        """Replace a task record's flexible payload (used by the dict adapter)."""

        identifier = _validate_identifier(task_id, "task_id")
        raw = dict(value)
        if raw.get("task_id", identifier) != identifier:
            raise ValueError("task_id cannot be changed")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RecordNotFoundError(identifier)
            status = str(raw.get("status", row["status"]))
            self._validate_task_status(status)
            task_type = _validate_identifier(
                str(raw.get("task_type", row["task_type"])), "task_type"
            )
            connection.execute(
                """UPDATE tasks
                   SET task_type = ?, status = ?, payload_json = ?, updated_at = ?
                   WHERE task_id = ?""",
                (
                    task_type,
                    status,
                    _json_dump(_payload_without(raw, _TASK_RESERVED)),
                    _utc_now(),
                    identifier,
                ),
            )
            connection.commit()
        return self.get_task(identifier)

    def transition_task(
        self,
        task_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically change task state, optionally using compare-and-set."""

        identifier = _validate_identifier(task_id, "task_id")
        self._validate_task_status(status)
        if expected_status is not None:
            self._validate_task_status(expected_status)
        changes = dict(updates or {})
        changes["status"] = status
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RecordNotFoundError(identifier)
            if expected_status is not None and row["status"] != expected_status:
                connection.rollback()
                raise ConcurrentStateError(
                    f"task {identifier} is {row['status']}, expected {expected_status}"
                )
            payload = _json_load(row["payload_json"])
            payload.update(_payload_without(changes, _TASK_RESERVED))
            connection.execute(
                """UPDATE tasks SET status = ?, payload_json = ?, updated_at = ?
                   WHERE task_id = ?""",
                (status, _json_dump(payload), _utc_now(), identifier),
            )
            connection.commit()
        return self.get_task(identifier)

    def list_tasks(
        self,
        *,
        project_id: str | None = None,
        snapshot_id: str | None = None,
        status: str | None = None,
        task_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit, offset = _validate_limit(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("project_id", project_id),
            ("snapshot_id", snapshot_id),
            ("task_type", task_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(_validate_identifier(value, column))
        if status is not None:
            self._validate_task_status(status)
            clauses.append("status = ?")
            parameters.append(status)
        rows = self._list_rows("tasks", clauses, parameters, limit, offset)
        return [self._row_task(row) for row in rows]

    def recover_interrupted_tasks(self, *, as_status: str = "queued") -> list[dict[str, Any]]:
        """Move orphaned ``running`` tasks to a restart-safe state.

        The API may resubmit returned queued tasks, or choose ``failed`` and
        expose them for an explicit retry.  Recovery is never run implicitly.
        """

        if as_status not in {"queued", "failed", "cancelled"}:
            raise ValueError("interrupted tasks may recover as queued, failed, or cancelled")
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM tasks WHERE status = 'running'"
            ).fetchall()
            recovered_ids = [str(row["task_id"]) for row in rows]
            for row in rows:
                payload = _json_load(row["payload_json"])
                payload["recovery"] = {
                    "previous_status": "running",
                    "recovered_at": now,
                }
                connection.execute(
                    """UPDATE tasks
                       SET status = ?, payload_json = ?, updated_at = ?
                       WHERE task_id = ?""",
                    (as_status, _json_dump(payload), now, row["task_id"]),
                )
            connection.commit()
        if not recovered_ids:
            return []
        return [self.get_task(task_id) for task_id in recovered_ids]

    def create_artifact_bundle(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        bundle_id: str | None = None,
        project_id: str | None = None,
        snapshot_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        raw = dict(payload or {})
        identifier = _validate_identifier(
            bundle_id or str(raw.get("bundle_id") or uuid.uuid4().hex), "bundle_id"
        )
        project = project_id if project_id is not None else raw.get("project_id")
        snapshot = snapshot_id if snapshot_id is not None else raw.get("snapshot_id")
        task = task_id if task_id is not None else raw.get("task_id")
        now = _utc_now()
        values = (
            identifier,
            _validate_identifier(str(project), "project_id") if project else None,
            _validate_identifier(str(snapshot), "snapshot_id") if snapshot else None,
            _validate_identifier(str(task), "task_id") if task else None,
            _json_dump(_payload_without(raw, _BUNDLE_RESERVED)),
            now,
            now,
        )
        with self._lock, self._connect() as connection:
            try:
                connection.execute("INSERT INTO artifact_bundles VALUES (?, ?, ?, ?, ?, ?, ?)", values)
            except sqlite3.IntegrityError as exc:
                raise StateStoreError(f"cannot create artifact bundle: {identifier}") from exc
        return self.get_artifact_bundle(identifier)

    def get_artifact_bundle(self, bundle_id: str) -> dict[str, Any]:
        identifier = _validate_identifier(bundle_id, "bundle_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifact_bundles WHERE bundle_id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(identifier)
        return self._row_bundle(row)

    def update_artifact_bundle(
        self, bundle_id: str, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._update_json_record(
            table="artifact_bundles",
            id_column="bundle_id",
            identifier=bundle_id,
            updates=updates,
            reserved=_BUNDLE_RESERVED,
            reader=self.get_artifact_bundle,
        )

    def list_artifact_bundles(
        self,
        *,
        project_id: str | None = None,
        snapshot_id: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit, offset = _validate_limit(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("project_id", project_id),
            ("snapshot_id", snapshot_id),
            ("task_id", task_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(_validate_identifier(value, column))
        rows = self._list_rows("artifact_bundles", clauses, parameters, limit, offset)
        return [self._row_bundle(row) for row in rows]

    def task_mapping(self) -> PersistentTaskMapping:
        """Return a ``MutableMapping`` adapter for legacy ``_tasks`` call sites."""

        return PersistentTaskMapping(self)

    def _update_json_record(
        self,
        *,
        table: str,
        id_column: str,
        identifier: str,
        updates: Mapping[str, Any],
        reserved: frozenset[str],
        reader: Any,
    ) -> dict[str, Any]:
        normalized = _validate_identifier(identifier, id_column)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE {id_column} = ?", (normalized,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RecordNotFoundError(normalized)
            payload = _json_load(row["payload_json"])
            payload.update(_payload_without(dict(updates), reserved))
            connection.execute(
                f"UPDATE {table} SET payload_json = ?, updated_at = ? WHERE {id_column} = ?",
                (_json_dump(payload), _utc_now(), normalized),
            )
            connection.commit()
        return reader(normalized)

    def _list_rows(
        self,
        table: str,
        clauses: list[str],
        parameters: list[Any],
        limit: int | None,
        offset: int,
    ) -> list[sqlite3.Row]:
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY updated_at DESC, {table[:-1]}_id"
        # artifact_bundles is the only table whose id does not follow the
        # singular-table convention.
        if table == "artifact_bundles":
            sql = sql.rsplit("artifact_bundle_id", 1)[0] + "bundle_id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            parameters.append(offset)
        with self._lock, self._connect() as connection:
            return connection.execute(sql, tuple(parameters)).fetchall()

    @staticmethod
    def _validate_task_status(status: str) -> None:
        if status not in TASK_STATUSES:
            allowed = ", ".join(sorted(TASK_STATUSES))
            raise ValueError(f"invalid task status {status!r}; expected one of: {allowed}")


class _PersistentTaskRecord(dict[str, Any]):
    """A dict whose top-level mutations are flushed back to SQLite."""

    def __init__(self, store: SQLiteStateStore, value: Mapping[str, Any]) -> None:
        super().__init__(value)
        self._store = store
        self._task_id = str(value["task_id"])

    def _flush(self) -> None:
        persisted = self._store.replace_task(self._task_id, self)
        dict.clear(self)
        dict.update(self, persisted)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "task_id" and value != self._task_id:
            raise ValueError("task_id cannot be changed")
        dict.__setitem__(self, key, value)
        self._flush()

    def __delitem__(self, key: str) -> None:
        if key in _TASK_RESERVED:
            raise KeyError(f"cannot delete task control field: {key}")
        dict.__delitem__(self, key)
        self._flush()

    def update(self, *args: Any, **kwargs: Any) -> None:
        changes = dict(*args, **kwargs)
        if "task_id" in changes and changes["task_id"] != self._task_id:
            raise ValueError("task_id cannot be changed")
        dict.update(self, changes)
        self._flush()

    def pop(self, key: str, *default: Any) -> Any:
        if key in _TASK_RESERVED:
            raise KeyError(f"cannot delete task control field: {key}")
        value = dict.pop(self, key, *default)
        self._flush()
        return value

    def popitem(self) -> tuple[str, Any]:
        removable = [key for key in self if key not in _TASK_RESERVED]
        if not removable:
            raise KeyError("task has no removable payload fields")
        key = removable[-1]
        value = dict.pop(self, key)
        self._flush()
        return key, value

    def clear(self) -> None:
        for key in tuple(self):
            if key not in _TASK_RESERVED:
                dict.__delitem__(self, key)
        self._flush()

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default


class PersistentTaskMapping(MutableMapping[str, dict[str, Any]]):
    """Compatibility adapter for the API's historical in-memory task dict.

    Existing call sites such as ``tasks[id] = {...}``, ``tasks.get(id)`` and
    ``tasks[id].update(...)`` continue to work while records become durable.
    Deletion is intentionally unsupported so provenance is not lost silently.
    """

    def __init__(self, store: SQLiteStateStore) -> None:
        self.store = store

    def __getitem__(self, task_id: str) -> dict[str, Any]:
        try:
            return _PersistentTaskRecord(self.store, self.store.get_task(task_id))
        except RecordNotFoundError as exc:
            raise KeyError(task_id) from exc

    def __setitem__(self, task_id: str, value: dict[str, Any]) -> None:
        try:
            self.store.get_task(task_id)
        except RecordNotFoundError:
            self.store.create_task(value, task_id=task_id)
        else:
            self.store.replace_task(task_id, value)

    def __delitem__(self, task_id: str) -> None:
        raise TypeError("persistent task records cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(task["task_id"] for task in self.store.list_tasks())

    def __len__(self) -> int:
        with self.store._lock, self.store._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()
        return int(row["count"] if row is not None else 0)


__all__ = [
    "TASK_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "ConcurrentStateError",
    "PersistentTaskMapping",
    "RecordNotFoundError",
    "SQLiteStateStore",
    "StateStoreError",
    "default_state_db_path",
]
