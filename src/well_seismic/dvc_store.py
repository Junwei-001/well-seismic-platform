"""Owner-scoped immutable versions backed by real local DVC objects.

This module is a backend adapter, not a generic file or shell API. Source paths
must be supplied by trusted application code and lie under configured roots.
DVC runs without SCM in private runtime directories; it never stages Git files.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from configparser import ConfigParser
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .content_identity import file_sha256

SCHEMA = "well-seismic.dvc-version.v1"
RESTORE_SCHEMA = "well-seismic.dvc-restore.v1"
_REVISION = re.compile(r"dv_[0-9a-f]{32}\Z")
_RESTORE = re.compile(r"dr_[0-9a-f]{32}\Z")
_PROJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ARCHIVE_KEY = re.compile(r"[0-9a-f]{64}\Z")
_SECRET = re.compile(
    r"password|secret|credential|authorization|api[_-]?key|access[_-]?token",
    re.IGNORECASE,
)
_DRIVE = re.compile(r"(?:^|\s)[A-Za-z]:[\\/]")


class DvcStoreError(RuntimeError):
    """A safe, stable failure code suitable for a transport boundary."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fail(code: str, message: str) -> None:
    raise DvcStoreError(code, message)


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise DvcStoreError(
            "invalid_metadata", "Metadata must be finite JSON."
        ) from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_links(path: Path) -> Path:
    """Check every existing component before resolving Windows junctions."""
    absolute = Path(os.path.abspath(path))
    for component in [*reversed(absolute.parents), absolute]:
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            _fail("unsafe_path", "Symbolic links and reparse points are not allowed.")
    return absolute


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@contextmanager
def archive_key_lock(directory: Path, key: str, *, timeout: float = 180):
    """Coordinate one content key across processes without holding a DB lock."""
    if not isinstance(key, str) or not _ARCHIVE_KEY.fullmatch(key):
        _fail("invalid_archive_key", "Archive identity is invalid.")
    directory = _no_links(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _no_links(directory / (key + ".lock"))
    descriptor = os.open(
        path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _fail("unsafe_path", "Archive coordination requires a regular lock file.")
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise DvcStoreError(
                        "archive_lock_failed", "Archive coordination failed."
                    ) from exc
                if time.monotonic() >= deadline:
                    raise DvcStoreError(
                        "archive_in_progress",
                        "This archive is still being prepared; retry after it completes.",
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            if locked:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _logical_name(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240 or "\\" in value:
        _fail("invalid_name", "Artifact names must be bounded relative paths.")
    components = value.split("/")
    if any(
        not part
        or part in {".", ".."}
        or part.startswith(".")
        or ":" in part
        or any(ord(c) < 32 for c in part)
        or part.endswith((" ", "."))
        for part in components
    ):
        _fail("invalid_name", "Artifact names must be safe relative paths.")
    if PurePosixPath(value).is_absolute():
        _fail("invalid_name", "Absolute artifact paths are not allowed.")
    for part in components:
        if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part):
            _fail("invalid_name", "Reserved artifact names are not allowed.")
    return value


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_metadata", "Metadata must be a JSON object.")
    result = json.loads(_json_bytes(dict(value)))
    if len(_json_bytes(result)) > 65536:
        _fail("invalid_metadata", "Metadata exceeds the size limit.")

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8:
            _fail("invalid_metadata", "Metadata exceeds the nesting limit.")
        if isinstance(item, dict):
            for key, child in item.items():
                if _SECRET.search(key):
                    _fail(
                        "invalid_metadata",
                        "Credentials cannot be stored in version metadata.",
                    )
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, str) and (
            item.startswith(("/", "\\")) or _DRIVE.search(item)
        ):
            _fail(
                "invalid_metadata", "Local paths cannot be stored in public metadata."
            )

    visit(result)
    return result


class DvcVersionStore:
    """Store one immutable DVC workspace per authenticated owner and version."""

    def __init__(
        self,
        root: Path,
        *,
        allowed_source_roots: Iterable[Path] | None = None,
        command: Sequence[str] | None = None,
        max_files: int = 128,
        max_total_bytes: int = 2 * 1024**3,
        timeout: float = 180,
    ) -> None:
        self.root = _no_links(Path(root))
        self.source_roots = tuple(
            _no_links(Path(path))
            for path in (
                allowed_source_roots
                if allowed_source_roots is not None
                else [self.root]
            )
        )
        if not self.source_roots:
            _fail(
                "invalid_configuration", "At least one trusted source root is required."
            )
        self.command = tuple(command or (sys.executable, "-m", "dvc"))
        if not self.command or not all(isinstance(x, str) and x for x in self.command):
            _fail("invalid_configuration", "A trusted DVC command is required.")
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.timeout = timeout

    def _run(self, args: Sequence[str], cwd: Path | None = None) -> str:
        environment = dict(os.environ)
        environment.update(DVC_NO_ANALYTICS="1", PYTHONDONTWRITEBYTECODE="1")
        try:
            result = subprocess.run(
                [*self.command, *args],
                cwd=cwd,
                env=environment,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DvcStoreError(
                "dvc_unavailable", "DVC could not be executed."
            ) from exc
        if result.returncode:
            # DVC diagnostics can contain local paths or credential-bearing URLs.
            _fail("dvc_failed", "The DVC operation failed; no version was published.")
        return result.stdout

    def status(self) -> dict[str, Any]:
        try:
            version = self._run(["--version"]).strip()
            if not re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)", version):
                _fail(
                    "dvc_unavailable", "DVC returned an unsupported version response."
                )
            return {
                "available": True,
                "version": version,
                "mode": "no_scm",
                "git_revision_history": False,
            }
        except DvcStoreError:
            return {
                "available": False,
                "version": None,
                "mode": "no_scm",
                "git_revision_history": False,
                "error": "dvc_unavailable",
            }

    def available(self) -> bool:
        return bool(self.status()["available"])

    def _require(self) -> str:
        status = self.status()
        if not status["available"]:
            _fail("dvc_unavailable", "DVC is not installed or cannot be executed.")
        return str(status["version"])

    def _scope(
        self, subject: str, project_id: str, *, create: bool = False
    ) -> tuple[Path, str]:
        if not isinstance(subject, str) or not subject or len(subject) > 256:
            _fail("invalid_owner", "An authenticated owner is required.")
        if not isinstance(project_id, str) or not _PROJECT.fullmatch(project_id):
            _fail("invalid_project", "Invalid project identifier.")
        owner = _sha(subject.encode("utf-8"))
        # A full SHA-256 of owner + project keeps private paths short on Windows.
        scope = _sha(_json_bytes([owner, project_id]))
        path = _no_links(self.root / "s" / scope)
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            _no_links(path)
        return path, owner

    @staticmethod
    def _initialize(workspace: Path, runner: Any, *, cache: Path | None = None) -> None:
        workspace.mkdir(mode=0o700, parents=True, exist_ok=False)
        runner(["init", "--no-scm", "--quiet"], workspace)
        config = ConfigParser()
        config["core"] = {
            "no_scm": "true",
            "analytics": "false",
            "check_update": "false",
        }
        config["cache"] = {"type": "copy"}
        if cache is not None:
            config["cache"]["dir"] = os.path.relpath(cache, workspace / ".dvc").replace(
                "\\", "/"
            )
        with (workspace / ".dvc" / "config").open("w", encoding="utf-8") as stream:
            config.write(stream)

    def _source(self, source: Path) -> Path:
        path = _no_links(source)
        if not any(_under(path, root) for root in self.source_roots):
            _fail("unsafe_path", "Artifact source is outside the trusted source roots.")
        if not path.is_file():
            _fail("invalid_source", "Artifact source must be a regular file.")
        return path

    def archive(
        self,
        subject: str,
        project_id: str,
        files: Mapping[str, Path],
        metadata: Mapping[str, Any],
        parent_revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Archive trusted server-side files; request paths must never reach here."""
        if not isinstance(files, Mapping) or not 1 <= len(files) <= self.max_files:
            _fail("input_limit", "Artifact count is outside the allowed range.")
        safe_metadata = _metadata(metadata)
        prepared = [
            (_logical_name(name), self._source(Path(source)))
            for name, source in files.items()
        ]
        names = [name.casefold() for name, _ in prepared]
        if len(set(names)) != len(names) or any(
            other.startswith(name + "/")
            for name in names
            for other in names
            if name != other
        ):
            _fail("invalid_name", "Artifact paths conflict.")
        if sum(path.stat().st_size for _, path in prepared) > self.max_total_bytes:
            _fail("input_limit", "Artifact data exceeds the size limit.")
        if parent_revision_id is not None:
            self.get(subject, project_id, parent_revision_id)
        archive_key = safe_metadata.get("archive_key")
        if archive_key is not None:
            existing = self.find_archive(subject, project_id, archive_key)
            if existing is not None:
                self._check_archive_request(
                    existing, prepared, safe_metadata, parent_revision_id
                )
                return existing
        version = self._require()
        project, owner = self._scope(subject, project_id, create=True)
        revision = "dv_" + (
            archive_key[:32] if archive_key is not None else uuid.uuid4().hex
        )
        versions = _no_links(project / "v")
        versions.mkdir(mode=0o700, exist_ok=True)
        # An interrupted attempt must not permanently occupy the content key.
        # Only the final immutable directory is the published version identity.
        pending_name = ".p-" + uuid.uuid4().hex[:24]
        pending = _no_links(versions / pending_name)
        pending.mkdir(mode=0o700, exist_ok=False)
        workspace = pending / "w"
        if (
            os.name == "nt"
            and max(
                # DVC atomically writes a directory cache via a 22-character
                # random suffix, not just the final 30-character MD5 + .dir.
                len(
                    str(
                        workspace
                        / ".dvc/cache/files/md5/00"
                        / ("0" * 30 + ".dir." + "0" * 22 + ".tmp")
                    )
                ),
                *(len(str(workspace / "data" / name)) for name, _ in prepared),
            )
            > 250
        ):
            _fail(
                "path_limit",
                "Use a shorter private DVC root or shorter artifact names on Windows.",
            )
        self._initialize(workspace, self._run)
        entries = []
        total = 0
        for name, source in sorted(prepared):
            source = self._source(source)
            target = workspace / "data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            # Copy into a fresh destination: source data and old versions stay intact.
            descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
            )
            with os.fdopen(descriptor, "rb") as incoming, target.open("xb") as outgoing:
                opened = os.fstat(incoming.fileno())
                if not stat.S_ISREG(opened.st_mode) or (
                    before.st_dev,
                    before.st_ino,
                ) != (opened.st_dev, opened.st_ino):
                    _fail(
                        "unsafe_path", "Artifact source changed identity while opening."
                    )
                while block := incoming.read(1024 * 1024):
                    total += len(block)
                    if total > self.max_total_bytes:
                        _fail("input_limit", "Artifact data exceeds the size limit.")
                    outgoing.write(block)
            after = self._source(source).stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ) or file_sha256(source) != file_sha256(target):
                _fail("source_changed", "Artifact source changed during archival.")
            entries.append(
                {
                    "path": name,
                    "size": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
        self._run(["add", "data", "--quiet"], workspace)
        try:
            changes = json.loads(self._run(["status", "--json"], workspace))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DvcStoreError(
                "dvc_failed", "DVC status could not be verified."
            ) from exc
        if changes:
            _fail("integrity_error", "DVC reported unarchived changes.")
        pointer = workspace / "data.dvc"
        if not pointer.is_file() or not (workspace / ".dvc" / "cache").is_dir():
            _fail(
                "integrity_error", "DVC did not produce the required pointer and cache."
            )
        record = {
            "schema_version": SCHEMA,
            "revision_id": revision,
            "parent_revision_id": parent_revision_id,
            "project_id": project_id,
            "owner_scope": owner,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "archived",
            "files": entries,
            "metadata": safe_metadata,
            "dvc": {
                "version": version,
                "mode": "no_scm",
                "git_revision_history": False,
                "pointer_sha256": file_sha256(pointer),
                "config_sha256": file_sha256(workspace / ".dvc" / "config"),
            },
        }
        record["receipt_sha256"] = _sha(_json_bytes(record))
        (pending / "receipt.json").write_bytes(_json_bytes(record))
        try:
            pending.rename(versions / revision)
        except OSError:
            if archive_key is None or not (versions / revision).exists():
                raise
            # A caller outside the Harness coordinator may race publication.
            # Reuse only an identical, fully verified immutable winner.
            existing = self.find_archive(subject, project_id, archive_key)
            if existing is None:
                raise
            self._check_archive_request(
                existing, prepared, safe_metadata, parent_revision_id
            )
            return existing
        return self.get(subject, project_id, revision)

    @staticmethod
    def _check_archive_request(record, prepared, metadata, parent_revision_id):
        expected = {name: file_sha256(path) for name, path in prepared}
        actual = {item["path"]: item["sha256"] for item in record["files"]}
        if (
            record.get("metadata") != metadata
            or record.get("parent_revision_id") != parent_revision_id
            or actual != expected
        ):
            _fail(
                "archive_identity_conflict",
                "Archive identity does not match this request.",
            )

    def find_archive(self, subject: str, project_id: str, archive_key: str):
        """Resolve one publication key; unrelated historical versions are not read."""
        if not isinstance(archive_key, str) or not _ARCHIVE_KEY.fullmatch(archive_key):
            _fail("invalid_archive_key", "Archive identity is invalid.")
        revision = "dv_" + archive_key[:32]
        project, _ = self._scope(subject, project_id)
        directory = _no_links(project / "v" / revision)
        if not directory.exists():
            return None
        record = self.get(subject, project_id, revision)
        if (record.get("metadata") or {}).get("archive_key") != archive_key:
            _fail(
                "archive_identity_conflict",
                "Archive identity does not match this version.",
            )
        return record

    def _load(
        self, subject: str, project_id: str, revision_id: str
    ) -> tuple[Path, dict[str, Any]]:
        if not isinstance(revision_id, str) or not _REVISION.fullmatch(revision_id):
            _fail("version_not_found", "Version was not found.")
        project, owner = self._scope(subject, project_id)
        directory = _no_links(project / "v" / revision_id)
        path = _no_links(directory / "receipt.json")
        try:
            if path.stat().st_size > 256 * 1024:
                _fail("integrity_error", "Version receipt exceeds the size limit.")
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DvcStoreError("version_not_found", "Version was not found.") from exc
        except (OSError, ValueError) as exc:
            raise DvcStoreError(
                "integrity_error", "Version receipt is invalid."
            ) from exc
        if not isinstance(record, dict):
            _fail("integrity_error", "Version receipt is invalid.")
        digest = record.get("receipt_sha256")
        unsigned = {k: v for k, v in record.items() if k != "receipt_sha256"}
        if (
            digest != _sha(_json_bytes(unsigned))
            or record.get("schema_version") != SCHEMA
            or (
                record.get("owner_scope"),
                record.get("project_id"),
                record.get("revision_id"),
            )
            != (owner, project_id, revision_id)
        ):
            _fail("integrity_error", "Version receipt integrity check failed.")
        workspace = directory / "w"
        if not isinstance(record.get("dvc"), dict):
            _fail("integrity_error", "DVC metadata is invalid.")
        for relative, key in [
            ("data.dvc", "pointer_sha256"),
            (".dvc/config", "config_sha256"),
        ]:
            item = _no_links(workspace / relative)
            if not item.is_file() or file_sha256(item) != record.get("dvc", {}).get(
                key
            ):
                _fail("integrity_error", "DVC metadata integrity check failed.")
        self._verify_files(workspace / "data", record.get("files"))
        return directory, record

    @staticmethod
    def _verify_files(root: Path, entries: Any) -> None:
        if not isinstance(entries, list) or not entries:
            _fail("integrity_error", "Version has no artifact manifest.")
        expected = set()
        for entry in entries:
            if not isinstance(entry, dict):
                _fail("integrity_error", "Artifact manifest is invalid.")
            name = _logical_name(entry.get("path"))
            if name.casefold() in expected:
                _fail("integrity_error", "Artifact manifest contains duplicate paths.")
            expected.add(name.casefold())
            path = _no_links(root / name)
            if (
                not path.is_file()
                or path.stat().st_size != entry.get("size")
                or file_sha256(path) != entry.get("sha256")
            ):
                _fail("integrity_error", "Version artifact integrity check failed.")
        actual = set()
        for item in root.rglob("*"):
            _no_links(item)
            if item.is_file():
                actual.add(item.relative_to(root).as_posix().casefold())
        if actual != expected:
            _fail("integrity_error", "Version contains unlisted artifact files.")

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "owner_scope"}

    def get(self, subject: str, project_id: str, revision_id: str) -> dict[str, Any]:
        return self._public(self._load(subject, project_id, revision_id)[1])

    def list_versions(self, subject: str, project_id: str) -> list[dict[str, Any]]:
        project, _ = self._scope(subject, project_id)
        directory = _no_links(project / "v")
        if not directory.exists():
            return []
        return sorted(
            [
                self.get(subject, project_id, p.name)
                for p in directory.iterdir()
                if _REVISION.fullmatch(p.name)
            ],
            key=lambda item: (item["created_at"], item["revision_id"]),
        )

    def compare(
        self, subject: str, project_id: str, from_revision_id: str, to_revision_id: str
    ) -> dict[str, Any]:
        first = self.get(subject, project_id, from_revision_id)
        second = self.get(subject, project_id, to_revision_id)
        old = {entry["path"]: entry for entry in first["files"]}
        new = {entry["path"]: entry for entry in second["files"]}
        return {
            "from_revision_id": from_revision_id,
            "to_revision_id": to_revision_id,
            "added": sorted(new.keys() - old.keys()),
            "removed": sorted(old.keys() - new.keys()),
            "changed": sorted(
                name for name in old.keys() & new.keys() if old[name] != new[name]
            ),
            "metadata_changed": first["metadata"] != second["metadata"],
        }

    def restore_owned(
        self, subject: str, project_id: str, revision_id: str
    ) -> dict[str, Any]:
        """Materialize a historical version into a new directory, never in place."""
        self._require()
        directory, record = self._load(subject, project_id, revision_id)
        project, _ = self._scope(subject, project_id)
        cache = _no_links(directory / "w" / ".dvc" / "cache")
        if not cache.is_dir():
            _fail("cache_missing", "The DVC cache for this version is missing.")
        for item in cache.rglob("*"):
            _no_links(item)
        restore_id = "dr_" + uuid.uuid4().hex
        restored = _no_links(project / "r" / restore_id)
        workspace = restored / "w"
        self._initialize(workspace, self._run, cache=cache)
        shutil.copyfile(directory / "w" / "data.dvc", workspace / "data.dvc")
        self._run(["checkout", "data.dvc", "--quiet"], workspace)
        self._verify_files(workspace / "data", record["files"])
        receipt = {
            "schema_version": RESTORE_SCHEMA,
            "restore_id": restore_id,
            "revision_id": revision_id,
            "project_id": project_id,
            "status": "restored",
            "verified": True,
            "files": record["files"],
            "created_at": datetime.now(UTC).isoformat(),
        }
        receipt["receipt_sha256"] = _sha(_json_bytes(receipt))
        (restored / "receipt.json").write_bytes(_json_bytes(receipt))
        return receipt

    def resolve_files(
        self, subject: str, project_id: str, revision_id: str
    ) -> dict[str, Path]:
        """Backend-only resolver: never serialize these paths in an API response."""
        directory, record = self._load(subject, project_id, revision_id)
        return {
            entry["path"]: directory / "w" / "data" / entry["path"]
            for entry in record["files"]
        }

    def resolve_restored_files(
        self, subject: str, project_id: str, restore_id: str
    ) -> dict[str, Path]:
        """Backend-only input resolver for a new execution from restored data."""
        if not isinstance(restore_id, str) or not _RESTORE.fullmatch(restore_id):
            _fail("version_not_found", "Restore was not found.")
        project, _ = self._scope(subject, project_id)
        restored = _no_links(project / "r" / restore_id)
        try:
            receipt_path = _no_links(restored / "receipt.json")
            if receipt_path.stat().st_size > 256 * 1024:
                _fail("integrity_error", "Restore receipt exceeds the size limit.")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DvcStoreError("version_not_found", "Restore was not found.") from exc
        except (OSError, ValueError) as exc:
            raise DvcStoreError(
                "integrity_error", "Restore receipt is invalid."
            ) from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != RESTORE_SCHEMA
            or receipt.get("status") != "restored"
            or receipt.get("verified") is not True
        ):
            _fail("integrity_error", "Restore receipt is invalid.")
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != _sha(_json_bytes(unsigned))
            or receipt.get("restore_id") != restore_id
            or receipt.get("project_id") != project_id
        ):
            _fail("integrity_error", "Restore receipt integrity check failed.")
        original = self.get(subject, project_id, receipt.get("revision_id"))
        if receipt.get("files") != original["files"]:
            _fail(
                "integrity_error", "Restore manifest differs from its source version."
            )
        self._verify_files(restored / "w" / "data", receipt["files"])
        return {
            entry["path"]: restored / "w" / "data" / entry["path"]
            for entry in receipt["files"]
        }
