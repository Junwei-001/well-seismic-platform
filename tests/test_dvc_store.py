"""DVC service boundary and real-cache regression tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import uuid
from pathlib import Path

import pytest

from well_seismic.dvc_store import DvcStoreError, DvcVersionStore

OWNER = "a" * 64
OTHER = "b" * 64
PROJECT = "test-project"
HAS_DVC = importlib.util.find_spec("dvc") is not None
real_dvc = pytest.mark.skipif(
    not HAS_DVC, reason="Real DVC is not installed in this interpreter"
)


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    return DvcVersionStore(
        tmp_path.parent / ("dv_" + uuid.uuid4().hex[:8]), allowed_source_roots=[source]
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "sources" / "input.las"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"~Version\nVERS. 2.0\n~ASCII\n100 2.4\n")
    return path


def assert_code(code, call):
    with pytest.raises(DvcStoreError) as caught:
        call()
    assert caught.value.code == code
    return str(caught.value)


def test_missing_dvc_is_explicit_and_never_claims_an_archive(tmp_path, source):
    missing = DvcVersionStore(
        tmp_path / "store",
        allowed_source_roots=[source.parent],
        command=[str(tmp_path / "nonexistent-executable")],
    )
    assert missing.status()["available"] is False
    assert missing.status()["error"] == "dvc_unavailable"
    assert_code(
        "dvc_unavailable",
        lambda: missing.archive(OWNER, PROJECT, {"input.las": source}, {}),
    )
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../input",
        "/input",
        "a/../b",
        "a//b",
        "a\\b",
        "C:/input",
        ".dvc/config",
        "data/A:stream",
        "CON",
        "a.",
    ],
)
def test_unsafe_artifact_names_are_rejected_before_execution(store, source, name):
    assert_code(
        "invalid_name", lambda: store.archive(OWNER, PROJECT, {name: source}, {})
    )


def test_outside_source_root_and_path_leak_are_rejected(store, tmp_path):
    outsider = tmp_path / "outside.las"
    outsider.write_text("outside")
    message = assert_code(
        "unsafe_path", lambda: store.archive(OWNER, PROJECT, {"x": outsider}, {})
    )
    assert str(tmp_path) not in message


def test_limits_and_case_collisions_are_rejected_before_execution(store, source):
    store.max_total_bytes = 1
    assert_code(
        "input_limit", lambda: store.archive(OWNER, PROJECT, {"input": source}, {})
    )
    assert_code(
        "invalid_name",
        lambda: store.archive(OWNER, PROJECT, {"a": source, "A": source}, {}),
    )
    assert_code(
        "invalid_name",
        lambda: store.archive(OWNER, PROJECT, {"a": source, "a/b": source}, {}),
    )
    assert_code("input_limit", lambda: store.archive(OWNER, PROJECT, {}, {}))


@pytest.mark.parametrize(
    "metadata",
    [
        {"password": "example"},
        {"nested": {"api_key": "example"}},
        {"source": "C:\\private\\input.las"},
        {"source": "/private/input.las"},
        {"metric": float("nan")},
        {"long": "x" * 65536},
    ],
)
def test_private_or_invalid_metadata_is_not_archived(store, source, metadata):
    assert_code(
        "invalid_metadata",
        lambda: store.archive(OWNER, PROJECT, {"input": source}, metadata),
    )


def test_symlink_source_is_rejected(store, source):
    link = source.with_name("linked.las")
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("OS does not permit creating symlinks for this test account")
    assert_code(
        "unsafe_path", lambda: store.archive(OWNER, PROJECT, {"input": link}, {})
    )


def test_windows_reparse_attribute_is_rejected(store, source, monkeypatch):
    original = Path.lstat

    class ReparseInfo:
        st_mode = stat.S_IFREG
        st_file_attributes = 0x400

    def replacement(path):
        return ReparseInfo() if path == source else original(path)

    monkeypatch.setattr(Path, "lstat", replacement)
    assert_code(
        "unsafe_path", lambda: store.archive(OWNER, PROJECT, {"input": source}, {})
    )


def test_other_owner_has_no_listing_and_no_access(store):
    assert store.list_versions(OTHER, PROJECT) == []
    assert_code(
        "version_not_found", lambda: store.get(OTHER, PROJECT, "dv_" + "1" * 32)
    )
    assert_code("invalid_project", lambda: store.list_versions(OWNER, "../project"))
    assert_code("invalid_owner", lambda: store.list_versions("", PROJECT))


@real_dvc
def test_real_dvc_archive_compare_restore_and_owner_isolation(store, source):
    first_bytes = source.read_bytes()
    first = store.archive(
        OWNER, PROJECT, {"inputs/input.las": source}, {"task_id": "task-a"}
    )
    assert first["status"] == "archived"
    assert first["dvc"]["mode"] == "no_scm"
    assert first["files"][0]["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert str(source.parent.parent) not in json.dumps(first)
    assert "owner_scope" not in first
    archived_files = store.resolve_files(OWNER, PROJECT, first["revision_id"])
    assert archived_files["inputs/input.las"].read_bytes() == first_bytes
    workspace = archived_files["inputs/input.las"].parents[2]
    assert (workspace / "data.dvc").is_file()
    assert (workspace / ".dvc" / "cache").is_dir()
    assert not list(store.root.rglob(".git"))

    source.write_bytes(first_bytes + b"101 2.5\n")
    second = store.archive(
        OWNER,
        PROJECT,
        {"inputs/input.las": source},
        {"task_id": "task-b"},
        first["revision_id"],
    )
    assert first["revision_id"] != second["revision_id"]
    assert second["parent_revision_id"] == first["revision_id"]
    difference = store.compare(
        OWNER, PROJECT, first["revision_id"], second["revision_id"]
    )
    assert difference["changed"] == ["inputs/input.las"]
    assert difference["metadata_changed"] is True
    assert len(store.list_versions(OWNER, PROJECT)) == 2
    assert store.list_versions(OTHER, PROJECT) == []
    assert_code(
        "version_not_found", lambda: store.get(OTHER, PROJECT, first["revision_id"])
    )
    assert_code(
        "version_not_found",
        lambda: store.restore_owned(OTHER, PROJECT, first["revision_id"]),
    )

    restored = store.restore_owned(OWNER, PROJECT, first["revision_id"])
    assert restored["verified"] is True
    paths = store.resolve_restored_files(OWNER, PROJECT, restored["restore_id"])
    assert paths["inputs/input.las"].read_bytes() == first_bytes
    assert paths["inputs/input.las"] != archived_files["inputs/input.las"]
    assert source.read_bytes() != first_bytes  # Never overwrite the current input.
    assert store.get(OWNER, PROJECT, first["revision_id"]) == first
    assert_code(
        "version_not_found",
        lambda: store.resolve_restored_files(OTHER, PROJECT, restored["restore_id"]),
    )


@real_dvc
@pytest.mark.parametrize(
    "target", ["artifact", "pointer", "receipt", "config", "extra_file"]
)
def test_sealed_version_tampering_is_detected(store, source, target):
    version = store.archive(OWNER, PROJECT, {"input": source}, {})
    artifact = store.resolve_files(OWNER, PROJECT, version["revision_id"])["input"]
    workspace = artifact.parents[1]
    path = {
        "artifact": artifact,
        "pointer": workspace / "data.dvc",
        "receipt": workspace.parent / "receipt.json",
        "config": workspace / ".dvc" / "config",
        "extra_file": workspace / "data" / "unexpected",
    }[target]
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"tampered")
    assert_code(
        "integrity_error", lambda: store.get(OWNER, PROJECT, version["revision_id"])
    )


@real_dvc
def test_missing_cache_cannot_be_claimed_restored(store, source):
    version = store.archive(OWNER, PROJECT, {"input": source}, {})
    artifact = store.resolve_files(OWNER, PROJECT, version["revision_id"])["input"]
    cache = artifact.parents[1] / ".dvc" / "cache"
    cache.rename(cache.with_name("cache-moved-for-negative-test"))
    assert_code(
        "cache_missing",
        lambda: store.restore_owned(OWNER, PROJECT, version["revision_id"]),
    )


@real_dvc
def test_corrupted_cache_object_fails_restore_hash_verification(store, source):
    version = store.archive(OWNER, PROJECT, {"input": source}, {})
    artifact = store.resolve_files(OWNER, PROJECT, version["revision_id"])["input"]
    cache = artifact.parents[1] / ".dvc" / "cache"
    objects = [
        path
        for path in cache.rglob("*")
        if path.is_file() and path.read_bytes() == source.read_bytes()
    ]
    assert len(objects) == 1
    objects[0].chmod(stat.S_IREAD | stat.S_IWRITE)
    objects[0].write_bytes(b"corrupt")
    with pytest.raises(DvcStoreError) as caught:
        store.restore_owned(OWNER, PROJECT, version["revision_id"])
    assert caught.value.code in {"dvc_failed", "integrity_error"}
    project = artifact.parents[4]
    assert not list((project / "r").rglob("receipt.json"))


@real_dvc
def test_failed_dvc_run_never_publishes_version(store, source, monkeypatch):
    original = store._run

    def fail_add(args, cwd=None):
        if args[0] == "add":
            raise DvcStoreError("dvc_failed", "DVC operation failed.")
        return original(args, cwd)

    monkeypatch.setattr(store, "_run", fail_add)
    assert_code(
        "dvc_failed", lambda: store.archive(OWNER, PROJECT, {"input": source}, {})
    )
    assert store.list_versions(OWNER, PROJECT) == []


@real_dvc
def test_corrupted_restore_receipt_is_rejected(store, source):
    version = store.archive(OWNER, PROJECT, {"input": source}, {})
    restored = store.restore_owned(OWNER, PROJECT, version["revision_id"])
    restored_file = store.resolve_restored_files(
        OWNER, PROJECT, restored["restore_id"]
    )["input"]
    receipt_path = restored_file.parents[2] / "receipt.json"
    receipt_path.write_text("[]", encoding="utf-8")
    assert_code(
        "integrity_error",
        lambda: store.resolve_restored_files(OWNER, PROJECT, restored["restore_id"]),
    )
