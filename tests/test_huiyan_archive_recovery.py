"""Review regressions for retriable uploads and complete native result archives."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from well_seismic.dvc_store import DvcStoreError, DvcVersionStore
from well_seismic.huiyan_harness import HuiyanHarness
from test_huiyan_harness import OWNER, OTHER, ROOT, harness, project_asset


class RecordingStore:
    def __init__(self):
        self.fail = False
        self.attempts = 0
        self.versions = []

    def archive(self, subject, project, files, metadata, parent_revision_id=None):
        self.attempts += 1
        if self.fail:
            raise DvcStoreError("dvc_unavailable", "synthetic unavailable")
        receipt = {
            "revision_id": "dv_" + uuid.uuid4().hex,
            "project_id": project,
            "status": "archived",
            "metadata": metadata,
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                }
                for name, path in sorted(files.items())
            ],
        }
        self.versions.append((subject, project, receipt))
        return receipt

    def list_versions(self, subject, project):
        return [
            value
            for owner, pid, value in self.versions
            if (owner, pid) == (subject, project)
        ]

    def get(self, subject, project, revision):
        return next(
            v
            for v in self.list_versions(subject, project)
            if v["revision_id"] == revision
        )

    def find_archive(self, subject, project, key):
        return next(
            (
                value
                for owner, pid, value in self.versions
                if (owner, pid) == (subject, project)
                and value["metadata"].get("archive_key") == key
            ),
            None,
        )


def test_failed_upload_retries_same_asset_after_restart_and_only_once(harness):
    store = harness.dvc = RecordingStore()
    pid = harness.execute(OWNER, "project_create", {"name": "retry in same project"})[
        "project_id"
    ]
    other_project = harness.execute(
        OWNER, "project_create", {"name": "unrelated project"}
    )["project_id"]
    content = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    store.fail = True
    first = harness.upload(OWNER, pid, "log", "same.las", content)
    assert first["versioning"]["status"] == "blocked"
    original = harness.asset(OWNER, pid, first["asset"]["id"])
    original_bytes = Path(original["path"]).read_bytes()
    store.fail = False
    restarted = HuiyanHarness(
        harness.root, config=harness.config, native=harness.native, dvc_store=store
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        replies = list(
            pool.map(
                lambda _: restarted.upload(OWNER, pid, "log", "same.las", content),
                range(3),
            )
        )
    assert all(reply["versioning"]["status"] == "archived" for reply in replies)
    assert all(reply == replies[0] for reply in replies)
    assert replies[0]["asset"] == first["asset"]
    assert len(restarted.assets(OWNER, pid)) == 1
    assert not restarted.assets(OWNER, other_project)
    assert Path(original["path"]).read_bytes() == original_bytes
    assert len(store.list_versions(OWNER, pid)) == 1 and store.attempts == 2
    with pytest.raises(HTTPException) as caught:
        restarted.upload(OTHER, pid, "log", "same.las", content)
    assert caught.value.status_code == 404


def test_two_service_instances_recover_one_failed_upload_without_duplicate_version(
    harness,
):
    store = harness.dvc = RecordingStore()
    pid = harness.execute(OWNER, "project_create", {"name": "parallel recovery"})[
        "project_id"
    ]
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    store.fail = True
    failed = harness.upload(OWNER, pid, "log", "retry.las", payload)
    store.fail = False
    another = HuiyanHarness(
        harness.root, config=harness.config, native=harness.native, dvc_store=store
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(harness.upload, OWNER, pid, "log", "retry.las", payload)
        second = pool.submit(another.upload, OWNER, pid, "log", "retry.las", payload)
        replies = first.result(10), second.result(10)
    assert replies[0] == replies[1]
    assert replies[0]["asset"] == failed["asset"]
    assert store.attempts == 2 and len(store.versions) == 1


def test_slow_archive_does_not_hold_database_write_lock_for_unrelated_project(harness):
    store = harness.dvc = RecordingStore()
    pid = harness.execute(OWNER, "project_create", {"name": "slow archive"})[
        "project_id"
    ]
    another = HuiyanHarness(
        harness.root, config=harness.config, native=harness.native, dvc_store=store
    )
    entered, release = threading.Event(), threading.Event()
    original = store.archive

    def slow_archive(*args, **kwargs):
        entered.set()
        assert release.wait(8)
        return original(*args, **kwargs)

    store.archive = slow_archive
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload = pool.submit(harness.upload, OWNER, pid, "log", "slow.las", payload)
        try:
            assert entered.wait(3)
            create = pool.submit(
                another.execute,
                OTHER,
                "project_create",
                {"name": "unrelated owner and project"},
            )
            unrelated = create.result(timeout=2)
            assert unrelated["status"] == "completed" and not upload.done()
        finally:
            release.set()
        assert upload.result(timeout=5)["versioning"]["status"] == "archived"


def test_new_archive_does_not_enumerate_or_validate_unrelated_history(
    harness, monkeypatch
):
    store = harness.dvc = RecordingStore()
    pid, _ = project_asset(harness)

    def unrelated_corruption(*args):
        raise DvcStoreError(
            "integrity_error", "unrelated historical version is corrupt"
        )

    monkeypatch.setattr(store, "list_versions", unrelated_corruption)
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    newer = harness.upload(OWNER, pid, "log", "healthy-new.las", payload)
    assert newer["versioning"]["status"] == "archived"
    assert len(store.versions) == 2


def test_archive_coordination_is_per_key_and_released_on_process_exit(tmp_path):
    from well_seismic.dvc_store import archive_key_lock

    locks = tmp_path / "key-locks"
    marker = tmp_path / "child-acquired"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from well_seismic.dvc_store import archive_key_lock; "
            "lock = archive_key_lock(Path(sys.argv[1]), 'a' * 64); lock.__enter__(); "
            "Path(sys.argv[2]).write_text('acquired'); sys.stdin.readline()",
            str(locks),
            str(marker),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while (
            not marker.exists() and child.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists(), "isolated lock child failed to acquire the first key"
        with archive_key_lock(locks, "b" * 64, timeout=0.2):
            pass
        with pytest.raises(DvcStoreError) as caught:
            with archive_key_lock(locks, "a" * 64, timeout=0.2):
                pass
        assert caught.value.code == "archive_in_progress"
    finally:
        child.terminate()
        child.wait(timeout=5)
        if child.stdin:
            child.stdin.close()
        if child.stderr:
            child.stderr.close()
    with archive_key_lock(locks, "a" * 64, timeout=0.2):
        pass


def test_published_version_is_reused_after_receipt_save_was_interrupted(harness):
    store = harness.dvc = RecordingStore()
    pid = harness.execute(OWNER, "project_create", {"name": "lost publication reply"})[
        "project_id"
    ]
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    original_archive = store.archive

    def interrupted(*args, **kwargs):
        original_archive(*args, **kwargs)
        raise DvcStoreError(
            "dvc_failed", "synthetic lost reply after durable publication"
        )

    store.archive = interrupted
    first = harness.upload(OWNER, pid, "log", "input.las", payload)
    assert first["versioning"]["status"] == "blocked" and len(store.versions) == 1
    store.archive = original_archive
    recovered = harness.upload(OWNER, pid, "log", "input.las", payload)
    assert recovered["versioning"]["status"] == "archived"
    assert recovered["asset"] == first["asset"]
    assert len(store.versions) == 1 and store.attempts == 1


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_archive_limits_are_checked_before_hashing_any_source(
    harness, monkeypatch, limit
):
    harness.dvc = RecordingStore()
    pid, run, _ = completed_task(harness)
    if limit == "count":
        harness.dvc.max_files = 1
    else:
        harness.dvc.max_total_bytes = 1

    def no_hash_read(*args):
        pytest.fail("Over-limit artifacts must be rejected before hashing")

    monkeypatch.setattr("well_seismic.huiyan_harness.file_sha256", no_hash_read)
    before = len(harness.dvc.versions)
    result = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert result["status"] == "blocked" and result["error"]["code"] == "input_limit"
    assert len(harness.dvc.versions) == before


def test_recovery_in_one_project_does_not_touch_another_projects_failed_receipt(
    harness,
):
    store = harness.dvc = RecordingStore()
    projects = [
        harness.execute(OWNER, "project_create", {"name": name})["project_id"]
        for name in ["target", "other"]
    ]
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    store.fail = True
    failed = [
        harness.upload(OWNER, pid, "log", "same.las", payload) for pid in projects
    ]
    store.fail = False
    recovered = harness.upload(OWNER, projects[0], "log", "same.las", payload)
    assert recovered["asset"] == failed[0]["asset"]
    assert len(store.list_versions(OWNER, projects[0])) == 1
    assert not store.list_versions(OWNER, projects[1])
    with harness.db() as db:
        other = json.loads(
            db.execute(
                "SELECT receipt FROM uploads WHERE project_id=?", (projects[1],)
            ).fetchone()[0]
        )
    assert other == failed[1]
    assert all(len(harness.assets(OWNER, pid)) == 1 for pid in projects)


def completed_task(harness):
    pid, aid = project_asset(harness)
    run = harness.execute(
        OWNER, "data_prepare", {"project_id": pid, "asset_ids": [aid]}
    )
    output = harness.root / "native-results"
    output.mkdir()
    matrix = output / "sample.npy"
    np.save(matrix, np.arange(160, dtype=np.float32).reshape(5, 32))
    report = output / "report.json"
    report.write_text('{"synthetic":true}', encoding="utf-8")
    manifest = output / "manifest.csv"
    manifest.write_text("sample,count\nsynthetic,160\n", encoding="utf-8")
    task = harness.test_tasks[run["task_id"]]
    task.update(
        status="completed",
        result={
            "sample": {"outputs": {"sample_array": str(matrix)}},
            "output_files": {"report": str(report)},
            "prepared_view": {
                "artifacts": [
                    {
                        "name": "manifest",
                        "path": str(manifest),
                        "media_type": "text/csv",
                    }
                ]
            },
        },
    )
    return pid, run, [matrix, report, manifest]


def test_manual_archive_collects_same_native_outputs_and_reuses_version(harness):
    harness.dvc = RecordingStore()
    pid, run, outputs = completed_task(harness)
    auto = harness.get_run(OWNER, run["execution_id"])
    manual = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    repeated = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    expected = {hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    assert expected <= {f["sha256"] for f in manual["result"]["files"]}
    assert {f["path"] for f in auto["result"]["versioning"]["files"]} <= {
        f["path"] for f in manual["result"]["files"]
    }
    assert repeated["revision_id"] == manual["revision_id"]


def test_assistant_wait_stops_at_deadline_and_archives_a_later_completion_once(harness, monkeypatch):
    import well_seismic.huiyan_harness as module

    store = harness.dvc = RecordingStore()
    pid, aid = project_asset(harness)
    operation_id = str(uuid.uuid4())
    arguments = {"project_id": pid, "asset_ids": [aid]}
    run = harness.execute(OWNER, "data_prepare", arguments, operation_id, defer_settlement=True)
    assert run["status"] == "running"
    assert run["result"]["native_task_submitted"]
    clock = SimpleNamespace(now=0.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    before = store.attempts
    pending = harness.wait_for_fast_run(OWNER, run, deadline=1.0)
    assert sleeps == pytest.approx([0.4, 0.4, 0.2])
    assert pending["status"] == "running"
    assert pending["result"]["native_status"] == "queued"
    assert store.attempts == before

    harness.test_tasks[run["task_id"]]["status"] = "completed"
    assert harness.wait_for_fast_run(OWNER, pending, deadline=1.0) == pending
    assert store.attempts == before
    with pytest.raises(HTTPException) as denied:
        harness.wait_for_fast_run(OTHER, pending, deadline=2.0)
    assert denied.value.status_code == 404

    settled = harness.wait_for_fast_run(OWNER, pending, deadline=2.0)
    assert settled["status"] == "completed" and settled["revision_id"]
    assert store.attempts == before + 1
    repeated = harness.execute(OWNER, "data_prepare", arguments, operation_id, defer_settlement=True)
    assert settled == repeated
    assert store.attempts == before + 1


@pytest.mark.parametrize("mode", ["automatic", "manual"])
def test_missing_declared_result_never_reports_archive_success(harness, mode):
    harness.dvc = RecordingStore()
    pid, run, outputs = completed_task(harness)
    outputs[0].unlink()
    before = len(harness.dvc.versions)
    result = (
        harness.get_run(OWNER, run["execution_id"])
        if mode == "automatic"
        else harness.execute(
            OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
        )
    )
    assert result["status"] == "blocked"
    assert not result.get("revision_id") and len(harness.dvc.versions) == before
    assert str(harness.root) not in json.dumps(result)


def test_task_status_exposes_recovery_without_rewriting_failed_execution(harness):
    harness.dvc = RecordingStore()
    pid, run, _ = completed_task(harness)
    harness.dvc.fail = True
    failed = harness.get_run(OWNER, run["execution_id"])
    assert failed["status"] == "blocked"
    harness.dvc.fail = False
    recovered = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert recovered["status"] == "completed"
    assert run["execution_id"] in recovered["result"]["source_execution_ids"]
    status = harness.execute(
        OWNER, "task_status", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert status["status"] == "completed"
    assert status["revision_id"] == recovered["revision_id"]
    assert (
        status["result"]["archival_recovery"]["execution_id"]
        == recovered["execution_id"]
    )
    assert status["result"]["original_executions"][0]["status"] == "blocked"
    assert harness.get_run(OWNER, run["execution_id"]) == failed


def test_archive_recovery_never_clears_business_gate_or_cross_project_ownership(
    harness,
):
    harness.dvc = RecordingStore()
    pid, run, _ = completed_task(harness)
    harness.test_tasks[run["task_id"]]["result"]["registration"] = {
        "business_status": "blocked"
    }
    failed = harness.get_run(OWNER, run["execution_id"])
    assert failed["error"]["code"] == "native_business_blocked"
    archive = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert archive["status"] == "completed"
    status = harness.execute(
        OWNER, "task_status", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert (
        status["status"] == "blocked"
        and status["error"]["code"] == "native_business_blocked"
    )
    assert harness.get_run(OWNER, run["execution_id"]) == failed
    other_project = harness.execute(
        OWNER, "project_create", {"name": "not the task owner"}
    )["project_id"]
    for subject, project in [(OTHER, pid), (OWNER, other_project)]:
        with pytest.raises(HTTPException) as caught:
            harness.execute(
                subject,
                "dvc_archive",
                {"project_id": project, "task_id": run["task_id"]},
            )
        assert caught.value.status_code == 404


@pytest.mark.parametrize("kind", ["outside", "relative", "too_many"])
def test_invalid_or_excess_declared_outputs_are_not_silently_omitted(
    harness, tmp_path, kind
):
    harness.dvc = RecordingStore()
    pid, run, outputs = completed_task(harness)
    if kind == "outside":
        outsider = tmp_path / "outside.npy"
        outsider.write_bytes(b"not a trusted artifact")
        value = str(outsider)
    elif kind == "relative":
        value = "missing-relative.npy"
    else:
        value = [str(outputs[0])] * 101
    harness.test_tasks[run["task_id"]]["result"]["outputs"] = {"invalid": value}
    before = len(harness.dvc.versions)
    result = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert result["status"] == "blocked" and not result.get("revision_id")
    assert len(harness.dvc.versions) == before


@pytest.mark.skipif(
    importlib.util.find_spec("dvc") is None, reason="Real DVC is not installed"
)
def test_real_dvc_published_version_recovers_after_lost_reply_and_rejects_key_collision(
    harness, tmp_path, monkeypatch
):
    store = harness.dvc = DvcVersionStore(
        tmp_path.parent / ("dvl_" + uuid.uuid4().hex[:8]),
        allowed_source_roots=[harness.root],
    )
    pid = harness.execute(
        OWNER, "project_create", {"name": "real lost publication reply"}
    )["project_id"]
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    original_get = store.get
    published = []

    def lose_first_published_reply(subject, project, revision):
        verified = original_get(subject, project, revision)
        if not published:
            published.append(verified)
            raise DvcStoreError(
                "reply_interrupted", "synthetic fault after publication"
            )
        return verified

    monkeypatch.setattr(store, "get", lose_first_published_reply)
    failed = harness.upload(OWNER, pid, "log", "published.las", payload)
    assert failed["versioning"]["status"] == "blocked" and len(published) == 1
    with harness.db() as db:
        assert db.execute("SELECT COUNT(*) FROM archives").fetchone()[0] == 0
    monkeypatch.setattr(store, "get", original_get)
    restarted = HuiyanHarness(
        harness.root, config=harness.config, native=harness.native, dvc_store=store
    )
    recovered = restarted.upload(OWNER, pid, "log", "published.las", payload)
    assert recovered["versioning"]["status"] == "archived"
    assert recovered["asset"] == failed["asset"]
    assert recovered["revision_id"] == published[0]["revision_id"]
    assert len(store.list_versions(OWNER, pid)) == 1
    version = original_get(OWNER, pid, recovered["revision_id"])
    assert {entry["sha256"] for entry in version["files"]} == {
        hashlib.sha256(payload).hexdigest()
    }
    key = version["metadata"]["archive_key"]
    collision = key[:32] + ("0" if key[32:] != "0" * 32 else "1") * 32
    with pytest.raises(DvcStoreError) as caught:
        store.find_archive(OWNER, pid, collision)
    assert caught.value.code == "archive_identity_conflict"


@pytest.mark.skipif(
    importlib.util.find_spec("dvc") is None, reason="Real DVC is not installed"
)
def test_real_dvc_failed_upload_recovers_same_asset_and_only_one_version(
    harness, tmp_path
):
    store = harness.dvc = DvcVersionStore(
        tmp_path.parent / ("dvu_" + uuid.uuid4().hex[:8]),
        allowed_source_roots=[harness.root],
    )
    pid = harness.execute(OWNER, "project_create", {"name": "real upload recovery"})[
        "project_id"
    ]
    payload = (ROOT / "tests/fixtures/gaia_synthetic.las").read_bytes()
    real_command = store.command
    store.command = (str(tmp_path / "not-installed"),)
    failed = harness.upload(OWNER, pid, "log", "real.las", payload)
    assert failed["versioning"]["status"] == "blocked"
    asset = harness.asset(OWNER, pid, failed["asset"]["id"])
    original_bytes = Path(asset["path"]).read_bytes()
    store.command = real_command
    recovered = harness.upload(OWNER, pid, "log", "real.las", payload)
    repeated = harness.upload(OWNER, pid, "log", "real.las", payload)
    assert repeated == recovered and recovered["asset"] == failed["asset"]
    assert recovered["versioning"]["status"] == "archived"
    assert len(store.list_versions(OWNER, pid)) == len(harness.assets(OWNER, pid)) == 1
    restored = store.restore_owned(OWNER, pid, recovered["revision_id"])
    files = store.resolve_restored_files(OWNER, pid, restored["restore_id"])
    assert len(files) == 1 and next(iter(files.values())).read_bytes() == original_bytes
    assert Path(asset["path"]).read_bytes() == original_bytes
    # A damaged older version must fail closed only when it is requested. It
    # cannot force every healthy new input through an all-history verification.
    old_key = recovered["versioning"]["metadata"]["archive_key"]
    old_file = next(
        iter(store.resolve_files(OWNER, pid, recovered["revision_id"]).values())
    )
    old_file.write_bytes(b"deliberately corrupted isolated old version")
    healthy = harness.upload(OWNER, pid, "log", "healthy.las", payload)
    assert healthy["versioning"]["status"] == "archived"
    assert (
        store.find_archive(
            OWNER, pid, healthy["versioning"]["metadata"]["archive_key"]
        )["revision_id"]
        == healthy["revision_id"]
    )
    with pytest.raises(DvcStoreError) as caught:
        store.find_archive(OWNER, pid, old_key)
    assert caught.value.code == "integrity_error"


@pytest.mark.skipif(
    importlib.util.find_spec("dvc") is None, reason="Real DVC is not installed"
)
def test_real_dvc_manual_restore_contains_every_native_result_and_same_hash(
    harness, tmp_path
):
    harness.dvc = DvcVersionStore(
        tmp_path.parent / ("dvr_" + uuid.uuid4().hex[:8]),
        allowed_source_roots=[harness.root],
    )
    pid, run, outputs = completed_task(harness)
    # Simulate the reported storage outage at the native automatic archive.
    real_command = harness.dvc.command
    harness.dvc.command = (str(tmp_path / "not-installed"),)
    failed = harness.get_run(OWNER, run["execution_id"])
    assert failed["status"] == "blocked"
    harness.dvc.command = real_command
    manual = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert manual["status"] == "completed", manual
    repeat = harness.execute(
        OWNER, "dvc_archive", {"project_id": pid, "task_id": run["task_id"]}
    )
    assert repeat["revision_id"] == manual["revision_id"]
    receipt = harness.dvc.restore_owned(OWNER, pid, manual["revision_id"])
    restored = harness.dvc.resolve_restored_files(OWNER, pid, receipt["restore_id"])
    manifest = {item["path"]: item["sha256"] for item in manual["result"]["files"]}
    assert set(restored) == set(manifest)
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name]
        for name, path in restored.items()
    )
    assert {hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs} <= set(
        manifest.values()
    )
    assert (
        len(harness.dvc.list_versions(OWNER, pid)) == 2
    )  # original input + recovered complete task
