"""Authenticated Gaia transport for the existing data-preparation engine."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from .api_models import InspectionRequest
from .llm.settings import _load_local_env

SCHEMA = "well-seismic.gaia-result.v1"
MAX_BYTES = 20 * 1024 * 1024
_LOCK = threading.Lock()


def transport_configured() -> bool:
    _load_local_env()
    return bool(os.environ.get("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", ""))


def service_token() -> str:
    _load_local_env()
    path = os.environ.get("WELL_SEISMIC_GAIA_SERVICE_TOKEN_FILE", "")
    try:
        value = Path(path).read_text(encoding="utf-8").strip() if path else ""
    except OSError:
        value = ""
    return value if len(value) >= 32 else ""


def authenticate(request: Request) -> str:
    token = service_token()
    if not token:
        raise HTTPException(503, "Gaia task connector is not configured")
    if not hmac.compare_digest(
        request.headers.get("authorization", ""), f"Bearer {token}"
    ):
        raise HTTPException(401, "Invalid service credential")
    subject = request.headers.get("x-gaia-subject", "")
    if not re.fullmatch(r"[0-9a-f]{64}", subject):
        raise HTTPException(400, "Missing Gaia subject")
    return subject


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def result_report(task: dict, receipt: dict) -> dict:
    result = task.get("result") or {}
    assets = result.get("assets") or []
    if len(assets) != 1 or assets[0].get("sha256") != receipt["input_sha256"]:
        raise ValueError("Sealed input does not match uploaded LAS")
    summary = result.get("summary") or {}
    counts = {
        key: int(summary.get(key, 0))
        for key in (
            "assets",
            "wells",
            "log_files",
            "seismic_files",
            "errors",
            "uncertain",
        )
    }
    wells = []
    for well in result.get("wells", [])[:100]:
        wells.append(
            {
                "name": str(well.get("name", ""))[:128],
                "logs": [
                    {
                        "samples": int(log.get("samples", 0)),
                        "curves": [str(c)[:64] for c in log.get("curves", [])[:100]],
                        "issue_count": len(log.get("issues", [])),
                    }
                    for log in well.get("logs", [])[:100]
                ],
            }
        )
    report = {
        "schema_version": SCHEMA,
        "operation": "las_data_preparation",
        "run_id": receipt["run_id"],
        "task_id": receipt["task_id"],
        "input_sha256": receipt["input_sha256"],
        "summary": counts,
        "wells": wells,
        "scope": "LAS 文件解析、数据登记与检查；未执行模型预测或井震标定。",
    }
    report["evidence_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


def build_router(root: Path, submit: Callable, get_task: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/v1/gaia", tags=["Gaia task connector"])

    @contextmanager
    def database():
        root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(root / "receipts.sqlite3")
        db.row_factory = sqlite3.Row
        db.execute(
            "CREATE TABLE IF NOT EXISTS receipts (run_id TEXT PRIMARY KEY, subject TEXT NOT NULL, input_sha256 TEXT NOT NULL, task_id TEXT NOT NULL)"
        )
        try:
            yield db
        finally:
            db.close()

    def owned(run_id: str, subject: str) -> dict:
        with database() as db:
            row = db.execute(
                "SELECT * FROM receipts WHERE run_id=? AND subject=?", (run_id, subject)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Task not found")
        return dict(row)

    def public_receipt(receipt: dict) -> dict:
        task = get_task(receipt["task_id"])
        status = task["status"]
        output = {
            "schema_version": SCHEMA,
            "run_id": receipt["run_id"],
            "task_id": receipt["task_id"],
            "input_sha256": receipt["input_sha256"],
            "status": status,
            "progress": int(task.get("progress", 0)),
        }
        if status == "completed":
            try:
                output["report"] = result_report(task, receipt)
            except ValueError:
                output.update(
                    status="failed", error="慧眼封存输入与盖亚提交内容不一致。"
                )
        elif status == "failed":
            output["error"] = "慧眼数据准备失败；请在慧眼本机查看任务诊断。"
        return output

    @router.get("/capabilities")
    def capabilities(request: Request):
        authenticate(request)
        return {
            "schema_version": SCHEMA,
            "operations": ["las_data_preparation"],
            "max_input_bytes": MAX_BYTES,
        }

    @router.post("/runs/{run_id}", status_code=202)
    async def create_run(run_id: str, request: Request):
        subject = authenticate(request)
        if not re.fullmatch(r"run_[0-9a-f]{16}", run_id):
            raise HTTPException(422, "Invalid run ID")
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_BYTES:
                raise HTTPException(413, "LAS input exceeds 20 MB")
            content.extend(chunk)
        if not content:
            raise HTTPException(422, "Empty LAS input")
        return await run_in_threadpool(persist_run, run_id, subject, bytes(content))

    def persist_run(run_id: str, subject: str, content: bytes):
        # The original parser remains authoritative; reject obvious non-LAS input early.
        if not re.search(rb"(?im)^\s*~v", content) or not re.search(
            rb"(?im)^\s*~a", content
        ):
            raise HTTPException(
                422, "Expected a LAS file with version and data sections"
            )
        digest = hashlib.sha256(content).hexdigest()
        with _LOCK, database() as db:
            existing = db.execute(
                "SELECT * FROM receipts WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing:
                if existing["subject"] != subject:
                    raise HTTPException(404, "Task not found")
                if existing["input_sha256"] != digest:
                    raise HTTPException(
                        409, "Run ID already belongs to different input"
                    )
                return public_receipt(dict(existing))
            folder = root / subject / run_id
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / "input.las"
            source.write_bytes(content)
            created = submit(
                InspectionRequest(log_paths=[str(source)], use_llm_fallback=False)
            )
            receipt = {
                "run_id": run_id,
                "subject": subject,
                "input_sha256": digest,
                "task_id": created.task_id,
            }
            db.execute(
                "INSERT INTO receipts VALUES (:run_id, :subject, :input_sha256, :task_id)",
                receipt,
            )
            db.commit()
            return public_receipt(receipt)

    @router.get("/runs/{run_id}")
    def read_run(run_id: str, request: Request):
        return public_receipt(owned(run_id, authenticate(request)))

    return router
