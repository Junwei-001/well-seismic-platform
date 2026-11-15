"""Authenticated, owner-scoped access to the native Huiyan workbench."""

from __future__ import annotations

import html
import json
import mimetypes
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .api_models import AssistantChatRequest
from .gaia_tasks import authenticate

_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>，；]+|/(?:home|data|tmp|var|srv|opt|mnt|workspace|Users|root)/[^\s\"'<>，；]+"
)


def public_value(value):
    """Keep native result shapes while withholding backend filesystem locations."""
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        return _PATH.sub(
            lambda match: match.group().replace("\\", "/").rsplit("/", 1)[-1], value
        )
    return value


def visualization_query(request: Request) -> dict:
    """The task identity belongs to the owned path, never to query parameters."""
    allowed = {"asset", "block", "layerpulse_output"}
    pairs = list(request.query_params.multi_items())
    if any(key not in allowed for key, _ in pairs) or len(
        {key for key, _ in pairs}
    ) != len(pairs):
        raise HTTPException(422, "Unsupported or repeated visualization query")
    result = {}
    for key, value in pairs:
        pattern = r"[0-9]{1,6}" if key == "asset" else r"[A-Za-z0-9_-]{1,80}"
        if not re.fullmatch(pattern, value):
            raise HTTPException(422, "Invalid visualization selector")
        result[key] = int(value) if key == "asset" else value
    return result


def owned_visualization_html(
    document: str, task_id: str, *, status_code: int = 200
) -> str:
    """Rewrite only native viewer navigation to the current owned resource.

    Relative query links work behind both supported Gaia gateways. Native task
    IDs never grant access to another task through an embedded link.
    """

    def replace(match):
        raw = match.group(1) or ""
        query = parse_qs(html.unescape(raw), keep_blank_values=True)
        tasks = query.pop("task_id", [task_id])
        if tasks != [task_id]:
            return "#unavailable"
        query.pop("embed", None)
        selected = {}
        for key in ("asset", "block", "layerpulse_output"):
            values = query.get(key)
            if values is None:
                continue
            pattern = r"[0-9]{1,6}" if key == "asset" else r"[A-Za-z0-9_-]{1,80}"
            if len(values) != 1 or not re.fullmatch(pattern, values[0]):
                return "#unavailable"
            selected[key] = values[0]
        encoded = "?" + urlencode(selected)
        return encoded.replace("&", "&amp;") if "&amp;" in raw else encoded

    document = re.sub(r"""/统一数据可视化(?:\?([^\s"'<>]*))?""", replace, document)
    # The native unavailable/error document has no workbench. An iframe load
    # event alone is not proof that a visualization was served successfully.
    if (
        200 <= status_code < 300
        and re.fullmatch(r"[0-9a-f]{32}", task_id)
        and re.search(r"""\bid=["']workbench["']""", document)
        and "</body>" in document
    ):
        message = json.dumps({"type": "huiyan:visualization-ready", "taskId": task_id})
        script = (
            "<script>window.parent.postMessage("
            + message
            + ",window.location.origin);</script>"
        )
        document = document.replace("</body>", script + "</body>", 1)
    return document


def require_owned_task(root: Path, subject: str, task_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise HTTPException(404, "慧眼任务不存在或不属于当前账号")
    if (root / "receipts.sqlite3").is_file():
        with sqlite3.connect(root / "receipts.sqlite3") as db:
            row = db.execute(
                "SELECT run_id FROM receipts WHERE task_id=? AND subject=?",
                (task_id, subject),
            ).fetchone()
        if row:
            return row[0]
    harness_db = root.parent / "huiyan_harness" / "harness.sqlite3"
    if harness_db.is_file():
        with sqlite3.connect(harness_db) as db:
            row = db.execute(
                "SELECT project_id FROM tasks WHERE task_id=? AND subject=?",
                (task_id, subject),
            ).fetchone()
        if row:
            return row[0]
    raise HTTPException(404, "慧眼任务不存在或不属于当前账号")


def build_router(
    root: Path,
    dist: Path,
    *,
    get_task: Callable,
    capabilities: Callable,
    health: Callable,
    assistant: Callable,
    manifest: Callable | None = None,
    assistant_owned: Callable | None = None,
    execution: Callable | None = None,
    visualization: Callable | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/gaia/ui", tags=["Gaia workbench"])

    @router.get("/")
    def index(request: Request):
        authenticate(request)
        source = dist / "index.html"
        if not source.is_file():
            raise HTTPException(503, "慧眼前端尚未构建")
        # Relative assets stay within whichever authenticated Gaia gateway
        # served this document, including the already-running local demo.
        html = source.read_text(encoding="utf-8").replace('="/assets/', '="./assets/')
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @router.get("/assets/{asset_path:path}")
    def asset(asset_path: str, request: Request):
        authenticate(request)
        asset_root = (dist / "assets").resolve()
        candidate = (asset_root / asset_path).resolve()
        if not candidate.is_relative_to(asset_root) or not candidate.is_file():
            raise HTTPException(404, "Asset not found")
        if candidate.suffix.lower() not in {
            ".js",
            ".css",
            ".jpg",
            ".jpeg",
            ".png",
            ".svg",
            ".woff",
            ".woff2",
            ".webp",
        }:
            raise HTTPException(404, "Asset not found")
        return FileResponse(
            candidate, media_type=mimetypes.guess_type(candidate.name)[0]
        )

    @router.get("/api/v1/health")
    def ui_health(request: Request):
        authenticate(request)
        return {
            key: value
            for key, value in health().items()
            if key
            in {"status", "service", "version", "runtime_mode", "task_models_enabled"}
        }

    @router.get("/api/v1/capabilities")
    def ui_capabilities(request: Request):
        authenticate(request)
        return public_value(capabilities())

    @router.get("/api/v1/tasks/{task_id}")
    def ui_task(task_id: str, request: Request):
        subject = authenticate(request)
        require_owned_task(root, subject, task_id)
        return public_value(get_task(task_id))

    @router.get("/api/v1/tasks/{task_id}/artifacts/snapshot_manifest")
    def ui_manifest(task_id: str, request: Request):
        require_owned_task(root, authenticate(request), task_id)
        if manifest is None:
            raise HTTPException(404, "数据清单不可用")
        artifact = manifest(task_id)
        path = Path(artifact.path)
        if path.stat().st_size > 8 * 1024 * 1024:
            raise HTTPException(413, "数据清单过大")
        value = public_value(json.loads(path.read_text(encoding="utf-8")))
        return JSONResponse(
            value,
            headers={
                "Content-Disposition": 'attachment; filename="huiyan-snapshot.json"'
            },
        )

    @router.post("/api/v1/assistant/chat")
    def ui_chat(payload: AssistantChatRequest, request: Request):
        subject = authenticate(request)
        if payload.task_id:
            require_owned_task(root, subject, payload.task_id)
        root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(root / "ui_sessions.sqlite3") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, subject TEXT NOT NULL)"
            )
            if payload.conversation_id:
                row = db.execute(
                    "SELECT subject FROM sessions WHERE id=?",
                    (str(payload.conversation_id),),
                ).fetchone()
                if row is None or row[0] != subject:
                    raise HTTPException(404, "对话不存在或不属于当前账号，请开启新对话")
        result = (
            assistant_owned(payload, subject=subject)
            if assistant_owned
            else assistant(payload)
        )
        if result.get("conversation_id"):
            with sqlite3.connect(root / "ui_sessions.sqlite3") as db:
                db.execute(
                    "INSERT OR IGNORE INTO sessions VALUES (?, ?)",
                    (str(result["conversation_id"]), subject),
                )
        return public_value(result)

    @router.get("/api/v1/tasks/{task_id}/visualization")
    def ui_visualization(task_id: str, request: Request):
        require_owned_task(root, authenticate(request), task_id)
        if visualization is None:
            raise HTTPException(404, "可视化结果不可用")
        result = visualization(task_id, **visualization_query(request))
        document = owned_visualization_html(
            public_value(result.body.decode("utf-8")),
            task_id,
            status_code=result.status_code,
        )
        return HTMLResponse(
            document,
            status_code=result.status_code,
            headers={"Cache-Control": "private, no-store"},
        )

    @router.get("/api/v1/assistant/executions/{execution_id}")
    def ui_execution(execution_id: str, request: Request):
        subject = authenticate(request)
        if execution is None:
            raise HTTPException(404, "执行记录不可用")
        return public_value(execution(subject, execution_id))

    return router
