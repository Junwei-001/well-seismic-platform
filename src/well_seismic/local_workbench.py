"""Loopback-only project/asset bridge for the standalone desktop workbench."""
import hashlib

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .gaia_tasks import is_loopback
from .huiyan_harness import ActionRequest, HuiyanHarness, MAX_UPLOAD_BYTES

LOCAL_WORKBENCH_SUBJECT = hashlib.sha256(b"well-seismic:local-workbench").hexdigest()


def build_local_workbench_router(harness: HuiyanHarness) -> APIRouter:
    router = APIRouter(prefix="/api/v1/workbench", tags=["Local desktop workbench"])

    def require_local(request: Request):
        # Never trust forwarded headers or accept a principal from the browser.
        if not request.client or not is_loopback(request.client.host):
            raise HTTPException(403, "Desktop workbench routes are loopback only")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Desktop workbench requires same origin")

    @router.post("/actions")
    def action(payload: ActionRequest, request: Request):
        require_local(request)
        if payload.operation not in {"project_create", "project_list", "asset_list"}:
            raise HTTPException(422, "Use the scoped assistant for this operation")
        return harness.execute(LOCAL_WORKBENCH_SUBJECT, payload.operation,
                               payload.arguments, str(payload.operation_id))

    @router.post("/projects/{project_id}/assets/log", status_code=201)
    async def upload(project_id: str, request: Request, filename: str):
        require_local(request)
        await run_in_threadpool(harness.project, LOCAL_WORKBENCH_SUBJECT, project_id)
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Asset exceeds 120 MiB")
            content.extend(chunk)
        return await run_in_threadpool(
            harness.upload, LOCAL_WORKBENCH_SUBJECT, project_id, "log", filename, bytes(content)
        )

    return router
