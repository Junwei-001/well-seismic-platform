"""Server-side bridge to Gaia's authenticated chat API."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

MAX_STREAM_BYTES = 2 * 1024 * 1024
DEFAULT_PROVIDER = "minimax-token-plan/MiniMax-M2.7"


class GaiaChatError(RuntimeError):
    """A Gaia chat request did not produce a complete assistant response."""


def _validated_base_url(value: str) -> str:
    origin = value.strip().rstrip("/")
    if any(char.isspace() or ord(char) < 32 for char in origin) or "?" in origin or "#" in origin:
        raise ValueError("Gaia chat URL must be an HTTPS origin or a loopback HTTP origin")
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Gaia chat URL must be an HTTPS origin or a loopback HTTP origin") from exc
    host = (parsed.hostname or "").casefold()
    if (
        not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}))
    ):
        raise ValueError("Gaia chat URL must be an HTTPS origin or a loopback HTTP origin")
    return origin


@dataclass(frozen=True)
class GaiaChatSettings:
    base_url: str
    api_key: str
    provider_id: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> GaiaChatSettings:
        raw_url = os.getenv("WELL_SEISMIC_GAIA_CHAT_BASE_URL", "").strip()
        timeout = float(os.getenv("WELL_SEISMIC_GAIA_CHAT_TIMEOUT_SECONDS", "120"))
        if not math.isfinite(timeout):
            raise ValueError("Gaia chat timeout must be finite")
        return cls(
            base_url=_validated_base_url(raw_url) if raw_url else "",
            api_key=os.getenv("WELL_SEISMIC_GAIA_CHAT_API_KEY", "").strip(),
            provider_id=os.getenv("WELL_SEISMIC_GAIA_PROVIDER_ID", DEFAULT_PROVIDER).strip(),
            timeout_seconds=max(5.0, timeout),
        )

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.provider_id)

    def public_status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "provider": "gaia_open_api",
            "model": self.provider_id,
            "credential_configured": bool(self.api_key),
            "credential_policy": "Gaia chat API key is read only by the backend and never returned to the browser",
        }


def _read_chat_stream(response: Any) -> str:
    chunks: list[str] = []
    consumed = 0
    ended = False
    for raw_line in response:
        consumed += len(raw_line)
        if consumed > MAX_STREAM_BYTES:
            raise GaiaChatError("Gaia chat response exceeded its size limit")
        if not raw_line.startswith(b"data:"):
            continue
        try:
            event = json.loads(raw_line[5:].strip().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GaiaChatError("Gaia chat returned an invalid event") from exc
        if not isinstance(event, dict):
            raise GaiaChatError("Gaia chat returned an invalid event")
        event_type = event.get("type")
        if event_type == "error":
            raise GaiaChatError("Gaia chat reported an error")
        if event_type == "end":
            ended = True
            break
        if event_type == "plain" and event.get("chain_type") not in {
            "tool_call", "tool_call_result", "reasoning", "expert_stream"
        }:
            data = event.get("data")
            if isinstance(data, str):
                chunks.append(data)
    answer = "".join(chunks).strip()
    if not ended or not answer:
        raise GaiaChatError("Gaia chat ended without a complete answer")
    return answer[:3000]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Gaia bearer credential to a redirect destination.
        return None


def chat_with_gaia(
    settings: GaiaChatSettings,
    *,
    question: str,
    context: dict[str, Any],
    conversation_id: UUID | None,
) -> tuple[str, str]:
    if not settings.available:
        raise GaiaChatError("Gaia chat is not configured")
    session_id = str(conversation_id or uuid4())
    message = (
        "你是地层慧眼平台的井震数据工程助手。只依据以下经过脱敏的任务摘要回答，"
        "不要编造数据或声称看过原始SEG-Y/LAS；涉及数据修改时先说明需要人工确认。\n"
        f"任务摘要：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
        f"用户问题：{question}"
    )
    body = {
        "username": "well-seismic-assistant",
        "session_id": session_id,
        "message": message,
        "selected_provider": settings.provider_id,
        "enable_streaming": True,
    }
    request = urllib.request.Request(
        f"{settings.base_url}/api/v1/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=settings.timeout_seconds) as response:
            return _read_chat_stream(response), session_id
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GaiaChatError("Gaia chat is unavailable") from exc
