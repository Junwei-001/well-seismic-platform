"""Contract tests for the server-side Gaia chat bridge."""

from __future__ import annotations

import io
import json
import unittest
import urllib.request
import urllib.response
from contextlib import contextmanager
from email.message import Message
from unittest.mock import patch
from uuid import UUID

from well_seismic.gaia_chat import (
    GaiaChatError,
    GaiaChatSettings,
    _read_chat_stream,
    _validated_base_url,
    chat_with_gaia,
)


def event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def transport_response(request, *, code=200, body=b"", location=None):
    headers = Message()
    if location is not None:
        headers["Location"] = location
    response = urllib.response.addinfourl(io.BytesIO(body), headers, request.full_url, code)
    response.msg = "Synthetic transport response"
    return response


@contextmanager
def fake_http_transport(respond):
    """Keep urllib's real opener/error/redirect chain; replace only network I/O."""
    requests = []

    def exchange(handler, request):
        requests.append(request)
        return respond(request, len(requests))

    with (
        patch.object(urllib.request, "_opener", None),
        patch.object(urllib.request.HTTPHandler, "http_open", exchange),
        patch.object(urllib.request.HTTPSHandler, "https_open", exchange),
        patch("socket.create_connection", side_effect=AssertionError("No real network is allowed")),
    ):
        yield requests


class GaiaChatTests(unittest.TestCase):
    def test_remote_chat_requires_https(self) -> None:
        self.assertEqual(_validated_base_url("https://gaia.example.org/"), "https://gaia.example.org")
        self.assertEqual(_validated_base_url("http://127.0.0.1:6185"), "http://127.0.0.1:6185")
        for invalid in (
            "http://gaia.example.org",
            "https://user:pass@gaia.example.org",
            "https://gaia.example.org/api/v1/chat",
            "https://gaia.example.org/?key=secret",
            "https://gaia.example.org:invalid",
            "https://gaia.example.org?",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validated_base_url(invalid)

    def test_stream_collects_answer_and_rejects_incomplete_output(self) -> None:
        body = b"".join(
            (
                event({"type": "session_id", "session_id": "x"}),
                event({"type": "plain", "data": "井震", "streaming": True}),
                event({"type": "plain", "data": "结果", "streaming": True}),
                event({"type": "plain", "data": "internal", "chain_type": "tool_call_result"}),
                event({"type": "end"}),
            )
        )
        self.assertEqual(_read_chat_stream(io.BytesIO(body)), "井震结果")
        with self.assertRaises(GaiaChatError):
            _read_chat_stream(io.BytesIO(event({"type": "plain", "data": "partial"})))
        with self.assertRaises(GaiaChatError):
            _read_chat_stream(io.BytesIO(event({"type": "error", "data": "private detail"})))

    def test_request_reuses_session_and_keeps_key_out_of_public_status(self) -> None:
        settings = GaiaChatSettings(
            base_url="https://gaia.example.org",
            api_key="server-only-secret",
            provider_id="minimax-token-plan/MiniMax-M2.7",
            timeout_seconds=10,
        )
        session_id = UUID("6fdc9411-7d8b-4608-bcb0-8f533861e24d")
        body = event({"type": "plain", "data": "完成"}) + event({"type": "end"})
        with fake_http_transport(lambda request, count: transport_response(request, body=body)) as requests:
            answer, returned_id = chat_with_gaia(
                settings,
                question="当前任务如何处理？",
                context={"task": "数据准备"},
                conversation_id=session_id,
            )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        payload = json.loads(request.data)
        self.assertEqual(answer, "完成")
        self.assertEqual(returned_id, str(session_id))
        self.assertEqual(payload["session_id"], str(session_id))
        self.assertEqual(payload["selected_provider"], settings.provider_id)
        self.assertEqual(request.get_header("Authorization"), "Bearer server-only-secret")
        self.assertNotIn("server-only-secret", json.dumps(settings.public_status()))

    def test_authenticated_redirects_never_follow_or_forward_credentials(self) -> None:
        settings = GaiaChatSettings(
            base_url="https://gaia.example.org",
            api_key="synthetic-redirect-key-not-a-real-credential",
            provider_id="synthetic-provider",
            timeout_seconds=10,
        )
        answer = event({"type": "plain", "data": "redirect destination"}) + event({"type": "end"})
        for code in (301, 302, 303):
            for target in (
                "https://other.example.invalid/collect",
                "http://gaia.example.org/collect",
                "https://gaia.example.org/another-path",
            ):
                with self.subTest(code=code, target=target):
                    def respond(request, count, *, code=code, target=target):
                        if count == 1:
                            return transport_response(request, code=code, location=target)
                        return transport_response(request, body=answer)

                    with fake_http_transport(respond) as requests, self.assertRaises(GaiaChatError) as error:
                        chat_with_gaia(settings, question="test", context={}, conversation_id=None)
                    self.assertEqual(len(requests), 1, "A redirect destination must never be contacted")
                    self.assertEqual(requests[0].full_url, "https://gaia.example.org/api/v1/chat")
                    self.assertEqual(requests[0].get_header("Authorization"), f"Bearer {settings.api_key}")
                    self.assertNotIn(settings.api_key, str(error.exception))
                    self.assertNotIn(target, str(error.exception))

    def test_assistant_route_sanitizes_local_paths_before_gaia_call(self) -> None:
        from well_seismic import api
        from well_seismic.api_models import AssistantChatRequest

        settings = GaiaChatSettings(
            base_url="https://gaia.example.org",
            api_key="server-only-secret",
            provider_id="minimax-token-plan/MiniMax-M2.7",
            timeout_seconds=10,
        )
        with (
            patch.object(GaiaChatSettings, "from_environment", return_value=settings),
            patch.object(api, "_assistant_context", return_value={"source": "C:\\private\\well.las"}),
            patch.object(api, "chat_with_gaia", return_value=("已检查摘要", "6fdc9411-7d8b-4608-bcb0-8f533861e24d")) as chat_mock,
        ):
            answer = api.assistant_chat(
                AssistantChatRequest(message="检查 C:\\private\\well.las", task_id="task-1")
            )
        self.assertEqual(answer["answer"], "已检查摘要")
        self.assertEqual(answer["provider"], "gaia_open_api")
        self.assertNotIn("server-only-secret", json.dumps(answer))
        self.assertNotIn("C:\\private\\well.las", json.dumps(chat_mock.call_args.kwargs))


if __name__ == "__main__":
    unittest.main()
