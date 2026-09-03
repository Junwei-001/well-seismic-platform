from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .contracts import LLMDecision, LLMDecisionRequest
from .settings import LLMSettings, validate_kimi_base_url


MAX_LLM_RESPONSE_BYTES = 2 * 1024 * 1024


class LLMProviderError(RuntimeError):
    pass


def _open(request_object: urllib.request.Request, settings: LLMSettings):
    if settings.use_system_proxy:
        return urllib.request.urlopen(request_object, timeout=settings.timeout_seconds)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request_object, timeout=settings.timeout_seconds)


def _read_bounded_json_response(
    response: Any,
    *,
    max_bytes: int = MAX_LLM_RESPONSE_BYTES,
) -> dict[str, Any]:
    """Read at most max_bytes plus one sentinel byte, then parse one JSON object."""
    raw_content_length = response.headers.get("Content-Length")
    if raw_content_length is not None:
        try:
            content_length = int(str(raw_content_length).strip())
        except ValueError as exc:
            raise LLMProviderError("LLM响应的Content-Length无效") from exc
        if content_length < 0:
            raise LLMProviderError("LLM响应的Content-Length无效")
        if content_length > max_bytes:
            raise LLMProviderError(
                f"LLM响应体超过{max_bytes}字节上限（Content-Length={content_length}）"
            )
    body = response.read(max_bytes + 1)
    if not isinstance(body, bytes):
        raise LLMProviderError("LLM响应体不是字节数据")
    if len(body) > max_bytes:
        raise LLMProviderError(f"LLM响应体超过{max_bytes}字节上限")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMProviderError("LLM响应体不是有效UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise LLMProviderError("LLM响应体不是JSON对象")
    return payload


def _configure_chat_output(
    body: dict[str, Any],
    settings: LLMSettings,
    *,
    schema_name: str,
    schema: dict[str, Any],
) -> None:
    """Apply the provider-specific structured-output and token contract."""
    if settings.provider == "kimi":
        body["max_completion_tokens"] = settings.max_output_tokens
        body["reasoning_effort"] = settings.reasoning_effort
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
        return
    body["max_tokens"] = settings.max_output_tokens
    body["response_format"] = {"type": "json_object"}


def _validate_chat_finish(payload: dict[str, Any]) -> None:
    try:
        finish_reason = str(payload["choices"][0]["finish_reason"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError("LLM响应缺少finish_reason，不能确认结构化输出完整") from exc
    if finish_reason != "stop":
        raise LLMProviderError(f"LLM响应未完整结束：finish_reason={finish_reason}")


class OpenAIResponsesProvider:
    """Minimal Responses API adapter with strict structured output and no SDK dependency."""

    provider_name = "openai_responses"

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.model = settings.model

    def decide(self, request: LLMDecisionRequest) -> LLMDecision:
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": list(request.options)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["choice", "confidence", "reason", "warnings"],
            "additionalProperties": False,
        }
        body = {
            "model": self.model,
            "store": False,
            "max_output_tokens": self.settings.max_output_tokens,
            "instructions": (
                "你是井震数据预处理中的受限分类器。只能从提供的候选项中选择；"
                "不得补造井名、坐标、单位、深度、时间或任何数值；证据不足时选择保留待确认/保留原始曲线。"
                "confidence表示对所选候选的把握，不表示原始数据质量。"
            ),
            "input": json.dumps(
                {
                    "decision_type": request.decision_type,
                    "question": request.question,
                    "options": request.options,
                    "evidence": request.evidence,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "well_seismic_preprocessing_decision",
                    "description": "受控的井震预处理候选选择",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        client_request_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "X-Client-Request-Id": client_request_id,
        }
        if self.settings.organization:
            headers["OpenAI-Organization"] = self.settings.organization
        if self.settings.project:
            headers["OpenAI-Project"] = self.settings.project
        request_object = urllib.request.Request(
            f"{self.settings.base_url}/responses",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                with _open(request_object, self.settings) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    response_id = response.headers.get("x-request-id", "") or str(payload.get("id", ""))
                parsed = self._structured_text(payload)
                return LLMDecision(
                    decision_type=request.decision_type,
                    choice=str(parsed["choice"]),
                    confidence=float(parsed["confidence"]),
                    reason=str(parsed["reason"])[:500],
                    provider=self.provider_name,
                    model=self.model,
                    request_id=response_id or client_request_id,
                    source_hash=request.source_hash,
                    warnings=[str(item)[:200] for item in parsed.get("warnings", [])],
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(2.0, 0.35 * (2 ** attempt)))
        raise LLMProviderError(f"LLM判断请求失败：{last_error}")

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Generate schema-constrained JSON for reviewed transformations and chat."""
        body = {
            "model": self.model,
            "store": False,
            "max_output_tokens": self.settings.max_output_tokens,
            "instructions": system_prompt,
            "input": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        client_request_id = str(uuid.uuid4())
        request_object = urllib.request.Request(
            f"{self.settings.base_url}/responses",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "X-Client-Request-Id": client_request_id,
            },
            method="POST",
        )
        try:
            with _open(request_object, self.settings) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
                request_id = response.headers.get("x-request-id", "") or str(response_payload.get("id", ""))
            return self._structured_text(response_payload), {
                "provider": self.provider_name,
                "model": self.model,
                "request_id": request_id or client_request_id,
            }
        except Exception as exc:
            raise LLMProviderError(f"LLM结构化生成失败：{exc}") from exc

    @staticmethod
    def _structured_text(payload: dict[str, Any]) -> dict[str, Any]:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return json.loads(direct)
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return json.loads(content["text"])
        raise LLMProviderError("LLM响应中没有结构化文本")


class OpenAICompatibleChatProvider:
    """OpenAI Chat Completions compatible adapter used by Kimi and compatible services."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.model = settings.model
        self.provider_name = f"{settings.provider}_chat_completions"
        self.base_url = (
            validate_kimi_base_url(settings.base_url)
            if settings.provider == "kimi"
            else settings.base_url.rstrip("/")
        )

    def decide(self, request: LLMDecisionRequest) -> LLMDecision:
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": list(request.options)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["choice", "confidence", "reason", "warnings"],
            "additionalProperties": False,
        }
        system_prompt = (
            "你是井震数据预处理中的受限分类器。只能从用户提供的候选项中选择；"
            "不得补造井名、坐标、单位、深度、时间或任何数值；证据不足时选择保留待确认或保留原始曲线。"
            "confidence表示对所选候选的把握，不表示原始数据质量。"
            "必须只返回一个JSON对象，且只能包含choice、confidence、reason、warnings四个字段。"
        )
        user_payload = {
            "decision_type": request.decision_type,
            "question": request.question,
            "options": request.options,
            "evidence": request.evidence,
            "output_contract": {
                "choice": "必须是options中的一个字符串",
                "confidence": "0到1之间的数字",
                "reason": "简短依据",
                "warnings": "字符串数组",
            },
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "stream": False,
        }
        _configure_chat_output(
            body,
            self.settings,
            schema_name="well_seismic_preprocessing_decision",
            schema=schema,
        )
        if self.settings.provider == "glm":
            body["thinking"] = {"type": "disabled"}
            body["do_sample"] = False

        client_request_id = str(uuid.uuid4())
        request_object = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "X-Client-Request-Id": client_request_id,
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                with _open(request_object, self.settings) as response:
                    payload = _read_bounded_json_response(response)
                    header_request_id = response.headers.get("x-request-id", "")
                _validate_chat_finish(payload)
                parsed = self._structured_text(
                    payload,
                    strict=self.settings.provider == "kimi",
                )
                response_id = str(payload.get("request_id") or payload.get("id") or header_request_id)
                return LLMDecision(
                    decision_type=request.decision_type,
                    choice=str(parsed["choice"]),
                    confidence=float(parsed["confidence"]),
                    reason=str(parsed["reason"])[:500],
                    provider=self.provider_name,
                    model=self.model,
                    request_id=response_id or client_request_id,
                    source_hash=request.source_hash,
                    warnings=[str(item)[:200] for item in parsed.get("warnings", [])],
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(2.0, 0.35 * (2 ** attempt)))
        raise LLMProviderError(f"LLM判断请求失败：{last_error}")

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"input": payload, "output_json_schema": schema},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "stream": False,
        }
        _configure_chat_output(
            body,
            self.settings,
            schema_name=schema_name,
            schema=schema,
        )
        if self.settings.provider == "glm":
            body["thinking"] = {"type": "disabled"}
            body["do_sample"] = False
        client_request_id = str(uuid.uuid4())
        request_object = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "X-Client-Request-Id": client_request_id,
            },
            method="POST",
        )
        try:
            with _open(request_object, self.settings) as response:
                response_payload = _read_bounded_json_response(response)
                header_request_id = response.headers.get("x-request-id", "")
            _validate_chat_finish(response_payload)
            parsed = self._structured_text(
                response_payload,
                strict=self.settings.provider == "kimi",
            )
            return parsed, {
                "provider": self.provider_name,
                "model": self.model,
                "request_id": str(response_payload.get("request_id") or response_payload.get("id") or header_request_id or client_request_id),
            }
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"LLM结构化生成失败：{exc}") from exc

    @staticmethod
    def _structured_text(
        payload: dict[str, Any],
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM响应中没有结构化文本")
        text = content.strip()
        if strict:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise LLMProviderError("LLM结构化响应不是JSON对象")
            return parsed
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise LLMProviderError("LLM结构化响应不是JSON对象")
        return parsed
