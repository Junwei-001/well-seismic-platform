from __future__ import annotations

from typing import Any, Protocol

from .providers import OpenAICompatibleChatProvider, OpenAIResponsesProvider
from .settings import LLMSettings


class StructuredGenerator(Protocol):
    provider_name: str
    model: str

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        ...


def build_structured_generator(settings: LLMSettings) -> StructuredGenerator | None:
    if not settings.available:
        return None
    if settings.api_mode == "responses":
        return OpenAIResponsesProvider(settings)
    if settings.api_mode == "chat_completions":
        return OpenAICompatibleChatProvider(settings)
    return None
