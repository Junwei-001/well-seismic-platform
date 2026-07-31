from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "是", "启用"}


def _env(name: str, fallback: Any) -> Any:
    value = os.getenv(name)
    return fallback if value is None else value


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _load_local_env() -> None:
    """Load an ignored project-local .env without overriding process environment variables."""
    path = Path(__file__).resolve().parents[3] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    provider: str
    api_mode: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int
    min_confidence: float
    max_calls_per_task: int
    max_context_chars: int
    send_file_names: bool
    use_system_proxy: bool
    audit_decisions: bool
    allowed_decisions: tuple[str, ...]
    organization: str = ""
    project: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    @property
    def available(self) -> bool:
        return self.enabled and self.configured

    def public_status(self) -> dict[str, Any]:
        missing = []
        if not self.api_key:
            missing.append("GLM_API_KEY" if self.provider == "glm" else "WELL_SEISMIC_LLM_API_KEY")
        if not self.model:
            missing.append("WELL_SEISMIC_LLM_MODEL")
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "available": self.available,
            "provider": self.provider,
            "api_mode": self.api_mode,
            "base_url": self.base_url,
            "model": self.model or "未配置",
            "api_key_configured": bool(self.api_key),
            "min_confidence": self.min_confidence,
            "max_calls_per_task": self.max_calls_per_task,
            "max_context_chars": self.max_context_chars,
            "send_file_names": self.send_file_names,
            "use_system_proxy": self.use_system_proxy,
            "allowed_decisions": list(self.allowed_decisions),
            "missing": missing,
            "data_policy": "仅发送字段名、单位、候选项和统计摘要；不发送SEG-Y振幅体或完整LAS样点",
        }


def load_llm_settings(config: dict[str, Any]) -> LLMSettings:
    _load_local_env()
    raw = config.get("llm", {})
    api_key = _first_env(
        "WELL_SEISMIC_LLM_API_KEY",
        "GLM_API_KEY",
        "ZHIPUAI_API_KEY",
        "OPENAI_API_KEY",
    )
    return LLMSettings(
        enabled=_bool(_env("WELL_SEISMIC_LLM_ENABLED", raw.get("enabled", False))),
        provider=str(_env("WELL_SEISMIC_LLM_PROVIDER", raw.get("provider", "glm"))).strip().lower(),
        api_mode=str(_env("WELL_SEISMIC_LLM_API_MODE", raw.get("api_mode", "chat_completions"))).strip().lower(),
        base_url=str(
            _env("WELL_SEISMIC_LLM_BASE_URL", raw.get("base_url", "https://open.bigmodel.cn/api/paas/v4"))
        ).rstrip("/"),
        model=str(_env("WELL_SEISMIC_LLM_MODEL", raw.get("model", ""))).strip(),
        api_key=api_key,
        timeout_seconds=max(1.0, float(_env("WELL_SEISMIC_LLM_TIMEOUT_SECONDS", raw.get("timeout_seconds", 25)))),
        max_retries=max(0, int(_env("WELL_SEISMIC_LLM_MAX_RETRIES", raw.get("max_retries", 2)))),
        max_output_tokens=max(64, int(_env("WELL_SEISMIC_LLM_MAX_OUTPUT_TOKENS", raw.get("max_output_tokens", 240)))),
        min_confidence=min(1.0, max(0.5, float(_env("WELL_SEISMIC_LLM_MIN_CONFIDENCE", raw.get("min_confidence", 0.82))))),
        max_calls_per_task=max(0, int(_env("WELL_SEISMIC_LLM_MAX_CALLS_PER_TASK", raw.get("max_calls_per_task", 24)))),
        max_context_chars=max(1000, int(_env("WELL_SEISMIC_LLM_MAX_CONTEXT_CHARS", raw.get("max_context_chars", 6000)))),
        send_file_names=_bool(_env("WELL_SEISMIC_LLM_SEND_FILE_NAMES", raw.get("send_file_names", False))),
        use_system_proxy=_bool(
            _env("WELL_SEISMIC_LLM_USE_SYSTEM_PROXY", raw.get("use_system_proxy", False))
        ),
        audit_decisions=_bool(raw.get("audit_decisions", True), True),
        allowed_decisions=tuple(str(item) for item in raw.get("allowed_decisions", ["metadata_role", "curve_mapping"])),
        organization=os.getenv("OPENAI_ORGANIZATION", "").strip(),
        project=os.getenv("OPENAI_PROJECT", "").strip(),
    )
