from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_KIMI_API_HOSTS = frozenset({"api.moonshot.cn", "api.moonshot.ai"})


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
    path = _project_env_path()
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


def _project_env_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".env"


def validate_kimi_base_url(value: str) -> str:
    """Return a canonical Kimi API URL or reject non-official network targets."""
    candidate = str(value).strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Kimi base_url不是合法HTTPS地址") from exc
    hostname = (parsed.hostname or "").casefold()
    valid = (
        parsed.scheme.casefold() == "https"
        and hostname in _KIMI_API_HOSTS
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.path == "/v1"
        and not parsed.query
        and not parsed.fragment
    )
    if not valid:
        raise ValueError(
            "Kimi base_url只允许https://api.moonshot.cn/v1或"
            "https://api.moonshot.ai/v1"
        )
    return f"https://{hostname}/v1"


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
    reasoning_effort: str = "high"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    @property
    def available(self) -> bool:
        return self.enabled and self.configured

    def public_status(self) -> dict[str, Any]:
        missing = []
        if not self.api_key:
            key_names = {
                "kimi": "KIMI_API_KEY（兼容 MOONSHOT_API_KEY）",
                "glm": "GLM_API_KEY",
            }
            missing.append(key_names.get(self.provider, "WELL_SEISMIC_LLM_API_KEY"))
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
            "reasoning_effort": self.reasoning_effort,
            "api_key_configured": bool(self.api_key),
            "credential_variable": "KIMI_API_KEY" if self.provider == "kimi" else "WELL_SEISMIC_LLM_API_KEY",
            "credential_file": str(_project_env_path()),
            "credential_template_file": str(_project_env_path().with_name(".env.example")),
            "credential_policy": "密钥仅由后端环境读取；状态接口只返回是否已配置，绝不返回密钥内容",
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
    provider = str(_env("WELL_SEISMIC_LLM_PROVIDER", raw.get("provider", "kimi"))).strip().lower()
    if provider == "kimi":
        api_key = _first_env("WELL_SEISMIC_LLM_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")
    elif provider == "glm":
        api_key = _first_env("WELL_SEISMIC_LLM_API_KEY", "GLM_API_KEY", "ZHIPUAI_API_KEY")
    else:
        api_key = _first_env("WELL_SEISMIC_LLM_API_KEY", "OPENAI_API_KEY")
    base_url = _first_env("WELL_SEISMIC_LLM_BASE_URL", "KIMI_BASE_URL") or str(
        raw.get("base_url", "https://api.moonshot.cn/v1")
    )
    base_url = (
        validate_kimi_base_url(base_url)
        if provider == "kimi"
        else base_url.rstrip("/")
    )
    model = _first_env("WELL_SEISMIC_LLM_MODEL", "KIMI_MODEL") or str(raw.get("model", "kimi-k3"))
    reasoning_effort = str(
        _env("WELL_SEISMIC_LLM_REASONING_EFFORT", raw.get("reasoning_effort", "high"))
    ).strip().lower()
    if reasoning_effort not in {"low", "high", "max"}:
        reasoning_effort = "high"
    return LLMSettings(
        enabled=_bool(_env("WELL_SEISMIC_LLM_ENABLED", raw.get("enabled", False))),
        provider=provider,
        api_mode=str(_env("WELL_SEISMIC_LLM_API_MODE", raw.get("api_mode", "chat_completions"))).strip().lower(),
        base_url=base_url,
        model=model.strip(),
        api_key=api_key,
        timeout_seconds=max(1.0, float(_env("WELL_SEISMIC_LLM_TIMEOUT_SECONDS", raw.get("timeout_seconds", 60)))),
        max_retries=max(0, int(_env("WELL_SEISMIC_LLM_MAX_RETRIES", raw.get("max_retries", 2)))),
        max_output_tokens=max(
            64,
            int(_env("WELL_SEISMIC_LLM_MAX_OUTPUT_TOKENS", raw.get("max_output_tokens", 4096))),
        ),
        min_confidence=min(1.0, max(0.5, float(_env("WELL_SEISMIC_LLM_MIN_CONFIDENCE", raw.get("min_confidence", 0.82))))),
        max_calls_per_task=max(0, int(_env("WELL_SEISMIC_LLM_MAX_CALLS_PER_TASK", raw.get("max_calls_per_task", 24)))),
        max_context_chars=max(
            1000,
            int(_env("WELL_SEISMIC_LLM_MAX_CONTEXT_CHARS", raw.get("max_context_chars", 16000))),
        ),
        send_file_names=_bool(_env("WELL_SEISMIC_LLM_SEND_FILE_NAMES", raw.get("send_file_names", False))),
        use_system_proxy=_bool(
            _env("WELL_SEISMIC_LLM_USE_SYSTEM_PROXY", raw.get("use_system_proxy", False))
        ),
        audit_decisions=_bool(raw.get("audit_decisions", True), True),
        allowed_decisions=tuple(str(item) for item in raw.get("allowed_decisions", ["metadata_role", "curve_mapping"])),
        organization=os.getenv("OPENAI_ORGANIZATION", "").strip(),
        project=os.getenv("OPENAI_PROJECT", "").strip(),
        reasoning_effort=reasoning_effort,
    )
