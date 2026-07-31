from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMDecisionRequest:
    decision_type: str
    question: str
    options: tuple[str, ...]
    evidence: dict[str, Any]
    source_hash: str


@dataclass
class LLMDecision:
    decision_type: str
    choice: str
    confidence: float
    reason: str
    accepted: bool = False
    provider: str = ""
    model: str = ""
    request_id: str = ""
    source_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "判断类型": self.decision_type,
            "选择": self.choice,
            "置信度": round(float(self.confidence), 4),
            "是否采纳": self.accepted,
            "理由摘要": self.reason,
            "提供方": self.provider,
            "模型": self.model,
            "请求ID": self.request_id,
            "来源摘要": self.source_hash,
            "警告": self.warnings,
        }


class LLMProvider(Protocol):
    provider_name: str
    model: str

    def decide(self, request: LLMDecisionRequest) -> LLMDecision:
        ...

