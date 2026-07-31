from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import LLMDecision, LLMDecisionRequest, LLMProvider
from .providers import OpenAICompatibleChatProvider, OpenAIResponsesProvider
from .settings import LLMSettings, load_llm_settings


class DecisionResolver:
    """Rule fallback orchestrator. It never accepts options outside the caller's allow-list."""

    def __init__(self, settings: LLMSettings, provider: LLMProvider | None, requested: bool):
        self.settings = settings
        self.provider = provider
        self.requested = requested
        self.records: list[dict[str, Any]] = []
        self._cache: dict[str, LLMDecision] = {}
        self._calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.requested and self.settings.available and self.provider)

    def resolve(
        self,
        decision_type: str,
        question: str,
        options: list[str],
        evidence: dict[str, Any],
    ) -> LLMDecision | None:
        if not self.enabled or decision_type not in self.settings.allowed_decisions:
            return None
        unique_options = tuple(dict.fromkeys(str(item) for item in options if str(item).strip()))
        if len(unique_options) < 2 or self._calls >= self.settings.max_calls_per_task:
            return None
        compact = self._compact(evidence)
        digest = hashlib.sha256(
            json.dumps([decision_type, unique_options, compact], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if digest in self._cache:
            return self._cache[digest]
        request = LLMDecisionRequest(decision_type, question, unique_options, compact, digest)
        self._calls += 1
        try:
            assert self.provider is not None
            decision = self.provider.decide(request)
            decision.accepted = (
                decision.choice in unique_options
                and decision.confidence >= self.settings.min_confidence
            )
            if decision.choice not in unique_options:
                decision.warnings.append("模型返回了候选集合之外的值，已拒绝采纳")
                decision.choice = ""
            elif not decision.accepted:
                decision.warnings.append(
                    f"置信度低于采纳阈值 {self.settings.min_confidence:.2f}，保留原规则结果"
                )
        except Exception as exc:
            decision = LLMDecision(
                decision_type=decision_type,
                choice="",
                confidence=0.0,
                reason="外部判断失败，已安全回退到规则结果",
                provider=self.settings.provider,
                model=self.settings.model,
                source_hash=digest,
                warnings=[str(exc)[:300]],
            )
        self._cache[digest] = decision
        if self.settings.audit_decisions:
            self.records.append(decision.to_audit_dict())
        return decision

    def resolve_metadata(self, path: Path, result: Any) -> LLMDecision | None:
        candidates: list[str] = []
        if result.heads:
            candidates.append("井位与海拔")
        if result.trajectories:
            candidates.append("井轨迹")
        if result.time_depth and not any("未归属井" in role for role in result.detected_roles):
            candidates.append("时深关系")
        candidates.append("保留待确认")
        return self.resolve(
            "metadata_role",
            "根据结构证据选择该井相关文件的唯一主角色。证据不足必须保留待确认。",
            candidates,
            {
                "file": path.name if self.settings.send_file_names else f"*{path.suffix.lower()}",
                "rule_roles": list(result.detected_roles),
                "rule_confidence": round(float(result.confidence), 4),
                "evidence": list(result.evidence),
                "available_payloads": {
                    "well_heads": len(result.heads),
                    "trajectories": len(result.trajectories),
                    "time_depth_groups": len(result.time_depth),
                },
            },
        )

    def resolve_curve(
        self,
        *,
        mnemonic: str,
        unit: str,
        description: str,
        values: np.ndarray,
        candidates: list[dict[str, Any]],
    ) -> LLMDecision | None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        stats = {
            "count": int(finite.size),
            "min": float(np.min(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
        options = [str(item["standard_name"]) for item in candidates]
        options.append("保留原始曲线")
        return self.resolve(
            "curve_mapping",
            "将未识别测井曲线映射到最可能的标准曲线；不能确定时保留原始曲线。",
            options,
            {
                "mnemonic": mnemonic,
                "unit": unit or "unknown",
                "description": description[:300],
                "statistics": stats,
                "rule_candidates": candidates,
            },
        )

    def resolve_issue_action(
        self,
        issue: dict[str, Any],
        candidates: list[str],
    ) -> LLMDecision | None:
        """从代码给定的安全处理方案中推荐一项，不执行任何数据修改。"""
        return self.resolve(
            "issue_action",
            "根据问题证据，从安全候选方案中推荐最稳妥的一项。只做建议，等待人工确认。",
            candidates,
            {
                "stage": issue.get("stage"),
                "severity": issue.get("severity"),
                "blocking": bool(issue.get("blocking")),
                "title": issue.get("title"),
                "message": issue.get("message"),
                "source_type": Path(str(issue.get("source", ""))).suffix.lower() or "workflow",
            },
        )

    def _compact(self, evidence: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(evidence, ensure_ascii=False, default=str)
        if len(serialized) <= self.settings.max_context_chars:
            return evidence
        return {
            "truncated_summary": serialized[: self.settings.max_context_chars],
            "truncated": True,
        }


def build_decision_resolver(
    config: dict[str, Any],
    *,
    requested: bool = False,
    provider: LLMProvider | None = None,
) -> DecisionResolver:
    settings = load_llm_settings(config)
    active_provider = provider
    if active_provider is None and requested and settings.available:
        if settings.provider not in {"glm", "openai", "openai_compatible"}:
            raise ValueError(f"暂不支持的LLM提供方：{settings.provider}")
        if settings.api_mode == "responses":
            active_provider = OpenAIResponsesProvider(settings)
        elif settings.api_mode == "chat_completions":
            active_provider = OpenAICompatibleChatProvider(settings)
        else:
            raise ValueError(f"暂不支持的LLM接口模式：{settings.api_mode}")
    return DecisionResolver(settings, active_provider, requested)
