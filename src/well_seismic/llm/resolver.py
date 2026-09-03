from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import LLMDecision, LLMDecisionRequest, LLMProvider
from .providers import OpenAICompatibleChatProvider, OpenAIResponsesProvider
from .parse_repair import TRAJECTORY_PARSE_PATCH_SCHEMA
from .settings import LLMSettings, load_llm_settings


class DecisionResolver:
    """Rule fallback orchestrator. It never accepts options outside the caller's allow-list."""

    def __init__(self, settings: LLMSettings, provider: LLMProvider | None, requested: bool):
        self.settings = settings
        self.provider = provider
        self.requested = requested
        self.records: list[dict[str, Any]] = []
        self._cache: dict[str, LLMDecision] = {}
        self._structured_cache: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
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
        """Select one bounded autofill action; callers validate it before applying."""
        return self.resolve(
            "issue_action",
            "根据问题证据选择最稳妥的结构化补全方案。不得虚构坐标系、datum或单位；证据充分时由规则校验后自动应用。",
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

    def resolve_file_role(
        self,
        *,
        path: Path,
        root: Path,
        suffix: str,
        size: int,
        header_tokens: list[str],
        rule_category: str,
        rule_reason: str,
    ) -> LLMDecision | None:
        """Classify one ambiguous local file within a fixed, auditable role list."""
        evidence: dict[str, Any] = {
            "suffix": suffix,
            "size_bytes": int(size),
            "header_fields": header_tokens[:40],
            "rule_candidate": rule_category,
            "rule_reason": rule_reason,
        }
        if self.settings.send_file_names:
            evidence["relative_path"] = str(path.relative_to(root)).replace("\\", "/")
        return self.resolve(
            "input_file_role",
            "判断本地数据文件最可能的数据角色；证据不足时必须选择待人工分类。",
            [
                "地震数据",
                "测区网格与坐标",
                "测井曲线",
                "井位、海拔与井轨迹",
                "解释成果与标签",
                "其他辅助数据",
                "待人工分类",
            ],
            evidence,
        )

    def resolve_trajectory_parse_patch(
        self,
        *,
        original_error: str,
        evidence: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]] | None:
        """Request one schema-bound parser patch, never code or file actions.

        ``issue_action`` remains the feature-policy gate so existing deployments
        do not gain a new external-call category merely by upgrading.  The
        returned JSON is still untrusted and must pass ``parse_repair``'s local
        allow-list plus a fresh parser/physics validation before use.
        """

        if (
            not self.enabled
            or "issue_action" not in self.settings.allowed_decisions
            or self._calls >= self.settings.max_calls_per_task
        ):
            return None
        generator = getattr(self.provider, "generate_json", None)
        if not callable(generator):
            return None
        compact = self._compact(evidence)
        digest = hashlib.sha256(
            json.dumps(
                ["trajectory_parse_patch", str(original_error)[:500], compact],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        cached = self._structured_cache.get(digest)
        if cached is not None:
            return cached
        self._calls += 1
        try:
            proposal, metadata = generator(
                system_prompt=(
                    "你是井轨迹文本解析补丁生成器。只能返回给定JSON Schema中的字段；"
                    "只允许选择分隔符、已有列的索引以及来源长度单位m/ft。"
                    "不得输出代码、命令、路径、文件操作、网络操作、井名、坐标值、深度值或默认零；"
                    "不得补造TVDSS基准或符号。证据不足的单位必须返回unknown。"
                    "原文件将保持只读，补丁会在内存重解析并经过确定性物理门校验。"
                ),
                payload={
                    "asset_role": "trajectory",
                    "original_error": str(original_error)[:500],
                    "structural_evidence": compact,
                    "allowed_patch_surface": ["delimiter", "columns", "field_units"],
                    "unknown_policy": "return_unknown_never_zero",
                },
                schema_name="well_seismic_trajectory_parse_patch",
                schema=TRAJECTORY_PARSE_PATCH_SCHEMA,
            )
            if not isinstance(proposal, dict):
                raise TypeError("structured parse patch is not a JSON object")
            metadata = {
                "provider": str(metadata.get("provider") or ""),
                "model": str(metadata.get("model") or ""),
                "request_id": str(metadata.get("request_id") or ""),
                "source_hash": digest,
            }
            result = (proposal, metadata)
            self._structured_cache[digest] = result
            if self.settings.audit_decisions:
                confidence = proposal.get("confidence", 0.0)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                self.records.append(
                    {
                        "判断类型": "trajectory_parse_patch",
                        "选择": "结构化解析补丁",
                        "置信度": round(confidence, 4),
                        "是否采纳": False,
                        "理由摘要": str(proposal.get("reason") or "")[:500],
                        "提供方": metadata["provider"],
                        "模型": metadata["model"],
                        "请求ID": metadata["request_id"],
                        "来源摘要": digest,
                        "警告": [
                            str(item)[:200]
                            for item in (
                                proposal.get("warnings")
                                if isinstance(proposal.get("warnings"), list)
                                else []
                            )[:8]
                        ],
                        "验证状态": "等待本地格式与物理门",
                    }
                )
            return result
        except Exception as exc:
            if self.settings.audit_decisions:
                self.records.append(
                    {
                        "判断类型": "trajectory_parse_patch",
                        "选择": "",
                        "置信度": 0.0,
                        "是否采纳": False,
                        "理由摘要": "结构化补丁生成失败，原读取错误保持不变",
                        "提供方": self.settings.provider,
                        "模型": self.settings.model,
                        "请求ID": "",
                        "来源摘要": digest,
                        "警告": [str(exc)[:300]],
                        "验证状态": "生成失败",
                    }
                )
            return None

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
        if settings.provider not in {"kimi", "glm", "openai", "openai_compatible"}:
            raise ValueError(f"暂不支持的LLM提供方：{settings.provider}")
        if settings.api_mode == "responses":
            active_provider = OpenAIResponsesProvider(settings)
        elif settings.api_mode == "chat_completions":
            active_provider = OpenAICompatibleChatProvider(settings)
        else:
            raise ValueError(f"暂不支持的LLM接口模式：{settings.api_mode}")
    return DecisionResolver(settings, active_provider, requested)
