from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..content_identity import canonical_sha256
from .generation import StructuredGenerator
from .privacy import issue_local_paths, sanitize_llm_payload


# Persistent transformations are intentionally narrower than the in-memory
# decision helpers.  A model may select an existing unit conversion, but it may
# not invent aliases, null sentinels, or new numerical transforms that would
# affect every later task after activation.
ALLOWED_OPERATIONS = {"unit_scale"}
TRANSFORMATION_POLICY_CONTRACT_VERSION = "well-seismic.llm-transformation-policy.v1"

TRANSFORMATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "explanation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "operations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": sorted(ALLOWED_OPERATIONS)},
                    "target": {"type": "string"},
                    "from_value": {"type": "string"},
                    "to_value": {"type": "string"},
                    "scale": {"type": "number"},
                    "offset": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["op", "target", "from_value", "to_value", "scale", "offset", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "explanation", "confidence", "operations"],
    "additionalProperties": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transformation_config_sha256(config: dict[str, Any]) -> str:
    """Bind a draft to the transformation policy that validated it."""

    return canonical_sha256(
        {
            "contract_version": TRANSFORMATION_POLICY_CONTRACT_VERSION,
            "allowed_operations": sorted(ALLOWED_OPERATIONS),
            "conversions": config.get("conversions", {}),
        }
    )


def _operations_sha256(operations: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        {
            "contract_version": TRANSFORMATION_POLICY_CONTRACT_VERSION,
            "operations": operations,
        }
    )


def _known_unit_operation(message: str, config: dict[str, Any]) -> dict[str, Any] | None:
    match = re.search(r"unit_conversion_unavailable:(.+?)->(.+)$", message.strip())
    if not match:
        return None
    source, target = (part.strip() for part in match.groups())
    rule = config.get("conversions", {}).get(f"{source}->{target}")
    if not rule:
        return None
    return {
        "op": "unit_scale",
        "target": "测井曲线",
        "from_value": source,
        "to_value": target,
        "scale": float(rule.get("scale", 1.0)),
        "offset": float(rule.get("offset", 0.0)),
        "reason": "单位知识库已有可验证换算",
    }


def _fallback_plan(issue: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    known_unit = _known_unit_operation(str(issue.get("message", "")), config)
    operations = [known_unit] if known_unit else []
    return {
        "title": "受控转换适配器草案" if operations else "保留待人工建模",
        "explanation": (
            "平台知识库已编译出确定性转换；启用前仍会执行样例测试。"
            if operations
            else "当前证据不足以自动生成安全转换，未创建任何数据修改操作。"
        ),
        "confidence": 1.0 if operations else 0.0,
        "operations": operations,
    }


def _validate_operation(
    operation: dict[str, Any],
    conversions: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    op = str(operation.get("op", ""))
    if op not in ALLOWED_OPERATIONS:
        return [f"不允许的操作：{op or '空'}"]
    if not str(operation.get("from_value", "")).strip() or not str(operation.get("to_value", "")).strip():
        errors.append("转换源值和目标值不能为空")
    try:
        scale = float(operation.get("scale", 1.0))
        offset = float(operation.get("offset", 0.0))
        if not math.isfinite(scale) or not math.isfinite(offset):
            errors.append("数值参数必须有限")
        if abs(scale) < 1e-12 or abs(scale) > 1e12:
            errors.append("比例因子超出安全范围")
        if op != "unit_scale" and (scale != 1.0 or offset != 0.0):
            errors.append("非单位操作不得携带数值变换")
        if op == "unit_scale":
            source = str(operation.get("from_value", "")).strip()
            target = str(operation.get("to_value", "")).strip()
            rule = (conversions or {}).get(f"{source}->{target}")
            if not isinstance(rule, dict):
                errors.append("单位换算不在平台知识库白名单中")
            else:
                expected_scale = float(rule.get("scale", 1.0))
                expected_offset = float(rule.get("offset", 0.0))
                if not math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append("比例因子与平台知识库不一致")
                if not math.isclose(offset, expected_offset, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append("偏移量与平台知识库不一致")
    except (TypeError, ValueError):
        errors.append("比例或偏移不是有效数字")
    return errors


def _auto_tests(
    operations: list[dict[str, Any]],
    conversions: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for index, operation in enumerate(operations, start=1):
        errors = _validate_operation(operation, conversions)
        if operation.get("op") == "unit_scale" and not errors:
            scale = float(operation["scale"])
            offset = float(operation["offset"])
            values = [0.0, 1.0, 1000.0]
            outputs = [value * scale + offset for value in values]
            if not all(math.isfinite(value) for value in outputs):
                errors.append("样例输出包含非有限值")
            if scale <= 0:
                errors.append("物理单位换算必须保持单调递增")
        tests.append({
            "name": f"操作 {index} · {operation.get('op', 'unknown')}",
            "passed": not errors,
            "details": "；".join(errors) if errors else "结构、数值范围与样例执行通过",
        })
    if not operations:
        tests.append({"name": "安全空操作检查", "passed": False, "details": "没有可启用的转换操作"})
    return tests


def _code_preview(operations: list[dict[str, Any]]) -> str:
    lines = ["# 地层慧眼受控转换适配器 v1", "def transform(adapter):"]
    if not operations:
        lines.append("    pass  # 证据不足，不修改数据")
    for operation in operations:
        args = (
            f"target={operation['target']!r}, from_value={operation['from_value']!r}, "
            f"to_value={operation['to_value']!r}, scale={float(operation['scale'])!r}, "
            f"offset={float(operation['offset'])!r}"
        )
        lines.append(f"    adapter.{operation['op']}({args})")
    return "\n".join(lines)


def create_transformation_draft(
    *,
    task_id: str,
    issue: dict[str, Any],
    config: dict[str, Any],
    generator: StructuredGenerator | None,
) -> dict[str, Any]:
    plan: dict[str, Any] | None = None
    metadata = {"provider": "平台规则编译器", "model": "deterministic-v1", "request_id": ""}
    generation_error = ""
    if generator is not None:
        try:
            outbound_issue = sanitize_llm_payload(
                {
                    "stage": issue.get("stage"),
                    "severity": issue.get("severity"),
                    "title": issue.get("title"),
                    "message": issue.get("message"),
                },
                known_paths=issue_local_paths(issue),
            )
            plan, metadata = generator.generate_json(
                system_prompt=(
                    "你是井震数据工程转换适配器生成器。只能输出给定JSON契约中的白名单操作；"
                    "不得生成脚本、表达式、文件操作、网络调用，不得猜测坐标、深度或时深关系。"
                    "证据不足时operations返回空数组。所有数值都将被本地校验并等待人工启用。"
                ),
                payload={
                    **outbound_issue,
                    "allowed_operations": sorted(ALLOWED_OPERATIONS),
                    "known_unit_conversions": config.get("conversions", {}),
                },
                schema_name="well_seismic_transformation_adapter",
                schema=TRANSFORMATION_SCHEMA,
            )
        except Exception as exc:
            generation_error = str(exc)[:400]
    if not isinstance(plan, dict):
        plan = _fallback_plan(issue, config)

    raw_operations = plan.get("operations", [])
    operations = [dict(item) for item in raw_operations if isinstance(item, dict)][:6]
    tests = _auto_tests(operations, config.get("conversions", {}))
    valid = bool(operations) and all(test["passed"] for test in tests)
    config_sha256 = transformation_config_sha256(config)
    operations_sha256 = _operations_sha256(operations)
    return {
        "id": uuid.uuid4().hex,
        "task_id": task_id,
        "issue_id": issue.get("id"),
        "title": str(plan.get("title", "受控转换适配器草案"))[:120],
        "explanation": str(plan.get("explanation", ""))[:800],
        "confidence": max(0.0, min(1.0, float(plan.get("confidence", 0.0)))),
        "operations": operations,
        "config_sha256": config_sha256,
        "operations_sha256": operations_sha256,
        "validation_contract": TRANSFORMATION_POLICY_CONTRACT_VERSION,
        "generated_code": _code_preview(operations),
        "tests": tests,
        "valid": valid,
        "status": "待人工启用" if valid else "未通过验证",
        "provider": metadata.get("provider", ""),
        "model": metadata.get("model", ""),
        "request_id": metadata.get("request_id", ""),
        "generation_error": generation_error,
        "created_at": _utc_now(),
        "activated_at": "",
    }


def activate_transformation(
    draft: dict[str, Any],
    registry_path: Path,
    config: dict[str, Any],
) -> None:
    if not draft.get("valid"):
        raise ValueError("转换草案未通过自动验证")
    current_config_sha256 = transformation_config_sha256(config)
    if draft.get("config_sha256") != current_config_sha256:
        raise ValueError("转换配置已变化，请重新生成并验证草案")
    raw_operations = draft.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations or len(raw_operations) > 6:
        raise ValueError("转换草案操作结构无效")
    operations = [dict(item) for item in raw_operations if isinstance(item, dict)]
    if len(operations) != len(raw_operations):
        raise ValueError("转换草案操作结构无效")
    if draft.get("operations_sha256") != _operations_sha256(operations):
        raise ValueError("转换草案内容已变化，请重新生成并验证草案")
    tests = _auto_tests(operations, config.get("conversions", {}))
    if not all(test["passed"] for test in tests):
        raise ValueError("转换草案未通过当前配置的自动验证")
    draft["tests"] = tests
    draft["config_sha256"] = current_config_sha256
    draft["operations_sha256"] = _operations_sha256(operations)
    draft["validation_contract"] = TRANSFORMATION_POLICY_CONTRACT_VERSION
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if registry_path.is_file():
        loaded = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, list):
            existing = loaded
    draft["status"] = "已启用"
    draft["activated_at"] = _utc_now()
    existing = [item for item in existing if item.get("id") != draft.get("id")]
    existing.append(draft)
    registry_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_active_transformations(config: dict[str, Any], config_dir: str | Path) -> None:
    registry_path = Path(config_dir).resolve().parent / "输出结果" / "智能转换插件" / "已启用转换.json"
    if not registry_path.is_file():
        return
    try:
        drafts = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return
    for draft in drafts if isinstance(drafts, list) else []:
        if draft.get("status") != "已启用":
            continue
        for operation in draft.get("operations", []):
            op = operation.get("op")
            source = str(operation.get("from_value", "")).strip()
            target = str(operation.get("to_value", "")).strip()
            if not source or not target or _validate_operation(
                operation,
                config.get("conversions", {}),
            ):
                continue
            if op == "unit_scale":
                config.setdefault("conversions", {})[f"{source}->{target}"] = {
                    "scale": float(operation.get("scale", 1.0)),
                    "offset": float(operation.get("offset", 0.0)),
                }
            elif op == "curve_alias" and target in config.get("curve_knowledge", {}):
                aliases = config["curve_knowledge"][target].setdefault("aliases", {}).setdefault("exact", [])
                if source not in aliases:
                    aliases.append(source)
            elif op == "well_alias":
                config.setdefault("manifest", {}).setdefault("well_aliases", {})[source] = target
            elif op == "field_alias":
                config.setdefault("well_schema", {}).setdefault("fields", {}).setdefault(target, []).append(source)
            elif op == "null_value":
                try:
                    config.setdefault("preprocessing", {}).setdefault("null_values", []).append(float(source))
                except ValueError:
                    continue
