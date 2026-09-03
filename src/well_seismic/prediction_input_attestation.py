"""Attest whether a prediction runner actually consumed registration inputs.

Making a sealed registration product available to a runner is not evidence that
the runner joined it to model features.  This module turns an explicit runner
receipt into a small, model-neutral audit record.  Missing or incomplete
receipts remain compatible with legacy runners, but are never reported as
``used``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

CONTRACT_VERSION = "well-seismic.prediction-registration-consumption.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_USED_STATUSES = {"consumed", "used"}
_NOT_USED_STATUSES = {
    "available_not_used",
    "ignored",
    "lineage_only",
    "not_requested",
    "not_used",
}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _explicit_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _receipt_claim(receipt: Mapping[str, Any]) -> bool | None:
    for key in ("registration_consumed", "consumed"):
        claim = _explicit_bool(receipt.get(key))
        if claim is not None:
            return claim
    status = str(receipt.get("status") or "").strip().casefold()
    if status in _USED_STATUSES:
        return True
    if status in _NOT_USED_STATUSES:
        return False
    # A dedicated receipt containing join evidence is itself an explicit
    # consumption claim.  An empty metadata object is not.
    evidence_keys = {
        "feature_channels",
        "join_coverage_fraction",
        "joined_row_count",
        "joined_rows",
        "joined_well_ids",
        "joined_wells",
    }
    return True if evidence_keys.intersection(receipt) else None


def _runner_claims(
    result: Mapping[str, Any],
) -> tuple[list[tuple[str, bool]], list[tuple[str, Mapping[str, Any]]]]:
    claims: list[tuple[str, bool]] = []
    receipts: list[tuple[str, Mapping[str, Any]]] = []
    for container_name in ("input", "provenance"):
        container = _mapping(result.get(container_name))
        if container is None:
            continue
        explicit = _explicit_bool(container.get("registration_consumed"))
        if explicit is not None:
            claims.append((f"{container_name}.registration_consumed", explicit))
        receipt = _mapping(container.get("registration_consumption"))
        if receipt is None:
            continue
        receipts.append((f"{container_name}.registration_consumption", receipt))
        receipt_claim = _receipt_claim(receipt)
        if receipt_claim is not None:
            claims.append(
                (f"{container_name}.registration_consumption", receipt_claim)
            )
    return claims, receipts


def _first(receipt: Mapping[str, Any], *keys: str) -> Any:
    return next((receipt[key] for key in keys if key in receipt), None)


def _well_ids(value: Any) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    values = [str(item).strip() for item in value]
    if not values or any(not item for item in values) or len(set(values)) != len(values):
        return None
    return values


def _feature_channels(value: Any) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    values = [str(item).strip() for item in value]
    if not values or any(not item for item in values):
        return None
    return values


def _positive_integer(value: Any) -> int | None:
    if type(value) is int and value > 0:
        return value
    return None


def _coverage(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 < number <= 1.0 else None


def _sha256(value: Any) -> str | None:
    digest = str(value or "").strip().casefold()
    return digest if _SHA256.fullmatch(digest) else None


def _validate_used_receipt(
    receipt: Mapping[str, Any],
    registration_context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    manifest_sha = _sha256(
        _first(
            receipt,
            "registration_manifest_sha256",
            "manifest_sha256",
        )
    )
    points_sha = _sha256(
        _first(
            receipt,
            "registration_points_sha256",
            "points_sha256",
        )
    )
    joined_well_ids = _well_ids(
        _first(receipt, "joined_well_ids", "joined_wells")
    )
    joined_row_count = _positive_integer(
        _first(receipt, "joined_row_count", "joined_rows")
    )
    coverage = _coverage(
        _first(receipt, "join_coverage_fraction", "coverage_fraction")
    )
    feature_channels = _feature_channels(receipt.get("feature_channels"))

    issues: list[str] = []
    for value, label in (
        (manifest_sha, "registration_manifest_sha256"),
        (points_sha, "registration_points_sha256"),
        (joined_well_ids, "joined_well_ids"),
        (joined_row_count, "joined_row_count"),
        (coverage, "join_coverage_fraction"),
        (feature_channels, "feature_channels"),
    ):
        if value is None:
            issues.append(f"missing_or_invalid:{label}")

    expected_manifest_sha = _sha256(
        registration_context.get("registration_manifest_sha256")
    )
    expected_points_sha = _sha256(
        registration_context.get("registration_points_sha256")
    )
    if manifest_sha is not None and manifest_sha != expected_manifest_sha:
        issues.append("registration_manifest_sha256_mismatch")
    if points_sha is not None and points_sha != expected_points_sha:
        issues.append("registration_points_sha256_mismatch")

    allowed_wells = {
        str(item)
        for item in (
            registration_context.get("registration_fusion_ready_well_ids") or ()
        )
    }
    if joined_well_ids is not None and (
        not allowed_wells or not set(joined_well_ids).issubset(allowed_wells)
    ):
        issues.append("joined_well_ids_not_in_fusion_ready_registration")
    if (
        joined_well_ids is not None
        and joined_row_count is not None
        and joined_row_count < len(joined_well_ids)
    ):
        issues.append("joined_row_count_smaller_than_joined_well_count")

    evidence = {
        "registration_manifest_sha256": manifest_sha,
        "registration_points_sha256": points_sha,
        "joined_well_ids": joined_well_ids,
        "joined_row_count": joined_row_count,
        "join_coverage_fraction": coverage,
        "feature_channels": feature_channels,
    }
    product_sha = _sha256(registration_context.get("registration_product_sha256"))
    product_role = str(
        registration_context.get("registration_product_role") or ""
    ).strip()
    if product_sha:
        evidence["registration_product_sha256"] = product_sha
    if product_role:
        evidence["registration_product_role"] = product_role
    return evidence, list(dict.fromkeys(issues))


def attest_prediction_registration_consumption(
    result: Mapping[str, Any],
    *,
    registration_requested: bool,
    registration_context: Mapping[str, Any] | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return an honest registration-availability and consumption decision."""

    context = dict(registration_context or {})
    available = bool(
        registration_requested
        and context.get("registration_manifest_sha256")
        and context.get("registration_points_sha256")
    )
    claims, receipts = _runner_claims(result)
    claim_values = {claim for _, claim in claims}
    claim_sources = [source for source, _ in claims]
    issues: list[str] = []

    if not registration_requested:
        # Raw/fast adapters may truthfully emit ``registration_consumed=false``
        # for every run.  That is not a claim that an unrequested registration
        # existed.  Only a positive claim is anomalous in this branch.
        if True in claim_values:
            issues.append("runner_reported_registration_without_request")
        status = "not_requested"
        evidence = None
    elif not available:
        status = "unattested"
        evidence = None
        issues.append("registration_requested_but_sealed_context_unavailable")
    elif len(claim_values) > 1:
        status = "unattested"
        evidence = None
        issues.append("conflicting_runner_consumption_claims")
    elif claim_values == {True}:
        if not receipts:
            status = "unattested"
            evidence = None
            issues.append(
                "used_claim_requires_registration_consumption_receipt"
            )
        else:
            validations = [
                _validate_used_receipt(receipt, context) for _, receipt in receipts
            ]
            validated = [item[0] for item in validations]
            for _, receipt_issues in validations:
                issues.extend(receipt_issues)
            evidence = validated[0]
            if any(item != evidence for item in validated[1:]):
                issues.append("conflicting_registration_consumption_receipts")
            issues = list(dict.fromkeys(issues))
            status = "used" if not issues else "unattested"
    elif str(task_id or "").casefold() == "horizon":
        # Platform registration rows contain MD/TWT lineage, not named horizon
        # picks.  A horizon runner must submit a full positive receipt before
        # this can become numerical consumption.
        status = "lineage_only"
        evidence = None
    elif claim_values == {False}:
        status = "available_not_used"
        evidence = None
    else:
        status = "unattested"
        evidence = None
        issues.append("runner_did_not_attest_registration_consumption")

    return {
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "registration_requested": bool(registration_requested),
        "registration_available": available,
        "registration_consumed": status == "used",
        "claims": [
            {"source": source, "registration_consumed": claim}
            for source, claim in claims
        ],
        "claim_sources": claim_sources,
        "evidence": evidence,
        "issues": issues,
    }


__all__ = [
    "CONTRACT_VERSION",
    "attest_prediction_registration_consumption",
]
