"""Immutable source-snapshot manifests and semantic drift checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from .api_models import (
    SURVEY_ATTESTATION_CONTRACT_VERSION,
    survey_attestation_declaration,
)
from .content_identity import canonical_sha256, source_snapshot_fingerprint

SOURCE_SNAPSHOT_CONTRACT_VERSION = "well-seismic.source-snapshot.v3"
SYSTEM_EVIDENCE_RECEIPT_CONTRACT_VERSION = (
    "well-seismic.system-evidence-receipt.v1"
)
RUNTIME_CONTRACT_CONFIRMATION_VERSION = (
    "well-seismic.runtime-contract-confirmation.v1"
)
RUNTIME_CONTRACT_REVIEW_VERSION = "well-seismic.runtime-contract-review.v1"
REGISTRATION_EVIDENCE_CONTRACT_VERSION = (
    "well-seismic.registration-evidence.v1"
)
SYSTEM_EVIDENCE_MIN_CONFIDENCE = 0.95
SYSTEM_EVIDENCE_ALLOWED_RULE_VERSIONS = {
    "well-seismic.registration-preflight.rules.v1",
}
SYSTEM_EVIDENCE_ALLOWED_RULE_IDS = {
    "sealed_request_semantics.v1",
    "verified_auto_applied.vertical_crs_id.v1",
    "verified_auto_applied.seismic_srd_elevation_m.v1",
    "verified_auto_applied.seismic_time_domain.v1",
    "verified_auto_applied.seismic_correction_state.v1",
}

_SEMANTIC_FIELDS = (
    "recursive",
    "seismic_srd_elevation_m",
    "vertical_crs_id",
    "horizontal_crs_id",
    "well_source_crs_id",
    "seismic_source_crs_id",
    "horizontal_unit",
    "horizontal_axis_order",
    "coordinate_reference_verified",
    "seismic_replacement_velocity_mps",
    "seismic_time_domain",
    "seismic_correction_state",
    "segy_geometry_profile",
    "segy_inline_byte",
    "segy_crossline_byte",
    "segy_x_byte",
    "segy_y_byte",
    "segy_coordinate_scalar_byte",
    "well_coordinate_source_unit",
    "well_vertical_datum_source_unit",
    "las_twt_source_unit",
    "time_depth_default_depth_domain",
    "time_depth_default_depth_unit",
    "time_depth_default_time_unit",
    "time_depth_default_depth_datum",
    "time_depth_default_depth_convention",
    "time_depth_default_time_reference",
    "time_depth_default_time_domain",
    "time_depth_default_correction_state",
)

_POLICY_FIELDS = (
    "lightweight",
    "use_llm_fallback",
    "target_task_id",
    "target_model_id",
)


def _plain_request(request: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(request, Mapping):
        return dict(request)
    dumper = getattr(request, "model_dump", None)
    if callable(dumper):
        return dict(dumper(mode="json"))
    raise TypeError("snapshot request must be a mapping or Pydantic model")


def snapshot_semantics(
    request: Mapping[str, Any] | Any,
    *,
    effective_config_sha256: str | None = None,
    transformation_registry_sha256: str | None = None,
) -> dict[str, Any]:
    raw = _plain_request(request)
    semantics = {key: raw.get(key) for key in _SEMANTIC_FIELDS}
    semantics.update(
        {
            "horizontal_unit": str(
                raw.get("horizontal_unit") or "unknown"
            ).lower(),
            "horizontal_axis_order": str(
                raw.get("horizontal_axis_order") or "unknown"
            ).upper(),
            "seismic_time_domain": str(
                raw.get("seismic_time_domain") or "unknown"
            ).upper(),
            "seismic_correction_state": str(
                raw.get("seismic_correction_state") or "unknown"
            ).lower(),
            "effective_config_sha256": effective_config_sha256,
            "transformation_registry_sha256": transformation_registry_sha256,
        }
    )
    return semantics


def snapshot_inspection_policy(request: Mapping[str, Any] | Any) -> dict[str, Any]:
    raw = _plain_request(request)
    return {key: raw.get(key) for key in _POLICY_FIELDS}


def _asset_record(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "id",
            "role",
            "path",
            "format",
            "name",
            "size",
            "sha256",
            "geometry_fingerprint",
            "asset_options_sha256",
        )
        if item.get(key) is not None
    }


def _precomputed_seismic_bindings(
    source_assets: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bound_assets: list[dict[str, Any]] = []
    for item in source_assets:
        if str(item.get("role") or "").casefold() != "seismic":
            continue
        digest = str(item.get("sha256") or "").strip().casefold()
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(
                "survey attestation cannot bind a seismic asset without a valid "
                "precomputed full-file SHA-256"
            )
        bound_assets.append(
            {
                "asset_id": item.get("id"),
                "name": item.get("name"),
                "size": item.get("size"),
                "full_content_sha256": digest,
            }
        )
    bound_assets.sort(
        key=lambda item: (
            str(item.get("full_content_sha256") or ""),
            str(item.get("asset_id") or ""),
        )
    )
    return bound_assets


def survey_attestation_reuse_key(
    request: Mapping[str, Any] | Any,
    source_assets: Iterable[Mapping[str, Any]],
) -> str | None:
    """Return a path-independent key for safe reuse of a human declaration."""

    raw = _plain_request(request)
    try:
        srd_elevation = float(raw.get("seismic_srd_elevation_m"))
    except (TypeError, ValueError):
        return None
    if not isfinite(srd_elevation):
        return None
    if str(raw.get("seismic_time_domain") or "").upper() != "TWT":
        return None
    if (
        str(raw.get("seismic_correction_state") or "").lower()
        != "corrected_to_srd"
    ):
        return None
    vertical_crs_id = str(raw.get("vertical_crs_id") or "").strip()
    if "MSL" not in vertical_crs_id.upper():
        return None
    bound_assets = _precomputed_seismic_bindings(source_assets)
    if not bound_assets:
        return None
    return canonical_sha256(
        {
            "contract_version": SURVEY_ATTESTATION_CONTRACT_VERSION,
            "declaration_text": survey_attestation_declaration(srd_elevation),
            "declared_contract": {
                "vertical_reference": "MSL",
                "vertical_crs_id": vertical_crs_id,
                "seismic_srd_elevation_m": srd_elevation,
                "seismic_time_domain": "TWT",
                "seismic_correction_state": "corrected_to_srd",
            },
            "bound_segy_full_content_sha256s": [
                item["full_content_sha256"] for item in bound_assets
            ],
        }
    )


def _survey_attestation_receipt(
    *,
    snapshot_id: str,
    request: Mapping[str, Any] | Any,
    source_assets: Iterable[Mapping[str, Any]],
    recorded_at: str,
    reuse_basis: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bind a human survey declaration to already-computed SEG-Y digests.

    This function intentionally consumes only ``source_assets[].sha256``.  It
    never opens the source files, so sealing an attestation cannot trigger a
    second full read of a large SEG-Y volume after the inventory hashing pass.
    """

    raw = _plain_request(request)
    source_records = [dict(item) for item in source_assets]
    attestation = raw.get("survey_attestation")
    if attestation is None and reuse_basis is None:
        return None
    reused_from: dict[str, Any] | None = None
    if attestation is None:
        prior_receipt = reuse_basis.get("receipt") if reuse_basis else None
        prior_sha256 = str(
            (reuse_basis or {}).get("receipt_sha256") or ""
        ).casefold()
        if not isinstance(prior_receipt, Mapping):
            raise TypeError("survey attestation reuse basis has no receipt")
        if len(prior_sha256) != 64 or canonical_sha256(prior_receipt) != prior_sha256:
            raise ValueError("survey attestation reuse receipt identity is invalid")
        prior_contract = prior_receipt.get("declared_contract")
        if not isinstance(prior_contract, Mapping):
            raise ValueError("survey attestation reuse receipt has no contract")
        attestation = {
            "contract_version": prior_receipt.get("contract_version"),
            "declaration_text": prior_receipt.get("declaration_text"),
            "declared_srd_elevation_m": prior_contract.get(
                "seismic_srd_elevation_m"
            ),
            "vertical_reference": prior_contract.get("vertical_reference"),
            "time_domain": prior_contract.get("seismic_time_domain"),
            "correction_state": prior_contract.get("seismic_correction_state"),
            "source": prior_receipt.get("source"),
            "confirmation_channel": prior_receipt.get("confirmation_channel"),
            "confirmed_at": prior_receipt.get("confirmed_at"),
        }
        reused_from = {
            "snapshot_id": str(
                (reuse_basis or {}).get("snapshot_id")
                or prior_receipt.get("snapshot_id")
                or ""
            ),
            "receipt_sha256": prior_sha256,
        }
    if not isinstance(attestation, Mapping):
        dumper = getattr(attestation, "model_dump", None)
        if not callable(dumper):
            raise TypeError("survey attestation must be a mapping")
        attestation = dumper(mode="json")

    if attestation.get("contract_version") != SURVEY_ATTESTATION_CONTRACT_VERSION:
        raise ValueError("survey attestation contract version is incompatible")
    try:
        declared_srd_elevation = float(attestation.get("declared_srd_elevation_m"))
    except (TypeError, ValueError) as exc:
        raise ValueError("survey attestation SRD elevation must be numeric") from exc
    if not isfinite(declared_srd_elevation):
        raise ValueError("survey attestation SRD elevation must be finite")
    if attestation.get("vertical_reference") != "MSL":
        raise ValueError("survey attestation vertical reference must be MSL")
    if attestation.get("time_domain") != "TWT":
        raise ValueError("survey attestation time domain must be TWT")
    if attestation.get("correction_state") != "corrected_to_srd":
        raise ValueError("survey attestation correction state is invalid")
    declaration_text = survey_attestation_declaration(declared_srd_elevation)
    if attestation.get("declaration_text") not in {None, declaration_text}:
        raise ValueError(
            "survey attestation declaration text does not match structured fields"
        )
    if attestation.get("source") != "human_user":
        raise ValueError("survey attestation must be signed by a human user")
    if attestation.get("confirmation_channel") not in {"user_ui", "user_api"}:
        raise ValueError("survey attestation confirmation channel is invalid")
    confirmed_at = str(attestation.get("confirmed_at") or "").strip()
    if not confirmed_at:
        raise ValueError("survey attestation confirmed_at is required")
    try:
        parsed_confirmed_at = datetime.fromisoformat(
            confirmed_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "survey attestation confirmed_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed_confirmed_at.tzinfo is None or parsed_confirmed_at.utcoffset() is None:
        raise ValueError("survey attestation confirmed_at must include timezone")
    confirmed_at = (
        parsed_confirmed_at.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    try:
        request_srd_elevation = float(raw.get("seismic_srd_elevation_m"))
    except (TypeError, ValueError) as exc:
        raise ValueError("survey attestation requires a numeric request SRD") from exc
    if request_srd_elevation != declared_srd_elevation:
        raise ValueError("survey attestation SRD differs from the request contract")
    if str(raw.get("seismic_time_domain") or "").upper() != "TWT":
        raise ValueError("survey attestation TWT differs from the request contract")
    if str(raw.get("seismic_correction_state") or "").lower() != (
        "corrected_to_srd"
    ):
        raise ValueError(
            "survey attestation correction state differs from the request contract"
        )
    vertical_crs_id = str(raw.get("vertical_crs_id") or "").strip()
    if "MSL" not in vertical_crs_id.upper():
        raise ValueError("survey attestation requires an explicit MSL vertical CRS")

    bound_assets = _precomputed_seismic_bindings(source_records)
    if not bound_assets:
        raise ValueError("survey attestation requires at least one registered SEG-Y asset")
    reuse_key = survey_attestation_reuse_key(raw, source_records)
    if reuse_key is None:
        raise ValueError("survey attestation request contract is not reusable")
    if reused_from is not None:
        prior_key = str((reuse_basis or {}).get("reuse_key") or "")
        if prior_key != reuse_key:
            raise ValueError("survey attestation reuse key does not match SEG-Y content")

    receipt: dict[str, Any] = {
        "contract_version": SURVEY_ATTESTATION_CONTRACT_VERSION,
        "snapshot_id": str(snapshot_id),
        "declaration_text": declaration_text,
        "source": "human_user",
        "confirmation_channel": str(attestation["confirmation_channel"]),
        "confirmed_at": confirmed_at,
        "recorded_at": recorded_at,
        "application_mode": (
            "reused_receipt" if reused_from is not None else "direct_confirmation"
        ),
        "reuse_key": reuse_key,
        "declared_contract": {
            "vertical_reference": "MSL",
            "vertical_crs_id": vertical_crs_id,
            "seismic_srd_elevation_m": declared_srd_elevation,
            "seismic_time_domain": "TWT",
            "seismic_correction_state": "corrected_to_srd",
        },
        "bound_segy_assets": bound_assets,
        "binding_policy": "reuse_precomputed_source_asset_full_file_sha256",
    }
    if reused_from is not None:
        receipt["reused_from"] = reused_from
    if len(bound_assets) == 1:
        receipt["bound_segy_full_content_sha256"] = bound_assets[0][
            "full_content_sha256"
        ]
    return receipt


def _system_evidence_receipt(
    *,
    snapshot_id: str,
    parent_snapshot_id: str,
    request: Mapping[str, Any] | Any,
    source_assets: Iterable[Mapping[str, Any]],
    recorded_at: str,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a bounded deterministic decision to precomputed source identities."""

    raw_request = _plain_request(request)
    raw_decision = dict(decision)
    if raw_decision.get("decision_type") != "bounded_machine_decision":
        raise ValueError(
            "system evidence receipt decision_type must be bounded_machine_decision"
        )
    if raw_decision.get("source") != "deterministic_rule":
        raise ValueError(
            "system evidence receipt source must be deterministic_rule"
        )
    rule_version = str(raw_decision.get("rule_version") or "").strip()
    if rule_version not in SYSTEM_EVIDENCE_ALLOWED_RULE_VERSIONS:
        raise ValueError("system evidence receipt rule version is not allowlisted")
    if str(raw_decision.get("parent_snapshot_id") or "") != parent_snapshot_id:
        raise ValueError("system evidence receipt parent snapshot differs")
    parent_snapshot_sha256 = str(
        raw_decision.get("parent_snapshot_sha256") or ""
    ).casefold()
    if len(parent_snapshot_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in parent_snapshot_sha256
    ):
        raise ValueError(
            "system evidence receipt requires a valid parent snapshot SHA-256"
        )

    required_fields = {
        "vertical_crs_id",
        "seismic_srd_elevation_m",
        "seismic_time_domain",
        "seismic_correction_state",
    }
    effective_patch = raw_decision.get("effective_patch")
    if not isinstance(effective_patch, Mapping) or set(effective_patch) != required_fields:
        raise ValueError(
            "system evidence receipt effective_patch must contain the four "
            "registration semantic fields"
        )
    for field, value in effective_patch.items():
        if raw_request.get(field) != value:
            raise ValueError(
                f"system evidence receipt field {field} differs from the request"
            )

    raw_candidates = raw_decision.get("selected_candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError("system evidence receipt selected_candidates must be a list")
    selected_candidates: list[dict[str, Any]] = []
    selected_fields: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, Mapping):
            raise TypeError("system evidence receipt candidate must be an object")
        candidate = dict(item)
        field = str(candidate.get("field") or "")
        if field not in required_fields or field in selected_fields:
            raise ValueError(
                "system evidence receipt must select each semantic field exactly once"
            )
        if candidate.get("value") != effective_patch[field]:
            raise ValueError(
                f"system evidence candidate {field} differs from the effective patch"
            )
        if str(candidate.get("inference_source") or "") not in {
            "explicit_input",
            "rule",
        }:
            raise ValueError(
                "system evidence receipt cannot promote an LLM-only candidate"
            )
        if candidate.get("status") != "verified":
            raise ValueError(
                "system evidence receipt candidate must be verified"
            )
        if candidate.get("requires_human_confirmation") is not False:
            raise ValueError(
                "system evidence receipt cannot bypass human confirmation"
            )
        if candidate.get("auto_applied") is not True:
            raise ValueError(
                "system evidence receipt candidate must already be auto-applied"
            )
        if str(candidate.get("rule_id") or "") not in (
            SYSTEM_EVIDENCE_ALLOWED_RULE_IDS
        ):
            raise ValueError(
                "system evidence receipt candidate rule is not allowlisted"
            )
        try:
            confidence = float(candidate.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "system evidence receipt candidate confidence must be numeric"
            ) from exc
        if not isfinite(confidence):
            raise ValueError(
                "system evidence receipt candidate confidence must be finite"
            )
        if confidence < SYSTEM_EVIDENCE_MIN_CONFIDENCE:
            raise ValueError(
                "system evidence receipt candidate confidence is below policy"
            )
        evidence = candidate.get("evidence")
        if not isinstance(evidence, list) or not any(
            str(value).strip() for value in evidence
        ):
            raise ValueError(
                "system evidence receipt candidate must preserve concrete evidence"
            )
        candidate["confidence"] = confidence
        selected_fields.add(field)
        selected_candidates.append(candidate)
    if selected_fields != required_fields:
        raise ValueError(
            "system evidence receipt must select all registration semantic fields"
        )
    selected_candidates.sort(key=lambda item: str(item["field"]))

    bound_assets = _precomputed_seismic_bindings(source_assets)
    if len(bound_assets) != 1:
        raise ValueError(
            "system evidence receipt requires exactly one registered SEG-Y asset"
        )
    decision_basis_sha256 = canonical_sha256(
        {
            "rule_version": rule_version,
            "parent_snapshot_id": parent_snapshot_id,
            "parent_snapshot_sha256": parent_snapshot_sha256,
            "effective_patch": dict(effective_patch),
            "selected_candidates": selected_candidates,
        }
    )
    declared_basis_sha256 = str(
        raw_decision.get("decision_basis_sha256") or ""
    ).casefold()
    if declared_basis_sha256 != decision_basis_sha256:
        raise ValueError("system evidence receipt decision basis identity differs")

    return {
        "contract_version": SYSTEM_EVIDENCE_RECEIPT_CONTRACT_VERSION,
        "decision_type": "bounded_machine_decision",
        "source": "deterministic_rule",
        "rule_version": rule_version,
        "snapshot_id": str(snapshot_id),
        "parent_snapshot_id": parent_snapshot_id,
        "parent_snapshot_sha256": parent_snapshot_sha256,
        "recorded_at": recorded_at,
        "effective_patch": dict(effective_patch),
        "selected_candidates": selected_candidates,
        "selected_candidates_sha256": canonical_sha256(selected_candidates),
        "decision_basis_sha256": decision_basis_sha256,
        "bound_segy_assets": bound_assets,
        "binding_policy": "reuse_precomputed_source_asset_full_file_sha256",
    }


def _runtime_contract_receipt(
    *,
    snapshot_id: str,
    parent_snapshot_id: str,
    request: Mapping[str, Any] | Any,
    source_assets: Iterable[Mapping[str, Any]],
    recorded_at: str,
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a human-edited run-first contract into the snapshot identity."""

    raw = dict(confirmation)
    if raw.get("source") != "human_user":
        raise ValueError("runtime contract confirmation source must be human_user")
    if raw.get("confirmation") != "CONFIRM_RUNTIME_CONTRACT":
        raise ValueError("runtime contract confirmation token is invalid")
    if raw.get("profile_id") != "CN_GENERAL_RUN_FIRST_V1":
        raise ValueError("runtime contract profile is not supported")
    if str(raw.get("parent_snapshot_id") or "") != parent_snapshot_id:
        raise ValueError("runtime contract parent snapshot differs")
    parent_snapshot_sha256 = str(
        raw.get("parent_snapshot_sha256") or ""
    ).casefold()
    if len(parent_snapshot_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in parent_snapshot_sha256
    ):
        raise ValueError("runtime contract requires a valid parent snapshot SHA-256")
    runtime_contract_review_sha256 = str(
        raw.get("runtime_contract_review_sha256") or ""
    ).casefold()
    if len(runtime_contract_review_sha256) != 64 or any(
        char not in "0123456789abcdef"
        for char in runtime_contract_review_sha256
    ):
        raise ValueError("runtime contract requires the immutable review SHA-256")
    confirmed_at = datetime.fromisoformat(
        str(raw.get("confirmed_at") or "").replace("Z", "+00:00")
    )
    if confirmed_at.tzinfo is None or confirmed_at.utcoffset() is None:
        raise ValueError("runtime contract confirmed_at must include timezone")
    confirmed_at_text = confirmed_at.astimezone(timezone.utc).isoformat()
    values = raw.get("confirmed_values")
    if not isinstance(values, Mapping):
        raise TypeError("runtime contract confirmed_values must be an object")
    values = dict(values)
    required_fields = {
        "horizontal_crs_id",
        "horizontal_unit",
        "horizontal_axis_order",
        "coordinate_reference_verified",
        "vertical_crs_id",
        "seismic_srd_elevation_m",
        "seismic_time_domain",
        "seismic_correction_state",
        "seismic_replacement_velocity_mps",
        "well_coordinate_source_unit",
        "well_vertical_datum_source_unit",
    }
    optional_fields = {
        "time_depth_default_depth_domain",
        "time_depth_default_depth_unit",
        "time_depth_default_time_unit",
        "time_depth_default_depth_datum",
        "time_depth_default_depth_convention",
        "time_depth_default_time_reference",
        "time_depth_default_time_domain",
        "time_depth_default_correction_state",
    }
    if not required_fields.issubset(values) or set(values) - (
        required_fields | optional_fields
    ):
        raise ValueError("runtime contract confirmed fields are incomplete or unknown")
    raw_request = _plain_request(request)
    for field, value in values.items():
        if raw_request.get(field) != value:
            raise ValueError(
                f"runtime contract field {field} differs from the sealed request"
            )
    attestation = raw.get("attestation")
    if not isinstance(attestation, Mapping):
        raise TypeError("runtime contract requires the explicit user attestation")
    attestation = dict(attestation)
    if raw_request.get("survey_attestation") != attestation:
        raise ValueError("runtime contract attestation differs from the sealed request")
    if str(attestation.get("source") or "") != "human_user":
        raise ValueError("runtime contract attestation source must be human_user")
    attestation_confirmed_at = datetime.fromisoformat(
        str(attestation.get("confirmed_at") or "").replace("Z", "+00:00")
    )
    if attestation_confirmed_at.tzinfo is None or (
        attestation_confirmed_at.astimezone(timezone.utc).isoformat()
        != confirmed_at_text
    ):
        raise ValueError("runtime contract confirmed_at differs from attestation")
    user_attestation_sha256 = canonical_sha256(attestation)
    decision_basis_sha256 = canonical_sha256(
        {
            "profile_id": raw["profile_id"],
            "parent_snapshot_id": parent_snapshot_id,
            "parent_snapshot_sha256": parent_snapshot_sha256,
            "runtime_contract_review_sha256": runtime_contract_review_sha256,
            "confirmation": raw["confirmation"],
            "confirmed_values": values,
            "user_attestation_sha256": user_attestation_sha256,
        }
    )
    if str(raw.get("decision_basis_sha256") or "").casefold() != (
        decision_basis_sha256
    ):
        raise ValueError("runtime contract decision basis identity differs")
    bound_assets = _precomputed_seismic_bindings(source_assets)
    if not bound_assets:
        raise ValueError("runtime contract confirmation requires a seismic asset")
    return {
        "contract_version": RUNTIME_CONTRACT_CONFIRMATION_VERSION,
        "decision_type": "human_runtime_contract_confirmation",
        "source": "human_user",
        "profile_id": raw["profile_id"],
        "snapshot_id": str(snapshot_id),
        "parent_snapshot_id": parent_snapshot_id,
        "parent_snapshot_sha256": parent_snapshot_sha256,
        "runtime_contract_review_sha256": runtime_contract_review_sha256,
        "confirmed_at": confirmed_at_text,
        "recorded_at": recorded_at,
        "confirmation": raw["confirmation"],
        "confirmed_values": values,
        "user_attestation_sha256": user_attestation_sha256,
        "decision_basis_sha256": decision_basis_sha256,
        "bound_segy_assets": bound_assets,
        "binding_policy": "reuse_precomputed_source_asset_full_file_sha256",
    }


def _registration_evidence_record(
    raw_evidence: Mapping[str, Any] | None,
    *,
    source_assets: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Freeze preflight evidence so task-result mutation cannot change a decision."""

    if raw_evidence is None:
        return None
    raw_candidates = raw_evidence.get("candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError("registration evidence candidates must be a list")
    allowed_fields = {
        "vertical_crs_id",
        "seismic_srd_elevation_m",
        "seismic_time_domain",
        "seismic_correction_state",
    }
    normalized: list[dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, Mapping):
            continue
        field = str(item.get("field") or "")
        if field not in allowed_fields:
            continue
        evidence = item.get("evidence")
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []
        inference_source = str(item.get("inference_source") or "rule")
        rule_id = str(item.get("rule_id") or "").strip()
        if not rule_id:
            rule_id = (
                "sealed_request_semantics.v1"
                if inference_source == "explicit_input"
                else f"verified_auto_applied.{field}.v1"
            )
        normalized.append(
            {
                "field": field,
                "value": item.get("value"),
                "confidence": item.get("confidence"),
                "status": str(item.get("status") or ""),
                "source": str(item.get("source") or ""),
                "evidence": [str(value) for value in evidence if str(value).strip()],
                "inference_source": inference_source,
                "requires_human_confirmation": bool(
                    item.get("requires_human_confirmation", True)
                ),
                "auto_applied": item.get("auto_applied") is True,
                "rule_id": rule_id,
            }
        )
    normalized.sort(key=canonical_sha256)
    request_patch = raw_evidence.get("request_patch")
    if not isinstance(request_patch, Mapping):
        request_patch = {}
    request_patch = {
        str(key): value
        for key, value in request_patch.items()
        if str(key) in allowed_fields
    }
    bound_segy_assets = _precomputed_seismic_bindings(source_assets)
    return {
        "contract_version": REGISTRATION_EVIDENCE_CONTRACT_VERSION,
        "candidate_schema_version": str(
            raw_evidence.get("candidate_schema_version") or ""
        ),
        "candidates": normalized,
        "request_patch": request_patch,
        "policy": str(raw_evidence.get("policy") or ""),
        "bound_segy_assets": bound_segy_assets,
        "binding_policy": "precomputed_full_file_sha256",
    }


def _runtime_contract_review_record(
    raw_review: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Freeze the exact editable review shown before a human confirmation."""

    if raw_review is None:
        return None
    if raw_review.get("contract_version") != RUNTIME_CONTRACT_REVIEW_VERSION:
        raise ValueError("runtime contract review version is incompatible")
    profile_id = str(raw_review.get("profile_id") or "")
    if profile_id != "CN_GENERAL_RUN_FIRST_V1":
        raise ValueError("runtime contract review profile is unsupported")
    raw_values = raw_review.get("values")
    raw_fields = raw_review.get("fields")
    if not isinstance(raw_values, Mapping) or not isinstance(raw_fields, list):
        raise TypeError("runtime contract review values/fields are invalid")
    values = {str(key): value for key, value in raw_values.items()}
    fields: list[dict[str, Any]] = []
    field_keys: set[str] = set()
    for raw in raw_fields:
        if not isinstance(raw, Mapping):
            raise TypeError("runtime contract review field must be an object")
        key = str(raw.get("key") or "").strip()
        control = str(raw.get("control") or "").strip()
        label = str(raw.get("label") or "").strip()
        if not key or key in field_keys or key not in values:
            raise ValueError("runtime contract review field keys must be unique values")
        if control not in {"text", "number", "select"} or not label:
            raise ValueError("runtime contract review field presentation is invalid")
        if raw.get("value") != values[key]:
            raise ValueError("runtime contract review field differs from baseline values")
        item: dict[str, Any] = {
            "key": key,
            "label": label,
            "value": values[key],
            "control": control,
        }
        for optional in ("unit", "group", "helper"):
            if raw.get(optional) is not None:
                item[optional] = str(raw[optional])
        choices = raw.get("choices")
        if choices is not None:
            if not isinstance(choices, list):
                raise TypeError("runtime contract review choices must be a list")
            item["choices"] = [
                {
                    "value": choice.get("value"),
                    "label": str(choice.get("label") or ""),
                }
                for choice in choices
                if isinstance(choice, Mapping)
            ]
        fields.append(item)
        field_keys.add(key)
    required = raw_review.get("required") is True
    if required != bool(fields):
        raise ValueError("runtime contract review required flag differs from fields")
    try:
        time_depth_asset_count = int(raw_review.get("time_depth_asset_count") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime contract time_depth_asset_count is invalid") from exc
    if time_depth_asset_count < 0:
        raise ValueError("runtime contract time_depth_asset_count cannot be negative")
    return {
        "contract_version": RUNTIME_CONTRACT_REVIEW_VERSION,
        "required": required,
        "profile_id": profile_id,
        "fields": fields,
        "values": values,
        "time_depth_asset_count": time_depth_asset_count,
    }


def build_source_snapshot_manifest(
    *,
    snapshot_id: str,
    project_id: str,
    created_by_task_id: str,
    request: Mapping[str, Any] | Any,
    assets: Iterable[Mapping[str, Any]],
    effective_config_sha256: str | None = None,
    transformation_registry_sha256: str | None = None,
    qc_summary: Mapping[str, Any] | None = None,
    parse_repairs: Iterable[Mapping[str, Any]] | None = None,
    registration_evidence: Mapping[str, Any] | None = None,
    survey_attestation_reuse_basis: Mapping[str, Any] | None = None,
    parent_snapshot_id: str | None = None,
    system_evidence_decision: Mapping[str, Any] | None = None,
    runtime_contract_confirmation: Mapping[str, Any] | None = None,
    runtime_contract_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = [_asset_record(item) for item in assets]
    repairs = [dict(item) for item in (parse_repairs or ())]
    sealed_at = datetime.now(timezone.utc).isoformat()
    semantics = snapshot_semantics(
        request,
        effective_config_sha256=effective_config_sha256,
        transformation_registry_sha256=transformation_registry_sha256,
    )
    policy = snapshot_inspection_policy(request)
    hashes = source_snapshot_fingerprint(
        records,
        semantics=semantics,
        inspection_policy=policy,
    )
    repairs_sha256 = canonical_sha256(repairs)
    # Parse repairs are immutable snapshot semantics but are not part of the
    # user request.  Bind them to the overall identity without pretending that
    # a later request can re-declare or reinterpret them.
    base_snapshot_sha256 = hashes["snapshot_sha256"]
    hashes["parse_repairs_sha256"] = repairs_sha256
    hashes["snapshot_sha256"] = canonical_sha256(
        {
            "base_snapshot_sha256": base_snapshot_sha256,
            "parse_repairs_sha256": repairs_sha256,
        }
    )
    frozen_registration_evidence = _registration_evidence_record(
        registration_evidence,
        source_assets=records,
    )
    if frozen_registration_evidence is not None:
        registration_evidence_sha256 = canonical_sha256(
            frozen_registration_evidence
        )
        hashes["registration_evidence_sha256"] = registration_evidence_sha256
        hashes["snapshot_sha256"] = canonical_sha256(
            {
                "base_snapshot_sha256": hashes["snapshot_sha256"],
                "registration_evidence_sha256": registration_evidence_sha256,
            }
        )
    frozen_runtime_contract_review = _runtime_contract_review_record(
        runtime_contract_review
    )
    if frozen_runtime_contract_review is not None:
        runtime_contract_review_sha256 = canonical_sha256(
            frozen_runtime_contract_review
        )
        hashes["runtime_contract_review_sha256"] = (
            runtime_contract_review_sha256
        )
        hashes["snapshot_sha256"] = canonical_sha256(
            {
                "base_snapshot_sha256": hashes["snapshot_sha256"],
                "runtime_contract_review_sha256": runtime_contract_review_sha256,
            }
        )
    attestation_receipt = _survey_attestation_receipt(
        snapshot_id=snapshot_id,
        request=request,
        source_assets=records,
        recorded_at=sealed_at,
        reuse_basis=survey_attestation_reuse_basis,
    )
    if attestation_receipt is not None and system_evidence_decision is not None:
        raise ValueError(
            "a source snapshot cannot combine human attestation with a system "
            "evidence decision"
        )
    if runtime_contract_confirmation is not None and system_evidence_decision is not None:
        raise ValueError(
            "a source snapshot cannot combine human runtime confirmation with "
            "a system evidence decision"
        )
    if attestation_receipt is not None:
        attestation_sha256 = canonical_sha256(attestation_receipt)
        hashes["survey_attestation_sha256"] = attestation_sha256
        hashes["snapshot_sha256"] = canonical_sha256(
            {
                "base_snapshot_sha256": hashes["snapshot_sha256"],
                "survey_attestation_sha256": attestation_sha256,
            }
        )
    system_receipt: dict[str, Any] | None = None
    if system_evidence_decision is not None:
        normalized_parent_snapshot_id = str(parent_snapshot_id or "").strip()
        if not normalized_parent_snapshot_id:
            raise ValueError(
                "system evidence decision requires a parent snapshot id"
            )
        system_receipt = _system_evidence_receipt(
            snapshot_id=snapshot_id,
            parent_snapshot_id=normalized_parent_snapshot_id,
            request=request,
            source_assets=records,
            recorded_at=sealed_at,
            decision=system_evidence_decision,
        )
        system_receipt_sha256 = canonical_sha256(system_receipt)
        hashes["system_evidence_receipt_sha256"] = system_receipt_sha256
        hashes["snapshot_sha256"] = canonical_sha256(
            {
                "base_snapshot_sha256": hashes["snapshot_sha256"],
                "parent_snapshot_id": normalized_parent_snapshot_id,
                "system_evidence_receipt_sha256": system_receipt_sha256,
            }
        )
    runtime_receipt: dict[str, Any] | None = None
    if runtime_contract_confirmation is not None:
        normalized_parent_snapshot_id = str(parent_snapshot_id or "").strip()
        if not normalized_parent_snapshot_id:
            raise ValueError(
                "runtime contract confirmation requires a parent snapshot id"
            )
        runtime_receipt = _runtime_contract_receipt(
            snapshot_id=snapshot_id,
            parent_snapshot_id=normalized_parent_snapshot_id,
            request=request,
            source_assets=records,
            recorded_at=sealed_at,
            confirmation=runtime_contract_confirmation,
        )
        runtime_receipt_sha256 = canonical_sha256(runtime_receipt)
        hashes["runtime_contract_receipt_sha256"] = runtime_receipt_sha256
        hashes["snapshot_sha256"] = canonical_sha256(
            {
                "base_snapshot_sha256": hashes["snapshot_sha256"],
                "parent_snapshot_id": normalized_parent_snapshot_id,
                "runtime_contract_receipt_sha256": runtime_receipt_sha256,
            }
        )
    manifest = {
        "contract_version": SOURCE_SNAPSHOT_CONTRACT_VERSION,
        "snapshot_id": str(snapshot_id),
        "project_id": str(project_id),
        "state": "sealed",
        "created_by_task_id": str(created_by_task_id),
        "sealed_at": sealed_at,
        "source_assets": records,
        "semantics": semantics,
        "inspection_policy": policy,
        "parse_repairs": repairs,
        "hashes": {
            "legacy_asset_set_sha256": hashes["asset_set_sha256"],
            "source_content_sha256": hashes["source_content_sha256"],
            "semantics_sha256": hashes["semantics_sha256"],
            "inspection_policy_sha256": hashes[
                "inspection_policy_sha256"
            ],
            "parse_repairs_sha256": hashes["parse_repairs_sha256"],
            "snapshot_sha256": hashes["snapshot_sha256"],
        },
        "qc_summary": dict(qc_summary or {}),
    }
    if frozen_registration_evidence is not None:
        manifest["registration_evidence"] = frozen_registration_evidence
        manifest["hashes"]["registration_evidence_sha256"] = hashes[
            "registration_evidence_sha256"
        ]
    if frozen_runtime_contract_review is not None:
        manifest["runtime_contract_review"] = frozen_runtime_contract_review
        manifest["hashes"]["runtime_contract_review_sha256"] = hashes[
            "runtime_contract_review_sha256"
        ]
    if attestation_receipt is not None:
        manifest["survey_attestation_receipt"] = attestation_receipt
        manifest["hashes"]["survey_attestation_sha256"] = hashes[
            "survey_attestation_sha256"
        ]
    if system_receipt is not None:
        manifest["parent_snapshot_id"] = str(parent_snapshot_id)
        manifest["derivation"] = {
            "kind": "bounded_machine_decision",
            "rule_version": system_receipt["rule_version"],
        }
        manifest["system_evidence_receipt"] = system_receipt
        manifest["hashes"]["system_evidence_receipt_sha256"] = hashes[
            "system_evidence_receipt_sha256"
        ]
    if runtime_receipt is not None:
        manifest["parent_snapshot_id"] = str(parent_snapshot_id)
        manifest["derivation"] = {
            "kind": "human_runtime_contract_confirmation",
            "profile_id": runtime_receipt["profile_id"],
        }
        manifest["runtime_contract_receipt"] = runtime_receipt
        manifest["hashes"]["runtime_contract_receipt_sha256"] = hashes[
            "runtime_contract_receipt_sha256"
        ]
    return manifest


def validate_snapshot_request_semantics(
    manifest: Mapping[str, Any],
    request: Mapping[str, Any] | Any,
    *,
    effective_config_sha256: str | None = None,
    transformation_registry_sha256: str | None = None,
) -> None:
    """Reject a derivative request that reinterprets a sealed snapshot."""

    if manifest.get("contract_version") != SOURCE_SNAPSHOT_CONTRACT_VERSION:
        raise ValueError("source snapshot contract version is incompatible")
    expected_semantics = snapshot_semantics(
        request,
        effective_config_sha256=effective_config_sha256,
        transformation_registry_sha256=transformation_registry_sha256,
    )
    expected_policy = snapshot_inspection_policy(request)
    hashes = manifest.get("hashes") or {}
    if canonical_sha256(expected_semantics) != str(
        hashes.get("semantics_sha256") or ""
    ):
        recorded_semantics = manifest.get("semantics")
        if not isinstance(recorded_semantics, Mapping):
            raise ValueError(
                "source snapshot semantic contract changed; the sealed snapshot "
                "does not contain readable semantics, rerun data preparation"
            )
        changed_fields = sorted(
            key
            for key in set(recorded_semantics) | set(expected_semantics)
            if recorded_semantics.get(key) != expected_semantics.get(key)
            or (key not in recorded_semantics) != (key not in expected_semantics)
        )
        configuration_fields = {
            "effective_config_sha256",
            "transformation_registry_sha256",
        }
        if configuration_fields.intersection(changed_fields):
            reason = "parser or transformation configuration changed"
        else:
            reason = "request fields changed"
        field_list = ", ".join(changed_fields) or "unknown fields"
        raise ValueError(
            "source snapshot semantic contract changed because "
            f"{reason} ({field_list}); rerun data preparation to create a new "
            "immutable snapshot"
        )
    if canonical_sha256(expected_policy) != str(
        hashes.get("inspection_policy_sha256") or ""
    ):
        raise ValueError("source snapshot inspection policy changed")


__all__ = [
    "REGISTRATION_EVIDENCE_CONTRACT_VERSION",
    "RUNTIME_CONTRACT_CONFIRMATION_VERSION",
    "RUNTIME_CONTRACT_REVIEW_VERSION",
    "SOURCE_SNAPSHOT_CONTRACT_VERSION",
    "SYSTEM_EVIDENCE_ALLOWED_RULE_IDS",
    "SYSTEM_EVIDENCE_ALLOWED_RULE_VERSIONS",
    "SYSTEM_EVIDENCE_MIN_CONFIDENCE",
    "SYSTEM_EVIDENCE_RECEIPT_CONTRACT_VERSION",
    "build_source_snapshot_manifest",
    "snapshot_inspection_policy",
    "snapshot_semantics",
    "survey_attestation_reuse_key",
    "validate_snapshot_request_semantics",
]
