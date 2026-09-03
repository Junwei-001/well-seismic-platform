"""Fail-closed acceptance contract for displaying prediction results.

The visualization layer contract describes *what* artifacts are available.  It
does not decide whether a scientific result is fit to display.  This module
keeps that decision separate and deterministic: a prediction is displayable
only when both quantitative and visual gates passed and the four comparison
panels are declared.  A failed result may expose diagnostics, but never the
prediction panels.  Missing or malformed evidence is unavailable by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal


CONTRACT_VERSION = "well-seismic.result-display-acceptance.v1"
PANEL_NAMES = ("raw", "truth", "prediction", "error")

GateStatus = Literal["passed", "failed", "not_run", "unavailable"]
DisplayStatus = Literal["accepted", "failed_diagnostic", "unavailable"]
PANEL_ROLE_CANDIDATES = {
    "raw": ("source", "baseline"),
    "truth": ("observation",),
    "prediction": ("prediction",),
    "error": ("quality",),
}


@dataclass(frozen=True)
class ResultDisplayDecision:
    """Normalized, JSON-safe decision returned to API and UI adapters."""

    contract_version: str
    display_status: DisplayStatus
    quantitative_status: GateStatus
    visual_status: GateStatus
    panels: dict[str, dict[str, Any]]
    diagnostics: tuple[str, ...]
    reason_codes: tuple[str, ...]
    missing_panels: tuple[str, ...]

    @property
    def panels_visible(self) -> bool:
        return self.display_status == "accepted"

    @property
    def diagnostics_visible(self) -> bool:
        return self.display_status == "failed_diagnostic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diagnostics"] = list(self.diagnostics)
        payload["reason_codes"] = list(self.reason_codes)
        payload["missing_panels"] = list(self.missing_panels)
        payload["panels_visible"] = self.panels_visible
        payload["diagnostics_visible"] = self.diagnostics_visible
        return payload


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _gate_status(value: object) -> GateStatus:
    if not isinstance(value, Mapping):
        return "unavailable"
    status = str(value.get("status") or "unavailable").strip().casefold()
    if status in {"passed", "failed", "not_run", "unavailable"}:
        return status  # type: ignore[return-value]
    return "unavailable"


def _panel(value: object) -> dict[str, Any] | None:
    if isinstance(value, str):
        identifier = value.strip()
        return {"layer_id": identifier} if identifier else None
    if not isinstance(value, Mapping):
        return None
    payload = {str(key): item for key, item in value.items()}
    references = (
        payload.get("layer_id"),
        payload.get("artifact_id"),
        payload.get("access_url"),
    )
    if not any(isinstance(item, str) and item.strip() for item in references):
        return None
    return payload


def comparison_panels_from_layers(
    layers: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select deterministic four-panel references from a generic layer bundle.

    Runners remain model-neutral: they label source, observation, prediction and
    quality layers, while this adapter assigns the UI panel names.  Uncertainty
    and validity layers deliberately do not substitute for an error layer.
    """

    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for layer in layers:
        role = str(layer.get("role") or "").strip().casefold()
        if role:
            by_role.setdefault(role, []).append(layer)
    panels: dict[str, dict[str, Any]] = {}
    for panel_name, roles in PANEL_ROLE_CANDIDATES.items():
        selected = next(
            (candidate for role in roles for candidate in by_role.get(role, ())),
            None,
        )
        if selected is None:
            continue
        reference = {
            key: selected[key]
            for key in ("id", "artifact_id", "access_url", "kind", "name")
            if selected.get(key) is not None
        }
        if "id" in reference:
            reference["layer_id"] = reference.pop("id")
        if _panel(reference) is not None:
            panels[panel_name] = reference
    return panels


def _passed_quantitative_gate_is_evidenced(gate: object) -> bool:
    if not isinstance(gate, Mapping):
        return False
    metrics = gate.get("metrics")
    criteria = gate.get("criteria")
    return bool(isinstance(metrics, Mapping) and metrics) and bool(
        isinstance(criteria, Mapping) and criteria
    )


def _passed_visual_gate_is_evidenced(gate: object) -> bool:
    if not isinstance(gate, Mapping):
        return False
    checks = gate.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)) or not checks:
        return False
    for check in checks:
        if not isinstance(check, Mapping):
            return False
        if str(check.get("status") or "").strip().casefold() != "passed":
            return False
        if not str(check.get("id") or "").strip():
            return False
    return True


def evaluate_result_display_acceptance(
    result_or_contract: Mapping[str, Any],
) -> ResultDisplayDecision:
    """Evaluate one result without trusting a caller-declared display status.

    A full prediction document should place the contract under
    ``display_acceptance``.  Passing the contract object directly is supported
    for validation tools.  ``display_status`` is always recomputed so a stale
    or hand-edited manifest cannot bypass either gate.
    """

    nested = result_or_contract.get("display_acceptance")
    if isinstance(nested, Mapping):
        contract: Mapping[str, Any] = nested
    elif result_or_contract.get("contract_version") == CONTRACT_VERSION:
        contract = result_or_contract
    else:
        return ResultDisplayDecision(
            contract_version=CONTRACT_VERSION,
            display_status="unavailable",
            quantitative_status="unavailable",
            visual_status="unavailable",
            panels={},
            diagnostics=(),
            reason_codes=("acceptance_contract_missing",),
            missing_panels=PANEL_NAMES,
        )

    reason_codes: list[str] = []
    declared_version = str(contract.get("contract_version") or "").strip()
    if declared_version != CONTRACT_VERSION:
        reason_codes.append("acceptance_contract_version_unsupported")

    quantitative_gate = contract.get("quantitative_gate")
    visual_gate = contract.get("visual_gate")
    quantitative_status = _gate_status(quantitative_gate)
    visual_status = _gate_status(visual_gate)

    diagnostics: list[str] = []
    for gate_name, gate, status in (
        ("quantitative", quantitative_gate, quantitative_status),
        ("visual", visual_gate, visual_status),
    ):
        if isinstance(gate, Mapping):
            gate_diagnostics = _strings(gate.get("diagnostics"))
        else:
            gate_diagnostics = ()
        diagnostics.extend(f"{gate_name}: {item}" for item in gate_diagnostics)
        if status == "failed" and not gate_diagnostics:
            reason_codes.append(f"{gate_name}_failure_diagnostics_missing")

    panels_payload = contract.get("panels")
    panels_payload = panels_payload if isinstance(panels_payload, Mapping) else {}
    panels: dict[str, dict[str, Any]] = {}
    for name in PANEL_NAMES:
        normalized = _panel(panels_payload.get(name))
        if normalized is not None:
            panels[name] = normalized
    missing_panels = tuple(name for name in PANEL_NAMES if name not in panels)

    failed_statuses = {
        name: status
        for name, status in (
            ("quantitative", quantitative_status),
            ("visual", visual_status),
        )
        if status == "failed"
    }
    failures_have_diagnostics = bool(failed_statuses) and all(
        isinstance(gate, Mapping) and bool(_strings(gate.get("diagnostics")))
        for gate in (quantitative_gate, visual_gate)
        if isinstance(gate, Mapping)
        and str(gate.get("status") or "").strip().casefold() == "failed"
    )
    if failed_statuses and failures_have_diagnostics and not any(
        code.endswith("diagnostics_missing") for code in reason_codes
    ):
        return ResultDisplayDecision(
            contract_version=CONTRACT_VERSION,
            display_status="failed_diagnostic",
            quantitative_status=quantitative_status,
            visual_status=visual_status,
            panels={},
            diagnostics=tuple(diagnostics),
            reason_codes=tuple(reason_codes or ("acceptance_gate_failed",)),
            missing_panels=missing_panels,
        )

    both_passed = quantitative_status == visual_status == "passed"
    if both_passed:
        if not _passed_quantitative_gate_is_evidenced(quantitative_gate):
            reason_codes.append("quantitative_pass_evidence_missing")
        if not _passed_visual_gate_is_evidenced(visual_gate):
            reason_codes.append("visual_pass_evidence_missing")
        if missing_panels:
            reason_codes.append("comparison_panels_incomplete")
        if declared_version == CONTRACT_VERSION and not reason_codes:
            return ResultDisplayDecision(
                contract_version=CONTRACT_VERSION,
                display_status="accepted",
                quantitative_status=quantitative_status,
                visual_status=visual_status,
                panels=panels,
                diagnostics=tuple(diagnostics),
                reason_codes=(),
                missing_panels=(),
            )
    else:
        if quantitative_status not in {"passed", "failed"}:
            reason_codes.append(f"quantitative_gate_{quantitative_status}")
        if visual_status not in {"passed", "failed"}:
            reason_codes.append(f"visual_gate_{visual_status}")

    return ResultDisplayDecision(
        contract_version=CONTRACT_VERSION,
        display_status="unavailable",
        quantitative_status=quantitative_status,
        visual_status=visual_status,
        panels={},
        diagnostics=tuple(diagnostics),
        reason_codes=tuple(dict.fromkeys(reason_codes or ("acceptance_not_satisfied",))),
        missing_panels=missing_panels,
    )


__all__ = [
    "CONTRACT_VERSION",
    "DisplayStatus",
    "GateStatus",
    "PANEL_NAMES",
    "PANEL_ROLE_CANDIDATES",
    "ResultDisplayDecision",
    "comparison_panels_from_layers",
    "evaluate_result_display_acceptance",
]
