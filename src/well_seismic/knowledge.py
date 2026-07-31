from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np

from .models import CurveInfo


def normalize_unit(unit: str, aliases: dict[str, str]) -> str:
    raw = (unit or "").strip()
    key = (
        raw.upper()
        .replace(" ", "")
        .replace("Μ", "Μ")
        .replace("μ", "Μ")
        .replace("µ", "Μ")
    )
    return aliases.get(key, raw or "unknown")


def convert_unit(values: np.ndarray, source: str, target: str, cfg: dict[str, Any]) -> tuple[np.ndarray, bool]:
    if source.casefold() == target.casefold() or target in ("", "unknown"):
        return values, True
    if source in ("", "unknown"):
        return values, False
    rule = cfg.get("conversions", {}).get(f"{source}->{target}")
    if rule:
        return values * float(rule.get("scale", 1.0)) + float(rule.get("offset", 0.0)), True

    # Composite petrophysical units occur frequently in derived LAS curves, for
    # example acoustic impedance: (m/s)*(kg/m3).  Resolve those units from the
    # same atomic conversion library instead of adding one hard-coded rule per
    # derived curve.  Affine conversions are intentionally rejected because an
    # offset cannot be distributed safely through a product.
    source_factors = _product_factors(source)
    target_factors = _product_factors(target)
    if len(source_factors) > 1 and len(source_factors) == len(target_factors):
        remaining_source = Counter(source_factors)
        remaining_target = Counter(target_factors)
        common = remaining_source & remaining_target
        remaining_source -= common
        remaining_target -= common
        scale = 1.0
        unmatched_targets = list(remaining_target.elements())
        for source_factor in remaining_source.elements():
            match_index = next(
                (
                    index
                    for index, target_factor in enumerate(unmatched_targets)
                    if _multiplicative_rule(source_factor, target_factor, cfg) is not None
                ),
                None,
            )
            if match_index is None:
                return values, False
            factor_rule = _multiplicative_rule(
                source_factor,
                unmatched_targets.pop(match_index),
                cfg,
            )
            assert factor_rule is not None
            scale *= factor_rule
        if not unmatched_targets:
            return values * scale, True
    return values, False


def _product_factors(unit: str) -> list[str]:
    """Split a product unit while preserving divisions inside each factor."""
    return [
        factor.strip().strip("()").strip()
        for factor in re.split(r"\s*\*\s*", unit)
        if factor.strip().strip("()")
    ]


def _multiplicative_rule(source: str, target: str, cfg: dict[str, Any]) -> float | None:
    if source.casefold() == target.casefold():
        return 1.0
    rule = cfg.get("conversions", {}).get(f"{source}->{target}")
    if not rule or float(rule.get("offset", 0.0)) != 0.0:
        return None
    return float(rule.get("scale", 1.0))


class CurveKnowledgeBase:
    def __init__(self, config: dict[str, Any]):
        self.rules = config.get("curve_knowledge", {})
        self.units = {
            "unit_aliases": config.get("unit_aliases", {}),
            "conversions": config.get("conversions", {}),
        }

    def identify(self, name: str, unit: str = "", description: str = "", values: np.ndarray | None = None) -> CurveInfo:
        raw = name.strip()
        token = re.sub(r"[\s.\-]+", "_", raw.upper()).strip("_")
        normalized_unit = normalize_unit(unit, self.units["unit_aliases"])
        candidates = self.candidate_scores(name, unit, description, values)
        best = candidates[0] if candidates else {
            "score": 0.0,
            "standard_name": f"UNKNOWN__{token or 'CURVE'}",
            "evidence": {},
        }
        standard = best["standard_name"] if best["score"] >= 0.40 else f"UNKNOWN__{token or 'CURVE'}"
        target_unit = self.rules.get(standard, {}).get("canonical_unit", normalized_unit)
        return CurveInfo(raw, standard, normalized_unit, target_unit, min(1.0, best["score"]), best["evidence"])

    def candidate_scores(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        values: np.ndarray | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        token = re.sub(r"[\s.\-]+", "_", name.strip().upper()).strip("_")
        normalized_unit = normalize_unit(unit, self.units["unit_aliases"])
        scored: list[dict[str, Any]] = []
        for standard, rule in self.rules.items():
            evidence: dict[str, float] = {}
            exact = {str(x).upper() for x in rule.get("aliases", {}).get("exact", [])}
            if token in exact:
                evidence["name"] = 1.0
            else:
                regex_match = any(re.search(pattern, token, re.I) for pattern in rule.get("aliases", {}).get("regex", []))
                evidence["name"] = 0.8 if regex_match else 0.0
            keywords = [str(x).lower() for x in rule.get("descriptions", [])]
            desc = description.lower()
            evidence["description"] = 1.0 if keywords and any(k in desc for k in keywords) else 0.0
            accepted = {normalize_unit(str(x), self.units["unit_aliases"]) for x in rule.get("accepted_units", [])}
            evidence["unit"] = 1.0 if normalized_unit in accepted else (0.25 if normalized_unit == "unknown" else 0.0)
            evidence["range"] = self._range_score(values, rule.get("range", {}))
            score = 0.58 * evidence["name"] + 0.14 * evidence["description"] + 0.18 * evidence["unit"] + 0.10 * evidence["range"]
            scored.append({
                "standard_name": standard,
                "score": round(float(score), 4),
                "evidence": {key: round(float(value), 4) for key, value in evidence.items()},
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit] if limit is not None else scored

    def reclassify(self, info: CurveInfo, standard_name: str, confidence: float) -> CurveInfo:
        if standard_name not in self.rules:
            return info
        return replace(
            info,
            standard_name=standard_name,
            standard_unit=str(self.rules[standard_name].get("canonical_unit", info.original_unit)),
            confidence=float(confidence),
            evidence={**info.evidence, "llm": float(confidence)},
        )

    def infer_high_confidence_unit(self, info: CurveInfo, values: np.ndarray) -> tuple[CurveInfo, str | None]:
        """Only infer units where value scale makes the conversion unambiguous enough for automation."""
        if info.original_unit not in ("", "unknown"):
            return info, None
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return info, None
        median = float(np.median(np.abs(finite)))
        inferred = None
        if info.standard_name in {"NPHI", "POR", "SW"} and 1.5 < median <= 100:
            inferred = "percent"
        elif info.standard_name == "RHOB" and 500 < median < 6000:
            inferred = "kg/m3"
        if inferred is None:
            return info, None
        return replace(info, original_unit=inferred, evidence={**info.evidence, "unit_scale": 0.95}), inferred

    @staticmethod
    def _range_score(values: np.ndarray | None, bounds: dict[str, Any]) -> float:
        if values is None or not bounds:
            return 0.5
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0
        lo = bounds.get("hard_min", -np.inf)
        hi = bounds.get("hard_max", np.inf)
        return float(np.mean((finite >= lo) & (finite <= hi)))

    def standardize(self, info: CurveInfo, values: np.ndarray) -> tuple[CurveInfo, np.ndarray, np.ndarray, list[str]]:
        issues: list[str] = []
        standardized, converted = convert_unit(values.astype(float, copy=True), info.original_unit, info.standard_unit, self.units)
        if (
            info.original_unit.casefold() != info.standard_unit.casefold()
            and info.original_unit not in ("", "unknown")
            and not converted
        ):
            issues.append(f"unit_conversion_unavailable:{info.original_unit}->{info.standard_unit}")
        rule = self.rules.get(info.standard_name, {})
        bounds = rule.get("range", {})
        mask = np.isfinite(standardized)
        if bounds:
            valid_range = (standardized >= bounds.get("hard_min", -np.inf)) & (standardized <= bounds.get("hard_max", np.inf))
            mask &= valid_range
            standardized[~valid_range] = np.nan
        return replace(info, source=info.source), standardized, mask, issues
