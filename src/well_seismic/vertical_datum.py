"""Deterministic vertical-datum normalization for well/seismic alignment.

The canonical coordinate is absolute elevation in metres relative to mean sea
level (MSL), positive upward.  Domain labels such as KB, GL and SRD are kept as
evidence; they are never treated as interchangeable names for the same value.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


CANONICAL_REFERENCE = "MSL"
DATUM_ALIASES = {
    "SRD": ("SRD", "SEISMIC REFERENCE DATUM", "SEISMIC DATUM", "PROCESSING DATUM", "地震处理基准面", "地震基准面"),
    "KB": ("EKB", "KBELEV", "KB_ELEVATION", "KB", "KELLY BUSHING", "WELL DATUM", "补心高"),
    "GL": ("GROUND ELEVATION", "GROUND LEVEL", "GL", "地面海拔", "地面高程"),
    "DF": ("DERRICK FLOOR", "DF", "钻台面"),
    "RT": ("ROTARY TABLE", "RT", "转盘面"),
}
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_EXPLICIT_MSL = re.compile(r"\bMSL\b|MEAN\s+SEA\s+LEVEL|平均海平面|海平面", re.IGNORECASE)
_UNIT_PATTERNS = (
    ("km", re.compile(r"\bKM\b|千米|公里", re.IGNORECASE)),
    ("cm", re.compile(r"\bCM\b|厘米", re.IGNORECASE)),
    ("mm", re.compile(r"\bMM\b|毫米", re.IGNORECASE)),
    ("dm", re.compile(r"\bDM\b|分米", re.IGNORECASE)),
    ("ft", re.compile(r"\b(?:FT|FEET|FOOT)\b|英尺", re.IGNORECASE)),
    ("in", re.compile(r"\b(?:IN|INCH|INCHES)\b|英寸", re.IGNORECASE)),
    ("yd", re.compile(r"\b(?:YD|YARD|YARDS)\b|码", re.IGNORECASE)),
    ("m", re.compile(r"\b(?:M|METER|METRE|METERS|METRES)\b|米", re.IGNORECASE)),
)
_RESISTIVITY_EVIDENCE = re.compile(
    r"\b(?:OHMS?|RESISTIVITY)\b|"
    r"OHMS?\s*[-._*/ ]?\s*(?:M|METERS?|METRES?|FT|FEET|FOOT)\b|"
    r"Ω\s*[·.*\-/ ]?\s*(?:M|FT)\b|电阻率",
    re.IGNORECASE,
)
_SEISMIC_DATUM_STATEMENT = re.compile(
    r"\b(?:FINAL|PROCESSING|SEISMIC(?:\s+REFERENCE)?)\s+DATUM\b",
    re.IGNORECASE,
)
_PSTM_EVIDENCE = re.compile(
    r"\bPSTM\b|PRE[- ]?STACK\s+TIME\s+MIGRATION|POST[- ]?STACK\s+TIME\s+MIGRATION",
    re.IGNORECASE,
)
_FLOATING_DATUM_STATIC_EVIDENCE = re.compile(
    r"\bFLOAT(?:ING|TING)?\s+DATUM\s+STATICS?\b|"
    r"\bFOLATING\s+DATUM\s+STATICS?\b|浮动基准(?:面)?静校正",
    re.IGNORECASE,
)
_REPLACEMENT_VELOCITY_EVIDENCE = re.compile(
    r"(?:REPLACEMENT|PEPLACEMENT|WEATHERING|DATUM)\s+VELOCITY|"
    r"替换速度|基准面校正速度",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatumObservation:
    entity_kind: str
    entity_name: str | None
    datum: str
    value: float
    unit: str
    relation: str
    reference: str
    source: str
    evidence: str
    confidence: float
    review_required: bool = False
    is_depth_reference: bool = False

    @property
    def absolute_elevation_m(self) -> float | None:
        value_m = _to_metres(self.value, self.unit)
        if value_m is None or self.reference != CANONICAL_REFERENCE:
            return None
        if self.relation == "elevation_above_reference":
            return value_m
        if self.relation == "depth_below_reference":
            return -value_m
        return None

    def with_entity(self, name: str) -> "DatumObservation":
        return replace(self, entity_name=name)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["absolute_elevation_m"] = self.absolute_elevation_m
        return result


@dataclass(frozen=True)
class ResolvedVerticalDatum:
    entity_kind: str
    entity_name: str
    datum: str | None
    absolute_elevation_m: float | None
    source: str
    evidence: str
    confidence: float
    ready: bool
    review_required: bool
    conflicts: tuple[str, ...]
    observations: tuple[DatumObservation, ...]
    canonical_reference: str = CANONICAL_REFERENCE
    positive_direction: str = "up"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_kind": self.entity_kind,
            "entity_name": self.entity_name,
            "datum": self.datum,
            "absolute_elevation_m": self.absolute_elevation_m,
            "canonical_reference": self.canonical_reference,
            "positive_direction": self.positive_direction,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 6),
            "ready": self.ready,
            "review_required": self.review_required,
            "conflicts": list(self.conflicts),
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass(frozen=True)
class TimeReferenceMetadata:
    time_reference: str = "unknown"
    time_domain: str = "unknown"
    correction_state: str = "unknown"
    depth_unit: str = "unknown"
    time_unit: str = "unknown"
    depth_datum: str | None = None
    depth_convention: str | None = None
    replacement_velocity_mps: float | None = None
    md_offset_to_trajectory_m: float | None = None
    provenance: str = ""
    evidence: tuple[str, ...] = ()
    contract_candidates: tuple["ContractEvidenceCandidate", ...] = ()

    @property
    def ready_for_srd_time(self) -> bool:
        return (
            self.time_domain in {"TWT", "OWT"}
            and self.correction_state in {"corrected_to_srd", "uncorrected"}
            and (
                self.correction_state == "uncorrected"
                or self.time_reference == "SRD"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_candidates"] = [
            item.to_dict() for item in self.contract_candidates
        ]
        return payload


@dataclass(frozen=True)
class ContractEvidenceCandidate:
    """One bounded contract value with its evidence and application policy.

    A candidate is deliberately separate from the effective seismic contract.
    Rules may mark exact declarations as ``verified`` and apply them.  A model
    may only add an advisory candidate; it can never turn an ambiguous header
    into an effective physical reference by itself.
    """

    field: str
    value: Any
    confidence: float
    status: str
    source: str
    evidence: tuple[str, ...] = ()
    inference_source: str = "rule"
    requires_human_confirmation: bool = True
    auto_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "confidence": round(float(self.confidence), 6),
            "status": self.status,
            "source": self.source,
            "evidence": list(self.evidence),
            "inference_source": self.inference_source,
            "requires_human_confirmation": self.requires_human_confirmation,
            "auto_applied": self.auto_applied,
        }


def _to_metres(value: float, unit: str) -> float | None:
    if not math.isfinite(float(value)):
        return None
    normalized = str(unit).strip().casefold()
    if normalized in {"m", "meter", "metre", "meters", "metres", "米"}:
        return float(value)
    # LAS 2.x commonly emits ``.F`` for foot on depth/elevation header items.
    # This helper is length-specific, so accepting F here cannot reinterpret a
    # Fahrenheit curve (curve-unit handling uses the separate knowledge base).
    if normalized in {"f", "ft", "feet", "foot", "英尺"}:
        return float(value) * 0.3048
    if normalized in {"cm", "centimeter", "centimetre", "厘米"}:
        return float(value) * 0.01
    if normalized in {"mm", "millimeter", "millimetre", "毫米"}:
        return float(value) * 0.001
    if normalized in {"dm", "decimeter", "decimetre", "分米"}:
        return float(value) * 0.1
    if normalized in {"km", "kilometer", "kilometre", "千米", "公里"}:
        return float(value) * 1000.0
    if normalized in {"in", "inch", "inches", "英寸"}:
        return float(value) * 0.0254
    if normalized in {"yd", "yard", "yards", "码"}:
        return float(value) * 0.9144
    return None


def length_to_metres(value: Any, unit: str) -> Any:
    factor = _to_metres(1.0, unit)
    if factor is None:
        raise ValueError(f"无法确认长度单位：{unit}")
    return value * factor


def time_to_milliseconds(value: Any, unit: str) -> Any:
    normalized = str(unit).strip().casefold()
    factors = {
        "ms": 1.0,
        "millisecond": 1.0,
        "milliseconds": 1.0,
        "毫秒": 1.0,
        "s": 1000.0,
        "sec": 1000.0,
        "second": 1000.0,
        "seconds": 1000.0,
        "秒": 1000.0,
        "us": 0.001,
        "µs": 0.001,
        "μs": 0.001,
        "microsecond": 0.001,
        "微秒": 0.001,
    }
    if normalized not in factors:
        raise ValueError(f"无法确认时间单位：{unit}")
    return value * factors[normalized]


def _detect_unit(text: str) -> str:
    for unit, pattern in _UNIT_PATTERNS:
        if pattern.search(text):
            return unit
    return "unknown"


def observation_from_value(
    *,
    datum: str,
    value: float,
    source: str,
    entity_kind: str,
    entity_name: str | None = None,
    unit: str = "m",
    evidence: str = "",
    confidence: float = 0.9,
    review_required: bool = False,
    is_depth_reference: bool | None = None,
) -> DatumObservation:
    canonical = str(datum).strip().upper()
    if canonical not in DATUM_ALIASES:
        raise ValueError(f"不支持的垂向基准：{datum}")
    return DatumObservation(
        entity_kind=str(entity_kind),
        entity_name=entity_name,
        datum=canonical,
        value=float(value),
        unit=str(unit).strip() if unit is not None and str(unit).strip() else "unknown",
        relation="elevation_above_reference",
        reference=CANONICAL_REFERENCE,
        source=str(source),
        evidence=evidence or f"{canonical}={value} {unit or 'unknown'}",
        confidence=float(max(0.0, min(1.0, confidence))),
        review_required=bool(review_required),
        is_depth_reference=(canonical in {"KB", "DF", "RT"}) if is_depth_reference is None else bool(is_depth_reference),
    )


_BINARY_CONTAINER_SUFFIXES = frozenset(
    {
        ".doc",
        ".docx",
        ".pdf",
        ".xls",
        ".xlsb",
        ".xlsm",
        ".xlsx",
        ".xltm",
        ".xltx",
        ".zip",
    }
)


def _decode_prefix(path: Path, max_bytes: int = 131072) -> str:
    # Office documents and other containers are binary/ZIP packages, not text
    # headers.  Decoding their compressed bytes and applying KB/GL/SRD regexes
    # can manufacture plausible-looking datum values from arbitrary payload
    # bytes.  Dedicated tabular readers remain responsible for spreadsheet
    # cells and their explicit unit columns; the generic header scanner fails
    # closed for containers it cannot interpret semantically.
    if path.suffix.casefold() in _BINARY_CONTAINER_SUFFIXES:
        return ""
    with path.open("rb") as handle:
        raw = handle.read(3200 if path.suffix.casefold() in {".sgy", ".segy"} else max_bytes)
    candidates: list[str] = []
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "cp500", "latin1"):
        try:
            candidates.append(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    if not candidates:
        return raw.decode("latin1", errors="replace")
    keywords = ("SRD", "DATUM", "MSL", "KB", "GL", "ELEVATION")
    decoded = max(
        candidates,
        key=lambda text: (
            sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
            + 100 * sum(keyword in text.upper() for keyword in keywords)
        ),
    )
    if path.suffix.casefold() in {".sgy", ".segy"} and "\n" not in decoded:
        return "\n".join(
            decoded[index : index + 80]
            for index in range(0, len(decoded), 80)
        )
    return decoded


def _datum_hits(text: str) -> list[tuple[int, int, str]]:
    hits: list[tuple[int, int, str]] = []
    upper = text.upper()
    for datum, aliases in DATUM_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias.upper())
            # Short aliases such as RT, GL, DF and KB are complete header
            # tokens.  Substring matching turned LAS fields like STRT,
            # EXPORT_DATE and TRUEVERTICALTHICKNESS into fake datums.
            pattern = re.compile(rf"(?<![A-Z0-9_]){escaped}(?![A-Z0-9_])")
            match = pattern.search(upper)
            if match is not None:
                hits.append((match.start(), match.end(), datum))
    return sorted(hits, key=lambda item: (item[0], -(item[1] - item[0])))


def extract_vertical_datum_observations(
    path: str | Path,
    *,
    entity_kind: str,
    entity_name: str | None = None,
) -> list[DatumObservation]:
    """Extract explicit KB/GL/SRD-style elevation statements from file headers.

    Bare ``# KB 123.4`` fields are retained with a review flag.  Statements
    explicitly saying ``from MSL`` or ``m MSL`` are deterministic and do not
    require an LLM.
    """

    source_path = Path(path)
    try:
        text = _decode_prefix(source_path)
    except OSError:
        return []
    observations: list[DatumObservation] = []
    seen: set[tuple[str, float, str, str]] = set()
    las_section = ""
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line or len(line) > 500:
            continue
        if source_path.suffix.casefold() == ".las" and line.startswith("~"):
            section_token = line[1:].split(maxsplit=1)[0].strip().upper()
            if section_token.startswith("C"):
                las_section = "curve"
            elif section_token.startswith("A"):
                las_section = "ascii"
            else:
                las_section = "header"
            continue
        # LAS curve definitions and sample rows are not well-head metadata.
        # In particular, the standard resistivity mnemonic ``RT .ohm-m`` must
        # never be offered to an LLM as a Rotary Table elevation candidate.
        if las_section in {"curve", "ascii"}:
            continue
        hits = _datum_hits(line)
        if not hits and entity_kind == "seismic":
            final_datum = _SEISMIC_DATUM_STATEMENT.search(line)
            if final_datum is not None:
                hits = [(final_datum.start(), final_datum.end(), "SRD")]
        if not hits:
            continue
        distinct = {item[2] for item in hits}
        if len(distinct) > 1 and "WELL DATUM" not in line.upper():
            # Expressions such as KB-GL are offsets, not absolute elevations.
            continue
        start, end, datum = hits[0]
        if "WELL DATUM" in line.upper() and "KB" in distinct:
            datum = "KB"
        if datum == "RT" and _RESISTIVITY_EVIDENCE.search(line):
            continue
        explicit_msl = bool(_EXPLICIT_MSL.search(line))
        after = line[end : end + 100]
        match = _NUMBER.search(after)
        if match is None:
            before_matches = list(_NUMBER.finditer(line[max(0, start - 40) : start]))
            if before_matches:
                match = before_matches[-1]
                value_text = match.group(0)
            elif datum == "SRD" and explicit_msl and re.search(r"DATUM[^.;]{0,30}(?:IS|=|:)\s*(?:MSL|MEAN SEA LEVEL)", line, re.IGNORECASE):
                value_text = "0"
            else:
                continue
        else:
            value_text = match.group(0)
        try:
            value = float(value_text.replace(",", ""))
        except ValueError:
            continue
        unit = _detect_unit(line)
        if unit == "unknown":
            attached_unit = re.search(
                rf"{re.escape(value_text)}\s*(KM|CM|MM|DM|FT|FEET|FOOT|IN|YD|M)(?![A-Z])",
                line,
                re.IGNORECASE,
            )
            if attached_unit is not None:
                token = attached_unit.group(1).casefold()
                unit = {
                    "feet": "ft",
                    "foot": "ft",
                }.get(token, token)
        below = bool(re.search(r"BELOW\s+(?:MSL|MEAN\s+SEA\s+LEVEL)|低于(?:平均)?海平面", line, re.IGNORECASE))
        relation = "depth_below_reference" if below else "elevation_above_reference"
        confidence = 0.96 if explicit_msl and unit != "unknown" else 0.78
        review_required = not explicit_msl or unit == "unknown"
        key = (datum, value, unit, relation)
        if key in seen:
            continue
        seen.add(key)
        observations.append(DatumObservation(
            entity_kind=str(entity_kind),
            entity_name=entity_name,
            datum=datum,
            value=value,
            unit=unit,
            relation=relation,
            reference=CANONICAL_REFERENCE,
            source=str(source_path),
            evidence=line[:300],
            confidence=confidence,
            review_required=review_required,
            is_depth_reference=datum in {"KB", "DF", "RT"} or (datum == "SRD" and entity_kind == "seismic"),
        ))
    return observations


def extract_time_reference_metadata(path: str | Path) -> TimeReferenceMetadata:
    """Read declarative depth/time reference semantics without guessing nulls as zero."""

    source_path = Path(path)
    try:
        text = _decode_prefix(source_path)
    except OSError:
        return TimeReferenceMetadata(provenance=str(source_path))
    upper = text.upper()
    evidence: list[str] = []
    candidates: list[ContractEvidenceCandidate] = []
    semantic_lines = [
        " ".join(line.strip().split())[:300]
        for line in text.splitlines()
        if re.search(
            r"\b(?:TWT|TWTT|OWT|PSTM|SRD|KB|GL|DF|RT|DATUM|VELOCITY|STATIC|STATICS|TVDSS|MD)\b|"
            r"双程|单程|基准|校正|替换速度|高程",
            line,
            re.IGNORECASE,
        )
    ][:12]
    evidence.extend(f"原始头:{line}" for line in semantic_lines if line)

    def add_candidate(
        field: str,
        value: Any,
        *,
        confidence: float,
        status: str,
        candidate_evidence: Iterable[str],
        requires_confirmation: bool,
        auto_applied: bool,
    ) -> None:
        candidates.append(
            ContractEvidenceCandidate(
                field=field,
                value=value,
                confidence=confidence,
                status=status,
                source=str(source_path),
                evidence=tuple(str(item)[:300] for item in candidate_evidence if item),
                requires_human_confirmation=requires_confirmation,
                auto_applied=auto_applied,
            )
        )

    if re.search(r"\b(?:TWT|TWTT)(?:[_ (\[]?MS)?\b|TWO[-_ ]?WAY[_ ]+TIME|双程时", upper):
        time_domain = "TWT"
        evidence.append("时间域识别为TWT")
        add_candidate(
            "seismic_time_domain",
            "TWT",
            confidence=0.98,
            status="verified",
            candidate_evidence=[*semantic_lines, "文件明确声明TWT/双程时"],
            requires_confirmation=False,
            auto_applied=True,
        )
    elif re.search(r"\bOWT\b|ONE[- ]?WAY\s+TIME|单程时", upper):
        time_domain = "OWT"
        evidence.append("时间域识别为OWT")
        add_candidate(
            "seismic_time_domain",
            "OWT",
            confidence=0.98,
            status="verified",
            candidate_evidence=[*semantic_lines, "文件明确声明OWT/单程时"],
            requires_confirmation=False,
            auto_applied=True,
        )
    else:
        time_domain = "unknown"
        if _PSTM_EVIDENCE.search(text):
            add_candidate(
                "seismic_time_domain",
                "TWT",
                confidence=0.74,
                status="candidate",
                candidate_evidence=[
                    *semantic_lines,
                    "PSTM是时间域处理证据，但不能单独证明文件采用TWT约定",
                ],
                requires_confirmation=True,
                auto_applied=False,
            )

    depth_unit_match = re.search(
        r"(?:DEPTH\s*UNIT|MD\s*[\[(]|TVD\s*[\[(])\s*[:=]?\s*(M|FT|CM|MM|DM|KM|IN|YD)\b",
        upper,
    )
    depth_unit = depth_unit_match.group(1).lower() if depth_unit_match else "unknown"
    time_unit_match = re.search(
        r"(?:TIME\s*UNIT|TWT\s*[\[(]|OWT\s*[\[(])\s*[:=]?\s*(MS|S|US|ΜS|ΜS)\b",
        upper,
    )
    time_unit = time_unit_match.group(1).lower() if time_unit_match else "unknown"

    corrected = re.search(
        r"CORRECTED[_ ]+(?:TO|AT)[_ ]+(?:THE[_ ]+)?SRD|"
        r"DATUM[-_ ]?CORRECTED[_ ]+TO[_ ]+SRD|已校正至?\s*SRD",
        upper,
    )
    uncorrected = re.search(r"\bUNCORRECTED\b|NOT\s+CORRECTED|RAW\s+(?:TIME|CHECKSHOT)|未校正", upper)
    if uncorrected:
        correction_state = "uncorrected"
        reference_hits = _datum_hits(upper)
        time_reference = next((datum for _, _, datum in reference_hits if datum in {"KB", "GL", "DF", "RT"}), "unknown")
        evidence.append("文件声明时间尚未校正到SRD")
        add_candidate(
            "seismic_correction_state",
            "uncorrected",
            confidence=0.98,
            status="verified",
            candidate_evidence=[*semantic_lines, "文件明确声明未校正"],
            requires_confirmation=False,
            auto_applied=True,
        )
    elif corrected:
        correction_state = "corrected_to_srd"
        time_reference = "SRD"
        evidence.append("文件声明时间已校正到SRD")
        add_candidate(
            "seismic_correction_state",
            "corrected_to_srd",
            confidence=0.99,
            status="verified",
            candidate_evidence=[*semantic_lines, "文件明确声明时间已校正到SRD"],
            requires_confirmation=False,
            auto_applied=True,
        )
        add_candidate(
            "seismic_time_reference",
            "SRD",
            confidence=0.99,
            status="verified",
            candidate_evidence=[*semantic_lines, "corrected_to_srd明确绑定SRD"],
            requires_confirmation=False,
            auto_applied=True,
        )
    else:
        correction_state = "unknown"
        time_reference = "SRD" if re.search(r"TIME[_ ]+(?:REFERENCE|DATUM)[^\r\n]{0,30}\bSRD\b|时间基准[^\r\n]{0,20}SRD", upper) else "unknown"
        if time_reference == "SRD":
            add_candidate(
                "seismic_time_reference",
                "SRD",
                confidence=0.96,
                status="verified",
                candidate_evidence=[*semantic_lines, "文件明确声明TIME REFERENCE/DATUM=SRD"],
                requires_confirmation=False,
                auto_applied=True,
            )
        reviewable_processed_stack = bool(
            _PSTM_EVIDENCE.search(text)
            and re.search(r"\bSTACK(?:ED)?\b", upper)
            and _SEISMIC_DATUM_STATEMENT.search(text)
            and _REPLACEMENT_VELOCITY_EVIDENCE.search(text)
            and _FLOATING_DATUM_STATIC_EVIDENCE.search(text)
        )
        if reviewable_processed_stack:
            add_candidate(
                "seismic_correction_state",
                "corrected_to_srd",
                confidence=0.86,
                status="review_required",
                candidate_evidence=[
                    *semantic_lines,
                    (
                        "PSTM叠后成品同时声明Final/Processing Datum、替换速度和逐道"
                        "Floating Datum Statics；组合证据支持一次人工确认已校正到最终SRD，"
                        "但不会自动改写有效合同"
                    ),
                ],
                requires_confirmation=True,
                auto_applied=False,
            )
        elif _PSTM_EVIDENCE.search(text) or _SEISMIC_DATUM_STATEMENT.search(text):
            add_candidate(
                "seismic_correction_state",
                "unknown",
                confidence=0.0,
                status="insufficient",
                candidate_evidence=[
                    *semantic_lines,
                    "PSTM或Final Datum只说明处理阶段/目标基准；没有明确静校正声明，禁止推断corrected_to_srd",
                ],
                requires_confirmation=True,
                auto_applied=False,
            )

    datum_match = re.search(
        r"\b(?:FINAL|PROCESSING|SEISMIC(?:\s+REFERENCE)?)\s+DATUM\b"
        r"[^\r\n]{0,30}?[:=]?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
        r"(M|FT|FEET|FOOT)\b",
        text,
        re.IGNORECASE,
    )
    if datum_match is not None:
        raw_value = float(datum_match.group(1).replace(",", ""))
        raw_unit = datum_match.group(2).lower()
        value_m = _to_metres(raw_value, raw_unit)
        datum_line = next(
            (
                " ".join(line.strip().split())[:300]
                for line in text.splitlines()
                if _SEISMIC_DATUM_STATEMENT.search(line)
            ),
            datum_match.group(0)[:300],
        )
        explicit_msl = bool(_EXPLICIT_MSL.search(datum_line))
        add_candidate(
            "seismic_srd_elevation_m",
            value_m,
            confidence=0.97 if explicit_msl else 0.82,
            status="verified" if explicit_msl else "candidate",
            candidate_evidence=[
                datum_line,
                (
                    "基准高程明确相对MSL"
                    if explicit_msl
                    else "Final/Processing Datum数值明确，但相对MSL关系需一次确认"
                ),
            ],
            requires_confirmation=not explicit_msl,
            auto_applied=explicit_msl,
        )

    depth_datum_match = re.search(
        r"(?:DEPTH[_ ]+(?:REFERENCE|DATUM)|深度(?:参考|基准))"
        r"[^\r\n]{0,40}[^A-Z0-9](KB|GL|DF|RT)(?:[^A-Z0-9]|$)",
        upper,
    )
    depth_datum = depth_datum_match.group(1) if depth_datum_match else None
    if depth_datum:
        evidence.append(f"深度起算面识别为{depth_datum}")

    if re.search(r"TVDSS[^\r\n]{0,50}(?:POSITIVE\s+DOWN|DEPTH\s+BELOW\s+MSL|向下为正|海平面以下深度)", upper):
        depth_convention = "depth_below_msl_positive_down"
    elif re.search(r"TVDSS[^\r\n]{0,50}(?:POSITIVE\s+UP|ELEVATION|向上为正|绝对高程)", upper):
        depth_convention = "elevation_positive_up"
    else:
        depth_convention = None

    velocity_match = _REPLACEMENT_VELOCITY_EVIDENCE.search(upper)
    replacement_velocity_mps = None
    if velocity_match:
        tail = upper[velocity_match.end() : velocity_match.end() + 80]
        number = _NUMBER.search(tail)
        if number:
            value = float(number.group(0).replace(",", ""))
            replacement_velocity_mps = value * 0.3048 if re.search(r"FT\s*/\s*S|FPS", tail) else value
            evidence.append(f"替换速度={replacement_velocity_mps:g}m/s")

    offset_match = re.search(r"MD_OFFSET_TO_TRAJECTORY(?:_M)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", upper)
    md_offset = float(offset_match.group(1)) if offset_match else None
    if md_offset is not None:
        evidence.append(f"md_offset_to_trajectory_m={md_offset:g}")

    return TimeReferenceMetadata(
        time_reference=time_reference,
        time_domain=time_domain,
        correction_state=correction_state,
        depth_unit=depth_unit,
        time_unit=time_unit,
        depth_datum=depth_datum,
        depth_convention=depth_convention,
        replacement_velocity_mps=replacement_velocity_mps,
        md_offset_to_trajectory_m=md_offset,
        provenance=str(source_path),
        evidence=tuple(evidence),
        contract_candidates=tuple(candidates),
    )


def correct_time_to_srd(
    time_ms: Any,
    *,
    source_elevation_msl_m: float,
    srd_elevation_msl_m: float,
    replacement_velocity_mps: float,
    time_domain: str,
) -> Any:
    """Apply only the near-surface datum static needed to reference time to SRD."""

    velocity = float(replacement_velocity_mps)
    if velocity <= 0:
        raise ValueError("替换速度必须大于0")
    fold = {"OWT": 1.0, "TWT": 2.0}.get(str(time_domain).upper())
    if fold is None:
        raise ValueError("时间域必须明确为OWT或TWT")
    correction_ms = fold * (float(srd_elevation_msl_m) - float(source_elevation_msl_m)) / velocity * 1000.0
    return time_ms + correction_ms


def observations_from_asset_options(
    options: dict[str, Any],
    *,
    source: str,
    entity_kind: str,
    entity_name: str | None = None,
) -> list[DatumObservation]:
    mappings = (
        ("srd_elevation_m", "SRD"),
        ("processing_datum_elevation_m", "SRD"),
        ("kb_elevation_m", "KB"),
        ("ground_elevation_m", "GL"),
    )
    result: list[DatumObservation] = []
    for key, datum in mappings:
        value = options.get(key)
        if value is None:
            continue
        result.append(observation_from_value(
            datum=datum,
            value=float(value),
            source=source,
            entity_kind=entity_kind,
            entity_name=entity_name,
            unit="m",
            evidence=f"asset.options.{key}={value} m MSL",
            confidence=1.0,
        ))
    return result


def resolve_vertical_datum(
    observations: Iterable[DatumObservation],
    *,
    entity_kind: str,
    entity_name: str,
    tolerance_m: float = 0.5,
    allow_gl_as_well_depth_reference: bool = False,
    maximum_kb_gl_offset_m: float = 30.0,
) -> ResolvedVerticalDatum:
    items = tuple(item for item in observations if item.absolute_elevation_m is not None)
    if entity_kind == "seismic":
        priority = ("SRD",)
    else:
        priority = ("KB", "DF", "RT") + (("GL",) if allow_gl_as_well_depth_reference else ())
    chosen_datum = next((name for name in priority if any(item.datum == name for item in items)), None)
    candidates = [item for item in items if item.datum == chosen_datum]
    chosen = max(candidates, key=lambda item: (not item.review_required, item.confidence)) if candidates else None
    conflicts: list[str] = []
    if chosen is not None:
        chosen_value = float(chosen.absolute_elevation_m)
        for item in candidates:
            value = float(item.absolute_elevation_m)
            if abs(value - chosen_value) > float(tolerance_m):
                conflicts.append(
                    f"{chosen_datum}绝对高程冲突:{chosen_value:.3f}m 与 {value:.3f}m；来源={chosen.source} | {item.source}"
                )

    if entity_kind == "well":
        kb_values = [float(item.absolute_elevation_m) for item in items if item.datum == "KB"]
        gl_values = [float(item.absolute_elevation_m) for item in items if item.datum == "GL"]
        if kb_values and gl_values:
            offset = max(kb_values, key=lambda value: value) - max(gl_values, key=lambda value: value)
            if offset < -float(tolerance_m) or offset > float(maximum_kb_gl_offset_m):
                conflicts.append(f"KB-GL高差异常:{offset:.3f}m，允许范围约为0~{maximum_kb_gl_offset_m:g}m")

    review_required = bool(chosen.review_required) if chosen is not None else True
    ready = bool(chosen is not None and not review_required and not conflicts)
    return ResolvedVerticalDatum(
        entity_kind=entity_kind,
        entity_name=entity_name,
        datum=chosen_datum,
        absolute_elevation_m=None if chosen is None else float(chosen.absolute_elevation_m),
        source="" if chosen is None else chosen.source,
        evidence="" if chosen is None else chosen.evidence,
        confidence=0.0 if chosen is None else chosen.confidence,
        ready=ready,
        review_required=review_required,
        conflicts=tuple(dict.fromkeys(conflicts)),
        observations=items,
    )


def absolute_elevation_from_tvd(datum_elevation_m: float, tvd_m: Any) -> Any:
    """Convert TVD measured downward from a datum to MSL elevation (up positive)."""

    return float(datum_elevation_m) - tvd_m


def depth_below_srd(srd_elevation_m: float, absolute_elevation_m: Any) -> Any:
    """Convert an MSL elevation to depth below seismic reference datum."""

    return float(srd_elevation_m) - absolute_elevation_m
