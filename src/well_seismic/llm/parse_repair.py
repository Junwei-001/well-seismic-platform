"""Fail-closed, in-memory repair contracts for unusual trajectory tables.

The LLM is deliberately kept outside the parser.  It receives a bounded
structural summary and may only propose delimiter, column-index and source-unit
metadata.  Deterministic code validates the proposal and reparses the original
read-only bytes; no generated code or filesystem operation is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..models import Trajectory


REPAIR_CONTRACT_VERSION = "well-seismic.llm-parse-repair.v1"

_DELIMITERS: dict[str, str | None] = {
    "comma": ",",
    "tab": "\t",
    "semicolon": ";",
    "pipe": "|",
    "whitespace": None,
}

_COLUMN_FIELDS = (
    "well_name",
    "md",
    "tvd",
    "tvdss",
    "inclination",
    "azimuth",
    "x",
    "y",
    "x_offset",
    "y_offset",
)
_LENGTH_FIELDS = ("md", "tvd", "tvdss", "x", "y", "x_offset", "y_offset")

_TRAJECTORY_HEADER_ALIASES = {
    "md": {
        "MD", "MDM", "MDFT", "MEASUREDDEPTH", "MEASUREDDEPTHM",
        "MEASUREDDEPTHFT", "DEPTH", "测深",
    },
    "tvd": {
        "TVD", "TVDM", "TVDFT", "VERTICALDEPTH", "VERTICALDEPTHM",
        "VERTICALDEPTHFT", "垂深",
    },
    "tvdss": {"TVDSS", "TVDSSM", "TVDSSFT", "SUBSEADEPTH", "海拔校正垂深"},
    "inclination": {"INC", "INCL", "INCLINATION", "DEV", "井斜"},
    "azimuth": {"AZI", "AZIM", "AZIMUTH", "方位"},
    "x": {"X", "XM", "XFT", "XCRD", "EASTING"},
    "y": {"Y", "YM", "YFT", "YCRD", "NORTHING"},
    "x_offset": {"XOFFSET", "XOFFSETM", "XOFFSETFT", "EASTINGOFFSET", "DEPARTUREEW"},
    "y_offset": {"YOFFSET", "YOFFSETM", "YOFFSETFT", "NORTHINGOFFSET", "DEPARTURENS"},
}

_WELL_NAME_HEADER_ALIASES = {
    "WELL", "WELLNAME", "WELLID", "UWI", "API", "井名", "井号",
}


TRAJECTORY_PARSE_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "delimiter": {"type": "string", "enum": sorted(_DELIMITERS)},
        "columns": {
            "type": "object",
            "properties": {
                field: {"type": ["integer", "null"], "minimum": 0, "maximum": 63}
                for field in _COLUMN_FIELDS
            },
            "required": list(_COLUMN_FIELDS),
            "additionalProperties": False,
        },
        "field_units": {
            "type": "object",
            "properties": {
                field: {"type": "string", "enum": ["m", "ft", "unknown"]}
                for field in _LENGTH_FIELDS
            },
            "required": list(_LENGTH_FIELDS),
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
        "warnings": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 200},
        },
    },
    "required": ["delimiter", "columns", "field_units", "confidence", "reason", "warnings"],
    "additionalProperties": False,
}


def _decode_prefix(path: Path, *, max_bytes: int = 65_536) -> str:
    with path.open("rb") as stream:
        raw = stream.read(max_bytes)
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(line: str, delimiter_name: str) -> list[str]:
    delimiter = _DELIMITERS[delimiter_name]
    if delimiter is None:
        return [item for item in re.split(r"[\s,\t]+", line.strip()) if item]
    return [item.strip() for item in line.split(delimiter)]


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _normalize_header_token(token: str) -> str:
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]", "", str(token).upper())


def _header_field(token: str) -> str | None:
    normalized = _normalize_header_token(token)
    if normalized in _WELL_NAME_HEADER_ALIASES:
        return "well_name"
    matches = [
        field
        for field, aliases in _TRAJECTORY_HEADER_ALIASES.items()
        if normalized in aliases
    ]
    return matches[0] if len(matches) == 1 else None


def _header_has_trajectory_semantics(header: Sequence[str]) -> bool:
    normalized = {_normalize_header_token(token) for token in header}
    has_md = bool(normalized & _TRAJECTORY_HEADER_ALIASES["md"])
    geometry_aliases = set().union(
        *(
            _TRAJECTORY_HEADER_ALIASES[field]
            for field in _COLUMN_FIELDS
            if field not in {"well_name", "md"}
        )
    )
    return has_md and bool(normalized & geometry_aliases)


def summarize_tabular_source(path: str | Path) -> dict[str, Any]:
    """Return bounded schema/statistical evidence without sending raw rows."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    text = _decode_prefix(source)
    lines = [line.strip() for line in text.splitlines() if line.strip()][:160]
    candidates: list[dict[str, Any]] = []
    delimiter_names = ("comma", "tab", "semicolon", "pipe", "whitespace")
    for delimiter_name in delimiter_names:
        best: dict[str, Any] | None = None
        for header_index, raw_header in enumerate(lines[:80]):
            header_text = raw_header.lstrip("#").strip()
            header = _split(header_text, delimiter_name)
            if not (2 <= len(header) <= 64) or not any(
                re.search(r"[A-Za-z\u4e00-\u9fff]", token) for token in header
            ):
                continue
            if not _header_has_trajectory_semantics(header):
                continue
            numeric_rows: list[list[float | None]] = []
            for raw_row in lines[header_index + 1 : header_index + 41]:
                if raw_row.startswith("#"):
                    continue
                tokens = _split(raw_row, delimiter_name)
                if len(tokens) != len(header):
                    continue
                values = [_number(token) for token in tokens]
                # A well-name column is allowed; all remaining columns should
                # still be predominantly numeric.
                if sum(value is not None for value in values) >= max(1, len(header) - 1):
                    numeric_rows.append(values)
            score = len(numeric_rows) * len(header)
            if not numeric_rows or (best is not None and score <= int(best["score"])):
                continue
            column_stats: list[dict[str, Any]] = []
            for index in range(len(header)):
                values = np.asarray(
                    [row[index] for row in numeric_rows if row[index] is not None],
                    dtype=float,
                )
                column_stats.append(
                    {
                        "index": index,
                        "header": header[index][:80],
                        "numeric_count": int(values.size),
                        "min": float(np.min(values)) if values.size else None,
                        "median": float(np.median(values)) if values.size else None,
                        "max": float(np.max(values)) if values.size else None,
                        "nondecreasing": bool(
                            values.size >= 2 and np.all(np.diff(values) >= 0)
                        ),
                    }
                )
            best = {
                "delimiter": delimiter_name,
                "header_line_index": header_index,
                "column_count": len(header),
                "sampled_data_rows": len(numeric_rows),
                "columns": column_stats,
                "score": score,
            }
        if best is not None:
            best.pop("score", None)
            candidates.append(best)
    candidates.sort(
        key=lambda item: (int(item["sampled_data_rows"]), int(item["column_count"])),
        reverse=True,
    )
    return {
        "suffix": source.suffix.casefold(),
        "size_bytes": source.stat().st_size,
        "source_sha256": _sha256_file(source),
        "candidate_tables": candidates[:6],
        "evidence_policy": "headers_and_column_statistics_only_no_raw_rows",
    }


def validate_trajectory_parse_patch(
    proposal: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    manifest_columns: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Compile untrusted JSON into the tiny options surface accepted by readers."""

    errors: list[str] = []
    allowed_top = {"delimiter", "columns", "field_units", "confidence", "reason", "warnings"}
    unexpected = sorted(set(proposal) - allowed_top)
    if unexpected:
        errors.append("unexpected_keys:" + ",".join(unexpected))

    missing_top = sorted(allowed_top - set(proposal))
    if missing_top:
        errors.append("missing_required_keys:" + ",".join(missing_top))

    raw_delimiter = proposal.get("delimiter")
    delimiter_name = raw_delimiter if isinstance(raw_delimiter, str) else ""
    if not isinstance(raw_delimiter, str):
        errors.append("delimiter_must_be_string")
    elif delimiter_name not in _DELIMITERS:
        errors.append("delimiter_not_allowlisted")

    raw_candidates = evidence.get("candidate_tables")
    candidates = (
        [item for item in raw_candidates if isinstance(item, Mapping)]
        if isinstance(raw_candidates, list)
        else []
    )
    if not candidates:
        errors.append("no_trajectory_table_candidate")
    matching_candidates = [
        item for item in candidates
        if str(item.get("delimiter") or "") == delimiter_name
    ]
    if len(matching_candidates) != 1:
        errors.append("delimiter_does_not_select_one_candidate")
        selected_candidate: Mapping[str, Any] | None = None
    else:
        selected_candidate = matching_candidates[0]
    selected_width = (
        int(selected_candidate.get("column_count") or 0)
        if selected_candidate is not None
        else 0
    )
    columns_raw = proposal.get("columns")
    if not isinstance(columns_raw, Mapping):
        errors.append("columns_must_be_object")
        columns_raw = {}
    unexpected_columns = sorted(set(columns_raw) - set(_COLUMN_FIELDS))
    if unexpected_columns:
        errors.append("column_fields_not_allowlisted:" + ",".join(unexpected_columns))
    missing_columns = sorted(set(_COLUMN_FIELDS) - set(columns_raw))
    if missing_columns:
        errors.append("missing_column_fields:" + ",".join(missing_columns))
    columns: dict[str, int] = {}
    for field in _COLUMN_FIELDS:
        value = columns_raw.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 63:
            errors.append(f"invalid_column_index:{field}")
            continue
        if selected_width <= 0 or value >= selected_width:
            errors.append(f"column_index_out_of_evidence:{field}")
            continue
        columns[field] = value
    if columns and "md" not in columns:
        errors.append("explicit_columns_require_md")
    duplicate_indexes = {
        index for index in columns.values() if list(columns.values()).count(index) > 1
    }
    if duplicate_indexes:
        errors.append("duplicate_column_indexes")

    declared_columns = {
        str(field): int(index)
        for field, index in (manifest_columns or {}).items()
        if (
            str(field) in _COLUMN_FIELDS
            and isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index <= 63
        )
    }
    if selected_candidate is not None:
        candidate_columns = selected_candidate.get("columns")
        header_by_index = {
            int(item["index"]): str(item.get("header") or "")
            for item in candidate_columns
            if (
                isinstance(candidate_columns, list)
                and isinstance(item, Mapping)
                and isinstance(item.get("index"), int)
                and not isinstance(item.get("index"), bool)
            )
        } if isinstance(candidate_columns, list) else {}
        recognized_by_index = {
            index: _header_field(header)
            for index, header in header_by_index.items()
        }
        for field, index in columns.items():
            recognized = recognized_by_index.get(index)
            if recognized is not None and recognized != field:
                errors.append(
                    f"column_semantic_mismatch:{field}:{index}:{recognized}"
                )
            elif recognized is None and declared_columns.get(field) != index:
                errors.append(f"column_semantic_unknown:{field}:{index}")
        for index, recognized in recognized_by_index.items():
            if recognized is None:
                continue
            if columns.get(recognized) != index:
                errors.append(
                    f"known_header_not_mapped:{recognized}:{index}"
                )

    units_raw = proposal.get("field_units")
    if not isinstance(units_raw, Mapping):
        errors.append("field_units_must_be_object")
        units_raw = {}
    unexpected_units = sorted(set(units_raw) - set(_LENGTH_FIELDS))
    if unexpected_units:
        errors.append("unit_fields_not_allowlisted:" + ",".join(unexpected_units))
    missing_units = sorted(set(_LENGTH_FIELDS) - set(units_raw))
    if missing_units:
        errors.append("missing_unit_fields:" + ",".join(missing_units))
    units: dict[str, str] = {}
    for field in _LENGTH_FIELDS:
        raw_value = units_raw.get(field)
        if not isinstance(raw_value, str):
            errors.append(f"unit_must_be_string:{field}")
            continue
        value = raw_value.casefold()
        if value == "unknown":
            continue
        if value not in {"m", "ft"}:
            errors.append(f"unit_not_allowlisted:{field}")
            continue
        units[field] = value

    raw_confidence = proposal.get("confidence")
    if (
        isinstance(raw_confidence, bool)
        or not isinstance(raw_confidence, (int, float))
        or not math.isfinite(float(raw_confidence))
        or not 0 <= float(raw_confidence) <= 1
    ):
        confidence = 0.0
        errors.append("invalid_confidence")
    else:
        confidence = float(raw_confidence)

    raw_reason = proposal.get("reason")
    if not isinstance(raw_reason, str):
        errors.append("reason_must_be_string")
        reason = ""
    elif len(raw_reason) > 500:
        errors.append("reason_too_long")
        reason = raw_reason[:500]
    else:
        reason = raw_reason

    raw_warnings = proposal.get("warnings")
    warnings: list[str] = []
    if not isinstance(raw_warnings, list):
        errors.append("warnings_must_be_array")
    else:
        if len(raw_warnings) > 8:
            errors.append("too_many_warnings")
        for index, item in enumerate(raw_warnings[:8]):
            if not isinstance(item, str):
                errors.append(f"warning_must_be_string:{index}")
                continue
            if len(item) > 200:
                errors.append(f"warning_too_long:{index}")
            warnings.append(item[:200])

    # An empty patch is not a repair.  Unknown is deliberately omitted rather
    # than converted to zero or another guessed value.
    if not columns and not units and delimiter_name == "whitespace":
        errors.append("empty_patch")

    options: dict[str, Any] = {}
    if delimiter_name in _DELIMITERS and _DELIMITERS[delimiter_name] is not None:
        options["delimiter"] = _DELIMITERS[delimiter_name]
    if columns:
        options["columns"] = columns
    if units:
        options["field_units"] = units
    compiled = {
        "options": options,
        "delimiter_name": delimiter_name,
        "confidence": confidence,
        "reason": reason,
        "warnings": warnings,
    }
    return compiled, errors


def validate_trajectory_physics(
    trajectories: Sequence[Trajectory],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Apply conservative format and physical gates after isolated reparse."""

    errors: list[str] = []
    spans: list[float] = []
    for trajectory in trajectories:
        md = np.asarray(trajectory.md, dtype=float)
        if md.size < 2 or not np.isfinite(md).all():
            errors.append(f"{trajectory.well_name}:md_requires_two_finite_rows")
            continue
        delta = np.diff(md)
        if np.any(delta < 0) or float(md[-1] - md[0]) <= 0:
            errors.append(f"{trajectory.well_name}:md_not_monotonic_with_positive_span")
        span = float(md[-1] - md[0])
        spans.append(span)
        if float(np.nanmin(md)) < -100 or float(np.nanmax(md)) > 20_000:
            errors.append(f"{trajectory.well_name}:md_outside_supported_physical_range_m")
        if "md" not in trajectory.source_units:
            errors.append(f"{trajectory.well_name}:md_source_unit_unresolved")
        if "missing_deviation_survey_no_md_to_tvd_conversion" in trajectory.issues:
            errors.append(f"{trajectory.well_name}:geometry_unknown_not_zero_filled")

        tvd = np.asarray(trajectory.tvd, dtype=float)
        finite_tvd = np.isfinite(tvd)
        if finite_tvd.any():
            if np.nanmax(np.abs(tvd[finite_tvd])) > 20_000:
                errors.append(f"{trajectory.well_name}:tvd_outside_supported_physical_range_m")
            if tvd.size == md.size and np.all(finite_tvd):
                tvd_step = np.abs(np.diff(tvd))
                if np.any(tvd_step > np.abs(delta) * 1.05 + 1e-6):
                    errors.append(f"{trajectory.well_name}:tvd_step_exceeds_md_step")

        if trajectory.inclination is not None:
            inclination = np.asarray(trajectory.inclination, dtype=float)
            if not np.isfinite(inclination).all() or np.any((inclination < 0) | (inclination > 180)):
                errors.append(f"{trajectory.well_name}:inclination_out_of_range")
        if trajectory.azimuth is not None:
            azimuth = np.asarray(trajectory.azimuth, dtype=float)
            if not np.isfinite(azimuth).all() or np.any(np.abs(azimuth) > 720):
                errors.append(f"{trajectory.well_name}:azimuth_out_of_range")
        for field in ("x", "y", "x_offset", "y_offset"):
            value = getattr(trajectory, field)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all() or np.any(np.abs(array) > 1e8):
                errors.append(f"{trajectory.well_name}:{field}_invalid")

    summary = {
        "trajectory_count": len(trajectories),
        "well_names": [trajectory.well_name for trajectory in trajectories[:50]],
        "md_span_m_min": min(spans) if spans else None,
        "md_span_m_max": max(spans) if spans else None,
    }
    if not trajectories:
        errors.append("reparse_produced_no_trajectory")
    return not errors, errors, summary


def repair_fingerprint(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "REPAIR_CONTRACT_VERSION",
    "TRAJECTORY_PARSE_PATCH_SCHEMA",
    "repair_fingerprint",
    "summarize_tabular_source",
    "validate_trajectory_parse_patch",
    "validate_trajectory_physics",
]
