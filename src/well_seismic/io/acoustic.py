from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..knowledge import CurveKnowledgeBase
from ..models import Evidence, WellLog


_DEPTH_HEADERS = {"MEASUREDDEPTH", "MD"}
_SONIC_HEADERS = {"P_AC", "PAC", "DT", "DTC", "AC"}


def _tokens(line: str) -> list[str]:
    return [token for token in re.split(r"[\s,;]+", line.lstrip("#").strip()) if token]


def _normalized_header(line: str) -> list[str]:
    return [
        re.sub(r"[^A-Za-z0-9_]", "", token).upper()
        for token in _tokens(line)
    ]


def _float_row(line: str) -> list[float] | None:
    try:
        return [
            float(value.replace("D", "E").replace("d", "e"))
            for value in _tokens(line)
        ]
    except ValueError:
        return None


def read_acoustic_text(
    path: str | Path,
    knowledge: CurveKnowledgeBase,
    preprocessing: dict[str, Any] | None = None,
) -> WellLog:
    """Read the explicit two-column ``measuredDepth p_ac`` text contract.

    ``.ac`` is not treated as a generic delimited file. A recognized header is
    mandatory so column order and physical meaning are never guessed. This
    legacy contract defines measured depth in metres and P-wave slowness in
    microseconds per metre.
    """

    source = Path(path)
    if source.suffix.casefold() != ".ac":
        raise ValueError(f"Acoustic text reader only accepts .ac files: {source}")
    raw = source.read_bytes()
    text = None
    encoding = ""
    for candidate in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeError(f"Cannot decode acoustic text file: {source}")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty acoustic text file: {source}")
    normalized = _normalized_header(lines[0])
    well_name = source.stem
    surface_xy: tuple[float, float] | None = None
    declared_depth_range: tuple[float, float] | None = None
    data_start = 1
    if (
        len(normalized) == 2
        and normalized[0] in _DEPTH_HEADERS
        and normalized[1] in _SONIC_HEADERS
    ):
        curve_name = normalized[1]
    elif len(lines) >= 8:
        # Explicit Landmark-style acoustic export:
        # WELL / X Y / 1 / depth range / 1 / Depth AC ORIGnnn / values.
        well_tokens = _tokens(lines[0])
        xy = _float_row(lines[1])
        depth_range = _float_row(lines[3])
        columns = _normalized_header(lines[5])
        control_one = _tokens(lines[2]) == ["1"] and _tokens(lines[4]) == ["1"]
        if (
            len(well_tokens) == 1
            and re.fullmatch(r"[A-Za-z0-9_.-]+", well_tokens[0])
            and xy is not None
            and len(xy) == 2
            and depth_range is not None
            and len(depth_range) == 2
            and control_one
            and len(columns) == 3
            and columns[0] in {"DEPTH", "MEASUREDDEPTH", "MD"}
            and columns[1] in _SONIC_HEADERS
            and columns[2].startswith("ORIG")
        ):
            well_name = well_tokens[0]
            surface_xy = (xy[0], xy[1])
            declared_depth_range = (depth_range[0], depth_range[1])
            curve_name = columns[1]
            data_start = 6
        else:
            curve_name = ""
    else:
        curve_name = ""
    if not curve_name:
        raise ValueError(
            "Unsupported .ac contract; expected 'measuredDepth p_ac' or the "
            f"explicit Landmark acoustic layout: {source}"
        )

    rows: list[tuple[float, float]] = []
    for line_number, line in enumerate(lines[data_start:], start=data_start + 1):
        if line.lstrip().startswith("#"):
            continue
        values = _tokens(line)
        if len(values) != 2:
            raise ValueError(f"Malformed .ac row {line_number}: expected 2 columns")
        try:
            md, sonic = (float(value.replace("D", "E").replace("d", "e")) for value in values)
        except ValueError as exc:
            raise ValueError(f"Malformed .ac row {line_number}: non-numeric value") from exc
        rows.append((md, sonic))
    if len(rows) < 2:
        raise ValueError(f"Acoustic text file requires at least 2 samples: {source}")

    data = np.asarray(rows, dtype=float)
    md = data[:, 0]
    sonic = data[:, 1]
    if not np.all(np.isfinite(md)):
        raise ValueError(f"Acoustic text measured depth contains non-finite values: {source}")
    if np.any(np.diff(md) <= 0.0):
        raise ValueError(f"Acoustic text measured depth must be strictly increasing: {source}")
    if declared_depth_range is not None:
        step = float(np.median(np.diff(md)))
        if (
            abs(float(md[0]) - declared_depth_range[0]) > max(step, 1e-6)
            or abs(float(md[-1]) - declared_depth_range[1]) > max(step, 1e-6)
        ):
            raise ValueError(f"Acoustic text samples conflict with declared depth range: {source}")

    nulls = [float(value) for value in (preprocessing or {}).get("null_values", [])]
    tolerance = float((preprocessing or {}).get("null_tolerance", 1e-6))
    for null in nulls:
        sonic[np.isclose(sonic, null, rtol=0.0, atol=tolerance)] = np.nan

    info = knowledge.identify(curve_name, "us/m", "P-wave acoustic slowness", sonic)
    if info.standard_name != "DT":
        raise ValueError(f"Explicit .ac sonic column did not resolve to DT: {source}")
    info.source = str(source)
    info, standardized, mask, issues = knowledge.standardize(info, sonic)
    if int(np.sum(mask)) < 2:
        raise ValueError(f"Acoustic text file has fewer than 2 valid DT samples: {source}")

    evidence = {
        "encoding": Evidence(encoding, "INFERRED", str(source), 0.8),
        "WELL": Evidence(
            well_name,
            "FILE_HEADER" if well_name != source.stem else "FILENAME",
            str(source),
            1.0 if well_name != source.stem else 0.95,
        ),
        "DEPTH_UNIT": Evidence("m", "FORMAT_CONTRACT", str(source), 1.0),
        "DT_UNIT": Evidence("us/m", "FORMAT_CONTRACT", str(source), 1.0),
    }
    if surface_xy is not None:
        evidence["XCRD"] = Evidence(surface_xy[0], "FILE_HEADER", str(source), 1.0)
        evidence["YCRD"] = Evidence(surface_xy[1], "FILE_HEADER", str(source), 1.0)
    return WellLog(
        well_name=well_name,
        depth=md,
        curves={"DT": standardized},
        masks={"DT": mask},
        curve_info={"DT": info},
        header=evidence,
        source=str(source),
        version="explicit_acoustic_text_v1",
        issues=[f"P_AC:{issue}" for issue in issues],
        processing_steps=[
            "explicit_acoustic_text_contract:measuredDepth_m+p_ac_us_per_m",
        ],
    )


__all__ = ["read_acoustic_text"]
