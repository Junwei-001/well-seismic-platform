from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..coordinate_reference import (
    CoordinateReferenceError,
    canonical_crs_id,
    parse_crs,
    transform_xy,
)
from ..models import Trajectory, WellHead
from ..trajectory import minimum_curvature
from ..vertical_datum import length_to_metres
from ..well_identity import canonical_api12_values
from .ooxml import OOXMLReadError, read_single_table_sheet
from .trajectory_canonicalization import canonicalize_station_rows

_PETREL_DEV_ALIASES = {
    "well_name": ("WELL", "WELLNAME", "UWI", "UWID"),
    "well_identifier": (
        "WELLAPI",
        "API",
        "APIID",
        "APINUMBER",
        "APIN",
    ),
    "md": ("MD", "MEASUREDDEPTH", "DEPTH"),
    "x": ("X", "EASTING"),
    "y": ("Y", "NORTHING"),
    "tvd": ("TVD", "VERTICALDEPTH"),
    "tvdss": ("TVDSS", "SUBSEADEPTH"),
    "x_offset": ("DX", "XOFFSET", "DEPARTUREEW"),
    "y_offset": ("DY", "YOFFSET", "DEPARTURENS"),
    "inclination": ("INCL", "INC", "INCLINATION", "DEV"),
    "azimuth": ("AZIMTN", "AZIMGN", "AZI", "AZIMUTH"),
}

_LENGTH_UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "yd": "yd",
    "yard": "yd",
    "yards": "yd",
    "米": "m",
    "公尺": "m",
    "英尺": "ft",
    "厘米": "cm",
    "公分": "cm",
    "毫米": "mm",
    "千米": "km",
    "公里": "km",
}
_HEADER_UNIT = "|".join(
    sorted((re.escape(item) for item in _LENGTH_UNIT_ALIASES), key=len, reverse=True)
)
_BRACKETED_HEADER_UNIT = re.compile(
    rf"^\s*(.*?)\s*[\(\[{{]\s*({_HEADER_UNIT})\s*[\)\]}}]\s*$",
    re.IGNORECASE,
)
_SUFFIX_HEADER_UNIT = re.compile(
    rf"^\s*(.*?)\s*[_\-.]\s*({_HEADER_UNIT})\s*$",
    re.IGNORECASE,
)


def _normalize_length_unit(value: Any) -> str | None:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip().casefold().replace(".", "")
    return _LENGTH_UNIT_ALIASES.get(text)


def _crs_source_and_target(
    options: dict[str, Any],
    dev_metadata: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    metadata = dev_metadata or {}
    source = metadata.get("crs") or options.get("source_crs") or options.get("crs")
    target = options.get("target_crs")
    return (
        str(source).strip() if source else None,
        str(target).strip() if target else None,
    )


def _source_crs_unit(source_crs: str) -> str:
    crs = parse_crs(source_crs, field="源CRS")
    if crs.is_geographic:
        return "degree"
    axes = tuple(crs.axis_info[:2])
    if len(axes) != 2 or any(axis.unit_conversion_factor is None for axis in axes):
        raise CoordinateReferenceError("源CRS缺少可用的水平轴单位")
    factors = [float(axis.unit_conversion_factor) for axis in axes]
    if all(np.isclose(value, 1.0, rtol=0.0, atol=1e-12) for value in factors):
        return "m"
    if all(
        np.isclose(value, 0.3048, rtol=0.0, atol=1e-8)
        or np.isclose(value, 1200.0 / 3937.0, rtol=0.0, atol=1e-8)
        for value in factors
    ):
        return "ft"
    return "crs_native"


def _validate_coordinate_unit_against_crs(
    field: str,
    *,
    source_crs: str,
    column_index: int | None,
    header: list[str] | None,
    options: dict[str, Any],
) -> str:
    native = _source_crs_unit(source_crs)
    header_unit, _ = _header_unit(header, column_index)
    mapping_unit, _ = _mapping_unit(options, field, context="Horizontal coordinate")
    declared = header_unit or mapping_unit
    if declared is None:
        for key in ("coordinate_unit", "horizontal_unit"):
            candidate = _normalize_length_unit(options.get(key))
            if candidate:
                declared = candidate
                break
    if native == "degree" and declared is not None:
        raise CoordinateReferenceError(
            f"{field}源CRS是经纬度，但字段声明了长度单位{declared}；拒绝把米/英尺当作度"
        )
    if native in {"m", "ft"} and declared is not None and declared != native:
        raise CoordinateReferenceError(
            f"{field}字段单位{declared}与源CRS轴单位{native}冲突"
        )
    return native


def _column_name_and_unit(value: str) -> tuple[str, str | None]:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    for pattern in (_BRACKETED_HEADER_UNIT, _SUFFIX_HEADER_UNIT):
        match = pattern.match(text)
        if match and match.group(1).strip():
            return match.group(1).strip(), _normalize_length_unit(match.group(2))
    return text, None


def _normalized_column(value: str) -> str:
    semantic_name, _ = _column_name_and_unit(value)
    normalized = unicodedata.normalize("NFKC", semantic_name).upper()
    # ``str.isalnum`` keeps CJK and other Unicode letters while still
    # removing separators.  The previous ASCII-only regex collapsed every
    # Chinese header to the empty string, so unrelated columns silently
    # overwrote each other in mapping dictionaries.
    return "".join(character for character in normalized if character.isalnum())


class ColumnMappingConflict(ValueError):
    """Raised when an automatic field mapping has more than one safe answer."""

    def __init__(self, conflicts: list[str]):
        self.conflicts = tuple(dict.fromkeys(conflicts))
        super().__init__("ambiguous_column_mapping:" + "|".join(self.conflicts))


class VerticalDatumUnitConflict(ValueError):
    """Raised when tabular elevation-unit evidence cannot be reconciled."""


def _map_columns_from_aliases(
    header: list[str],
    aliases: dict[str, list[str] | tuple[str, ...]],
    *,
    priority_fields: set[str] | frozenset[str] = frozenset({"depth"}),
) -> dict[str, int]:
    """Map an unambiguous header and fail closed when human review is needed.

    ``depth`` is a deliberate dispatcher alias used to retain the concrete
    MD/TVD/TVDSS domain for time-depth tables.  Its configured alias order is
    therefore a deterministic priority, while ordinary physical fields must
    never silently choose between two source columns.
    """

    normalized_aliases = {
        _normalized_column(str(name))
        for raw_names in aliases.values()
        for name in ([raw_names] if isinstance(raw_names, str) else raw_names)
    }
    normalized_aliases.discard("")

    available: dict[str, list[int]] = {}
    conflicts: list[str] = []
    for index, name in enumerate(header):
        key = _normalized_column(name)
        if not key:
            continue
        available.setdefault(key, []).append(index)
    for key, indices in available.items():
        if len(indices) > 1 and key in normalized_aliases:
            labels = ",".join(f"{index}:{header[index]}" for index in indices)
            conflicts.append(f"duplicate_normalized_header:{key}={labels}")

    columns: dict[str, int] = {}
    for field, raw_names in aliases.items():
        names = [raw_names] if isinstance(raw_names, str) else list(raw_names)
        candidates: list[int] = []
        for name in names:
            key = _normalized_column(str(name))
            if not key:
                continue
            for index in available.get(key, []):
                if index not in candidates:
                    candidates.append(index)
        if not candidates:
            continue
        if len(candidates) > 1 and field not in priority_fields:
            labels = ",".join(f"{index}:{header[index]}" for index in candidates)
            conflicts.append(f"field_matches_multiple_columns:{field}={labels}")
            continue
        columns[field] = candidates[0]

    fields_by_column: dict[int, list[str]] = {}
    for field, index in columns.items():
        fields_by_column.setdefault(index, []).append(field)
    for index, fields in fields_by_column.items():
        if len(fields) <= 1:
            continue
        field_set = set(fields)
        # The generic ``depth`` field intentionally shares the concrete depth
        # column so _time_depth_domain can report MD/TVD/TVDSS without guessing.
        allowed_depth_share = (
            "depth" in field_set
            and len(field_set - {"depth", "md", "tvd", "tvdss"}) == 0
            and len(field_set - {"depth"}) == 1
        )
        if not allowed_depth_share:
            conflicts.append(
                "column_matches_multiple_fields:"
                f"{index}:{header[index]}={','.join(sorted(fields))}"
            )

    if conflicts:
        raise ColumnMappingConflict(conflicts)
    return columns


def _header_unit(header: list[str] | None, index: int | None) -> tuple[str | None, str | None]:
    if header is None or index is None or index < 0 or index >= len(header):
        return None, None
    _, unit = _column_name_and_unit(header[index])
    return unit, f"header:{header[index]}" if unit else None


def _mapping_unit(
    options: dict[str, Any],
    field: str,
    *,
    context: str = "Trajectory",
) -> tuple[str | None, str | None]:
    for option_name in ("field_units", "trajectory_units", "units"):
        raw = options.get(option_name)
        if not isinstance(raw, dict):
            continue
        normalized = {
            _normalized_column(str(key)): value
            for key, value in raw.items()
        }
        value = normalized.get(_normalized_column(field))
        if value is not None:
            if str(value).strip().casefold() in {"", "unknown"}:
                continue
            unit = _normalize_length_unit(value)
            if unit is None:
                raise ValueError(
                    f"{context} {field} has unsupported declared unit {value!r} "
                    f"in options.{option_name}"
                )
            return unit, f"options.{option_name}.{field}"
    exact_key = f"{field}_unit"
    if exact_key in options:
        if str(options[exact_key]).strip().casefold() in {"", "unknown"}:
            return None, None
        unit = _normalize_length_unit(options[exact_key])
        if unit is None:
            raise ValueError(
                f"{context} {field} has unsupported declared unit "
                f"{options[exact_key]!r} in options.{exact_key}"
            )
        return unit, f"options.{exact_key}"
    return None, None


def _fallback_unit(
    options: dict[str, Any],
    field: str,
    *,
    context: str = "Trajectory",
) -> tuple[str | None, str | None]:
    if field in {"md", "tvd", "tvdss"}:
        keys = ("depth_unit", "vertical_unit", "length_unit")
    else:
        keys = ("coordinate_unit", "horizontal_unit", "length_unit")
    for key in keys:
        if key not in options:
            continue
        if str(options[key]).strip().casefold() in {"", "unknown"}:
            continue
        unit = _normalize_length_unit(options[key])
        if unit is None:
            raise ValueError(
                f"{context} {field} has unsupported declared unit "
                f"{options[key]!r} in options.{key}"
            )
        return unit, f"options.{key}"
    return None, None


def _resolve_field_unit(
    field: str,
    *,
    column_index: int | None,
    header: list[str] | None,
    options: dict[str, Any],
    dev_metadata: dict[str, Any],
    issues: list[str],
    context: str = "Trajectory",
) -> tuple[str, str]:
    option_unit, option_source = _mapping_unit(
        options, field, context=context
    )
    header_unit, header_source = _header_unit(header, column_index)
    if option_unit is not None:
        if header_unit is not None and header_unit != option_unit:
            issues.append(
                f"unit_conflict:{field}:options={option_unit}:header={header_unit}:options_precedence"
            )
        return option_unit, str(option_source)
    if header_unit is not None:
        return header_unit, str(header_source)
    fallback_unit, fallback_source = _fallback_unit(
        options, field, context=context
    )
    if fallback_unit is not None:
        return fallback_unit, str(fallback_source)
    dev_unit = _normalize_length_unit(
        dev_metadata.get(f"{field}_unit") or dev_metadata.get("length_unit")
    )
    if dev_unit is not None:
        return dev_unit, f"petrel_header:{field}_unit"
    source_header = (
        header[column_index]
        if header is not None and column_index is not None and column_index < len(header)
        else field
    )
    raise ValueError(
        f"{context} {field} length unit is unknown for column {source_header!r}; "
        "declare options.field_units or use a machine-readable header such as MD(m)/X(ft)"
    )


def _resolve_well_head_vertical_unit(
    *,
    row: list[str],
    columns: dict[str, int],
    header: list[str] | None,
    options: dict[str, Any],
    populated_fields: set[str],
) -> str | None:
    """Resolve the common KB/GL unit without silently overriding evidence.

    ``WellHead`` currently carries one unit for its elevation fields.  A
    machine-readable unit in ``KB(m)`` or ``GROUND_ELEVATION(ft)`` is therefore
    authoritative evidence, just like a ``DATUM_UNIT`` column.  If populated
    elevation fields disagree, or an option/row declaration contradicts the
    header, the row cannot be represented safely and parsing fails closed.

    Empty elevation cells deliberately contribute no declaration.  In
    particular, a blank ground-elevation column never creates or changes a GL
    value merely because its header contains a unit.
    """

    declarations: list[tuple[str, str]] = []

    def add_declaration(raw: Any, source: str) -> None:
        text = str(raw or "").strip()
        if text.casefold() in {"", "unknown", "na", "n/a", "null", "none"}:
            return
        unit = _normalize_length_unit(text)
        if unit is None:
            raise VerticalDatumUnitConflict(
                f"Well-head vertical datum has unsupported declared unit "
                f"{text!r} from {source}"
            )
        declarations.append((unit, source))

    unit_index = columns.get("vertical_datum_unit")
    if unit_index is not None and 0 <= unit_index < len(row):
        add_declaration(row[unit_index], "column:vertical_datum_unit")

    for option_name in ("vertical_datum_unit", "elevation_unit", "vertical_unit"):
        if option_name in options:
            add_declaration(options[option_name], f"options.{option_name}")

    for field in ("kb", "ground_elevation"):
        if field not in populated_fields:
            continue
        column_index = columns.get(field)
        header_unit, header_source = _header_unit(header, column_index)
        mapping_unit, mapping_source = _mapping_unit(
            options,
            field,
            context="Well-head vertical datum",
        )
        if header_unit is not None:
            declarations.append((header_unit, str(header_source)))
        if mapping_unit is not None:
            declarations.append((mapping_unit, str(mapping_source)))

    distinct_units = {unit for unit, _ in declarations}
    if len(distinct_units) > 1:
        evidence = ",".join(f"{source}={unit}" for unit, source in declarations)
        raise VerticalDatumUnitConflict(f"vertical_datum_unit_conflict:{evidence}")
    return declarations[0][0] if declarations else None


def _convert_field_to_metres(
    values: np.ndarray,
    field: str,
    unit: str,
    conversions: list[str],
) -> np.ndarray:
    converted = np.asarray(length_to_metres(values, unit), dtype=float)
    if unit != "m":
        factor = float(length_to_metres(1.0, unit))
        conversions.append(f"{field}:{unit}->m(*{factor:g})")
    return converted


def read_petrel_dev_header(path: str | Path) -> dict[str, Any]:
    """Read common Petrel DEV metadata without assuming a project-specific name."""

    path = Path(path)
    metadata: dict[str, Any] = {}
    patterns = {
        "well_name": r"^#\s*WELL\s+NAME\s*:\s*(.+?)\s*$",
        "x": r"^#\s*WELL\s+HEAD\s+X-COORDINATE\s*:\s*([-+0-9.Ee]+)",
        "y": r"^#\s*WELL\s+HEAD\s+Y-COORDINATE\s*:\s*([-+0-9.Ee]+)",
        "kb": r"^#\s*WELL\s+DATUM.*?:\s*([-+0-9.Ee]+)",
    }
    declared_units: list[str] = []
    for line in _read_lines(path):
        for field, pattern in patterns.items():
            match = re.match(pattern, line.strip(), re.I)
            if not match:
                continue
            metadata[field] = match.group(1).strip() if field == "well_name" else float(match.group(1))
            if field in {"x", "y", "kb"}:
                unit_match = re.search(
                    rf"\(\s*({_HEADER_UNIT})\s*\)\s*$",
                    line,
                    re.IGNORECASE,
                )
                unit = _normalize_length_unit(unit_match.group(1)) if unit_match else None
                if unit:
                    metadata[f"{field}_unit"] = unit
                    declared_units.append(unit)
        epsg = re.search(r"\bEPSG\s*[,=: ]\s*(\d{4,6})\b", line, re.I)
        if epsg:
            metadata["crs"] = f"EPSG:{epsg.group(1)}"
    if declared_units and len(set(declared_units)) == 1:
        # Petrel DEV headers state the project length unit on the well-head and
        # datum records even when the subsequent MD/TVD table uses bare names.
        metadata["length_unit"] = declared_units[0]
    return metadata


def _petrel_dev_rows(path: Path) -> tuple[list[str], list[list[str]]] | None:
    """Locate the actual tabular header in a Petrel-style DEV file.

    Petrel writes many comment lines before the data header, so treating the
    first ``#`` line as a table header silently loses the trajectory.
    """

    lines = [line.strip() for line in _read_lines(path) if line.strip()]
    for index, line in enumerate(lines):
        if line.startswith("#") or set(line) <= {"=", "-"}:
            continue
        tokens = re.split(r"[\s,\t]+", line)
        normalized = {_normalized_column(token) for token in tokens}
        if "MD" not in normalized or not normalized.intersection({"TVD", "INCL", "INC", "X", "DX"}):
            continue
        rows: list[list[str]] = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith("#") or set(candidate) <= {"=", "-"}:
                continue
            values = re.split(r"[\s,\t]+", candidate)
            if len(values) >= len(tokens):
                rows.append(values)
        return tokens, rows
    return None


def _petrel_dev_columns(header: list[str]) -> dict[str, int]:
    # Petrel commonly exports both true-north and grid-north azimuth.  The
    # alias order explicitly prefers AZIM_TN and is therefore deterministic.
    return _map_columns_from_aliases(
        header,
        _PETREL_DEV_ALIASES,
        priority_fields={"azimuth"},
    )


def _read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1", errors="replace").splitlines()


def _trim_ooxml_row(row: list[Any]) -> list[Any]:
    values = list(row)
    while values and (
        values[-1] is None or not str(values[-1]).strip()
    ):
        values.pop()
    return values


def _well_block_marker(row: list[Any]) -> str | None:
    if len(row) < 2:
        return None
    marker = re.sub(r"[^A-Z]", "", str(row[0]).upper())
    name = str(row[1] or "").strip()
    if marker not in {"WELL", "WELLNAME"} or not name:
        return None
    if any(value is not None and str(value).strip() for value in row[2:]):
        return None
    return name


def _well_block_columns(header: list[Any]) -> dict[str, int] | None:
    normalized = [_normalized_column(str(value or "")) for value in header]

    def locate(names: set[str]) -> int | None:
        matches = [index for index, value in enumerate(normalized) if value in names]
        if len(matches) > 1:
            raise OOXMLReadError(
                "OOXML repeated well block has duplicate semantic columns: "
                + ",".join(str(header[index]) for index in matches)
            )
        return matches[0] if matches else None

    columns = {
        "api": locate({"API", "APIID", "APINUMBER"}),
        "md": locate({"MD", "MEASUREDDEPTH", "DEPTH"}),
        "inclination": locate({"INC", "INCL", "INCLINATION", "DEVIATION"}),
        "azimuth": locate({"AZI", "AZIM", "AZIMUTH", "BEARING"}),
    }
    if all(columns[field] is not None for field in ("md", "inclination", "azimuth")):
        return {field: int(index) for field, index in columns.items() if index is not None}
    return None


_STRICT_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_QUADRANT_AZIMUTH = re.compile(
    r"^\s*([NS])\s*(\d+(?:\.\d+)?)\s*(?:°|DEG(?:REE)?S?)?\s*([EW])\s*$",
    re.IGNORECASE,
)


def _strict_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field}:missing")
    if isinstance(value, (int, float)):
        converted = float(value)
    else:
        text = str(value).strip()
        if not _STRICT_NUMBER.fullmatch(text):
            raise ValueError(f"{field}:not_strict_numeric:{text}")
        converted = float(text)
    if not np.isfinite(converted):
        raise ValueError(f"{field}:not_finite")
    return converted


def _azimuth_degrees(value: Any) -> float:
    try:
        numeric = _strict_float(value, field="azimuth")
    except ValueError:
        match = _QUADRANT_AZIMUTH.fullmatch(str(value or ""))
        if match is None:
            raise
        north_south, angle_text, east_west = match.groups()
        angle = float(angle_text)
        if not 0.0 <= angle <= 90.0:
            raise ValueError(f"azimuth:quadrant_angle_out_of_range:{angle:g}")
        if north_south.upper() == "N":
            numeric = angle if east_west.upper() == "E" else 360.0 - angle
        else:
            numeric = 180.0 - angle if east_west.upper() == "E" else 180.0 + angle
    if not 0.0 <= numeric <= 360.0:
        raise ValueError(f"azimuth:out_of_range:{numeric:g}")
    return 0.0 if np.isclose(numeric, 360.0) else numeric


def _api_text(value: Any) -> str:
    """Preserve an API identifier without retaining spreadsheet ``.0`` noise."""

    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+)\.0+", text)
    return match.group(1) if match is not None else text


def _source_decimal_precision(value: Any) -> int:
    """Return a conservative precision score for a numeric OOXML cell."""

    text = str(value or "").strip().casefold()
    mantissa = text.split("e", 1)[0]
    if "." not in mantissa:
        return 0
    return len(mantissa.partition(".")[2].rstrip("0"))


def _congruent_base_api_duplicate(
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    """Resolve one exact duplicate survey exported under API10 and API10+00.

    The API relationship alone is not enough.  Both blocks must carry the same
    normalized well marker, identical MD stations, and effectively identical
    inclination/azimuth geometry.  The more precise source block is retained;
    equal precision stays ambiguous and fails closed.
    """

    if len(blocks) != 2 or any(block["errors"] for block in blocks):
        return None
    if any(len(block["api_values"]) != 1 for block in blocks):
        return None
    api_by_block = [next(iter(block["api_values"])) for block in blocks]
    if not all(re.fullmatch(r"\d+", api) for api in api_by_block):
        return None
    shorter, longer = sorted(api_by_block, key=len)
    if len(shorter) != 10 or len(longer) != 12 or longer != shorter + "00":
        return None

    station_sets = [block["stations"] for block in blocks]
    if any(len(stations) < 2 for stations in station_sets):
        return None
    left = np.asarray(
        [(md, inclination, azimuth) for _, md, inclination, azimuth in station_sets[0]],
        dtype=float,
    )
    right = np.asarray(
        [(md, inclination, azimuth) for _, md, inclination, azimuth in station_sets[1]],
        dtype=float,
    )
    if left.shape != right.shape or not np.array_equal(left[:, 0], right[:, 0]):
        return None
    if not np.all(np.diff(left[:, 0]) > 0.0):
        return None
    inclination_delta = np.abs(left[:, 1] - right[:, 1])
    azimuth_delta = np.abs(left[:, 2] - right[:, 2])
    azimuth_delta = np.minimum(azimuth_delta, 360.0 - azimuth_delta)
    max_angle_delta = float(
        max(np.max(inclination_delta), np.max(azimuth_delta))
    )
    # 0.005 degree is the maximum quantisation error when one export rounds a
    # more precise survey to two decimals.  A tiny floating tolerance only
    # accommodates binary representation of that exact boundary.
    if max_angle_delta > 0.0050001:
        return None
    precision = [int(block["precision_score"]) for block in blocks]
    if precision[0] == precision[1]:
        return None
    selected_index = 0 if precision[0] > precision[1] else 1
    selected = blocks[selected_index]
    return selected, (
        "congruent_api10_api12_duplicate:"
        f"{shorter}~{longer}:stations={left.shape[0]}:"
        f"max_angle_delta_deg={max_angle_delta:.7g}:"
        f"selected={api_by_block[selected_index]}:higher_numeric_precision"
    )


def _ooxml_well_blocks(
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]], list[str]] | None:
    markers = [
        index
        for index, row in enumerate(rows)
        if _well_block_marker(_trim_ooxml_row(row)) is not None
    ]
    if not markers:
        return None

    blocks: list[dict[str, Any]] = []
    declared_md_units: set[str] = set()
    for marker_offset, marker_index in enumerate(markers):
        end = markers[marker_offset + 1] if marker_offset + 1 < len(markers) else len(rows)
        marker_row = _trim_ooxml_row(rows[marker_index])
        name = str(_well_block_marker(marker_row))
        header_index = next(
            (
                index
                for index in range(marker_index + 1, end)
                if _well_block_columns(_trim_ooxml_row(rows[index])) is not None
            ),
            None,
        )
        if header_index is None:
            raise OOXMLReadError(
                f"OOXML well block {name!r} has no MD/inclination/azimuth header"
            )
        header = _trim_ooxml_row(rows[header_index])
        columns = _well_block_columns(header)
        assert columns is not None
        md_unit, _ = _header_unit(
            [str(value or "") for value in header], columns["md"]
        )
        if md_unit:
            declared_md_units.add(md_unit)
        blocks.append(
            {
                "name": name,
                "marker_row": marker_index + 1,
                "header": header,
                "columns": columns,
                "md_unit": md_unit,
                "rows": [
                    (index + 1, _trim_ooxml_row(rows[index]))
                    for index in range(header_index + 1, end)
                    if _trim_ooxml_row(rows[index])
                ],
            }
        )

    if len(declared_md_units) > 1:
        raise OOXMLReadError(
            "OOXML well blocks declare conflicting MD units: "
            + ",".join(sorted(declared_md_units))
        )
    consensus_md_unit = next(iter(declared_md_units), None)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for block in blocks:
        key = _normalized_column(block["name"])
        grouped[key].append(block)
        display_names.setdefault(key, block["name"])

    logical_rows: list[list[Any]] = []
    rejected: list[str] = []
    quarantined: list[str] = []
    canonicalized: list[str] = []
    duplicate_receipts: list[str] = []
    quadrant_count = 0
    inherited_unit_blocks = 0
    for key, well_blocks in grouped.items():
        name = display_names[key]
        parsed_blocks: list[dict[str, Any]] = []
        for block in well_blocks:
            if block["md_unit"] is None and consensus_md_unit is not None:
                inherited_unit_blocks += 1
            columns = block["columns"]
            block_stations: list[tuple[str | None, float, float, float]] = []
            block_station_rows: list[int] = []
            block_apis: set[str] = set()
            block_errors: list[str] = []
            precision_score = 0
            for row_number, row in block["rows"]:
                try:
                    required_width = max(columns.values()) + 1
                    if len(row) < required_width:
                        raise ValueError("missing_required_cells")
                    api = (
                        None
                        if "api" not in columns
                        else _api_text(row[columns["api"]])
                    )
                    if "api" in columns and not api:
                        raise ValueError("api:missing")
                    md = _strict_float(row[columns["md"]], field="md")
                    inclination = _strict_float(
                        row[columns["inclination"]], field="inclination"
                    )
                    raw_azimuth = row[columns["azimuth"]]
                    azimuth = _azimuth_degrees(raw_azimuth)
                    if not isinstance(raw_azimuth, (int, float)):
                        quadrant_count += 1
                    if not 0.0 <= inclination <= 180.0:
                        raise ValueError(
                            f"inclination:out_of_range:{inclination:g}"
                        )
                except (IndexError, ValueError) as exc:
                    block_errors.append(f"row={row_number}:{exc}")
                    continue
                if api:
                    block_apis.add(api)
                precision_score += _source_decimal_precision(
                    row[columns["inclination"]]
                )
                precision_score += _source_decimal_precision(raw_azimuth)
                block_stations.append((api, md, inclination, azimuth))
                block_station_rows.append(row_number)
            parsed_blocks.append(
                {
                    "marker_row": block["marker_row"],
                    "stations": block_stations,
                    "station_rows": block_station_rows,
                    "api_values": block_apis,
                    "errors": block_errors,
                    "precision_score": precision_score,
                }
            )

        errors = [
            error
            for block in parsed_blocks
            for error in block["errors"]
        ]
        if len(errors) > 1:
            rejected.append(
                f"{name}:invalid_station:{'|'.join(sorted(set(errors))[:3])}"
            )
            continue
        selected_blocks = parsed_blocks
        api_values = {
            api
            for block in selected_blocks
            for api in block["api_values"]
        }
        if len(api_values) > 1:
            duplicate = _congruent_base_api_duplicate(parsed_blocks)
            if duplicate is None:
                rejected.append(
                    f"{name}:conflicting_api:{'|'.join(sorted(api_values))}"
                )
                continue
            selected, receipt = duplicate
            selected_blocks = [selected]
            duplicate_receipts.append(f"{name}:{receipt}")
        # One later duplicate MD can be isolated when it is the sole source of
        # an ordering reversal and removing that occurrence restores the
        # source block's monotonic survey order.  This does not guess a depth
        # or average conflicting angles; the anomalous duplicate is discarded
        # with an explicit physical-gate receipt.
        for block in selected_blocks:
            md_sequence = np.asarray(
                [station[1] for station in block["stations"]],
                dtype=float,
            )
            decreases = np.flatnonzero(np.diff(md_sequence) < 0.0) + 1
            if decreases.size != 1:
                continue
            duplicate_index = int(decreases[0])
            duplicate_md = float(md_sequence[duplicate_index])
            earlier = np.flatnonzero(
                md_sequence[:duplicate_index] == duplicate_md
            )
            if earlier.size != 1 or int(np.sum(md_sequence == duplicate_md)) != 2:
                continue
            retained_md = np.delete(md_sequence, duplicate_index)
            if retained_md.size < 2 or not np.all(np.diff(retained_md) >= 0.0):
                continue
            source_row = int(block["station_rows"][duplicate_index])
            del block["stations"][duplicate_index]
            del block["station_rows"][duplicate_index]
            quarantined.append(
                f"{name}:out_of_order_duplicate_md_quarantined:"
                f"row={source_row}:md={duplicate_md:g}:"
                f"remaining_valid={len(block['stations'])}"
            )
        # Export order is not a physical constraint.  Reorder the complete
        # station tuple (MD, INC, AZI and its source-row identity) by MD while
        # preserving stable order for equal MD.  Conflicting equal-MD geometry
        # is still rejected below; no first/last/mean value is invented.
        for block in selected_blocks:
            md_sequence = np.asarray(
                [station[1] for station in block["stations"]], dtype=float
            )
            order = np.argsort(md_sequence, kind="stable")
            if not np.array_equal(order, np.arange(md_sequence.size)):
                source_rows = list(block["station_rows"])
                block["stations"] = [
                    block["stations"][int(index)] for index in order
                ]
                block["station_rows"] = [
                    block["station_rows"][int(index)] for index in order
                ]
                source_preview = ",".join(str(value) for value in source_rows[:24])
                canonical_preview = ",".join(
                    str(value) for value in block["station_rows"][:24]
                )
                canonicalized.append(
                    f"{name}:source_station_order_canonicalized:"
                    f"marker_row={block['marker_row']}:"
                    f"station_count={len(source_rows)}:"
                    f"source_rows={source_preview}:"
                    f"canonical_rows={canonical_preview}:source_values_modified=false"
                )
        station_rows = [
            station
            for block in selected_blocks
            for station in block["stations"]
        ]
        if errors:
            quarantined.append(
                f"{name}:invalid_station_quarantined:{errors[0]}:"
                f"remaining_valid={len(station_rows)}"
            )
        by_md: dict[float, set[tuple[float, float]]] = defaultdict(set)
        for _, md, inclination, azimuth in station_rows:
            by_md[md].add((inclination, azimuth))
        conflicts = [md for md, values in by_md.items() if len(values) > 1]
        if conflicts:
            rejected.append(
                f"{name}:conflicting_duplicate_md:{'|'.join(f'{value:g}' for value in conflicts[:5])}"
            )
            continue
        unique_stations = sorted(
            (
                (md, next(iter(values))[0], next(iter(values))[1])
                for md, values in by_md.items()
            ),
            key=lambda item: item[0],
        )
        if len(unique_stations) < 2:
            rejected.append(f"{name}:fewer_than_two_unique_md_stations")
            continue
        selected_api_values = {
            api
            for block in selected_blocks
            for api in block["api_values"]
        }
        api_value = next(iter(selected_api_values), "")
        logical_rows.extend(
            [
                [name, api_value, md, inclination, azimuth]
                for md, inclination, azimuth in unique_stations
            ]
        )

    md_label = "MD" if consensus_md_unit is None else f"MD({consensus_md_unit})"
    evidence = [
        f"OOXML重复井段表：{len(blocks)}个区段，{len(grouped)}个井名",
        f"OOXML有效井轨迹：{len(grouped) - len(rejected)}；隔离拒绝：{len(rejected)}",
    ]
    if inherited_unit_blocks:
        evidence.append(
            f"{inherited_unit_blocks}个井段的MD单位由工作簿一致表头继承为{consensus_md_unit}"
        )
    if quadrant_count:
        evidence.append(f"{quadrant_count}个象限方位角按N/S角度E/W确定性换算")
    evidence.extend(f"OOXML井点顺序已规范化：{item}" for item in canonicalized)
    evidence.extend(f"OOXML坏站已隔离：{item}" for item in quarantined)
    evidence.extend(f"OOXML重复轨迹已核验归并：{item}" for item in duplicate_receipts)
    evidence.extend(f"OOXML井段已隔离：{item}" for item in rejected)
    return (
        ["WELL_NAME", "WELL_API", md_label, "INCLINATION", "AZIMUTH"],
        logical_rows,
        evidence,
    )


def _ooxml_table_rows(path: Path) -> tuple[list[str] | None, list[list[Any]], list[str]]:
    sheet = read_single_table_sheet(path)
    physical_rows = [_trim_ooxml_row(row) for row in sheet.rows]
    block_table = _ooxml_well_blocks(physical_rows)
    if block_table is not None:
        header, rows, evidence = block_table
        return header, rows, [f"OOXML工作表：{sheet.sheet_name}", *evidence]

    candidates: list[tuple[int, int, int]] = []
    for index, row in enumerate(physical_rows[:100]):
        nonblank = [value for value in row if value is not None and str(value).strip()]
        string_count = sum(
            isinstance(value, str) and any(character.isalpha() for character in value)
            for value in nonblank
        )
        if string_count >= 2:
            candidates.append((string_count, len(nonblank), -index))
    if not candidates:
        return None, [], [f"OOXML工作表：{sheet.sheet_name}", "未找到可靠的多列表头"]
    _, _, negative_index = max(candidates)
    header_index = -negative_index
    header = [str(value or "").strip() for value in physical_rows[header_index]]
    rows = [
        row
        for row in physical_rows[header_index + 1 :]
        if any(value is not None and str(value).strip() for value in row)
    ]
    return (
        header,
        rows,
        [
            f"OOXML工作表：{sheet.sheet_name}",
            f"OOXML表头位于第{header_index + 1}行",
        ],
    )


def _text_rows(
    path: Path,
    delimiter: str | None = None,
) -> tuple[list[str] | None, list[list[str]], list[str]]:
    lines = [line.strip() for line in _read_lines(path) if line.strip()]
    header = None
    result: list[list[str]] = []

    def split_row(line: str) -> list[str]:
        if delimiter:
            return [value.strip() for value in line.split(delimiter)]
        if path.suffix.casefold() == ".csv":
            # A regex split collapses consecutive commas and shifts every
            # column after a blank value.  That previously turned an empty
            # GROUND_ELEVATION cell into TOTAL_DEPTH for standard CSV input.
            return [value.strip() for value in next(csv.reader([line]))]
        return re.split(r"[\s,\t,]+", line)

    for line in lines:
        if line.startswith("#"):
            if header is None:
                header = re.split(r"[\s,\t]+", line.lstrip("#").strip())
            continue
        tokens = split_row(line)
        if header is None and tokens and any(
            any(character.isalpha() for character in token)
            for token in tokens[1:]
        ):
            header = tokens
            continue
        result.append(tokens)
    return header, result, []


def _rows_with_evidence(
    path: Path,
    delimiter: str | None = None,
) -> tuple[list[str] | None, list[list[Any]], list[str]]:
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm"}:
        return _ooxml_table_rows(path)
    if suffix == ".xls":
        raise ValueError(
            "Legacy .xls is not OOXML and cannot be parsed without an explicit xlrd adapter"
        )
    return _text_rows(path, delimiter)


def _rows(
    path: Path,
    delimiter: str | None = None,
) -> tuple[list[str] | None, list[list[Any]]]:
    header, rows, _ = _rows_with_evidence(path, delimiter)
    return header, rows


def read_well_heads(path: str | Path, options: dict[str, Any]) -> list[WellHead]:
    path = Path(path)
    if path.suffix.casefold() in {".well", ".track"}:
        from .opendtect_well import read_opendtect_well_track

        return [read_opendtect_well_track(path, options).head]
    header, rows = _rows(path, options.get("delimiter"))
    columns = options.get("columns")
    if columns is None and header:
        aliases = options.get("field_aliases", {})
        columns = _map_columns_from_aliases(header, aliases)
    if columns is None:
        columns = {"well_name": 0, "x": 1, "y": 2, "kb": 3, "total_depth_md": 4}
    source_crs, target_crs = _crs_source_and_target(options)
    use_crs_transform = bool(source_crs and target_crs)
    coordinate_units: dict[str, str | None] = {}
    coordinate_issues: list[str] = []
    for field in ("x", "y"):
        if columns.get(field) is None:
            coordinate_units[field] = None
            continue
        if use_crs_transform:
            coordinate_units[field] = _validate_coordinate_unit_against_crs(
                field,
                source_crs=str(source_crs),
                column_index=columns.get(field),
                header=header,
                options=options,
            )
            continue
        try:
            unit, _ = _resolve_field_unit(
                field,
                column_index=columns.get(field),
                header=header,
                options=options,
                dev_metadata={},
                issues=coordinate_issues,
                context="Well-head",
            )
            coordinate_units[field] = unit
        except ValueError as exc:
            if "length unit is unknown" not in str(exc):
                raise
            coordinate_units[field] = None
            coordinate_issues.append(
                f"horizontal_coordinate_unit_unknown:{field}"
            )
    heads: list[WellHead] = []
    for row in rows:
        try:
            def number(field: str) -> float | None:
                index = columns.get(field)
                if index is None or index < 0 or index >= len(row):
                    return None
                raw = str(row[index]).strip()
                if raw.casefold() in {
                    "",
                    "na",
                    "n/a",
                    "null",
                    "none",
                    "nan",
                }:
                    return None
                value = float(raw)
                return value if np.isfinite(value) else None

            kb = number("kb")
            ground_elevation = number("ground_elevation")
            datum_type_index = columns.get("vertical_datum_type")
            datum_type = (
                ""
                if datum_type_index is None
                or datum_type_index < 0
                or datum_type_index >= len(row)
                else _normalized_column(str(row[datum_type_index] or ""))
            )
            if datum_type:
                if datum_type in {"KB", "RKB", "KELLYBUSHING"}:
                    pass
                elif datum_type in {
                    "GL",
                    "GR",
                    "GROUND",
                    "GROUNDLEVEL",
                    "GROUNDREFERENCE",
                }:
                    if ground_elevation is None:
                        ground_elevation = kb
                    kb = None
                elif datum_type in {"DF", "DRILLINGFLOOR"}:
                    # The current WellHead contract has no DF field.  Keeping
                    # it as KB would silently shift every derived elevation.
                    kb = None
                else:
                    raise VerticalDatumUnitConflict(
                        f"unsupported_vertical_datum_type:{row[datum_type_index]}"
                    )
            vertical_datum_unit = _resolve_well_head_vertical_unit(
                row=row,
                columns=columns,
                header=header,
                options=options,
                populated_fields={
                    field
                    for field, value in (
                        ("kb", kb),
                        ("ground_elevation", ground_elevation),
                    )
                    if value is not None
                },
            )
            raw_x = number("x")
            raw_y = number("y")
            coordinate_transform: dict[str, Any] = {}
            resolved_crs = source_crs
            if raw_x is not None and raw_y is not None and use_crs_transform:
                transformed = transform_xy(
                    [raw_x],
                    [raw_y],
                    source_crs=str(source_crs),
                    target_crs=str(target_crs),
                )
                x = float(transformed.x[0])
                y = float(transformed.y[0])
                resolved_crs = transformed.target_crs
                coordinate_transform = transformed.provenance()
            else:
                x = (
                    None
                    if raw_x is None or coordinate_units.get("x") is None
                    else float(length_to_metres(raw_x, str(coordinate_units["x"])))
                )
                y = (
                    None
                    if raw_y is None or coordinate_units.get("y") is None
                    else float(length_to_metres(raw_y, str(coordinate_units["y"])))
                )
            coordinates_ready = x is not None and y is not None
            identifier_index = columns.get("well_identifier")
            raw_identifier = (
                None
                if identifier_index is None
                or identifier_index < 0
                or identifier_index >= len(row)
                else row[identifier_index]
            )
            identifiers, invalid_identifiers = canonical_api12_values(
                [raw_identifier]
            )
            head_coordinate_issues = list(coordinate_issues)
            if invalid_identifiers:
                head_coordinate_issues.append(
                    "well_identifier_invalid:"
                    + "|".join(invalid_identifiers)
                )
                identifiers.extend(invalid_identifiers)
            name_index = columns.get("well_name")
            well_name = (
                row[name_index]
                if name_index is not None and 0 <= name_index < len(row)
                else (
                    identifiers[0].partition(":")[2]
                    if identifiers
                    else options.get("well_name", path.stem)
                )
            )
            heads.append(WellHead(
                well_name=well_name,
                x=x, y=y, kb=kb,
                ground_elevation=ground_elevation, total_depth_md=number("total_depth_md"),
                crs=resolved_crs, source=str(path), confidence=(1.0 if header or options.get("columns") else 0.7) if coordinates_ready else 0.5,
                vertical_datum_unit=vertical_datum_unit,
                horizontal_unit="m" if coordinates_ready else "unknown",
                coordinate_issues=list(dict.fromkeys(head_coordinate_issues)),
                source_x=raw_x,
                source_y=raw_y,
                source_crs=(
                    canonical_crs_id(str(source_crs), field="井位源CRS")
                    if source_crs
                    else None
                ),
                coordinate_transform=coordinate_transform,
                identifiers=identifiers,
            ))
        except (CoordinateReferenceError, VerticalDatumUnitConflict):
            raise
        except (ValueError, IndexError):
            continue
    return heads


def read_trajectory(path: str | Path, options: dict[str, Any]) -> list[Trajectory]:
    path = Path(path)
    if path.suffix.casefold() in {".well", ".track"}:
        from .opendtect_well import read_opendtect_well_track

        return [read_opendtect_well_track(path, options).trajectory]
    dev_table = _petrel_dev_rows(path) if path.suffix.casefold() == ".dev" else None
    header, rows = dev_table or _rows(path, options.get("delimiter"))
    columns = dict(options.get("columns") or {})
    dev_metadata = read_petrel_dev_header(path) if dev_table else {}
    source_crs, target_crs = _crs_source_and_target(options, dev_metadata)
    use_crs_transform = bool(source_crs and target_crs)
    if not columns and dev_table:
        columns = _petrel_dev_columns(header)
    if not columns and header:
        aliases = {
            field: list(names)
            for field, names in _PETREL_DEV_ALIASES.items()
        }
        for field, names in options.get("field_aliases", {}).items():
            existing = aliases.setdefault(field, [])
            for name in names:
                if name not in existing:
                    existing.append(name)
        columns = _map_columns_from_aliases(header, aliases)
    if "md" not in columns:
        raise ValueError(f"Trajectory schema needs an MD column: {path}")
    if "tvdss" in columns and "tvd" not in columns:
        convention = str(
            options.get("tvdss_convention")
            or options.get("tvdss_positive_direction")
            or ""
        ).strip().casefold()
        datum = str(options.get("tvdss_datum") or "").strip().upper()
        reference = options.get("well_reference_elevation_m")
        allowed = {
            "depth_below_msl": "depth_below_msl",
            "depth_below_msl_m": "depth_below_msl",
            "positive_down": "depth_below_msl",
            "down": "depth_below_msl",
            "z_msl": "z_msl",
            "z_msl_m": "z_msl",
            "elevation_positive_up": "z_msl",
            "positive_up": "z_msl",
            "up": "z_msl",
        }
        if convention not in allowed or datum != "MSL" or reference is None:
            raise ValueError(
                "Trajectory has TVDSS but no TVD. TVDSS is not TVD; declare "
                "tvdss_convention (depth_below_msl or z_msl), tvdss_datum=MSL, "
                "and well_reference_elevation_m before conversion"
            )
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        name_index = columns.get("well_name")
        identifier_index = columns.get("well_identifier")
        row_identifiers, _ = canonical_api12_values(
            [
                row[identifier_index]
                if identifier_index is not None
                and 0 <= identifier_index < len(row)
                else None
            ]
        )
        name = (
            row[name_index]
            if name_index is not None and 0 <= name_index < len(row)
            else (
                row_identifiers[0].partition(":")[2]
                if row_identifiers
                else options.get(
                    "well_name", dev_metadata.get("well_name", path.stem)
                )
            )
        )
        grouped.setdefault(name, []).append(row)
    trajectories: list[Trajectory] = []
    for name, group in grouped.items():
        def array(field: str) -> np.ndarray | None:
            idx = columns.get(field)
            if idx is None:
                return None
            try:
                return np.asarray([float(row[idx]) for row in group], dtype=float)
            except (ValueError, IndexError):
                return None
        md = array("md")
        if md is None or md.size == 0:
            continue
        issues: list[str] = []
        identifier_index = columns.get("well_identifier")
        raw_identifiers = [
            row[identifier_index]
            for row in group
            if identifier_index is not None
            and 0 <= identifier_index < len(row)
            and str(row[identifier_index] or "").strip()
        ]
        identifiers, invalid_identifiers = canonical_api12_values(
            raw_identifiers
        )
        if invalid_identifiers:
            issues.append(
                "well_identifier_invalid:"
                + "|".join(invalid_identifiers)
            )
            identifiers.extend(invalid_identifiers)
        if len(identifiers) > 1:
            issues.append(
                "well_identifier_conflict:" + "|".join(identifiers)
            )
        source_units: dict[str, str] = {}
        unit_conversions: list[str] = []
        unit_provenance: dict[str, str] = {}
        physical_arrays: dict[str, np.ndarray | None] = {
            "md": md,
            "tvd": array("tvd"),
            "tvdss": array("tvdss"),
            "x": array("x"),
            "y": array("y"),
            "x_offset": array("x_offset"),
            "y_offset": array("y_offset"),
        }
        for field, values in physical_arrays.items():
            if values is None:
                continue
            if field in {"x", "y"} and use_crs_transform:
                native_unit = _validate_coordinate_unit_against_crs(
                    field,
                    source_crs=str(source_crs),
                    column_index=columns.get(field),
                    header=header,
                    options=options,
                )
                source_units[field] = native_unit
                unit_provenance[field] = (
                    f"source_crs_native:{canonical_crs_id(str(source_crs), field='轨迹源CRS')}"
                )
                continue
            unit, provenance = _resolve_field_unit(
                field,
                column_index=columns.get(field),
                header=header,
                options=options,
                dev_metadata=dev_metadata,
                issues=issues,
            )
            physical_arrays[field] = _convert_field_to_metres(
                values,
                field,
                unit,
                unit_conversions,
            )
            source_units[field] = unit
            unit_provenance[field] = provenance
        md = physical_arrays["md"]
        if md is None:
            continue
        inc, azi = array("inclination"), array("azimuth")
        station_arrays: dict[str, np.ndarray | None] = {
            field: values
            for field, values in physical_arrays.items()
            if field != "md"
        }
        station_arrays.update({"inclination": inc, "azimuth": azi})
        canonical = canonicalize_station_rows(md, station_arrays)
        order = canonical.source_indices
        if order.size == 0:
            # The bad well is isolated here; other wells in the same table can
            # still contribute heads, logs, trajectories and calibration.
            continue
        md = md[order]
        inc = None if inc is None else inc[order]
        azi = None if azi is None else azi[order]
        tvd = physical_arrays["tvd"]
        tvdss = physical_arrays["tvdss"]
        xo, yo = physical_arrays["x_offset"], physical_arrays["y_offset"]
        x, y = physical_arrays["x"], physical_arrays["y"]
        tvd = None if tvd is None else tvd[order]
        tvdss = None if tvdss is None else tvdss[order]
        xo = None if xo is None else xo[order]
        yo = None if yo is None else yo[order]
        x = None if x is None else x[order]
        y = None if y is None else y[order]
        vertical_semantics: dict[str, Any] = {"canonical_depth": "TVD_m_positive_down"}
        if canonical.receipt is not None:
            vertical_semantics["station_normalization"] = canonical.receipt
            if canonical.receipt["dropped_nonfinite_md_source_rows_1_based"]:
                issues.append("nonfinite_md_stations_quarantined")
        if canonical.conflicting_duplicate_md:
            # Retaining the duplicate identity is intentional: the strict-MD
            # formal/P13 contracts will isolate only this ambiguous trajectory.
            issues.append("conflicting_duplicate_md_stations_not_auto_resolved")
            vertical_semantics.update(
                {
                    "registration_eligible": False,
                    "registration_block_reason": "conflicting_duplicate_md_station_geometry",
                }
            )
        if tvd is None and tvdss is not None:
            convention = str(
                options.get("tvdss_convention")
                or options.get("tvdss_positive_direction")
            ).strip().casefold()
            convention = {
                "depth_below_msl_m": "depth_below_msl",
                "positive_down": "depth_below_msl",
                "down": "depth_below_msl",
                "z_msl_m": "z_msl",
                "elevation_positive_up": "z_msl",
                "positive_up": "z_msl",
                "up": "z_msl",
            }.get(convention, convention)
            reference_elevation_m = float(options["well_reference_elevation_m"])
            if not np.isfinite(reference_elevation_m):
                raise ValueError("well_reference_elevation_m must be finite")
            z_msl = -tvdss if convention == "depth_below_msl" else tvdss
            tvd = reference_elevation_m - z_msl
            source_units["tvd"] = "m"
            unit_provenance["tvd"] = "computed_from_declared_tvdss_semantics"
            unit_conversions.append(f"tvdss:{convention}->tvd(reference={reference_elevation_m:g}m)")
            vertical_semantics.update(
                {
                    "source_depth": "TVDSS",
                    "tvdss_convention": convention,
                    "tvdss_datum": "MSL",
                    "well_reference_elevation_m": reference_elevation_m,
                }
            )
        coordinate_transform: dict[str, Any] = {}
        resolved_horizontal_crs = (
            canonical_crs_id(str(source_crs), field="轨迹源CRS")
            if source_crs
            else None
        )
        if use_crs_transform:
            if (x is None) != (y is None):
                raise CoordinateReferenceError("轨迹绝对X/Y必须同时存在才能重投影")
            if x is not None and y is not None:
                transformed = transform_xy(
                    x,
                    y,
                    source_crs=str(source_crs),
                    target_crs=str(target_crs),
                )
                x = transformed.x
                y = transformed.y
                resolved_horizontal_crs = transformed.target_crs
                coordinate_transform = transformed.provenance()
            else:
                # Relative offsets are canonical metres.  Once composed with
                # the already-transformed well head they live in the target
                # projected CRS, even though the source head coordinates are
                # retained for the exact offset-path operation below.
                resolved_horizontal_crs = canonical_crs_id(
                    str(target_crs), field="轨迹目标CRS"
                )
                coordinate_transform = {
                    "source_crs": canonical_crs_id(str(source_crs), field="轨迹源CRS"),
                    "target_crs": canonical_crs_id(str(target_crs), field="轨迹目标CRS"),
                    "transformed": False,
                    "operation": "pending_absolute_path_from_wellhead_and_offsets",
                    "axis_contract": "always_xy",
                }
        if tvd is None or xo is None or yo is None:
            if inc is not None and azi is not None:
                missing_tvd = tvd is None
                missing_offsets = xo is None or yo is None
                tvd_calc, east, north = minimum_curvature(md, inc, azi)
                if tvd is None:
                    tvd = tvd_calc
                    source_units["tvd"] = "m"
                    unit_provenance["tvd"] = "computed:minimum_curvature"
                if xo is None:
                    xo = east
                    source_units["x_offset"] = "m"
                    unit_provenance["x_offset"] = "computed:minimum_curvature"
                if yo is None:
                    yo = north
                    source_units["y_offset"] = "m"
                    unit_provenance["y_offset"] = "computed:minimum_curvature"
                if missing_tvd:
                    issues.append(
                        "missing_tvd_reconstructed_with_minimum_curvature"
                    )
                if missing_offsets:
                    issues.append(
                        "missing_coordinates_reconstructed_with_minimum_curvature"
                    )
            else:
                if tvd is None:
                    tvd = np.full_like(md, np.nan)
                    unit_provenance["tvd"] = "unavailable:no_deviation_survey"
                if xo is None:
                    xo = np.zeros_like(md)
                    source_units["x_offset"] = "m"
                    unit_provenance["x_offset"] = "generated_zero:no_deviation_survey"
                if yo is None:
                    yo = np.zeros_like(md)
                    source_units["y_offset"] = "m"
                    unit_provenance["y_offset"] = "generated_zero:no_deviation_survey"
                issues.append("missing_deviation_survey_no_md_to_tvd_conversion")
        trajectories.append(
            Trajectory(
                well_name=name,
                md=md,
                tvd=tvd,
                x_offset=xo,
                y_offset=yo,
                inclination=inc,
                azimuth=azi,
                x=x,
                y=y,
                source=str(path),
                confidence=0.5 if issues else 1.0,
                issues=issues,
                source_units=source_units,
                unit_conversions=unit_conversions,
                unit_provenance=unit_provenance,
                vertical_semantics=vertical_semantics,
                source_crs=(
                    canonical_crs_id(str(source_crs), field="轨迹源CRS")
                    if source_crs
                    else None
                ),
                horizontal_crs=resolved_horizontal_crs,
                coordinate_transform=coordinate_transform,
                identifiers=identifiers,
            )
        )
    return trajectories


def read_time_depth(path: str | Path, options: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = Path(path)
    _, rows = _rows(path, options.get("delimiter"))
    columns = options.get("columns", {"well_name": 0, "depth": 1, "time": 2})
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            name_index = columns.get("well_name")
            name = row[name_index] if name_index is not None else options.get("well_name", path.stem)
            grouped.setdefault(name, []).append((float(row[columns["depth"]]), float(row[columns["time"]])))
        except (ValueError, IndexError, KeyError):
            continue
    return {name: (np.asarray([x[0] for x in values]), np.asarray([x[1] for x in values])) for name, values in grouped.items()}
