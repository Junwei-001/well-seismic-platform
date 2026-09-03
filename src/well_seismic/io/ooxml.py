from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


class OOXMLReadError(ValueError):
    """Raised when an OOXML workbook cannot be represented safely."""


@dataclass(frozen=True)
class OOXMLSheetRows:
    sheet_name: str
    rows: list[list[Any]]


_MAX_XML_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ROWS = 1_048_576
_MAX_COLUMNS = 16_384


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise OOXMLReadError(f"OOXML workbook is missing required member {name}") from exc
    if info.file_size > _MAX_XML_MEMBER_BYTES:
        raise OOXMLReadError(
            f"OOXML member is too large to inspect safely: {name} ({info.file_size} bytes)"
        )
    return archive.read(info)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    try:
        root = ElementTree.fromstring(_safe_member(archive, "xl/sharedStrings.xml"))
    except ElementTree.ParseError as exc:
        raise OOXMLReadError("OOXML shared-string table is malformed") from exc
    values: list[str] = []
    for item in root.iter():
        if _local_name(item.tag) != "si":
            continue
        values.append(
            "".join(
                node.text or ""
                for node in item.iter()
                if _local_name(node.tag) == "t"
            )
        )
    return values


def _worksheet_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ElementTree.fromstring(
            _safe_member(archive, "xl/workbook.xml")
        )
        relationships = ElementTree.fromstring(
            _safe_member(archive, "xl/_rels/workbook.xml.rels")
        )
    except ElementTree.ParseError as exc:
        raise OOXMLReadError("OOXML workbook relationships are malformed") from exc

    targets: dict[str, str] = {}
    for relation in relationships.iter():
        if _local_name(relation.tag) != "Relationship":
            continue
        relation_id = relation.attrib.get("Id")
        target = relation.attrib.get("Target")
        relation_type = relation.attrib.get("Type", "")
        if not relation_id or not target or not relation_type.endswith("/worksheet"):
            continue
        normalized = posixpath.normpath(
            target.lstrip("/") if target.startswith("/xl/") else posixpath.join("xl", target)
        )
        if normalized.startswith("../") or not normalized.startswith("xl/"):
            raise OOXMLReadError(f"OOXML worksheet target escapes xl/: {target}")
        targets[relation_id] = normalized

    sheets: list[tuple[str, str]] = []
    for sheet in workbook.iter():
        if _local_name(sheet.tag) != "sheet":
            continue
        relation_id = next(
            (
                value
                for key, value in sheet.attrib.items()
                if _local_name(key) == "id"
            ),
            None,
        )
        if relation_id is None or relation_id not in targets:
            raise OOXMLReadError(
                f"OOXML worksheet {sheet.attrib.get('name', '<unnamed>')} has no relationship"
            )
        sheets.append((str(sheet.attrib.get("name") or relation_id), targets[relation_id]))
    if not sheets:
        raise OOXMLReadError("OOXML workbook contains no worksheets")
    return sheets


def _column_index(reference: str) -> int:
    match = re.match(r"([A-Za-z]+)", reference)
    if match is None:
        raise OOXMLReadError(f"OOXML cell has an invalid reference: {reference!r}")
    value = 0
    for character in match.group(1).upper():
        value = value * 26 + ord(character) - ord("A") + 1
    if value < 1 or value > _MAX_COLUMNS:
        raise OOXMLReadError(f"OOXML cell column is outside Excel limits: {reference}")
    return value - 1


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iter()
            if _local_name(node.tag) == "t"
        )
    value_node = next(
        (node for node in cell if _local_name(node.tag) == "v"),
        None,
    )
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError) as exc:
            raise OOXMLReadError(
                f"OOXML cell references an invalid shared-string index: {raw}"
            ) from exc
    if cell_type == "b":
        if raw not in {"0", "1"}:
            raise OOXMLReadError(f"OOXML boolean cell has invalid value {raw!r}")
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return int(numeric) if numeric.is_integer() else numeric


def _sheet_rows(
    archive: zipfile.ZipFile,
    member: str,
    shared_strings: list[str],
) -> list[list[Any]]:
    try:
        info = archive.getinfo(member)
    except KeyError as exc:
        raise OOXMLReadError(f"OOXML worksheet member is missing: {member}") from exc
    if info.file_size > _MAX_XML_MEMBER_BYTES:
        raise OOXMLReadError(
            f"OOXML worksheet is too large to inspect safely: {member} ({info.file_size} bytes)"
        )
    rows: list[list[Any]] = []
    try:
        with archive.open(info) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                if _local_name(element.tag) != "row":
                    continue
                if len(rows) >= _MAX_ROWS:
                    raise OOXMLReadError("OOXML worksheet exceeds Excel row limits")
                values: dict[int, Any] = {}
                for cell in element:
                    if _local_name(cell.tag) != "c":
                        continue
                    column = _column_index(cell.attrib.get("r", ""))
                    values[column] = _cell_value(cell, shared_strings)
                width = max(values, default=-1) + 1
                row = [values.get(index) for index in range(width)]
                while row and row[-1] is None:
                    row.pop()
                rows.append(row)
                element.clear()
    except ElementTree.ParseError as exc:
        raise OOXMLReadError(f"OOXML worksheet XML is malformed: {member}") from exc
    return rows


def read_ooxml_workbook(path: str | Path) -> list[OOXMLSheetRows]:
    """Read OOXML values without requiring an optional Excel runtime.

    The reader intentionally returns raw cell values.  Domain-specific header,
    unit and well-block decisions remain in :mod:`well_seismic.io.tabular`.
    """

    source = Path(path)
    if source.suffix.casefold() not in {".xlsx", ".xlsm"}:
        raise OOXMLReadError(f"OOXML reader does not support {source.suffix or '<no suffix>'}")
    try:
        with zipfile.ZipFile(source) as archive:
            shared_strings = _shared_strings(archive)
            return [
                OOXMLSheetRows(name, _sheet_rows(archive, member, shared_strings))
                for name, member in _worksheet_members(archive)
            ]
    except (OSError, zipfile.BadZipFile) as exc:
        raise OOXMLReadError(f"Unreadable OOXML workbook: {source}") from exc


def read_single_table_sheet(path: str | Path) -> OOXMLSheetRows:
    """Return the only non-empty worksheet, failing closed on ambiguity."""

    sheets = read_ooxml_workbook(path)
    populated = [
        sheet
        for sheet in sheets
        if any(any(value is not None and str(value).strip() for value in row) for row in sheet.rows)
    ]
    if not populated:
        return OOXMLSheetRows(sheets[0].sheet_name, [])
    if len(populated) != 1:
        names = ",".join(sheet.sheet_name for sheet in populated)
        raise OOXMLReadError(
            "OOXML metadata workbook has multiple populated worksheets; "
            f"sheet ownership must be explicit: {names}"
        )
    return populated[0]
