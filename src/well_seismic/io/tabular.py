from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Trajectory, WellHead
from ..trajectory import minimum_curvature


def _read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1", errors="replace").splitlines()


def _rows(path: Path, delimiter: str | None = None) -> tuple[list[str] | None, list[list[str]]]:
    lines = [line.strip() for line in _read_lines(path) if line.strip()]
    header = None
    result: list[list[str]] = []
    for line in lines:
        if line.startswith("#"):
            if header is None:
                header = re.split(r"[\s,\t]+", line.lstrip("#").strip())
            continue
        tokens = line.split(delimiter) if delimiter else re.split(r"[\s,\t,]+", line)
        if header is None and tokens and any(re.search(r"[A-Za-z]", token) for token in tokens[1:]):
            header = tokens
            continue
        result.append(tokens)
    return header, result


def read_well_heads(path: str | Path, options: dict[str, Any]) -> list[WellHead]:
    path = Path(path)
    header, rows = _rows(path, options.get("delimiter"))
    columns = options.get("columns")
    if columns is None and header:
        aliases = options.get("field_aliases", {})
        normalized = {re.sub(r"[^A-Z0-9]", "", name.upper()): i for i, name in enumerate(header)}
        columns = {}
        for field, names in aliases.items():
            for name in names:
                key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
                if key in normalized:
                    columns[field] = normalized[key]
                    break
    columns = columns or {"well_name": 0, "x": 1, "y": 2, "kb": 3, "total_depth_md": 4}
    heads: list[WellHead] = []
    for row in rows:
        try:
            def number(field: str) -> float | None:
                index = columns.get(field)
                return float(row[index]) if index is not None and index < len(row) and row[index] not in ("", "NA", "NULL") else None
            heads.append(WellHead(
                well_name=row[columns["well_name"]] if columns.get("well_name") is not None else options.get("well_name", path.stem),
                x=number("x"), y=number("y"), kb=number("kb"),
                ground_elevation=number("ground_elevation"), total_depth_md=number("total_depth_md"),
                crs=options.get("crs"), source=str(path), confidence=1.0 if header or options.get("columns") else 0.7,
            ))
        except (ValueError, IndexError):
            continue
    return heads


def read_trajectory(path: str | Path, options: dict[str, Any]) -> list[Trajectory]:
    path = Path(path)
    header, rows = _rows(path, options.get("delimiter"))
    columns = options.get("columns") or {}
    if not columns and header:
        normalized = {re.sub(r"[^A-Z0-9]", "", name.upper()): i for i, name in enumerate(header)}
        aliases = options.get("field_aliases", {})
        for field, names in aliases.items():
            for name in names:
                key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
                if key in normalized:
                    columns[field] = normalized[key]
                    break
    if "md" not in columns:
        raise ValueError(f"Trajectory schema needs an MD column: {path}")
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        name_index = columns.get("well_name")
        name = row[name_index] if name_index is not None else options.get("well_name", path.stem)
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
        order = np.argsort(md, kind="stable")
        md = md[order]
        inc, azi = array("inclination"), array("azimuth")
        tvd, xo, yo = array("tvd"), array("x_offset"), array("y_offset")
        x, y = array("x"), array("y")
        issues: list[str] = []
        if tvd is None or xo is None or yo is None:
            if inc is not None and azi is not None:
                tvd_calc, east, north = minimum_curvature(md, inc[order], azi[order])
                tvd = tvd_calc if tvd is None else tvd[order]
                xo = east if xo is None else xo[order]
                yo = north if yo is None else yo[order]
                issues.append("missing_coordinates_reconstructed_with_minimum_curvature")
            else:
                tvd = md.copy() if tvd is None else tvd[order]
                xo = np.zeros_like(md) if xo is None else xo[order]
                yo = np.zeros_like(md) if yo is None else yo[order]
                issues.append("vertical_well_fallback_due_to_missing_deviation_survey")
        else:
            tvd, xo, yo = tvd[order], xo[order], yo[order]
        trajectories.append(Trajectory(name, md, tvd, xo, yo, None if inc is None else inc[order], None if azi is None else azi[order], None if x is None else x[order], None if y is None else y[order], str(path), 0.5 if issues else 1.0, issues))
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
