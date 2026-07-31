from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..knowledge import CurveKnowledgeBase, convert_unit
from ..models import Evidence, WellLog


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace"), "latin1-replace"


def _parse_las_item(line: str) -> tuple[str, str, str, str] | None:
    if not line.strip() or line.lstrip().startswith("#") or "." not in line:
        return None
    left, _, description = line.partition(":")
    mnemonic, rest = left.split(".", 1)
    if rest[:1].isspace():
        unit, value = "", rest.strip()
    else:
        parts = rest.strip().split(None, 1)
        unit = parts[0].strip() if parts else ""
        value = parts[1].strip() if len(parts) > 1 else ""
    return mnemonic.strip(), unit, value, description.strip()


def _collapse_duplicate_depths(data: np.ndarray, strategy: str) -> np.ndarray:
    depth = data[:, 0]
    unique_depth, first_indices, inverse, counts = np.unique(
        depth,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if np.all(counts == 1):
        return data
    if strategy == "first":
        return data[first_indices]
    if strategy == "last":
        last_indices = np.zeros_like(first_indices)
        for row_index, group_index in enumerate(inverse):
            last_indices[group_index] = row_index
        return data[last_indices]
    if strategy != "mean":
        raise ValueError(f"未知重复深度处理策略：{strategy}")
    collapsed = np.full((len(unique_depth), data.shape[1]), np.nan, dtype=float)
    collapsed[:, 0] = unique_depth
    for group_index in range(len(unique_depth)):
        rows = data[inverse == group_index]
        for column in range(1, data.shape[1]):
            valid = rows[:, column][np.isfinite(rows[:, column])]
            if valid.size:
                collapsed[group_index, column] = float(np.mean(valid))
    return collapsed


def _numeric_tokens(lines: list[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.replace("\t", " ").split():
            try:
                values.append(float(token.replace("D", "E").replace("d", "e").replace(",", "")))
            except ValueError:
                continue
    return values


def read_las(
    path: str | Path,
    knowledge: CurveKnowledgeBase,
    preprocessing: dict[str, Any] | None = None,
    decision_resolver: Any | None = None,
) -> WellLog:
    path = Path(path)
    text, encoding = _read_text(path)
    sections: dict[str, list[str]] = {}
    current = "PREAMBLE"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("~"):
            current = stripped[1:].split()[0].upper()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    header: dict[str, Evidence] = {"encoding": Evidence(encoding, "INFERRED", str(path), 0.8)}
    version = "unknown"
    for section_name in ("VERSION", "V", "WELL", "W", "PARAMETER", "P"):
        for line in sections.get(section_name, []):
            item = _parse_las_item(line)
            if not item:
                continue
            key, unit, value, description = item
            key_u = key.upper()
            header[key_u] = Evidence(value, "FILE_HEADER", str(path), 1.0, [unit, description])
            if key_u == "VERS":
                version = value.split()[0] if value else "unknown"

    curve_lines = sections.get("CURVE", sections.get("C", []))
    definitions: list[tuple[str, str, str]] = []
    for line in curve_lines:
        item = _parse_las_item(line)
        if item:
            definitions.append((item[0], item[1], item[3]))

    ascii_lines = sections.get("A", sections.get("ASCII", []))
    rows: list[list[float]] = []
    wrap = str(header.get("WRAP", Evidence("NO", "DEFAULT", str(path))).value).strip().upper().startswith("Y")
    if wrap and definitions:
        tokens = _numeric_tokens(ascii_lines)
        width_hint = len(definitions)
        rows = [tokens[start:start + width_hint] for start in range(0, len(tokens), width_hint) if len(tokens[start:start + width_hint]) == width_hint]
    else:
        for line in ascii_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(token.replace("D", "E").replace("d", "e").replace(",", "")) for token in stripped.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric LAS data found: {path}")
    width = min(len(row) for row in rows)
    data = np.asarray([row[:width] for row in rows if len(row) >= width], dtype=float)
    if len(definitions) < width:
        definitions.extend((f"CURVE_{i+1}", "", "") for i in range(len(definitions), width))
    definitions = definitions[:width]

    nulls = list((preprocessing or {}).get("null_values", []))
    null_header = header.get("NULL")
    if null_header:
        try:
            nulls.append(float(str(null_header.value).split()[0]))
        except ValueError:
            pass
    null_tolerance = float((preprocessing or {}).get("null_tolerance", 1e-6))
    for null in nulls:
        data[np.isclose(data, null, rtol=0, atol=null_tolerance)] = np.nan

    settings = preprocessing or {}
    depth_index = 0
    for index, (name, unit, description) in enumerate(definitions):
        depth_info = knowledge.identify(name, unit, description, data[:, index])
        if depth_info.standard_name == "DEPTH":
            depth_index = index
            break
    if depth_index != 0:
        order = [depth_index] + [index for index in range(data.shape[1]) if index != depth_index]
        data = data[:, order]
        definitions = [definitions[index] for index in order]
    normalize_depth_unit = knowledge.identify(
        definitions[0][0], definitions[0][1], definitions[0][2], data[:, 0]
    ).original_unit
    depth_values, depth_converted = convert_unit(data[:, 0], normalize_depth_unit, "m", knowledge.units)
    data[:, 0] = depth_values
    depth = data[:, 0]
    issues: list[str] = []
    processing_steps: list[str] = []
    if normalize_depth_unit not in ("m", "", "unknown"):
        if depth_converted:
            processing_steps.append(f"depth_unit_converted:{normalize_depth_unit}->m")
        else:
            issues.append(f"depth_unit_conversion_unavailable:{normalize_depth_unit}->m")
    if np.any(np.diff(depth[np.isfinite(depth)]) < 0) and settings.get("sort_depth", True):
        order = np.argsort(depth, kind="stable")
        data = data[order]
        depth = data[:, 0]
        processing_steps.append("depth_was_not_monotonic_and_was_sorted")
    if len(np.unique(depth[np.isfinite(depth)])) < np.sum(np.isfinite(depth)):
        strategy = str(settings.get("depth_duplicate", "mean")).lower()
        data = _collapse_duplicate_depths(data, strategy)
        depth = data[:, 0]
        processing_steps.append(f"duplicate_depth_samples_collapsed:{strategy}")

    well_name = str(header.get("WELL", Evidence(path.stem, "FILENAME", str(path), 0.6)).value).strip() or path.stem
    curves: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    infos = {}
    candidates: dict[str, list[tuple[float, str, np.ndarray, np.ndarray, Any]]] = {}
    for index, (name, unit, description) in enumerate(definitions[1:], start=1):
        values = data[:, index]
        info = knowledge.identify(name, unit, description, values)
        if info.standard_name.startswith("UNKNOWN__") and decision_resolver is not None:
            decision = decision_resolver.resolve_curve(
                mnemonic=name,
                unit=unit,
                description=description,
                values=values,
                candidates=knowledge.candidate_scores(name, unit, description, values, limit=5),
            )
            if decision and decision.accepted and decision.choice != "保留原始曲线":
                info = knowledge.reclassify(info, decision.choice, decision.confidence)
                processing_steps.append(f"llm_curve_mapping:{name}->{info.standard_name}:{decision.confidence:.3f}")
        info, inferred_unit = knowledge.infer_high_confidence_unit(info, values)
        if inferred_unit:
            processing_steps.append(f"unit_inferred_from_scale:{name}:{inferred_unit}")
        info.source = str(path)
        info, standardized, mask, curve_issues = knowledge.standardize(info, values)
        issues.extend(f"{name}:{issue}" for issue in curve_issues)
        valid_ratio = float(mask.mean()) if mask.size else 0.0
        candidates.setdefault(info.standard_name, []).append((info.confidence + 0.15 * valid_ratio, name, standardized, mask, info))
    for standard, options in candidates.items():
        options.sort(key=lambda item: item[0], reverse=True)
        _, _, values, mask, info = options[0]
        curves[standard] = values
        masks[standard] = mask
        infos[standard] = info
        if len(options) > 1:
            source_names = [item[1] for item in options]
            if len(set(source_names)) == 1:
                processing_steps.append(
                    f"duplicate_curve_columns_resolved:{standard}:selected={source_names[0]}:count={len(source_names)}"
                )
            else:
                score_margin = options[0][0] - options[1][0]
                if score_margin >= float(settings.get("curve_conflict_score_margin", 0.08)):
                    processing_steps.append(
                        f"curve_conflict_auto_resolved:{standard}:selected={source_names[0]}:candidates={','.join(source_names)}"
                    )
                else:
                    issues.append(
                        f"curve_conflict:{standard}:selected={source_names[0]}:candidates={','.join(source_names)}"
                    )
    return WellLog(
        well_name,
        depth,
        curves,
        masks,
        infos,
        header,
        str(path),
        version,
        issues,
        processing_steps,
    )
