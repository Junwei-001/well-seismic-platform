"""Integrity-bound CIGVis payloads for measured-depth downstream results.

The adapter consumes only current public standard-result contracts.  It never
derives a measured-depth axis from row number, and it never republishes class
probabilities, confidence values, entropy, or fracture ranking scores.
"""

from __future__ import annotations

import re
import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .standard_results import (
    build_standard_result_bundle,
    resolve_standard_result_artifact,
    supports_standard_well_sequence_view,
)


WELL_SEQUENCE_VISUALIZATION_CONTRACT_VERSION = (
    "well-seismic.cigvis-well-sequence.v1"
)
_MAXIMUM_SEQUENCE_ROWS = 200000
_MAXIMUM_SEQUENCE_BYTES = 64 * 1024 * 1024
_MAXIMUM_RENDERED_CURVE_POINTS = 1200
_PROPERTY_UNITS = {
    "DEN": "g/cm³",
    "POR": "fraction",
    "LOG_PERM": "log10(mD)",
    "SW": "fraction",
    "VSH": "fraction",
}
_FRACTURE_LEVELS = {
    0: ("low", "相对较弱"),
    1: ("medium", "相对中等"),
    2: ("high", "相对较强"),
}
_FLUID_CLASSES = {
    0: ("Dry", "干层"),
    1: ("Water", "水层"),
    2: ("Oil", "油层"),
    3: ("Gas", "气层"),
    4: ("Mixed", "混合层"),
}


def _field(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _complete_bounded_text(path: Path) -> str:
    if path.stat().st_size > _MAXIMUM_SEQUENCE_BYTES:
        raise ValueError("well-sequence artifact exceeds the 64 MiB safety limit")
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("well-sequence artifact has no supported text encoding")


def _complete_delimited_rows(
    path: Path, *, delimiter: str
) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(_complete_bounded_text(path)), delimiter=delimiter)
    try:
        raw_headers = next(reader)
    except StopIteration:
        return [], []
    rows: list[list[str]] = []
    width = len(raw_headers)
    for row in reader:
        if len(rows) >= _MAXIMUM_SEQUENCE_ROWS:
            raise ValueError("well-sequence artifact exceeds the 200000-row safety limit")
        width = max(width, len(row))
        rows.append(list(row))
    headers = [
        str(value).strip() or f"COLUMN_{index + 1}"
        for index, value in enumerate(raw_headers + [""] * (width - len(raw_headers)))
    ]
    return headers, [row + [""] * (width - len(row)) for row in rows]


def _complete_las_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    section = ""
    curve_names: list[str] = []
    rows: list[list[str]] = []
    for raw_line in _complete_bounded_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1:].split(maxsplit=1)[0].casefold()
            continue
        if section.startswith("curve"):
            mnemonic = line.split(".", 1)[0].strip().split(maxsplit=1)[0]
            if mnemonic:
                curve_names.append(mnemonic)
        elif section.startswith("ascii") or section == "a":
            if len(rows) >= _MAXIMUM_SEQUENCE_ROWS:
                raise ValueError(
                    "well-sequence LAS exceeds the 200000-row safety limit"
                )
            rows.append(line.replace(",", " ").split())
    width = max((len(row) for row in rows), default=len(curve_names))
    headers = [
        curve_names[index] if index < len(curve_names) else f"CURVE_{index + 1}"
        for index in range(width)
    ]
    return headers, [row + [""] * (width - len(row)) for row in rows]


def _column_index(headers: Sequence[str], candidates: Sequence[str]) -> int | None:
    by_name = {_field(name): index for index, name in enumerate(headers)}
    for candidate in candidates:
        if (index := by_name.get(_field(candidate))) is not None:
            return index
    return None


def _numeric_column(
    rows: Sequence[Sequence[str]], index: int, *, field_name: str
) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[index])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric value") from exc
        if not np.isfinite(value) or abs(value) >= 1e30 or value in {-999.25, -999.0}:
            raise ValueError(f"{field_name} contains a null or non-finite value")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _numeric_cell(row: Sequence[str], index: int) -> float | None:
    try:
        value = float(row[index])
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) >= 1e30 or value in {-999.25, -999.0}:
        return None
    return value


def _extrema_preserving_indices(
    primary: np.ndarray,
    standard_deviation: np.ndarray | None,
    *,
    maximum_points: int = _MAXIMUM_RENDERED_CURVE_POINTS,
) -> np.ndarray:
    """Bound SVG size while retaining endpoints and local extrema."""

    count = int(primary.size)
    if count <= maximum_points:
        return np.arange(count, dtype=np.int64)
    values_per_bucket = 4 if standard_deviation is not None else 2
    bucket_count = max(1, (maximum_points - 2) // values_per_bucket)
    boundaries = np.linspace(1, count - 1, bucket_count + 1, dtype=np.int64)
    selected = {0, count - 1}
    lower = primary - standard_deviation if standard_deviation is not None else None
    upper = primary + standard_deviation if standard_deviation is not None else None
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop <= start:
            continue
        local = slice(int(start), int(stop))
        selected.add(int(start + np.argmin(primary[local])))
        selected.add(int(start + np.argmax(primary[local])))
        if lower is not None and upper is not None:
            selected.add(int(start + np.argmin(lower[local])))
            selected.add(int(start + np.argmax(upper[local])))
    return np.asarray(sorted(selected), dtype=np.int64)


def _well_identity_from_result(
    result: Mapping[str, Any], *, output_key: str
) -> dict[str, str]:
    match = re.search(r"well[_-]?(\d+)", output_key, re.IGNORECASE)
    raw_index = int(match.group(1)) - 1 if match else -1
    candidates = result.get("well_outputs")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = result.get("wells")
    if (
        isinstance(candidates, Sequence)
        and not isinstance(candidates, (str, bytes))
        and 0 <= raw_index < len(candidates)
        and isinstance(candidates[raw_index], Mapping)
    ):
        item = candidates[raw_index]
        identity: dict[str, str] = {}
        for source_key, target_key in (
            ("well_uid", "wellUid"),
            ("well_id", "wellId"),
            ("well_name", "name"),
        ):
            value = str(item.get(source_key) or "").strip()
            if value:
                identity[target_key] = value
        return identity
    return {}


def _well_label_from_result(
    result: Mapping[str, Any], *, output_key: str, fallback: str
) -> str:
    identity = _well_identity_from_result(result, output_key=output_key)
    return str(
        identity.get("wellId")
        or identity.get("name")
        or identity.get("wellUid")
        or fallback
    )


def _las_well_name(path: Path) -> str | None:
    section = ""
    for raw_line in _complete_bounded_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith("~"):
            section = line[1:].split(maxsplit=1)[0].casefold()
            continue
        if section.startswith("well") and _field(line.split(".", 1)[0]) == "well":
            payload = line.split(":", 1)[0]
            value = payload.split(".", 1)[-1].strip()
            if value:
                return value
    return None


def _property_column_candidates(
    target: str, contract: Mapping[str, Any], *, suffix: str
) -> tuple[list[str], list[str]]:
    artifact_fields = contract.get("artifact_fields")
    artifact_fields = artifact_fields if isinstance(artifact_fields, Mapping) else {}
    primary = str(contract.get("primary_output") or "")
    explicit_primary = (
        str(artifact_fields.get("las_primary") or "") if suffix == ".las" else ""
    )
    main = [
        explicit_primary,
        primary,
        f"{target}_PRED_PHYSICAL_BOUNDED",
        f"{target}_physical_bounded",
        f"{target}_PRED_MEAN",
        f"{target}_prediction",
        f"{target}_PRED",
    ]
    uncertainty = [
        f"{target}_PRED_STD",
        f"{target}_ensemble_standard_deviation",
        f"{target}_STD",
        f"{target}_prediction_std",
    ]
    return [item for item in main if item], uncertainty


def _property_sequence(
    *,
    result: Mapping[str, Any],
    path: Path,
    artifact: Mapping[str, Any],
    sha256: str,
) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix == ".las":
        headers, rows = _complete_las_rows(path)
    elif suffix in {".csv", ".tsv"}:
        headers, rows = _complete_delimited_rows(
            path,
            delimiter="\t" if suffix == ".tsv" else ",",
        )
    else:
        raise ValueError("well-property CIGVis accepts only sealed CSV, TSV or LAS")
    if len(rows) < 2:
        raise ValueError("well-property result has fewer than two MD samples")
    contract = result["output_contract"]
    assert isinstance(contract, Mapping)
    target = str(contract.get("target") or result.get("target") or "").strip().upper()
    if target not in _PROPERTY_UNITS:
        raise ValueError("well-property target is not a registered physical property")
    main_candidates, uncertainty_candidates = _property_column_candidates(
        target, contract, suffix=suffix
    )
    md_index = _column_index(headers, ("MD_m", "MD", "DEPT", "DEPTH_M"))
    main_index = _column_index(headers, main_candidates)
    std_index = _column_index(headers, uncertainty_candidates)
    if md_index is None or main_index is None:
        raise ValueError("well-property artifact lacks declared MD and primary columns")
    mask_index = _column_index(
        headers, ("PRED_MASK", "prediction_mask", "valid_mask")
    )
    md_values: list[float] = []
    primary_values: list[float] = []
    std_values: list[float | None] = []
    for row in rows:
        if mask_index is not None:
            mask_value = _numeric_cell(row, mask_index)
            if mask_value is None or mask_value <= 0:
                continue
        md_value = _numeric_cell(row, md_index)
        primary_value = _numeric_cell(row, main_index)
        if md_value is None or primary_value is None:
            continue
        md_values.append(md_value)
        primary_values.append(primary_value)
        std_values.append(
            _numeric_cell(row, std_index) if std_index is not None else None
        )
    if len(md_values) < 2:
        raise ValueError("well-property artifact has fewer than two valid MD samples")
    md = np.asarray(md_values, dtype=np.float64)
    primary = np.asarray(primary_values, dtype=np.float64)
    standard_deviation = None
    if std_index is not None and all(value is not None for value in std_values):
        standard_deviation = np.asarray(std_values, dtype=np.float64)
        if np.any(standard_deviation < 0):
            standard_deviation = None
    delta = np.diff(md)
    if np.all(delta < 0):
        md = md[::-1]
        primary = primary[::-1]
        if standard_deviation is not None:
            standard_deviation = standard_deviation[::-1]
    elif not np.all(delta > 0):
        raise ValueError("well-property MD must be strictly monotonic")
    source_sample_count = int(md.size)
    rendered_indices = _extrema_preserving_indices(primary, standard_deviation)
    md = md[rendered_indices]
    primary = primary[rendered_indices]
    if standard_deviation is not None:
        standard_deviation = standard_deviation[rendered_indices]
    output_key = str(artifact.get("output_key") or "")
    fallback = re.sub(
        r"(?i)(?:_prediction(?:_derived)?|_por|_den|_sw|_vsh|_log_perm)$",
        "",
        path.stem,
    ).strip("_- ") or "预测井"
    well_id = (
        _las_well_name(path)
        if suffix == ".las"
        else _well_label_from_result(result, output_key=output_key, fallback=fallback)
    ) or fallback
    unit = str(result.get("units") or _PROPERTY_UNITS[target])
    payload: dict[str, Any] = {
        "kind": "property_curve",
        "name": f"{well_id} · {target} 储层物性",
        "wellId": well_id,
        "taskId": "well_property",
        "scientificStatus": str(result.get("scientific_status") or "unknown"),
        "verticalAxis": {
            "kind": "measured_depth",
            "label": "MD",
            "unit": "m",
            "values": md.tolist(),
            "source": "sealed_result_column",
        },
        "curve": {
            "target": target,
            "label": target,
            "unit": unit,
            "primaryValues": primary.tolist(),
            "primarySemantics": str(contract.get("primary_semantics") or "model_regression"),
            "primaryColumn": str(headers[main_index]),
            "uncertaintySemantics": (
                "pointwise_predictive_standard_deviation"
                if standard_deviation is not None
                else None
            ),
        },
        "sampleCount": source_sample_count,
        "renderedSampleCount": int(md.size),
        "sourceArtifact": {
            "artifactId": str(artifact.get("artifact_id") or ""),
            "sha256": sha256,
        },
        "display": {
            "mainPlot": "physical_primary_curve_with_mean_plus_minus_std"
            if standard_deviation is not None
            else "physical_primary_curve",
            "probabilityDisplayed": False,
        },
    }
    if standard_deviation is not None:
        payload["curve"]["standardDeviationValues"] = standard_deviation.tolist()
        payload["curve"]["lowerValues"] = (primary - standard_deviation).tolist()
        payload["curve"]["upperValues"] = (primary + standard_deviation).tolist()
    return payload


def _categorical_sequence(
    *,
    result: Mapping[str, Any],
    task_id: str,
    path: Path,
    artifact: Mapping[str, Any],
    sha256: str,
) -> dict[str, Any]:
    if path.suffix.casefold() != ".csv":
        raise ValueError("categorical well sequence requires a sealed interval CSV")
    headers, rows = _complete_delimited_rows(path, delimiter=",")
    by_name = {_field(name): index for index, name in enumerate(headers)}
    sample_count_index = by_name.get("sample_count")
    common = {"top_md_m", "bottom_md_m"}
    if task_id == "facies_1d":
        required = common | {"facies_code", "facies_name"}
        code_name, label_name = "facies_code", "facies_name"
    elif task_id == "fluid_interpretation":
        required = common | {
            "well_id",
            "fluid_class_code",
            "fluid_class",
            "fluid_class_zh",
        }
        code_name, label_name = "fluid_class_code", "fluid_class"
    elif task_id == "fracture_development":
        required = common | {"fracture_level_code", "fracture_level"}
        code_name, label_name = "fracture_level_code", "fracture_level"
    else:
        raise ValueError("categorical well sequence task is not registered")
    if not required <= set(by_name):
        raise ValueError("categorical interval artifact lacks its deterministic schema")
    forbidden = ("probability", "confidence", "entropy", "uncertainty")
    if any(token in name for name in by_name for token in forbidden):
        raise ValueError("categorical interval artifact exposes forbidden probability data")
    intervals: list[dict[str, Any]] = []
    well_ids: set[str] = set()
    for row in rows:
        try:
            top = float(row[by_name["top_md_m"]])
            bottom = float(row[by_name["bottom_md_m"]])
            numeric_code = float(row[by_name[code_name]])
            label = str(row[by_name[label_name]]).strip()
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("categorical interval row is invalid") from exc
        if not np.isfinite(numeric_code) or not numeric_code.is_integer():
            raise ValueError("categorical interval class code must be an integer")
        code = int(numeric_code)
        sample_count: int | None = None
        if sample_count_index is not None:
            raw_sample_count = _numeric_cell(row, sample_count_index)
            if (
                raw_sample_count is None
                or not float(raw_sample_count).is_integer()
                or raw_sample_count < 1
            ):
                raise ValueError("categorical interval sample_count is invalid")
            sample_count = int(raw_sample_count)
        if (
            not np.isfinite(top)
            or not np.isfinite(bottom)
            or bottom < top
            or (bottom == top and sample_count != 1)
            or not label
        ):
            raise ValueError("categorical interval has an invalid MD boundary or label")
        well_id = (
            str(row[by_name["well_id"]]).strip()
            if "well_id" in by_name and by_name["well_id"] < len(row)
            else ""
        )
        if task_id == "fluid_interpretation" and not well_id:
            raise ValueError("fluid interval row has no well_id")
        if well_id:
            well_ids.add(well_id)
        if task_id == "fracture_development":
            expected = _FRACTURE_LEVELS.get(code)
            if expected is None or label.casefold() != expected[0]:
                raise ValueError("fracture interval level is outside low/medium/high")
            label_zh = (
                str(row[by_name["fracture_level_zh"]]).strip()
                if "fracture_level_zh" in by_name
                else expected[1]
            )
        elif task_id == "fluid_interpretation":
            expected = _FLUID_CLASSES.get(code)
            if expected is None or label.casefold() != expected[0].casefold():
                raise ValueError("fluid interval class is outside the frozen ontology")
            label_zh = str(row[by_name["fluid_class_zh"]]).strip()
            if label_zh != expected[1]:
                raise ValueError("fluid interval Chinese label differs from its class code")
            label, label_zh = expected
        else:
            label_zh = label
        intervals.append(
            {
                "topMdM": top,
                "bottomMdM": bottom,
                "code": code,
                "label": label,
                "labelZh": label_zh,
                "_sourceSampleCount": sample_count,
            }
        )
    if not intervals:
        raise ValueError("categorical interval result is empty")
    if len(well_ids) > 1:
        raise ValueError("one interval artifact must describe exactly one well")
    intervals.sort(key=lambda item: (float(item["topMdM"]), float(item["bottomMdM"])))
    expanded_point_intervals = 0
    for index, interval in enumerate(intervals):
        top = float(interval["topMdM"])
        bottom = float(interval["bottomMdM"])
        if bottom == top:
            display_top = top
            display_bottom = bottom
            if index > 0:
                previous_bottom = float(intervals[index - 1]["bottomMdM"])
                if previous_bottom < top:
                    display_top = (previous_bottom + top) * 0.5
            if index + 1 < len(intervals):
                next_top = float(intervals[index + 1]["topMdM"])
                if next_top > bottom:
                    display_bottom = (bottom + next_top) * 0.5
            if display_bottom <= display_top:
                raise ValueError(
                    "single-sample categorical interval has no displayable MD support"
                )
            interval["topMdM"] = display_top
            interval["bottomMdM"] = display_bottom
            expanded_point_intervals += 1
        interval.pop("_sourceSampleCount", None)
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if float(current["topMdM"]) < float(previous["bottomMdM"]) - 1e-6:
            raise ValueError("categorical intervals overlap on the MD axis")
    output_key = str(artifact.get("output_key") or "")
    fallback = re.sub(r"(?i)_(?:facies|fluid|fracture)_intervals$", "", path.stem)
    row_well_id = next(iter(well_ids), "")
    well_identity = _well_identity_from_result(result, output_key=output_key)
    if row_well_id:
        well_identity["wellId"] = row_well_id
    if not well_identity:
        well_identity = {"name": fallback or "解释井"}
    well_id = str(
        well_identity.get("wellId")
        or well_identity.get("name")
        or well_identity.get("wellUid")
    )
    subject = {
        "facies_1d": "一维地震相",
        "fluid_interpretation": "流体解释",
        "fracture_development": "裂缝相对发育",
    }[task_id]
    sequence = {
        "kind": "categorical_intervals",
        "name": f"{well_id} · {subject}",
        "wellId": well_id,
        "wellIdentity": well_identity,
        "taskId": task_id,
        "scientificStatus": str(result.get("scientific_status") or "unknown"),
        "verticalAxis": {
            "kind": "measured_depth",
            "label": "MD",
            "unit": "m",
            "minimum": min(float(item["topMdM"]) for item in intervals),
            "maximum": max(float(item["bottomMdM"]) for item in intervals),
            "source": "sealed_interval_boundaries",
        },
        "intervals": intervals,
        "intervalCount": len(intervals),
        "sourceArtifact": {
            "artifactId": str(artifact.get("artifact_id") or ""),
            "sha256": sha256,
        },
        "display": {
            "mainPlot": "deterministic_md_intervals",
            "probabilityDisplayed": False,
            "scoreDisplayed": False,
        },
    }
    if expanded_point_intervals:
        sequence["displayCompatibility"] = {
            "legacySingleSampleIntervalsExpanded": expanded_point_intervals,
            "boundaryRule": "neighbor_midpoint_support",
        }
    return sequence


def build_standard_well_sequence_preview(
    result: Mapping[str, Any], *, execution_task_id: str
) -> dict[str, Any]:
    """Build bounded CIGVis well assets from verified standard artifacts."""

    if not supports_standard_well_sequence_view(result):
        raise ValueError("prediction lacks a current CIGVis well-sequence contract")
    task_id = str(result.get("task_id") or "")
    bundle = build_standard_result_bundle(result, execution_task_id=execution_task_id)
    downloads = list(bundle["downloads"]["artifacts"])
    downloads.sort(
        key=lambda item: (
            0 if Path(str(item.get("filename") or "")).suffix.casefold() == ".csv" else 1,
            str(item.get("output_key") or ""),
            str(item.get("relative_path") or ""),
        )
    )
    sequences: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    failures: list[str] = []
    for artifact in downloads:
        suffix = Path(str(artifact.get("filename") or "")).suffix.casefold()
        if task_id == "well_property" and suffix not in {".csv", ".tsv", ".las"}:
            continue
        if task_id != "well_property" and suffix != ".csv":
            continue
        try:
            resolved = resolve_standard_result_artifact(
                result,
                execution_task_id=execution_task_id,
                artifact_id=str(artifact["artifact_id"]),
            )
            if task_id == "well_property":
                sequence = _property_sequence(
                    result=result,
                    path=resolved.path,
                    artifact=artifact,
                    sha256=resolved.sha256,
                )
            else:
                sequence = _categorical_sequence(
                    result=result,
                    task_id=task_id,
                    path=resolved.path,
                    artifact=artifact,
                    sha256=resolved.sha256,
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append(f"{artifact.get('output_key')}: {exc}")
            continue
        well_key = str(sequence["wellId"]).strip().casefold()
        if well_key in seen_wells:
            continue
        seen_wells.add(well_key)
        sequences.append(sequence)
    if not sequences:
        detail = "; ".join(failures[:3])
        raise ValueError(
            "no sealed standard artifact satisfies the CIGVis well-sequence schema"
            + (f": {detail}" if detail else "")
        )
    raw_well_count = result.get("well_count")
    expected_well_count = len(sequences)
    if raw_well_count is not None:
        try:
            expected_well_count = int(raw_well_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("well_count is invalid for CIGVis well-sequence coverage") from exc
        if expected_well_count <= 0 or len(sequences) > expected_well_count:
            raise ValueError(
                "CIGVis well-sequence coverage is inconsistent: "
                f"expected {expected_well_count}, resolved {len(sequences)}"
            )
    return {
        "contractVersion": WELL_SEQUENCE_VISUALIZATION_CONTRACT_VERSION,
        "taskId": execution_task_id,
        "interpretationTaskId": task_id,
        "scientificStatus": str(result.get("scientific_status") or "unknown"),
        "wellSequences": sequences,
        "coverage": {
            "expectedWellCount": expected_well_count,
            "resolvedWellCount": len(sequences),
            "missingWellCount": max(0, expected_well_count - len(sequences)),
            "complete": len(sequences) == expected_well_count,
            "rejectedArtifactCount": len(failures),
            "diagnostics": failures[:5],
        },
        "displayPolicy": {
            "verticalAxis": "measured_depth_m",
            "probabilityAsPrimaryPlot": False,
            "fractureScoreDisplayed": False,
        },
    }


__all__ = [
    "WELL_SEQUENCE_VISUALIZATION_CONTRACT_VERSION",
    "build_standard_well_sequence_preview",
]
