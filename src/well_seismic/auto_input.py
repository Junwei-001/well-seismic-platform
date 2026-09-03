from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .coordinate_reference import (
    CoordinateReferenceError,
    canonical_crs_id,
    parse_crs,
    require_projected_metre_crs,
)

SEISMIC_SUFFIXES = {".sgy", ".segy"}
LOG_SUFFIXES = {".las", ".ac"}
METADATA_SUFFIXES = {
    ".dat", ".txt", ".csv", ".tsv", ".prn", ".path", ".dev", ".well", ".track"
}

DISCOVERY_CATEGORIES = {
    "seismic": "地震数据",
    "survey": "测区网格与坐标",
    "logs": "测井曲线",
    "wells": "井位、海拔与井轨迹",
    "interpretations": "解释成果与标签",
    "auxiliary": "其他辅助数据",
    "unclassified": "待人工分类",
}

_DISCOVERY_EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

_SURVEY_HINTS = (
    "survey", "grid", "geometry", "coordinate", "inline", "crossline", "binning", "2dline",
    "测区", "网格", "坐标", "线道", "线号", "道号", "测线",
)
_WELL_HINTS = (
    "well", "wellinfo", "well_info", "wellloc", "well_loc", "wellhead", "trajectory", "deviation",
    "井数据", "井相关", "井位", "井口", "海拔", "井轨迹", "井斜", "完钻", "测斜", "time_depth", "时深",
)
_INTERPRETATION_HINTS = (
    "fault", "horizon", "hor", "surface", "facies", "lith", "label", "interpretation", "polygon", "tops",
    "断层", "层位", "地层", "沉积相", "岩性", "标签", "解释成果", "储层",
)
_AUXILIARY_HINTS = (
    "wavelet", "velocity", "ricker", "config", "readme", "index", "session",
    "子波", "速度", "配置", "说明", "索引", "会话",
)


def _text_header_tokens(path: Path) -> list[str]:
    if path.suffix.lower() not in METADATA_SUFFIXES | {".xlsx", ".xls"}:
        return []
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return []
    try:
        with path.open("rb") as handle:
            raw = handle.read(8192)
    except OSError:
        return []
    text = ""
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", first_line)[:40]]


def _classify_discovery_file(path: Path, root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    relative = str(path.relative_to(root)).replace("\\", "/")
    hint = relative.casefold()
    header_tokens = _text_header_tokens(path)
    header = " ".join(header_tokens)

    if suffix in SEISMIC_SUFFIXES:
        category, confidence, evidence = "seismic", 0.99, "SEG-Y扩展名"
    elif suffix in LOG_SUFFIXES:
        category, confidence, evidence = "logs", 0.99, "LAS扩展名"
    elif any(token in hint for token in _INTERPRETATION_HINTS):
        category, confidence, evidence = "interpretations", 0.92, "目录或文件名包含解释成果特征"
    elif suffix in {".path", ".prn", ".dev", ".well", ".track"} or any(
        token in hint for token in _WELL_HINTS
    ):
        category, confidence, evidence = "wells", 0.91, "目录、文件名或扩展名符合井基础资料"
    elif any(token in hint for token in _SURVEY_HINTS):
        category, confidence, evidence = "survey", 0.92, "目录或文件名包含测区几何特征"
    elif any(token in hint for token in _AUXILIARY_HINTS):
        category, confidence, evidence = "auxiliary", 0.86, "目录或文件名包含辅助资料特征"
    elif ("well" in header or "wellname" in header or "井名" in header) and {"x", "y"}.issubset(header_tokens):
        category, confidence, evidence = "wells", 0.90, "表头包含井名及X/Y字段"
    elif ({"inline", "crossline"}.intersection(header_tokens) and {"x", "y"}.issubset(header_tokens)):
        category, confidence, evidence = "survey", 0.90, "表头包含Inline/Crossline及X/Y字段"
    elif any(token in header for token in ("fault", "horizon", "facies", "lith", "断层", "层位", "岩性")):
        category, confidence, evidence = "interpretations", 0.86, "表头包含解释成果字段"
    elif suffix in METADATA_SUFFIXES | {".xlsx", ".xls", ".npy", ".tif", ".tiff", ".json"}:
        category, confidence, evidence = "unclassified", 0.45, "通用数据文件，现有证据不足"
    else:
        category, confidence, evidence = "auxiliary", 0.55, "非核心格式，暂列辅助资料"

    return {
        "path": str(path.resolve()),
        "relative_path": relative,
        "name": path.name,
        "suffix": suffix or "无扩展名",
        "size": path.stat().st_size if path.exists() else 0,
        "category": category,
        "category_label": DISCOVERY_CATEGORIES[category],
        "confidence": confidence,
        "source": "规则",
        "evidence": evidence,
        "header_tokens": header_tokens,
        "status": "已识别" if confidence >= 0.82 and category != "unclassified" else "待人工确认",
    }


def discover_input_root(
    input_root: str | Path,
    *,
    recursive: bool = True,
    max_files: int = 20000,
    decision_resolver: Any | None = None,
) -> dict[str, Any]:
    """Scan one local root and return reviewed, allow-listed file-role suggestions.

    Deterministic readers classify known formats first. An optional decision
    resolver may only choose from the platform's fixed categories for ambiguous
    files; it never receives file contents or seismic/log sample arrays.
    """
    root = Path(input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据根目录不存在：{root}")
    if not root.is_dir():
        raise NotADirectoryError(f"智能识别入口必须是目录：{root}")
    if max_files < 1:
        raise ValueError("max_files必须大于0")

    iterator = root.rglob("*") if recursive else root.glob("*")
    files: list[Path] = []
    truncated = False
    for path in iterator:
        if any(part.casefold() in _DISCOVERY_EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts[:-1]):
            continue
        if not path.is_file():
            continue
        if len(files) >= max_files:
            truncated = True
            break
        files.append(path)

    candidates: list[dict[str, Any]] = []
    llm_calls = 0
    llm_accepted = 0
    for path in sorted(files):
        item = _classify_discovery_file(path, root)
        if item["status"] == "待人工确认" and decision_resolver is not None:
            decision = decision_resolver.resolve_file_role(
                path=path,
                root=root,
                suffix=item["suffix"],
                size=item["size"],
                header_tokens=item.pop("header_tokens"),
                rule_category=item["category_label"],
                rule_reason=item["evidence"],
            )
            if decision is not None:
                llm_calls += 1
                if decision.accepted and decision.choice in DISCOVERY_CATEGORIES.values():
                    reverse = {label: key for key, label in DISCOVERY_CATEGORIES.items()}
                    item["category"] = reverse[decision.choice]
                    item["category_label"] = decision.choice
                    item["confidence"] = float(decision.confidence)
                    item["source"] = "LLM"
                    item["evidence"] = decision.reason
                    item["status"] = "已识别" if item["category"] != "unclassified" else "待人工确认"
                    llm_accepted += int(item["status"] == "已识别")
        else:
            item.pop("header_tokens", None)
        candidates.append(item)

    accepted = [item for item in candidates if item["status"] == "已识别"]
    suggested_paths = {
        key: [item["path"] for item in accepted if item["category"] == key]
        for key in ("seismic", "survey", "logs", "wells", "interpretations", "auxiliary")
    }
    counts = Counter(item["category"] for item in candidates)
    return {
        "root_path": str(root),
        "recursive": recursive,
        "scanned_files": len(candidates),
        "classified_files": len(accepted),
        "review_required": len(candidates) - len(accepted),
        "truncated": truncated,
        "max_files": max_files,
        "category_counts": {
            key: {"label": label, "count": counts.get(key, 0)}
            for key, label in DISCOVERY_CATEGORIES.items()
        },
        "suggested_paths": suggested_paths,
        "candidates": candidates,
        "llm": {
            "enabled": bool(decision_resolver and decision_resolver.enabled),
            "calls": llm_calls,
            "accepted": llm_accepted,
            "policy": "规则优先；仅将字段名、格式与统计摘要交给LLM，分类结果仍需人工确认",
        },
    }


def _find_named_directory(root: Path, candidates: list[str], required: bool = False) -> Path | None:
    normalized = {re.sub(r"[\s_\-]+", "", p.name).lower(): p for p in root.iterdir() if p.is_dir()}
    for candidate in candidates:
        key = re.sub(r"[\s_\-]+", "", candidate).lower()
        if key in normalized:
            return normalized[key]
    if required:
        raise FileNotFoundError(f"输入目录缺少必要文件夹，候选名称：{candidates}")
    return None


def _contains_suffix(
    paths: Sequence[Path],
    suffixes: set[str],
    *,
    recursive: bool,
) -> bool:
    accepted = {suffix.casefold() for suffix in suffixes}
    for path in paths:
        if path.is_file():
            if path.suffix.casefold() in accepted:
                return True
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        if any(candidate.is_file() and candidate.suffix.casefold() in accepted for candidate in iterator):
            return True
    return False


def build_automatic_manifest(input_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an in-memory manifest from only the four coarse input folders.

    This inventory does not infer detailed well metadata roles. All metadata-like
    files are assigned the neutral ``well_metadata`` role and classified later by
    content-aware readers.
    """
    root = Path(input_root).resolve()
    seismic = _find_named_directory(root, ["01_地震数据", "地震数据", "seismic"], required=True)
    logs = _find_named_directory(root, ["02_测井数据", "测井数据", "logs", "well_logs"], required=True)
    metadata = _find_named_directory(
        root,
        [
            "03_井基础信息与井轨迹",
            "井基础信息与井轨迹",
            "03_井相关数据",
            "井相关数据",
            "井数据",
            "well_metadata",
        ],
        required=False,
    )
    auxiliary = _find_named_directory(
        root,
        ["04_其他辅助数据", "其他辅助数据", "04_辅助数据", "辅助数据", "auxiliary"],
        required=False,
    )

    inputs: list[dict[str, Any]] = [
        {"dataset": "competition", "role": "seismic", "directory": str(seismic.relative_to(root)), "patterns": ["*.sgy", "*.segy", "*.SGY", "*.SEGY"], "recursive": True, "stage": "UNKNOWN", "options": {"profile": "standard_3d", "profile_source": "automatic_default"}},
        {"dataset": "competition", "role": "well_logs", "directory": str(logs.relative_to(root)), "patterns": ["*.las", "*.LAS", "*.ac", "*.AC"], "recursive": True, "stage": "UNKNOWN"},
    ]
    if metadata:
        if _contains_suffix([metadata], {".ac"}, recursive=True):
            inputs.append({"dataset": "competition", "role": "well_logs", "directory": str(metadata.relative_to(root)), "patterns": ["*.ac", "*.AC"], "recursive": True, "stage": "UNKNOWN"})
        inputs.append({"dataset": "competition", "role": "well_metadata", "directory": str(metadata.relative_to(root)), "patterns": ["*.dat", "*.txt", "*.csv", "*.tsv", "*.prn", "*.path", "*.dev", "*.DEV", "*.well", "*.WELL", "*.track", "*.TRACK"], "recursive": True, "stage": "UNKNOWN"})
    if auxiliary:
        inputs.append({"dataset": "competition", "role": "auxiliary", "directory": str(auxiliary.relative_to(root)), "patterns": ["*"], "recursive": True, "stage": "UNKNOWN"})
    manifest = {
        "schema_version": "2.0-auto",
        "root": str(root),
        "deduplication": {"skip_duplicates": True, "quick_signature": True},
        "well_aliases": {},
        "inputs": inputs,
    }
    inventory = {
        "输入根目录": str(root),
        "地震目录": str(seismic),
        "测井目录": str(logs),
        "井相关目录": str(metadata) if metadata else None,
        "辅助目录": str(auxiliary) if auxiliary else None,
        "说明": "清单由程序运行时自动生成，用户无需预先提供。",
    }
    return manifest, inventory


PathInput = str | Path | Sequence[str | Path] | None


def _input_paths(value: PathInput, label: str, required: bool) -> list[Path]:
    if value is None:
        raw_paths: list[str | Path] = []
    elif isinstance(value, (str, Path)):
        raw_paths = [value]
    else:
        raw_paths = list(value)

    paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if not str(raw_path).strip():
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在：{path}")
        key = str(path).casefold()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    if required and not paths:
        raise ValueError(f"{label}至少需要一个有效路径")
    return paths


def _path_groups(
    paths: list[Path],
    *,
    role: str,
    patterns: list[str],
    recursive: bool,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for path in paths:
        group: dict[str, Any] = {
            "dataset": "competition",
            "role": role,
            "stage": "UNKNOWN",
        }
        if path.is_file():
            group["path"] = str(path)
        else:
            group.update({
                "directory": str(path),
                "patterns": patterns,
                "recursive": recursive,
            })
        if options:
            group["options"] = dict(options)
        groups.append(group)
    return groups


def build_explicit_paths_manifest(
    seismic_directory: PathInput,
    log_directory: PathInput,
    metadata_directory: PathInput = None,
    auxiliary_directory: PathInput = None,
    survey_directory: PathInput = None,
    interpretation_directory: PathInput = None,
    time_depth_directory: PathInput = None,
    *,
    recursive: bool = True,
    require_seismic: bool = True,
    require_logs: bool = True,
    seismic_srd_elevation_m: float | None = None,
    vertical_crs_id: str | None = None,
    horizontal_crs_id: str | None = None,
    well_source_crs_id: str | None = None,
    seismic_source_crs_id: str | None = None,
    horizontal_unit: str = "unknown",
    horizontal_axis_order: str = "unknown",
    coordinate_reference_verified: bool = False,
    seismic_replacement_velocity_mps: float | None = None,
    seismic_time_domain: str = "unknown",
    seismic_correction_state: str = "unknown",
    segy_geometry_profile: str | None = None,
    segy_inline_byte: int | None = None,
    segy_crossline_byte: int | None = None,
    segy_x_byte: int | None = None,
    segy_y_byte: int | None = None,
    segy_coordinate_scalar_byte: int | None = None,
    well_coordinate_source_unit: str | None = None,
    well_vertical_datum_source_unit: str | None = None,
    las_twt_source_unit: str | None = None,
    time_depth_default_depth_domain: str | None = None,
    time_depth_default_depth_unit: str | None = None,
    time_depth_default_time_unit: str | None = None,
    time_depth_default_depth_datum: str | None = None,
    time_depth_default_depth_convention: str | None = None,
    time_depth_default_time_reference: str | None = None,
    time_depth_default_time_domain: str | None = None,
    time_depth_default_correction_state: str | None = None,
    target_task_id: str | None = None,
    target_model_id: str | None = None,
    target_scope_explicit: bool = False,
    target_required_modalities: Sequence[str] | None = None,
    target_model_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an in-memory manifest from independently selected paths.

    Every category accepts one or many folders/files. Source files remain in
    place. Catalog construction only scans file names, file metadata and quick
    signatures; SEG-Y samples continue to be read lazily.
    """
    seismic = _input_paths(seismic_directory, "地震数据路径", required=require_seismic)
    logs = _input_paths(log_directory, "测井数据路径", required=require_logs)
    metadata = _input_paths(metadata_directory, "井基础信息与井轨迹路径", required=False)
    auxiliary = _input_paths(auxiliary_directory, "其他辅助数据路径", required=False)
    survey = _input_paths(survey_directory, "测区网格与坐标路径", required=False)
    interpretations = _input_paths(interpretation_directory, "解释成果与标签路径", required=False)
    time_depth = _input_paths(
        time_depth_directory,
        "时深/checkshot/VSP路径",
        required=False,
    )
    if not any((seismic, logs, metadata, time_depth, auxiliary, survey, interpretations)):
        raise ValueError("至少需要登记一个有效数据路径")

    seismic_options: dict[str, Any] = {
        "profile": "standard_3d",
        "profile_source": "automatic_default",
    }
    well_coordinate_options: dict[str, Any] = {}
    declared_target_crs = str(horizontal_crs_id or "").strip()
    native_local_grid = declared_target_crs.upper().startswith("LOCAL_SURVEY_XY_")

    def projected_crs_or_empty(value: Any) -> str:
        declared = str(value or "").strip()
        if not declared or declared.upper().startswith("LOCAL_SURVEY_XY_"):
            return ""
        try:
            return canonical_crs_id(
                require_projected_metre_crs(declared, field="资产坐标CRS")
            )
        except CoordinateReferenceError:
            # Bare WGS84/Beijing54 names and local namespaces are declarations,
            # not executable pyproj transforms.
            return ""

    def source_crs_or_empty(value: Any) -> str:
        """Return an executable 2-D source CRS without imposing target units."""

        declared = str(value or "").strip()
        if not declared or declared.upper().startswith("LOCAL_SURVEY_XY_"):
            return ""
        try:
            source = parse_crs(declared, field="资产源CRS")
        except CoordinateReferenceError:
            return ""
        if not (source.is_projected or source.is_geographic):
            return ""
        return canonical_crs_id(source, field="资产源CRS")

    canonical_target_crs = (
        "" if native_local_grid else projected_crs_or_empty(declared_target_crs)
    )
    well_source_crs = (
        ""
        if native_local_grid
        else source_crs_or_empty(well_source_crs_id or canonical_target_crs)
    )
    seismic_source_crs = (
        ""
        if native_local_grid
        else source_crs_or_empty(seismic_source_crs_id or canonical_target_crs)
    )
    if canonical_target_crs:
        well_coordinate_options["target_crs"] = canonical_target_crs
        seismic_options["target_crs"] = canonical_target_crs
    if well_source_crs:
        well_coordinate_options["source_crs"] = well_source_crs
    if seismic_source_crs:
        seismic_options["source_crs"] = seismic_source_crs
    if segy_geometry_profile and segy_geometry_profile.strip():
        seismic_options.update(
            {
                "profile": segy_geometry_profile.strip(),
                "profile_source": "explicit_user",
            }
        )
    for key, value in (
        ("inline_byte", segy_inline_byte),
        ("crossline_byte", segy_crossline_byte),
        ("x_byte", segy_x_byte),
        ("y_byte", segy_y_byte),
        ("coordinate_scalar_byte", segy_coordinate_scalar_byte),
    ):
        if value is not None:
            seismic_options[key] = int(value)
    source_coordinate_unit = (
        str(well_coordinate_source_unit).lower()
        if well_coordinate_source_unit is not None
        else (
            "unknown"
            if well_source_crs
            else str(horizontal_unit or "unknown").lower()
        )
    )
    if source_coordinate_unit != "unknown":
        # Raw well-head X/Y are converted to the platform's canonical metres
        # by the readers; this records the source unit used for that conversion.
        well_coordinate_options["coordinate_unit"] = source_coordinate_unit
    source_vertical_datum_unit = str(
        well_vertical_datum_source_unit or "unknown"
    ).lower()
    if source_vertical_datum_unit != "unknown":
        # Do not inherit this from the horizontal CRS/source unit.  KB/GL/DF/RT
        # are a separate physical contract and are converted independently.
        well_coordinate_options["vertical_datum_unit"] = source_vertical_datum_unit
    if las_twt_source_unit and str(las_twt_source_unit).casefold() != "unknown":
        well_coordinate_options["twt_unit"] = str(las_twt_source_unit).lower()
    time_depth_fallbacks = {
        "default_depth_domain": time_depth_default_depth_domain,
        "default_depth_unit": time_depth_default_depth_unit,
        "default_time_unit": time_depth_default_time_unit,
        "default_depth_datum": time_depth_default_depth_datum,
        "default_depth_convention": time_depth_default_depth_convention,
        "default_time_reference": time_depth_default_time_reference,
        "default_time_domain": time_depth_default_time_domain,
        "default_correction_state": time_depth_default_correction_state,
    }
    for key, value in time_depth_fallbacks.items():
        if value is not None and str(value).strip().casefold() != "unknown":
            well_coordinate_options[key] = str(value).strip()
    if seismic_srd_elevation_m is not None:
        seismic_options["srd_elevation_m"] = float(seismic_srd_elevation_m)
    if seismic_replacement_velocity_mps is not None:
        seismic_options["replacement_velocity_mps"] = float(seismic_replacement_velocity_mps)
        well_coordinate_options["default_replacement_velocity_mps"] = float(
            seismic_replacement_velocity_mps
        )
    if seismic_time_domain != "unknown":
        seismic_options["time_domain"] = str(seismic_time_domain).upper()
    if seismic_correction_state != "unknown":
        seismic_options["correction_state"] = str(seismic_correction_state).lower()
    inputs = _path_groups(
        seismic,
        role="seismic",
        patterns=["*.sgy", "*.segy", "*.SGY", "*.SEGY"],
        recursive=recursive,
        options=seismic_options,
    )
    inputs.extend(_path_groups(
        logs,
        role="well_logs",
        patterns=["*.las", "*.LAS", "*.ac", "*.AC"],
        recursive=recursive,
        options=well_coordinate_options,
    ))
    # Specialized acoustic curves are often stored beside trajectory/metadata
    # files rather than inside the LAS directory. Keep their role explicit so
    # the generic metadata reader never guesses their columns.
    acoustic_metadata = [
        path for path in metadata
        if _contains_suffix([path], {".ac"}, recursive=recursive)
    ]
    inputs.extend(_path_groups(
        acoustic_metadata,
        role="well_logs",
        patterns=["*.ac", "*.AC"],
        recursive=recursive,
        options=well_coordinate_options,
    ))
    inputs.extend(_path_groups(
        metadata,
        role="well_metadata",
        patterns=["*.dat", "*.txt", "*.csv", "*.tsv", "*.prn", "*.path", "*.dev", "*.DEV", "*.well", "*.WELL", "*.track", "*.TRACK", "*.xlsx", "*.xls"],
        recursive=recursive,
        options=well_coordinate_options,
    ))
    # A dedicated time-depth slot must remain on the well-side parsing path.
    # ``well_metadata`` deliberately uses adaptive header/shape recognition,
    # which is safer for mixed checkshot/VSP exports than assuming a fixed
    # three-column table.  It must never be catalogued as survey geometry.
    time_depth_options = {
        **well_coordinate_options,
        "declared_input_kind": "time_depth_checkshot_vsp",
    }
    inputs.extend(_path_groups(
        time_depth,
        role="well_metadata",
        patterns=["*.dat", "*.txt", "*.csv", "*.tsv", "*.prn", "*.xlsx", "*.xls"],
        recursive=recursive,
        options=time_depth_options,
    ))
    inputs.extend(_path_groups(
        survey,
        role="survey_geometry",
        patterns=["*.dat", "*.txt", "*.csv", "*.tsv", "*.xlsx", "*.xls"],
        recursive=recursive,
    ))
    inputs.extend(_path_groups(
        interpretations,
        role="interpretation",
        patterns=["*.dat", "*.txt", "*.csv", "*.tsv", "*.xlsx", "*.xls", "*.sgy", "*.segy", "*.npy", "*.tif", "*.tiff", "*.json"],
        recursive=recursive,
    ))
    inputs.extend(_path_groups(
        auxiliary,
        role="auxiliary",
        patterns=["*"],
        recursive=recursive,
    ))

    manifest = {
        "schema_version": "2.3-reviewed-discovery",
        "root": ".",
        "deduplication": {"skip_duplicates": True, "quick_signature": True},
        "well_aliases": {},
        "vertical_crs": {
            "id": str(vertical_crs_id or "LOCAL_MSL_UNSPECIFIED"),
            "unit": "m",
            "axis": "elevation_positive_up",
        },
        "horizontal_crs": {
            "id": str(horizontal_crs_id or "UNSPECIFIED"),
            "unit": str(horizontal_unit or "unknown").lower(),
            "axis_order": str(horizontal_axis_order or "unknown").upper(),
            "verified": bool(coordinate_reference_verified),
        },
        "inputs": inputs,
        "target_task": {
            "task_id": target_task_id,
            "model_id": target_model_id,
            "scope_explicit": bool(target_scope_explicit),
            "required_modalities": list(target_required_modalities or ()),
            "model_contract": dict(target_model_contract or {}),
        },
    }
    inventory = {
        "输入方式": "四类数据分别指定一个或多个绝对路径",
        "地震数据路径": [str(path) for path in seismic],
        "测井数据路径": [str(path) for path in logs],
        "井基础信息与井轨迹路径": [str(path) for path in metadata],
        "时深_checkshot_VSP路径": [str(path) for path in time_depth],
        "测区网格与坐标路径": [str(path) for path in survey],
        "解释成果与标签路径": [str(path) for path in interpretations],
        "其他辅助数据路径": [str(path) for path in auxiliary],
        "递归读取": recursive,
        "人工确认地震SRD高程_m_MSL": seismic_srd_elevation_m,
        "垂向CRS": str(vertical_crs_id or "LOCAL_MSL_UNSPECIFIED"),
        "水平CRS": str(horizontal_crs_id or "UNSPECIFIED"),
        "井数据源CRS": well_source_crs or "auto_or_target",
        "地震数据源CRS": seismic_source_crs or "target",
        "水平单位": str(horizontal_unit or "unknown").lower(),
        "水平轴序": str(horizontal_axis_order or "unknown").upper(),
        "水平坐标参考已核验": bool(coordinate_reference_verified),
        "地震时间域": seismic_time_domain,
        "地震时间校正状态": seismic_correction_state,
        "地震替换速度_mps": seismic_replacement_velocity_mps,
        "时深缺省语义": {
            key: value
            for key, value in time_depth_fallbacks.items()
            if value is not None and str(value).strip().casefold() != "unknown"
        },
        "SEG-Y几何Profile": seismic_options.get("profile"),
        "SEG-Y几何Profile来源": seismic_options.get("profile_source"),
        "SEG-Y道头字节": {
            key: seismic_options.get(key)
            for key in (
                "inline_byte",
                "crossline_byte",
                "x_byte",
                "y_byte",
                "coordinate_scalar_byte",
            )
        },
        "井位源坐标单位": source_coordinate_unit,
        "井口高程源单位": source_vertical_datum_unit,
        "LAS_TWT源单位": str(las_twt_source_unit or "unknown").lower(),
        "目标任务": target_task_id,
        "目标模型": target_model_id,
        "说明": "路径内允许混合文件和任意子目录；直接读取原始位置，不复制大型数据文件；SEG-Y保持按需读取。",
    }
    return manifest, inventory
