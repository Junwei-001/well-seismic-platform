from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


SEISMIC_SUFFIXES = {".sgy", ".segy"}
LOG_SUFFIXES = {".las"}
METADATA_SUFFIXES = {".dat", ".txt", ".csv", ".tsv", ".prn", ".path"}


def _find_named_directory(root: Path, candidates: list[str], required: bool = False) -> Path | None:
    normalized = {re.sub(r"[\s_\-]+", "", p.name).lower(): p for p in root.iterdir() if p.is_dir()}
    for candidate in candidates:
        key = re.sub(r"[\s_\-]+", "", candidate).lower()
        if key in normalized:
            return normalized[key]
    if required:
        raise FileNotFoundError(f"输入目录缺少必要文件夹，候选名称：{candidates}")
    return None


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
        {"dataset": "competition", "role": "seismic", "directory": str(seismic.relative_to(root)), "patterns": ["*.sgy", "*.segy", "*.SGY", "*.SEGY"], "recursive": True, "stage": "UNKNOWN", "options": {"profile": "standard_3d"}},
        {"dataset": "competition", "role": "well_logs", "directory": str(logs.relative_to(root)), "patterns": ["*.las", "*.LAS"], "recursive": True, "stage": "UNKNOWN"},
    ]
    if metadata:
        inputs.append({"dataset": "competition", "role": "well_metadata", "directory": str(metadata.relative_to(root)), "patterns": ["*.dat", "*.txt", "*.csv", "*.tsv", "*.prn", "*.path"], "recursive": True, "stage": "UNKNOWN"})
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
    *,
    recursive: bool = True,
    require_seismic: bool = True,
    require_logs: bool = True,
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
    if not any((seismic, logs, metadata, auxiliary)):
        raise ValueError("至少需要登记一个有效数据路径")

    inputs = _path_groups(
        seismic,
        role="seismic",
        patterns=["*.sgy", "*.segy", "*.SGY", "*.SEGY"],
        recursive=recursive,
        options={"profile": "standard_3d"},
    )
    inputs.extend(_path_groups(
        logs,
        role="well_logs",
        patterns=["*.las", "*.LAS"],
        recursive=recursive,
    ))
    inputs.extend(_path_groups(
        metadata,
        role="well_metadata",
        patterns=["*.dat", "*.txt", "*.csv", "*.tsv", "*.prn", "*.path", "*.xlsx", "*.xls"],
        recursive=recursive,
    ))
    inputs.extend(_path_groups(
        auxiliary,
        role="auxiliary",
        patterns=["*"],
        recursive=recursive,
    ))

    manifest = {
        "schema_version": "2.2-multi-paths",
        "root": ".",
        "deduplication": {"skip_duplicates": True, "quick_signature": True},
        "well_aliases": {},
        "inputs": inputs,
    }
    inventory = {
        "输入方式": "四类数据分别指定一个或多个绝对路径",
        "地震数据路径": [str(path) for path in seismic],
        "测井数据路径": [str(path) for path in logs],
        "井基础信息与井轨迹路径": [str(path) for path in metadata],
        "其他辅助数据路径": [str(path) for path in auxiliary],
        "递归读取": recursive,
        "说明": "路径内允许混合文件和任意子目录；直接读取原始位置，不复制大型数据文件；SEG-Y保持按需读取。",
    }
    return manifest, inventory
