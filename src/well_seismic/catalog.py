from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Asset


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return int(stat.st_dev), int(stat.st_ino)
    except OSError:
        return None


def _quick_signature(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(65536))
        if size > 65536:
            handle.seek(max(0, size - 65536))
            digest.update(handle.read(65536))
    return digest.hexdigest()


def build_catalog(manifest: dict[str, Any], manifest_path: str | Path) -> tuple[list[Asset], list[dict[str, Any]]]:
    base = Path(manifest_path).resolve().parent
    root = (base / manifest.get("root", ".")).resolve()
    assets: list[Asset] = []
    duplicates: list[dict[str, Any]] = []
    seen_identity: dict[tuple[int, int], str] = {}
    seen_signature: dict[str, str] = {}

    groups = list(manifest.get("inputs", []))
    asset_table = manifest.get("asset_table")
    if asset_table:
        table_path = (base / asset_table).resolve()
        rows = None
        for encoding in ("utf-8-sig", "gb18030", "cp1252"):
            try:
                with table_path.open("r", encoding=encoding, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                break
            except UnicodeDecodeError:
                continue
        if rows is None:
            raise UnicodeError(f"Cannot decode asset table: {table_path}")
        for row in rows:
            if not row.get("path") or not row.get("role"):
                continue
            options_text = row.get("options_json", "").strip()
            groups.append({
                "dataset": row.get("dataset") or "default",
                "role": row["role"],
                "path": row["path"],
                "stage": row.get("stage") or "UNKNOWN",
                "version": row.get("version") or None,
                "options": json.loads(options_text) if options_text else {},
            })

    for group in groups:
        role = group["role"]
        dataset = group.get("dataset", "default")
        declared_path = group.get("path")
        target = (root / declared_path).resolve() if declared_path else (root / group["directory"]).resolve()
        if not target.exists():
            if group.get("required", False):
                raise FileNotFoundError(f"Required input path does not exist: {target}")
            continue
        if target.is_file():
            files: set[Path] = {target}
        else:
            patterns = group.get("patterns", ["*"])
            files = set()
            for pattern in patterns:
                iterator = target.rglob(pattern) if group.get("recursive", False) else target.glob(pattern)
                files.update(path for path in iterator if path.is_file())
        for path in sorted(files):
            identity = _file_identity(path)
            duplicate_of = seen_identity.get(identity) if identity else None
            signature = None
            if duplicate_of is None and manifest.get("deduplication", {}).get("quick_signature", True):
                signature = _quick_signature(path)
                duplicate_of = seen_signature.get(signature)
            if duplicate_of:
                duplicates.append({"path": str(path), "duplicate_of": duplicate_of, "role": role})
                if manifest.get("deduplication", {}).get("skip_duplicates", True):
                    continue
            asset_id = f"{dataset}:{role}:{len(assets)+1:04d}"
            asset = Asset(
                asset_id=asset_id,
                role=role,
                path=path,
                dataset=dataset,
                version=group.get("version"),
                stage=group.get("stage", "UNKNOWN"),
                # Directory groups can expand into many assets.  Each asset
                # needs an independent effective-options document because a
                # validated per-file parse repair must never leak into a
                # sibling file from the same directory.
                options=copy.deepcopy(group.get("options", {})),
                identity=identity,
            )
            assets.append(asset)
            if identity:
                seen_identity[identity] = str(path)
            if signature:
                seen_signature[signature] = str(path)
    return assets, duplicates
