"""Content-addressed manifests for data derived from a source snapshot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .content_identity import canonical_sha256, file_sha256

PREPARED_VIEW_CONTRACT_VERSION = "well-seismic.prepared-view.v1"


def _artifact_record(name: str, raw_path: str | Path) -> dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prepared-view artifact is missing: {path}")
    return {
        "name": str(name),
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _view_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = []
    for item in manifest.get("artifacts") or []:
        artifacts.append(
            {
                key: item.get(key)
                for key in ("name", "role", "schema_version", "size", "sha256")
            }
        )
    artifacts.sort(key=lambda item: str(item.get("name", "")))
    parents = [
        {
            key: item.get(key)
            for key in ("kind", "view_id", "view_sha256")
        }
        for item in (manifest.get("parents") or [])
    ]
    parents.sort(key=lambda item: (str(item.get("kind", "")), str(item.get("view_id", ""))))
    return {
        "contract_version": manifest.get("contract_version"),
        "view_id": manifest.get("view_id"),
        "kind": manifest.get("kind"),
        "source_snapshot_id": manifest.get("source_snapshot_id"),
        "source_snapshot_sha256": manifest.get("source_snapshot_sha256"),
        "producer_task_id": manifest.get("producer_task_id"),
        "parents": parents,
        "producer": manifest.get("producer") or {},
        "artifacts": artifacts,
        "gates": manifest.get("gates") or {},
    }


def build_prepared_view_manifest(
    *,
    view_id: str,
    kind: str,
    source_snapshot_id: str,
    source_snapshot_sha256: str,
    producer_task_id: str,
    artifacts: Mapping[str, str | Path],
    artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    parents: list[dict[str, Any]] | None = None,
    producer: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a path-independent identity over verified derivative artifacts."""

    if not view_id or not source_snapshot_id or not source_snapshot_sha256:
        raise ValueError("prepared view requires view, snapshot id and snapshot SHA")
    metadata = artifact_metadata or {}
    records: list[dict[str, Any]] = []
    for name, path in artifacts.items():
        record = _artifact_record(str(name), path)
        record.update(dict(metadata.get(str(name)) or {}))
        records.append(record)
    records.sort(key=lambda item: str(item["name"]))
    manifest: dict[str, Any] = {
        "contract_version": PREPARED_VIEW_CONTRACT_VERSION,
        "view_id": str(view_id),
        "kind": str(kind),
        "state": "ready",
        "source_snapshot_id": str(source_snapshot_id),
        "source_snapshot_sha256": str(source_snapshot_sha256),
        "producer_task_id": str(producer_task_id),
        "parents": list(parents or []),
        "producer": dict(producer or {}),
        "artifacts": records,
        "gates": dict(gates or {}),
    }
    manifest["view_sha256"] = canonical_sha256(_view_identity_payload(manifest))
    return manifest


def write_prepared_view_manifest(
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write and return a manifest plus its own file identity."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_prepared_view_manifest(**kwargs)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
    }


def validate_prepared_view_manifest(
    manifest_path: str | Path,
    *,
    expected_view_id: str | None = None,
    expected_source_snapshot_id: str | None = None,
    expected_source_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed if lineage, view identity or any artifact has changed."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"prepared-view manifest is invalid: {path}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("prepared-view manifest must be a JSON object")
    if manifest.get("contract_version") != PREPARED_VIEW_CONTRACT_VERSION:
        raise ValueError("prepared-view contract version is incompatible")
    for expected, key in (
        (expected_view_id, "view_id"),
        (expected_source_snapshot_id, "source_snapshot_id"),
        (expected_source_snapshot_sha256, "source_snapshot_sha256"),
    ):
        if expected is not None and str(manifest.get(key)) != str(expected):
            raise ValueError(f"prepared-view {key} lineage mismatch")
    expected_view_sha = canonical_sha256(_view_identity_payload(manifest))
    if str(manifest.get("view_sha256")) != expected_view_sha:
        raise ValueError("prepared-view identity mismatch")
    for item in manifest.get("artifacts") or []:
        artifact = Path(str(item.get("path") or "")).expanduser().resolve()
        if not artifact.is_file():
            raise ValueError(f"prepared-view artifact is missing: {artifact}")
        if artifact.stat().st_size != int(item.get("size") or -1):
            raise ValueError(f"prepared-view artifact size changed: {artifact}")
        if file_sha256(artifact) != str(item.get("sha256") or ""):
            raise ValueError(f"prepared-view artifact content changed: {artifact}")
    return {
        **manifest,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
    }


__all__ = [
    "PREPARED_VIEW_CONTRACT_VERSION",
    "build_prepared_view_manifest",
    "validate_prepared_view_manifest",
    "write_prepared_view_manifest",
]
