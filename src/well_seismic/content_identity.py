"""Content-addressed identities for immutable data snapshots.

Paths are locations, not identities.  The helpers in this module deliberately
keep the byte-content digest separate from the SEG-Y geometry digest so a
downstream product can state both which file it consumed and which grid/time
axis it interpreted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: str | Path, progress: Any = None) -> str:
    source = Path(path)
    total = source.stat().st_size
    completed = 0
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            completed += len(chunk)
            if progress:
                progress(completed, total)
    if progress and completed == 0:
        progress(0, total)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    if contiguous.size:
        digest.update(memoryview(contiguous).cast("B"))
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def seismic_geometry_identity(geometry: Any) -> dict[str, Any]:
    """Return a full header/grid identity without hashing seismic amplitudes."""

    payload = {
        "contract_version": "well-seismic.geometry.v1",
        "revision": getattr(geometry, "revision", None),
        "endian": getattr(geometry, "endian", None),
        "sample_format": int(getattr(geometry, "sample_format", 0)),
        "sample_interval_ms": float(getattr(geometry, "sample_interval", 0.0)),
        "samples_per_trace": int(getattr(geometry, "samples_per_trace", 0)),
        "trace_count": int(getattr(geometry, "trace_count", 0)),
        "profile": str(getattr(geometry, "profile", "")),
        "arrays": {
            name: _array_identity(getattr(geometry, name, None))
            for name in (
                "time_axis",
                "inline",
                "crossline",
                "x",
                "y",
                "trace_offsets",
                "coordinate_scalar",
            )
        },
    }
    return {**payload, "geometry_fingerprint": canonical_sha256(payload)}


def snapshot_assets_fingerprint(assets: Iterable[Mapping[str, Any]]) -> str:
    """Bind normalized locations, bytes and interpreted geometry as one set."""

    records = []
    for item in assets:
        records.append(
            {
                key: item.get(key)
                for key in (
                    "id",
                    "role",
                    "path",
                    "size",
                    "sha256",
                    "geometry_fingerprint",
                )
            }
        )
    records.sort(key=lambda item: (str(item.get("path", "")).casefold(), str(item.get("id", ""))))
    return canonical_sha256(records)


def snapshot_source_content_fingerprint(
    assets: Iterable[Mapping[str, Any]],
) -> str:
    """Hash source bytes/roles without making filesystem locations identity."""

    records = []
    for item in assets:
        records.append(
            {
                key: item.get(key)
                for key in (
                    "role",
                    "size",
                    "sha256",
                    "geometry_fingerprint",
                    "asset_options_sha256",
                )
            }
        )
    records.sort(
        key=lambda item: (
            str(item.get("role", "")),
            str(item.get("sha256", "")),
            int(item.get("size") or 0),
        )
    )
    return canonical_sha256(records)


def snapshot_semantics_fingerprint(semantics: Mapping[str, Any]) -> str:
    """Hash the interpretation rules that turn source bytes into data values.

    Source-file equality is insufficient when the same SEG-Y axis can be
    declared OWT or TWT, or when coordinates/units are interpreted under a
    different reference system.  This digest deliberately stays separate from
    the legacy path-bound asset-set digest so existing v2 snapshots remain
    readable while new derivatives can fail closed on semantic drift.
    """

    return canonical_sha256(dict(semantics))


def source_snapshot_fingerprint(
    assets: Iterable[Mapping[str, Any]],
    *,
    semantics: Mapping[str, Any],
    inspection_policy: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return content, semantics, policy and aggregate snapshot identities."""

    asset_set_sha256 = snapshot_assets_fingerprint(assets)
    source_content_sha256 = snapshot_source_content_fingerprint(assets)
    semantics_sha256 = snapshot_semantics_fingerprint(semantics)
    inspection_policy_sha256 = canonical_sha256(dict(inspection_policy or {}))
    snapshot_sha256 = canonical_sha256(
        {
            "contract_version": "well-seismic.source-snapshot.v3",
            "source_content_sha256": source_content_sha256,
            "semantics_sha256": semantics_sha256,
            "inspection_policy_sha256": inspection_policy_sha256,
        }
    )
    return {
        "asset_set_sha256": asset_set_sha256,
        "source_content_sha256": source_content_sha256,
        "semantics_sha256": semantics_sha256,
        "inspection_policy_sha256": inspection_policy_sha256,
        "snapshot_sha256": snapshot_sha256,
    }


__all__ = [
    "canonical_sha256",
    "file_sha256",
    "seismic_geometry_identity",
    "snapshot_assets_fingerprint",
    "snapshot_semantics_fingerprint",
    "snapshot_source_content_fingerprint",
    "source_snapshot_fingerprint",
]
