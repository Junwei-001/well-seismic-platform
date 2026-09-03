"""Content-bound, non-executable cache for sealed SEG-Y geometry.

The expensive part of replaying a sealed SourceSnapshot is scanning every
SEG-Y trace header to reconstruct the full inline/crossline/XY arrays.  This
module persists only that deterministic geometry in NumPy archives.  It never
serializes a pipeline object and never enables pickle loading.

Every cache generation is bound to the sealed snapshot fingerprint, source
asset identities, parser options, effective configuration, transformation
registry, and service build.  A missing or invalid cache is therefore a cache
miss, never an authority for changing SourceSnapshot semantics.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import uuid
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .content_identity import (
    canonical_sha256,
    file_sha256,
    seismic_geometry_identity,
)
from .models import Asset, SeismicGeometry


SEALED_GEOMETRY_CACHE_CONTRACT_VERSION = (
    "well-seismic.sealed-segy-geometry-cache.v1"
)
_ARRAY_NAMES = frozenset(
    {
        "time_axis",
        "inline",
        "crossline",
        "x",
        "y",
        "trace_offsets",
        "coordinate_scalar",
    }
)
_ARCHIVE_NAME = re.compile(r"g[0-9]{3}_[0-9a-f]{32}\.npz\Z")


class SealedGeometryCacheMiss(RuntimeError):
    """The cache is absent or was produced for another immutable contract."""


class SealedGeometryCacheIntegrityError(ValueError):
    """The cache exists but fails its non-executable integrity contract."""


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SealedGeometryCacheIntegrityError(
            f"{field} must be a lowercase-compatible SHA-256"
        )
    return normalized


def _cache_key(
    *,
    snapshot_id: str,
    source_snapshot_sha256: str,
    effective_config_sha256: str,
    transformation_registry_sha256: str,
    service_build_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
            "snapshot_id": str(snapshot_id),
            "source_snapshot_sha256": _require_sha256(
                source_snapshot_sha256,
                field="source_snapshot_sha256",
            ),
            "effective_config_sha256": _require_sha256(
                effective_config_sha256,
                field="effective_config_sha256",
            ),
            "transformation_registry_sha256": _require_sha256(
                transformation_registry_sha256,
                field="transformation_registry_sha256",
            ),
            "service_build_sha256": _require_sha256(
                service_build_sha256,
                field="service_build_sha256",
            ),
        }
    )


def cache_manifest_path(
    cache_root: str | Path,
    *,
    snapshot_id: str,
    source_snapshot_sha256: str,
    effective_config_sha256: str,
    transformation_registry_sha256: str,
    service_build_sha256: str,
) -> Path:
    """Return a content-addressed path without embedding a caller-owned id."""

    key = _cache_key(
        snapshot_id=snapshot_id,
        source_snapshot_sha256=source_snapshot_sha256,
        effective_config_sha256=effective_config_sha256,
        transformation_registry_sha256=transformation_registry_sha256,
        service_build_sha256=service_build_sha256,
    )
    root = Path(cache_root).expanduser().resolve()
    # Keep enough content-address bits for collision resistance while staying
    # below the legacy Windows MAX_PATH limit in deeply nested workspaces.
    return root / key[:2] / key[:32] / "manifest.json"


def _expected_seismic_assets(
    snapshot_assets: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for raw in snapshot_assets:
        if str(raw.get("role") or "").casefold() != "seismic":
            continue
        path = str(raw.get("path") or "").strip()
        if not path:
            raise SealedGeometryCacheIntegrityError(
                "sealed seismic asset is missing its path"
            )
        key = _path_key(path)
        if key in expected:
            raise SealedGeometryCacheIntegrityError(
                "sealed snapshot contains duplicate seismic paths"
            )
        expected[key] = dict(raw)
    return expected


def _catalog_seismic_assets(
    catalog_assets: Iterable[Asset],
) -> dict[str, Asset]:
    result: dict[str, Asset] = {}
    for asset in catalog_assets:
        if str(asset.role).casefold() != "seismic":
            continue
        key = _path_key(asset.path)
        if key in result:
            raise SealedGeometryCacheIntegrityError(
                "pipeline catalog contains duplicate seismic paths"
            )
        result[key] = asset
    return result


def _asset_options_sha256(asset: Asset) -> str:
    return canonical_sha256(_json_safe(asset.options))


def _geometry_metadata(geometry: SeismicGeometry) -> dict[str, Any]:
    return {
        "path": str(Path(geometry.path).expanduser().resolve()),
        "revision": geometry.revision,
        "endian": geometry.endian,
        "sample_format": int(geometry.sample_format),
        "sample_interval": float(geometry.sample_interval),
        "samples_per_trace": int(geometry.samples_per_trace),
        "trace_count": int(geometry.trace_count),
        "profile": str(geometry.profile),
        "confidence": float(geometry.confidence),
        "issues": [str(item) for item in geometry.issues],
        "source_crs": geometry.source_crs,
        "horizontal_crs": geometry.horizontal_crs,
        "coordinate_transform": _json_safe(geometry.coordinate_transform),
    }


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_sealed_geometry_cache(
    cache_root: str | Path,
    *,
    snapshot_id: str,
    source_snapshot_sha256: str,
    snapshot_assets: Iterable[Mapping[str, Any]],
    catalog_assets: Iterable[Asset],
    seismic: Iterable[tuple[Asset, Any]],
    effective_config_sha256: str,
    transformation_registry_sha256: str,
    service_build_sha256: str,
) -> dict[str, Any]:
    """Persist readable geometry and return an auditable cache receipt."""

    manifest_path = cache_manifest_path(
        cache_root,
        snapshot_id=snapshot_id,
        source_snapshot_sha256=source_snapshot_sha256,
        effective_config_sha256=effective_config_sha256,
        transformation_registry_sha256=transformation_registry_sha256,
        service_build_sha256=service_build_sha256,
    )
    cache_directory = manifest_path.parent
    cache_directory.mkdir(parents=True, exist_ok=True)
    expected = _expected_seismic_assets(snapshot_assets)
    catalog = _catalog_seismic_assets(catalog_assets)
    entries: list[dict[str, Any]] = []

    for index, (asset, reader) in enumerate(seismic):
        geometry = getattr(reader, "geometry", None)
        if geometry is None:
            continue
        path_key = _path_key(asset.path)
        sealed = expected.get(path_key)
        current = catalog.get(path_key)
        if sealed is None or current is None:
            raise SealedGeometryCacheIntegrityError(
                f"readable geometry is absent from the sealed catalog: {asset.path}"
            )
        source_sha256 = _require_sha256(
            sealed.get("sha256"), field=f"source asset {asset.path} sha256"
        )
        geometry_fingerprint = _require_sha256(
            sealed.get("geometry_fingerprint"),
            field=f"source asset {asset.path} geometry_fingerprint",
        )
        observed_identity = seismic_geometry_identity(geometry)
        if observed_identity["geometry_fingerprint"] != geometry_fingerprint:
            raise SealedGeometryCacheIntegrityError(
                f"in-memory SEG-Y geometry differs from sealed snapshot: {asset.path}"
            )
        options_sha256 = _asset_options_sha256(current)
        sealed_options_sha256 = str(
            sealed.get("asset_options_sha256") or ""
        ).casefold()
        if sealed_options_sha256 and sealed_options_sha256 != options_sha256:
            raise SealedGeometryCacheIntegrityError(
                f"SEG-Y parser options differ from sealed snapshot: {asset.path}"
            )

        arrays = {
            name: np.ascontiguousarray(getattr(geometry, name))
            for name in sorted(_ARRAY_NAMES)
            if getattr(geometry, name) is not None
        }
        if "time_axis" not in arrays or "trace_offsets" not in arrays:
            raise SealedGeometryCacheIntegrityError(
                f"SEG-Y geometry lacks required axes: {asset.path}"
            )
        temporary_archive = cache_directory / (
            f".{index:03d}.{uuid.uuid4().hex[:8]}.npz"
        )
        np.savez(temporary_archive, **arrays)
        archive_sha256 = file_sha256(temporary_archive)
        archive_name = f"g{index:03d}_{archive_sha256[:32]}.npz"
        archive_path = cache_directory / archive_name
        if archive_path.is_file() and file_sha256(archive_path) == archive_sha256:
            temporary_archive.unlink()
        else:
            os.replace(temporary_archive, archive_path)

        entry = {
            "asset_path": str(Path(asset.path).expanduser().resolve()),
            "source_asset_size": int(sealed.get("size") or 0),
            "source_asset_sha256": source_sha256,
            "asset_options_sha256": options_sha256,
            "geometry_fingerprint": geometry_fingerprint,
            "archive": archive_name,
            "archive_sha256": archive_sha256,
            "array_names": sorted(arrays),
            "geometry": _geometry_metadata(geometry),
        }
        entry["entry_sha256"] = canonical_sha256(entry)
        entries.append(entry)

    document: dict[str, Any] = {
        "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
        "snapshot_id": str(snapshot_id),
        "source_snapshot_sha256": _require_sha256(
            source_snapshot_sha256, field="source_snapshot_sha256"
        ),
        "effective_config_sha256": _require_sha256(
            effective_config_sha256, field="effective_config_sha256"
        ),
        "transformation_registry_sha256": _require_sha256(
            transformation_registry_sha256,
            field="transformation_registry_sha256",
        ),
        "service_build_sha256": _require_sha256(
            service_build_sha256, field="service_build_sha256"
        ),
        "serialization": "numpy_npz_allow_pickle_false",
        "entries": entries,
    }
    document["cache_identity_sha256"] = canonical_sha256(document)
    _write_json_atomic(manifest_path, document)
    manifest_sha256 = file_sha256(manifest_path)
    return {
        "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
        "state": "available",
        "source_snapshot_id": str(snapshot_id),
        "source_snapshot_sha256": str(source_snapshot_sha256).casefold(),
        "manifest_sha256": manifest_sha256,
        "cache_identity_sha256": document["cache_identity_sha256"],
        "seismic_geometry_count": len(entries),
        "serialization": document["serialization"],
    }


def _read_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except FileNotFoundError as exc:
        raise SealedGeometryCacheMiss("sealed SEG-Y geometry cache is absent") from exc
    except OSError as exc:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest is unreadable"
        ) from exc
    before_signature = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
        int(getattr(before, "st_dev", 0)),
        int(getattr(before, "st_ino", 0)),
    )
    after_signature = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
        int(getattr(after, "st_dev", 0)),
        int(getattr(after, "st_ino", 0)),
    )
    if before_signature != after_signature:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest changed while it was read"
        )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest differs from its trusted "
            "SourceSnapshot task receipt"
        )
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest is unreadable"
        ) from exc
    if not isinstance(raw, dict):
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest must be an object"
        )
    return raw


def _validate_array(
    name: str,
    value: np.ndarray,
    *,
    trace_count: int,
    samples_per_trace: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype.hasobject
        or array.dtype.kind not in "biuf"
        or array.dtype.itemsize > 8
    ):
        raise SealedGeometryCacheIntegrityError(
            f"cached geometry array {name} has an unsafe dtype"
        )
    if array.ndim != 1:
        raise SealedGeometryCacheIntegrityError(
            f"cached geometry array {name} must be one-dimensional"
        )
    expected_size = samples_per_trace if name == "time_axis" else trace_count
    if array.shape != (expected_size,):
        raise SealedGeometryCacheIntegrityError(
            f"cached geometry array {name} has an invalid shape"
        )
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _load_sealed_geometry_cache(
    cache_root: str | Path,
    *,
    snapshot_id: str,
    source_snapshot_sha256: str,
    snapshot_assets: Iterable[Mapping[str, Any]],
    catalog_assets: Iterable[Asset],
    effective_config_sha256: str,
    transformation_registry_sha256: str,
    service_build_sha256: str,
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, SeismicGeometry], dict[str, Any]]:
    """Load only geometry that exactly matches the sealed source contract."""

    manifest_path = cache_manifest_path(
        cache_root,
        snapshot_id=snapshot_id,
        source_snapshot_sha256=source_snapshot_sha256,
        effective_config_sha256=effective_config_sha256,
        transformation_registry_sha256=transformation_registry_sha256,
        service_build_sha256=service_build_sha256,
    )
    if not expected_manifest_sha256:
        raise SealedGeometryCacheMiss(
            "sealed SEG-Y geometry cache has no trusted SourceSnapshot task receipt"
        )
    trusted_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        field="expected_manifest_sha256",
    )
    document = _read_manifest(
        manifest_path,
        expected_sha256=trusted_manifest_sha256,
    )
    expected_header = {
        "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
        "snapshot_id": str(snapshot_id),
        "source_snapshot_sha256": _require_sha256(
            source_snapshot_sha256, field="source_snapshot_sha256"
        ),
        "effective_config_sha256": _require_sha256(
            effective_config_sha256, field="effective_config_sha256"
        ),
        "transformation_registry_sha256": _require_sha256(
            transformation_registry_sha256,
            field="transformation_registry_sha256",
        ),
        "service_build_sha256": _require_sha256(
            service_build_sha256, field="service_build_sha256"
        ),
        "serialization": "numpy_npz_allow_pickle_false",
    }
    for field, expected_value in expected_header.items():
        if document.get(field) != expected_value:
            raise SealedGeometryCacheMiss(
                f"sealed SEG-Y geometry cache does not match {field}"
            )
    declared_identity = _require_sha256(
        document.get("cache_identity_sha256"), field="cache_identity_sha256"
    )
    identity_payload = dict(document)
    identity_payload.pop("cache_identity_sha256", None)
    if canonical_sha256(identity_payload) != declared_identity:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache manifest identity mismatch"
        )

    expected = _expected_seismic_assets(snapshot_assets)
    catalog = _catalog_seismic_assets(catalog_assets)
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise SealedGeometryCacheMiss("sealed SEG-Y geometry cache has no entries")
    loaded: dict[str, SeismicGeometry] = {}
    loaded_path_keys: set[str] = set()
    cache_directory = manifest_path.parent.resolve()

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise SealedGeometryCacheIntegrityError(
                "sealed SEG-Y geometry cache entry must be an object"
            )
        entry = dict(raw_entry)
        declared_entry_sha256 = _require_sha256(
            entry.pop("entry_sha256", None), field="entry_sha256"
        )
        if canonical_sha256(entry) != declared_entry_sha256:
            raise SealedGeometryCacheIntegrityError(
                "sealed SEG-Y geometry cache entry identity mismatch"
            )
        asset_path = str(entry.get("asset_path") or "")
        path_key = _path_key(asset_path)
        if path_key in loaded_path_keys:
            raise SealedGeometryCacheIntegrityError(
                "sealed SEG-Y geometry cache contains duplicate paths"
            )
        sealed = expected.get(path_key)
        current = catalog.get(path_key)
        if sealed is None or current is None:
            raise SealedGeometryCacheMiss(
                "cached SEG-Y asset is absent from the current sealed catalog"
            )
        if (
            int(entry.get("source_asset_size") or -1)
            != int(sealed.get("size") or -2)
            or str(entry.get("source_asset_sha256") or "").casefold()
            != str(sealed.get("sha256") or "").casefold()
            or str(entry.get("geometry_fingerprint") or "").casefold()
            != str(sealed.get("geometry_fingerprint") or "").casefold()
            or str(entry.get("asset_options_sha256") or "").casefold()
            != _asset_options_sha256(current)
        ):
            raise SealedGeometryCacheMiss(
                f"cached SEG-Y asset contract changed: {asset_path}"
            )
        archive_name = str(entry.get("archive") or "")
        archive_sha256 = _require_sha256(
            entry.get("archive_sha256"), field="archive_sha256"
        )
        if (
            not _ARCHIVE_NAME.fullmatch(archive_name)
            or not archive_name.endswith(f"_{archive_sha256[:32]}.npz")
        ):
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y archive name is unsafe"
            )
        archive_path = (cache_directory / archive_name).resolve()
        if archive_path.parent != cache_directory:
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y archive escapes its fixed cache directory"
            )
        if not archive_path.is_file() or file_sha256(archive_path) != archive_sha256:
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y archive content identity mismatch"
            )
        declared_arrays = entry.get("array_names")
        if (
            not isinstance(declared_arrays, list)
            or set(declared_arrays) - _ARRAY_NAMES
            or "time_axis" not in declared_arrays
            or "trace_offsets" not in declared_arrays
        ):
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y array declaration is invalid"
            )
        metadata = entry.get("geometry")
        if not isinstance(metadata, dict):
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y geometry metadata is missing"
            )
        if _path_key(str(metadata.get("path") or "")) != path_key:
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y geometry path does not match its asset"
            )
        trace_count = int(metadata.get("trace_count") or 0)
        samples_per_trace = int(metadata.get("samples_per_trace") or 0)
        if trace_count <= 0 or samples_per_trace <= 0:
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y geometry has invalid dimensions"
            )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                members = archive.infolist()
                expected_members = {f"{name}.npy" for name in declared_arrays}
                if (
                    {member.filename for member in members} != expected_members
                    or any(
                        member.is_dir()
                        or member.compress_type != zipfile.ZIP_STORED
                        for member in members
                    )
                ):
                    raise SealedGeometryCacheIntegrityError(
                        "cached SEG-Y archive member contract is unsafe"
                    )
                maximum_payload_bytes = sum(
                    (
                        samples_per_trace
                        if name == "time_axis"
                        else trace_count
                    )
                    * 8
                    + 1024
                    for name in declared_arrays
                )
                if sum(member.file_size for member in members) > maximum_payload_bytes:
                    raise SealedGeometryCacheIntegrityError(
                        "cached SEG-Y archive exceeds its declared geometry bounds"
                    )
            with np.load(archive_path, allow_pickle=False) as payload:
                if set(payload.files) != set(declared_arrays):
                    raise SealedGeometryCacheIntegrityError(
                        "cached SEG-Y archive arrays differ from its manifest"
                    )
                arrays = {
                    name: _validate_array(
                        name,
                        payload[name],
                        trace_count=trace_count,
                        samples_per_trace=samples_per_trace,
                    )
                    for name in declared_arrays
                }
        except SealedGeometryCacheIntegrityError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y archive cannot be decoded safely"
            ) from exc

        geometry = SeismicGeometry(
            path=str(Path(asset_path).expanduser().resolve()),
            revision=metadata.get("revision"),
            endian=str(metadata.get("endian") or ""),
            sample_format=int(metadata.get("sample_format") or 0),
            sample_interval=float(metadata.get("sample_interval") or 0.0),
            samples_per_trace=samples_per_trace,
            trace_count=trace_count,
            time_axis=arrays["time_axis"],
            inline=arrays.get("inline"),
            crossline=arrays.get("crossline"),
            x=arrays.get("x"),
            y=arrays.get("y"),
            trace_offsets=arrays["trace_offsets"],
            coordinate_scalar=arrays.get("coordinate_scalar"),
            profile=str(metadata.get("profile") or ""),
            confidence=float(metadata.get("confidence") or 0.0),
            issues=[str(item) for item in (metadata.get("issues") or [])],
            source_crs=(
                str(metadata["source_crs"])
                if metadata.get("source_crs") is not None
                else None
            ),
            horizontal_crs=(
                str(metadata["horizontal_crs"])
                if metadata.get("horizontal_crs") is not None
                else None
            ),
            coordinate_transform=dict(metadata.get("coordinate_transform") or {}),
        )
        observed_fingerprint = seismic_geometry_identity(geometry)[
            "geometry_fingerprint"
        ]
        if observed_fingerprint != str(
            sealed.get("geometry_fingerprint") or ""
        ).casefold():
            raise SealedGeometryCacheIntegrityError(
                "cached SEG-Y geometry differs from sealed geometry fingerprint"
            )
        loaded_path_keys.add(path_key)
        loaded[str(Path(asset_path).expanduser().resolve())] = geometry

    receipt = {
        "contract_version": SEALED_GEOMETRY_CACHE_CONTRACT_VERSION,
        "state": "hit",
        "source_snapshot_id": str(snapshot_id),
        "source_snapshot_sha256": str(source_snapshot_sha256).casefold(),
        "manifest_sha256": trusted_manifest_sha256,
        "cache_identity_sha256": declared_identity,
        "seismic_geometry_count": len(loaded),
        "serialization": document["serialization"],
        "validation": (
            "snapshot+asset_sha256+options+config+transform+service_build+"
            "archive_sha256+shape+dtype+geometry_fingerprint"
        ),
    }
    return loaded, receipt


def load_sealed_geometry_cache(
    cache_root: str | Path,
    *,
    snapshot_id: str,
    source_snapshot_sha256: str,
    snapshot_assets: Iterable[Mapping[str, Any]],
    catalog_assets: Iterable[Asset],
    effective_config_sha256: str,
    transformation_registry_sha256: str,
    service_build_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, SeismicGeometry], dict[str, Any]]:
    """Load a cache while normalizing every malformed schema into rejection."""

    try:
        return _load_sealed_geometry_cache(
            cache_root,
            snapshot_id=snapshot_id,
            source_snapshot_sha256=source_snapshot_sha256,
            snapshot_assets=snapshot_assets,
            catalog_assets=catalog_assets,
            effective_config_sha256=effective_config_sha256,
            transformation_registry_sha256=transformation_registry_sha256,
            service_build_sha256=service_build_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except (SealedGeometryCacheMiss, SealedGeometryCacheIntegrityError):
        raise
    except (
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise SealedGeometryCacheIntegrityError(
            "sealed SEG-Y geometry cache schema is malformed"
        ) from exc


__all__ = [
    "SEALED_GEOMETRY_CACHE_CONTRACT_VERSION",
    "SealedGeometryCacheIntegrityError",
    "SealedGeometryCacheMiss",
    "cache_manifest_path",
    "load_sealed_geometry_cache",
    "write_sealed_geometry_cache",
]
