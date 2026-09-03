from __future__ import annotations

import re
import os
from collections.abc import Iterable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any

from .content_identity import canonical_sha256


RECEIPT_CONTRACT_VERSION = "well-seismic.segy-geometry-receipt.v1"
SOURCE_SNAPSHOT_CONTRACT_VERSION = "well-seismic.source-snapshot.v3"
DEFAULT_MINIMUM_HEADER_CONFIDENCE = 0.9
# GeoPathTie consumes spatial samples along well trajectories, so its automatic
# promotion threshold is intentionally stricter than the generic SEG-Y reader's
# 0.35 readability floor.
DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE = 0.75

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEADER_ISSUE_PATTERN = re.compile(
    r"^(inline|crossline|x|y)_byte=(\d+):confidence=(\d+(?:\.\d+)?)$"
)
_SCALAR_ISSUE_PATTERN = re.compile(r"^coordinate_scalar_byte=(\d+):configured$")
_GRID_ISSUE_PATTERN = re.compile(
    r"^inline_crossline_grid=.*(?:^|,)confidence:(\d+(?:\.\d+)?)$"
)
_HEADER_FIELDS = ("inline", "crossline", "x", "y")


def _sha256(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return text if _SHA256_PATTERN.fullmatch(text) else ""


def _normalized_path(value: Any) -> str:
    try:
        return os.path.normcase(str(Path(str(value)).expanduser().resolve()))
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _parse_inspection_issues(
    issues: Any,
    *,
    minimum_header_confidence: float,
) -> tuple[dict[str, int], dict[str, float], int, float] | None:
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        return None
    bytes_by_field: dict[str, int] = {}
    confidence_by_field: dict[str, float] = {}
    scalar_bytes: list[int] = []
    grid_confidences: list[float] = []
    for issue in issues:
        header_match = _HEADER_ISSUE_PATTERN.fullmatch(issue)
        if header_match:
            field, raw_byte, raw_confidence = header_match.groups()
            if field in bytes_by_field:
                return None
            header_byte = int(raw_byte)
            confidence = float(raw_confidence)
            if (
                not 1 <= header_byte <= 237
                or not isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                return None
            bytes_by_field[field] = header_byte
            confidence_by_field[field] = confidence
            continue
        scalar_match = _SCALAR_ISSUE_PATTERN.fullmatch(issue)
        if scalar_match:
            scalar_bytes.append(int(scalar_match.group(1)))
            continue
        grid_match = _GRID_ISSUE_PATTERN.fullmatch(issue)
        if grid_match:
            grid_confidences.append(float(grid_match.group(1)))
    if set(bytes_by_field) != set(_HEADER_FIELDS):
        return None
    if len(set(bytes_by_field.values())) != len(bytes_by_field):
        return None
    if any(
        confidence_by_field[field] < minimum_header_confidence
        for field in _HEADER_FIELDS
    ):
        return None
    if len(scalar_bytes) != 1 or not 1 <= scalar_bytes[0] <= 239:
        return None
    scalar_range = set(range(scalar_bytes[0], scalar_bytes[0] + 2))
    if any(
        scalar_range.intersection(range(value, value + 4))
        for value in bytes_by_field.values()
    ):
        return None
    if (
        len(grid_confidences) != 1
        or not isfinite(grid_confidences[0])
        or not 0.0 <= grid_confidences[0] <= 1.0
    ):
        return None
    return bytes_by_field, confidence_by_field, scalar_bytes[0], grid_confidences[0]


def build_verified_snapshot_segy_geometry_receipts(
    result: Mapping[str, Any],
    *,
    snapshot_id: str,
    snapshot_contract_version: str,
    source_snapshot_fingerprint: str,
    snapshot_assets: Iterable[Mapping[str, Any]],
    minimum_header_confidence: float = DEFAULT_MINIMUM_HEADER_CONFIDENCE,
    minimum_geometry_confidence: float = DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE,
) -> list[dict[str, Any]]:
    """Project deterministic SEG-Y inspection evidence into a fail-closed receipt.

    The receipt does not mutate SourceSnapshot semantics. It authorizes replay of
    the exact geometry interpretation already bound by the sealed asset geometry
    fingerprint. Consumers still have to reopen the SEG-Y and reproduce that
    fingerprint before using the inferred header bytes.
    """

    snapshot_sha256 = _sha256(source_snapshot_fingerprint)
    if (
        snapshot_contract_version != SOURCE_SNAPSHOT_CONTRACT_VERSION
        or not snapshot_id
        or not snapshot_sha256
        or not isfinite(minimum_header_confidence)
        or not isfinite(minimum_geometry_confidence)
    ):
        return []
    if not 0.0 <= minimum_header_confidence <= 1.0:
        return []
    if not 0.0 <= minimum_geometry_confidence <= 1.0:
        return []

    def unique_assets(records: Iterable[Any]) -> dict[str, dict[str, Any]] | None:
        by_path: dict[str, dict[str, Any]] = {}
        for item in records:
            if (
                not isinstance(item, Mapping)
                or str(item.get("role") or "").casefold() != "seismic"
            ):
                continue
            normalized = _normalized_path(item.get("path"))
            if not normalized:
                continue
            if normalized in by_path:
                return None
            by_path[normalized] = dict(item)
        return by_path

    sealed_by_path = unique_assets(snapshot_assets)
    result_assets = unique_assets(result.get("assets") or [])
    seismic_summaries = result.get("seismic")
    if (
        sealed_by_path is None
        or result_assets is None
        or len(sealed_by_path) != 1
        or len(result_assets) != 1
        or not isinstance(seismic_summaries, list)
        or len(seismic_summaries) != 1
        or not isinstance(seismic_summaries[0], Mapping)
    ):
        return []
    receipts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for summary in seismic_summaries:
        normalized_path = _normalized_path(summary.get("path"))
        if not normalized_path:
            continue
        if normalized_path in seen_paths:
            return []
        seen_paths.add(normalized_path)
        sealed_asset = sealed_by_path.get(normalized_path)
        result_asset = result_assets.get(normalized_path)
        if sealed_asset is None or result_asset is None:
            continue
        if sealed_asset.get("integrity_status") != "sha256_verified":
            continue

        asset_sha256 = _sha256(sealed_asset.get("sha256"))
        geometry_fingerprint = _sha256(sealed_asset.get("geometry_fingerprint"))
        asset_options_sha256 = _sha256(sealed_asset.get("asset_options_sha256"))
        if not all((asset_sha256, geometry_fingerprint, asset_options_sha256)):
            continue
        if any(
            _sha256(result_asset.get(field)) != expected
            for field, expected in (
                ("sha256", asset_sha256),
                ("geometry_fingerprint", geometry_fingerprint),
                ("asset_options_sha256", asset_options_sha256),
            )
        ):
            continue
        geometry_identity = result_asset.get("geometry_identity")
        if not isinstance(geometry_identity, Mapping):
            continue
        if _sha256(geometry_identity.get("geometry_fingerprint")) != geometry_fingerprint:
            continue
        profile_name = str(geometry_identity.get("profile") or "").strip()
        if not profile_name:
            continue
        try:
            geometry_confidence = float(summary.get("confidence"))
        except (TypeError, ValueError):
            continue
        if (
            not isfinite(geometry_confidence)
            or not 0.0 <= geometry_confidence <= 1.0
            or geometry_confidence < minimum_geometry_confidence
        ):
            continue
        parsed = _parse_inspection_issues(
            summary.get("issues"),
            minimum_header_confidence=minimum_header_confidence,
        )
        if parsed is None:
            continue
        header_bytes, header_confidence, scalar_byte, grid_confidence = parsed
        # The summary confidence is capped by the grid assessment. A mismatch
        # means the mutable presentation envelope no longer reflects the
        # deterministic reader result and must not be promoted to a receipt.
        expected_geometry_confidence = min(
            sum(header_confidence.values()) / len(header_confidence),
            grid_confidence,
        )
        if abs(expected_geometry_confidence - geometry_confidence) > 0.002:
            continue
        receipt: dict[str, Any] = {
            "contract_version": RECEIPT_CONTRACT_VERSION,
            "authority": "sealed_automatic_geometry_inspection",
            "source_snapshot_id": str(snapshot_id),
            "source_snapshot_sha256": snapshot_sha256,
            "snapshot_contract_version": snapshot_contract_version,
            "source_asset_id": sealed_asset.get("id"),
            "source_asset_path": str(sealed_asset.get("path")),
            "source_asset_sha256": asset_sha256,
            "source_asset_geometry_fingerprint": geometry_fingerprint,
            "source_asset_options_sha256": asset_options_sha256,
            "profile_name": profile_name,
            "resolved_header_bytes": header_bytes,
            "coordinate_scalar_byte": scalar_byte,
            "header_confidence": header_confidence,
            "geometry_confidence": geometry_confidence,
            "minimum_header_confidence": float(minimum_header_confidence),
            "minimum_geometry_confidence": float(minimum_geometry_confidence),
            "inspection_issues_sha256": canonical_sha256(summary.get("issues")),
            "replay_policy": "reinspect_and_match_sealed_geometry_fingerprint",
        }
        receipt["receipt_sha256"] = _receipt_hash(receipt)
        receipts.append(receipt)
    receipts.sort(key=lambda item: str(item.get("source_asset_path") or "").casefold())
    return receipts


def validate_snapshot_segy_geometry_receipt(
    receipt: Mapping[str, Any],
    *,
    source_path: str | Path,
    source_snapshot_id: str,
    source_snapshot_fingerprint: str,
    snapshot_contract_version: str,
    snapshot_assets: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a projected receipt and return the six replay semantics."""

    if receipt.get("contract_version") != RECEIPT_CONTRACT_VERSION:
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt version is invalid")
    if _sha256(receipt.get("receipt_sha256")) != _receipt_hash(receipt):
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt hash mismatch")
    expected_snapshot_sha256 = _sha256(source_snapshot_fingerprint)
    if (
        snapshot_contract_version != SOURCE_SNAPSHOT_CONTRACT_VERSION
        or receipt.get("snapshot_contract_version") != snapshot_contract_version
        or str(receipt.get("source_snapshot_id") or "") != str(source_snapshot_id)
        or _sha256(receipt.get("source_snapshot_sha256")) != expected_snapshot_sha256
        or not expected_snapshot_sha256
    ):
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt snapshot binding mismatch")
    if receipt.get("authority") != "sealed_automatic_geometry_inspection":
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt authority is invalid")
    if receipt.get("replay_policy") != "reinspect_and_match_sealed_geometry_fingerprint":
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt replay policy is invalid")

    normalized_source = _normalized_path(source_path)
    if _normalized_path(receipt.get("source_asset_path")) != normalized_source:
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt path mismatch")
    matched_asset = next(
        (
            item
            for item in snapshot_assets
            if isinstance(item, Mapping)
            and str(item.get("role") or "").casefold() == "seismic"
            and _normalized_path(item.get("path")) == normalized_source
        ),
        None,
    )
    if matched_asset is None or matched_asset.get("integrity_status") != "sha256_verified":
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt has no verified sealed asset")
    for receipt_field, asset_field in (
        ("source_asset_sha256", "sha256"),
        ("source_asset_geometry_fingerprint", "geometry_fingerprint"),
        ("source_asset_options_sha256", "asset_options_sha256"),
    ):
        receipt_digest = _sha256(receipt.get(receipt_field))
        asset_digest = _sha256(matched_asset.get(asset_field))
        if not receipt_digest or receipt_digest != asset_digest:
            raise ValueError(
                f"GeoPathTie automatic SEG-Y geometry receipt {asset_field} binding mismatch"
            )
    if receipt.get("source_asset_id") != matched_asset.get("id"):
        raise ValueError("GeoPathTie automatic SEG-Y geometry receipt asset id mismatch")

    try:
        geometry_confidence = float(receipt.get("geometry_confidence"))
        minimum_geometry_confidence = float(receipt.get("minimum_geometry_confidence"))
        minimum_header_confidence = float(receipt.get("minimum_header_confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("GeoPathTie automatic SEG-Y geometry confidence is invalid") from exc
    if (
        not all(
            isfinite(value)
            for value in (
                geometry_confidence,
                minimum_geometry_confidence,
                minimum_header_confidence,
            )
        )
        or not 0.0 <= geometry_confidence <= 1.0
        or not 0.0 <= minimum_geometry_confidence <= 1.0
        or not 0.0 <= minimum_header_confidence <= 1.0
        or geometry_confidence < minimum_geometry_confidence
        or minimum_geometry_confidence < DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE
        or minimum_header_confidence < DEFAULT_MINIMUM_HEADER_CONFIDENCE
    ):
        raise ValueError("GeoPathTie automatic SEG-Y geometry confidence is below policy")
    header_bytes = receipt.get("resolved_header_bytes")
    header_confidence = receipt.get("header_confidence")
    if not isinstance(header_bytes, Mapping) or set(header_bytes) != set(_HEADER_FIELDS):
        raise ValueError("GeoPathTie automatic SEG-Y geometry header bytes are incomplete")
    if not isinstance(header_confidence, Mapping) or set(header_confidence) != set(_HEADER_FIELDS):
        raise ValueError("GeoPathTie automatic SEG-Y geometry confidence fields are incomplete")
    normalized_bytes: dict[str, int] = {}
    for field in _HEADER_FIELDS:
        value = header_bytes.get(field)
        confidence = header_confidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 237:
            raise ValueError(f"GeoPathTie automatic SEG-Y {field} byte is invalid")
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GeoPathTie automatic SEG-Y {field} confidence is invalid") from exc
        if (
            not isfinite(numeric_confidence)
            or not 0.0 <= numeric_confidence <= 1.0
            or numeric_confidence < minimum_header_confidence
        ):
            raise ValueError(f"GeoPathTie automatic SEG-Y {field} confidence is below policy")
        normalized_bytes[field] = value
    if len(set(normalized_bytes.values())) != len(normalized_bytes):
        raise ValueError("GeoPathTie automatic SEG-Y header bytes conflict")
    scalar_byte = receipt.get("coordinate_scalar_byte")
    if isinstance(scalar_byte, bool) or not isinstance(scalar_byte, int) or not 1 <= scalar_byte <= 239:
        raise ValueError("GeoPathTie automatic SEG-Y coordinate scalar byte is invalid")
    scalar_range = set(range(scalar_byte, scalar_byte + 2))
    if any(
        scalar_range.intersection(range(value, value + 4))
        for value in normalized_bytes.values()
    ):
        raise ValueError("GeoPathTie automatic SEG-Y coordinate scalar byte conflicts")
    profile_name = str(receipt.get("profile_name") or "").strip()
    if not profile_name:
        raise ValueError("GeoPathTie automatic SEG-Y profile is missing")
    return {
        "segy_geometry_profile": profile_name,
        "segy_inline_byte": normalized_bytes["inline"],
        "segy_crossline_byte": normalized_bytes["crossline"],
        "segy_x_byte": normalized_bytes["x"],
        "segy_y_byte": normalized_bytes["y"],
        "segy_coordinate_scalar_byte": scalar_byte,
    }


__all__ = [
    "DEFAULT_MINIMUM_GEOMETRY_CONFIDENCE",
    "DEFAULT_MINIMUM_HEADER_CONFIDENCE",
    "RECEIPT_CONTRACT_VERSION",
    "build_verified_snapshot_segy_geometry_receipts",
    "validate_snapshot_segy_geometry_receipt",
]
