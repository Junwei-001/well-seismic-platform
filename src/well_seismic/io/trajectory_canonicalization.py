"""Auditable, capability-first canonicalisation of trajectory station rows.

Source row order is not a physical trajectory contract: MD is.  This module
therefore provides the one repair that is always deterministic -- move whole
station tuples into stable MD order -- and folds only geometrically congruent
duplicates.  Conflicting values at the same canonical MD are deliberately
retained so the existing formal/P13 gates isolate that trajectory instead of
silently choosing a first/last/mean value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_LINEAR_TOLERANCE_M = 1e-6
_ANGLE_TOLERANCE_DEG = 1e-6


@dataclass(frozen=True)
class StationCanonicalization:
    """Selected source rows and an immutable normalisation receipt."""

    source_indices: np.ndarray
    receipt: dict[str, Any] | None
    conflicting_duplicate_md: bool


def _value_token(value: float) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if np.isposinf(numeric):
        return "+inf"
    if np.isneginf(numeric):
        return "-inf"
    return numeric.hex()


def _station_digest(
    md: np.ndarray,
    arrays: Mapping[str, np.ndarray | None],
    source_indices: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    named_arrays: dict[str, np.ndarray] = {"md": np.asarray(md, dtype=float)}
    named_arrays.update(
        {
            str(name): np.asarray(values, dtype=float)
            for name, values in arrays.items()
            if values is not None
        }
    )
    for name in sorted(named_arrays):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        values = named_arrays[name]
        for index in source_indices:
            digest.update(_value_token(values[int(index)]).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _equivalent(values: np.ndarray, field: str) -> bool:
    numeric = np.asarray(values, dtype=float)
    finite = np.isfinite(numeric)
    if not np.all(finite):
        # A missing value and a measured value are not interchangeable.  All
        # missing values are congruent only when they share the same kind.
        if np.any(finite):
            return False
        return bool(
            np.all(np.isnan(numeric))
            or np.all(np.isposinf(numeric))
            or np.all(np.isneginf(numeric))
        )
    if field == "azimuth":
        delta = np.abs(numeric - numeric[0]) % 360.0
        delta = np.minimum(delta, 360.0 - delta)
        return bool(np.all(delta <= _ANGLE_TOLERANCE_DEG))
    tolerance = (
        _ANGLE_TOLERANCE_DEG
        if field in {"inclination"}
        else _LINEAR_TOLERANCE_M
    )
    return bool(np.all(np.abs(numeric - numeric[0]) <= tolerance))


def canonicalize_station_rows(
    md: np.ndarray,
    station_arrays: Mapping[str, np.ndarray | None],
    *,
    md_identity_decimals: int = 8,
) -> StationCanonicalization:
    """Return stable-MD source indices plus a structured audit receipt.

    Non-finite MD rows are quarantined because they cannot identify a station.
    Duplicate MD identities are folded only when every supplied station field
    is congruent.  Conflicting duplicate groups remain in the returned rows so
    downstream strict-MD contracts continue to fail that one trajectory.
    """

    md_values = np.asarray(md, dtype=float)
    if md_values.ndim != 1:
        raise ValueError("Trajectory MD must be one-dimensional")
    for name, values in station_arrays.items():
        if values is None:
            continue
        numeric = np.asarray(values, dtype=float)
        if numeric.ndim != 1 or numeric.shape != md_values.shape:
            raise ValueError(
                f"Trajectory station field {name} must be one-dimensional and match MD"
            )

    all_indices = np.arange(md_values.size, dtype=int)
    finite_indices = all_indices[np.isfinite(md_values)]
    order = np.argsort(md_values[finite_indices], kind="stable")
    sorted_indices = finite_indices[order]
    sorted_identity = np.round(
        md_values[sorted_indices], decimals=int(md_identity_decimals)
    )

    keep = np.ones(sorted_indices.size, dtype=bool)
    collapsed: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    start = 0
    while start < sorted_indices.size:
        end = start + 1
        while (
            end < sorted_indices.size
            and sorted_identity[end] == sorted_identity[start]
        ):
            end += 1
        if end - start > 1:
            group_positions = np.arange(start, end, dtype=int)
            group_indices = sorted_indices[group_positions]
            conflict_fields = [
                str(name)
                for name, values in station_arrays.items()
                if values is not None
                and not _equivalent(np.asarray(values)[group_indices], str(name))
            ]
            group_receipt = {
                "canonical_md_m": float(sorted_identity[start]),
                "source_rows_1_based": [int(index) + 1 for index in group_indices],
            }
            if conflict_fields:
                group_receipt["conflicting_fields"] = conflict_fields
                conflicting.append(group_receipt)
            else:
                keep[group_positions[1:]] = False
                group_receipt["retained_source_row_1_based"] = int(group_indices[0]) + 1
                collapsed.append(group_receipt)
        start = end

    canonical_indices = sorted_indices[keep]
    nonfinite_rows = [
        int(index) + 1 for index in all_indices[~np.isfinite(md_values)]
    ]
    source_order_changed = not np.array_equal(finite_indices, sorted_indices)
    changed = bool(nonfinite_rows or source_order_changed or collapsed)
    needs_receipt = bool(changed or conflicting)
    receipt: dict[str, Any] | None = None
    if needs_receipt:
        canonical_rows = [int(index) + 1 for index in canonical_indices]
        receipt = {
            "policy": "capability_first_station_canonicalization_v1",
            "md_identity_decimals": int(md_identity_decimals),
            "input_station_count": int(md_values.size),
            "canonical_station_count": int(canonical_indices.size),
            "source_order_changed": source_order_changed,
            "dropped_nonfinite_md_source_rows_1_based": nonfinite_rows,
            "collapsed_congruent_duplicate_md_groups": collapsed,
            "conflicting_duplicate_md_groups": conflicting,
            "canonical_source_rows_1_based": (
                canonical_rows if len(canonical_rows) <= 256 else None
            ),
            "raw_station_sha256": _station_digest(
                md_values, station_arrays, all_indices
            ),
            "canonical_station_sha256": _station_digest(
                md_values, station_arrays, canonical_indices
            ),
            "source_values_modified": False,
            "formal_md_ready": bool(
                canonical_indices.size >= 2
                and not conflicting
                and np.all(
                    np.diff(
                        np.round(
                            md_values[canonical_indices],
                            decimals=int(md_identity_decimals),
                        )
                    )
                    > 0.0
                )
            ),
        }
    return StationCanonicalization(
        source_indices=canonical_indices,
        receipt=receipt,
        conflicting_duplicate_md=bool(conflicting),
    )
