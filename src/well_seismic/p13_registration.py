"""Online orchestration for the frozen P13 probabilistic well tie.

The PyTorch model lives in the sibling WellFuse checkout and is executed in
its CUDA environment.  This module owns only the platform boundary: select
label-free inputs from the current snapshot, invoke the immutable ensemble,
validate its product, and convert an accepted candidate into the platform's
registration-track contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .alignment import build_spatial_aligner
from .content_identity import canonical_sha256, file_sha256
from .model_applicability import (
    MAXIMUM_APPLICABILITY_PROFILE_TRACES,
    evaluate_applicability,
    observe_align_wells,
    observe_seismic_reader,
    resolve_training_envelope,
    write_applicability_manifest,
)
from .prediction import _wellfuse_runtime_paths, _wellfuse_subprocess_environment
from .p13_trace_support import (
    build_p13_trace_support_aligner,
    write_p13_trace_support,
)
from .registration_contract import (
    FUSION_FEATURE_TRACK_CONTRACT_VERSION,
    TRAJECTORY_STATIONWISE_TWT_POLICY,
    validate_fusion_feature_track_v3,
)
from .task_runtime import managed_run

_OUTPUT_COLUMNS = (
    "MD",
    "TVDSS",
    "X",
    "Y",
    "Z",
    "TWT_mean",
    "TWT_std",
    "quality",
)
_PROHIBITED_REQUEST_KEYS = (
    "time_depth",
    "checkshot",
    "target_twt",
    "twt_label",
    "legacy_test",
)
_P13_APPLICABILITY_DECISION_SCHEMA = "well-seismic.p13-applicability-decision.v1"
_P13_APPLICABILITY_EVIDENCE_KEYS = {
    "manifest_path",
    "manifest_sha256",
    "decision_sha256",
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _csv_flag(row: dict[str, str], column: str, *, fallback: bool) -> bool:
    value = row.get(column)
    if value is None or not str(value).strip():
        return fallback
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _safe_directory_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized or "well"


def _p13_entry_item(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    item = entry.get("item")
    return item if isinstance(item, Mapping) else entry


def _group_p13_entries_by_seismic_asset(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group wells without ever combining different SEG-Y readers/assets."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in entries:
        item = _p13_entry_item(entry)
        reader = item.get("reader")
        if reader is None:
            raise ValueError("P13 applicability grouping requires a SEG-Y reader")
        asset = item.get("asset")
        raw_path = getattr(asset, "path", None)
        asset_path = (
            os.path.normcase(str(Path(raw_path).expanduser().resolve()))
            if raw_path is not None
            else ""
        )
        grouped.setdefault((asset_path, id(reader)), []).append(entry)

    ordered = sorted(
        grouped.items(),
        key=lambda pair: (
            pair[0][0],
            sorted(
                str(
                    getattr(
                        _p13_entry_item(entry).get("entity"),
                        "well_uid",
                        "",
                    )
                )
                for entry in pair[1]
            ),
            pair[0][1],
        ),
    )
    result: list[dict[str, Any]] = []
    for ordinal, ((asset_path, _reader_identity), members) in enumerate(
        ordered, start=1
    ):
        item = _p13_entry_item(members[0])
        well_uids = sorted(
            str(getattr(_p13_entry_item(member).get("entity"), "well_uid", ""))
            for member in members
        )
        group_id = hashlib.sha256(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "asset_path": asset_path,
                    "well_uids": well_uids,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        result.append(
            {
                "group_id": group_id,
                "asset_path": asset_path or None,
                "reader": item["reader"],
                "entries": members,
                "well_uids": well_uids,
            }
        )
    return result


def _p13_applicability_decision_payload(
    applicability: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": _P13_APPLICABILITY_DECISION_SCHEMA,
        "model_id": str(applicability.get("model_id") or ""),
        "status": str(applicability.get("status") or ""),
        "decision": str(applicability.get("decision") or ""),
        "route": str(applicability.get("route") or ""),
        "score": applicability.get("score"),
        "envelope_sha256": applicability.get("envelope_sha256"),
        "manifest_sha256": str(manifest_sha256).casefold(),
    }


def _p13_applicability_evidence(
    applicability: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, str]:
    manifest = manifest_path.expanduser().resolve()
    manifest_sha256 = file_sha256(manifest)
    decision_sha256 = canonical_sha256(
        _p13_applicability_decision_payload(
            applicability,
            manifest_sha256=manifest_sha256,
        )
    )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "decision_sha256": decision_sha256,
    }


def _bind_p13_request_applicability_evidence(
    request_path: Path,
    evidence: Mapping[str, Any],
) -> None:
    if set(evidence) != _P13_APPLICABILITY_EVIDENCE_KEYS:
        raise ValueError("P13 applicability evidence fields are incomplete")
    normalized = {key: str(evidence[key]) for key in sorted(evidence)}
    for key in ("manifest_sha256", "decision_sha256"):
        digest = normalized[key].casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"P13 applicability evidence {key} is not SHA-256")
        normalized[key] = digest
    manifest = Path(normalized["manifest_path"]).expanduser().resolve()
    if not manifest.is_file() or file_sha256(manifest) != normalized["manifest_sha256"]:
        raise ValueError("P13 applicability evidence manifest binding is invalid")
    normalized["manifest_path"] = str(manifest)

    destination = request_path.expanduser().resolve()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    current = payload.get("applicability_evidence")
    if current is not None and current != normalized:
        raise ValueError("P13 request already binds different applicability evidence")
    payload["applicability_evidence"] = normalized
    _assert_label_free_request(payload)
    temporary = destination.with_suffix(destination.suffix + ".applicability.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def registration_evidence_priority(evidence: dict[str, Any]) -> int:
    """Return the fixed authority tier for one registration result.

    A learned P13 proposal is inference evidence, not a replacement for an
    available checkshot/VSP or sonic tie. A fusion-ready P13 proposal may only
    outrank lower-authority physical initializations such as constant velocity;
    an inference-only proposal may not replace any usable physical track.
    """

    source = str(
        evidence.get("registrationSource") or evidence.get("method") or ""
    ).casefold()
    status = str(
        evidence.get("registrationStatus") or evidence.get("status") or ""
    ).casefold()
    if (
        status == "provided_tie"
        or "provided_time_depth" in source
        or "checkshot" in source
        or "vsp" in source
    ):
        return 400
    if "sonic" in source:
        return 300
    if "p13" in source:
        fusion_ready = bool(
            evidence.get("fusionReady", evidence.get("fusion_ready", False))
        )
        inference_ready = bool(
            evidence.get(
                "inferenceEligible",
                evidence.get("inference_eligible", False),
            )
        )
        if fusion_ready:
            return 200
        if inference_ready:
            return 50
        return 0
    # A finite physical initialization remains authoritative over a P13 result
    # that has not passed the fusion gate.
    return 100


def arbitrate_registration_tracks(
    physical_tracks: dict[str, dict[str, Any]],
    p13_tracks: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Merge tracks using provided/VSP > sonic > fusion-ready P13 authority."""

    selected = {key: dict(track) for key, track in physical_tracks.items()}

    def identity(track: dict[str, Any], fallback: str) -> str:
        well_uid = str(track.get("well_uid") or "").strip().casefold()
        if well_uid:
            return f"uid:{well_uid}"
        well_name = str(track.get("well_name") or fallback).strip().casefold()
        return f"name:{well_name}"

    selected_by_identity = {
        identity(track, key): key for key, track in selected.items()
    }
    decisions: list[dict[str, Any]] = []
    for p13_key, p13_track in p13_tracks.items():
        track_identity = identity(p13_track, p13_key)
        selected_key = selected_by_identity.get(track_identity)
        incumbent = selected.get(selected_key) if selected_key is not None else None
        incumbent_priority = (
            registration_evidence_priority(incumbent) if incumbent is not None else -1
        )
        p13_priority = registration_evidence_priority(p13_track)
        choose_p13 = incumbent is None or p13_priority > incumbent_priority
        output_key = selected_key or p13_key
        chosen = dict(p13_track if choose_p13 else incumbent)
        diagnostics = dict(chosen.get("diagnostics") or {})
        diagnostics["registration_arbitration"] = {
            "policy": "provided_checkshot_vsp>sonic_tie>fusion_ready_p13>physical_initial>inference_only_p13",
            "selected_source": chosen.get("registrationSource"),
            "selected_priority": max(p13_priority, incumbent_priority),
            "physical_source": (
                incumbent.get("registrationSource") if incumbent is not None else None
            ),
            "physical_priority": incumbent_priority if incumbent is not None else None,
            "p13_source": p13_track.get("registrationSource"),
            "p13_priority": p13_priority,
            "p13_fusion_ready": bool(p13_track.get("fusionReady", False)),
        }
        chosen["diagnostics"] = diagnostics
        selected[output_key] = chosen
        selected_by_identity[track_identity] = output_key
        decisions.append(
            {
                "well_uid": chosen.get("well_uid"),
                "well_name": chosen.get("well_name"),
                "selected_source": chosen.get("registrationSource"),
                "selected_priority": max(p13_priority, incumbent_priority),
                "physical_source": (
                    incumbent.get("registrationSource")
                    if incumbent is not None
                    else None
                ),
                "p13_source": p13_track.get("registrationSource"),
                "p13_fusion_ready": bool(p13_track.get("fusionReady", False)),
                "reason": (
                    "p13_selected_by_authority"
                    if choose_p13
                    else "physical_registration_preserved"
                ),
            }
        )
    return selected, decisions


def _registration_identity(track: Mapping[str, Any], fallback: str = "") -> str:
    """Return a stable identity without silently joining different wells."""

    well_uid = str(track.get("well_uid") or "").strip().casefold()
    if well_uid:
        return f"uid:{well_uid}"
    well_name = str(track.get("well_name") or fallback).strip().casefold()
    return f"name:{well_name}" if well_name else ""


def _finite_series(
    track: Mapping[str, Any], column: str, count: int
) -> tuple[list[float] | None, str | None]:
    raw = track.get(column)
    if raw is None:
        return None, f"{column}_missing"
    values = list(raw)
    if len(values) != count:
        return None, f"{column}_length_mismatch"
    converted: list[float] = []
    for value in values:
        if not _finite(value):
            return None, f"{column}_not_finite"
        converted.append(float(value))
    return converted, None


def _strictly_increasing(values: list[float]) -> bool:
    return len(values) >= 2 and all(
        right > left for left, right in zip(values, values[1:])
    )


def _aligned_physical_geometry(
    physical: Mapping[str, Any], target_md: list[float]
) -> tuple[dict[str, Any] | None, str | None]:
    """Select original-geometry rows on the P13 MD grid without interpolation.

    P13 legitimately drops LAS rows whose model inputs are invalid.  Requiring
    equal track lengths therefore rejects otherwise exact subsets.  This gate
    accepts only target MD samples that already exist in the physical primary
    (within one micrometre); it never invents a position between stations.
    """

    raw_source_md = list(physical.get("md") or ())
    source_md, source_error = _finite_series(physical, "md", len(raw_source_md))
    if source_error or source_md is None or not _strictly_increasing(source_md):
        return None, "physical_md_not_strictly_increasing"
    source = np.asarray(source_md, dtype=float)
    target = np.asarray(target_md, dtype=float)
    insertion = np.searchsorted(source, target, side="left")
    right = np.clip(insertion, 0, source.size - 1)
    left = np.clip(insertion - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    indices = np.where(choose_right, right, left)
    if np.any(np.abs(source[indices] - target) > 1e-6):
        return None, "md_grid_mismatch"
    if len(set(int(index) for index in indices)) != len(indices):
        return None, "md_grid_not_one_to_one"

    def selected(column: str) -> tuple[list[float] | None, str | None]:
        raw = physical.get(column)
        if raw is None:
            return None, f"{column}_missing"
        values = list(raw)
        if len(values) != len(source_md):
            return None, f"{column}_length_mismatch"
        chosen: list[float] = []
        for index in indices:
            value = values[int(index)]
            if not _finite(value):
                return None, f"{column}_not_finite"
            chosen.append(float(value))
        return chosen, None

    def selected_optional(column: str) -> tuple[list[Any] | None, str | None]:
        raw = physical.get(column)
        if raw is None:
            return None, None
        values = list(raw)
        if len(values) != len(source_md):
            return None, f"{column}_length_mismatch"
        return [values[int(index)] for index in indices], None

    tvd, tvd_error = selected("tvd")
    x, x_error = selected("x")
    y, y_error = selected("y")
    if tvd_error or tvd is None:
        return None, f"physical_{tvd_error or 'tvd_missing'}"
    if x_error or y_error or x is None or y is None:
        return None, "physical_xy_not_finite"

    z_msl, z_error = selected("zMsl")
    depth_below_srd, depth_error = selected("depthBelowSrd")
    absolute_reference_ready = bool(physical.get("absoluteReferenceReady", True))
    inline, inline_error = selected_optional("inline")
    crossline, crossline_error = selected_optional("crossline")
    if inline_error or crossline_error:
        return None, inline_error or crossline_error
    reference = physical.get("wellReferenceElevationM")
    srd = physical.get("seismicSrdElevationM")
    if z_error:
        if not _finite(reference):
            return None, "physical_z_msl_missing_without_reference_elevation"
        z_msl = [float(reference) - value for value in tvd]
    if depth_error:
        if absolute_reference_ready:
            if not _finite(srd):
                return None, "physical_depth_below_srd_missing_without_srd_elevation"
            assert z_msl is not None
            depth_below_srd = [float(srd) - value for value in z_msl]
        else:
            # Native-relative inference has valid well TVD/z-MSL geometry but
            # intentionally no absolute seismic SRD.  Keep this column missing;
            # never substitute a fake 0 m datum.
            depth_below_srd = [float("nan")] * len(target_md)

    assert z_msl is not None and depth_below_srd is not None
    return {
        "md": target_md,
        "tvd": tvd,
        "zMsl": z_msl,
        "depthBelowSrd": depth_below_srd,
        "x": x,
        "y": y,
        "inline": inline,
        "crossline": crossline,
    }, None


def _build_p13_fusion_feature_track(
    physical: Mapping[str, Any], p13_track: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Build a separate feature track while retaining physical registration.

    Geometry comes only from the physical/original track and TWT, uncertainty,
    and quality only from the accepted P13 product. It never mutates
    ``physical`` and never opens a TD/checkshot/label input.
    """

    if not bool(p13_track.get("fusionReady")):
        return None, "p13_fusion_ready_required"
    if not bool(p13_track.get("inferenceEligible")):
        return None, "p13_inference_eligible_required"
    physical_identity = _registration_identity(physical)
    p13_identity = _registration_identity(p13_track)
    if not physical_identity or not p13_identity:
        return None, "well_identity_missing"
    if physical_identity != p13_identity:
        return None, "well_identity_mismatch"
    physical_name = str(physical.get("well_name") or "").strip().casefold()
    p13_name = str(p13_track.get("well_name") or "").strip().casefold()
    if physical_name and p13_name and physical_name != p13_name:
        return None, "well_name_mismatch"
    physical_source = str(
        physical.get("registrationSource") or physical.get("method") or ""
    ).casefold()
    if "p13" in physical_source or "wellfuse_align" in physical_source:
        return None, "physical_primary_cannot_be_learned_p13"
    if (
        str(physical.get("trajectoryTimePolicy") or "")
        == TRAJECTORY_STATIONWISE_TWT_POLICY
    ):
        # The frozen P13 feature contract is globally monotonic in MD.  It must
        # not replace a physical-primary trajectory whose time legitimately
        # reverses with measured TVD until a stationwise P13 model exists.
        return None, "p13_global_monotonic_twt_incompatible_with_stationwise_primary"

    raw_p13_md = list(p13_track.get("md") or ())
    p13_md, p13_md_error = _finite_series(p13_track, "md", len(raw_p13_md))
    if p13_md_error or p13_md is None:
        return None, "p13_md_missing_or_not_finite"
    if not _strictly_increasing(p13_md):
        return None, "p13_md_not_strictly_increasing"

    physical_geometry, geometry_error = _aligned_physical_geometry(physical, p13_md)
    if geometry_error or physical_geometry is None:
        return None, geometry_error or "physical_geometry_unavailable"
    md = physical_geometry["md"]
    twt, twt_error = _finite_series(p13_track, "twtMean", len(md))
    if twt_error or twt is None or not _strictly_increasing(twt):
        return None, "p13_twt_not_finite_strictly_increasing"

    twt_std, _ = _finite_series(p13_track, "twtStd", len(md))
    quality, _ = _finite_series(p13_track, "registrationQuality", len(md))
    p13_source = str(p13_track.get("registrationSource") or "")
    primary_source = str(physical.get("registrationSource") or "physical_primary")
    diagnostics = dict(p13_track.get("diagnostics") or {})
    diagnostics["fusion_feature_track"] = {
        "contract_version": FUSION_FEATURE_TRACK_CONTRACT_VERSION,
        "role": "experimental_p13_feature_track",
        "physical_primary_identity": physical_identity,
        "physical_primary_source": primary_source,
        "p13_source": p13_source,
        "physical_primary_preserved": True,
        "time_depth_supervision_is_model_input": False,
        "absolute_reference_ready": bool(physical.get("absoluteReferenceReady", True)),
    }
    feature_track: dict[str, Any] = {
        **dict(physical),
        "well_uid": physical.get("well_uid") or p13_track.get("well_uid"),
        "well_name": physical.get("well_name") or p13_track.get("well_name"),
        "md": md,
        "tvd": physical_geometry["tvd"],
        "zMsl": physical_geometry["zMsl"],
        "depthBelowSrd": physical_geometry["depthBelowSrd"],
        "x": physical_geometry["x"],
        "y": physical_geometry["y"],
        "inline": physical_geometry["inline"],
        "crossline": physical_geometry["crossline"],
        "twtMean": twt,
        "twtStd": twt_std if twt_std is not None else [None] * len(md),
        "registrationQuality": quality if quality is not None else [None] * len(md),
        "validMask": [True] * len(md),
        "registrationSource": p13_source,
        "registrationStatus": "experimental_fusion_feature_candidate",
        "sourceAuthority": "learned_p13_fusion_ready",
        "inferenceEligible": True,
        "fusionReady": True,
        "supervisionEligible": False,
        "trainingEligible": False,
        "absoluteReferenceReady": bool(physical.get("absoluteReferenceReady", True)),
        "diagnostics": diagnostics,
    }
    try:
        validate_fusion_feature_track_v3(feature_track)
    except ValueError as exc:
        return None, f"feature_contract_invalid:{exc}"
    return feature_track, None


def build_p13_fusion_feature_tracks(
    physical_tracks: Mapping[str, Mapping[str, Any]],
    p13_tracks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return immutable physical primaries plus eligible P13 feature tracks.

    Consumers retain ``physical_primary`` as registration authority and may
    seal ``fusion_feature_tracks`` separately for experimental feature use.
    Every non-promoted P13 track remains an auditable decision with a reason.
    """

    primary = {key: dict(track) for key, track in physical_tracks.items()}
    primary_by_identity = {
        _registration_identity(track, key): track
        for key, track in primary.items()
        if _registration_identity(track, key)
    }
    features: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    for key, p13_track in p13_tracks.items():
        identity = _registration_identity(p13_track, key)
        physical = primary_by_identity.get(identity)
        if physical is None:
            decisions.append(
                {
                    "well_identity": identity or key,
                    "well_uid": p13_track.get("well_uid"),
                    "well_name": p13_track.get("well_name"),
                    "eligible": False,
                    "reason": "physical_primary_missing",
                }
            )
            continue
        feature, reason = _build_p13_fusion_feature_track(physical, p13_track)
        decisions.append(
            {
                "well_identity": identity,
                "well_uid": p13_track.get("well_uid"),
                "well_name": p13_track.get("well_name"),
                "eligible": feature is not None,
                "reason": reason,
                "physical_primary_source": physical.get("registrationSource"),
                "p13_source": p13_track.get("registrationSource"),
            }
        )
        if feature is not None:
            feature_key = str(
                feature.get("well_uid") or feature.get("well_name") or key
            )
            features[feature_key] = feature
    return {
        "contract_version": FUSION_FEATURE_TRACK_CONTRACT_VERSION,
        "physical_primary": primary,
        "fusion_feature_tracks": features,
        "decisions": decisions,
        "fusion_feature_well_count": len(features),
        "time_depth_supervision_is_model_input": False,
    }


def _assert_label_free_request(payload: dict[str, Any]) -> None:
    keys: list[str] = []

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                keys.append(name.casefold())
                visit(item, name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload)
    offenders = [
        key for key in keys if any(token in key for token in _PROHIBITED_REQUEST_KEYS)
    ]
    if offenders:
        raise ValueError(f"P13请求包含监督字段：{offenders}")


def _header_byte(geometry: Any, field: str, default: int) -> int:
    prefix = f"{field}_byte="
    for issue in getattr(geometry, "issues", ()):
        text = str(issue)
        if text.startswith(prefix):
            try:
                return int(text[len(prefix) :].split(":", 1)[0])
            except (TypeError, ValueError):
                break
    return int(default)


def _coordinate_fields(pipeline: Any, asset: Any, reader: Any) -> dict[str, int]:
    geometry = reader.geometry
    if geometry is None:
        geometry = reader.inspect()
    profiles = pipeline.config.get("segy", {}).get("profiles", {})
    profile = profiles.get(
        asset.options.get("profile", "standard_3d"),
        profiles.get("standard_3d", {}),
    )
    return {
        "x_byte": _header_byte(geometry, "x", 181),
        "y_byte": _header_byte(geometry, "y", 185),
        "scalar_byte": int(
            asset.options.get(
                "coordinate_scalar_byte",
                profile.get("coordinate_scalar", 71),
            )
        ),
    }


def _trajectory_is_complete(trajectory: Any) -> bool:
    vertical_semantics = dict(getattr(trajectory, "vertical_semantics", {}) or {})
    if vertical_semantics.get("registration_eligible") is False:
        return False
    md = np.asarray(trajectory.md, dtype=float)
    tvd = np.asarray(trajectory.tvd, dtype=float)
    if md.ndim != 1 or tvd.shape != md.shape or md.size < 2:
        return False
    if trajectory.x is not None and trajectory.y is not None:
        x = np.asarray(trajectory.x, dtype=float)
        y = np.asarray(trajectory.y, dtype=float)
    else:
        x = np.asarray(trajectory.x_offset, dtype=float)
        y = np.asarray(trajectory.y_offset, dtype=float)
    return bool(
        x.shape == md.shape
        and y.shape == md.shape
        and np.all(np.isfinite(md))
        and np.all(np.isfinite(tvd))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.diff(md) > 0.0)
    )


def _best_acoustic_log(entity: Any) -> Any | None:
    candidates: list[tuple[int, float, Any]] = []
    for log in entity.logs:
        if "DT" not in log.curves or "DT" not in log.masks:
            continue
        values = np.asarray(log.curves["DT"], dtype=float)
        mask = np.asarray(log.masks["DT"], dtype=bool)
        depth = np.asarray(log.depth, dtype=float)
        if values.shape != depth.shape or mask.shape != depth.shape:
            continue
        info = log.curve_info.get("DT")
        unit = (
            str(getattr(info, "standard_unit", ""))
            .casefold()
            .replace("µ", "u")
            .replace("μ", "u")
        )
        if unit != "us/m":
            continue
        valid = mask & np.isfinite(depth) & np.isfinite(values) & (values > 0.0)
        count = int(valid.sum())
        if count >= 8:
            candidates.append((count, float(getattr(info, "confidence", 0.0)), log))
    return max(candidates, default=(0, 0.0, None), key=lambda item: item[:2])[2]


def _write_canonical_well_inputs(
    directory: Path,
    *,
    entity: Any,
    head: Any,
    log: Any,
    trajectory: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Serialize parsed canonical arrays, never labels, for the frozen CLI."""

    directory.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(log.depth, dtype=float)
    values = np.asarray(log.curves["DT"], dtype=float)
    mask = np.asarray(log.masks["DT"], dtype=bool)
    valid = mask & np.isfinite(depth) & np.isfinite(values) & (values > 0.0)
    acoustic = np.column_stack((depth[valid], values[valid]))
    acoustic = acoustic[np.argsort(acoustic[:, 0])]
    _, unique = np.unique(acoustic[:, 0], return_index=True)
    acoustic = acoustic[np.sort(unique)]
    if acoustic.shape[0] < 8 or bool((np.diff(acoustic[:, 0]) <= 0.0).any()):
        raise ValueError(f"{entity.canonical_name}规范声波曲线有效样点不足")

    md = np.asarray(trajectory.md, dtype=float)
    tvd = np.asarray(trajectory.tvd, dtype=float)
    if trajectory.x is not None and trajectory.y is not None:
        x_offset = np.asarray(trajectory.x, dtype=float) - float(head.x)
        y_offset = np.asarray(trajectory.y, dtype=float) - float(head.y)
    else:
        x_offset = np.asarray(trajectory.x_offset, dtype=float)
        y_offset = np.asarray(trajectory.y_offset, dtype=float)
    valid_trajectory = (
        np.isfinite(md)
        & np.isfinite(tvd)
        & np.isfinite(x_offset)
        & np.isfinite(y_offset)
    )
    path_values = np.column_stack(
        (
            md[valid_trajectory],
            tvd[valid_trajectory],
            x_offset[valid_trajectory],
            y_offset[valid_trajectory],
        )
    )
    path_values = path_values[np.argsort(path_values[:, 0])]
    _, unique = np.unique(path_values[:, 0], return_index=True)
    path_values = path_values[np.sort(unique)]
    if path_values.shape[0] < 2 or bool((np.diff(path_values[:, 0]) <= 0.0).any()):
        raise ValueError(f"{entity.canonical_name}规范轨迹有效站点不足")
    # The frozen P13 runtime integrates sonic along a non-decreasing TVD axis.
    # Near-horizontal measured surveys can legitimately contain centimetre-scale
    # TVD reversals (inclination slightly above 90 degrees).  Preserve the
    # authoritative DEV/trajectory everywhere in the platform and project only
    # this P13-specific serialized input onto its monotonic envelope.  MD and XY
    # remain bit-for-bit derived from the measured trajectory, so the model
    # product can still be mapped back onto the original 3-D well path by MD.
    measured_tvd = path_values[:, 1].copy()
    original_geometry_sha256 = hashlib.sha256(
        np.asarray(path_values, dtype="<f8").tobytes()
    ).hexdigest()
    runtime_tvd = np.maximum.accumulate(measured_tvd)
    tvd_correction = runtime_tvd - measured_tvd
    negative_intervals = np.diff(measured_tvd) < -1e-5
    corrected_stations = tvd_correction > 1e-8
    path_values[:, 1] = runtime_tvd
    trajectory_input_audit = {
        "schema": "well-seismic.p13-canonical-trajectory.v1",
        "runtime_tvd_policy": "monotonic_envelope_for_frozen_p13_only",
        "original_platform_trajectory_preserved": True,
        "md_xy_preserved": True,
        "station_count": int(path_values.shape[0]),
        "negative_tvd_interval_count": int(np.count_nonzero(negative_intervals)),
        "corrected_station_count": int(np.count_nonzero(corrected_stations)),
        "maximum_tvd_correction_m": float(
            np.max(tvd_correction) if tvd_correction.size else 0.0
        ),
        "total_negative_tvd_increment_m": float(
            -np.sum(np.minimum(np.diff(measured_tvd), 0.0))
        ),
        "original_geometry_sha256": original_geometry_sha256,
        "runtime_geometry_sha256": hashlib.sha256(
            np.asarray(path_values, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    # The frozen runtime evaluates only the measured MD intersection of the
    # acoustic log and trajectory. Requiring the entire LAS interval to sit
    # inside the trajectory rejects otherwise well-constrained wells merely
    # because a few log samples extend below the last survey station. Keep the
    # original samples for auditability, but fail closed when there is no
    # sufficiently sampled measured intersection; no trajectory extrapolation
    # is introduced here or in the runtime.
    overlap_lower = max(float(acoustic[0, 0]), float(path_values[0, 0]))
    overlap_upper = min(float(acoustic[-1, 0]), float(path_values[-1, 0]))
    overlap_samples = int(
        np.count_nonzero(
            (acoustic[:, 0] >= overlap_lower - 1e-6)
            & (acoustic[:, 0] <= overlap_upper + 1e-6)
        )
    )
    if overlap_upper <= overlap_lower or overlap_samples < 8:
        raise ValueError(
            f"{entity.canonical_name}声波曲线与完整实测轨迹没有足够的MD交集"
        )

    safe_name = _safe_directory_name(entity.well_uid)
    acoustic_path = directory / f"{safe_name}.ac"
    trajectory_path = directory / f"{safe_name}.path"
    np.savetxt(
        acoustic_path,
        acoustic,
        fmt="%.10g",
        header="measuredDepth p_ac",
        comments="#",
    )
    np.savetxt(
        trajectory_path,
        path_values,
        fmt="%.10g",
        header="measuredDepth verticalDepth xOffset yOffset",
        comments="#",
    )
    return acoustic_path, trajectory_path, trajectory_input_audit


def _optional_full_las_path(log: Any) -> tuple[Path | None, dict[str, Any]]:
    """Return the parsed log's original LAS when it is a real full-log file."""

    source = Path(str(getattr(log, "source", ""))).expanduser()
    if source.suffix.casefold() != ".las":
        return None, {
            "full_las_supplied": False,
            "mode": "canonical_dt_only",
            "degradation_reason": "selected acoustic source is not LAS",
        }
    if not source.is_file():
        return None, {
            "full_las_supplied": False,
            "mode": "canonical_dt_only",
            "degradation_reason": "original LAS path is unavailable",
        }
    return source.resolve(), {
        "full_las_supplied": True,
        "mode": "original_full_las_optional_density_evidence",
        "degradation_reason": None,
    }


def _eligible_wells(pipeline: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    matching = pipeline.config.get("matching", {})
    coordinate_reference = dict(matching.get("coordinate_reference") or {})
    coordinate_contract_ready = bool(
        coordinate_reference.get("verified") is True
        and str(coordinate_reference.get("crs") or "").strip()
        and str(coordinate_reference.get("horizontal_unit") or "").casefold() == "m"
        and str(coordinate_reference.get("axis_order") or "").upper() in {"XY", "YX"}
    )
    max_horizontal_distance = float(matching.get("max_horizontal_distance", 500.0))
    sources = pipeline._selected_seismic_sources(matching)
    aligner = build_spatial_aligner(matching).fit(sources)
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for entity in pipeline.registry.entities.values():
        reason = ""
        head = entity.preferred_head
        trajectory = entity.preferred_trajectory
        acoustic = _best_acoustic_log(entity)
        well_datum = None
        if not coordinate_contract_ready:
            reason = "unverified_horizontal_coordinate_contract"
        elif acoustic is None:
            reason = "missing_unambiguous_dt_us_per_m"
        elif trajectory is None:
            reason = "missing_measured_trajectory"
        elif (
            dict(getattr(trajectory, "vertical_semantics", {}) or {}).get(
                "registration_eligible"
            )
            is False
        ):
            reason = "unresolved_trajectory_vertical_datum_conflict"
        elif not _trajectory_is_complete(trajectory):
            reason = "incomplete_measured_trajectory"
        elif head is None or not (_finite(head.x) and _finite(head.y)):
            reason = "missing_surface_xy"
        else:
            try:
                well_datum = pipeline._well_datum(entity)
            except (AttributeError, TypeError, ValueError):
                well_datum = None
            if (
                well_datum is None
                or not bool(getattr(well_datum, "ready", False))
                or str(getattr(well_datum, "datum", "") or "").upper()
                not in {"KB", "DF", "RT"}
                or not _finite(getattr(well_datum, "absolute_elevation_m", None))
            ):
                reason = "unresolved_non_gl_well_vertical_datum"
        reference = None
        if not reason:
            reference = aligner.match(float(head.x), float(head.y))
            if reference is None:
                reason = "well_outside_seismic_geometry"
            elif float(reference.distance) > max_horizontal_distance:
                reason = "well_exceeds_max_horizontal_distance"
        trajectory_distances: list[float] = []
        trajectory_trace_indices: list[int] = []
        if not reason:
            assert head is not None and trajectory is not None and reference is not None
            if trajectory.x is not None and trajectory.y is not None:
                trajectory_x = np.asarray(trajectory.x, dtype=float)
                trajectory_y = np.asarray(trajectory.y, dtype=float)
            else:
                trajectory_x = float(head.x) + np.asarray(
                    trajectory.x_offset, dtype=float
                )
                trajectory_y = float(head.y) + np.asarray(
                    trajectory.y_offset, dtype=float
                )
            for station_x, station_y in zip(trajectory_x, trajectory_y):
                station_reference = aligner.match(
                    float(station_x),
                    float(station_y),
                    asset=reference.asset,
                )
                if station_reference is None:
                    reason = "trajectory_outside_seismic_geometry"
                    break
                trajectory_distances.append(float(station_reference.distance))
                station_trace_index = getattr(station_reference, "trace_index", None)
                if (
                    not isinstance(station_trace_index, bool)
                    and isinstance(station_trace_index, (int, np.integer))
                    and int(station_trace_index) >= 0
                ):
                    trajectory_trace_indices.append(int(station_trace_index))
            if (
                trajectory_distances
                and max(trajectory_distances) > max_horizontal_distance
            ):
                reason = "trajectory_exceeds_max_horizontal_distance"
        if reason:
            skipped.append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "reason": reason,
                }
            )
            continue
        assert head is not None and trajectory is not None and acoustic is not None
        assert reference is not None
        assert well_datum is not None
        eligible.append(
            {
                "entity": entity,
                "head": head,
                "trajectory": trajectory,
                "acoustic": acoustic,
                "surface_elevation_m": float(well_datum.absolute_elevation_m),
                "well_depth_datum": str(well_datum.datum),
                "asset": reference.asset,
                "reader": reference.reader,
                "_spatial_aligner": aligner,
                "surface_trace_index": (
                    int(reference.trace_index)
                    if not isinstance(getattr(reference, "trace_index", None), bool)
                    and isinstance(
                        getattr(reference, "trace_index", None), (int, np.integer)
                    )
                    and int(reference.trace_index) >= 0
                    else None
                ),
                "trajectory_trace_indices": tuple(trajectory_trace_indices),
                "nearest_trace_distance_m": float(reference.distance),
                "trajectory_p95_trace_distance_m": float(
                    np.quantile(trajectory_distances, 0.95)
                ),
                "trajectory_max_trace_distance_m": float(max(trajectory_distances)),
            }
        )
    return eligible, skipped


def _balanced_applicability_trace_indices(
    eligible: list[dict[str, Any]],
    *,
    reader: Any,
    maximum_traces: int = 16,
) -> np.ndarray:
    """Select bounded seismic traces beside eligible wells without labels."""

    if maximum_traces < 1:
        raise ValueError("maximum_traces must be positive")
    candidates_by_well: list[np.ndarray] = []
    for item in eligible:
        if item.get("reader") is not reader:
            continue
        candidates: list[int] = []
        surface = item.get("surface_trace_index")
        if (
            not isinstance(surface, bool)
            and isinstance(surface, (int, np.integer))
            and int(surface) >= 0
        ):
            candidates.append(int(surface))
        for value in item.get("trajectory_trace_indices") or ():
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, np.integer))
                and int(value) >= 0
            ):
                candidates.append(int(value))
        unique = np.unique(np.asarray(candidates, dtype=np.int64))
        if unique.size:
            candidates_by_well.append(unique)
    if not candidates_by_well:
        return np.asarray([], dtype=np.int64)

    selected_well_positions = np.unique(
        np.linspace(
            0,
            len(candidates_by_well) - 1,
            min(maximum_traces, len(candidates_by_well)),
            dtype=np.int64,
        )
    )
    selected_wells = [candidates_by_well[int(pos)] for pos in selected_well_positions]
    base, remainder = divmod(maximum_traces, len(selected_wells))
    selected: list[int] = []
    for index, candidates in enumerate(selected_wells):
        budget = base + (1 if index < remainder else 0)
        positions = np.unique(
            np.linspace(
                0,
                len(candidates) - 1,
                min(budget, len(candidates)),
                dtype=np.int64,
            )
        )
        selected.extend(int(candidates[int(pos)]) for pos in positions)

    chosen = np.unique(np.asarray(selected, dtype=np.int64))
    if chosen.size < maximum_traces:
        all_candidates = np.unique(np.concatenate(candidates_by_well))
        remaining = np.setdiff1d(all_candidates, chosen, assume_unique=True)
        if remaining.size:
            positions = np.unique(
                np.linspace(
                    0,
                    len(remaining) - 1,
                    min(maximum_traces - len(chosen), len(remaining)),
                    dtype=np.int64,
                )
            )
            chosen = np.unique(np.concatenate((chosen, remaining[positions]))).astype(
                np.int64, copy=False
            )
    return chosen[:maximum_traces]


def _observe_p13_seismic(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile the seismic support consumed beside eligible well trajectories."""

    sealed_support = [item.get("trace_support") for item in eligible]
    if any(sealed_support):
        if not all(isinstance(item, Mapping) for item in sealed_support):
            raise ValueError(
                "P13 sealed trace support is incomplete across eligible wells"
            )
        readers = {id(item["reader"]): item["reader"] for item in eligible}
        if len(readers) != 1:
            raise ValueError(
                "P13 multiple SEG-Y assets require separate applicability decisions"
            )
        reader = next(iter(readers.values()))
        trace_indices = np.unique(
            np.concatenate(
                [
                    np.asarray(item["trace_support"]["trace_indices"], dtype=np.int64)
                    for item in eligible
                ]
            )
        )
        if trace_indices.size < 1:
            raise ValueError("P13 sealed trace support contains no source traces")
        profile_trace_indices = _balanced_applicability_trace_indices(
            [
                {
                    "reader": item["reader"],
                    "trajectory_trace_indices": tuple(
                        np.asarray(
                            item["trace_support"]["trace_indices"], dtype=np.int64
                        ).tolist()
                    ),
                }
                for item in eligible
            ],
            reader=reader,
            maximum_traces=MAXIMUM_APPLICABILITY_PROFILE_TRACES,
        )
        if profile_trace_indices.size < 1:
            raise ValueError("P13 sealed trace support contains no profileable traces")
        observations = observe_seismic_reader(
            reader,
            maximum_traces=int(profile_trace_indices.size),
            trace_indices=profile_trace_indices,
        )
        geometry = reader.geometry or reader.inspect()
        support_receipts = sorted(
            str(item["trace_support"]["receipt_sha256"]) for item in eligible
        )
        observations.update(
            {
                "seismic.observation_scope": "sealed_p13_consumed_trace_support",
                "seismic.observation_trace_count": int(trace_indices.size),
                "seismic.survey_trace_count": int(geometry.trace_count),
                "seismic.observation_well_count": len(eligible),
                "seismic.observation_trace_indices_sha256": hashlib.sha256(
                    np.asarray(trace_indices, dtype="<i8").tobytes()
                ).hexdigest(),
                "seismic.trace_support_receipts_sha256": hashlib.sha256(
                    "\n".join(support_receipts).encode("ascii")
                ).hexdigest(),
            }
        )
        return observations

    reader = eligible[0]["reader"]
    trace_indices = _balanced_applicability_trace_indices(
        eligible, reader=reader, maximum_traces=16
    )
    if not trace_indices.size:
        observations = observe_seismic_reader(reader)
        observations["seismic.observation_scope"] = "survey_evenly_spaced_fallback"
        return observations

    observations = observe_seismic_reader(
        reader,
        maximum_traces=int(trace_indices.size),
        trace_indices=trace_indices,
    )
    geometry = reader.geometry or reader.inspect()
    observations.update(
        {
            "seismic.observation_scope": ("eligible_well_trajectory_nearest_traces"),
            "seismic.observation_trace_count": int(trace_indices.size),
            "seismic.survey_trace_count": int(geometry.trace_count),
            "seismic.observation_well_count": int(
                sum(item.get("reader") is reader for item in eligible)
            ),
            "seismic.observation_trace_indices_sha256": hashlib.sha256(
                np.asarray(trace_indices, dtype="<i8").tobytes()
            ).hexdigest(),
        }
    )
    return observations


def _prepare_p13_request(
    pipeline: Any,
    *,
    item: dict[str, Any],
    cache_directory: Path,
    output_directory: Path,
    coordinate_reference: str,
    source_snapshot_context: Mapping[str, Any] | None,
    trace_support_aligner: Any | None,
) -> dict[str, Any]:
    """Serialize one label-free request and its sealed trace support."""

    entity = item["entity"]
    safe_name = _safe_directory_name(entity.well_uid)
    request_path = cache_directory / f"{safe_name}_request.json"
    candidate_root = output_directory / "p13_candidates" / safe_name
    acoustic_path, trajectory_path, trajectory_input_audit = (
        _write_canonical_well_inputs(
            cache_directory / "canonical_inputs",
            entity=entity,
            head=item["head"],
            log=item["acoustic"],
            trajectory=item["trajectory"],
        )
    )
    tie_las_path, tie_curve_input = _optional_full_las_path(item["acoustic"])
    coordinate_fields = _coordinate_fields(pipeline, item["asset"], item["reader"])
    request: dict[str, Any] = {
        "schema": "wellfuse.p13_unknown_survey_request.v1",
        "seismic_path": str(Path(item["asset"].path).resolve()),
        "well": {
            "well_id": entity.canonical_name,
            "acoustic_path": str(acoustic_path.resolve()),
            "trajectory_path": str(trajectory_path.resolve()),
            "surface_x_m": float(item["head"].x),
            "surface_y_m": float(item["head"].y),
            "surface_elevation_m": float(item["surface_elevation_m"]),
            "tie_curve_mode": tie_curve_input["mode"],
        },
        "cache_root": str((cache_directory / safe_name).resolve()),
        "output_root": str(candidate_root.resolve()),
        "coordinate_fields": coordinate_fields,
        "coordinate_reference": str(coordinate_reference),
        "vertical_datum": (
            f"canonical MSL elevation from resolved {item['well_depth_datum']}"
        ),
        "tie_curve_input_audit": tie_curve_input,
        "trajectory_input_audit": trajectory_input_audit,
    }
    trace_support = None
    if source_snapshot_context is not None:
        if trace_support_aligner is None:
            raise ValueError("sealed P13 trace-support aligner is unavailable")
        trace_support = write_p13_trace_support(
            cache_directory / "trace_support",
            pipeline=pipeline,
            aligner=trace_support_aligner,
            item=item,
            acoustic_path=acoustic_path,
            trajectory_path=trajectory_path,
            coordinate_fields=coordinate_fields,
            source_snapshot_context=source_snapshot_context,
        )
        item["trace_support"] = trace_support
        request["requires_sealed_trace_support"] = True
        request["seismic_trace_support"] = dict(trace_support["request"])
    if tie_las_path is not None:
        request["well"]["tie_las_path"] = str(tie_las_path)
    _assert_label_free_request(request)
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "item": item,
        "entity": entity,
        "request_path": request_path.resolve(),
        "candidate_root": candidate_root.resolve(),
        "trace_support": trace_support,
        "trajectory_input_audit": trajectory_input_audit,
    }


def _load_candidate(
    path: Path,
    *,
    entity: Any,
    trajectory: Any,
    head: Any,
    surface_elevation_m: float | None = None,
    expected_trace_support_receipt_sha256: str | None = None,
    expected_trace_support_identity_sha256: str | None = None,
    expected_applicability_manifest_sha256: str | None = None,
    expected_applicability_decision_sha256: str | None = None,
    trajectory_input_audit: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    csv_path = path / "time_depth.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "wellfuse.align.unknown-survey-candidate.v1":
        raise ValueError("P13候选manifest schema不匹配")
    if expected_trace_support_receipt_sha256 is not None:
        if (
            str(manifest.get("consumed_trace_support_receipt_sha256") or "").casefold()
            != str(expected_trace_support_receipt_sha256).casefold()
            or str(
                manifest.get("consumed_trace_support_identity_sha256") or ""
            ).casefold()
            != str(expected_trace_support_identity_sha256 or "").casefold()
        ):
            raise ValueError("P13候选没有消费当前请求封存的地震道支持")
    applicability_expectations = (
        expected_applicability_manifest_sha256,
        expected_applicability_decision_sha256,
    )
    if any(value is not None for value in applicability_expectations):
        if not all(value is not None for value in applicability_expectations):
            raise ValueError("P13候选适用性证据预期不完整")
        if (
            str(manifest.get("consumed_applicability_manifest_sha256") or "").casefold()
            != str(expected_applicability_manifest_sha256).casefold()
            or str(
                manifest.get("consumed_applicability_decision_sha256") or ""
            ).casefold()
            != str(expected_applicability_decision_sha256).casefold()
        ):
            raise ValueError("P13候选没有消费当前分组验签的适用性证据")
    registration_source = str(manifest.get("registration_source") or "")
    allowed_sources = {
        "wellfuse_align_p13_prediction",
        "wellfuse_align_p13_factorized_v2",
        "wellfuse_align_p13_scientific_incumbent",
        "wellfuse_align_p13_factorized_v3",
    }
    if registration_source not in allowed_sources:
        raise ValueError("P13候选registration_source不匹配")
    if manifest.get("p13_checkpoint_executed") is not True:
        raise ValueError("P13候选没有实际执行checkpoint")
    if (
        registration_source == "wellfuse_align_p13_factorized_v2"
        and manifest.get("factorized_v2_checkpoint_executed") is not True
    ):
        raise ValueError("P13-Factorized候选没有实际执行第二阶段checkpoint")

    if (
        registration_source == "wellfuse_align_p13_scientific_incumbent"
        and not manifest.get("scientific_fallback")
    ):
        raise ValueError(
            "P13 scientific incumbent candidate is missing its frozen identity"
        )
    if registration_source == "wellfuse_align_p13_factorized_v3":
        if manifest.get("factorized_v3_checkpoint_executed") is not True:
            raise ValueError("P13-Factorized-v3 checkpoint was not executed")
        if manifest.get("factorized_v3_model_forward_executed") is not True:
            raise ValueError("P13-Factorized-v3 model forward was not executed")
        if manifest.get("factorized_v3_accepted") is not True:
            raise ValueError(
                "P13-Factorized-v3 runtime gate did not accept the product"
            )

    values = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if tuple(values.dtype.names or ()) != _OUTPUT_COLUMNS:
        raise ValueError("P13候选CSV列合同不匹配")
    columns = {
        name: np.atleast_1d(np.asarray(values[name], dtype=float))
        for name in _OUTPUT_COLUMNS
    }
    samples = columns["MD"].size
    if samples < 2 or any(column.size != samples for column in columns.values()):
        raise ValueError("P13候选样点不足或列长度不一致")
    if not all(np.all(np.isfinite(column)) for column in columns.values()):
        raise ValueError("P13候选包含NaN/Inf")
    if bool((np.diff(columns["MD"]) <= 0.0).any()) or bool(
        (np.diff(columns["TWT_mean"]) <= 0.0).any()
    ):
        raise ValueError("P13候选MD/TWT必须严格递增")
    if bool((columns["TWT_std"] < 0.0).any()) or bool(
        ((columns["quality"] < 0.0) | (columns["quality"] > 1.0)).any()
    ):
        raise ValueError("P13候选不确定性或质量越界")

    source_md = np.asarray(trajectory.md, dtype=float)
    if bool(
        (columns["MD"] < source_md[0] - 1e-6).any()
        or (columns["MD"] > source_md[-1] + 1e-6).any()
    ):
        raise ValueError("P13 candidate MD lies outside the measured trajectory")
    if trajectory.x is not None and trajectory.y is not None:
        source_x = np.asarray(trajectory.x, dtype=float)
        source_y = np.asarray(trajectory.y, dtype=float)
        source_x_offset = source_x - float(head.x)
        source_y_offset = source_y - float(head.y)
    else:
        # Preserve the authoritative offset arrays for the audit identity.
        # Reconstructing absolute XY and subtracting a large well-head coordinate
        # changes low-order float bits, so an offsets-only trajectory would fail
        # its own SHA-256 audit even though its physical geometry is unchanged.
        source_x_offset = np.asarray(trajectory.x_offset, dtype=float)
        source_y_offset = np.asarray(trajectory.y_offset, dtype=float)
        source_x = float(head.x) + source_x_offset
        source_y = float(head.y) + source_y_offset
    expected_x = np.interp(columns["MD"], source_md, source_x)
    expected_y = np.interp(columns["MD"], source_md, source_y)
    coordinate_gap = np.hypot(columns["X"] - expected_x, columns["Y"] - expected_y)
    if float(np.max(coordinate_gap)) > 2.0:
        raise ValueError("P13候选坐标与当前快照轨迹不一致")
    source_tvd = np.asarray(trajectory.tvd, dtype=float)
    expected_tvd = np.interp(columns["MD"], source_md, source_tvd)
    candidate_validation_tvd = expected_tvd
    runtime_proxy_applied = False
    if trajectory_input_audit is not None:
        audit = dict(trajectory_input_audit)
        if (
            audit.get("schema") != "well-seismic.p13-canonical-trajectory.v1"
            or audit.get("runtime_tvd_policy")
            != "monotonic_envelope_for_frozen_p13_only"
            or audit.get("original_platform_trajectory_preserved") is not True
            or audit.get("md_xy_preserved") is not True
        ):
            raise ValueError("P13 canonical trajectory audit is incompatible")
        original_geometry = np.column_stack(
            (
                source_md,
                source_tvd,
                source_x_offset,
                source_y_offset,
            )
        )
        original_geometry = original_geometry[
            np.all(np.isfinite(original_geometry), axis=1)
        ]
        original_geometry = original_geometry[np.argsort(original_geometry[:, 0])]
        _, unique = np.unique(original_geometry[:, 0], return_index=True)
        original_geometry = original_geometry[np.sort(unique)]
        runtime_geometry = original_geometry.copy()
        runtime_geometry[:, 1] = np.maximum.accumulate(runtime_geometry[:, 1])
        correction = runtime_geometry[:, 1] - original_geometry[:, 1]
        observed_original_sha256 = hashlib.sha256(
            np.asarray(original_geometry, dtype="<f8").tobytes()
        ).hexdigest()
        observed_runtime_sha256 = hashlib.sha256(
            np.asarray(runtime_geometry, dtype="<f8").tobytes()
        ).hexdigest()
        if (
            str(audit.get("original_geometry_sha256") or "").casefold()
            != observed_original_sha256
            or str(audit.get("runtime_geometry_sha256") or "").casefold()
            != observed_runtime_sha256
            or int(audit.get("station_count", -1)) != original_geometry.shape[0]
            or int(audit.get("negative_tvd_interval_count", -1))
            != int(np.count_nonzero(np.diff(original_geometry[:, 1]) < -1e-5))
            or int(audit.get("corrected_station_count", -1))
            != int(np.count_nonzero(correction > 1e-8))
            or not math.isclose(
                float(audit.get("maximum_tvd_correction_m", float("nan"))),
                float(np.max(correction) if correction.size else 0.0),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("P13 canonical trajectory audit does not match this well")
        runtime_proxy_applied = bool(np.any(correction > 1e-8))
        if runtime_proxy_applied:
            candidate_validation_tvd = np.interp(
                columns["MD"], runtime_geometry[:, 0], runtime_geometry[:, 1]
            )
    surface_elevation = (
        surface_elevation_m
        if _finite(surface_elevation_m)
        else head.kb
        if _finite(head.kb)
        else None
    )
    if not _finite(surface_elevation):
        raise ValueError(
            "P13 candidate validation requires a canonical non-GL well datum"
        )
    expected_tvdss = float(surface_elevation) - candidate_validation_tvd
    if not (
        np.allclose(columns["TVDSS"], expected_tvdss, rtol=0.0, atol=1e-3)
        and np.allclose(columns["Z"], expected_tvdss, rtol=0.0, atol=1e-3)
    ):
        raise ValueError(
            "P13 candidate TVDSS/Z is inconsistent with trajectory and datum"
        )

    accepted = manifest.get("p13_raw_candidate_accepted") is True
    rejection_reason = manifest.get("rejection_reason")
    physics_sanity = (manifest.get("execution") or {}).get("physics_sanity") or {}
    physics_passed = physics_sanity.get("passed")
    physics_reason = physics_sanity.get("rejection_reason")
    if not isinstance(physics_passed, bool) or accepted is not physics_passed:
        raise ValueError("P13 candidate acceptance conflicts with physics_sanity")
    if accepted:
        if rejection_reason is not None or physics_reason is not None:
            raise ValueError(
                "accepted P13 candidate unexpectedly has a rejection reason"
            )
    elif not rejection_reason or str(rejection_reason) != str(physics_reason):
        raise ValueError("rejected P13 candidate has inconsistent rejection evidence")
    model_gate = manifest.get("model_execution_gate") or {}
    if model_gate and model_gate.get("passed") is not True:
        raise ValueError("P13 candidate model execution gate did not pass")
    execution_ready = bool(
        registration_source in allowed_sources
        and (
            registration_source != "wellfuse_align_p13_factorized_v2"
            or manifest.get("factorized_v2_checkpoint_executed") is True
        )
        and (
            registration_source != "wellfuse_align_p13_factorized_v3"
            or (
                manifest.get("factorized_v3_checkpoint_executed") is True
                and manifest.get("factorized_v3_model_forward_executed") is True
                and manifest.get("factorized_v3_accepted") is True
            )
        )
    )
    inference_ready = bool(accepted and execution_ready)
    declared_inference_ready = manifest.get("inference_ready")
    if (
        declared_inference_ready is not None
        and bool(declared_inference_ready) != inference_ready
    ):
        raise ValueError(
            "P13 candidate inference_ready conflicts with physics/model gates"
        )

    # ``alignment_qc`` is deliberately an optional, product-side diagnostic.
    # The frozen P13 predictions remain the source of the TWT curve; QC must
    # never rewrite it.  A QC result may become a downstream hard gate only
    # when its separate scientific gate has passed.  Older candidate products
    # predate this contract and therefore remain loadable without it.
    alignment_qc = manifest.get("alignment_qc")
    qc_policy: str | None = None
    qc_scientific_gate_passed: bool | None = None
    qc_hard_gate_active = False
    if alignment_qc is not None:
        if not isinstance(alignment_qc, dict):
            raise ValueError("P13 candidate alignment_qc must be an object")
        required_qc_fields = {
            "prediction_rewritten",
            "labels_opened",
            "accepted",
            "reasons",
        }
        missing_qc_fields = sorted(required_qc_fields - set(alignment_qc))
        if missing_qc_fields:
            raise ValueError(
                "P13 candidate alignment_qc is missing required fields: "
                + ", ".join(missing_qc_fields)
            )
        if alignment_qc["prediction_rewritten"] is not False:
            raise ValueError("P13 alignment_qc must not rewrite the prediction")
        if alignment_qc["labels_opened"] is not False:
            raise ValueError("P13 alignment_qc must not open labels")
        if type(alignment_qc["accepted"]) is not bool:
            raise ValueError("P13 alignment_qc.accepted must be a bool")
        if not isinstance(alignment_qc["reasons"], list):
            raise ValueError("P13 alignment_qc.reasons must be a list")

        declared_policy = alignment_qc.get("policy")
        declared_activation = alignment_qc.get("activation")
        if declared_policy is None and declared_activation is None:
            raise ValueError(
                "P13 alignment_qc must explicitly declare policy or activation"
            )
        if (
            declared_policy is not None
            and declared_activation is not None
            and declared_policy != declared_activation
        ):
            raise ValueError(
                "P13 alignment_qc.policy and activation must agree when both are present"
            )
        qc_policy = str(
            declared_policy if declared_policy is not None else declared_activation
        )
        if qc_policy not in {"diagnostics_only", "hard_gate"}:
            raise ValueError(
                "P13 alignment_qc policy must be diagnostics_only or hard_gate"
            )
        qc_scientific_gate_passed = alignment_qc.get("scientific_gate_passed", False)
        if type(qc_scientific_gate_passed) is not bool:
            raise ValueError("P13 alignment_qc.scientific_gate_passed must be a bool")
        qc_hard_gate_active = bool(
            qc_policy == "hard_gate" and qc_scientific_gate_passed
        )

    declared_fusion_ready = manifest.get("fusion_ready", False)
    if (
        declared_fusion_ready is True
        and qc_hard_gate_active
        and alignment_qc is not None
        and alignment_qc["accepted"] is not True
    ):
        raise ValueError(
            "P13 candidate fusion_ready requires accepted alignment_qc when its hard gate is active"
        )
    fusion_ready = bool(declared_fusion_ready and inference_ready)
    if (
        manifest.get("training_label_eligible") is True
        or manifest.get("supervision_eligible") is True
    ):
        raise ValueError("unknown-survey P13 inference cannot become supervision")
    tvd = np.interp(columns["MD"], source_md, np.asarray(trajectory.tvd, dtype=float))
    track = {
        "well_name": entity.canonical_name,
        "well_uid": entity.well_uid,
        "md": columns["MD"].round(8).tolist(),
        "tvd": tvd.round(8).tolist(),
        "x": expected_x.round(8).tolist(),
        "y": expected_y.round(8).tolist(),
        "twtMean": columns["TWT_mean"].round(8).tolist(),
        "twtStd": columns["TWT_std"].round(8).tolist(),
        "registrationQuality": columns["quality"].round(8).tolist(),
        "registrationSource": registration_source,
        "registrationStatus": (
            "experimental_fusion_ready"
            if fusion_ready
            else "experimental_inference_candidate"
        ),
        "registrationCoverage": 1.0,
        "inferenceEligible": inference_ready,
        "fusionReady": fusion_ready,
        "supervisionEligible": False,
        "trainingEligible": False,
        "diagnostics": {
            "candidate_manifest": str(manifest_path.resolve()),
            "uncertainty_definition": manifest.get("uncertainty_definition"),
            "raw_candidate_accepted": accepted,
            "rejection_reason": rejection_reason,
            "physics_sanity": physics_sanity,
            "factorized_v2": (manifest.get("execution") or {}).get("factorized_v2"),
            "factorized_v3": (manifest.get("execution") or {}).get("factorized_v3"),
            "inference_ready": inference_ready,
            "fusion_ready": fusion_ready,
            "alignment_qc": alignment_qc,
            "alignment_qc_accepted": (
                bool(alignment_qc["accepted"]) if alignment_qc is not None else None
            ),
            "alignment_qc_policy": qc_policy,
            "alignment_qc_scientific_gate_passed": qc_scientific_gate_passed,
            "alignment_qc_hard_gate_active": qc_hard_gate_active,
            "alignment_qc_reasons": (
                list(alignment_qc["reasons"]) if alignment_qc is not None else []
            ),
            "training_label_eligible": False,
            "fusion_semantics": (
                "experimental inference evidence; never time-depth supervision"
            ),
            "trajectory_input_audit": (
                dict(trajectory_input_audit)
                if trajectory_input_audit is not None
                else None
            ),
            "runtime_tvd_proxy_applied": runtime_proxy_applied,
            "candidate_geometry_mapped_to_original_trajectory": True,
            "consumed_trace_support_receipt_sha256": manifest.get(
                "consumed_trace_support_receipt_sha256"
            ),
            "consumed_trace_support_identity_sha256": manifest.get(
                "consumed_trace_support_identity_sha256"
            ),
        },
    }
    validation = {
        "registration_source": registration_source,
        "factorized_v2_checkpoint_executed": bool(
            manifest.get("factorized_v2_checkpoint_executed")
        ),
        "raw_candidate_accepted": accepted,
        "rejection_reason": rejection_reason,
        "physics_sanity": physics_sanity,
        "inference_ready": inference_ready,
        "fusion_ready": fusion_ready,
        "alignment_qc": alignment_qc,
        "alignment_qc_accepted": (
            bool(alignment_qc["accepted"]) if alignment_qc is not None else None
        ),
        "alignment_qc_reasons": (
            list(alignment_qc["reasons"]) if alignment_qc is not None else []
        ),
        "alignment_qc_policy": qc_policy,
        "alignment_qc_scientific_gate_passed": qc_scientific_gate_passed,
        "alignment_qc_hard_gate_active": qc_hard_gate_active,
        "training_label_eligible": False,
        "runtime_tvd_proxy_applied": runtime_proxy_applied,
        "candidate_geometry_mapped_to_original_trajectory": True,
        "consumed_trace_support_receipt_sha256": manifest.get(
            "consumed_trace_support_receipt_sha256"
        ),
        "consumed_trace_support_identity_sha256": manifest.get(
            "consumed_trace_support_identity_sha256"
        ),
        "consumed_applicability_manifest_sha256": manifest.get(
            "consumed_applicability_manifest_sha256"
        ),
        "consumed_applicability_decision_sha256": manifest.get(
            "consumed_applicability_decision_sha256"
        ),
        "manifest": str(manifest_path.resolve()),
        "time_depth_csv": str(csv_path.resolve()),
        "time_depth_las": (
            str((path / "time_depth.las").resolve())
            if (path / "time_depth.las").is_file()
            else None
        ),
        "sample_count": samples,
        "median_quality": float(np.median(columns["quality"])),
        "median_twt_std_ms": float(np.median(columns["TWT_std"])),
        "twt_range_ms": [float(columns["TWT_mean"][0]), float(columns["TWT_mean"][-1])],
    }
    return track, validation


def _run_p13_registration_candidates_legacy(
    pipeline: Any,
    *,
    project_root: Path,
    output_directory: Path,
    cache_directory: Path,
    progress: Any = None,
) -> dict[str, Any]:
    """Run P13 for every well satisfying the frozen runtime input contract."""

    eligible, skipped = _eligible_wells(pipeline)
    result: dict[str, Any] = {
        "attempted": bool(eligible),
        "checkpoint_executed": False,
        "eligible_well_count": len(eligible),
        "accepted_well_count": 0,
        "executed_well_count": 0,
        "skipped": skipped,
        "failures": [],
        "records": [],
        "tracks": {},
        "accepted_well_uids": [],
    }
    if not eligible:
        result["runtime_status"] = "not_eligible"
        return result

    try:
        wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        result["runtime_status"] = "runtime_unavailable"
        result["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        return result
    output_directory.mkdir(parents=True, exist_ok=True)
    observations: dict[str, Any] = {}
    observation_error: str | None = None
    try:
        observations.update(_observe_p13_seismic(eligible))
        observations.update(observe_align_wells(eligible))
    except Exception as exc:  # noqa: BLE001 - a missing required observation abstains safely
        observation_error = f"{type(exc).__name__}: {exc}"
    envelope, envelope_path = resolve_training_envelope(
        wellfuse_root, "wellfuse_align_p13"
    )
    applicability = evaluate_applicability(
        envelope, observations, model_id="wellfuse_align_p13"
    )
    if observation_error:
        applicability["issues"].append(
            f"label_free_observation_error:{observation_error}"
        )
    applicability_manifest = write_applicability_manifest(
        output_directory / "p13_applicability.json",
        applicability,
        envelope_path=envelope_path,
    )
    result["applicability"] = applicability
    result["applicability_manifest"] = str(applicability_manifest)
    if applicability["decision"] == "do_not_execute_model":
        result["runtime_status"] = "applicability_abstained_physics_fallback"
        result["applicability_route"] = applicability["route"]
        return result
    wrapper = wellfuse_root / "scripts" / "infer_p13_unknown_survey.py"
    if not wrapper.is_file():
        result["runtime_status"] = "runner_missing"
        result["failures"].append({"error": f"P13 runner not found: {wrapper}"})
        return result

    cache_directory.mkdir(parents=True, exist_ok=True)
    coordinate_reference = (
        pipeline.config.get("matching", {}).get("coordinate_reference", {}).get("crs")
        or "unknown_projected_crs"
    )
    for index, item in enumerate(eligible, start=1):
        entity = item["entity"]
        safe_name = _safe_directory_name(entity.well_uid)
        request_path = cache_directory / f"{safe_name}_request.json"
        candidate_root = output_directory / "p13_candidates" / safe_name
        runtime_log = output_directory / "p13_candidates" / f"{safe_name}_runtime.log"
        try:
            acoustic_path, trajectory_path, trajectory_input_audit = (
                _write_canonical_well_inputs(
                    cache_directory / "canonical_inputs",
                    entity=entity,
                    head=item["head"],
                    log=item["acoustic"],
                    trajectory=item["trajectory"],
                )
            )
            tie_las_path, tie_curve_input = _optional_full_las_path(item["acoustic"])
            request = {
                "schema": "wellfuse.p13_unknown_survey_request.v1",
                "seismic_path": str(Path(item["asset"].path).resolve()),
                "well": {
                    "well_id": entity.canonical_name,
                    "acoustic_path": str(acoustic_path.resolve()),
                    "trajectory_path": str(trajectory_path.resolve()),
                    "surface_x_m": float(item["head"].x),
                    "surface_y_m": float(item["head"].y),
                    "surface_elevation_m": float(item["surface_elevation_m"]),
                    "tie_curve_mode": tie_curve_input["mode"],
                },
                "cache_root": str((cache_directory / safe_name).resolve()),
                "output_root": str(candidate_root.resolve()),
                "coordinate_fields": _coordinate_fields(
                    pipeline, item["asset"], item["reader"]
                ),
                "coordinate_reference": str(coordinate_reference),
                "vertical_datum": (
                    f"canonical MSL elevation from resolved {item['well_depth_datum']}"
                ),
                "tie_curve_input_audit": tie_curve_input,
                "trajectory_input_audit": trajectory_input_audit,
            }
            if tie_las_path is not None:
                request["well"]["tie_las_path"] = str(tie_las_path)
            _assert_label_free_request(request)
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if progress:
                progress(
                    30 + int(24 * (index - 1) / max(len(eligible), 1)),
                    f"正在执行冻结概率标定集成：{entity.canonical_name} ({index}/{len(eligible)})",
                )
            command = [
                str(python_executable),
                str(wrapper),
                "--request",
                str(request_path.resolve()),
            ]
            completed = managed_run(
                command,
                cwd=wellfuse_root,
                env=_wellfuse_subprocess_environment(wellfuse_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            runtime_log.parent.mkdir(parents=True, exist_ok=True)
            runtime_log.write_text(
                (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
            )
            if completed.returncode:
                tail = "\n".join(
                    ((completed.stdout or "") + (completed.stderr or "")).splitlines()[
                        -20:
                    ]
                )
                raise RuntimeError(
                    f"P13 subprocess exited {completed.returncode}: {tail}"
                )
            # A zero exit from the frozen runner means its checkpoint ensemble
            # really executed.  Product validation is a separate decision and
            # must not erase that execution evidence.
            result["checkpoint_executed"] = True
            result["executed_well_count"] += 1
            track, validation = _load_candidate(
                candidate_root,
                entity=entity,
                trajectory=item["trajectory"],
                head=item["head"],
                surface_elevation_m=item["surface_elevation_m"],
                trajectory_input_audit=trajectory_input_audit,
            )
            record = {
                "well_uid": entity.well_uid,
                "well_name": entity.canonical_name,
                "request": str(request_path.resolve()),
                "runtime_log": str(runtime_log.resolve()),
                "nearest_trace_distance_m": item["nearest_trace_distance_m"],
                "trajectory_p95_trace_distance_m": item[
                    "trajectory_p95_trace_distance_m"
                ],
                "trajectory_max_trace_distance_m": item[
                    "trajectory_max_trace_distance_m"
                ],
                "trajectory_input_audit": trajectory_input_audit,
                **validation,
            }
            result["records"].append(record)
            if validation["inference_ready"]:
                result["tracks"][entity.canonical_name] = track
                result["accepted_well_uids"].append(entity.well_uid)
                result["accepted_well_count"] += 1
        except Exception as exc:  # noqa: BLE001 - isolate one vendor well/runtime
            result["failures"].append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": str(request_path.resolve()),
                    "runtime_log": str(runtime_log.resolve()),
                }
            )

    if result["accepted_well_count"]:
        result["runtime_status"] = "executed_candidate_accepted"
    elif result["checkpoint_executed"] and result["records"]:
        result["runtime_status"] = "executed_rejected_physics_fallback"
    elif result["checkpoint_executed"]:
        result["runtime_status"] = "executed_product_invalid_physics_fallback"
    elif result["failures"]:
        result["runtime_status"] = "execution_failed_physics_fallback"
    else:
        result["runtime_status"] = "not_eligible"
    return result


def run_p13_registration_candidates(
    pipeline: Any,
    *,
    project_root: Path,
    output_directory: Path,
    cache_directory: Path,
    source_snapshot_context: Mapping[str, Any] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Run P13 in independent SEG-Y applicability and inference groups."""

    eligible, skipped = _eligible_wells(pipeline)
    result: dict[str, Any] = {
        "attempted": bool(eligible),
        "checkpoint_executed": False,
        "eligible_well_count": len(eligible),
        "accepted_well_count": 0,
        "executed_well_count": 0,
        "skipped": skipped,
        "failures": [],
        "records": [],
        "tracks": {},
        "accepted_well_uids": [],
        "batch_execution": None,
        "applicability_groups": [],
    }
    if not eligible:
        result["runtime_status"] = "not_eligible"
        return result

    try:
        wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        result["runtime_status"] = "runtime_unavailable"
        result["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        return result
    output_directory.mkdir(parents=True, exist_ok=True)
    envelope, envelope_path = resolve_training_envelope(
        wellfuse_root, "wellfuse_align_p13"
    )

    def evaluate_groups(
        groups: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], set[int]]:
        accepted_groups: list[dict[str, Any]] = []
        accepted_entry_ids: set[int] = set()
        for index, group in enumerate(groups, start=1):
            group_entries = list(group["entries"])
            group_items = [dict(_p13_entry_item(entry)) for entry in group_entries]
            observations: dict[str, Any] = {}
            observation_error: str | None = None
            try:
                observations.update(_observe_p13_seismic(group_items))
                observations.update(observe_align_wells(group_items))
            except Exception as exc:  # noqa: BLE001 - one asset group abstains safely
                observation_error = f"{type(exc).__name__}: {exc}"
            applicability = evaluate_applicability(
                envelope,
                observations,
                model_id="wellfuse_align_p13",
            )
            if observation_error:
                applicability["issues"].append(
                    f"label_free_observation_error:{observation_error}"
                )
            manifest_path = (
                output_directory / "p13_applicability.json"
                if len(groups) == 1
                else output_directory
                / "p13_applicability"
                / f"{index:03d}_{group['group_id']}.json"
            )
            applicability_manifest = write_applicability_manifest(
                manifest_path,
                applicability,
                envelope_path=envelope_path,
            )
            evidence = _p13_applicability_evidence(
                applicability,
                manifest_path=applicability_manifest,
            )
            decision = str(applicability.get("decision") or "")
            permitted = decision in {"execute_model", "execute_with_warning"}
            if observation_error:
                permitted = False
            group_record = {
                "group_id": group["group_id"],
                "asset_path": group["asset_path"],
                "well_uids": list(group["well_uids"]),
                "applicability": applicability,
                "applicability_manifest": str(applicability_manifest),
                "applicability_manifest_sha256": evidence["manifest_sha256"],
                "applicability_decision_sha256": evidence["decision_sha256"],
                "observation_error": observation_error,
                "execute_model": permitted,
            }
            result["applicability_groups"].append(group_record)
            group["applicability_evidence"] = evidence
            group["applicability_record"] = group_record
            if permitted:
                accepted_groups.append(group)
                accepted_entry_ids.update(id(entry) for entry in group_entries)
                continue
            for entry in group_entries:
                item = _p13_entry_item(entry)
                entity = item.get("entity")
                result["skipped"].append(
                    {
                        "well_uid": getattr(entity, "well_uid", None),
                        "well_name": getattr(entity, "canonical_name", None),
                        "reason": "p13_asset_group_applicability_abstained",
                        "asset_path": group["asset_path"],
                        "applicability_manifest": str(applicability_manifest),
                        "applicability_decision": decision,
                        "observation_error": observation_error,
                    }
                )
        return accepted_groups, accepted_entry_ids

    legacy_evidence_by_item_id: dict[int, dict[str, str]] = {}
    if source_snapshot_context is None:
        legacy_groups = _group_p13_entries_by_seismic_asset(eligible)
        accepted_groups, accepted_ids = evaluate_groups(legacy_groups)
        for group in accepted_groups:
            evidence = dict(group["applicability_evidence"])
            for item in group["entries"]:
                legacy_evidence_by_item_id[id(item)] = evidence
        eligible = [item for item in eligible if id(item) in accepted_ids]
        if not eligible:
            first = result["applicability_groups"][0]
            result["applicability"] = first["applicability"]
            result["applicability_manifest"] = first["applicability_manifest"]
            result["runtime_status"] = "applicability_abstained_physics_fallback"
            result["applicability_route"] = first["applicability"].get("route")
            return result

    wrapper = wellfuse_root / "scripts" / "infer_p13_unknown_survey.py"
    if not wrapper.is_file():
        result["runtime_status"] = "runner_missing"
        result["failures"].append({"error": f"P13 runner not found: {wrapper}"})
        return result

    cache_directory.mkdir(parents=True, exist_ok=True)
    coordinate_reference = (
        pipeline.config.get("matching", {}).get("coordinate_reference", {}).get("crs")
        or "unknown_projected_crs"
    )
    trace_support_aligner = None
    if source_snapshot_context is not None:
        existing_aligner = eligible[0].get("_spatial_aligner")
        trace_support_aligner = (
            existing_aligner
            if int(getattr(existing_aligner, "neighbor_count", 0)) == 9
            else build_p13_trace_support_aligner(pipeline)
        )
    pending: list[dict[str, Any]] = []
    for item in eligible:
        entity = item["entity"]
        request_path = (
            cache_directory / f"{_safe_directory_name(entity.well_uid)}_request.json"
        )
        try:
            entry = _prepare_p13_request(
                pipeline,
                item=item,
                cache_directory=cache_directory,
                output_directory=output_directory,
                coordinate_reference=str(coordinate_reference),
                source_snapshot_context=source_snapshot_context,
                trace_support_aligner=trace_support_aligner,
            )
            if source_snapshot_context is None:
                evidence = legacy_evidence_by_item_id[id(item)]
                _bind_p13_request_applicability_evidence(
                    entry["request_path"], evidence
                )
                entry["applicability_evidence"] = evidence
            pending.append(entry)
        except Exception as exc:  # noqa: BLE001 - isolate one vendor well
            result["failures"].append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": str(request_path.resolve()),
                    "runtime_log": None,
                }
            )

    if not pending:
        result["runtime_status"] = "not_eligible"
        return result

    if source_snapshot_context is not None:
        sealed_groups = _group_p13_entries_by_seismic_asset(pending)
        accepted_groups, accepted_ids = evaluate_groups(sealed_groups)
        for group in accepted_groups:
            evidence = dict(group["applicability_evidence"])
            for entry in group["entries"]:
                _bind_p13_request_applicability_evidence(
                    entry["request_path"], evidence
                )
                entry["applicability_evidence"] = evidence
        pending = [entry for entry in pending if id(entry) in accepted_ids]
        if not pending:
            first = result["applicability_groups"][0]
            result["applicability"] = first["applicability"]
            result["applicability_manifest"] = first["applicability_manifest"]
            has_observation_error = any(
                group["observation_error"] for group in result["applicability_groups"]
            )
            result["runtime_status"] = (
                "applicability_observation_failed_physics_fallback"
                if has_observation_error
                else "applicability_abstained_physics_fallback"
            )
            result["applicability_route"] = "physics_or_abstain"
            return result

    if len(result["applicability_groups"]) == 1:
        only_group = result["applicability_groups"][0]
        result["applicability"] = only_group["applicability"]
        result["applicability_manifest"] = only_group["applicability_manifest"]
    else:
        result["applicability"] = {
            "schema_version": "well-seismic.p13-grouped-applicability.v1",
            "group_count": len(result["applicability_groups"]),
            "groups": result["applicability_groups"],
        }
        result["applicability_manifest"] = None
    result["applicability_manifests"] = [
        group["applicability_manifest"] for group in result["applicability_groups"]
    ]

    execution_groups = _group_p13_entries_by_seismic_asset(pending)
    microbatch_size = int(
        pipeline.config.get("registration", {}).get("p13_microbatch_size", 8)
    )
    if microbatch_size < 1:
        raise ValueError("registration.p13_microbatch_size must be positive")
    batch_summaries: list[dict[str, Any]] = []
    numeric_batch_fields = (
        "request_count",
        "completed_count",
        "failed_count",
        "sfm_load_count",
        "p13_checkpoint_load_count",
        "legacy_p13_checkpoint_load_count",
        "coordinate_index_count",
        "trace_support_load_count",
        "trace_support_validation_count",
    )
    for group_index, group in enumerate(execution_groups, start=1):
        group_entries = list(group["entries"])
        suffix = (
            ""
            if len(execution_groups) == 1
            else f"_{group_index:03d}_{group['group_id']}"
        )
        batch_request_path = cache_directory / f"p13_batch_request{suffix}.json"
        batch_runtime_log = (
            output_directory / "p13_candidates" / f"batch_runtime{suffix}.log"
        )
        batch_request_path.write_text(
            json.dumps(
                {
                    "schema": "wellfuse.p13_unknown_survey_batch_request.v1",
                    "requests": [str(entry["request_path"]) for entry in group_entries],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if progress:
            progress(
                30 + int(20 * (group_index - 1) / max(len(execution_groups), 1)),
                (
                    "正在按SEG-Y分组执行冻结概率标定集成："
                    f"{group_index}/{len(execution_groups)}组，{len(group_entries)}口井"
                ),
            )
        completed = managed_run(
            [
                str(python_executable),
                str(wrapper),
                "--batch-request",
                str(batch_request_path.resolve()),
                "--microbatch-size",
                str(microbatch_size),
            ],
            cwd=wellfuse_root,
            env=_wellfuse_subprocess_environment(wellfuse_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        batch_runtime_log.parent.mkdir(parents=True, exist_ok=True)
        batch_runtime_log.write_text(
            (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
        )
        output_lines = [
            line.strip()
            for line in (completed.stdout or "").splitlines()
            if line.strip()
        ]
        try:
            batch_result = (
                json.loads(output_lines[-1])
                if not completed.returncode and output_lines
                else None
            )
        except json.JSONDecodeError:
            batch_result = None
        if not isinstance(batch_result, dict):
            tail = "\n".join(
                ((completed.stdout or "") + (completed.stderr or "")).splitlines()[-20:]
            )
            for entry in group_entries:
                entity = entry["entity"]
                result["failures"].append(
                    {
                        "well_uid": entity.well_uid,
                        "well_name": entity.canonical_name,
                        "error": (
                            f"P13 batch subprocess exited {completed.returncode}: {tail}"
                            if completed.returncode
                            else "P13 batch subprocess returned no valid result manifest"
                        ),
                        "request": str(entry["request_path"]),
                        "runtime_log": str(batch_runtime_log.resolve()),
                    }
                )
            batch_summaries.append(
                {
                    "group_id": group["group_id"],
                    "asset_path": group["asset_path"],
                    "request": str(batch_request_path.resolve()),
                    "runtime_log": str(batch_runtime_log.resolve()),
                    "error": "invalid_batch_result",
                }
            )
            continue

        summary = {
            key: batch_result.get(key) for key in ("schema", *numeric_batch_fields)
        }
        summary.update(
            {
                "group_id": group["group_id"],
                "asset_path": group["asset_path"],
                "request": str(batch_request_path.resolve()),
                "runtime_log": str(batch_runtime_log.resolve()),
            }
        )
        batch_summaries.append(summary)
        result["checkpoint_executed"] = bool(
            result["checkpoint_executed"]
            or int(batch_result.get("p13_checkpoint_load_count", 0)) > 0
        )
        batch_items = {
            str(Path(str(item.get("request", ""))).resolve()): item
            for item in batch_result.get("items", [])
            if isinstance(item, dict) and item.get("request")
        }
        for entry in group_entries:
            item = entry["item"]
            entity = entry["entity"]
            request_key = str(entry["request_path"])
            batch_item = batch_items.get(request_key)
            if not batch_item or batch_item.get("passed") is not True:
                result["failures"].append(
                    {
                        "well_uid": entity.well_uid,
                        "well_name": entity.canonical_name,
                        "error": str(
                            (batch_item or {}).get(
                                "error", "P13 batch result is missing this well"
                            )
                        ),
                        "request": request_key,
                        "runtime_log": str(batch_runtime_log.resolve()),
                    }
                )
                continue
            result["executed_well_count"] += 1
            try:
                expectations: dict[str, Any] = {}
                if entry.get("trace_support"):
                    expectations.update(
                        {
                            "expected_trace_support_receipt_sha256": entry[
                                "trace_support"
                            ]["receipt_sha256"],
                            "expected_trace_support_identity_sha256": entry[
                                "trace_support"
                            ]["support_identity_sha256"],
                        }
                    )
                applicability_evidence = entry.get("applicability_evidence")
                if not isinstance(applicability_evidence, Mapping):
                    raise ValueError(
                        "P13 request has no grouped applicability evidence"
                    )
                expectations.update(
                    {
                        "expected_applicability_manifest_sha256": applicability_evidence[
                            "manifest_sha256"
                        ],
                        "expected_applicability_decision_sha256": applicability_evidence[
                            "decision_sha256"
                        ],
                    }
                )
                track, validation = _load_candidate(
                    entry["candidate_root"],
                    entity=entity,
                    trajectory=item["trajectory"],
                    head=item["head"],
                    surface_elevation_m=item["surface_elevation_m"],
                    trajectory_input_audit=entry["trajectory_input_audit"],
                    **expectations,
                )
            except Exception as exc:  # noqa: BLE001 - isolate invalid product
                result["failures"].append(
                    {
                        "well_uid": entity.well_uid,
                        "well_name": entity.canonical_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "request": request_key,
                        "runtime_log": str(batch_runtime_log.resolve()),
                    }
                )
                continue
            result["records"].append(
                {
                    "well_uid": entity.well_uid,
                    "well_name": entity.canonical_name,
                    "request": request_key,
                    "runtime_log": str(batch_runtime_log.resolve()),
                    "nearest_trace_distance_m": item["nearest_trace_distance_m"],
                    "trajectory_p95_trace_distance_m": item[
                        "trajectory_p95_trace_distance_m"
                    ],
                    "trajectory_max_trace_distance_m": item[
                        "trajectory_max_trace_distance_m"
                    ],
                    "trace_support_receipt": (
                        entry["trace_support"]["receipt_path"]
                        if entry.get("trace_support")
                        else None
                    ),
                    "trace_support_receipt_sha256": (
                        entry["trace_support"]["receipt_sha256"]
                        if entry.get("trace_support")
                        else None
                    ),
                    "applicability_manifest": applicability_evidence["manifest_path"],
                    "applicability_manifest_sha256": applicability_evidence[
                        "manifest_sha256"
                    ],
                    "applicability_decision_sha256": applicability_evidence[
                        "decision_sha256"
                    ],
                    "trajectory_input_audit": entry["trajectory_input_audit"],
                    **validation,
                }
            )
            if validation["inference_ready"]:
                result["tracks"][entity.canonical_name] = track
                result["accepted_well_uids"].append(entity.well_uid)
                result["accepted_well_count"] += 1

    aggregate: dict[str, Any] = {
        "schema": (
            batch_summaries[0].get("schema")
            if len(batch_summaries) == 1
            else "well-seismic.p13-grouped-batch-execution.v1"
        ),
        "group_count": len(batch_summaries),
        "microbatch_size": microbatch_size,
        "groups": batch_summaries,
    }
    for field in numeric_batch_fields:
        aggregate[field] = sum(
            int(summary.get(field) or 0) for summary in batch_summaries
        )
    if len(batch_summaries) == 1:
        aggregate["request"] = batch_summaries[0].get("request")
        aggregate["runtime_log"] = batch_summaries[0].get("runtime_log")
    result["batch_execution"] = aggregate

    if result["accepted_well_count"]:
        result["runtime_status"] = "executed_candidate_accepted"
    elif result["checkpoint_executed"] and result["records"]:
        result["runtime_status"] = "executed_rejected_physics_fallback"
    elif result["checkpoint_executed"]:
        result["runtime_status"] = "executed_product_invalid_physics_fallback"
    elif result["failures"]:
        result["runtime_status"] = "execution_failed_physics_fallback"
    else:
        result["runtime_status"] = "not_eligible"
    return result


def load_registration_points(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load V3 products, retaining the V2 branch for historical read-only use."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = tuple(csv.DictReader(handle).fieldnames or ())
    if "contract_version" in fieldnames:
        from .registration_contract import (
            read_registration_points_v3,
            registration_points_to_tracks_v3,
        )

        points, _ = read_registration_points_v3(source)
        tracks = registration_points_to_tracks_v3(points)
        for track in tracks.values():
            diagnostics = dict(track.get("diagnostics") or {})
            diagnostics["source_registration_points"] = str(source)
            track["diagnostics"] = diagnostics
        return tracks

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "well_uid",
            "well_name",
            "md_m",
            "tvd_m",
            "x",
            "y",
            "twt_mean_ms",
            "twt_std_ms",
            "quality",
            "method",
            "status",
            "training_eligible",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("registration_points.csv列合同不完整")
        for row in reader:
            grouped.setdefault((row["well_uid"], row["well_name"]), []).append(row)

    tracks: dict[str, dict[str, Any]] = {}
    for (well_uid, well_name), rows in grouped.items():
        rows.sort(key=lambda row: float(row["md_m"]))
        numeric = {
            key: [float(row[column]) for row in rows]
            for key, column in (
                ("md", "md_m"),
                ("tvd", "tvd_m"),
                ("x", "x"),
                ("y", "y"),
                ("twtMean", "twt_mean_ms"),
                ("twtStd", "twt_std_ms"),
                ("registrationQuality", "quality"),
            )
        }
        md = np.asarray(numeric["md"], dtype=float)
        twt = np.asarray(numeric["twtMean"], dtype=float)
        if md.size < 2 or not np.all(np.isfinite(md)) or not np.all(np.isfinite(twt)):
            continue
        if bool((np.diff(md) <= 0.0).any()) or bool((np.diff(twt) <= 0.0).any()):
            continue
        legacy_training_eligible = _csv_flag(
            rows[0], "training_eligible", fallback=False
        )
        inference_eligible = _csv_flag(rows[0], "inference_eligible", fallback=True)
        fusion_ready = (
            _csv_flag(rows[0], "fusion_ready", fallback=legacy_training_eligible)
            and inference_eligible
        )
        supervision_eligible = _csv_flag(
            rows[0], "supervision_eligible", fallback=legacy_training_eligible
        )
        track = {
            "well_uid": well_uid,
            "well_name": well_name,
            **numeric,
            "registrationSource": rows[0]["method"],
            "registrationStatus": rows[0]["status"],
            "registrationCoverage": 1.0,
            "inferenceEligible": inference_eligible,
            "fusionReady": fusion_ready,
            "supervisionEligible": supervision_eligible,
            "trainingEligible": legacy_training_eligible,
            "diagnostics": {
                "external_registration_reused": True,
                "source_registration_points": str(source),
                "registration_is_time_depth_supervision": supervision_eligible,
                "fusion_ready": fusion_ready,
            },
        }
        tracks[well_uid] = track
    return tracks


__all__ = [
    "arbitrate_registration_tracks",
    "build_p13_fusion_feature_tracks",
    "load_registration_points",
    "registration_evidence_priority",
    "run_p13_registration_candidates",
]
