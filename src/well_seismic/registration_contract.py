"""Authoritative full-resolution Registration V3 product contract.

Registration V2 mixed a browser preview with the reusable MD--TWT product.
V3 keeps every source-depth row in ``registration_points.csv`` and writes a
separate, explicitly non-authoritative preview.  Missing uncertainty remains
missing; it is never serialized as zero.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .content_identity import canonical_sha256, file_sha256


REGISTRATION_CONTRACT_VERSION = "well-seismic.registration.v3"
FUSION_FEATURE_TRACK_CONTRACT_VERSION = "well-seismic.p13-fusion-feature-track.v1"
STRICT_MD_TWT_POLICY = "strict_md_twt_v1"
TRAJECTORY_STATIONWISE_TWT_POLICY = "trajectory_stationwise_twt_v1"
TRAJECTORY_TIME_POLICIES = frozenset(
    {STRICT_MD_TWT_POLICY, TRAJECTORY_STATIONWISE_TWT_POLICY}
)
REGISTRATION_POINTS_FILENAME = "registration_points.csv"
REGISTRATION_PREVIEW_FILENAME = "registration_preview.csv"
REGISTRATION_MANIFEST_FILENAME = "registration_manifest.json"

REGISTRATION_POINT_COLUMNS = (
    "contract_version",
    "well_uid",
    "well_name",
    "point_index",
    "md_m",
    "tvd_m",
    "z_msl_m",
    "depth_below_srd_m",
    "x",
    "y",
    "inline",
    "crossline",
    "twt_mean_ms",
    "twt_std_ms",
    "quality",
    "valid_mask",
    "trajectory_time_policy",
    "trajectory_segment_id",
    "track_coverage",
    "method",
    "status",
    "source_authority",
    "well_depth_datum",
    "well_reference_elevation_m",
    "horizontal_crs_id",
    "horizontal_unit",
    "horizontal_axis_order",
    "vertical_crs_id",
    "seismic_srd_elevation_m",
    "absolute_reference_ready",
    "time_domain",
    "time_reference",
    "correction_state",
    "uncertainty_calibrated",
    "uncertainty_source",
    "inference_eligible",
    "fusion_ready",
    "supervision_eligible",
    "training_eligible",
)

_BACKWARD_COMPATIBLE_OPTIONAL_POINT_COLUMNS = frozenset(
    {
        "absolute_reference_ready",
        "trajectory_time_policy",
        "trajectory_segment_id",
    }
)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    numeric = _finite_or_none(value)
    return None if numeric is None else int(round(numeric))


def _truth(value: Any) -> bool:
    if type(value) is bool:
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    return value


def _series(
    track: Mapping[str, Any],
    key: str,
    count: int,
    *,
    default: Any = None,
) -> list[Any]:
    raw = track.get(key)
    if raw is None:
        return [default] * count
    values = list(raw)
    if len(values) != count:
        raise ValueError(
            f"registration track {track.get('well_uid') or track.get('well_name')} "
            f"column {key} has {len(values)} rows; expected {count}"
        )
    return values


def trajectory_stationwise_segment_ids(
    tvd_values: Sequence[Any],
    *,
    identity: str = "registration track",
) -> list[int]:
    """Return deterministic TVD-direction segments without rewriting a station.

    A trajectory can cross 90 degrees inclination, so TVD and the corresponding
    stationwise time can reverse while MD remains strictly increasing.  Segment
    ids make those direction changes explicit for audit and future consumers;
    they are not a license to sort, nudge, accumulate, or otherwise repair the
    physical trajectory.
    """

    values = [_finite_or_none(value) for value in tvd_values]
    if len(values) < 2 or any(value is None for value in values):
        raise ValueError(
            f"{identity}: trajectory-stationwise TWT requires finite TVD at every row"
        )
    numeric = [float(value) for value in values if value is not None]
    segment_ids = [0] * len(numeric)
    segment = 0
    direction = 0
    epsilon = 5e-9
    for index, (left, right) in enumerate(zip(numeric, numeric[1:]), start=1):
        delta = right - left
        next_direction = 1 if delta > epsilon else (-1 if delta < -epsilon else 0)
        if next_direction and direction and next_direction != direction:
            segment += 1
        if next_direction:
            direction = next_direction
        segment_ids[index] = segment
    return segment_ids


def validate_trajectory_stationwise_twt(
    tvd_values: Sequence[Any],
    twt_values: Sequence[Any],
    *,
    segment_ids: Sequence[Any] | None = None,
    identity: str = "registration track",
) -> dict[str, Any]:
    """Validate a stationwise trajectory time without imposing MD monotonic TWT.

    The accepted curve must follow the sign of the measured TVD change at every
    station interval.  This permits a real horizontal-well TVD reversal, but
    rejects an arbitrary/non-physical time reversal.  Every row must be valid so
    downstream MD interpolation never bridges an unobserved gap.
    """

    tvd = [_finite_or_none(value) for value in tvd_values]
    twt = [_finite_or_none(value) for value in twt_values]
    if len(tvd) != len(twt) or len(tvd) < 2:
        raise ValueError(
            f"{identity}: trajectory-stationwise TVD/TWT columns must have equal length"
        )
    if any(value is None for value in tvd) or any(value is None for value in twt):
        raise ValueError(
            f"{identity}: trajectory-stationwise TVD/TWT requires every row to be finite"
        )
    numeric_tvd = [float(value) for value in tvd if value is not None]
    numeric_twt = [float(value) for value in twt if value is not None]
    expected_segments = trajectory_stationwise_segment_ids(
        numeric_tvd,
        identity=identity,
    )
    if segment_ids is not None:
        parsed_segments = [_int_or_none(value) for value in segment_ids]
        if any(value is None or value < 0 for value in parsed_segments):
            raise ValueError(
                f"{identity}: trajectory segment ids must be finite non-negative integers"
            )
        if [int(value) for value in parsed_segments if value is not None] != expected_segments:
            raise ValueError(
                f"{identity}: trajectory segment ids do not match measured TVD direction changes"
            )

    epsilon = 5e-9
    reversal_intervals = 0
    flat_intervals = 0
    for index, ((left_tvd, right_tvd), (left_twt, right_twt)) in enumerate(
        zip(zip(numeric_tvd, numeric_tvd[1:]), zip(numeric_twt, numeric_twt[1:]))
    ):
        tvd_delta = right_tvd - left_tvd
        twt_delta = right_twt - left_twt
        if tvd_delta > epsilon and twt_delta <= epsilon:
            raise ValueError(
                f"{identity}: stationwise TWT does not follow increasing TVD at interval {index}"
            )
        if tvd_delta < -epsilon and twt_delta >= -epsilon:
            raise ValueError(
                f"{identity}: stationwise TWT does not follow decreasing TVD at interval {index}"
            )
        if abs(tvd_delta) <= epsilon and abs(twt_delta) > epsilon:
            raise ValueError(
                f"{identity}: stationwise TWT changes while TVD is flat at interval {index}"
            )
        reversal_intervals += int(tvd_delta < -epsilon)
        flat_intervals += int(abs(tvd_delta) <= epsilon)

    return {
        "trajectory_time_policy": TRAJECTORY_STATIONWISE_TWT_POLICY,
        "trajectory_segment_count": max(expected_segments) + 1,
        "tvd_reversal_interval_count": reversal_intervals,
        "flat_tvd_interval_count": flat_intervals,
        "trajectory_segment_ids": expected_segments,
    }


def validate_fusion_feature_track_v3(track: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an experimental P13 feature track without promoting it.

    A feature track is a separate, reusable TWT feature proposal. It is not
    the selected physical registration and it must never become supervision.
    The caller is responsible for sealing it separately from the physical
    primary product.
    """

    identity = str(track.get("well_uid") or track.get("well_name") or "").strip()
    if not identity:
        raise ValueError("fusion feature track has no well identity")
    md = list(track.get("md") or ())
    if len(md) < 2:
        raise ValueError(f"{identity}: fusion feature track has fewer than two MD rows")
    count = len(md)
    absolute_reference_ready = _truth(
        track.get("absoluteReferenceReady", True)
    )
    required_columns = {
        "tvd": _series(track, "tvd", count),
        "zMsl": _series(track, "zMsl", count),
        "x": _series(track, "x", count),
        "y": _series(track, "y", count),
        "twtMean": _series(track, "twtMean", count),
        "registrationQuality": _series(track, "registrationQuality", count),
    }
    if absolute_reference_ready:
        required_columns["depthBelowSrd"] = _series(
            track, "depthBelowSrd", count
        )
    numeric = {"md": md, **required_columns}
    for name, values in numeric.items():
        if any(_finite_or_none(value) is None for value in values):
            raise ValueError(f"{identity}: fusion feature {name} must be finite")
    md_values = [float(value) for value in md]
    twt_values = [float(value) for value in required_columns["twtMean"]]
    if any(right <= left for left, right in zip(md_values, md_values[1:])):
        raise ValueError(f"{identity}: fusion feature MD must be strictly increasing")
    if any(right <= left for left, right in zip(twt_values, twt_values[1:])):
        raise ValueError(f"{identity}: fusion feature TWT must be strictly increasing")
    quality_values = [
        float(value) for value in required_columns["registrationQuality"]
    ]
    if any(value < 0.0 or value > 1.0 for value in quality_values):
        raise ValueError(f"{identity}: fusion feature quality must be within [0, 1]")
    raw_std = track.get("twtStd")
    if raw_std is not None:
        std_values = list(raw_std)
        if len(std_values) != count:
            raise ValueError(f"{identity}: fusion feature twtStd length differs from MD")
        for value in std_values:
            if value is None or str(value).strip() == "":
                continue
            try:
                numeric_std = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{identity}: fusion feature uncertainty must be finite and non-negative"
                ) from exc
            if not math.isfinite(numeric_std) or numeric_std < 0.0:
                raise ValueError(
                    f"{identity}: fusion feature uncertainty must be finite and non-negative"
                )
    valid_mask = track.get("validMask")
    if valid_mask is not None and not all(_truth(value) for value in _series(track, "validMask", count)):
        raise ValueError(f"{identity}: fusion feature valid_mask must cover every feature row")
    if not _truth(track.get("inferenceEligible")) or not _truth(track.get("fusionReady")):
        raise ValueError(f"{identity}: fusion feature requires inferenceEligible and fusionReady")
    if _truth(track.get("supervisionEligible")) or _truth(track.get("trainingEligible")):
        raise ValueError(f"{identity}: fusion feature must not become supervision or training data")
    source = str(track.get("registrationSource") or "").casefold()
    if "p13" not in source:
        raise ValueError(f"{identity}: fusion feature must be sourced from P13")
    metadata = dict((track.get("diagnostics") or {}).get("fusion_feature_track") or {})
    if metadata.get("contract_version") != FUSION_FEATURE_TRACK_CONTRACT_VERSION:
        raise ValueError(f"{identity}: fusion feature contract metadata is missing")
    if metadata.get("role") != "experimental_p13_feature_track":
        raise ValueError(f"{identity}: fusion feature role is invalid")
    if not metadata.get("physical_primary_identity"):
        raise ValueError(f"{identity}: fusion feature does not attest its physical primary")
    return {
        "contract_version": FUSION_FEATURE_TRACK_CONTRACT_VERSION,
        "well_identity": identity,
        "point_count": count,
        "md_range_m": [md_values[0], md_values[-1]],
        "twt_range_ms": [twt_values[0], twt_values[-1]],
        "physical_primary_identity": str(metadata["physical_primary_identity"]),
    }


def infer_source_authority(track: Mapping[str, Any]) -> str:
    """Return a stable, human-readable evidence tier for one selected track."""

    source = str(track.get("registrationSource") or track.get("method") or "").casefold()
    status = str(track.get("registrationStatus") or track.get("status") or "").casefold()
    diagnostics = dict(track.get("diagnostics") or {})
    scientific_source_kind = str(
        track.get("sourceKind")
        or diagnostics.get("source_kind")
        or (diagnostics.get("time_depth_qc") or {}).get("selected_source_kind")
        or ""
    ).strip().casefold()
    if scientific_source_kind in {"checkshot", "vsp", "checkshot_vsp"}:
        return "provided_checkshot_vsp"
    if scientific_source_kind == "provided_time_depth":
        return "provided_time_depth"
    if scientific_source_kind == "well_twt_curve":
        return "well_twt_curve"
    if (
        "checkshot" in source
        or "vsp" in source
    ):
        return "provided_checkshot_vsp"
    if status == "provided_tie" or "provided_time_depth" in source:
        return "provided_time_depth"
    if "sonic" in source:
        return "sonic_physical_tie"
    if "p13" in source:
        return (
            "learned_p13_fusion_ready"
            if _truth(track.get("fusionReady", track.get("fusion_ready")))
            else "learned_p13_candidate"
        )
    if "external" in source:
        return "external_registration"
    return "physical_initialization"


@dataclass(frozen=True)
class RegistrationPointV3:
    contract_version: str
    well_uid: str
    well_name: str
    point_index: int
    md_m: float
    tvd_m: float | None
    z_msl_m: float | None
    depth_below_srd_m: float | None
    x: float | None
    y: float | None
    inline: int | None
    crossline: int | None
    twt_mean_ms: float | None
    twt_std_ms: float | None
    quality: float | None
    valid_mask: bool
    trajectory_time_policy: str
    trajectory_segment_id: int | None
    track_coverage: float
    method: str
    status: str
    source_authority: str
    well_depth_datum: str | None
    well_reference_elevation_m: float | None
    horizontal_crs_id: str
    horizontal_unit: str
    horizontal_axis_order: str
    vertical_crs_id: str
    seismic_srd_elevation_m: float | None
    absolute_reference_ready: bool
    time_domain: str
    time_reference: str
    correction_state: str
    uncertainty_calibrated: bool
    uncertainty_source: str | None
    inference_eligible: bool
    fusion_ready: bool
    supervision_eligible: bool
    training_eligible: bool

    def csv_row(self) -> dict[str, Any]:
        return {key: _csv_value(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class RegistrationWriteResult:
    points_path: Path
    preview_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    points_sha256: str
    preview_sha256: str
    manifest_sha256: str
    product_sha256: str


@dataclass(frozen=True)
class RegistrationProductV3:
    manifest_path: Path
    points_path: Path
    preview_path: Path | None
    manifest: dict[str, Any]
    points: tuple[RegistrationPointV3, ...]
    tracks: dict[str, dict[str, Any]]
    validation: dict[str, Any]


def _points_for_track(
    track: Mapping[str, Any],
    semantics: Mapping[str, Any],
) -> list[RegistrationPointV3]:
    md = list(track.get("md") or ())
    if len(md) < 2:
        raise ValueError("registration track must contain at least two full-resolution MD rows")
    count = len(md)
    tvd = _series(track, "tvd", count)
    z_msl = _series(track, "zMsl", count)
    depth_below_srd = _series(track, "depthBelowSrd", count)
    x = _series(track, "x", count)
    y = _series(track, "y", count)
    inline = _series(track, "inline", count)
    crossline = _series(track, "crossline", count)
    twt = _series(track, "twtMean", count)
    std = _series(track, "twtStd", count)
    quality = _series(track, "registrationQuality", count)
    explicit_valid = track.get("validMask")
    if explicit_valid is None:
        valid = [
            all(_finite_or_none(value) is not None for value in values)
            for values in zip(md, x, y, twt)
        ]
    else:
        valid = [_truth(value) for value in _series(track, "validMask", count)]
    coverage = float(sum(valid) / count)

    diagnostics = dict(track.get("diagnostics") or {})
    trajectory_time_policy = str(
        track.get("trajectoryTimePolicy")
        or track.get("trajectory_time_policy")
        or diagnostics.get("trajectory_time_policy")
        or STRICT_MD_TWT_POLICY
    ).strip()
    if trajectory_time_policy not in TRAJECTORY_TIME_POLICIES:
        raise ValueError(
            "unsupported trajectory time policy: " + trajectory_time_policy
        )
    explicit_segment_ids = track.get(
        "trajectorySegmentId", track.get("trajectory_segment_id")
    )
    if trajectory_time_policy == TRAJECTORY_STATIONWISE_TWT_POLICY:
        valid_indices = [index for index, selected in enumerate(valid) if selected]
        if len(valid_indices) < 2 or valid_indices != list(
            range(valid_indices[0], valid_indices[-1] + 1)
        ):
            raise ValueError(
                f"{track.get('well_uid') or track.get('well_name')}: "
                "trajectory-stationwise TWT requires one contiguous valid support interval"
            )
        supplied_segment_ids: list[Any] | None = None
        if explicit_segment_ids is not None:
            full_supplied_segment_ids = list(explicit_segment_ids)
            if len(full_supplied_segment_ids) != count:
                raise ValueError(
                    f"{track.get('well_uid') or track.get('well_name')}: "
                    "trajectory segment id length differs from MD"
                )
            if any(
                full_supplied_segment_ids[index] is not None
                and str(full_supplied_segment_ids[index]).strip() != ""
                for index, selected in enumerate(valid)
                if not selected
            ):
                raise ValueError(
                    f"{track.get('well_uid') or track.get('well_name')}: "
                    "trajectory segment ids must be null outside valid support"
                )
            supplied_segment_ids = [
                full_supplied_segment_ids[index] for index in valid_indices
            ]
        stationwise_validation = validate_trajectory_stationwise_twt(
            [tvd[index] for index in valid_indices],
            [twt[index] for index in valid_indices],
            segment_ids=supplied_segment_ids,
            identity=str(track.get("well_uid") or track.get("well_name") or "track"),
        )
        trajectory_segment_ids = [None] * count
        for index, segment_id in zip(
            valid_indices,
            stationwise_validation["trajectory_segment_ids"],
        ):
            trajectory_segment_ids[index] = int(segment_id)
    else:
        if explicit_segment_ids is not None:
            raise ValueError(
                f"{track.get('well_uid') or track.get('well_name')}: "
                "strict MD-TWT track must not declare trajectory segment ids"
            )
        trajectory_segment_ids = [None] * count
    source_authority = str(
        track.get("sourceAuthority") or infer_source_authority(track)
    )
    horizontal_crs_id = str(
        track.get("horizontalCrsId") or semantics.get("horizontal_crs_id") or ""
    )
    horizontal_unit = str(
        track.get("horizontalUnit") or semantics.get("horizontal_unit") or ""
    )
    axis_order = str(
        track.get("horizontalAxisOrder")
        or semantics.get("horizontal_axis_order")
        or ""
    )
    vertical_crs_id = str(
        track.get("verticalCrsId") or semantics.get("vertical_crs_id") or ""
    )
    seismic_srd = _finite_or_none(
        track.get("seismicSrdElevationM", semantics.get("seismic_srd_elevation_m"))
    )
    absolute_reference_ready = _truth(
        track.get(
            "absoluteReferenceReady",
            semantics.get("absolute_reference_ready", True),
        )
    )
    if not horizontal_crs_id or horizontal_unit != "m" or axis_order not in {"XY", "YX"}:
        raise ValueError("Registration V3 requires a verified metre-based horizontal CRS contract")
    if not vertical_crs_id:
        raise ValueError("Registration V3 requires a vertical survey namespace")
    if absolute_reference_ready and seismic_srd is None:
        raise ValueError(
            "absolute-reference Registration V3 requires a finite seismic SRD"
        )
    time_domain = str(track.get("timeDomain") or semantics.get("time_domain") or "")
    time_reference = str(
        track.get("timeReference") or semantics.get("time_reference") or ""
    )
    correction_state = str(
        track.get("correctionState") or semantics.get("correction_state") or ""
    )
    time_axis_ready = _truth(
        track.get(
            "timeAxisReady",
            semantics.get("time_axis_ready", time_domain == "TWT"),
        )
    )
    if absolute_reference_ready:
        if (
            time_domain != "TWT"
            or time_reference != "SRD"
            or correction_state != "corrected_to_srd"
        ):
            raise ValueError("absolute-reference Registration V3 requires TWT corrected to SRD")
    elif (
        time_reference != "native_segy_sample_zero"
        or correction_state != "native_unmodified"
        or (time_axis_ready and time_domain != "TWT")
        or (not time_axis_ready and time_domain != "native_time_unknown")
    ):
        raise ValueError(
            "native-relative Registration V3 requires the unmodified native SEG-Y sample axis"
        )

    method = str(track.get("registrationSource") or "unknown")
    status = str(track.get("registrationStatus") or "unregistered")
    inference_eligible = _truth(track.get("inferenceEligible", True))
    fusion_ready = _truth(
        track.get("fusionReady", track.get("trainingEligible", False))
    ) and inference_eligible
    supervision_eligible = _truth(
        track.get("supervisionEligible", track.get("trainingEligible", False))
    )
    training_eligible = _truth(track.get("trainingEligible", False))
    if trajectory_time_policy == TRAJECTORY_STATIONWISE_TWT_POLICY and (
        supervision_eligible or training_eligible
    ):
        raise ValueError(
            "trajectory-stationwise inferred TWT cannot become supervision or training data"
        )
    if not absolute_reference_ready and (
        supervision_eligible or training_eligible
    ):
        raise ValueError(
            "native-relative registration cannot become supervision or training data"
        )
    if not time_axis_ready and fusion_ready:
        raise ValueError(
            "registration with an unknown native time axis cannot be fusion-ready"
        )
    well_depth_datum = track.get("wellDepthDatum")
    well_reference = _finite_or_none(track.get("wellReferenceElevationM"))
    uncertainty_calibrated = _truth(
        track.get(
            "uncertaintyCalibrated",
            diagnostics.get("uncertainty_calibrated", False),
        )
    )
    uncertainty_source = track.get("uncertaintySource") or diagnostics.get(
        "uncertainty_definition"
    )

    points: list[RegistrationPointV3] = []
    for index in range(count):
        md_value = _finite_or_none(md[index])
        if md_value is None:
            raise ValueError(f"Registration V3 MD is not finite at row {index}")
        point_valid = bool(valid[index])
        twt_value = _finite_or_none(twt[index])
        x_value = _finite_or_none(x[index])
        y_value = _finite_or_none(y[index])
        if point_valid and (twt_value is None or x_value is None or y_value is None):
            raise ValueError(f"valid_mask=true row {index} lacks finite XY or TWT")
        std_value = _finite_or_none(std[index])
        if std_value is not None and std_value < 0.0:
            raise ValueError(f"negative TWT uncertainty at row {index}")
        quality_value = _finite_or_none(quality[index])
        if quality_value is not None and not 0.0 <= quality_value <= 1.0:
            raise ValueError(f"registration quality outside [0, 1] at row {index}")
        points.append(
            RegistrationPointV3(
                contract_version=REGISTRATION_CONTRACT_VERSION,
                well_uid=str(track.get("well_uid") or ""),
                well_name=str(track.get("well_name") or ""),
                point_index=index,
                md_m=md_value,
                tvd_m=_finite_or_none(tvd[index]),
                z_msl_m=_finite_or_none(z_msl[index]),
                depth_below_srd_m=_finite_or_none(depth_below_srd[index]),
                x=x_value,
                y=y_value,
                inline=_int_or_none(inline[index]),
                crossline=_int_or_none(crossline[index]),
                twt_mean_ms=twt_value,
                twt_std_ms=std_value,
                quality=quality_value,
                valid_mask=point_valid,
                trajectory_time_policy=trajectory_time_policy,
                trajectory_segment_id=trajectory_segment_ids[index],
                track_coverage=coverage,
                method=method,
                status=status,
                source_authority=source_authority,
                well_depth_datum=(
                    str(well_depth_datum) if well_depth_datum is not None else None
                ),
                well_reference_elevation_m=well_reference,
                horizontal_crs_id=horizontal_crs_id,
                horizontal_unit=horizontal_unit,
                horizontal_axis_order=axis_order,
                vertical_crs_id=vertical_crs_id,
                seismic_srd_elevation_m=seismic_srd,
                absolute_reference_ready=absolute_reference_ready,
                time_domain=time_domain,
                time_reference=time_reference,
                correction_state=correction_state,
                uncertainty_calibrated=uncertainty_calibrated,
                uncertainty_source=(
                    str(uncertainty_source) if uncertainty_source else None
                ),
                inference_eligible=inference_eligible,
                fusion_ready=fusion_ready,
                supervision_eligible=supervision_eligible,
                training_eligible=training_eligible,
            )
        )
    return points


def build_registration_points_v3(
    tracks: Iterable[Mapping[str, Any]],
    *,
    semantics: Mapping[str, Any],
) -> list[RegistrationPointV3]:
    points: list[RegistrationPointV3] = []
    identities: set[str] = set()
    for track in tracks:
        identity = str(track.get("well_uid") or track.get("well_name") or "").strip()
        if not identity:
            raise ValueError("registration track has no well identity")
        if identity.casefold() in identities:
            raise ValueError(f"duplicate selected registration track: {identity}")
        identities.add(identity.casefold())
        points.extend(_points_for_track(track, semantics))
    validate_registration_points_v3(points)
    return points


def validate_registration_points_v3(
    points: Sequence[RegistrationPointV3],
) -> dict[str, Any]:
    if not points:
        return {
            "contract_version": REGISTRATION_CONTRACT_VERSION,
            "point_count": 0,
            "valid_point_count": 0,
            "well_count": 0,
            "wells": {},
        }
    grouped: dict[str, list[RegistrationPointV3]] = {}
    for point in points:
        if point.contract_version != REGISTRATION_CONTRACT_VERSION:
            raise ValueError("mixed or unsupported registration point contract")
        identity = point.well_uid or point.well_name
        if not identity:
            raise ValueError("registration point has no well identity")
        grouped.setdefault(identity, []).append(point)

    wells: dict[str, dict[str, Any]] = {}
    for identity, rows in grouped.items():
        indices = [row.point_index for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"{identity}: point_index is not complete and sequential")
        md = [row.md_m for row in rows]
        if any(not math.isfinite(value) for value in md) or any(
            right <= left for left, right in zip(md, md[1:])
        ):
            raise ValueError(f"{identity}: MD must be finite and strictly increasing")
        valid_rows = [row for row in rows if row.valid_mask]
        valid_twt = [float(row.twt_mean_ms) for row in valid_rows if row.twt_mean_ms is not None]
        if len(valid_rows) < 2 or len(valid_twt) != len(valid_rows):
            raise ValueError(f"{identity}: fewer than two valid MD--TWT points")
        trajectory_time_policy = rows[0].trajectory_time_policy or STRICT_MD_TWT_POLICY
        stationwise_validation: dict[str, Any] | None = None
        if trajectory_time_policy == STRICT_MD_TWT_POLICY:
            if any(right <= left for left, right in zip(valid_twt, valid_twt[1:])):
                raise ValueError(f"{identity}: valid TWT must be strictly increasing")
            if any(row.trajectory_segment_id is not None for row in rows):
                raise ValueError(
                    f"{identity}: strict MD-TWT track contains trajectory segment ids"
                )
        elif trajectory_time_policy == TRAJECTORY_STATIONWISE_TWT_POLICY:
            valid_indices = [index for index, row in enumerate(rows) if row.valid_mask]
            if valid_indices != list(range(valid_indices[0], valid_indices[-1] + 1)):
                raise ValueError(
                    f"{identity}: trajectory-stationwise TWT valid support contains an internal gap"
                )
            if any(
                row.trajectory_segment_id is not None
                for row in rows
                if not row.valid_mask
            ):
                raise ValueError(
                    f"{identity}: trajectory segment id exists outside valid support"
                )
            stationwise_validation = validate_trajectory_stationwise_twt(
                [row.tvd_m for row in valid_rows],
                [row.twt_mean_ms for row in valid_rows],
                segment_ids=[row.trajectory_segment_id for row in valid_rows],
                identity=identity,
            )
            if rows[0].supervision_eligible or rows[0].training_eligible:
                raise ValueError(
                    f"{identity}: trajectory-stationwise inferred TWT cannot be supervision or training"
                )
        else:
            raise ValueError(
                f"{identity}: unsupported trajectory time policy {trajectory_time_policy}"
            )
        if rows[0].fusion_ready and any(
            row.tvd_m is None
            or row.z_msl_m is None
            or (
                row.absolute_reference_ready
                and row.depth_below_srd_m is None
            )
            for row in valid_rows
        ):
            raise ValueError(
                f"{identity}: fusion_ready requires TVD/z_msl and, when absolute, depth_below_srd"
            )
        expected_coverage = len(valid_rows) / len(rows)
        if any(abs(row.track_coverage - expected_coverage) > 5e-7 for row in rows):
            raise ValueError(f"{identity}: track_coverage does not match valid_mask")
        invariant_fields = (
            "method",
            "status",
            "source_authority",
            "horizontal_crs_id",
            "horizontal_unit",
            "horizontal_axis_order",
            "vertical_crs_id",
            "seismic_srd_elevation_m",
            "absolute_reference_ready",
            "time_domain",
            "time_reference",
            "correction_state",
            "trajectory_time_policy",
            "inference_eligible",
            "fusion_ready",
            "supervision_eligible",
            "training_eligible",
        )
        for field in invariant_fields:
            if len({getattr(row, field) for row in rows}) != 1:
                raise ValueError(f"{identity}: track-level field {field} changes by row")
        first = rows[0]
        wells[identity] = {
            "well_uid": first.well_uid,
            "well_name": first.well_name,
            "point_count": len(rows),
            "valid_point_count": len(valid_rows),
            "coverage": round(expected_coverage, 8),
            "md_range_m": [md[0], md[-1]],
            "twt_range_ms": [min(valid_twt), max(valid_twt)],
            "nullable_uncertainty_count": sum(
                row.twt_std_ms is None for row in rows
            ),
            "method": first.method,
            "status": first.status,
            "source_authority": first.source_authority,
            "fusion_ready": first.fusion_ready,
            "trajectory_time_policy": trajectory_time_policy,
            "trajectory_segment_count": (
                stationwise_validation["trajectory_segment_count"]
                if stationwise_validation is not None
                else 1
            ),
            "tvd_reversal_interval_count": (
                stationwise_validation["tvd_reversal_interval_count"]
                if stationwise_validation is not None
                else 0
            ),
        }
    return {
        "contract_version": REGISTRATION_CONTRACT_VERSION,
        "point_count": len(points),
        "valid_point_count": sum(point.valid_mask for point in points),
        "well_count": len(wells),
        "wells": wells,
    }


def _write_points(path: Path, points: Sequence[RegistrationPointV3]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRATION_POINT_COLUMNS)
        writer.writeheader()
        writer.writerows(point.csv_row() for point in points)


def _preview_points(
    points: Sequence[RegistrationPointV3],
    *,
    per_well_limit: int,
) -> list[RegistrationPointV3]:
    grouped: dict[str, list[RegistrationPointV3]] = {}
    for point in points:
        grouped.setdefault(point.well_uid or point.well_name, []).append(point)
    preview: list[RegistrationPointV3] = []
    for rows in grouped.values():
        if len(rows) <= per_well_limit:
            preview.extend(rows)
            continue
        indices = {
            int(round(index * (len(rows) - 1) / (per_well_limit - 1)))
            for index in range(per_well_limit)
        }
        preview.extend(rows[index] for index in sorted(indices))
    return preview


def registration_product_sha256(
    manifest: Mapping[str, Any],
    *,
    points_sha256: str,
    preview_sha256: str,
) -> str:
    stable_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"registration_product_sha256"}
    }
    return canonical_sha256(
        {
            "contract_version": REGISTRATION_CONTRACT_VERSION,
            "registration_points_sha256": points_sha256,
            "registration_preview_sha256": preview_sha256,
            "manifest": stable_manifest,
        }
    )


def write_registration_product_v3(
    output_directory: str | Path,
    tracks: Iterable[Mapping[str, Any]],
    *,
    semantics: Mapping[str, Any],
    manifest_fields: Mapping[str, Any] | None = None,
    preview_limit: int = 240,
) -> RegistrationWriteResult:
    if preview_limit < 2:
        raise ValueError("registration preview_limit must be at least 2")
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    points = build_registration_points_v3(tracks, semantics=semantics)
    validation = validate_registration_points_v3(points)
    points_path = directory / REGISTRATION_POINTS_FILENAME
    preview_path = directory / REGISTRATION_PREVIEW_FILENAME
    manifest_path = directory / REGISTRATION_MANIFEST_FILENAME
    _write_points(points_path, points)
    _write_points(preview_path, _preview_points(points, per_well_limit=preview_limit))
    points_sha = file_sha256(points_path)
    preview_sha = file_sha256(preview_path)
    absolute_reference_ready = _truth(
        semantics.get("absolute_reference_ready", True)
    )
    time_axis_ready = _truth(
        semantics.get(
            "time_axis_ready",
            str(semantics.get("time_domain") or "") == "TWT",
        )
    )
    seismic_srd_elevation_m = _finite_or_none(
        semantics.get("seismic_srd_elevation_m")
    )

    protected = {
        "contract_version",
        "schema_version",
        "point_contract",
        "coordinate_contract",
        "vertical_contract",
        "time_contract",
        "registration_points_sha256",
        "registration_preview_sha256",
        "registration_product_sha256",
        "outputs",
        "wells",
        "point_count",
        "valid_point_count",
    }
    extra = {
        key: value
        for key, value in dict(manifest_fields or {}).items()
        if key not in protected
    }
    manifest: dict[str, Any] = {
        **extra,
        "contract_version": REGISTRATION_CONTRACT_VERSION,
        "schema_version": 3,
        "point_contract": {
            "authoritative_file": REGISTRATION_POINTS_FILENAME,
            "full_resolution": True,
            "preview_is_non_authoritative": True,
            "valid_mask_definition": "finite MD, XY and TWT selected by registration",
            "nullable_columns": [
                "tvd_m",
                "z_msl_m",
                "depth_below_srd_m",
                "inline",
                "crossline",
                "twt_mean_ms",
                "twt_std_ms",
                "quality",
            ],
            "missing_uncertainty_semantics": "unknown_not_zero",
            "trajectory_time_policies": {
                STRICT_MD_TWT_POLICY: (
                    "default; valid TWT is strictly increasing along MD"
                ),
                TRAJECTORY_STATIONWISE_TWT_POLICY: (
                    "explicit measured-trajectory station mapping; local TWT reversal "
                    "must follow the sign of the preserved TVD reversal"
                ),
            },
            "trajectory_segment_semantics": (
                "direction-run audit id derived from unchanged TVD stations; "
                "not permission to sort, repair, or bridge invalid rows"
            ),
        },
        "coordinate_contract": {
            "horizontal_crs_id": str(semantics.get("horizontal_crs_id") or ""),
            "horizontal_unit": str(semantics.get("horizontal_unit") or ""),
            "horizontal_axis_order": str(
                semantics.get("horizontal_axis_order") or ""
            ),
        },
        "vertical_contract": {
            "vertical_crs_id": str(semantics.get("vertical_crs_id") or ""),
            "canonical_coordinate": "z_msl_m",
            "canonical_reference": "MSL",
            "positive_direction": "up",
            "absolute_reference_ready": absolute_reference_ready,
            "seismic_srd_elevation_m": seismic_srd_elevation_m,
            "depth_below_srd_definition": (
                "seismic_srd_elevation_m - z_msl_m"
                if absolute_reference_ready
                else None
            ),
            "native_relative_policy": (
                None
                if absolute_reference_ready
                else "SRD remains unknown; z_msl comes only from resolved well KB/DF/RT"
            ),
        },
        "time_contract": {
            "time_domain": str(semantics.get("time_domain") or ""),
            "time_reference": str(semantics.get("time_reference") or ""),
            "correction_state": str(semantics.get("correction_state") or ""),
            "unit": "ms",
            "absolute_reference_ready": absolute_reference_ready,
            "time_axis_ready": time_axis_ready,
        },
        "point_count": validation["point_count"],
        "valid_point_count": validation["valid_point_count"],
        "wells": list(validation["wells"].values()),
        "registration_points_sha256": points_sha,
        "registration_preview_sha256": preview_sha,
        "outputs": {
            **dict((manifest_fields or {}).get("outputs") or {}),
            "manifest": str(manifest_path),
            "registration_points": str(points_path),
            "registration_preview": str(preview_path),
        },
    }
    product_sha = registration_product_sha256(
        manifest,
        points_sha256=points_sha,
        preview_sha256=preview_sha,
    )
    manifest["registration_product_sha256"] = product_sha
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return RegistrationWriteResult(
        points_path=points_path,
        preview_path=preview_path,
        manifest_path=manifest_path,
        manifest=manifest,
        points_sha256=points_sha,
        preview_sha256=preview_sha,
        manifest_sha256=file_sha256(manifest_path),
        product_sha256=product_sha,
    )


def _parse_optional_float(row: Mapping[str, str], key: str) -> float | None:
    value = str(row.get(key) or "").strip()
    return None if not value else _finite_or_none(value)


def _point_from_row(row: Mapping[str, str]) -> RegistrationPointV3:
    return RegistrationPointV3(
        contract_version=str(row["contract_version"]),
        well_uid=str(row["well_uid"]),
        well_name=str(row["well_name"]),
        point_index=int(row["point_index"]),
        md_m=float(row["md_m"]),
        tvd_m=_parse_optional_float(row, "tvd_m"),
        z_msl_m=_parse_optional_float(row, "z_msl_m"),
        depth_below_srd_m=_parse_optional_float(row, "depth_below_srd_m"),
        x=_parse_optional_float(row, "x"),
        y=_parse_optional_float(row, "y"),
        inline=_int_or_none(row.get("inline")),
        crossline=_int_or_none(row.get("crossline")),
        twt_mean_ms=_parse_optional_float(row, "twt_mean_ms"),
        twt_std_ms=_parse_optional_float(row, "twt_std_ms"),
        quality=_parse_optional_float(row, "quality"),
        valid_mask=_truth(row["valid_mask"]),
        trajectory_time_policy=(
            str(row.get("trajectory_time_policy") or "").strip()
            or STRICT_MD_TWT_POLICY
        ),
        trajectory_segment_id=_int_or_none(row.get("trajectory_segment_id")),
        track_coverage=float(row["track_coverage"]),
        method=str(row["method"]),
        status=str(row["status"]),
        source_authority=str(row["source_authority"]),
        well_depth_datum=(str(row.get("well_depth_datum") or "").strip() or None),
        well_reference_elevation_m=_parse_optional_float(
            row, "well_reference_elevation_m"
        ),
        horizontal_crs_id=str(row["horizontal_crs_id"]),
        horizontal_unit=str(row["horizontal_unit"]),
        horizontal_axis_order=str(row["horizontal_axis_order"]),
        vertical_crs_id=str(row["vertical_crs_id"]),
        seismic_srd_elevation_m=_parse_optional_float(
            row, "seismic_srd_elevation_m"
        ),
        absolute_reference_ready=(
            _truth(row.get("absolute_reference_ready"))
            if "absolute_reference_ready" in row
            else True
        ),
        time_domain=str(row["time_domain"]),
        time_reference=str(row["time_reference"]),
        correction_state=str(row["correction_state"]),
        uncertainty_calibrated=_truth(row["uncertainty_calibrated"]),
        uncertainty_source=(str(row.get("uncertainty_source") or "").strip() or None),
        inference_eligible=_truth(row["inference_eligible"]),
        fusion_ready=_truth(row["fusion_ready"]),
        supervision_eligible=_truth(row["supervision_eligible"]),
        training_eligible=_truth(row["training_eligible"]),
    )


def read_registration_points_v3(
    path: str | Path,
) -> tuple[tuple[RegistrationPointV3, ...], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(
            (
                set(REGISTRATION_POINT_COLUMNS)
                - _BACKWARD_COMPATIBLE_OPTIONAL_POINT_COLUMNS
            )
            - set(reader.fieldnames or ())
        )
        if missing:
            raise ValueError(
                "Registration V3 points columns are incomplete: " + ", ".join(missing)
            )
        points = tuple(_point_from_row(row) for row in reader)
    validation = validate_registration_points_v3(points)
    return points, validation


def registration_points_to_tracks_v3(
    points: Sequence[RegistrationPointV3],
) -> dict[str, dict[str, Any]]:
    validation = validate_registration_points_v3(points)
    grouped: dict[str, list[RegistrationPointV3]] = {}
    for point in points:
        grouped.setdefault(point.well_uid or point.well_name, []).append(point)
    tracks: dict[str, dict[str, Any]] = {}
    for identity, rows in grouped.items():
        first = rows[0]

        def optional(values: Iterable[float | int | None]) -> list[float]:
            return [float("nan") if value is None else float(value) for value in values]

        tracks[identity] = {
            "well_uid": first.well_uid,
            "well_name": first.well_name,
            "md": [row.md_m for row in rows],
            "tvd": optional(row.tvd_m for row in rows),
            "zMsl": optional(row.z_msl_m for row in rows),
            "depthBelowSrd": optional(row.depth_below_srd_m for row in rows),
            "x": optional(row.x for row in rows),
            "y": optional(row.y for row in rows),
            "inline": optional(row.inline for row in rows),
            "crossline": optional(row.crossline for row in rows),
            "twtMean": optional(row.twt_mean_ms for row in rows),
            "twtStd": optional(row.twt_std_ms for row in rows),
            "registrationQuality": optional(row.quality for row in rows),
            "validMask": [row.valid_mask for row in rows],
            "trajectoryTimePolicy": first.trajectory_time_policy,
            "trajectorySegmentId": (
                [row.trajectory_segment_id for row in rows]
                if first.trajectory_time_policy
                == TRAJECTORY_STATIONWISE_TWT_POLICY
                else None
            ),
            "registrationSource": first.method,
            "registrationStatus": first.status,
            "registrationCoverage": validation["wells"][identity]["coverage"],
            "sourceAuthority": first.source_authority,
            "wellDepthDatum": first.well_depth_datum,
            "wellReferenceElevationM": first.well_reference_elevation_m,
            "horizontalCrsId": first.horizontal_crs_id,
            "horizontalUnit": first.horizontal_unit,
            "horizontalAxisOrder": first.horizontal_axis_order,
            "verticalCrsId": first.vertical_crs_id,
            "seismicSrdElevationM": first.seismic_srd_elevation_m,
            "absoluteReferenceReady": first.absolute_reference_ready,
            "timeDomain": first.time_domain,
            "timeReference": first.time_reference,
            "correctionState": first.correction_state,
            "uncertaintyCalibrated": first.uncertainty_calibrated,
            "uncertaintySource": first.uncertainty_source,
            "inferenceEligible": first.inference_eligible,
            "fusionReady": first.fusion_ready and first.inference_eligible,
            "supervisionEligible": first.supervision_eligible,
            "trainingEligible": first.training_eligible,
            "diagnostics": {
                "external_registration_reused": True,
                "registration_contract_version": REGISTRATION_CONTRACT_VERSION,
                "registration_is_time_depth_supervision": first.supervision_eligible,
                "fusion_ready": first.fusion_ready and first.inference_eligible,
                "trajectory_time_policy": first.trajectory_time_policy,
                "trajectory_segment_count": validation["wells"][identity][
                    "trajectory_segment_count"
                ],
                "tvd_reversal_interval_count": validation["wells"][identity][
                    "tvd_reversal_interval_count"
                ],
            },
        }
    return tracks


def read_registration_product_v3(
    manifest_path: str | Path,
) -> RegistrationProductV3:
    manifest_source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != REGISTRATION_CONTRACT_VERSION:
        raise ValueError(
            "Registration V3 reader is read-only for V3 products; use the legacy "
            "V2 loader for historical outputs"
        )
    outputs = dict(manifest.get("outputs") or {})
    points_path = Path(outputs.get("registration_points") or "")
    if not points_path.is_absolute():
        points_path = manifest_source.parent / points_path
    # A migrated Windows installation can expose the same sealed product
    # through both the legacy checkout path and the runtime junction target.
    # Return canonical paths so downstream identity checks compare the file,
    # not two aliases for that file.
    points_path = points_path.expanduser().resolve()
    points, validation = read_registration_points_v3(points_path)
    points_sha = file_sha256(points_path)
    if points_sha != manifest.get("registration_points_sha256"):
        raise ValueError("Registration V3 points content hash mismatch")
    preview_value = outputs.get("registration_preview")
    preview_path = Path(preview_value) if preview_value else None
    if preview_path is not None and not preview_path.is_absolute():
        preview_path = manifest_source.parent / preview_path
    if preview_path is not None:
        preview_path = preview_path.expanduser().resolve()
    preview_sha = file_sha256(preview_path) if preview_path is not None else ""
    if preview_sha != str(manifest.get("registration_preview_sha256") or ""):
        raise ValueError("Registration V3 preview content hash mismatch")
    expected_product_sha = registration_product_sha256(
        manifest,
        points_sha256=points_sha,
        preview_sha256=preview_sha,
    )
    if expected_product_sha != manifest.get("registration_product_sha256"):
        raise ValueError("Registration V3 product hash mismatch")
    return RegistrationProductV3(
        manifest_path=manifest_source,
        points_path=points_path,
        preview_path=preview_path,
        manifest=manifest,
        points=points,
        tracks=registration_points_to_tracks_v3(points),
        validation=validation,
    )


__all__ = [
    "FUSION_FEATURE_TRACK_CONTRACT_VERSION",
    "REGISTRATION_CONTRACT_VERSION",
    "REGISTRATION_MANIFEST_FILENAME",
    "REGISTRATION_POINT_COLUMNS",
    "REGISTRATION_POINTS_FILENAME",
    "REGISTRATION_PREVIEW_FILENAME",
    "STRICT_MD_TWT_POLICY",
    "TRAJECTORY_STATIONWISE_TWT_POLICY",
    "TRAJECTORY_TIME_POLICIES",
    "RegistrationPointV3",
    "RegistrationProductV3",
    "RegistrationWriteResult",
    "build_registration_points_v3",
    "infer_source_authority",
    "read_registration_points_v3",
    "read_registration_product_v3",
    "registration_points_to_tracks_v3",
    "registration_product_sha256",
    "trajectory_stationwise_segment_ids",
    "validate_trajectory_stationwise_twt",
    "validate_registration_points_v3",
    "validate_fusion_feature_track_v3",
    "write_registration_product_v3",
]
