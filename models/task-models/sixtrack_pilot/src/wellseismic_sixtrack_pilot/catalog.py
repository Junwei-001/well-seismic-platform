"""Model ids, payload versions and request validation for the six-track pilot.

Validation runs twice: at submission (the platform's ``validate_prediction_request``
hook turns ``ValueError`` into HTTP 422) and again inside the runner, so a
request that bypasses the HTTP layer is still rejected before any work starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FAULT_MODEL_ID = "sixtrack_fault_local_logistic_audited_v2"
RECONSTRUCTION_MODEL_ID = "sixtrack_reconstruction_p21"
SWEETSPOT_MODEL_ID = "sixtrack_sweetspot_p7_t3"
LITHOFACIES_MODEL_ID = "sixtrack_lithofacies_p76"
FACIES_MODEL_IDS = ("sixtrack_facies_f3_p81", "sixtrack_facies_penobscot_p81_reference")
PROPERTY_MODEL_IDS = {
    "sixtrack_property_tabicl_phif": "PHIF",
    "sixtrack_property_tabicl_klogh": "KLOGH",
    "sixtrack_property_tabicl_sw": "SW",
}
MODEL_VERSIONS = {
    FAULT_MODEL_ID: "p78-retain-incumbent-v1",
    RECONSTRUCTION_MODEL_ID: "p21-development-prepared-v1",
    LITHOFACIES_MODEL_ID: "p76-development-windows-v1",
    SWEETSPOT_MODEL_ID: "p7-development-observation-v1",
    **{model_id: "p81-development-native-v1" for model_id in FACIES_MODEL_IDS},
    **{model_id: "p67-frozen-tabicl-v1" for model_id in PROPERTY_MODEL_IDS},
}
ALL_MODEL_IDS = tuple(MODEL_VERSIONS)

FAULT_AXES = ("Z", "INLINE", "CROSSLINE")
FAULT_OPTION_KEYS = frozenset({"inline_range", "crossline_range", "sample_range", "axis_order"})
FAULT_MIN_EDGE = 3
DEVELOPMENT_FAMILIES = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12")
FROZEN_TEST_FAMILY = "15/9-F-15"
PROPERTY_OPTION_KEYS = frozenset({"query_family", "query_well_ids", "seed"})
CPU_DEVICES = frozenset({"", "auto", "cpu"})
_RAW_SOURCE_FIELDS = ("raw_well_paths", "raw_well_root", "trajectory_paths")
_LINEAGE_FIELDS = ("source_task_id", "registration_task_id", "prepared_view_task_id")


def _inclusive_range(options: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = options.get(key)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"options.{key} must be [first, last] integers")
    first, last = int(value[0]), int(value[1])
    if last - first + 1 < FAULT_MIN_EDGE:
        raise ValueError(f"options.{key} spans fewer than {FAULT_MIN_EDGE} samples (box too small)")
    return first, last


def _require_cpu(request: Mapping[str, Any]) -> None:
    device = str(request.get("device") or "").strip().casefold()
    if device not in CPU_DEVICES:
        raise ValueError("six-track pilot models run on CPU only; use device=auto or cpu")


def _reject_fields(request: Mapping[str, Any], fields: tuple[str, ...], reason: str) -> None:
    supplied = [name for name in fields if request.get(name)]
    if supplied:
        raise ValueError(f"{reason}: {', '.join(supplied)}")


def _reject_unknown_options(options: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(str(key) for key in options if key not in allowed)
    if unknown:
        raise ValueError(f"unsupported options for this model: {', '.join(unknown)}")


def validate_fault_request(request: Mapping[str, Any]) -> None:
    """Reject malformed fault requests; the SEG-Y itself is checked by the platform."""

    _require_cpu(request)
    _reject_fields(request, _RAW_SOURCE_FIELDS, "fault model consumes one SEG-Y only")
    # Snapshot-bound runs would carry header-byte semantics this worker does not
    # consume (it always addresses traces by INLINE_3D/CROSSLINE_3D), so they are
    # refused explicitly instead of being run with silently ignored metadata.
    _reject_fields(
        request,
        _LINEAGE_FIELDS,
        "fault extension does not support snapshot-bound runs yet; submit seismic_path directly",
    )
    if request.get("patch_size") or request.get("overlap"):
        raise ValueError("fault baseline works per inline slice; patch_size/overlap are not used")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, FAULT_OPTION_KEYS)
    axis_order = options.get("axis_order")
    if axis_order is not None and [str(item).upper() for item in axis_order] != list(FAULT_AXES):
        raise ValueError(f"axis_order must be {list(FAULT_AXES)}")
    range_keys = ("inline_range", "crossline_range", "sample_range")
    given = [key for key in range_keys if key in options]
    crop_start, crop_size = request.get("crop_start"), request.get("crop_size")
    if given:
        if len(given) != len(range_keys):
            raise ValueError("inline_range, crossline_range and sample_range must be given together")
        if crop_start is not None or crop_size is not None:
            raise ValueError("options.*_range and crop_start/crop_size are mutually exclusive")
        ranges = [_inclusive_range(options, key) for key in range_keys]
        if ranges[2][0] < 0:
            raise ValueError("options.sample_range must be non-negative sample indices")
    elif crop_start is not None or crop_size is not None:
        if crop_start is None or crop_size is None:
            raise ValueError("crop_start and crop_size must be given together")
        if any(int(value) < 0 for value in crop_start):
            raise ValueError("crop_start must be non-negative [Z, INLINE, CROSSLINE] indices")
        if any(int(value) < FAULT_MIN_EDGE for value in crop_size):
            raise ValueError(f"crop_size must be at least {FAULT_MIN_EDGE} on every axis (box too small)")
    else:
        raise ValueError(
            "an explicit box is required: options.inline_range/crossline_range/sample_range "
            "or crop_start/crop_size"
        )


def validate_property_request(request: Mapping[str, Any]) -> None:
    """Dataset-bound only: no raw paths, no snapshot lineage, no frozen-test family."""

    _require_cpu(request)
    if str(request.get("seismic_path") or "").strip():
        raise ValueError("dataset-bound property model does not accept seismic_path")
    _reject_fields(request, _RAW_SOURCE_FIELDS, "dataset-bound property model does not accept raw data paths")
    _reject_fields(request, _LINEAGE_FIELDS, "dataset-bound property model does not consume snapshot lineage")
    if any(request.get(name) is not None for name in ("crop_start", "crop_size", "patch_size", "overlap")):
        raise ValueError("crop/patch fields do not apply to the dataset-bound property model")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, PROPERTY_OPTION_KEYS)
    family = str(options.get("query_family") or "").strip()
    if not family:
        raise ValueError(
            "options.query_family is required; one of " + ", ".join(DEVELOPMENT_FAMILIES)
        )
    if family == FROZEN_TEST_FAMILY:
        raise ValueError("15/9-F-15 is the frozen test family and is never predicted")
    if family not in DEVELOPMENT_FAMILIES:
        raise ValueError("options.query_family must be one of " + ", ".join(DEVELOPMENT_FAMILIES))
    wells = options.get("query_well_ids")
    if wells is not None and (
        not isinstance(wells, list) or not wells or not all(isinstance(item, str) and item for item in wells)
    ):
        raise ValueError("options.query_well_ids must be a non-empty list of well ids")
    seed = options.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("options.seed must be a non-negative integer")


def validate_reconstruction_request(request: Mapping[str, Any]) -> None:
    """Only select a pinned original fold; never silently accept new-area data."""
    _require_cpu(request)
    _reject_fields(request, ("seismic_path", *_RAW_SOURCE_FIELDS, *_LINEAGE_FIELDS),
                   "dataset-bound P21 does not accept raw data or snapshot lineage")
    if any(request.get(name) is not None for name in ("crop_start", "crop_size", "patch_size", "overlap", "threshold")):
        raise ValueError("crop/patch/threshold fields do not apply to dataset-bound P21")
    options = dict(request.get("options") or {})
    _reject_unknown_options(options, frozenset({"fold_id"}))
    fold = options.get("fold_id")
    if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
        raise ValueError("options.fold_id must be an integer 0..4")
