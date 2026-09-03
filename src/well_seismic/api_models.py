"""Stable HTTP request and response contracts for the platform API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SURVEY_ATTESTATION_CONTRACT_VERSION = "well-seismic.survey-attestation.v1"
REGISTRATION_PREFLIGHT_CONTRACT_VERSION = (
    "well-seismic.registration-preflight.v1"
)
RUNTIME_CONTRACT_REVIEW_VERSION = "well-seismic.runtime-contract-review.v1"
RUNTIME_CONTRACT_CONFIRMATION_VERSION = (
    "well-seismic.runtime-contract-confirmation.v1"
)


def survey_attestation_declaration(srd_elevation_m: float) -> str:
    """Render one unambiguous statement from the structured survey contract."""

    value = float(srd_elevation_m)
    if not isfinite(value):
        raise ValueError("survey attestation SRD elevation must be finite")
    if value == 0:
        value = 0.0
    rendered = format(value, ".15g")
    return (
        "本人确认：本次数据准备中登记的全部 SEG-Y 地震数据，其处理基准面（SRD）"
        f"为平均海平面（MSL）{rendered} m，地震时间域为 TWT，且均已校正到该 SRD。"
    )


class SurveyAttestation(BaseModel):
    """An explicit human declaration for otherwise unavailable survey metadata.

    The fixed statement prevents a generic acknowledgement from being mistaken
    for a physical-data contract.  ``source`` deliberately has no LLM/Kimi
    option: an automated suggestion can prepare the form but cannot sign it.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["well-seismic.survey-attestation.v1"] = (
        SURVEY_ATTESTATION_CONTRACT_VERSION
    )
    declared_srd_elevation_m: float = Field(allow_inf_nan=False)
    vertical_reference: Literal["MSL"] = "MSL"
    time_domain: Literal["TWT"] = "TWT"
    correction_state: Literal["corrected_to_srd"] = "corrected_to_srd"
    declaration_text: str | None = None
    source: Literal["human_user"] = "human_user"
    confirmation_channel: Literal["user_ui", "user_api"] = "user_ui"
    confirmed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("confirmed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("survey attestation confirmed_at must include timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def render_or_validate_declaration(self) -> "SurveyAttestation":
        expected = survey_attestation_declaration(self.declared_srd_elevation_m)
        if self.declaration_text is None:
            self.declaration_text = expected
        elif self.declaration_text != expected:
            raise ValueError(
                "survey attestation declaration text does not match structured fields"
            )
        return self


class InspectionRequest(BaseModel):
    seismic_paths: list[str] = Field(default_factory=list)
    survey_paths: list[str] = Field(default_factory=list)
    log_paths: list[str] = Field(default_factory=list)
    well_paths: list[str] = Field(default_factory=list)
    # Optional checkshot/VSP/time-depth controls are kept separate from survey
    # geometry so their well/time semantics can be parsed and audited.
    time_depth_paths: list[str] = Field(default_factory=list)
    interpretation_paths: list[str] = Field(default_factory=list)
    auxiliary_paths: list[str] = Field(default_factory=list)
    recursive: bool = True
    lightweight: bool = True
    use_llm_fallback: bool = False
    seismic_srd_elevation_m: float | None = None
    vertical_crs_id: str | None = None
    # Canonical projected-metre target, normally the SEG-Y survey CRS.
    horizontal_crs_id: str | None = None
    # Optional source overrides.  When omitted, explicit file evidence (for
    # example Petrel DEV EPSG) wins, otherwise source is assumed to equal target.
    well_source_crs_id: str | None = None
    seismic_source_crs_id: str | None = None
    horizontal_unit: Literal["m", "ft", "unknown"] = "unknown"
    horizontal_axis_order: Literal["XY", "YX", "unknown"] = "unknown"
    coordinate_reference_verified: bool = False
    seismic_replacement_velocity_mps: float | None = Field(default=None, gt=0)
    seismic_time_domain: Literal["TWT", "OWT", "unknown"] = "unknown"
    seismic_correction_state: Literal["corrected_to_srd", "uncorrected", "unknown"] = (
        "unknown"
    )
    # Advanced source-contract overrides.  These are intentionally separate
    # from the canonical CRS/time declarations above: e.g. a source table may
    # be in feet while the sealed platform coordinate contract is metres.
    segy_geometry_profile: str | None = None
    segy_inline_byte: int | None = Field(default=None, ge=1, le=237)
    segy_crossline_byte: int | None = Field(default=None, ge=1, le=237)
    segy_x_byte: int | None = Field(default=None, ge=1, le=237)
    segy_y_byte: int | None = Field(default=None, ge=1, le=237)
    segy_coordinate_scalar_byte: int | None = Field(default=None, ge=1, le=239)
    well_coordinate_source_unit: Literal["m", "ft", "unknown"] | None = None
    # Source unit for well-head elevation/datum fields such as KB, GL, DF and
    # RT.  This is deliberately independent from X/Y units: projected
    # coordinates and well datums can legitimately use different units.
    well_vertical_datum_source_unit: Literal["m", "ft", "unknown"] | None = None
    las_twt_source_unit: Literal["ms", "s", "us", "unknown"] | None = None
    # Human-reviewed fallbacks are applied only when a parsed time-depth table
    # does not declare the corresponding semantic.  They never create a
    # time-depth asset and never override explicit file evidence.
    time_depth_default_depth_domain: Literal[
        "md", "tvd", "tvdss", "unknown"
    ] | None = None
    time_depth_default_depth_unit: Literal["m", "ft", "unknown"] | None = None
    time_depth_default_time_unit: Literal["ms", "s", "us", "unknown"] | None = None
    time_depth_default_depth_datum: Literal[
        "KB", "GL", "DF", "RT", "MSL", "unknown"
    ] | None = None
    time_depth_default_depth_convention: Literal[
        "depth_below_msl_positive_down", "elevation_positive_up", "unknown"
    ] | None = None
    time_depth_default_time_reference: Literal[
        "SRD", "KB", "GL", "DF", "RT", "unknown"
    ] | None = None
    time_depth_default_time_domain: Literal["TWT", "OWT", "unknown"] | None = None
    time_depth_default_correction_state: Literal[
        "corrected_to_srd", "uncorrected", "unknown"
    ] | None = None
    target_task_id: str | None = None
    target_model_id: str | None = None
    target_scope_explicit: bool = False
    survey_attestation: SurveyAttestation | None = None

    @model_validator(mode="after")
    def validate_survey_attestation_contract(self) -> "InspectionRequest":
        if self.survey_attestation is None:
            return self
        if self.seismic_srd_elevation_m is None or float(
            self.seismic_srd_elevation_m
        ) != float(self.survey_attestation.declared_srd_elevation_m):
            raise ValueError(
                "survey attestation SRD elevation must equal the request contract"
            )
        if self.seismic_time_domain != self.survey_attestation.time_domain:
            raise ValueError(
                "survey attestation time domain must equal the request contract"
            )
        if self.seismic_correction_state != self.survey_attestation.correction_state:
            raise ValueError(
                "survey attestation correction state must equal the request contract"
            )
        if "MSL" not in str(self.vertical_crs_id or "").upper():
            raise ValueError(
                "survey attestation requires an explicit MSL vertical_crs_id"
            )
        return self


class InputDiscoveryRequest(BaseModel):
    root_path: str
    recursive: bool = True
    use_llm_fallback: bool = False
    max_files: int = Field(default=20000, ge=1, le=100000)


class ActiveSnapshotRequest(BaseModel):
    """Persist the desktop project's currently selected immutable snapshot."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)

    @field_validator("snapshot_id")
    @classmethod
    def normalize_snapshot_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("snapshot_id must not be empty")
        return normalized


class PreprocessingRequest(InspectionRequest):
    output_directory: str | None = None
    source_snapshot_id: str | None = None
    registration_task_id: str | None = None


class HorizontalRegistrationRequest(BaseModel):
    """Create a plan-view derivative from one sealed source snapshot.

    Paths and coordinate semantics are intentionally not repeated here.  The
    worker rehydrates the exact sealed preparation request, which removes a
    common source of semantic-drift failures and prevents callers from silently
    reinterpreting the same XY numbers under another CRS.
    """

    model_config = ConfigDict(extra="forbid")

    source_snapshot_id: str = Field(min_length=1)
    output_directory: str | None = None


class RegistrationPreflightRequest(BaseModel):
    """Resolve one sealed snapshot into a registration-ready immutable snapshot."""

    model_config = ConfigDict(extra="forbid")

    source_snapshot_id: str = Field(min_length=1)


class RuntimeContractValues(BaseModel):
    """Editable values accepted from the concise post-ingest review dialog."""

    model_config = ConfigDict(extra="forbid")

    horizontal_crs_id: str = Field(min_length=1)
    horizontal_unit: Literal["m"]
    horizontal_axis_order: Literal["XY", "YX"]
    coordinate_reference_verified: Literal[True]
    vertical_crs_id: str = Field(min_length=1)
    seismic_srd_elevation_m: float = Field(allow_inf_nan=False)
    seismic_time_domain: Literal["TWT"]
    seismic_correction_state: Literal["corrected_to_srd"]
    seismic_replacement_velocity_mps: float = Field(gt=0, allow_inf_nan=False)
    well_coordinate_source_unit: Literal["m", "ft"]
    well_vertical_datum_source_unit: Literal["m", "ft"]
    time_depth_default_depth_domain: Literal["md", "tvd", "tvdss"] | None = None
    time_depth_default_depth_unit: Literal["m", "ft"] | None = None
    time_depth_default_time_unit: Literal["ms", "s", "us"] | None = None
    time_depth_default_depth_datum: Literal[
        "KB", "GL", "DF", "RT", "MSL"
    ] | None = None
    time_depth_default_depth_convention: Literal[
        "depth_below_msl_positive_down", "elevation_positive_up"
    ] | None = None
    time_depth_default_time_reference: Literal[
        "SRD", "KB", "GL", "DF", "RT"
    ] | None = None
    time_depth_default_time_domain: Literal["TWT", "OWT"] | None = None
    time_depth_default_correction_state: Literal[
        "corrected_to_srd", "uncorrected"
    ] | None = None

    @model_validator(mode="before")
    @classmethod
    def require_explicit_non_null_fields(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        required = {
            "horizontal_crs_id",
            "horizontal_unit",
            "horizontal_axis_order",
            "coordinate_reference_verified",
            "vertical_crs_id",
            "seismic_srd_elevation_m",
            "seismic_time_domain",
            "seismic_correction_state",
            "seismic_replacement_velocity_mps",
            "well_coordinate_source_unit",
            "well_vertical_datum_source_unit",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(
                "runtime contract values must be submitted explicitly; missing: "
                + ", ".join(sorted(missing))
            )
        null_time_depth = sorted(
            key
            for key, item in value.items()
            if str(key).startswith("time_depth_default_") and item is None
        )
        if null_time_depth:
            raise ValueError(
                "runtime contract time-depth fields cannot be null: "
                + ", ".join(null_time_depth)
            )
        return value

    @model_validator(mode="after")
    def validate_time_depth_defaults(self) -> "RuntimeContractValues":
        state = self.time_depth_default_correction_state
        reference = self.time_depth_default_time_reference
        if state == "corrected_to_srd" and reference not in {None, "SRD"}:
            raise ValueError(
                "corrected_to_srd time-depth defaults require time_reference=SRD"
            )
        if state == "uncorrected" and reference == "SRD":
            raise ValueError(
                "uncorrected time-depth defaults require a well datum reference"
            )
        return self


class RuntimeContractConfirmationRequest(BaseModel):
    """Confirm editable defaults for one already-sealed source snapshot."""

    model_config = ConfigDict(extra="forbid")

    source_snapshot_id: str = Field(min_length=1)
    values: RuntimeContractValues
    confirmation: Literal["CONFIRM_RUNTIME_CONTRACT"]
    attestation: SurveyAttestation

    @model_validator(mode="before")
    @classmethod
    def require_explicit_user_attestation(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw_attestation = value.get("attestation")
        if not isinstance(raw_attestation, Mapping):
            raise ValueError("runtime contract requires an explicit user attestation")
        required = {
            "contract_version",
            "declared_srd_elevation_m",
            "vertical_reference",
            "time_domain",
            "correction_state",
            "declaration_text",
            "source",
            "confirmation_channel",
            "confirmed_at",
        }
        if not required.issubset(raw_attestation):
            missing = ", ".join(sorted(required - set(raw_attestation)))
            raise ValueError(
                "runtime contract user attestation must be fully supplied; "
                f"missing: {missing}"
            )
        if not str(raw_attestation.get("declaration_text") or "").strip():
            raise ValueError("runtime contract declaration_text must not be empty")
        if not str(raw_attestation.get("confirmed_at") or "").strip():
            raise ValueError("runtime contract confirmed_at must be client supplied")
        return value

    @model_validator(mode="after")
    def bind_attestation_to_values(self) -> "RuntimeContractConfirmationRequest":
        if float(self.attestation.declared_srd_elevation_m) != float(
            self.values.seismic_srd_elevation_m
        ):
            raise ValueError("runtime attestation SRD differs from confirmed values")
        if self.attestation.time_domain != self.values.seismic_time_domain:
            raise ValueError("runtime attestation time domain differs from values")
        if self.attestation.correction_state != (
            self.values.seismic_correction_state
        ):
            raise ValueError("runtime attestation correction state differs from values")
        if "MSL" not in self.values.vertical_crs_id.upper():
            raise ValueError("runtime attestation requires an MSL vertical_crs_id")
        return self


class RuntimeContractConfirmationResponse(BaseModel):
    """Immutable derived snapshot created from one human confirmation."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["well-seismic.runtime-contract-confirmation.v1"] = (
        RUNTIME_CONTRACT_CONFIRMATION_VERSION
    )
    source_snapshot_id: str
    derived_snapshot_id: str
    effective_request: PreprocessingRequest
    runtime_contract_receipt: dict[str, Any]


class RegistrationPreflightResponse(BaseModel):
    """Registration-ready request returned without weakening the registration API."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["well-seismic.registration-preflight.v1"] = (
        REGISTRATION_PREFLIGHT_CONTRACT_VERSION
    )
    source_snapshot_id: str
    derived_snapshot_id: str
    resolution: Literal[
        "reused_complete_snapshot",
        "derived_bounded_machine_decision",
        "reused_native_relative_snapshot",
    ]
    effective_request: PreprocessingRequest
    system_evidence_receipt: dict[str, Any] | None = None
    execution_contract: dict[str, Any] | None = None


class RegistrationPreflightFailureDetail(BaseModel):
    """Machine-actionable reason a sealed snapshot cannot enter formal tie."""

    model_config = ConfigDict(extra="forbid")

    code: str
    category: Literal[
        "formal_contract_unavailable",
        "source_snapshot_integrity",
        "source_snapshot_semantic_drift",
        "source_snapshot_unavailable",
        "source_quality_unavailable",
        "horizontal_contract_unavailable",
    ]
    horizontal_fallback_allowed: bool
    requires_new_snapshot: bool
    missing_fields: list[str] = Field(default_factory=list)
    reason: str


class PredictionRequest(BaseModel):
    """Model-neutral prediction request.

    Common volume controls remain first-class fields for backward compatibility.
    Task-specific runners can consume validated values from ``options``.
    """

    task_id: str = "fault"
    # Omit the model when the caller wants the platform to resolve the safest
    # runnable default for the selected geological task.  A fixed FaultSeg
    # default made otherwise valid requests for every other task fail with a
    # misleading task/model mismatch.
    model_id: str | None = None
    # Volume models require this field. Dataset-bound well models intentionally
    # leave it empty and resolve their registered well asset inside the runner.
    seismic_path: str = ""
    # Competition well-side tasks can run without SEG-Y.  Paths are explicit
    # first-class fields so callers do not have to smuggle local assets through
    # model-specific options.  ``raw_well_root`` lets the WellFuse CLI perform
    # its own deterministic recursive discovery; individual paths take
    # precedence when both are supplied.
    raw_well_paths: list[str] = Field(default_factory=list)
    raw_well_root: str | None = None
    trajectory_paths: list[str] = Field(default_factory=list)
    source_task_id: str | None = None
    # Optional derived view created by the sample-building stage.  The source
    # snapshot remains the immutable identity; this id only binds a verified
    # derivative and must never replace ``source_task_id``.
    prepared_view_task_id: str | None = None
    # Models that consume wells must be bound to the immutable registration
    # product created from the same data snapshot.  Keeping this lineage as a
    # first-class field prevents task-specific ``options`` from silently
    # substituting an unrelated time/depth table.
    registration_task_id: str | None = None
    crop_start: tuple[int, int, int] | None = None
    crop_size: tuple[int, int, int] | None = None
    patch_size: tuple[int, int, int] | None = None
    overlap: tuple[int, int, int] | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    device: str = "auto"
    output_directory: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_direct12b_supervision_parameters(cls, value: Any) -> Any:
        """Reject even unknown/nested supervision fields before Pydantic drops extras."""

        if not isinstance(value, Mapping):
            return value
        if value.get("model_id") != "WellFuse-GeoAlign-12B-Direct-v1":
            return value

        keys: list[str] = []

        def collect(item: object) -> None:
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    keys.append(
                        str(key).strip().casefold().replace("-", "_").replace(" ", "_")
                    )
                    collect(nested)
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                for nested in item:
                    collect(nested)

        collect(value)
        fragments = (
            "checkshot",
            "time_depth",
            "timedepth",
            "td_table",
            "target_time",
            "target_twt",
            "vsp",
        )
        forbidden = sorted(
            {key for key in keys if any(fragment in key for fragment in fragments)}
        )
        if forbidden:
            raise ValueError(
                "多模态井震对齐推理不接受TD/checkshot/VSP参数: "
                + ", ".join(forbidden)
            )
        return value


class TaskCreated(BaseModel):
    task_id: str
    status: str
    message: str


class SystemCacheClearRequest(BaseModel):
    """Explicit same-origin JSON confirmation for regenerable cache cleanup."""

    confirmation: Literal["CLEAR_REGENERABLE_CACHE"]


class RegistrationCandidateAcceptanceRequest(BaseModel):
    """Explicit human promotion of selected GeoPath candidate wells.

    Candidate inference intentionally writes ``fusion_ready=false``.  The
    caller must bind the review to the exact candidate manifest and name the
    wells being accepted; there is no implicit "accept all" path.
    """

    accepted_well_ids: list[str] = Field(min_length=1)
    expected_candidate_manifest_sha256: str = Field(min_length=64, max_length=64)
    confirmation: Literal["ACCEPT_GEOPATH_CANDIDATE"]
    review_note: str = Field(default="", max_length=1000)


class ViserSliceRequest(BaseModel):
    task_id: str
    asset_index: int = Field(default=0, ge=0)
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    z: int | None = Field(default=None, ge=0)


class ViserLayerModeRequest(BaseModel):
    task_id: str
    asset_index: int = Field(default=0, ge=0)
    mode: Literal["combined", "prediction"] = "combined"


class IssueConfirmationRequest(BaseModel):
    decision: str
    action: str = ""


class BatchIssueActionRequest(BaseModel):
    mode: Literal["apply_recommended", "rollback"] = "apply_recommended"
    batch_id: str | None = None
    stages: list[str] = Field(default_factory=list)


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    task_id: str | None = None


class TransformationActivationRequest(BaseModel):
    confirmation: str
