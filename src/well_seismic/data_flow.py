"""Machine-readable data-flow contracts for runnable downstream models.

The geological task registry answers *what* a task means.  This module answers
*which sealed data products one concrete model actually needs*.  Keeping the
contract at model level is important because a single task can have both a
well-only fast model and a registration-backed fusion model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

PRIMARY_PREDICTION_TASK_IDS = (
    "fault",
    "horizon",
    "well_property",
    "fluid_interpretation",
    "facies_1d",
    "facies_3d",
    "fracture_development",
)

_OUTPUT_CONTRACTS = {
    "alignment": "well-seismic.registration.v3-candidate",
    "fault": "well-seismic.fault-volume.v1",
    "horizon": "well-seismic.stratigraphic-instance-volume.v1",
    "well_property": "well-seismic.well-property-curves.v1",
    "fluid_interpretation": "well-seismic.fluid-classes.v1",
    "facies_1d": "well-seismic.well-facies-sequence.v1",
    "facies_3d": "well-seismic.facies-volume.v1",
    "fracture_development": "well-seismic.fracture-development.v1",
}

DATA_FLOW_TASK_IDS = (*PRIMARY_PREDICTION_TASK_IDS, "alignment")

# New platform runs for well-side downstream interpretation always consume the
# active sealed SourceSnapshot (and, when applicable, its verified
# PreparedView).  Legacy raw-file and registered-dataset results remain
# readable, but those source modes are not part of the public run contract.
SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS = frozenset(
    {"well_property", "fluid_interpretation", "facies_1d", "fracture_development"}
)


@dataclass(frozen=True)
class ModelDataFlowSpec:
    """Stable public input dependency contract for one model runner."""

    model_id: str
    task_id: str
    source_modes: tuple[str, ...]
    target_source_modes: tuple[str, ...]
    required_modalities: tuple[str, ...]
    optional_modalities: tuple[str, ...]
    registration_policy: str
    prepared_view_policy: str
    prepared_view_consumed: bool
    accepted_domains: tuple[str, ...]
    output_contract: str
    degradation_policy: str
    adapter_registered: bool
    runner_registered: bool
    runnable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boolean(capability: Mapping[str, Any], metadata: Mapping[str, Any], key: str) -> bool:
    if key in capability:
        return bool(capability[key])
    return bool(metadata.get(key))


def _requires_seismic(capability: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    if "requires_seismic" in capability or "requires_seismic" in metadata:
        return _boolean(capability, metadata, "requires_seismic")
    formats = {str(value).casefold() for value in capability.get("source_formats") or ()}
    return bool(formats & {"sgy", "segy"})


def _registration_policy(
    *,
    task_id: str,
    capability: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    explicit = metadata.get("registration_policy")
    if explicit in {"none", "optional_control", "required"}:
        return str(explicit)
    if _boolean(capability, metadata, "requires_registration"):
        return "required"
    return "none"


def _modalities(
    *,
    task_id: str,
    capability: Mapping[str, Any],
    metadata: Mapping[str, Any],
    registration_policy: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required: list[str] = []
    optional: list[str] = []
    if _requires_seismic(capability, metadata):
        required.append("seismic")
    if _boolean(capability, metadata, "requires_logs"):
        required.append("well_log")
    if _boolean(capability, metadata, "requires_complete_trajectory"):
        required.append("trajectory")
    elif task_id in {"well_property", "fluid_interpretation", "facies_1d", "fracture_development"}:
        optional.append("trajectory")
    if registration_policy == "required":
        required.append("registration")
    elif registration_policy == "optional_control":
        optional.append("registration_control")
    if task_id == "facies_3d":
        optional.append("well_facies_context")
    return tuple(dict.fromkeys(required)), tuple(dict.fromkeys(optional))


def _source_modes(
    *,
    requires_seismic: bool,
    registration_policy: str,
    capability: Mapping[str, Any],
) -> tuple[str, ...]:
    modes: list[str] = []
    if (
        requires_seismic
        or registration_policy == "required"
        or bool(capability.get("supports_snapshot_wells"))
    ):
        modes.append("sealed_snapshot")
    if bool(capability.get("supports_raw_wells")):
        modes.append("explicit_raw")
    if bool(capability.get("supports_dataset_bound")):
        modes.append("registered_dataset")
    if not modes:
        # Seismic adapters that predate the explicit flag still consume a
        # snapshot-bound SEG-Y path.
        modes.append("sealed_snapshot")
    return tuple(modes)


def _degradation_policy(task_id: str, registration_policy: str) -> str:
    if registration_policy == "required":
        return "fail_closed_when_registration_is_missing_or_unattested"
    if task_id == "facies_3d":
        return "cross-survey transfer must remain candidate or caution"
    if task_id == "fracture_development":
        return "out-of-domain MD mapping must remain experimental"
    return "missing optional modalities are reported; required modalities fail closed"


def build_model_data_flow_specs(
    model_specs: Iterable[Any],
    adapter_capabilities: Iterable[Mapping[str, Any]],
    runner_model_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Compose current model dependencies from registries without importing runners."""

    adapter_by_id = {
        str(item.get("model_id")): dict(item)
        for item in adapter_capabilities
        if item.get("model_id")
    }
    runner_ids = {str(value) for value in runner_model_ids}
    rows: list[ModelDataFlowSpec] = []
    for model in model_specs:
        metadata = dict(getattr(model, "metadata", {}) or {})
        task_id = str(metadata.get("prediction_task") or "")
        if task_id not in DATA_FLOW_TASK_IDS:
            continue
        model_id = str(model.id)
        capability = adapter_by_id.get(model_id, {})
        registration_policy = _registration_policy(
            task_id=task_id,
            capability=capability,
            metadata=metadata,
        )
        requires_seismic = _requires_seismic(capability, metadata)
        required, optional = _modalities(
            task_id=task_id,
            capability=capability,
            metadata=metadata,
            registration_policy=registration_policy,
        )
        source_modes = (
            ("sealed_snapshot",)
            if task_id in SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS
            else _source_modes(
                requires_seismic=requires_seismic,
                registration_policy=registration_policy,
                capability=capability,
            )
        )
        prepared_consumed = bool(metadata.get("prepared_view_consumed", False))
        prepared_view_policy = str(
            metadata.get("prepared_view_policy") or "optional"
        )
        if prepared_view_policy not in {"none", "optional", "preferred", "required"}:
            raise ValueError(
                f"unsupported prepared_view_policy for {model_id}: "
                f"{prepared_view_policy}"
            )
        rows.append(
            ModelDataFlowSpec(
                model_id=model_id,
                task_id=task_id,
                source_modes=source_modes,
                target_source_modes=tuple(
                    dict.fromkeys(("sealed_snapshot", *source_modes))
                ),
                required_modalities=required,
                optional_modalities=optional,
                registration_policy=registration_policy,
                prepared_view_policy=prepared_view_policy,
                prepared_view_consumed=prepared_consumed,
                accepted_domains=("TWT_MS",)
                if requires_seismic
                else ("MD_M",),
                output_contract=_OUTPUT_CONTRACTS[task_id],
                degradation_policy=_degradation_policy(
                    task_id, registration_policy
                ),
                adapter_registered=model_id in adapter_by_id,
                runner_registered=model_id in runner_ids,
                runnable=(
                    str(getattr(model, "runtime_status", "")) == "runnable"
                    and model_id in adapter_by_id
                    and model_id in runner_ids
                ),
            )
        )
    task_order = {
        task_id: index for index, task_id in enumerate(DATA_FLOW_TASK_IDS)
    }
    rows.sort(key=lambda item: (task_order[item.task_id], item.model_id))
    return [item.to_dict() for item in rows]


__all__ = [
    "PRIMARY_PREDICTION_TASK_IDS",
    "DATA_FLOW_TASK_IDS",
    "SNAPSHOT_ONLY_DOWNSTREAM_WELL_TASK_IDS",
    "ModelDataFlowSpec",
    "build_model_data_flow_specs",
]
