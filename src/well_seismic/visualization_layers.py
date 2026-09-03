"""Stable, model-neutral descriptors for visualization and exported artifacts.

Model runners and imported frozen releases describe *what* an artifact means.
Rendering engines decide *how* to draw it.  Keeping this contract free of
NumPy/PyTorch objects also makes it safe to persist in task manifests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePath
from typing import Any, Literal
from urllib.parse import quote


LayerKind = Literal[
    "volume",
    "surface",
    "points",
    "trajectory",
    "well_curve",
    "table",
    "report",
]
LayerRole = Literal[
    "source",
    "baseline",
    "prediction",
    "observation",
    "uncertainty",
    "valid_mask",
    "quality",
    "evidence",
]


@dataclass(frozen=True)
class VisualizationLayerDescriptor:
    """Portable description of one displayable result layer.

    ``artifact_id`` is intentionally an opaque catalog identifier.  Browsers
    never receive or construct an arbitrary local path.
    """

    id: str
    name: str
    kind: LayerKind
    role: LayerRole
    artifact_id: str
    access_url: str | None = None
    axis_order: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provisional: bool = False
    visible_by_default: bool = True

    def __post_init__(self) -> None:
        for field_name in ("id", "name", "artifact_id"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"visualization layer {field_name} cannot be empty")
        if len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError("visualization layer axis_order contains duplicates")
        if self.kind in {"volume", "surface", "points", "trajectory"}:
            coordinate_reference = self.geometry.get("crs") or self.geometry.get(
                "coordinate_reference"
            )
            if not coordinate_reference:
                raise ValueError(
                    f"{self.kind} layer must declare geometry.crs or "
                    "geometry.coordinate_reference"
                )
        if self.role == "uncertainty" and not self.uncertainty:
            raise ValueError("uncertainty layer must declare uncertainty semantics")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axis_order"] = list(self.axis_order)
        return payload


@dataclass(frozen=True)
class ArtifactBundleDescriptor:
    """A complete, immutable output bundle from one release or inference run."""

    id: str
    task_id: str
    release_id: str
    layers: tuple[VisualizationLayerDescriptor, ...]
    manifest_artifact_id: str
    source_snapshot_id: str | None = None
    scientific_status: str = "candidate"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.task_id.strip() or not self.release_id.strip():
            raise ValueError("artifact bundle id, task_id and release_id are required")
        layer_ids = [layer.id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("artifact bundle layer ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layers"] = [layer.to_dict() for layer in self.layers]
        payload["warnings"] = list(self.warnings)
        return payload


_VOLUME_SUFFIXES = frozenset({".npy", ".npz", ".zarr", ".sgy", ".segy"})
_SURFACE_SUFFIXES = frozenset({".xyz", ".obj", ".ply", ".vtp", ".vtk"})
_REPORT_SUFFIXES = frozenset({".json", ".md", ".txt", ".log", ".yaml", ".yml"})
_TABLE_SUFFIXES = frozenset({".csv", ".tsv"})
_WELL_CURVE_SUFFIXES = frozenset({".las"})


def _identifier_component(value: object, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    return text or fallback


def _artifact_path(value: object) -> str | None:
    """Return a declared artifact path without resolving or opening it."""

    if isinstance(value, (str, PurePath)):
        path = str(value).strip()
        return path or None
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        if isinstance(raw_path, (str, PurePath)):
            path = str(raw_path).strip()
            return path or None
    return None


def _declared_format(output_key: str, path: str) -> str:
    suffix = PurePath(path).suffix.lower()
    if suffix:
        return suffix.lstrip(".")
    key = output_key.lower()
    for marker in ("npy", "npz", "zarr", "sgy", "segy", "json", "csv", "las"):
        if key == marker or key.endswith(f"_{marker}"):
            return marker
    return ""


def _explicit_array_descriptor(value: object) -> tuple[str, tuple[str, ...]] | None:
    """Return the selected NPZ member and its axes when explicitly declared.

    An NPZ file is a container, not an array.  Requiring both fields prevents a
    raw/model-input package (or a Direct12/well result archive) from being
    advertised as one directly renderable 3-D volume merely because its file
    name ends in ``.npz``.
    """

    if not isinstance(value, Mapping):
        return None
    array_key = str(value.get("array_key") or "").strip()
    raw_axes = value.get("axis_order") or value.get("axes")
    if (
        not array_key
        or not isinstance(raw_axes, Sequence)
        or isinstance(raw_axes, (str, bytes))
    ):
        return None
    axes = tuple(str(axis).strip() for axis in raw_axes if str(axis).strip())
    if not axes or len(axes) != len(raw_axes):
        return None
    return array_key, axes


def _layer_kind(output_key: str, path: str, value: object = None) -> LayerKind | None:
    key = output_key.lower()
    suffix = PurePath(path).suffix.lower()
    # A canonical raw-well NPZ is an input/evidence package on the MD axis,
    # not a spatial prediction volume.  Treating it as a volume makes the UI
    # offer an invalid 3-D viewer for nine one-dimensional curves.
    if "canonical_input" in key and suffix == ".npz":
        return "table"
    # A trace-validity mask has only the two horizontal spatial axes.  It is a
    # support surface for volumes, not a TWT/depth volume of its own.
    if "valid_trace_mask" in key and suffix == ".npy":
        return "surface"
    if suffix in _REPORT_SUFFIXES or key.endswith(("_json", "_report", "_log")):
        return "report"
    if suffix in _TABLE_SUFFIXES:
        return "table"
    if suffix in _WELL_CURVE_SUFFIXES:
        return "well_curve"
    if suffix in _SURFACE_SUFFIXES:
        return "surface"
    if suffix == ".npz":
        explicit = _explicit_array_descriptor(value)
        if explicit is None:
            return "report"
        declared_kind = (
            str(value.get("layer_kind") or value.get("visualization_kind") or "")
            .strip()
            .casefold()
            if isinstance(value, Mapping)
            else ""
        )
        return "surface" if declared_kind == "surface" else "volume"
    if suffix in _VOLUME_SUFFIXES or key.endswith(
        ("_npy", "_npz", "_zarr", "_sgy", "_segy")
    ):
        # NPY/NPZ outputs from the preserved SurfaceSeg runner are dense
        # [INLINE, CROSSLINE, SAMPLE] label/confidence arrays, not picked
        # geometric surfaces.  Only an explicit surface-oriented key changes
        # an array artifact into a surface layer.
        if any(token in key for token in ("horizon", "surface_xyz", "surface_mesh")):
            return "surface"
        return "volume"
    return None


def _layer_role(output_key: str, kind: LayerKind) -> LayerRole:
    key = output_key.lower()
    # Fluid decisions are categorical prediction products even though CSV is
    # their most interoperable transport.  Probability and decision receipts
    # remain evidence, while deterministic point/interval tables are eligible
    # as the default result view.
    if kind == "table" and any(
        token in key for token in ("fluid_intervals_csv", "fluid_points_csv")
    ):
        return "prediction"
    if kind in {"report", "table"}:
        return "evidence"
    if key.startswith(("source_", "raw_", "input_")) or key in {
        "source",
        "raw",
        "input",
    }:
        return "source"
    if any(
        token in key
        for token in ("ground_truth", "reference_truth", "heldout_truth", "observation")
    ) or key.startswith("truth_"):
        return "observation"
    if key.startswith(("error_", "residual_", "difference_", "absolute_error_")) or any(
        token in key for token in ("_error_", "_residual_", "_absolute_difference_")
    ):
        return "quality"
    if any(
        token in key
        for token in (
            "valid_mask",
            "valid_trace_mask",
            "validity_mask",
            "coverage_mask",
            "effective_mask",
        )
    ):
        return "valid_mask"
    if any(
        token in key
        for token in (
            "uncertainty",
            "confidence",
            "sigma",
            "std",
            "variance",
            "aleatoric",
            "epistemic",
        )
    ):
        return "uncertainty"
    # A generic ``mask`` from FaultSeg is a binary prediction and a generic
    # ``mask`` from SurfaceSeg is a label prediction.  It must not silently be
    # re-labelled as data validity.
    return "prediction"


def _uncertainty_semantics(output_key: str) -> dict[str, Any]:
    key = output_key.lower()
    if "confidence" in key:
        return {
            "quantity": "confidence",
            "direction": "higher_is_more_certain",
            "expected_range": [0.0, 1.0],
        }
    if "variance" in key:
        quantity = "variance"
    elif "aleatoric" in key:
        quantity = "aleatoric_uncertainty"
    elif "epistemic" in key:
        quantity = "epistemic_uncertainty"
    elif "sigma" in key or "std" in key:
        quantity = "standard_deviation"
    else:
        quantity = "predictive_uncertainty"
    return {"quantity": quantity, "direction": "higher_is_less_certain"}


def _axes(result: Mapping[str, Any], model_id: str) -> tuple[str, ...]:
    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    segmentation = result.get("segmentation")
    segmentation = segmentation if isinstance(segmentation, Mapping) else {}
    if model_id == "seismic_surface_seg":
        declared = segmentation.get("axes") or input_metadata.get("axes")
    else:
        declared = input_metadata.get("axes") or input_metadata.get("model_order")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        return ()
    return tuple(str(value) for value in declared)


def _shape(result: Mapping[str, Any], model_id: str) -> list[int] | None:
    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    segmentation = result.get("segmentation")
    segmentation = segmentation if isinstance(segmentation, Mapping) else {}
    geometry = result.get("geometry")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    candidates: tuple[object, ...]
    if model_id == "seismic_surface_seg":
        candidates = (segmentation.get("shape_ics"), input_metadata.get("shape_ics"))
    else:
        # A bounded producer result describes the output ROI, while
        # ``shape_zyx`` can still be the complete source cube.
        candidates = (
            geometry.get("shape"),
            input_metadata.get("resolved_roi_size_zyx"),
            input_metadata.get("crop_size_zyx"),
            input_metadata.get("shape_zyx"),
        )
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            try:
                return [int(value) for value in candidate]
            except (TypeError, ValueError):
                continue
    return None


def _layer_axes(
    result: Mapping[str, Any],
    *,
    model_id: str,
    output_key: str,
    kind: LayerKind,
    value: object,
) -> tuple[str, ...]:
    explicit = _explicit_array_descriptor(value)
    if explicit is not None:
        return explicit[1]
    if kind not in {"volume", "surface"}:
        return ()
    base_axes = _axes(result, model_id)
    key = output_key.casefold()
    if "valid_trace_mask" in key:
        return tuple(
            axis
            for axis in base_axes
            if str(axis).strip().upper()
            not in {"Z", "T", "TIME", "TWT", "LOCAL_TWT", "DEPTH", "SAMPLE"}
        )
    # The F3 producer stores scalar volumes as [INLINE, XLINE, TWT] and the
    # probability tensor with one leading CLASS axis.  This is also the layout
    # consumed by facies_visualization.py.
    if model_id == "wellfuse_facies_3d_f3_fast":
        if "probability" in key:
            inference = result.get("inference")
            inference = inference if isinstance(inference, Mapping) else {}
            declared = inference.get("probability_axis_order")
            if isinstance(declared, Sequence) and not isinstance(
                declared, (str, bytes)
            ):
                return tuple(str(axis) for axis in declared)
            return ("CLASS", "INLINE", "XLINE", "TWT")
        return ("INLINE", "XLINE", "TWT")
    return base_axes


def _shape_for_axes(
    result: Mapping[str, Any],
    *,
    model_id: str,
    axes: tuple[str, ...],
) -> list[int] | None:
    base_shape = _shape(result, model_id)
    base_axes = _axes(result, model_id)
    if base_shape is None or not axes or len(base_shape) != len(base_axes):
        return base_shape
    dimensions = {
        str(axis).strip().upper(): int(size)
        for axis, size in zip(base_axes, base_shape, strict=True)
    }
    if "CROSSLINE" in dimensions:
        dimensions.setdefault("XLINE", dimensions["CROSSLINE"])
    if "XLINE" in dimensions:
        dimensions.setdefault("CROSSLINE", dimensions["XLINE"])
    if "CLASS" in axes:
        inference = result.get("inference")
        inference = inference if isinstance(inference, Mapping) else {}
        facies = result.get("facies")
        facies = facies if isinstance(facies, Mapping) else {}
        class_codes = inference.get("class_codes") or facies.get("class_codes")
        if isinstance(class_codes, Sequence) and not isinstance(
            class_codes, (str, bytes)
        ):
            dimensions["CLASS"] = len(class_codes)
    try:
        return [dimensions[str(axis).strip().upper()] for axis in axes]
    except KeyError:
        return base_shape


def _geometry(
    result: Mapping[str, Any],
    *,
    model_id: str,
    kind: LayerKind,
    axes: tuple[str, ...],
    value: object,
) -> dict[str, Any]:
    if kind not in {"volume", "surface", "points", "trajectory"}:
        return {}
    input_metadata = result.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    declared = result.get("geometry")
    declared = declared if isinstance(declared, Mapping) else {}
    coordinate_reference = (
        declared.get("crs")
        or declared.get("coordinate_reference")
        or input_metadata.get("crs")
        or input_metadata.get("coordinate_reference")
        or "source_seismic_grid_crs_unverified"
    )
    geometry: dict[str, Any] = {"coordinate_reference": str(coordinate_reference)}
    explicit_geometry = (
        value.get("geometry")
        if isinstance(value, Mapping) and isinstance(value.get("geometry"), Mapping)
        else {}
    )
    explicit_shape = (
        explicit_geometry.get("shape")
        if isinstance(explicit_geometry, Mapping)
        else None
    )
    if explicit_shape is None and isinstance(value, Mapping):
        explicit_shape = value.get("shape")
    shape: list[int] | None = None
    if isinstance(explicit_shape, Sequence) and not isinstance(
        explicit_shape, (str, bytes)
    ):
        try:
            shape = [int(item) for item in explicit_shape]
        except (TypeError, ValueError):
            shape = None
    if shape is None:
        shape = _shape_for_axes(result, model_id=model_id, axes=axes)
    if shape is not None:
        geometry["shape"] = shape
    for key in (
        "origin",
        "spacing",
        "inline_range",
        "crossline_range",
        "sample_interval_ms",
        "crop_start_zyx",
        "crop_size_zyx",
    ):
        geometry_value = explicit_geometry.get(
            key, declared.get(key, input_metadata.get(key))
        )
        if geometry_value is not None:
            geometry[key] = geometry_value
    return geometry


def _units(output_key: str) -> dict[str, str]:
    key = output_key.lower()
    if any(token in key for token in ("probability", "confidence")):
        return {"value": "1"}
    if "mask" in key or "label" in key:
        return {"value": "class_id"}
    return {}


def prediction_result_to_artifact_bundle(
    result: Mapping[str, Any],
    *,
    bundle_id: str | None = None,
    release_id: str | None = None,
    artifact_url_template: str | None = None,
) -> ArtifactBundleDescriptor:
    """Translate a legacy prediction result into the stable layer protocol.

    The adapter is deliberately metadata-only: declared output paths are never
    resolved, opened, memory-mapped, checksummed, or otherwise inspected, and
    only the basename enters the public layer metadata.  When an artifact URL
    template is supplied, clients load data through an opaque allow-listed id.
    The preserved FaultSeg/SurfaceSeg visualization builders remain responsible
    for bounded previews of their native arrays.
    """

    if not isinstance(result, Mapping):
        raise TypeError("prediction result must be a mapping")
    model_id = str(result.get("model_id") or "").strip()
    task_id = str(result.get("task_id") or "").strip()
    if not model_id or not task_id:
        raise ValueError("prediction result must declare model_id and task_id")
    outputs = result.get("outputs")
    input_metadata = result.get("input")
    if not isinstance(outputs, Mapping) or not isinstance(input_metadata, Mapping):
        raise ValueError("prediction result must contain input and outputs mappings")

    resolved_bundle_id = str(
        bundle_id
        or result.get("bundle_id")
        or result.get("prediction_id")
        or result.get("run_id")
        or f"{task_id}.{model_id}.prediction"
    ).strip()
    resolved_release_id = str(release_id or result.get("release_id") or model_id).strip()
    layers: list[VisualizationLayerDescriptor] = []
    artifact_ids: dict[str, str] = {}
    adapter_warnings: list[str] = []

    for index, (raw_key, raw_value) in enumerate(outputs.items()):
        output_key = str(raw_key).strip()
        path = _artifact_path(raw_value)
        if not output_key or path is None:
            adapter_warnings.append(f"ignored output {raw_key!r}: no declared path")
            continue
        kind = _layer_kind(output_key, path, raw_value)
        if kind is None:
            adapter_warnings.append(
                f"ignored output {output_key!r}: unsupported or ambiguous artifact format"
            )
            continue
        role = _layer_role(output_key, kind)
        layer_axes = _layer_axes(
            result,
            model_id=model_id,
            output_key=output_key,
            kind=kind,
            value=raw_value,
        )
        component = _identifier_component(output_key, fallback=f"output-{index}")
        artifact_id = f"{resolved_bundle_id}.artifact.{index}.{component}"
        artifact_ids[output_key] = artifact_id
        uncertainty = _uncertainty_semantics(output_key) if role == "uncertainty" else None
        access_url = None
        if artifact_url_template:
            if "{artifact_id}" not in artifact_url_template:
                raise ValueError("artifact_url_template must contain {artifact_id}")
            access_url = artifact_url_template.format(
                artifact_id=quote(artifact_id, safe="")
            )
        layers.append(
            VisualizationLayerDescriptor(
                id=f"{resolved_bundle_id}.layer.{index}.{component}",
                name=output_key.replace("_", " ").strip().title(),
                kind=kind,
                role=role,
                artifact_id=artifact_id,
                access_url=access_url,
                axis_order=layer_axes,
                units=_units(output_key),
                geometry=_geometry(
                    result,
                    model_id=model_id,
                    kind=kind,
                    axes=layer_axes,
                    value=raw_value,
                ),
                uncertainty=uncertainty,
                metadata={
                    "output_key": output_key,
                    "declared_name": PurePath(path).name,
                    "format": _declared_format(output_key, path),
                    "model_id": model_id,
                    "path_verified": False,
                    **(
                        {"array_key": _explicit_array_descriptor(raw_value)[0]}
                        if _explicit_array_descriptor(raw_value) is not None
                        else {}
                    ),
                },
                visible_by_default=role == "prediction",
            )
        )

    if not layers:
        raise ValueError("prediction result contains no supported output artifacts")

    manifest_artifact_id = ""
    for preferred_key in ("metadata_json", "manifest_json", "result_json"):
        if preferred_key in artifact_ids:
            manifest_artifact_id = artifact_ids[preferred_key]
            break
    if not manifest_artifact_id:
        manifest_artifact_id = f"{resolved_bundle_id}.result-metadata"

    result_warnings = result.get("warnings")
    if isinstance(result_warnings, Sequence) and not isinstance(result_warnings, (str, bytes)):
        adapter_warnings.extend(str(item) for item in result_warnings)
    scientific_status = str(result.get("scientific_status") or "candidate").strip()
    source_snapshot_id = result.get("source_snapshot_id") or input_metadata.get("snapshot_id")
    return ArtifactBundleDescriptor(
        id=resolved_bundle_id,
        task_id=task_id,
        release_id=resolved_release_id,
        layers=tuple(layers),
        manifest_artifact_id=manifest_artifact_id,
        source_snapshot_id=str(source_snapshot_id) if source_snapshot_id else None,
        scientific_status=scientific_status,
        warnings=tuple(adapter_warnings),
    )


__all__ = [
    "ArtifactBundleDescriptor",
    "LayerKind",
    "LayerRole",
    "VisualizationLayerDescriptor",
    "prediction_result_to_artifact_bundle",
]
