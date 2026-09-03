"""Integrity-bound, model-neutral result downloads and visualization assets.

Prediction runners keep ownership of their native output formats.  This module
only publishes files that were already sealed by the prediction output
integrity contract; it never invents a scientific result or exposes a local
filesystem path.  Spatial models can link to the existing 3-D workbench, while
well/log models expose LAS and tabular assets for a client-side curve viewer.
"""

from __future__ import annotations

import base64
import copy
import csv
import html
import io
import json
import mimetypes
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import numpy as np

from .content_identity import canonical_sha256, file_sha256
from .standard_export import _png_bytes


STANDARD_RESULT_BUNDLE_CONTRACT_VERSION = "well-seismic.standard-result-bundle.v1"
STANDARD_RESULT_MANIFEST_CONTRACT_VERSION = "well-seismic.standard-result-manifest.v1"
_PREDICTION_OUTPUT_INTEGRITY_CONTRACT_VERSION = (
    "well-seismic.prediction-output-integrity.v1"
)

_SPATIAL_LAYER_KINDS = frozenset({"volume", "surface", "points", "trajectory"})
_CLIENT_RENDERERS_BY_SUFFIX = {
    ".las": ("well_curve", "client_well_curve"),
    ".csv": ("table", "client_table"),
    ".tsv": ("table", "client_table"),
    ".png": ("image", "client_image"),
    ".jpg": ("image", "client_image"),
    ".jpeg": ("image", "client_image"),
    ".webp": ("image", "client_image"),
    ".json": ("diagnostic_table", "client_diagnostic"),
}
_MAX_BOUNDED_NPZ_MEMBER_BYTES = 256 * 1024 * 1024
_DIRECT12B_MODEL_ID = "WellFuse-GeoAlign-12B-Direct-v1"
_FAST_FLUID_MODEL_ID = "wellfuse_fluid_interpretation_fast"
_CIGVIS_WELL_SEQUENCE_CONTRACTS = {
    "well_property": "wellfuse.well-property-output.v2",
    "fluid_interpretation": "wellfuse.fluid-interpretation-output.v3",
    "facies_1d": "wellfuse.facies-1d-output.v3",
    "fracture_development": "wellfuse.fracture-development-output.v1",
}


def _contract_declares_public_outputs(value: object) -> bool:
    """Treat malformed declarations as public instead of raising from the gate."""

    if value is None:
        return False
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return True
    return bool(value)


def supports_standard_well_sequence_view(
    result: Mapping[str, Any],
    *,
    visualization_assets: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Admit only current deterministic well-output contracts to CIGVis.

    This gate is deliberately independent from the spatial candidate/display
    gate.  Older outputs without the current producer-owned contract continue
    to use the bounded standard file viewer instead of being guessed into a
    measured-depth product.
    """

    task_id = str(result.get("task_id") or "")
    expected_version = _CIGVIS_WELL_SEQUENCE_CONTRACTS.get(task_id)
    contract = result.get("output_contract")
    if expected_version is None or not isinstance(contract, Mapping):
        return False
    if str(contract.get("contract_version") or "") != expected_version:
        return False
    if task_id == "fluid_interpretation" and (
        str(contract.get("primary_output") or "") != "fluid_intervals_csv"
        or str(contract.get("primary_interval_output") or "")
        != "fluid_intervals_csv"
        or str(contract.get("published_granularity") or "")
        != "continuous_md_intervals"
        or str(contract.get("primary_decision_rule") or "")
        != "minimum_continuous_thickness_same_class_bridge"
        or str(contract.get("probability_usage") or "")
        != "internal_decoding_only_not_persisted"
        or contract.get("point_output_persisted") is not False
        or contract.get("class_probability_persisted") is not False
        or _contract_declares_public_outputs(
            contract.get("public_probability_outputs")
        )
        or _contract_declares_public_outputs(
            contract.get("public_uncertainty_outputs")
        )
    ):
        return False
    if task_id == "facies_1d":
        decision_contract = (
            str(contract.get("primary_output") or ""),
            str(contract.get("primary_decision_rule") or ""),
        )
        if (
            _contract_declares_public_outputs(
                contract.get("public_probability_outputs")
            )
            or _contract_declares_public_outputs(
                contract.get("public_uncertainty_outputs")
            )
            or str(contract.get("primary_interval_output") or "")
            != "facies_intervals_csv"
            or decision_contract
            not in {
                ("facies_code", "sequence_viterbi"),
                ("predicted_code", "argmax_probability"),
            }
        ):
            return False
    if task_id == "fracture_development" and (
        str(contract.get("primary_output") or "") != "fracture_intervals_csv"
        or str(contract.get("published_granularity") or "")
        != "continuous_md_intervals"
        or str(contract.get("score_semantics") or "")
        != "internal_relative_ranking_not_probability"
        or str(contract.get("spatial_scope") or "")
        != "well_side_only_not_3d_fracture_segmentation"
    ):
        return False
    if task_id == "well_property":
        target = str(contract.get("target") or "").strip().upper()
        primary_output = str(contract.get("primary_output") or "").strip()
        primary_semantics = str(contract.get("primary_semantics") or "")
        physical_bounding_applied = contract.get("physical_bounding_applied")
        expects_physical_bound = target in {"POR", "SW", "VSH"}
        if (
            target not in {"DEN", "POR", "LOG_PERM", "SW", "VSH"}
            or not primary_output
            or primary_semantics
            != (
                "physical_bounded_model_regression"
                if expects_physical_bound
                else "raw_model_regression"
            )
            or physical_bounding_applied is not expects_physical_bound
        ):
            return False
    if visualization_assets is None:
        return True
    allowed_kinds = (
        {"well_curve", "table"} if task_id == "well_property" else {"table"}
    )
    return any(str(item.get("kind") or "") in allowed_kinds for item in visualization_assets)


def _is_fluid_interval_download(item: Mapping[str, Any]) -> bool:
    """Expose only the deterministic interval table for the fluid model."""

    output_key = str(item.get("output_key") or "").casefold()
    filename = str(item.get("filename") or "").casefold()
    valid_output_key = output_key == "fluid_intervals_csv" or bool(
        re.fullmatch(r"well_\d+_fluid_intervals_csv", output_key)
    )
    return valid_output_key and filename.endswith(".csv")


def _is_facies_interval_download(item: Mapping[str, Any]) -> bool:
    """Expose only the deterministic contiguous-interval CSV for Facies-1D."""

    output_key = str(item.get("output_key") or "").casefold()
    filename = str(item.get("filename") or "").casefold()
    return filename.endswith(".csv") and (
        "facies_intervals" in output_key or "facies_intervals" in filename
    )


def _is_facies_3d_class_download(item: Mapping[str, Any]) -> bool:
    """Keep the categorical class volume and its bounded standard previews."""

    output_key = str(item.get("output_key") or "").casefold()
    filename = str(item.get("filename") or "").casefold()
    return (
        output_key in {
            "class_code_npy",
            "argmax_code_npy",
            "standard_preview_png",
            "standard_slice_bundle_zip",
        }
        or filename in {
            "class_code.npy",
            "argmax_code.npy",
            "standard_preview.png",
            "standard_slice_bundle.zip",
        }
    )


def _is_fracture_interval_download(item: Mapping[str, Any]) -> bool:
    """Expose only deterministic low/medium/high fracture-development intervals."""

    output_key = str(item.get("output_key") or "").casefold()
    filename = str(item.get("filename") or "").casefold()
    return (
        "fracture_intervals_csv" in output_key
        or filename == "fracture_intervals.csv"
    )


def _is_fault_mask_download(
    item: Mapping[str, Any], *, verified_mask_segy: bool
) -> bool:
    """Expose the deterministic fault mask and reproducible standard exports.

    The probability cube stays sealed for technical integrity checks but is
    deliberately absent from the competition-facing business result.
    """

    output_key = str(item.get("output_key") or "").casefold()
    filename = str(item.get("filename") or "").casefold()
    if output_key == "fault_mask_sgy":
        return verified_mask_segy
    if output_key == "representative_grid_receipt_json":
        return filename.endswith(".json")
    if output_key == "representative_blocks_directory":
        # Publish each independently sealed binary mask and its block receipt,
        # but keep probability arrays technical-only.  No child is advertised
        # as a global/full-volume mask.
        return filename == "faultseg_mask.npy" or filename == "block_receipt.json"
    return output_key in {
        "mask_npy",
        "standard_slice_bundle_zip",
        "standard_preview_png",
        "fault_mask_slice_summary_csv",
        "fault_mask_audit_json",
        "standard_result_manifest_json",
    }


def _declared_output_paths(value: object, *, prefix: str = "") -> dict[str, Path]:
    declared: dict[str, Path] = {}
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            declared[prefix or "output"] = Path(raw_path).expanduser().resolve()
            return declared
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            declared.update(_declared_output_paths(item, prefix=label))
    elif isinstance(value, str) and value.strip():
        declared[prefix or "output"] = Path(value).expanduser().resolve()
    return declared


@dataclass(frozen=True)
class ResolvedStandardResultArtifact:
    """One verified file selected through an opaque standard-result id."""

    path: Path
    filename: str
    media_type: str
    sha256: str
    size_bytes: int


def _integrity_document(
    result: Mapping[str, Any], *, execution_task_id: str
) -> Mapping[str, Any]:
    integrity = result.get("output_integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("prediction has no sealed output-integrity manifest")
    if (
        str(integrity.get("contract_version") or "")
        != _PREDICTION_OUTPUT_INTEGRITY_CONTRACT_VERSION
    ):
        raise ValueError("prediction output-integrity contract is unsupported")
    claimed = str(integrity.get("integrity_sha256") or "").casefold()
    unsigned = copy.deepcopy(dict(integrity))
    unsigned.pop("integrity_sha256", None)
    if len(claimed) != 64 or canonical_sha256(unsigned).casefold() != claimed:
        raise ValueError("prediction output-integrity manifest is not self-consistent")
    if str(integrity.get("producer_task_id") or "") != execution_task_id:
        raise ValueError("prediction output-integrity producer task does not match")
    bindings = (
        ("interpretation_task_id", "task_id"),
        ("model_id", "model_id"),
        ("source_snapshot_id", "source_snapshot_id"),
    )
    for integrity_key, result_key in bindings:
        expected = str(result.get(result_key) or "")
        observed = str(integrity.get(integrity_key) or "")
        if expected != observed:
            raise ValueError(
                f"prediction output-integrity {integrity_key} does not match result"
            )
    artifacts = integrity.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("prediction output-integrity manifest has no artifacts")
    declared_paths = _declared_output_paths(result.get("outputs") or {})
    if set(declared_paths) != {str(key) for key in artifacts}:
        raise ValueError(
            "prediction outputs and output-integrity artifact keys do not match"
        )
    for raw_key, raw_record in artifacts.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError("prediction output-integrity artifact is invalid")
        sealed_path = Path(str(raw_record.get("path") or "")).expanduser().resolve()
        if declared_paths[str(raw_key)] != sealed_path:
            raise ValueError(f"prediction output path drifted after sealing: {raw_key}")
    return integrity


def _artifact_id(
    *,
    execution_task_id: str,
    integrity_sha256: str,
    output_key: str,
    relative_path: str | None,
    artifact_sha256: str,
) -> str:
    digest = canonical_sha256(
        {
            "execution_task_id": execution_task_id,
            "output_integrity_sha256": integrity_sha256,
            "output_key": output_key,
            "relative_path": relative_path,
            "artifact_sha256": artifact_sha256,
        }
    )
    return f"standard-result-{digest[:32]}"


def _format_from_name(filename: str) -> str:
    suffix = Path(filename).suffix.casefold()
    return suffix[1:] if suffix else "binary"


def _media_type(filename: str) -> str:
    suffix = Path(filename).suffix.casefold()
    explicit = {
        ".las": "application/x-las",
        ".npy": "application/x-npy",
        ".npz": "application/x-npz",
        ".sgy": "application/x-segy",
        ".segy": "application/x-segy",
    }
    return (
        explicit.get(suffix)
        or mimetypes.guess_type(filename)[0]
        or ("application/octet-stream")
    )


def _safe_filename(raw: object, *, fallback: str) -> str:
    name = Path(str(raw or "")).name.strip()
    return name or fallback


def _layer_by_output_key(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bundle = result.get("artifact_bundle")
    layers = bundle.get("layers") if isinstance(bundle, Mapping) else None
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for raw in layers:
        if not isinstance(raw, Mapping):
            continue
        metadata = raw.get("metadata")
        output_key = (
            str(metadata.get("output_key") or "").strip()
            if isinstance(metadata, Mapping)
            else ""
        )
        if output_key:
            mapped[output_key] = copy.deepcopy(dict(raw))
    return mapped


def _download_record(
    *,
    execution_task_id: str,
    integrity_sha256: str,
    output_key: str,
    relative_path: str | None,
    filename: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    digest = str(record.get("sha256") or "").casefold()
    size = int(record.get("size") or 0)
    if len(digest) != 64 or size < 0:
        raise ValueError(f"sealed prediction artifact is invalid: {output_key}")
    artifact_id = _artifact_id(
        execution_task_id=execution_task_id,
        integrity_sha256=integrity_sha256,
        output_key=output_key,
        relative_path=relative_path,
        artifact_sha256=digest,
    )
    return {
        "artifact_id": artifact_id,
        "output_key": output_key,
        "source_kind": "directory_child" if relative_path is not None else "file",
        "relative_path": relative_path,
        "filename": filename,
        "format": _format_from_name(filename),
        "media_type": _media_type(filename),
        "size_bytes": size,
        "sha256": digest,
        "download_url": (
            f"/api/v1/tasks/{quote(execution_task_id, safe='')}"
            f"/standard-results/artifacts/{quote(artifact_id, safe='')}"
        ),
    }


def _visualization_asset_from_layer(
    layer: Mapping[str, Any], download: Mapping[str, Any]
) -> dict[str, Any] | None:
    kind = str(layer.get("kind") or "").strip()
    suffix = Path(str(download.get("filename") or "")).suffix.casefold()
    if kind in _SPATIAL_LAYER_KINDS:
        renderer = "platform_spatial_layer"
    elif kind == "well_curve":
        renderer_info = _CLIENT_RENDERERS_BY_SUFFIX.get(suffix)
        if renderer_info != ("well_curve", "client_well_curve"):
            return None
        renderer = renderer_info[1]
    elif kind == "table":
        # A layer's semantic kind is not enough to choose a parser.  In
        # particular, canonical_input NPZ files are downloadable evidence on
        # the MD axis, not delimited text.  Advertising them as client_table
        # makes the viewer feed binary ZIP bytes to csv.reader and return 500.
        renderer_info = _CLIENT_RENDERERS_BY_SUFFIX.get(suffix)
        if renderer_info != ("table", "client_table"):
            return None
        renderer = renderer_info[1]
    else:
        renderer_info = _CLIENT_RENDERERS_BY_SUFFIX.get(suffix)
        if renderer_info is None:
            return None
        kind, renderer = renderer_info
    metadata = layer.get("metadata")
    public_metadata = (
        copy.deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
    )
    public_metadata.pop("path", None)
    public_metadata.pop("declared_path", None)
    return {
        "asset_id": str(layer.get("id") or download["artifact_id"]),
        "name": str(layer.get("name") or download["filename"]),
        "kind": kind,
        "role": str(layer.get("role") or "prediction"),
        "renderer": renderer,
        "artifact_id": download["artifact_id"],
        "download_url": download["download_url"],
        "filename": download["filename"],
        "format": download["format"],
        "size_bytes": download["size_bytes"],
        "sha256": download["sha256"],
        "axis_order": list(layer.get("axis_order") or []),
        "units": copy.deepcopy(dict(layer.get("units") or {})),
        "geometry": copy.deepcopy(dict(layer.get("geometry") or {})),
        "style": copy.deepcopy(dict(layer.get("style") or {})),
        "uncertainty": copy.deepcopy(layer.get("uncertainty")),
        "metadata": public_metadata,
        "visible_by_default": bool(layer.get("visible_by_default", False)),
    }


def _visualization_asset_from_file(
    download: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    renderer_info = _CLIENT_RENDERERS_BY_SUFFIX.get(
        Path(str(download.get("filename") or "")).suffix.casefold()
    )
    axis_order: list[str] = []
    if renderer_info is None:
        suffix = Path(str(download.get("filename") or "")).suffix.casefold()
        model_id = str(result.get("model_id") or "")
        output_key = str(download.get("output_key") or "")
        if (
            model_id == _DIRECT12B_MODEL_ID
            and output_key == "prediction_npz"
            and suffix == ".npz"
        ):
            renderer_info = ("well_curve", "server_npz_sequence")
            axis_order = ["SAMPLE", "TRAJECTORY_SAMPLE_INDEX"]
    if renderer_info is None:
        standard_spatial_export = result.get("standard_spatial_export")
        standard_spatial_export = (
            standard_spatial_export
            if isinstance(standard_spatial_export, Mapping)
            else {}
        )
        suffix = Path(str(download.get("filename") or "")).suffix.casefold()
        if suffix in {".npy", ".npz"} and str(download.get("output_key") or "") == str(
            standard_spatial_export.get("authoritative_output_key") or ""
        ):
            model_id = str(result.get("model_id") or "").casefold()
            renderer_info = (
                "surface" if model_id == "wellfuse_horizon_p17" else "volume",
                "server_spatial_slices",
            )
            input_metadata = result.get("input")
            input_metadata = (
                input_metadata if isinstance(input_metadata, Mapping) else {}
            )
            axis_order = [
                str(axis)
                for axis in (
                    standard_spatial_export.get("axis_order")
                    or input_metadata.get("axes")
                    or []
                )
            ]
    if renderer_info is None:
        return None
    kind, renderer = renderer_info
    return {
        "asset_id": f"visualization-{download['artifact_id']}",
        "name": str(download["filename"]),
        "kind": kind,
        "role": "prediction",
        "renderer": renderer,
        "artifact_id": download["artifact_id"],
        "download_url": download["download_url"],
        "filename": download["filename"],
        "format": download["format"],
        "size_bytes": download["size_bytes"],
        "sha256": download["sha256"],
        "axis_order": axis_order,
        "units": {},
        "geometry": {},
        "style": {},
        "uncertainty": None,
        "metadata": {
            "output_key": download["output_key"],
            "relative_path": download.get("relative_path"),
        },
        "visible_by_default": True,
    }


def build_standard_result_bundle(
    result: Mapping[str, Any],
    *,
    execution_task_id: str,
    interactive_model_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a path-free manifest from one sealed prediction result."""

    integrity = _integrity_document(result, execution_task_id=execution_task_id)
    integrity_sha256 = str(integrity["integrity_sha256"]).casefold()
    artifacts = integrity["artifacts"]
    assert isinstance(artifacts, Mapping)  # validated by _integrity_document
    layers = _layer_by_output_key(result)
    model_id = str(result.get("model_id") or "")
    interpretation_task_id = str(result.get("task_id") or "")
    downloads: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for raw_output_key, raw_record in artifacts.items():
        output_key = str(raw_output_key)
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"sealed prediction artifact is invalid: {output_key}")
        kind = str(raw_record.get("kind") or "")
        if kind == "file":
            downloads.append(
                _download_record(
                    execution_task_id=execution_task_id,
                    integrity_sha256=integrity_sha256,
                    output_key=output_key,
                    relative_path=None,
                    filename=_safe_filename(
                        raw_record.get("path"), fallback=f"{output_key}.bin"
                    ),
                    record=raw_record,
                )
            )
            continue
        if kind != "directory":
            raise ValueError(f"sealed prediction artifact kind is unsupported: {kind}")
        raw_children = raw_record.get("children")
        if not isinstance(raw_children, Mapping) or not raw_children:
            raise ValueError(f"sealed prediction directory is empty: {output_key}")
        child_ids: list[str] = []
        for raw_relative, raw_child in raw_children.items():
            relative = PurePosixPath(str(raw_relative))
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not isinstance(raw_child, Mapping)
            ):
                raise ValueError(
                    f"sealed prediction directory child is unsafe: {output_key}"
                )
            download = _download_record(
                execution_task_id=execution_task_id,
                integrity_sha256=integrity_sha256,
                output_key=output_key,
                relative_path=relative.as_posix(),
                filename=relative.name,
                record=raw_child,
            )
            downloads.append(download)
            child_ids.append(str(download["artifact_id"]))
        groups.append(
            {
                "output_key": output_key,
                "kind": "directory",
                "size_bytes": int(raw_record.get("size") or 0),
                "file_count": int(raw_record.get("file_count") or len(child_ids)),
                "sha256": str(raw_record.get("sha256") or "").casefold(),
                "download_strategy": "sealed_individual_files",
                "child_artifact_ids": child_ids,
                "archive_artifact_id": None,
            }
        )

    by_output_key = {str(item["output_key"]): item for item in downloads}
    for group in groups:
        output_key = str(group["output_key"])
        archive_candidates: list[str] = []
        if output_key.endswith("_directory"):
            archive_candidates.append(f"{output_key[: -len('_directory')]}_archive_zip")
        archive_candidates.extend((f"{output_key}_archive_zip", f"{output_key}_zip"))
        archive = next(
            (by_output_key[key] for key in archive_candidates if key in by_output_key),
            None,
        )
        if archive is not None:
            group["download_strategy"] = "archive_and_sealed_individual_files"
            group["archive_artifact_id"] = archive["artifact_id"]

    # The probability vectors, pointwise classifications, LAS and technical
    # receipts remain outside the fluid business-result surface.  Integrity is
    # still checked against every sealed native output above; this filter only
    # controls what users can visualize and download as the standard result.
    if interpretation_task_id == "fluid_interpretation":
        downloads = [item for item in downloads if _is_fluid_interval_download(item)]
        groups = []
    elif interpretation_task_id == "facies_1d":
        downloads = [item for item in downloads if _is_facies_interval_download(item)]
        groups = []
    elif interpretation_task_id == "facies_3d":
        downloads = [item for item in downloads if _is_facies_3d_class_download(item)]
        groups = []
    elif interpretation_task_id == "fracture_development":
        downloads = [
            item for item in downloads if _is_fracture_interval_download(item)
        ]
        groups = []
    elif interpretation_task_id == "fault":
        spatial_export = result.get("standard_spatial_export")
        spatial_export = (
            spatial_export if isinstance(spatial_export, Mapping) else {}
        )
        fault_audit = spatial_export.get("fault_mask_audit")
        fault_audit = fault_audit if isinstance(fault_audit, Mapping) else {}
        segy_export = fault_audit.get("segy_export")
        segy_export = segy_export if isinstance(segy_export, Mapping) else {}
        verified_mask_segy = (
            segy_export.get("status") == "roi_mask_segy_available"
            and segy_export.get("public_download_ready", True) is not False
        )
        downloads = [
            item
            for item in downloads
            if _is_fault_mask_download(
                item, verified_mask_segy=verified_mask_segy
            )
        ]
        groups = []

    visualization_assets: list[dict[str, Any]] = []
    seen_visualization_ids: set[str] = set()
    for download in downloads:
        download_output_key = str(download["output_key"])
        layerpulse_download_only = (
            model_id == "layerpulse_geochronograph_f3x200cf"
            and (
                download_output_key.endswith("_download_sgy")
                or download_output_key.endswith("_class_legend_csv")
            )
        )
        if layerpulse_download_only:
            # The NPY head is the sole interactive layer.  SEG-Y and its class
            # map are alternate transports for download, not extra tasks.
            continue
        layer = layers.get(download_output_key)
        asset = None
        if layer is not None and download.get("source_kind") == "file":
            asset = _visualization_asset_from_layer(layer, download)
        if asset is None:
            asset = _visualization_asset_from_file(download, result=result)
        if asset is None or str(asset["asset_id"]) in seen_visualization_ids:
            continue
        seen_visualization_ids.add(str(asset["asset_id"]))
        visualization_assets.append(asset)

    spatial_assets = [
        item
        for item in visualization_assets
        if item.get("kind") in _SPATIAL_LAYER_KINDS
    ]
    has_spatial_platform_viewer = bool(
        spatial_assets and model_id in {str(item) for item in interactive_model_ids}
    )
    representative_grid = result.get("representative_grid")
    representative_contract = (
        str(representative_grid.get("contract_version") or "")
        if isinstance(representative_grid, Mapping)
        else ""
    )
    representative_block_count = (
        len(representative_grid.get("blocks") or [])
        if isinstance(representative_grid, Mapping)
        else 0
    )
    has_representative_fault_viewer = bool(
        model_id == "faultseg_3d"
        and model_id in {str(item) for item in interactive_model_ids}
        and isinstance(representative_grid, Mapping)
        and (representative_contract, representative_block_count)
        in {
            ("well-seismic.faultseg-representative-grid.v1", 36),
            ("well-seismic.faultseg-representative-grid.v2", 128),
        }
        and representative_grid.get("scope") == "representative_sampling"
        and representative_grid.get("is_full_volume") is False
    )
    has_well_sequence_viewer = supports_standard_well_sequence_view(
        result,
        visualization_assets=visualization_assets,
    )
    has_platform_viewer = (
        has_spatial_platform_viewer
        or has_representative_fault_viewer
        or has_well_sequence_viewer
    )
    if not has_spatial_platform_viewer:
        for asset in spatial_assets:
            asset["renderer"] = "server_spatial_slices"

    def preferred_order(item: Mapping[str, Any]) -> tuple[int, int, str]:
        output_key = str((item.get("metadata") or {}).get("output_key") or "")
        if model_id == _FAST_FLUID_MODEL_ID:
            fluid_rank = 0 if "fluid_intervals_csv" in output_key else 7
        else:
            fluid_rank = 0
        visible_rank = (
            0
            if item.get("role") == "prediction"
            and item.get("visible_by_default")
            else 1
        )
        kind_rank = {
            "volume": 0,
            "surface": 0,
            "points": 0,
            "trajectory": 0,
            "well_curve": 1,
            "table": 2,
            "image": 3,
            "diagnostic_table": 4,
        }.get(str(item.get("kind") or ""), 9)
        return fluid_rank * 10 + visible_rank, kind_rank, str(
            item.get("asset_id") or ""
        )

    preferred = (
        min(
            visualization_assets,
            key=preferred_order,
        )
        if visualization_assets
        else None
    )
    manifest_url = (
        f"/api/v1/tasks/{quote(execution_task_id, safe='')}"
        "/standard-results/manifest"
    )
    visualization_url = (
        f"/api/v1/tasks/{quote(execution_task_id, safe='')}"
        "/standard-results/visualization"
    )
    for asset in visualization_assets:
        asset["visualization_url"] = (
            f"{visualization_url}?artifact_id="
            f"{quote(str(asset['artifact_id']), safe='')}"
        )
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_snapshot_sha256 = str(provenance.get("source_snapshot_fingerprint") or "")
    document: dict[str, Any] = {
        "contract_version": STANDARD_RESULT_BUNDLE_CONTRACT_VERSION,
        "manifest_contract_version": STANDARD_RESULT_MANIFEST_CONTRACT_VERSION,
        "bundle_id": f"standard-result-{execution_task_id}",
        "execution_task_id": execution_task_id,
        "interpretation_task_id": str(result.get("task_id") or ""),
        "model_id": model_id,
        "source_snapshot": {
            "id": str(result.get("source_snapshot_id") or ""),
            "sha256": source_snapshot_sha256 or None,
        },
        "output_integrity": {
            "contract_version": str(integrity.get("contract_version") or ""),
            "sha256": integrity_sha256,
        },
        "visualization": {
            "available": bool(visualization_assets),
            "entry_url": visualization_url if visualization_assets else None,
            "platform_viewer_url": (
                f"/统一数据可视化?task_id={quote(execution_task_id, safe='')}&embed=0"
                if has_platform_viewer
                else None
            ),
            "preferred_asset_id": (
                str(preferred.get("asset_id")) if preferred is not None else None
            ),
            "renderers": sorted(
                {str(item["renderer"]) for item in visualization_assets}
            ),
            "assets": visualization_assets,
        },
        "downloads": {
            "manifest_url": manifest_url,
            "artifact_count": len(downloads),
            "artifacts": downloads,
            "groups": groups,
        },
    }
    document["bundle_sha256"] = canonical_sha256(document)
    return document


def _bounded_text(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> str:
    with path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        payload = payload[:maximum_bytes]
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _bounded_delimited_rows(
    path: Path, *, delimiter: str, maximum_rows: int = 2000
) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(_bounded_text(path)), delimiter=delimiter)
    raw_rows = [list(row) for _, row in zip(range(maximum_rows + 1), reader)]
    if not raw_rows:
        return [], []
    width = max(len(row) for row in raw_rows)
    headers = [
        str(value).strip() or f"COLUMN_{index + 1}"
        for index, value in enumerate(raw_rows[0] + [""] * (width - len(raw_rows[0])))
    ]
    rows = [row + [""] * (width - len(row)) for row in raw_rows[1:]]
    return headers, rows


def _bounded_las_rows(
    path: Path, *, maximum_rows: int = 2000
) -> tuple[list[str], list[list[str]]]:
    text = _bounded_text(path)
    section = ""
    curve_names: list[str] = []
    data_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1:].split(maxsplit=1)[0].casefold()
            continue
        if section.startswith("curve"):
            mnemonic = line.split(".", 1)[0].strip().split(maxsplit=1)[0]
            if mnemonic:
                curve_names.append(mnemonic)
        elif section.startswith("ascii") or section == "a":
            data_lines.append(line)
            if len(data_lines) >= maximum_rows:
                break
    rows = [line.replace(",", " ").split() for line in data_lines]
    width = max((len(row) for row in rows), default=len(curve_names))
    headers = [
        curve_names[index] if index < len(curve_names) else f"CURVE_{index + 1}"
        for index in range(width)
    ]
    return headers, [row + [""] * (width - len(row)) for row in rows]


def _numeric_value(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) >= 1e30 or number in {-999.25, -999.0}:
        return None
    return number


def _curve_svg(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not headers or not rows:
        return "<p class='empty'>没有可绘制的数据行。</p>"
    numeric: list[tuple[int, list[float | None]]] = []
    for index in range(len(headers)):
        values = [
            _numeric_value(row[index] if index < len(row) else None) for row in rows
        ]
        available = sum(value is not None for value in values)
        if available >= max(2, len(rows) // 3):
            numeric.append((index, values))
    if len(numeric) < 2:
        return "<p class='empty'>数值列不足，已在下方提供有界表格预览。</p>"
    preferred_axis_names = {
        "md",
        "dept",
        "depth",
        "tvd",
        "tvdss",
        "twt",
        "time",
        "sample",
    }
    axis_index, axis_values = next(
        (
            item
            for item in numeric
            if str(headers[item[0]]).strip().casefold() in preferred_axis_names
        ),
        numeric[0],
    )
    curves = [item for item in numeric if item[0] != axis_index][:6]
    if not curves:
        return "<p class='empty'>只有一个数值轴，已在下方提供表格预览。</p>"
    width, height, top, bottom = 1120, 600, 48, 36
    left, right, gap = 86, 28, 14
    plot_height = height - top - bottom
    track_width = (width - left - right - gap * (len(curves) - 1)) / len(curves)
    valid_axis = [value for value in axis_values if value is not None]
    axis_min, axis_max = min(valid_axis), max(valid_axis)
    if axis_max == axis_min:
        axis_max = axis_min + 1.0
    colors = ("#42d3ff", "#ffb84d", "#9cf57a", "#ff79c6", "#bd93f9", "#f87171")
    fragments = [
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='测井或预测曲线有界预览'>",
        f"<text x='8' y='{top}' class='axis-label'>{html.escape(str(headers[axis_index]))}: {axis_min:.4g}</text>",
        f"<text x='8' y='{top + plot_height}' class='axis-label'>{axis_max:.4g}</text>",
    ]
    for curve_number, (column_index, values) in enumerate(curves):
        x0 = left + curve_number * (track_width + gap)
        finite = [value for value in values if value is not None]
        minimum, maximum = min(finite), max(finite)
        if maximum == minimum:
            maximum = minimum + 1.0
        color = colors[curve_number % len(colors)]
        fragments.extend(
            (
                f"<rect x='{x0:.1f}' y='{top}' width='{track_width:.1f}' height='{plot_height}' class='track'/>",
                f"<text x='{x0 + 4:.1f}' y='22' class='curve-name'>{html.escape(str(headers[column_index]))}</text>",
                f"<text x='{x0 + 4:.1f}' y='39' class='range'>{minimum:.4g} – {maximum:.4g}</text>",
            )
        )
        points: list[str] = []
        for axis, value in zip(axis_values, values, strict=False):
            if axis is None or value is None:
                continue
            x = x0 + (value - minimum) / (maximum - minimum) * track_width
            y = top + (axis - axis_min) / (axis_max - axis_min) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        fragments.append(
            f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='1.4'/>"
        )
    fragments.append("</svg>")
    return "".join(fragments)


def _fluid_interval_svg(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str | None:
    """Render a categorical MD interval column instead of probability curves."""

    index_by_name = {
        str(name).strip().casefold(): index for index, name in enumerate(headers)
    }
    required = {"top_md_m", "bottom_md_m", "fluid_class_code", "fluid_class"}
    if not required <= set(index_by_name):
        return None
    parsed: list[tuple[float, float, int, str, str]] = []
    for row in rows:
        try:
            top = float(row[index_by_name["top_md_m"]])
            bottom = float(row[index_by_name["bottom_md_m"]])
            code = int(float(row[index_by_name["fluid_class_code"]]))
            label = str(row[index_by_name["fluid_class"]])
            label_zh = (
                str(row[index_by_name["fluid_class_zh"]])
                if "fluid_class_zh" in index_by_name
                else label
            )
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom < top:
            continue
        parsed.append((top, bottom, code, label, label_zh))
    if not parsed:
        return None
    top_md = min(item[0] for item in parsed)
    bottom_md = max(item[1] for item in parsed)
    if bottom_md <= top_md:
        bottom_md = top_md + 1.0
    width, height, plot_top, plot_bottom = 760, 680, 54, 38
    plot_height = height - plot_top - plot_bottom
    palette = {
        0: "#64748b",  # Dry
        1: "#38bdf8",  # Water
        2: "#fbbf24",  # Oil
        3: "#fb7185",  # Gas
        4: "#a78bfa",  # Mixed
    }
    fragments = [
        "<section class='fluid-interval-view'>",
        "<h2>确定性流体层段</h2>",
        (
            "<p class='meta'>颜色表示最终流体类型；深度段为 MD 中点边界。"
            "标准结果仅包含连续深度段及其确定流体类型。</p>"
        ),
        (
            f"<svg viewBox='0 0 {width} {height}' role='img' "
            "aria-label='确定性流体层段 MD 柱状图'>"
        ),
        f"<text x='16' y='{plot_top}' class='axis-label'>MD {top_md:.4g} m</text>",
        f"<text x='16' y='{plot_top + plot_height}' class='axis-label'>MD {bottom_md:.4g} m</text>",
        f"<rect x='126' y='{plot_top}' width='270' height='{plot_height}' class='track'/>",
    ]
    for top, bottom, code, label, label_zh in parsed:
        y0 = plot_top + (top - top_md) / (bottom_md - top_md) * plot_height
        y1 = plot_top + (bottom - top_md) / (bottom_md - top_md) * plot_height
        block_height = max(y1 - y0, 0.8)
        color = palette.get(code, "#94a3b8")
        fragments.append(
            f"<rect x='127' y='{y0:.2f}' width='268' height='{block_height:.2f}' "
            f"fill='{color}' stroke='#07111f' stroke-width='0.6'/>"
        )
        if block_height >= 13.0:
            fragments.append(
                f"<text x='410' y='{y0 + min(block_height - 2, 13):.2f}' "
                f"class='curve-name'>{html.escape(label_zh)} / "
                f"{html.escape(label)}</text>"
            )
    fragments.append("</svg>")
    legend = "".join(
        "<span><i style='background:"
        f"{palette[code]}'></i>{html.escape(label_zh)} / {html.escape(label)}</span>"
        for code, label, label_zh in (
            (0, "Dry", "干层"),
            (1, "Water", "水层"),
            (2, "Oil", "油层"),
            (3, "Gas", "气层"),
            (4, "Mixed", "混合层"),
        )
    )
    fragments.extend((f"<div class='fluid-legend'>{legend}</div>", "</section>"))
    return "".join(fragments)


def _facies_interval_svg(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str | None:
    """Render deterministic sedimentary-facies intervals as one categorical MD track."""

    index_by_name = {
        str(name).strip().casefold(): index for index, name in enumerate(headers)
    }
    required = {"top_md_m", "bottom_md_m", "facies_code", "facies_name"}
    if not required <= set(index_by_name):
        return None
    parsed: list[tuple[float, float, int, str]] = []
    for row in rows:
        try:
            top = float(row[index_by_name["top_md_m"]])
            bottom = float(row[index_by_name["bottom_md_m"]])
            code = int(float(row[index_by_name["facies_code"]]))
            label = str(row[index_by_name["facies_name"]])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom < top:
            continue
        parsed.append((top, bottom, code, label))
    if not parsed:
        return None
    top_md = min(item[0] for item in parsed)
    bottom_md = max(item[1] for item in parsed)
    if bottom_md <= top_md:
        bottom_md = top_md + 1.0
    width, height, plot_top, plot_bottom = 760, 680, 54, 38
    plot_height = height - plot_top - plot_bottom
    palette = {
        0: "#64748b",
        1: "#38bdf8",
        2: "#34d399",
        3: "#fbbf24",
        5: "#a78bfa",
        6: "#fb7185",
    }
    fragments = [
        "<section class='facies-interval-view'>",
        "<h2>确定性沉积相层段</h2>",
        (
            "<p class='meta'>颜色表示 Viterbi 解码后的最终相类型；"
            "标准结果仅展示连续 MD 层段，不展示逐类概率、置信度或熵。</p>"
        ),
        (
            f"<svg viewBox='0 0 {width} {height}' role='img' "
            "aria-label='常规测井图版式确定性沉积相层段'>"
        ),
        "<text x='112' y='31' text-anchor='end' class='log-track-header'>MD / m</text>",
        "<text x='261' y='31' text-anchor='middle' class='log-track-header'>"
        "地震相分类（确定性）</text>",
        f"<rect x='126' y='{plot_top}' width='270' height='{plot_height}' class='track'/>",
    ]
    grid_fragments: list[str] = []
    for tick_index in range(11):
        fraction = tick_index / 10.0
        y = plot_top + fraction * plot_height
        md_value = top_md + fraction * (bottom_md - top_md)
        fragments.append(
            f"<text x='112' y='{y + 4:.2f}' text-anchor='end' "
            f"class='axis-label'>{md_value:.4g}</text>"
        )
        grid_fragments.append(
            f"<line x1='120' y1='{y:.2f}' x2='396' y2='{y:.2f}' "
            "class='facies-log-grid'/>"
        )
    for division in range(1, 4):
        x = 126 + division * 270 / 4.0
        grid_fragments.append(
            f"<line x1='{x:.2f}' y1='{plot_top}' x2='{x:.2f}' "
            f"y2='{plot_top + plot_height}' class='facies-log-grid'/>"
        )
    legend_values: dict[int, str] = {}
    for top, bottom, code, label in parsed:
        legend_values.setdefault(code, label)
        y0 = plot_top + (top - top_md) / (bottom_md - top_md) * plot_height
        y1 = plot_top + (bottom - top_md) / (bottom_md - top_md) * plot_height
        block_height = max(y1 - y0, 0.8)
        color = palette.get(code, "#94a3b8")
        fragments.append(
            f"<rect x='127' y='{y0:.2f}' width='268' height='{block_height:.2f}' "
            f"fill='{color}' stroke='#07111f' stroke-width='0.6'/>"
        )
        if block_height >= 13.0:
            fragments.append(
                f"<text x='410' y='{y0 + min(block_height - 2, 13):.2f}' "
                f"class='curve-name'>{html.escape(label)}（代码 {code}）</text>"
            )
    fragments.extend(grid_fragments)
    fragments.append("</svg>")
    legend = "".join(
        "<span><i style='background:"
        f"{palette.get(code, '#94a3b8')}'></i>{html.escape(label)}（{code}）</span>"
        for code, label in sorted(legend_values.items())
    )
    fragments.extend((f"<div class='fluid-legend'>{legend}</div>", "</section>"))
    return "".join(fragments)


def _fracture_interval_svg(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str | None:
    """Render deterministic relative-development intervals without score curves."""

    index_by_name = {
        str(name).strip().casefold(): index for index, name in enumerate(headers)
    }
    required = {
        "top_md_m",
        "bottom_md_m",
        "fracture_level_code",
        "fracture_level",
    }
    if not required <= set(index_by_name):
        return None
    parsed: list[tuple[float, float, int, str, str]] = []
    for row in rows:
        try:
            top = float(row[index_by_name["top_md_m"]])
            bottom = float(row[index_by_name["bottom_md_m"]])
            code = int(float(row[index_by_name["fracture_level_code"]]))
            label = str(row[index_by_name["fracture_level"]])
            label_zh = (
                str(row[index_by_name["fracture_level_zh"]])
                if "fracture_level_zh" in index_by_name
                else label
            )
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom < top:
            continue
        parsed.append((top, bottom, code, label, label_zh))
    if not parsed:
        return None
    top_md = min(item[0] for item in parsed)
    bottom_md = max(item[1] for item in parsed)
    if bottom_md <= top_md:
        bottom_md = top_md + 1.0
    width, height, plot_top, plot_bottom = 760, 680, 54, 38
    plot_height = height - plot_top - plot_bottom
    palette = {0: "#8fb4c9", 1: "#dda32b", 2: "#d74735"}
    fragments = [
        "<section class='fracture-interval-view'>",
        "<h2>井侧裂缝相对发育层段</h2>",
        (
            "<p class='meta'>颜色仅表示本井内相对较弱、中等、较强三个等级；"
            "不是裂缝概率、绝对密度或三维裂缝体。</p>"
        ),
        (
            f"<svg viewBox='0 0 {width} {height}' role='img' "
            "aria-label='井侧裂缝相对发育连续MD层段'>"
        ),
        f"<text x='16' y='{plot_top}' class='axis-label'>MD {top_md:.4g} m</text>",
        f"<text x='16' y='{plot_top + plot_height}' class='axis-label'>MD {bottom_md:.4g} m</text>",
        f"<rect x='126' y='{plot_top}' width='270' height='{plot_height}' class='track'/>",
    ]
    legend_values: dict[int, tuple[str, str]] = {}
    for top, bottom, code, label, label_zh in parsed:
        legend_values.setdefault(code, (label, label_zh))
        y0 = plot_top + (top - top_md) / (bottom_md - top_md) * plot_height
        y1 = plot_top + (bottom - top_md) / (bottom_md - top_md) * plot_height
        block_height = max(y1 - y0, 0.8)
        color = palette.get(code, "#94a3b8")
        fragments.append(
            f"<rect x='127' y='{y0:.2f}' width='268' height='{block_height:.2f}' "
            f"fill='{color}' stroke='#07111f' stroke-width='0.6'/>"
        )
        if block_height >= 13.0:
            fragments.append(
                f"<text x='410' y='{y0 + min(block_height - 2, 13):.2f}' "
                f"class='curve-name'>{html.escape(label_zh)} / "
                f"{html.escape(label)}</text>"
            )
    fragments.append("</svg>")
    legend = "".join(
        "<span><i style='background:"
        f"{palette.get(code, '#94a3b8')}'></i>{html.escape(label_zh)} / "
        f"{html.escape(label)}</span>"
        for code, (label, label_zh) in sorted(legend_values.items())
    )
    fragments.extend((f"<div class='fluid-legend'>{legend}</div>", "</section>"))
    return "".join(fragments)


def _table_html(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, maximum_rows: int = 80
) -> str:
    if not headers:
        return "<p class='empty'>没有结构化表格列。</p>"
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers[:20])
    body = []
    for row in rows[:maximum_rows]:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in list(row)[:20])
            + "</tr>"
        )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _diagnostic_html(path: Path) -> str:
    text = _bounded_text(path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return f"<pre>{html.escape(text[:100000])}</pre>"
    tables: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, Mapping) for item in value[:100])
            ):
                headers = sorted(
                    {str(column) for item in value[:100] for column in item.keys()}
                )
                rows = [
                    [item.get(column, "") for column in headers] for item in value[:100]
                ]
                tables.append(
                    f"<h2>{html.escape(str(key))}</h2>{_table_html(headers, rows)}"
                )
    pretty = json.dumps(document, ensure_ascii=False, indent=2)[:100000]
    return "".join(tables) + f"<h2>诊断 JSON</h2><pre>{html.escape(pretty)}</pre>"


def _bounded_npz_spatial_array(
    path: Path, *, model_id: str, output_key: str
) -> tuple[np.ndarray, str]:
    """Load one documented numeric 3-D NPZ member under a strict size cap."""

    preferred_by_model = {
        "wellfuse_horizon_p17": ("prediction_twt_ms",),
        "wellfuse_facies_3d_p17": ("argmax_code",),
    }
    preferred = [
        *preferred_by_model.get(model_id.casefold(), ()),
        "argmax_code",
        "class_code",
        "label",
        "labels",
        "mask",
        "prediction_twt_ms",
        "prediction",
        "probability",
    ]
    try:
        with zipfile.ZipFile(path) as archive:
            info_by_key = {
                PurePosixPath(info.filename).with_suffix("").as_posix(): info
                for info in archive.infolist()
                if info.filename.casefold().endswith(".npy")
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("spatial NPZ is not a readable NumPy archive") from exc
    with np.load(path, allow_pickle=False) as arrays:
        candidates = list(dict.fromkeys([*preferred, *arrays.files]))
        for key in candidates:
            if key not in arrays.files:
                continue
            info = info_by_key.get(key)
            if info is None or info.file_size > _MAX_BOUNDED_NPZ_MEMBER_BYTES:
                continue
            try:
                array = np.asarray(arrays[key])
            except (EOFError, OSError, TypeError, ValueError):
                continue
            if array.ndim != 3 or array.dtype.hasobject:
                continue
            if not (
                np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
            ):
                continue
            return array, key
    raise ValueError(
        f"spatial NPZ output {output_key} has no bounded numeric 3-D member"
    )


def _direct12b_sequence_html(path: Path) -> str:
    """Render bounded Direct-12B trajectory-index curves from its sealed NPZ."""

    try:
        with zipfile.ZipFile(path) as archive:
            info_by_key = {
                PurePosixPath(info.filename).with_suffix("").as_posix(): info
                for info in archive.infolist()
                if info.filename.casefold().endswith(".npy")
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Direct-12B NPZ is not a readable NumPy archive") from exc

    def member(
        arrays: Any,
        key: str,
        *,
        maximum_bytes: int = _MAX_BOUNDED_NPZ_MEMBER_BYTES,
    ) -> np.ndarray | None:
        info = info_by_key.get(key)
        if key not in arrays.files or info is None or info.file_size > maximum_bytes:
            return None
        try:
            value = np.asarray(arrays[key])
        except (EOFError, OSError, TypeError, ValueError):
            return None
        return None if value.dtype.hasobject else value

    with np.load(path, allow_pickle=False) as arrays:
        predicted = member(arrays, "predicted_twt_ms")
        if predicted is None:
            raise ValueError("Direct-12B NPZ has no bounded predicted_twt_ms member")
        predicted = np.asarray(predicted, dtype=float)
        if predicted.ndim == 1:
            predicted = predicted[None, :]
        if predicted.ndim != 2 or predicted.shape[1] < 1:
            raise ValueError("Direct-12B predicted_twt_ms must be SAMPLE x TRAJECTORY")
        uncertainty = member(arrays, "uncertainty_twt_ms")
        confidence = member(arrays, "alignment_confidence")
        probability = member(arrays, "alignment_probability")
        trajectory_mask = member(arrays, "trajectory_mask")
        sample_ids = member(arrays, "sample_ids", maximum_bytes=16 * 1024 * 1024)

        optional_curves: list[tuple[str, np.ndarray]] = []
        for name, value in (
            ("uncertainty_twt_ms", uncertainty),
            ("alignment_confidence", confidence),
        ):
            if value is None:
                continue
            numeric = np.asarray(value, dtype=float)
            if numeric.ndim == 1:
                numeric = numeric[None, :]
            if numeric.shape == predicted.shape:
                optional_curves.append((name, numeric))
        if probability is not None:
            numeric_probability = np.asarray(probability, dtype=float)
            if (
                numeric_probability.ndim == 3
                and numeric_probability.shape[:2] == predicted.shape
            ):
                optional_curves.append(
                    (
                        "peak_alignment_probability",
                        np.nanmax(numeric_probability, axis=-1),
                    )
                )
        valid_mask: np.ndarray | None = None
        if trajectory_mask is not None:
            candidate_mask = np.asarray(trajectory_mask, dtype=bool)
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask[None, :]
            if candidate_mask.shape == predicted.shape:
                valid_mask = candidate_mask
        labels = (
            [str(value) for value in np.asarray(sample_ids).reshape(-1)]
            if sample_ids is not None
            else []
        )

    cards: list[str] = []
    for sample_index in range(min(predicted.shape[0], 8)):
        count = int(predicted.shape[1])
        indices = np.unique(
            np.rint(np.linspace(0, max(count - 1, 0), min(count, 2000))).astype(int)
        )
        headers = ["trajectory_sample_index", "predicted_twt_ms"] + [
            name for name, _ in optional_curves
        ]
        rows: list[list[str]] = []
        for index in indices:
            if valid_mask is not None and not bool(valid_mask[sample_index, index]):
                continue
            row = [str(int(index)), str(float(predicted[sample_index, index]))]
            row.extend(
                str(float(values[sample_index, index])) for _, values in optional_curves
            )
            rows.append(row)
        label = (
            labels[sample_index] if sample_index < len(labels) else str(sample_index)
        )
        cards.append(
            f"<section><h2>样本 {html.escape(label)}</h2>"
            "<p class='meta'>纵轴为轨迹样点序号；未提供 MD 时不冒充 MD。"
            "概率曲线为各样点对地震时间轴的峰值概率。</p>"
            f"{_curve_svg(headers, rows)}{_table_html(headers, rows)}</section>"
        )
    if not cards:
        raise ValueError("Direct-12B NPZ contains no displayable trajectory rows")
    return "".join(cards)


def render_standard_result_visualization(
    result: Mapping[str, Any],
    *,
    execution_task_id: str,
    artifact_id: str | None = None,
    interactive_model_ids: Sequence[str] = (),
) -> str:
    """Render a bounded standalone viewer for non-spatial result files."""

    bundle = build_standard_result_bundle(
        result,
        execution_task_id=execution_task_id,
        interactive_model_ids=interactive_model_ids,
    )
    assets = bundle["visualization"]["assets"]
    if not assets:
        raise ValueError("prediction has no standard visualization asset")
    selected = next(
        (
            item
            for item in assets
            if artifact_id is not None
            and str(item.get("artifact_id") or "") == artifact_id
        ),
        None,
    )
    if artifact_id is not None and selected is None:
        raise KeyError("standard visualization artifact is not registered")
    if selected is None:
        preferred_id = bundle["visualization"].get("preferred_asset_id")
        selected = next(
            (item for item in assets if item.get("asset_id") == preferred_id),
            assets[0],
        )
    resolved = resolve_standard_result_artifact(
        result,
        execution_task_id=execution_task_id,
        artifact_id=str(selected["artifact_id"]),
        interactive_model_ids=interactive_model_ids,
    )
    renderer = str(selected.get("renderer") or "")
    if renderer == "client_well_curve":
        headers, rows = _bounded_las_rows(resolved.path)
        content = (
            _curve_svg(headers, rows) + "<h2>数据预览</h2>" + _table_html(headers, rows)
        )
    elif renderer == "client_table":
        delimiter = "\t" if resolved.path.suffix.casefold() == ".tsv" else ","
        headers, rows = _bounded_delimited_rows(resolved.path, delimiter=delimiter)
        fluid_intervals = _fluid_interval_svg(headers, rows)
        facies_intervals = _facies_interval_svg(headers, rows)
        fracture_intervals = _fracture_interval_svg(headers, rows)
        categorical_intervals = (
            fluid_intervals or facies_intervals or fracture_intervals
        )
        content = (
            categorical_intervals
            if categorical_intervals
            else _curve_svg(headers, rows)
            + "<h2>数据预览</h2>"
            + _table_html(headers, rows)
        )
    elif renderer == "client_diagnostic":
        content = _diagnostic_html(resolved.path)
    elif renderer == "client_image":
        content = (
            f"<img class='preview-image' src='{html.escape(str(selected['download_url']))}' "
            f"alt='{html.escape(str(selected['name']))}'>"
        )
    elif renderer == "server_npz_sequence":
        content = _direct12b_sequence_html(resolved.path)
    elif renderer == "server_spatial_slices":
        suffix = resolved.path.suffix.casefold()
        member_name: str | None = None
        if suffix == ".npy":
            array = np.load(resolved.path, mmap_mode="r", allow_pickle=False)
        elif suffix == ".npz":
            array, member_name = _bounded_npz_spatial_array(
                resolved.path,
                model_id=str(bundle.get("model_id") or ""),
                output_key=str(
                    (selected.get("metadata") or {}).get("output_key") or ""
                ),
            )
        else:
            raise ValueError(
                "this spatial format is downloadable but has no safe bounded renderer"
            )
        if array.ndim != 3:
            raise ValueError("bounded spatial viewer requires a three-dimensional NPY")
        categorical = bool(
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.bool_)
            or any(
                token in str(selected.get("name") or "").casefold()
                for token in ("class", "label", "mask", "argmax")
            )
        )
        axes = list(selected.get("axis_order") or [])
        if len(axes) != 3:
            axes = ["AXIS_0", "AXIS_1", "AXIS_2"]
        cards: list[str] = []
        for axis_index, axis_name in enumerate(axes):
            index = int(array.shape[axis_index] // 2)
            plane = np.asarray(np.take(array, index, axis=axis_index))
            payload = base64.b64encode(
                _png_bytes(plane, categorical=categorical)
            ).decode("ascii")
            cards.append(
                "<figure><figcaption>"
                f"{html.escape(str(axis_name))} = {index} · "
                f"{html.escape(member_name + ' · ' if member_name else '')}"
                f"{html.escape(str(list(plane.shape)))}</figcaption>"
                f"<img class='preview-image' src='data:image/png;base64,{payload}' "
                f"alt='{html.escape(str(axis_name))} 中央切片'></figure>"
            )
        content = "<div class='slice-grid'>" + "".join(cards) + "</div>"
    else:
        raise ValueError("spatial result must use the platform visualization workbench")
    navigation = "".join(
        f"<a class='asset-link' href='{html.escape(str(item['visualization_url']))}'>"
        f"{html.escape(str(item['name']))}</a>"
        for item in assets[:100]
    )
    return f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(str(selected['name']))} · 标准结果可视化</title>
<style>
body{{margin:0;background:radial-gradient(circle at 48% 42%,#f8fbfd 0%,#eef3f7 58%,#e3eaf0 100%);color:#203246;font:14px/1.5 system-ui,sans-serif}}main{{padding:22px;max-width:1500px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin-top:24px}}.meta{{color:#60758a;margin-bottom:16px}}
nav{{display:flex;gap:8px;overflow:auto;padding:12px 0 18px}}.asset-link{{white-space:nowrap;color:#315f82;background:#e8f3fb;border:1px solid #b9d6eb;border-radius:8px;padding:7px 10px;text-decoration:none}}
svg{{width:100%;background:#eef3f7;border:1px solid #b9c9d6;border-radius:10px}}.track{{fill:#f8fbfd;stroke:#9eb2c3}}.curve-name{{fill:#203246;font-size:13px}}.range,.axis-label{{fill:#60758a;font-size:12px}}.log-track-header{{fill:#203246;font-size:12px;font-weight:700}}.facies-log-grid{{stroke:#71879a;stroke-width:1;stroke-opacity:.42;vector-effect:non-scaling-stroke}}
.table-wrap{{max-height:460px;overflow:auto;background:rgba(255,255,255,.72);border:1px solid #b9c9d6;border-radius:9px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:6px 8px;border-bottom:1px solid #d7e1e9;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#e6edf3}}
pre{{max-height:700px;overflow:auto;background:#eef3f7;border:1px solid #b9c9d6;border-radius:9px;padding:14px}}.preview-image{{max-width:100%;height:auto}}.empty{{color:#60758a}}
.slice-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}figure{{margin:0;background:rgba(248,251,253,.94);border:1px solid #b9c9d6;border-radius:9px;padding:10px}}figcaption{{margin-bottom:8px;color:#60758a}}
.fluid-interval-view svg,.facies-interval-view svg{{max-width:760px}}.fluid-legend{{display:flex;flex-wrap:wrap;gap:8px 14px;margin:10px 0 18px}}.fluid-legend span{{display:inline-flex;align-items:center;gap:6px;color:#435a70}}.fluid-legend i{{width:12px;height:12px;border-radius:3px}}
</style></head><body><main><h1>{html.escape(str(selected['name']))}</h1>
<div class='meta'>标准模型成果 · SHA-256 {html.escape(str(resolved.sha256))} · 有界预览最多 2000 行</div>
<nav>{navigation}</nav>{content}</main></body></html>"""


def resolve_standard_result_artifact(
    result: Mapping[str, Any],
    *,
    execution_task_id: str,
    artifact_id: str,
    interactive_model_ids: Sequence[str] = (),
) -> ResolvedStandardResultArtifact:
    """Resolve and re-verify an artifact selected from the public manifest."""

    bundle = build_standard_result_bundle(
        result,
        execution_task_id=execution_task_id,
        interactive_model_ids=interactive_model_ids,
    )
    public_artifacts = bundle["downloads"]["artifacts"]
    selected = next(
        (item for item in public_artifacts if item.get("artifact_id") == artifact_id),
        None,
    )
    if selected is None:
        raise KeyError("standard result artifact is not registered")
    integrity = _integrity_document(result, execution_task_id=execution_task_id)
    private_record = integrity["artifacts"].get(selected["output_key"])
    if not isinstance(private_record, Mapping):
        raise ValueError("standard result artifact lost its integrity binding")
    relative_path = selected.get("relative_path")
    if relative_path is None:
        if private_record.get("kind") != "file":
            raise ValueError("standard result file binding has the wrong kind")
        path = Path(str(private_record.get("path") or "")).expanduser().resolve()
    else:
        if private_record.get("kind") != "directory":
            raise ValueError("standard result child binding has the wrong kind")
        relative = PurePosixPath(str(relative_path))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("standard result child path is unsafe")
        root = Path(str(private_record.get("path") or "")).expanduser().resolve()
        path = root.joinpath(*relative.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "standard result child escapes its sealed directory"
            ) from exc
    if not path.is_file():
        raise FileNotFoundError("standard result artifact no longer exists")
    before = path.stat()
    expected_size = int(selected["size_bytes"])
    if before.st_size != expected_size:
        raise ValueError("standard result artifact size changed after completion")
    observed_sha256 = file_sha256(path).casefold()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or observed_sha256 != str(selected["sha256"]).casefold()
    ):
        raise ValueError("standard result artifact integrity verification failed")
    return ResolvedStandardResultArtifact(
        path=path,
        filename=str(selected["filename"]),
        media_type=str(selected["media_type"]),
        sha256=observed_sha256,
        size_bytes=after.st_size,
    )


__all__ = [
    "STANDARD_RESULT_BUNDLE_CONTRACT_VERSION",
    "STANDARD_RESULT_MANIFEST_CONTRACT_VERSION",
    "ResolvedStandardResultArtifact",
    "build_standard_result_bundle",
    "render_standard_result_visualization",
    "resolve_standard_result_artifact",
    "supports_standard_well_sequence_view",
]
