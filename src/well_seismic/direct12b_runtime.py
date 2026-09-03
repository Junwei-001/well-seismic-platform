"""Platform adapter and subprocess runner for the inference-safe Direct-12B model.

The platform consumes either an external manifest of sealed eleven-key NPZ inputs
or a strict no-TD raw bundle that the WellFuse CPU/device prepare CLI materialises
into that same sealed contract before inference.  Training supervision,
time-depth tables, checkshots and VSP are not part of either mode.  The active
adapter is selected through a small pointer document that is re-read for every
request, so the final F5 adapter can replace the provisional D1 adapter without
changing API code or restarting the platform.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .modeling.input_adapters import (
    ModelInputAdapterRegistry,
    ModelInputBatch,
    ModelInputRequest,
)
from .task_runtime import managed_run

DIRECT12B_MODEL_ID = "WellFuse-GeoAlign-12B-Direct-v1"
DIRECT12B_RELEASE_ID = "wellfuse_geoalign_12b_direct_v1"
DIRECT12B_POINTER_SCHEMA = "well-seismic.direct12b-release-pointer.v1"
DIRECT12B_CORE_LOGICAL_PARAMETERS = 11_666_744_866
DIRECT12B_PRODUCTION_ANCHOR_MEMBERS = 18
DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS = 82_833_030
DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS = (
    DIRECT12B_CORE_LOGICAL_PARAMETERS
    + DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS
)
DIRECT12B_P13_SCIENTIFIC_MANIFEST_SCHEMA = "wellfuse.align.p13-ensemble.v1"
DIRECT12B_RAW_BUNDLE_SCHEMA = "wellfuse.direct12b_raw_bundle.v1"
DIRECT12B_RAW_PREPARE_RECEIPT_SCHEMA = (
    "wellfuse.direct12b_raw_prepare_receipt.v1"
)
DIRECT12B_FORWARD_INPUT_KEYS = (
    "curve_values",
    "curve_masks",
    "moment_tokens",
    "moment_patch_mask",
    "sfm_tokens",
    "sfm_view_mask",
    "trajectory_xy",
    "trajectory_features",
    "trajectory_mask",
    "seismic_time_ms",
    "modality_presence",
)


def _write_direct12b_prediction_preview(
    prediction_path: Path, destination: Path
) -> Path:
    """Materialize an auditable trajectory-point curve table from the sealed NPZ."""

    with np.load(prediction_path, allow_pickle=False) as payload:
        if "predicted_twt_ms" not in payload.files:
            raise ValueError("Direct-12B prediction NPZ has no predicted_twt_ms")
        predicted = np.asarray(payload["predicted_twt_ms"], dtype=np.float64)
        if predicted.ndim == 1:
            predicted = predicted[None, :]
        if predicted.ndim != 2 or not np.all(np.isfinite(predicted)):
            raise ValueError("Direct-12B predicted_twt_ms must be a finite 2-D array")
        sample_count, point_count = predicted.shape

        def optional(name: str, *, boolean: bool = False) -> np.ndarray | None:
            if name not in payload.files:
                return None
            array = np.asarray(payload[name])
            if array.ndim == 1 and sample_count == 1:
                array = array[None, :]
            if array.shape != predicted.shape:
                raise ValueError(f"Direct-12B {name} shape differs from predicted_twt_ms")
            if not boolean and not np.all(np.isfinite(array)):
                raise ValueError(f"Direct-12B {name} contains non-finite values")
            return array.astype(bool if boolean else np.float64, copy=False)

        uncertainty = optional("uncertainty_twt_ms")
        confidence = optional("alignment_confidence")
        valid = optional("trajectory_mask", boolean=True)
        if "sample_ids" in payload.files:
            raw_ids = np.asarray(payload["sample_ids"])
            if raw_ids.ndim != 1 or raw_ids.size != sample_count:
                raise ValueError("Direct-12B sample_ids do not match prediction rows")
            sample_ids = [str(value) for value in raw_ids.tolist()]
        else:
            sample_ids = [f"sample_{index + 1:04d}" for index in range(sample_count)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".csv.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "sample_id",
                "trajectory_point_index",
                "predicted_twt_ms",
                "uncertainty_twt_ms",
                "alignment_confidence",
                "trajectory_valid",
            )
        )
        for sample_index, sample_id in enumerate(sample_ids):
            for point_index in range(point_count):
                writer.writerow(
                    (
                        sample_id,
                        point_index,
                        f"{predicted[sample_index, point_index]:.9g}",
                        (
                            f"{uncertainty[sample_index, point_index]:.9g}"
                            if uncertainty is not None
                            else ""
                        ),
                        (
                            f"{confidence[sample_index, point_index]:.9g}"
                            if confidence is not None
                            else ""
                        ),
                        int(valid[sample_index, point_index]) if valid is not None else 1,
                    )
                )
    os.replace(temporary, destination)
    return destination
DIRECT12B_MANIFEST_FIELDS = {
    "sample_id",
    "survey",
    "well",
    "cache_path",
    "geometry",
    "label_role",
}
DIRECT12B_PUBLIC_OPTION_KEYS = {
    "all_samples",
    "batch_size",
    "external_cache_manifest",
    "raw_bundle",
    "raw_prepare_device",
    "raw_prepare_force_base",
    "raw_prepare_num_threads",
    "sample_ids",
    "timeout_seconds",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "checkshot",
    "time_depth",
    "timedepth",
    "td_table",
    "target_time",
    "target_twt",
    "vsp",
    "synthetic",
    "supervision",
    "teacher",
    "velocity_model",
)
_FORBIDDEN_EXACT_KEYS = {
    "adapter_path",
    "checkpoint_path",
    "checkpoint_paths",
    "data_config",
    "model_config",
    "model_path",
    "supervision",
    "supervision_path",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cli_json_object(stdout: str, *, stage: str) -> dict[str, Any]:
    """Read the final JSON object while tolerating dependency status lines.

    MOMENT/transformers may print messages such as ``Loading weights`` to
    stdout before the CLI emits its structured summary.  Requiring the entire
    stream to be JSON makes a successful inference look like a runtime failure.
    Only a JSON object that consumes the non-whitespace tail is accepted, so a
    partial or stale object still fails closed.
    """

    content = str(stdout or "").strip()
    decoder = json.JSONDecoder()
    for index in reversed(
        [position for position, character in enumerate(content) if character == "{"]
    ):
        try:
            payload, end = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            continue
        if content[end:].strip() or not isinstance(payload, dict):
            continue
        return payload
    raise RuntimeError(f"Direct-12B {stage} did not return a final JSON object")


def _normalized_key(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _walk_option_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(_normalized_key(key))
            keys.extend(_walk_option_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            keys.extend(_walk_option_keys(nested))
    return keys


def _walk_string_values(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            values.extend(_walk_string_values(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            values.extend(_walk_string_values(nested))
    elif isinstance(value, str):
        values.append(value)
    return values


def _read_raw_bundle_boundary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Direct-12B raw bundle is missing: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Direct-12B raw bundle must be JSON, YAML, or YML")
    if not isinstance(payload, Mapping):
        raise TypeError("Direct-12B raw bundle must be an object")
    if payload.get("schema") != DIRECT12B_RAW_BUNDLE_SCHEMA:
        raise ValueError("unsupported Direct-12B raw bundle schema")
    forbidden_keys = sorted(
        {
            key
            for key in _walk_option_keys(payload)
            if any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
            or key in {"label", "labels", "target", "targets"}
        }
    )
    forbidden_value_fragments = (
        "time_depth",
        "timedepth",
        "checkshot",
        "/vsp/",
        "synthetic",
        "supervision",
        "/labels/",
        "/targets/",
    )
    forbidden_values = sorted(
        {
            value
            for value in _walk_string_values(payload)
            if any(
                fragment
                in value.replace("\\", "/")
                .casefold()
                .replace("no_time_depth", "no_td")
                .replace("no_timedepth", "no_td")
                for fragment in forbidden_value_fragments
            )
        }
    )
    if forbidden_keys or forbidden_values:
        raise ValueError(
            "Direct-12B raw bundle exposes TD/checkshot/VSP, synthetic, label, "
            "target, or supervision material"
        )
    survey = payload.get("survey")
    wells = payload.get("wells")
    if not isinstance(survey, Mapping) or not str(survey.get("id") or "").strip():
        raise ValueError("Direct-12B raw bundle survey.id is required")
    if not isinstance(wells, list) or not wells:
        raise ValueError("Direct-12B raw bundle requires wells")
    well_ids = [
        str(well.get("well_id") or "").strip()
        for well in wells
        if isinstance(well, Mapping)
    ]
    if len(well_ids) != len(wells) or any(not value for value in well_ids):
        raise ValueError("Direct-12B raw bundle well_id entries are invalid")
    return {
        "path": path,
        "sha256": _sha256(path),
        "survey": str(survey["id"]),
        "well_ids": well_ids,
    }


def validate_direct12b_request_options(
    options: Mapping[str, Any],
    *,
    public_request: bool = True,
) -> dict[str, Any]:
    """Validate the small inference-only public option surface.

    ``public_request=False`` permits model-neutral platform provenance fields
    that are appended after HTTP validation, while preserving the same explicit
    supervision-key rejection.
    """

    normalized_keys = _walk_option_keys(options)
    forbidden = sorted(
        {
            key
            for key in normalized_keys
            if key in _FORBIDDEN_EXACT_KEYS
            or any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS)
        }
    )
    if forbidden:
        raise ValueError(
            "多模态井震对齐推理不接受TD/checkshot/VSP、监督信息或运行时权重覆盖参数: "
            + ", ".join(forbidden)
        )
    if public_request:
        unknown = sorted(set(options) - DIRECT12B_PUBLIC_OPTION_KEYS)
        if unknown:
            raise ValueError(
                "unsupported Direct-12B inference options: " + ", ".join(unknown)
            )

    manifest_value = str(options.get("external_cache_manifest") or "").strip()
    raw_bundle_value = str(options.get("raw_bundle") or "").strip()
    if bool(manifest_value) == bool(raw_bundle_value):
        raise ValueError(
            "select exactly one of external_cache_manifest or raw_bundle for "
            "Direct-12B inference"
        )
    raw_bundle = (
        _read_raw_bundle_boundary(Path(raw_bundle_value).expanduser().resolve())
        if raw_bundle_value
        else None
    )

    raw_sample_ids = options.get("sample_ids")
    if raw_sample_ids is None:
        sample_ids: list[str] = []
    elif isinstance(raw_sample_ids, Sequence) and not isinstance(
        raw_sample_ids, (str, bytes, bytearray)
    ):
        sample_ids = [str(value).strip() for value in raw_sample_ids]
        if any(not value for value in sample_ids):
            raise ValueError("sample_ids cannot contain empty values")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids cannot contain duplicates")
    else:
        raise TypeError("sample_ids must be a list of sample ids")
    all_samples = bool(options.get("all_samples", False))
    if all_samples == bool(sample_ids):
        raise ValueError("select exactly one of sample_ids or all_samples=true")

    batch_size = int(options.get("batch_size", 1))
    if batch_size < 1 or batch_size > 16:
        raise ValueError("batch_size must be between 1 and 16")
    timeout_seconds = int(options.get("timeout_seconds", 21600))
    if timeout_seconds < 60 or timeout_seconds > 86400:
        raise ValueError("timeout_seconds must be between 60 and 86400")
    raw_prepare_device = str(options.get("raw_prepare_device") or "cpu").casefold()
    if raw_prepare_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("raw_prepare_device must be cpu, cuda, or auto")
    raw_prepare_threads = int(options.get("raw_prepare_num_threads", 8))
    if raw_prepare_threads < 1 or raw_prepare_threads > 64:
        raise ValueError("raw_prepare_num_threads must be between 1 and 64")
    return {
        "input_mode": "raw_bundle" if raw_bundle is not None else "sealed_manifest",
        "external_cache_manifest": (
            Path(manifest_value).expanduser().resolve() if manifest_value else None
        ),
        "raw_bundle": raw_bundle,
        "raw_prepare_device": raw_prepare_device,
        "raw_prepare_num_threads": raw_prepare_threads,
        "raw_prepare_force_base": bool(options.get("raw_prepare_force_base", False)),
        "sample_ids": sample_ids,
        "all_samples": all_samples,
        "batch_size": batch_size,
        "timeout_seconds": timeout_seconds,
    }


def _read_external_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Direct-12B external cache manifest is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "Direct-12B external cache manifest must be a non-empty JSON array"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, Mapping):
            raise TypeError(f"Direct-12B manifest record {index} must be an object")
        if set(value) != DIRECT12B_MANIFEST_FIELDS:
            raise ValueError(
                f"Direct-12B manifest record {index} must contain exactly "
                f"{sorted(DIRECT12B_MANIFEST_FIELDS)}"
            )
        record = dict(value)
        sample_id = str(record["sample_id"]).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(
                f"Direct-12B manifest sample_id is empty or duplicated: {sample_id}"
            )
        seen.add(sample_id)
        if str(record["label_role"]).strip().casefold() not in {
            "unlabeled",
            "inference_only",
        }:
            raise ValueError(
                f"Direct-12B production manifest record {sample_id} is not inference-only"
            )
        cache_path = Path(str(record["cache_path"])).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Direct-12B eleven-key cache is missing: {cache_path}"
            )
        record["sample_id"] = sample_id
        record["cache_path"] = str(cache_path)
        records.append(record)
    return records


def _validate_selected_caches(
    records: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]
) -> None:
    selected = set(sample_ids)
    for record in records:
        if str(record["sample_id"]) not in selected:
            continue
        cache_path = Path(str(record["cache_path"]))
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != set(DIRECT12B_FORWARD_INPUT_KEYS):
                raise ValueError(
                    f"Direct-12B cache {cache_path} is not the exact sealed eleven-key contract"
                )


class Direct12BInputAdapter:
    """Validate a sealed external manifest without materialising GPU tensors."""

    model_id = DIRECT12B_MODEL_ID

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_formats": [
                "direct12b_raw_bundle_json_or_yaml",
                "direct12b_external_cache_manifest_json",
                "sealed_direct12b_11key_npz",
            ],
            "array_axes": ["SAMPLE", "SEALED_TENSOR"],
            "tensor_axes": ["BATCH", "TOKEN_OR_DEPTH", "CHANNEL"],
            "dtype": "sealed_mixed_float32_bool",
            "patch_size": [],
            "overlap": [],
            "normalization": "presealed_inference_safe_direct12b_contract",
            "requires_logs": True,
            "requires_seismic": True,
            "requires_registration": False,
            "supports_crop": False,
            "input_mode": "sealed_manifest_or_raw_bundle_prepare",
            "forward_input_keys": list(DIRECT12B_FORWARD_INPUT_KEYS),
            "forward_input_key_count": len(DIRECT12B_FORWARD_INPUT_KEYS),
            "required_options_one_of": [
                ["external_cache_manifest", "sample_ids"],
                ["external_cache_manifest", "all_samples"],
                ["raw_bundle", "sample_ids"],
                ["raw_bundle", "all_samples"],
            ],
            "forbidden_inference_parameters": ["TD", "checkshot", "VSP"],
        }

    def compatibility(
        self,
        geometry: Any,
        *,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del geometry
        try:
            validated = validate_direct12b_request_options(
                options or {}, public_request=False
            )
            if validated["input_mode"] == "raw_bundle":
                raw_bundle = validated["raw_bundle"]
                available = {
                    f"{raw_bundle['survey']}::{well}"
                    for well in raw_bundle["well_ids"]
                }
            else:
                records = _read_external_manifest(
                    validated["external_cache_manifest"]
                )
                available = {str(record["sample_id"]) for record in records}
            selected = (
                available if validated["all_samples"] else set(validated["sample_ids"])
            )
            missing = sorted(selected - available)
            if missing:
                raise ValueError(
                    "selected Direct-12B samples are absent from the manifest: "
                    + ", ".join(missing)
                )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            return {
                "ready": False,
                "reason": str(exc),
                "adapter": type(self).__name__,
                "expected_axes": ["SAMPLE", "SEALED_TENSOR"],
                "requires_registration": False,
            }
        return {
            "ready": True,
            "reason": f"sealed eleven-key manifest selected {len(selected)} sample(s)",
            "adapter": type(self).__name__,
            "expected_axes": ["SAMPLE", "SEALED_TENSOR"],
            "sample_count": len(selected),
            "requires_registration": False,
        }

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        validated = validate_direct12b_request_options(
            request.options, public_request=False
        )
        if validated["input_mode"] == "raw_bundle":
            raw_bundle = validated["raw_bundle"]
            available = [
                f"{raw_bundle['survey']}::{well}"
                for well in raw_bundle["well_ids"]
            ]
            sample_ids = (
                sorted(available)
                if validated["all_samples"]
                else validated["sample_ids"]
            )
            missing = sorted(set(sample_ids) - set(available))
            if missing:
                raise ValueError(
                    "selected Direct-12B raw samples are absent from the bundle: "
                    + ", ".join(missing)
                )
            return ModelInputBatch(
                model_id=self.model_id,
                array=None,
                valid_mask=None,
                axes=("SAMPLE", "RAW_BUNDLE"),
                provenance={
                    "input_mode": "raw_bundle_prepare_then_exact11",
                    "raw_bundle": str(raw_bundle["path"]),
                    "raw_bundle_sha256": raw_bundle["sha256"],
                    "sample_ids": sample_ids,
                    "sample_count": len(sample_ids),
                    "surveys": [raw_bundle["survey"]],
                    "forward_input_keys": list(DIRECT12B_FORWARD_INPUT_KEYS),
                    "forward_input_key_count": len(DIRECT12B_FORWARD_INPUT_KEYS),
                    "materialization": "wellfuse_direct12b_raw_prepare_cli",
                    "labels_or_time_depth_opened": False,
                    "time_depth_parameter_accepted": False,
                    "checkshot_parameter_accepted": False,
                    "vsp_parameter_accepted": False,
                    "registration_consumed": False,
                },
            )
        manifest_path = validated["external_cache_manifest"]
        records = _read_external_manifest(manifest_path)
        by_id = {str(record["sample_id"]): record for record in records}
        sample_ids = (
            sorted(by_id) if validated["all_samples"] else validated["sample_ids"]
        )
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                "selected Direct-12B samples are absent from the manifest: "
                + ", ".join(missing)
            )
        _validate_selected_caches(records, sample_ids)
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=None,
            axes=("SAMPLE", "SEALED_TENSOR"),
            provenance={
                "input_mode": "sealed_external_manifest",
                "external_cache_manifest": str(manifest_path),
                "external_cache_manifest_sha256": _sha256(manifest_path),
                "sample_ids": list(sample_ids),
                "sample_count": len(sample_ids),
                "surveys": sorted(
                    {str(by_id[value]["survey"]) for value in sample_ids}
                ),
                "forward_input_keys": list(DIRECT12B_FORWARD_INPUT_KEYS),
                "forward_input_key_count": len(DIRECT12B_FORWARD_INPUT_KEYS),
                "materialization": "wellfuse_direct12b_external_manifest_cli",
                "labels_or_time_depth_opened": False,
                "time_depth_parameter_accepted": False,
                "checkshot_parameter_accepted": False,
                "vsp_parameter_accepted": False,
                "registration_consumed": False,
            },
        )


@dataclass(frozen=True)
class Direct12BReleasePointer:
    path: Path
    sha256: str
    wellfuse_root: Path
    python_executable: Path
    adapter_path: Path
    model_config: Path
    data_config: Path
    model_path: str | None
    core: str
    local_files_only: bool
    stage: str
    effect_status: str
    production_anchor_config: Path | None
    production_anchor_project_root: Path | None
    production_anchor_scientific_manifest: Path | None
    production_anchor_scientific_manifest_sha256: str | None
    production_anchor_member_count: int
    core_logical_parameter_count: int
    production_anchor_logical_parameter_count: int
    deployment_logical_parameter_count: int


def _resolve_pointer_path(project_root: Path) -> Path:
    configured = os.getenv("WELLFUSE_DIRECT12B_POINTER")
    return (
        Path(
            configured
            or project_root
            / "models"
            / "wellfuse"
            / "direct12b"
            / "release_pointer_v1.json"
        )
        .expanduser()
        .resolve()
    )


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_direct12b_release_pointer(project_root: Path) -> Direct12BReleasePointer:
    pointer_path = _resolve_pointer_path(project_root)
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"Direct-12B release pointer is missing: {pointer_path}"
        )
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Direct-12B release pointer must be a JSON object")
    if payload.get("schema_version") != DIRECT12B_POINTER_SCHEMA:
        raise ValueError("unsupported Direct-12B release pointer schema")
    if payload.get("model_id") != DIRECT12B_MODEL_ID:
        raise ValueError(
            "Direct-12B release pointer model_id differs from the platform model"
        )

    configured_root = os.getenv("WELLFUSE_DIRECT12B_PROJECT_ROOT")
    root_value = (
        configured_root
        or payload.get("wellfuse_project_root")
        or os.getenv("WELLFUSE_PROJECT_ROOT")
        or "runtime/wellfuse"
    )
    wellfuse_root = _resolve_path(root_value, base=project_root)
    inference_script = wellfuse_root / "scripts" / "infer_wellfuse_12b_direct.py"
    if not inference_script.is_file():
        raise FileNotFoundError(
            f"Direct-12B inference script is missing: {inference_script}"
        )

    python_value = (
        os.getenv("WELLFUSE_DIRECT12B_PYTHON")
        or payload.get("python_path")
        or os.getenv("WELLFUSE_PYTHON")
        or ""
    )
    python_executable = _resolve_path(python_value, base=project_root)
    if not python_executable.is_file():
        raise FileNotFoundError(
            "Direct-12B CUDA Python is missing; set WELLFUSE_DIRECT12B_PYTHON or "
            f"update the pointer: {python_executable}"
        )

    adapter_path = _resolve_path(payload.get("adapter_path"), base=wellfuse_root)
    model_config = _resolve_path(payload.get("model_config"), base=wellfuse_root)
    data_config = _resolve_path(payload.get("data_config"), base=wellfuse_root)
    for required in (
        adapter_path / "bridge.pt",
        adapter_path / "manifest.json",
        model_config,
        data_config,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Direct-12B pointer target is missing: {required}")
    core = str(payload.get("core") or "hf")
    if core not in {"hf", "tiny"}:
        raise ValueError(f"unsupported Direct-12B core: {core}")
    effect_status = str(payload.get("effect_status") or "provisional_non_final")
    if effect_status not in {"provisional_non_final", "final_release"}:
        raise ValueError(f"unsupported Direct-12B effect status: {effect_status}")
    anchor_config_value = str(payload.get("production_anchor_config") or "").strip()
    anchor_project_root_value = str(
        payload.get("production_anchor_project_root") or ""
    ).strip()
    anchor_manifest_value = str(
        payload.get("production_anchor_scientific_manifest") or ""
    ).strip()
    anchor_manifest_sha256 = str(
        payload.get("production_anchor_scientific_manifest_sha256") or ""
    ).strip().casefold()
    anchor_project_root = (
        _resolve_path(anchor_project_root_value, base=wellfuse_root)
        if anchor_project_root_value
        else wellfuse_root
    )
    production_anchor_config = (
        _resolve_path(anchor_config_value, base=wellfuse_root)
        if anchor_config_value
        else None
    )
    production_anchor_manifest = (
        _resolve_path(anchor_manifest_value, base=anchor_project_root)
        if anchor_manifest_value
        else None
    )
    anchor_member_count = int(
        payload.get(
            "production_anchor_member_count",
            DIRECT12B_PRODUCTION_ANCHOR_MEMBERS,
        )
    )
    core_parameter_count = int(
        payload.get(
            "core_logical_parameter_count",
            DIRECT12B_CORE_LOGICAL_PARAMETERS,
        )
    )
    anchor_parameter_count = int(
        payload.get(
            "production_anchor_logical_parameter_count",
            DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS,
        )
    )
    deployment_parameter_count = int(
        payload.get(
            "deployment_logical_parameter_count",
            DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS,
        )
    )
    expected_parameter_counts = (
        core_parameter_count == DIRECT12B_CORE_LOGICAL_PARAMETERS
        and anchor_parameter_count
        == DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS
        and deployment_parameter_count == DIRECT12B_DEPLOYMENT_LOGICAL_PARAMETERS
        and deployment_parameter_count
        == core_parameter_count + anchor_parameter_count
    )
    if not expected_parameter_counts:
        raise ValueError("Direct-12B deployment parameter breakdown drifted")
    if anchor_member_count != DIRECT12B_PRODUCTION_ANCHOR_MEMBERS:
        raise ValueError("Direct-12B production anchor must contain 18 members")
    if effect_status == "final_release":
        if production_anchor_config is None or not production_anchor_config.is_file():
            raise FileNotFoundError(
                "final Direct-12B pointer requires production_anchor_config"
            )
        if production_anchor_manifest is None or not production_anchor_manifest.is_file():
            raise FileNotFoundError(
                "final Direct-12B pointer requires the formal P13 ensemble manifest"
            )
        if len(anchor_manifest_sha256) != 64:
            raise ValueError(
                "final Direct-12B pointer requires the P13 manifest SHA-256"
            )
        if _sha256(production_anchor_manifest).casefold() != anchor_manifest_sha256:
            raise ValueError("Direct-12B P13 scientific manifest SHA-256 drifted")
        anchor_manifest = json.loads(
            production_anchor_manifest.read_text(encoding="utf-8")
        )
        members = anchor_manifest.get("members") or []
        if (
            anchor_manifest.get("schema_version")
            != DIRECT12B_P13_SCIENTIFIC_MANIFEST_SCHEMA
            or anchor_manifest.get("member_count")
            != DIRECT12B_PRODUCTION_ANCHOR_MEMBERS
            or len(members) != DIRECT12B_PRODUCTION_ANCHOR_MEMBERS
            or anchor_manifest.get("time_depth_supervision_is_model_input") is not False
        ):
            raise ValueError("Direct-12B formal P13 ensemble manifest is invalid")
        adapter_manifest = json.loads(
            (adapter_path / "manifest.json").read_text(encoding="utf-8")
        )
        sealed_anchor = (adapter_manifest.get("metadata") or {}).get(
            "production_protected_anchor"
        )
        if not isinstance(sealed_anchor, Mapping) or not sealed_anchor.get(
            "ensemble_id"
        ):
            raise ValueError(
                "final Direct-12B adapter does not seal the production P13 anchor"
            )
        sealed_parameters = sealed_anchor.get("parameter_summary") or {}
        if (
            int(sealed_parameters.get("member_count") or 0)
            != DIRECT12B_PRODUCTION_ANCHOR_MEMBERS
            or int(sealed_parameters.get("total_model_parameters") or 0)
            != DIRECT12B_PRODUCTION_ANCHOR_LOGICAL_PARAMETERS
            or sealed_anchor.get("time_depth_model_input") is not False
            or str(
                sealed_anchor.get("scientific_manifest_sha256") or ""
            ).casefold()
            != anchor_manifest_sha256
        ):
            raise ValueError("final Direct-12B adapter P13 anchor identity drifted")
    model_path_value = str(
        os.getenv("WELLFUSE_DIRECT12B_MODEL_PATH")
        or payload.get("model_path")
        or ""
    ).strip() or None
    local_files_only = bool(payload.get("local_files_only", True))
    if model_path_value and local_files_only:
        local_model_path = Path(model_path_value).expanduser()
        if local_model_path.is_absolute() and not local_model_path.is_dir():
            raise FileNotFoundError(
                "Direct-12B local base-model directory is missing; set "
                "WELLFUSE_DIRECT12B_MODEL_PATH to a valid directory or a "
                f"cached Hugging Face model id: {local_model_path}"
            )
    return Direct12BReleasePointer(
        path=pointer_path,
        sha256=_sha256(pointer_path),
        wellfuse_root=wellfuse_root,
        python_executable=python_executable,
        adapter_path=adapter_path,
        model_config=model_config,
        data_config=data_config,
        model_path=model_path_value,
        core=core,
        local_files_only=local_files_only,
        stage=str(payload.get("stage") or "unknown"),
        effect_status=effect_status,
        production_anchor_config=production_anchor_config,
        production_anchor_project_root=(
            anchor_project_root if production_anchor_config is not None else None
        ),
        production_anchor_scientific_manifest=production_anchor_manifest,
        production_anchor_scientific_manifest_sha256=(
            anchor_manifest_sha256 or None
        ),
        production_anchor_member_count=anchor_member_count,
        core_logical_parameter_count=core_parameter_count,
        production_anchor_logical_parameter_count=anchor_parameter_count,
        deployment_logical_parameter_count=deployment_parameter_count,
    )


def direct12b_pointer_public_state(project_root: Path) -> dict[str, Any]:
    """Return validated pointer state for capability and release views."""

    try:
        pointer = load_direct12b_release_pointer(project_root.resolve())
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "effect_status": "pointer_invalid",
            "final_release_active": False,
            "error": str(exc),
        }
    return {
        "valid": True,
        "path": str(pointer.path),
        "sha256": pointer.sha256,
        "stage": pointer.stage,
        "effect_status": pointer.effect_status,
        "final_release_active": pointer.effect_status == "final_release",
        "production_anchor_configured": pointer.production_anchor_config is not None,
        "production_anchor_member_count": pointer.production_anchor_member_count,
        "core_logical_parameter_count": pointer.core_logical_parameter_count,
        "production_anchor_logical_parameter_count": (
            pointer.production_anchor_logical_parameter_count
        ),
        "deployment_logical_parameter_count": (
            pointer.deployment_logical_parameter_count
        ),
    }


def _subprocess_environment(wellfuse_root: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    if os.name == "nt":
        path_keys = [key for key in environment if key.casefold() == "path"]
        if len(path_keys) > 1:
            preferred = min(
                path_keys, key=lambda key: (key != "Path", -len(environment[key]))
            )
            value = environment[preferred]
            for key in path_keys:
                del environment[key]
            environment["Path"] = value
    environment["PYTHONUTF8"] = "1"
    if wellfuse_root is not None:
        bundled_source = str((wellfuse_root / "src").resolve())
        existing_pythonpath = environment.get("PYTHONPATH", "").strip()
        environment["PYTHONPATH"] = (
            bundled_source
            if not existing_pythonpath
            else os.pathsep.join((bundled_source, existing_pythonpath))
        )
    return environment


def _prepare_direct12b_raw_bundle(
    validated: Mapping[str, Any],
    *,
    pointer: Direct12BReleasePointer,
) -> dict[str, Any]:
    """Materialise and verify the inference-only exact11 cache for a raw bundle."""

    raw_bundle = validated.get("raw_bundle")
    if not isinstance(raw_bundle, Mapping):
        raise TypeError("Direct-12B raw prepare requires validated raw bundle metadata")
    prepare_script = (
        pointer.wellfuse_root / "scripts" / "prepare_wellfuse_12b_direct_raw.py"
    )
    if not prepare_script.is_file():
        raise FileNotFoundError(
            f"Direct-12B raw prepare script is missing: {prepare_script}"
        )
    command = [
        str(pointer.python_executable),
        str(prepare_script),
        "--bundle",
        str(raw_bundle["path"]),
        "--device",
        str(validated["raw_prepare_device"]),
        "--num-threads",
        str(validated["raw_prepare_num_threads"]),
    ]
    if validated["raw_prepare_force_base"]:
        command.append("--force-base")
    completed = managed_run(
        command,
        cwd=str(pointer.wellfuse_root),
        env=_subprocess_environment(pointer.wellfuse_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=validated["timeout_seconds"],
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
        raise RuntimeError(
            "Direct-12B raw prepare subprocess failed with code "
            f"{completed.returncode}:\n{tail}"
        )
    summary = _parse_cli_json_object(completed.stdout, stage="raw prepare")
    if (
        summary.get("passed") is not True
        or summary.get("mode") != "exact11"
        or summary.get("labels_or_time_depth_opened") is not False
    ):
        raise RuntimeError(
            "Direct-12B raw prepare summary did not preserve exact11/no-TD"
        )

    receipt_path = _resolve_path(
        summary.get("receipt"), base=pointer.wellfuse_root
    )
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Direct-12B raw prepare receipt is missing: {receipt_path}"
        )
    receipt_sha256 = _sha256(receipt_path)
    if str(summary.get("receipt_sha256") or "").casefold() != receipt_sha256:
        raise RuntimeError("Direct-12B raw prepare receipt SHA-256 drifted")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise TypeError("Direct-12B raw prepare receipt must be a JSON object")
    boundary = receipt.get("input_boundary") or {}
    required_false = (
        "raw_bundle_forbidden_fields_present",
        "supervision_repository_constructed",
        "time_depth_parameter_accepted",
        "checkshot_parameter_accepted",
        "vsp_parameter_accepted",
        "synthetic_parameter_accepted",
        "label_parameter_accepted",
        "labels_or_time_depth_opened",
    )
    external_manifests = receipt.get("external_manifests") or {}
    exact_records = receipt.get("exact_records") or []
    if (
        receipt.get("schema") != DIRECT12B_RAW_PREPARE_RECEIPT_SCHEMA
        or receipt.get("passed") is not True
        or receipt.get("mode") != "exact11"
        or receipt.get("bundle_sha256") != raw_bundle["sha256"]
        or receipt.get("survey") != raw_bundle["survey"]
        or int(receipt.get("well_count") or 0) != len(raw_bundle["well_ids"])
        or not isinstance(boundary, Mapping)
        or any(boundary.get(key) is not False for key in required_false)
        or boundary.get("model_forward_keys_exact") is not True
        or not isinstance(external_manifests, Mapping)
        or external_manifests.get("passed") is not True
        or external_manifests.get("labels_or_time_depth_opened") is not False
        or not isinstance(exact_records, list)
        or len(exact_records) != len(raw_bundle["well_ids"])
    ):
        raise RuntimeError(
            "Direct-12B raw prepare receipt did not preserve the raw-to-exact11 "
            "inference boundary"
        )
    for record in exact_records:
        if (
            not isinstance(record, Mapping)
            or record.get("passed") is not True
            or record.get("input_only") is not True
            or record.get("labels_or_time_depth_opened") is not False
            or int(record.get("forward_key_count") or 0)
            != len(DIRECT12B_FORWARD_INPUT_KEYS)
            or set(record.get("forward_keys") or [])
            != set(DIRECT12B_FORWARD_INPUT_KEYS)
        ):
            raise RuntimeError("Direct-12B raw exact11 record failed verification")
    budget = receipt.get("budget_after") or {}
    if (
        not isinstance(budget, Mapping)
        or budget.get("strictly_below_1tb") is not True
        or budget.get("within_configured_budget") is not True
    ):
        raise RuntimeError("Direct-12B raw prepare exceeded its survey storage budget")

    manifest_path = _resolve_path(
        summary.get("inference_manifest"), base=pointer.wellfuse_root
    )
    receipt_manifest_path = _resolve_path(
        external_manifests.get("inference_manifest"), base=pointer.wellfuse_root
    )
    if manifest_path != receipt_manifest_path or not manifest_path.is_file():
        raise FileNotFoundError(
            "Direct-12B raw prepare did not write its sealed inference manifest"
        )
    manifest_sha256 = _sha256(manifest_path)
    if (
        str(external_manifests.get("inference_manifest_sha256") or "").casefold()
        != manifest_sha256
    ):
        raise RuntimeError("Direct-12B raw inference manifest SHA-256 drifted")
    return {
        "command": command,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
        "inference_manifest": manifest_path,
        "inference_manifest_sha256": manifest_sha256,
        "gpu_used": bool(receipt.get("gpu_used", False)),
    }


def run_direct12b_prediction(
    request: ModelInputRequest,
    *,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Execute the existing WellFuse production CLI without importing CUDA here."""

    del config, threshold, patch_size, overlap
    runtime_options = {**request.options, **dict(options or {})}
    validated = validate_direct12b_request_options(
        runtime_options, public_request=False
    )
    adapter = adapters.get(DIRECT12B_MODEL_ID)
    batch = adapter.prepare(
        ModelInputRequest(source=request.source, options=runtime_options)
    )
    pointer = load_direct12b_release_pointer(project_root.resolve())
    if progress:
        progress(10, "已验证多模态井震对齐输入边界与运行指针")
    raw_prepare: dict[str, Any] | None = None
    inference_manifest = validated["external_cache_manifest"]
    if validated["input_mode"] == "raw_bundle":
        if progress:
            progress(20, "正在用CPU/selected device将raw bundle封存为11-key输入")
        raw_prepare = _prepare_direct12b_raw_bundle(validated, pointer=pointer)
        inference_manifest = raw_prepare["inference_manifest"]
        sealed_options = {
            "external_cache_manifest": str(inference_manifest),
            "sample_ids": validated["sample_ids"],
            "all_samples": validated["all_samples"],
            "batch_size": validated["batch_size"],
            "timeout_seconds": validated["timeout_seconds"],
        }
        sealed_batch = adapter.prepare(
            ModelInputRequest(source=request.source, options=sealed_options)
        )
        batch = ModelInputBatch(
            model_id=sealed_batch.model_id,
            array=sealed_batch.array,
            valid_mask=sealed_batch.valid_mask,
            axes=sealed_batch.axes,
            provenance={
                **batch.provenance,
                "materialized_external_cache_manifest": str(inference_manifest),
                "materialized_external_cache_manifest_sha256": raw_prepare[
                    "inference_manifest_sha256"
                ],
                "raw_prepare_receipt": str(raw_prepare["receipt_path"]),
                "raw_prepare_receipt_sha256": raw_prepare["receipt_sha256"],
                "sealed_cache_validation": "passed",
            },
        )
        if progress:
            progress(35, "原始数据包已转换并验证为十一张量井震输入")
    output = output_directory.expanduser().resolve()
    command = [
        str(pointer.python_executable),
        str(pointer.wellfuse_root / "scripts" / "infer_wellfuse_12b_direct.py"),
        "--model-config",
        str(pointer.model_config),
        "--data-config",
        str(pointer.data_config),
        "--external-cache-manifest",
        str(inference_manifest),
        "--core",
        pointer.core,
        "--adapter-path",
        str(pointer.adapter_path),
        "--batch-size",
        str(validated["batch_size"]),
        "--device",
        "cuda"
        if str(device_name).casefold().startswith("cuda")
        else "cpu"
        if str(device_name).casefold().startswith("cpu")
        else "auto",
        "--output",
        str(output),
    ]
    if pointer.local_files_only:
        command.append("--local-files-only")
    if pointer.model_path:
        command.extend(("--model-path", pointer.model_path))
    if pointer.production_anchor_config is not None:
        command.extend(
            (
                "--production-anchor-config",
                str(pointer.production_anchor_config),
                "--production-anchor-project-root",
                str(pointer.production_anchor_project_root),
            )
        )
    if validated["all_samples"]:
        command.append("--all-samples")
    else:
        command.append("--samples")
        command.extend(validated["sample_ids"])
    if progress:
        progress(45 if raw_prepare is not None else 20, "正在执行多模态井震对齐生产推理")
    completed = managed_run(
        command,
        cwd=str(pointer.wellfuse_root),
        env=_subprocess_environment(pointer.wellfuse_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=validated["timeout_seconds"],
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
        raise RuntimeError(
            f"Direct-12B subprocess failed with code {completed.returncode}:\n{tail}"
        )
    receipt = _parse_cli_json_object(completed.stdout, stage="inference")
    if receipt.get("model_id") != DIRECT12B_MODEL_ID:
        raise RuntimeError(
            "Direct-12B receipt model_id differs from the selected model"
        )
    boundary = receipt.get("input_boundary") or {}
    if any(
        boundary.get(key) is not False
        for key in (
            "labels_or_time_depth_opened",
            "time_depth_parameter_accepted",
            "checkshot_parameter_accepted",
            "vsp_parameter_accepted",
        )
    ):
        raise RuntimeError(
            "Direct-12B receipt did not preserve the inference-only boundary"
        )
    checks = receipt.get("checks") or {}
    if not checks or not all(bool(value) for value in checks.values()):
        raise RuntimeError(f"Direct-12B receipt checks failed: {checks}")
    production_anchor = receipt.get("production_protected_anchor")
    if pointer.effect_status == "final_release":
        if not isinstance(production_anchor, Mapping):
            raise RuntimeError(
                "final Direct-12B inference did not execute the production P13 anchor"
            )
        parameter_summary = production_anchor.get("parameter_summary") or {}
        if (
            len(production_anchor.get("members") or [])
            != pointer.production_anchor_member_count
            or int(parameter_summary.get("member_count") or 0)
            != pointer.production_anchor_member_count
            or int(parameter_summary.get("total_model_parameters") or 0)
            != pointer.production_anchor_logical_parameter_count
            or production_anchor.get("time_depth_model_input") is not False
            or str(
                production_anchor.get("scientific_manifest_sha256") or ""
            ).casefold()
            != pointer.production_anchor_scientific_manifest_sha256
        ):
            raise RuntimeError(
                "final Direct-12B receipt production anchor identity is incomplete"
            )
    prediction_path = Path(
        str((receipt.get("outputs") or {}).get("prediction_npz"))
    ).resolve()
    receipt_path = output / "direct12b_inference_receipt.json"
    if not prediction_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            "Direct-12B subprocess did not write its sealed outputs"
        )
    preview_path = _write_direct12b_prediction_preview(
        prediction_path,
        output / "direct12b_prediction_preview.csv",
    )
    if progress:
        progress(95, "多模态井震对齐输出与无时深监督边界已验证")
    warnings = []
    if pointer.effect_status != "final_release":
        warnings.append(
            "当前运行指针使用D1临时adapter，仅证明12B工程链路可运行，不代表最终多工区效果。"
        )
    outputs = {
        "prediction_npz": str(prediction_path),
        "prediction_preview_csv": str(preview_path),
        "inference_receipt": str(receipt_path),
    }
    if raw_prepare is not None:
        outputs["raw_prepare_receipt"] = str(raw_prepare["receipt_path"])
    return {
        "model_id": DIRECT12B_MODEL_ID,
        "model_name": "多模态井震对齐（SEG-Y+LAS+完整轨迹→TWT与联合表征）",
        "model_executed": True,
        "input": {
            **batch.provenance,
            "axes": list(batch.axes),
            "time_depth_supervision_is_model_input": False,
        },
        "outputs": outputs,
        "provenance": {
            "runner": (
                "wellfuse_direct12b_raw_prepare_then_inference_subprocess"
                if raw_prepare is not None
                else "wellfuse_direct12b_external_manifest_subprocess"
            ),
            "release_id": DIRECT12B_RELEASE_ID,
            "release_pointer": str(pointer.path),
            "release_pointer_sha256": pointer.sha256,
            "adapter_path": str(pointer.adapter_path),
            "adapter_stage": pointer.stage,
            "effect_status": pointer.effect_status,
            "production_protected_anchor": production_anchor,
            "parameter_breakdown": {
                "core_logical_parameter_count": pointer.core_logical_parameter_count,
                "production_anchor_member_count": pointer.production_anchor_member_count,
                "production_anchor_logical_parameter_count": (
                    pointer.production_anchor_logical_parameter_count
                ),
                "deployment_logical_parameter_count": (
                    pointer.deployment_logical_parameter_count
                ),
            },
            "labels_or_time_depth_opened": False,
            "registration_consumed": False,
            "raw_prepare": (
                {
                    "receipt": str(raw_prepare["receipt_path"]),
                    "receipt_sha256": raw_prepare["receipt_sha256"],
                    "inference_manifest": str(raw_prepare["inference_manifest"]),
                    "inference_manifest_sha256": raw_prepare[
                        "inference_manifest_sha256"
                    ],
                    "device": raw_prepare["receipt"].get("foundation_device"),
                    "gpu_used": raw_prepare["gpu_used"],
                    "labels_or_time_depth_opened": False,
                }
                if raw_prepare is not None
                else None
            ),
        },
        "runtime": {
            "device": receipt.get("device"),
            "core_kind": receipt.get("core_kind"),
            "sample_count": receipt.get("sample_count"),
            "raw_prepare_gpu_used": (
                raw_prepare["gpu_used"] if raw_prepare is not None else None
            ),
        },
        "warnings": warnings,
    }
