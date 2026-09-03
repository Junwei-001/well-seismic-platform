"""Formal model inference services, separate from shared preprocessing."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np

from .fault_models import FAULTNET_MODEL_ID, FAULTSEG_MODEL_ID
from .faultseg import FaultSegInputSpec
from .alignment.geopath_runtime import MODEL_ID as GEOPATH_TIE_V1_MODEL_ID
from .alignment.geopath_runtime import run_geopath_tie_v1
from .model_applicability import (
    evaluate_applicability,
    resolve_training_envelope,
    write_applicability_manifest,
)
from .platform_mode import interface_only_enabled
from .modeling.input_adapters import (
    FAULTSEG_CENTER_BLOCK_SCOPE,
    FAULTSEG_REPRESENTATIVE_GRID_CONTRACT_VERSION,
    FAULTSEG_REPRESENTATIVE_SCOPE,
    ModelInputAdapterRegistry,
    ModelInputRequest,
    _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT,
    _FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX,
    faultseg_execution_scope_contract,
    faultseg_execution_scope_metadata,
)
from .io.segy import SegyReader
from .surface_horizon_display_contract import (
    DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
    DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
    HORIZON_DISPLAY_GATE_SCHEMA,
    validate_surface_horizon_display_contract,
)
from .surface_seg_visualization import validate_surface_window_inference_receipt
from .task_runtime import managed_popen, managed_run

Progress = Callable[[int, str], None]
PredictionRunner = Callable[..., dict[str, Any]]

FAST_FLUID_MODEL_ID = "wellfuse_fluid_interpretation_fast"
FAST_FACIES_1D_MODEL_ID = "wellfuse_facies_1d_chengdu_fast"
F3_FACIES_3D_MODEL_ID = "wellfuse_facies_3d_f3_fast"
FRACTURE_DEVELOPMENT_MODEL_ID = "wellfuse_fracture_development_utah_fast"
FAST_PROPERTY_MODEL_TARGETS = {
    "wellfuse_den_northwest_fast": "DEN",
    "wellfuse_por_northwest_fast": "POR",
    "wellfuse_log_perm_northwest_fast": "LOG_PERM",
    "wellfuse_sw_northwest_fast": "SW",
    "wellfuse_vsh_northwest_fast": "VSH",
}
RAW_WELL_TASK_BY_MODEL = {
    FAST_FLUID_MODEL_ID: "fluid_interpretation",
    FAST_FACIES_1D_MODEL_ID: "facies_1d",
    **{model_id: "well_property" for model_id in FAST_PROPERTY_MODEL_TARGETS},
    FRACTURE_DEVELOPMENT_MODEL_ID: "fracture_development",
}
PUBLIC_RUNTIME_MODEL_NAMES = {
    "layerpulse_geochronograph_f3x200cf": "LayerPulse 智能解释（单 checkpoint 多任务基础模型）",
    FAULTSEG_MODEL_ID: "SEG-Y地震体→慧眼三维断层识别结果",
    FAULTNET_MODEL_ID: "SEG-Y地震体→慧眼区域增强断层结果",
    "seismic_surface_seg": "地层实例分割（SEG-Y地震体→地层标签体与置信度）",
    "wellfuse_horizon_p17": "历史四层位追踪（既有成果→只读下载与可视化）",
    "wellfuse_facies_1d_p17": "井侧沉积相分类（九线LAS+完整轨迹→确定性相层段）",
    "wellfuse_facies_3d_p17": "三维地震相分割（SEG-Y地震体→离散相体）",
    FAST_FACIES_1D_MODEL_ID: "登记井侧沉积相分类（九线LAS→确定性相层段）",
    F3_FACIES_3D_MODEL_ID: "六类三维地震相分割（时间域SEG-Y→离散相体）",
    "wellfuse_channel_p17": "河道地质体识别（SEG-Y地震体→河道概率与几何属性）",
    "wellfuse_karst_p17": "岩溶地质体识别（SEG-Y地震体→岩溶概率与几何属性）",
    FRACTURE_DEVELOPMENT_MODEL_ID: "井侧裂缝发育排序（LAS/CSV测井→连续发育层段）",
    "wellfuse_den_p18": "历史封存井曲线→DEN只读成果",
    "wellfuse_por_p18": "历史封存井曲线→POR只读成果",
    "wellfuse_log_perm_p18": "历史封存井曲线→LOG_PERM只读成果",
    "wellfuse_sw_p18": "历史封存井曲线→SW只读成果",
    "wellfuse_vsh_p18": "历史封存井曲线→VSH只读成果",
    FAST_FLUID_MODEL_ID: "五类流体解释（登记九线LAS→连续MD流体层段）",
    "wellfuse_den_northwest_fast": "井侧密度预测（封存快照井资产→DEN整井曲线）",
    "wellfuse_por_northwest_fast": "井侧孔隙度预测（封存快照井资产→POR整井曲线）",
    "wellfuse_log_perm_northwest_fast": "井侧渗透率预测（封存快照井资产→LOG_PERM整井曲线）",
    "wellfuse_sw_northwest_fast": "井侧含水饱和度预测（封存快照井资产→SW整井曲线）",
    "wellfuse_vsh_northwest_fast": "井侧泥质含量预测（封存快照井资产→VSH整井曲线）",
}
FACIES3D_CHENGDU_HEADER_BYTES = (9, 21)
FACIES3D_TRACE_COUNT_RELATIVE_TOLERANCE = 1e-3
SURFACE_CHECKPOINT_NATIVE_PREPROCESSING_POLICY = (
    "checkpoint-native-anisotropic-resize"
)
SURFACE_EXPERIMENTAL_ASPECT_PRESERVING_POLICY = (
    "experimental-bounded-aspect-preserving-overlap"
)
PHYSICAL_BOUNDED_PROPERTY_TARGETS = frozenset({"POR", "SW", "VSH"})
F3_FACIES_3D_VALIDATED_IDENTITY = {
    "dataset_id": "f3_netherlands_dense",
    "source_sha256": "a2e9167c3b7ecf618bb852e96262090a632066cad6df3ce464bd1164552158e8",
    "checkpoint_sha256": "d5fa8da8bb328ea859895181070698ae530e8f08391d841d6a3e158a52fbce29",
    "metrics_sha256": "542b021110126bd4f934badbac4e6b4ac5eb29ca46e40de7494cf54ad719d838",
}
_F3_TRUSTED_SEALED_HEADER_AUTHORITIES = frozenset(
    {
        "sealed_automatic_geometry_inspection",
        "sealed_source_snapshot_semantics",
    }
)


def public_runtime_model_name(model_id: str) -> str:
    """Return a task/input/output name while keeping the stable model id separate."""

    return PUBLIC_RUNTIME_MODEL_NAMES.get(model_id, "已登记预测模型")


def _write_deterministic_directory_zip(
    source_directory: Path,
    output_path: Path,
) -> Path:
    """Materialize a directory result as one stable downloadable artifact."""

    source = source_directory.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"prediction output directory is missing: {source}")
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix().casefold(),
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as input_handle, archive.open(info, "w") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    os.replace(temporary, destination)
    return destination


def _resolve_facies_3d_header_bytes(
    options: Mapping[str, Any], provenance: Mapping[str, Any]
) -> tuple[int, int]:
    """Prefer explicit/detected bytes, with the audited Chengdu bytes as fallback."""

    recommended = provenance.get("recommended_options", {})
    if not isinstance(recommended, Mapping):
        recommended = {}
    values = (
        options.get("iline_byte", recommended.get("iline_byte")),
        options.get("xline_byte", recommended.get("xline_byte")),
    )
    defaults = FACIES3D_CHENGDU_HEADER_BYTES
    resolved: list[int] = []
    for value, default in zip(values, defaults, strict=True):
        parsed = default if value is None else int(value)
        if parsed < 1 or parsed > 237:
            raise ValueError(f"invalid Facies-3D trace-header byte: {parsed}")
        resolved.append(parsed)
    return resolved[0], resolved[1]


def _validate_facies_3d_geometry(
    seismic: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject collapsed/reused geometry before a Facies-3D result is exposed."""

    shape = seismic.get("shape_t_inline_xline")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise RuntimeError("Facies-3D runtime did not report a TWT/inline/xline shape")
    shape_t_inline_xline = tuple(int(value) for value in shape)
    if any(value < 1 for value in shape_t_inline_xline):
        raise RuntimeError(f"Facies-3D runtime reported invalid shape: {shape}")
    if shape_t_inline_xline[1] <= 1 or shape_t_inline_xline[2] <= 1:
        raise RuntimeError(
            "Facies-3D geometry collapsed to a non-3D volume; both inline and "
            f"xline dimensions must exceed one, got {list(shape_t_inline_xline)}"
        )

    source_shape = provenance.get("source_shape_zyx")
    if isinstance(source_shape, (list, tuple)) and len(source_shape) == 3:
        expected_samples = int(source_shape[0])
        if shape_t_inline_xline[0] != expected_samples:
            raise RuntimeError(
                "Facies-3D runtime sample count differs from adapter preflight: "
                f"{shape_t_inline_xline[0]} != {expected_samples}"
            )

    expected_trace_count = int(provenance.get("trace_count", 0))
    valid_fraction = float(seismic.get("valid_trace_fraction", float("nan")))
    if expected_trace_count < 1 or not np.isfinite(valid_fraction):
        raise RuntimeError("Facies-3D trace-count lineage is incomplete")
    if valid_fraction <= 0.0 or valid_fraction > 1.0:
        raise RuntimeError(
            f"Facies-3D valid_trace_fraction is invalid: {valid_fraction}"
        )
    inferred_trace_count = int(
        round(valid_fraction * shape_t_inline_xline[1] * shape_t_inline_xline[2])
    )
    tolerance = max(
        1,
        int(np.ceil(expected_trace_count * FACIES3D_TRACE_COUNT_RELATIVE_TOLERANCE)),
    )
    if abs(inferred_trace_count - expected_trace_count) > tolerance:
        raise RuntimeError(
            "Facies-3D runtime grid does not account for the inspected SEG-Y traces: "
            f"inferred={inferred_trace_count}, expected={expected_trace_count}, "
            f"tolerance={tolerance}"
        )
    return {
        "shape_t_inline_xline": list(shape_t_inline_xline),
        "expected_trace_count": expected_trace_count,
        "inferred_trace_count": inferred_trace_count,
        "trace_count_tolerance": tolerance,
        "passed": True,
    }


def _f3_exact_identity_evidence(
    *,
    source: Path,
    checkpoint: Path,
    metrics_path: Path,
    runtime_options: Mapping[str, Any],
    expected_identity: Mapping[str, str] = F3_FACIES_3D_VALIDATED_IDENTITY,
) -> dict[str, Any]:
    """Bind dataset validation only to API-verified source and exact artifacts."""

    source = source.expanduser().resolve()
    source_identity = runtime_options.get("prediction_source_identity")
    identity = dict(source_identity) if isinstance(source_identity, Mapping) else {}
    expected_source_sha = str(expected_identity.get("source_sha256", "")).casefold()
    expected_checkpoint_sha = str(
        expected_identity.get("checkpoint_sha256", "")
    ).casefold()
    expected_metrics_sha = str(expected_identity.get("metrics_sha256", "")).casefold()
    checkpoint_stat = checkpoint.stat()
    metrics_stat = metrics_path.stat()
    checkpoint_sha = _checkpoint_sha256(
        str(checkpoint), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size
    )
    metrics_sha = _checkpoint_sha256(
        str(metrics_path), metrics_stat.st_mtime_ns, metrics_stat.st_size
    )
    identity_path_matches = False
    try:
        identity_path_matches = (
            Path(str(identity.get("path", ""))).expanduser().resolve() == source
        )
    except (OSError, TypeError, ValueError):
        identity_path_matches = False
    try:
        identity_size_matches = int(identity.get("size", -1)) == source.stat().st_size
    except (OSError, TypeError, ValueError):
        identity_size_matches = False
    observed_source_sha = str(identity.get("sha256", "")).casefold()
    checks = {
        "api_prediction_source_identity_present": bool(identity),
        "source_kind_matches": identity.get("kind") == "seismic_file",
        "source_path_matches": identity_path_matches,
        "source_size_matches": identity_size_matches,
        "source_integrity_verified_by_api": (
            identity.get("integrity_status") == "sha256_verified"
        ),
        "source_sha256_matches": observed_source_sha == expected_source_sha,
        "checkpoint_sha256_matches": checkpoint_sha == expected_checkpoint_sha,
        "metrics_sha256_matches": metrics_sha == expected_metrics_sha,
    }
    exact_match = all(checks.values())
    return {
        "contract_version": "well-seismic.f3-exact-validation-identity.v1",
        "status": (
            "validated_within_dataset_exact_identity"
            if exact_match
            else "candidate_identity_unknown_or_drifted"
        ),
        "exact_match": exact_match,
        "dataset_id": str(expected_identity.get("dataset_id", "f3_netherlands_dense")),
        "checks": checks,
        "source": {
            "path": str(source),
            "sha256": observed_source_sha or None,
            "sha256_authority": "api_verified_prediction_source_identity",
            "integrity_status": identity.get("integrity_status"),
            "expected_sha256": expected_source_sha,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "expected_sha256": expected_checkpoint_sha,
        },
        "metrics": {
            "path": str(metrics_path),
            "sha256": metrics_sha,
            "expected_sha256": expected_metrics_sha,
        },
        "reason_codes": [name for name, passed in checks.items() if not passed],
    }


def _sealed_f3_recommended_header_bytes(
    runtime_options: Mapping[str, Any],
    *,
    source: Path,
    prediction_source_identity: Mapping[str, Any] | None,
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    """Resolve the immutable snapshot recommendation without using filenames."""

    identity = (
        dict(prediction_source_identity)
        if isinstance(prediction_source_identity, Mapping)
        else {}
    )
    identity_sha = str(identity.get("sha256", "")).casefold()
    receipts = runtime_options.get("source_snapshot_segy_geometry_receipts")
    if isinstance(receipts, list):
        for raw_receipt in receipts:
            if not isinstance(raw_receipt, Mapping):
                continue
            try:
                path_matches = (
                    Path(str(raw_receipt.get("source_asset_path", "")))
                    .expanduser()
                    .resolve()
                    == source.expanduser().resolve()
                )
            except (OSError, TypeError, ValueError):
                path_matches = False
            resolved = raw_receipt.get("resolved_header_bytes")
            if not isinstance(resolved, Mapping):
                continue
            if (
                raw_receipt.get("authority")
                != "sealed_automatic_geometry_inspection"
                or not path_matches
                or not identity_sha
                or str(raw_receipt.get("source_asset_sha256", "")).casefold()
                != identity_sha
            ):
                continue
            try:
                pair = (int(resolved["inline"]), int(resolved["crossline"]))
            except (KeyError, TypeError, ValueError):
                continue
            return pair, {
                "authority": "sealed_automatic_geometry_inspection",
                "receipt_sha256": raw_receipt.get("receipt_sha256"),
                "source_asset_sha256": identity_sha,
            }

    semantics = runtime_options.get("source_snapshot_semantics")
    if isinstance(semantics, Mapping):
        try:
            pair = (
                int(semantics["segy_inline_byte"]),
                int(semantics["segy_crossline_byte"]),
            )
        except (KeyError, TypeError, ValueError):
            pair = None
        if pair is not None:
            return pair, {
                "authority": "sealed_source_snapshot_semantics",
                "source_snapshot_semantics_sha256": runtime_options.get(
                    "source_snapshot_semantics_sha256"
                ),
                "source_asset_sha256": identity_sha or None,
            }
    return None, {"authority": "unavailable"}


def _f3_header_pair_digest(inline: np.ndarray, crossline: np.ndarray) -> str:
    pairs = np.column_stack((inline, crossline)).astype(">i4", copy=False)
    return hashlib.sha256(pairs.tobytes(order="C")).hexdigest()


def _audit_f3_header_pair_equivalence(
    *,
    source: Path,
    config: Mapping[str, Any],
    resolved_header_bytes: tuple[int, int],
    sealed_header_bytes: tuple[int, int] | None,
    sealed_authority: Mapping[str, Any],
    prediction_source_identity: Mapping[str, Any] | None,
    expected_trace_count: int,
    receipt_path: Path,
) -> dict[str, Any]:
    """Fail closed unless alternate line-number byte pairs agree on every trace."""

    source = source.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    identity = (
        dict(prediction_source_identity)
        if isinstance(prediction_source_identity, Mapping)
        else {}
    )
    authority = str(sealed_authority.get("authority", ""))
    sealed_recommendation_available = sealed_header_bytes is not None
    trusted_sealed_recommendation_authority = (
        authority in _F3_TRUSTED_SEALED_HEADER_AUTHORITIES
    )

    def eligibility(
        *,
        identical_positions: bool,
        full_scan: bool,
        trace_count_matches: bool,
        zero_mismatches: bool,
        pair_sha256_matches: bool,
    ) -> dict[str, Any]:
        identical_basis = (
            sealed_recommendation_available
            and trusted_sealed_recommendation_authority
            and identical_positions
        )
        equivalence_basis = (
            sealed_recommendation_available
            and trusted_sealed_recommendation_authority
            and not identical_positions
            and full_scan
            and trace_count_matches
            and zero_mismatches
            and pair_sha256_matches
        )
        checks = {
            "sealed_recommendation_available": sealed_recommendation_available,
            "trusted_sealed_recommendation_authority": (
                trusted_sealed_recommendation_authority
            ),
            "identical_header_positions": identical_positions,
            "full_trace_header_scan_performed": full_scan,
            "trace_count_matches": trace_count_matches,
            "zero_mismatch_trace_count": zero_mismatches,
            "pair_sha256_matches": pair_sha256_matches,
        }
        required_checks = {
            "sealed_recommendation_available": sealed_recommendation_available,
            "trusted_sealed_recommendation_authority": (
                trusted_sealed_recommendation_authority
            ),
        }
        if sealed_recommendation_available and identical_positions:
            required_checks["identical_header_positions"] = identical_positions
        elif sealed_recommendation_available:
            required_checks.update(
                {
                    "full_trace_header_scan_performed": full_scan,
                    "trace_count_matches": trace_count_matches,
                    "zero_mismatch_trace_count": zero_mismatches,
                    "pair_sha256_matches": pair_sha256_matches,
                }
            )
        return {
            "eligible": bool(identical_basis or equivalence_basis),
            "basis": (
                "trusted_identical_header_positions"
                if identical_basis
                else (
                    "trusted_all_trace_pair_equivalence"
                    if equivalence_basis
                    else None
                )
            ),
            "trusted_authorities": sorted(_F3_TRUSTED_SEALED_HEADER_AUTHORITIES),
            "checks": checks,
            "reason_codes": [
                name for name, passed in required_checks.items() if not passed
            ],
        }

    base: dict[str, Any] = {
        "contract_version": "well-seismic.f3-header-pair-equivalence.v1",
        "source": str(source),
        "source_sha256": identity.get("sha256"),
        "source_sha256_authority": "api_prediction_source_identity",
        "resolved_header_bytes": {
            "iline": int(resolved_header_bytes[0]),
            "xline": int(resolved_header_bytes[1]),
        },
        "sealed_recommended_header_bytes": (
            None
            if sealed_header_bytes is None
            else {
                "iline": int(sealed_header_bytes[0]),
                "xline": int(sealed_header_bytes[1]),
            }
        ),
        "sealed_recommendation": dict(sealed_authority),
        "expected_trace_count": int(expected_trace_count),
    }

    def write(document: Mapping[str, Any]) -> None:
        receipt_path.write_text(
            json.dumps(dict(document), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if sealed_header_bytes is None:
        validation_eligibility = eligibility(
            identical_positions=False,
            full_scan=False,
            trace_count_matches=False,
            zero_mismatches=False,
            pair_sha256_matches=False,
        )
        receipt = {
            **base,
            "status": "not_applicable_no_sealed_recommendation",
            "comparison_required": False,
            "full_trace_header_scan_performed": False,
            "validation_eligible": validation_eligibility["eligible"],
            "validation_eligibility": validation_eligibility,
            "passed": True,
        }
        write(receipt)
        return receipt
    if tuple(resolved_header_bytes) == tuple(sealed_header_bytes):
        validation_eligibility = eligibility(
            identical_positions=True,
            full_scan=False,
            trace_count_matches=False,
            zero_mismatches=False,
            pair_sha256_matches=False,
        )
        receipt = {
            **base,
            "status": "identical_header_positions",
            "comparison_required": False,
            "full_trace_header_scan_performed": False,
            "validation_eligible": validation_eligibility["eligible"],
            "validation_eligibility": validation_eligibility,
            "passed": True,
        }
        write(receipt)
        return receipt

    before = source.stat()
    try:
        reader = SegyReader(
            source,
            dict(config),
            {
                "profile": "standard_3d",
                "inline_byte": int(resolved_header_bytes[0]),
                "crossline_byte": int(resolved_header_bytes[1]),
            },
        )
        geometry = reader.inspect()
        fields = reader._read_header_fields(
            np.asarray(geometry.trace_offsets, dtype=np.int64),
            {
                "resolved_inline": (int(resolved_header_bytes[0]), 4),
                "resolved_crossline": (int(resolved_header_bytes[1]), 4),
                "sealed_inline": (int(sealed_header_bytes[0]), 4),
                "sealed_crossline": (int(sealed_header_bytes[1]), 4),
            },
            str(geometry.endian),
        )
        after = source.stat()
        if (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError("SEG-Y source changed during full header equivalence scan")
        resolved_inline = fields["resolved_inline"]
        resolved_crossline = fields["resolved_crossline"]
        sealed_inline = fields["sealed_inline"]
        sealed_crossline = fields["sealed_crossline"]
        mismatch = (resolved_inline != sealed_inline) | (
            resolved_crossline != sealed_crossline
        )
        mismatch_indices = np.flatnonzero(mismatch)
        observed_trace_count = int(len(resolved_inline))
        trace_count_matches = observed_trace_count == int(expected_trace_count)
        zero_mismatches = mismatch_indices.size == 0
        resolved_pair_sha256 = _f3_header_pair_digest(
            resolved_inline, resolved_crossline
        )
        sealed_pair_sha256 = _f3_header_pair_digest(sealed_inline, sealed_crossline)
        pair_sha256_matches = resolved_pair_sha256 == sealed_pair_sha256
        passed = trace_count_matches and zero_mismatches
        validation_eligibility = eligibility(
            identical_positions=False,
            full_scan=True,
            trace_count_matches=trace_count_matches,
            zero_mismatches=zero_mismatches,
            pair_sha256_matches=pair_sha256_matches,
        )
        first_mismatch = None
        if mismatch_indices.size:
            index = int(mismatch_indices[0])
            first_mismatch = {
                "trace_index": index,
                "resolved_pair": [
                    int(resolved_inline[index]),
                    int(resolved_crossline[index]),
                ],
                "sealed_pair": [
                    int(sealed_inline[index]),
                    int(sealed_crossline[index]),
                ],
            }
        receipt = {
            **base,
            "status": (
                "verified_all_trace_pairs_equivalent"
                if passed
                else "failed_header_pair_equivalence"
            ),
            "comparison_required": True,
            "full_trace_header_scan_performed": True,
            "observed_trace_count": observed_trace_count,
            "compared_trace_count": observed_trace_count,
            "mismatch_trace_count": int(mismatch_indices.size),
            "first_mismatch": first_mismatch,
            "resolved_pair_sha256": resolved_pair_sha256,
            "sealed_pair_sha256": sealed_pair_sha256,
            "source_stat_unchanged_during_scan": True,
            "validation_eligible": validation_eligibility["eligible"],
            "validation_eligibility": validation_eligibility,
            "passed": passed,
        }
        write(receipt)
    except Exception as exc:
        validation_eligibility = eligibility(
            identical_positions=False,
            full_scan=False,
            trace_count_matches=False,
            zero_mismatches=False,
            pair_sha256_matches=False,
        )
        receipt = {
            **base,
            "status": "failed_header_pair_equivalence_scan",
            "comparison_required": True,
            "full_trace_header_scan_performed": False,
            "validation_eligible": validation_eligibility["eligible"],
            "validation_eligibility": validation_eligibility,
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write(receipt)
        raise ValueError(
            "F3 SEG-Y alternate line-number header pair could not be proven equivalent"
        ) from exc
    if not receipt["passed"]:
        raise ValueError(
            "F3 SEG-Y resolved line-number bytes differ from the sealed recommendation; "
            "all-trace pair equivalence failed"
        )
    return receipt


def _normalized_subprocess_environment(
    source: dict[str, str] | None = None,
    *,
    windows: bool | None = None,
) -> dict[str, str]:
    """Return an environment safe for Windows child-process creation.

    Some launchers inject both ``Path`` and ``PATH``. Windows treats those
    names as identical, while Python can retain both spellings in
    ``os.environ``. Passing that mapping unchanged to a child process causes
    PowerShell and some Win32 wrappers to reject the duplicate key.
    """

    environment = dict(os.environ if source is None else source)
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        path_keys = [key for key in environment if key.casefold() == "path"]
        if len(path_keys) > 1:
            preferred = min(
                path_keys,
                key=lambda key: (key != "Path", -len(environment[key])),
            )
            path_value = environment[preferred]
            for key in path_keys:
                del environment[key]
            environment["Path"] = path_value
    environment["PYTHONUTF8"] = "1"
    return environment


def _wellfuse_subprocess_environment(wellfuse_root: Path) -> dict[str, str]:
    """Make the selected WellFuse checkout authoritative in child Python.

    The platform intentionally uses an isolated interpreter for inference.  That
    interpreter can also contain an older installed ``wellfuse5090`` package,
    so relying on its ambient import order can silently bind checkpoints to the
    interpreter directory instead of the configured WellFuse project.
    """

    source_root = (wellfuse_root / "src").resolve()
    if not (source_root / "wellfuse5090").is_dir():
        raise FileNotFoundError(f"WellFuse project source not found: {source_root}")
    environment = _normalized_subprocess_environment()
    existing = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + existing if existing else ""
    )
    environment["WELLFUSE_EXPECTED_SOURCE_ROOT"] = str(source_root)
    return environment


class PredictionRunnerRegistry:
    """Dispatch inference without coupling the API to a specific model."""

    entry_point_group = "well_seismic.prediction_runners"

    def __init__(self) -> None:
        self._runners: dict[str, PredictionRunner] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(
        self, model_id: str, runner: PredictionRunner, *, replace: bool = False
    ) -> None:
        if model_id in self._runners and not replace:
            raise ValueError(f"prediction runner already registered: {model_id}")
        self._runners[model_id] = runner

    def run(
        self, model_id: str, request: ModelInputRequest, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            runner = self._runners[model_id]
        except KeyError as exc:
            raise KeyError(
                f"no prediction runner registered for model: {model_id}"
            ) from exc
        return runner(request, **kwargs)

    def model_ids(self) -> list[str]:
        return list(self._runners)

    def load_entry_points(self) -> list[str]:
        """Use the entry-point name as ``model_id`` and its object as runner."""
        if interface_only_enabled():
            return []
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                self.register(entry_point.name, entry_point.load())
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {
                        "plugin": entry_point.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return loaded


def build_default_prediction_runners() -> PredictionRunnerRegistry:
    from .direct12b_runtime import DIRECT12B_MODEL_ID, run_direct12b_prediction
    from .layerpulse_contract import LAYERPULSE_MODEL_ID
    from .layerpulse_runtime import run_layerpulse_prediction

    registry = PredictionRunnerRegistry()
    # Keep the public runner registry present, but do not import or register
    # task executors in the extracted no-weight shell.  The API will therefore
    # return a structured "adapter required" state instead of failing later
    # while trying to open a missing checkpoint.
    if interface_only_enabled():
        return registry
    registry.register(LAYERPULSE_MODEL_ID, run_layerpulse_prediction)
    registry.register(FAULTSEG_MODEL_ID, run_faultseg_prediction)
    registry.register(
        FAULTNET_MODEL_ID,
        lambda request, **kwargs: run_faultseg_prediction(
            request,
            runtime_model_id=FAULTNET_MODEL_ID,
            **kwargs,
        ),
    )
    registry.register("seismic_surface_seg", run_surface_seg_prediction)
    registry.register(
        "wellfuse_channel_p17",
        lambda request, **kwargs: run_wellfuse_geobody_prediction(
            request, model_id="wellfuse_channel_p17", **kwargs
        ),
    )
    registry.register(
        "wellfuse_karst_p17",
        lambda request, **kwargs: run_wellfuse_geobody_prediction(
            request, model_id="wellfuse_karst_p17", **kwargs
        ),
    )
    # The four-surface implementation remains importable for historical
    # reproducibility, but is intentionally not registered for new requests.
    registry.register(GEOPATH_TIE_V1_MODEL_ID, run_geopath_tie_v1_prediction)
    direct12b_disabled = os.getenv("WELLFUSE_DISABLE_DIRECT12B", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not direct12b_disabled:
        registry.register(DIRECT12B_MODEL_ID, run_direct12b_prediction)
    registry.register("wellfuse_facies_3d_p17", run_wellfuse_facies_3d_prediction)
    registry.register(
        FAST_FACIES_1D_MODEL_ID, run_wellfuse_facies_1d_chengdu_fast_prediction
    )
    registry.register(F3_FACIES_3D_MODEL_ID, run_wellfuse_facies_3d_f3_prediction)
    registry.register(
        FRACTURE_DEVELOPMENT_MODEL_ID,
        lambda request, **kwargs: run_unified_raw_well_prediction(
            request, model_id=FRACTURE_DEVELOPMENT_MODEL_ID, **kwargs
        ),
    )
    for model_id in (
        "wellfuse_facies_1d_p17",
        "wellfuse_den_p18",
        "wellfuse_por_p18",
        "wellfuse_log_perm_p18",
        "wellfuse_sw_p18",
        "wellfuse_vsh_p18",
    ):
        registry.register(
            model_id,
            lambda request, _model_id=model_id, **kwargs: run_wellfuse_well_prediction(
                request, model_id=_model_id, **kwargs
            ),
        )
    for model_id in (FAST_FLUID_MODEL_ID, *FAST_PROPERTY_MODEL_TARGETS):
        registry.register(
            model_id,
            lambda request, _model_id=model_id, **kwargs: run_fast_downstream_prediction(
                request, model_id=_model_id, **kwargs
            ),
        )
    return registry


def _wellfuse_runtime_paths(project_root: Path) -> tuple[Path, Path]:
    configured_root = os.getenv("WELLFUSE_PROJECT_ROOT")
    wellfuse_root = Path(
        configured_root or project_root / "runtime" / "wellfuse"
    ).expanduser().resolve()
    configured_python = os.getenv("WELLFUSE_PYTHON")
    if configured_python:
        python_executable = Path(configured_python).expanduser().resolve()
    else:
        python_executable = (
            project_root / "runtime" / "python-wellfuse" / "python.exe"
        ).resolve()
    if not (wellfuse_root / "src" / "wellfuse5090").is_dir():
        raise FileNotFoundError(f"WellFuse project source not found: {wellfuse_root}")
    if not python_executable.is_file():
        raise FileNotFoundError(
            "WellFuse inference Python not found; set WELLFUSE_PYTHON to the "
            f"CUDA environment executable: {python_executable}"
        )
    return wellfuse_root, python_executable


def _faultseg_subprocess_python(project_root: Path) -> Path:
    """Resolve the isolated CUDA-capable Python used by FaultSeg."""

    configured = os.getenv("WELLFUSE_PYTHON")
    if configured:
        executable = Path(configured).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"WELLFUSE_PYTHON does not point to a Python executable: {executable}"
            )
        return executable
    bundled_python = (
        project_root / "runtime" / "python-wellfuse" / "python.exe"
    ).resolve()
    if bundled_python.is_file():
        return bundled_python
    raise FileNotFoundError(
        "Bundled CUDA inference Python not found; set WELLFUSE_PYTHON explicitly "
        f"or restore the platform runtime: {bundled_python}"
    )


def _json_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} must contain a JSON object: {path}")
    return payload


def _stdout_json_mapping(stdout: str, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        tail = "\n".join(stdout.splitlines()[-30:])
        raise ValueError(f"{description} did not emit one JSON object:\n{tail}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} must emit a JSON object")
    return payload


def _fast_downstream_validation(
    wellfuse_root: Path,
    *,
    model_id: str,
    dataset: str,
    features: str | None,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_id == FAST_FLUID_MODEL_ID:
        evidence_dataset = "chengdu" if dataset == "chengdu" else "northwest_all"
        assert features is not None
        metrics_path = (
            wellfuse_root
            / "artifacts"
            / "fluid_interpretation"
            / evidence_dataset
            / f"{features}_fixed_last"
            / "metrics.json"
        ).resolve()
        metrics = _json_mapping(metrics_path, description="fluid fixed-last metrics")
        configuration = metrics["configuration"]
        oof = metrics["oof"]
        final_model = metrics["final_model"]
        if not all(
            isinstance(item, Mapping) for item in (configuration, oof, final_model)
        ):
            raise TypeError("fluid fixed-last metrics have an invalid structure")
        selection = str(configuration.get("selection", ""))
        if selection != "fixed_last":
            raise ValueError(f"fluid checkpoint selection is not fixed_last: {selection}")
        pooled = oof["pooled"]
        baseline = oof["majority_baseline"]
        if not isinstance(pooled, Mapping) or not isinstance(baseline, Mapping):
            raise TypeError("fluid OOF metrics have an invalid structure")
        hydrocarbon = pooled["hydrocarbon_binary"]
        baseline_pooled = baseline["pooled"]
        if not isinstance(hydrocarbon, Mapping) or not isinstance(
            baseline_pooled, Mapping
        ):
            raise TypeError("fluid OOF validation metrics have an invalid structure")
        validation = {
            "metric_name": "material_macro_f1",
            "metric_value": float(pooled["macro_f1_material"]),
            "hydrocarbon_f1": float(hydrocarbon["f1"]),
            "baseline_value": float(baseline_pooled["macro_f1_material"]),
            "evidence_dataset": evidence_dataset,
            "metrics_json": str(metrics_path),
        }
        checkpoint_selection = {
            "selection": selection,
            "epochs": int(final_model["epochs"]),
            "ensemble_size": 1,
            "checkpoints": [str(summary["checkpoint"])],
        }
        return validation, checkpoint_selection

    target = FAST_PROPERTY_MODEL_TARGETS[model_id]
    epochs = 8 if target in {"DEN", "LOG_PERM", "VSH"} else 4
    metrics_path = (
        wellfuse_root
        / "artifacts"
        / "fast_downstream"
        / "northwest_property"
        / f"northwest_all_fixed_last_{epochs}ep"
        / "metrics.json"
    ).resolve()
    metrics = _json_mapping(metrics_path, description="property fixed-last metrics")
    selection = str(metrics.get("selection", ""))
    if selection != "fixed_last":
        raise ValueError(f"property checkpoint selection is not fixed_last: {selection}")
    targets = metrics["targets"]
    if not isinstance(targets, Mapping) or not isinstance(targets.get(target), Mapping):
        raise TypeError(f"property fixed-last metrics do not contain target {target}")
    target_metrics = targets[target]
    model_metrics = target_metrics["oof_model"]
    baseline_metrics = target_metrics["oof_baseline"]
    if not isinstance(model_metrics, Mapping) or not isinstance(
        baseline_metrics, Mapping
    ):
        raise TypeError(f"property OOF metrics have an invalid structure for {target}")
    validation = {
        "metric_name": "mae",
        "metric_value": float(model_metrics["mae"]),
        "baseline_value": float(baseline_metrics["mae"]),
        "r2": float(model_metrics["r2"]),
        "evidence_dataset": "northwest_all",
        "metrics_json": str(metrics_path),
    }
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise TypeError("property inference summary must list checkpoints")
    checkpoint_selection = {
        "selection": selection,
        "epochs": epochs,
        "ensemble_size": int(summary.get("ensemble_size", len(checkpoints))),
        "checkpoints": [str(path) for path in checkpoints],
    }
    return validation, checkpoint_selection


def _resolved_cli_artifact(raw_path: object, root: Path, *, description: str) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def run_unified_raw_well_prediction(
    request: ModelInputRequest,
    *,
    model_id: str,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the stable WellFuse raw-well CLI contract for one downstream model."""

    del config, threshold, patch_size, overlap
    try:
        task_id = RAW_WELL_TASK_BY_MODEL[model_id]
    except KeyError as exc:
        raise KeyError(f"unsupported raw-well model: {model_id}") from exc
    requested_device = str(device_name).strip().casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported WellFuse device: {requested_device}")
    runtime_options = {**request.options, **dict(options or {})}
    batch = adapters.get(model_id).prepare(
        ModelInputRequest(source=request.source, options=runtime_options)
    )
    if batch.provenance.get("input_mode") != "raw_wells":
        raise ValueError(f"{model_id} raw-well runner requires raw_well_paths or raw_well_root")

    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    script = wellfuse_root / "scripts" / "infer_downstream_raw_wells.py"
    if not script.is_file():
        raise FileNotFoundError(f"WellFuse unified raw-well CLI not found: {script}")
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_root = output_directory / "predictions"
    runtime_log = output_directory / "raw_well_runtime.log"
    command = [
        str(python_executable),
        str(script),
        "--task-id",
        task_id,
        "--model-id",
        model_id,
        "--output-dir",
        str(prediction_root),
        "--device",
        requested_device,
    ]
    for path in batch.provenance.get("raw_well_paths") or []:
        command.extend(("--raw-well-path", str(path)))
    if batch.provenance.get("raw_well_root"):
        command.extend(("--raw-well-root", str(batch.provenance["raw_well_root"])))
    for path in batch.provenance.get("trajectory_paths") or []:
        command.extend(("--trajectory-path", str(path)))
    for path in batch.provenance.get("wellhead_paths") or []:
        command.extend(("--wellhead-path", str(path)))
    if runtime_options.get("batch_size") is not None:
        command.extend(("--batch-size", str(int(runtime_options["batch_size"]))))

    if progress:
        progress(12, "已核验赛事原始井文件，正在执行统一九线推理合同")
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
    runtime_log.write_text(
        "\n".join(
            (
                f"command={json.dumps(command, ensure_ascii=False)}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
            )
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"WellFuse raw-well inference failed (exit {completed.returncode}):\n{tail}"
        )

    summary_path = prediction_root / "summary.json"
    summary = _json_mapping(summary_path, description="raw-well summary")
    raw_summary_schema = str(summary.get("schema_version", ""))
    if raw_summary_schema not in {
        "wellfuse.raw_well_downstream.inference.v1",
        "wellfuse.raw_well_downstream.inference.v2",
    }:
        raise ValueError("raw-well summary schema_version is incompatible")
    if str(summary.get("task_id")) != task_id or str(summary.get("model_id")) != model_id:
        raise ValueError("raw-well summary task/model binding differs from request")
    raw_wells = summary.get("wells")
    if not isinstance(raw_wells, list) or not raw_wells:
        raise ValueError("raw-well summary contains no well outputs")

    outputs: dict[str, str] = {
        "summary_json": str(summary_path.resolve()),
        "runtime_log": str(runtime_log.resolve()),
    }
    summary_outputs = summary.get("outputs")
    manifest_reference = (
        summary_outputs.get("manifest_json") or summary_outputs.get("manifest")
        if isinstance(summary_outputs, Mapping)
        else None
    )
    if manifest_reference:
        manifest_path = _resolved_cli_artifact(
            manifest_reference, prediction_root, description="raw-well manifest"
        )
    else:
        manifest_path = _resolved_cli_artifact(
            "manifest.json", prediction_root, description="raw-well manifest"
        )
    manifest_document = _json_mapping(
        manifest_path, description="raw-well manifest"
    )
    outputs["manifest_json"] = str(manifest_path)
    property_target = FAST_PROPERTY_MODEL_TARGETS.get(model_id)
    requires_v2_output_contract = bool(
        model_id
        in {
            FAST_FLUID_MODEL_ID,
            FAST_FACIES_1D_MODEL_ID,
            FRACTURE_DEVELOPMENT_MODEL_ID,
        }
        or property_target in PHYSICAL_BOUNDED_PROPERTY_TARGETS
    )
    if requires_v2_output_contract:
        if raw_summary_schema != "wellfuse.raw_well_downstream.inference.v2":
            raise ValueError(
                "raw-well facies/physical-property output requires the explicit v2 contract"
            )
        if (
            manifest_document.get("schema_version")
            != "wellfuse.raw_well_downstream.manifest.v2"
            or str(manifest_document.get("task_id")) != task_id
            or str(manifest_document.get("model_id")) != model_id
        ):
            raise ValueError("raw-well manifest does not carry the v2 output contract")
        summary_contract = summary.get("output_contract")
        if not isinstance(summary_contract, Mapping) or dict(
            manifest_document.get("output_contract") or {}
        ) != dict(summary_contract):
            raise ValueError("raw-well summary/manifest output contracts differ")
        if model_id == FAST_FLUID_MODEL_ID:
            if (
                summary.get("primary_output") != "fluid_intervals_csv"
                or summary.get("diagnostic_output")
                != "fluid_decision_receipt_json"
                or summary_contract.get("contract_version")
                != "wellfuse.fluid-interpretation-output.v3"
                or summary_contract.get("primary_interval_output")
                != "fluid_intervals_csv"
                or summary_contract.get("primary_decision_rule")
                != "minimum_continuous_thickness_same_class_bridge"
                or summary_contract.get("published_granularity")
                != "continuous_md_intervals"
                or summary_contract.get("probability_usage")
                != "internal_decoding_only_not_persisted"
                or summary_contract.get("point_output_persisted") is not False
                or summary_contract.get("class_probability_persisted") is not False
            ):
                raise ValueError("raw-well fluid decision contract is invalid")
        elif model_id == FAST_FACIES_1D_MODEL_ID:
            if (
                summary.get("primary_output") != "facies_code"
                or summary.get("diagnostic_output") is not None
                or summary_contract.get("contract_version")
                != "wellfuse.facies-1d-output.v3"
                or summary_contract.get("primary_interval_output")
                != "facies_intervals_csv"
                or summary_contract.get("primary_decision_rule")
                != "sequence_viterbi"
                or list(summary_contract.get("public_probability_outputs") or [])
                or list(summary_contract.get("public_uncertainty_outputs") or [])
            ):
                raise ValueError("raw-well facies primary/diagnostic contract is invalid")
        elif model_id == FRACTURE_DEVELOPMENT_MODEL_ID:
            if (
                summary.get("primary_output") != "fracture_intervals_csv"
                or summary.get("diagnostic_output")
                != "fracture_development_score"
                or summary_contract.get("contract_version")
                != "wellfuse.fracture-development-output.v1"
                or summary_contract.get("decision_rule")
                != "rolling_median_9_then_within_well_tertiles"
                or summary_contract.get("score_semantics")
                != "internal_relative_ranking_not_probability"
                or summary_contract.get("spatial_scope")
                != "well_side_only_not_3d_fracture_segmentation"
            ):
                raise ValueError("raw-well fracture interval contract is invalid")
        else:
            assert property_target is not None
            if (
                summary.get("primary_output")
                != f"{property_target}_physical_bounded"
                or summary_contract.get("raw_evidence_output")
                != f"{property_target}_prediction_raw"
                or summary_contract.get("legacy_raw_output")
                != f"{property_target}_prediction"
                or list(summary_contract.get("physical_bounds") or []) != [0.0, 1.0]
                or summary_contract.get("transform") != "clip"
            ):
                raise ValueError("raw-well physical property output contract is invalid")
    rejected_inputs: dict[str, Any] | None = None
    rejected_reference = (
        summary_outputs.get("rejected_inputs")
        if isinstance(summary_outputs, Mapping)
        else None
    )
    if rejected_reference:
        rejected_path = _resolved_cli_artifact(
            rejected_reference,
            prediction_root,
            description="raw-well rejected-input receipt",
        )
        rejected_inputs = _json_mapping(
            rejected_path,
            description="raw-well rejected-input receipt",
        )
        if (
            rejected_inputs.get("schema_version")
            != "wellfuse.raw_well_downstream.rejected_inputs.v1"
            or str(rejected_inputs.get("task_id")) != task_id
            or str(rejected_inputs.get("model_id")) != model_id
        ):
            raise ValueError("raw-well rejected-input receipt binding is incompatible")
        if int(rejected_inputs.get("accepted_input_count", -1)) != len(raw_wells):
            raise ValueError("raw-well rejected-input receipt accepted count differs from outputs")
        outputs["rejected_inputs_json"] = str(rejected_path)
    well_outputs: list[dict[str, Any]] = []
    for index, raw_well in enumerate(raw_wells, start=1):
        if not isinstance(raw_well, Mapping):
            raise TypeError("raw-well output must be a JSON object")
        raw_artifacts = raw_well.get("artifacts")
        artifacts = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
        csv_reference = (
            artifacts.get("fluid_intervals_csv")
            if model_id == FAST_FLUID_MODEL_ID
            else raw_well.get("prediction_csv") or artifacts.get("prediction_csv")
        )
        csv_path = _resolved_cli_artifact(
            csv_reference,
            prediction_root,
            description=(
                f"raw-well fluid interval CSV {index}"
                if model_id == FAST_FLUID_MODEL_ID
                else f"raw-well prediction CSV {index}"
            ),
        )
        if requires_v2_output_contract:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            if model_id == FAST_FACIES_1D_MODEL_ID:
                required_columns = {
                    "well_id",
                    "MD_m",
                    "interval_id",
                    "facies_code",
                    "facies_name",
                }
            elif model_id == FAST_FLUID_MODEL_ID:
                required_columns = {
                    "well_id",
                    "interval_id",
                    "top_md_m",
                    "bottom_md_m",
                    "thickness_m",
                    "fluid_class_code",
                    "fluid_class",
                    "fluid_class_zh",
                    "sample_count",
                    "merged_sample_count",
                }
            elif model_id == FRACTURE_DEVELOPMENT_MODEL_ID:
                required_columns = {
                    "well_id",
                    "MD_m",
                    "interval_id",
                    "fracture_level_code",
                    "fracture_level",
                    "fracture_level_zh",
                    "relative_score_audit",
                }
            else:
                assert property_target is not None
                required_columns = {
                    f"{property_target}_prediction",
                    f"{property_target}_prediction_raw",
                    f"{property_target}_physical_bounded",
                }
            if not required_columns <= set(header):
                raise ValueError(
                    "raw-well prediction CSV is missing the declared v2 primary/evidence fields"
                )
            if model_id == FAST_FLUID_MODEL_ID and any(
                token in str(column).casefold()
                for column in header
                for token in ("prob", "confidence")
            ):
                raise ValueError(
                    "fluid interval CSV must not expose probability or confidence columns"
                )
            if model_id == FAST_FACIES_1D_MODEL_ID and any(
                token in str(column).casefold()
                for column in header
                for token in ("prob", "confidence", "entropy", "uncertainty")
            ):
                raise ValueError(
                    "facies public point CSV must contain deterministic classes only"
                )
            if model_id == FRACTURE_DEVELOPMENT_MODEL_ID and any(
                "prob" in str(column).casefold() for column in header
            ):
                raise ValueError(
                    "fracture point evidence must not expose probability fields"
                )
        if model_id != FAST_FLUID_MODEL_ID:
            outputs[f"well_{index:03d}_prediction_csv"] = str(csv_path)
        fluid_interval_path: Path | None = None
        fluid_decision_receipt_path: Path | None = None
        fracture_interval_path: Path | None = None
        for artifact_name, output_name in (
            ("prediction_las", "prediction_las"),
            ("canonical_npz", "canonical_input_npz"),
            ("facies_intervals_csv", "facies_intervals_csv"),
            ("fluid_intervals_csv", "fluid_intervals_csv"),
            ("fracture_intervals_csv", "fracture_intervals_csv"),
            (
                "fluid_decision_receipt_json",
                "fluid_decision_receipt_json",
            ),
        ):
            if artifacts.get(artifact_name):
                artifact_path = _resolved_cli_artifact(
                    artifacts[artifact_name],
                    prediction_root,
                    description=f"raw-well {artifact_name} {index}",
                )
                outputs[f"well_{index:03d}_{output_name}"] = str(artifact_path)
                if artifact_name == "fluid_intervals_csv":
                    fluid_interval_path = artifact_path
                elif artifact_name == "fluid_decision_receipt_json":
                    fluid_decision_receipt_path = artifact_path
                elif artifact_name == "fracture_intervals_csv":
                    fracture_interval_path = artifact_path
        if model_id == FAST_FLUID_MODEL_ID:
            if fluid_interval_path != csv_path or fluid_decision_receipt_path is None:
                raise ValueError(
                    "raw-well fluid output must bind one interval CSV and decision receipt"
                )
            receipt = _json_mapping(
                fluid_decision_receipt_path,
                description=f"raw-well fluid decision receipt {index}",
            )
            if (
                receipt.get("contract_version")
                != "wellfuse.fluid-sequence-decision-receipt.v1"
                or receipt.get("probability_persisted") is not False
                or dict(receipt.get("outputs") or {})
                != {"fluid_intervals_csv": fluid_interval_path.name}
            ):
                raise ValueError("raw-well fluid decision receipt is incompatible")
        if model_id == FRACTURE_DEVELOPMENT_MODEL_ID:
            if fracture_interval_path is None:
                raise ValueError("raw-well fracture output has no interval CSV")
            with fracture_interval_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                interval_header = next(csv.reader(stream), [])
            required_interval_columns = {
                "well_id",
                "interval_id",
                "top_md_m",
                "bottom_md_m",
                "thickness_m",
                "fracture_level_code",
                "fracture_level",
                "fracture_level_zh",
                "sample_count",
                "mean_relative_score",
            }
            if not required_interval_columns <= set(interval_header) or any(
                "prob" in str(column).casefold() for column in interval_header
            ):
                raise ValueError("raw-well fracture interval CSV contract is invalid")
        well_outputs.append(
            {
                "well_id": str(raw_well.get("well_id", "")),
                "sample_count": int(raw_well.get("sample_count", 0)),
                **(
                    {"fluid_intervals_csv": str(fluid_interval_path)}
                    if fluid_interval_path is not None
                    else {"prediction_csv": str(csv_path)}
                ),
                **(
                    {
                        "fluid_decision_receipt_json": str(
                            fluid_decision_receipt_path
                        )
                    }
                    if fluid_decision_receipt_path is not None
                    else {}
                ),
                **(
                    {"fracture_intervals_csv": str(fracture_interval_path)}
                    if fracture_interval_path is not None
                    else {}
                ),
                "source_path": str(raw_well.get("source_path", "")),
                "observed_curve_count": int(raw_well.get("observed_curve_count", 0)),
                "missing_curves": [
                    str(value)
                    for value in (
                        raw_well.get("missing_curves")
                        or raw_well.get("missing_curve_list")
                        or []
                    )
                ],
                "trajectory_status": str(raw_well.get("trajectory_status", "")),
                "applicability_status": str(
                    raw_well.get("applicability_status", "")
                ),
                **(
                    {"physical_bounds_audit": dict(raw_well["physical_bounds_audit"])}
                    if isinstance(raw_well.get("physical_bounds_audit"), Mapping)
                    else {}
                ),
                **(
                    {"predicted_class_counts": dict(raw_well["predicted_class_counts"])}
                    if isinstance(raw_well.get("predicted_class_counts"), Mapping)
                    else {}
                ),
                **(
                    {"decoded_class_counts": dict(raw_well["decoded_class_counts"])}
                    if isinstance(raw_well.get("decoded_class_counts"), Mapping)
                    else {}
                ),
                **(
                    {
                        "fracture_level_counts": dict(
                            raw_well["fracture_level_counts"]
                        )
                    }
                    if isinstance(raw_well.get("fracture_level_counts"), Mapping)
                    else {}
                ),
                **(
                    {
                        "relative_level_thresholds": dict(
                            raw_well["relative_level_thresholds"]
                        )
                    }
                    if isinstance(
                        raw_well.get("relative_level_thresholds"), Mapping
                    )
                    else {}
                ),
                **(
                    {
                        "raw_argmax_class_counts": dict(
                            raw_well["raw_argmax_class_counts"]
                        )
                    }
                    if isinstance(
                        raw_well.get("raw_argmax_class_counts"), Mapping
                    )
                    else {}
                ),
                **(
                    {
                        "interval_count": int(raw_well["interval_count"]),
                        "merged_spike_sample_count": int(
                            raw_well.get("merged_spike_sample_count", 0)
                        ),
                    }
                    if raw_well.get("interval_count") is not None
                    else {}
                ),
            }
        )

    if property_target in PHYSICAL_BOUNDED_PROPERTY_TARGETS:
        aggregate_audit = summary.get("physical_bounds_audit")
        if not isinstance(aggregate_audit, Mapping):
            raise ValueError("raw-well bounded property summary lacks aggregate audit")
        if (
            int(aggregate_audit.get("sample_count", -1))
            != sum(
                int(item["physical_bounds_audit"].get("sample_count", 0))
                for item in well_outputs
            )
            or int(aggregate_audit.get("out_of_bounds_count", -1))
            != sum(
                int(item["physical_bounds_audit"].get("out_of_bounds_count", 0))
                for item in well_outputs
            )
            or aggregate_audit.get("raw_evidence_preserved") is not True
        ):
            raise ValueError("raw-well bounded property aggregate audit differs from wells")

    result_path = output_directory / "result.json"
    outputs["result_json"] = str(result_path.resolve())
    raw_checkpoints = summary.get("checkpoints")
    checkpoints = (
        [str(value) for value in raw_checkpoints]
        if isinstance(raw_checkpoints, list)
        else []
    )
    snapshot_property_run = bool(
        model_id in FAST_PROPERTY_MODEL_TARGETS
        and batch.provenance.get("source_mode") == "sealed_snapshot"
    )
    result: dict[str, Any] = {
        "model_id": model_id,
        "model_name": public_runtime_model_name(model_id),
        "model_executed": True,
        "execution_status": (
            "completed_sealed_snapshot_inference"
            if snapshot_property_run
            else "completed_raw_well_inference"
        ),
        "scientific_status": (
            "candidate_single_well_validation"
            if model_id == FRACTURE_DEVELOPMENT_MODEL_ID
            else "experimental_unknown_well_transfer"
        ),
        "scope": (
            "sealed_snapshot_whole_wells"
            if snapshot_property_run
            else "raw_unknown_wells"
        ),
        "dataset": "sealed_snapshot" if snapshot_property_run else "raw_wells",
        "well_count": int(summary.get("well_count", len(well_outputs))),
        "discovered_input_count": int(
            summary.get("discovered_input_count", len(well_outputs))
        ),
        "accepted_input_count": int(
            summary.get("accepted_input_count", len(well_outputs))
        ),
        "rejected_input_count": int(summary.get("rejected_input_count", 0)),
        "device": str(summary.get("device", requested_device)),
        "input": {"source": str(request.source), "axes": ["MD"], **batch.provenance},
        "inference": {
            "mode": (
                "sealed_snapshot_well_sequence"
                if snapshot_property_run
                else "raw_well_sequence"
            ),
            "schema_version": summary["schema_version"],
        },
        "well_outputs": well_outputs,
        "checkpoint_selection": {
            "checkpoint": summary.get("checkpoint"),
            "checkpoints": checkpoints,
        },
        "outputs": outputs,
        "warnings": [
            (
                "当前封存快照井资产属于新工区应用，既有验证指标不自动外推。"
                if snapshot_property_run
                else "原始井推理属于未知工区迁移，数据集内验证指标不自动外推。"
            ),
            *(str(value) for value in (summary.get("warnings") or [])),
        ],
    }
    if rejected_inputs is not None:
        result["rejected_inputs"] = rejected_inputs
        if result["rejected_input_count"]:
            result["warnings"].append(
                f"已逐文件隔离 {result['rejected_input_count']} 个无法解析的原始井输入；"
                "有效井已继续推理，详见 rejected_inputs_json。"
            )
    for key in (
        "target",
        "units",
        "classes",
        "input_contract",
        "primary_output",
        "diagnostic_output",
        "output_contract",
        "physical_bounds_audit",
        "applicability",
    ):
        if summary.get(key) is not None:
            result[key] = summary[key]
    if model_id == FRACTURE_DEVELOPMENT_MODEL_ID:
        result["warnings"].append(
            "裂缝输出为相对发育排序，不是精确裂缝点或绝对事件密度。"
        )
    elif model_id == FAST_FACIES_1D_MODEL_ID:
        result["warnings"].append(
            "主成果为Viterbi确定性相序列与连续层段；类别概率仅在进程内参与解码，不作为公开文件。"
        )
    elif model_id == FAST_FLUID_MODEL_ID:
        result["warnings"].append(
            "流体主成果仅为连续MD层段；五类概率只在进程内参与解码，不落盘或公开下载。"
        )
    elif property_target in PHYSICAL_BOUNDED_PROPERTY_TARGETS:
        result["warnings"].append(
            f"{property_target}正式主曲线按[0,1]裁剪；原始回归值和越界计数完整保留供审计。"
        )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress:
        progress(98, f"已写出{len(well_outputs)}口井的逐深度CSV与输入QC清单")
    return result


def run_fast_downstream_prediction(
    request: ModelInputRequest,
    *,
    model_id: str,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run retained dataset-bound fluid or Northwest property checkpoints."""

    del config, threshold, patch_size, overlap
    if model_id != FAST_FLUID_MODEL_ID and model_id not in FAST_PROPERTY_MODEL_TARGETS:
        raise KeyError(f"unsupported fast downstream model: {model_id}")
    requested_device = str(device_name).strip().casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported WellFuse device: {requested_device}")
    runtime_options = {**request.options, **dict(options or {})}
    adapter_request = ModelInputRequest(
        source=request.source,
        crop_start=request.crop_start,
        crop_size=request.crop_size,
        options=runtime_options,
    )
    batch = adapters.get(model_id).prepare(adapter_request)
    if batch.provenance.get("input_mode") == "raw_wells":
        return run_unified_raw_well_prediction(
            adapter_request,
            model_id=model_id,
            adapters=adapters,
            config={},
            project_root=project_root,
            output_directory=output_directory,
            device_name=device_name,
            options=runtime_options,
            progress=progress,
        )
    dataset = str(batch.provenance["dataset"])
    well_ids = [str(value) for value in batch.provenance["well_ids"]]
    if progress:
        progress(8, f"已锁定{dataset}中的{len(well_ids)}口整井与fixed-last权重")

    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_root = output_directory / "predictions"
    runtime_log = output_directory / "fast_downstream_runtime.log"
    script: Path
    command: list[str]
    features: str | None = None
    target = FAST_PROPERTY_MODEL_TARGETS.get(model_id)
    if model_id == FAST_FLUID_MODEL_ID:
        features = str(
            runtime_options.get(
                "features", "seismic_fusion" if dataset == "chengdu" else "well_only"
            )
        ).strip().casefold()
        if features not in {"well_only", "seismic_fusion"}:
            raise ValueError("fluid features must be well_only or seismic_fusion")
        if dataset != "chengdu" and features != "well_only":
            raise ValueError("Northwest fluid inference supports well_only features")
        script = wellfuse_root / "scripts" / "infer_fluid_interpretation.py"
        command = [
            str(python_executable),
            str(script),
            "--dataset",
            dataset,
            "--features",
            features,
            "--output-root",
            str(prediction_root),
        ]
        for well_id in well_ids:
            command.extend(("--well-id", well_id))
        if requested_device == "cpu":
            command.append("--cpu")
        if runtime_options.get("northwest_root"):
            command.extend(("--northwest-root", str(runtime_options["northwest_root"])))
        if runtime_options.get("chengdu_asset_root"):
            command.extend(
                ("--chengdu-asset-root", str(runtime_options["chengdu_asset_root"]))
            )
    else:
        assert target is not None
        script = wellfuse_root / "scripts" / "infer_northwest_property.py"
        command = [
            str(python_executable),
            str(script),
            "--dataset",
            dataset,
            "--target",
            target,
            "--well-ids",
            *well_ids,
            "--output-dir",
            str(prediction_root),
            "--device",
            requested_device,
        ]
        data_root = runtime_options.get("data_root", runtime_options.get("northwest_root"))
        if data_root:
            command.extend(("--data-root", str(data_root)))
    if not script.is_file():
        raise FileNotFoundError(f"WellFuse fast downstream script not found: {script}")
    if runtime_options.get("batch_size") is not None:
        command.extend(("--batch-size", str(int(runtime_options["batch_size"]))))

    if progress:
        progress(18, "正在执行WellFuse数据集内整井推理")
    completed = managed_run(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    runtime_log.write_text(
        "\n".join(
            (
                f"command={json.dumps(command, ensure_ascii=False)}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
            )
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"WellFuse fast downstream inference failed (exit {completed.returncode}):\n{tail}"
        )

    summary_path = prediction_root / "summary.json"
    summary = _json_mapping(summary_path, description="fast downstream summary")
    if str(summary.get("dataset", "")) != dataset:
        raise ValueError("fast downstream summary dataset differs from the request")
    if target is not None and str(summary.get("target", "")) != target:
        raise ValueError("fast downstream summary target differs from the requested model")
    bounded_property_contract: Mapping[str, Any] | None = None
    fluid_output_contract: Mapping[str, Any] | None = None
    if model_id == FAST_FLUID_MODEL_ID:
        if (
            summary.get("schema_version")
            != "wellfuse.fluid_interpretation.batch_inference.v3"
        ):
            raise ValueError(
                "fluid inference requires the interval-only schema v3"
            )
        raw_contract = summary.get("output_contract")
        if not isinstance(raw_contract, Mapping):
            raise ValueError("fluid summary is missing output_contract")
        if (
            summary.get("primary_output") != "fluid_intervals_csv"
            or summary.get("diagnostic_output")
            != "fluid_decision_receipt_json"
            or raw_contract.get("contract_version")
            != "wellfuse.fluid-interpretation-output.v3"
            or raw_contract.get("primary_interval_output")
            != "fluid_intervals_csv"
            or raw_contract.get("primary_decision_rule")
            != "minimum_continuous_thickness_same_class_bridge"
            or raw_contract.get("published_granularity")
            != "continuous_md_intervals"
            or raw_contract.get("probability_usage")
            != "internal_decoding_only_not_persisted"
            or raw_contract.get("point_output_persisted") is not False
            or raw_contract.get("class_probability_persisted") is not False
        ):
            raise ValueError("fluid summary output contract is incompatible")
        fluid_output_contract = raw_contract
    if target in PHYSICAL_BOUNDED_PROPERTY_TARGETS:
        if summary.get("schema_version") != "wellfuse.northwest-property-inference.v2":
            raise ValueError(
                "bounded POR/SW/VSH output requires northwest-property inference schema v2"
            )
        raw_contract = summary.get("output_contract")
        if not isinstance(raw_contract, Mapping):
            raise ValueError("bounded property summary is missing output_contract")
        expected_primary = f"{target}_PRED_PHYSICAL_BOUNDED"
        if (
            raw_contract.get("primary_output") != expected_primary
            or raw_contract.get("raw_evidence_output") != f"{target}_PRED_MEAN_RAW"
            or list(raw_contract.get("physical_bounds") or []) != [0.0, 1.0]
            or raw_contract.get("transform") != "clip"
        ):
            raise ValueError("bounded property summary output contract is incompatible")
        bounded_property_contract = raw_contract
    if requested_device == "cuda" and model_id == FAST_FLUID_MODEL_ID:
        if str(summary.get("device", "")).casefold() != "cuda":
            raise RuntimeError("CUDA was requested but fluid inference executed on CPU")

    raw_wells = summary.get("wells")
    if not isinstance(raw_wells, list) or not raw_wells:
        raise ValueError("fast downstream summary contains no well outputs")
    csv_field = "csv" if model_id == FAST_FLUID_MODEL_ID else "prediction_csv"
    outputs: dict[str, str] = {
        "summary_json": str(summary_path.resolve()),
        "runtime_log": str(runtime_log.resolve()),
    }
    well_outputs: list[dict[str, Any]] = []
    for index, raw_well in enumerate(raw_wells, start=1):
        if not isinstance(raw_well, Mapping):
            raise TypeError("fast downstream well summary must be a JSON object")
        csv_path = Path(str(raw_well.get(csv_field, ""))).expanduser()
        if not csv_path.is_absolute():
            csv_path = summary_path.parent / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"fast downstream well CSV not found: {csv_path}")
        if fluid_output_contract is not None:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            required_columns = {
                "well_id",
                "interval_id",
                "top_md_m",
                "bottom_md_m",
                "thickness_m",
                "fluid_class_code",
                "fluid_class",
                "fluid_class_zh",
                "sample_count",
                "merged_sample_count",
            }
            if not required_columns <= set(header):
                raise ValueError(
                    "fluid interval CSV is missing deterministic layer fields"
                )
            if any(
                token in str(column).casefold()
                for column in header
                for token in ("prob", "confidence")
            ):
                raise ValueError(
                    "fluid interval CSV must not expose probability or confidence columns"
                )
        if bounded_property_contract is not None:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            required_columns = {
                f"{target}_PRED_MEAN",
                f"{target}_PRED_MEAN_RAW",
                f"{target}_PRED_PHYSICAL_BOUNDED",
            }
            if not required_columns <= set(header):
                raise ValueError(
                    "bounded property CSV is missing raw or physical primary output"
                )
            bounds_audit = raw_well.get("physical_bounds_audit")
            if not isinstance(bounds_audit, Mapping):
                raise ValueError("bounded property well output is missing clipping audit")
        else:
            bounds_audit = None
        if fluid_output_contract is None:
            outputs[f"well_{index:03d}_prediction_csv"] = str(csv_path)
        fluid_artifacts: dict[str, str] = {}
        if fluid_output_contract is not None:
            for summary_key, output_suffix in (
                ("interval_csv", "fluid_intervals_csv"),
                ("decision_receipt_json", "fluid_decision_receipt_json"),
            ):
                artifact_path = Path(str(raw_well.get(summary_key, ""))).expanduser()
                if not artifact_path.is_absolute():
                    artifact_path = summary_path.parent / artifact_path
                artifact_path = artifact_path.resolve()
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"fluid {summary_key} not found: {artifact_path}"
                    )
                outputs[f"well_{index:03d}_{output_suffix}"] = str(artifact_path)
                fluid_artifacts[summary_key] = str(artifact_path)
            if fluid_artifacts.get("interval_csv") != str(csv_path):
                raise ValueError("fluid summary CSV and interval artifact differ")
            receipt_path = Path(fluid_artifacts["decision_receipt_json"])
            receipt = _json_mapping(
                receipt_path,
                description=f"fluid decision receipt {index}",
            )
            if (
                receipt.get("contract_version")
                != "wellfuse.fluid-sequence-decision-receipt.v1"
                or receipt.get("probability_persisted") is not False
                or dict(receipt.get("outputs") or {})
                != {"fluid_intervals_csv": csv_path.name}
            ):
                raise ValueError("fluid decision receipt is incompatible")
        well_outputs.append(
            {
                "well_id": str(raw_well.get("well_id", "")),
                "sample_count": int(
                    raw_well.get("sample_count", raw_well.get("samples", 0))
                ),
                **(
                    {"fluid_intervals_csv": str(csv_path)}
                    if fluid_output_contract is not None
                    else {"prediction_csv": str(csv_path)}
                ),
                **fluid_artifacts,
                **(
                    {
                        "interval_count": int(raw_well.get("interval_count", 0)),
                        "merged_spike_sample_count": int(
                            raw_well.get("merged_spike_sample_count", 0)
                        ),
                        "predicted_class_counts": dict(
                            raw_well.get("predicted_class_counts") or {}
                        ),
                        "raw_argmax_class_counts": dict(
                            raw_well.get("raw_argmax_class_counts") or {}
                        ),
                    }
                    if fluid_output_contract is not None
                    else {}
                ),
                **(
                    {"physical_bounds_audit": dict(bounds_audit)}
                    if bounds_audit is not None
                    else {}
                ),
            }
        )

    validation, checkpoint_selection = _fast_downstream_validation(
        wellfuse_root,
        model_id=model_id,
        dataset=dataset,
        features=features,
        summary=summary,
    )
    if bounded_property_contract is not None:
        aggregate_audit = summary.get("physical_bounds_audit")
        if not isinstance(aggregate_audit, Mapping):
            raise ValueError("bounded property summary is missing aggregate clipping audit")
        observed_out_of_bounds = sum(
            int(item["physical_bounds_audit"].get("out_of_bounds_count", 0))
            for item in well_outputs
        )
        observed_samples = sum(
            int(item["physical_bounds_audit"].get("sample_count", 0))
            for item in well_outputs
        )
        if (
            int(aggregate_audit.get("out_of_bounds_count", -1))
            != observed_out_of_bounds
            or int(aggregate_audit.get("sample_count", -1)) != observed_samples
            or aggregate_audit.get("raw_evidence_preserved") is not True
        ):
            raise ValueError("bounded property aggregate clipping audit differs from wells")
    well_count = int(summary.get("well_count", len(raw_wells)))
    result_path = output_directory / "result.json"
    outputs["result_json"] = str(result_path.resolve())
    result: dict[str, Any] = {
        "model_id": model_id,
        "model_name": public_runtime_model_name(model_id),
        "model_executed": True,
        "execution_status": "completed_fixed_last_dataset_inference",
        "scientific_status": "validated_within_dataset",
        "scope": "within_dataset",
        "dataset": dataset,
        "well_count": well_count,
        "device": str(summary.get("device", requested_device)),
        "input": {
            "source": str(request.source),
            "axes": ["MD"],
            **batch.provenance,
        },
        "well_outputs": well_outputs,
        "validation": validation,
        "checkpoint_selection": checkpoint_selection,
        "outputs": outputs,
        "warnings": ["验证范围仅限已登记数据集，不代表跨工区泛化。"],
    }
    if target is not None:
        result["target"] = target
        result["units"] = str(summary["units"])
        if bounded_property_contract is not None:
            result["primary_output"] = str(summary["primary_output"])
            result["diagnostic_output"] = str(summary["diagnostic_output"])
            result["output_contract"] = dict(bounded_property_contract)
            result["physical_bounds_audit"] = dict(
                summary.get("physical_bounds_audit") or {}
            )
            result["warnings"].append(
                f"{target}正式主曲线按[0,1]裁剪；原始回归值和越界计数完整保留供审计。"
            )
    else:
        result["features"] = features
        result["classes"] = [str(value) for value in summary.get("classes", [])]
        if fluid_output_contract is not None:
            result["primary_output"] = str(summary["primary_output"])
            result["diagnostic_output"] = str(summary["diagnostic_output"])
            result["output_contract"] = dict(fluid_output_contract)
            result["warnings"].append(
                "流体主成果仅为连续MD层段；五类概率只在进程内参与解码，不落盘或公开下载。"
            )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress:
        progress(98, f"已写出{well_count}口井的CSV、验证指标与fixed-last证据")
    return result


def _facies_deterministic_intervals(
    *,
    well_id: str,
    md_m: np.ndarray,
    facies_code: np.ndarray,
    class_names: Mapping[int, str],
) -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """Build contiguous business intervals from a deterministic facies sequence."""

    md = np.asarray(md_m, dtype=np.float64)
    code = np.asarray(facies_code, dtype=np.int16)
    if md.ndim != 1 or code.shape != md.shape or md.size < 2:
        raise ValueError(
            "Facies-1D deterministic MD/code must be matching vectors "
            "with at least two samples"
        )
    if not np.all(np.isfinite(md)) or np.any(np.diff(md) <= 0.0):
        raise ValueError("Facies-1D deterministic MD must be finite and strictly increasing")
    boundaries = np.flatnonzero(np.r_[True, code[1:] != code[:-1], True])
    edges = np.empty(md.size + 1, dtype=np.float64)
    edges[0] = md[0]
    edges[-1] = md[-1]
    edges[1:-1] = (md[:-1] + md[1:]) * 0.5
    interval_id = np.empty(md.shape, dtype=np.int32)
    rows: list[tuple[Any, ...]] = []
    for ordinal, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True), start=1
    ):
        interval_id[start:stop] = ordinal
        value = int(code[start])
        if value not in class_names:
            raise ValueError(f"Facies-1D deterministic sequence contains class {value}")
        top = float(edges[start])
        bottom = float(edges[stop])
        rows.append(
            (
                well_id,
                ordinal,
                top,
                bottom,
                bottom - top,
                value,
                class_names[value],
                int(stop - start),
            )
        )
    return interval_id, rows


def run_wellfuse_facies_1d_chengdu_fast_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the frozen Chengdu 46-well three-member Facies-1D ensemble."""

    del config, threshold, patch_size, overlap
    runtime_options = {**request.options, **dict(options or {})}
    batch = adapters.get(FAST_FACIES_1D_MODEL_ID).prepare(
        ModelInputRequest(source=request.source, options=runtime_options)
    )
    if batch.provenance.get("input_mode") == "raw_wells":
        return run_unified_raw_well_prediction(
            ModelInputRequest(source=request.source, options=runtime_options),
            model_id=FAST_FACIES_1D_MODEL_ID,
            adapters=adapters,
            config={},
            project_root=project_root,
            output_directory=output_directory,
            device_name=device_name,
            options=runtime_options,
            progress=progress,
        )
    dataset = str(batch.provenance["dataset"])
    well_ids = [str(value) for value in batch.provenance["well_ids"]]
    requested_device = str(device_name).strip().casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported Facies-1D device: {requested_device}")
    execution_device = "cpu" if requested_device == "cpu" else "cuda"

    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    script = wellfuse_root / "scripts" / "run_facies1d_fast.py"
    checkpoint = Path(
        str(
            runtime_options.get(
                "checkpoint_path",
                wellfuse_root
                / "artifacts/facies1d_fast/legacy_viterbi_v1/facies1d_fast_v1.pt",
            )
        )
    ).expanduser().resolve()
    contract_root = Path(
        str(
            runtime_options.get(
                "contract_root",
                wellfuse_root / "artifacts/p17/chengdu_facies/contract_v1",
            )
        )
    ).expanduser().resolve()
    metrics_path = checkpoint.parent / "metrics.json"
    config_path = checkpoint.parent / "config.json"
    for path, description in (
        (script, "Facies-1D script"),
        (checkpoint, "Facies-1D checkpoint"),
        (contract_root, "Facies-1D Chengdu contract"),
        (metrics_path, "Facies-1D metrics"),
        (config_path, "Facies-1D config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(runtime_options.get("batch_size", 16)))
    command = [
        str(python_executable),
        str(script),
        "--mode",
        "legacy-infer",
        "--checkpoint",
        str(checkpoint),
        "--contract-root",
        str(contract_root),
        "--output-root",
        str(output_directory),
        "--device",
        execution_device,
        "--batch-size",
        str(batch_size),
    ]
    for well_id in well_ids:
        command.extend(("--well", well_id))
    if progress:
        progress(10, f"已锁定成都{len(well_ids)}口井与三成员Viterbi部署checkpoint")
    completed = managed_run(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    runtime_log = output_directory / "facies1d_fast_runtime.log"
    runtime_log.write_text(
        "\n".join(
            (
                f"command={json.dumps(command, ensure_ascii=False)}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
            )
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"WellFuse Facies-1D inference failed (exit {completed.returncode}):\n{tail}"
        )
    summary = _stdout_json_mapping(
        completed.stdout, description="WellFuse Facies-1D inference"
    )
    if summary.get("schema_version") != "wellfuse.facies1d_fast.legacy_inference.v1":
        raise ValueError("unsupported Facies-1D inference summary")
    actual_device = str(summary.get("device", ""))
    if requested_device == "cuda" and not actual_device.casefold().startswith("cuda"):
        raise RuntimeError("CUDA was requested but Facies-1D executed on CPU")

    metrics_document = _json_mapping(metrics_path, description="Facies-1D metrics")
    config_document = _json_mapping(config_path, description="Facies-1D config")
    measured = metrics_document.get("metrics")
    if not isinstance(measured, Mapping):
        raise TypeError("Facies-1D metrics document is missing metrics")
    per_class = measured.get("pooled_decoded", {})
    per_class = per_class.get("per_class", []) if isinstance(per_class, Mapping) else []
    classes = [
        {"code": int(item["code"]), "name": str(item["name"])}
        for item in per_class
        if isinstance(item, Mapping)
    ]
    active_codes = [int(value) for value in config_document.get("active_codes", [])]
    if len(active_codes) != 6 or [item["code"] for item in classes] != active_codes:
        raise ValueError("Facies-1D class ontology differs between config and metrics")
    class_names = {int(item["code"]): str(item["name"]) for item in classes}

    output_root = Path(str(summary["output_root"])).expanduser().resolve()
    outputs: dict[str, str] = {
        "metrics_json": str(metrics_path),
        "config_json": str(config_path),
        "runtime_log": str(runtime_log),
    }
    well_outputs: list[dict[str, Any]] = []
    for index, well_id in enumerate(well_ids, start=1):
        npz_path = (output_root / f"{well_id}.npz").resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"Facies-1D well output not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as payload:
            required = {
                "md_m",
                "probability",
                "decoded_index",
                "decoded_code",
                "predictive_entropy_nats",
                "epistemic_mutual_information_nats",
            }
            if not required <= set(payload.files):
                raise ValueError(f"Facies-1D output keys are incomplete: {npz_path}")
            md_m = np.asarray(payload["md_m"], dtype=np.float32)
            probability = np.asarray(payload["probability"], dtype=np.float32)
            primary_index = probability.argmax(axis=0).astype(np.int16)
            primary_code = np.asarray(
                [active_codes[item] for item in primary_index], dtype=np.int16
            )
            decoded_index = np.asarray(payload["decoded_index"], dtype=np.int16)
            decoded_code = np.asarray(payload["decoded_code"], dtype=np.int16)
            if "raw_index" in payload.files:
                stored_raw_index = np.asarray(payload["raw_index"], dtype=np.int16)
                if not np.array_equal(stored_raw_index, primary_index):
                    raise ValueError(
                        f"Facies-1D stored raw_index is not probability argmax: {npz_path}"
                    )
            if "raw_code" in payload.files:
                stored_raw_code = np.asarray(payload["raw_code"], dtype=np.int16)
                if not np.array_equal(stored_raw_code, primary_code):
                    raise ValueError(
                        f"Facies-1D stored raw_code is not mapped probability argmax: {npz_path}"
                    )
            predictive_entropy = np.asarray(
                payload["predictive_entropy_nats"], dtype=np.float32
            )
            epistemic_mi = np.asarray(
                payload["epistemic_mutual_information_nats"], dtype=np.float32
            )
        if probability.shape != (len(active_codes), len(md_m)) or any(
            values.shape != md_m.shape
            for values in (decoded_index, decoded_code, predictive_entropy, epistemic_mi)
        ):
            raise ValueError(f"Facies-1D output shapes are inconsistent: {npz_path}")
        interval_id, interval_rows = _facies_deterministic_intervals(
            well_id=well_id,
            md_m=md_m,
            facies_code=decoded_code,
            class_names=class_names,
        )
        csv_path = output_directory / f"well_{index:03d}_facies.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "well_id",
                    "md_m",
                    "interval_id",
                    "facies_code",
                    "facies_name",
                ]
            )
            for sample_index in range(len(md_m)):
                writer.writerow(
                    [
                        well_id,
                        float(md_m[sample_index]),
                        int(interval_id[sample_index]),
                        int(decoded_code[sample_index]),
                        class_names[int(decoded_code[sample_index])],
                    ]
                )
        interval_csv_path = output_directory / f"well_{index:03d}_facies_intervals.csv"
        with interval_csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "well_id",
                    "interval_id",
                    "top_md_m",
                    "bottom_md_m",
                    "thickness_m",
                    "facies_code",
                    "facies_name",
                    "sample_count",
                ]
            )
            writer.writerows(interval_rows)
        outputs[f"well_{index:03d}_prediction_csv"] = str(csv_path.resolve())
        outputs[f"well_{index:03d}_facies_intervals_csv"] = str(
            interval_csv_path.resolve()
        )
        well_outputs.append(
            {
                "well_id": well_id,
                "sample_count": len(md_m),
                "prediction_csv": str(csv_path.resolve()),
                "facies_intervals_csv": str(interval_csv_path.resolve()),
                "interval_count": len(interval_rows),
                "predicted_class_counts": {
                    str(code): int((decoded_code == code).sum()) for code in active_codes
                },
            }
        )

    member_count = int(summary.get("member_count", 0))
    result_path = output_directory / "result.json"
    outputs["result_json"] = str(result_path.resolve())
    result: dict[str, Any] = {
        "schema_version": "well-seismic.facies-1d-chengdu-fast-runtime.v3",
        "model_id": FAST_FACIES_1D_MODEL_ID,
        "model_name": public_runtime_model_name(FAST_FACIES_1D_MODEL_ID),
        "model_executed": True,
        "checkpoint_forward_calls": member_count
        * max(1, (len(well_ids) + batch_size - 1) // batch_size),
        "execution_status": "completed_deterministic_viterbi_facies_intervals",
        "scientific_status": "validated_within_dataset",
        "scope": "within_dataset",
        "dataset": dataset,
        "well_count": len(well_ids),
        "device": actual_device,
        "requested_device": requested_device,
        "checkpoint": str(checkpoint),
        "input": {"source": str(request.source), "axes": ["MD"], **batch.provenance},
        "inference": {
            "active_codes": active_codes,
            "classes": classes,
            "primary_decision_rule": "sequence_viterbi",
            "sequence_decoder": str(
                config_document.get("decoder", "first_order_viterbi")
            ),
            "member_count": member_count,
            "transition_strength": float(summary.get("transition_strength", 0.2)),
            "minimum_segment_samples": 1,
            "minimum_segment_rule": "no_posthoc_merge_after_global_viterbi",
            "internal_probability_use": "argmax_and_viterbi_only_not_exported",
        },
        "primary_output": "facies_code",
        "diagnostic_output": None,
        "output_contract": {
            "contract_version": "wellfuse.facies-1d-output.v3",
            "primary_output": "facies_code",
            "primary_interval_output": "facies_intervals_csv",
            "secondary_point_output": "prediction_csv",
            "primary_decision_rule": "sequence_viterbi",
            "sequence_smoothing": {
                "method": "first_order_viterbi",
                "transition_strength": float(summary.get("transition_strength", 0.2)),
                "minimum_segment_samples": 1,
                "minimum_segment_rule": "no_posthoc_merge_after_global_viterbi",
            },
            "public_probability_outputs": [],
            "public_uncertainty_outputs": [],
            "internal_probability_use": "argmax_and_viterbi_only_not_exported",
        },
        "classes": classes,
        "well_outputs": well_outputs,
        "validation": {
            "metric_name": "whole_well_decoded_macro_f1",
            "metric_value": float(measured["whole_well_decoded_macro_f1"]),
            "evidence_dataset": "chengdu_46_well_oof",
            "metrics_json": str(metrics_path),
            "output_semantics": "deterministic_viterbi_sequence_primary",
        },
        "checkpoint_selection": {
            "selection": str(metrics_document.get("status", "")),
            "ensemble_size": member_count,
            "checkpoint": str(checkpoint),
        },
        "outputs": outputs,
        "warnings": [
            "仅验证成都登记数据集内的46口井，不代表未知井或跨工区泛化。",
            "当前冻结模型使用九线测井、轨迹与Align特征，不消费地震振幅分支。",
            "业务成果只公开Viterbi确定性相序列与连续层段；逐类概率、置信度和熵不进入下载资产。",
        ],
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress:
        progress(98, f"已写出{len(well_ids)}口井的确定性相序列与连续层段CSV")
    return result


def _validated_f3_primary_confidence(
    probability: np.ndarray,
    class_code: np.ndarray,
    confidence: np.ndarray,
    valid_trace_mask: np.ndarray,
) -> np.ndarray:
    """Validate and derive the formal confidence from the exported probabilities."""

    if probability.dtype != np.dtype(np.float32):
        raise ValueError("F3 Facies-3D formal probability must be stored as float32")
    if confidence.dtype != np.dtype(np.float32):
        raise ValueError("F3 Facies-3D formal confidence must be stored as float32")
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(confidence)):
        raise ValueError("F3 Facies-3D probability/confidence contains non-finite values")
    valid_voxels = np.broadcast_to(valid_trace_mask[..., None], class_code.shape)
    probability_argmax = probability.argmax(axis=0).astype(np.int16)
    if not np.array_equal(class_code[valid_voxels], probability_argmax[valid_voxels]):
        raise ValueError(
            "F3 Facies-3D class_code is not the float32 probability argmax"
        )
    if not np.all(class_code[~valid_voxels] == -1):
        raise ValueError("F3 Facies-3D invalid traces must use class_code -1")
    probability_confidence = probability.max(axis=0)
    if not np.array_equal(confidence, probability_confidence):
        raise ValueError(
            "F3 Facies-3D confidence is not from the formal float32 probability"
        )
    return probability_confidence


def run_wellfuse_facies_3d_f3_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the frozen dense-F3 checkpoint on a bounded SEG-Y ROI."""

    del threshold, overlap
    runtime_options = {**request.options, **dict(options or {})}
    requested_device = str(device_name).strip().casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported F3 Facies-3D device: {requested_device}")
    adapter_request = ModelInputRequest(
        source=request.source,
        crop_start=request.crop_start,
        crop_size=request.crop_size,
        options=runtime_options,
    )
    batch = adapters.get(F3_FACIES_3D_MODEL_ID).prepare(adapter_request)
    source_shape = tuple(int(value) for value in batch.provenance["source_shape_zyx"])
    if len(source_shape) != 3:
        raise ValueError("F3 Facies-3D source shape must be TWT,inline,xline")

    selection_receipt = batch.provenance.get("unlabeled_roi_selection")
    selection_resolved = (
        selection_receipt.get("resolved")
        if isinstance(selection_receipt, Mapping)
        else None
    )
    if selection_receipt is not None and not isinstance(selection_resolved, Mapping):
        raise ValueError("F3 Facies-3D unlabeled ROI selection receipt is malformed")
    if isinstance(selection_resolved, Mapping):
        t_start = int(selection_resolved["sample_start"])
        inline_start = int(selection_resolved["inline_start"])
        crossline_start = int(selection_resolved["crossline_start"])
        t_count = int(selection_resolved["sample_count"])
        inline_count = int(selection_resolved["inline_count"])
        crossline_count = int(selection_resolved["crossline_count"])
    else:
        starts = request.crop_start or (0, 0, 0)
        default_counts = tuple(
            min(size, limit)
            for size, limit in zip(
                source_shape, (256, 64, 64), strict=True
            )
        )
        requested_counts = request.crop_size or patch_size or default_counts
        t_start = int(runtime_options.get("t_start", starts[0]))
        inline_start = int(runtime_options.get("inline_start", starts[1]))
        crossline_start = int(runtime_options.get("crossline_start", starts[2]))
        t_count = int(runtime_options.get("t_count", requested_counts[0]))
        inline_count = int(runtime_options.get("inline_count", requested_counts[1]))
        crossline_count = int(
            runtime_options.get("crossline_count", requested_counts[2])
        )
    roi_start = (t_start, inline_start, crossline_start)
    roi_count = (t_count, inline_count, crossline_count)
    if any(value < 0 for value in roi_start) or any(value < 1 for value in roi_count):
        raise ValueError("F3 Facies-3D ROI starts must be non-negative and counts positive")
    if any(
        start + count > size
        for start, count, size in zip(roi_start, roi_count, source_shape, strict=True)
    ):
        raise ValueError(
            f"F3 Facies-3D ROI {roi_start}+{roi_count} exceeds source {source_shape}"
        )
    iline_byte, xline_byte = _resolve_facies_3d_header_bytes(
        runtime_options, batch.provenance
    )
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_source_identity = runtime_options.get("prediction_source_identity")
    prediction_source_identity = (
        dict(raw_source_identity)
        if isinstance(raw_source_identity, Mapping)
        else None
    )
    sealed_header_bytes, sealed_header_authority = (
        _sealed_f3_recommended_header_bytes(
            runtime_options,
            source=request.source,
            prediction_source_identity=prediction_source_identity,
        )
    )
    header_equivalence_path = (
        output_directory / "f3_header_pair_equivalence_receipt.json"
    )
    header_equivalence = _audit_f3_header_pair_equivalence(
        source=request.source,
        config=config,
        resolved_header_bytes=(iline_byte, xline_byte),
        sealed_header_bytes=sealed_header_bytes,
        sealed_authority=sealed_header_authority,
        prediction_source_identity=prediction_source_identity,
        expected_trace_count=int(batch.provenance.get("trace_count", 0)),
        receipt_path=header_equivalence_path,
    )

    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    script = wellfuse_root / "scripts" / "run_facies3d_fast_f3.py"
    checkpoint = Path(
        str(
            runtime_options.get(
                "checkpoint_path",
                project_root
                / "models"
                / "wellfuse"
                / "structural"
                / "facies_3d"
                / "f3_facies3d_fast_best.pt",
            )
        )
    ).expanduser().resolve()
    benchmark_path = Path(
        str(
            runtime_options.get(
                "benchmark_path",
                wellfuse_root
                / "artifacts/facies3d_fast/f3_dense_25d_unet_tune"
                / "metrics_test_once.json",
            )
        )
    ).expanduser().resolve()
    for path, description in (
        (script, "F3 Facies-3D script"),
        (checkpoint, "F3 Facies-3D checkpoint"),
        (benchmark_path, "F3 frozen test metrics"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    validation_identity = _f3_exact_identity_evidence(
        source=request.source,
        checkpoint=checkpoint,
        metrics_path=benchmark_path,
        runtime_options=runtime_options,
    )
    validated_within_dataset = bool(
        validation_identity["exact_match"]
        and header_equivalence["validation_eligible"]
    )

    raw_npz = output_directory / "facies3d_f3_roi.npz"
    batch_size = max(1, int(runtime_options.get("batch_size", 4)))
    command = [
        str(python_executable),
        str(script),
        "--predict-segy",
        str(request.source.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--prediction-output",
        str(raw_npz),
        "--device",
        requested_device,
        "--t-start",
        str(t_start),
        "--t-count",
        str(t_count),
        "--inline-start",
        str(inline_start),
        "--inline-count",
        str(inline_count),
        "--crossline-start",
        str(crossline_start),
        "--crossline-count",
        str(crossline_count),
        "--inline-byte",
        str(iline_byte),
        "--crossline-byte",
        str(xline_byte),
        "--batch-size",
        str(batch_size),
    ]
    if progress:
        mode_label = "单道" if inline_count == crossline_count == 1 else "ROI"
        progress(10, f"已锁定公开稠密基准权重，正在执行{mode_label}六类地震相推理")
    completed = managed_run(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    runtime_log = output_directory / "facies3d_f3_runtime.log"
    runtime_log.write_text(
        "\n".join(
            (
                f"command={json.dumps(command, ensure_ascii=False)}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout.rstrip(),
                "[stderr]",
                completed.stderr.rstrip(),
            )
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"WellFuse F3 Facies-3D inference failed (exit {completed.returncode}):\n{tail}"
        )
    summary = _stdout_json_mapping(
        completed.stdout, description="WellFuse F3 Facies-3D inference"
    )
    if summary.get("schema_version") != "wellfuse.facies3d_fast.f3_segy_roi_prediction.v1":
        raise ValueError("unsupported F3 Facies-3D inference summary")
    actual_device = str(summary.get("device", ""))
    if requested_device == "cuda" and not actual_device.casefold().startswith("cuda"):
        raise RuntimeError("CUDA was requested but F3 Facies-3D executed on CPU")
    if not raw_npz.is_file():
        raise FileNotFoundError("F3 Facies-3D did not create its ROI NPZ")

    with np.load(raw_npz, allow_pickle=False) as payload:
        required = {
            "probability",
            "class_code",
            "confidence",
            "uncertainty",
            "valid_trace_mask",
        }
        if not required <= set(payload.files):
            raise ValueError("F3 Facies-3D ROI NPZ is missing required fields")
        probability = np.asarray(payload["probability"])
        class_code = np.asarray(payload["class_code"], dtype=np.int16)
        confidence = np.asarray(payload["confidence"])
        uncertainty = np.asarray(payload["uncertainty"], dtype=np.float16)
        valid_trace_mask = np.asarray(payload["valid_trace_mask"], dtype=np.bool_)
    expected_class_shape = (inline_count, crossline_count, t_count)
    if class_code.shape != expected_class_shape:
        raise ValueError(
            f"F3 Facies-3D class shape {class_code.shape} != {expected_class_shape}"
        )
    if probability.shape != (6, *expected_class_shape) or any(
        array.shape != expected_class_shape for array in (confidence, uncertainty)
    ):
        raise ValueError("F3 Facies-3D probability/confidence/uncertainty shapes differ")
    if valid_trace_mask.shape != expected_class_shape[:2]:
        raise ValueError("F3 Facies-3D valid trace mask shape differs from the ROI")
    if not np.any(valid_trace_mask):
        raise ValueError(
            "F3 Facies-3D ROI contains no real SEG-Y trace; choose another "
            "inline/crossline start"
        )
    confidence = _validated_f3_primary_confidence(
        probability,
        class_code,
        confidence,
        valid_trace_mask,
    )
    if isinstance(selection_receipt, Mapping):
        selection_grid = selection_receipt.get("grid")
        selection_probe = selection_receipt.get("amplitude_probe")
        selection_thresholds = selection_receipt.get("thresholds")
        if not all(
            isinstance(value, Mapping)
            for value in (selection_grid, selection_probe, selection_thresholds)
        ):
            raise ValueError("F3 Facies-3D ROI selection evidence is incomplete")
        observed_valid_fraction = float(valid_trace_mask.mean())
        expected_valid_fraction = float(selection_grid["valid_trace_fraction"])
        if not np.isclose(
            observed_valid_fraction,
            expected_valid_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "F3 Facies-3D runtime valid grid differs from the unlabeled ROI receipt"
            )
        if observed_valid_fraction < float(
            selection_thresholds["minimum_valid_trace_fraction"]
        ):
            raise ValueError("F3 Facies-3D ROI valid trace support is below threshold")
        if float(selection_probe["active_trace_fraction"]) < float(
            selection_thresholds["minimum_active_trace_fraction"]
        ) or float(selection_probe["nonzero_sample_fraction"]) < float(
            selection_thresholds["minimum_nonzero_sample_fraction"]
        ) or float(selection_probe["finite_sample_fraction"]) < float(
            selection_thresholds["minimum_finite_sample_fraction"]
        ):
            raise ValueError("F3 Facies-3D ROI has insufficient nonzero amplitude support")

    flattened = {
        "class_code_npy": output_directory / "class_code.npy",
        "probability_npy": output_directory / "probability.npy",
        "confidence_npy": output_directory / "confidence.npy",
        "uncertainty_npy": output_directory / "uncertainty.npy",
        "valid_trace_mask_npy": output_directory / "valid_trace_mask.npy",
    }
    for path, array in (
        (flattened["class_code_npy"], class_code),
        (flattened["probability_npy"], probability),
        (flattened["confidence_npy"], confidence),
        (flattened["uncertainty_npy"], uncertainty),
        (flattened["valid_trace_mask_npy"], valid_trace_mask),
    ):
        np.save(path, array, allow_pickle=False)

    benchmark = _json_mapping(benchmark_path, description="F3 frozen test metrics")
    test_metrics = benchmark.get("test_metrics")
    combined = test_metrics.get("combined") if isinstance(test_metrics, Mapping) else None
    if not isinstance(combined, Mapping):
        raise TypeError("F3 frozen test metrics do not contain combined results")
    valid_voxels = np.broadcast_to(valid_trace_mask[..., None], class_code.shape)
    confidence_mean = float(
        np.asarray(confidence, dtype=np.float32)[valid_voxels].mean()
    )
    uncertainty_mean = float(
        np.asarray(uncertainty, dtype=np.float32)[valid_voxels].mean()
    )
    runtime_manifest_path = raw_npz.with_suffix(".json").resolve()
    if not runtime_manifest_path.is_file():
        raise FileNotFoundError("F3 Facies-3D sidecar manifest was not created")
    platform_manifest_path = (
        output_directory / "facies3d_f3_prediction_manifest.json"
    ).resolve()
    result_path = output_directory / "result.json"
    outputs = {
        **{name: str(path.resolve()) for name, path in flattened.items()},
        "raw_prediction_npz": str(raw_npz.resolve()),
        "manifest_json": str(platform_manifest_path),
        "runtime_manifest_json": str(runtime_manifest_path),
        "header_pair_equivalence_receipt_json": str(
            header_equivalence_path.resolve()
        ),
        "benchmark_metrics_json": str(benchmark_path.resolve()),
        "runtime_log": str(runtime_log.resolve()),
        "result_json": str(result_path.resolve()),
    }
    section_batches = max(1, (inline_count + batch_size - 1) // batch_size)
    inference_mode = "single_trace" if inline_count == crossline_count == 1 else "roi"
    result: dict[str, Any] = {
        "schema_version": "well-seismic.facies-3d-f3-fast-runtime.v2",
        "model_id": F3_FACIES_3D_MODEL_ID,
        "model_name": public_runtime_model_name(F3_FACIES_3D_MODEL_ID),
        "model_executed": True,
        "checkpoint_forward_calls": section_batches,
        "execution_status": "completed_f3_dense_checkpoint_transfer_inference",
        "scientific_status": (
            "validated_within_dataset"
            if validated_within_dataset
            else "experimental_transfer_candidate"
        ),
        "validated_scope": (
            "F3_dense_exact_source_checkpoint_metrics"
            if validated_within_dataset
            else "candidate_identity_or_header_validation_ineligible"
        ),
        "device": actual_device,
        "requested_device": requested_device,
        "checkpoint": str(checkpoint),
        "input": {
            "source": str(request.source),
            "axes": ["TWT", "INLINE", "XLINE"],
            "shape_zyx": list(source_shape),
            "crop_start_zyx": list(roi_start),
            "crop_size_zyx": list(roi_count),
            "unlabeled_roi_selection": selection_receipt,
            "resolved_header_bytes": {"iline": iline_byte, "xline": xline_byte},
            "prediction_source_identity": prediction_source_identity,
            "validation_identity": validation_identity,
            "header_pair_equivalence": header_equivalence,
            **batch.provenance,
        },
        "geometry": {
            "coordinate_reference": batch.provenance.get(
                "coordinate_reference", "source_seismic_grid_crs_unverified"
            ),
            "shape": list(roi_count),
            "crop_start_zyx": list(roi_start),
            "crop_size_zyx": list(roi_count),
            "unlabeled_roi_selection": selection_receipt,
            "resolved_header_bytes": {"iline": iline_byte, "xline": xline_byte},
            "header_pair_equivalence": header_equivalence,
        },
        "inference": {
            "mode": inference_mode,
            "single_trace_classification": inference_mode == "single_trace",
            "t_start": t_start,
            "t_count": t_count,
            "inline_start": inline_start,
            "inline_count": inline_count,
            "crossline_start": crossline_start,
            "crossline_count": crossline_count,
            "unlabeled_roi_selection": selection_receipt,
            "class_codes": [0, 1, 2, 3, 4, 5],
            "probability_axis_order": ["CLASS", "INLINE", "XLINE", "TWT"],
            "probability_dtype": "float32",
            "primary_decision_rule": "argmax_probability",
            "confidence_dtype": "float32",
            "confidence_semantics": "maximum_probability_for_argmax_class",
            "valid_trace_fraction": float(summary["valid_trace_fraction"]),
            "reference_checkpoint_validation_miou": float(
                summary["checkpoint_validation_miou"]
            ),
            "current_roi_labels_opened": False,
            "current_roi_metrics_computed": False,
        },
        "facies": {
            "shape_t_inline_xline": [t_count, inline_count, crossline_count],
            "class_codes": [0, 1, 2, 3, 4, 5],
            "confidence_mean": confidence_mean,
            "uncertainty_mean": uncertainty_mean,
            "valid_trace_fraction": float(summary["valid_trace_fraction"]),
            "reference_f3_frozen_test_macro_f1": float(combined["macro_f1"]),
            "reference_f3_frozen_test_miou": float(combined["miou"]),
            "current_roi_labels_opened": False,
            "current_roi_metrics_claimed": False,
        },
        "validation": {
            "status": (
                "validated_within_dataset"
                if validated_within_dataset
                else "candidate_identity_or_header_validation_ineligible"
            ),
            "identity_status": validation_identity["status"],
            "metric_name": "F3_frozen_test_mIoU",
            "metric_value": float(combined["miou"]),
            "macro_f1": float(combined["macro_f1"]),
            "evidence_dataset": "F3_dense_frozen_test_once",
            "metrics_json": str(benchmark_path.resolve()),
            "identity": validation_identity,
            "header_validation_eligible": header_equivalence[
                "validation_eligible"
            ],
            "validated_within_dataset_eligible": validated_within_dataset,
            "applies_to_exact_source_dataset": validated_within_dataset,
            "applies_to_current_roi": False,
            "current_roi_labels_opened": False,
            "current_roi_metrics_computed": False,
            "metric_scope": "frozen_reference_test_split_not_current_roi",
        },
        "outputs": outputs,
        "warnings": [
            (
                "源文件、checkpoint与metrics的SHA均精确匹配F3冻结合同；"
                "数据集内验证成立，但当前ROI未打开标签且不声明ROI指标。"
                if validated_within_dataset
                else (
                    "源文件、冻结证据身份或封存header验证资格未成立；"
                    "当前SEG-Y结果保持迁移候选。"
                )
            ),
            "类别0至5沿用训练数据本体，不自动映射为其他工区井侧沉积相代码。",
        ],
    }
    platform_manifest = {
        "schema_version": "well-seismic.facies-3d-f3-prediction-manifest.v1",
        "model_id": F3_FACIES_3D_MODEL_ID,
        "scientific_status": result["scientific_status"],
        "validated_scope": result["validated_scope"],
        "prediction_source_identity": prediction_source_identity,
        "validation_identity": validation_identity,
        "geometry": result["geometry"],
        "current_roi_evaluation": {
            "labels_opened": False,
            "metrics_computed": False,
            "reference_metrics_only": True,
        },
        "runtime_manifest_json": str(runtime_manifest_path),
        "header_pair_equivalence_receipt_json": str(
            header_equivalence_path.resolve()
        ),
        "outputs": {
            name: value
            for name, value in outputs.items()
            if name not in {"manifest_json", "result_json"}
        },
    }
    platform_manifest_path.write_text(
        json.dumps(platform_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress:
        progress(98, "六类离散相体、有效掩码与标准分类切片已写出")
    return result


def _sample_volume_statistics(path: Path, *, threshold: float) -> dict[str, float]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"WellFuse probability output must be 3D: {array.shape}")
    steps = tuple(max(1, int(np.ceil(size / 96))) for size in array.shape)
    sample = np.asarray(
        array[tuple(slice(None, None, step) for step in steps)], dtype=np.float32
    )
    if not np.all(np.isfinite(sample)):
        raise ValueError("WellFuse probability output contains non-finite values")
    return {
        "min": float(sample.min()),
        "max": float(sample.max()),
        "mean": float(sample.mean(dtype=np.float64)),
        "positive_fraction": float((sample >= threshold).mean()),
    }


def run_wellfuse_facies_3d_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Execute the real weakly supervised P17 Facies-3D checkpoint."""

    del threshold
    options = dict(options or {})
    requested_device = str(device_name).casefold()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported Facies-3D device: {requested_device}")
    if progress:
        progress(8, "正在核验Facies-3D规则SEG-Y与弱监督候选合同")
    batch = adapters.get("wellfuse_facies_3d_p17").prepare(request)
    iline_byte, xline_byte = _resolve_facies_3d_header_bytes(
        options, batch.provenance
    )
    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    checkpoint = Path(
        str(
            options.get(
                "checkpoint_path",
                wellfuse_root
                / "artifacts/p17/chengdu_facies/volume_3d_v1"
                / "seed_20260817/spatial_fold1/best.pt",
            )
        )
    ).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"P17 Facies-3D checkpoint not found: {checkpoint}")
    mode = str(options.get("facies_3d_mode", "sample")).casefold()
    if mode not in {"sample", "tiles", "full"}:
        raise ValueError("facies_3d_mode must be sample, tiles, or full")
    runtime_patch = tuple(int(value) for value in (patch_size or (96, 64, 64)))
    runtime_overlap = tuple(int(value) for value in (overlap or (48, 32, 32)))
    if len(runtime_patch) != 3 or len(runtime_overlap) != 3:
        raise ValueError("Facies-3D patch and overlap must be TWT,inline,xline triples")
    if any(value < 1 for value in runtime_patch) or any(
        value < 0 or value >= size
        for value, size in zip(runtime_overlap, runtime_patch, strict=True)
    ):
        raise ValueError("invalid Facies-3D patch/overlap")
    wrapper = project_root / "scripts" / "run_wellfuse_facies_3d.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"P17 Facies-3D wrapper not found: {wrapper}")
    output_directory.mkdir(parents=True, exist_ok=True)
    request_document = {
        "schema_version": "well-seismic.p17-facies-3d-runtime-request.v1",
        "checkpoint_path": str(checkpoint),
        "segy_path": str(request.source.resolve()),
        "output_directory": str(output_directory.resolve()),
        "mode": mode,
        "patch_shape": list(runtime_patch),
        "overlap": list(runtime_overlap),
        "iline_byte": iline_byte,
        "xline_byte": xline_byte,
        "requested_device": requested_device,
        "expected_trace_count": int(batch.provenance["trace_count"]),
        "expected_sample_count": int(batch.provenance["source_shape_zyx"][0]),
    }
    request_path = output_directory / "facies_3d_runtime_request.json"
    temporary = request_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(request_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, request_path)
    runtime_log = output_directory / "wellfuse_facies_3d_runtime.log"
    command = [
        str(python_executable),
        str(wrapper),
        "--wellfuse-root",
        str(wellfuse_root),
        "--request",
        str(request_path),
    ]
    if progress:
        progress(18, f"Facies-3D真实checkpoint已锁定，正在执行{mode}候选推理")
    with managed_popen(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        lines: list[str] = []
        result_document: dict[str, Any] | None = None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            lines.append(line)
            if line.startswith("WELLFUSE_FACIES3D_RESULT="):
                result_document = json.loads(line.split("=", 1)[1])
        return_code = process.wait()
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if return_code or result_document is None:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(f"P17 Facies-3D inference failed (exit {return_code}):\n{tail}")
    if result_document.get("model_executed") is not True:
        raise RuntimeError("P17 Facies-3D returned without executing its checkpoint")
    seismic = result_document.get("seismic")
    if not isinstance(seismic, Mapping):
        raise TypeError("P17 Facies-3D returned no seismic geometry")
    geometry_validation = _validate_facies_3d_geometry(seismic, batch.provenance)
    manifest_path = Path(str(result_document["manifest"])).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("P17 Facies-3D manifest was not created")
    fields = result_document.get("outputs", {}).get("fields", {})
    if not isinstance(fields, dict) or (not fields and mode != "tiles"):
        raise ValueError("P17 Facies-3D output fields are missing")
    output_paths = {
        f"{name}_artifact": str(Path(str(path)).resolve())
        for name, path in fields.items()
    }
    output_paths.update(
        {
            "metadata_json": str(manifest_path),
            "manifest_json": str(manifest_path),
            "tile_directory": str(
                Path(str(result_document["outputs"]["tile_root"])).resolve()
            ),
            "runtime_log": str(runtime_log.resolve()),
        }
    )
    tile_preview: dict[str, Any] | None = None
    if mode == "tiles":
        tile_root = Path(str(result_document["outputs"]["tile_root"])).resolve()
        tile_paths = sorted(tile_root.glob("*.npz"))
        if not tile_paths:
            raise FileNotFoundError("P17 Facies-3D tiles mode returned no tile arrays")
        with np.load(tile_paths[0], allow_pickle=False) as tile_payload:
            required_members = {"probability", "valid_mask", "origin_t_inline_xline"}
            if not required_members.issubset(tile_payload.files):
                raise ValueError("P17 Facies-3D tile lacks the declared preview members")
            probability = np.asarray(tile_payload["probability"], dtype=np.float32)
            valid_mask = np.asarray(tile_payload["valid_mask"], dtype=np.bool_)
            origin = np.asarray(tile_payload["origin_t_inline_xline"], dtype=np.int64)
        if probability.ndim != 4 or valid_mask.shape != probability.shape[1:]:
            raise ValueError("P17 Facies-3D tile probability/valid-mask shape drifted")
        active_codes = np.asarray(
            result_document["outputs"]["active_codes"], dtype=np.int16
        )
        if active_codes.ndim != 1 or active_codes.size != probability.shape[0]:
            raise ValueError("P17 Facies-3D tile class table does not match probability")
        time_index = probability.shape[1] // 2
        winners = active_codes[np.argmax(probability[:, time_index], axis=0)]
        winners = np.asarray(winners, dtype=np.int16)
        winners[~valid_mask[time_index]] = -1
        from .standard_export import _png_bytes

        preview_path = output_directory / "facies_3d_tiles_preview.png"
        preview_temporary = preview_path.with_suffix(f".png.tmp.{os.getpid()}")
        preview_temporary.write_bytes(_png_bytes(winners, categorical=True))
        os.replace(preview_temporary, preview_path)
        output_paths["tile_preview_png"] = str(preview_path.resolve())
        tile_preview = {
            "source_tile": tile_paths[0].name,
            "origin_t_inline_xline": origin.tolist(),
            "local_time_index": time_index,
            "scope": "representative_tile_only",
            "quantitative_use": False,
        }
    result = {
        "model_id": "wellfuse_facies_3d_p17",
        "model_name": public_runtime_model_name("wellfuse_facies_3d_p17"),
        "model_executed": True,
        "checkpoint_forward_calls": int(result_document["completed_tiles"]),
        "execution_status": result_document["execution_status"],
        "scientific_status": "experimental_unknown_survey_weak_candidate",
        "device": str(result_document.get("device", "cpu")),
        "requested_device": requested_device,
        "input": {
            "source": str(request.source),
            "axes": ["TWT", "INLINE", "XLINE"],
            "resolved_header_bytes": {
                "iline": iline_byte,
                "xline": xline_byte,
            },
            **batch.provenance,
        },
        "inference": {
            "mode": mode,
            "patch_size": list(runtime_patch),
            "overlap": list(runtime_overlap),
            "active_codes": result_document["outputs"]["active_codes"],
            "zero_support_codes": result_document["outputs"]["zero_support_codes"],
            "geometry_validation": geometry_validation,
            "tile_preview": tile_preview,
        },
        "outputs": output_paths,
        "warnings": [
            "这是稀疏井弱监督的三维地震相候选，不是未知工区dense 3D精度证明。",
            "默认sample模式只运行中心crop；全体积必须显式选择facies_3d_mode=full。",
            "现有Facies-3D检查点仅作为单折实验部署，不替代已验证的井侧Facies-1D。",
            *(
                [
                    f"请求设备为{requested_device}，当前WellFuse Facies-3D接口实际在CPU执行。"
                ]
                if requested_device != "cpu"
                and str(result_document.get("device", "cpu")) == "cpu"
                else []
            ),
        ],
    }
    if progress:
        progress(98, "Facies-3D类别/概率/不确定性/有效mask候选已写出")
    return result


def run_wellfuse_well_prediction(
    request: ModelInputRequest,
    *,
    model_id: str,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run frozen P17 Facies-1D or P18 well models on registered unknown wells."""

    del config, threshold, patch_size, overlap
    options = dict(options or {})
    if progress:
        progress(8, "正在核验LAS、完整轨迹与井震标定lineage")
    batch = adapters.get(model_id).prepare(request)
    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    _, training_envelope_path = resolve_training_envelope(wellfuse_root, model_id)
    wrapper = project_root / "scripts" / "run_wellfuse_well_models.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"WellFuse well-side wrapper not found: {wrapper}")
    requested_device = "auto" if device_name == "auto" else str(device_name)
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported WellFuse device: {requested_device}")
    output_directory.mkdir(parents=True, exist_ok=True)
    prepared_view_consumed = bool(
        batch.provenance.get("prepared_view_consumed", False)
    )
    well_las_option = options.get("well_las")
    if prepared_view_consumed and isinstance(well_las_option, Mapping):
        allowed_las = {
            str(Path(path).expanduser().resolve()).casefold()
            for path in batch.provenance["las_paths"]
        }
        well_las_option = {
            str(well_id): str(Path(str(path)).expanduser().resolve())
            for well_id, path in well_las_option.items()
            if str(Path(str(path)).expanduser().resolve()).casefold() in allowed_las
        }
    request_document: dict[str, Any] = {
        "schema_version": "well-seismic.unknown-well-runtime-request.v1",
        "model_id": model_id,
        "wellfuse_root": str(wellfuse_root),
        "output_directory": str(output_directory.resolve()),
        "device": requested_device,
        "las_paths": list(batch.provenance["las_paths"]),
        "registration_points_path": batch.provenance["registration_points_path"],
        "registration_manifest_path": batch.provenance.get(
            "registration_manifest_path"
        ),
        "registration_task_id": batch.provenance.get("registration_task_id"),
        "source_snapshot_id": batch.provenance.get("source_snapshot_id"),
        "well_las": well_las_option,
        "force_baseline": bool(options.get("force_baseline", False)),
        "applicability_unknown_policy": str(
            options.get("applicability_unknown_policy", "execute_unassessed")
        ),
    }
    if prepared_view_consumed:
        request_document["prepared_view_receipt"] = {
            key: batch.provenance.get(key)
            for key in (
                "prepared_view_id",
                "prepared_view_manifest_sha256",
                "prepared_view_sha256",
                "prepared_view_kind",
                "prepared_view_input_contract",
                "prepared_view_roles_used",
                "prepared_view_artifact_names_used",
                "prepared_view_artifacts_used",
            )
        }
    if training_envelope_path is not None:
        request_document["training_envelope_path"] = str(training_envelope_path)
    explicit_checkpoints = options.get("checkpoint_paths")
    if explicit_checkpoints is not None:
        request_document["checkpoint_paths"] = explicit_checkpoints
    request_path = output_directory / "well_runtime_request.json"
    temporary = request_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(request_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, request_path)
    runtime_log = output_directory / "wellfuse_well_runtime.log"
    command = [
        str(python_executable),
        str(wrapper),
        "--wellfuse-root",
        str(wellfuse_root),
        "--request",
        str(request_path),
    ]
    if progress:
        progress(18, "输入合同已通过，正在执行冻结checkpoint或合法基线")
    with managed_popen(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        lines: list[str] = []
        result_document: dict[str, Any] | None = None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            lines.append(line)
            if line.startswith("WELLFUSE_WELL_RESULT="):
                result_document = json.loads(line.split("=", 1)[1])
        return_code = process.wait()
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if return_code or result_document is None:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(
            f"WellFuse well-side inference failed (exit {return_code}):\n{tail}"
        )
    upstream_outputs = result_document.get("outputs")
    if not isinstance(upstream_outputs, dict):
        raise TypeError("WellFuse well-side outputs must be a mapping")
    manifest_path = Path(str(upstream_outputs.get("manifest", ""))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "WellFuse well-side inference did not create manifest.json"
        )
    manifest_document = _json_mapping(
        manifest_path, description="WellFuse well-side manifest"
    )
    runtime_provenance = result_document.get("provenance")
    if not isinstance(runtime_provenance, dict):
        raise RuntimeError(
            "WellFuse aligned well runtime did not return registration consumption provenance"
        )
    registration_consumption = runtime_provenance.get("registration_consumption")
    if not isinstance(registration_consumption, dict):
        raise RuntimeError(
            "WellFuse aligned well runtime did not attest Registration V3 feature consumption"
        )
    if model_id == "wellfuse_facies_1d_p17":
        expected_codes = [0, 1, 2, 3, 5, 6]
        result_contract = result_document.get("output_contract")
        manifest_contract = manifest_document.get("output_contract")
        if not isinstance(result_contract, Mapping) or not isinstance(
            manifest_contract, Mapping
        ):
            raise ValueError("aligned facies output is missing its deterministic contract")
        required_contract = {
            "primary_output": "predicted_code",
            "primary_decision_rule": "argmax_probability",
            "primary_interval_output": "facies_intervals_csv",
            "active_codes": expected_codes,
            "background_status": "no_background_class",
        }
        for key, expected in required_contract.items():
            if (
                result_contract.get(key) != expected
                or manifest_contract.get(key) != expected
            ):
                raise ValueError(f"aligned facies output contract drifted at {key}")
        if (
            result_contract.get("contract_version") != "wellfuse.facies-1d-output.v3"
            or manifest_contract.get("contract_version")
            != "wellfuse.facies-1d-output.v3"
            or list(result_contract.get("public_probability_outputs") or [])
            or list(manifest_contract.get("public_probability_outputs") or [])
            or list(result_contract.get("public_uncertainty_outputs") or [])
            or list(manifest_contract.get("public_uncertainty_outputs") or [])
        ):
            raise ValueError("aligned facies public output contract exposes probabilities")
        result_classes = result_document.get("classes")
        manifest_classes = manifest_document.get("classes")
        if (
            not isinstance(result_classes, list)
            or result_classes != manifest_classes
            or [item.get("code") for item in result_classes if isinstance(item, Mapping)]
            != expected_codes
            or any(
                not isinstance(item, Mapping)
                or str(item.get("role") or "") != "class"
                or not str(item.get("name") or "").strip()
                for item in result_classes
            )
        ):
            raise ValueError("aligned facies class ontology is incomplete or inconsistent")
        raw_well_outputs = result_document.get("well_outputs")
        if not isinstance(raw_well_outputs, list) or not raw_well_outputs:
            raise ValueError("aligned facies result contains no auditable well outputs")
        for raw_well in raw_well_outputs:
            if not isinstance(raw_well, Mapping):
                raise TypeError("aligned facies well output must be a mapping")
            npz_path = Path(str(raw_well.get("npz", ""))).expanduser().resolve()
            if not npz_path.is_file():
                raise FileNotFoundError(f"aligned facies NPZ not found: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as payload:
                required = {
                    "active_codes",
                    "probability",
                    "predicted_code",
                    "valid_mask",
                }
                if not required <= set(payload.files):
                    raise ValueError("aligned facies NPZ lacks argmax evidence")
                active_codes = np.asarray(payload["active_codes"], dtype=np.int64)
                probability = np.asarray(payload["probability"], dtype=np.float64)
                predicted_code = np.asarray(payload["predicted_code"], dtype=np.int64)
                valid_mask = np.asarray(payload["valid_mask"], dtype=np.bool_)
            if active_codes.tolist() != expected_codes:
                raise ValueError("aligned facies NPZ active codes drifted")
            if (
                probability.ndim != 2
                or probability.shape[1] != len(active_codes)
                or predicted_code.shape != (probability.shape[0],)
                or valid_mask.shape != (probability.shape[0],)
                or not np.isfinite(probability).all()
                or np.any(probability < 0.0)
            ):
                raise ValueError("aligned facies probability arrays are invalid")
            if np.any(valid_mask) and not np.allclose(
                probability[valid_mask].sum(axis=1), 1.0, atol=1e-5
            ):
                raise ValueError("aligned facies probabilities are not normalized")
            expected_primary = active_codes[probability.argmax(axis=1)]
            if not np.array_equal(predicted_code[valid_mask], expected_primary[valid_mask]):
                raise ValueError("aligned facies primary code is not probability argmax")
            intervals_path = Path(
                str(raw_well.get("facies_intervals_csv", ""))
            ).expanduser().resolve()
            if not intervals_path.is_file():
                raise FileNotFoundError(
                    f"aligned facies interval CSV not found: {intervals_path}"
                )
            with intervals_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                interval_rows = list(csv.DictReader(stream))
            expected_interval_columns = {
                "well_id",
                "interval_id",
                "top_md_m",
                "bottom_md_m",
                "thickness_m",
                "facies_code",
                "facies_name",
                "sample_count",
            }
            if not interval_rows or set(interval_rows[0]) != expected_interval_columns:
                raise ValueError("aligned facies interval CSV schema is invalid")
            if any(
                token in column.casefold()
                for column in interval_rows[0]
                for token in ("prob", "confidence", "entropy", "uncertainty")
            ):
                raise ValueError("aligned facies interval CSV exposes probability fields")
            if int(raw_well.get("interval_count", -1)) != len(interval_rows):
                raise ValueError("aligned facies interval count differs from CSV")
    property_target = {
        "wellfuse_por_p18": "POR",
        "wellfuse_sw_p18": "SW",
        "wellfuse_vsh_p18": "VSH",
    }.get(model_id)
    if property_target is not None:
        expected_contract = {
            "primary_output": "prediction_physical_bounded",
            "raw_evidence_output": "prediction_raw",
            "legacy_raw_output": "prediction",
            "physical_bounds": [0.0, 1.0],
            "transform": "clip",
        }
        result_contract = result_document.get("output_contract")
        manifest_contract = manifest_document.get("output_contract")
        if not isinstance(result_contract, Mapping) or not isinstance(
            manifest_contract, Mapping
        ):
            raise ValueError("bounded P18 output is missing the physical/raw contract")
        for key, expected in expected_contract.items():
            if result_contract.get(key) != expected or manifest_contract.get(key) != expected:
                raise ValueError(f"bounded P18 output contract drifted at {key}")
        raw_well_outputs = result_document.get("well_outputs")
        if not isinstance(raw_well_outputs, list) or not raw_well_outputs:
            raise ValueError("bounded P18 result contains no auditable well outputs")
        observed_samples = 0
        observed_out_of_bounds = 0
        for raw_well in raw_well_outputs:
            if not isinstance(raw_well, Mapping):
                raise TypeError("bounded P18 well output must be a mapping")
            bounds_audit = raw_well.get("physical_bounds_audit")
            if not isinstance(bounds_audit, Mapping):
                raise ValueError("bounded P18 well output lacks clipping audit")
            npz_path = Path(str(raw_well.get("npz", ""))).expanduser().resolve()
            if not npz_path.is_file():
                raise FileNotFoundError(f"bounded P18 NPZ not found: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as payload:
                required = {
                    "prediction",
                    "prediction_raw",
                    "prediction_physical_bounded",
                }
                if not required <= set(payload.files):
                    raise ValueError("bounded P18 NPZ lacks raw or physical primary evidence")
                legacy_raw = np.asarray(payload["prediction"], dtype=np.float64)
                explicit_raw = np.asarray(payload["prediction_raw"], dtype=np.float64)
                bounded = np.asarray(
                    payload["prediction_physical_bounded"], dtype=np.float64
                )
            if not np.array_equal(legacy_raw, explicit_raw, equal_nan=True):
                raise ValueError("bounded P18 legacy prediction no longer preserves raw semantics")
            if not np.allclose(bounded, np.clip(explicit_raw, 0.0, 1.0), equal_nan=True):
                raise ValueError("bounded P18 primary curve is not clip(raw, 0, 1)")
            observed_samples += int(bounds_audit.get("sample_count", 0))
            observed_out_of_bounds += int(bounds_audit.get("out_of_bounds_count", 0))
        aggregate_audit = result_document.get("physical_bounds_audit")
        if (
            not isinstance(aggregate_audit, Mapping)
            or int(aggregate_audit.get("sample_count", -1)) != observed_samples
            or int(aggregate_audit.get("out_of_bounds_count", -1))
            != observed_out_of_bounds
            or aggregate_audit.get("raw_evidence_preserved") is not True
        ):
            raise ValueError("bounded P18 aggregate clipping audit differs from wells")
    prepared_view_receipt = {
        key: batch.provenance.get(key)
        for key in (
            "prepared_view_consumed",
            "prepared_view_id",
            "prepared_view_manifest_path",
            "prepared_view_manifest_sha256",
            "prepared_view_sha256",
            "prepared_view_kind",
            "prepared_view_input_contract",
            "prepared_view_roles_used",
            "prepared_view_artifact_names_used",
            "prepared_view_artifacts_used",
        )
        if key in batch.provenance
    }
    las_directory = Path(str(upstream_outputs["las_directory"])).resolve()
    npz_directory = Path(str(upstream_outputs["npz_directory"])).resolve()
    las_archive = _write_deterministic_directory_zip(
        las_directory,
        output_directory / "well_las_outputs.zip",
    )
    npz_archive = _write_deterministic_directory_zip(
        npz_directory,
        output_directory / "well_npz_outputs.zip",
    )
    facies_intervals_directory: Path | None = None
    facies_intervals_archive: Path | None = None
    if model_id == "wellfuse_facies_1d_p17":
        facies_intervals_directory = Path(
            str(upstream_outputs["facies_intervals_directory"])
        ).resolve()
        facies_intervals_archive = _write_deterministic_directory_zip(
            facies_intervals_directory,
            output_directory / "facies_intervals.zip",
        )
    result = {
        **result_document,
        "model_id": model_id,
        "model_name": public_runtime_model_name(model_id),
        "device": requested_device,
        "input": {
            "source": str(request.source),
            "axes": ["MD"],
            **batch.provenance,
            "registration_consumed": True,
            "registration_consumption": registration_consumption,
            **prepared_view_receipt,
        },
        "provenance": {
            **runtime_provenance,
            "registration_consumed": True,
            "registration_consumption": registration_consumption,
            **prepared_view_receipt,
        },
        "outputs": {
            "metadata_json": str(manifest_path),
            "manifest_json": str(manifest_path),
            "las_directory": str(las_directory),
            "npz_directory": str(npz_directory),
            "las_archive_zip": str(las_archive),
            "npz_archive_zip": str(npz_archive),
            **(
                {
                    "facies_intervals_directory": str(facies_intervals_directory),
                    "facies_intervals_archive_zip": str(facies_intervals_archive),
                }
                if facies_intervals_directory is not None
                and facies_intervals_archive is not None
                else {}
            ),
            "runtime_log": str(runtime_log.resolve()),
            "result_json": str((output_directory / "result.json").resolve()),
        },
        "warnings": [
            *(str(item) for item in result_document.get("warnings", [])),
            "model_executed字段必须与manifest一起解读；合法基线降级不会伪装成checkpoint执行。",
            *(
                [
                    "井侧沉积相业务成果只公开确定性连续层段CSV；概率、置信度和熵仅保留在内部技术审计文件。"
                ]
                if model_id == "wellfuse_facies_1d_p17"
                else []
            ),
            *(
                [
                    f"{property_target}正式主曲线按[0,1]裁剪；原始回归值和越界计数完整保留供审计。"
                ]
                if property_target is not None
                else []
            ),
        ],
    }
    if progress:
        progress(98, "井侧确定性业务成果与内部技术审计已写出，正在登记manifest")
    return result


def run_geopath_tie_v1_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Execute the frozen GeoPathTie-V1 candidate over sealed Registration V3."""

    del threshold, patch_size, overlap
    runtime_options = {**request.options, **dict(options or {})}
    if progress:
        progress(8, "正在核验轨迹感知井震校正所需的封存SEG-Y、轨迹与标定先验")
    batch = adapters.get(GEOPATH_TIE_V1_MODEL_ID).prepare(
        ModelInputRequest(source=request.source, options=runtime_options)
    )
    checkpoint_value = runtime_options.get("checkpoint_path")
    checkpoint_path = (
        Path(str(checkpoint_value)).expanduser().resolve()
        if checkpoint_value
        else None
    )
    if progress:
        progress(20, "正在执行轨迹感知井震校正冻结权重与路径求解")
    result = run_geopath_tie_v1(
        seismic_path=request.source,
        registration_manifest_path=Path(
            str(batch.provenance["registration_manifest_path"])
        ),
        output_directory=output_directory,
        segy_profile_receipt=dict(batch.provenance["segy_profile_receipt"]),
        canonical_las_config=config,
        checkpoint_path=checkpoint_path,
        device=device_name,
        source_snapshot_id=batch.provenance.get("source_snapshot_id"),
        project_root=project_root,
        las_paths=[Path(str(path)) for path in batch.provenance["las_paths"]],
    )
    result["input"] = {
        **result.get("input", {}),
        **batch.provenance,
        "registration_consumed": True,
        "registration_consumption": result["provenance"]["registration_consumption"],
    }
    result["provenance"] = {
        **result.get("provenance", {}),
        "registration_consumed": True,
        "registration_consumption": result["provenance"]["registration_consumption"],
    }
    if progress:
        progress(98, "轨迹感知实验候选与消费回执已写出")
    return result


def run_wellfuse_horizon_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the real P17 three-seed local horizon-event ensemble.

    The runtime is intentionally delegated to the CUDA-enabled WellFuse
    environment.  The wrapper receives only SEG-Y, optional fixed named
    horizon controls, and Align lineage.  No target surface is accepted.
    """

    del threshold
    options = {**request.options, **dict(options or {})}
    if progress:
        progress(8, "正在核验未知工区四层位候选输入与标定来源链")
    adapter_request = ModelInputRequest(
        source=request.source,
        crop_start=request.crop_start,
        crop_size=request.crop_size,
        options=options,
    )
    batch = adapters.get("wellfuse_horizon_p17").prepare(adapter_request)
    recommended_options = batch.provenance.get("recommended_options")
    if isinstance(recommended_options, Mapping):
        for name in (
            "iline_byte",
            "xline_byte",
            "x_byte",
            "y_byte",
            "coordinate_scalar_byte",
        ):
            if recommended_options.get(name) is not None:
                options.setdefault(name, recommended_options[name])
    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    checkpoint_paths = [
        (
            project_root
            / "models"
            / "wellfuse"
            / "structural"
            / "horizon"
            / f"seed_{seed}"
            / "best.pt"
        ).resolve()
        for seed in (20260817, 20260829, 20260841)
    ]
    envelope, envelope_path = resolve_training_envelope(
        wellfuse_root, "wellfuse_horizon_p17"
    )
    observations: dict[str, Any] = {}
    if envelope is not None:
        profiled = batch.provenance.get("applicability_observations")
        observations = dict(profiled) if isinstance(profiled, dict) else {}
    checkpoint_binding: dict[str, Any]
    effective_envelope = envelope
    if envelope is None:
        checkpoint_binding = {"status": "unknown_no_training_envelope"}
    else:
        expected_digests = sorted(
            str(item["sha256"]) for item in envelope["checkpoint_refs"]
        )
        actual_digests: list[str] = []
        for path in checkpoint_paths:
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_digests.append(digest.hexdigest())
        actual_digests.sort()
        verified = (
            len(actual_digests) == len(checkpoint_paths)
            and actual_digests == expected_digests
        )
        checkpoint_binding = {
            "status": "verified" if verified else "unbound_checkpoint_set",
            "expected_sha256": expected_digests,
            "actual_sha256": actual_digests,
        }
        if not verified:
            effective_envelope = None
    applicability = evaluate_applicability(
        effective_envelope, observations, model_id="wellfuse_horizon_p17"
    )
    applicability["checkpoint_binding"] = checkpoint_binding
    if envelope is not None and effective_envelope is None:
        applicability.update(
            {
                "decision": "do_not_execute_model",
                "route": "abstain_unbound_checkpoint_set",
            }
        )
        applicability["issues"].append(
            "checkpoint_set_not_bound_to_training_envelope"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    applicability_manifest = write_applicability_manifest(
        output_directory / "applicability.json",
        applicability,
        envelope_path=envelope_path,
    )
    if applicability["decision"] == "do_not_execute_model":
        return {
            "model_id": "wellfuse_horizon_p17",
            "model_name": public_runtime_model_name("wellfuse_horizon_p17"),
            "model_executed": False,
            "execution_status": "abstained_out_of_training_envelope",
            "scientific_status": "not_evaluated_on_this_input",
            "applicability": applicability,
            "input": {
                "source": str(request.source),
                "axes": ["TWT", "INLINE", "XLINE"],
                **batch.provenance,
            },
            "outputs": {
                "metadata_json": str(applicability_manifest),
                "applicability_json": str(applicability_manifest),
            },
            "warnings": [
                "Horizon checkpoint execution was abstained because required label-free input "
                "features were outside its content-addressed training envelope."
            ],
        }
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P17 Horizon checkpoints not found: {missing}")
    wrapper = project_root / "scripts" / "run_wellfuse_horizon.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"P17 Horizon platform wrapper not found: {wrapper}")

    requested_device = "cuda" if device_name == "auto" else str(device_name)
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported WellFuse device: {requested_device}")
    spatial_tile = tuple(
        int(value) for value in (patch_size[-2:] if patch_size else (64, 64))
    )
    spatial_overlap = tuple(
        int(value) for value in (overlap[-2:] if overlap else (16, 16))
    )
    if len(spatial_tile) != 2 or len(spatial_overlap) != 2:
        raise ValueError("P17 Horizon patch/overlap must resolve to inline,xline pairs")
    if any(value < 1 for value in spatial_tile) or any(
        value < 0 or value >= size
        for value, size in zip(spatial_overlap, spatial_tile, strict=True)
    ):
        raise ValueError("invalid P17 Horizon spatial patch/overlap")

    source_shape = tuple(
        int(value) for value in batch.provenance.get("source_shape_zyx", ())
    )
    if len(source_shape) != 3 or any(value < 1 for value in source_shape):
        raise ValueError(
            "P17 Horizon adapter must report a positive TWT,inline,xline source shape"
        )
    crop_start = request.crop_start or (0, 0, 0)
    crop_size = request.crop_size
    selection_receipt = batch.provenance.get("unlabeled_roi_selection")
    selection_resolved = (
        selection_receipt.get("resolved")
        if isinstance(selection_receipt, Mapping)
        else None
    )
    if selection_receipt is not None and not isinstance(selection_resolved, Mapping):
        raise ValueError("P17 Horizon unlabeled ROI selection receipt is malformed")

    def _roi_value(name: str, fallback: int) -> int:
        value = options.get(name)
        return fallback if value is None else int(value)

    if isinstance(selection_resolved, Mapping):
        inline_start = int(selection_resolved["inline_start"])
        crossline_start = int(selection_resolved["crossline_start"])
        inline_count = int(selection_resolved["inline_count"])
        crossline_count = int(selection_resolved["crossline_count"])
    else:
        inline_start = _roi_value("inline_start", int(crop_start[1]))
        crossline_start = _roi_value("crossline_start", int(crop_start[2]))
        inline_count = _roi_value(
            "inline_count",
            int(crop_size[1])
            if crop_size is not None
            else source_shape[1] - inline_start,
        )
        crossline_count = _roi_value(
            "crossline_count",
            int(crop_size[2])
            if crop_size is not None
            else source_shape[2] - crossline_start,
        )
    roi_start = (inline_start, crossline_start)
    roi_count = (inline_count, crossline_count)
    if any(value < 0 for value in roi_start) or any(value < 1 for value in roi_count):
        raise ValueError(
            "P17 Horizon ROI starts must be non-negative and counts must be positive"
        )
    if inline_start + inline_count > source_shape[1] or (
        crossline_start + crossline_count > source_shape[2]
    ):
        raise ValueError(
            "P17 Horizon ROI exceeds source grid: "
            f"start={roi_start}, count={roi_count}, source={source_shape[1:]}"
        )
    spatial_roi_requested = bool(
        selection_receipt
        or
        request.crop_start
        or request.crop_size
        or any(
            options.get(name) is not None
            for name in (
                "inline_start",
                "inline_count",
                "crossline_start",
                "crossline_count",
            )
        )
    )

    request_document = {
        "schema_version": "well-seismic.p17-horizon-runtime-request.v2",
        "segy_path": str(request.source.resolve()),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "output_root": str(output_directory.resolve()),
        "survey_name": str(options.get("survey_name", request.source.stem)),
        "device": requested_device,
        "tile_size": list(spatial_tile),
        "overlap": list(spatial_overlap),
        "iline_byte": int(options.get("iline_byte", 189)),
        "xline_byte": int(options.get("xline_byte", 193)),
        "x_byte": int(options.get("x_byte", 181)),
        "y_byte": int(options.get("y_byte", 185)),
        "coordinate_scalar_byte": int(options.get("coordinate_scalar_byte", 71)),
        "inline_start": inline_start,
        "inline_count": inline_count,
        "crossline_start": crossline_start,
        "crossline_count": crossline_count,
        "spatial_roi_requested": spatial_roi_requested,
        "spatial_roi_selection_semantics": (
            "zero_based_indices_into_sorted_unique_grid_axes"
        ),
        "unlabeled_roi_selection": selection_receipt,
        "context_horizons_path": options.get("context_horizons_path"),
        "registration_points_path": options.get("registration_points_path"),
        "registration_manifest_path": options.get("registration_manifest_path"),
        "registration_points": options.get("registration_points"),
        "registration_source": options.get(
            "registration_source",
            "wellfuse_align_prediction"
            if options.get("registration_task_id")
            else None,
        ),
        "applicability_envelope_sha256": applicability.get("envelope_sha256"),
        "applicability_status": applicability["status"],
    }
    request_path = output_directory / "runtime_request.json"
    temporary = request_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(request_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, request_path)
    runtime_log = output_directory / "wellfuse_horizon_runtime.log"
    command = [
        str(python_executable),
        str(wrapper),
        "--wellfuse-root",
        str(wellfuse_root),
        "--request-json",
        str(request_path),
    ]
    if progress:
        progress(18, "三组四层位追踪权重已核验，正在执行未知工区候选推理")
    with managed_popen(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        lines: list[str] = []
        result_document: dict[str, Any] | None = None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            lines.append(line)
            if line.startswith("WELLFUSE_HORIZON_RESULT="):
                result_document = json.loads(line.split("=", 1)[1])
        return_code = process.wait()
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if return_code or result_document is None:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(
            f"P17 Horizon inference failed (exit {return_code}):\n{tail}"
        )
    if not result_document.get("checkpoint_forward_calls"):
        raise RuntimeError(
            "P17 Horizon returned without executing a real checkpoint forward"
        )

    runtime_roi = result_document.get("spatial_roi")
    if not isinstance(runtime_roi, Mapping):
        raise TypeError("P17 Horizon runtime result must contain a spatial ROI receipt")
    resolved_runtime_roi = runtime_roi.get("resolved")
    expected_runtime_roi = {
        "inline_start": inline_start,
        "inline_count": inline_count,
        "crossline_start": crossline_start,
        "crossline_count": crossline_count,
    }
    if not isinstance(resolved_runtime_roi, Mapping) or any(
        int(resolved_runtime_roi.get(name, -1)) != expected
        for name, expected in expected_runtime_roi.items()
    ):
        raise RuntimeError(
            "P17 Horizon runtime ignored or changed the requested spatial ROI"
        )

    upstream_outputs = result_document.get("outputs")
    if not isinstance(upstream_outputs, dict):
        raise TypeError("P17 Horizon result outputs must be a mapping")
    candidate_path = Path(str(upstream_outputs.get("candidate_npz", ""))).resolve()
    uncertainty_path = Path(str(upstream_outputs.get("uncertainty_npy", ""))).resolve()
    valid_mask_path = Path(str(upstream_outputs.get("valid_mask_npy", ""))).resolve()
    manifest_path = Path(str(result_document.get("manifest", ""))).resolve()
    required = (candidate_path, uncertainty_path, valid_mask_path, manifest_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "P17 Horizon did not create its candidate/uncertainty/mask/manifest"
        )
    with np.load(candidate_path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["prediction_twt_ms"])
        valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
        horizons = [str(value) for value in archive["horizon_names"]]
        ilines = np.asarray(archive["ilines"], dtype=np.int64)
        xlines = np.asarray(archive["xlines"], dtype=np.int64)
        x_coordinates = np.asarray(archive["x"], dtype=np.float64)
        y_coordinates = np.asarray(archive["y"], dtype=np.float64)
    if (
        prediction.ndim != 3
        or prediction.shape != valid_mask.shape
        or len(horizons) != 4
        or prediction.shape != (4, inline_count, crossline_count)
        or ilines.shape != (inline_count,)
        or xlines.shape != (crossline_count,)
        or x_coordinates.shape != (inline_count, crossline_count)
        or y_coordinates.shape != (inline_count, crossline_count)
    ):
        raise ValueError(
            f"P17 Horizon output contract mismatch: prediction={prediction.shape}, mask={valid_mask.shape}"
        )
    petrel = upstream_outputs.get("petrel_xyz")
    if not isinstance(petrel, dict) or set(petrel) != set(horizons):
        raise RuntimeError("P17 Horizon did not export all four Petrel XYZ surfaces")
    petrel_paths = {name: Path(str(petrel[name])).resolve() for name in horizons}
    if not all(path.is_file() for path in petrel_paths.values()):
        raise FileNotFoundError(
            "one or more P17 Horizon Petrel XYZ outputs are missing"
        )

    output_paths = {
        "horizon_candidates_npz": str(candidate_path),
        "uncertainty_sigma_npy": str(uncertainty_path),
        "valid_mask_npy": str(valid_mask_path),
        **{
            f"horizon_{name.casefold()}_surface_xyz": str(path)
            for name, path in petrel_paths.items()
        },
        "metadata_json": str(manifest_path),
        "runtime_request_json": str(request_path.resolve()),
        "runtime_log": str(runtime_log.resolve()),
    }
    result = {
        "model_id": "wellfuse_horizon_p17",
        "model_name": public_runtime_model_name("wellfuse_horizon_p17"),
        "model_executed": True,
        "checkpoint_forward_calls": int(result_document["checkpoint_forward_calls"]),
        "scientific_status": "experimental_candidate",
        "checkpoint": [str(path) for path in checkpoint_paths],
        "device": requested_device,
        "input": {
            "source": str(request.source),
            "axes": ["HORIZON", "INLINE", "XLINE"],
            "shape_zyx": [int(value) for value in prediction.shape],
            "source_shape_zyx": batch.provenance.get("source_shape_zyx"),
            "spatial_roi": dict(runtime_roi),
            **batch.provenance,
        },
        "geometry": {
            "shape": [int(value) for value in prediction.shape],
            "inline_range": [int(ilines[0]), int(ilines[-1])],
            "crossline_range": [int(xlines[0]), int(xlines[-1])],
            "inline_values": ilines.tolist(),
            "crossline_values": xlines.tolist(),
            "coordinate_shape_inline_xline": list(x_coordinates.shape),
            "spatial_roi": dict(runtime_roi),
            "unlabeled_roi_selection": selection_receipt,
            "sample_interval_ms": batch.provenance.get("sample_interval_ms"),
            "coordinate_reference": "source_seismic_grid_crs_unverified",
        },
        "inference": {
            "result_status": "experimental_runnable_candidate",
            "adaptation_mode": result_document.get("adaptation_mode"),
            "checkpoint_forward_calls": int(
                result_document["checkpoint_forward_calls"]
            ),
            "actual_checkpoint_loaded_and_forward_executed": True,
            "target_survey_weight_updates": 0,
            "valid_fraction": float(valid_mask.mean()),
            "applicability": applicability,
            "spatial_roi": dict(runtime_roi),
            "unlabeled_roi_selection": selection_receipt,
        },
        "outputs": {
            **output_paths,
            "applicability_json": str(applicability_manifest),
        },
        "warnings": [
            *list(result_document.get("warnings") or ()),
            *(
                [
                    "Training envelope was unavailable; applicability is explicitly unknown, "
                    "and legacy experimental execution was retained."
                ]
                if applicability["status"] == "unknown"
                else []
            ),
        ],
    }
    if progress:
        progress(98, "四层位实验候选已输出，正在登记NPZ、XYZ与不确定性图层")
    return result


def run_wellfuse_geobody_prediction(
    request: ModelInputRequest,
    *,
    model_id: str,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the frozen P17 Channel/Karst expert in its CUDA environment."""

    del config
    options = dict(options or {})
    task = {
        "wellfuse_channel_p17": "channel",
        "wellfuse_karst_p17": "karst",
    }.get(model_id)
    if task is None:
        raise ValueError(f"unsupported WellFuse geobody model: {model_id}")

    if progress:
        progress(8, f"正在核验{'河道' if task == 'channel' else '岩溶'}地质体识别的三维 SEG-Y 输入")
    batch = adapters.get(model_id).prepare(request)
    wellfuse_root, python_executable = _wellfuse_runtime_paths(project_root)
    checkpoint = (
        wellfuse_root
        / "artifacts"
        / "p17"
        / f"{task}_topology"
        / "rounds"
        / "seed_20260817"
        / "best.pt"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"WellFuse P17 {task} checkpoint not found: {checkpoint}"
        )

    runtime_patch = tuple(int(value) for value in (patch_size or (64, 96, 96)))
    runtime_overlap = tuple(int(value) for value in (overlap or (32, 48, 48)))
    runtime_threshold = 0.5 if threshold is None else float(threshold)
    minimum_voxels = int(options.get("minimum_voxels", 256))
    requested_device = "cuda" if device_name == "auto" else str(device_name)
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported WellFuse device: {requested_device}")
    output_directory.mkdir(parents=True, exist_ok=True)
    runtime_log = output_directory / "wellfuse_runtime.log"
    wrapper = project_root / "scripts" / "run_wellfuse_geobody.py"
    command = [
        str(python_executable),
        str(wrapper),
        "--wellfuse-root",
        str(wellfuse_root),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--segy",
        str(request.source),
        "--output",
        str(output_directory),
        "--survey-name",
        str(options.get("survey_name", request.source.stem)),
        "--patch",
        ",".join(map(str, runtime_patch)),
        "--overlap",
        ",".join(map(str, runtime_overlap)),
        "--threshold",
        str(runtime_threshold),
        "--minimum-voxels",
        str(minimum_voxels),
        "--device",
        requested_device,
        "--iline-byte",
        str(int(options.get("iline_byte", 189))),
        "--xline-byte",
        str(int(options.get("xline_byte", 193))),
    ]
    if progress:
        progress(18, f"{'河道' if task == 'channel' else '岩溶'}地质体识别已启动，正在执行确定性滑窗推理")
    with managed_popen(
        command,
        cwd=wellfuse_root,
        env=_normalized_subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        lines: list[str] = []
        result_document: dict[str, Any] | None = None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            lines.append(line)
            if line.startswith("WELLFUSE_GEOBODY_RESULT="):
                result_document = json.loads(line.split("=", 1)[1])
        return_code = process.wait()
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if return_code or result_document is None:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(
            f"WellFuse P17 {task} inference failed (exit {return_code}):\n{tail}"
        )

    output_fields = result_document.get("outputs", {}).get("fields", {})
    if not isinstance(output_fields, dict) or "probability" not in output_fields:
        raise RuntimeError("WellFuse P17 result is missing probability output")
    probability_path = Path(str(output_fields["probability"])).resolve()
    probability_array = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    shape = [int(value) for value in probability_array.shape]
    del probability_array
    statistics = _sample_volume_statistics(
        probability_path, threshold=runtime_threshold
    )
    output_paths = {
        f"{name}_npy": str(Path(str(path)).resolve())
        for name, path in output_fields.items()
    }
    output_paths.update(
        {
            "candidate_catalog_csv": str(
                Path(str(result_document["outputs"]["candidate_catalog"])).resolve()
            ),
            "metadata_json": str((output_directory / "manifest.json").resolve()),
            "runtime_log": str(runtime_log.resolve()),
        }
    )
    result = {
        "model_id": model_id,
        "model_name": public_runtime_model_name(model_id),
        "checkpoint": str(checkpoint),
        "device": requested_device,
        "input": {
            "source": str(request.source),
            "axes": ["Z", "INLINE", "CROSSLINE"],
            "shape_zyx": shape,
            "source_shape_zyx": shape,
            "crop_start_zyx": [0, 0, 0],
            "crop_size_zyx": shape,
            **batch.provenance,
        },
        "inference": {
            "patch_size": list(runtime_patch),
            "overlap": list(runtime_overlap),
            "threshold": runtime_threshold,
            "minimum_voxels": minimum_voxels,
            "deterministic_order": "TWT_then_inline_then_xline",
            "result_status": "provisional_real_survey_candidates",
        },
        "probability": {"shape_zyx": shape, **statistics},
        "outputs": output_paths,
        "warnings": [
            "真实工区输出为候选解释；合成数据指标不是当前工区精度。",
            "不确定性尚未在当前真实工区完成校准。",
        ],
    }
    if progress:
        progress(98, f"{'河道' if task == 'channel' else '岩溶'}地质体识别完成，正在登记三维图层")
    return result


def _run_faultseg_in_process(
    volume: np.ndarray,
    *,
    faultseg_root: Path,
    checkpoint: Path,
    runtime_spec: FaultSegInputSpec,
    device_name: str,
    progress: Progress | None,
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Retain the original in-process implementation for explicit CPU use."""

    if str(faultseg_root) not in sys.path:
        sys.path.insert(0, str(faultseg_root))
    try:
        importlib.import_module("torch")
        checkpoint_module = importlib.import_module("src.checkpoint")
        inference_module = importlib.import_module("src.inference")
    except ImportError as exc:
        raise RuntimeError(
            "FaultSeg inference requires PyTorch; install the project faultseg "
            "optional dependencies"
        ) from exc

    device = inference_module.choose_device(device_name)
    model, checkpoint_metadata = checkpoint_module.load_model(checkpoint, device)
    if progress:
        progress(55, f"FaultSeg 已加载到 {device}，开始滑窗推理")

    def patch_progress(index: int, total: int, origin: tuple[int, int, int]) -> None:
        if progress:
            progress(
                55 + int(35 * index / max(total, 1)),
                f"FaultSeg patch {index}/{total}，起点 {origin}",
            )

    probability = inference_module.predict_volume(
        model,
        volume,
        device,
        runtime_spec.patch_size,
        runtime_spec.overlap,
        progress=patch_progress,
        normalize_patches=True,
        weighted_blending=True,
        invalid_value=np.nan,
    )
    return probability, dict(checkpoint_metadata), str(device)


def _run_faultseg_subprocess(
    volume: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    project_root: Path,
    faultseg_root: Path,
    checkpoint: Path,
    output_directory: Path,
    runtime_spec: FaultSegInputSpec,
    device_name: str,
    progress: Progress | None,
    input_volume_path: Path | None = None,
    weighted_blending: bool = True,
    allow_patch_fallback: bool = False,
    cuda_patch_fallbacks: Sequence[int | Sequence[int]] = (),
    threshold_source: str | None = None,
    scope: str | None = None,
    training_context_shape: Sequence[int] | None = None,
    normalization_mode: str | None = None,
    allow_uncalibrated_normalization_experiment: bool = False,
    runtime_model_id: str = FAULTSEG_MODEL_ID,
    checkpoint_loader: str = "state_dict",
    output_activation: str = "sigmoid",
    patch_multiple: int = 8,
    training_context_authority: str = "checkpoint_metadata",
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Execute FaultSeg with the CUDA-capable WellFuse Python interpreter."""

    wrapper = project_root / "scripts" / "run_faultseg_subprocess.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"FaultSeg subprocess wrapper not found: {wrapper}")
    python_executable = _faultseg_subprocess_python(project_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    probability_path = output_directory / "faultseg_probability.npy"
    mask_path = output_directory / "faultseg_mask.npy"
    with tempfile.TemporaryDirectory(
        prefix="faultseg_subprocess_", dir=output_directory
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = (
            input_volume_path.expanduser().resolve()
            if input_volume_path is not None
            else temporary_root / "input_volume.npy"
        )
        valid_mask_path = temporary_root / "valid_mask.npy"
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        if input_volume_path is None:
            np.save(input_path, np.asarray(volume, dtype=np.float32))
        elif not input_path.is_file():
            raise FileNotFoundError(f"staged FaultSeg input not found: {input_path}")
        if valid_mask is not None:
            np.save(valid_mask_path, np.asarray(valid_mask, dtype=np.uint8))
        request_document = {
            "schema_version": "well-seismic.faultseg-subprocess-request.v1",
            "model_id": runtime_model_id,
            "faultseg_root": str(faultseg_root.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "input_volume_npy": str(input_path.resolve()),
            "valid_mask_npy": (
                str(valid_mask_path.resolve()) if valid_mask is not None else None
            ),
            "probability_npy": str(probability_path.resolve()),
            "mask_npy": str(mask_path.resolve()),
            "result_json": str(result_path.resolve()),
            "device": device_name,
            "patch_size": list(runtime_spec.patch_size),
            "overlap": list(runtime_spec.overlap),
            "threshold": runtime_spec.threshold,
            "weighted_blending": bool(weighted_blending),
            "allow_patch_fallback": bool(allow_patch_fallback),
            "cuda_patch_fallbacks": [
                list(value) if isinstance(value, Sequence) else int(value)
                for value in cuda_patch_fallbacks
            ],
            "threshold_source": threshold_source,
            "scope": scope,
            "normalization_mode": normalization_mode,
            "checkpoint_loader": checkpoint_loader,
            "output_activation": output_activation,
            "patch_multiple": int(patch_multiple),
            "training_context_authority": training_context_authority,
            "allow_uncalibrated_normalization_experiment": bool(
                allow_uncalibrated_normalization_experiment
            ),
            "training_context_shape": (
                list(training_context_shape)
                if training_context_shape is not None
                else None
            ),
        }
        request_path.write_text(
            json.dumps(request_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            str(python_executable),
            str(wrapper),
            "--request-json",
            str(request_path),
        ]
        runtime_label = (
            "FaultNet" if runtime_model_id == FAULTNET_MODEL_ID else "FaultSeg"
        )
        if progress:
            progress(48, f"正在通过独立 {python_executable.name} 启动 {runtime_label}")
        with managed_popen(
            command,
            cwd=project_root,
            env=_normalized_subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            assert process.stdout is not None
            output_tail: list[str] = []
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                output_tail.append(line)
                output_tail = output_tail[-40:]
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(event, Mapping)
                    or event.get("event") != "faultseg_progress"
                ):
                    continue
                index = int(event.get("index", 0))
                total = max(int(event.get("total", 0)), 1)
                if progress:
                    progress(
                        55 + int(35 * index / total),
                        f"{runtime_label} CUDA patch {index}/{total}，起点 {event.get('origin')}",
                    )
            return_code = process.wait()
        if return_code:
            details = "\n".join(output_tail) or "subprocess emitted no output"
            raise RuntimeError(
                f"FaultSeg subprocess failed with exit code {return_code}:\n{details}"
            )
        runtime_result = _json_mapping(
            result_path, description="FaultSeg subprocess result"
        )
        if (
            runtime_result.get("schema_version")
            != "well-seismic.faultseg-subprocess.v1"
        ):
            raise ValueError("FaultSeg subprocess returned an unsupported schema")
        actual_shape = tuple(int(value) for value in runtime_result["shape_zyx"])
        if actual_shape != volume.shape:
            raise ValueError(
                f"FaultSeg subprocess output shape {actual_shape} differs from {volume.shape}"
            )
        probability = np.load(probability_path, mmap_mode="r+", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if probability.shape != volume.shape or mask.shape != volume.shape:
            raise ValueError("FaultSeg subprocess arrays do not match the input volume")
        if isinstance(mask, np.memmap):
            mask._mmap.close()
        actual_device = str(runtime_result.get("device", ""))
        if device_name.startswith("cuda") and not actual_device.startswith("cuda"):
            raise RuntimeError(
                f"CUDA was requested but FaultSeg executed on {actual_device or 'unknown'}"
            )
        checkpoint_metadata = {
            "epoch": runtime_result.get("checkpoint_epoch"),
            "runtime": {
                "selected_patch_size": runtime_result.get("selected_patch_size"),
                "selected_overlap": runtime_result.get("selected_overlap"),
                "weighted_blending": runtime_result.get("weighted_blending"),
                "patch_attempts": runtime_result.get("patch_attempts") or [],
                "patch_count": runtime_result.get("patch_count"),
                "inference_context_degraded": bool(
                    runtime_result.get("inference_context_degraded")
                ),
                "degradation_reasons": runtime_result.get("degradation_reasons")
                or [],
                "normalization": runtime_result.get("normalization"),
                "normalization_statistics": runtime_result.get(
                    "normalization_statistics"
                ),
                "checkpoint_loader": runtime_result.get("checkpoint_loader"),
                "output_activation": runtime_result.get("output_activation"),
                "threshold_source": runtime_result.get("threshold_source"),
                "statistics": runtime_result.get("statistics") or {},
                "checkpoint_training_shape": runtime_result.get(
                    "checkpoint_training_shape"
                ),
                "configured_training_context_shape": runtime_result.get(
                    "configured_training_context_shape"
                ),
                "training_context_validated": bool(
                    runtime_result.get("training_context_validated")
                ),
                "training_context_policy": runtime_result.get(
                    "training_context_policy"
                ),
                "primary_patch_matches_checkpoint": bool(
                    runtime_result.get("primary_patch_matches_checkpoint")
                ),
                "debug_context_deviation": bool(
                    runtime_result.get("debug_context_deviation")
                ),
            },
        }
        return (
            probability,
            checkpoint_metadata,
            actual_device,
        )


def _run_faultseg_representative_grid_subprocess(
    *,
    project_root: Path,
    faultseg_root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    output_directory: Path,
    representative_grid: Mapping[str, Any],
    device_name: str,
    progress: Progress | None,
) -> tuple[dict[str, Any], str]:
    """Run the sealed 128-block request in one model-loading subprocess."""

    wrapper = project_root / "scripts" / "run_faultseg_subprocess.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"FaultSeg subprocess wrapper not found: {wrapper}")
    python_executable = _faultseg_subprocess_python(project_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    blocks_directory = output_directory / "representative_blocks"
    blocks_directory.mkdir(parents=True, exist_ok=True)
    raw_blocks = representative_grid.get("blocks")
    if (
        not isinstance(raw_blocks, list)
        or len(raw_blocks) != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
    ):
        raise ValueError(
            "FaultSeg representative adapter must provide exactly 128 blocks"
        )
    request_blocks: list[dict[str, Any]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            raise TypeError("FaultSeg representative adapter block must be an object")
        block = dict(raw_block)
        block_id = str(block.get("block_id") or "")
        block_directory = blocks_directory / block_id
        public_block = {
            key: value for key, value in block.items() if not str(key).startswith("_")
        }
        public_block.update(
            {
                "input_volume_npy": str(
                    Path(str(block["_staged_input_volume_npy"])).resolve()
                ),
                "valid_mask_npy": str(
                    Path(str(block["_staged_valid_mask_npy"])).resolve()
                ),
                "probability_npy": str(
                    (block_directory / "faultseg_probability.npy").resolve()
                ),
                "mask_npy": str(
                    (block_directory / "faultseg_mask.npy").resolve()
                ),
                "metadata_json": str(
                    (block_directory / "block_receipt.json").resolve()
                ),
            }
        )
        request_blocks.append(public_block)

    with tempfile.TemporaryDirectory(
        prefix="faultseg_grid_subprocess_", dir=output_directory
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        request_document = {
            "schema_version": "well-seismic.faultseg-subprocess-request.v1",
            "faultseg_root": str(faultseg_root.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "result_json": str(result_path.resolve()),
            "device": device_name,
            "patch_size": [128, 128, 128],
            "overlap": [0, 0, 0],
            "threshold": 0.518,
            "weighted_blending": False,
            "allow_patch_fallback": False,
            "cuda_patch_fallbacks": [],
            "threshold_source": "sealed_representative_grid_policy_v2",
            "scope": FAULTSEG_REPRESENTATIVE_SCOPE,
            "representative_grid_contract_version": (
                FAULTSEG_REPRESENTATIVE_GRID_CONTRACT_VERSION
            ),
            "normalization_mode": "per_patch_zscore",
            "allow_uncalibrated_normalization_experiment": False,
            "training_context_shape": [128, 128, 128],
            "source_shape_zyx": list(
                representative_grid.get("source_shape_zyx") or []
            ),
            "representative_grid_shape_zyx": list(
                _FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX
            ),
            "grid_order": "Z_then_INLINE_then_CROSSLINE",
            "representative_blocks": request_blocks,
        }
        request_path.write_text(
            json.dumps(request_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            str(python_executable),
            str(wrapper),
            "--request-json",
            str(request_path),
        ]
        if progress:
            progress(48, "正在单次加载 FaultSeg 并执行 128 个代表性体块")
        with managed_popen(
            command,
            cwd=project_root,
            env=_normalized_subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            assert process.stdout is not None
            output_tail: list[str] = []
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                output_tail.append(line)
                output_tail = output_tail[-40:]
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(event, Mapping)
                    or event.get("event") != "faultseg_progress"
                ):
                    continue
                block_index = int(event.get("block_index", 0))
                if progress:
                    progress(
                        50
                        + int(
                            40
                            * block_index
                            / _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
                        ),
                        "FaultSeg 代表块 "
                        f"{block_index}/{_FAULTSEG_REPRESENTATIVE_BLOCK_COUNT} · "
                        f"{event.get('block_id')}",
                    )
            return_code = process.wait()
        if return_code:
            details = "\n".join(output_tail) or "subprocess emitted no output"
            raise RuntimeError(
                f"FaultSeg representative subprocess failed with exit code "
                f"{return_code}:\n{details}"
            )
        runtime_result = _json_mapping(
            result_path, description="FaultSeg representative subprocess result"
        )
    if runtime_result.get("schema_version") != "well-seismic.faultseg-subprocess.v1":
        raise ValueError("FaultSeg representative subprocess returned unsupported schema")
    runtime_grid = runtime_result.get("representative_grid")
    if not isinstance(runtime_grid, Mapping):
        raise ValueError("FaultSeg representative subprocess omitted its grid receipt")
    runtime_blocks = runtime_grid.get("blocks")
    if (
        not isinstance(runtime_blocks, list)
        or len(runtime_blocks) != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
    ):
        raise ValueError(
            "FaultSeg representative subprocess did not return 128 blocks"
        )
    if (
        int(runtime_result.get("forward_calls") or 0)
        != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
    ) or any(
        int(block.get("forward_calls") or 0) != 1
        for block in runtime_blocks
        if isinstance(block, Mapping)
    ):
        raise ValueError("FaultSeg representative forward-call receipt is invalid")
    actual_device = str(runtime_result.get("device") or "")
    if device_name.startswith("cuda") and not actual_device.startswith("cuda"):
        raise RuntimeError(
            f"CUDA was requested but FaultSeg executed on {actual_device or 'unknown'}"
        )
    return dict(runtime_result), actual_device


_FAULTNET_CANDIDATE_MANIFEST_SCHEMA = "wellfuse.faultnet-finetune-result.v1"
_FAULTNET_LOCAL_CANDIDATE_TYPE = "local_accepted_candidate"
_FAULTNET_LOCAL_CANDIDATE_FIELDS = frozenset(
    {
        "type",
        "path",
        "sha256",
        "size_bytes",
        "candidate_manifest",
        "candidate_manifest_sha256",
        "candidate_manifest_size_bytes",
        "local_only",
        "redistribution_authorized",
        "release_registered",
    }
)
_FAULTNET_FORBIDDEN_PATH_OPTIONS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "faultnet_checkpoint",
        "faultnet_checkpoint_path",
        "faultnet_model_path",
        "faultnet_weights",
        "faultnet_weights_path",
        "model_path",
        "weights_path",
    }
)


def _validated_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return digest


def _validated_file_size(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _contained_local_faultnet_file(
    project_root: Path,
    value: Any,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty project-relative path")
    raw_path = Path(value.strip()).expanduser()
    if raw_path.is_absolute():
        raise ValueError(f"{field} must remain project-relative for portability")
    root = (project_root / "model_outputs" / "faultnet_finetune").resolve()
    path = (project_root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the local FaultNet output root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def _file_identity(path: Path) -> tuple[int, str]:
    stat = path.stat()
    return stat.st_size, _checkpoint_sha256(str(path), stat.st_mtime_ns, stat.st_size)


def _resolve_bundled_faultnet_variant(
    project_root: Path,
    *,
    variant: str,
    configured_path: str,
) -> tuple[Path, dict[str, Any]]:
    checkpoint = _project_path(project_root, configured_path)
    try:
        relative_checkpoint = checkpoint.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("bundled FaultNet checkpoint escapes the project root") from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FaultNet checkpoint not found: {checkpoint}")
    manifest_path = (project_root / "models" / "manifest.json").resolve()
    manifest = _read_json_object(
        manifest_path,
        description="FaultNet bundled weight manifest",
    )
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise ValueError("FaultNet bundled weight manifest has no models list")
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, Mapping)
            and str(item.get("path") or "").replace("\\", "/")
            == relative_checkpoint
        ),
        None,
    )
    if entry is None:
        raise RuntimeError(
            f"FaultNet variant {variant} is not registered in models/manifest.json"
        )
    expected_sha = _validated_sha256(
        entry.get("sha256"), field="models/manifest.json checkpoint sha256"
    )
    expected_size = _validated_file_size(
        entry.get("size_bytes", entry.get("size")),
        field="models/manifest.json checkpoint size",
    )
    observed_size, observed_sha = _file_identity(checkpoint)
    if observed_size != expected_size:
        raise RuntimeError(
            f"FaultNet variant {variant} size differs from models/manifest.json"
        )
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"FaultNet variant {variant} SHA-256 differs from models/manifest.json"
        )
    return checkpoint, {
        "schema_version": "well-seismic.faultnet-checkpoint-integrity.v1",
        "variant": variant,
        "variant_type": "bundled_release_checkpoint",
        "selection_source": "configs/faultseg.yaml#faultnet.checkpoint_variants",
        "integrity_source": "models/manifest.json",
        "integrity_status": "sha256_verified",
        "checkpoint": {
            "path": str(checkpoint),
            "relative_path": relative_checkpoint,
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "expected_size_bytes": expected_size,
            "observed_size_bytes": observed_size,
        },
        "candidate_manifest": None,
        "local_only": False,
        "redistribution_authorized": True,
        "release_registered": True,
        "explicit_ab_selection": variant != "0.7",
    }


def _resolve_local_faultnet_candidate(
    project_root: Path,
    *,
    variant: str,
    declaration: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    unknown_fields = set(declaration) - _FAULTNET_LOCAL_CANDIDATE_FIELDS
    missing_fields = _FAULTNET_LOCAL_CANDIDATE_FIELDS - set(declaration)
    if unknown_fields or missing_fields:
        details = []
        if missing_fields:
            details.append("missing=" + ",".join(sorted(missing_fields)))
        if unknown_fields:
            details.append("unknown=" + ",".join(sorted(unknown_fields)))
        raise ValueError(
            "FaultNet local candidate declaration differs from its sealed contract: "
            + "; ".join(details)
        )
    if declaration.get("type") != _FAULTNET_LOCAL_CANDIDATE_TYPE:
        raise ValueError("FaultNet local candidate has an unsupported declaration type")
    if declaration.get("local_only") is not True:
        raise ValueError("FaultNet local candidate must be explicitly local_only")
    if declaration.get("redistribution_authorized") is not False:
        raise ValueError("FaultNet local candidate cannot authorize redistribution")
    if declaration.get("release_registered") is not False:
        raise ValueError("FaultNet local candidate cannot masquerade as a release asset")

    checkpoint = _contained_local_faultnet_file(
        project_root, declaration.get("path"), field="FaultNet candidate checkpoint"
    )
    candidate_manifest = _contained_local_faultnet_file(
        project_root,
        declaration.get("candidate_manifest"),
        field="FaultNet candidate manifest",
    )
    expected_checkpoint_sha = _validated_sha256(
        declaration.get("sha256"), field="FaultNet candidate checkpoint sha256"
    )
    expected_checkpoint_size = _validated_file_size(
        declaration.get("size_bytes"), field="FaultNet candidate checkpoint size"
    )
    expected_manifest_sha = _validated_sha256(
        declaration.get("candidate_manifest_sha256"),
        field="FaultNet candidate manifest sha256",
    )
    expected_manifest_size = _validated_file_size(
        declaration.get("candidate_manifest_size_bytes"),
        field="FaultNet candidate manifest size",
    )
    observed_manifest_size, observed_manifest_sha = _file_identity(candidate_manifest)
    if observed_manifest_size != expected_manifest_size:
        raise RuntimeError("FaultNet candidate manifest size differs from its declaration")
    if observed_manifest_sha != expected_manifest_sha:
        raise RuntimeError("FaultNet candidate manifest SHA-256 differs from its declaration")

    manifest = _read_json_object(
        candidate_manifest,
        description="FaultNet candidate acceptance manifest",
    )
    if manifest.get("schema_version") != _FAULTNET_CANDIDATE_MANIFEST_SCHEMA:
        raise RuntimeError("FaultNet candidate manifest has an unsupported schema")
    if manifest.get("status") != "candidate_accepted":
        raise RuntimeError("FaultNet candidate manifest is not in candidate_accepted status")
    if manifest.get("candidate_accepted") is not True:
        raise RuntimeError("FaultNet candidate manifest did not pass its local A/B gate")
    if manifest.get("automatic_release_promotion") is not False:
        raise RuntimeError(
            "FaultNet local candidate must explicitly disable automatic release promotion"
        )
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
        raise RuntimeError("FaultNet candidate acceptance receipt did not pass")
    rights = manifest.get("rights")
    if not isinstance(rights, Mapping):
        raise RuntimeError("FaultNet candidate manifest omitted its local-use rights receipt")
    if rights.get("usage_scope") != "local_research_only":
        raise RuntimeError("FaultNet candidate is not sealed for local research use")
    if rights.get("redistribution_authorized") is not False:
        raise RuntimeError("FaultNet candidate manifest unexpectedly authorizes redistribution")
    checkpoint_receipt = manifest.get("checkpoint")
    if not isinstance(checkpoint_receipt, Mapping):
        raise RuntimeError("FaultNet candidate manifest omitted its checkpoint receipt")
    manifest_checkpoint_sha = _validated_sha256(
        checkpoint_receipt.get("candidate_sha256"),
        field="FaultNet candidate manifest checkpoint sha256",
    )
    if manifest_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(
            "FaultNet candidate declaration and manifest name different checkpoint SHA-256 values"
        )
    if checkpoint_receipt.get("torchscript_reload_verified") is not True:
        raise RuntimeError("FaultNet candidate did not pass TorchScript reload verification")
    manifest_candidate_path = str(checkpoint_receipt.get("candidate_path") or "").strip()
    if not manifest_candidate_path or Path(manifest_candidate_path).name != checkpoint.name:
        raise RuntimeError("FaultNet candidate manifest names a different checkpoint file")

    observed_checkpoint_size, observed_checkpoint_sha = _file_identity(checkpoint)
    if observed_checkpoint_size != expected_checkpoint_size:
        raise RuntimeError("FaultNet candidate checkpoint size differs from its declaration")
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("FaultNet candidate checkpoint SHA-256 differs from its declaration")
    return checkpoint, {
        "schema_version": "well-seismic.faultnet-checkpoint-integrity.v1",
        "variant": variant,
        "variant_type": _FAULTNET_LOCAL_CANDIDATE_TYPE,
        "selection_source": "configs/faultseg.yaml#faultnet.checkpoint_variants",
        "integrity_source": "pinned_candidate_manifest_and_checkpoint",
        "integrity_status": "sha256_verified",
        "checkpoint": {
            "path": str(checkpoint),
            "relative_path": checkpoint.relative_to(project_root.resolve()).as_posix(),
            "manifest_declared_path": manifest_candidate_path,
            "expected_sha256": expected_checkpoint_sha,
            "manifest_sha256": manifest_checkpoint_sha,
            "observed_sha256": observed_checkpoint_sha,
            "expected_size_bytes": expected_checkpoint_size,
            "observed_size_bytes": observed_checkpoint_size,
            "sha256_matches_manifest": True,
        },
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "relative_path": candidate_manifest.relative_to(
                project_root.resolve()
            ).as_posix(),
            "expected_sha256": expected_manifest_sha,
            "observed_sha256": observed_manifest_sha,
            "expected_size_bytes": expected_manifest_size,
            "observed_size_bytes": observed_manifest_size,
            "schema_version": manifest["schema_version"],
            "status": manifest["status"],
            "candidate_accepted": True,
            "acceptance_passed": True,
            "automatic_release_promotion": False,
        },
        "local_only": True,
        "usage_scope": rights["usage_scope"],
        "redistribution_authorized": False,
        "release_registered": False,
        "explicit_ab_selection": True,
    }


def _resolve_faultnet_checkpoint(
    project_root: Path,
    faultnet_config: Mapping[str, Any],
    runtime_options: Mapping[str, Any],
) -> tuple[str, Path, dict[str, Any]]:
    supplied_path_options = sorted(
        key for key in _FAULTNET_FORBIDDEN_PATH_OPTIONS if key in runtime_options
    )
    if supplied_path_options:
        raise ValueError(
            "FaultNet checkpoint paths cannot be supplied by a prediction request; "
            "select a configured named faultnet_variant instead"
        )
    variants = faultnet_config.get("checkpoint_variants")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("FaultNet checkpoint_variants must be configured")
    raw_variant = runtime_options.get("faultnet_variant")
    if raw_variant is None:
        raw_variant = runtime_options.get("faultnet_gamma")
    if raw_variant is None:
        raw_variant = faultnet_config.get("default_gamma") or "0.7"
    if not isinstance(raw_variant, str):
        raise ValueError("FaultNet variant must be a configured string name")
    variant = raw_variant.strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", variant) is None:
        raise ValueError("FaultNet variant must be a simple configured name, not a path")
    declaration = variants.get(variant)
    if declaration is None:
        raise ValueError(
            "unsupported FaultNet checkpoint variant; expected one of "
            + ", ".join(sorted(str(value) for value in variants))
        )
    if isinstance(declaration, str):
        checkpoint, receipt = _resolve_bundled_faultnet_variant(
            project_root,
            variant=variant,
            configured_path=declaration,
        )
    elif isinstance(declaration, Mapping):
        checkpoint, receipt = _resolve_local_faultnet_candidate(
            project_root,
            variant=variant,
            declaration=declaration,
        )
    else:
        raise ValueError("FaultNet checkpoint variant declaration is invalid")
    return variant, checkpoint, receipt


def run_faultseg_prediction(
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
    progress: Progress | None = None,
    runtime_model_id: str = FAULTSEG_MODEL_ID,
) -> dict[str, Any]:
    if runtime_model_id not in {FAULTSEG_MODEL_ID, FAULTNET_MODEL_ID}:
        raise ValueError(f"unsupported full-volume fault model: {runtime_model_id}")
    model_label = "FaultNet" if runtime_model_id == FAULTNET_MODEL_ID else "FaultSeg"
    config_key = "faultnet" if runtime_model_id == FAULTNET_MODEL_ID else "faultseg"
    runtime_options = dict(request.options)
    runtime_options.update(options or {})
    requested_device = str(device_name).strip().casefold()
    if requested_device not in {"auto", "cpu"} and not requested_device.startswith(
        "cuda"
    ):
        raise ValueError(f"unsupported FaultSeg device: {device_name}")
    faultseg_config = dict(config.get(config_key, {}))
    if not faultseg_config:
        raise ValueError(f"missing seismic fault model config section: {config_key}")
    spec = FaultSegInputSpec.from_config({"faultseg": faultseg_config})
    explicit_scope = str(runtime_options.get("faultseg_scope") or "").strip()
    scope_contract = faultseg_execution_scope_contract(runtime_model_id)
    scope = (
        explicit_scope.casefold()
        if explicit_scope
        else (
            "debug_crop"
            if request.crop_start or request.crop_size
            else str(scope_contract["default_value"])
        )
    )
    scope = {
        "legacy_crop": "debug_crop",
        "auto_roi": "automatic_valid_roi",
        "automatic_roi": "automatic_valid_roi",
    }.get(scope, scope)
    adaptive_small_volume_candidate = bool(
        scope == "debug_crop"
        and runtime_options.get("adaptive_small_volume_candidate") is True
    )
    if scope == FAULTSEG_REPRESENTATIVE_SCOPE:
        raise ValueError(
            "FaultSeg representative_grid_128 is historical read-only scope; "
            "new predictions must use center_block_1 or full_volume"
        )
    if scope not in {
        "full_volume",
        FAULTSEG_CENTER_BLOCK_SCOPE,
        "automatic_valid_roi",
        "debug_crop",
    }:
        raise ValueError(f"unsupported FaultSeg scope: {scope}")
    if scope == FAULTSEG_CENTER_BLOCK_SCOPE and runtime_model_id != FAULTSEG_MODEL_ID:
        raise ValueError(
            "FaultSeg center_block_1 is only supported by the "
            "registered FaultSeg checkpoint"
        )
    if scope in {"full_volume", FAULTSEG_CENTER_BLOCK_SCOPE} and (
        request.crop_start is not None or request.crop_size is not None
    ):
        raise ValueError(
            f"FaultSeg {scope} cannot include crop controls"
        )

    ignored_formal_parameters: list[str] = []
    if scope == "debug_crop":
        selected_threshold = spec.threshold if threshold is None else float(threshold)
        requested_patch = patch_size
        requested_overlap = overlap
        threshold_source = (
            str(faultseg_config.get("threshold_source", "configured"))
            if threshold is None
            else "explicit_debug_override"
        )
    else:
        selected_threshold = spec.threshold
        requested_patch = spec.patch_size
        requested_overlap = (
            (0, 0, 0)
            if scope == FAULTSEG_CENTER_BLOCK_SCOPE
            else spec.overlap
        )
        threshold_source = str(
            faultseg_config.get(
                "threshold_source", "checkpoint_validation_dice_calibrated"
            )
        )
        for name, value in (
            ("patch_size", patch_size),
            ("overlap", overlap),
            ("threshold", threshold),
        ):
            if value is not None:
                ignored_formal_parameters.append(name)

    training_context = tuple(
        int(value)
        for value in faultseg_config.get(
            "training_context_shape", (128, 128, 128)
        )
    )
    if len(training_context) != 3:
        raise ValueError("FaultSeg training_context_shape must contain three axes")
    if scope == "full_volume":
        configured_formal_scope = str(
            faultseg_config.get("formal_scope", "full_volume")
        ).strip().casefold()
        if configured_formal_scope != "full_volume":
            raise ValueError("FaultSeg formal_scope must remain full_volume")
        if spec.patch_size != (128, 128, 128):
            raise ValueError("FaultSeg full_volume patch_size must remain 128x128x128")
        if spec.overlap != (64, 64, 64):
            raise ValueError("FaultSeg full_volume overlap must remain 64x64x64")
        if training_context != (128, 128, 128):
            raise ValueError(
                "FaultSeg full_volume training_context_shape must remain 128x128x128"
            )
    elif scope == FAULTSEG_CENTER_BLOCK_SCOPE:
        if spec.patch_size != (128, 128, 128):
            raise ValueError(
                "FaultSeg center block patch_size must remain 128x128x128"
            )
        if training_context != (128, 128, 128):
            raise ValueError(
                "FaultSeg center block training_context_shape must remain "
                "128x128x128"
            )
        if float(spec.threshold) != 0.518:
            raise ValueError(
                "FaultSeg center block threshold must remain 0.518"
            )
    normalization_key = (
        "debug_normalization" if scope == "debug_crop" else "normalization"
    )
    normalization_mode = str(
        faultseg_config.get(normalization_key, "per_patch_zscore")
    ).strip().casefold()
    normalization_mode = {
        "patch_zscore": "per_patch_zscore",
        "patch_minmax": "per_patch_minmax",
        "shared_roi_zscore": "roi_shared_zscore",
        "formal_roi_zscore": "roi_shared_zscore",
    }.get(normalization_mode, normalization_mode)
    required_normalization = (
        "per_patch_minmax"
        if runtime_model_id == FAULTNET_MODEL_ID
        else "per_patch_zscore"
    )
    if normalization_mode != required_normalization:
        raise ValueError(
            f"{model_label} platform prediction requires {required_normalization}; "
            f"received {normalization_mode}"
        )
    weighted_blending = bool(faultseg_config.get("weighted_blending", True))
    if scope == "full_volume" and not weighted_blending:
        raise ValueError("FaultSeg full_volume requires weighted_blending=true")
    if scope == FAULTSEG_CENTER_BLOCK_SCOPE:
        weighted_blending = False
    output_directory.mkdir(parents=True, exist_ok=True)

    faultseg_root = project_root / "runtime" / "wellfuse" / "third_party" / "faultseg"
    checkpoint_variant: str | None = None
    checkpoint_integrity: dict[str, Any] | None = None
    if runtime_model_id == FAULTNET_MODEL_ID:
        checkpoint_variant, checkpoint, checkpoint_integrity = (
            _resolve_faultnet_checkpoint(
                project_root,
                faultseg_config,
                runtime_options,
            )
        )
    else:
        checkpoint_value = faultseg_config.get(
            "checkpoint",
            "models/wellfuse/structural/fault/faultseg-best.pt",
        )
        checkpoint = _project_path(project_root, str(checkpoint_value))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{model_label} checkpoint not found: {checkpoint}")
    checkpoint_loader = str(
        faultseg_config.get("checkpoint_loader", "state_dict")
    ).strip().casefold()
    output_activation = str(
        faultseg_config.get("output_activation", "sigmoid")
    ).strip().casefold()
    training_context_authority = str(
        faultseg_config.get(
            "training_context_authority", "checkpoint_metadata"
        )
    ).strip().casefold()
    probability_path = output_directory / "faultseg_probability.npy"
    mask_path = output_directory / "faultseg_mask.npy"
    if progress:
        scope_label = (
            "工区中心单个128³块"
            if scope == FAULTSEG_CENTER_BLOCK_SCOPE
            else "完整工区"
        )
        progress(10, f"正在检查 {model_label} {scope_label}范围与磁盘流式推理预算")
    with tempfile.TemporaryDirectory(
        prefix="faultseg_staging_", dir=output_directory
    ) as staging_value:
        prepared_options = {
            **runtime_options,
            "_faultseg_scope_resolved": scope,
            "_faultseg_staging_directory": staging_value,
            "_faultseg_runtime_patch_size": list(requested_patch or spec.patch_size),
        }
        prepared_request = ModelInputRequest(
            source=request.source,
            crop_start=request.crop_start,
            crop_size=request.crop_size,
            options=prepared_options,
        )
        if progress:
            progress(
                12,
                (
                    "正在从SEG-Y三轴中心封存单个128³块"
                    if scope == FAULTSEG_CENTER_BLOCK_SCOPE
                    else "正在将SEG-Y完整工区流式写入磁盘后备三维体"
                ),
            )
        batch = adapters.get(runtime_model_id).prepare(prepared_request)
        batch_provenance = dict(batch.provenance)
        resolved_scope = str(batch_provenance.get("scope") or scope)
        batch_provenance.setdefault("scope", resolved_scope)
        if scope in {"full_volume", FAULTSEG_CENTER_BLOCK_SCOPE} and resolved_scope != scope:
            raise ValueError(
                f"FaultSeg input adapter changed sealed scope {scope} to {resolved_scope}"
            )
        if scope == FAULTSEG_REPRESENTATIVE_SCOPE:
            if batch.array is not None or batch.valid_mask is not None:
                raise RuntimeError(
                    "FaultSeg representative adapter must return independent staged blocks"
                )
            if resolved_scope != FAULTSEG_REPRESENTATIVE_SCOPE:
                raise ValueError("FaultSeg representative adapter changed the sealed scope")
            adapter_grid = batch_provenance.get("representative_grid")
            if not isinstance(adapter_grid, Mapping):
                raise ValueError("FaultSeg representative adapter omitted its grid plan")
            if (
                adapter_grid.get("contract_version")
                != FAULTSEG_REPRESENTATIVE_GRID_CONTRACT_VERSION
                or adapter_grid.get("grid_shape_zyx")
                != list(_FAULTSEG_REPRESENTATIVE_GRID_SHAPE_ZYX)
                or int(adapter_grid.get("block_count") or 0)
                != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
            ):
                raise ValueError("FaultSeg representative adapter grid contract drifted")
            checkpoint_digest = _checkpoint_sha256(
                str(checkpoint),
                checkpoint.stat().st_mtime_ns,
                checkpoint.stat().st_size,
            )
            runtime_result, device = _run_faultseg_representative_grid_subprocess(
                project_root=project_root,
                faultseg_root=faultseg_root,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_digest,
                output_directory=output_directory,
                representative_grid=adapter_grid,
                device_name=requested_device,
                progress=progress,
            )
            runtime_grid = runtime_result.get("representative_grid")
            if not isinstance(runtime_grid, Mapping):
                raise ValueError("FaultSeg representative runtime omitted its grid receipt")
            runtime_blocks = runtime_grid.get("blocks")
            if (
                not isinstance(runtime_blocks, list)
                or len(runtime_blocks) != _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT
            ):
                raise ValueError(
                    "FaultSeg representative runtime did not return 128 blocks"
                )
            representative_grid = {
                key: value
                for key, value in adapter_grid.items()
                if key != "blocks"
            }
            representative_grid.update(
                {
                    key: value
                    for key, value in runtime_grid.items()
                    if key != "blocks"
                }
            )
            representative_grid.update(
                {
                    "scope": "representative_sampling",
                    "is_full_volume": False,
                    "source_segy": str(request.source.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_epoch": runtime_result.get("checkpoint_epoch"),
                    "threshold": 0.518,
                    "threshold_source": "sealed_representative_grid_policy_v2",
                    "normalization": "per_patch_zscore",
                    "inference_overlap_zyx": [0, 0, 0],
                    "weighted_blending": False,
                    "inter_block_stitching": False,
                    "forward_calls_total": _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT,
                    "blocks": runtime_blocks,
                }
            )
            sampled_with_overlap = int(
                representative_grid.get("sampled_voxel_count_with_overlap") or 0
            )
            union_voxels = int(
                representative_grid.get("representative_union_voxel_count") or 0
            )
            representative_grid["source_blocks_may_overlap"] = bool(
                sampled_with_overlap > union_voxels
            )
            blocks_directory = output_directory / "representative_blocks"
            receipt_path = output_directory / "faultseg_representative_grid_receipt.json"
            metadata_path = output_directory / "faultseg_result.json"
            receipt_path.write_text(
                json.dumps(representative_grid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            public_provenance = {
                key: value
                for key, value in batch_provenance.items()
                if key not in {"representative_grid", "materialized_volume_npy"}
            }
            result = {
                "schema_version": "well-seismic.faultseg-runtime.v1",
                "model_id": "faultseg_3d",
                "model_name": public_runtime_model_name("faultseg_3d"),
                "model_executed": True,
                "checkpoint_forward_calls": _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT,
                "scientific_status": "representative_sampling_candidate",
                "execution_scope": faultseg_execution_scope_metadata(
                    FAULTSEG_REPRESENTATIVE_SCOPE
                ),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_epoch": runtime_result.get("checkpoint_epoch"),
                "device": str(device),
                "input": {
                    "source_shape_zyx": list(
                        representative_grid.get("source_shape_zyx") or []
                    ),
                    "axes": ["Z", "INLINE", "CROSSLINE"],
                    **public_provenance,
                },
                "inference": {
                    "training_context_shape": [128, 128, 128],
                    "checkpoint_training_context_shape": runtime_result.get(
                        "checkpoint_training_shape"
                    ),
                    "training_context_validated": bool(
                        runtime_result.get("training_context_validated")
                    ),
                    "training_context_policy": runtime_result.get(
                        "training_context_policy"
                    ),
                    "primary_patch_matches_checkpoint": bool(
                        runtime_result.get("primary_patch_matches_checkpoint")
                    ),
                    "patch_size": [128, 128, 128],
                    "overlap": [0, 0, 0],
                    "weighted_blending": False,
                    "normalization": "per_patch_zscore",
                    "normalization_statistics": runtime_result.get(
                        "normalization_statistics"
                    ),
                    "threshold": 0.518,
                    "threshold_source": "sealed_representative_grid_policy_v2",
                    "scope": FAULTSEG_REPRESENTATIVE_SCOPE,
                    "faultseg_scope": FAULTSEG_REPRESENTATIVE_SCOPE,
                    "patch_attempts": runtime_result.get("patch_attempts") or [],
                    "inference_context_degraded": False,
                    "degradation_reasons": [],
                    "ignored_non_debug_parameters": [],
                    "forward_calls": _FAULTSEG_REPRESENTATIVE_BLOCK_COUNT,
                    "stitching": False,
                },
                "representative_grid": representative_grid,
                "outputs": {
                    "representative_blocks_directory": str(
                        blocks_directory.resolve()
                    ),
                    "representative_grid_receipt_json": str(receipt_path.resolve()),
                    "metadata_json": str(metadata_path.resolve()),
                },
                "warnings": [
                    "该结果是128个代表性体块的独立抽样，不是完整工区预测。",
                    "体块之间不拼接；不得将这些输出解释为full-volume。",
                ],
            }
            metadata_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if progress:
                progress(98, "FaultSeg 128 个代表性体块已完成并封存")
            return result
        if batch.array is None:
            raise RuntimeError(
                "FaultSeg input adapter did not materialize a seismic array"
            )
        batch_shape = tuple(int(value) for value in batch.array.shape)
        batch_dtype = str(batch.array.dtype)
        batch_axes = tuple(batch.axes)

        if scope == "debug_crop" and requested_patch is None:
            resolved_patch = tuple(
                max(
                    spec.patch_multiple,
                    (min(size, configured) // spec.patch_multiple)
                    * spec.patch_multiple,
                )
                for size, configured in zip(
                    batch_shape, spec.patch_size, strict=True
                )
            )
        else:
            resolved_patch = tuple(int(value) for value in (requested_patch or spec.patch_size))
        if any(size > available for size, available in zip(resolved_patch, batch_shape)):
            if scope != "debug_crop":
                raise ValueError(
                    "FaultSeg formal ROI is smaller than its training-context patch: "
                    f"ROI {batch_shape}, patch {resolved_patch}"
                )
            resolved_patch = tuple(
                (min(size, available) // spec.patch_multiple) * spec.patch_multiple
                for size, available in zip(resolved_patch, batch_shape)
            )
        resolved_overlap = tuple(
            int(value)
            for value in (
                requested_overlap
                or tuple(min(value, size // 2) for value, size in zip(spec.overlap, resolved_patch))
            )
        )
        runtime_spec = FaultSegInputSpec(
            resolved_patch,
            resolved_overlap,
            spec.patch_multiple,
            selected_threshold,
        ).validated()
        if adaptive_small_volume_candidate and (
            request.crop_start is not None
            or tuple(batch_shape) != tuple(runtime_spec.patch_size)
            or tuple(runtime_spec.overlap) != (0, 0, 0)
        ):
            raise ValueError(
                "FaultSeg small-volume adaptive candidate must use one deterministic "
                "center crop whose shape equals the inference patch and has zero overlap"
            )

        if progress:
            progress(40, f"地震范围已封存，正在加载 {model_label} 权重")
        staged_path_value = batch_provenance.get("materialized_volume_npy")
        staged_path = (
            Path(str(staged_path_value)).expanduser().resolve()
            if staged_path_value
            else None
        )
        use_in_process = (
            runtime_model_id == FAULTSEG_MODEL_ID
            and checkpoint_loader == "state_dict"
            and requested_device == "cpu"
            and resolved_scope == "debug_crop"
            and int(batch.array.nbytes)
            <= int(faultseg_config.get("api_process_in_memory_bytes", 256 * 1024**2))
        )
        try:
            if use_in_process:
                probability, checkpoint_metadata, device = _run_faultseg_in_process(
                    batch.array,
                    faultseg_root=faultseg_root,
                    checkpoint=checkpoint,
                    runtime_spec=runtime_spec,
                    device_name="cpu",
                    progress=progress,
                )
            else:
                probability, checkpoint_metadata, device = _run_faultseg_subprocess(
                    batch.array,
                    batch.valid_mask,
                    project_root=project_root,
                    faultseg_root=faultseg_root,
                    checkpoint=checkpoint,
                    output_directory=output_directory,
                    runtime_spec=runtime_spec,
                    device_name=requested_device,
                    progress=progress,
                    input_volume_path=staged_path,
                    weighted_blending=weighted_blending,
                    allow_patch_fallback=False,
                    cuda_patch_fallbacks=(),
                    threshold_source=threshold_source,
                    scope=resolved_scope,
                    training_context_shape=training_context,
                    normalization_mode=normalization_mode,
                    allow_uncalibrated_normalization_experiment=False,
                    runtime_model_id=runtime_model_id,
                    checkpoint_loader=checkpoint_loader,
                    output_activation=output_activation,
                    patch_multiple=spec.patch_multiple,
                    training_context_authority=training_context_authority,
                )
        finally:
            if isinstance(batch.array, np.memmap):
                batch.array.flush()
                batch.array._mmap.close()

    if probability.shape != batch_shape:
        raise ValueError(
            f"FaultSeg probability shape {probability.shape} differs from input "
            f"{batch_shape}"
        )
    subprocess_runtime = checkpoint_metadata.get("runtime") or {}
    if use_in_process:
        probability = np.asarray(probability, dtype=np.float32)
        mask = probability >= runtime_spec.threshold
        if batch.valid_mask is not None:
            probability[:, ~batch.valid_mask] = 0.0
            mask[:, ~batch.valid_mask] = False
        np.save(probability_path, probability.astype(np.float32, copy=False))
        np.save(mask_path, mask.astype(np.uint8, copy=False))
        statistics = {
            "min": float(probability.min()),
            "max": float(probability.max()),
            "mean": float(probability.mean()),
            "positive_fraction": float(mask.mean()),
        }
        actual_patch = list(runtime_spec.patch_size)
        actual_overlap = list(runtime_spec.overlap)
        patch_attempts = [
            {
                "patch_size": actual_patch,
                "overlap": actual_overlap,
                "status": "selected",
            }
        ]
        context_degraded = tuple(actual_patch) != training_context
        degradation_reasons = (
            ["explicit_debug_crop_context"] if context_degraded else []
        )
    else:
        statistics = dict(subprocess_runtime.get("statistics") or {})
        if not probability_path.is_file() or not mask_path.is_file():
            # Compatibility for third-party/test subprocess adapters that
            # still return an in-memory array under the v1 runner contract.
            probability = np.asarray(probability, dtype=np.float32)
            mask = probability >= runtime_spec.threshold
            if batch.valid_mask is not None:
                probability[:, ~batch.valid_mask] = 0.0
                mask[:, ~batch.valid_mask] = False
            np.save(probability_path, probability.astype(np.float32, copy=False))
            np.save(mask_path, mask.astype(np.uint8, copy=False))
            statistics = {
                "min": float(probability.min()),
                "max": float(probability.max()),
                "mean": float(probability.mean()),
                "positive_fraction": float(mask.mean()),
            }
        actual_patch = list(
            subprocess_runtime.get("selected_patch_size") or runtime_spec.patch_size
        )
        actual_overlap = list(
            subprocess_runtime.get("selected_overlap") or runtime_spec.overlap
        )
        patch_attempts = list(subprocess_runtime.get("patch_attempts") or [])
        context_degraded = bool(
            subprocess_runtime.get("inference_context_degraded")
            or tuple(actual_patch) != training_context
        )
        degradation_reasons = list(
            subprocess_runtime.get("degradation_reasons") or []
        )
        if tuple(actual_patch) != training_context and not degradation_reasons:
            degradation_reasons.append("runtime_patch_differs_from_training_context")

    if resolved_scope == FAULTSEG_CENTER_BLOCK_SCOPE and (
        batch_shape != (128, 128, 128)
        or tuple(actual_patch) != (128, 128, 128)
        or tuple(actual_overlap) != (0, 0, 0)
        or bool(context_degraded)
        or int(subprocess_runtime.get("patch_count") or 0) != 1
    ):
        raise ValueError(
            "FaultSeg center_block_1 runtime must preserve one exact 128x128x128 forward"
        )

    batch_provenance["materialized_volume_npy"] = None
    batch_provenance["staged_input_ephemeral"] = True
    batch_provenance["staged_input_removed_after_inference"] = True
    execution_scope = faultseg_execution_scope_metadata(
        resolved_scope,
        model_id=runtime_model_id,
    )
    if adaptive_small_volume_candidate:
        execution_scope.update(
            {
                "label": "小测区自适应单块",
                "description": "从真实地震体中心截取可容纳的最大模型倍数单块；不补造地震道",
                "estimated_duration_class": "adaptive_single_center_block",
                "adaptive_small_volume_candidate": True,
                "block_shape_zyx": list(batch_shape),
            }
        )
    result = {
        "schema_version": "well-seismic.faultseg-runtime.v1",
        "model_id": runtime_model_id,
        "model_name": public_runtime_model_name(runtime_model_id),
        "model_executed": True,
        "checkpoint_forward_calls": int(
            subprocess_runtime.get("patch_count") or 1
        ),
        "scientific_status": (
            "full_volume_fault_probability_candidate"
            if resolved_scope == "full_volume"
            else (
                "center_block_fault_probability_candidate"
                if resolved_scope == FAULTSEG_CENTER_BLOCK_SCOPE
                else (
                    "adaptive_small_volume_fault_candidate"
                    if adaptive_small_volume_candidate
                    else "historical_roi_or_debug_candidate"
                )
            )
        ),
        "execution_scope": execution_scope,
        "checkpoint": str(checkpoint),
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_sha256": _checkpoint_sha256(
            str(checkpoint), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size
        ),
        "checkpoint_integrity": checkpoint_integrity,
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "device": str(device),
        "input": {
            "shape_zyx": list(batch_shape),
            "dtype": batch_dtype,
            "axes": list(batch_axes),
            **batch_provenance,
        },
        "inference": {
            "training_context_shape": list(training_context),
            "checkpoint_training_context_shape": subprocess_runtime.get(
                "checkpoint_training_shape"
            ),
            "training_context_validated": (
                bool(subprocess_runtime.get("training_context_validated"))
                if not use_in_process
                else False
            ),
            "training_context_policy": (
                subprocess_runtime.get("training_context_policy")
                or "debug_in_process_configuration_only"
            ),
            "primary_patch_matches_checkpoint": (
                bool(subprocess_runtime.get("primary_patch_matches_checkpoint"))
                if not use_in_process
                else None
            ),
            "debug_context_deviation": (
                bool(subprocess_runtime.get("debug_context_deviation"))
                if not use_in_process
                else tuple(runtime_spec.patch_size) != training_context
            ),
            "patch_size": actual_patch,
            "overlap": actual_overlap,
            "weighted_blending": weighted_blending,
            "normalization": (
                subprocess_runtime.get("normalization")
                or required_normalization
            ),
            "checkpoint_loader": (
                subprocess_runtime.get("checkpoint_loader")
                or checkpoint_loader
            ),
            "output_activation": (
                subprocess_runtime.get("output_activation")
                or output_activation
            ),
            "normalization_statistics": subprocess_runtime.get(
                "normalization_statistics"
            ),
            "threshold": runtime_spec.threshold,
            "threshold_source": threshold_source,
            "scope": batch_provenance.get("scope"),
            "faultseg_scope": batch_provenance.get("scope"),
            "stitching": resolved_scope == "full_volume",
            "full_volume_reconstructed": resolved_scope == "full_volume",
            "forward_calls": int(subprocess_runtime.get("patch_count") or 1),
            "patch_attempts": patch_attempts,
            "inference_context_degraded": context_degraded,
            "degradation_reasons": degradation_reasons,
            "ignored_non_debug_parameters": ignored_formal_parameters,
            "adaptive_small_volume_candidate": adaptive_small_volume_candidate,
            "adaptive_patch_policy": (
                "center_crop_to_largest_available_patch_multiple_v1"
                if adaptive_small_volume_candidate
                else None
            ),
        },
        "probability": {
            "shape_zyx": list(batch_shape),
            **statistics,
        },
        "outputs": {
            "probability_npy": str(probability_path),
            "mask_npy": str(mask_path),
        },
    }
    if resolved_scope == FAULTSEG_CENTER_BLOCK_SCOPE:
        result["warnings"] = [
            "该结果仅覆盖工区三轴中心的单个128³块，不代表连续全区断层概率体。"
        ]
    elif adaptive_small_volume_candidate:
        result["warnings"] = [
            "该结果是小测区自适应实验候选，仅覆盖声明的中心单块。",
            "推理上下文小于checkpoint训练上下文，不得申报为全工区正式定量成果。",
        ]
    metadata_path = output_directory / "faultseg_result.json"
    result["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress:
        progress(98, f"{model_label} 推理完成，正在登记结果")
    if isinstance(probability, np.memmap):
        probability._mmap.close()
    return result


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@lru_cache(maxsize=16)
def _checkpoint_sha256(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_surface_checkpoint(
    checkpoint: Path,
    expected: dict[str, Any] | None = None,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SurfaceSeg checkpoint not found: {checkpoint}")
    stat = checkpoint.stat()
    if stat.st_size < 1024:
        header = checkpoint.read_bytes()[:64]
        if header.startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(
                f"SurfaceSeg checkpoint is still a Git LFS pointer: {checkpoint}"
            )
        raise RuntimeError(f"SurfaceSeg checkpoint is unexpectedly small: {checkpoint}")
    expected = expected or {}
    expected_size = expected.get("size")
    if expected_size is not None and stat.st_size != int(expected_size):
        raise RuntimeError(
            f"SurfaceSeg checkpoint size mismatch: {checkpoint} "
            f"({stat.st_size} != {int(expected_size)})"
        )
    expected_hash = str(expected.get("sha256", "")).strip().lower()
    if expected_hash:
        actual_hash = _checkpoint_sha256(
            str(checkpoint.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SurfaceSeg checkpoint SHA256 mismatch: {checkpoint} "
                f"({actual_hash} != {expected_hash})"
            )


def _load_surface_seg_runtime(surface_root: Path) -> Any:
    """Load the bundled package under a private name to avoid module collisions."""
    module_name = "_well_seismic_surface_seg_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    package_root = surface_root / "minimal_sgy"
    init_path = package_root / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"SurfaceSeg Python package not found: {init_path}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SurfaceSeg package: {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit):
        sys.modules.pop(module_name, None)
        raise
    return module


def _surface_runtime_options(
    surface_config: dict[str, Any],
    options: dict[str, Any],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        return options[name] if name in options else surface_config.get(name, default)

    mask_threshold = (
        float(threshold)
        if threshold is not None
        else float(value("mask_threshold", 0.5))
    )
    inline_count = value("inline_count", None)
    inline_byte = value("inline_byte", None)
    crossline_byte = value("crossline_byte", None)
    max_inlines = value("max_inlines", None)
    display_gate_raw = surface_config.get("horizon_display_gate", {})
    display_gate = (
        dict(display_gate_raw) if isinstance(display_gate_raw, Mapping) else {}
    )
    return {
        "device": str(value("device", "auto")),
        "segformer_batch_size": int(value("segformer_batch_size", 2)),
        "mask2former_batch_size": int(value("mask2former_batch_size", 1)),
        "amplitude_mode": str(value("amplitude_mode", "auto")),
        "query_threshold": float(value("query_threshold", 0.35)),
        "mask_threshold": mask_threshold,
        "confidence_threshold": float(
            surface_config.get("confidence_threshold", 0.35)
        ),
        # These parameters define the sealed scale/reconciliation policy.  They
        # intentionally come from service configuration rather than a public
        # prediction request so two identical inputs cannot silently use
        # different global-label semantics.
        "preprocessing_policy": str(
            surface_config.get(
                "preprocessing_policy",
                SURFACE_CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
            )
        ),
        "window_overlap": float(surface_config.get("window_overlap", 0.0)),
        "max_tiles_per_inline": int(
            surface_config.get("max_tiles_per_inline", 1)
        ),
        "minimum_uniform_scale": float(
            surface_config.get("minimum_uniform_scale", 0.35)
        ),
        "association_max_gap": int(
            surface_config.get("association_max_gap", 2)
        ),
        "horizon_display_gate_schema": str(
            display_gate.get("schema_version", HORIZON_DISPLAY_GATE_SCHEMA)
        ),
        "minimum_horizon_finite_trace_fraction": float(
            display_gate.get(
                "minimum_finite_trace_fraction",
                DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
            )
        ),
        "minimum_horizon_largest_component_fraction": float(
            display_gate.get(
                "minimum_largest_component_fraction",
                DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION,
            )
        ),
        "horizon_surface_connectivity": int(
            display_gate.get("surface_connectivity", 4)
        ),
        "num_visualizations": int(value("num_visualizations", 5)),
        "inline_count": None if inline_count is None else int(inline_count),
        "inline_byte": None if inline_byte is None else int(inline_byte),
        "crossline_byte": None if crossline_byte is None else int(crossline_byte),
        "max_inlines": None if max_inlines is None else int(max_inlines),
        "write_mask_sgy": bool(value("write_mask_sgy", True)),
    }


def _surface_cli_command(
    python_executable: Path,
    request: ModelInputRequest,
    output_directory: Path,
    models_dir: Path,
    runtime_options: dict[str, Any],
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "minimal_sgy",
        "--input",
        str(request.source),
        "--output-dir",
        str(output_directory),
        "--models-dir",
        str(models_dir),
        "--device",
        str(runtime_options["device"]),
        "--segformer-batch-size",
        str(runtime_options["segformer_batch_size"]),
        "--mask2former-batch-size",
        str(runtime_options["mask2former_batch_size"]),
        "--amplitude-mode",
        str(runtime_options["amplitude_mode"]),
        "--query-threshold",
        str(runtime_options["query_threshold"]),
        "--mask-threshold",
        str(runtime_options["mask_threshold"]),
        "--confidence-threshold",
        str(runtime_options["confidence_threshold"]),
        "--preprocessing-policy",
        str(runtime_options["preprocessing_policy"]),
        "--window-overlap",
        str(runtime_options["window_overlap"]),
        "--max-tiles-per-inline",
        str(runtime_options["max_tiles_per_inline"]),
        "--minimum-uniform-scale",
        str(runtime_options["minimum_uniform_scale"]),
        "--association-max-gap",
        str(runtime_options["association_max_gap"]),
        "--minimum-horizon-finite-trace-fraction",
        str(runtime_options["minimum_horizon_finite_trace_fraction"]),
        "--minimum-horizon-largest-component-fraction",
        str(runtime_options["minimum_horizon_largest_component_fraction"]),
        "--num-visualizations",
        str(runtime_options["num_visualizations"]),
    ]
    if runtime_options["inline_count"] is not None:
        command.extend(["--inline-count", str(runtime_options["inline_count"])])
    if runtime_options.get("inline_byte") is not None:
        command.extend(["--inline-byte", str(runtime_options["inline_byte"])])
    if runtime_options.get("crossline_byte") is not None:
        command.extend(["--crossline-byte", str(runtime_options["crossline_byte"])])
    if runtime_options["max_inlines"] is not None:
        command.extend(["--max-inlines", str(runtime_options["max_inlines"])])
    if not runtime_options["write_mask_sgy"]:
        command.append("--no-mask-sgy")
    return command


def _run_surface_seg_external(
    *,
    python_executable: Path,
    surface_root: Path,
    request: ModelInputRequest,
    output_directory: Path,
    models_dir: Path,
    runtime_options: dict[str, Any],
    runtime_log: Path,
    direct_error: str,
    progress: Progress | None,
) -> dict[str, Any]:
    if not python_executable.is_file():
        raise FileNotFoundError(
            "SurfaceSeg dependencies are unavailable in the API process and "
            f"external_python does not exist: {python_executable}; direct import: {direct_error}"
        )
    command = _surface_cli_command(
        python_executable,
        request,
        output_directory,
        models_dir,
        runtime_options,
    )
    environment = _normalized_subprocess_environment()
    # The released SurfaceSeg bundle is intentionally self-contained.  Keep
    # common model hubs offline so a missing local asset cannot turn a formal
    # prediction into an unsealed network download.
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    lines = [
        "execution_backend=external_python",
        f"python={python_executable}",
        f"direct_import_error={direct_error}",
        f"command={json.dumps(command, ensure_ascii=False)}",
    ]
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage_progress = {
        "[1/6]": 18,
        "[2/6]": 28,
        "[3/6]": 45,
        "[4/6]": 62,
        "[5/6]": 88,
        "[6/6]": 94,
    }
    stage_ranges = {
        "[2/6]": (28, 44),
        "[3/6]": (45, 61),
        "[4/6]": (62, 87),
    }
    launched = False
    try:
        with managed_popen(
            command,
            cwd=surface_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            launched = True
            assert process.stdout is not None
            current_stage: str | None = None
            last_progress = 0
            with runtime_log.open("a", encoding="utf-8") as sink:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    lines.append(line)
                    sink.write(line + "\n")
                    sink.flush()
                    if not progress:
                        continue
                    stage_updated = False
                    for marker, percent in stage_progress.items():
                        if line.startswith(marker):
                            current_stage = marker
                            last_progress = max(last_progress, percent)
                            progress(percent, f"SurfaceSeg {line}")
                            stage_updated = True
                            break
                    if stage_updated or current_stage not in stage_ranges:
                        continue
                    match = re.search(r"(\d+)\s*/\s*(\d+)\s+inlines", line)
                    if match is None:
                        continue
                    completed = int(match.group(1))
                    total = int(match.group(2))
                    if total < 1 or completed < 0:
                        continue
                    start, stop = stage_ranges[current_stage]
                    bounded = min(completed, total)
                    percent = start + int((stop - start) * bounded / total)
                    if percent > last_progress:
                        last_progress = percent
                        progress(percent, f"SurfaceSeg {line.strip()}")
            return_code = process.wait()
    except OSError as exc:
        if launched:
            raise
        lines.append(f"launch_error={type(exc).__name__}: {exc}")
        with runtime_log.open("a", encoding="utf-8") as sink:
            sink.write(lines[-1] + "\n")
        raise RuntimeError(f"无法启动 SurfaceSeg 外部 Python：{exc}") from exc
    lines.append(f"return_code={return_code}")
    with runtime_log.open("a", encoding="utf-8") as sink:
        sink.write(lines[-1] + "\n")
    if return_code:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(f"SurfaceSeg 外部推理失败（退出码 {return_code}）：\n{tail}")
    summary_path = output_directory / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"SurfaceSeg 推理未生成 summary.json：{summary_path}")
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("SurfaceSeg summary.json 必须是 JSON 对象")
    return document


def run_surface_seg_prediction(
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
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the bundled three-stage stratigraphic instance segmentation model."""
    del patch_size, overlap
    options = dict(options or {})
    surface_config = dict(config.get("surface_seg", {}))
    options.setdefault("device", device_name)
    forbidden_policy_overrides = sorted(
        key
        for key in (
            "execution_backend",
            "external_python",
            "preprocessing_policy",
            "window_overlap",
            "max_tiles_per_inline",
            "minimum_uniform_scale",
            "association_max_gap",
            "confidence_threshold",
            "horizon_display_gate",
            "minimum_horizon_finite_trace_fraction",
            "minimum_horizon_largest_component_fraction",
            "horizon_surface_connectivity",
        )
        if key in options
    )
    if forbidden_policy_overrides:
        raise ValueError(
            "SurfaceSeg execution and global-label policy are controlled by the "
            "sealed service configuration and cannot be overridden by prediction "
            "options: "
            + ", ".join(forbidden_policy_overrides)
        )
    backend_policy = str(
        surface_config.get("execution_backend", "auto")
    ).strip().casefold()
    if backend_policy not in {"auto", "in_process", "external_python"}:
        raise ValueError(
            "SurfaceSeg execution_backend must be auto, in_process, or external_python"
        )
    runtime_options = _surface_runtime_options(
        surface_config,
        options,
        threshold=threshold,
    )
    if not 0.35 <= runtime_options["confidence_threshold"] <= 1.0:
        raise ValueError(
            "SurfaceSeg sealed confidence_threshold must stay within [0.35, 1.0]"
        )
    preprocessing_policy = str(runtime_options["preprocessing_policy"])
    if preprocessing_policy not in {
        SURFACE_CHECKPOINT_NATIVE_PREPROCESSING_POLICY,
        SURFACE_EXPERIMENTAL_ASPECT_PRESERVING_POLICY,
    }:
        raise ValueError("SurfaceSeg sealed preprocessing_policy is unsupported")
    if not 0.0 <= runtime_options["window_overlap"] < 0.5:
        raise ValueError("SurfaceSeg sealed window_overlap must stay within [0, 0.5)")
    if not 1 <= runtime_options["max_tiles_per_inline"] <= 4:
        raise ValueError(
            "SurfaceSeg sealed max_tiles_per_inline must stay within [1, 4]"
        )
    if not 0.0 < runtime_options["minimum_uniform_scale"] <= 1.0:
        raise ValueError(
            "SurfaceSeg sealed minimum_uniform_scale must stay within (0, 1]"
        )
    if preprocessing_policy == SURFACE_CHECKPOINT_NATIVE_PREPROCESSING_POLICY and (
        runtime_options["window_overlap"] != 0.0
        or runtime_options["max_tiles_per_inline"] != 1
    ):
        raise ValueError(
            "SurfaceSeg checkpoint-native policy requires window_overlap=0 and "
            "max_tiles_per_inline=1"
        )
    if (
        runtime_options["horizon_display_gate_schema"]
        != HORIZON_DISPLAY_GATE_SCHEMA
        or runtime_options["horizon_surface_connectivity"] != 4
    ):
        raise ValueError("SurfaceSeg sealed horizon_display_gate contract is invalid")
    if not (
        DEFAULT_MINIMUM_FINITE_TRACE_FRACTION
        <= runtime_options["minimum_horizon_finite_trace_fraction"]
        <= 1.0
    ):
        raise ValueError(
            "SurfaceSeg minimum horizon finite trace fraction must stay within "
            "[0.10, 1]"
        )
    if not (
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
        <= runtime_options["minimum_horizon_largest_component_fraction"]
        <= 1.0
    ):
        raise ValueError(
            "SurfaceSeg minimum horizon largest component fraction must stay "
            "within [0.05, 1]"
        )
    surface_root = _project_path(
        project_root,
        options.get(
            "model_root",
            surface_config.get("model_root", "接口模型/seismic_surface_seg"),
        ),
    )
    models_dir = _project_path(
        project_root,
        options.get(
            "models_dir",
            surface_config.get("models_dir", surface_root / "models"),
        ),
    )
    required_checkpoints = (
        ("segformer-base", models_dir / "segformer-base" / "best.pt"),
        ("segformer-refine", models_dir / "segformer-refine" / "best.pt"),
        ("mask2former", models_dir / "mask2former" / "best.pt"),
    )
    checkpoint_manifest = surface_config.get("checkpoint_manifest", {})
    for stage, checkpoint in required_checkpoints:
        expected = (
            checkpoint_manifest.get(stage)
            if isinstance(checkpoint_manifest, dict)
            and isinstance(checkpoint_manifest.get(stage), dict)
            else None
        )
        _verify_surface_checkpoint(checkpoint, expected)

    if progress:
        progress(10, "正在核验地层分割三维后叠加 SEG-Y 输入")
    batch = adapters.get("seismic_surface_seg").prepare(request)
    if runtime_options["inline_count"] is None:
        inferred_inline_count = batch.provenance.get("native_inline_count")
        if inferred_inline_count is not None:
            runtime_options["inline_count"] = int(inferred_inline_count)
    recommended_options = batch.provenance.get("recommended_options", {})
    if isinstance(recommended_options, Mapping):
        for option_name in ("inline_byte", "crossline_byte"):
            if runtime_options[option_name] is None and recommended_options.get(option_name) is not None:
                runtime_options[option_name] = int(recommended_options[option_name])
    output_directory.mkdir(parents=True, exist_ok=True)
    runtime_log = output_directory / "surface_seg_runtime.log"

    execution_backend = "in_process"
    direct_error = ""
    runtime = None
    if backend_policy != "external_python":
        try:
            runtime = _load_surface_seg_runtime(surface_root)
        except (ImportError, ModuleNotFoundError, SystemExit) as exc:
            direct_error = f"{type(exc).__name__}: {exc}"
            if backend_policy == "in_process":
                raise RuntimeError(
                    "SurfaceSeg was configured for in-process execution but its "
                    f"dependencies could not be loaded: {direct_error}"
                ) from exc
    else:
        direct_error = "configured_external_python"
    if runtime is not None:
        if progress:
            progress(18, "主服务依赖可用，正在进程内启动 SurfaceSeg 三阶段推理")
        upstream = runtime.run_inference(
            request.source,
            output_directory,
            models_dir=models_dir,
            **runtime_options,
        )
        runtime_log.write_text(
            "\n".join(
                (
                    "execution_backend=in_process",
                    f"python={sys.executable}",
                    "status=completed",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        execution_backend = "external_python"
        external_python = _project_path(
            project_root,
            os.getenv("WELL_SEISMIC_SURFACE_PYTHON")
            or os.getenv("WELLFUSE_PYTHON")
            or surface_config.get("external_python", sys.executable),
        )
        if progress:
            progress(15, f"主服务缺少模型依赖，切换外部环境：{external_python}")
        upstream = _run_surface_seg_external(
            python_executable=external_python,
            surface_root=surface_root,
            request=request,
            output_directory=output_directory,
            models_dir=models_dir,
            runtime_options=runtime_options,
            runtime_log=runtime_log,
            direct_error=direct_error,
            progress=progress,
        )

    if not isinstance(upstream, dict):
        raise RuntimeError("SurfaceSeg Python API must return a mapping")
    expected_prior_mode = str(
        surface_config.get(
            "prior_compatibility_mode", "segformer-base-as-refine-prior"
        )
    ).strip()
    actual_prior_mode = str(upstream.get("prior_compatibility_mode") or "").strip()
    if actual_prior_mode != expected_prior_mode:
        raise RuntimeError(
            "SurfaceSeg Refine prior mode differs from the sealed prediction "
            f"configuration: expected={expected_prior_mode!r}, "
            f"actual={actual_prior_mode!r}"
        )
    artifacts = upstream.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("SurfaceSeg result is missing artifacts")
    global_reconciliation_raw = upstream.get("global_reconciliation")
    abstention_raw = upstream.get("abstention")
    window_inference_raw = upstream.get("window_inference")
    for value, label in (
        (global_reconciliation_raw, "global_reconciliation"),
        (abstention_raw, "abstention"),
        (window_inference_raw, "window_inference"),
    ):
        if not isinstance(value, Mapping) or not value:
            raise RuntimeError(
                f"SurfaceSeg new inference is missing required {label} receipt"
            )
    global_reconciliation = dict(global_reconciliation_raw)
    abstention = dict(abstention_raw)
    window_inference = dict(window_inference_raw)
    if (
        global_reconciliation.get("schema_version")
        != "surface-seg.global-reconciliation.v1"
        or abstention.get("schema_version") != "surface-seg.abstention.v1"
        or window_inference.get("schema_version")
        != "surface-seg.window-inference.v1"
    ):
        raise RuntimeError("SurfaceSeg producer receipt schema is incompatible")
    producer_global_ready = global_reconciliation.get("global_display_ready") is True
    if producer_global_ready:
        if not artifacts.get("global_mask_npy"):
            raise RuntimeError("SurfaceSeg global mask artifact is missing")
        selected_mask_artifact = "global_mask_npy"
    else:
        if global_reconciliation.get("output_semantics") != "local_inline_fallback":
            raise RuntimeError(
                "SurfaceSeg degraded reconciliation lacks a sealed local fallback"
            )
        selected_mask_artifact = (
            "local_mask_npy" if artifacts.get("local_mask_npy") else "mask_npy"
        )
        if not artifacts.get(selected_mask_artifact):
            raise RuntimeError("SurfaceSeg local Inline fallback artifact is missing")
    mask_artifact = artifacts.get(selected_mask_artifact)
    mask_path = Path(str(mask_artifact or "")).expanduser().resolve()
    confidence_path = (
        Path(str(artifacts.get("confidence_npy", ""))).expanduser().resolve()
    )
    if not mask_path.is_file() or not confidence_path.is_file():
        raise FileNotFoundError("SurfaceSeg did not create mask.npy and confidence.npy")
    labels = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    confidence = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 3 or confidence.shape != labels.shape:
        raise ValueError(
            f"SurfaceSeg output shape mismatch: labels={labels.shape}, confidence={confidence.shape}"
        )
    shape_ics = [int(value) for value in labels.shape]
    window_receipt_reasons = validate_surface_window_inference_receipt(
        window_inference,
        expected_native_shape=(shape_ics[2], shape_ics[1]),
        expected_max_tiles=runtime_options["max_tiles_per_inline"],
        expected_overlap=runtime_options["window_overlap"],
        expected_minimum_uniform_scale=(
            runtime_options["minimum_uniform_scale"]
            if preprocessing_policy
            == SURFACE_EXPERIMENTAL_ASPECT_PRESERVING_POLICY
            else None
        ),
        expected_preprocessing_policy=preprocessing_policy,
        expected_tile_selection_policy=(
            "maximum-uniform-scale-within-tile-budget"
            if preprocessing_policy
            == SURFACE_EXPERIMENTAL_ASPECT_PRESERVING_POLICY
            else "whole-inline-single-window"
        ),
        hard_max_tiles=4,
    )
    if window_receipt_reasons:
        raise RuntimeError(
            "SurfaceSeg window inference receipt is invalid: "
            + ", ".join(window_receipt_reasons)
        )

    try:
        unknown_label = int(abstention.get("unknown_label"))
        confidence_threshold = float(abstention.get("confidence_threshold"))
        valid_voxel_count = int(abstention.get("valid_voxel_count"))
        unknown_voxel_count = int(abstention.get("unknown_voxel_count"))
        invalid_grid_voxel_count = int(
            abstention.get("invalid_grid_voxel_count", 0)
        )
        unknown_fraction = float(abstention.get("unknown_fraction"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("SurfaceSeg abstention receipt is incomplete") from exc
    total_voxel_count = int(labels.size)
    actual_unknown_count = sum(
        int(np.count_nonzero(np.asarray(labels[index]) == unknown_label))
        for index in range(shape_ics[0])
    )
    if (
        unknown_label != -1
        or not np.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
        or not np.isclose(
            confidence_threshold,
            runtime_options["confidence_threshold"],
            rtol=0.0,
            atol=1e-9,
        )
        or min(valid_voxel_count, unknown_voxel_count, invalid_grid_voxel_count) < 0
        or unknown_voxel_count > valid_voxel_count
        or valid_voxel_count + invalid_grid_voxel_count != total_voxel_count
        or actual_unknown_count
        != unknown_voxel_count + invalid_grid_voxel_count
        or not np.isfinite(unknown_fraction)
        or not np.isclose(
            unknown_fraction,
            unknown_voxel_count / valid_voxel_count if valid_voxel_count else 1.0,
            rtol=1e-6,
            atol=1e-9,
        )
    ):
        raise RuntimeError("SurfaceSeg abstention counts or threshold are inconsistent")
    for index in range(shape_ics[0]):
        unknown = np.asarray(labels[index]) == unknown_label
        inline_confidence = np.asarray(confidence[index])
        if (
            np.any(~np.isfinite(inline_confidence))
            or np.any(inline_confidence < 0.0)
            or np.any(inline_confidence > 1.0)
        ):
            raise RuntimeError("SurfaceSeg confidence must be finite and within [0, 1]")
        if np.any(inline_confidence[unknown] != 0.0):
            raise RuntimeError("SurfaceSeg unknown voxels must have zero confidence")
        if np.any(inline_confidence[~unknown] < confidence_threshold):
            raise RuntimeError(
                "SurfaceSeg labels below the confidence threshold must be unknown"
            )

    global_display_reasons: list[str] = []
    if not producer_global_ready:
        global_display_reasons.append("global_display_ready_not_attested")
    if global_reconciliation.get("output_semantics") != "global_ordered_package_id":
        global_display_reasons.append("global_output_semantics_invalid")
    if global_reconciliation.get("non_crossing_verified") is not True:
        global_display_reasons.append("global_non_crossing_not_verified")
    if global_reconciliation.get("order_graph_acyclic") is not True:
        global_display_reasons.append("global_order_graph_not_acyclic")
    try:
        global_package_count = int(
            global_reconciliation.get("global_package_count", -1)
        )
        global_horizon_count = int(
            global_reconciliation.get("global_horizon_count", -1)
        )
        matched_transition_count = int(
            global_reconciliation.get("matched_transition_count", -1)
        )
        processed_inline_count = int(
            global_reconciliation.get("processed_inline_count", -1)
        )
        dominant_package_fraction = float(
            global_reconciliation.get("dominant_global_package_fraction")
        )
        maximum_dominant_package_fraction = float(
            global_reconciliation.get("maximum_dominant_package_fraction")
        )
    except (TypeError, ValueError, OverflowError):
        global_package_count = global_horizon_count = matched_transition_count = -1
        processed_inline_count = -1
        dominant_package_fraction = maximum_dominant_package_fraction = float("nan")
    if global_package_count < 2 or global_horizon_count < 1 or global_horizon_count != max(
        global_package_count - 1, 0
    ):
        global_display_reasons.append("global_reconciliation_counts_invalid")
    if (
        not np.isfinite(dominant_package_fraction)
        or not np.isfinite(maximum_dominant_package_fraction)
        or not np.isclose(
            maximum_dominant_package_fraction, 0.98, rtol=0.0, atol=1e-12
        )
        or not 0.0 <= dominant_package_fraction < maximum_dominant_package_fraction
    ):
        global_display_reasons.append("global_dominant_package_gate_invalid")
    transition_receipts = global_reconciliation.get("transition_receipts")
    if (
        shape_ics[0] < 2
        or processed_inline_count != shape_ics[0]
        or matched_transition_count <= 0
        or not isinstance(transition_receipts, Sequence)
        or isinstance(transition_receipts, (str, bytes))
        or len(transition_receipts) != shape_ics[0]
    ):
        global_display_reasons.append("global_transition_evidence_insufficient")
    if not str(global_reconciliation.get("association_scope") or "").strip():
        global_display_reasons.append("global_association_scope_missing")
    if global_horizon_count > 0 and not artifacts.get("horizon_surfaces_npz"):
        global_display_reasons.append("global_horizon_artifact_missing")
    horizon_display_contract: dict[str, Any] = {
        "valid": False,
        "reason_codes": ["horizon_display_contract_not_evaluated"],
        "raw_horizon_count": 0,
        "display_horizon_count": 0,
        "eligible_horizon_ids": [],
        "suppressed_horizon_ids": [],
    }
    if producer_global_ready and artifacts.get("horizon_surfaces_npz"):
        horizon_display_contract = validate_surface_horizon_display_contract(
            global_reconciliation,
            str(artifacts["horizon_surfaces_npz"]),
            global_mask_path=mask_path,
            expected_shape_ics=tuple(shape_ics),
            expected_minimum_finite_trace_fraction=runtime_options[
                "minimum_horizon_finite_trace_fraction"
            ],
            expected_minimum_largest_component_fraction=runtime_options[
                "minimum_horizon_largest_component_fraction"
            ],
        )
        if not horizon_display_contract["valid"]:
            raise RuntimeError(
                "SurfaceSeg horizon display receipt is inconsistent with raw "
                "NPZ/mask artifacts: "
                + ", ".join(horizon_display_contract["reason_codes"])
            )
    elif not producer_global_ready:
        gate = global_reconciliation.get("horizon_display_gate")
        expected_empty_gate = bool(
            isinstance(gate, Mapping)
            and gate.get("schema_version") == HORIZON_DISPLAY_GATE_SCHEMA
            and gate.get("surface_connectivity") == 4
            and gate.get("volume_connectivity_analogue") == 6
            and gate.get("finite_trace_fraction_denominator")
            == "dense_inline_xline_grid"
            and gate.get("axis_support_fraction_denominator")
            == "dense_axis_count"
            and gate.get("largest_component_fraction_denominator")
            == "finite_trace_count"
            and np.isclose(
                float(gate.get("minimum_finite_trace_fraction", -1.0)),
                runtime_options["minimum_horizon_finite_trace_fraction"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                float(gate.get("minimum_largest_component_fraction", -1.0)),
                runtime_options[
                    "minimum_horizon_largest_component_fraction"
                ],
                rtol=0.0,
                atol=1e-12,
            )
            and global_reconciliation.get("horizon_surface_receipts") == []
            and global_reconciliation.get("display_horizon_count") == 0
            and global_reconciliation.get("suppressed_horizon_ids") == []
        )
        if not expected_empty_gate:
            raise RuntimeError(
                "SurfaceSeg degraded horizon display receipt is incomplete"
            )
        horizon_display_contract = {
            "valid": True,
            "reason_codes": [],
            "raw_horizon_count": 0,
            "display_horizon_count": 0,
            "eligible_horizon_ids": [],
            "suppressed_horizon_ids": [],
        }
    global_display_ready = not global_display_reasons

    local_fallback_semantics = (
        not producer_global_ready
        and global_reconciliation.get("output_semantics") == "local_inline_fallback"
    )
    local_mask_artifact = None
    if local_fallback_semantics:
        local_mask_artifact = (
            "local_mask_npy" if artifacts.get("local_mask_npy") else "mask_npy"
        )
    local_mask_path: Path | None = None
    if local_mask_artifact and artifacts.get(local_mask_artifact):
        local_mask_path = (
            Path(str(artifacts[local_mask_artifact])).expanduser().resolve()
        )
        if not local_mask_path.is_file():
            local_mask_path = None
    display_labels = labels
    display_mask_artifact = "global_mask_npy" if global_display_ready else None
    if not global_display_ready and local_mask_path is not None:
        local_labels = np.load(local_mask_path, mmap_mode="r", allow_pickle=False)
        if local_labels.shape != labels.shape or not np.issubdtype(
            local_labels.dtype, np.integer
        ):
            raise RuntimeError("SurfaceSeg local fallback mask is incompatible")
        display_labels = local_labels
        display_mask_artifact = local_mask_artifact
    local_fallback_available = (
        not global_display_ready
        and display_mask_artifact is not None
        and display_mask_artifact == local_mask_artifact
    )
    label_range = [int(display_labels.min()), int(display_labels.max())]
    confidence_summary = {
        "shape_ics": shape_ics,
        "dtype": str(confidence.dtype),
        "min": float(confidence.min()),
        "max": float(confidence.max()),
        "mean": float(confidence.mean(dtype=np.float64)),
    }
    del display_labels, labels, confidence

    output_paths = {
        str(name): str(value)
        for name, value in artifacts.items()
        if value not in (None, "")
    }
    upstream_summary_path = output_directory / "summary.json"
    output_paths["upstream_summary_json"] = str(upstream_summary_path)
    output_paths["runtime_log"] = str(runtime_log)
    result = {
        "schema_version": "well-seismic.surface-seg-runtime.v1",
        "model_id": "seismic_surface_seg",
        "model_name": public_runtime_model_name("seismic_surface_seg"),
        "model_executed": True,
        "checkpoint_forward_calls": len(required_checkpoints),
        "scientific_status": "legacy_engineering_candidate",
        "checkpoint": str(models_dir),
        "checkpoint_evidence": [
            {
                "stage": stage,
                "path": str(checkpoint),
                "sha256": _checkpoint_sha256(
                    str(checkpoint),
                    checkpoint.stat().st_mtime_ns,
                    checkpoint.stat().st_size,
                ),
            }
            for stage, checkpoint in required_checkpoints
        ],
        "device": str(upstream.get("device", runtime_options["device"])),
        "input": {
            "source": str(request.source),
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "shape_ics": shape_ics,
            "source_shape_ics": list(batch.provenance["shape_ics"]),
            "source_shape_zyx": list(batch.provenance["source_shape_zyx"]),
            **{
                key: value
                for key, value in batch.provenance.items()
                if key not in {"source", "shape_ics", "source_shape_zyx"}
            },
        },
        "inference": {
            "execution_backend": execution_backend,
            "execution_backend_policy": backend_policy,
            "visualization_backend": upstream.get("visualization_backend"),
            "amplitude_mode": upstream.get("amplitude_scaling", {}).get(
                "effective",
                runtime_options["amplitude_mode"],
            ),
            "amplitude_mode_requested": runtime_options["amplitude_mode"],
            "query_threshold": runtime_options["query_threshold"],
            "mask_threshold": runtime_options["mask_threshold"],
            "segformer_batch_size": runtime_options["segformer_batch_size"],
            "mask2former_batch_size": runtime_options["mask2former_batch_size"],
            "inline_count": runtime_options["inline_count"],
            "inline_byte": runtime_options["inline_byte"],
            "crossline_byte": runtime_options["crossline_byte"],
            "max_inlines": runtime_options["max_inlines"],
            "write_mask_sgy": runtime_options["write_mask_sgy"],
            "prior_compatibility_mode": upstream.get("prior_compatibility_mode"),
            "window_inference": window_inference,
            "minimum_horizon_finite_trace_fraction": runtime_options[
                "minimum_horizon_finite_trace_fraction"
            ],
            "minimum_horizon_largest_component_fraction": runtime_options[
                "minimum_horizon_largest_component_fraction"
            ],
            "crop_policy": "模型按完整 Inline 切片推理；通用 crop_start/crop_size 不适用",
        },
        "segmentation": {
            "shape_ics": shape_ics,
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "dtype": str(upstream.get("mask_dtype", "int16")),
            "label_range": label_range,
            "instance_count": max(0, label_range[1] + 1),
            "max_instances_per_inline": max(0, label_range[1] + 1),
            "cross_inline_consistent": global_display_ready,
            "label_scope": (
                "global_packages"
                if global_display_ready
                else "inline_local"
                if local_fallback_available
                else "unavailable_global_reconciliation"
            ),
            "display_mask_artifact": display_mask_artifact,
            "global_display_gate": {
                "ready": global_display_ready,
                "reason_codes": list(dict.fromkeys(global_display_reasons)),
                "local_fallback_available": local_fallback_available,
                "raw_horizon_count": horizon_display_contract[
                    "raw_horizon_count"
                ],
                "display_horizon_count": horizon_display_contract[
                    "display_horizon_count"
                ],
                "suppressed_horizon_ids": horizon_display_contract[
                    "suppressed_horizon_ids"
                ],
            },
            "horizon_display_contract": horizon_display_contract,
            "global_reconciliation": global_reconciliation,
            "abstention": abstention,
            "invalid_label": unknown_label,
            "confidence_min": confidence_summary["min"],
            "confidence_max": confidence_summary["max"],
            "confidence_mean": confidence_summary["mean"],
            "confidence": confidence_summary,
        },
        "geometry": upstream.get("geometry", {}),
        "checkpoints": upstream.get("checkpoints", {}),
        "elapsed_seconds": upstream.get("elapsed_seconds"),
        "outputs": output_paths,
    }
    metadata_path = output_directory / "surface_seg_result.json"
    result["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress(98, "地层分割推理完成，正在登记标签体与置信度体")
    return result
