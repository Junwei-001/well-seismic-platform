"""Training and evaluation utilities for the unified GeoPathTie-V1 model.

The module deliberately keeps the learned component small.  The upstream
``geopath_tie`` operator owns trajectory sampling, seismic-tube construction
and physical candidate generation.  This module ranks those candidates,
predicts a bounded residual and an aleatoric scale.  Vertical, deviated and
horizontal wells share one model; geometry is used for balanced sampling and
stratified reporting, never for selecting a separate network.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geopath_tie import solve_geopath_path

try:  # Torch is an optional platform dependency until training is requested.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in the minimal installation.
    torch = None
    nn = None


GEOPATH_DATASET_SCHEMA = "well-seismic.geopath-training.v1"
GEOPATH_CHECKPOINT_SCHEMA = "well-seismic.geopath-residual-checkpoint.v1"
GEOPATH_OOF_SCHEMA = "well-seismic.geopath-oof-prediction.v1"

GEOMETRIES = ("vertical", "deviated", "horizontal")
MODEL_ARMS = ("trajectory_tube", "geopath_full")
REPORT_ARMS = ("sonic_prior", "p13_baseline", *MODEL_ARMS)

POINT_FEATURE_NAMES = (
    "sonic_minus_reference_ms",
    "p13_minus_reference_ms",
    "p13_confidence",
    "kappa",
    "inclination_fraction",
    "dogleg_fraction",
    "curve_quality",
    "tube_coherence",
    "tube_std",
    "tube_phase_dispersion",
    "md_fraction",
    "anchor_distance_scaled",
)

CANDIDATE_FEATURE_NAMES = (
    "local_score",
    "correlation",
    "score_margin_to_best",
    "rank_fraction",
    "shift_minus_sonic_ms",
    "shift_minus_p13_ms",
    "absolute_shift_ms",
    "evidence_valid",
)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "GeoPathTie training requires PyTorch. Use the wellfusing CUDA "
            "environment or install the project 'faultseg' extra."
        )


def _as_vector(values: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


@dataclass(frozen=True)
class GeoPathTensorContract:
    """Portable point/candidate tensor contract between physics and training."""

    point_features: np.ndarray
    candidate_features: np.ndarray
    candidate_shift_ms: np.ndarray
    candidate_valid_mask: np.ndarray
    reference_twt_ms: np.ndarray
    sonic_twt_ms: np.ndarray
    p13_twt_ms: np.ndarray
    target_twt_ms: np.ndarray
    supervision_mask: np.ndarray
    md_m: np.ndarray
    dominant_period_ms: np.ndarray
    well_id: np.ndarray
    family_id: np.ndarray
    geometry: np.ndarray
    point_feature_names: tuple[str, ...] = POINT_FEATURE_NAMES
    candidate_feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES
    schema_version: str = GEOPATH_DATASET_SCHEMA

    @property
    def point_count(self) -> int:
        return int(np.asarray(self.point_features).shape[0])

    @property
    def candidate_count(self) -> int:
        return int(np.asarray(self.candidate_features).shape[1])

    def validate(self) -> None:
        point = np.asarray(self.point_features)
        candidate = np.asarray(self.candidate_features)
        if self.schema_version != GEOPATH_DATASET_SCHEMA:
            raise ValueError(f"unsupported GeoPath dataset schema: {self.schema_version}")
        if point.ndim != 2:
            raise ValueError(f"point_features must be [N,P], got {point.shape}")
        if candidate.ndim != 3 or candidate.shape[0] != point.shape[0]:
            raise ValueError(
                "candidate_features must be [N,K,C] with the same N as point_features"
            )
        if tuple(self.point_feature_names) != POINT_FEATURE_NAMES:
            raise ValueError("point feature order does not match GeoPathTie-V1")
        if tuple(self.candidate_feature_names) != CANDIDATE_FEATURE_NAMES:
            raise ValueError("candidate feature order does not match GeoPathTie-V1")
        if point.shape[1] != len(POINT_FEATURE_NAMES):
            raise ValueError("point feature width does not match feature names")
        if candidate.shape[2] != len(CANDIDATE_FEATURE_NAMES):
            raise ValueError("candidate feature width does not match feature names")

        n_points, n_candidates = candidate.shape[:2]
        shifts = np.asarray(self.candidate_shift_ms)
        mask = np.asarray(self.candidate_valid_mask, dtype=bool)
        if shifts.shape != (n_points, n_candidates):
            raise ValueError("candidate_shift_ms shape must equal [N,K]")
        if mask.shape != shifts.shape:
            raise ValueError("candidate_valid_mask shape must equal candidate_shift_ms")
        if np.any(~np.any(mask, axis=1)):
            raise ValueError("each point must have at least one valid candidate")
        if not np.all(np.isfinite(point)):
            raise ValueError("point_features contains non-finite values")
        if not np.all(np.isfinite(candidate[mask])):
            raise ValueError("valid candidate features contain non-finite values")
        if not np.all(np.isfinite(shifts[mask])):
            raise ValueError("valid candidate shifts contain non-finite values")

        vectors = {
            "reference_twt_ms": self.reference_twt_ms,
            "sonic_twt_ms": self.sonic_twt_ms,
            "p13_twt_ms": self.p13_twt_ms,
            "target_twt_ms": self.target_twt_ms,
            "supervision_mask": self.supervision_mask,
            "md_m": self.md_m,
            "dominant_period_ms": self.dominant_period_ms,
            "well_id": self.well_id,
            "family_id": self.family_id,
            "geometry": self.geometry,
        }
        for name, values in vectors.items():
            _as_vector(values, name=name, length=n_points)
        supervised = np.asarray(self.supervision_mask, dtype=bool)
        target = np.asarray(self.target_twt_ms, dtype=float)
        if np.any(supervised & ~np.isfinite(target)):
            raise ValueError("supervised target_twt_ms must be finite")
        if not np.all(np.isfinite(np.asarray(self.reference_twt_ms, dtype=float))):
            raise ValueError("reference_twt_ms must be finite")
        if np.any(np.asarray(self.dominant_period_ms, dtype=float) <= 0.0):
            raise ValueError("dominant_period_ms must be positive")
        geometry = np.asarray(self.geometry, dtype=str)
        unknown = sorted(set(geometry) - set(GEOMETRIES))
        if unknown:
            raise ValueError(f"unknown well geometry labels: {unknown}")
        if np.any(np.char.str_len(np.asarray(self.well_id, dtype=str)) == 0):
            raise ValueError("well_id cannot be empty")
        if np.any(np.char.str_len(np.asarray(self.family_id, dtype=str)) == 0):
            raise ValueError("family_id cannot be empty")

    def subset(self, selection: np.ndarray) -> GeoPathTensorContract:
        indices = np.asarray(selection)
        if indices.dtype == bool and indices.shape != (self.point_count,):
            raise ValueError("boolean subset mask must match point count")
        fields = {
            field: np.asarray(getattr(self, field))[indices]
            for field in (
                "point_features",
                "candidate_features",
                "candidate_shift_ms",
                "candidate_valid_mask",
                "reference_twt_ms",
                "sonic_twt_ms",
                "p13_twt_ms",
                "target_twt_ms",
                "supervision_mask",
                "md_m",
                "dominant_period_ms",
                "well_id",
                "family_id",
                "geometry",
            )
        }
        result = replace(self, **fields)
        result.validate()
        return result

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema_version=np.asarray(self.schema_version),
            point_feature_names=np.asarray(self.point_feature_names),
            candidate_feature_names=np.asarray(self.candidate_feature_names),
            point_features=np.asarray(self.point_features, dtype=np.float32),
            candidate_features=np.asarray(self.candidate_features, dtype=np.float32),
            candidate_shift_ms=np.asarray(self.candidate_shift_ms, dtype=np.float32),
            candidate_valid_mask=np.asarray(self.candidate_valid_mask, dtype=np.uint8),
            reference_twt_ms=np.asarray(self.reference_twt_ms, dtype=np.float32),
            sonic_twt_ms=np.asarray(self.sonic_twt_ms, dtype=np.float32),
            p13_twt_ms=np.asarray(self.p13_twt_ms, dtype=np.float32),
            target_twt_ms=np.asarray(self.target_twt_ms, dtype=np.float32),
            supervision_mask=np.asarray(self.supervision_mask, dtype=np.uint8),
            md_m=np.asarray(self.md_m, dtype=np.float32),
            dominant_period_ms=np.asarray(self.dominant_period_ms, dtype=np.float32),
            well_id=np.asarray(self.well_id, dtype=str),
            family_id=np.asarray(self.family_id, dtype=str),
            geometry=np.asarray(self.geometry, dtype=str),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> GeoPathTensorContract:
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "schema_version",
                "point_feature_names",
                "candidate_feature_names",
                "point_features",
                "candidate_features",
                "candidate_shift_ms",
                "candidate_valid_mask",
                "reference_twt_ms",
                "sonic_twt_ms",
                "p13_twt_ms",
                "target_twt_ms",
                "supervision_mask",
                "md_m",
                "dominant_period_ms",
                "well_id",
                "family_id",
                "geometry",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"GeoPath dataset is missing fields: {missing}")
            result = cls(
                schema_version=str(payload["schema_version"]),
                point_feature_names=tuple(str(item) for item in payload["point_feature_names"]),
                candidate_feature_names=tuple(
                    str(item) for item in payload["candidate_feature_names"]
                ),
                point_features=payload["point_features"].astype(np.float32),
                candidate_features=payload["candidate_features"].astype(np.float32),
                candidate_shift_ms=payload["candidate_shift_ms"].astype(np.float32),
                candidate_valid_mask=payload["candidate_valid_mask"].astype(bool),
                reference_twt_ms=payload["reference_twt_ms"].astype(np.float32),
                sonic_twt_ms=payload["sonic_twt_ms"].astype(np.float32),
                p13_twt_ms=payload["p13_twt_ms"].astype(np.float32),
                target_twt_ms=payload["target_twt_ms"].astype(np.float32),
                supervision_mask=payload["supervision_mask"].astype(bool),
                md_m=payload["md_m"].astype(np.float32),
                dominant_period_ms=payload["dominant_period_ms"].astype(np.float32),
                well_id=payload["well_id"].astype(str),
                family_id=payload["family_id"].astype(str),
                geometry=payload["geometry"].astype(str),
            )
        result.validate()
        return result


def contract_from_geopath_outputs(
    *,
    reference_twt_ms: np.ndarray,
    sonic_twt_ms: np.ndarray,
    p13_twt_ms: np.ndarray,
    p13_confidence: np.ndarray,
    kappa: np.ndarray,
    inclination_deg: np.ndarray,
    dogleg_deg_per_30m: np.ndarray,
    curve_quality: np.ndarray,
    tube_coherence: np.ndarray,
    tube_std: np.ndarray,
    tube_phase: np.ndarray,
    candidate_shift_ms: np.ndarray,
    candidate_score: np.ndarray,
    candidate_correlation: np.ndarray,
    candidate_valid_mask: np.ndarray,
    target_twt_ms: np.ndarray,
    supervision_mask: np.ndarray,
    md_m: np.ndarray,
    dominant_period_ms: np.ndarray,
    well_id: Sequence[str] | np.ndarray,
    family_id: Sequence[str] | np.ndarray,
    geometry: Sequence[str] | np.ndarray,
    anchor_distance_m: np.ndarray | None = None,
) -> GeoPathTensorContract:
    """Adapt physics outputs into the frozen V1 tensor order.

    ``tube_*`` accepts either one summary value per depth or an ``[N,T]``
    tube.  In the latter case robust depth-wise summaries are calculated here.
    Candidate shifts are residuals relative to ``reference_twt_ms``.
    """

    reference = np.asarray(reference_twt_ms, dtype=float)
    n_points = reference.size
    if reference.ndim != 1:
        raise ValueError("reference_twt_ms must be one-dimensional")

    def vector(values: Any, name: str) -> np.ndarray:
        return np.asarray(_as_vector(values, name=name, length=n_points), dtype=float)

    def tube_summary(values: Any, name: str, *, dispersion: bool = False) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            return vector(array, name)
        if array.ndim != 2 or array.shape[0] != n_points:
            raise ValueError(f"{name} must be [N] or [N,T], got {array.shape}")
        if dispersion:
            centered = np.angle(np.exp(1j * array).mean(axis=1))
            return np.nanmedian(np.abs(array - centered[:, None]), axis=1)
        return np.nanmedian(array, axis=1)

    sonic = vector(sonic_twt_ms, "sonic_twt_ms")
    p13 = vector(p13_twt_ms, "p13_twt_ms")
    md = vector(md_m, "md_m")
    shift = np.asarray(candidate_shift_ms, dtype=float)
    score = np.asarray(candidate_score, dtype=float)
    correlation = np.asarray(candidate_correlation, dtype=float)
    valid = np.asarray(candidate_valid_mask, dtype=bool)
    if shift.ndim != 2 or score.shape != shift.shape or correlation.shape != shift.shape:
        raise ValueError("candidate shift, score and correlation must share [N,K]")
    if valid.shape != shift.shape or shift.shape[0] != n_points:
        raise ValueError("candidate_valid_mask must share candidate [N,K]")

    md_min = np.zeros(n_points, dtype=float)
    md_span = np.ones(n_points, dtype=float)
    wells = np.asarray(well_id, dtype=str)
    for name in np.unique(wells):
        selection = wells == name
        lower = float(np.nanmin(md[selection]))
        span = max(float(np.nanmax(md[selection]) - lower), 1.0)
        md_min[selection] = lower
        md_span[selection] = span
    anchor = (
        np.full(n_points, 1000.0, dtype=float)
        if anchor_distance_m is None
        else vector(anchor_distance_m, "anchor_distance_m")
    )
    coherence_summary = tube_summary(tube_coherence, "tube_coherence")
    std_summary = tube_summary(tube_std, "tube_std")
    phase_dispersion = tube_summary(tube_phase, "tube_phase", dispersion=True)
    point_features = np.column_stack(
        (
            sonic - reference,
            p13 - reference,
            vector(p13_confidence, "p13_confidence"),
            vector(kappa, "kappa"),
            vector(inclination_deg, "inclination_deg") / 90.0,
            vector(dogleg_deg_per_30m, "dogleg_deg_per_30m") / 30.0,
            vector(curve_quality, "curve_quality"),
            coherence_summary,
            std_summary,
            phase_dispersion,
            (md - md_min) / md_span,
            np.clip(anchor / 1000.0, 0.0, 10.0),
        )
    )

    masked_score = np.where(valid, score, -np.inf)
    best_score = np.max(masked_score, axis=1, keepdims=True)
    order = np.argsort(np.argsort(-masked_score, axis=1), axis=1)
    rank_fraction = order / max(shift.shape[1] - 1, 1)
    candidate_features = np.stack(
        (
            score,
            correlation,
            best_score - score,
            rank_fraction,
            shift - (sonic - reference)[:, None],
            shift - (p13 - reference)[:, None],
            np.abs(shift),
            valid.astype(float),
        ),
        axis=-1,
    )
    candidate_features[~valid] = 0.0
    shift = np.where(valid, shift, 0.0)
    point_features = np.nan_to_num(point_features, nan=0.0, posinf=0.0, neginf=0.0)

    result = GeoPathTensorContract(
        point_features=point_features.astype(np.float32),
        candidate_features=candidate_features.astype(np.float32),
        candidate_shift_ms=shift.astype(np.float32),
        candidate_valid_mask=valid,
        reference_twt_ms=reference.astype(np.float32),
        sonic_twt_ms=sonic.astype(np.float32),
        p13_twt_ms=p13.astype(np.float32),
        target_twt_ms=np.asarray(target_twt_ms, dtype=np.float32),
        supervision_mask=np.asarray(supervision_mask, dtype=bool),
        md_m=md.astype(np.float32),
        dominant_period_ms=np.asarray(dominant_period_ms, dtype=np.float32),
        well_id=wells,
        family_id=np.asarray(family_id, dtype=str),
        geometry=np.asarray(geometry, dtype=str),
    )
    result.validate()
    return result


def concatenate_contracts(parts: Sequence[GeoPathTensorContract]) -> GeoPathTensorContract:
    if not parts:
        raise ValueError("at least one GeoPath tensor part is required")
    for part in parts:
        part.validate()
    candidate_counts = {part.candidate_count for part in parts}
    if len(candidate_counts) != 1:
        raise ValueError("all contracts must use the same candidate count")
    fields = {
        name: np.concatenate([np.asarray(getattr(part, name)) for part in parts], axis=0)
        for name in (
            "point_features",
            "candidate_features",
            "candidate_shift_ms",
            "candidate_valid_mask",
            "reference_twt_ms",
            "sonic_twt_ms",
            "p13_twt_ms",
            "target_twt_ms",
            "supervision_mask",
            "md_m",
            "dominant_period_ms",
            "well_id",
            "family_id",
            "geometry",
        )
    }
    result = GeoPathTensorContract(**fields)
    result.validate()
    return result


def build_group_folds(
    contract: GeoPathTensorContract,
    *,
    n_splits: int = 5,
    seed: int = 20260819,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assign complete well families to approximately geometry-balanced folds."""

    contract.validate()
    families = np.asarray(contract.family_id, dtype=str)
    wells = np.asarray(contract.well_id, dtype=str)
    geometries = np.asarray(contract.geometry, dtype=str)
    unique_families = np.unique(families)
    if not 2 <= n_splits <= unique_families.size:
        raise ValueError("n_splits must be between 2 and the number of families")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for family in unique_families:
        selection = families == family
        geometry_values, counts = np.unique(geometries[selection], return_counts=True)
        geometry = str(geometry_values[int(np.argmax(counts))])
        records.append(
            {
                "family_id": str(family),
                "geometry": geometry,
                "point_count": int(np.count_nonzero(selection)),
                "wells": sorted(set(wells[selection])),
                "tie_break": float(rng.random()),
            }
        )

    fold_load = np.zeros(n_splits, dtype=float)
    fold_geometry = {name: np.zeros(n_splits, dtype=float) for name in GEOMETRIES}
    assignment: dict[str, int] = {}
    for geometry in GEOMETRIES:
        group = [record for record in records if record["geometry"] == geometry]
        group.sort(key=lambda item: (-item["point_count"], item["tie_break"]))
        for record in group:
            geometry_load = fold_geometry[geometry]
            candidates = np.flatnonzero(geometry_load == geometry_load.min())
            if candidates.size > 1:
                total = fold_load[candidates]
                candidates = candidates[total == total.min()]
            fold = int(candidates[0])
            assignment[str(record["family_id"])] = fold
            weight = float(record["point_count"])
            fold_load[fold] += weight
            fold_geometry[geometry][fold] += weight
            record["fold"] = fold

    point_folds = np.asarray([assignment[family] for family in families], dtype=np.int16)
    manifest: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["family_id"])):
        for well in record["wells"]:
            manifest.append(
                {
                    "well_id": well,
                    "family_id": record["family_id"],
                    "geometry": record["geometry"],
                    "fold": int(record["fold"]),
                    "family_point_count": int(record["point_count"]),
                }
            )
    return point_folds, manifest


def load_fixed_fold_assignment(
    contract: GeoPathTensorContract,
    path: str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Map the checked-in Chengdu well/family split contract onto tensor rows."""

    contract.validate()
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or len(folds) != 5:
        raise ValueError("GeoPathTie-V1 fixed split must contain exactly five folds")
    well_to_fold: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    for fold_index, fold_name in enumerate(sorted(folds)):
        fold_payload = folds[fold_name]
        if not isinstance(fold_payload, Mapping):
            raise TypeError(f"invalid fold payload: {fold_name}")
        validation_wells = fold_payload.get("validation_wells")
        if not isinstance(validation_wells, list):
            raise TypeError(f"{fold_name} has no validation_wells list")
        for well in validation_wells:
            name = str(well)
            if name in well_to_fold:
                raise ValueError(f"well appears in multiple fixed folds: {name}")
            well_to_fold[name] = fold_index

    point_folds = np.full(contract.point_count, -1, dtype=np.int16)
    wells = np.asarray(contract.well_id, dtype=str)
    families = np.asarray(contract.family_id, dtype=str)
    geometries = np.asarray(contract.geometry, dtype=str)
    for well in np.unique(wells):
        selection = wells == well
        if well not in well_to_fold:
            if np.any(contract.supervision_mask[selection]):
                raise ValueError(f"supervised well is absent from fixed folds: {well}")
            continue
        fold = well_to_fold[well]
        point_folds[selection] = fold
        manifest.append(
            {
                "well_id": str(well),
                "family_id": str(families[selection][0]),
                "geometry": str(geometries[selection][0]),
                "fold": int(fold),
                "point_count": int(np.count_nonzero(selection)),
                "split_source": str(source),
            }
        )
    for family in np.unique(families[point_folds >= 0]):
        assigned = np.unique(point_folds[(families == family) & (point_folds >= 0)])
        if assigned.size != 1:
            raise ValueError(f"family crosses fixed folds: {family}")
    return point_folds, manifest


def balanced_sample_weights(
    well_id: Sequence[str] | np.ndarray,
    geometry: Sequence[str] | np.ndarray,
    selection: np.ndarray | None = None,
) -> np.ndarray:
    """Equalize geometry totals, wells within geometry, and points within well."""

    wells = np.asarray(well_id, dtype=str)
    geometries = np.asarray(geometry, dtype=str)
    if wells.shape != geometries.shape or wells.ndim != 1:
        raise ValueError("well_id and geometry must be same-length vectors")
    selected = (
        np.ones(wells.size, dtype=bool)
        if selection is None
        else np.asarray(selection, dtype=bool)
    )
    if selected.shape != wells.shape or not np.any(selected):
        raise ValueError("sample selection must contain at least one point")
    result = np.zeros(wells.size, dtype=np.float64)
    present_geometries = sorted(set(geometries[selected]))
    for geometry_name in present_geometries:
        geometry_mask = selected & (geometries == geometry_name)
        geometry_wells = sorted(set(wells[geometry_mask]))
        for well in geometry_wells:
            points = geometry_mask & (wells == well)
            result[points] = 1.0 / (
                len(present_geometries) * len(geometry_wells) * np.count_nonzero(points)
            )
    result[selected] /= float(np.mean(result[selected]))
    return result


@dataclass(frozen=True)
class GeoPathModelConfig:
    hidden_dim: int = 64
    dropout: float = 0.10
    max_residual_ms: float = 24.0
    min_sigma_ms: float = 2.0
    max_sigma_ms: float = 160.0


@dataclass(frozen=True)
class GeoPathTrainConfig:
    epochs: int = 24
    batch_size: int = 1024
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 20260819
    classification_weight: float = 0.35
    regression_weight: float = 1.0
    uncertainty_weight: float = 0.08
    gradient_clip_norm: float = 5.0


DEFAULT_MODEL_CONFIG = GeoPathModelConfig()
DEFAULT_TRAIN_CONFIG = GeoPathTrainConfig()


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    location = np.nanmedian(array, axis=0)
    lower = np.nanquantile(array, 0.25, axis=0)
    upper = np.nanquantile(array, 0.75, axis=0)
    scale = upper - lower
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-5), scale, 1.0)
    location = np.where(np.isfinite(location), location, 0.0)
    return location.astype(np.float32), scale.astype(np.float32)


if nn is not None:

    class GeoPathResidualModel(nn.Module):
        """Small shared candidate-ranker and bounded residual head."""

        def __init__(
            self,
            config: GeoPathModelConfig,
            *,
            point_location: np.ndarray,
            point_scale: np.ndarray,
            candidate_location: np.ndarray,
            candidate_scale: np.ndarray,
        ) -> None:
            super().__init__()
            self.config = config
            hidden = int(config.hidden_dim)
            self.register_buffer(
                "point_location", torch.as_tensor(point_location, dtype=torch.float32)
            )
            self.register_buffer(
                "point_scale", torch.as_tensor(point_scale, dtype=torch.float32)
            )
            self.register_buffer(
                "candidate_location",
                torch.as_tensor(candidate_location, dtype=torch.float32),
            )
            self.register_buffer(
                "candidate_scale", torch.as_tensor(candidate_scale, dtype=torch.float32)
            )
            self.point_encoder = nn.Sequential(
                nn.Linear(len(POINT_FEATURE_NAMES), hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.candidate_encoder = nn.Sequential(
                nn.Linear(len(CANDIDATE_FEATURE_NAMES) + 1, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.candidate_score = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            self.residual_head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            self.sigma_head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )

        def forward(
            self,
            point_features: Any,
            candidate_features: Any,
            candidate_shift_ms: Any,
            candidate_valid_mask: Any,
        ) -> dict[str, Any]:
            point = torch.nan_to_num(
                (point_features - self.point_location) / self.point_scale
            ).clamp(-12.0, 12.0)
            candidate = torch.nan_to_num(
                (candidate_features - self.candidate_location)
                / self.candidate_scale
            ).clamp(-12.0, 12.0)
            shift_scaled = (candidate_shift_ms / 120.0).unsqueeze(-1)
            point_embedding = self.point_encoder(point)
            candidate_embedding = self.candidate_encoder(
                torch.cat((candidate, shift_scaled), dim=-1)
            )
            context = point_embedding.unsqueeze(1).expand_as(candidate_embedding)
            logits = self.candidate_score(
                torch.cat((context, candidate_embedding), dim=-1)
            ).squeeze(-1)
            logits = logits.masked_fill(~candidate_valid_mask.bool(), -1.0e9)
            probability = torch.softmax(logits, dim=-1)
            pooled = torch.sum(probability.unsqueeze(-1) * candidate_embedding, dim=1)
            joint = torch.cat((point_embedding, pooled), dim=-1)
            residual = self.config.max_residual_ms * torch.tanh(
                self.residual_head(joint).squeeze(-1)
            )
            expected_candidate = torch.sum(probability * candidate_shift_ms, dim=-1)
            sigma_fraction = torch.sigmoid(self.sigma_head(joint).squeeze(-1))
            sigma = self.config.min_sigma_ms + sigma_fraction * (
                self.config.max_sigma_ms - self.config.min_sigma_ms
            )
            return {
                "candidate_logits": logits,
                "candidate_probability": probability,
                "expected_candidate_shift_ms": expected_candidate,
                "residual_ms": residual,
                "predicted_shift_ms": expected_candidate + residual,
                "sigma_ms": sigma,
            }

else:  # pragma: no cover - only used by a minimal NumPy-only installation.

    class GeoPathResidualModel:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def _arm_point_features(point_features: np.ndarray, arm: str) -> np.ndarray:
    result = np.asarray(point_features, dtype=np.float32).copy()
    if arm == "trajectory_tube":
        result[:, POINT_FEATURE_NAMES.index("kappa")] = 0.0
        result[:, POINT_FEATURE_NAMES.index("anchor_distance_scaled")] = 0.0
    elif arm != "geopath_full":
        raise ValueError(f"unknown trainable GeoPath arm: {arm}")
    return result


def _fit_normalization(
    contract: GeoPathTensorContract,
    train_mask: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_location, point_scale = _robust_location_scale(
        _arm_point_features(contract.point_features[train_mask], arm)
    )
    candidate_mask = contract.candidate_valid_mask[train_mask]
    candidates = contract.candidate_features[train_mask][candidate_mask]
    candidate_location, candidate_scale = _robust_location_scale(candidates)
    return point_location, point_scale, candidate_location, candidate_scale


def resolve_device(requested: str = "auto") -> str:
    _require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this Python environment has no CUDA torch")
    return requested


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_geopath_model(
    contract: GeoPathTensorContract,
    train_mask: np.ndarray,
    *,
    arm: Literal["trajectory_tube", "geopath_full"],
    model_config: GeoPathModelConfig = DEFAULT_MODEL_CONFIG,
    train_config: GeoPathTrainConfig = DEFAULT_TRAIN_CONFIG,
    device: str = "auto",
) -> tuple[GeoPathResidualModel, list[dict[str, float]]]:
    """Fit exactly ``epochs`` epochs and return the final, not best, weights."""

    _require_torch()
    contract.validate()
    selected = np.asarray(train_mask, dtype=bool) & np.asarray(
        contract.supervision_mask, dtype=bool
    )
    if selected.shape != (contract.point_count,) or np.count_nonzero(selected) < 8:
        raise ValueError("GeoPath training requires at least eight supervised points")
    resolved_device = resolve_device(device)
    _seed_everything(train_config.seed)
    normalization = _fit_normalization(contract, selected, arm)
    model = GeoPathResidualModel(
        model_config,
        point_location=normalization[0],
        point_scale=normalization[1],
        candidate_location=normalization[2],
        candidate_scale=normalization[3],
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_indices = np.flatnonzero(selected)
    weights = balanced_sample_weights(
        contract.well_id, contract.geometry, selected
    )[train_indices]
    probability = weights / weights.sum()
    target_shift = contract.target_twt_ms - contract.reference_twt_ms
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    point_view = _arm_point_features(contract.point_features, arm)

    for epoch in range(train_config.epochs):
        model.train()
        sampled = rng.choice(
            train_indices,
            size=train_indices.size,
            replace=True,
            p=probability,
        )
        losses: list[float] = []
        for start in range(0, sampled.size, train_config.batch_size):
            index = sampled[start : start + train_config.batch_size]
            point = torch.as_tensor(
                point_view[index], dtype=torch.float32, device=resolved_device
            )
            candidate = torch.as_tensor(
                contract.candidate_features[index],
                dtype=torch.float32,
                device=resolved_device,
            )
            shifts = torch.as_tensor(
                contract.candidate_shift_ms[index],
                dtype=torch.float32,
                device=resolved_device,
            )
            candidate_mask = torch.as_tensor(
                contract.candidate_valid_mask[index],
                dtype=torch.bool,
                device=resolved_device,
            )
            target = torch.as_tensor(
                target_shift[index], dtype=torch.float32, device=resolved_device
            )
            output = model(point, candidate, shifts, candidate_mask)
            distance = torch.abs(shifts - target.unsqueeze(-1)).masked_fill(
                ~candidate_mask, 1.0e9
            )
            target_candidate = torch.argmin(distance, dim=-1)
            classification = torch.nn.functional.cross_entropy(
                output["candidate_logits"], target_candidate
            )
            error = output["predicted_shift_ms"] - target
            regression = torch.nn.functional.smooth_l1_loss(
                output["predicted_shift_ms"], target, beta=12.0
            )
            sigma = output["sigma_ms"]
            uncertainty = torch.mean(0.5 * torch.square(error / sigma) + torch.log(sigma))
            loss = (
                train_config.classification_weight * classification
                + train_config.regression_weight * regression
                + train_config.uncertainty_weight * uncertainty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return model, history


def predict_geopath_model(
    model: GeoPathResidualModel,
    contract: GeoPathTensorContract,
    selection: np.ndarray,
    *,
    arm: Literal["trajectory_tube", "geopath_full"],
    device: str = "auto",
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    _require_torch()
    resolved_device = resolve_device(device)
    index = np.flatnonzero(np.asarray(selection, dtype=bool))
    point_view = _arm_point_features(contract.point_features, arm)
    output_parts: dict[str, list[np.ndarray]] = {
        "candidate_logits": [],
        "candidate_probability": [],
        "expected_candidate_shift_ms": [],
        "residual_ms": [],
        "predicted_shift_ms": [],
        "sigma_ms": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, index.size, batch_size):
            batch = index[start : start + batch_size]
            prediction = model(
                torch.as_tensor(
                    point_view[batch], dtype=torch.float32, device=resolved_device
                ),
                torch.as_tensor(
                    contract.candidate_features[batch],
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                torch.as_tensor(
                    contract.candidate_shift_ms[batch],
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                torch.as_tensor(
                    contract.candidate_valid_mask[batch],
                    dtype=torch.bool,
                    device=resolved_device,
                ),
            )
            for name, parts in output_parts.items():
                parts.append(prediction[name].detach().cpu().numpy())
    result = {
        name: np.concatenate(parts, axis=0) for name, parts in output_parts.items()
    }
    probability = result["candidate_probability"]
    entropy = -np.sum(probability * np.log(np.clip(probability, 1.0e-8, 1.0)), axis=1)
    entropy /= math.log(max(contract.candidate_count, 2))
    result["normalized_entropy"] = entropy.astype(np.float32)
    result["accepted"] = (
        (result["sigma_ms"] <= 80.0)
        & (np.max(probability, axis=1) >= 0.30)
        & (entropy <= 0.92)
    )
    result["indices"] = index
    return result


def regularize_geopath_full_paths(
    contract: GeoPathTensorContract,
    prediction: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply the deterministic trajectory-wide path solver to full-arm logits.

    OOF grouping guarantees an entire well lies in one held-out fold.  The
    learned residual is retained after replacing the pointwise expected
    candidate by the physically consistent selected path.
    """

    index = np.asarray(prediction["indices"], dtype=int)
    result = {name: np.asarray(values).copy() for name, values in prediction.items()}
    predicted_shift = np.asarray(result["predicted_shift_ms"], dtype=float)
    expected_shift = np.asarray(result["expected_candidate_shift_ms"], dtype=float)
    residual = predicted_shift - expected_shift
    sigma = np.asarray(result["sigma_ms"], dtype=float)
    accepted = np.asarray(result["accepted"], dtype=bool)
    path_applied = np.zeros(index.size, dtype=bool)
    indexed_wells = np.asarray(contract.well_id, dtype=str)[index]
    kappa_column = POINT_FEATURE_NAMES.index("kappa")
    for well in np.unique(indexed_wells):
        local = np.flatnonzero(indexed_wells == well)
        global_index = index[local]
        order = np.argsort(contract.md_m[global_index], kind="stable")
        ordered_local = local[order]
        ordered_global = global_index[order]
        coordinate = np.asarray(contract.md_m[ordered_global], dtype=float)
        if coordinate.size > 1 and not np.all(np.diff(coordinate) > 0.0):
            coordinate = np.arange(coordinate.size, dtype=float)
        logits = np.asarray(result["candidate_logits"])[ordered_local].copy()
        valid = np.asarray(contract.candidate_valid_mask[ordered_global], dtype=bool)
        logits[~valid] = -1.0e6
        prior = np.asarray(
            contract.p13_twt_ms[ordered_global]
            - contract.reference_twt_ms[ordered_global],
            dtype=float,
        )
        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        path = solve_geopath_path(
            logits,
            contract.candidate_shift_ms[ordered_global],
            kappa=contract.point_features[ordered_global, kappa_column],
            prior_shift_ms=prior,
            coordinate_m=coordinate,
        )
        if not path.accepted or path.selected_shift_ms is None:
            accepted[local] = False
            continue
        selected = np.asarray(path.selected_shift_ms, dtype=float)
        predicted_shift[ordered_local] = selected + residual[ordered_local]
        if path.posterior_std_ms is not None:
            sigma[ordered_local] = np.hypot(
                sigma[ordered_local], np.asarray(path.posterior_std_ms, dtype=float)
            )
        path_applied[local] = True
    result["predicted_shift_ms"] = predicted_shift.astype(np.float32)
    result["sigma_ms"] = sigma.astype(np.float32)
    # The path posterior broadens sigma. Re-apply the fixed uncertainty ceiling
    # after that update so a point cannot remain accepted solely because its
    # pre-path point estimate was over-confident.
    result["accepted"] = accepted & path_applied & (sigma <= 80.0)
    result["path_applied"] = path_applied
    return result


def apply_geopath_acceptance_fallback(
    reference_twt_ms: np.ndarray,
    p13_twt_ms: np.ndarray,
    predicted_shift_ms: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and deployable TWT tracks without using target labels.

    GeoPath is a residual challenger around a sealed prior. Where its fixed
    uncertainty, ambiguity, or path gate rejects the learned correction, the
    deployable result retains that prior instead of emitting an explicitly
    rejected correction. The raw track is returned separately for diagnosis.
    """

    reference = np.asarray(reference_twt_ms, dtype=float)
    p13 = np.asarray(p13_twt_ms, dtype=float)
    shift = np.asarray(predicted_shift_ms, dtype=float)
    use_model = np.asarray(accepted, dtype=bool)
    if not (reference.shape == p13.shape == shift.shape == use_model.shape):
        raise ValueError("GeoPath fallback arrays must have identical shapes")
    raw = reference + shift
    prior = np.where(np.isfinite(p13), p13, reference)
    final = np.where(use_model & np.isfinite(raw), raw, prior)
    return raw.astype(np.float32), final.astype(np.float32)


def save_checkpoint(
    path: str | Path,
    model: GeoPathResidualModel,
    *,
    arm: str,
    fold: int,
    model_config: GeoPathModelConfig,
    train_config: GeoPathTrainConfig,
    history: Sequence[Mapping[str, float]],
) -> Path:
    _require_torch()
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": GEOPATH_CHECKPOINT_SCHEMA,
            "arm": arm,
            "fold": int(fold),
            "selected_epoch_policy": "fixed_final_epoch",
            "selected_epoch": int(train_config.epochs),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "point_feature_names": list(POINT_FEATURE_NAMES),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "model_state_dict": model.state_dict(),
            "history": list(history),
        },
        output,
    )
    return output


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> tuple[GeoPathResidualModel, dict[str, Any]]:
    _require_torch()
    resolved_device = resolve_device(device)
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location=resolved_device, weights_only=False
    )
    if payload.get("schema_version") != GEOPATH_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported GeoPath checkpoint schema")
    if tuple(payload.get("point_feature_names", ())) != POINT_FEATURE_NAMES:
        raise ValueError("checkpoint point feature order is incompatible")
    if tuple(payload.get("candidate_feature_names", ())) != CANDIDATE_FEATURE_NAMES:
        raise ValueError("checkpoint candidate feature order is incompatible")
    state = payload["model_state_dict"]
    model = GeoPathResidualModel(
        GeoPathModelConfig(**payload["model_config"]),
        point_location=state["point_location"].cpu().numpy(),
        point_scale=state["point_scale"].cpu().numpy(),
        candidate_location=state["candidate_location"].cpu().numpy(),
        candidate_scale=state["candidate_scale"].cpu().numpy(),
    ).to(resolved_device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _prediction_metrics(
    contract: GeoPathTensorContract,
    predicted_twt_ms: np.ndarray,
    *,
    sigma_ms: np.ndarray | None = None,
    accepted: np.ndarray | None = None,
    selection: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(contract.target_twt_ms, dtype=float)
    predicted = np.asarray(predicted_twt_ms, dtype=float)
    mask = np.asarray(contract.supervision_mask, dtype=bool) & np.isfinite(predicted)
    if selection is not None:
        mask &= np.asarray(selection, dtype=bool)
    if not np.any(mask):
        return {"point_count": 0, "well_count": 0}
    error = predicted - target
    absolute = np.abs(error)
    wells = np.asarray(contract.well_id, dtype=str)
    well_mae = [float(np.mean(absolute[mask & (wells == well)])) for well in np.unique(wells[mask])]
    result: dict[str, Any] = {
        "point_count": int(np.count_nonzero(mask)),
        "well_count": len(well_mae),
        "mae_ms": float(np.mean(absolute[mask])),
        "median_ae_ms": float(np.median(absolute[mask])),
        "p90_ae_ms": float(np.quantile(absolute[mask], 0.90)),
        "well_macro_mae_ms": float(np.mean(well_mae)),
        "well_median_mae_ms": float(np.median(well_mae)),
        "well_p90_mae_ms": float(np.quantile(well_mae, 0.90)),
        "cycle_jump_rate": _path_cycle_jump_rate(
            contract, predicted, selection=selection
        ),
    }
    if sigma_ms is not None:
        sigma = np.asarray(sigma_ms, dtype=float)
        result["interval_90_coverage"] = float(
            np.mean(absolute[mask] <= 1.6448536269514722 * sigma[mask])
        )
        result["median_sigma_ms"] = float(np.median(sigma[mask]))
    if accepted is not None:
        accept = np.asarray(accepted, dtype=bool)
        result["accepted_coverage"] = float(np.mean(accept[mask]))
        accepted_mask = mask & accept
        result["accepted_well_macro_mae_ms"] = (
            float(
                np.mean(
                    [
                        np.mean(absolute[accepted_mask & (wells == well)])
                        for well in np.unique(wells[accepted_mask])
                    ]
                )
            )
            if np.any(accepted_mask)
            else None
        )
    return result


def _path_cycle_jump_rate(
    contract: GeoPathTensorContract,
    predicted_twt_ms: np.ndarray,
    *,
    selection: np.ndarray | None = None,
) -> float | None:
    """Measure discontinuous correction jumps after removing the physical prior."""

    predicted = np.asarray(predicted_twt_ms, dtype=float)
    p13 = np.asarray(contract.p13_twt_ms, dtype=float)
    reference = np.asarray(contract.reference_twt_ms, dtype=float)
    prior = np.where(np.isfinite(p13), p13, reference)
    usable = np.isfinite(predicted) & np.isfinite(prior)
    if selection is not None:
        usable &= np.asarray(selection, dtype=bool)
    jump_count = 0
    transition_count = 0
    wells = np.asarray(contract.well_id, dtype=str)
    for well in np.unique(wells[usable]):
        indices = np.flatnonzero(usable & (wells == well))
        if indices.size < 2:
            continue
        indices = indices[np.argsort(contract.md_m[indices], kind="stable")]
        correction = predicted[indices] - prior[indices]
        jump = np.abs(np.diff(correction))
        period = 0.5 * (
            contract.dominant_period_ms[indices[:-1]]
            + contract.dominant_period_ms[indices[1:]]
        )
        jump_count += int(np.count_nonzero(jump > period))
        transition_count += int(jump.size)
    return None if transition_count == 0 else float(jump_count / transition_count)


def evaluate_predictions(
    contract: GeoPathTensorContract,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce overall, per-geometry, and per-well metrics for every arm."""

    contract.validate()
    geometry = np.asarray(contract.geometry, dtype=str)
    metrics: dict[str, Any] = {"schema_version": GEOPATH_OOF_SCHEMA, "arms": {}}
    per_well: list[dict[str, Any]] = []
    for arm, payload in predictions.items():
        predicted = np.asarray(payload["predicted_twt_ms"], dtype=float)
        sigma = payload.get("sigma_ms")
        accepted = payload.get("accepted")
        md_fraction = np.zeros(contract.point_count, dtype=float)
        for well in np.unique(contract.well_id):
            well_mask = np.asarray(contract.well_id) == well
            md_values = np.asarray(contract.md_m[well_mask], dtype=float)
            span = max(float(np.max(md_values) - np.min(md_values)), 1.0)
            md_fraction[well_mask] = (md_values - float(np.min(md_values))) / span
        arm_metrics: dict[str, Any] = {
            "overall": _prediction_metrics(
                contract, predicted, sigma_ms=sigma, accepted=accepted
            ),
            "by_geometry": {},
            "by_trajectory_segment": {},
        }
        for geometry_name in GEOMETRIES:
            arm_metrics["by_geometry"][geometry_name] = _prediction_metrics(
                contract,
                predicted,
                sigma_ms=sigma,
                accepted=accepted,
                selection=geometry == geometry_name,
            )
        segment_masks = {
            "heel": md_fraction < (1.0 / 3.0),
            "middle": (md_fraction >= (1.0 / 3.0)) & (md_fraction < (2.0 / 3.0)),
            "toe": md_fraction >= (2.0 / 3.0),
        }
        for segment_name, segment_mask in segment_masks.items():
            arm_metrics["by_trajectory_segment"][segment_name] = _prediction_metrics(
                contract,
                predicted,
                sigma_ms=sigma,
                accepted=accepted,
                selection=segment_mask,
            )
        metrics["arms"][arm] = arm_metrics
        drift_values: list[float] = []
        for well in np.unique(contract.well_id):
            selection = np.asarray(contract.well_id) == well
            well_metrics = _prediction_metrics(
                contract,
                predicted,
                sigma_ms=sigma,
                accepted=accepted,
                selection=selection,
            )
            md = contract.md_m[selection & contract.supervision_mask]
            residual = (
                predicted[selection & contract.supervision_mask]
                - contract.target_twt_ms[selection & contract.supervision_mask]
            )
            if md.size >= 2 and float(np.ptp(md)) >= 1.0:
                slope = np.polyfit(md, residual, deg=1)[0]
                drift_per_km = float(abs(slope) * 1000.0)
            else:
                drift_per_km = None
            if drift_per_km is not None:
                drift_values.append(drift_per_km)
            per_well.append(
                {
                    "arm": arm,
                    "well_id": str(well),
                    "family_id": str(np.asarray(contract.family_id)[selection][0]),
                    "geometry": str(np.asarray(contract.geometry)[selection][0]),
                    **well_metrics,
                    "absolute_time_drift_ms_per_km": drift_per_km,
                }
            )
        arm_metrics["time_drift_ms_per_km"] = {
            "well_count": len(drift_values),
            "median": float(np.median(drift_values)) if drift_values else None,
            "p90": float(np.quantile(drift_values, 0.90)) if drift_values else None,
            "maximum": float(np.max(drift_values)) if drift_values else None,
        }
    metrics["relative_gains"] = _relative_gains(metrics)
    metrics["promotion"] = promotion_decision(metrics)
    return metrics, per_well


def _gain(baseline: Mapping[str, Any], challenger: Mapping[str, Any]) -> float | None:
    baseline_mae = baseline.get("well_macro_mae_ms")
    challenger_mae = challenger.get("well_macro_mae_ms")
    if baseline_mae is None or challenger_mae is None or float(baseline_mae) <= 0.0:
        return None
    return float((float(baseline_mae) - float(challenger_mae)) / float(baseline_mae))


def _relative_gains(metrics: Mapping[str, Any]) -> dict[str, Any]:
    arms = metrics.get("arms", {})
    result: dict[str, Any] = {}
    for challenger_name in MODEL_ARMS:
        if challenger_name not in arms:
            continue
        result[challenger_name] = {}
        for baseline_name in ("sonic_prior", "p13_baseline"):
            if baseline_name not in arms:
                continue
            comparison: dict[str, Any] = {
                "overall": _gain(
                    arms[baseline_name]["overall"],
                    arms[challenger_name]["overall"],
                ),
                "by_geometry": {},
            }
            for geometry in GEOMETRIES:
                comparison["by_geometry"][geometry] = _gain(
                    arms[baseline_name]["by_geometry"][geometry],
                    arms[challenger_name]["by_geometry"][geometry],
                )
            result[challenger_name][f"vs_{baseline_name}"] = comparison
    return result


def promotion_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the predeclared Chengdu gates without hiding missing evidence."""

    gains = metrics.get("relative_gains", {}).get("geopath_full", {}).get(
        "vs_p13_baseline", {}
    )
    overall_gain = gains.get("overall")
    geometry_gains = gains.get("by_geometry", {})
    deviated_gain = geometry_gains.get("deviated")
    horizontal_gain = geometry_gains.get("horizontal")
    full = metrics.get("arms", {}).get("geopath_full", {}).get("overall", {})
    geometry_pass = (
        deviated_gain is not None
        and horizontal_gain is not None
        and (
            (deviated_gain >= 0.10 and horizontal_gain >= -0.05)
            or (horizontal_gain >= 0.10 and deviated_gain >= -0.05)
        )
    )
    gates = {
        "overall_vs_p13_not_worse_than_2pct": {
            "passed": overall_gain is not None and overall_gain >= -0.02,
            "value": overall_gain,
            "threshold": -0.02,
        },
        "deviated_or_horizontal_gain_10pct_other_not_worse_5pct": {
            "passed": geometry_pass,
            "deviated_gain": deviated_gain,
            "horizontal_gain": horizontal_gain,
        },
        "cycle_jump_rate_below_2pct": {
            "passed": full.get("cycle_jump_rate") is not None
            and float(full["cycle_jump_rate"]) < 0.02,
            "value": full.get("cycle_jump_rate"),
            "threshold": 0.02,
        },
        "accepted_coverage_at_least_60pct": {
            "passed": full.get("accepted_coverage") is not None
            and float(full["accepted_coverage"]) >= 0.60,
            "value": full.get("accepted_coverage"),
            "threshold": 0.60,
        },
        "f3_external_regression": {
            "passed": None,
            "status": "not_evaluated_in_chengdu_oof",
        },
    }
    evaluated = [item["passed"] for item in gates.values() if item["passed"] is not None]
    return {
        "passed_chengdu_gates": bool(evaluated) and all(evaluated),
        "production_promotion_ready": False,
        "production_blocker": "F3 external regression has not been evaluated",
        "gates": gates,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    seen = set(columns)
    for row in rows[1:]:
        for name in row:
            if name not in seen:
                columns.append(name)
                seen.add(name)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_well_predictions(
    output: Path,
    contract: GeoPathTensorContract,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    point_folds: np.ndarray,
) -> None:
    wells = np.asarray(contract.well_id, dtype=str)
    for well in np.unique(wells):
        selection = wells == well
        indices = np.flatnonzero(selection)
        order = np.argsort(contract.md_m[indices], kind="stable")
        indices = indices[order]
        well_root = output / "wells" / str(well)
        well_root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "md_m": np.asarray(contract.md_m[indices], dtype=np.float32),
            "target_twt_ms": np.asarray(
                contract.target_twt_ms[indices], dtype=np.float32
            ),
            "supervision_mask": np.asarray(
                contract.supervision_mask[indices], dtype=np.uint8
            ),
            "reference_twt_ms": np.asarray(
                contract.reference_twt_ms[indices], dtype=np.float32
            ),
            "fold": np.asarray(point_folds[indices], dtype=np.int16),
        }
        for arm, payload in predictions.items():
            arrays[f"{arm}__predicted_twt_ms"] = np.asarray(
                payload["predicted_twt_ms"][indices], dtype=np.float32
            )
            if "raw_predicted_twt_ms" in payload:
                arrays[f"{arm}__raw_predicted_twt_ms"] = np.asarray(
                    payload["raw_predicted_twt_ms"][indices], dtype=np.float32
                )
            if "sigma_ms" in payload:
                arrays[f"{arm}__sigma_ms"] = np.asarray(
                    payload["sigma_ms"][indices], dtype=np.float32
                )
            if "accepted" in payload:
                arrays[f"{arm}__accepted"] = np.asarray(
                    payload["accepted"][indices], dtype=np.uint8
                )
        np.savez_compressed(
            well_root / "prediction.npz",
            schema_version=np.asarray(GEOPATH_OOF_SCHEMA),
            well_id=np.asarray(str(well)),
            family_id=np.asarray(str(contract.family_id[indices[0]])),
            geometry=np.asarray(str(contract.geometry[indices[0]])),
            **arrays,
        )
        rows: list[dict[str, Any]] = []
        for local, global_index in enumerate(indices):
            row: dict[str, Any] = {
                "well_id": str(well),
                "family_id": str(contract.family_id[global_index]),
                "geometry": str(contract.geometry[global_index]),
                "fold": int(point_folds[global_index]),
                "md_m": float(contract.md_m[global_index]),
                "target_twt_ms": (
                    float(contract.target_twt_ms[global_index])
                    if contract.supervision_mask[global_index]
                    else ""
                ),
                "supervision_mask": int(contract.supervision_mask[global_index]),
                "reference_twt_ms": float(contract.reference_twt_ms[global_index]),
            }
            for arm, payload in predictions.items():
                row[f"{arm}__predicted_twt_ms"] = float(
                    payload["predicted_twt_ms"][global_index]
                )
                if "raw_predicted_twt_ms" in payload:
                    row[f"{arm}__raw_predicted_twt_ms"] = float(
                        payload["raw_predicted_twt_ms"][global_index]
                    )
                if "sigma_ms" in payload:
                    row[f"{arm}__sigma_ms"] = float(payload["sigma_ms"][global_index])
                if "accepted" in payload:
                    row[f"{arm}__accepted"] = int(payload["accepted"][global_index])
            rows.append(row)
        _write_csv(well_root / "prediction.csv", rows)


def _display_metric(
    metrics: Mapping[str, Any], name: str, *, percent: bool = False
) -> str:
    value = metrics.get(name)
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%" if percent else f"{float(value):.3f}"


def write_report(path: str | Path, metrics: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    lines = [
        "# GeoPathTie-V1 OOF Report",
        "",
        "One shared residual model was used for vertical, deviated and horizontal wells.",
        "",
        "| Arm | Well-macro MAE (ms) | Median AE (ms) | P90 AE (ms) | Cycle jump | Accepted | 90% interval coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in REPORT_ARMS:
        overall = metrics.get("arms", {}).get(arm, {}).get("overall", {})
        lines.append(
            f"| {arm} | {_display_metric(overall, 'well_macro_mae_ms')} | "
            f"{_display_metric(overall, 'median_ae_ms')} | "
            f"{_display_metric(overall, 'p90_ae_ms')} | "
            f"{_display_metric(overall, 'cycle_jump_rate', percent=True)} | "
            f"{_display_metric(overall, 'accepted_coverage', percent=True)} | "
            f"{_display_metric(overall, 'interval_90_coverage', percent=True)} |"
        )
    lines.extend(["", "## Geometry", ""])
    for geometry in GEOMETRIES:
        lines.extend(
            [
                f"### {geometry}",
                "",
                "| Arm | Well count | Well-macro MAE (ms) | Median AE (ms) | P90 AE (ms) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for arm in REPORT_ARMS:
            item = (
                metrics.get("arms", {})
                .get(arm, {})
                .get("by_geometry", {})
                .get(geometry, {})
            )
            lines.append(
                f"| {arm} | {item.get('well_count', 0)} | "
                f"{item.get('well_macro_mae_ms', 'n/a')} | "
                f"{item.get('median_ae_ms', 'n/a')} | {item.get('p90_ae_ms', 'n/a')} |"
            )
        lines.append("")
    full = metrics.get("arms", {}).get("geopath_full", {})
    lines.extend(
        [
            "## Trajectory segments (GeoPath full)",
            "",
            "| Segment | Well-macro MAE (ms) | Median AE (ms) | P90 AE (ms) | Accepted |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for segment in ("heel", "middle", "toe"):
        item = full.get("by_trajectory_segment", {}).get(segment, {})
        lines.append(
            f"| {segment} | {_display_metric(item, 'well_macro_mae_ms')} | "
            f"{_display_metric(item, 'median_ae_ms')} | "
            f"{_display_metric(item, 'p90_ae_ms')} | "
            f"{_display_metric(item, 'accepted_coverage', percent=True)} |"
        )
    drift = full.get("time_drift_ms_per_km", {})
    lines.extend(
        [
            "",
            "## Time drift and uncertainty",
            "",
            (
                f"- Absolute time-drift median/P90/max: "
                f"{_display_metric(drift, 'median')} / "
                f"{_display_metric(drift, 'p90')} / "
                f"{_display_metric(drift, 'maximum')} ms/km."
            ),
            (
                f"- Nominal 90% interval empirical coverage: "
                f"{_display_metric(full.get('overall', {}), 'interval_90_coverage', percent=True)}."
            ),
            "- Sigma remains uncalibrated until an independent calibration set is available.",
            "",
        ]
    )
    lines.extend(["## Promotion gates", ""])
    promotion = metrics.get("promotion", {})
    for name, gate in promotion.get("gates", {}).items():
        status = gate.get("passed")
        label = "PASS" if status is True else "FAIL" if status is False else "NOT EVALUATED"
        lines.append(f"- `{name}`: {label}")
    lines.extend(
        [
            "",
            f"Chengdu gates passed: `{promotion.get('passed_chengdu_gates', False)}`.",
            f"Production promotion ready: `{promotion.get('production_promotion_ready', False)}`.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_grouped_oof(
    contract: GeoPathTensorContract,
    output_root: str | Path,
    *,
    n_splits: int = 5,
    model_config: GeoPathModelConfig = DEFAULT_MODEL_CONFIG,
    train_config: GeoPathTrainConfig = DEFAULT_TRAIN_CONFIG,
    device: str = "auto",
    point_folds: np.ndarray | None = None,
    fold_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train tube/full arms on identical family folds and write OOF artifacts."""

    _require_torch()
    contract.validate()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if point_folds is None:
        resolved_folds, resolved_manifest = build_group_folds(
            contract, n_splits=n_splits, seed=train_config.seed
        )
    else:
        resolved_folds = np.asarray(point_folds, dtype=np.int16)
        if resolved_folds.shape != (contract.point_count,):
            raise ValueError("point_folds must match the GeoPath point count")
        supervised_folds = resolved_folds[np.asarray(contract.supervision_mask, dtype=bool)]
        if np.any(supervised_folds < 0) or set(np.unique(supervised_folds)) != set(
            range(n_splits)
        ):
            raise ValueError("fixed point_folds must assign every supervised point")
        resolved_manifest = list(fold_manifest or [])
    _write_csv(output / "fold_manifest.csv", resolved_manifest)
    n_points = contract.point_count
    predictions: dict[str, dict[str, np.ndarray]] = {
        "sonic_prior": {
            "predicted_twt_ms": np.asarray(contract.sonic_twt_ms, dtype=np.float32),
            "accepted": np.isfinite(contract.sonic_twt_ms),
        },
        "p13_baseline": {
            "predicted_twt_ms": np.asarray(contract.p13_twt_ms, dtype=np.float32),
            "accepted": np.isfinite(contract.p13_twt_ms),
        },
    }
    histories: dict[str, Any] = {}
    for arm in MODEL_ARMS:
        predictions[arm] = {
            "predicted_twt_ms": np.full(n_points, np.nan, dtype=np.float32),
            "raw_predicted_twt_ms": np.full(n_points, np.nan, dtype=np.float32),
            "sigma_ms": np.full(n_points, np.nan, dtype=np.float32),
            "accepted": np.zeros(n_points, dtype=bool),
            "normalized_entropy": np.full(n_points, np.nan, dtype=np.float32),
        }
        for fold in range(n_splits):
            test_mask = resolved_folds == fold
            train_mask = (resolved_folds >= 0) & ~test_mask
            fold_train_config = replace(
                train_config, seed=train_config.seed + 1009 * fold
            )
            model, history = fit_geopath_model(
                contract,
                train_mask,
                arm=arm,
                model_config=model_config,
                train_config=fold_train_config,
                device=device,
            )
            checkpoint = output / "checkpoints" / arm / f"fold_{fold}.pt"
            save_checkpoint(
                checkpoint,
                model,
                arm=arm,
                fold=fold,
                model_config=model_config,
                train_config=fold_train_config,
                history=history,
            )
            fold_prediction = predict_geopath_model(
                model, contract, test_mask, arm=arm, device=device
            )
            if arm == "geopath_full":
                fold_prediction = regularize_geopath_full_paths(
                    contract, fold_prediction
                )
            index = fold_prediction["indices"]
            raw_twt, final_twt = apply_geopath_acceptance_fallback(
                contract.reference_twt_ms[index],
                contract.p13_twt_ms[index],
                fold_prediction["predicted_shift_ms"],
                fold_prediction["accepted"],
            )
            predictions[arm]["raw_predicted_twt_ms"][index] = raw_twt
            predictions[arm]["predicted_twt_ms"][index] = final_twt
            for name in ("sigma_ms", "accepted", "normalized_entropy"):
                predictions[arm][name][index] = fold_prediction[name]
            histories[f"{arm}/fold_{fold}"] = history

        final_train_config = replace(train_config, seed=train_config.seed + 7919)
        final_model, final_history = fit_geopath_model(
            contract,
            np.asarray(contract.supervision_mask, dtype=bool),
            arm=arm,
            model_config=model_config,
            train_config=final_train_config,
            device=device,
        )
        save_checkpoint(
            output / "checkpoints" / arm / "final_all.pt",
            final_model,
            arm=arm,
            fold=-1,
            model_config=model_config,
            train_config=final_train_config,
            history=final_history,
        )
        histories[f"{arm}/final_all"] = final_history

    metrics, per_well = evaluate_predictions(contract, predictions)
    metrics["fold_count"] = int(n_splits)
    metrics["point_count"] = int(n_points)
    metrics["supervised_point_count"] = int(
        np.count_nonzero(contract.supervision_mask)
    )
    metrics["well_count"] = int(np.unique(contract.well_id).size)
    metrics["family_count"] = int(np.unique(contract.family_id).size)
    metrics["device"] = resolve_device(device)
    metrics["model_config"] = asdict(model_config)
    metrics["train_config"] = asdict(train_config)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "training_history.json", histories)
    _write_csv(output / "per_well_metrics.csv", per_well)
    write_report(output / "REPORT.md", metrics)
    _write_per_well_predictions(output, contract, predictions, resolved_folds)
    np.savez_compressed(
        output / "oof_predictions.npz",
        schema_version=np.asarray(GEOPATH_OOF_SCHEMA),
        fold=resolved_folds,
        well_id=np.asarray(contract.well_id, dtype=str),
        target_twt_ms=np.asarray(contract.target_twt_ms, dtype=np.float32),
        **{
            f"{arm}__{name}": np.asarray(values)
            for arm, payload in predictions.items()
            for name, values in payload.items()
        },
    )
    return metrics


def evaluate_external_contract(
    contract: GeoPathTensorContract,
    run_root: str | Path,
    *,
    output_root: str | Path,
    dataset_name: str,
    device: str = "auto",
) -> dict[str, Any]:
    """Run frozen final-all checkpoints on a second contract (for example F3)."""

    contract.validate()
    run = Path(run_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    n_points = contract.point_count
    predictions: dict[str, dict[str, np.ndarray]] = {
        "sonic_prior": {
            "predicted_twt_ms": np.asarray(contract.sonic_twt_ms, dtype=np.float32),
            "accepted": np.isfinite(contract.sonic_twt_ms),
        },
        "p13_baseline": {
            "predicted_twt_ms": np.asarray(contract.p13_twt_ms, dtype=np.float32),
            "accepted": np.isfinite(contract.p13_twt_ms),
        },
    }
    selection = np.ones(n_points, dtype=bool)
    for arm in MODEL_ARMS:
        model, checkpoint = load_checkpoint(
            run / "checkpoints" / arm / "final_all.pt", device=device
        )
        if checkpoint.get("selected_epoch_policy") != "fixed_final_epoch":
            raise ValueError("external evaluation requires a fixed-final checkpoint")
        prediction = predict_geopath_model(
            model, contract, selection, arm=arm, device=device
        )
        if arm == "geopath_full":
            prediction = regularize_geopath_full_paths(contract, prediction)
        raw_twt, final_twt = apply_geopath_acceptance_fallback(
            contract.reference_twt_ms,
            contract.p13_twt_ms,
            prediction["predicted_shift_ms"],
            prediction["accepted"],
        )
        predictions[arm] = {
            "predicted_twt_ms": final_twt,
            "raw_predicted_twt_ms": raw_twt,
            "sigma_ms": prediction["sigma_ms"],
            "accepted": prediction["accepted"],
            "normalized_entropy": prediction["normalized_entropy"],
        }
    metrics, per_well = evaluate_predictions(contract, predictions)
    metrics["dataset_name"] = dataset_name
    metrics["evaluation_kind"] = "external_final_checkpoint_regression"
    metrics["blind_status"] = "non_blind_regression"
    _write_json(output / "metrics.json", metrics)
    _write_csv(output / "per_well_metrics.csv", per_well)
    write_report(output / "REPORT.md", metrics)
    np.savez_compressed(
        output / "predictions.npz",
        schema_version=np.asarray(GEOPATH_OOF_SCHEMA),
        well_id=np.asarray(contract.well_id, dtype=str),
        target_twt_ms=np.asarray(contract.target_twt_ms, dtype=np.float32),
        **{
            f"{arm}__{name}": np.asarray(values)
            for arm, payload in predictions.items()
            for name, values in payload.items()
        },
    )
    return metrics


def attach_external_regression(
    training_metrics: dict[str, Any],
    external_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the non-blind F3 gate and update promotion readiness."""

    result = json.loads(json.dumps(training_metrics, ensure_ascii=False))
    gain = (
        external_metrics.get("relative_gains", {})
        .get("geopath_full", {})
        .get("vs_sonic_prior", {})
        .get("overall")
    )
    passed = gain is not None and float(gain) >= -0.05
    result["external_regression"] = {
        "dataset_name": external_metrics.get("dataset_name"),
        "blind_status": external_metrics.get("blind_status", "non_blind_regression"),
        "geopath_full_gain_vs_sonic": gain,
        "passed_not_worse_than_sonic_5pct": passed,
    }
    promotion = result.setdefault("promotion", {})
    gates = promotion.setdefault("gates", {})
    gates["f3_external_regression"] = {
        "passed": passed,
        "value": gain,
        "threshold": -0.05,
        "status": "non_blind_regression",
    }
    promotion["production_promotion_ready"] = bool(
        promotion.get("passed_chengdu_gates", False) and passed
    )
    promotion["production_blocker"] = (
        None
        if promotion["production_promotion_ready"]
        else "one or more Chengdu/F3 promotion gates failed"
    )
    return result


def load_oof_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        if str(payload["schema_version"]) != GEOPATH_OOF_SCHEMA:
            raise ValueError("unsupported GeoPath OOF schema")
        result: dict[str, dict[str, np.ndarray]] = {}
        for name in payload.files:
            if "__" not in name:
                continue
            arm, field = name.split("__", 1)
            result.setdefault(arm, {})[field] = payload[name]
    return result


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "GEOMETRIES",
    "GEOPATH_CHECKPOINT_SCHEMA",
    "GEOPATH_DATASET_SCHEMA",
    "GEOPATH_OOF_SCHEMA",
    "MODEL_ARMS",
    "POINT_FEATURE_NAMES",
    "REPORT_ARMS",
    "GeoPathModelConfig",
    "GeoPathResidualModel",
    "GeoPathTensorContract",
    "GeoPathTrainConfig",
    "apply_geopath_acceptance_fallback",
    "attach_external_regression",
    "balanced_sample_weights",
    "build_group_folds",
    "concatenate_contracts",
    "contract_from_geopath_outputs",
    "evaluate_external_contract",
    "evaluate_predictions",
    "fit_geopath_model",
    "load_checkpoint",
    "load_fixed_fold_assignment",
    "load_oof_predictions",
    "predict_geopath_model",
    "regularize_geopath_full_paths",
    "resolve_device",
    "run_grouped_oof",
    "save_checkpoint",
    "write_report",
]
