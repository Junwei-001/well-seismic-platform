"""Thin Chengdu asset adapter for GeoPathTie-V1.

The adapter deliberately reuses the existing immutable WellFuse caches instead
of rebuilding the 263 GB source survey.  It joins four inputs needed by the
GeoPathTie runner:

* 118 cached LAS/trajectory/trajectory-seismic inputs;
* 92 cached SMI time-depth targets;
* the 92-well P13 out-of-fold prior; and
* the frozen Chengdu sonic/current-static predictions.

The split contract is a separate checked-in JSON document.  Families, rather
than individual borehole names, are the validation unit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Geometry = Literal["vertical", "deviated", "horizontal"]
FoldRole = Literal["train", "validation", "unlabeled_train"]

PLATFORM_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WELLFUSE_ROOT = PLATFORM_ROOT.parent / "wellfuse"
DEFAULT_FOLD_CONTRACT = PLATFORM_ROOT / "configs/geopath_tie_v1_chengdu_folds.json"
DEFAULT_STATIC_GATE_ROOT = (
    PLATFORM_ROOT / "model_outputs/chengdu_static_shift_gate_eval_20260819_v2"
)

_SIDETRACK_FAMILY = re.compile(r"^((?:EBSH\d*|EBSK|EBST)-\d+)-", re.IGNORECASE)


def mother_well_family(well_id: str) -> str:
    """Return the fixed Chengdu mother/pad family for split isolation."""

    value = re.sub(r"_Pilot$", "", str(well_id).strip(), flags=re.IGNORECASE)
    match = _SIDETRACK_FAMILY.match(value)
    return (match.group(1) if match else value).upper()


def geometry_class(maximum_inclination_deg: float, maximum_lateral_offset_m: float) -> Geometry:
    """Use the geometry definition already used by the 92-well Chengdu model."""

    if float(maximum_inclination_deg) >= 60.0:
        return "horizontal"
    if float(maximum_inclination_deg) >= 10.0 or float(maximum_lateral_offset_m) > 50.0:
        return "deviated"
    return "vertical"


@dataclass(frozen=True)
class GeoPathWellRecord:
    well_id: str
    family_id: str
    geometry: Geometry
    supervised: bool
    surface_x_m: float
    surface_y_m: float
    input_cache: Path
    label_cache: Path | None
    p13_prior: Path | None
    sonic_gate_prediction: Path | None


@dataclass(frozen=True)
class GeoPathFold:
    fold_id: str
    train_wells: tuple[str, ...]
    validation_wells: tuple[str, ...]
    unlabeled_train_wells: tuple[str, ...]


@dataclass(frozen=True)
class GeoPathWellArrays:
    well_id: str
    family_id: str
    geometry: Geometry
    md_m: np.ndarray
    curve_values: np.ndarray
    curve_masks: np.ndarray
    seismic: np.ndarray
    seismic_time_ms: np.ndarray
    trajectory_features: np.ndarray
    trajectory_mask: np.ndarray
    trajectory_xy_m: np.ndarray
    trajectory_seismic: np.ndarray
    target_twt_ms: np.ndarray
    supervision_mask: np.ndarray
    p13_twt_ms: np.ndarray
    p13_std_ms: np.ndarray
    p13_confidence: np.ndarray
    sonic_twt_ms: np.ndarray
    current_gated_twt_ms: np.ndarray
    input_metadata: dict[str, Any]


class ChengduGeoPathCatalog:
    """Training-facing reader over the existing Chengdu caches and fixed folds."""

    def __init__(
        self,
        records: dict[str, GeoPathWellRecord],
        folds: tuple[GeoPathFold, ...],
        *,
        seismic_path: Path,
        source_manifest_path: Path,
        fold_contract_path: Path,
    ) -> None:
        self.records = records
        self.folds = folds
        self.seismic_path = seismic_path
        self.source_manifest_path = source_manifest_path
        self.fold_contract_path = fold_contract_path
        self._validate_split()

    @property
    def well_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))

    @property
    def supervised_well_ids(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, item in self.records.items() if item.supervised))

    @property
    def unlabeled_well_ids(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, item in self.records.items() if not item.supervised))

    def fold(self, fold_id: str) -> GeoPathFold:
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold
        raise KeyError(f"unknown GeoPathTie fold: {fold_id}")

    def wells_for_fold(self, fold_id: str, role: FoldRole) -> tuple[str, ...]:
        fold = self.fold(fold_id)
        if role == "train":
            return fold.train_wells
        if role == "validation":
            return fold.validation_wells
        if role == "unlabeled_train":
            return fold.unlabeled_train_wells
        raise ValueError(f"unsupported GeoPathTie fold role: {role}")

    def geometry_counts(self, well_ids: tuple[str, ...] | list[str]) -> dict[str, int]:
        return {
            geometry: sum(self.records[well_id].geometry == geometry for well_id in well_ids)
            for geometry in ("vertical", "deviated", "horizontal")
        }

    def load_well(self, well_id: str) -> GeoPathWellArrays:
        record = self.records[well_id]
        with np.load(record.input_cache, allow_pickle=False) as payload:
            md_m = np.asarray(payload["depth_axis_m"], dtype=np.float64)
            curve_values = np.asarray(payload["curve_values"], dtype=np.float32)
            curve_masks = np.asarray(payload["curve_masks"], dtype=np.float32)
            seismic = np.asarray(payload["seismic"], dtype=np.float32)
            seismic_time_ms = np.asarray(payload["seismic_time_ms"], dtype=np.float64)
            trajectory_features = np.asarray(payload["trajectory_features"], dtype=np.float32)
            trajectory_mask = np.asarray(payload["trajectory_mask"], dtype=np.float32)
            trajectory_xy_m = np.asarray(payload["trajectory_xy"], dtype=np.float64)
            trajectory_seismic = np.asarray(payload["trajectory_seismic"], dtype=np.float32)
            input_metadata = json.loads(str(payload["metadata"].item()))

        length = len(md_m)
        target_twt_ms = np.full(length, np.nan, dtype=np.float64)
        supervision_mask = np.zeros(length, dtype=np.bool_)
        if record.label_cache is not None:
            with np.load(record.label_cache, allow_pickle=False) as payload:
                target_twt_ms = np.asarray(payload["target_time_ms"], dtype=np.float64)
                supervision_mask = np.asarray(payload["valid_target"] > 0.5, dtype=np.bool_)

        p13_twt_ms = np.full(length, np.nan, dtype=np.float64)
        p13_std_ms = np.full(length, np.nan, dtype=np.float32)
        p13_confidence = np.zeros(length, dtype=np.float32)
        if record.p13_prior is not None:
            with np.load(record.p13_prior, allow_pickle=False) as payload:
                p13_md = np.asarray(payload["md"], dtype=np.float64)
                p13_twt_ms = _interpolate_inside(
                    md_m, p13_md, np.asarray(payload["twt_mean_ms"], dtype=np.float64)
                )
                p13_std_ms = _interpolate_inside(
                    md_m, p13_md, np.asarray(payload["twt_std_ms"], dtype=np.float64)
                ).astype(np.float32)
                p13_confidence = _interpolate_inside(
                    md_m, p13_md, np.asarray(payload["confidence"], dtype=np.float64)
                ).astype(np.float32)

        sonic_twt_ms = np.full(length, np.nan, dtype=np.float64)
        current_gated_twt_ms = np.full(length, np.nan, dtype=np.float64)
        if record.sonic_gate_prediction is not None:
            with np.load(record.sonic_gate_prediction, allow_pickle=False) as payload:
                sonic_md = np.asarray(payload["md_m"], dtype=np.float64)
                sonic_twt_ms = _interpolate_inside(
                    md_m, sonic_md, np.asarray(payload["sonic_prior_twt_ms"], dtype=np.float64)
                )
                current_gated_twt_ms = _interpolate_inside(
                    md_m,
                    sonic_md,
                    np.asarray(payload["current_gated_twt_ms"], dtype=np.float64),
                )

        return GeoPathWellArrays(
            well_id=record.well_id,
            family_id=record.family_id,
            geometry=record.geometry,
            md_m=md_m,
            curve_values=curve_values,
            curve_masks=curve_masks,
            seismic=seismic,
            seismic_time_ms=seismic_time_ms,
            trajectory_features=trajectory_features,
            trajectory_mask=trajectory_mask,
            trajectory_xy_m=trajectory_xy_m,
            trajectory_seismic=trajectory_seismic,
            target_twt_ms=target_twt_ms,
            supervision_mask=supervision_mask,
            p13_twt_ms=p13_twt_ms,
            p13_std_ms=p13_std_ms,
            p13_confidence=p13_confidence,
            sonic_twt_ms=sonic_twt_ms,
            current_gated_twt_ms=current_gated_twt_ms,
            input_metadata=input_metadata,
        )

    def _validate_split(self) -> None:
        supervised = set(self.supervised_well_ids)
        validation_seen: list[str] = []
        if len(self.folds) != 5:
            raise ValueError("GeoPathTie-V1 requires exactly five fixed folds")
        for fold in self.folds:
            train = set(fold.train_wells)
            validation = set(fold.validation_wells)
            unlabeled_train = set(fold.unlabeled_train_wells)
            if train & validation or train | validation != supervised:
                raise ValueError(f"invalid supervised partition in {fold.fold_id}")
            validation_families = {self.records[name].family_id for name in validation}
            train_families = {self.records[name].family_id for name in train}
            unlabeled_families = {self.records[name].family_id for name in unlabeled_train}
            if validation_families & (train_families | unlabeled_families):
                raise ValueError(f"mother-well leakage in {fold.fold_id}")
            if any(value == 0 for value in self.geometry_counts(fold.validation_wells).values()):
                raise ValueError(f"geometry stratum absent from {fold.fold_id}")
            validation_seen.extend(fold.validation_wells)
        if len(validation_seen) != len(supervised) or set(validation_seen) != supervised:
            raise ValueError("five-fold validation must cover every supervised well once")


def _interpolate_inside(query: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_x) & np.isfinite(source_y)
    x = source_x[valid]
    y = source_y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    x, unique = np.unique(x, return_index=True)
    y = y[unique]
    result = np.full(query.shape, np.nan, dtype=np.float64)
    if len(x) < 2:
        return result
    inside = (query >= x[0]) & (query <= x[-1])
    result[inside] = np.interp(query[inside], x, y)
    return result


def load_chengdu_geopath_catalog(
    *,
    wellfuse_root: str | Path = DEFAULT_WELLFUSE_ROOT,
    fold_contract_path: str | Path = DEFAULT_FOLD_CONTRACT,
    static_gate_root: str | Path = DEFAULT_STATIC_GATE_ROOT,
) -> ChengduGeoPathCatalog:
    """Join the checked-in fold contract to the existing Chengdu assets."""

    wellfuse = Path(wellfuse_root).resolve()
    contract_path = Path(fold_contract_path).resolve()
    gate_root = Path(static_gate_root).resolve()
    source_manifest = wellfuse / "data/external_blocks/chengdu_module1/well_manifest.csv"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    observed_manifest_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if observed_manifest_sha256 != str(contract["source_manifest_sha256"]):
        raise ValueError("Chengdu source manifest changed after the five-fold contract was fixed")
    rows = list(csv.DictReader(source_manifest.open(encoding="utf-8-sig", newline="")))

    input_root = wellfuse / "artifacts/cache_v3/chengdu_module1/inputs"
    label_root = wellfuse / "artifacts/cache_v3/chengdu_module1/labels"
    p13_root = wellfuse / "artifacts/align_x_reranker_v1/p13_oof_v1/wells"
    records: dict[str, GeoPathWellRecord] = {}
    for row in rows:
        well_id = row["well_id"]
        supervised = row["role"] == "supervised_training_candidate"
        label_cache = label_root / f"chengdu_all_wells__{well_id}.npz"
        p13_prior = p13_root / well_id / "p13_oof_prior.npz"
        gate_prediction = gate_root / "predictions" / well_id / "prediction.npz"
        records[well_id] = GeoPathWellRecord(
            well_id=well_id,
            family_id=mother_well_family(well_id),
            geometry=geometry_class(
                float(row["maximum_inclination_deg"]),
                float(row["maximum_lateral_offset_m"]),
            ),
            supervised=supervised,
            surface_x_m=float(row["surface_x"]),
            surface_y_m=float(row["surface_y"]),
            input_cache=input_root / f"chengdu_all_wells__{well_id}.npz",
            label_cache=label_cache if supervised else None,
            p13_prior=p13_prior if p13_prior.is_file() else None,
            sonic_gate_prediction=gate_prediction if gate_prediction.is_file() else None,
        )

    supervised_ids = {name for name, item in records.items() if item.supervised}
    unlabeled_ids = {name for name, item in records.items() if not item.supervised}
    folds: list[GeoPathFold] = []
    for fold_id, payload in contract["folds"].items():
        validation = tuple(sorted(map(str, payload["validation_wells"])))
        validation_families = {records[name].family_id for name in validation}
        if validation_families != set(map(str, payload["validation_families"])):
            raise ValueError(f"fixed family membership drifted in {fold_id}")
        observed_geometry = {
            geometry: sum(records[name].geometry == geometry for name in validation)
            for geometry in ("vertical", "deviated", "horizontal")
        }
        if observed_geometry != payload["geometry_counts"]:
            raise ValueError(f"fixed geometry strata drifted in {fold_id}")
        folds.append(
            GeoPathFold(
                fold_id=str(fold_id),
                train_wells=tuple(sorted(supervised_ids.difference(validation))),
                validation_wells=validation,
                unlabeled_train_wells=tuple(
                    sorted(
                        name
                        for name in unlabeled_ids
                        if records[name].family_id not in validation_families
                    )
                ),
            )
        )

    seismic_path = wellfuse.parent / "0模块一预处理模型测试" / Path(
        str(contract["seismic_relative_path"])
    )
    return ChengduGeoPathCatalog(
        records,
        tuple(folds),
        seismic_path=seismic_path,
        source_manifest_path=source_manifest,
        fold_contract_path=contract_path,
    )


__all__ = [
    "ChengduGeoPathCatalog",
    "GeoPathFold",
    "GeoPathWellArrays",
    "GeoPathWellRecord",
    "geometry_class",
    "load_chengdu_geopath_catalog",
    "mother_well_family",
]
