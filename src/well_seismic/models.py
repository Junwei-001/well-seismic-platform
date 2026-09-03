from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Evidence:
    value: Any
    source_type: str
    source: str
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Asset:
    asset_id: str
    role: str
    path: Path
    dataset: str
    version: str | None = None
    stage: str = "UNKNOWN"
    options: dict[str, Any] = field(default_factory=dict)
    identity: tuple[int, int] | None = None


@dataclass
class CurveInfo:
    original_name: str
    standard_name: str
    original_unit: str = ""
    standard_unit: str = ""
    confidence: float = 0.0
    evidence: dict[str, float] = field(default_factory=dict)
    source: str = ""


@dataclass
class WellLog:
    well_name: str
    depth: np.ndarray
    curves: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    curve_info: dict[str, CurveInfo]
    header: dict[str, Evidence]
    source: str
    version: str = "unknown"
    issues: list[str] = field(default_factory=list)
    processing_steps: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)


@dataclass
class WellHead:
    well_name: str
    x: float | None = None
    y: float | None = None
    kb: float | None = None
    ground_elevation: float | None = None
    total_depth_md: float | None = None
    crs: str | None = None
    source: str = ""
    confidence: float = 1.0
    vertical_datum_unit: str | None = None
    # X/Y are canonical metres only.  Unknown source units leave X/Y unset so
    # the record can still contribute non-spatial metadata without entering a
    # spatial well-to-seismic match.
    horizontal_unit: str = "unknown"
    coordinate_issues: list[str] = field(default_factory=list)
    source_x: float | None = None
    source_y: float | None = None
    source_crs: str | None = None
    coordinate_transform: dict[str, Any] = field(default_factory=dict)
    identifiers: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    well_name: str
    md: np.ndarray
    tvd: np.ndarray
    x_offset: np.ndarray
    y_offset: np.ndarray
    inclination: np.ndarray | None = None
    azimuth: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    source: str = ""
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    source_units: dict[str, str] = field(default_factory=dict)
    unit_conversions: list[str] = field(default_factory=list)
    unit_provenance: dict[str, str] = field(default_factory=dict)
    vertical_semantics: dict[str, Any] = field(default_factory=dict)
    source_crs: str | None = None
    horizontal_crs: str | None = None
    coordinate_transform: dict[str, Any] = field(default_factory=dict)
    identifiers: list[str] = field(default_factory=list)


@dataclass
class TimeDepthTable:
    """带坐标域和单位来源的时深控制点，避免把MD表误用于TVD。"""

    source: str
    depth: np.ndarray
    time: np.ndarray
    source_kind: str = "unknown"
    depth_domain: str = "md"
    depth_unit: str = "unknown"
    time_unit: str = "unknown"
    confidence: float = 0.95
    depth_datum: str | None = None
    depth_convention: str | None = None
    time_reference: str = "unknown"
    time_domain: str = "unknown"
    correction_state: str = "unknown"
    replacement_velocity_mps: float | None = None
    md_offset_to_trajectory_m: float | None = None


@dataclass
class SeismicGeometry:
    path: str
    revision: float | None
    endian: str
    sample_format: int
    sample_interval: float
    samples_per_trace: int
    trace_count: int
    time_axis: np.ndarray
    inline: np.ndarray | None
    crossline: np.ndarray | None
    x: np.ndarray | None
    y: np.ndarray | None
    trace_offsets: np.ndarray
    coordinate_scalar: np.ndarray | None
    profile: str
    confidence: float
    issues: list[str] = field(default_factory=list)
    source_crs: str | None = None
    horizontal_crs: str | None = None
    coordinate_transform: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchRecord:
    well_uid: str
    well_name: str
    log_source: str
    seismic_source: str
    md: float
    tvd: float | None
    z_msl_m: float | None
    depth_below_msl_m: float | None
    depth_below_srd_m: float | None
    x: float
    y: float
    trace_index: int
    inline: int | None
    crossline: int | None
    distance: float
    seismic_coordinate: float | None
    horizontal_confidence: float
    vertical_method: str
    vertical_confidence: float
    well_features: dict[str, float]
    well_mask: dict[str, bool]
    seismic_window: list[float] | None
    provenance: dict[str, Any]
    vertical_status: str = "horizontal_only"
    vertical_uncertainty_ms: float | None = None
    seismic_window_valid: bool = False
    coordinate_reference_verified: bool = False
    vertical_datum_verified: bool = False
    training_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
