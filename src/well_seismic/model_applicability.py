"""Label-free training-envelope checks at the platform inference boundary.

The canonical envelope is produced by WellFuse and content addressed.  This
module deliberately has no dependency on the WellFuse Python environment so
the platform can reject or route an out-of-domain request *before* launching a
CUDA subprocess.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

TRAINING_ENVELOPE_SCHEMA = "wellfuse.model-training-envelope.v1"
APPLICABILITY_SCHEMA = "wellfuse.model-applicability-evaluation.v1"
ROI_SELECTION_SCHEMA = "well-seismic.unlabeled-seismic-roi-selection.v1"
MAXIMUM_APPLICABILITY_PROFILE_TRACES = 64


class ModelApplicabilityError(ValueError):
    """Raised when the applicability contract is malformed or tampered."""


@dataclass(frozen=True)
class UnlabeledSeismicRoiSelection:
    """A label-free, trace-backed bounded SEG-Y selection."""

    start_zyx: tuple[int, int, int]
    size_zyx: tuple[int, int, int]
    trace_indices: np.ndarray
    receipt: dict[str, Any]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("envelope_sha256", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_training_envelope(path: str | Path, *, model_id: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelApplicabilityError("training envelope must be a JSON object")
    if payload.get("schema_version") != TRAINING_ENVELOPE_SCHEMA:
        raise ModelApplicabilityError("training envelope schema mismatch")
    if payload.get("model_id") != model_id:
        raise ModelApplicabilityError("training envelope model_id mismatch")
    declared = str(payload.get("envelope_sha256", ""))
    if len(declared) != 64 or declared != _canonical_sha256(payload):
        raise ModelApplicabilityError("training envelope SHA-256 drifted")
    if not isinstance(payload.get("features"), dict) or not payload["features"]:
        raise ModelApplicabilityError("training envelope features are missing")
    policy = payload.get("policy")
    if not isinstance(policy, dict) or not str(policy.get("ood_route", "")):
        raise ModelApplicabilityError("training envelope policy is missing")
    return payload


def resolve_training_envelope(
    wellfuse_root: str | Path, model_id: str
) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(wellfuse_root).expanduser().resolve() / "artifacts" / "model_applicability_v1"
    index_path = root / "index.json"
    if not index_path.is_file():
        return None, None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "wellfuse.model-training-envelope-index.v1":
        raise ModelApplicabilityError("training envelope index schema mismatch")
    record = (index.get("models") or {}).get(model_id)
    if not isinstance(record, Mapping):
        return None, None
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelApplicabilityError("training envelope path is not confined")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ModelApplicabilityError("training envelope path escaped its root")
    payload = load_training_envelope(path, model_id=model_id)
    if payload["envelope_sha256"] != record.get("sha256"):
        raise ModelApplicabilityError("training envelope index digest drifted")
    return payload, path


def _numeric_score(value: float, rule: Mapping[str, Any]) -> tuple[float, bool, str]:
    reference_low, reference_high = map(float, rule["reference"])
    hard_low, hard_high = map(float, rule["hard"])
    if value < hard_low or value > hard_high:
        return 0.0, True, "outside_hard_envelope"
    if reference_low <= value <= reference_high:
        return 1.0, False, "inside_reference_envelope"
    if value < reference_low:
        score = (value - hard_low) / max(reference_low - hard_low, 1e-12)
    else:
        score = (hard_high - value) / max(hard_high - reference_high, 1e-12)
    return float(np.clip(score, 0.0, 1.0)), False, "inside_soft_margin"


def evaluate_applicability(
    envelope: Mapping[str, Any] | None,
    observations: Mapping[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    """Score observations without labels and return execute/route/abstain."""

    if envelope is None:
        return {
            "schema_version": APPLICABILITY_SCHEMA,
            "model_id": model_id,
            "status": "unknown",
            "score": None,
            "decision": "unassessed",
            "route": "caller_policy_required",
            "envelope_sha256": None,
            "envelope_path": None,
            "label_free": True,
            "observations": dict(observations),
            "feature_results": {},
            "issues": ["training_envelope_missing"],
        }
    if envelope.get("model_id") != model_id:
        raise ModelApplicabilityError("envelope model_id mismatch")
    feature_results: dict[str, Any] = {}
    missing: list[str] = []
    violations: list[str] = []
    numerator = 0.0
    denominator = 0.0
    for name, rule in envelope["features"].items():
        required = bool(rule.get("required", True))
        weight = float(rule.get("weight", 1.0))
        raw = observations.get(name)
        if raw is None:
            feature_results[name] = {
                "observed": None,
                "score": None,
                "required": required,
                "status": "missing",
            }
            if required:
                missing.append(name)
            continue
        denominator += weight
        if rule.get("kind") == "categorical":
            observed: Any = str(raw)
            hard = observed not in {str(item) for item in rule.get("accepted", [])}
            score = 0.0 if hard else 1.0
            state = "unaccepted_category" if hard else "accepted_category"
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            observed = value if math.isfinite(value) else str(raw)
            if math.isfinite(value):
                score, hard, state = _numeric_score(value, rule)
            else:
                score, hard, state = 0.0, True, "non_finite"
        numerator += weight * score
        if hard:
            violations.append(name)
        feature_results[name] = {
            "observed": observed,
            "score": score,
            "weight": weight,
            "required": required,
            "status": state,
            "hard_violation": hard,
        }
    score = numerator / denominator if denominator else 0.0
    policy = envelope["policy"]
    if missing or violations:
        status = "out_of_domain"
    elif score >= float(policy["execute_threshold"]):
        status = "applicable"
    elif score >= float(policy["caution_threshold"]):
        status = "caution"
    else:
        status = "out_of_domain"
    if status == "applicable":
        decision, route = "execute_model", "model"
    elif status == "caution":
        decision, route = "execute_with_warning", "model_with_ood_warning"
    else:
        decision, route = "do_not_execute_model", str(policy["ood_route"])
    return {
        "schema_version": APPLICABILITY_SCHEMA,
        "model_id": model_id,
        "task_kind": envelope.get("task_kind"),
        "status": status,
        "score": float(score),
        "decision": decision,
        "route": route,
        "envelope_sha256": envelope["envelope_sha256"],
        "label_free": True,
        "observations": dict(observations),
        "feature_results": feature_results,
        "issues": [
            *(f"missing_required:{name}" for name in missing),
            *(f"hard_violation:{name}" for name in violations),
        ],
    }


def _spectral_and_amplitude_features(traces: np.ndarray, sample_interval_ms: float) -> dict[str, float]:
    values = np.asarray(traces, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 16:
        return {}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)) * 1.4826)
    scale = max(mad, float(np.std(finite)), 1e-12)
    normalized = np.nan_to_num((values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    spectrum = np.square(np.abs(np.fft.rfft(normalized, axis=1)))
    spectrum[:, 0] = 0.0
    energy = spectrum.sum(axis=0)
    total = float(energy.sum())
    features = {
        "seismic.amplitude_p99_over_mad": float(
            np.percentile(np.abs(finite - median), 99.0) / max(mad, 1e-12)
        ),
        "seismic.zero_fraction": float(np.mean(finite == 0.0)),
    }
    if total <= 0.0:
        return features
    cumulative = np.cumsum(energy) / total
    frequency = np.fft.rfftfreq(values.shape[1], d=sample_interval_ms / 1000.0)
    for quantile, name in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
        index = min(int(np.searchsorted(cumulative, quantile)), len(frequency) - 1)
        features[f"seismic.frequency_{name}_hz"] = float(frequency[index])
    return features


def _axis_spacing_m(geometry: Any) -> tuple[float | None, float | None]:
    required = (geometry.inline, geometry.crossline, geometry.x, geometry.y)
    if any(item is None for item in required):
        return None, None
    inline, crossline, x, y = (np.asarray(item) for item in required)
    if not (inline.shape == crossline.shape == x.shape == y.shape) or inline.size < 2:
        return None, None
    def one_line(primary: np.ndarray, secondary: np.ndarray) -> float | None:
        # Sampling a few complete grid lines avoids sorting a million-trace
        # survey while remaining independent of SEG-Y trace traversal order.
        candidates = np.unique(primary)
        if not candidates.size:
            return None
        sample = candidates[
            np.unique(np.linspace(0, len(candidates) - 1, min(7, len(candidates)), dtype=int))
        ]
        values: list[np.ndarray] = []
        for primary_value in sample:
            selected = np.flatnonzero(primary == primary_value)
            if selected.size < 2:
                continue
            order = np.argsort(secondary[selected], kind="stable")
            indices = selected[order]
            distance = np.hypot(np.diff(x[indices]), np.diff(y[indices]))
            distance = distance[np.isfinite(distance) & (distance > 0.0)]
            if distance.size:
                values.append(distance)
        return float(np.median(np.concatenate(values))) if values else None

    return one_line(inline, crossline), one_line(crossline, inline)


def _axis_candidate_starts(length: int, count: int, explicit: int | None) -> list[int]:
    maximum = length - count
    if maximum < 0:
        raise ValueError(f"ROI count {count} exceeds source axis length {length}")
    if explicit is not None:
        start = int(explicit)
        if start < 0 or start > maximum:
            raise ValueError(
                f"explicit ROI start {start} with count {count} exceeds axis {length}"
            )
        return [start]
    center = maximum // 2
    samples = np.linspace(0, maximum, min(9, maximum + 1), dtype=np.int64)
    return sorted(
        {center, *(int(value) for value in samples)},
        key=lambda value: (abs(value - center), value),
    )


def _trace_index_grid(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geometry.inline is None or geometry.crossline is None:
        raise ValueError("unlabeled ROI selection requires Inline/Crossline headers")
    inline = np.asarray(geometry.inline, dtype=np.int64)
    crossline = np.asarray(geometry.crossline, dtype=np.int64)
    if inline.shape != crossline.shape or inline.size != int(geometry.trace_count):
        raise ValueError("SEG-Y Inline/Crossline headers do not match trace count")
    inline_values = np.unique(inline)
    crossline_values = np.unique(crossline)
    inline_index = np.searchsorted(inline_values, inline)
    crossline_index = np.searchsorted(crossline_values, crossline)
    flat = inline_index * len(crossline_values) + crossline_index
    if np.unique(flat).size != flat.size:
        raise ValueError("unlabeled ROI selection rejects duplicate Inline/Crossline bins")
    grid = np.full(
        (len(inline_values), len(crossline_values)), -1, dtype=np.int64
    )
    grid.ravel()[flat] = np.arange(inline.size, dtype=np.int64)
    return grid, inline_values, crossline_values


def _densest_window_start(
    trace_grid: np.ndarray,
    *,
    inline_count: int,
    crossline_count: int,
) -> tuple[int, int]:
    valid = (trace_grid >= 0).astype(np.int32, copy=False)
    integral = np.pad(valid, ((1, 0), (1, 0))).cumsum(
        axis=0, dtype=np.int32
    ).cumsum(axis=1, dtype=np.int32)
    counts = (
        integral[inline_count:, crossline_count:]
        - integral[:-inline_count, crossline_count:]
        - integral[inline_count:, :-crossline_count]
        + integral[:-inline_count, :-crossline_count]
    )
    center = (
        (trace_grid.shape[0] - inline_count) // 2,
        (trace_grid.shape[1] - crossline_count) // 2,
    )
    maximum_count = int(counts.max())
    if int(counts[center]) == maximum_count:
        return center
    maximum_start = (
        trace_grid.shape[0] - inline_count,
        trace_grid.shape[1] - crossline_count,
    )
    best: tuple[float, int, int] | None = None
    for inline_start in range(counts.shape[0]):
        crossline_candidates = np.flatnonzero(
            counts[inline_start] == maximum_count
        )
        for crossline_start in crossline_candidates:
            distance = (
                (inline_start - center[0]) / max(1, maximum_start[0])
            ) ** 2 + (
                (int(crossline_start) - center[1])
                / max(1, maximum_start[1])
            ) ** 2
            candidate = (float(distance), inline_start, int(crossline_start))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def _amplitude_probe(
    reader: Any,
    trace_indices: np.ndarray,
    *,
    sample_start: int,
    sample_count: int,
    maximum_traces: int,
) -> dict[str, Any]:
    if trace_indices.size < 1:
        return {
            "sampled_trace_count": 0,
            "active_trace_count": 0,
            "active_trace_fraction": 0.0,
            "finite_sample_fraction": 0.0,
            "nonzero_sample_fraction": 0.0,
        }
    positions = np.unique(
        np.linspace(
            0,
            len(trace_indices) - 1,
            min(maximum_traces, len(trace_indices)),
            dtype=np.int64,
        )
    )
    selected = trace_indices[positions]
    finite_count = 0
    nonzero_count = 0
    total_count = 0
    active_count = 0
    for trace_index in selected:
        values = np.asarray(
            reader.read_trace(
                int(trace_index), slice(sample_start, sample_start + sample_count)
            ),
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        total_count += int(values.size)
        finite_count += int(finite.size)
        if finite.size < 2:
            continue
        nonzero = finite != 0.0
        nonzero_count += int(np.count_nonzero(nonzero))
        median = float(np.median(finite))
        centered = finite - median
        mad = float(1.4826 * np.median(np.abs(centered)))
        rms = float(np.sqrt(np.mean(np.square(centered))))
        numerical_floor = max(1.0, abs(median)) * np.finfo(np.float64).eps * 16.0
        has_dynamic_amplitude = max(mad, rms) > numerical_floor
        if float(np.mean(nonzero)) >= 0.01 and has_dynamic_amplitude:
            active_count += 1
    sampled_count = int(len(selected))
    return {
        "sampled_trace_count": sampled_count,
        "active_trace_count": active_count,
        "active_trace_fraction": float(active_count / max(1, sampled_count)),
        "finite_sample_fraction": float(finite_count / max(1, total_count)),
        "nonzero_sample_fraction": float(nonzero_count / max(1, finite_count)),
    }


def select_unlabeled_seismic_roi(
    reader: Any,
    *,
    size_zyx: Sequence[int],
    explicit_start_zyx: Sequence[int | None] | None = None,
    minimum_valid_trace_fraction: float = 0.5,
    minimum_active_trace_fraction: float = 0.5,
    minimum_nonzero_sample_fraction: float = 0.01,
    minimum_finite_sample_fraction: float = 0.99,
    maximum_probe_traces: int = 16,
    maximum_candidate_probes: int = 128,
) -> UnlabeledSeismicRoiSelection:
    """Select the nearest dense, active bounded volume without labels.

    Explicit start components are never moved.  Missing components are tried
    from the geometric centre outwards on a deterministic nine-point lattice.
    Header occupancy is checked before a bounded sample window is read, and a
    window is accepted only when real, varying amplitudes are present.
    """

    geometry = reader.geometry or reader.inspect()
    requested_size = tuple(int(value) for value in size_zyx)
    if len(requested_size) != 3 or any(value < 1 for value in requested_size):
        raise ValueError("unlabeled SEG-Y ROI size must be a positive Z/Inline/Xline triple")
    source_shape = (
        int(geometry.samples_per_trace),
        int(np.unique(geometry.inline).size) if geometry.inline is not None else 0,
        int(np.unique(geometry.crossline).size)
        if geometry.crossline is not None
        else 0,
    )
    if any(size > available for size, available in zip(requested_size, source_shape)):
        raise ValueError(
            f"unlabeled SEG-Y ROI {requested_size} exceeds source {source_shape}"
        )
    if not 0.0 < minimum_valid_trace_fraction <= 1.0:
        raise ValueError("minimum_valid_trace_fraction must be in (0,1]")
    if not 0.0 < minimum_active_trace_fraction <= 1.0:
        raise ValueError("minimum_active_trace_fraction must be in (0,1]")
    if not 0.0 <= minimum_nonzero_sample_fraction <= 1.0:
        raise ValueError("minimum_nonzero_sample_fraction must be in [0,1]")
    if not 0.0 <= minimum_finite_sample_fraction <= 1.0:
        raise ValueError("minimum_finite_sample_fraction must be in [0,1]")
    if maximum_probe_traces < 1:
        raise ValueError("maximum_probe_traces must be positive")
    if maximum_candidate_probes < 1:
        raise ValueError("maximum_candidate_probes must be positive")

    if explicit_start_zyx is None:
        explicit_start: tuple[int | None, int | None, int | None] = (
            None,
            None,
            None,
        )
    else:
        if len(explicit_start_zyx) != 3:
            raise ValueError("explicit SEG-Y ROI start must be a Z/Inline/Xline triple")
        explicit_start = tuple(
            None if value is None else int(value) for value in explicit_start_zyx
        )

    trace_grid, inline_values, crossline_values = _trace_index_grid(geometry)
    z_count, inline_count, crossline_count = requested_size
    z_starts = _axis_candidate_starts(
        source_shape[0], z_count, explicit_start[0]
    )
    inline_starts = _axis_candidate_starts(
        source_shape[1], inline_count, explicit_start[1]
    )
    crossline_starts = _axis_candidate_starts(
        source_shape[2], crossline_count, explicit_start[2]
    )
    spatial_starts = set(product(inline_starts, crossline_starts))
    if explicit_start[1] is None and explicit_start[2] is None:
        spatial_starts.add(
            _densest_window_start(
                trace_grid,
                inline_count=inline_count,
                crossline_count=crossline_count,
            )
        )
    center = tuple(
        (available - count) // 2
        for available, count in zip(source_shape, requested_size)
    )
    maximum_start = tuple(
        available - count
        for available, count in zip(source_shape, requested_size)
    )

    candidates: list[tuple[int, float, int, int, int, np.ndarray]] = []
    for z_start, (inline_start, crossline_start) in product(
        z_starts, spatial_starts
    ):
        window = trace_grid[
            inline_start : inline_start + inline_count,
            crossline_start : crossline_start + crossline_count,
        ]
        trace_indices = window[window >= 0]
        valid_count = int(trace_indices.size)
        normalized_distance = sum(
            ((start - midpoint) / max(1, limit)) ** 2
            for start, midpoint, limit in zip(
                (z_start, inline_start, crossline_start),
                center,
                maximum_start,
            )
        )
        candidates.append(
            (
                -valid_count,
                float(normalized_distance),
                z_start,
                inline_start,
                crossline_start,
                trace_indices,
            )
        )
    candidates.sort(key=lambda item: item[:5])

    evaluated = 0
    best_rejection: dict[str, Any] | None = None
    total_cells = inline_count * crossline_count
    for (
        negative_valid_count,
        _distance,
        z_start,
        inline_start,
        crossline_start,
        trace_indices,
    ) in candidates[:maximum_candidate_probes]:
        valid_count = -negative_valid_count
        valid_fraction = float(valid_count / total_cells)
        evaluated += 1
        if valid_fraction < minimum_valid_trace_fraction:
            probe = {
                "sampled_trace_count": 0,
                "active_trace_count": 0,
                "active_trace_fraction": 0.0,
                "finite_sample_fraction": 0.0,
                "nonzero_sample_fraction": 0.0,
            }
        else:
            probe = _amplitude_probe(
                reader,
                trace_indices,
                sample_start=z_start,
                sample_count=z_count,
                maximum_traces=maximum_probe_traces,
            )
        accepted = bool(
            valid_fraction >= minimum_valid_trace_fraction
            and probe["active_trace_fraction"] >= minimum_active_trace_fraction
            and probe["nonzero_sample_fraction"]
            >= minimum_nonzero_sample_fraction
            and probe["finite_sample_fraction"] >= minimum_finite_sample_fraction
        )
        rejection = {
            "start_zyx": [z_start, inline_start, crossline_start],
            "valid_trace_fraction": valid_fraction,
            **probe,
        }
        if best_rejection is None or (
            rejection["active_trace_fraction"],
            rejection["nonzero_sample_fraction"],
            rejection["valid_trace_fraction"],
        ) > (
            best_rejection["active_trace_fraction"],
            best_rejection["nonzero_sample_fraction"],
            best_rejection["valid_trace_fraction"],
        ):
            best_rejection = rejection
        if not accepted:
            continue

        requested = {
            "sample_start": explicit_start[0],
            "sample_count": z_count,
            "inline_start": explicit_start[1],
            "inline_count": inline_count,
            "crossline_start": explicit_start[2],
            "crossline_count": crossline_count,
        }
        resolved = {
            "sample_start": z_start,
            "sample_count": z_count,
            "inline_start": inline_start,
            "inline_count": inline_count,
            "crossline_start": crossline_start,
            "crossline_count": crossline_count,
        }
        raw_source_path = str(
            getattr(reader, "path", getattr(geometry, "path", ""))
        ).strip()
        source_path = Path(raw_source_path).expanduser() if raw_source_path else None
        source_stat = (
            source_path.stat()
            if source_path is not None and source_path.is_file()
            else None
        )
        receipt: dict[str, Any] = {
            "contract_version": ROI_SELECTION_SCHEMA,
            "selection_semantics": (
                "zero_based_indices_into_sorted_unique_grid_and_sample_axes"
            ),
            "selection_policy": (
                "explicit_start_components_validated"
                if any(value is not None for value in explicit_start)
                else "auto_nearest_center_active_dense"
            ),
            "label_free": True,
            "target_or_supervision_open_count": 0,
            "source": {
                "path": str(source_path.resolve()) if source_path is not None else "",
                "size_bytes": source_stat.st_size if source_stat is not None else None,
                "mtime_ns": source_stat.st_mtime_ns if source_stat is not None else None,
                "trace_count": int(geometry.trace_count),
                "geometry_profile": str(getattr(geometry, "profile", "")),
                "geometry_confidence": float(
                    getattr(geometry, "confidence", 0.0)
                ),
            },
            "source_shape_t_inline_xline": list(source_shape),
            "selected_shape_t_inline_xline": list(requested_size),
            "requested": requested,
            "resolved": resolved,
            "inline_value_range": [
                int(inline_values[inline_start]),
                int(inline_values[inline_start + inline_count - 1]),
            ],
            "crossline_value_range": [
                int(crossline_values[crossline_start]),
                int(crossline_values[crossline_start + crossline_count - 1]),
            ],
            "grid": {
                "cell_count": total_cells,
                "valid_trace_count": valid_count,
                "valid_trace_fraction": valid_fraction,
            },
            "amplitude_probe": {
                "sample_start": z_start,
                "sample_count": z_count,
                **probe,
            },
            "thresholds": {
                "minimum_valid_trace_fraction": minimum_valid_trace_fraction,
                "minimum_active_trace_fraction": minimum_active_trace_fraction,
                "minimum_nonzero_sample_fraction": (
                    minimum_nonzero_sample_fraction
                ),
                "minimum_finite_sample_fraction": minimum_finite_sample_fraction,
                "maximum_probe_traces": maximum_probe_traces,
                "maximum_candidate_probes": maximum_candidate_probes,
            },
            "candidate_count": len(candidates),
            "evaluated_candidate_count": evaluated,
        }
        encoded = json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
        return UnlabeledSeismicRoiSelection(
            start_zyx=(z_start, inline_start, crossline_start),
            size_zyx=requested_size,
            trace_indices=np.asarray(trace_indices, dtype=np.int64),
            receipt=receipt,
        )

    mode = (
        "explicit ROI"
        if any(value is not None for value in explicit_start)
        else "automatic ROI candidates"
    )
    raise ValueError(
        f"{mode} contains insufficient real SEG-Y amplitude support; "
        f"best_label_free_probe={best_rejection}"
    )


def observe_seismic_reader(
    reader: Any,
    *,
    maximum_traces: int = 16,
    trace_indices: Sequence[int] | np.ndarray | None = None,
    inline_count: int | None = None,
    crossline_count: int | None = None,
) -> dict[str, Any]:
    """Extract a bounded, target-free survey profile from an inspected reader."""

    geometry = reader.geometry or reader.inspect()
    resolved_inline_count = (
        int(inline_count)
        if inline_count is not None
        else int(np.unique(geometry.inline).size)
        if geometry.inline is not None
        else 0
    )
    resolved_crossline_count = (
        int(crossline_count)
        if crossline_count is not None
        else int(np.unique(geometry.crossline).size)
        if geometry.crossline is not None
        else 0
    )
    if maximum_traces < 1:
        raise ValueError("maximum_traces must be positive")
    if trace_indices is None:
        available_trace_count = int(geometry.trace_count)
        if available_trace_count < 1:
            raise ValueError("applicability survey contains no SEG-Y traces")
        selected_count = min(
            maximum_traces,
            MAXIMUM_APPLICABILITY_PROFILE_TRACES,
            available_trace_count,
        )
        indices = np.unique(
            np.linspace(
                0,
                available_trace_count - 1,
                selected_count,
                dtype=np.int64,
            )
        )
    else:
        available_indices = np.unique(np.asarray(trace_indices, dtype=np.int64))
        if available_indices.size < 1:
            raise ValueError("applicability ROI contains no SEG-Y traces")
        if available_indices[0] < 0 or available_indices[-1] >= int(
            geometry.trace_count
        ):
            raise ValueError("applicability ROI trace index exceeds the SEG-Y")
        available_trace_count = int(available_indices.size)
        selected_count = min(
            maximum_traces,
            MAXIMUM_APPLICABILITY_PROFILE_TRACES,
            available_trace_count,
        )
        positions = np.unique(
            np.linspace(
                0,
                available_trace_count - 1,
                selected_count,
                dtype=np.int64,
            )
        )
        indices = available_indices[positions]
    traces = np.stack(
        [
            np.asarray(reader.read_trace(int(index)), dtype=np.float32)
            for index in indices
        ]
    )
    spacing_inline, spacing_crossline = _axis_spacing_m(geometry)
    time = np.asarray(geometry.time_axis, dtype=np.float64)
    result: dict[str, Any] = {
        "seismic.sample_interval_ms": float(geometry.sample_interval),
        "seismic.time_start_ms": float(time[0]),
        "seismic.time_end_ms": float(time[-1]),
        "seismic.time_span_ms": float(time[-1] - time[0]),
        "seismic.sample_count": int(geometry.samples_per_trace),
        "seismic.trace_count": available_trace_count,
        "seismic.profiled_trace_count": int(indices.size),
        "seismic.profile_trace_limit": MAXIMUM_APPLICABILITY_PROFILE_TRACES,
        "seismic.inline_count": resolved_inline_count,
        "seismic.crossline_count": resolved_crossline_count,
        "seismic.geometry_confidence": float(geometry.confidence),
        **_spectral_and_amplitude_features(traces, float(geometry.sample_interval)),
    }
    if spacing_inline is not None:
        result["seismic.trace_spacing_inline_m"] = spacing_inline
    if spacing_crossline is not None:
        result["seismic.trace_spacing_crossline_m"] = spacing_crossline
    return result


def _inclination_degrees(trajectory: Any) -> np.ndarray:
    md = np.asarray(trajectory.md, dtype=np.float64)
    tvd = np.asarray(trajectory.tvd, dtype=np.float64)
    if trajectory.x is not None and trajectory.y is not None:
        x = np.asarray(trajectory.x, dtype=np.float64)
        y = np.asarray(trajectory.y, dtype=np.float64)
    else:
        x = np.asarray(trajectory.x_offset, dtype=np.float64)
        y = np.asarray(trajectory.y_offset, dtype=np.float64)
    if md.size < 2:
        return np.empty(0, dtype=np.float64)
    delta_md = np.maximum(np.diff(md), 1e-9)
    vertical_fraction = np.clip(np.abs(np.diff(tvd)) / delta_md, 0.0, 1.0)
    inclination = np.degrees(np.arccos(vertical_fraction))
    finite = np.isfinite(inclination) & np.isfinite(np.diff(x)) & np.isfinite(np.diff(y))
    return inclination[finite]


def observe_align_wells(eligible: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate complete-trajectory, acoustic and trace-support observations."""

    if not eligible:
        return {"well.count": 0}
    dt_coverages: list[float] = []
    md_spans: list[float] = []
    inclinations: list[float] = []
    lateral_spans: list[float] = []
    trace_distances: list[float] = []
    for item in eligible:
        log = item["acoustic"]
        values = np.asarray(log.curves["DT"], dtype=np.float64)
        mask = np.asarray(log.masks["DT"], dtype=bool)
        dt_coverages.append(float(np.mean(mask & np.isfinite(values) & (values > 0.0))))
        trajectory = item["trajectory"]
        md = np.asarray(trajectory.md, dtype=np.float64)
        md_spans.append(float(md[-1] - md[0]))
        angle = _inclination_degrees(trajectory)
        inclinations.append(float(np.percentile(angle, 95.0)) if angle.size else 0.0)
        if trajectory.x is not None and trajectory.y is not None:
            x = np.asarray(trajectory.x, dtype=np.float64)
            y = np.asarray(trajectory.y, dtype=np.float64)
        else:
            x = np.asarray(trajectory.x_offset, dtype=np.float64)
            y = np.asarray(trajectory.y_offset, dtype=np.float64)
        lateral_spans.append(float(np.hypot(x[-1] - x[0], y[-1] - y[0])))
        trace_distances.append(float(item["nearest_trace_distance_m"]))
    return {
        "well.count": len(eligible),
        "well.dt_coverage_p10": float(np.percentile(dt_coverages, 10.0)),
        "well.md_span_p10_m": float(np.percentile(md_spans, 10.0)),
        "well.md_span_p90_m": float(np.percentile(md_spans, 90.0)),
        "well.max_inclination_p95_deg": float(np.percentile(inclinations, 95.0)),
        "well.lateral_displacement_p95_m": float(np.percentile(lateral_spans, 95.0)),
        "well.nearest_trace_distance_p90_m": float(np.percentile(trace_distances, 90.0)),
        "well.complete_trajectory_fraction": 1.0,
    }


def write_applicability_manifest(
    path: str | Path,
    evaluation: Mapping[str, Any],
    *,
    envelope_path: str | Path | None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = dict(evaluation)
    payload["envelope_path"] = str(Path(envelope_path).resolve()) if envelope_path else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if destination.exists():
        current = destination.read_text(encoding="utf-8")
        if current != encoded:
            raise FileExistsError(f"applicability manifest already exists: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "APPLICABILITY_SCHEMA",
    "MAXIMUM_APPLICABILITY_PROFILE_TRACES",
    "ModelApplicabilityError",
    "evaluate_applicability",
    "load_training_envelope",
    "observe_align_wells",
    "observe_seismic_reader",
    "resolve_training_envelope",
    "write_applicability_manifest",
]
