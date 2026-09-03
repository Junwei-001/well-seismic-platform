"""Conservative cross-inline reconciliation for ordered stratigraphic packages.

The neural network emits per-inline instance ids.  Those ids are local to one
section and therefore cannot be compared across the volume.  This module links
the ordered instances with a monotone dynamic program, preserves abstentions,
and derives horizon surfaces from the globally reconciled package volume.

The implementation deliberately has no scipy dependency so it can run in the
small offline inference environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations
from typing import Any

import numpy as np


UNKNOWN_LABEL = -1
RECONCILIATION_SCHEMA = "surface-seg.global-reconciliation.v1"
MAXIMUM_DOMINANT_PACKAGE_FRACTION = 0.98
NEIGHBOUR_CONSENSUS_PASSES = 6
REFERENCE_UNARY_WEIGHT = 0.10
LOCAL_FRAGMENT_COALESCING_PENALTY = 0.10
HORIZON_DISPLAY_GATE_SCHEMA = "surface-seg.horizon-display-gate.v1"
DEFAULT_MINIMUM_FINITE_TRACE_FRACTION = 0.10
DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION = 0.05
SURFACE_COMPONENT_CONNECTIVITY = 4


@dataclass(frozen=True)
class _Descriptor:
    """Depth statistics for one local package or one active global track."""

    identifier: int
    midpoint: float
    thickness: float
    support: int
    mean_confidence: float


def _describe_inline(labels: np.ndarray, confidence: np.ndarray) -> list[_Descriptor]:
    sample_count = max(int(labels.shape[1]), 1)
    flat_labels = labels.reshape(-1)
    known = flat_labels >= 0
    if not np.any(known):
        return []
    values = flat_labels[known].astype(np.int64, copy=False)
    weights = np.clip(confidence.reshape(-1)[known].astype(np.float64), 1e-6, 1.0)
    sample_indexes = np.broadcast_to(
        np.arange(sample_count, dtype=np.int32), labels.shape
    ).reshape(-1)[known]
    size = int(values.max()) + 1
    counts = np.bincount(values, minlength=size)
    weight_sums = np.bincount(values, weights=weights, minlength=size)
    sums = np.bincount(values, weights=sample_indexes * weights, minlength=size)
    confidence_sums = np.bincount(values, weights=weights, minlength=size)
    minimums = np.full(size, sample_count, dtype=np.int32)
    maximums = np.full(size, -1, dtype=np.int32)
    np.minimum.at(minimums, values, sample_indexes)
    np.maximum.at(maximums, values, sample_indexes)
    descriptors: list[_Descriptor] = []
    minimum_support = max(2, int(labels.size * 1e-5))
    for value in np.flatnonzero(counts):
        if counts[value] < minimum_support:
            continue
        descriptors.append(
            _Descriptor(
                identifier=int(value),
                midpoint=float(sums[value] / weight_sums[value] / sample_count),
                thickness=float(
                    max((maximums[value] - minimums[value]) / sample_count, 1.0 / sample_count)
                ),
                support=int(counts[value]),
                mean_confidence=float(confidence_sums[value] / counts[value]),
            )
        )
    return sorted(descriptors, key=lambda item: (item.midpoint, item.identifier))


def _monotone_match(
    previous: list[_Descriptor],
    current: list[_Descriptor],
    *,
    common_shift: float,
    gap_penalty: float = 0.78,
    maximum_match_cost: float = 1.35,
) -> list[tuple[int, int]]:
    """Needleman-Wunsch matching that can never reverse layer order."""

    rows, columns = len(previous), len(current)
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    moves = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    costs[0, 0] = 0.0
    costs[1:, 0] = np.arange(1, rows + 1) * gap_penalty
    costs[0, 1:] = np.arange(1, columns + 1) * gap_penalty
    moves[1:, 0] = 1
    moves[0, 1:] = 2

    pair_costs = np.full((rows, columns), np.inf, dtype=np.float64)
    for old_index, old in enumerate(previous):
        for new_index, new in enumerate(current):
            depth_delta = abs((new.midpoint - common_shift) - old.midpoint)
            thickness_delta = abs(new.thickness - old.thickness)
            confidence_delta = abs(new.mean_confidence - old.mean_confidence)
            pair_costs[old_index, new_index] = (
                depth_delta / 0.09
                + 0.35 * thickness_delta / 0.12
                + 0.10 * confidence_delta
            )

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                costs[row - 1, column - 1] + pair_costs[row - 1, column - 1],
                costs[row - 1, column] + gap_penalty,
                costs[row, column - 1] + gap_penalty,
            )
            move = int(np.argmin(candidates))
            costs[row, column] = candidates[move]
            moves[row, column] = move

    matches: list[tuple[int, int]] = []
    row, column = rows, columns
    while row or column:
        move = int(moves[row, column])
        if row and column and move == 0:
            if pair_costs[row - 1, column - 1] <= maximum_match_cost:
                matches.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and (not column or move == 1):
            row -= 1
        else:
            column -= 1
    return list(reversed(matches))


def _topological_order(
    track_ids: set[int], order_edges: set[tuple[int, int]], medians: dict[int, float]
) -> tuple[list[int], bool]:
    outgoing: dict[int, set[int]] = {item: set() for item in track_ids}
    indegree = {item: 0 for item in track_ids}
    for upper, lower in order_edges:
        if upper == lower or lower in outgoing[upper]:
            continue
        outgoing[upper].add(lower)
        indegree[lower] += 1

    ready: list[tuple[float, int]] = []
    for item, degree in indegree.items():
        if degree == 0:
            heappush(ready, (medians.get(item, 0.5), item))
    ordered: list[int] = []
    while ready:
        _, item = heappop(ready)
        ordered.append(item)
        for child in outgoing[item]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(ready, (medians.get(child, 0.5), child))
    acyclic = len(ordered) == len(track_ids)
    if not acyclic:
        # A cycle indicates contradictory local evidence.  The fallback is
        # deterministic; the subsequent per-trace gate abstains on crossings.
        ordered = sorted(track_ids, key=lambda item: (medians.get(item, 0.5), item))
    return ordered, acyclic


def _abstain_on_crossings(labels: np.ndarray) -> int:
    rejected = 0
    for inline_index in range(labels.shape[0]):
        slab = labels[inline_index]
        known_or_sentinel = np.where(slab >= 0, slab, UNKNOWN_LABEL)
        prefix_deepest = np.maximum.accumulate(known_or_sentinel, axis=1)
        crossing = (slab >= 0) & (slab < prefix_deepest)
        rejected += int(np.count_nonzero(crossing))
        slab[crossing] = UNKNOWN_LABEL
    return rejected


def _build_reference(
    rows: list[list[_Descriptor]], maximum_global_packages: int
) -> tuple[list[_Descriptor], int]:
    counts = [len(row) for row in rows if row]
    if not counts:
        return [], 0
    package_count = int(np.rint(np.median(counts)))
    if package_count < 1 or package_count > maximum_global_packages:
        return [], package_count
    cohort = [row for row in rows if len(row) == package_count]
    if not cohort:
        return [], package_count
    reference: list[_Descriptor] = []
    for ordinal in range(package_count):
        items = [row[ordinal] for row in cohort]
        reference.append(
            _Descriptor(
                identifier=ordinal,
                midpoint=float(np.median([item.midpoint for item in items])),
                thickness=float(np.median([item.thickness for item in items])),
                support=int(np.median([item.support for item in items])),
                mean_confidence=float(
                    np.median([item.mean_confidence for item in items])
                ),
            )
        )
    return reference, package_count


def _align_to_reference(
    reference: list[_Descriptor],
    current: list[_Descriptor],
    *,
    maximum_combinations: int = 50_000,
) -> tuple[list[tuple[int, int]], float, bool]:
    """Exactly align the shorter ordered sequence to a subset of the longer.

    Jointly scoring the subset and its common vertical shift avoids the track
    birth cascade seen when a local package disappears and every deeper local id
    is renumbered.  Official outputs have 4--10 packages, so exhaustive monotone
    subset search is both deterministic and tiny.  Fragmented query sets fail
    closed instead of falling back to an unverified heuristic.
    """

    package_count, local_count = len(reference), len(current)
    if not package_count or not local_count:
        return [], 0.0, False
    if local_count <= package_count:
        candidate_count = math.comb(package_count, local_count)
        if candidate_count > maximum_combinations:
            return [], 0.0, True
        candidates = (
            (tuple(range(local_count)), slots)
            for slots in combinations(range(package_count), local_count)
        )
    else:
        candidate_count = math.comb(local_count, package_count)
        if candidate_count > maximum_combinations:
            return [], 0.0, True
        candidates = (
            (kept, tuple(range(package_count)))
            for kept in combinations(range(local_count), package_count)
        )

    best: tuple[float, tuple[int, ...], tuple[int, ...], float] | None = None
    for local_indexes, package_indexes in candidates:
        shifts = np.asarray(
            [
                current[local_index].midpoint
                - reference[package_index].midpoint
                for local_index, package_index in zip(
                    local_indexes, package_indexes, strict=True
                )
            ],
            dtype=np.float64,
        )
        common_shift = float(np.median(shifts))
        depth_residual = float(np.mean(np.abs(shifts - common_shift)))
        thickness_residual = float(
            np.mean(
                [
                    abs(
                        current[local_index].thickness
                        - reference[package_index].thickness
                    )
                    for local_index, package_index in zip(
                        local_indexes, package_indexes, strict=True
                    )
                ]
            )
        )
        confidence_residual = float(
            np.mean(
                [
                    abs(
                        current[local_index].mean_confidence
                        - reference[package_index].mean_confidence
                    )
                    for local_index, package_index in zip(
                        local_indexes, package_indexes, strict=True
                    )
                ]
            )
        )
        cost = (
            depth_residual
            + 0.10 * thickness_residual
            + 0.01 * confidence_residual
            + 0.15 * abs(common_shift)
        )
        candidate = (cost, local_indexes, package_indexes, common_shift)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, local_indexes, package_indexes, common_shift = best
    return (
        [
            (package_index, local_index)
            for local_index, package_index in zip(
                local_indexes, package_indexes, strict=True
            )
        ],
        common_shift,
        False,
    )


def _initial_complete_mapping(
    reference: list[_Descriptor], current: list[_Descriptor]
) -> tuple[np.ndarray, float, bool]:
    """Return an ordered package assignment for every supported local instance.

    The exact reference alignment provides the stable anchors.  A local query
    omitted by that injective alignment is conservatively attached to the
    nearest anchored package.  This is important for Mask2Former output: one
    geological package is sometimes split into adjacent local query fragments,
    and discarding the extra fragment creates an avoidable unknown stripe.
    """

    matches, common_shift, overflow = _align_to_reference(reference, current)
    mapping = np.full(len(current), UNKNOWN_LABEL, dtype=np.int16)
    for package_index, local_index in matches:
        mapping[local_index] = reference[package_index].identifier
    if overflow or not reference or not current:
        return mapping, common_shift, overflow

    reference_midpoints = np.asarray(
        [item.midpoint for item in reference], dtype=np.float64
    )
    for local_index, item in enumerate(current):
        if mapping[local_index] >= 0:
            continue
        adjusted_midpoint = item.midpoint - common_shift
        mapping[local_index] = int(
            np.argmin(np.abs(reference_midpoints - adjusted_midpoint))
        )
    # Descriptor order is shallow-to-deep.  The accumulate is a deterministic
    # projection onto the non-decreasing (never crossing) assignment space.
    mapping = np.maximum.accumulate(mapping)
    return mapping, common_shift, False


def _reference_cost_matrix(
    reference: list[_Descriptor],
    current: list[_Descriptor],
    common_shift: float,
) -> np.ndarray:
    costs = np.zeros((len(current), len(reference)), dtype=np.float64)
    for local_index, local in enumerate(current):
        for package_index, package in enumerate(reference):
            costs[local_index, package_index] = (
                abs((local.midpoint - common_shift) - package.midpoint) / 0.09
                + 0.35 * abs(local.thickness - package.thickness) / 0.12
                + 0.10
                * abs(local.mean_confidence - package.mean_confidence)
            )
    return costs


def _descriptor_overlap(
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    previous: list[_Descriptor],
    current: list[_Descriptor],
) -> tuple[np.ndarray, int]:
    """Count same-voxel evidence between two local descriptor sequences."""

    overlap = np.zeros((len(previous), len(current)), dtype=np.int64)
    if not previous or not current:
        return overlap, 0
    previous_lookup = np.full(
        max(item.identifier for item in previous) + 1, UNKNOWN_LABEL, dtype=np.int16
    )
    current_lookup = np.full(
        max(item.identifier for item in current) + 1, UNKNOWN_LABEL, dtype=np.int16
    )
    for index, item in enumerate(previous):
        previous_lookup[item.identifier] = index
    for index, item in enumerate(current):
        current_lookup[item.identifier] = index

    previous_flat = previous_labels.reshape(-1)
    current_flat = current_labels.reshape(-1)
    supported = (
        (previous_flat >= 0)
        & (current_flat >= 0)
        & (previous_flat < len(previous_lookup))
        & (current_flat < len(current_lookup))
    )
    if not np.any(supported):
        return overlap, 0
    previous_indexes = previous_lookup[previous_flat[supported]]
    current_indexes = current_lookup[current_flat[supported]]
    supported_descriptors = (previous_indexes >= 0) & (current_indexes >= 0)
    previous_indexes = previous_indexes[supported_descriptors].astype(
        np.int64, copy=False
    )
    current_indexes = current_indexes[supported_descriptors].astype(
        np.int64, copy=False
    )
    comparable_count = int(len(previous_indexes))
    if comparable_count:
        overlap = np.bincount(
            previous_indexes * len(current) + current_indexes,
            minlength=len(previous) * len(current),
        ).reshape(len(previous), len(current))
    return overlap, comparable_count


def _solve_monotone_consensus_row(
    unary_costs: np.ndarray,
    *,
    previous_mapping: np.ndarray | None = None,
    previous_overlap: tuple[np.ndarray, int] | None = None,
    next_mapping: np.ndarray | None = None,
    next_overlap: tuple[np.ndarray, int] | None = None,
    reference_unary_weight: float = REFERENCE_UNARY_WEIGHT,
    coalescing_penalty: float = LOCAL_FRAGMENT_COALESCING_PENALTY,
) -> np.ndarray:
    """Best non-decreasing row assignment under fixed neighbour evidence.

    Equal adjacent package ids are allowed and explicitly penalised.  They mean
    that adjacent local query fragments are coalesced into one global package;
    decreasing ids are impossible, so this operation cannot cross horizons.
    """

    local_count, package_count = unary_costs.shape
    if not local_count or not package_count:
        return np.full(local_count, UNKNOWN_LABEL, dtype=np.int16)
    scores = -reference_unary_weight * unary_costs
    package_ids = range(package_count)
    if previous_mapping is not None and previous_overlap is not None:
        overlap, comparable = previous_overlap
        if comparable:
            for package_id in package_ids:
                selected = previous_mapping == package_id
                if np.any(selected):
                    scores[:, package_id] += (
                        overlap[selected, :].sum(axis=0) / comparable
                    )
    if next_mapping is not None and next_overlap is not None:
        overlap, comparable = next_overlap
        if comparable:
            for package_id in package_ids:
                selected = next_mapping == package_id
                if np.any(selected):
                    scores[:, package_id] += (
                        overlap[:, selected].sum(axis=1) / comparable
                    )

    dynamic = np.full((local_count, package_count), -np.inf, dtype=np.float64)
    backtrack = np.zeros((local_count, package_count), dtype=np.int16)
    dynamic[0] = scores[0]
    for local_index in range(1, local_count):
        previous_row = dynamic[local_index - 1]
        best_strict_value = -np.inf
        best_strict_index = 0
        for package_id in package_ids:
            if package_id:
                candidate_index = package_id - 1
                candidate_value = previous_row[candidate_index]
                if candidate_value > best_strict_value:
                    best_strict_value = candidate_value
                    best_strict_index = candidate_index
            same_value = previous_row[package_id] - coalescing_penalty
            # A tie selects the strict transition and therefore avoids an
            # unnecessary merge while remaining fully deterministic.
            if same_value > best_strict_value or package_id == 0:
                predecessor = package_id
                predecessor_value = same_value
            else:
                predecessor = best_strict_index
                predecessor_value = best_strict_value
            dynamic[local_index, package_id] = (
                predecessor_value + scores[local_index, package_id]
            )
            backtrack[local_index, package_id] = predecessor

    package_id = int(np.argmax(dynamic[-1]))
    mapping = np.empty(local_count, dtype=np.int16)
    mapping[-1] = package_id
    for local_index in range(local_count - 1, 0, -1):
        package_id = int(backtrack[local_index, package_id])
        mapping[local_index - 1] = package_id
    return mapping


def _neighbor_consensus_mappings(
    labels: np.ndarray,
    rows: list[list[_Descriptor]],
    reference: list[_Descriptor],
    *,
    fault_shift_threshold: float,
) -> tuple[list[dict[int, int]], list[float], set[int], bool, int]:
    """Jointly refine ordered mappings using stable reference and neighbours."""

    arrays: list[np.ndarray] = []
    common_shifts: list[float] = []
    alignment_overflow = False
    for current in rows:
        mapping, common_shift, overflow = _initial_complete_mapping(reference, current)
        arrays.append(mapping)
        common_shifts.append(common_shift)
        alignment_overflow = alignment_overflow or overflow
    fault_transitions = {
        inline_index
        for inline_index in range(1, len(rows))
        if len(arrays[inline_index - 1])
        and len(arrays[inline_index])
        and abs(common_shifts[inline_index] - common_shifts[inline_index - 1])
        >= fault_shift_threshold
    }
    if reference and not alignment_overflow:
        overlaps = [
            _descriptor_overlap(
                labels[inline_index - 1],
                labels[inline_index],
                rows[inline_index - 1],
                rows[inline_index],
            )
            for inline_index in range(1, len(rows))
        ]
        unary_costs = [
            _reference_cost_matrix(reference, current, common_shift)
            for current, common_shift in zip(rows, common_shifts, strict=True)
        ]
        for pass_index in range(NEIGHBOUR_CONSENSUS_PASSES):
            inline_indexes = (
                range(len(rows))
                if pass_index % 2 == 0
                else range(len(rows) - 1, -1, -1)
            )
            for inline_index in inline_indexes:
                if not rows[inline_index]:
                    continue
                has_previous_evidence = (
                    inline_index > 0 and inline_index not in fault_transitions
                )
                has_next_evidence = (
                    inline_index + 1 < len(rows)
                    and inline_index + 1 not in fault_transitions
                )
                arrays[inline_index] = _solve_monotone_consensus_row(
                    unary_costs[inline_index],
                    previous_mapping=(
                        arrays[inline_index - 1]
                        if has_previous_evidence
                        else None
                    ),
                    previous_overlap=(
                        overlaps[inline_index - 1]
                        if has_previous_evidence
                        else None
                    ),
                    next_mapping=(
                        arrays[inline_index + 1] if has_next_evidence else None
                    ),
                    next_overlap=(
                        overlaps[inline_index] if has_next_evidence else None
                    ),
                )

    mappings = [
        {
            descriptor.identifier: int(package_id)
            for descriptor, package_id in zip(current, mapping, strict=True)
            if package_id >= 0
        }
        for current, mapping in zip(rows, arrays, strict=True)
    ]
    coalesced_instance_count = sum(
        max(
            int(np.count_nonzero(mapping >= 0))
            - len(np.unique(mapping[mapping >= 0])),
            0,
        )
        for mapping in arrays
        if len(mapping)
    )
    return (
        mappings,
        common_shifts,
        fault_transitions,
        alignment_overflow,
        int(coalesced_instance_count),
    )


def _adjacent_agreement(
    labels: np.ndarray,
    mappings: list[dict[int, int]] | None = None,
    excluded_transitions: set[int] | None = None,
) -> tuple[float | None, int]:
    previous: np.ndarray | None = None
    equal_count = 0
    comparable_count = 0
    for inline_index in range(len(labels)):
        source = labels[inline_index]
        if mappings is None:
            current = source
        else:
            current = np.full(source.shape, UNKNOWN_LABEL, dtype=np.int16)
            for local_id, package_id in mappings[inline_index].items():
                current[source == local_id] = package_id
        if previous is not None and (
            excluded_transitions is None or inline_index not in excluded_transitions
        ):
            comparable = (previous >= 0) & (current >= 0)
            comparable_count += int(np.count_nonzero(comparable))
            equal_count += int(np.count_nonzero(comparable & (previous == current)))
        previous = current.copy() if mappings is None else current
    if not comparable_count:
        return None, 0
    return float(equal_count / comparable_count), comparable_count


def _mapping_voxel_retention(
    labels: np.ndarray, mappings: list[dict[int, int]]
) -> tuple[int, int, float]:
    input_known = 0
    retained = 0
    for inline_index, mapping in enumerate(mappings):
        slab = labels[inline_index]
        input_known += int(np.count_nonzero(slab >= 0))
        for local_id in mapping:
            retained += int(np.count_nonzero(slab == local_id))
    fraction = float(retained / input_known) if input_known else 0.0
    return input_known, retained, fraction


def _mapped_quality_after_crossing_gate(
    labels: np.ndarray,
    mappings: list[dict[int, int]],
    fault_transitions: set[int],
) -> dict[str, float | int | None]:
    previous: np.ndarray | None = None
    equal = comparable = 0
    fault_equal = fault_comparable = 0
    input_known = retained = crossing_rejected = 0
    package_voxel_counts: dict[int, int] = {}
    for inline_index, mapping in enumerate(mappings):
        source = labels[inline_index]
        input_known += int(np.count_nonzero(source >= 0))
        current = np.full(source.shape, UNKNOWN_LABEL, dtype=np.int16)
        for local_id, package_id in mapping.items():
            current[source == local_id] = package_id
        mapped_known = int(np.count_nonzero(current >= 0))
        retained += mapped_known
        prefix = np.maximum.accumulate(
            np.where(current >= 0, current, UNKNOWN_LABEL), axis=1
        )
        crossing = (current >= 0) & (current < prefix)
        crossing_count = int(np.count_nonzero(crossing))
        crossing_rejected += crossing_count
        current[crossing] = UNKNOWN_LABEL
        values, counts = np.unique(current[current >= 0], return_counts=True)
        for package_id, count in zip(values, counts, strict=True):
            key = int(package_id)
            package_voxel_counts[key] = package_voxel_counts.get(key, 0) + int(count)
        if previous is not None:
            shared = (previous >= 0) & (current >= 0)
            shared_count = int(np.count_nonzero(shared))
            shared_equal = int(np.count_nonzero(shared & (previous == current)))
            comparable += shared_count
            equal += shared_equal
            if inline_index not in fault_transitions:
                fault_comparable += shared_count
                fault_equal += shared_equal
        previous = current
    final_retained = retained - crossing_rejected
    retained_package_voxels = sum(package_voxel_counts.values())
    dominant_package_fraction = (
        float(max(package_voxel_counts.values()) / retained_package_voxels)
        if retained_package_voxels
        else None
    )
    return {
        "agreement": float(equal / comparable) if comparable else None,
        "comparable": comparable,
        "fault_exempt_agreement": (
            float(fault_equal / fault_comparable) if fault_comparable else None
        ),
        "fault_exempt_comparable": fault_comparable,
        "input_known": input_known,
        "retained_before_crossing": retained,
        "crossing_rejected": crossing_rejected,
        "retained_fraction": (
            float(final_retained / input_known) if input_known else 0.0
        ),
        "dominant_package_fraction": dominant_package_fraction,
    }


def reconcile_global_packages(
    local_labels: np.ndarray,
    confidence: np.ndarray,
    *,
    max_gap_inlines: int = 2,
    fault_shift_threshold: float = 0.08,
    in_place: bool = False,
    maximum_global_packages: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map local instance ids to globally ordered package ids.

    Unknown voxels remain ``-1``.  Matching is monotone, so an association can
    skip a missing package but can never swap the stratigraphic order.
    """

    labels = np.asarray(local_labels)
    scores = np.asarray(confidence)
    if labels.ndim != 3 or scores.shape != labels.shape:
        raise ValueError("local_labels and confidence must share [inline,xline,sample]")
    inline_descriptors = [
        _describe_inline(labels[index], scores[index]) for index in range(len(labels))
    ]
    reference, detected_package_count = _build_reference(
        inline_descriptors, maximum_global_packages
    )
    reference_cohort_count = sum(
        len(row) == detected_package_count for row in inline_descriptors
    )
    (
        mappings,
        common_shifts,
        fault_transition_indexes,
        alignment_overflow,
        coalesced_instance_count,
    ) = _neighbor_consensus_mappings(
        labels,
        inline_descriptors,
        reference,
        fault_shift_threshold=fault_shift_threshold,
    )

    transition_rows: list[dict[str, Any]] = []
    fault_edges = 0
    matched_count = 0
    unmatched_count = 0
    for inline_index, (current, mapping, common_shift) in enumerate(
        zip(inline_descriptors, mappings, common_shifts, strict=True)
    ):
        is_fault_shift = inline_index in fault_transition_indexes
        fault_edges += int(is_fault_shift)
        if inline_index:
            matched_count += len(mapping)
        unmatched_count += max(len(current) - len(mapping), 0)
        transition_rows.append(
            {
                "inline_index": inline_index,
                "local_package_count": len(current),
                "matched_package_count": len(mapping),
                "unmatched_package_count": max(len(current) - len(mapping), 0),
                "estimated_common_shift_fraction": round(common_shift, 6),
                "fault_or_shift_transition": is_fault_shift,
            }
        )
    longest_mapping_gap = 0
    current_mapping_gap = 0
    for mapping in mappings:
        if mapping:
            current_mapping_gap = 0
        else:
            current_mapping_gap += 1
            longest_mapping_gap = max(longest_mapping_gap, current_mapping_gap)

    before_agreement, before_comparable = _adjacent_agreement(labels)
    candidate_agreement, candidate_comparable = _adjacent_agreement(labels, mappings)
    fault_exempt_before, fault_exempt_before_comparable = _adjacent_agreement(
        labels, excluded_transitions=fault_transition_indexes
    )
    fault_exempt_candidate, fault_exempt_candidate_comparable = _adjacent_agreement(
        labels,
        mappings,
        excluded_transitions=fault_transition_indexes,
    )
    _, _, candidate_voxel_retention = (
        _mapping_voxel_retention(labels, mappings)
    )
    comparable_retention_candidate = (
        float(candidate_comparable / before_comparable)
        if before_comparable
        else 0.0
    )
    predicted = _mapped_quality_after_crossing_gate(
        labels, mappings, fault_transition_indexes
    )
    predicted_agreement = predicted["agreement"]
    predicted_fault_exempt_agreement = predicted["fault_exempt_agreement"]
    predicted_comparable = int(predicted["comparable"])
    predicted_comparable_retention = (
        float(predicted_comparable / before_comparable)
        if before_comparable
        else 0.0
    )
    predicted_dominant_fraction = predicted["dominant_package_fraction"]
    evidence_ready = bool(
        len(labels) >= 2
        and detected_package_count >= 2
        and len(reference) >= 2
        and reference
        and matched_count > 0
        and predicted_agreement is not None
        and predicted_agreement >= max(float(before_agreement or 0.0), 0.65)
        and predicted_fault_exempt_agreement is not None
        and predicted_fault_exempt_agreement
        >= max(float(fault_exempt_before or 0.0), 0.65)
        and float(predicted["retained_fraction"]) >= 0.85
        and predicted_comparable_retention >= 0.80
        and predicted_dominant_fraction is not None
        and predicted_dominant_fraction < MAXIMUM_DOMINANT_PACKAGE_FRACTION
        and longest_mapping_gap <= max_gap_inlines
    )
    if not reference or alignment_overflow or not evidence_ready:
        fallback = labels if in_place else labels.astype(np.int16, copy=True)
        if detected_package_count > maximum_global_packages:
            reason = "global_package_limit_exceeded"
        elif detected_package_count < 2:
            reason = "insufficient_global_packages"
        elif alignment_overflow:
            reason = "reference_alignment_combination_budget_exceeded"
        elif not reference:
            reason = "global_reference_unavailable"
        elif float(predicted["retained_fraction"]) < 0.85:
            reason = "post_crossing_retention_below_floor"
        elif predicted_comparable_retention < 0.80:
            reason = "post_crossing_comparable_coverage_below_floor"
        elif (
            predicted_dominant_fraction is not None
            and predicted_dominant_fraction >= MAXIMUM_DOMINANT_PACKAGE_FRACTION
        ):
            reason = "dominant_global_package_fraction_above_ceiling"
        elif longest_mapping_gap > max_gap_inlines:
            reason = "reference_alignment_gap_exceeded"
        else:
            reason = "adjacent_inline_consistency_not_improved"
        return fallback, {
            "schema_version": RECONCILIATION_SCHEMA,
            "method": "neighbour-consensus-monotone-coalescing",
            "association_scope": "volume-reference-plus-adjacent-evidence",
            "degraded": True,
            "global_display_ready": False,
            "output_semantics": "local_inline_fallback",
            "degraded_reason": reason,
            "maximum_global_packages": int(maximum_global_packages),
            "detected_global_package_count": int(detected_package_count),
            "processed_inline_count": len(labels),
            "inline_with_packages_count": sum(bool(item) for item in inline_descriptors),
            "global_package_count": 0,
            "global_horizon_count": 0,
            "matched_transition_count": matched_count,
            "unmatched_instance_count": unmatched_count,
            "coalesced_local_instance_count": coalesced_instance_count,
            "fault_or_shift_transition_count": fault_edges,
            "continuity_ratio": 0.0,
            "order_graph_acyclic": True,
            "non_crossing_verified": False,
            "crossing_voxel_abstention_count": 0,
            "adjacent_known_voxel_agreement_before": before_agreement,
            "adjacent_known_voxel_agreement_candidate": candidate_agreement,
            "adjacent_comparable_voxel_count_before": before_comparable,
            "adjacent_comparable_voxel_count_candidate": candidate_comparable,
            "association_retained_voxel_fraction_candidate": candidate_voxel_retention,
            "adjacent_comparable_voxel_retention_candidate": comparable_retention_candidate,
            "fault_exempt_adjacent_agreement_before": fault_exempt_before,
            "fault_exempt_adjacent_agreement_candidate": fault_exempt_candidate,
            "fault_exempt_comparable_voxel_count_before": fault_exempt_before_comparable,
            "fault_exempt_comparable_voxel_count_candidate": fault_exempt_candidate_comparable,
            "predicted_post_crossing_agreement": predicted_agreement,
            "predicted_post_crossing_fault_exempt_agreement": predicted_fault_exempt_agreement,
            "predicted_post_crossing_retained_voxel_fraction": predicted[
                "retained_fraction"
            ],
            "predicted_post_crossing_comparable_voxel_retention": predicted_comparable_retention,
            "dominant_global_package_fraction": predicted_dominant_fraction,
            "maximum_dominant_package_fraction": MAXIMUM_DOMINANT_PACKAGE_FRACTION,
            "predicted_crossing_voxel_abstention_count": predicted[
                "crossing_rejected"
            ],
            "max_gap_inlines": int(max_gap_inlines),
            "maximum_observed_mapping_gap_inlines": longest_mapping_gap,
            "transition_receipts": transition_rows,
        }

    global_labels = (
        labels
        if in_place and labels.dtype == np.int16 and labels.flags.writeable
        else np.full(labels.shape, UNKNOWN_LABEL, dtype=np.int16)
    )
    input_known_voxel_count = 0
    retained_voxel_count = 0
    for inline_index, mapping in enumerate(mappings):
        source_inline = labels[inline_index].copy() if global_labels is labels else labels[inline_index]
        input_known_voxel_count += int(np.count_nonzero(source_inline >= 0))
        if global_labels is labels:
            global_labels[inline_index].fill(UNKNOWN_LABEL)
        for local_id, package_id in mapping.items():
            selected = source_inline == local_id
            retained_voxel_count += int(np.count_nonzero(selected))
            global_labels[inline_index][selected] = package_id
    crossing_rejected = _abstain_on_crossings(global_labels)

    package_voxel_counts = np.zeros(detected_package_count, dtype=np.int64)
    for inline_index in range(global_labels.shape[0]):
        known_values = global_labels[inline_index]
        known_values = known_values[known_values >= 0]
        if len(known_values):
            package_voxel_counts += np.bincount(
                known_values.astype(np.int64, copy=False),
                minlength=detected_package_count,
            )[:detected_package_count]
    final_known_voxel_count = int(package_voxel_counts.sum())
    dominant_package_fraction = (
        float(package_voxel_counts.max() / final_known_voxel_count)
        if final_known_voxel_count
        else None
    )

    non_crossing = True
    for inline_index in range(global_labels.shape[0]):
        slab = global_labels[inline_index]
        prefix = np.maximum.accumulate(
            np.where(slab >= 0, slab, UNKNOWN_LABEL), axis=1
        )
        if np.any((slab >= 0) & (slab < prefix)):
            non_crossing = False
            break
    after_agreement, after_comparable = _adjacent_agreement(global_labels)
    fault_exempt_after, fault_exempt_after_comparable = _adjacent_agreement(
        global_labels, excluded_transitions=fault_transition_indexes
    )
    final_voxel_retention = (
        float((retained_voxel_count - crossing_rejected) / input_known_voxel_count)
        if input_known_voxel_count
        else 0.0
    )
    comparable_retention_after = (
        float(after_comparable / before_comparable) if before_comparable else 0.0
    )
    descriptor_denominator = sum(len(row) for row in inline_descriptors[1:])
    final_contract_ready = bool(
        non_crossing
        and detected_package_count >= 2
        and len(labels) >= 2
        and matched_count > 0
        and after_agreement is not None
        and after_agreement >= max(float(before_agreement or 0.0), 0.65)
        and fault_exempt_after is not None
        and fault_exempt_after >= max(float(fault_exempt_before or 0.0), 0.65)
        and final_voxel_retention >= 0.85
        and comparable_retention_after >= 0.80
        and dominant_package_fraction is not None
        and dominant_package_fraction < MAXIMUM_DOMINANT_PACKAGE_FRACTION
    )
    if not final_contract_ready:
        raise RuntimeError(
            "Global reconciliation changed after its sealed preflight; no result "
            "was published. This indicates an internal deterministic-contract error."
        )
    receipt: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA,
        "method": "neighbour-consensus-monotone-coalescing",
        "degraded": False,
        "global_display_ready": final_contract_ready,
        "output_semantics": "global_ordered_package_id",
        "maximum_global_packages": int(maximum_global_packages),
        "processed_inline_count": len(labels),
        "inline_with_packages_count": sum(bool(item) for item in inline_descriptors),
        "association_scope": "volume-reference-plus-adjacent-evidence",
        "neighbor_consensus_pass_count": NEIGHBOUR_CONSENSUS_PASSES,
        "reference_unary_weight": REFERENCE_UNARY_WEIGHT,
        "local_fragment_coalescing_penalty": LOCAL_FRAGMENT_COALESCING_PENALTY,
        "input_confidence_gate_applied": True,
        "descriptor_weighting": "confidence-weighted-depth",
        "minimum_instance_support": "max(2, inline_voxels*1e-5)",
        "gap_handling": "volume-reference-with-bounded-empty-inline-run",
        "max_gap_inlines": int(max_gap_inlines),
        "maximum_observed_mapping_gap_inlines": longest_mapping_gap,
        "reference_package_count_strategy": "median-local-package-count",
        "reference_cohort_inline_count": reference_cohort_count,
        "reference_packages": [
            {
                "package_id": item.identifier,
                "normalized_midpoint": item.midpoint,
                "normalized_thickness": item.thickness,
                "mean_confidence": item.mean_confidence,
            }
            for item in reference
        ],
        "global_package_count": detected_package_count,
        "global_horizon_count": max(detected_package_count - 1, 0),
        "matched_transition_count": matched_count,
        "unmatched_instance_count": unmatched_count,
        "coalesced_local_instance_count": coalesced_instance_count,
        "fault_or_shift_transition_count": fault_edges,
        "continuity_ratio": (
            float(matched_count / descriptor_denominator)
            if descriptor_denominator
            else 0.0
        ),
        "association_retained_voxel_fraction": final_voxel_retention,
        "dominant_global_package_fraction": dominant_package_fraction,
        "maximum_dominant_package_fraction": MAXIMUM_DOMINANT_PACKAGE_FRACTION,
        "association_retained_voxel_fraction_before_crossing_gate": (
            float(retained_voxel_count / input_known_voxel_count)
            if input_known_voxel_count
            else 0.0
        ),
        "adjacent_known_voxel_agreement_before": before_agreement,
        "adjacent_known_voxel_agreement_after": after_agreement,
        "adjacent_known_voxel_agreement_improvement": (
            float(after_agreement - before_agreement)
            if after_agreement is not None and before_agreement is not None
            else None
        ),
        "adjacent_comparable_voxel_count_before": before_comparable,
        "adjacent_comparable_voxel_count_after": after_comparable,
        "adjacent_comparable_voxel_retention": comparable_retention_after,
        "fault_exempt_adjacent_agreement_before": fault_exempt_before,
        "fault_exempt_adjacent_agreement_after": fault_exempt_after,
        "fault_exempt_comparable_voxel_count_before": fault_exempt_before_comparable,
        "fault_exempt_comparable_voxel_count_after": fault_exempt_after_comparable,
        "fault_transition_indexes": sorted(fault_transition_indexes),
        "preflight_predicted_crossing_voxel_abstention_count": predicted[
            "crossing_rejected"
        ],
        "order_graph_acyclic": True,
        "non_crossing_verified": non_crossing,
        "crossing_voxel_abstention_count": crossing_rejected,
        "transition_receipts": transition_rows,
    }
    return global_labels, receipt


def extract_horizon_surfaces(
    global_labels: np.ndarray,
    confidence: np.ndarray,
    global_package_count: int,
    maximum_dense_bytes: int = 512 * 1024 * 1024,
    *,
    minimum_finite_trace_fraction: float = DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
    minimum_largest_component_fraction: float = (
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
    ),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Derive ordered surfaces and an auditable, per-surface display receipt.

    The dense NPZ payload remains authoritative and complete: this function
    never drops or renumbers a horizon.  ``display_eligible`` is metadata for
    downstream viewers.  Connected support is evaluated on the 2-D
    Inline/Crossline finite-trace mask with deterministic 4-neighbour
    connectivity (the surface analogue of 6-neighbour voxel connectivity).
    """

    labels = np.asarray(global_labels)
    scores = np.asarray(confidence, dtype=np.float32)
    horizon_count = max(int(global_package_count) - 1, 0)
    required_bytes = horizon_count * labels.shape[0] * labels.shape[1] * 4 * 2
    if required_bytes > maximum_dense_bytes:
        raise RuntimeError(
            "Dense horizon surface budget exceeded: "
            f"required_bytes={required_bytes}, maximum_dense_bytes={maximum_dense_bytes}. "
            "Global surface export was not allocated."
        )
    depths = np.full((horizon_count, labels.shape[0], labels.shape[1]), np.nan, np.float32)
    horizon_scores = np.full_like(depths, np.nan)
    xline_count, sample_count = labels.shape[1], labels.shape[2]
    trace_indexes = np.repeat(np.arange(xline_count, dtype=np.int64), sample_count)
    sample_indexes = np.tile(np.arange(sample_count, dtype=np.int32), xline_count)
    for inline_index in range(labels.shape[0]):
        flat_labels = labels[inline_index].reshape(-1)
        known = (flat_labels >= 1) & (flat_labels <= horizon_count)
        if not np.any(known):
            continue
        positions = (
            (flat_labels[known].astype(np.int64) - 1) * xline_count
            + trace_indexes[known]
        )
        first = np.full(horizon_count * xline_count, sample_count, dtype=np.int32)
        np.minimum.at(first, positions, sample_indexes[known])
        first = first.reshape(horizon_count, xline_count)
        present = first < sample_count
        depths[:, inline_index, :][present] = first[present].astype(np.float32)

        inline_scores = scores[inline_index]
        padded = np.pad(inline_scores, ((0, 0), (1, 1)), mode="edge")
        neighbourhood = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
        safe_first = np.minimum(first, sample_count - 1)
        sampled = np.take_along_axis(
            neighbourhood[None, :, :], safe_first[:, :, None], axis=2
        )[:, :, 0]
        horizon_scores[:, inline_index, :][present] = sampled[present]
    horizon_ids = np.arange(horizon_count, dtype=np.int16)
    lower_ids = np.arange(1, int(global_package_count), dtype=np.int16)
    display_receipt = build_horizon_display_receipt(
        depths,
        horizon_ids,
        lower_ids,
        minimum_finite_trace_fraction=minimum_finite_trace_fraction,
        minimum_largest_component_fraction=minimum_largest_component_fraction,
    )
    return depths, horizon_scores, horizon_ids, lower_ids, display_receipt


def _validate_display_fraction(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must stay within [0, 1]")
    return result


def _four_connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Return row-major deterministic component sizes without scipy.

    Each contiguous run in one Inline row is one union-find node.  Runs in
    adjacent rows are joined only when their Crossline intervals overlap,
    which is exactly 4-neighbour connectivity and avoids a Python object per
    finite trace on full F3 volumes.
    """

    supported = np.asarray(mask, dtype=bool)
    if supported.ndim != 2:
        raise ValueError("surface support mask must use [INLINE, CROSSLINE]")
    parents: list[int] = []
    sizes: list[int] = []

    def find(node: int) -> int:
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            parent = parents[node]
            parents[node] = root
            node = parent
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        # A stable minimum-node root makes the result independent of hash or
        # set iteration order.
        root, child = sorted((first_root, second_root))
        parents[child] = root
        sizes[root] += sizes[child]
        sizes[child] = 0

    previous_runs: list[tuple[int, int, int]] = []
    for row in supported:
        padded = np.pad(row, (1, 1), constant_values=False)
        transitions = np.diff(padded.astype(np.int8, copy=False))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        current_runs: list[tuple[int, int, int]] = []
        previous_cursor = 0
        for start_raw, stop_raw in zip(starts, stops, strict=True):
            start, stop = int(start_raw), int(stop_raw)
            node = len(parents)
            parents.append(node)
            sizes.append(stop - start)
            while (
                previous_cursor < len(previous_runs)
                and previous_runs[previous_cursor][1] <= start
            ):
                previous_cursor += 1
            overlap_cursor = previous_cursor
            while (
                overlap_cursor < len(previous_runs)
                and previous_runs[overlap_cursor][0] < stop
            ):
                union(node, previous_runs[overlap_cursor][2])
                overlap_cursor += 1
            current_runs.append((start, stop, node))
        previous_runs = current_runs
    return [sizes[index] for index, parent in enumerate(parents) if parent == index]


def build_horizon_display_receipt(
    depths: np.ndarray,
    horizon_ids: np.ndarray,
    lower_package_ids: np.ndarray,
    *,
    minimum_finite_trace_fraction: float = DEFAULT_MINIMUM_FINITE_TRACE_FRACTION,
    minimum_largest_component_fraction: float = (
        DEFAULT_MINIMUM_LARGEST_COMPONENT_FRACTION
    ),
) -> dict[str, Any]:
    """Measure every raw surface and decide display eligibility without editing it."""

    surfaces = np.asarray(depths)
    ids = np.asarray(horizon_ids)
    lower_ids = np.asarray(lower_package_ids)
    if surfaces.ndim != 3:
        raise ValueError("depths must use [HORIZON, INLINE, CROSSLINE]")
    horizon_count, inline_count, xline_count = surfaces.shape
    if ids.shape != (horizon_count,) or lower_ids.shape != (horizon_count,):
        raise ValueError("horizon_ids and lower_package_ids must match depths")
    if len(np.unique(ids)) != horizon_count:
        raise ValueError("horizon_ids must be unique")
    minimum_finite = _validate_display_fraction(
        minimum_finite_trace_fraction, "minimum_finite_trace_fraction"
    )
    minimum_largest = _validate_display_fraction(
        minimum_largest_component_fraction,
        "minimum_largest_component_fraction",
    )
    total_trace_count = int(inline_count * xline_count)
    rows: list[dict[str, Any]] = []
    suppressed: list[int] = []
    for index in range(horizon_count):
        finite = np.isfinite(surfaces[index])
        finite_trace_count = int(np.count_nonzero(finite))
        inline_support_count = int(np.count_nonzero(np.any(finite, axis=1)))
        xline_support_count = int(np.count_nonzero(np.any(finite, axis=0)))
        component_sizes = _four_connected_component_sizes(finite)
        largest_component_trace_count = max(component_sizes, default=0)
        finite_fraction = (
            float(finite_trace_count / total_trace_count)
            if total_trace_count
            else 0.0
        )
        largest_fraction = (
            float(largest_component_trace_count / finite_trace_count)
            if finite_trace_count
            else 0.0
        )
        reasons: list[str] = []
        if finite_fraction < minimum_finite:
            reasons.append("finite_trace_fraction_below_minimum")
        if largest_fraction < minimum_largest:
            reasons.append("largest_component_fraction_below_minimum")
        horizon_id = int(ids[index])
        eligible = not reasons
        if not eligible:
            suppressed.append(horizon_id)
        rows.append(
            {
                "horizon_id": horizon_id,
                "lower_package_id": int(lower_ids[index]),
                "finite_trace_count": finite_trace_count,
                "total_trace_count": total_trace_count,
                "finite_trace_fraction": finite_fraction,
                "inline_support_count": inline_support_count,
                "inline_support_fraction": (
                    float(inline_support_count / inline_count)
                    if inline_count
                    else 0.0
                ),
                "xline_support_count": xline_support_count,
                "xline_support_fraction": (
                    float(xline_support_count / xline_count)
                    if xline_count
                    else 0.0
                ),
                "connected_component_count": len(component_sizes),
                "largest_component_trace_count": largest_component_trace_count,
                "largest_component_fraction": largest_fraction,
                "display_eligible": eligible,
                "reasons": reasons,
            }
        )
    return {
        "schema_version": HORIZON_DISPLAY_GATE_SCHEMA,
        "minimum_finite_trace_fraction": minimum_finite,
        "minimum_largest_component_fraction": minimum_largest,
        "surface_connectivity": SURFACE_COMPONENT_CONNECTIVITY,
        "volume_connectivity_analogue": 6,
        "finite_trace_fraction_denominator": "dense_inline_xline_grid",
        "axis_support_fraction_denominator": "dense_axis_count",
        "largest_component_fraction_denominator": "finite_trace_count",
        "raw_horizon_count": horizon_count,
        "display_horizon_count": horizon_count - len(suppressed),
        "suppressed_horizon_ids": suppressed,
        "horizon_surface_receipts": rows,
    }
