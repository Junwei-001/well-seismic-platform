"""Typed, read-only contracts for published model and result artifacts.

The release catalog deliberately does not imply that a checkpoint is runnable or
scientifically supported.  Those are two independent states and are kept
separate in the public contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ScientificStatus = Literal[
    "unassessed",
    "candidate",
    "selected_for_refit",
    "validated",
    "conditional",
    "failed",
    "rejected",
]
RuntimeStatus = Literal[
    "runnable",
    "adapter_required",
    "precomputed_only",
    "blocked",
    "unavailable",
]
LayerKind = Literal["surface", "volume", "well_curve", "table", "report"]


@dataclass(frozen=True, kw_only=True)
class ReleaseArtifact:
    """One explicitly allow-listed artifact exposed by a release."""

    id: str
    name: str
    role: str
    kind: str
    path: str
    relative_path: str
    exists: bool
    media_type: str
    layer: LayerKind | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    integrity_status: str = "not_checked"
    unit: str | None = None
    axis_order: str | None = None
    uncertainty_definition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ArtifactRelease:
    """A versioned result bundle with explicit scientific/runtime semantics."""

    id: str
    name: str
    version: str
    task_id: str
    description: str
    scientific_status: ScientificStatus
    runtime_status: RuntimeStatus
    evidence_class: str
    scope: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    artifacts: tuple[ReleaseArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "wellfuse"
    release_type: str = "artifact"

    @property
    def available(self) -> bool:
        return any(artifact.exists for artifact in self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available"] = self.available
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        # Compatibility alias for clients introduced before the generic layer
        # descriptor was finalised.
        payload["precomputed_artifacts"] = payload["artifacts"]
        return payload


@dataclass(frozen=True, kw_only=True)
class ModelRelease(ArtifactRelease):
    """A release backed by a model, even if no online adapter is installed yet."""

    model_id: str
    runner_id: str | None = None
    release_type: str = "model"
