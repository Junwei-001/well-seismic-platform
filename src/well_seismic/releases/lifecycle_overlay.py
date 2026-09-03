"""Read-only, fail-closed projection of a WellFuse lifecycle registry.

The platform never opens the lifecycle SQLite database and never mutates a
pointer.  It consumes only a content-addressed registry snapshot.  A candidate
is exposed only when its task/split manifests and every declared artifact can
be verified under the configured WellFuse artifact root.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import ModelRelease, ReleaseArtifact


REGISTRY_SCHEMA = "wellfuse.lifecycle.registry.v1"
CANDIDATE_SCHEMA = "wellfuse.lifecycle.candidate.v1"
ARTIFACT_SCHEMA = "wellfuse.lifecycle.artifact_ref.v1"
TASK_SPEC_SCHEMA = "wellfuse.lifecycle.task_spec.v1"
SPLIT_MANIFEST_SCHEMA = "wellfuse.lifecycle.split_manifest.v1"

_POINTER_TYPES = frozenset(("scientific_incumbent", "runtime_default"))
_SCIENTIFIC_STATUSES = frozenset(
    (
        "unassessed",
        "candidate",
        "selected_for_refit",
        "validated",
        "conditional",
        "failed",
        "rejected",
    )
)
_RUNTIME_STATUSES = frozenset(
    ("runnable", "adapter_required", "precomputed_only", "blocked", "unavailable")
)
_DISALLOWED_PATH_PARTS = frozenset(
    ("label", "labels", "target", "targets", "cache", "pytest", ".pytest_cache", "tmp", "temp")
)
_DISALLOWED_ROLES = frozenset(("label", "labels", "target", "targets", "ground_truth"))
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE_STAGES = (
    "validate",
    "prepare",
    "baseline",
    "optimize",
    "promote",
    "refit",
    "verify",
)
_PRIMARY_PARTITIONS = ("selection_train", "selection_validation", "frozen_test")


class LifecycleOverlayError(RuntimeError):
    """The sealed lifecycle snapshot cannot be safely projected."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LifecycleOverlayError("lifecycle document is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LifecycleOverlayError(f"{context} must be an object")
    return dict(value)


def _sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise LifecycleOverlayError(f"{context} must be an array")
    return value


def _token(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleOverlayError(f"{context} must be a non-empty string")
    return value.strip()


def _sha(value: object, context: str) -> str:
    result = _token(value, context)
    if _HEX_SHA256.fullmatch(result) is None:
        raise LifecycleOverlayError(f"{context} is not a lowercase SHA-256")
    return result


def _strict_fields(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    unknown = set(payload).difference(expected)
    missing = expected.difference(payload)
    if unknown or missing:
        raise LifecycleOverlayError(
            f"{context} fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _unique_tokens(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    values = _sequence(value, context)
    result = [_token(item, f"{context}[]") for item in values]
    if not allow_empty and not result:
        raise LifecycleOverlayError(f"{context} must not be empty")
    if len(set(result)) != len(result):
        raise LifecycleOverlayError(f"{context} contains duplicate values")
    return result


def _content_digest(payload: Mapping[str, Any], digest_field: str, context: str) -> str:
    recorded = _sha(payload.get(digest_field), f"{context}.{digest_field}")
    content = {key: value for key, value in payload.items() if key != digest_field}
    if _canonical_sha256(content) != recorded:
        raise LifecycleOverlayError(f"{context} canonical SHA-256 differs")
    return recorded


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()
    if not normalized:
        raise LifecycleOverlayError("candidate cannot form a safe release id")
    return normalized


def _infer_layer(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in {".json", ".md", ".txt", ".yaml", ".yml"}:
        return "report"
    if suffix in {".csv", ".tsv", ".xlsx"}:
        return "table"
    if suffix == ".las":
        return "well_curve"
    if suffix in {".npy", ".npz", ".zarr", ".sgy", ".segy"}:
        return "volume"
    return None


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    role: str
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


class LifecycleRegistryOverlay:
    """Strict projection of selected candidates from one immutable snapshot."""

    schema_version = "well-seismic.lifecycle-overlay.v1"

    def __init__(self, snapshot_path: str | Path, *, artifact_root: str | Path) -> None:
        self.snapshot_path = Path(snapshot_path).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.registry_sha256 = ""
        self._releases: tuple[ModelRelease, ...] = ()
        self._candidates: dict[tuple[str, str], ModelRelease] = {}
        self._pointers: dict[tuple[str, str], str] = {}
        self._load()

    def list_releases(self) -> list[ModelRelease]:
        return list(self._releases)

    def resolve_candidate(self, *, task_id: str, candidate_id: str) -> ModelRelease | None:
        """Return one verified immutable candidate, selected or not.

        Static platform registries may describe adapters and capabilities, but
        they must not invent scientific/runtime state.  This lookup is the
        narrow bridge used to bind a legacy public model id to the candidate
        that is actually present in the sealed lifecycle snapshot.
        """

        return self._candidates.get((str(task_id).strip(), str(candidate_id).strip()))

    def resolve_pointer(self, *, task_id: str, pointer_type: str) -> ModelRelease | None:
        """Resolve a verified task pointer without falling back to static state."""

        task = str(task_id).strip()
        pointer = str(pointer_type).strip()
        if pointer not in _POINTER_TYPES:
            raise ValueError(f"unknown lifecycle pointer: {pointer}")
        candidate_id = self._pointers.get((task, pointer))
        if candidate_id is None:
            return None
        return self._candidates.get((task, candidate_id))

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": True,
            "read_only": True,
            "fail_closed": True,
            "snapshot_path": str(self.snapshot_path),
            "registry_sha256": self.registry_sha256,
            "release_count": len(self._releases),
            "candidate_count": len(self._candidates),
            "pointer_count": len(self._pointers),
        }

    def _load(self) -> None:
        if not self.snapshot_path.is_file():
            raise LifecycleOverlayError(f"lifecycle snapshot is missing: {self.snapshot_path}")
        if not _within(self.snapshot_path, self.artifact_root):
            raise LifecycleOverlayError("lifecycle snapshot escapes the artifact root")
        try:
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LifecycleOverlayError("lifecycle snapshot is not readable JSON") from exc
        document = _mapping(payload, "registry")
        _strict_fields(
            document,
            {
                "schema_version",
                "candidate_count",
                "candidates",
                "pointers",
                "pointer_transitions",
                "registry_sha256",
            },
            "registry",
        )
        if document["schema_version"] != REGISTRY_SCHEMA:
            raise LifecycleOverlayError("unsupported lifecycle registry schema")
        self.registry_sha256 = _content_digest(document, "registry_sha256", "registry")
        candidates_raw = _sequence(document["candidates"], "registry.candidates")
        count = document["candidate_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count != len(candidates_raw):
            raise LifecycleOverlayError("registry candidate_count differs from candidates")
        candidates: dict[str, tuple[dict[str, Any], tuple[_VerifiedArtifact, ...]]] = {}
        for index, value in enumerate(candidates_raw):
            candidate = _mapping(value, f"registry.candidates[{index}]")
            identifier, artifacts = self._verify_candidate(candidate, index=index)
            if identifier in candidates:
                raise LifecycleOverlayError(f"duplicate lifecycle candidate: {identifier}")
            candidates[identifier] = (candidate, artifacts)

        pointers = self._verify_pointers(document["pointers"], candidates)
        if not isinstance(document["pointer_transitions"], list):
            raise LifecycleOverlayError("registry.pointer_transitions must be an array")
        selected: dict[str, set[str]] = {}
        generations: dict[tuple[str, str], int] = {}
        for pointer in pointers:
            candidate_id = pointer["candidate_id"]
            selected.setdefault(candidate_id, set()).add(pointer["pointer_type"])
            generations[(candidate_id, pointer["pointer_type"])] = pointer["generation"]
        all_releases = {
            (str(candidate["task_id"]), candidate_id): self._release(
                candidate,
                artifacts,
                selected.get(candidate_id, set()),
                generations,
            )
            for candidate_id, (candidate, artifacts) in candidates.items()
        }
        self._candidates = all_releases
        self._pointers = {
            (str(pointer["task_id"]), str(pointer["pointer_type"])): str(
                pointer["candidate_id"]
            )
            for pointer in pointers
        }
        self._releases = tuple(
            all_releases[(str(candidate["task_id"]), candidate_id)]
            for candidate_id, (candidate, _artifacts) in candidates.items()
            if candidate_id in selected
        )
        if len({release.id for release in self._releases}) != len(self._releases):
            raise LifecycleOverlayError("selected candidates produce colliding release ids")

    def _verify_candidate(
        self, payload: dict[str, Any], *, index: int
    ) -> tuple[str, tuple[_VerifiedArtifact, ...]]:
        context = f"registry.candidates[{index}]"
        _strict_fields(
            payload,
            {
                "schema_version",
                "candidate_id",
                "task_id",
                "task_spec_sha256",
                "split_manifest_sha256",
                "scientific_status",
                "runtime_status",
                "evidence_class",
                "artifacts",
                "metrics",
                "metadata",
                "candidate_sha256",
            },
            context,
        )
        if payload["schema_version"] != CANDIDATE_SCHEMA:
            raise LifecycleOverlayError(f"{context} has unsupported schema")
        _content_digest(payload, "candidate_sha256", context)
        candidate_id = _token(payload["candidate_id"], f"{context}.candidate_id")
        _token(payload["task_id"], f"{context}.task_id")
        task_sha = _sha(payload["task_spec_sha256"], f"{context}.task_spec_sha256")
        split_sha = _sha(payload["split_manifest_sha256"], f"{context}.split_manifest_sha256")
        if payload["scientific_status"] not in _SCIENTIFIC_STATUSES:
            raise LifecycleOverlayError(f"{context} has unknown scientific_status")
        if payload["runtime_status"] not in _RUNTIME_STATUSES:
            raise LifecycleOverlayError(f"{context} has unknown runtime_status")
        _token(payload["evidence_class"], f"{context}.evidence_class")
        _mapping(payload["metrics"], f"{context}.metrics")
        _mapping(payload["metadata"], f"{context}.metadata")

        artifacts_raw = _sequence(payload["artifacts"], f"{context}.artifacts")
        if not artifacts_raw:
            raise LifecycleOverlayError(f"{context}.artifacts is empty")
        artifacts = tuple(
            self._verify_artifact(_mapping(item, f"{context}.artifacts[{position}]"), context)
            for position, item in enumerate(artifacts_raw)
        )
        roles = [artifact.role for artifact in artifacts]
        if len(set(roles)) != len(roles):
            raise LifecycleOverlayError(f"{context} contains duplicate artifact roles")
        if len({_safe_identifier(role) for role in roles}) != len(roles):
            raise LifecycleOverlayError(f"{context} artifact roles produce colliding public ids")
        by_role = {artifact.role: artifact for artifact in artifacts}
        for required in ("task_spec", "split_manifest"):
            if required not in by_role:
                raise LifecycleOverlayError(f"{context} is missing required {required} artifact")
        self._verify_task_spec(
            by_role["task_spec"].path,
            expected_digest=task_sha,
            expected_task_id=str(payload["task_id"]),
        )
        self._verify_split_manifest(
            by_role["split_manifest"].path,
            expected_digest=split_sha,
        )
        for artifact in artifacts:
            if artifact.path.suffix.casefold() == ".json":
                try:
                    parsed = json.loads(artifact.path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise LifecycleOverlayError(
                        f"declared JSON artifact is invalid: {artifact.relative_path}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise LifecycleOverlayError(
                        f"declared JSON artifact is not an object: {artifact.relative_path}"
                    )
        return candidate_id, artifacts

    def _verify_artifact(self, payload: dict[str, Any], context: str) -> _VerifiedArtifact:
        _strict_fields(
            payload,
            {"schema_version", "path", "sha256", "size_bytes", "role"},
            f"{context}.artifact",
        )
        if payload["schema_version"] != ARTIFACT_SCHEMA:
            raise LifecycleOverlayError(f"{context} has unsupported artifact schema")
        role = _token(payload["role"], f"{context}.artifact.role")
        role_parts = set(re.split(r"[^a-z0-9]+", role.casefold()))
        if role_parts & _DISALLOWED_ROLES:
            raise LifecycleOverlayError(f"{context} declares a prohibited artifact role")
        raw_path = Path(_token(payload["path"], f"{context}.artifact.path"))
        if ".." in raw_path.parts:
            raise LifecycleOverlayError(f"artifact path contains traversal: {raw_path}")
        path = raw_path.resolve() if raw_path.is_absolute() else (self.artifact_root / raw_path).resolve()
        if not _within(path, self.artifact_root):
            raise LifecycleOverlayError(f"artifact escapes the artifact root: {raw_path}")
        relative = path.relative_to(self.artifact_root)
        lowered = {part.casefold() for part in relative.parts}
        if lowered & _DISALLOWED_PATH_PARTS:
            raise LifecycleOverlayError(f"artifact uses a prohibited path: {relative.as_posix()}")
        if not path.is_file():
            raise LifecycleOverlayError(f"artifact is missing: {relative.as_posix()}")
        size = payload["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise LifecycleOverlayError(f"{context}.artifact.size_bytes is invalid")
        expected_sha = _sha(payload["sha256"], f"{context}.artifact.sha256")
        if path.stat().st_size != size:
            raise LifecycleOverlayError(f"artifact size differs: {relative.as_posix()}")
        if _file_sha256(path) != expected_sha:
            raise LifecycleOverlayError(f"artifact SHA-256 differs: {relative.as_posix()}")
        return _VerifiedArtifact(role, path, relative.as_posix(), expected_sha, size)

    @staticmethod
    def _read_contract(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LifecycleOverlayError(f"contract manifest is invalid JSON: {path.name}") from exc
        return _mapping(payload, path.name)

    @classmethod
    def _verify_task_spec(
        cls, path: Path, *, expected_digest: str, expected_task_id: str
    ) -> None:
        document = cls._read_contract(path)
        _strict_fields(
            document,
            {
                "schema_version",
                "task_id",
                "task_family",
                "objective",
                "input_allowlist",
                "input_denylist",
                "target_fields",
                "baseline_ids",
                "split_policy",
                "frozen_seeds",
                "metrics",
                "promotion_gates",
                "verify_gates",
                "runtime_policy",
                "lifecycle_stages",
                "task_spec_sha256",
            },
            path.name,
        )
        if document["schema_version"] != TASK_SPEC_SCHEMA:
            raise LifecycleOverlayError(f"contract manifest schema differs: {path.name}")
        observed = _content_digest(document, "task_spec_sha256", path.name)
        if observed != expected_digest:
            raise LifecycleOverlayError(f"candidate digest differs from {path.name}")
        if _token(document["task_id"], f"{path.name}.task_id") != expected_task_id:
            raise LifecycleOverlayError("task spec belongs to a different task")
        _token(document["task_family"], f"{path.name}.task_family")
        _token(document["objective"], f"{path.name}.objective")
        allow = set(_unique_tokens(document["input_allowlist"], "input_allowlist", allow_empty=True))
        deny = set(_unique_tokens(document["input_denylist"], "input_denylist", allow_empty=True))
        targets = set(_unique_tokens(document["target_fields"], "target_fields", allow_empty=True))
        if allow & deny or allow & targets:
            raise LifecycleOverlayError("task spec input/deny/target fields overlap")
        _unique_tokens(document["baseline_ids"], "baseline_ids")
        for field in ("split_policy", "promotion_gates", "verify_gates", "runtime_policy"):
            _mapping(document[field], f"{path.name}.{field}")
        seeds = _sequence(document["frozen_seeds"], f"{path.name}.frozen_seeds")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise LifecycleOverlayError("task spec frozen_seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise LifecycleOverlayError("task spec frozen_seeds contains duplicates")
        metrics = _sequence(document["metrics"], f"{path.name}.metrics")
        if not metrics:
            raise LifecycleOverlayError("task spec has no frozen metric")
        metric_names: list[str] = []
        for index, raw in enumerate(metrics):
            metric = _mapping(raw, f"{path.name}.metrics[{index}]")
            name = _token(metric.get("name"), f"{path.name}.metrics[{index}].name")
            if metric.get("direction") not in {"minimize", "maximize"}:
                raise LifecycleOverlayError("task spec metric direction is invalid")
            _token(metric.get("aggregation"), f"{path.name}.metrics[{index}].aggregation")
            metric_names.append(name)
        if len(set(metric_names)) != len(metric_names):
            raise LifecycleOverlayError("task spec metric names are duplicated")
        if tuple(document["lifecycle_stages"]) != _LIFECYCLE_STAGES:
            raise LifecycleOverlayError("task spec lifecycle stages differ")

    @classmethod
    def _verify_split_manifest(cls, path: Path, *, expected_digest: str) -> None:
        document = cls._read_contract(path)
        _strict_fields(
            document,
            {
                "schema_version",
                "split_id",
                "split_unit",
                "entities",
                "partitions",
                "group_assignments",
                "source_digests",
                "folds",
                "metadata",
                "split_manifest_sha256",
            },
            path.name,
        )
        if document["schema_version"] != SPLIT_MANIFEST_SCHEMA:
            raise LifecycleOverlayError(f"contract manifest schema differs: {path.name}")
        observed = _content_digest(document, "split_manifest_sha256", path.name)
        if observed != expected_digest:
            raise LifecycleOverlayError(f"candidate digest differs from {path.name}")
        _token(document["split_id"], f"{path.name}.split_id")
        _token(document["split_unit"], f"{path.name}.split_unit")
        entities = set(_unique_tokens(document["entities"], f"{path.name}.entities"))
        partitions_raw = _mapping(document["partitions"], f"{path.name}.partitions")
        required = {*_PRIMARY_PARTITIONS, "refit_pool"}
        if not required.issubset(partitions_raw):
            raise LifecycleOverlayError("split manifest is missing primary partitions")
        partitions = {
            str(name): set(
                _unique_tokens(members, f"partitions.{name}", allow_empty=True)
            )
            for name, members in partitions_raw.items()
        }
        primary = [partitions[name] for name in _PRIMARY_PARTITIONS]
        if any(primary[first] & primary[second] for first in range(3) for second in range(first + 1, 3)):
            raise LifecycleOverlayError("split manifest primary partitions overlap")
        if set().union(*primary) != entities:
            raise LifecycleOverlayError("split manifest primary partitions do not cover entities")
        if partitions["refit_pool"] != primary[0] | primary[1]:
            raise LifecycleOverlayError("split manifest refit_pool differs")
        if any(members - entities for members in partitions.values()):
            raise LifecycleOverlayError("split manifest partition has unknown entities")
        groups_raw = _mapping(document["group_assignments"], f"{path.name}.group_assignments")
        if set(groups_raw) != entities:
            raise LifecycleOverlayError("split group assignments do not cover entities")
        group_partitions: dict[str, set[str]] = {}
        for partition_name in _PRIMARY_PARTITIONS:
            for entity in partitions[partition_name]:
                group = _token(groups_raw[entity], f"group_assignments.{entity}")
                group_partitions.setdefault(group, set()).add(partition_name)
        if any(len(names) > 1 for names in group_partitions.values()):
            raise LifecycleOverlayError("split leakage group crosses primary partitions")
        digests = _mapping(document["source_digests"], f"{path.name}.source_digests")
        if not digests:
            raise LifecycleOverlayError("split manifest has no source digests")
        for name, digest in digests.items():
            _sha(digest, f"source_digests.{name}")
        folds = _mapping(document["folds"], f"{path.name}.folds")
        validation_seen: set[str] = set()
        refit = partitions["refit_pool"]
        for fold_name, raw in folds.items():
            fold = _mapping(raw, f"folds.{fold_name}")
            if set(fold) != {"train", "validation"}:
                raise LifecycleOverlayError("split fold fields differ")
            train = set(_unique_tokens(fold["train"], f"folds.{fold_name}.train", allow_empty=True))
            validation = set(
                _unique_tokens(
                    fold["validation"], f"folds.{fold_name}.validation", allow_empty=True
                )
            )
            if train & validation or train | validation != refit:
                raise LifecycleOverlayError("split fold is not exclusive and complete")
            if validation_seen & validation:
                raise LifecycleOverlayError("split entity validates in multiple folds")
            validation_seen |= validation
        if folds and validation_seen != refit:
            raise LifecycleOverlayError("split fold validation is not exact-once")
        _mapping(document["metadata"], f"{path.name}.metadata")

    @staticmethod
    def _verify_pointers(
        value: object,
        candidates: Mapping[str, tuple[dict[str, Any], tuple[_VerifiedArtifact, ...]]],
    ) -> list[dict[str, Any]]:
        pointers_raw = _sequence(value, "registry.pointers")
        pointers: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw in enumerate(pointers_raw):
            pointer = _mapping(raw, f"registry.pointers[{index}]")
            _strict_fields(
                pointer,
                {"task_id", "pointer_type", "candidate_id", "generation", "updated_at"},
                f"registry.pointers[{index}]",
            )
            task_id = _token(pointer["task_id"], f"registry.pointers[{index}].task_id")
            pointer_type = _token(
                pointer["pointer_type"], f"registry.pointers[{index}].pointer_type"
            )
            candidate_id = _token(
                pointer["candidate_id"], f"registry.pointers[{index}].candidate_id"
            )
            generation = pointer["generation"]
            if pointer_type not in _POINTER_TYPES:
                raise LifecycleOverlayError("registry pointer type is unknown")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise LifecycleOverlayError("registry pointer generation is invalid")
            if (task_id, pointer_type) in seen:
                raise LifecycleOverlayError("registry has duplicate task/pointer entries")
            seen.add((task_id, pointer_type))
            if candidate_id not in candidates:
                raise LifecycleOverlayError("registry pointer references an unknown candidate")
            candidate = candidates[candidate_id][0]
            if candidate["task_id"] != task_id:
                raise LifecycleOverlayError("registry pointer crosses task boundaries")
            if pointer_type == "scientific_incumbent" and candidate["scientific_status"] not in {
                "validated",
                "conditional",
            }:
                raise LifecycleOverlayError("scientific incumbent is not scientifically accepted")
            if pointer_type == "runtime_default" and candidate["runtime_status"] != "runnable":
                raise LifecycleOverlayError("runtime default is not runnable")
            pointers.append(
                {
                    **pointer,
                    "task_id": task_id,
                    "pointer_type": pointer_type,
                    "candidate_id": candidate_id,
                    "generation": generation,
                }
            )
        return pointers

    def _release(
        self,
        candidate: dict[str, Any],
        artifacts: tuple[_VerifiedArtifact, ...],
        pointer_types: set[str],
        generations: Mapping[tuple[str, str], int],
    ) -> ModelRelease:
        metadata = dict(candidate["metadata"])
        candidate_id = str(candidate["candidate_id"])
        task_id = str(candidate["task_id"])
        release_id = f"lifecycle__{_safe_identifier(task_id)}__{_safe_identifier(candidate_id)}"
        release_artifacts = tuple(
            ReleaseArtifact(
                id=_safe_identifier(artifact.role),
                name=artifact.path.name,
                role=artifact.role,
                kind=(
                    "checkpoint"
                    if artifact.path.suffix.casefold() in {".pt", ".pth", ".ckpt"}
                    else "file"
                ),
                path=str(artifact.path),
                relative_path=artifact.relative_path,
                exists=True,
                media_type=(
                    mimetypes.guess_type(artifact.path.name)[0] or "application/octet-stream"
                ),
                layer=_infer_layer(artifact.path),
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                integrity_status="sha256_verified",
            )
            for artifact in artifacts
        )
        lifecycle_metadata = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "registry_sha256": self.registry_sha256,
            "candidate_sha256": candidate["candidate_sha256"],
            "task_spec_sha256": candidate["task_spec_sha256"],
            "split_manifest_sha256": candidate["split_manifest_sha256"],
            "pointer_types": sorted(pointer_types),
            "scientific_incumbent": "scientific_incumbent" in pointer_types,
            "runtime_default": "runtime_default" in pointer_types,
            "pointer_generations": {
                pointer: generations[(candidate_id, pointer)] for pointer in sorted(pointer_types)
            },
            "read_only": True,
            "catalog_only": True,
        }
        raw_scope = metadata.get("scope", ())
        scope = (
            tuple(str(item) for item in raw_scope)
            if isinstance(raw_scope, list) and all(isinstance(item, str) for item in raw_scope)
            else (task_id,)
        )
        raw_warnings = metadata.get("warnings", ())
        warnings = (
            tuple(str(item) for item in raw_warnings)
            if isinstance(raw_warnings, list) and all(isinstance(item, str) for item in raw_warnings)
            else ()
        )
        return ModelRelease(
            id=release_id,
            name=str(metadata.get("name") or candidate_id),
            version=str(metadata.get("version") or candidate["candidate_sha256"][:12]),
            task_id=task_id,
            description=str(metadata.get("description") or "Verified WellFuse lifecycle candidate."),
            scientific_status=candidate["scientific_status"],
            runtime_status=candidate["runtime_status"],
            evidence_class=str(candidate["evidence_class"]),
            scope=scope,
            warnings=warnings,
            artifacts=release_artifacts,
            metadata={**metadata, "lifecycle": lifecycle_metadata},
            source="wellfuse_lifecycle",
            model_id=str(metadata.get("model_id") or candidate_id),
            runner_id=(str(metadata["runner_id"]) if metadata.get("runner_id") else None),
        )


__all__ = ["LifecycleOverlayError", "LifecycleRegistryOverlay"]
