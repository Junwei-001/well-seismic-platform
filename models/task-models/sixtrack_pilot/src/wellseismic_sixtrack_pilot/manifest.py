"""Payload manifests shared by the platform-side plugin and the isolated workers.

Stdlib only: the platform process and the model interpreter both import this
module, so it must not pull numpy, torch or any platform package.

Layout (``<task-models>/<model_id>/<version>/``)::

    manifest.json        tracked; declares every payload file and its SHA-256
    payload/<files>      git-ignored copies made by ``tools/materialize_payloads.py``

Any missing file, size drift or SHA-256 mismatch raises ``PayloadIntegrityError``;
callers never fall back to another location.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "well-seismic.sixtrack-model-payload.v1"
MANIFEST_NAME = "manifest.json"
PACKAGE_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = PACKAGE_DIRECTORY.parents[1]
PROJECT_ROOT_ENV = "WELL_SEISMIC_MODEL_PROJECT_ROOT"


def resolve_project_root(package_directory: Path, environ: dict) -> Path | None:
    """Editable checkout or explicit deployment root; never infer from cwd/site-packages."""
    configured = environ.get(PROJECT_ROOT_ENV)
    if configured is not None:
        path = Path(configured.strip())
        if not configured.strip() or not path.is_absolute():
            raise RuntimeError(f"{PROJECT_ROOT_ENV} must be an absolute project directory")
        root = path.resolve()
        if not (root / "models/task-models").is_dir():
            raise RuntimeError(f"{PROJECT_ROOT_ENV} has no models/task-models directory")
        return root
    package_root = package_directory.parents[1]
    if (package_directory.parent.name == "src" and package_root.name == "sixtrack_pilot"
            and (package_root / "pyproject.toml").is_file()
            and package_root.parent.name == "task-models"
            and package_root.parent.parent.name == "models"):
        return package_root.parents[2]
    return None


PLATFORM_ROOT = resolve_project_root(PACKAGE_DIRECTORY, os.environ)
TASK_MODELS_ROOT = PLATFORM_ROOT / "models/task-models" if PLATFORM_ROOT is not None else None


class PayloadIntegrityError(RuntimeError):
    """A packaged payload is absent, altered or escapes its model directory."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_manifest(model_dir: Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PayloadIntegrityError(f"unsupported payload manifest schema in {model_dir}")
    return manifest


def model_dir_for(model_id: str, version: str) -> Path:
    if TASK_MODELS_ROOT is None:
        raise RuntimeError(f"Installed model registration requires {PROJECT_ROOT_ENV}; standalone workers use explicit input paths")
    for value in (model_id, version):
        if not value or value in (".", "..") or Path(value).name != value:
            raise ValueError("model ID and version must be single directory names")
    return TASK_MODELS_ROOT / model_id / version


def verify_payload(model_dir: Path | str) -> dict[str, Path]:
    """Verify every declared payload file and return ``{role: absolute_path}``."""

    model_dir = Path(model_dir).resolve()
    manifest = read_manifest(model_dir)
    verified: dict[str, Path] = {}
    for entry in manifest["payload"]:
        role = str(entry["role"])
        path = (model_dir / entry["path"]).resolve()
        try:
            path.relative_to(model_dir)
        except ValueError as exc:
            raise PayloadIntegrityError(f"{role} escapes {model_dir}") from exc
        if not path.is_file():
            raise PayloadIntegrityError(
                f"{manifest['model_id']}: payload {role} is missing ({path}); "
                "run tools/materialize_payloads.py"
            )
        size = path.stat().st_size
        if size != int(entry["size"]):
            raise PayloadIntegrityError(
                f"{manifest['model_id']}: payload {role} size {size} != manifest {entry['size']}"
            )
        observed = sha256_file(path)
        if observed != str(entry["sha256"]).lower():
            raise PayloadIntegrityError(
                f"{manifest['model_id']}: payload {role} sha256 {observed} != manifest {entry['sha256']}"
            )
        verified[role] = path
    return verified


def load_request(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fail(result_path: Path, exc: BaseException) -> int:
    """Write a structured failure record for the platform runner and exit non-zero."""

    atomic_json(
        result_path,
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        },
    )
    return 2
