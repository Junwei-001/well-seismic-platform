"""Verify the extracted no-weight platform boundary.

The check is intentionally independent from the full migration-package
verifier, which expects model checkpoints and CUDA archives.  It can be run
without importing FastAPI; ``--runtime`` additionally exercises the public
health/capability contract in interfaces-only mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".bin",
    ".npy",
    ".npz",
    ".sgy",
    ".segy",
    ".las",
    ".dlis",
    ".zip",
    ".7z",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".h5",
    ".hdf5",
}
FORBIDDEN_PARTS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?:\b[A-Z]:\\Users\\|\b[A-Z]:\\[^\r\n\\]+\\[^\r\n\\]+)"
)
FRONTEND_EXPECTED = {
    "src/App.vue": (506194, "195e2b66ad1a01909f1e37a0a9090c6317690df5c45e9f1d8e1917cd8309c0e2"),
    "src/api.ts": (57247, "392ada120677c2f237e3d56503263dc166aa7ad02f0d49e4ea514b907aade56b"),
    "src/styles.css": (134168, "e59e8ec01dd09266bcc2df96f4aa37a3a437c028c90339d7fb5c73b703fa4bcb"),
    "src/product-theme.css": (87526, "ba82c1a0e91648468ee1866bccfd309207576a65225a866bcd9d1d6157219471"),
    "dist/assets/index-aUVSo6eS.js": (410304, "6e366261a8f4cfa7b893ca4fef686d9e66e3258a5d8db4165925af0918cfc6e6"),
    "dist/assets/index-h6p1ay4M.css": (228138, "b53e4d04da744aaee7608412626d9d96361ed9ad882b30c8434b51444c261c8c"),
    "dist/assets/首屏_井震智能解释中心-CGp4ykid.jpg": (226848, "f5a6c12a0cca2785b27348d7b0c2183e8c03411ec18a75e9fe01a4e403ee3092"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def verify_static() -> list[str]:
    errors: list[str] = []
    required = (
        "README.md",
        ".env.example",
        "pyproject.toml",
        "frontend/index.html",
        "frontend/dist/index.html",
        "src/well_seismic/api.py",
        "src/well_seismic/platform_mode.py",
        "configs/layerpulse.yaml",
        "interfaces/model_registry.json",
        "models/INTERFACE_ONLY",
        "models/manifest.json",
        "抽离清单.json",
        "启动接口平台.ps1",
        "停止接口平台.ps1",
    )
    for item in required:
        if not (ROOT / item).is_file():
            errors.append(f"missing required file: {item}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = relative(path)
        if ".git" in Path(rel).parts:
            continue
        # The verifier contains the redaction markers as literals; do not flag
        # its own source while scanning the package for leaked source paths.
        if path.resolve() == Path(__file__).resolve():
            continue
        # The running API may leave only these regenerable control-plane files
        # under runtime/state.  They are not part of the extracted payload.
        if rel.startswith("runtime/state/") and (
            path.name == "platform_state.sqlite3"
            or path.name in {"platform_state.sqlite3-shm", "platform_state.sqlite3-wal", "interface-platform.pid.json"}
            or path.name.startswith("interface-platform-")
        ):
            continue
        parts = {part.casefold() for part in Path(rel).parts}
        if parts & {part.casefold() for part in FORBIDDEN_PARTS}:
            errors.append(f"generated/cache path present: {rel}")
        if path.suffix.casefold() in WEIGHT_SUFFIXES:
            errors.append(f"model/data binary present: {rel}")
        if path.name.casefold() in {".env", ".env.local", ".env.production"}:
            errors.append(f"private environment file present: {rel}")
        if path.suffix.casefold() in {".md", ".json", ".yaml", ".yml", ".toml", ".ps1", ".py"} and "frontend" not in Path(rel).parts:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if PRIVATE_PATH_PATTERN.search(text):
                errors.append(f"source-machine path present in {rel}")

    for rel, (size, expected_hash) in FRONTEND_EXPECTED.items():
        path = ROOT / "frontend" / rel
        if not path.is_file():
            errors.append(f"frontend baseline file missing: {rel}")
            continue
        if path.stat().st_size != size:
            errors.append(f"frontend size drift: {rel} ({path.stat().st_size} != {size})")
        observed = sha256(path)
        if observed.casefold() != expected_hash.casefold():
            errors.append(f"frontend hash drift: {rel} ({observed} != {expected_hash})")

    frontend_files = [
        path
        for path in (ROOT / "frontend").rglob("*")
        if path.is_file()
        and "node_modules" not in path.parts
        and ".ruff_cache" not in path.parts
    ]
    frontend_bytes = sum(path.stat().st_size for path in frontend_files)
    if len(frontend_files) != 50 or frontend_bytes != 4266081:
        errors.append(
            "frontend baseline count/bytes drift: "
            f"{len(frontend_files)} files / {frontend_bytes} bytes"
        )

    try:
        manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"models/manifest.json invalid: {exc}")
    else:
        if manifest.get("mode") != "interfaces_only" or manifest.get("models") != []:
            errors.append("models/manifest.json must remain an empty interfaces-only manifest")

    return errors


def verify_runtime() -> list[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["WELLFUSE_MODEL_MODE"] = "interfaces_only"
    env["WELLFUSE_DISABLE_TASK_MODELS"] = "1"
    fd, probe_db = tempfile.mkstemp(prefix="strata-interface-verify-", suffix=".sqlite3")
    os.close(fd)
    try:
        os.unlink(probe_db)
    except FileNotFoundError:
        pass
    env["WELL_SEISMIC_STATE_DB"] = probe_db
    code = (
        "import json; import uvicorn; import well_seismic.api as a; "
        "c=a.capabilities(); h=a.health(); "
        "assert h['runtime_mode']=='interfaces_only'; "
        "assert c['runtime_mode']['task_models_enabled'] is False; "
        "assert c['runtime_mode']['weights_attached'] is False; "
        "assert c['prediction_runner_model_ids']==[]; "
        "assert all(not t['runnable_model_ids'] for t in c['prediction_tasks']); "
        "print(json.dumps({'models':len(c['models']), 'tasks':len(c['prediction_tasks'])}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                Path(probe_db + suffix).unlink()
            except FileNotFoundError:
                pass
    if completed.returncode != 0:
        return [
            "runtime contract failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="also import API and exercise capabilities")
    args = parser.parse_args()
    errors = verify_static()
    if args.runtime:
        errors.extend(verify_runtime())
    result = {"ok": not errors, "root": str(ROOT), "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
