from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the portable release tree.")
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--allow-lfs-pointers", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    required = [
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "pyproject.toml",
        "frontend/package-lock.json",
        "src/well_seismic/api.py",
        "configs/faultseg.yaml",
        "configs/surface_seg.yaml",
        "models/manifest.json",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"缺少必要文件: {relative}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        path = ROOT / model["path"]
        if not path.exists():
            errors.append(f"缺少模型权重: {model['path']}")
            continue
        if is_lfs_pointer(path):
            if not (args.allow_lfs_pointers and model["git_lfs"]):
                errors.append(f"模型仍是 Git LFS pointer，请执行 git lfs pull: {model['path']}")
            continue
        if path.stat().st_size != model["size"]:
            errors.append(
                f"模型大小不匹配: {model['path']} "
                f"({path.stat().st_size} != {model['size']})"
            )
        if not args.skip_model_hash:
            actual = sha256(path)
            if actual != model["sha256"]:
                errors.append(f"模型 SHA-256 不匹配: {model['path']}")

    if args.runtime:
        sys.path.insert(0, str(ROOT / "src"))
        for module_name in ("well_seismic.api", "well_seismic.cli", "numpy", "yaml"):
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"运行时导入失败 {module_name}: {type(exc).__name__}: {exc}")
        if not (ROOT / "frontend" / "dist" / "index.html").exists():
            errors.append("缺少 frontend/dist/index.html，请先执行前端构建")

    if errors:
        print("发布自检失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"发布自检通过：{len(manifest['models'])} 个模型权重已登记。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
