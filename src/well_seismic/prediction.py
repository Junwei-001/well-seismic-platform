"""Formal model inference services, separate from shared preprocessing."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .faultseg import FaultSegInputSpec
from .modeling.input_adapters import ModelInputAdapterRegistry, ModelInputRequest


Progress = Callable[[int, str], None]
PredictionRunner = Callable[..., dict[str, Any]]


class PredictionRunnerRegistry:
    """Dispatch inference without coupling the API to a specific model."""

    entry_point_group = "well_seismic.prediction_runners"

    def __init__(self) -> None:
        self._runners: dict[str, PredictionRunner] = {}
        self.plugin_load_errors: list[dict[str, str]] = []

    def register(self, model_id: str, runner: PredictionRunner, *, replace: bool = False) -> None:
        if model_id in self._runners and not replace:
            raise ValueError(f"prediction runner already registered: {model_id}")
        self._runners[model_id] = runner

    def run(self, model_id: str, request: ModelInputRequest, **kwargs: Any) -> dict[str, Any]:
        try:
            runner = self._runners[model_id]
        except KeyError as exc:
            raise KeyError(f"no prediction runner registered for model: {model_id}") from exc
        return runner(request, **kwargs)

    def model_ids(self) -> list[str]:
        return list(self._runners)

    def load_entry_points(self) -> list[str]:
        """Use the entry-point name as ``model_id`` and its object as runner."""
        loaded: list[str] = []
        for entry_point in entry_points(group=self.entry_point_group):
            try:
                self.register(entry_point.name, entry_point.load())
                loaded.append(entry_point.name)
            except Exception as exc:
                self.plugin_load_errors.append(
                    {"plugin": entry_point.name, "error": f"{type(exc).__name__}: {exc}"}
                )
        return loaded


def build_default_prediction_runners() -> PredictionRunnerRegistry:
    registry = PredictionRunnerRegistry()
    registry.register("faultseg_3d", run_faultseg_prediction)
    registry.register("seismic_surface_seg", run_surface_seg_prediction)
    return registry


def run_faultseg_prediction(
    request: ModelInputRequest,
    *,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    del options  # Reserved for task/model-specific runner parameters.
    spec = FaultSegInputSpec.from_config(config)
    patch_size = patch_size or spec.patch_size
    overlap = overlap or spec.overlap
    threshold = spec.threshold if threshold is None else float(threshold)
    runtime_spec = FaultSegInputSpec(patch_size, overlap, spec.patch_multiple, threshold).validated()

    if progress:
        progress(12, "正在按 FaultSeg 输入契约重建三维地震体")
    batch = adapters.get("faultseg_3d").prepare(request)
    if batch.array is None:
        raise RuntimeError("FaultSeg input adapter did not materialize a seismic array")
    if progress:
        progress(40, "三维体块已就绪，正在加载 FaultSeg 权重")

    faultseg_root = project_root / "接口模型" / "faultSeg-main"
    checkpoint = faultseg_root / "model" / "faultseg-best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FaultSeg checkpoint not found: {checkpoint}")
    if str(faultseg_root) not in sys.path:
        sys.path.insert(0, str(faultseg_root))
    try:
        torch = importlib.import_module("torch")
        checkpoint_module = importlib.import_module("src.checkpoint")
        inference_module = importlib.import_module("src.inference")
    except ImportError as exc:
        raise RuntimeError("FaultSeg inference requires PyTorch; install the project faultseg optional dependencies") from exc

    device = inference_module.choose_device(device_name)
    model, checkpoint_metadata = checkpoint_module.load_model(checkpoint, device)
    if progress:
        progress(55, f"FaultSeg 已加载到 {device}，开始滑窗推理")

    def patch_progress(index: int, total: int, origin: tuple[int, int, int]) -> None:
        if progress:
            progress(55 + int(35 * index / max(total, 1)), f"FaultSeg patch {index}/{total}，起点 {origin}")

    probability = inference_module.predict_volume(
        model,
        batch.array,
        device,
        runtime_spec.patch_size,
        runtime_spec.overlap,
        progress=patch_progress,
        normalize_patches=True,
        invalid_value=np.nan,
    )
    mask = probability >= runtime_spec.threshold
    if batch.valid_mask is not None:
        probability[:, ~batch.valid_mask] = 0.0
        mask[:, ~batch.valid_mask] = False

    output_directory.mkdir(parents=True, exist_ok=True)
    probability_path = output_directory / "faultseg_probability.npy"
    mask_path = output_directory / "faultseg_mask.npy"
    np.save(probability_path, probability.astype(np.float32))
    np.save(mask_path, mask.astype(np.uint8))
    result = {
        "model_id": "faultseg_3d",
        "model_name": "FaultSeg 三维断层分割",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "device": str(device),
        "input": {
            "shape_zyx": list(batch.array.shape),
            "dtype": str(batch.array.dtype),
            "axes": list(batch.axes),
            **batch.provenance,
        },
        "inference": {
            "patch_size": list(runtime_spec.patch_size),
            "overlap": list(runtime_spec.overlap),
            "normalization": "per-patch z-score",
            "threshold": runtime_spec.threshold,
        },
        "probability": {
            "shape_zyx": list(probability.shape),
            "min": float(probability.min()),
            "max": float(probability.max()),
            "mean": float(probability.mean()),
            "positive_fraction": float(mask.mean()),
        },
        "outputs": {
            "probability_npy": str(probability_path),
            "mask_npy": str(mask_path),
        },
    }
    metadata_path = output_directory / "faultseg_result.json"
    result["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if progress:
        progress(98, "FaultSeg 推理完成，正在登记结果")
    return result


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@lru_cache(maxsize=16)
def _checkpoint_sha256(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_surface_checkpoint(
    checkpoint: Path,
    expected: dict[str, Any] | None = None,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SurfaceSeg checkpoint not found: {checkpoint}")
    stat = checkpoint.stat()
    if stat.st_size < 1024:
        header = checkpoint.read_bytes()[:64]
        if header.startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(f"SurfaceSeg checkpoint is still a Git LFS pointer: {checkpoint}")
        raise RuntimeError(f"SurfaceSeg checkpoint is unexpectedly small: {checkpoint}")
    expected = expected or {}
    expected_size = expected.get("size")
    if expected_size is not None and stat.st_size != int(expected_size):
        raise RuntimeError(
            f"SurfaceSeg checkpoint size mismatch: {checkpoint} "
            f"({stat.st_size} != {int(expected_size)})"
        )
    expected_hash = str(expected.get("sha256", "")).strip().lower()
    if expected_hash:
        actual_hash = _checkpoint_sha256(
            str(checkpoint.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SurfaceSeg checkpoint SHA256 mismatch: {checkpoint} "
                f"({actual_hash} != {expected_hash})"
            )


def _load_surface_seg_runtime(surface_root: Path) -> Any:
    """Load the bundled package under a private name to avoid module collisions."""
    module_name = "_well_seismic_surface_seg_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    package_root = surface_root / "minimal_sgy"
    init_path = package_root / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"SurfaceSeg Python package not found: {init_path}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SurfaceSeg package: {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit):
        sys.modules.pop(module_name, None)
        raise
    return module


def _surface_runtime_options(
    surface_config: dict[str, Any],
    options: dict[str, Any],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        return options[name] if name in options else surface_config.get(name, default)

    mask_threshold = (
        float(threshold)
        if threshold is not None
        else float(value("mask_threshold", 0.5))
    )
    inline_count = value("inline_count", None)
    max_inlines = value("max_inlines", None)
    return {
        "device": str(value("device", "auto")),
        "segformer_batch_size": int(value("segformer_batch_size", 2)),
        "mask2former_batch_size": int(value("mask2former_batch_size", 1)),
        "amplitude_mode": str(value("amplitude_mode", "auto")),
        "query_threshold": float(value("query_threshold", 0.35)),
        "mask_threshold": mask_threshold,
        "num_visualizations": int(value("num_visualizations", 5)),
        "inline_count": None if inline_count is None else int(inline_count),
        "max_inlines": None if max_inlines is None else int(max_inlines),
        "write_mask_sgy": bool(value("write_mask_sgy", True)),
    }


def _surface_cli_command(
    python_executable: Path,
    request: ModelInputRequest,
    output_directory: Path,
    models_dir: Path,
    runtime_options: dict[str, Any],
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "minimal_sgy",
        "--input",
        str(request.source),
        "--output-dir",
        str(output_directory),
        "--models-dir",
        str(models_dir),
        "--device",
        str(runtime_options["device"]),
        "--segformer-batch-size",
        str(runtime_options["segformer_batch_size"]),
        "--mask2former-batch-size",
        str(runtime_options["mask2former_batch_size"]),
        "--amplitude-mode",
        str(runtime_options["amplitude_mode"]),
        "--query-threshold",
        str(runtime_options["query_threshold"]),
        "--mask-threshold",
        str(runtime_options["mask_threshold"]),
        "--num-visualizations",
        str(runtime_options["num_visualizations"]),
    ]
    if runtime_options["inline_count"] is not None:
        command.extend(["--inline-count", str(runtime_options["inline_count"])])
    if runtime_options["max_inlines"] is not None:
        command.extend(["--max-inlines", str(runtime_options["max_inlines"])])
    if not runtime_options["write_mask_sgy"]:
        command.append("--no-mask-sgy")
    return command


def _run_surface_seg_external(
    *,
    python_executable: Path,
    surface_root: Path,
    request: ModelInputRequest,
    output_directory: Path,
    models_dir: Path,
    runtime_options: dict[str, Any],
    runtime_log: Path,
    direct_error: str,
    progress: Progress | None,
) -> dict[str, Any]:
    if not python_executable.is_file():
        raise FileNotFoundError(
            "SurfaceSeg dependencies are unavailable in the API process and "
            f"external_python does not exist: {python_executable}; direct import: {direct_error}"
        )
    command = _surface_cli_command(
        python_executable,
        request,
        output_directory,
        models_dir,
        runtime_options,
    )
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    lines = [
        "execution_backend=external_python",
        f"python={python_executable}",
        f"direct_import_error={direct_error}",
        f"command={json.dumps(command, ensure_ascii=False)}",
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=surface_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        lines.append(f"launch_error={type(exc).__name__}: {exc}")
        runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError(f"无法启动 SurfaceSeg 外部 Python：{exc}") from exc
    assert process.stdout is not None
    stage_progress = {
        "[1/6]": 18,
        "[2/6]": 28,
        "[3/6]": 45,
        "[4/6]": 62,
        "[5/6]": 88,
        "[6/6]": 94,
    }
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        lines.append(line)
        if progress:
            for marker, percent in stage_progress.items():
                if line.startswith(marker):
                    progress(percent, f"SurfaceSeg {line}")
                    break
    return_code = process.wait()
    lines.append(f"return_code={return_code}")
    runtime_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if return_code:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(f"SurfaceSeg 外部推理失败（退出码 {return_code}）：\n{tail}")
    summary_path = output_directory / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"SurfaceSeg 推理未生成 summary.json：{summary_path}")
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("SurfaceSeg summary.json 必须是 JSON 对象")
    return document


def run_surface_seg_prediction(
    request: ModelInputRequest,
    *,
    adapters: ModelInputAdapterRegistry,
    config: dict[str, Any],
    project_root: Path,
    output_directory: Path,
    device_name: str = "auto",
    threshold: float | None = None,
    patch_size: tuple[int, int, int] | None = None,
    overlap: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the bundled three-stage stratigraphic instance segmentation model."""
    del patch_size, overlap
    options = dict(options or {})
    surface_config = dict(config.get("surface_seg", {}))
    options.setdefault("device", device_name)
    runtime_options = _surface_runtime_options(
        surface_config,
        options,
        threshold=threshold,
    )
    surface_root = _project_path(
        project_root,
        options.get(
            "model_root",
            surface_config.get("model_root", "接口模型/seismic_surface_seg"),
        ),
    )
    models_dir = _project_path(
        project_root,
        options.get(
            "models_dir",
            surface_config.get("models_dir", surface_root / "models"),
        ),
    )
    required_checkpoints = (
        ("segformer-base", models_dir / "segformer-base" / "best.pt"),
        ("segformer-refine", models_dir / "segformer-refine" / "best.pt"),
        ("mask2former", models_dir / "mask2former" / "best.pt"),
    )
    checkpoint_manifest = surface_config.get("checkpoint_manifest", {})
    for stage, checkpoint in required_checkpoints:
        expected = (
            checkpoint_manifest.get(stage)
            if isinstance(checkpoint_manifest, dict)
            and isinstance(checkpoint_manifest.get(stage), dict)
            else None
        )
        _verify_surface_checkpoint(checkpoint, expected)

    if progress:
        progress(10, "正在核验地层分割三维后叠加 SEG-Y 输入")
    batch = adapters.get("seismic_surface_seg").prepare(request)
    if runtime_options["inline_count"] is None:
        inferred_inline_count = batch.provenance.get("native_inline_count")
        if inferred_inline_count is not None:
            runtime_options["inline_count"] = int(inferred_inline_count)
    output_directory.mkdir(parents=True, exist_ok=True)
    runtime_log = output_directory / "surface_seg_runtime.log"

    execution_backend = "in_process"
    direct_error = ""
    try:
        runtime = _load_surface_seg_runtime(surface_root)
    except (ImportError, ModuleNotFoundError, SystemExit) as exc:
        runtime = None
        direct_error = f"{type(exc).__name__}: {exc}"
    if runtime is not None:
        if progress:
            progress(18, "主服务依赖可用，正在进程内启动 SurfaceSeg 三阶段推理")
        upstream = runtime.run_inference(
            request.source,
            output_directory,
            models_dir=models_dir,
            **runtime_options,
        )
        runtime_log.write_text(
            "\n".join(
                (
                    "execution_backend=in_process",
                    f"python={sys.executable}",
                    "status=completed",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        execution_backend = "external_python"
        external_python = _project_path(
            project_root,
            options.get(
                "external_python",
                surface_config.get("external_python", sys.executable),
            ),
        )
        if progress:
            progress(15, f"主服务缺少模型依赖，切换外部环境：{external_python}")
        upstream = _run_surface_seg_external(
            python_executable=external_python,
            surface_root=surface_root,
            request=request,
            output_directory=output_directory,
            models_dir=models_dir,
            runtime_options=runtime_options,
            runtime_log=runtime_log,
            direct_error=direct_error,
            progress=progress,
        )

    if not isinstance(upstream, dict):
        raise RuntimeError("SurfaceSeg Python API must return a mapping")
    artifacts = upstream.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("SurfaceSeg result is missing artifacts")
    mask_path = Path(str(artifacts.get("mask_npy", ""))).expanduser().resolve()
    confidence_path = Path(str(artifacts.get("confidence_npy", ""))).expanduser().resolve()
    if not mask_path.is_file() or not confidence_path.is_file():
        raise FileNotFoundError("SurfaceSeg did not create mask.npy and confidence.npy")
    labels = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    confidence = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 3 or confidence.shape != labels.shape:
        raise ValueError(
            f"SurfaceSeg output shape mismatch: labels={labels.shape}, confidence={confidence.shape}"
        )
    shape_ics = [int(value) for value in labels.shape]
    label_range = [int(labels.min()), int(labels.max())]
    confidence_summary = {
        "shape_ics": shape_ics,
        "dtype": str(confidence.dtype),
        "min": float(confidence.min()),
        "max": float(confidence.max()),
        "mean": float(confidence.mean(dtype=np.float64)),
    }
    del labels, confidence

    output_paths = {
        str(name): str(value)
        for name, value in artifacts.items()
        if value not in (None, "")
    }
    upstream_summary_path = output_directory / "summary.json"
    output_paths["upstream_summary_json"] = str(upstream_summary_path)
    output_paths["runtime_log"] = str(runtime_log)
    result = {
        "model_id": "seismic_surface_seg",
        "model_name": "Seismic Surface Seg 地层分割",
        "checkpoint": str(models_dir),
        "device": str(upstream.get("device", runtime_options["device"])),
        "input": {
            "source": str(request.source),
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "shape_ics": shape_ics,
            "source_shape_ics": list(batch.provenance["shape_ics"]),
            "source_shape_zyx": list(batch.provenance["source_shape_zyx"]),
            **{
                key: value
                for key, value in batch.provenance.items()
                if key not in {"source", "shape_ics", "source_shape_zyx"}
            },
        },
        "inference": {
            "execution_backend": execution_backend,
            "amplitude_mode": upstream.get("amplitude_scaling", {}).get(
                "effective",
                runtime_options["amplitude_mode"],
            ),
            "amplitude_mode_requested": runtime_options["amplitude_mode"],
            "query_threshold": runtime_options["query_threshold"],
            "mask_threshold": runtime_options["mask_threshold"],
            "segformer_batch_size": runtime_options["segformer_batch_size"],
            "mask2former_batch_size": runtime_options["mask2former_batch_size"],
            "inline_count": runtime_options["inline_count"],
            "max_inlines": runtime_options["max_inlines"],
            "write_mask_sgy": runtime_options["write_mask_sgy"],
            "prior_compatibility_mode": upstream.get("prior_compatibility_mode"),
            "crop_policy": "模型按完整 Inline 切片推理；通用 crop_start/crop_size 不适用",
        },
        "segmentation": {
            "shape_ics": shape_ics,
            "axes": ["INLINE", "CROSSLINE", "SAMPLE"],
            "dtype": str(upstream.get("mask_dtype", "int16")),
            "label_range": label_range,
            "instance_count": max(0, label_range[1] + 1),
            "max_instances_per_inline": max(0, label_range[1] + 1),
            "cross_inline_consistent": False,
            "invalid_label": -1,
            "confidence_min": confidence_summary["min"],
            "confidence_max": confidence_summary["max"],
            "confidence_mean": confidence_summary["mean"],
            "confidence": confidence_summary,
        },
        "geometry": upstream.get("geometry", {}),
        "checkpoints": upstream.get("checkpoints", {}),
        "elapsed_seconds": upstream.get("elapsed_seconds"),
        "outputs": output_paths,
    }
    metadata_path = output_directory / "surface_seg_result.json"
    result["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress(98, "地层分割推理完成，正在登记标签体与置信度体")
    return result
