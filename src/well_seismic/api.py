from __future__ import annotations

import copy
import html
import json
import os
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .api_models import (
    AssistantChatRequest,
    InspectionRequest,
    IssueConfirmationRequest,
    PredictionRequest,
    PreprocessingRequest,
    TaskCreated,
    TransformationActivationRequest,
    ViserLayerModeRequest,
    ViserSliceRequest,
)
from .auto_input import build_explicit_paths_manifest
from .cigvis_adapter import (
    plotly_javascript,
    render_cigvis_workbench,
    update_viser_layer_mode,
    update_viser_slices,
)
from .config import load_config
from .fusion import build_default_fusion_registry
from .interpretation import build_default_interpretation_registry
from .llm import build_structured_generator, load_llm_settings
from .llm.transformation import activate_transformation, create_transformation_draft
from .modeling import ModelInputRequest, build_default_input_adapters, build_default_registry
from .pipeline import WellSeismicPipeline
from .platform_capabilities import build_platform_capabilities
from .prediction import build_default_prediction_runners
from .prediction_visualization import build_prediction_visualization_payload
from .visualization_preview import build_visualization_preview
from .workflow import build_preparation_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


app = FastAPI(
    title="地层慧眼",
    version="0.3.0",
    description="油气甜点智能识别的地震—测井多模态统一表征大模型平台",
)

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


_tasks: dict[str, dict[str, Any]] = {}
_transformation_drafts: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="well-seismic-inspection")
_model_registry = build_default_registry()
_model_registry.load_entry_points()
_interpretation_registry = build_default_interpretation_registry()
_interpretation_registry.load_entry_points()
_fusion_registry = build_default_fusion_registry()
_fusion_registry.load_entry_points()
_platform_config = load_config(CONFIG_DIR, {"inputs": []})
_input_adapters = build_default_input_adapters(_platform_config)
_input_adapters.load_entry_points(_platform_config)
_prediction_runners = build_default_prediction_runners()
_prediction_runners.load_entry_points()
TRANSFORMATION_REGISTRY = PROJECT_ROOT / "输出结果" / "智能转换插件" / "已启用转换.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_task(task_id: str, **values: Any) -> None:
    with _tasks_lock:
        task = _tasks[task_id]
        task.update(values)
        task["updated_at"] = _now()


def _get_task(task_id: str) -> dict[str, Any]:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return dict(task)


def _seismic_dimension(geometry: Any) -> tuple[str, str]:
    inline = geometry.inline
    crossline = geometry.crossline
    if inline is not None and crossline is not None:
        unique_inline = int(np.unique(inline).size)
        unique_crossline = int(np.unique(crossline).size)
        if unique_inline > 1 and unique_crossline > 1:
            return "三维地震体", f"{unique_inline} 个 Inline × {unique_crossline} 个 Crossline"
        if unique_inline > 1 or unique_crossline > 1:
            return "二维地震测线", "仅一个方向形成有效网格"
    return "地震数据（待确认维度）", "Inline/Crossline 道头不足"


def _inspection_result(pipeline: WellSeismicPipeline) -> dict[str, Any]:
    report = pipeline.quality_report()
    role_counts = Counter(asset.role for asset in pipeline.assets)
    seismic_items: list[dict[str, Any]] = []
    dimension_counts: Counter[str] = Counter()
    result_rows: list[dict[str, Any]] = []

    for asset, reader in pipeline.seismic:
        geometry = reader.geometry
        if geometry is None:
            continue
        dimension, evidence = _seismic_dimension(geometry)
        inline_count = int(np.unique(geometry.inline).size) if geometry.inline is not None else 0
        crossline_count = int(np.unique(geometry.crossline).size) if geometry.crossline is not None else 0
        grid_cells = inline_count * crossline_count
        grid_coverage = min(1.0, float(geometry.trace_count / grid_cells)) if grid_cells else 0.0
        dimension_counts[dimension] += 1
        seismic_items.append({
            "name": asset.path.name,
            "path": str(asset.path),
            "dimension": dimension,
            "evidence": evidence,
            "trace_count": geometry.trace_count,
            "samples_per_trace": geometry.samples_per_trace,
            "sample_interval_ms": geometry.sample_interval,
            "shape_zyx": [geometry.samples_per_trace, inline_count, crossline_count],
            "inline_count": inline_count,
            "crossline_count": crossline_count,
            "grid_coverage": grid_coverage,
            "confidence": geometry.confidence,
            "issues": geometry.issues,
            "model_compatibility": _input_adapters.compatibilities(geometry),
        })

    # Keep every registered seismic asset in the model-neutral inventory, even
    # when a reader cannot reconstruct its geometry. Visualization and model
    # adapters can then explain why an asset is unavailable instead of silently
    # dropping it from the task.
    inspected_paths = {item["path"] for item in seismic_items}
    seismic_errors = {
        item.get("path", ""): item.get("error", "读取失败")
        for item in report["errors"]
        if item.get("role") == "seismic"
    }
    for asset in pipeline.assets:
        asset_path = str(asset.path)
        if asset.role != "seismic" or asset_path in inspected_paths:
            continue
        error = seismic_errors.get(asset_path, "未能重建 Inline/Crossline 几何")
        dimension_counts["地震数据（待确认维度）"] += 1
        seismic_items.append({
            "name": asset.path.name,
            "path": asset_path,
            "dimension": "地震数据（待确认维度）",
            "evidence": error,
            "trace_count": 0,
            "samples_per_trace": 0,
            "sample_interval_ms": 0.0,
            "shape_zyx": [0, 0, 0],
            "inline_count": 0,
            "crossline_count": 0,
            "grid_coverage": 0.0,
            "confidence": 0.0,
            "issues": [error],
            "model_compatibility": _input_adapters.unavailable_compatibilities(
                f"源文件已登记，但尚未形成模型所需的地震网格：{error}"
            ),
        })

    for dimension in ("二维地震测线", "三维地震体", "地震数据（待确认维度）"):
        count = dimension_counts.get(dimension, 0)
        if count:
            result_rows.append({
                "type": dimension,
                "count": count,
                "evidence": "SEG-Y 二进制头、道头和网格规律",
                "status": "待确认" if "待确认" in dimension else "可读取",
            })

    log_count = role_counts.get("well_logs", 0)
    if log_count:
        result_rows.append({
            "type": "LAS 测井",
            "count": log_count,
            "evidence": "LAS 版本段、井信息段和曲线定义段",
            "status": "可读取",
        })

    metadata_count = role_counts.get("well_metadata", 0)
    uncertain_metadata = sum(
        1 for item in pipeline.metadata_detection if item.get("状态") == "待确认"
    )
    if metadata_count:
        result_rows.append({
            "type": "井基础信息与井轨迹",
            "count": metadata_count,
            "evidence": "字段映射、数据结构和井名关联",
            "status": f"{uncertain_metadata} 个待确认" if uncertain_metadata else "已识别",
        })

    auxiliary_count = role_counts.get("auxiliary", 0)
    if auxiliary_count:
        result_rows.append({
            "type": "其他辅助数据",
            "count": auxiliary_count,
            "evidence": "登记来源，不参与基础匹配",
            "status": "已登记",
        })

    preparation = build_preparation_report(pipeline)
    visualization_preview = build_visualization_preview(pipeline)
    visualization_preview["seismicInventory"] = seismic_items
    registered_seismic_count = sum(1 for asset in pipeline.assets if asset.role == "seismic")
    data_snapshot = {
        "contract_version": "1.0",
        "snapshot_id": None,
        "semantics": "model_neutral",
        "source_assets": {
            "seismic": registered_seismic_count,
            "well_logs": log_count,
            "well_metadata": metadata_count,
            "auxiliary": auxiliary_count,
        },
        "canonical_data": {
            "seismic_geometry": {
                "axes": ["Z", "INLINE", "CROSSLINE"],
                "registered": registered_seismic_count,
                "readable": sum(1 for item in seismic_items if item["trace_count"] > 0),
                "renderable_3d": len(visualization_preview.get("volumes", [])),
                "renderable_2d": len(visualization_preview.get("lines2d", [])),
            },
            "well_entities": {"count": report["summary"]["wells"]},
            "well_logs": {"count": log_count},
        },
        "derived_views": {
            "visualization_preview": {
                "status": "available"
                if visualization_preview.get("volumes") or visualization_preview.get("lines2d") or visualization_preview.get("wellLogs")
                else "unavailable",
                "model_specific": False,
            },
            "well_seismic_samples": {
                "status": "available" if preparation["gates"]["can_build_samples"] else "blocked",
                "optional": True,
                "model_specific": False,
            },
        },
        "downstream_policy": "下游模型必须通过各自输入适配器从该快照派生输入，不得反向修改源数据或通用预处理结果。",
    }

    return {
        "summary": {
            "assets": report["summary"]["assets"],
            "duplicates_skipped": report["summary"]["duplicates_skipped"],
            "wells": report["summary"]["wells"],
            "seismic_files": report["summary"]["seismic_files"],
            "registered_seismic_files": registered_seismic_count,
            "log_files": log_count,
            "metadata_files": metadata_count,
            "auxiliary_files": auxiliary_count,
            "uncertain": uncertain_metadata + dimension_counts.get("地震数据（待确认维度）", 0),
            "errors": report["summary"]["errors"],
        },
        "rows": result_rows,
        "seismic": seismic_items,
        "wells": report["wells"],
        "well_entities": report["wells"],
        "assets": report["assets"],
        "metadata_detection": pipeline.metadata_detection,
        "errors": report["errors"],
        "duplicates": report["duplicates"],
        "inventory": pipeline.automatic_inventory,
        "preparation": preparation,
        "data_snapshot": data_snapshot,
        "visualization_preview": visualization_preview,
    }


def inspect_paths(request: InspectionRequest, progress: Any = None) -> dict[str, Any]:
    if not any((request.seismic_paths, request.log_paths, request.well_paths, request.auxiliary_paths)):
        raise ValueError("至少需要登记一个有效数据路径")
    if progress:
        progress(10, "正在校验绝对路径")

    manifest, inventory = build_explicit_paths_manifest(
        seismic_directory=request.seismic_paths,
        log_directory=request.log_paths,
        metadata_directory=request.well_paths,
        auxiliary_directory=request.auxiliary_paths,
        recursive=request.recursive,
        require_seismic=False,
        require_logs=False,
    )
    if progress:
        progress(30, "正在建立数据资产目录和检查重复文件")

    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    pipeline.automatic_inventory = inventory
    if progress:
        progress(45, f"已登记 {len(pipeline.assets)} 个数据资产，正在读取文件头和井数据")

    pipeline.ingest()
    if progress:
        progress(90, "正在汇总识别结果、置信度和异常信息")
    return _inspection_result(pipeline)


def preprocess_paths(
    request: PreprocessingRequest,
    *,
    task_id: str,
    progress: Any = None,
) -> dict[str, Any]:
    if not request.seismic_paths:
        raise ValueError("至少需要一个地震数据路径")
    if not request.log_paths:
        raise ValueError("至少需要一个测井数据路径")
    if progress:
        progress(8, "正在校验路径并建立数据资产目录")

    manifest, inventory = build_explicit_paths_manifest(
        seismic_directory=request.seismic_paths,
        log_directory=request.log_paths,
        metadata_directory=request.well_paths,
        auxiliary_directory=request.auxiliary_paths,
        recursive=request.recursive,
    )
    pipeline = WellSeismicPipeline(
        manifest,
        CONFIG_DIR,
        use_llm_fallback=request.use_llm_fallback,
    )
    pipeline.automatic_inventory = inventory
    if progress:
        progress(25, f"已登记 {len(pipeline.assets)} 个资产，正在读取和标准化数据")

    pipeline.ingest()
    if progress:
        progress(62, "正在重建井轨迹并构建井震匹配样本")

    samples = pipeline.build_samples()
    tie_status_counts = Counter(
        str(item.get("status", "horizontal_only")) for item in pipeline.well_ties
    )
    valid_window_count = sum(1 for item in samples if item.get("seismic_window_valid"))
    training_eligible_count = sum(1 for item in samples if item.get("training_eligible"))
    coordinate_reference_verified = bool(
        pipeline.config.get("matching", {}).get("coordinate_reference", {}).get("verified", False)
    )
    inspection = _inspection_result(pipeline)
    output_directory = (
        Path(request.output_directory).expanduser().resolve()
        if request.output_directory and request.output_directory.strip()
        else PROJECT_ROOT / "输出结果" / f"前端任务_{task_id[:8]}"
    )
    if progress:
        progress(88, "正在写入中文质量报告、样本索引和多模态样本")
    output_files = pipeline.write_outputs(output_directory)
    return {
        **inspection,
        "matching": {
            "sample_count": len(samples),
            "valid_window_count": valid_window_count,
            "training_eligible_count": training_eligible_count,
            "coordinate_reference_verified": coordinate_reference_verified,
            "vertical_alignment_counts": dict(tie_status_counts),
            "output_directory": str(output_directory),
            "output_files": {name: str(path) for name, path in output_files.items()},
        },
    }


def _run_inspection(task_id: str, request: InspectionRequest) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="任务已开始")
        result = inspect_paths(request, update)
        result["data_snapshot"]["snapshot_id"] = task_id
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="路径预检与数据识别完成",
            result=result,
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="路径预检失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _run_preprocessing(task_id: str, request: PreprocessingRequest) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="预处理任务已开始")
        result = preprocess_paths(request, task_id=task_id, progress=update)
        result["data_snapshot"]["snapshot_id"] = task_id
        result["data_snapshot"]["derived_views"]["well_seismic_samples"].update({
            "status": "generated",
            "sample_count": result.get("matching", {}).get("sample_count", 0),
            "valid_window_count": result.get("matching", {}).get("valid_window_count", 0),
            "training_eligible_count": result.get("matching", {}).get("training_eligible_count", 0),
            "output_directory": result.get("matching", {}).get("output_directory", ""),
        })
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message="数据预处理与多模态样本构建完成",
            result=result,
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="数据预处理与匹配失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )


def _run_prediction(task_id: str, request: PredictionRequest) -> None:
    def update(progress: int, message: str) -> None:
        _set_task(task_id, status="running", progress=progress, message=message)

    try:
        _set_task(task_id, status="running", progress=1, message="模型推理任务已开始")
        source = Path(request.seismic_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"地震文件不存在：{source}")
        if source.suffix.lower() not in {".sgy", ".segy"}:
            raise ValueError("当前三维体推理接口接收 .sgy 或 .segy 文件")
        output = (
            Path(request.output_directory).expanduser().resolve()
            if request.output_directory and request.output_directory.strip()
            else PROJECT_ROOT / "model_outputs" / f"{request.task_id}_{request.model_id}_{task_id[:8]}"
        )
        result = _prediction_runners.run(
            request.model_id,
            ModelInputRequest(
                source=source,
                crop_start=request.crop_start,
                crop_size=request.crop_size,
                options=dict(request.options),
            ),
            adapters=_input_adapters,
            config=_platform_config,
            project_root=PROJECT_ROOT,
            output_directory=output,
            device_name=request.device,
            threshold=request.threshold,
            patch_size=request.patch_size,
            overlap=request.overlap,
            options=request.options,
            progress=update,
        )
        result["task_id"] = request.task_id
        result["task_name"] = _interpretation_registry.get(request.task_id).name
        _set_task(
            task_id,
            status="completed",
            progress=100,
            message=f"{result['task_name']}推理完成",
            result={"prediction": result, "source_task_id": request.source_task_id},
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            progress=100,
            message="模型推理失败",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
def _queue_task(task_type: str, message: str) -> str:
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "progress": 0,
            "message": message,
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
        }
    return task_id


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "地层慧眼", "version": app.version}


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return build_platform_capabilities(
        project_root=PROJECT_ROOT,
        platform_config=_platform_config,
        model_registry=_model_registry,
        interpretation_registry=_interpretation_registry,
        fusion_registry=_fusion_registry,
        input_adapters=_input_adapters,
        prediction_runners=_prediction_runners,
    )


@app.get("/api/v1/llm/status")
def llm_status() -> dict[str, Any]:
    return load_llm_settings(_platform_config).public_status()


@app.get("/api/v1/demo-paths")
def demo_paths() -> dict[str, Any]:
    reference = PROJECT_ROOT.parent / "整理版_地震解释与储层反演数据"
    seismic_2d = reference / "01_综合解释训练数据" / "01_二维地震解释"
    seismic_3d = reference / "01_综合解释训练数据" / "02_三维地震解释"
    logs = reference / "01_综合解释训练数据" / "03_井数据" / "02_LAS测井曲线"
    wells = reference / "01_综合解释训练数据" / "03_井数据"
    available = all(path.exists() for path in (seismic_2d, seismic_3d, logs, wells))
    return {
        "available": available,
        "seismic_paths": [str(seismic_2d), str(seismic_3d)] if available else [],
        "log_paths": [str(logs)] if available else [],
        "well_paths": [str(wells)] if available else [],
        "auxiliary_paths": [],
    }


@app.post("/api/v1/data-preparation/tasks", response_model=TaskCreated, status_code=202)
def create_data_preparation_task(request: InspectionRequest) -> TaskCreated:
    if not any((request.seismic_paths, request.log_paths, request.well_paths, request.auxiliary_paths)):
        raise HTTPException(status_code=422, detail="至少需要登记一个有效数据路径")
    task_id = _queue_task("data_preparation", "数据准备任务已进入本地队列")
    _executor.submit(_run_inspection, task_id, request)
    return TaskCreated(task_id=task_id, status="queued", message="数据准备任务已进入本地队列")


@app.post("/api/v1/data-inspection/tasks", response_model=TaskCreated, status_code=202, deprecated=True)
def create_inspection_task(request: InspectionRequest) -> TaskCreated:
    return create_data_preparation_task(request)


@app.post("/api/v1/sample-building/tasks", response_model=TaskCreated, status_code=202)
def create_sample_building_task(request: PreprocessingRequest) -> TaskCreated:
    if not request.seismic_paths:
        raise HTTPException(status_code=422, detail="至少需要一个地震数据路径")
    if not request.log_paths:
        raise HTTPException(status_code=422, detail="至少需要一个测井数据路径")
    task_id = _queue_task("sample_building", "样本构建任务已进入本地队列")
    _executor.submit(_run_preprocessing, task_id, request)
    return TaskCreated(task_id=task_id, status="queued", message="样本构建任务已进入本地队列")


@app.post("/api/v1/data-preparation/multimodal-view-tasks", response_model=TaskCreated, status_code=202)
def create_multimodal_data_view_task(request: PreprocessingRequest) -> TaskCreated:
    """Build the optional well-seismic view inside the data-preparation layer."""
    return create_sample_building_task(request)


@app.post("/api/v1/preprocessing/tasks", response_model=TaskCreated, status_code=202, deprecated=True)
def create_preprocessing_task(request: PreprocessingRequest) -> TaskCreated:
    return create_sample_building_task(request)


@app.post("/api/v1/prediction/tasks", response_model=TaskCreated, status_code=202)
def create_prediction_task(request: PredictionRequest) -> TaskCreated:
    if not request.seismic_path.strip():
        raise HTTPException(status_code=422, detail="请选择一个三维 SEG-Y 文件")
    try:
        _interpretation_registry.get(request.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        model_spec = next(spec for spec in _model_registry.list_specs() if spec.id == request.model_id)
    except StopIteration as exc:
        raise HTTPException(status_code=422, detail=f"模型尚未注册：{request.model_id}") from exc
    if model_spec.metadata.get("prediction_task") != request.task_id:
        raise HTTPException(
            status_code=422,
            detail=f"模型 {request.model_id} 不属于解释任务 {request.task_id}",
        )
    if request.model_id not in {item["model_id"] for item in _input_adapters.capabilities()}:
        raise HTTPException(status_code=422, detail=f"模型没有已注册的输入适配器：{request.model_id}")
    if request.model_id not in _prediction_runners.model_ids():
        raise HTTPException(status_code=422, detail=f"模型没有已注册的推理运行器：{request.model_id}")
    task_id = _queue_task("model_prediction", "模型推理任务已进入本地队列")
    _set_task(task_id, parent_task_id=request.source_task_id)
    _executor.submit(_run_prediction, task_id, request)
    return TaskCreated(task_id=task_id, status="queued", message="模型推理任务已进入本地队列")


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    try:
        return _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc


@app.get("/api/v1/tasks/{task_id}/artifacts/{artifact_name}")
def get_prediction_artifact(task_id: str, artifact_name: str) -> FileResponse:
    """Serve only an artifact explicitly registered by a completed prediction."""
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    if task.get("task_type") != "model_prediction" or task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="仅已完成的模型推理任务可以读取产物")
    prediction = (task.get("result") or {}).get("prediction")
    outputs = prediction.get("outputs") if isinstance(prediction, dict) else None
    if not isinstance(outputs, dict) or artifact_name not in outputs:
        raise HTTPException(status_code=404, detail=f"推理产物不存在：{artifact_name}")
    raw_path = outputs.get(artifact_name)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(status_code=404, detail=f"推理产物不可用：{artifact_name}")
    artifact_path = Path(raw_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"推理产物文件不存在：{artifact_name}")
    return FileResponse(
        artifact_path,
        filename=artifact_path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, no-store"},
    )


@app.post("/api/v1/tasks/{task_id}/issues/{issue_id}/confirmation")
def confirm_issue(
    task_id: str,
    issue_id: str,
    request: IssueConfirmationRequest,
) -> dict[str, Any]:
    if request.decision not in {"确认采用", "暂不采用"}:
        raise HTTPException(status_code=422, detail="decision必须为“确认采用”或“暂不采用”")
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在或服务已重启")
        result = task.get("result") or {}
        preparation = result.get("preparation") or {}
        issue = next((item for item in preparation.get("issues", []) if item.get("id") == issue_id), None)
        if issue is None:
            raise HTTPException(status_code=404, detail="当前任务中没有该问题")
        action = request.action.strip() or str(issue.get("recommended_action", ""))
        candidates = set(issue.get("candidate_actions", []))
        if request.decision == "确认采用" and (not action or action not in candidates):
            raise HTTPException(status_code=422, detail="只能确认后端提供的安全候选方案")
        issue["confirmation_status"] = "已确认采用" if request.decision == "确认采用" else "暂不采用"
        issue["confirmed_action"] = action if request.decision == "确认采用" else ""
        issue["confirmed_at"] = _now()
        task["updated_at"] = _now()
        return dict(issue)


@app.post("/api/v1/tasks/{task_id}/issues/{issue_id}/transformation-drafts")
def generate_transformation_draft(task_id: str, issue_id: str) -> dict[str, Any]:
    try:
        task = _get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启") from exc
    preparation = (task.get("result") or {}).get("preparation") or {}
    issue = next((item for item in preparation.get("issues", []) if item.get("id") == issue_id), None)
    if issue is None:
        raise HTTPException(status_code=404, detail="当前任务中没有该问题")
    settings = load_llm_settings(_platform_config)
    draft = create_transformation_draft(
        task_id=task_id,
        issue=issue,
        config=_platform_config,
        generator=build_structured_generator(settings),
    )
    with _tasks_lock:
        _transformation_drafts[draft["id"]] = draft
    return draft


@app.post("/api/v1/transformation-drafts/{draft_id}/activation")
def activate_transformation_draft(
    draft_id: str,
    request: TransformationActivationRequest,
) -> dict[str, Any]:
    if request.confirmation != "确认启用":
        raise HTTPException(status_code=422, detail="必须明确提交“确认启用”")
    with _tasks_lock:
        draft = _transformation_drafts.get(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="转换草案不存在或服务已重启")
        try:
            activate_transformation(draft, TRANSFORMATION_REGISTRY)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        task = _tasks.get(str(draft.get("task_id")))
        if task is not None:
            issues = ((task.get("result") or {}).get("preparation") or {}).get("issues", [])
            issue = next((item for item in issues if item.get("id") == draft.get("issue_id")), None)
            if issue is not None:
                issue["transformation_draft_id"] = draft_id
                issue["confirmation_status"] = "已启用转换插件"
                issue["confirmed_action"] = draft.get("title", "受控转换适配器")
                issue["confirmed_at"] = draft.get("activated_at", "")
        return dict(draft)


def _assistant_context(task_id: str | None) -> dict[str, Any]:
    if not task_id:
        return {"task": "尚未选择任务"}
    try:
        task = _get_task(task_id)
    except KeyError:
        return {"task": "任务不存在或服务已重启"}
    result = task.get("result") or {}
    preparation = result.get("preparation") or {}
    return {
        "task_id": task_id,
        "task_type": task.get("task_type"),
        "status": task.get("status"),
        "summary": result.get("summary", {}),
        "stages": [
            {"id": stage.get("id"), "name": stage.get("name"), "status": stage.get("status"), "issue_count": stage.get("issue_count")}
            for stage in preparation.get("stages", [])
        ],
        "issues": [
            {
                "stage": issue.get("stage"),
                "severity": issue.get("severity"),
                "title": issue.get("title"),
                "message": issue.get("message"),
                "affected_count": issue.get("affected_count", 1),
                "status": issue.get("confirmation_status"),
            }
            for issue in preparation.get("issues", [])[:12]
        ],
        "gates": preparation.get("gates", {}),
    }


@app.post("/api/v1/assistant/chat")
def assistant_chat(request: AssistantChatRequest) -> dict[str, Any]:
    settings = load_llm_settings(_platform_config)
    generator = build_structured_generator(settings)
    context = _assistant_context(request.task_id)
    allowed_targets = ["overview", "preparation", "visualization", "samples", "models", "prediction", "evaluation", "settings"]
    if generator is not None:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "actions": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "target": {"type": "string", "enum": allowed_targets},
                        },
                        "required": ["label", "target"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answer", "actions"],
            "additionalProperties": False,
        }
        try:
            response, metadata = generator.generate_json(
                system_prompt=(
                    "你是地层慧眼平台的井震数据工程助手。请只依据提供的任务摘要回答，使用简洁中文；"
                    "不得声称看过未提供的原始SEG-Y或完整LAS，不得编造指标。"
                    "涉及数据修改时说明需要生成受控转换草案并由人工启用。"
                ),
                payload={"question": request.message, "workflow_context": context},
                schema_name="well_seismic_assistant_response",
                schema=schema,
            )
            return {
                "answer": str(response.get("answer", ""))[:3000],
                "actions": [item for item in response.get("actions", []) if item.get("target") in allowed_targets],
                "source": "LLM",
                **metadata,
            }
        except Exception as exc:
            fallback_error = str(exc)[:300]
    else:
        fallback_error = "LLM未启用或未完成配置"
    issue_count = len(context.get("issues", [])) if isinstance(context.get("issues"), list) else 0
    answer = (
        f"当前任务已记录 {issue_count} 项需要关注的问题。建议先进入“数据准备”，按阶段查看证据；"
        "对无法由知识库确定的映射，可生成受控转换适配器并在自动测试通过后人工启用。"
        if request.task_id
        else "请先执行数据准备。完成后我可以结合当前任务阶段、问题和放行条件给出诊断。"
    )
    return {
        "answer": answer,
        "actions": [{"label": "查看数据准备", "target": "preparation"}],
        "source": "本地工作流助手",
        "provider": "local",
        "model": "workflow-context-v1",
        "request_id": "",
        "warning": fallback_error,
    }


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="前端尚未构建，请先运行 npm run build")
    return FileResponse(index)


def _dashboard_document(path: Path) -> str:
    """兼容旧的srcdoc打包文件，并在服务端去掉多余iframe外壳。"""
    document = path.read_text(encoding="utf-8-sig")
    marker = 'srcdoc="'
    start = document.find(marker)
    finish = document.rfind('"></iframe>')
    if start >= 0 and finish > start:
        return html.unescape(document[start + len(marker):finish])
    return document


_EMBEDDED_VISUALIZATION_STYLE = """
<style id="well-seismic-embedded-mode">
html, body {
  min-width: 0 !important;
  color: #20242b !important;
  background: #f3f5f8 !important;
}
body { padding: 0 !important; }
.current-task-banner {
  display: flex;
  min-height: 46px;
  gap: 12px;
  align-items: center;
  padding: 9px 20px;
  color: #4f5966;
  font: 500 13px/1.4 Inter, "Microsoft YaHei UI", sans-serif;
  background: #fff;
  border-bottom: 1px solid #e5e8ee;
}
.current-task-banner strong { color: #1f5fd4; font-size: 14px; }
.current-task-banner span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#future-model-interfaces {
  --font-size-base: 15px;
  width: 100% !important;
  max-width: none !important;
  padding: 0 !important;
  font-size: 15px !important;
}
#future-model-interfaces .workspace {
  grid-template-columns: 276px minmax(0, 1fr) !important;
  gap: 0 !important;
  align-items: stretch !important;
  background: #f3f5f8 !important;
}
#future-model-interfaces .workspace > .sidebar {
  min-height: 100vh;
  padding: 24px 20px;
  background: #f8f9fb;
  border-right: 1px solid #e5e8ee;
  box-shadow: none;
}
#future-model-interfaces .workspace > .main {
  padding: 20px 22px 32px;
  background: #f3f5f8;
}
#future-model-interfaces .workspace > .sidebar > h2,
#future-model-interfaces #active-panel-label,
#future-model-interfaces .side-nav,
#future-model-interfaces .nav-group-label,
#future-model-interfaces .summary,
#future-model-interfaces [data-side-options]:not([data-side-options="volume"]),
#future-model-interfaces [data-panel-content]:not([data-panel-content="volume"]) {
  display: none !important;
}
#future-model-interfaces [data-side-options="volume"],
#future-model-interfaces [data-panel-content="volume"] {
  display: block !important;
}
#future-model-interfaces .side-options { margin: 0 !important; }
#future-model-interfaces .side-options h3 {
  margin: 0 0 16px;
  color: #252a31;
  font-size: 16px;
  font-weight: 650;
}
#future-model-interfaces .side-options .option-stack { gap: 18px; }
#future-model-interfaces .side-options .option-stack + h3 {
  margin-top: 30px;
  padding-top: 22px;
  border-top: 1px solid #e5e8ee;
}
#future-model-interfaces .form-label,
#future-model-interfaces .form-check-label {
  color: #4e5763;
  font-size: 13px;
}
#future-model-interfaces .form-select {
  min-height: 38px;
  background-color: #fff;
  border-color: #dfe3e9;
  border-radius: 8px;
}
#future-model-interfaces .form-range { accent-color: #2468f2; }
#future-model-interfaces #reset-view {
  min-height: 38px;
  color: #303640;
  background: #fff;
  border-color: #dce0e7;
  border-radius: 8px;
}
#future-model-interfaces [data-panel-content="volume"] {
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#future-model-interfaces #volume-detail {
  min-height: 30px;
  padding: 0 2px 10px;
  color: #7a838f;
  font-size: 12px;
}
#future-model-interfaces .volume3d-stage { margin-top: 0 !important; }
#future-model-interfaces .canvas-label {
  min-height: 32px;
  color: #2b3037;
  font-size: 13px;
  font-weight: 550;
}
#future-model-interfaces .volume3d-canvas {
  min-height: 650px;
  border: 1px solid #e1e5eb;
  border-radius: 10px;
  box-shadow: 0 8px 24px rgb(31 47 68 / 7%);
}
#future-model-interfaces .well-overlay { border-radius: 10px; }
#future-model-interfaces .volume-grid { gap: 12px; }
#future-model-interfaces .volume-grid > div {
  padding: 10px;
  background: #fff;
  border: 1px solid #e5e8ee;
  border-radius: 9px;
}
#future-model-interfaces .seismic-canvas {
  border-color: #e1e5eb;
  border-radius: 6px;
}
@media (min-width: 850px) {
  #future-model-interfaces .volume-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  }
}
@media (max-width: 1100px) {
  #future-model-interfaces .workspace { grid-template-columns: 240px minmax(0, 1fr) !important; }
  #future-model-interfaces .volume3d-canvas { min-height: 480px; }
}
</style>
"""

_EMBEDDED_VISUALIZATION_SCRIPT = """
<script id="well-seismic-embedded-activation">
window.addEventListener("load", () => {
  const root = document.getElementById("future-model-interfaces");
  const volumeButton = root && root.querySelector('[data-panel="volume"]');
  if (volumeButton) volumeButton.click();
});
</script>
"""


def _visualization_unavailable(title: str, message: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<style>body{margin:0;display:grid;min-height:100vh;place-items:center;background:#f3f5f8;"
        "font:500 15px/1.7 Inter,'Microsoft YaHei UI',sans-serif;color:#68717d}.state{max-width:560px;"
        "padding:32px 38px;text-align:center;background:#fff;border:1px solid #e2e6ed;border-radius:14px;"
        "box-shadow:0 14px 40px rgba(40,58,82,.08)}h2{margin:0 0 8px;color:#20252d;font-size:21px}"
        "p{margin:0}</style></head><body><div class='state'>"
        f"<h2>{html.escape(title)}</h2><p>{html.escape(message)}</p></div></body></html>"
    )


@app.post("/api/v1/visualization/viser-slices", include_in_schema=False)
def move_viser_slices(request: ViserSliceRequest) -> dict[str, Any]:
    if request.x is None and request.y is None and request.z is None:
        raise HTTPException(status_code=422, detail="至少提供一个切片索引")
    try:
        return update_viser_slices(
            request.task_id,
            request.asset_index,
            {"x": request.x, "y": request.y, "z": request.z},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/visualization/viser-layer-mode", include_in_schema=False)
def change_viser_layer_mode(request: ViserLayerModeRequest) -> dict[str, Any]:
    try:
        return update_viser_layer_mode(
            request.task_id,
            request.asset_index,
            request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/cigvis/plotly.min.js", include_in_schema=False)
def cigvis_plotly_bundle() -> Response:
    try:
        source = plotly_javascript(PROJECT_ROOT)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CIGVis Plotly资源不可用：{exc}") from exc
    return Response(
        content=source,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/统一数据可视化", include_in_schema=False)
def unified_data_visualization(
    embed: bool = True,
    task_id: str | None = None,
    asset: int = 0,
) -> HTMLResponse:
    if not task_id:
        return _visualization_unavailable(
            "尚未绑定当前任务",
            "请从平台完成数据准备后点击“查看当前任务数据”，系统不会再默认展示历史示例数据。",
        )
    try:
        requested_task = _get_task(task_id)
    except KeyError:
        return _visualization_unavailable("任务数据已失效", "服务可能已重启，请重新执行数据准备。")
    data_task = requested_task
    prediction_result: dict[str, Any] | None = None
    if requested_task.get("task_type") == "model_prediction":
        raw_prediction = (requested_task.get("result") or {}).get("prediction")
        if not isinstance(raw_prediction, dict):
            return _visualization_unavailable("预测结果尚未就绪", "请等待推理任务完成后再打开结果工作台。")
        prediction_result = raw_prediction
        source_task_id = requested_task.get("parent_task_id") or (requested_task.get("result") or {}).get("source_task_id")
        if not source_task_id:
            return _visualization_unavailable("缺少源数据任务", "该推理结果没有登记source_task_id，无法重建背景地震体。")
        try:
            data_task = _get_task(str(source_task_id))
        except KeyError:
            return _visualization_unavailable("源数据任务已失效", "请重新执行数据准备，再进入预测解释任务。")
    preview = copy.deepcopy((data_task.get("result") or {}).get("visualization_preview") or {})
    if prediction_result is not None:
        try:
            prediction_volume = build_prediction_visualization_payload(
                prediction_result,
                config=_platform_config,
                segy_options={"profile": str(prediction_result.get("input", {}).get("geometry_profile", "standard_3d"))},
                max_shape_zyx=(128, 96, 96),
            )
            prediction_descriptor = prediction_volume["predictionVisualization"]
            prediction_volume["embeddedWells"] = []
            preview["volumes"] = [prediction_volume, *preview.get("volumes", [])]
            preview["activePrediction"] = {
                "modelId": str(prediction_descriptor["modelId"]),
                "taskId": task_id,
                "preferredLayer": prediction_descriptor.get("preferredLayer"),
            }
        except Exception as exc:
            return _visualization_unavailable(
                "预测结果可视化失败",
                f"{type(exc).__name__}: {exc}",
            )
    if not preview.get("volumes") and not preview.get("lines2d"):
        detail = "；".join(str(item) for item in preview.get("issues", [])[:2])
        return _visualization_unavailable(
            "当前任务没有可渲染的地震数据",
            detail or "请先在数据准备中读取二维测线或形成可靠Inline/Crossline网格的三维SEG-Y。",
        )
    try:
        document = render_cigvis_workbench(
            PROJECT_ROOT,
            preview,
            task_id=task_id,
            asset_index=asset,
            embed=embed,
        )
    except Exception as exc:
        return _visualization_unavailable(
            "CIGVis可视化启动失败",
            f"{type(exc).__name__}: {exc}",
        )
    return HTMLResponse(document, headers={"Cache-Control": "no-store"})


@app.get("/三维地震看板", include_in_schema=False, deprecated=True)
def seismic_dashboard(task_id: str | None = None) -> HTMLResponse:
    return unified_data_visualization(task_id=task_id, embed=False)


def run() -> None:
    import uvicorn

    host = os.getenv("WELL_SEISMIC_HOST", "127.0.0.1")
    port = int(os.getenv("WELL_SEISMIC_PORT", "8000"))
    uvicorn.run("well_seismic.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
