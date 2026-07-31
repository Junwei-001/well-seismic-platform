from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from well_seismic.api_models import PredictionRequest
from well_seismic.config import load_config
from well_seismic.interpretation import build_default_interpretation_registry
from well_seismic.modeling import (
    ModelInputAdapterRegistry,
    ModelInputBatch,
    ModelInputRequest,
    SurfaceSegInputAdapter,
    build_default_registry,
)
from well_seismic.prediction import (
    _verify_surface_checkpoint,
    build_default_prediction_runners,
    run_surface_seg_prediction,
)


def _geometry(
    inline: list[int],
    crossline: list[int],
    *,
    sample_count: int = 24,
) -> SimpleNamespace:
    return SimpleNamespace(
        inline=np.asarray(inline, dtype=np.int64),
        crossline=np.asarray(crossline, dtype=np.int64),
        trace_count=len(inline),
        samples_per_trace=sample_count,
        sample_interval=2.0,
        profile="standard_3d",
        issues=[
            "inline_byte=189:confidence=1.000",
            "crossline_byte=193:confidence=1.000",
        ],
    )


def _fake_surface_project(root: Path) -> tuple[Path, Path]:
    surface_root = root / "surface"
    for stage in ("segformer-base", "segformer-refine", "mask2former"):
        checkpoint = surface_root / "models" / stage / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint" * 200)
    source = root / "input.sgy"
    source.write_bytes(b"test")
    return surface_root, source


class _PreparedSurfaceAdapter:
    model_id = "seismic_surface_seg"

    def __init__(self, native_inline_count: int | None = None) -> None:
        self.native_inline_count = native_inline_count

    def capabilities(self) -> dict[str, object]:
        return {"model_id": self.model_id}

    def prepare(self, request: ModelInputRequest) -> ModelInputBatch:
        provenance: dict[str, object] = {
            "source": str(request.source),
            "shape_ics": [2, 3, 4],
            "source_shape_zyx": [4, 2, 3],
            "geometry_profile": "standard_3d",
            "geometry_mode": (
                "platform_inferred_inline_count"
                if self.native_inline_count is not None
                else "standard_headers"
            ),
            "materialization": "model_native_segy_reader",
        }
        if self.native_inline_count is not None:
            provenance["native_inline_count"] = self.native_inline_count
        return ModelInputBatch(
            model_id=self.model_id,
            array=None,
            valid_mask=np.ones((2, 3), dtype=bool),
            axes=("INLINE", "CROSSLINE", "SAMPLE"),
            provenance=provenance,
        )


def _fake_upstream(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [
            [[0, 0, 1, 1], [0, 1, 1, 2], [0, 1, 2, 2]],
            [[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2]],
        ],
        dtype=np.int16,
    )
    confidence = np.full(labels.shape, 0.75, dtype=np.float16)
    mask_path = output_dir / "mask.npy"
    confidence_path = output_dir / "confidence.npy"
    overview_path = output_dir / "overview.png"
    np.save(mask_path, labels)
    np.save(confidence_path, confidence)
    overview_path.write_bytes(b"png")
    summary = {
        "volume_shape": list(labels.shape),
        "mask_dtype": str(labels.dtype),
        "label_range": [0, 2],
        "device": "cpu",
        "amplitude_scaling": {"requested": "auto", "effective": "robust"},
        "prior_compatibility_mode": "segformer-base-as-refine-prior",
        "geometry": {
            "inline_count": 2,
            "xline_count": 3,
            "sample_count": 4,
        },
        "checkpoints": {"segformer_base": {"epoch": 1}},
        "artifacts": {
            "mask_npy": str(mask_path),
            "mask_sgy": None,
            "confidence_npy": str(confidence_path),
            "overview": str(overview_path),
        },
        "elapsed_seconds": 1.25,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return summary


def test_surface_seg_config_task_model_adapter_and_runner_are_registered() -> None:
    config = load_config(Path(__file__).parents[1] / "configs", {"inputs": []})
    assert config["surface_seg"]["amplitude_mode"] == "auto"
    assert config["surface_seg"]["segformer_batch_size"] == 2
    assert config["surface_seg"]["mask2former_batch_size"] == 1

    models = {item.id: item for item in build_default_registry().list_specs()}
    assert models["seismic_surface_seg"].metadata["prediction_task"] == "strata"

    tasks = {item.id: item for item in build_default_interpretation_registry().list_specs()}
    assert tasks["strata"].short_name == "地层分割"
    assert "seismic_surface_seg" in build_default_prediction_runners().model_ids()


def test_surface_adapter_accepts_standard_poststack_and_rejects_duplicates() -> None:
    adapter = SurfaceSegInputAdapter({"surface_seg": {}})
    ready = adapter.compatibility(
        _geometry([100, 100, 100, 101, 101, 101], [20, 21, 22, 20, 21, 22])
    )
    assert ready["ready"] is True
    assert ready["shape_ics"] == [2, 3, 24]

    duplicate = adapter.compatibility(
        _geometry([100, 100, 101, 101, 101], [20, 21, 20, 21, 21])
    )
    assert duplicate["ready"] is False
    assert duplicate["duplicate_trace_count"] == 1
    assert "叠前" in duplicate["reason"]


def test_surface_adapter_allows_explicit_inline_count_fallback() -> None:
    adapter = SurfaceSegInputAdapter({"surface_seg": {}})
    geometry = _geometry([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    geometry.issues = []
    compatibility = adapter.compatibility(
        geometry,
        options={"inline_count": 2},
    )
    assert compatibility["ready"] is True
    assert compatibility["shape_ics"] == [2, 3, 24]
    assert compatibility["geometry_mode"] == "explicit_inline_count"


def test_surface_adapter_auto_passes_platform_verified_ordered_grid() -> None:
    adapter = SurfaceSegInputAdapter({"surface_seg": {}})
    geometry = _geometry(
        [500, 500, 500, 501, 501, 501],
        [220, 221, 222, 220, 221, 222],
    )
    geometry.issues = [
        "inline_byte=189:confidence=0.890",
        "crossline_byte=21:confidence=0.984",
    ]

    compatibility = adapter.compatibility(geometry)

    assert compatibility["ready"] is True
    assert compatibility["geometry_mode"] == "platform_inferred_inline_count"
    assert compatibility["native_inline_count"] == 2
    assert compatibility["recommended_options"] == {"inline_count": 2}
    assert compatibility["shape_ics"] == [2, 3, 24]


def test_surface_adapter_does_not_guess_unordered_nonstandard_grid() -> None:
    adapter = SurfaceSegInputAdapter({"surface_seg": {}})
    geometry = _geometry(
        [500, 501, 500, 501, 500, 501],
        [220, 220, 221, 221, 222, 222],
    )
    geometry.issues = [
        "inline_byte=189:confidence=0.890",
        "crossline_byte=21:confidence=0.984",
    ]

    compatibility = adapter.compatibility(geometry)

    assert compatibility["ready"] is False
    assert compatibility["platform_ordered_grid"] is False
    assert compatibility["native_inline_count"] is None


def test_surface_checkpoint_manifest_rejects_corrupted_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint" * 200)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _verify_surface_checkpoint(
            checkpoint,
            {"size": checkpoint.stat().st_size, "sha256": "0" * 64},
        )


def test_surface_runner_normalizes_upstream_result_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface_root, source = _fake_surface_project(tmp_path)
    adapters = ModelInputAdapterRegistry()
    adapters.register(_PreparedSurfaceAdapter(native_inline_count=2))

    class _Runtime:
        @staticmethod
        def run_inference(
            input_path: Path,
            output_dir: Path,
            **options: object,
        ) -> dict[str, object]:
            assert input_path == source
            assert options["mask_threshold"] == pytest.approx(0.6)
            assert options["inline_count"] == 2
            return _fake_upstream(output_dir)

    monkeypatch.setattr(
        "well_seismic.prediction._load_surface_seg_runtime",
        lambda _root: _Runtime(),
    )
    result = run_surface_seg_prediction(
        ModelInputRequest(
            source=source,
            crop_size=(32, 32, 32),
            options={"amplitude_mode": "auto"},
        ),
        adapters=adapters,
        config={
            "surface_seg": {
                "model_root": str(surface_root),
                "models_dir": str(surface_root / "models"),
            }
        },
        project_root=tmp_path,
        output_directory=tmp_path / "output",
        device_name="cpu",
        threshold=0.6,
    )

    assert result["model_id"] == "seismic_surface_seg"
    assert result["input"]["axes"] == ["INLINE", "CROSSLINE", "SAMPLE"]
    assert result["input"]["source_shape_zyx"] == [4, 2, 3]
    assert result["inference"]["execution_backend"] == "in_process"
    assert result["inference"]["mask_threshold"] == pytest.approx(0.6)
    assert result["inference"]["inline_count"] == 2
    assert result["segmentation"]["shape_ics"] == [2, 3, 4]
    assert result["segmentation"]["label_range"] == [0, 2]
    assert result["segmentation"]["max_instances_per_inline"] == 3
    assert result["segmentation"]["instance_count"] == 3
    assert result["segmentation"]["cross_inline_consistent"] is False
    assert result["segmentation"]["confidence_mean"] == pytest.approx(0.75)
    assert result["segmentation"]["confidence"]["mean"] == pytest.approx(0.75)
    assert Path(result["outputs"]["metadata_json"]).is_file()
    assert "mask_sgy" not in result["outputs"]


def test_api_passes_model_options_and_serves_only_registered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from well_seismic import api

    source = tmp_path / "source.sgy"
    source.write_bytes(b"sgy")
    overview = tmp_path / "overview.png"
    overview.write_bytes(b"png")
    captured: dict[str, object] = {}

    def fake_run(
        model_id: str,
        request: ModelInputRequest,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["model_id"] = model_id
        captured["options"] = request.options
        return {
            "model_id": model_id,
            "model_name": "Surface",
            "outputs": {"overview": str(overview)},
        }

    monkeypatch.setattr(api._prediction_runners, "run", fake_run)
    task_id = api._queue_task("model_prediction", "queued")
    try:
        api._run_prediction(
            task_id,
            PredictionRequest(
                task_id="strata",
                model_id="seismic_surface_seg",
                seismic_path=str(source),
                options={"inline_count": 2, "amplitude_mode": "robust"},
            ),
        )
        task = api._get_task(task_id)
        assert task["status"] == "completed"
        assert captured["model_id"] == "seismic_surface_seg"
        assert captured["options"] == {
            "inline_count": 2,
            "amplitude_mode": "robust",
        }

        response = api.get_prediction_artifact(task_id, "overview")
        assert Path(response.path) == overview.resolve()
        with pytest.raises(HTTPException) as error:
            api.get_prediction_artifact(task_id, "not_registered")
        assert error.value.status_code == 404
    finally:
        with api._tasks_lock:
            api._tasks.pop(task_id, None)
