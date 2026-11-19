"""Replay one locked P81 development model, retaining its trained state.

This is a fixed-protocol training replay, not recovery of historical GPU weights.
The research runner discarded those weights. No search or promotion occurs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellseismic_sixtrack_pilot.manifest import atomic_json, sha256_file

LOCK_PATH = Path(__file__).resolve().parents[1] / "source-locks/facies-p81-recovery-v1.json"
TRACK = "_pipelines/02_task_datasets/facies"


def validate_request(root: Path, output: Path, task: str, fold: int, repeat: int) -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    if task not in lock["tasks"]:
        raise ValueError("Only independent F3/Penobscot development tasks are supported")
    if type(fold) is not int or fold not in lock["folds"]:
        raise ValueError("Development fold must be an integer in 0..4")
    if type(repeat) is not int or repeat not in range(len(lock["seeds"])):
        raise ValueError("Original confirmation repeat must be an integer in 0..2")
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Recovery output must stay outside the read-only research repository")
    if output.exists():
        raise ValueError("Use a new output directory; recovery never overwrites evidence")
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"P81 recovery source hash mismatch: {relative}")
    return lock


def recover(root: Path, output: Path, task: str, fold: int, repeat: int, device: str = "cpu") -> dict:
    lock = validate_request(root, output, task, fold, repeat)
    if device not in ("cpu", "cuda:0"):
        raise ValueError("Recovery supports CPU or one explicitly isolated CUDA device")
    if device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in tuple(str(i) for i in range(8)):
        raise ValueError("GPU replay requires one explicitly selected CUDA_VISIBLE_DEVICES index")
    for name, version in lock["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Recovery runtime version mismatch: {name}")
    # GPU availability must be checked by the caller before a GPU replay.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    root, output = root.resolve(), output.resolve()
    sys.path.insert(0, str(root / TRACK))
    p = importlib.import_module("facies_p81_champion_confirmation")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    # Match the outer P81 runner as well as its individual training functions.
    p.p28._request_deterministic_execution()
    if p.CONFIGS != lock["configs"] or list(p.SEEDS) != lock["seeds"]:
        raise ValueError("P81 configuration or seeds drifted")
    if p.p11.MAX_UPDATES != lock["baseline_updates"] or p.p13.CANDIDATE_UPDATES != lock["candidate_updates"]:
        raise ValueError("P81 training budget drifted")
    p.p11.verify_git_source(p.p11.SAM2_SOURCE_ROOT, lock["sam2_source_revision"])
    checkpoint = p.p11.verify_checkpoint("facies", p.p11.SAM2_CHECKPOINT)
    if sha256_file(checkpoint) != lock["sam2_checkpoint_sha256"]:
        raise ValueError("SAM2 checkpoint mismatch")
    p.p11._prepare_sam2_dependency_path()
    p.p11.insert_import_root(p.p11.SAM2_SOURCE_ROOT, "sam2")
    prepared = p.p11.stage3.prepare_fold(
        task_id=task, fold_id=fold,
        manifest_path=root / TRACK / "_outputs/p81_facies_champion_confirmation/input" / f"{task}_split_manifest.json",
        processed_root=root / "_data/processed/facies", budget=p.p11.stage3.Stage3Budget(),
    )
    seed = lock["seeds"][repeat]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def progress(stage: str, **details):
        value = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **details}
        atomic_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)

    progress("baseline_training", updates=40)
    baseline = p.p11._build_seeded_small_model(task, prepared.num_classes, device, seed=seed)
    base_result = p.p11._train_ce_model(baseline, prepared, device=device, seed=seed, label="platform-p81-recovery")
    # Validation metrics remain inside the unchanged research function. They
    # never select a model, seed, stopping step or configuration in this tool.
    trained_baseline = base_result.pop("model").cpu()
    del base_result
    np.savez_compressed(output / "query.npz", query_images=prepared.validation_images,
                        sample_ids=np.asarray(prepared.validation_sample_ids))
    original_predict = p.p13._predict_logits
    original_batches = p.p13._candidate_batches
    captured = []

    def capture_predict(model, images, *, device):
        result = original_predict(model, images, device=device)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(state, output / "model_state.pt")
        captured.append(len(state))
        progress("trained_state_saved", tensors=len(state))
        return result

    def report_batches(*args, **kwargs):
        for index, batch in enumerate(original_batches(*args, **kwargs)):
            if index % 20 == 0:
                progress("candidate_training", update=index, total=160)
            yield batch

    p.p13._predict_logits = capture_predict
    p.p13._candidate_batches = report_batches
    try:
        result = p._train_cross_attention(prepared=prepared, trained_baseline=trained_baseline,
                                         task=task, device=device, seed=seed, weight_mode="pretrained")
    finally:
        p.p13._predict_logits = original_predict
        p.p13._candidate_batches = original_batches
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one final trained-state capture")
    del trained_baseline, baseline
    progress("checkpoint_reload")
    fresh_small = p.p11._build_seeded_small_model(task, prepared.num_classes, device, seed=seed)
    fresh_encoder = p.p13.build_sam2_encoder(task, prepared.num_classes, device, weight_mode="pretrained", seed=seed)
    fresh = p.p13.CrossAttentionSegmentationModel(fresh_small, fresh_encoder).to(device)
    fresh.load_state_dict(torch.load(output / "model_state.pt", map_location="cpu", weights_only=True), strict=True)
    reloaded_logits, _ = original_predict(fresh, prepared.validation_images, device=device)
    logits = result.pop("logits")
    if not np.array_equal(logits, reloaded_logits):
        raise RuntimeError("Reloaded trained model does not reproduce recovery logits exactly")
    np.savez_compressed(output / "reference.npz", logits=logits,
                        prediction=logits.argmax(axis=1).astype(np.uint8),
                        sample_ids=np.asarray(prepared.validation_sample_ids))
    # Historical GPU outputs are comparison evidence, never worker inputs.
    historical_path = root / TRACK / "_outputs/p81_facies_champion_confirmation/predictions" / task / lock["arm"] / f"fold_{fold}" / f"repeat_{repeat}.npz"
    with np.load(historical_path, allow_pickle=False) as historical:
        if not np.array_equal(historical["sample_ids"], np.asarray(prepared.validation_sample_ids)):
            raise RuntimeError("Recovered query roster differs from historical P81 confirmation")
        recovered_probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy().astype(np.float16)
        gpu_changes = int(np.count_nonzero(historical["prediction"] != recovered_probabilities.argmax(axis=1)))
        probability_difference = float(np.max(np.abs(historical["probabilities"].astype(np.float32) - recovered_probabilities.astype(np.float32))))
        pixels = int(historical["prediction"].size)
    # Recheck sources after training to detect another window changing them.
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Research source changed during recovery: {relative}")
    receipt = {"schema_version": "well-seismic.p81-trained-state-recovery.v1", "status": "completed",
        "task_id": task, "fold_id": fold, "repeat_id": repeat, "seed": seed, "device": device,
        "arm": lock["arm"], "label_version": lock["label_versions"][task],
        "num_classes": prepared.num_classes, "query_rows": len(prepared.validation_sample_ids),
        "fold_split_hash": prepared.fold_split_hash, "preprocessor": asdict(prepared.preprocessor),
        "training": result["training"], "wiring_gate": result["wiring_gate"],
        "payload_sha256": {n: sha256_file(output / n) for n in ("model_state.pt", "query.npz", "reference.npz")},
        "source_lock_sha256": sha256_file(LOCK_PATH), "runtime_versions": lock["runtime_versions"],
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "historical_prediction_sha256": sha256_file(historical_path),
        "historical_gpu_prediction_pixel_changes": gpu_changes, "historical_prediction_pixels": pixels,
        "historical_float16_probability_max_abs_difference": probability_difference,
        "historical_prediction_comparison_uses_same_float16_softmax": True,
        "checkpoint_reload_logits_exact": True, "original_gpu_weights_recovered": False,
        "query_targets_exported": False, "frozen_test_accessed": False, "research_sources_modified": False,
        "model_selection_performed": False, "research_default_changed": False, "registered": False,
        "deployed": False, "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "recovery.json", receipt)
    progress("completed", historical_gpu_pixel_changes=gpu_changes)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--task", choices=("facies_f3", "facies_penobscot"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    recover(args.competition_root, args.output_directory, args.task, args.fold, args.repeat, args.device)
