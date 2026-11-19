"""Frozen P76 development probability ensemble, local pretrained TabICLv2 on CPU.

The research runtime default remains separate until refit. No query targets,
guard/test data, downstream optimizer steps, or downloads are accepted here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from ..manifest import atomic_json, sha256_file

CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
ARM = "tabiclv2_compact70_e2e8_probability_fusion"
INPUT_KEYS = frozenset({"train_well", "train_seismic", "train_labels", "query_well", "query_seismic"})


def compact_features(well, seismic):
    well, seismic = np.asarray(well, dtype=np.float32), np.asarray(seismic, dtype=np.float32)
    if well.ndim != 3 or well.shape[1:] != (26, 33) or seismic.shape != (len(well), 3, 3, 33):
        raise ValueError("P76 requires aligned well [N,26,33] and seismic [N,3,3,33] windows")
    channels = np.concatenate((well, seismic.reshape(len(well), 9, 33)), axis=1)
    if not np.isfinite(channels).all():
        raise ValueError("P76 windows must be finite")
    return np.concatenate((channels[:, :, 16], channels.mean(axis=-1)), axis=1).astype(np.float32)


def fixed_logits(probabilities, classes):
    values, classes = np.asarray(probabilities, dtype=np.float64), np.asarray(classes)
    if classes.ndim != 1 or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("P76 class IDs must be integer codes")
    if values.ndim != 2 or values.shape[1] != len(classes) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("P76 probabilities/classes are invalid")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("P76 probability rows must sum to one")
    if np.any(classes < 0) or np.any(classes >= 9) or len(np.unique(classes)) != len(classes):
        raise ValueError("P76 requires distinct class IDs within fixed schema 0..8")
    full = np.full((len(values), 9), 1e-12, dtype=np.float64)
    full[:, classes] = np.clip(values, 1e-12, 1)
    full /= full.sum(axis=1, keepdims=True)
    return np.log(full).astype(np.float32)


def mix_logits(e2, e8):
    components = []
    for logits in (e2, e8):
        values = np.asarray(logits, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
            raise ValueError("P76 components require finite fixed nine-class logits")
        values -= values.max(axis=1, keepdims=True)
        probability = np.exp(values)
        components.append(probability / probability.sum(axis=1, keepdims=True))
    if components[0].shape != components[1].shape:
        raise ValueError("P76 component rows are misaligned")
    probability = .25 * components[0] + .75 * components[1]
    probability /= probability.sum(axis=1, keepdims=True)
    return np.log(np.clip(probability, 1e-12, 1))


def component_seed(component, fold, repeat):
    material = f"2693|P76|{ARM}|{component}|{fold}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1) or 2693


def predict(inputs, checkpoint: Path, *, fold: int, repeat: int, progress=None):
    if set(inputs) != INPUT_KEYS:
        raise ValueError("P76 accepts exactly support/query windows and training labels; query labels forbidden")
    if type(fold) is not int or fold not in range(3) or type(repeat) is not int or repeat not in (3, 4, 5):
        raise ValueError("P76 requires original development fold 0..2 and confirmation repeat 3..5")
    train = compact_features(inputs["train_well"], inputs["train_seismic"])
    query = compact_features(inputs["query_well"], inputs["query_seismic"])
    labels = np.asarray(inputs["train_labels"])
    if not 1 <= len(train) <= 360 or not 1 <= len(query) <= 360:
        raise ValueError("P76 worker is limited to the original development support/query budget")
    if labels.shape != (len(train),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= 9):
        raise ValueError("P76 training labels must be fixed-schema integer codes 0..8")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("P76 checkpoint hash mismatch")
    if importlib.metadata.version("tabicl") != "2.1.1":
        raise ValueError("P76 requires locked TabICL 2.1.1")
    from tabicl import TabICLClassifier
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    logits, seeds, classes = {}, {}, {}
    with threadpool_limits(limits=1):
        for component, views in (("e2", 2), ("e8", 8)):
            seeds[component] = component_seed(component, fold, repeat)
            estimator = TabICLClassifier(n_estimators=views, batch_size=1, model_path=str(checkpoint),
                allow_auto_download=False, checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                device="cpu", offload_mode="auto", random_state=seeds[component], verbose=False)
            estimator.fit(train, labels)
            logits[component] = fixed_logits(estimator.predict_proba(query), estimator.classes_).astype(np.float64)
            classes[component] = estimator.classes_.tolist()
            print(f"P76 {component} completed", flush=True)
            if progress: progress(component)
            del estimator
    fused = mix_logits(logits["e2"], logits["e8"])
    return {"logits": fused, "logits_e2": logits["e2"], "logits_e8": logits["e8"],
            "prediction": fused.argmax(axis=1).astype(np.uint8)}, {
        "query_rows": len(query), "support_rows": len(train), "feature_columns": 70,
        "component_seeds": seeds, "component_classes": classes, "checkpoint_sha256": CHECKPOINT_SHA256,
        "tabicl_version": "2.1.1", "downstream_optimizer_steps": 0, "query_targets_consumed": False,
        "auto_download": False, "execution": "cpu_pretrained_development_ensemble",
        "research_runtime_default_changed": False, "platform_registered": False,
        "frozen_test_accessed": False, "new_area_verified": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists(): raise ValueError("Use a new output directory")
    if sha256_file(args.input) != args.input_sha256: raise ValueError("P76 input hash mismatch")
    with np.load(args.input, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    predictions, receipt = predict(inputs, args.checkpoint, fold=args.fold, repeat=args.repeat)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    path = args.output_directory / "lithofacies_predictions.npz"
    np.savez_compressed(path, **predictions)
    receipt.update(status="completed", input_sha256=args.input_sha256, output_sha256=sha256_file(path))
    atomic_json(args.output_directory / "receipt.json", receipt)


if __name__ == "__main__": main()
