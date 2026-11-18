"""Standalone inference for a complete recovered P81 model state.

No research module, labels, optimizer or pretrained-weight download is used.
Inputs must already carry the original fold-train normalization.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..manifest import sha256_file
from ..facies_contract import SOURCE_LOCK_PATH, source_lock


CLASSES = {"facies_f3": 10, "facies_penobscot": 8}
LOCK_PATH = SOURCE_LOCK_PATH


def load_queries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"query_images", "sample_ids"}:
            raise ValueError("Facies queries must contain only images and sample IDs")
        images, ids = archive["query_images"], archive["sample_ids"]
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1:] != (1, 128, 128):
        raise ValueError("Facies queries require float32 [N,1,128,128] normalized slices")
    if not 1 <= len(images) <= 16 or not np.isfinite(images).all():
        raise ValueError("Facies inference accepts 1..16 finite development slices")
    if ids.dtype.kind != "U" or ids.shape != (len(images),) or not np.all(np.char.str_len(ids) > 0):
        raise ValueError("Every facies image requires one nonempty Unicode sample ID")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Facies query sample IDs must be unique")
    return images, ids


def predict(input_path: Path, state_path: Path, state_sha256: str,
            task: str, sam2_source: Path, device: str = "cpu", progress=None) -> dict[str, np.ndarray]:
    if task not in CLASSES:
        raise ValueError("F3 and Penobscot require separate label spaces")
    images, ids = load_queries(input_path)
    if len(state_sha256) != 64 or sha256_file(state_path) != state_sha256:
        raise ValueError("Complete trained state SHA-256 mismatch")
    lock = source_lock()
    for name in ("torch", "numpy", "segmentation-models-pytorch", "torchvision", "timm"):
        if importlib.metadata.version(name) != lock["runtime_versions"][name]:
            raise ValueError(f"Facies inference runtime mismatch: {name}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sam2_source, text=True).strip()
    if revision != lock["sam2_source_revision"]:
        raise ValueError("SAM2 source revision mismatch")
    for arguments in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(arguments, cwd=sam2_source).returncode:
            raise ValueError("SAM2 source has tracked modifications")
    if device not in ("cpu", "cuda:0"):
        raise ValueError("Facies inference supports CPU or one isolated CUDA device")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "") not in tuple(str(i) for i in range(8)):
        raise ValueError("GPU inference requires one explicit CUDA_VISIBLE_DEVICES index")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import segmentation_models_pytorch as smp
    from .facies_architecture import CrossAttentionSegmentationModel, _sam2_normalize_native

    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(sam2_source.resolve()))
    from sam2.build_sam import build_sam2

    class NativeEncoder(torch.nn.Module):
        def __init__(self, image_encoder):
            super().__init__()
            self.image_encoder = image_encoder

        def forward(self, inputs):
            encoded = self.image_encoder(_sam2_normalize_native(inputs))
            maps = list(encoded["backbone_fpn"])[-3:]
            if len(maps) != 3 or any(feature.ndim != 4 or feature.shape[1] != 256 for feature in maps):
                raise ValueError("Unexpected native SAM2 feature pyramid")
            return maps

    constructor = smp.FPN if task == "facies_f3" else smp.DeepLabV3Plus
    small = constructor(encoder_name="resnet18", encoder_weights=None, in_channels=1,
                        classes=CLASSES[task], activation=None, aux_params=None)
    # Every image encoder, CNN, fusion and head tensor is loaded below. The
    # base SAM2 checkpoint is unnecessary at inference and is never opened.
    backbone = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt_path=None,
                          device=device, mode="eval", apply_postprocessing=False)
    model = CrossAttentionSegmentationModel(small, NativeEncoder(backbone.image_encoder)).to(device)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()
    ):
        raise ValueError("Complete trained state contains invalid tensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    if progress:
        progress("trained_state_loaded", {"query_rows": len(images)})
    parts = []
    with torch.no_grad():
        for start in range(0, len(images), 2):
            logits, _ = model(torch.from_numpy(np.ascontiguousarray(images[start:start + 2])).to(device))
            parts.append(logits.cpu().numpy())
            if start == 0 and progress:
                progress("pretrained_forward_completed", {"completed_query_rows": len(logits), "query_rows": len(images)})
    logits = np.concatenate(parts, axis=0)
    if logits.shape != (len(images), CLASSES[task], 128, 128) or not np.isfinite(logits).all():
        raise RuntimeError("Facies model returned invalid logits")
    probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    return {"logits": logits, "probabilities": probabilities,
            "prediction": logits.argmax(axis=1).astype(np.uint8), "sample_ids": ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--task", choices=tuple(CLASSES), required=True)
    parser.add_argument("--sam2-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        raise ValueError("Facies prediction output must have the .npz suffix")
    if args.output.exists():
        raise ValueError("Refusing to overwrite facies predictions")
    output = predict(args.input, args.state, args.state_sha256, args.task, args.sam2_source, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
