"""Local-only frozen GFM encoding for aligned reconstruction seismic volumes.

CPU execution preserves P14's slice/token/channel selection. Historical CUDA
bfloat16 features are a separate numerical reference, not an exact CPU oracle.
No encoder training, random initialization, download or query labels are used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from ..manifest import atomic_json, sha256_file

SNAPSHOT = "d4a33965730a506cfdb4c85fa2a0a344c53216a2"
ASSETS = {
    "source": "ce6f97b5be889231107716d71175fe4482a678f8eb4e608c274265ca57612904",
    "config": "3abf4d69533c7aca3075deef079b28098eb9d3ba3ba10ed45e0733aebbb3fefc",
    "model_card": "3ca7d92f45bac244c7fa9fb8460b703636e853acbbaa35831ae4dc9bdcdbcf5f",
    "weights": "c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446",
}
SEEDS = (1867973658, 2137841944, 3902865753)


def load_encoder(source_root: Path, snapshot: Path):
    if snapshot.name != SNAPSHOT:
        raise ValueError("GFM snapshot revision mismatch")
    paths = {"source": source_root / "GFM/ElasticViTMAE.py", "config": snapshot / "config.json",
             "model_card": snapshot / "README.md", "weights": snapshot / "model.safetensors"}
    for role, path in paths.items():
        if sha256_file(path) != ASSETS[role]:
            raise ValueError(f"GFM {role} hash mismatch")
    spec = importlib.util.spec_from_file_location("wellseismic_locked_gfm_vendor", paths["source"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.ElasticViTMAE.from_pretrained(snapshot, local_files_only=True)
    network.eval().cpu()
    for parameter in network.parameters():
        parameter.requires_grad = False
    return network, {"snapshot_revision": SNAPSHOT, "asset_sha256": ASSETS,
                     "pretrained_weights_loaded": True, "encoder_frozen": True,
                     "auto_download": False, "execution": "cpu_float32",
                     "parameter_count": sum(parameter.numel() for parameter in network.parameters())}


def normalize_slice(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    values, active = np.asarray(values, dtype=np.float32), np.asarray(active, dtype=bool)
    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError("GFM slice/mask must be aligned two-dimensional arrays")
    selected = values[active]
    if selected.size < 2 or not np.isfinite(selected).all():
        raise ValueError("GFM slice lacks finite active samples")
    normalized = np.zeros_like(values)
    normalized[active] = (selected - float(np.mean(selected))) / max(float(np.std(selected)), 1e-6)
    return normalized


def selected_channels() -> np.ndarray:
    return np.stack([np.stack([np.sort(rng.choice(1200, 16, replace=False)) for _ in range(6)])
                     for rng in (np.random.default_rng(seed) for seed in SEEDS)])


def extract_features(seismic: np.ndarray, active: np.ndarray, requested: np.ndarray, network) -> np.ndarray:
    seismic, active, requested = np.asarray(seismic), np.asarray(active), np.asarray(requested)
    if active.ndim != 3 or active.dtype != np.bool_ or seismic.shape != (3, *active.shape):
        raise ValueError("GFM requires three aligned seismic volumes and a Boolean active mask")
    if requested.ndim != 2 or requested.shape[1] != 3 or not len(requested) or not np.issubdtype(requested.dtype, np.integer):
        raise ValueError("GFM requested cells must be integer [N,3] KJI indices")
    if np.any(requested < 0) or np.any(requested >= np.asarray(active.shape)):
        raise ValueError("GFM requested cell is outside the volume")
    if not active[tuple(requested.T)].all():
        raise ValueError("GFM requested cell is inactive")
    if any(parameter.requires_grad for parameter in network.parameters()):
        raise ValueError("GFM encoder must be frozen")
    selected = selected_channels()
    token_j = np.clip(np.rint((requested[:, 1].astype(np.float64) + .5) * 160 / active.shape[1] - .5), 0, 159).astype(np.int64)
    features = np.empty((3, len(requested), 96), dtype=np.float32)
    torch.set_num_threads(1)
    network.eval()
    for i in np.unique(requested[:, 2]):
        rows = np.flatnonzero(requested[:, 2] == i)
        for channel in range(3):
            normalized = normalize_slice(seismic[channel, :, :, i], active[:, :, i])
            images = functional.interpolate(torch.as_tensor(normalized[None, None], dtype=torch.float32),
                                            size=(400, 160), mode="bilinear", align_corners=False)
            order = torch.arange(160, dtype=torch.float32)[None]
            with torch.inference_mode():
                latent, mask, restored = network.forward_encoder(images, order, 160)
            if tuple(latent.shape) != (1, 161, 1200) or not torch.isfinite(latent).all():
                raise ValueError("GFM encoder produced invalid tokens")
            if torch.any(mask != 0) or not torch.equal(restored, torch.arange(160, dtype=restored.dtype)[None]):
                raise ValueError("GFM encoder masked or reordered traces")
            tokens = latent.float().cpu().numpy()[0]
            trace = tokens[1 + token_j[rows]]
            cls = np.broadcast_to(tokens[0], (len(rows), 1200))
            for seed in range(3):
                features[seed, rows, channel * 32:channel * 32 + 16] = trace[:, selected[seed, channel * 2]]
                features[seed, rows, channel * 32 + 16:channel * 32 + 32] = cls[:, selected[seed, channel * 2 + 1]]
        print(f"GFM completed fixed-I slice {int(i)}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("GFM produced non-finite features")
    return features.transpose(1, 0, 2).reshape(len(requested), 288).astype(np.float64)


def run(args) -> dict:
    if args.output_directory.exists():
        raise ValueError("Use a new output directory")
    if sha256_file(args.volume) != args.volume_sha256 or sha256_file(args.indices) != args.indices_sha256:
        raise ValueError("GFM volume/indices hash mismatch")
    with np.load(args.volume, allow_pickle=False) as archive:
        if set(archive.files) != {"seismic", "active", "normalized_coordinates"}:
            raise ValueError("GFM volume requires exactly target-free seismic/active/normalized_coordinates")
        seismic, active = archive["seismic"], archive["active"]
    requested = np.load(args.indices, allow_pickle=False)
    torch.set_num_threads(1)
    network, assets = load_encoder(args.source_root, args.snapshot)
    features = extract_features(seismic, active, requested, network)
    args.output_directory.mkdir(parents=True)
    output = args.output_directory / "gfm_features.npz"
    np.savez_compressed(output, indices_kji=requested, features=features)
    receipt = {"status": "completed", "assets": assets, "feature_shape": list(features.shape),
               "feature_sha256": sha256_file(output), "volume_sha256": args.volume_sha256,
               "indices_sha256": args.indices_sha256, "query_targets_consumed": False,
               "historical_gpu_bfloat16_parity_verified": False, "new_area_verified": False,
               "platform_registered": False, "deployed": False}
    atomic_json(args.output_directory / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("volume", "indices", "source-root", "snapshot", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--volume-sha256", required=True)
    parser.add_argument("--indices-sha256", required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
