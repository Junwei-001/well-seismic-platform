#!/usr/bin/env python3
"""Train or fine-tune FaultSeg3D with PyTorch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.checkpoint import load_model
from src.data import FaultVolumeDataset, discover_ids, parse_shape
from src.losses import balanced_bce_loss, bce_dice_loss, conservative_bce_tversky_loss
from src.metrics import SegmentationMetrics
from src.model import FaultSegNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seis", type=Path, default=Path("data/train/seis"))
    parser.add_argument("--train-fault", type=Path, default=Path("data/train/fault"))
    parser.add_argument("--val-seis", type=Path, default=Path("data/validation/seis"))
    parser.add_argument("--val-fault", type=Path, default=Path("data/validation/fault"))
    parser.add_argument("--shape", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--loss",
        choices=("bce-dice", "bce-tversky", "bce", "balanced-bce"),
        default="bce-dice",
    )
    parser.add_argument("--bottleneck", type=int, choices=(128, 512), default=128)
    parser.add_argument("--init", type=Path, help="initialize from a model checkpoint")
    parser.add_argument("--resume", type=Path, help="resume a training checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="freeze conv1-conv3 for the first N epochs of conservative fine-tuning",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after N epochs without validation-IoU improvement (0 disables)",
    )
    parser.add_argument("--max-train-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-samples", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def select_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "bce-dice":
        return bce_dice_loss
    if name == "bce-tversky":
        return conservative_bce_tversky_loss
    if name == "balanced-bce":
        return balanced_bce_loss
    return nn.functional.binary_cross_entropy_with_logits


def set_encoder_trainable(model: FaultSegNet, trainable: bool) -> None:
    for layer in (
        model.conv1a,
        model.conv1b,
        model.conv2a,
        model.conv2b,
        model.conv3a,
        model.conv3b,
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = trainable


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    metrics = SegmentationMetrics()
    for seismic, fault in loader:
        seismic = seismic.to(device, non_blocking=True)
        fault = fault.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(seismic)
                loss = criterion(logits, fault)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.detach() * seismic.shape[0]
        metrics.update(logits, fault)
    result = {"loss": float(loss_sum / len(loader.dataset))}
    result.update(metrics.compute())
    return result


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.resume.parent if args.resume else Path("runs/faultseg")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = choose_device(args.device)

    train_ids = discover_ids(args.train_seis, args.train_fault)
    val_ids = discover_ids(args.val_seis, args.val_fault)
    if args.max_train_samples is not None:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_ids = val_ids[: args.max_val_samples]
    if not train_ids or not val_ids:
        raise FileNotFoundError("paired training and validation .dat files are required")

    train_data = FaultVolumeDataset(
        args.train_seis, args.train_fault, train_ids, args.shape, augment=True
    )
    val_data = FaultVolumeDataset(args.val_seis, args.val_fault, val_ids, args.shape)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    start_epoch = 0
    best_iou = -1.0
    saved: dict = {}
    if args.resume:
        model, saved = load_model(args.resume, device)
        start_epoch = int(saved.get("epoch", -1)) + 1
        best_iou = float(saved.get("best_iou", -1.0))
    elif args.init:
        model, _ = load_model(args.init, device)
    else:
        model = FaultSegNet(args.bottleneck).to(device)

    if args.freeze_encoder_epochs < 0 or args.early_stop_patience < 0:
        raise ValueError("freeze and early-stop epoch counts must be non-negative")
    encoder_frozen = start_epoch < args.freeze_encoder_epochs
    set_encoder_trainable(model, not encoder_frozen)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    if args.resume:
        if "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])
        if "scheduler_state" in saved:
            scheduler.load_state_dict(saved["scheduler_state"])

    criterion = select_loss(args.loss)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if args.resume and "scaler_state" in saved:
        scaler.load_state_dict(saved["scaler_state"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    print(
        f"device={device} train={len(train_data)} validation={len(val_data)} "
        f"parameters={sum(p.numel() for p in model.parameters()):,} amp={amp_enabled}"
    )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        if encoder_frozen and epoch >= args.freeze_encoder_epochs:
            set_encoder_trainable(model, True)
            encoder_frozen = False
            print(f"unfroze encoder at epoch {epoch + 1}", flush=True)
        train_metrics = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, amp_enabled
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                model, val_loader, device, criterion, None, scaler, amp_enabled
            )
        scheduler.step(val_metrics["iou"])
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        print(json.dumps(record), flush=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        improved = val_metrics["iou"] > best_iou
        best_iou = max(best_iou, val_metrics["iou"])
        checkpoint = {
            "format": "faultseg3d-pytorch-v2",
            "epoch": epoch,
            "best_iou": best_iou,
            "bottleneck_channels": model.bottleneck_channels,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "args": serializable_args(args),
            "metrics": record,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if improved:
            torch.save(
                {
                    "format": checkpoint["format"],
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "bottleneck_channels": model.bottleneck_channels,
                    "model_state": model.state_dict(),
                    "args": checkpoint["args"],
                    "metrics": record,
                },
                args.output_dir / "best.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"early stopping after {epochs_without_improvement} epochs without improvement",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
