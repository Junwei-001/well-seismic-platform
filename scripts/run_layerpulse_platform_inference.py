"""Run one LayerPulse unified preview inference outside the FastAPI process.

The platform process owns SourceSnapshot validation and SEG-Y geometry.  This
subprocess receives one already materialised, deterministic T-I-X patch, loads
the single delivery checkpoint, executes exactly one forward, and writes all
eleven task products.  Classification products retain complete logits and are
decoded only by ``argmax(dim=1)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUEST_SCHEMA = "well-seismic.layerpulse-inference-request.v1"
RESULT_SCHEMA = "well-seismic.layerpulse-child-result.v1"
MODEL_ID = "layerpulse_geochronograph_f3x200cf"
PORTABLE_CHECKPOINT_SCHEMA = "wellfuse.layerpulse-portable-model-only.v1"
PORTABLE_CHECKPOINT_SUFFIX = "_portable_v4.pt"
FORBIDDEN_KEY_FRAGMENTS = (
    "checkshot",
    "time_depth",
    "timedepth",
    "td_table",
    "td_teacher",
    "target_twt",
    "target_time",
    "velocity_model",
    "vsp",
)
CLASSIFICATION_OUTPUTS = (
    "facies_logits",
    "fault_logits",
    "unconformity_logits",
    "channel_logits",
    "karst_logits",
    "connectivity_logits",
)
REGRESSION_OUTPUTS = (
    "rgt",
    "impedance",
    "porosity",
    "well_match",
    "uncertainty",
)
TASK_LABELS = {
    "facies_logits": "Facies (background + 6 classes)",
    "fault_logits": "Fault",
    "unconformity_logits": "Unconformity / erosion barrier",
    "channel_logits": "Channel (background + 4 classes)",
    "karst_logits": "Karst",
    "connectivity_logits": "Structural connectivity",
    "rgt": "Relative geological time",
    "impedance": "Impedance",
    "porosity": "Porosity",
    "well_match": "Well-seismic match",
    "uncertainty": "Uncertainty",
}
CLASS_NAMES = {
    "facies_logits": (
        "background",
        "upper_ns",
        "middle_ns",
        "lower_ns",
        "rijnland_chalk",
        "scruff",
        "zechstein",
    ),
    "fault_logits": ("background", "fault"),
    "unconformity_logits": ("background", "unconformity"),
    "channel_logits": (
        "background",
        "channel_1",
        "channel_2",
        "channel_3",
        "channel_4",
    ),
    "karst_logits": ("background", "karst"),
    "connectivity_logits": ("background", "connected"),
}
PREVIEW_SCOPE_BY_CROP_SELECTION = {
    "fusion_ready_well_trajectory_anchor": "well_anchored_preview_patch",
    "explicit_geometry_crop": "explicit_preview_patch",
    "fixed_geometry_center": "fixed_geometry_preview_patch",
    # Compatibility with requests persisted before crop_selection was explicit.
    "floor_center_with_lower_index_tie_break_v1": "fixed_geometry_preview_patch",
}
PREVIEW_LABEL_BY_SCOPE = {
    "well_anchored_preview_patch": "井轨迹锚定预览子体",
    "explicit_preview_patch": "用户指定预览子体",
    "fixed_geometry_preview_patch": "固定中心预览子体",
}
PALETTE = np.asarray(
    [
        (0, 0, 0),
        (239, 68, 68),
        (245, 158, 11),
        (250, 204, 21),
        (34, 197, 94),
        (6, 182, 212),
        (59, 130, 246),
        (139, 92, 246),
    ],
    dtype=np.uint8,
)


class LayerPulsePlatformRunnerError(RuntimeError):
    """Raised when the sealed platform inference contract is violated."""


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayerPulsePlatformRunnerError(f"{name} must be a JSON object")
    return dict(value)


def _path(value: object, *, name: str, must_exist: bool = True) -> Path:
    text = str(value or "").strip()
    if not text:
        raise LayerPulsePlatformRunnerError(f"{name} is required")
    path = Path(text).expanduser().resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalised_key(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _forbidden_request_keys(value: object) -> list[str]:
    found: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalised = _normalised_key(key)
                if any(fragment in normalised for fragment in FORBIDDEN_KEY_FRAGMENTS):
                    found.add(str(key))
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def _shape3(value: object, *, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise LayerPulsePlatformRunnerError(f"{name} must contain three integers")
    shape = tuple(int(item) for item in value)
    if any(item < 0 for item in shape):
        raise LayerPulsePlatformRunnerError(f"{name} cannot contain negatives")
    return shape  # type: ignore[return-value]


def _resolve_output_window(
    input_contract: Mapping[str, Any],
    *,
    model_input_shape_tix: tuple[int, int, int],
) -> tuple[dict[str, Any], tuple[slice, slice, slice]]:
    """Validate central-logit cropping without changing forward count.

    An absent contract preserves the historical identity window.  The enabled
    form must be a symmetric halo and may only crop tensors after the single
    shared-Backbone forward, before classification argmax.
    """

    raw = input_contract.get("output_window")
    if raw is None:
        slices = tuple(slice(0, size) for size in model_input_shape_tix)
        return (
            {
                "enabled": False,
                "model_input_shape_tix": list(model_input_shape_tix),
                "output_offset_tix": [0, 0, 0],
                "output_shape_tix": list(model_input_shape_tix),
                "halo_tix": [0, 0, 0],
                "boundary_mode": "none",
                "output_rule": "identity_complete_logits_then_direct_argmax",
                "single_checkpoint_forward_calls": 1,
            },
            slices,  # type: ignore[arg-type]
        )
    window = _mapping(raw, name="input.output_window")
    if window.get("schema_version") != (
        "well-seismic.layerpulse-context-halo-window.v1"
    ):
        raise LayerPulsePlatformRunnerError("unsupported output-window schema")
    if window.get("enabled") is not True:
        raise LayerPulsePlatformRunnerError("declared output window is not enabled")
    declared_input_shape = _shape3(
        window.get("model_input_shape_tix"),
        name="input.output_window.model_input_shape_tix",
    )
    if declared_input_shape != model_input_shape_tix:
        raise LayerPulsePlatformRunnerError(
            "output-window model input shape differs from the transported patch"
        )
    output_offset = _shape3(
        window.get("output_offset_tix"),
        name="input.output_window.output_offset_tix",
    )
    output_shape = _shape3(
        window.get("output_shape_tix"),
        name="input.output_window.output_shape_tix",
    )
    halo = _shape3(
        window.get("halo_tix"), name="input.output_window.halo_tix"
    )
    if any(size <= 0 for size in output_shape) or not any(halo):
        raise LayerPulsePlatformRunnerError("output-window halo/shape is empty")
    if output_offset != halo or any(
        output + 2 * margin != available
        for output, margin, available in zip(
            output_shape, halo, model_input_shape_tix, strict=True
        )
    ):
        raise LayerPulsePlatformRunnerError(
            "output window is not the declared symmetric central halo crop"
        )
    if window.get("boundary_mode") != "constant_zero_with_explicit_valid_mask":
        raise LayerPulsePlatformRunnerError("unsupported context-halo boundary mode")
    if window.get("output_rule") != "central_complete_logits_then_direct_argmax":
        raise LayerPulsePlatformRunnerError("unsupported context-halo output rule")
    if int(window.get("single_checkpoint_forward_calls") or 0) != 1:
        raise LayerPulsePlatformRunnerError("context halo must retain one forward call")
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(output_offset, output_shape, strict=True)
    )
    window.update(
        {
            "model_input_shape_tix": list(model_input_shape_tix),
            "output_offset_tix": list(output_offset),
            "output_shape_tix": list(output_shape),
            "halo_tix": list(halo),
        }
    )
    return window, slices  # type: ignore[return-value]


def _load_request(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    request = _mapping(document, name="request")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise LayerPulsePlatformRunnerError("unsupported LayerPulse request schema")
    if request.get("model_id") != MODEL_ID:
        raise LayerPulsePlatformRunnerError("LayerPulse request model_id differs")
    forbidden = _forbidden_request_keys(request)
    if forbidden:
        raise LayerPulsePlatformRunnerError(
            "LayerPulse inference forbids TD/checkshot/VSP inputs: "
            + ", ".join(forbidden)
        )
    return request


def _atomic_json(document: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(document), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _normalise_seismic(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    clean = np.where(valid, raw, 0.0).astype(np.float32, copy=False)
    selected = clean[valid]
    if selected.size == 0:
        raise LayerPulsePlatformRunnerError("input patch contains no finite valid samples")
    mean = float(selected.mean(dtype=np.float64))
    std = float(selected.std(dtype=np.float64))
    scale = max(std, 1.0e-6)
    clean = np.clip((clean - mean) / scale, -6.0, 6.0)
    clean[~valid] = 0.0
    if not np.isfinite(clean).all():
        raise LayerPulsePlatformRunnerError("normalised seismic contains non-finite values")
    return np.ascontiguousarray(clean, dtype=np.float32), {
        "mean": mean,
        "standard_deviation": std,
        "applied_scale": scale,
        "clip_min": -6.0,
        "clip_max": 6.0,
    }


def _load_well_bundle(
    path: Path,
    *,
    well_channels: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    actual_sha256 = _sha256_file(path)
    if actual_sha256 != str(expected_sha256 or "").strip().casefold():
        raise LayerPulsePlatformRunnerError("well bundle identity differs from request")
    required = {
        "wells",
        "well_curve_mask",
        "well_mask",
        "well_md",
        "well_md_m",
        "trajectory",
        "trajectory_metric_xyz",
        "trajectory_tvd_m",
        "trajectory_mask",
        "well_parent_index",
        "well_kickoff_md_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise LayerPulsePlatformRunnerError(
                "well bundle is missing tensors: " + ", ".join(missing)
            )
        arrays = {key: np.asarray(archive[key]) for key in required}
    if arrays["wells"].ndim != 3 or int(arrays["wells"].shape[1]) != well_channels:
        raise LayerPulsePlatformRunnerError(
            f"well bundle wells must be [W,{well_channels},S]"
        )
    station_shape = (int(arrays["wells"].shape[0]), int(arrays["wells"].shape[2]))
    if arrays["well_mask"].shape != station_shape:
        raise LayerPulsePlatformRunnerError("well_mask shape differs from wells")
    if arrays["trajectory"].shape != (*station_shape, 3):
        raise LayerPulsePlatformRunnerError("trajectory must be [W,S,3]")
    tensor_batch: dict[str, Any] = {}
    boolean_keys = {"well_curve_mask", "well_mask", "trajectory_mask"}
    integer_keys = {"well_parent_index"}
    for key, array in arrays.items():
        if not np.isfinite(array).all() and key not in boolean_keys:
            raise LayerPulsePlatformRunnerError(f"well tensor {key} is non-finite")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if key in boolean_keys:
            tensor = tensor.bool()
        elif key in integer_keys:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        tensor_batch[key] = tensor.unsqueeze(0)
    observed = int(np.count_nonzero(arrays["well_curve_mask"]))
    total = int(arrays["well_curve_mask"].size)
    receipt = {
        "mode": "snapshot_wells_md_trajectory_no_td",
        "bundle_sha256": actual_sha256,
        "well_count": station_shape[0],
        "station_count": station_shape[1],
        "station_capacity": station_shape[1],
        "valid_station_count": int(np.count_nonzero(arrays["well_mask"])),
        "well_channels": well_channels,
        "observed_curve_fraction": float(observed / max(total, 1)),
        "time_depth_table_consumed": False,
    }
    return tensor_batch, receipt


def _build_teacher_free_empty_batch(
    *, well_channels: int, spatial_shape: Sequence[int]
) -> dict[str, Any]:
    """Build the exact seismic-only inference shell without training modules."""

    import torch

    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value < 16 for value in shape):
        raise LayerPulsePlatformRunnerError(
            "LayerPulse spatial shape must contain three values >=16"
        )
    return {
        "seismic": torch.zeros((1, 1, *shape), dtype=torch.float32),
        "wells": torch.empty(
            (1, 0, int(well_channels), 1), dtype=torch.float32
        ),
        "well_curve_mask": torch.empty(
            (1, 0, int(well_channels), 1), dtype=torch.bool
        ),
        "well_mask": torch.empty((1, 0), dtype=torch.bool),
        "well_md": torch.empty((1, 0, 1), dtype=torch.float32),
        "well_md_m": torch.empty((1, 0, 1), dtype=torch.float32),
        "trajectory": torch.empty((1, 0, 1, 3), dtype=torch.float32),
        "trajectory_metric_xyz": torch.empty(
            (1, 0, 1, 3), dtype=torch.float32
        ),
        "trajectory_tvd_m": torch.empty((1, 0, 1), dtype=torch.float32),
        "trajectory_mask": torch.empty((1, 0, 1), dtype=torch.bool),
        "well_parent_index": torch.empty((1, 0), dtype=torch.long),
        "well_kickoff_md_m": torch.empty((1, 0), dtype=torch.float32),
    }


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def _finite_min_max(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


def _gray(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    selected = values[finite]
    if selected.size == 0:
        return np.full((*values.shape, 3), 32, dtype=np.uint8)
    low, high = np.percentile(selected, (2.0, 98.0))
    scale = max(float(high - low), 1.0e-6)
    unit = np.clip((values - low) / scale, 0.0, 1.0)
    byte = np.rint(unit * 255.0).astype(np.uint8)
    rgb = np.repeat(byte[..., None], 3, axis=-1)
    rgb[~finite] = (30, 35, 42)
    return rgb


def _heat(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    selected = values[finite]
    if selected.size == 0:
        return np.full((*values.shape, 3), 32, dtype=np.uint8)
    low, high = np.percentile(selected, (2.0, 98.0))
    unit = np.clip((values - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    anchors = np.asarray(
        [
            (15, 23, 42),
            (30, 64, 175),
            (6, 182, 212),
            (34, 197, 94),
            (250, 204, 21),
            (239, 68, 68),
        ],
        dtype=np.float32,
    )
    position = unit * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (position - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    rgb = np.rint(rgb).astype(np.uint8)
    rgb[~finite] = (30, 35, 42)
    return rgb


def _planes(volume: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    t, inline, xline = volume.shape
    return (
        (f"TWT {t // 2}", np.asarray(volume[t // 2, :, :])),
        (f"Inline {inline // 2}", np.asarray(volume[:, inline // 2, :])),
        (f"Xline {xline // 2}", np.asarray(volume[:, :, xline // 2])),
    )


def _resize_rgb(array: np.ndarray, size: tuple[int, int]) -> Any:
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(array), mode="RGB").resize(
        size, resample=Image.Resampling.BILINEAR
    )


def _classification_overlay(seismic: np.ndarray, labels: np.ndarray) -> np.ndarray:
    base = _gray(seismic).astype(np.float32)
    colors = PALETTE[np.asarray(labels, dtype=np.int64) % len(PALETTE)].astype(np.float32)
    foreground = np.asarray(labels) > 0
    mixed = base.copy()
    mixed[foreground] = 0.38 * base[foreground] + 0.62 * colors[foreground]
    return np.rint(mixed).astype(np.uint8)


def _render_task_preview(
    *,
    task_key: str,
    seismic: np.ndarray,
    values: np.ndarray,
    categorical: bool,
    destination: Path,
) -> None:
    from PIL import Image, ImageDraw

    width, panel_height = 300, 250
    header_height = 54
    canvas = Image.new("RGB", (width * 3, panel_height + header_height), (9, 17, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), TASK_LABELS[task_key], fill=(235, 241, 249))
    subtitle = (
        "complete logits -> direct argmax" if categorical else "continuous Backbone output"
    )
    draw.text((14, 29), subtitle, fill=(148, 163, 184))
    seismic_planes = _planes(seismic)
    value_planes = _planes(values)
    for index, ((label, seismic_plane), (_, value_plane)) in enumerate(
        zip(seismic_planes, value_planes, strict=True)
    ):
        rgb = (
            _classification_overlay(seismic_plane, value_plane)
            if categorical
            else _heat(value_plane)
        )
        image = _resize_rgb(rgb, (width, panel_height))
        canvas.paste(image, (index * width, header_height))
        draw.rectangle(
            (index * width + 7, header_height + 7, index * width + 105, header_height + 27),
            fill=(9, 17, 31),
        )
        draw.text((index * width + 12, header_height + 11), label, fill=(226, 232, 240))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _render_seismic_preview(seismic: np.ndarray, destination: Path) -> None:
    from PIL import Image, ImageDraw

    width, panel_height, header_height = 300, 250, 40
    canvas = Image.new("RGB", (width * 3, panel_height + header_height), (9, 17, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), "Registered seismic - fixed geometry preview", fill=(235, 241, 249))
    for index, (label, plane) in enumerate(_planes(seismic)):
        image = _resize_rgb(_gray(plane), (width, panel_height))
        canvas.paste(image, (index * width, header_height))
        draw.rectangle(
            (index * width + 7, header_height + 7, index * width + 105, header_height + 27),
            fill=(9, 17, 31),
        )
        draw.text((index * width + 12, header_height + 11), label, fill=(226, 232, 240))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _render_atlas(previews: Sequence[tuple[str, Path]], destination: Path) -> None:
    from PIL import Image, ImageDraw

    cell_width, cell_height = 330, 138
    columns = 3
    rows = math.ceil(len(previews) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (8, 15, 28))
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(previews):
        row, column = divmod(index, columns)
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (cell_width - 12, cell_height - 30), Image.Resampling.LANCZOS
            )
        x, y = column * cell_width + 6, row * cell_height + 25
        canvas.paste(image, (x, y))
        draw.text((x + 4, row * cell_height + 7), label, fill=(226, 232, 240))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _task_stem(output_key: str) -> str:
    return output_key.removesuffix("_logits")


def _tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash the exact portable model tensors without materialising a second copy."""

    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise LayerPulsePlatformRunnerError(
                f"portable checkpoint state {name!r} is not a tensor"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _load_portable_checkpoint_model_only(
    checkpoint: Path, *, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Load the inference-only checkpoint after verifying its sealed tensor digest."""

    import torch

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    document = _mapping(payload, name="portable checkpoint")
    state = _mapping(document.get("model"), name="portable checkpoint model")
    model_config = _mapping(
        document.get("model_config"), name="portable checkpoint model_config"
    )
    conversion = _mapping(
        document.get("conversion"), name="portable checkpoint conversion"
    )
    source = _mapping(
        document.get("source_checkpoint"), name="portable checkpoint source"
    )
    expected_state_hash = str(conversion.get("model_state_sha256") or "").casefold()
    if (
        document.get("schema_version") != PORTABLE_CHECKPOINT_SCHEMA
        or document.get("model_id") != MODEL_ID
        or int(document.get("parameter_count") or -1) != 174_697_519
        or int(document.get("state_tensor_count") or -1) != 626
        or len(state) != 626
        or int(document.get("f_final_channels") or -1) != 96
        or int(document.get("head_count") or -1) != 11
        or document.get("head_input_contract") != "shared_F_final_only"
        or document.get("one_forward_all_tasks") is not True
        or document.get("teacher_required_at_forward") is not False
        or document.get("td_table_required_at_forward") is not False
        or conversion.get("model_tensors_bitwise_preserved") is not True
        or conversion.get("optimizer_state_included") is not False
        or len(expected_state_hash) != 64
    ):
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse checkpoint identity differs"
        )
    actual_state_hash = _tensor_state_sha256(state)
    if actual_state_hash != expected_state_hash:
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse model tensor digest differs"
        )

    from torch import nn

    from layerpulse.modeling.heads import FinalFeatureTaskHead
    from layerpulse.modeling.model import build_model

    model = build_model(model_config)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse checkpoint strict load returned incompatible keys"
        )
    heads = getattr(model, "heads", None)
    if not isinstance(heads, nn.ModuleDict) or not heads:
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse model does not expose task heads"
        )
    invalid_heads = [
        name
        for name, head in heads.items()
        if not isinstance(head, FinalFeatureTaskHead)
    ]
    if invalid_heads:
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse heads are not F_final-only: "
            + ", ".join(invalid_heads)
        )
    f_final_channels = int(model.config.f_final_channels)
    for name, head in heads.items():
        first = head.projection[0]
        if not isinstance(first, nn.GroupNorm) or first.num_channels != f_final_channels:
            raise LayerPulsePlatformRunnerError(
                f"portable LayerPulse head {name!r} does not consume F_final"
            )
    contract = {
        "f_final_channels": f_final_channels,
        "head_count": len(heads),
        "head_input": "shared_F_final_only",
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        parameter_count != 174_697_519
        or int(contract["f_final_channels"]) != 96
        or int(contract["head_count"]) != 11
        or str(contract["head_input"]) != "shared_F_final_only"
    ):
        raise LayerPulsePlatformRunnerError(
            "portable LayerPulse architecture differs from the sealed contract"
        )
    model.eval().to(device)
    return model, {
        "path": str(checkpoint),
        "schema_version": PORTABLE_CHECKPOINT_SCHEMA,
        "family": document.get("family"),
        "strict_model_load": True,
        "optimizer_loaded": False,
        "optimizer_state_included": False,
        "single_checkpoint": True,
        "parameter_count": parameter_count,
        "f_final_channels": int(contract["f_final_channels"]),
        "head_count": int(contract["head_count"]),
        "head_input": str(contract["head_input"]),
        "classification_selection": "direct_argmax_dim1",
        "teacher_required_at_forward": False,
        "model_state_sha256": actual_state_hash,
        "source_checkpoint_filename": source.get("filename"),
        "source_checkpoint_sha256": source.get("sha256"),
    }


def _load_unified_checkpoint_model_only(
    checkpoint: Path, *, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Delegate all supported V1-V5 schemas to the strict model-only loader."""

    if checkpoint.name.endswith(PORTABLE_CHECKPOINT_SUFFIX):
        return _load_portable_checkpoint_model_only(checkpoint, device=device)

    from layerpulse.training.precision_multitask_train_fit_panel_v1 import (
        load_train_fit_checkpoint,
    )

    return load_train_fit_checkpoint(checkpoint, device=device)


def run(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    input_contract = _mapping(request.get("input"), name="input")
    inference = _mapping(request.get("inference") or {}, name="inference")
    crop_selection = str(
        input_contract.get("crop_selection")
        or input_contract.get("selection_policy")
        or "fixed_geometry_center"
    ).strip()
    try:
        preview_scope = PREVIEW_SCOPE_BY_CROP_SELECTION[crop_selection]
    except KeyError as exc:
        raise LayerPulsePlatformRunnerError(
            f"unsupported input crop_selection: {crop_selection}"
        ) from exc
    requested_scope = str(inference.get("scope") or preview_scope).strip()
    if requested_scope != preview_scope:
        raise LayerPulsePlatformRunnerError(
            "inference.scope differs from the input crop_selection contract"
        )
    checkpoint = _path(request.get("checkpoint"), name="checkpoint")
    expected_checkpoint = str(os.getenv("LAYERPULSE_CHECKPOINT") or "").strip()
    if expected_checkpoint and checkpoint != Path(expected_checkpoint).expanduser().resolve():
        raise LayerPulsePlatformRunnerError("checkpoint differs from sealed environment")
    patch_path = _path(
        input_contract.get("patch_npy") or input_contract.get("input_patch_npy"),
        name="input.patch_npy",
    )
    valid_path_value = input_contract.get("valid_mask_npy") or input_contract.get(
        "input_valid_mask_npy"
    )
    patch = np.asarray(np.load(patch_path, mmap_mode="r", allow_pickle=False))
    if patch.ndim != 3 or any(int(size) < 16 for size in patch.shape):
        raise LayerPulsePlatformRunnerError("input patch must be [T,I,X] with axes >=16")
    if valid_path_value:
        valid_path = _path(valid_path_value, name="input.valid_mask_npy")
        valid = np.asarray(np.load(valid_path, mmap_mode="r", allow_pickle=False)).astype(bool)
        if valid.shape != patch.shape:
            raise LayerPulsePlatformRunnerError("valid mask shape differs from input patch")
    else:
        valid = np.isfinite(patch)
    valid = np.asarray(valid & np.isfinite(patch), dtype=bool)
    seismic, normalisation = _normalise_seismic(patch, valid)
    output_window, output_slices = _resolve_output_window(
        input_contract,
        model_input_shape_tix=tuple(int(size) for size in seismic.shape),
    )
    output_seismic = np.ascontiguousarray(seismic[output_slices], dtype=np.float32)
    output_valid = np.ascontiguousarray(valid[output_slices], dtype=bool)
    output_shape = tuple(int(size) for size in output_seismic.shape)
    if output_shape != tuple(output_window["output_shape_tix"]):
        raise LayerPulsePlatformRunnerError("resolved output-window shape drifted")
    if not np.any(output_valid):
        raise LayerPulsePlatformRunnerError("central output ROI contains no valid samples")
    axes = tuple(str(axis).strip().upper() for axis in input_contract.get("axes", ()))
    if axes not in {("TWT", "INLINE", "XLINE"), ("T", "INLINE", "XLINE")}:
        raise LayerPulsePlatformRunnerError(
            "input axes must explicitly be TWT,INLINE,XLINE (or T,INLINE,XLINE)"
        )
    device_name = str(inference.get("device") or "cuda").strip().casefold()
    if not device_name.startswith("cuda"):
        raise LayerPulsePlatformRunnerError("formal LayerPulse platform runner requires CUDA")
    if not torch.cuda.is_available():
        raise LayerPulsePlatformRunnerError("CUDA is unavailable")
    device = torch.device(device_name)

    started = time.perf_counter()
    model, checkpoint_receipt = _load_unified_checkpoint_model_only(
        checkpoint, device=device
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    batch = _build_teacher_free_empty_batch(
        well_channels=int(model.config.well_channels),
        spatial_shape=tuple(int(size) for size in seismic.shape),
    )
    batch["seismic"] = torch.from_numpy(seismic).unsqueeze(0).unsqueeze(0)
    batch["seismic_mask"] = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)
    well_bundle_value = input_contract.get("well_bundle_npz")
    if well_bundle_value:
        expected_well_bundle_sha256 = str(
            input_contract.get("well_bundle_sha256") or ""
        ).strip().casefold()
        if len(expected_well_bundle_sha256) != 64:
            raise LayerPulsePlatformRunnerError(
                "input.well_bundle_sha256 is required for a well bundle"
            )
        well_batch, well_receipt = _load_well_bundle(
            _path(well_bundle_value, name="input.well_bundle_npz"),
            well_channels=int(model.config.well_channels),
            expected_sha256=expected_well_bundle_sha256,
        )
        batch.update(well_batch)
    else:
        well_receipt = {
            "mode": "seismic_only_no_td",
            "well_count": 0,
            "station_count": 0,
            "station_capacity": 0,
            "valid_station_count": 0,
            "well_channels": int(model.config.well_channels),
            "observed_curve_fraction": 0.0,
            "time_depth_table_consumed": False,
        }
    batch = _move_batch(batch, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        model_output = model(batch, return_aux=False)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    tasks = _mapping(model_output.get("tasks"), name="model tasks")
    predictions = _mapping(model_output.get("predictions"), name="model predictions")
    if set(tasks) != set(CLASSIFICATION_OUTPUTS) | set(REGRESSION_OUTPUTS):
        raise LayerPulsePlatformRunnerError("model did not return the exact eleven tasks")

    output_directory = _path(
        request.get("output_directory"), name="output_directory", must_exist=False
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    logits: dict[str, np.ndarray] = {}
    scalar_volumes: dict[str, np.ndarray] = {}
    task_catalog: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    preview_paths: list[tuple[str, Path]] = []

    # Facies is deliberately first so the model-neutral standard result viewer
    # uses the competition-facing seven-logit task as its primary 3-D layer.
    for output_key in CLASSIFICATION_OUTPUTS:
        full_tensor = tasks[output_key].detach().float().cpu()
        if (
            full_tensor.ndim != 5
            or full_tensor.shape[0] != 1
            or tuple(int(size) for size in full_tensor.shape[-3:]) != seismic.shape
        ):
            raise LayerPulsePlatformRunnerError(f"{output_key} shape is not [1,C,T,I,X]")
        tensor = full_tensor[(slice(None), slice(None), *output_slices)]
        array = tensor[0].numpy()
        if not np.isfinite(array).all():
            raise LayerPulsePlatformRunnerError(f"{output_key} contains non-finite logits")
        direct = tensor.argmax(dim=1)[0].numpy().astype(np.uint8)
        returned = predictions.get(output_key)
        returned_central = (
            returned.detach().cpu()[(slice(None), *output_slices)]
            if returned is not None
            else None
        )
        if returned_central is None or not torch.equal(
            returned_central, tensor.argmax(dim=1)
        ):
            raise LayerPulsePlatformRunnerError(
                f"{output_key} prediction is not direct argmax(dim=1)"
            )
        expected_classes = CLASS_NAMES[output_key]
        if int(array.shape[0]) != len(expected_classes):
            raise LayerPulsePlatformRunnerError(f"{output_key} class count differs")
        stem = _task_stem(output_key)
        argmax_path = output_directory / f"{stem}_argmax.npy"
        np.save(argmax_path, direct, allow_pickle=False)
        outputs[f"{stem}_argmax_npy"] = str(argmax_path)
        # Preserve the exact float32 values used by direct argmax.  Quantising
        # logits before export can flip near-tied classes and would make the
        # persisted complete-logit contract disagree with the class volume.
        logits[output_key] = np.ascontiguousarray(array, dtype=np.float32)
        preview_path = output_directory / f"{stem}_preview.png"
        _render_task_preview(
            task_key=output_key,
            seismic=output_seismic,
            values=direct,
            categorical=True,
            destination=preview_path,
        )
        outputs[f"{stem}_preview_png"] = str(preview_path)
        preview_paths.append((stem, preview_path))
        task_catalog.append(
            {
                "name": stem,
                "output_key": output_key,
                "kind": "classification",
                "channels": int(array.shape[0]),
                "background_index": 0,
                "class_names": list(expected_classes),
                "selection": "direct_argmax_dim1",
                "logits_shape_ctix": list(array.shape),
                "output_shape_tix": list(direct.shape),
                "class_min": int(direct.min()),
                "class_max": int(direct.max()),
                "finite": True,
                "artifact_key": f"{stem}_argmax_npy",
                "preview_artifact_key": f"{stem}_preview_png",
            }
        )

    logits_path = output_directory / "complete_classification_logits.npz"
    np.savez(logits_path, **logits)
    outputs["complete_logits_npz"] = str(logits_path)

    for output_key in REGRESSION_OUTPUTS:
        full_tensor = tasks[output_key].detach().float().cpu()
        if (
            full_tensor.shape[:2] != (1, 1)
            or full_tensor.ndim != 5
            or tuple(int(size) for size in full_tensor.shape[-3:]) != seismic.shape
        ):
            raise LayerPulsePlatformRunnerError(f"{output_key} shape is not [1,1,T,I,X]")
        tensor = full_tensor[(slice(None), slice(None), *output_slices)]
        array = tensor[0, 0].numpy()
        if not np.isfinite(array).all():
            raise LayerPulsePlatformRunnerError(f"{output_key} contains non-finite values")
        stored = np.ascontiguousarray(array, dtype=np.float32)
        scalar_volumes[output_key] = stored
        volume_path = output_directory / f"{output_key}.npy"
        np.save(volume_path, stored, allow_pickle=False)
        outputs[f"{output_key}_npy"] = str(volume_path)
        preview_path = output_directory / f"{output_key}_preview.png"
        _render_task_preview(
            task_key=output_key,
            seismic=output_seismic,
            values=array,
            categorical=False,
            destination=preview_path,
        )
        outputs[f"{output_key}_preview_png"] = str(preview_path)
        preview_paths.append((output_key, preview_path))
        minimum, maximum = _finite_min_max(array)
        task_catalog.append(
            {
                "name": output_key,
                "output_key": output_key,
                "kind": "regression",
                "channels": 1,
                "background_index": None,
                "selection": None,
                "output_shape_tix": list(array.shape),
                "minimum": minimum,
                "maximum": maximum,
                "finite": True,
                "artifact_key": f"{output_key}_npy",
                "preview_artifact_key": f"{output_key}_preview_png",
            }
        )

    seismic_preview = output_directory / "registered_seismic_preview.png"
    _render_seismic_preview(output_seismic, seismic_preview)
    outputs["registered_seismic_preview_png"] = str(seismic_preview)
    atlas_path = output_directory / "layerpulse_task_atlas.png"
    _render_atlas(preview_paths, atlas_path)
    outputs["task_atlas_png"] = str(atlas_path)
    valid_path = output_directory / "valid_mask.npy"
    np.save(valid_path, output_valid.astype(np.uint8), allow_pickle=False)
    outputs["valid_mask_npy"] = str(valid_path)

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    crop_start = _shape3(
        input_contract.get("crop_start_tix", (0, 0, 0)), name="input.crop_start_tix"
    )
    source_shape = _shape3(
        input_contract.get("source_shape_tix", seismic.shape),
        name="input.source_shape_tix",
    )
    input_geometry = _mapping(input_contract.get("geometry") or {}, name="input.geometry")
    receipt = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "task_id": "layerpulse",
        "model_id": MODEL_ID,
        "model_name": "LayerPulse-GeoChronoGraph unified intelligent interpretation",
        "model_executed": True,
        "single_checkpoint": True,
        "single_forward_calls": 1,
        "checkpoint_forward_calls": 1,
        "input_shape_tix": list(seismic.shape),
        "output_shape_tix": list(output_shape),
        "input": {
            "axes": ["TWT", "INLINE", "XLINE"],
            "model_order": ["TWT", "INLINE", "XLINE"],
            "shape_zyx": list(output_shape),
            "shape_tix": list(output_shape),
            "model_input_shape_zyx": list(seismic.shape),
            "model_input_shape_tix": list(seismic.shape),
            "model_input_origin_tix": list(
                output_window.get("model_input_origin_tix") or crop_start
            ),
            "output_offset_in_model_input_tix": list(
                output_window["output_offset_tix"]
            ),
            "context_halo": output_window,
            "crop_start_zyx": list(crop_start),
            "crop_start_tix": list(crop_start),
            "crop_size_zyx": list(output_shape),
            "crop_size_tix": list(output_shape),
            "resolved_roi_size_zyx": list(output_shape),
            "source_shape_zyx": list(source_shape),
            "source_shape_tix": list(source_shape),
            "crop_selection": crop_selection,
            "selection_policy": crop_selection,
            "well_anchor": input_contract.get("well_anchor"),
            "coordinate_reference": str(
                input_geometry.get("coordinate_reference")
                or input_geometry.get("crs")
                or "source_seismic_grid_crs_unverified"
            ),
            "geometry": input_geometry,
            "normalisation": normalisation,
            "finite_valid_fraction": float(output_valid.mean()),
            "model_input_finite_valid_fraction": float(valid.mean()),
            "time_depth_supervision_is_model_input": False,
            "well_input": well_receipt,
        },
        "geometry": {
            **input_geometry,
            "shape": list(output_shape),
            "crop_start_zyx": list(crop_start),
            "crop_size_zyx": list(output_shape),
            "coordinate_reference": str(
                input_geometry.get("coordinate_reference")
                or input_geometry.get("crs")
                or "source_seismic_grid_crs_unverified"
            ),
        },
        "inference": {
            "scope": preview_scope,
            "is_complete_volume": False,
            "selection_policy": crop_selection,
            "forward_calls": 1,
            "classification_selection": (
                "central_complete_logits_then_direct_argmax_dim1"
                if output_window["enabled"]
                else "complete_logits_direct_argmax_dim1"
            ),
            "complete_logits_retained": True,
            "model_input_shape_tix": list(seismic.shape),
            "output_shape_tix": list(output_shape),
            "context_halo": output_window,
            "context_halo_performance_commitment": False,
            "task_count": 11,
            "classification_task_count": 6,
            "regression_task_count": 5,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "checkpoint_load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "class_codes": list(range(7)),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "schema_version": checkpoint_receipt.get("schema_version"),
            "family": checkpoint_receipt.get("family"),
            "parameter_count": int(checkpoint_receipt.get("parameter_count", 0)),
            "f_final_channels": int(checkpoint_receipt.get("f_final_channels", 0)),
            "head_count": int(checkpoint_receipt.get("head_count", 0)),
            "strict_model_load": bool(checkpoint_receipt.get("strict_model_load")),
            "teacher_required_at_forward": bool(
                checkpoint_receipt.get("teacher_required_at_forward", False)
            ),
        },
        "task_catalog": task_catalog,
        "outputs": outputs,
        "scientific_status": "engineering_candidate",
        "warnings": [
            f"当前平台运行{PREVIEW_LABEL_BY_SCOPE[preview_scope]}，不代表完整工区定量验收。",
            *(
                ["封存快照未提供可消费的MD+轨迹井张量，本次按无井模式运行。"]
                if int(well_receipt["well_count"]) == 0
                else []
            ),
            *(
                ["上下文 halo 的显存与吞吐仅以独立 CUDA 预检收据为准，本结果不作性能承诺。"]
                if output_window["enabled"]
                else []
            ),
        ],
        "provenance": {
            "runner": "layerpulse_single_checkpoint_external_cuda_subprocess",
            "one_forward_returns_all_tasks": True,
            "head_input_contract": "shared_F_final_only",
            "teacher_or_td_opened": False,
            "classification_threshold_used": False,
            "connected_component_cleanup_used": False,
            "complete_logits_cropped_before_argmax": bool(output_window["enabled"]),
            "context_halo": output_window,
            "crop_selection": crop_selection,
            "preview_scope": preview_scope,
            "well_anchor": input_contract.get("well_anchor"),
        },
        "checks": {
            "checkpoint_loaded": True,
            "one_forward": True,
            "all_11_tasks_present": len(task_catalog) == 11,
            "all_outputs_finite": all(bool(item["finite"]) for item in task_catalog),
            "classification_direct_argmax": True,
            "classification_threshold_absent": True,
            "connected_component_cleanup_absent": True,
            "output_coordinate_origin_preserved": True,
            "background_included": True,
            "time_depth_input_absent": True,
        },
    }
    manifest_path = output_directory / "layerpulse_result_manifest.json"
    child_receipt_path = output_directory / "layerpulse_child_receipt.json"
    outputs["manifest_json"] = str(manifest_path)
    outputs["receipt_json"] = str(child_receipt_path)
    outputs["layerpulse_manifest_json"] = str(manifest_path)
    _atomic_json(receipt, manifest_path)
    _atomic_json(receipt, child_receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = _load_request(args.request.resolve(strict=True))
        result = run(request)
        _atomic_json(result, args.result.resolve())
    except Exception as exc:  # noqa: BLE001 - child must always emit a failure receipt
        failure = {
            "schema_version": RESULT_SCHEMA,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _atomic_json(failure, args.result.resolve())
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "pass", "result": str(args.result.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
