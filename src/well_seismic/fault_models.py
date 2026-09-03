"""Shared identities for full-volume seismic fault probability models."""

from __future__ import annotations

from typing import Any


FAULTSEG_MODEL_ID = "faultseg_3d"
FAULTNET_MODEL_ID = "faultnet_china_field"
FAULT_VOLUME_MODEL_IDS = frozenset({FAULTSEG_MODEL_ID, FAULTNET_MODEL_ID})


def is_fault_volume_model_id(value: Any) -> bool:
    return str(value or "").strip().casefold() in FAULT_VOLUME_MODEL_IDS
