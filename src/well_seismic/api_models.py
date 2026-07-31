"""Stable HTTP request and response contracts for the platform API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InspectionRequest(BaseModel):
    seismic_paths: list[str] = Field(default_factory=list)
    log_paths: list[str] = Field(default_factory=list)
    well_paths: list[str] = Field(default_factory=list)
    auxiliary_paths: list[str] = Field(default_factory=list)
    recursive: bool = True
    lightweight: bool = True
    use_llm_fallback: bool = False


class PreprocessingRequest(InspectionRequest):
    output_directory: str | None = None


class PredictionRequest(BaseModel):
    """Model-neutral prediction request.

    Common volume controls remain first-class fields for backward compatibility.
    Task-specific runners can consume validated values from ``options``.
    """

    task_id: str = "fault"
    model_id: str = "faultseg_3d"
    seismic_path: str
    source_task_id: str | None = None
    crop_start: tuple[int, int, int] | None = None
    crop_size: tuple[int, int, int] | None = None
    patch_size: tuple[int, int, int] | None = None
    overlap: tuple[int, int, int] | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    device: str = "auto"
    output_directory: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class TaskCreated(BaseModel):
    task_id: str
    status: str
    message: str


class ViserSliceRequest(BaseModel):
    task_id: str
    asset_index: int = Field(default=0, ge=0)
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    z: int | None = Field(default=None, ge=0)


class ViserLayerModeRequest(BaseModel):
    task_id: str
    asset_index: int = Field(default=0, ge=0)
    mode: Literal["combined", "prediction"] = "combined"


class IssueConfirmationRequest(BaseModel):
    decision: str
    action: str = ""


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    task_id: str | None = None


class TransformationActivationRequest(BaseModel):
    confirmation: str
