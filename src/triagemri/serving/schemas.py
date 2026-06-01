"""Pydantic request/response schemas for the Triage-MRI API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Response for a single volume prediction."""

    score: float = Field(..., description="Anomaly score in [0, 1]")
    logits: float = Field(..., description="Raw logit value")
    decision: str = Field(..., description="NORMAL, INCONCLUSIVE, or REVISAR")
    threshold: float = Field(..., description="Decision threshold used")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    anatomy: str = Field(default="brain", description="Anatomy used for prediction")
    patient_id: Optional[str] = Field(default=None)


class BatchResponse(BaseModel):
    """Response for batch prediction."""

    predictions: list[PredictResponse]
    total_time_s: float
    num_samples: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_loaded: bool
    gpu_available: bool
    anatomy_mode: str
