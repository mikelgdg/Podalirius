"""FastAPI application for Triage-MRI inference."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from triagemri.data.preprocessing import load_and_preprocess
from triagemri.serving.schemas import (
    BatchResponse,
    HealthResponse,
    PredictResponse,
)
from triagemri.serving.model_loader import get_model, get_device, get_anatomy_mode

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Triage-MRI",
    description="Normality screening for 3D MRI volumes",
    version="0.1.0",
)

MAX_UPLOAD_MB = 100


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    try:
        model = get_model()
        model_loaded = True
        anatomy_mode = model.anatomy_mode
    except RuntimeError:
        model_loaded = False
        anatomy_mode = "unknown"

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        gpu_available=torch.cuda.is_available(),
        anatomy_mode=anatomy_mode,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="NIfTI (.nii.gz) or MHA volume"),
    anatomy: str = Query("brain", description="Anatomy for prediction"),
    threshold: float = Query(0.5, description="Decision threshold"),
) -> PredictResponse:
    """Run triage on a single 3D MRI volume."""
    model = get_model()
    device = get_device()

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)")

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        volume = load_and_preprocess(tmp_path)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        if volume.dim() == 4:
            volume = volume.unsqueeze(0)
        volume = volume.to(device)

        t0 = time.time()
        with torch.no_grad():
            output = model(volume, anatomy=anatomy)
        elapsed_ms = (time.time() - t0) * 1000

        score = float(output["score"].item())
        logits_val = float(output["logits"].item())
        decision = "REVISAR" if score >= threshold else "NORMAL"

        return PredictResponse(
            score=score,
            logits=logits_val,
            decision=decision,
            threshold=threshold,
            inference_time_ms=elapsed_ms,
            anatomy=anatomy,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/batch", response_model=BatchResponse)
async def batch_predict(
    files: list[UploadFile] = File(...),
    anatomy: str = Query("brain"),
    threshold: float = Query(0.5),
) -> BatchResponse:
    """Run triage on multiple volumes."""
    model = get_model()
    device = get_device()
    predictions: list[PredictResponse] = []

    t_start = time.time()

    for file in files:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            volume = load_and_preprocess(tmp_path)
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)
            if volume.dim() == 4:
                volume = volume.unsqueeze(0)
            volume = volume.to(device)

            t0 = time.time()
            with torch.no_grad():
                output = model(volume, anatomy=anatomy)
            elapsed_ms = (time.time() - t0) * 1000

            score = float(output["score"].item())
            logits_val = float(output["logits"].item())
            decision = "REVISAR" if score >= threshold else "NORMAL"

            predictions.append(
                PredictResponse(
                    score=score,
                    logits=logits_val,
                    decision=decision,
                    threshold=threshold,
                    inference_time_ms=elapsed_ms,
                    anatomy=anatomy,
                )
            )
        except Exception as exc:
            predictions.append(
                PredictResponse(
                    score=-1.0,
                    logits=-1.0,
                    decision="ERROR",
                    threshold=threshold,
                    inference_time_ms=0.0,
                    anatomy=anatomy,
                    patient_id=f"error: {exc}",
                )
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    total_time = time.time() - t_start
    return BatchResponse(
        predictions=predictions,
        total_time_s=total_time,
        num_samples=len(predictions),
    )
