# VAJRA inference API — real model inference.
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from vajra.api.schemas import (
    AdversarialRequest,
    AdversarialResponse,
    FlowRequest,
    HealthResponse,
    PredictionResponse,
)
from vajra.config import load_config
from vajra.models.registry import get_model

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/checkpoints/baseline_model.pt")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/model_transformer.yaml")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_preprocessor = None


def _load() -> None:
    global _model, _preprocessor
    if not Path(MODEL_PATH).exists():
        print(f"[startup] Model not found at {MODEL_PATH}; /predict returns 503.")
        return
    if not Path(PREPROCESSOR_PATH).exists():
        print(f"[startup] Preprocessor not found at {PREPROCESSOR_PATH}.")
        return
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    config = load_config(CONFIG_PATH)
    model = get_model(config, len(_preprocessor.feature_columns)).to(_device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.eval()
    _model = model
    print(f"[startup] Model loaded on {_device} ({len(_preprocessor.feature_columns)} features)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="VAJRA NIDS API", version="0.2.0", lifespan=lifespan)


def _predict_one(features: list[float]) -> PredictionResponse:
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
    expected = len(_preprocessor.feature_columns)
    if arr.shape[1] != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {arr.shape[1]}",
        )
    scaled = _preprocessor.scaler.transform(arr).astype(np.float32)
    with torch.no_grad():
        logits = _model(torch.from_numpy(scaled).to(_device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label = "attack" if probs[1] > 0.5 else "benign"
    return PredictionResponse(
        label=label,
        confidence=float(max(probs[1], probs[0])),
        probabilities={"attack": float(probs[1]), "benign": float(probs[0])},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        is_model_loaded=_model is not None,
        is_preprocessor_loaded=_preprocessor is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: FlowRequest) -> PredictionResponse:
    return _predict_one(req.features)


@app.post("/predict/adversarial", response_model=AdversarialResponse)
def predict_adversarial(req: AdversarialRequest) -> AdversarialResponse:
    clean = _predict_one(req.features)
    # Phase 2 replaces this with a real FGSM/PGD attack.
    return AdversarialResponse(
        clean=clean,
        adversarial=clean,
        label_flipped=False,
        acs=0.0,
        acs_threshold=0.11,
        flagged=False,
    )