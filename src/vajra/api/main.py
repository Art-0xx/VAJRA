# VAJRA inference API — real model inference and real adversarial endpoint.
from __future__ import annotations

import json
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
from vajra.attacks.fgsm import fgsm_attack
from vajra.attacks.pgd import pgd_attack
from vajra.config import load_config
from vajra.models.registry import get_model

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/checkpoints/baseline_model.pt")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/model_transformer.yaml")
METRICS_PATH = os.environ.get("METRICS_PATH", "artifacts/metrics/table_A_clean.json")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_preprocessor = None
_threshold = 0.5


def _load() -> None:
    global _model, _preprocessor, _threshold
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
    if Path(METRICS_PATH).exists():
        m = json.loads(Path(METRICS_PATH).read_text())
        _threshold = float(m.get("optimal_threshold", 0.5))
    print(f"[startup] Model loaded on {_device} ({len(_preprocessor.feature_columns)} "
          f"features, threshold={_threshold:.3f})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="VAJRA NIDS API", version="0.3.0", lifespan=lifespan)


def _to_tensor(features: list[float]) -> torch.Tensor:
    arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if _preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    expected = len(_preprocessor.feature_columns)
    if arr.shape[1] != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {arr.shape[1]}",
        )
    scaled = _preprocessor.scaler.transform(arr).astype(np.float32)
    return torch.from_numpy(scaled).to(_device)


def _predict_tensor(x: torch.Tensor) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label = "attack" if probs[1] > _threshold else "benign"
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
    return _predict_tensor(_to_tensor(req.features))


@app.post("/predict/adversarial", response_model=AdversarialResponse)
def predict_adversarial(req: AdversarialRequest) -> AdversarialResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    x = _to_tensor(req.features)
    clean = _predict_tensor(x)

    # y placeholder: we generate the attack against the true label, but for
    # a single request we do not know it. FGSM/PGD with y=1 ("assume attack")
    # is standard for single-flow demos — we ask the attacker to make the
    # model say "benign" for a sample we believe is an attack.
    y = torch.tensor([1], dtype=torch.long, device=_device)

    if req.attack == "fgsm":
        result = fgsm_attack(_model, x, y, epsilon=req.epsilon)
        adv_x = result.adversarial
    elif req.attack == "pgd":
        result = pgd_attack(_model, x, y, epsilon=req.epsilon, num_iter=10)
        adv_x = result.adversarial
    else:
        raise HTTPException(status_code=400, detail=f"Unknown attack: {req.attack}")

    adv = _predict_tensor(adv_x)
    flipped = clean.label != adv.label

    return AdversarialResponse(
        clean=clean,
        adversarial=adv,
        label_flipped=flipped,
        acs=0.0,             # Phase 4 replaces this with the real divergence score
        acs_threshold=0.11,
        flagged=False,
    )