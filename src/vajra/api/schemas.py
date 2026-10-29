# Pydantic request/response models for the VAJRA API.
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class FlowRequest(BaseModel):
    features: list[float] = Field(..., description="Raw flow feature vector")


class PredictionResponse(BaseModel):
    label: Literal["attack", "benign"]
    confidence: float
    probabilities: dict[str, float]


class AdversarialRequest(BaseModel):
    features: list[float]
    attack: Literal["fgsm", "pgd"] = "fgsm"
    epsilon: float = Field(0.05, ge=0.0, le=0.5)


class AdversarialResponse(BaseModel):
    clean: PredictionResponse
    adversarial: PredictionResponse
    label_flipped: bool
    acs: float
    acs_threshold: float
    flagged: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    is_model_loaded: bool
    is_preprocessor_loaded: bool
