#!/usr/bin/env python
"""VAJRA P0 Bootstrap - creates the entire project structure and files.

Run once from the vajra/ folder:
    python bootstrap.py

Then delete this file.

NOTE: file contents use '#' comments instead of triple-quoted docstrings.
This avoids nested triple-quote parsing errors.
"""
import sys
from pathlib import Path

FOLDERS = [
    ".vscode",
    "configs",
    "data/raw", "data/processed",
    "src/vajra/data", "src/vajra/models", "src/vajra/attacks",
    "src/vajra/defenses", "src/vajra/justification", "src/vajra/evaluation",
    "src/vajra/api", "src/vajra/dashboard", "src/vajra/replay",
    "scripts",
    "artifacts/checkpoints", "artifacts/metrics", "artifacts/figures", "artifacts/replay",
    "tests",
    "docs",
]

FILES = {}

# ---------------- Root ----------------

FILES[".python-version"] = "3.11\n"

FILES[".gitignore"] = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
build/
dist/

# Data (never commit datasets)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Artifacts
artifacts/checkpoints/*
artifacts/metrics/*
artifacts/figures/*
artifacts/replay/*
artifacts/*.joblib
!artifacts/.gitkeep

# MLflow
mlruns/

# OS
.DS_Store
Thumbs.db
"""

FILES[".dockerignore"] = """.git
.venv
data/raw
data/processed
artifacts
tests
notebooks
__pycache__
*.pyc
bootstrap.py
"""

FILES["pyproject.toml"] = """[project]
name = "vajra"
version = "0.1.0"
description = "Vector-Attention Justification for Robust Anomaly Detection"
requires-python = ">=3.10,<3.12"

dependencies = [
    "torch==2.2.1",
    "numpy==1.26.4",
    "adversarial-robustness-toolbox==1.20.1",
    "pandas==2.2.2",
    "scikit-learn==1.4.2",
    "pyarrow==16.0.0",
    "PyYAML==6.0.1",
    "typer==0.12.3",
    "mlflow==2.12.2",
    "fastapi==0.111.0",
    "uvicorn[standard]==0.29.0",
    "pydantic==2.7.1",
    "streamlit==1.35.0",
    "plotly==5.22.0",
    "requests==2.31.0",
    "joblib==1.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.2.0",
    "pytest-cov==5.0.0",
    "httpx==0.27.0",
    "ruff==0.4.4",
    "mypy==1.10.0",
]

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
"""

FILES["requirements.txt"] = """torch==2.2.1
numpy==1.26.4
adversarial-robustness-toolbox==1.20.1
pandas==2.2.2
scikit-learn==1.4.2
pyarrow==16.0.0
PyYAML==6.0.1
typer==0.12.3
mlflow==2.12.2
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
streamlit==1.35.0
plotly==5.22.0
requests==2.31.0
joblib==1.4.0
pytest==8.2.0
httpx==0.27.0
"""

FILES["Dockerfile"] = """FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=/app/src

COPY src/ ./src/
COPY configs/ ./configs/
RUN mkdir -p artifacts/checkpoints artifacts/metrics artifacts/figures

CMD ["uvicorn", "vajra.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

FILES["docker-compose.yml"] = """services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/artifacts/checkpoints/baseline_model.pt
      - PREPROCESSOR_PATH=/app/artifacts/preprocessor.joblib
    volumes:
      - ./artifacts:/app/artifacts:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 3s
      retries: 5

  dashboard:
    build: .
    command: ["streamlit", "run", "src/vajra/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
"""

# ---------------- VS Code ----------------

FILES[".vscode/extensions.json"] = """{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-azuretools.vscode-docker",
    "tamasfe.even-better-toml",
    "redhat.vscode-yaml",
    "eamodio.gitlens"
  ]
}
"""

FILES[".vscode/settings.json"] = """{
  "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
  "python.analysis.typeCheckingMode": "basic",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.testing.unittestEnabled": false,
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.ruff_cache": true
  }
}
"""

FILES[".vscode/launch.json"] = """{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "VAJRA API (uvicorn)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["vajra.api.main:app", "--reload", "--port", "8000"],
      "env": { "PYTHONPATH": "${workspaceFolder}/src" },
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "VAJRA Tests (pytest)",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["-q", "tests"],
      "env": { "PYTHONPATH": "${workspaceFolder}/src" },
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "VAJRA Dashboard (streamlit)",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "src/vajra/dashboard/app.py"],
      "env": { "PYTHONPATH": "${workspaceFolder}/src" },
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    }
  ]
}
"""

FILES[".vscode/tasks.json"] = """{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "VAJRA: Run Tests",
      "type": "shell",
      "command": "pytest -q",
      "group": { "kind": "test", "isDefault": true },
      "problemMatcher": []
    },
    {
      "label": "VAJRA: Start API",
      "type": "shell",
      "command": "uvicorn vajra.api.main:app --reload --port 8000",
      "isBackground": true,
      "problemMatcher": [],
      "group": "build"
    },
    {
      "label": "VAJRA: Start Dashboard",
      "type": "shell",
      "command": "streamlit run src/vajra/dashboard/app.py",
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "VAJRA: Docker Compose Up",
      "type": "shell",
      "command": "docker compose up --build",
      "group": "build",
      "problemMatcher": []
    }
  ]
}
"""

# ---------------- Configs ----------------

FILES["configs/base.yaml"] = """seed: 42
device: cpu
paths:
  data_raw: data/raw
  data_processed: data/processed
  artifacts: artifacts
  preprocessor: artifacts/preprocessor.joblib
dataset:
  name: cicids2017
  n_features: 78
  label_column: Label
  test_size: 0.15
  val_size: 0.15
"""

FILES["configs/model_transformer.yaml"] = """inherits: base.yaml
model:
  architecture: transformer
  d_model: 128
  n_heads: 8
  n_layers: 4
  n_classes: 2
  dropout: 0.1
training:
  batch_size: 256
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-5
  early_stopping_patience: 5
  checkpoint_path: artifacts/checkpoints/baseline_model.pt
"""

FILES["configs/attack_fgsm.yaml"] = """inherits: base.yaml
attack:
  name: fgsm
  epsilon_grid: [0.01, 0.03, 0.05, 0.1]
  clip_min: 0.0
  clip_max: 1.0
checkpoint: artifacts/checkpoints/baseline_model.pt
output_csv: artifacts/metrics/table_B_fgsm.csv
"""

FILES["configs/attack_pgd.yaml"] = """inherits: base.yaml
attack:
  name: pgd
  epsilon_grid: [0.01, 0.03, 0.05, 0.1]
  eps_step: 0.01
  max_iter: 10
  clip_min: 0.0
  clip_max: 1.0
  subset_size: 50000
checkpoint: artifacts/checkpoints/baseline_model.pt
output_csv: artifacts/metrics/table_B_pgd.csv
"""

FILES["configs/defense_pgd_at.yaml"] = """inherits: base.yaml
defense:
  name: pgd_at
  epsilon: 0.05
  max_iter: 7
training:
  batch_size: 128
  epochs: 30
  lr: 5.0e-5
checkpoint_path: artifacts/checkpoints/defended_pgdat_model.pt
"""

FILES["configs/defense_trades.yaml"] = """inherits: base.yaml
defense:
  name: trades
  epsilon: 0.05
  beta: 6.0
  max_iter: 7
training:
  batch_size: 128
  epochs: 30
  lr: 5.0e-5
checkpoint_path: artifacts/checkpoints/defended_trades_model.pt
"""

FILES["configs/replay_mininet.yaml"] = """topology:
  attacker_ip: 10.0.0.1
  victim_ip: 10.0.0.2
  nids_ip: 10.0.0.3
  bridge: vajra-br0
replay:
  template_pcap: data/raw/sample_attack.pcap
  output_pcap: artifacts/replay/adv.pcap
  target_url: http://vajra-api:8000
"""

# ---------------- Package markers ----------------

FILES["src/vajra/__init__.py"] = (
    "# VAJRA - Vector-Attention Justification for Robust Anomaly Detection.\n"
    '__version__ = "0.1.0"\n'
)
FILES["src/vajra/data/__init__.py"] = ""
FILES["src/vajra/models/__init__.py"] = ""
FILES["src/vajra/attacks/__init__.py"] = ""
FILES["src/vajra/defenses/__init__.py"] = ""
FILES["src/vajra/justification/__init__.py"] = ""
FILES["src/vajra/evaluation/__init__.py"] = ""
FILES["src/vajra/api/__init__.py"] = ""
FILES["src/vajra/dashboard/__init__.py"] = ""
FILES["src/vajra/replay/__init__.py"] = ""
FILES["tests/__init__.py"] = ""

# ---------------- Python source files ----------------

FILES["src/vajra/config.py"] = '''# Load YAML configs into typed dataclasses.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if "inherits" in raw:
        parent = load_config(path.parent / raw.pop("inherits"))
        raw = {**parent.raw, **raw}
    return Config(raw=raw, path=path)
'''

FILES["src/vajra/api/schemas.py"] = '''# Pydantic request/response models for the VAJRA API.
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
    model_loaded: bool
    preprocessor_loaded: bool
'''

FILES["src/vajra/api/main.py"] = '''# VAJRA inference API (P0 placeholder implementation).
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from vajra.api.schemas import (
    AdversarialRequest,
    AdversarialResponse,
    FlowRequest,
    HealthResponse,
    PredictionResponse,
)

app = FastAPI(title="VAJRA NIDS API", version="0.1.0")

_MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/checkpoints/baseline_model.pt")
_PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")


def _model_loaded() -> bool:
    return Path(_MODEL_PATH).exists()


def _preprocessor_loaded() -> bool:
    return Path(_PREPROCESSOR_PATH).exists()


def _predict_placeholder(features: list[float]) -> PredictionResponse:
    arr = np.asarray(features, dtype=np.float32)
    if arr.size == 0:
        raise HTTPException(status_code=400, detail="features must be non-empty")
    score = float(1.0 / (1.0 + np.exp(-arr.mean() * 10.0)))
    label = "attack" if score > 0.5 else "benign"
    return PredictionResponse(
        label=label,
        confidence=max(score, 1.0 - score),
        probabilities={"attack": score, "benign": 1.0 - score},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded(),
        preprocessor_loaded=_preprocessor_loaded(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: FlowRequest) -> PredictionResponse:
    return _predict_placeholder(req.features)


@app.post("/predict/adversarial", response_model=AdversarialResponse)
def predict_adversarial(req: AdversarialRequest) -> AdversarialResponse:
    clean = _predict_placeholder(req.features)
    adv = _predict_placeholder(req.features)
    return AdversarialResponse(
        clean=clean,
        adversarial=adv,
        label_flipped=False,
        acs=0.0,
        acs_threshold=0.11,
        flagged=False,
    )
'''

FILES["src/vajra/dashboard/app.py"] = '''# VAJRA Streamlit dashboard (P0 placeholder implementation).
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEMO_FLOW = [0.03, 0.11, 0.42, 0.0, 0.88, 0.01, 0.07, 0.19] + [0.05] * 70

st.set_page_config(page_title="VAJRA NIDS Console", layout="wide")
st.title("VAJRA - Robust NIDS Console")

with st.sidebar:
    st.header("Controls")
    model_choice = st.selectbox("Model", ["Undefended", "PGD-AT", "TRADES"])
    attack_choice = st.selectbox("Attack", ["None", "FGSM", "PGD", "AutoAttack"])
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    run = st.button("Run Attack", type="primary")

try:
    health = requests.get(API_BASE_URL + "/health", timeout=3).json()
    st.caption(
        "API status: **" + health["status"] + "** | "
        + "model_loaded: `" + str(health["model_loaded"]) + "` | "
        + "preprocessor_loaded: `" + str(health["preprocessor_loaded"]) + "`"
    )
except Exception as exc:
    st.error("API unreachable at " + API_BASE_URL + ": " + str(exc))
    st.stop()

if run:
    try:
        if attack_choice == "None":
            resp = requests.post(
                API_BASE_URL + "/predict",
                json={"features": DEMO_FLOW},
                timeout=10,
            ).json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", resp["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["confidence"]))
            with col2:
                st.info("No attack selected. Switch attack to FGSM/PGD to test.")
        else:
            resp = requests.post(
                API_BASE_URL + "/predict/adversarial",
                json={
                    "features": DEMO_FLOW,
                    "attack": attack_choice.lower(),
                    "epsilon": epsilon,
                },
                timeout=30,
            ).json()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Clean")
                st.metric("Label", resp["clean"]["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["clean"]["confidence"]))
            with col2:
                st.subheader("Adversarial")
                st.metric("Label", resp["adversarial"]["label"].upper())
                st.metric("Confidence", "{:.2f}".format(resp["adversarial"]["confidence"]))
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Label flipped", str(resp["label_flipped"]))
            c2.metric("ACS", "{:.2f}".format(resp["acs"]))
            if resp["flagged"]:
                st.error("Adversarial input detected by attention divergence")
            else:
                st.success("Attention divergence within normal range")
    except Exception as exc:
        st.error("Request failed: " + str(exc))

st.divider()
st.caption("P0 skeleton - predictions are placeholders until Week 3.")
'''

# ---------------- Tests ----------------

FILES["tests/test_api.py"] = '''# Contract tests for the VAJRA API.
from fastapi.testclient import TestClient
from vajra.api.main import app

client = TestClient(app)
DEMO_FLOW = [0.03, 0.11, 0.42, 0.0, 0.88, 0.01, 0.07, 0.19] + [0.05] * 70


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["preprocessor_loaded"], bool)


def test_predict_returns_contract():
    r = client.post("/predict", json={"features": DEMO_FLOW})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"attack", "benign"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"attack", "benign"}


def test_predict_rejects_empty_features():
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 400


def test_adversarial_returns_contract():
    r = client.post(
        "/predict/adversarial",
        json={"features": DEMO_FLOW, "attack": "fgsm", "epsilon": 0.05},
    )
    assert r.status_code == 200
    body = r.json()
    assert "clean" in body and "adversarial" in body
    assert isinstance(body["label_flipped"], bool)
    assert isinstance(body["flagged"], bool)
'''

# ---------------- Docs ----------------

FILES["README.md"] = """# VAJRA

Vector-Attention Justification for Robust Anomaly Detection.

## Quick Start

    docker compose up --build

- API: http://localhost:8000/health
- Dashboard: http://localhost:8501

## Local Development

    pip install -e ".[dev]"
    pytest -q
    uvicorn vajra.api.main:app --reload
    streamlit run src/vajra/dashboard/app.py
"""

FILES["ETHICS.md"] = """# Ethics and Safety Statement

VAJRA includes a live-replay component that reconstructs adversarial network
flows into PCAP files and injects them into an emulated network. This is
defensive security research.

## Isolation Guarantees

1. All replay experiments run inside a Mininet virtual topology with no route
   to the host network, the university network, or the public internet.
2. No traffic is injected onto production interfaces.
3. Reconstructed PCAPs contain only traffic derived from public benchmarks.
4. The vajra-replay Docker profile is opt-in.
"""

FILES["REPRODUCE.md"] = """# Reproducing VAJRA Results

## Data Preparation

    python scripts/prepare_data.py --dataset cicids2017

## Table A - Clean Performance

    python scripts/evaluate_clean.py --checkpoint artifacts/checkpoints/baseline_model.pt

## Table B - Robustness Profile

    python scripts/run_attack.py --attack fgsm --eps 0.01 0.03 0.05 0.1 --seeds 1 2 3
    python scripts/run_attack.py --attack pgd  --eps 0.01 0.03 0.05 0.1 --seeds 1 2 3
"""

FILES["docs/architecture.md"] = """# VAJRA Architecture

## Services

| Service | Port | Responsibility |
|---|---|---|
| vajra-api | 8000 | Inference, attack endpoint, attention endpoint |
| vajra-dashboard | 8501 | Streamlit UI |
| vajra-replay | internal | PCAP + Mininet replay (opt-in) |

## Invariants

1. API never imports training code.
2. Preprocessing uses the same preprocessor.joblib as training.
3. Feature order is enforced.
4. No service has external network access.
"""

FILES["docs/threat_model.md"] = """# Threat Model

| Dimension | Specification |
|---|---|
| Attacker knowledge | White-box primary; grey-box via transfer |
| Perturbation space | Feature-space L-inf, eps in {0.01, 0.03, 0.05, 0.1} |
| Attacker goal | Evasion and induction |
| Attack families | FGSM, PGD, AutoAttack |
| Out of scope | Physical-layer, protocol-invalid, black-box queries |
"""

FILES["docs/evaluation_protocol.md"] = """# Evaluation Protocol

Metrics: clean F1, robust accuracy at eps, ASR, robust F1, delta F1,
security curve, FPR@1%, latency.

Tables A-D per VAJRA-TD-001 section 5.2. All results mean +/- std over 3 seeds.
"""

FILES["docs/dataset_card.md"] = """# Dataset Card

| Field | Value |
|---|---|
| Dataset | CIC-IDS2017 |
| Rows after cleaning | (fill after Week 2) |
| Features | 78 |
| Split | 70/15/15 stratified, seed=42 |
| Preprocessor fit | Train split only |
"""

FILES["docs/demo_guide.md"] = """# Demo Guide (90 seconds)

1. Open dashboard at http://localhost:8501
2. Attack = None -> Run. Shows clean ATTACK.
3. Attack = FGSM, eps = 0.05 -> Run. Label flips (Week 4+).
4. Attack = PGD, eps = 0.10 -> Run. Stronger flip.
5. Model = TRADES, same PGD eps=0.05 -> Run. Label stays ATTACK.
"""

# ---------------- Gitkeeps ----------------

FILES["data/raw/.gitkeep"] = ""
FILES["data/processed/.gitkeep"] = ""
FILES["artifacts/.gitkeep"] = ""


def main() -> None:
    root = Path.cwd()
    if (root / "pyproject.toml").exists():
        print("ERROR: " + str(root) + " already has pyproject.toml")
        print("Refusing to overwrite. Run this script in an empty folder.")
        sys.exit(1)

    print("Bootstrapping VAJRA in " + str(root) + "\n")

    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
        print("  mkdir  " + folder + "/")

    for rel_path, content in FILES.items():
        full = root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        print("  write  " + rel_path)

    print("\nDone. P0 structure created.\n")
    print("Next steps:")
    print("  1. python -m venv .venv")
    print("  2. .\\.venv\\Scripts\\Activate.ps1      (Windows PowerShell)")
    print("     source .venv/bin/activate          (Mac/Linux)")
    print("  3. pip install -e .[dev]")
    print("  4. pytest -q")
    print("  5. Delete bootstrap.py")


if __name__ == "__main__":
    main()