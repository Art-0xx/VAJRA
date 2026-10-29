# Contract tests for the VAJRA API.
#
# The prediction tests require artifacts/checkpoints/baseline_model.pt
# (created by scripts/train.py --smoke-test or full training). They skip
# cleanly when the model is missing so the suite works during Phase 1A
# and after Phase 1B training.
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from vajra.api.main import app

MODEL_READY = (
    Path("artifacts/checkpoints/baseline_model.pt").exists()
    and Path("artifacts/preprocessor.joblib").exists()
)

# 77 features — matches the preprocessor's feature_columns count.
DEMO_FLOW = [0.03, 0.11, 0.42, 0.0, 0.88, 0.01, 0.07, 0.19] + [0.05] * 69


@pytest.fixture(scope="module")
def client():
    # `with TestClient(app)` fires the lifespan startup handler, which
    # loads the model. Without the context manager the lifespan is skipped.
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["is_model_loaded"], bool)
    assert isinstance(body["is_preprocessor_loaded"], bool)


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained; run scripts/train.py --smoke-test")
def test_predict_returns_contract(client):
    r = client.post("/predict", json={"features": DEMO_FLOW})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"attack", "benign"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"attack", "benign"}


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained")
def test_predict_rejects_wrong_feature_count(client):
    r = client.post("/predict", json={"features": [0.1] * 5})
    assert r.status_code == 400


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained")
def test_adversarial_returns_contract(client):
    r = client.post(
        "/predict/adversarial",
        json={"features": DEMO_FLOW, "attack": "fgsm", "epsilon": 0.05},
    )
    assert r.status_code == 200
    body = r.json()
    assert "clean" in body and "adversarial" in body
    assert isinstance(body["label_flipped"], bool)
    assert isinstance(body["flagged"], bool)