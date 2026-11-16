# Integration and Contract Tests for VAJRA FastAPI Service (Blueprint Section 5.4)
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from vajra.service.main import app

MODEL_READY = (
    Path("artifacts/checkpoints/baseline_model.pt").exists()
    and Path("artifacts/preprocessor.joblib").exists()
)

DEMO_FLOW = [0.05] * 77


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["features_count"] == 77
    assert data["active_model"] in ["baseline", "defended", "ensemble"]


@pytest.mark.skipif(not MODEL_READY, reason="Model artifacts not available")
def test_predict_endpoint_contract(client):
    payload = {
        "features": DEMO_FLOW,
        "flow_id": "test-flow-001",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "dst_port": 80,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["flow_id"] == "test-flow-001"
    assert data["verdict"] in ["ATTACK", "BENIGN"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "top_features" in data
    assert isinstance(data["top_features"], list)
    assert "attention_divergence" in data
    assert isinstance(data["blocked"], bool)


@pytest.mark.skipif(not MODEL_READY, reason="Model artifacts not available")
def test_explain_endpoint(client):
    payload = {
        "features": DEMO_FLOW,
        "top_k": 3,
        "flow_id": "explain-flow-001",
    }
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["top_features"]) == 3
    assert "entropy" in data
    assert isinstance(data["attributions"], dict)


@pytest.mark.skipif(not MODEL_READY, reason="Model artifacts not available")
def test_model_swap_endpoint(client):
    # Swap to defended
    r = client.post("/model/swap", json={"target_model": "defended"})
    assert r.status_code == 200
    assert r.json()["active_model"] == "defended"
    assert r.json()["threshold"] == 0.50

    # Swap to ensemble
    r = client.post("/model/swap", json={"target_model": "ensemble"})
    assert r.status_code == 200
    assert r.json()["active_model"] == "ensemble"

    # Swap back to baseline
    r = client.post("/model/swap", json={"target_model": "baseline"})
    assert r.status_code == 200
    assert r.json()["active_model"] == "baseline"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "throughput_fps" in data
    assert "latency_p50_ms" in data
    assert "latency_p99_ms" in data
    assert "total_requests" in data

