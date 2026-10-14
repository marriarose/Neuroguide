import pytest
from fastapi.testclient import TestClient
import numpy as np

from backend.app.main import app
from backend.app.services.inference_service import InferenceService

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert data["total_subjects"] == 20
    assert data["total_windows"] == 69742

def test_dataset_status_endpoint():
    response = client.get("/api/dataset/status?feature_type=de_LDS")
    assert response.status_code == 200
    data = response.json()
    assert data["files_detected"] == 20
    assert data["training_permitted"] is True
    assert "audit_report" in data
    assert data["audit_report"]["total_eeg_windows"] == 69742

def test_predict_endpoint_rejection_when_untrained(monkeypatch):
    monkeypatch.setattr(InferenceService, "is_model_trained", lambda: False)

    sample_features = np.zeros((5, 62)).tolist()
    response = client.post("/api/predict/features", json={"features": sample_features})

    assert response.status_code == 400
    assert "not trained" in response.json()["detail"]
