import pytest
import numpy as np
import torch
from backend.app.schemas import TrainingConfigRequest
from backend.app.services.training_service import TrainingService
from backend.app.services.dataset_service import DatasetService
from backend.app.services.inference_service import InferenceService

def test_training_pipeline_and_inference(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("backend.app.services.training_service.MODELS_DIR", models_dir)
    monkeypatch.setattr("backend.app.services.training_service.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("backend.app.services.inference_service.MODELS_DIR", models_dir)

    config = TrainingConfigRequest(
        task_mode="7_class",
        epochs=2,
        batch_size=128,
        learning_rate=0.01,
        dropout=0.1
    )

    metrics = TrainingService.train_model_pipeline(config)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert len(metrics["per_class"]) == 7
    assert (models_dir / "eegnet_de_model.pt").exists()
    assert (reports_dir / "metrics.json").exists()

    loaded = InferenceService.load_trained_model()
    assert loaded is True

    sample_feat = np.random.randn(5, 62).astype(np.float32)
    pred_res = InferenceService.predict_features(sample_feat)

    assert pred_res.predicted_class in ["disgust", "fear", "sad", "neutral", "surprise", "happy", "anger"]
    assert 0.0 <= pred_res.confidence <= 1.0
    assert len(pred_res.probabilities) == 7
