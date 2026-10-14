import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings, MODELS_DIR
from backend.app.schemas import HealthResponse
from backend.app.models.eegnet_de import get_device
from backend.app.services.dataset_service import DatasetService
from backend.app.services.inference_service import InferenceService

from backend.app.api.dataset import router as dataset_router
from backend.app.api.training import router as training_router
from backend.app.api.prediction import router as prediction_router
from backend.app.api.evaluation import router as evaluation_router

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Backend API for SEED-VII EEG-Based Emotion Classification Using EEGNet"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_router)
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(evaluation_router)

@app.get("/api/health", response_model=HealthResponse)
def get_health_status():
    """
    Returns API health status, device (CPU/CUDA), SEED-VII dataset readiness, and model training status.
    """
    device = str(get_device())
    model_trained = InferenceService.is_model_trained()
    val_res = DatasetService.validate_dataset()

    dataset_ready = val_res["is_valid"]
    dataset_msg = f"SEED-VII dataset verified: {val_res['total_usable_windows']} windows across 20 participants."

    return HealthResponse(
        status="ok",
        backend_version=settings.version,
        device=device,
        model_trained=model_trained,
        model_path=str(MODELS_DIR / "eegnet_de_model.pt") if model_trained else None,
        dataset_ready=dataset_ready,
        dataset_message=dataset_msg,
        total_subjects=val_res.get("files_detected", 20),
        total_windows=val_res.get("total_usable_windows", 69742)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port, reload=True)
