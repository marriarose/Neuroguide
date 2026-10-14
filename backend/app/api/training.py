from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.schemas import TrainingConfigRequest, TrainingStatusResponse
from backend.app.services.dataset_service import DatasetService
from backend.app.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])

@router.post("/start", response_model=TrainingStatusResponse)
def start_model_training(config: TrainingConfigRequest, background_tasks: BackgroundTasks):
    """
    Starts model training in the background.
    First validates dataset integrity and labels. Returns HTTP 400 error if labels are missing or invalid.
    """
    # 1. Dataset Validation check
    val = DatasetService.validate_dataset(feature_type=config.feature_type)
    if not val["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Training blocked: Dataset validation failed or labels are missing.",
                "errors": val["errors"],
                "warnings": val["warnings"]
            }
        )

    # 2. Process dataset into numpy arrays
    try:
        DatasetService.process_and_save_dataset(feature_type=config.feature_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare processed dataset: {str(e)}")

    # 3. Check if already running
    curr_status = TrainingService.get_status()
    if curr_status.status == "running":
        raise HTTPException(status_code=409, detail="A training job is already in progress.")

    # 4. Launch training in background
    background_tasks.add_task(TrainingService.train_model_pipeline, config)

    return TrainingStatusResponse(
        status="running",
        total_epochs=config.epochs,
        current_epoch=0,
        elapsed_seconds=0.0
    )

@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status():
    """
    Returns live training progress, loss/accuracy curves history, and status.
    """
    return TrainingService.get_status()
