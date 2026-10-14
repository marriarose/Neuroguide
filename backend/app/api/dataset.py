from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas import DatasetStatusResponse, ValidateDatasetRequest, ValidateDatasetResponse
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

@router.get("/status", response_model=DatasetStatusResponse)
def get_dataset_status(feature_type: str = Query("de_LDS", description="Feature type prefix")):
    """
    Returns the dataset readiness status, file inventory, shape distribution, label validation errors,
    and whether training is currently permitted.
    """
    try:
        val = DatasetService.validate_dataset(feature_type=feature_type)
        return DatasetStatusResponse(**val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect dataset status: {str(e)}")

@router.post("/inspect")
def inspect_dataset_files():
    """
    Inspects all MATLAB files recursively in the configured raw dataset directory.
    """
    try:
        return DatasetService.inspect_raw_files()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset inspection error: {str(e)}")

@router.post("/validate", response_model=ValidateDatasetResponse)
def validate_and_process_dataset(request: ValidateDatasetRequest):
    """
    Validates dataset labels and feature shapes. If valid, preprocesses arrays into data/processed/.
    If invalid or labels are missing, blocks execution and returns detailed validation errors.
    """
    val = DatasetService.validate_dataset(feature_type=request.feature_type)
    if not val["is_valid"]:
        return ValidateDatasetResponse(
            is_valid=False,
            message=f"Dataset validation failed for feature type '{request.feature_type}'. Training is blocked.",
            errors=val["errors"],
            warnings=val["warnings"],
            total_observations=0,
            class_counts=val["class_distribution"]
        )

    try:
        X, y, groups, meta = DatasetService.process_and_save_dataset(feature_type=request.feature_type)
        return ValidateDatasetResponse(
            is_valid=True,
            message=f"Successfully validated and processed {len(y)} observations across {len(meta['manifest'])} arrays.",
            errors=[],
            warnings=val["warnings"],
            total_observations=len(y),
            class_counts=val["class_distribution"]
        )
    except Exception as e:
        return ValidateDatasetResponse(
            is_valid=False,
            message=f"Processing failed: {str(e)}",
            errors=[str(e)],
            warnings=val["warnings"],
            total_observations=0,
            class_counts=val["class_distribution"]
        )
