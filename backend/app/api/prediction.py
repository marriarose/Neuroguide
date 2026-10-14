from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.app.schemas import PredictFeaturesRequest, PredictionResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/predict", tags=["prediction"])

@router.post("/features", response_model=PredictionResponse)
def predict_features(request: PredictFeaturesRequest):
    """
    Predicts emotion class (negative, neutral, positive) from a 5-band x 62-channel feature array.
    Returns HTTP 400 error if model is not trained.
    """
    if not InferenceService.is_model_trained():
        raise HTTPException(
            status_code=400,
            detail="Model is not trained yet. Please complete training before requesting predictions."
        )

    try:
        return InferenceService.predict_features(request.features)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/file")
async def predict_file(file: UploadFile = File(...)):
    """
    Predicts emotion classes for compatible (N, 5, 62) feature arrays inside an uploaded MATLAB file.
    """
    if not InferenceService.is_model_trained():
        raise HTTPException(
            status_code=400,
            detail="Model is not trained yet. Please complete training before uploading files for prediction."
        )

    if not file.filename.endswith((".mat", ".MAT")):
        raise HTTPException(status_code=400, detail="Only MATLAB .mat files are supported.")

    try:
        contents = await file.read()
        return InferenceService.predict_file(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file prediction: {str(e)}")
