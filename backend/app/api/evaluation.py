import json
from fastapi import APIRouter, HTTPException
from backend.app.config import REPORTS_DIR, MODELS_DIR
from backend.app.schemas import EvaluationMetricsResponse

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

@router.get("/metrics", response_model=EvaluationMetricsResponse)
def get_evaluation_metrics():
    """
    Returns saved evaluation metrics (Accuracy, Macro F1, Per-class Precision/Recall/F1, Confusion Matrix, loss/accuracy history)
    from the latest completed model evaluation.
    """
    metrics_file = REPORTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation metrics found. Please train and evaluate the model first."
        )

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationMetricsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation metrics: {str(e)}")
