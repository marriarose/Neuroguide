from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# Health & Status
class HealthResponse(BaseModel):
    status: str
    backend_version: str
    device: str
    model_trained: bool
    model_path: Optional[str] = None
    dataset_ready: bool
    dataset_message: str
    total_subjects: int = 20
    total_windows: int = 69742

class AuditReportResponse(BaseModel):
    dataset_name: str = "SEED-VII"
    total_participants: int
    participants: List[int]
    sessions_per_participant: int
    trials_per_session: int
    total_trials: int
    total_eeg_windows: int
    total_eye_windows: int
    eeg_feature_shape: List[int]
    eye_feature_shape: List[int]
    temporal_alignment_mismatches: int
    nan_missing_values: int
    discrete_classes_count: int
    discrete_class_distribution: Dict[str, int]
    continuous_labels_available: bool
    dataset_valid: bool
    audit_passed: bool
    summary: str

class DatasetStatusResponse(BaseModel):
    files_detected: int
    files: List[str]
    selected_feature_type: str
    total_arrays: int
    labeled_arrays: int
    unlabeled_arrays: int
    total_usable_windows: int
    shape_distribution: Dict[str, int]
    class_distribution: Dict[str, int]
    validation_status: str = Field(alias="status", default="valid")
    errors: List[str]
    warnings: List[str]
    training_permitted: bool
    audit_report: Optional[AuditReportResponse] = None

    model_config = ConfigDict(populate_by_name=True)

class ValidateDatasetRequest(BaseModel):
    feature_type: str = "de_LDS"

class ValidateDatasetResponse(BaseModel):
    is_valid: bool
    message: str
    errors: List[str]
    warnings: List[str]
    total_observations: int
    class_counts: Dict[str, int]

# Training Schemas
class TrainingConfigRequest(BaseModel):
    feature_type: str = "de_LDS"  # "de_LDS", "de", "psd"
    task_mode: str = "7_class"   # "7_class", "3_class", "regression"
    modality: str = "eeg_only"   # "eeg_only", "eeg_eye_multimodal"
    batch_size: int = Field(default=64, ge=4, le=1024)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    epochs: int = Field(default=35, ge=1, le=200)
    dropout: float = Field(default=0.25, ge=0.0, le=0.9)
    optimizer: str = "adam"
    random_seed: int = 42
    early_stopping_patience: int = 8
    class_weighted_loss: bool = False

class TrainingStatusResponse(BaseModel):
    status: str  # "idle", "running", "completed", "failed", "blocked"
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    best_val_acc: Optional[float] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    history: List[Dict[str, Any]] = []

# Evaluation Schemas
class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int

class EvaluationMetricsResponse(BaseModel):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, PerClassMetrics]
    confusion_matrix: List[List[int]]
    classes: List[str]
    loss_history: Dict[str, List[float]]
    accuracy_history: Dict[str, List[float]]
    split_info: Dict[str, Any]
    limitations_warning: Optional[str] = None

# Prediction Schemas
class PredictFeaturesRequest(BaseModel):
    features: List[List[float]]  # (5, 62)
    eye_features: Optional[List[float]] = None  # (33,)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str
    disclaimer: str = "Predictions are based on SEED-VII EEG/Eye-derived features and do not constitute a clinical diagnosis."
