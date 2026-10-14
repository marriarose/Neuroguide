import os
from pathlib import Path
from pydantic import BaseModel

# Base project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EEG_RAW_DIR = RAW_DATA_DIR / "EEG_features"
EYE_RAW_DIR = RAW_DATA_DIR / "EYE_features"
CONTINUOUS_LABELS_DIR = RAW_DATA_DIR / "continuous_labels"
SAVE_INFO_DIR = RAW_DATA_DIR / "save_info"
METADATA_DIR = DATA_DIR / "metadata"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LABELS_FILE = BASE_DIR / "labels.csv"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard SEED-VII 7 Emotion Classes (Canonical order)
SEED7_EMOTION_CLASSES = ["disgust", "fear", "sad", "neutral", "surprise", "happy", "anger"]
SEED7_LABEL_MAP = {c: i for i, c in enumerate(SEED7_EMOTION_CLASSES)}
SEED7_REVERSE_MAP = {i: c for i, c in enumerate(SEED7_EMOTION_CLASSES)}

# 3-Class Mapping fallback
THREE_CLASS_MAP = {
    "disgust": 0, "fear": 0, "sad": 0, "anger": 0,  # Negative
    "neutral": 1,                                   # Neutral
    "happy": 2, "surprise": 2                       # Positive
}
THREE_CLASS_NAMES = ["negative", "neutral", "positive"]

# EEG & EYE Feature Config
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]  # 5 frequency bands
NUM_BANDS = 5
NUM_CHANNELS = 62
NUM_EYE_FEATURES = 33

class AppConfig(BaseModel):
    app_name: str = "NeuroGuide SEED-VII EEG API"
    version: str = "2.0.0"
    debug: bool = True
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

settings = AppConfig()
