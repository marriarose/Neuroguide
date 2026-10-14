import datetime
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from backend.app.config import (
    MODELS_DIR,
    SEED7_EMOTION_CLASSES,
    NUM_BANDS,
    NUM_CHANNELS,
    NUM_EYE_FEATURES,
)
from backend.app.models.eegnet_de import EEGNetDEClassifier, MultimodalEEGNetClassifier, get_device
from backend.app.schemas import PredictionResponse

class InferenceService:
    """
    Service for loading trained SEED-VII EEGNet checkpoints and running real-time 7-class emotion prediction
    on single feature vectors (5, 62) or uploaded MATLAB feature files.
    """

    _model: torch.nn.Module = None
    _scaler_mean: np.ndarray = None
    _scaler_std: np.ndarray = None
    _checkpoint_meta: Dict[str, Any] = {}
    _classes: List[str] = SEED7_EMOTION_CLASSES

    @classmethod
    def is_model_trained(cls) -> bool:
        model_path = MODELS_DIR / "eegnet_de_model.pt"
        return model_path.exists()

    @classmethod
    def load_trained_model(cls) -> bool:
        model_path = MODELS_DIR / "eegnet_de_model.pt"
        if not model_path.exists():
            return False

        device = get_device()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        config = checkpoint.get("config", {})
        classes = checkpoint.get("classes", SEED7_EMOTION_CLASSES)
        modality = config.get("modality", "eeg_only")
        dropout_rate = config.get("dropout", 0.25)
        num_classes = len(classes)

        if modality == "eeg_eye_multimodal":
            model = MultimodalEEGNetClassifier(
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        else:
            model = EEGNetDEClassifier(
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        cls._model = model
        cls._classes = classes
        cls._checkpoint_meta = checkpoint
        cls._scaler_mean = checkpoint.get("scaler_mean")
        cls._scaler_std = checkpoint.get("scaler_std")

        if cls._scaler_mean is None or cls._scaler_std is None:
            scaler_path = MODELS_DIR / "scaler.npz"
            if scaler_path.exists():
                sdata = np.load(scaler_path)
                cls._scaler_mean = sdata["mean"]
                cls._scaler_std = sdata["std"]

        return True

    @classmethod
    def predict_features(
        cls,
        features: Union[np.ndarray, List[List[float]]],
        eye_features: Optional[Union[np.ndarray, List[float]]] = None
    ) -> PredictionResponse:
        """
        Runs 7-class emotion prediction on a 5-band x 62-channel EEG feature array.
        """
        if not cls.is_model_trained():
            raise RuntimeError("No trained model checkpoint found. Please train the model before running inference.")

        if cls._model is None:
            cls.load_trained_model()

        arr = np.array(features, dtype=np.float32)

        if arr.ndim == 1 and arr.size == NUM_BANDS * NUM_CHANNELS:
            arr = arr.reshape(1, NUM_BANDS, NUM_CHANNELS)
        elif arr.ndim == 2 and arr.shape == (NUM_BANDS, NUM_CHANNELS):
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[1:] == (NUM_BANDS, NUM_CHANNELS):
            pass
        else:
            raise ValueError(f"Invalid feature shape {arr.shape}. Expected (5, 62) or flat 310 numbers.")

        if cls._scaler_mean is not None and cls._scaler_std is not None:
            arr_norm = (arr - cls._scaler_mean) / cls._scaler_std
        else:
            arr_norm = arr

        device = get_device()
        tensor_in = torch.tensor(arr_norm, dtype=torch.float32).unsqueeze(1).to(device)

        tensor_eye = None
        if eye_features is not None:
            arr_eye = np.array(eye_features, dtype=np.float32).reshape(1, -1)
            tensor_eye = torch.tensor(arr_eye, dtype=torch.float32).to(device)

        with torch.no_grad():
            if isinstance(cls._model, MultimodalEEGNetClassifier) and tensor_eye is not None:
                logits = cls._model(tensor_in, tensor_eye)
            else:
                logits = cls._model(tensor_in)

            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = cls._classes[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {
            cname: float(probs[idx])
            for idx, cname in enumerate(cls._classes)
        }

        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return PredictionResponse(
            predicted_class=pred_label,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            model_version="SEED-VII-EEGNet-v2.0",
            timestamp=ts
        )

    @classmethod
    def predict_file(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Parses an uploaded MATLAB file and predicts emotions for all feature arrays inside.
        """
        if not cls.is_model_trained():
            raise RuntimeError("No trained model checkpoint found.")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            mat_data = loadmat(tmp_path)
            extracted_predictions = []

            for key, val in mat_data.items():
                if key.startswith("__"):
                    continue

                if isinstance(val, np.ndarray) and val.ndim == 3:
                    arr = val
                    if arr.shape[1] == 5 and arr.shape[2] == 62:
                        arr_std = arr
                    elif arr.shape[0] == 5 and arr.shape[1] == 62:
                        arr_std = np.transpose(arr, (2, 0, 1))
                    elif arr.shape[0] == 62 and arr.shape[1] == 5:
                        arr_std = np.transpose(arr, (2, 1, 0))
                    else:
                        continue

                    window_predictions = []
                    for w in range(arr_std.shape[0]):
                        feat_win = arr_std[w]
                        res = cls.predict_features(feat_win)
                        window_predictions.append({
                            "window_index": w,
                            "predicted_class": res.predicted_class,
                            "confidence": res.confidence,
                            "probabilities": res.probabilities
                        })

                    classes = [wp["predicted_class"] for wp in window_predictions]
                    from collections import Counter
                    majority_class = Counter(classes).most_common(1)[0][0]

                    extracted_predictions.append({
                        "variable_name": key,
                        "num_windows": arr_std.shape[0],
                        "majority_predicted_class": majority_class,
                        "window_predictions": window_predictions
                    })

            if not extracted_predictions:
                raise ValueError("No compatible (N, 5, 62) feature arrays found in the uploaded file.")

            return {
                "filename": filename,
                "arrays_processed": len(extracted_predictions),
                "predictions": extracted_predictions
            }

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
