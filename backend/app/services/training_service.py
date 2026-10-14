import time
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from backend.app.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    SEED7_EMOTION_CLASSES,
    THREE_CLASS_NAMES,
)
from backend.app.schemas import TrainingConfigRequest, TrainingStatusResponse
from backend.app.models.eegnet_de import EEGNetDEClassifier, MultimodalEEGNetClassifier, get_device
from backend.app.services.dataset_service import DatasetService

class TrainingService:
    """
    Service for executing PyTorch training loops for SEED-VII 7-class emotion classification,
    3-class emotion mapping, or multimodal fusion.
    """

    _current_status = TrainingStatusResponse(status="idle")
    _is_running = False

    @classmethod
    def get_status(cls) -> TrainingStatusResponse:
        return cls._current_status

    @classmethod
    def train_model_pipeline(
        cls,
        config: TrainingConfigRequest,
        X_eeg: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        X_eye: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if cls._is_running:
            raise RuntimeError("A training job is already running.")

        cls._is_running = True
        cls._current_status = TrainingStatusResponse(
            status="running",
            total_epochs=config.epochs,
            current_epoch=0,
            elapsed_seconds=0.0,
            history=[]
        )

        start_time = time.time()

        try:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.random_seed)

            device = get_device()

            # Load processed data if not provided
            if X_eeg is None or y is None or groups is None:
                proc_file = PROCESSED_DATA_DIR / "dataset.npz"
                if not proc_file.exists():
                    # Run dataset preparation automatically
                    DatasetService.process_and_save_dataset(feature_type=config.feature_type)
                
                data_npz = np.load(proc_file)
                X_eeg = data_npz["X_eeg"]
                X_eye = data_npz["X_eye"] if config.modality == "eeg_eye_multimodal" else None
                groups = data_npz["groups"]
                
                if config.task_mode == "3_class":
                    y = data_npz["y_3"]
                    class_names = THREE_CLASS_NAMES
                elif config.task_mode == "regression":
                    y = data_npz["y_cont"]
                    class_names = ["continuous_val"]
                else:
                    y = data_npz["y_7"]
                    class_names = SEED7_EMOTION_CLASSES
            else:
                class_names = SEED7_EMOTION_CLASSES if config.task_mode == "7_class" else THREE_CLASS_NAMES

            num_classes = len(class_names) if config.task_mode != "regression" else 1

            # Subject-Independent Grouped Split
            (tr_tup, val_tup, test_tup, split_info) = DatasetService.split_data_group_aware(
                X_eeg, y, groups, X_eye=X_eye, random_seed=config.random_seed
            )

            X_tr, y_tr, X_eye_tr = tr_tup
            X_v, y_v, X_eye_v = val_tup
            X_te, y_te, X_eye_te = test_tup

            # Normalize EEG features strictly on train split
            scaler_file = MODELS_DIR / "scaler.npz"
            X_tr_norm, X_v_norm, X_te_norm, (mean_stat, std_stat) = DatasetService.normalize_splits(
                X_tr, X_v, X_te, save_path=scaler_file
            )

            # PyTorch Datasets
            if config.modality == "eeg_eye_multimodal" and X_eye_tr is not None:
                train_ds = TensorDataset(
                    torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(X_eye_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )
                val_ds = TensorDataset(
                    torch.tensor(X_v_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(X_eye_v, dtype=torch.float32),
                    torch.tensor(y_v, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )
                test_ds = TensorDataset(
                    torch.tensor(X_te_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(X_eye_te, dtype=torch.float32),
                    torch.tensor(y_te, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )
            else:
                train_ds = TensorDataset(
                    torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_tr, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )
                val_ds = TensorDataset(
                    torch.tensor(X_v_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_v, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )
                test_ds = TensorDataset(
                    torch.tensor(X_te_norm, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_te, dtype=torch.long if config.task_mode != "regression" else torch.float32)
                )

            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

            # Model Initialization
            if config.modality == "eeg_eye_multimodal":
                model = MultimodalEEGNetClassifier(
                    num_classes=num_classes,
                    dropout_rate=config.dropout
                ).to(device)
            else:
                model = EEGNetDEClassifier(
                    num_classes=num_classes,
                    dropout_rate=config.dropout
                ).to(device)

            criterion = nn.CrossEntropyLoss()

            # Optimizer
            opt_name = config.optimizer.lower()
            if opt_name == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
            elif opt_name == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )

            best_val_loss = float("inf")
            best_val_acc = 0.0
            patience_counter = 0
            best_model_state = None
            history = []

            for epoch in range(1, config.epochs + 1):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for batch in train_loader:
                    if config.modality == "eeg_eye_multimodal":
                        inputs, inputs_eye, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        outputs = model(inputs, inputs_eye)
                    else:
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                        outputs = model(inputs)

                    optimizer.zero_grad()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_train += (preds == targets).sum().item()
                    total_train += targets.size(0)

                epoch_train_loss = running_loss / max(1, total_train)
                epoch_train_acc = correct_train / max(1, total_train)

                # Validation Phase
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for batch in val_loader:
                        if config.modality == "eeg_eye_multimodal":
                            inputs, inputs_eye, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                            outputs = model(inputs, inputs_eye)
                        else:
                            inputs, targets = batch[0].to(device), batch[1].to(device)
                            outputs = model(inputs)

                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        correct_val += (preds == targets).sum().item()
                        total_val += targets.size(0)

                epoch_val_loss = val_loss / max(1, total_val)
                epoch_val_acc = correct_val / max(1, total_val)

                scheduler.step(epoch_val_loss)

                epoch_info = {
                    "epoch": epoch,
                    "train_loss": round(epoch_train_loss, 4),
                    "val_loss": round(epoch_val_loss, 4),
                    "train_acc": round(epoch_train_acc, 4),
                    "val_acc": round(epoch_val_acc, 4)
                }
                history.append(epoch_info)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_val_acc = epoch_val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                elapsed = time.time() - start_time
                cls._current_status = TrainingStatusResponse(
                    status="running",
                    current_epoch=epoch,
                    total_epochs=config.epochs,
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                    train_acc=epoch_train_acc,
                    val_acc=epoch_val_acc,
                    best_val_acc=best_val_acc,
                    elapsed_seconds=elapsed,
                    history=history
                )

                if patience_counter >= config.early_stopping_patience:
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Save Model Checkpoint
            model_save_path = MODELS_DIR / "eegnet_de_model.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.model_dump(),
                "classes": class_names,
                "scaler_mean": mean_stat,
                "scaler_std": std_stat
            }
            torch.save(checkpoint, model_save_path)

            # Test Evaluation
            model.eval()
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for batch in test_loader:
                    if config.modality == "eeg_eye_multimodal":
                        inputs, inputs_eye, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        outputs = model(inputs, inputs_eye)
                    else:
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    test_preds.extend(preds.cpu().numpy())
                    test_targets.extend(targets.numpy())

            test_preds = np.array(test_preds)
            test_targets = np.array(test_targets)

            acc = float(np.mean(test_preds == test_targets))
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                test_targets, test_preds, average="macro", zero_division=0
            )
            labels_range = list(range(num_classes))
            p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
                test_targets, test_preds, average=None, zero_division=0, labels=labels_range
            )
            cm = confusion_matrix(test_targets, test_preds, labels=labels_range).tolist()

            per_class_metrics = {}
            for idx, cname in enumerate(class_names):
                per_class_metrics[cname] = {
                    "precision": round(float(p_per[idx]), 4),
                    "recall": round(float(r_per[idx]), 4),
                    "f1_score": round(float(f1_per[idx]), 4),
                    "support": int(sup_per[idx])
                }

            metrics_report = {
                "accuracy": round(acc, 4),
                "macro_precision": round(float(p_macro), 4),
                "macro_recall": round(float(r_macro), 4),
                "macro_f1": round(float(f1_macro), 4),
                "per_class": per_class_metrics,
                "confusion_matrix": cm,
                "classes": class_names,
                "loss_history": {
                    "train": [h["train_loss"] for h in history],
                    "val": [h["val_loss"] for h in history]
                },
                "accuracy_history": {
                    "train": [h["train_acc"] for h in history],
                    "val": [h["val_acc"] for h in history]
                },
                "split_info": split_info
            }

            with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_report, f, indent=2)

            elapsed = time.time() - start_time
            cls._current_status = TrainingStatusResponse(
                status="completed",
                current_epoch=len(history),
                total_epochs=config.epochs,
                train_loss=history[-1]["train_loss"],
                val_loss=history[-1]["val_loss"],
                train_acc=history[-1]["train_acc"],
                val_acc=history[-1]["val_acc"],
                best_val_acc=best_val_acc,
                elapsed_seconds=elapsed,
                history=history
            )

            return metrics_report

        except Exception as e:
            cls._current_status = TrainingStatusResponse(
                status="failed",
                error=str(e),
                elapsed_seconds=time.time() - start_time
            )
            raise e
        finally:
            cls._is_running = False
