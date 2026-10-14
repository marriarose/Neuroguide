#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas import TrainingConfigRequest
from backend.app.services.training_service import TrainingService
from backend.app.services.dataset_service import DatasetService

def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on SEED-VII dataset.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--feature-type", type=str, default="de_LDS")
    parser.add_argument("--task-mode", type=str, default="7_class", choices=["7_class", "3_class", "regression"])
    parser.add_argument("--modality", type=str, default="eeg_only", choices=["eeg_only", "eeg_eye_multimodal"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 75)
    print(f" NeuroGuide Model Training (Task: {args.task_mode}, Modality: {args.modality}) ")
    print("=" * 75)

    config = TrainingConfigRequest(
        feature_type=args.feature_type,
        task_mode=args.task_mode,
        modality=args.modality,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        random_seed=args.seed
    )

    print("\nExecuting subject-independent training pipeline...")
    metrics = TrainingService.train_model_pipeline(config)

    print("\nTraining completed successfully!")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for cls_name, m in metrics["per_class"].items():
        print(f"  {cls_name:12s} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1_score']:.4f}")

    print("\nModel saved to models/eegnet_de_model.pt")
    print("Evaluation report saved to reports/metrics.json")
    print("=" * 75)

if __name__ == "__main__":
    main()
