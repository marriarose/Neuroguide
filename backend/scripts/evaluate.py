#!/usr/bin/env python3
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import REPORTS_DIR

def main():
    print("=" * 75)
    print(" NeuroGuide SEED-VII Model Evaluation Report ")
    print("=" * 75)

    metrics_file = REPORTS_DIR / "metrics.json"
    if not metrics_file.exists():
        print("\n[ERROR] No evaluation report found at reports/metrics.json.")
        print("Please train the model first using backend/scripts/train.py or via the Web UI.")
        sys.exit(1)

    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    print(f"\nOverall Test Accuracy : {m['accuracy'] * 100:.2f}%")
    print(f"Macro Precision       : {m['macro_precision']:.4f}")
    print(f"Macro Recall          : {m['macro_recall']:.4f}")
    print(f"Macro F1 Score        : {m['macro_f1']:.4f}")

    print("\nPer-Class Classification Report:")
    print("-" * 65)
    print(f"{'Class':14s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    print("-" * 65)
    for cname, p in m["per_class"].items():
        print(f"{cname:14s} | {p['precision']:10.4f} | {p['recall']:10.4f} | {p['f1_score']:10.4f} | {p['support']:<8d}")

    print("-" * 65)
    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    classes = m["classes"]
    header_str = f"{'':14s} " + " ".join([f"{c[:8]:>8s}" for c in classes])
    print(header_str)
    for idx, row in enumerate(m["confusion_matrix"]):
        row_str = f"{classes[idx]:14s} " + " ".join([f"{val:8d}" for val in row])
        print(row_str)

    print("\nSplit Info:", m.get("split_info", {}))
    print("=" * 75)

if __name__ == "__main__":
    main()
