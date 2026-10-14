#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import DatasetService

def main():
    parser = argparse.ArgumentParser(description="Prepare SEED-VII dataset features and labels.")
    parser.add_argument("--feature-type", type=str, default="de_LDS", choices=["de_LDS", "de", "psd"])
    args = parser.parse_args()

    print("=" * 75)
    print(f" NeuroGuide SEED-VII Dataset Preparation ({args.feature_type}) ")
    print("=" * 75)

    audit = DatasetService.generate_dataset_audit()
    if not audit["dataset_valid"]:
        print(f"\n[ERROR] Dataset audit failed: {audit['summary']}")
        sys.exit(1)

    print("\nAudit passed. Extracting EEG features, EYE features, and 7-class emotion labels...")
    X_eeg, X_eye, y_7, y_3, y_cont, groups, meta = DatasetService.process_and_save_dataset(
        feature_type=args.feature_type
    )

    print(f"\nSuccessfully processed and saved dataset to data/processed/dataset.npz")
    print(f"EEG Feature Matrix Shape: {X_eeg.shape}")
    print(f"Eye Feature Matrix Shape: {X_eye.shape}")
    print(f"7-Class Labels Vector   : {y_7.shape} (values: {set(y_7)})")
    print(f"Participants Represented: {len(set(groups))} subjects ({set(groups)})")
    print("=" * 75)

if __name__ == "__main__":
    main()
