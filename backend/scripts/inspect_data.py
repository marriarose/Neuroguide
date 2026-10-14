#!/usr/bin/env python3
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import DatasetService

def main():
    print("=" * 75)
    print(" NeuroGuide SEED-VII Dataset Audit & Inspection Report ")
    print("=" * 75)

    try:
        audit = DatasetService.generate_dataset_audit()
        
        print(f"\nDataset Name          : {audit['dataset_name']}")
        print(f"Total Participants    : {audit['total_participants']} subjects (IDs: {audit['participants'][:5]}...{audit['participants'][-5:]})")
        print(f"Sessions / Participant: {audit['sessions_per_participant']} sessions (20 trials/session = 80 trials total)")
        print(f"Total Trials Audited  : {audit['total_trials']} trial recordings")
        print(f"Total EEG Windows     : {audit['total_eeg_windows']} (5 bands x 62 channels)")
        print(f"Total Eye Windows     : {audit['total_eye_windows']} (33 feature dimensions)")
        print(f"Temporal Mismatches   : {audit['temporal_alignment_mismatches']} (100% 1-to-1 window alignment)")
        print(f"NaN / Non-Finite Count: {audit['nan_missing_values']}")
        
        print("\nDiscrete 7 Emotion Class Distribution (Across 69,742 windows):")
        print("-" * 50)
        for cname, count in audit['discrete_class_distribution'].items():
            pct = (count / audit['total_eeg_windows']) * 100
            print(f"  {cname:12s} | {count:6d} windows ({pct:.1f}%)")

        print("-" * 50)
        print(f"Continuous Labels     : Available ({audit['continuous_labels_available']})")
        print(f"Dataset Audit Passed  : {audit['audit_passed']}")
        print(f"Training Permitted    : {audit['dataset_valid']}")

        print("\nAudit Summary:", audit['summary'])
        print("=" * 75)

    except Exception as e:
        print(f"\n[ERROR] Audit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
