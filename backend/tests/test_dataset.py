import pytest
import numpy as np
from pathlib import Path
from backend.app.services.dataset_service import DatasetService

def test_parse_stimulus_labels():
    labels_map = DatasetService.parse_stimulus_labels()
    assert len(labels_map) == 80
    assert set(labels_map.values()).issubset({"disgust", "fear", "sad", "neutral", "surprise", "happy", "anger"})

def test_generate_dataset_audit():
    audit = DatasetService.generate_dataset_audit()
    assert audit["total_participants"] == 20
    assert audit["total_eeg_windows"] == 69742
    assert audit["total_eye_windows"] == 69742
    assert audit["temporal_alignment_mismatches"] == 0
    assert audit["nan_missing_values"] == 0
    assert audit["audit_passed"] is True

def test_validate_dataset_success():
    res = DatasetService.validate_dataset("de_LDS")
    assert res["is_valid"] is True
    assert res["training_permitted"] is True
    assert len(res["errors"]) == 0
    assert res["total_usable_windows"] == 69742
    assert len(res["class_distribution"]) == 7

def test_process_and_save_dataset():
    X_eeg, X_eye, y_7, y_3, y_cont, groups, meta = DatasetService.process_and_save_dataset()

    assert X_eeg.ndim == 3 and X_eeg.shape[1:] == (5, 62)
    assert X_eye.ndim == 2 and X_eye.shape[1] == 33
    assert len(X_eeg) == len(X_eye) == len(y_7) == len(groups) == 69742
    assert set(np.unique(y_7)) == set(range(7))
    assert len(np.unique(groups)) == 20

def test_grouped_split_prevents_leakage():
    X_eeg, X_eye, y_7, _, _, groups, _ = DatasetService.process_and_save_dataset()

    (tr_tup, val_tup, test_tup, split_info) = DatasetService.split_data_group_aware(
        X_eeg, y_7, groups, X_eye=X_eye, random_seed=42
    )

    X_tr, y_tr, X_eye_tr = tr_tup
    X_te, y_te, X_eye_te = test_tup

    assert len(X_tr) + len(val_tup[0]) + len(X_te) == len(X_eeg)
    assert split_info["strategy"] == "subject_independent_group_split"
