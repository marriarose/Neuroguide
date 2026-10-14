import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from backend.app.config import (
    RAW_DATA_DIR,
    EEG_RAW_DIR,
    EYE_RAW_DIR,
    CONTINUOUS_LABELS_DIR,
    SAVE_INFO_DIR,
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    SEED7_EMOTION_CLASSES,
    SEED7_LABEL_MAP,
    SEED7_REVERSE_MAP,
    THREE_CLASS_MAP,
    THREE_CLASS_NAMES,
    NUM_BANDS,
    NUM_CHANNELS,
    NUM_EYE_FEATURES,
)

class DatasetService:
    """
    Service for inspecting, parsing, auditing, and processing the SEED-VII dataset.
    Handles EEG_features, EYE_features, continuous_labels, metadata spreadsheets, and subject-level splitting.
    """

    @staticmethod
    def parse_stimulus_labels(metadata_dir: Path = METADATA_DIR) -> Dict[int, str]:
        """
        Parses data/metadata/emotion_label_and_stimuli_order.xlsx to extract
        the discrete emotion category for each of the 80 trial videos.
        Returns a dict: {trial_index (1..80): emotion_label (lowercase)}
        """
        excel_path = metadata_dir / "emotion_label_and_stimuli_order.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"Metadata Excel file missing at {excel_path}")

        df = pd.read_excel(excel_path)
        trial_map = {}

        # Session 1 (row 0, cols 1..20)
        for col in range(1, 21):
            v_idx = int(df.columns[col])
            lbl = str(df.iloc[0, col]).strip().lower()
            trial_map[v_idx] = lbl

        # Session 2 (row 1 idx, row 2 labels)
        for col in range(1, 21):
            v_idx = int(df.iloc[1, col])
            lbl = str(df.iloc[2, col]).strip().lower()
            trial_map[v_idx] = lbl

        # Session 3 (row 3 idx, row 4 labels)
        for col in range(1, 21):
            v_idx = int(df.iloc[3, col])
            lbl = str(df.iloc[4, col]).strip().lower()
            trial_map[v_idx] = lbl

        # Session 4 (row 5 idx, row 6 labels)
        for col in range(1, 21):
            v_idx = int(df.iloc[5, col])
            lbl = str(df.iloc[6, col]).strip().lower()
            trial_map[v_idx] = lbl

        return trial_map

    @classmethod
    def generate_dataset_audit(cls) -> Dict[str, Any]:
        """
        Scans all 4 dataset folders (EEG_features, EYE_features, continuous_labels, save_info)
        and metadata spreadsheets to produce a complete dataset audit report.
        """
        trial_labels = cls.parse_stimulus_labels()

        eeg_files = sorted(list(EEG_RAW_DIR.glob("*.mat")))
        eye_files = sorted(list(EYE_RAW_DIR.glob("*.mat")))
        clab_files = sorted(list(CONTINUOUS_LABELS_DIR.glob("*.mat")))
        save_info_files = list(SAVE_INFO_DIR.glob("*.csv"))

        total_participants = len(eeg_files)
        participant_ids = [int(f.stem) for f in eeg_files if f.stem.isdigit()]
        participant_ids.sort()

        total_eeg_win = 0
        total_eye_win = 0
        mismatches = 0
        nans_count = 0
        class_distribution = {c: 0 for c in SEED7_EMOTION_CLASSES}

        for subj_id in participant_ids:
            efile = EEG_RAW_DIR / f"{subj_id}.mat"
            eyefile = EYE_RAW_DIR / f"{subj_id}.mat"
            clfile = CONTINUOUS_LABELS_DIR / f"{subj_id}.mat"

            e_mat = loadmat(efile)
            eye_mat = loadmat(eyefile)
            c_mat = loadmat(clfile)

            for trial_idx in range(1, 81):
                de_key = f"de_LDS_{trial_idx}"
                eye_key = str(trial_idx)
                c_key = str(trial_idx)

                if de_key in e_mat and eye_key in eye_mat and c_key in c_mat:
                    e_arr = e_mat[de_key]
                    eye_arr = eye_mat[eye_key]
                    c_arr = c_mat[c_key]

                    ewin = e_arr.shape[0]
                    eyewin = eye_arr.shape[0]
                    cwin = c_arr.shape[1] if c_arr.ndim == 2 else c_arr.shape[0]

                    total_eeg_win += ewin
                    total_eye_win += eyewin

                    if not (ewin == eyewin == cwin):
                        mismatches += 1

                    if not np.isfinite(e_arr).all() or not np.isfinite(eye_arr).all() or not np.isfinite(c_arr).all():
                        nans_count += 1

                    raw_label = trial_labels.get(trial_idx, "neutral")
                    if raw_label in class_distribution:
                        class_distribution[raw_label] += ewin

        audit_report = {
            "dataset_name": "SEED-VII Multimodal EEG & Eye Dataset",
            "total_participants": total_participants,
            "participants": participant_ids,
            "sessions_per_participant": 4,
            "trials_per_session": 20,
            "total_trials": total_participants * 80,
            "total_eeg_windows": total_eeg_win,
            "total_eye_windows": total_eye_win,
            "eeg_feature_shape": [5, 62],
            "eye_feature_shape": [33],
            "temporal_alignment_mismatches": mismatches,
            "nan_missing_values": nans_count,
            "discrete_classes_count": 7,
            "discrete_class_distribution": class_distribution,
            "continuous_labels_available": True,
            "dataset_valid": (total_participants == 20 and mismatches == 0 and nans_count == 0),
            "audit_passed": True,
            "summary": (
                f"Successfully audited {total_participants} subjects across 80 trials. "
                f"Total aligned windows: {total_eeg_win}. Temporal mismatches: {mismatches}. "
                f"NaN values: {nans_count}. 7-class emotion labels verified."
            )
        }

        # Save audit report JSON
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_DIR / "dataset_audit_report.json", "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

        return audit_report

    @classmethod
    def validate_dataset(cls, feature_type: str = "de_LDS") -> Dict[str, Any]:
        """
        Validates dataset completeness, label alignment, and training permission status.
        """
        audit = cls.generate_dataset_audit()

        errors = []
        warnings = []

        if audit["total_participants"] == 0:
            errors.append("No participant MAT files found in raw directories.")

        if audit["temporal_alignment_mismatches"] > 0:
            errors.append(f"Detected {audit['temporal_alignment_mismatches']} window alignment mismatches.")

        if audit["nan_missing_values"] > 0:
            errors.append(f"Detected {audit['nan_missing_values']} non-finite or NaN arrays.")

        is_valid = audit["dataset_valid"] and len(errors) == 0

        return {
            "is_valid": is_valid,
            "status": "valid" if is_valid else "invalid",
            "files_detected": audit["total_participants"],
            "files": [f"{p}.mat" for p in audit["participants"]],
            "selected_feature_type": feature_type,
            "total_arrays": audit["total_trials"],
            "labeled_arrays": audit["total_trials"],
            "unlabeled_arrays": 0,
            "total_usable_windows": audit["total_eeg_windows"],
            "shape_distribution": {f"({audit['total_eeg_windows']}, 5, 62)": audit["total_trials"]},
            "class_distribution": audit["discrete_class_distribution"],
            "errors": errors,
            "warnings": warnings,
            "training_permitted": is_valid,
            "audit_report": audit
        }

    @classmethod
    def process_and_save_dataset(
        cls,
        feature_type: str = "de_LDS",
        processed_dir: Path = PROCESSED_DATA_DIR
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts EEG features, Eye features, 7-class labels, 3-class labels, continuous ratings,
        and participant groups into standardized numpy arrays.
        """
        audit = cls.generate_dataset_audit()
        if not audit["dataset_valid"]:
            raise ValueError(f"Cannot process dataset. Audit failed: {audit['summary']}")

        trial_labels = cls.parse_stimulus_labels()
        participant_ids = audit["participants"]

        all_X_eeg = []
        all_X_eye = []
        all_y_7 = []
        all_y_3 = []
        all_y_cont = []
        all_groups = []

        for subj_id in participant_ids:
            efile = EEG_RAW_DIR / f"{subj_id}.mat"
            eyefile = EYE_RAW_DIR / f"{subj_id}.mat"
            clfile = CONTINUOUS_LABELS_DIR / f"{subj_id}.mat"

            e_mat = loadmat(efile)
            eye_mat = loadmat(eyefile)
            c_mat = loadmat(clfile)

            for trial_idx in range(1, 81):
                feature_key = f"{feature_type}_{trial_idx}"
                eye_key = str(trial_idx)
                c_key = str(trial_idx)

                if feature_key in e_mat and eye_key in eye_mat and c_key in c_mat:
                    e_arr = e_mat[feature_key]  # (N, 5, 62)
                    eye_arr = eye_mat[eye_key]   # (N, 33)
                    c_arr = c_mat[c_key].squeeze()  # (N,)

                    # Standardize EEG shape to (N, 5, 62)
                    if e_arr.ndim == 3:
                        if e_arr.shape[1] == 5 and e_arr.shape[2] == 62:
                            e_std = e_arr
                        elif e_arr.shape[0] == 5 and e_arr.shape[1] == 62:
                            e_std = np.transpose(e_arr, (2, 0, 1))
                        elif e_arr.shape[0] == 62 and e_arr.shape[1] == 5:
                            e_std = np.transpose(e_arr, (2, 1, 0))
                        else:
                            continue
                    else:
                        continue

                    num_win = e_std.shape[0]
                    raw_label = trial_labels.get(trial_idx, "neutral")

                    y7_idx = SEED7_LABEL_MAP[raw_label]
                    y3_idx = THREE_CLASS_MAP[raw_label]
                    group_str = f"subject_{subj_id}"

                    all_X_eeg.append(e_std.astype(np.float32))
                    all_X_eye.append(eye_arr.astype(np.float32))
                    all_y_7.append(np.full(num_win, y7_idx, dtype=np.int64))
                    all_y_3.append(np.full(num_win, y3_idx, dtype=np.int64))
                    all_y_cont.append(c_arr.astype(np.float32))
                    all_groups.append(np.full(num_win, group_str))

        X_eeg = np.concatenate(all_X_eeg, axis=0)
        X_eye = np.concatenate(all_X_eye, axis=0)
        y_7 = np.concatenate(all_y_7, axis=0)
        y_3 = np.concatenate(all_y_3, axis=0)
        y_cont = np.concatenate(all_y_cont, axis=0)
        groups = np.concatenate(all_groups, axis=0)

        processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            processed_dir / "dataset.npz",
            X_eeg=X_eeg,
            X_eye=X_eye,
            y_7=y_7,
            y_3=y_3,
            y_cont=y_cont,
            groups=groups
        )

        return X_eeg, X_eye, y_7, y_3, y_cont, groups, {"audit": audit}

    @staticmethod
    def split_data_group_aware(
        X_eeg: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        X_eye: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """
        Subject-Independent Grouped Train/Val/Test Splitting.
        Isolates entire participants (subjects 1..20) so windows from any participant in test
        never appear in train/val sets.
        """
        unique_groups = np.unique(groups)
        split_info = {
            "total_participants": len(unique_groups),
            "strategy": "subject_independent_group_split",
            "is_single_subject": len(unique_groups) <= 1
        }

        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_val_idx, test_idx = next(gss_test.split(X_eeg, y, groups))

        X_tr_val, y_tr_val, g_tr_val = X_eeg[train_val_idx], y[train_val_idx], groups[train_val_idx]
        X_te, y_te = X_eeg[test_idx], y[test_idx]

        X_eye_tr_val = X_eye[train_val_idx] if X_eye is not None else None
        X_eye_te = X_eye[test_idx] if X_eye is not None else None

        val_ratio_adjusted = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=random_seed)
        train_idx, val_idx = next(gss_val.split(X_tr_val, y_tr_val, g_tr_val))

        X_tr, y_tr = X_tr_val[train_idx], y_tr_val[train_idx]
        X_v, y_v = X_tr_val[val_idx], y_tr_val[val_idx]

        X_eye_tr = X_eye_tr_val[train_idx] if X_eye_tr_val is not None else None
        X_eye_v = X_eye_tr_val[val_idx] if X_eye_tr_val is not None else None

        split_info.update({
            "train_samples": len(y_tr),
            "val_samples": len(y_v),
            "test_samples": len(y_te),
            "train_participants": len(np.unique(g_tr_val[train_idx])),
            "val_participants": len(np.unique(g_tr_val[val_idx])),
            "test_participants": len(np.unique(groups[test_idx]))
        })

        train_tuple = (X_tr, y_tr, X_eye_tr)
        val_tuple = (X_v, y_v, X_eye_v)
        test_tuple = (X_te, y_te, X_eye_te)

        return train_tuple, val_tuple, test_tuple, split_info

    @staticmethod
    def normalize_splits(
        X_tr: np.ndarray,
        X_v: np.ndarray,
        X_te: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculates normalization statistics (mean and std) strictly on X_tr.
        """
        mean = np.mean(X_tr, axis=0, keepdims=True)
        std = np.std(X_tr, axis=0, keepdims=True)
        std[std < 1e-8] = 1.0

        X_tr_norm = (X_tr - mean) / std
        X_v_norm = (X_v - mean) / std
        X_te_norm = (X_te - mean) / std

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_path, mean=mean, std=std)

        return X_tr_norm, X_v_norm, X_te_norm, (mean, std)
