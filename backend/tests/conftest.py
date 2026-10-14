import pytest
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import savemat

@pytest.fixture
def temp_seed7_dataset_dir():
    """
    Creates a temporary directory structure mimicking SEED-VII dataset layout
    for isolated unit testing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw"
        meta_path = Path(tmpdir) / "metadata"
        
        eeg_dir = raw_path / "EEG_features"
        eye_dir = raw_path / "EYE_features"
        clab_dir = raw_path / "continuous_labels"
        save_info_dir = raw_path / "save_info"

        eeg_dir.mkdir(parents=True, exist_ok=True)
        eye_dir.mkdir(parents=True, exist_ok=True)
        clab_dir.mkdir(parents=True, exist_ok=True)
        save_info_dir.mkdir(parents=True, exist_ok=True)
        meta_path.mkdir(parents=True, exist_ok=True)

        # Create 3 synthetic participant files
        for subj in [1, 2, 3]:
            eeg_dict = {}
            eye_dict = {}
            clab_dict = {}

            for trial in range(1, 81):
                n_win = 10 + (trial % 5)
                eeg_dict[f"de_LDS_{trial}"] = np.random.randn(n_win, 5, 62).astype(np.float32)
                eye_dict[str(trial)] = np.random.randn(n_win, 33).astype(np.float32)
                clab_dict[str(trial)] = np.random.uniform(0, 20, (1, n_win)).astype(np.float64)

            savemat(eeg_dir / f"{subj}.mat", eeg_dict)
            savemat(eye_dir / f"{subj}.mat", eye_dict)
            savemat(clab_dir / f"{subj}.mat", clab_dict)

        yield {
            "root": Path(tmpdir),
            "raw_dir": raw_path,
            "metadata_dir": meta_path,
            "eeg_dir": eeg_dir,
            "eye_dir": eye_dir,
            "clab_dir": clab_dir,
        }
