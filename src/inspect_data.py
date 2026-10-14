
from pathlib import Path
from scipy.io import loadmat
import numpy as np

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Find MATLAB files
mat_files = list(DATA_DIR.glob("*.mat"))

if not mat_files:
    raise FileNotFoundError(
        f"No .mat files found in {DATA_DIR}\n"
        "Download a SEED preprocessed EEG file and place it here."
    )

# Inspect the first file
file_path = mat_files[0]
print(f"\nInspecting: {file_path.name}")

data = loadmat(file_path)

print("\nVariables in the MATLAB file:")
for key, value in data.items():
    if key.startswith("__"):
        continue

    if isinstance(value, np.ndarray):
        print(
            f"{key:15s} | "
            f"shape={value.shape} | "
            f"dtype={value.dtype}"
        )
    else:
        print(f"{key:15s} | type={type(value)}")

# Inspect emotion labels
if "labels" in data:
    labels = np.asarray(data["labels"]).squeeze()
    print("\nEmotion labels:")
    print(labels)

    print("\nUnique labels:", np.unique(labels))

# Inspect one EEG trial
if "eeg_1" in data:
    eeg = np.asarray(data["eeg_1"])
    print("\nFirst EEG trial:")
    print("Shape:", eeg.shape)
    print("Minimum:", np.nanmin(eeg))
    print("Maximum:", np.nanmax(eeg))
    print("Contains NaN:", np.isnan(eeg).any())
