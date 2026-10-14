from pathlib import Path
from scipy.io import loadmat

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

mat_files = sorted(DATA_DIR.glob("*.mat"))

if not mat_files:
    raise FileNotFoundError(f"No .mat files found in {DATA_DIR}")

for file_path in mat_files:
    print(f"\n{'=' * 60}")
    print(f"FILE: {file_path.name}")

    data = loadmat(file_path)
    keys = [k for k in data if not k.startswith("__")]

    print("Variable count:", len(keys))
    print("Possible label/metadata variables:")

    candidates = [
        k for k in keys
        if any(word in k.lower()
               for word in ["label", "emotion", "subject", "session", "trial"])
    ]
    print(candidates if candidates else "None found")

    print("\nFeature arrays:")
    feature_keys = [
        k for k in keys
        if k.startswith(("de_LDS_", "de_", "psd_"))
    ]

    for key in feature_keys[:10]:
        print(f"  {key}: {data[key].shape}")

    if len(feature_keys) > 10:
        print(f"  ... and {len(feature_keys) - 10} more arrays")

print("\nInspection complete.")