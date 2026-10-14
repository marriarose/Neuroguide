# NeuroGuide: SEED-VII Multimodal EEG & Emotion Classification Platform

**NeuroGuide** is a full-stack AI research platform designed for **SEED-VII** multimodal EEG and eye-tracking emotion recognition across 20 participants.

The system classifies EEG-derived Differential Entropy (DE) and Linear Dynamical System (LDS) smoothed DE features into **7 discrete emotion categories**:
- **Disgust**
- **Fear**
- **Sad**
- **Neutral**
- **Surprise**
- **Happy**
- **Anger**

NeuroGuide features a PyTorch **EEGNet-inspired** convolutional neural network, a dual-branch **Multimodal Fusion Architecture** (EEG + Eye Features), participant-independent leakage-safe data splitting, a FastAPI REST API backend, and an interactive React Vite TypeScript dashboard frontend.

---

## 1. Verified SEED-VII Dataset Audit Summary

| Component | Audit Finding |
|---|---|
| **Raw Subdirectories** | `EEG_features/`, `EYE_features/`, `continuous_labels/`, `save_info/` |
| **Participants** | **20 Subjects** (`1.mat` through `20.mat`, confirmed in `subject info.xlsx`) |
| **Sessions & Trials** | **4 Sessions, 80 Trials total** (20 trials/session, defined in `emotion_label_and_stimuli_order.xlsx`) |
| **EEG Features** | **69,742 total windows** across 20 subjects (`de_LDS`, `de`, `psd` shape `(N, 5, 62)`) |
| **Eye Features** | **69,742 total windows** across 20 subjects (33 feature dimensions, shape `(N, 33)`) |
| **Temporal Alignment**| **100% 1-to-1 Window Alignment** (0 mismatches across all 20 subjects) |
| **Data Integrity** | **0 NaNs / 0 non-finite values** |
| **Discrete Emotion Labels** | **7 Classes**: Disgust (12 trials), Fear (12 trials), Sad (12 trials), Neutral (8 trials), Surprise (12 trials), Happy (12 trials), Anger (12 trials) |
| **Continuous Labels** | Dynamic self-report rating traces `(1, N)` per trial |

---

## 2. Architecture & Project Layout

```text
NeuroGuide/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point & CORS
│   │   ├── config.py             # SEED-VII paths, 7 emotion classes, hyperparams
│   │   ├── schemas.py            # Pydantic schemas & audit report models
│   │   ├── api/                  # API routers (dataset, training, prediction, evaluation)
│   │   ├── services/             # DatasetService (audit, parser, grouped split), TrainingService, InferenceService
│   │   └── models/
│   │       └── eegnet_de.py      # EEGNet & Multimodal Fusion PyTorch ConvNet
│   ├── scripts/                  # CLI tools
│   │   ├── inspect_data.py       # SEED-VII audit & inspection CLI
│   │   ├── prepare_data.py       # SEED-VII feature extraction & dataset.npz preparation
│   │   ├── train.py              # Subject-independent training CLI
│   │   └── evaluate.py           # 7-Class evaluation & 7x7 confusion matrix CLI
│   ├── tests/                    # PyTest test suite (12 tests)
│   └── requirements.txt
├── frontend/                     # React Vite TypeScript Dashboard
│   ├── src/
│   │   ├── components/           # Navbar, StatCard
│   │   ├── pages/                # Overview, DatasetExplorer, Training, Evaluation, Prediction, About
│   │   ├── services/             # Typed API client
│   │   └── App.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── data/
│   ├── raw/                      # EEG_features/, EYE_features/, continuous_labels/, save_info/
│   ├── metadata/                 # SEED-VII Excel spreadsheets & channel locs
│   └── processed/                # dataset.npz & dataset_audit_report.json
├── models/                       # Checkpoints (eegnet_de_model.pt) & scaler.npz
├── reports/                      # Evaluation metrics (metrics.json)
└── README.md
```

---

## 3. Running the Full-Stack Application

### 1. Backend Setup & Startup (Port 8000)
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Set PYTHONPATH and start FastAPI server
$env:PYTHONPATH="."
.venv\Scripts\uvicorn backend.app.main:app --reload --port 8000
```
- Interactive API Swagger Docs: `http://localhost:8000/docs`

### 2. Frontend Setup & Startup (Port 5173)
```powershell
cd frontend
npm run dev
```
- Open `http://localhost:5173` in your browser.

---

## 4. CLI Audit, Preparation, & Training

- **Run SEED-VII Dataset Audit**:
  ```powershell
  $env:PYTHONPATH="."
  python backend/scripts/inspect_data.py
  ```

- **Prepare Dataset Features**:
  ```powershell
  $env:PYTHONPATH="."
  python backend/scripts/prepare_data.py --feature-type de_LDS
  ```

- **Train 7-Class EEGNet Model (Participant-Independent Split)**:
  ```powershell
  $env:PYTHONPATH="."
  python backend/scripts/train.py --task-mode 7_class --modality eeg_only --epochs 35
  ```

- **Train Multimodal EEG + Eye Fusion Model**:
  ```powershell
  $env:PYTHONPATH="."
  python backend/scripts/train.py --task-mode 7_class --modality eeg_eye_multimodal --epochs 35
  ```

- **Generate 7x7 Evaluation Report**:
  ```powershell
  $env:PYTHONPATH="."
  python backend/scripts/evaluate.py
  ```

---

## 5. Automated Unit & Integration Tests

Run the complete PyTest test suite (12 tests covering SEED-VII audit, parsing, grouped splitting, PyTorch forward passes, multimodal fusion, and API endpoints):
```powershell
$env:PYTHONPATH="."
.venv\Scripts\pytest backend/tests -v
```

---

## 6. Academic Disclaimer

Model predictions are produced strictly for academic AI research demonstration on the SEED-VII dataset and do not constitute a medical or clinical psychological diagnosis.
