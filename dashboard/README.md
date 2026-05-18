# EEG Analysis Dashboard

Local full-stack dashboard for EEG ADHD classification using EEG-GPT and EEGNet.

## Setup

### 1. Copy model files
```
dashboard/models/eegpt_unified_last.pt
dashboard/models/eegnet_adhd_last.pt
```
If either file is missing the backend runs in **Mock Mode** for that model and returns
realistic simulated predictions so the frontend still works.

### 2. Install dependencies
```bash
cd D:/EEG-FM-Bench/dashboard
pip install -r requirements.txt
```

### 3. Start the backend
```bash
cd D:/EEG-FM-Bench/dashboard
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the frontend
Navigate to **http://localhost:8000** in your browser.
(The backend serves `frontend/index.html` at `/`.)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server & model status |
| GET | `/model_info/{model_type}` | Metadata for `eegpt` or `eegnet` |
| POST | `/predict` | Run EEG analysis (multipart/form-data) |

### POST /predict — request body
| Field | Type | Description |
|-------|------|-------------|
| `file` | file | CSV with 19 EEG channels |
| `model_type` | string | `eegpt` or `eegnet` |

---

## Supported CSV Format

The CSV must contain (case-insensitive) column headers for all 19 channels:
```
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz
```
Additional columns (e.g. `Class`, `ID`) are ignored automatically.

---

## Preprocessing Pipeline

1. Read CSV → extract 19 EEG channels (case-insensitive match)
2. Convert µV → V (`× 1e-6`)
3. MNE RawArray at 128 Hz
4. Notch filter at 50 Hz
5. Bandpass filter 0.1–100 Hz
6. Resample to 256 Hz
7. Slice into 4-second (1024-sample) non-overlapping windows
8. Output: `(N_windows, 19, 1024)` tensor

---

## Notes

- Models trained on ADHD dataset: 61 ADHD subjects, 60 Control subjects
- EEGPT: foundation transformer pretrained on large EEG corpus
- EEGNet: lightweight depthwise CNN (braindecode implementation)
- Band powers computed via Welch method (scipy.signal.welch)
