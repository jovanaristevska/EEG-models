import logging
import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make EEG-FM-Bench importable
BENCH_PATH = 'D:/EEG-FM-Bench'
if BENCH_PATH not in sys.path:
    sys.path.insert(0, BENCH_PATH)

from backend.model_loader import load_eegpt, load_eegnet
from backend.preprocessor import PreprocessorPipeline
from backend.inference import InferenceEngine

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
EEGPT_PATH = os.path.join(MODELS_DIR, 'eegpt_adhd_last.pt')
EEGNET_PATH = os.path.join(MODELS_DIR, 'eegnet_adhd_last.pt')

# ── model loading ─────────────────────────────────────────────────────────────
logger.info("Loading models...")
eegpt_model = load_eegpt(EEGPT_PATH) if os.path.exists(EEGPT_PATH) else None
eegnet_model = load_eegnet(EEGNET_PATH) if os.path.exists(EEGNET_PATH) else None

if eegpt_model is None:
    logger.warning("EEGPT model not loaded — running in MOCK mode for EEGPT")
if eegnet_model is None:
    logger.warning("EEGNet model not loaded — running in MOCK mode for EEGNet")

MOCK_MODE = eegpt_model is None and eegnet_model is None

preprocessor = PreprocessorPipeline()
engine = InferenceEngine()

# ── model metadata ────────────────────────────────────────────────────────────
MODEL_INFO = {
    'eegpt': {
        'name': 'EEG-GPT',
        'accuracy': 0.873,
        'auroc': 0.896,
        'balanced_acc': 0.881,
        'auc_pr': 0.954,
        'epochs': 50,
        'batch_size': 32,
        'description': 'Foundation model pretrained on large EEG corpus',
    },
    'eegnet': {
        'name': 'EEGNet',
        'accuracy': 0.751,
        'auroc': 0.861,
        'balanced_acc': 0.764,
        'auc_pr': 0.720,
        'epochs': 50,
        'batch_size': 1024,
        'description': 'Lightweight CNN architecture for EEG classification',
    },
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title='EEG Analysis Dashboard API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve frontend at root
@app.get('/', include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    raise HTTPException(status_code=404, detail='Frontend not found')


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'models_loaded': {
            'eegpt': eegpt_model is not None,
            'eegnet': eegnet_model is not None,
        },
        'mock_mode': MOCK_MODE,
    }


@app.get('/model_info/{model_type}')
async def model_info(model_type: str):
    if model_type not in MODEL_INFO:
        raise HTTPException(status_code=404, detail=f'Unknown model: {model_type}')
    return MODEL_INFO[model_type]


@app.post('/predict')
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...),
):
    if model_type not in ('eegpt', 'eegnet'):
        raise HTTPException(status_code=400,
                            detail=f"model_type must be 'eegpt' or 'eegnet', got '{model_type}'")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Preprocess
        try:
            preproc_result = preprocessor.preprocess(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

        windows_tensor = preproc_result['windows_tensor']
        eeg_signal = preproc_result['eeg_signal']
        band_powers = preproc_result['band_powers']
        n_windows = preproc_result['n_windows']

        # Select model
        model = eegpt_model if model_type == 'eegpt' else eegnet_model
        mock_this = model is None

        # Run inference
        try:
            result = engine.predict(windows_tensor, model, model_type)
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f'Inference error: {exc}')

        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'reliability': result['reliability'],
            'window_predictions': result['window_predictions'],
            'eeg_signal': eeg_signal,
            'band_powers': band_powers,
            'model_used': model_type,
            'windows_analyzed': n_windows,
            'mock_mode': mock_this,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
