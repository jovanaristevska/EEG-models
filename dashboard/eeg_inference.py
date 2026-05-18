"""
Standalone EEGPT inference script.
Run with eeg_env (numpy 1.26.4):
    D:/EEG-FM-Bench/eeg_env/Scripts/python.exe eeg_inference.py <csv_path>
"""
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

# Add EEG-FM-Bench to path
sys.path.insert(0, 'D:/EEG-FM-Bench')

def preprocess(csv_path):
    import pandas as pd
    import mne

    CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                'O1','O2','F7','F8','T7','T8','P7','P8','FZ','CZ','PZ']

    df = pd.read_csv(csv_path)
    col_map = {c.upper(): c for c in df.columns}
    missing = [ch for ch in CHANNELS if ch not in col_map]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    eeg_df = df[[col_map[ch] for ch in CHANNELS]]
    data = eeg_df.to_numpy(dtype=np.float64).T  # (19, n_timepoints)
    data = data * 1e-6  # µV → V  (match training pipeline)
    # NO unit conversion - model trained on µV

    info = mne.create_info(ch_names=CHANNELS, sfreq=128.0, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    raw.resample(256.0, verbose=False)
    raw.filter(l_freq=0.1, h_freq=100.0, verbose=False)

    data_resampled = raw.get_data()
    WINDOW_SAMPLES = 1024
    MAX_WINDOWS = 20
    n_total = data_resampled.shape[1]
    n_windows = min(n_total // WINDOW_SAMPLES, MAX_WINDOWS)
    trimmed = data_resampled[:, :n_windows * WINDOW_SAMPLES]
    windows = trimmed.reshape(19, n_windows, WINDOW_SAMPLES)
    windows_tensor = windows.transpose(1, 0, 2)  # (N, 19, 1024)
    return windows_tensor, n_windows


def load_model(path):
    import torch.nn as nn
    sys.path.insert(0, 'D:/EEG-FM-Bench')
    from baseline.eegpt.model import EEGTransformer
    from baseline.abstract.classifier import MultiHeadClassifier
    from baseline.eegpt.eegpt_config import EegptConfig

    cfg = EegptConfig()
    model_conf = cfg.model

    # Build encoder
    encoder = EEGTransformer(
        img_size=[64, 60 * 256],
        patch_size=model_conf.patch_size,
        patch_stride=model_conf.patch_stride,
        embed_num=model_conf.embed_num,
        embed_dim=model_conf.embed_dim,
        depth=model_conf.depth,
        num_heads=model_conf.num_heads,
        mlp_ratio=model_conf.mlp_ratio,
        drop_rate=model_conf.dropout_rate,
        attn_drop_rate=model_conf.attn_dropout_rate,
        drop_path_rate=model_conf.drop_path_rate,
        init_std=model_conf.init_std,
        qkv_bias=model_conf.qkv_bias,
        norm_layer=nn.LayerNorm,
    )

    # Load checkpoint
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Strip 'module.' prefix
    clean = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        clean[key] = v

    # Load encoder weights
    encoder_state = {k[8:]: v for k, v in clean.items() if k.startswith('encoder.')}
    missing, _ = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Encoder missing: {len(missing)} keys", file=sys.stderr)

    # Build classifier
    head_configs = {'adhd': 2}  # dataset_name: n_classes
    feature_dim = model_conf.embed_dim

    classifier = MultiHeadClassifier(
        embed_dim=feature_dim,
        head_configs=head_configs,
        head_cfg=cfg.model.classifier_head,
    )

    # Load classifier weights
    classifier_state = {k[11:]: v for k, v in clean.items() if k.startswith('classifier.')}
    classifier.load_state_dict(classifier_state, strict=False)

    class _Model(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, batch):
            x = batch['data']
            chan_ids = torch.arange(x.size(1), dtype=torch.long)
            features = self.encoder(x, chan_ids=chan_ids)
            return self.classifier(features, 'adhd')

    model = _Model(encoder, classifier)
    model.eval()
    return model


def predict(windows_tensor, model):
    tensor = torch.FloatTensor(windows_tensor)
    
    with torch.no_grad():
        batch = {'data': tensor}
        output = model(batch)
        probs = F.softmax(output, dim=1)
    
    probs_np = probs.numpy()
    predicted_classes = np.argmax(probs_np, axis=1)
    
    adhd_votes = int(np.sum(predicted_classes == 0))
    control_votes = int(np.sum(predicted_classes == 1))
    overall = "ADHD" if adhd_votes >= control_votes else "Control"
    
    winning_class_idx = 0 if overall == "ADHD" else 1
    confidence = float(np.mean(probs_np[:, winning_class_idx])) * 100
    reliability = "High" if confidence > 75 else "Medium" if confidence > 55 else "Low"
    
    window_predictions = [
        {
            "window": i + 1,
            "prediction": "ADHD" if cls == 0 else "Control",
            "confidence": round(float(probs_np[i, cls]), 3)
        }
        for i, cls in enumerate(predicted_classes)
    ]
    
    return {
        "prediction": overall,
        "confidence": round(confidence, 1),
        "reliability": reliability,
        "window_predictions": window_predictions,
        "windows_analyzed": len(predicted_classes)
    }


if __name__ == '__main__':
    csv_path = sys.argv[1]
    model_path = sys.argv[2]
    
    try:
        windows_tensor, n_windows = preprocess(csv_path)
        model = load_model(model_path)
        result = predict(windows_tensor, model)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)