import logging
import random
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Label mapping: index 0 = ADHD, index 1 = Control
LABELS = {0: 'ADHD', 1: 'Control'}


class InferenceEngine:

    def predict(self, windows_tensor, model, model_type: str) -> dict:
        if model is None:
            # Mock mode
            n_windows = windows_tensor.shape[0]
            confidences = np.random.uniform(0.55, 0.85, n_windows)
            predictions = (confidences > 0.65).astype(int)
            mean_conf = float(np.mean(confidences))
            overall = int(np.round(np.mean(predictions)))
            return {
                "prediction": "ADHD" if overall == 0 else "Control",
                "confidence": round(mean_conf * 100, 1),
                "reliability": "High" if mean_conf > 0.75 else "Medium",
                "window_predictions": [
                    {
                        "window": i + 1,
                        "prediction": "ADHD" if p == 0 else "Control",
                        "confidence": round(float(c), 3)
                    }
                    for i, (p, c) in enumerate(zip(predictions, confidences))
                ]
            }

        # BATCH inference - all windows at once
        tensor = torch.FloatTensor(windows_tensor)  # (N, 19, 1024)
        
        with torch.no_grad():
            if model_type == 'eegpt':
                batch = {'data': tensor * 0.001} 
                output = model(batch)
            elif model_type == 'neurogpt':
                output = model({'data': tensor * 0.001})
            else:
                # EEGNet expects tensor directly
                output = model(tensor)
            
            # Get probabilities
            probs = F.softmax(output, dim=1)  # (N, 2)
            
        probs_np = probs.numpy()
        predicted_classes = np.argmax(probs_np, axis=1)  # (N,)
        
        # Majority vote for overall prediction
        adhd_votes = int(np.sum(predicted_classes == 0))
        control_votes = int(np.sum(predicted_classes == 1))
        overall = "ADHD" if adhd_votes >= control_votes else "Control"
        
        # Confidence = mean probability of winning class
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
            "window_predictions": window_predictions
        }
   
    # def _real_predict(self, windows_tensor, model, model_type, n_windows):
    #     tensor = torch.FloatTensor(windows_tensor)  # (N, 19, 1024)

    #     window_preds = []
    #     all_probs = []

    #     with torch.no_grad():
    #         # Process in mini-batches of 8 to avoid OOM
    #         batch_size = 8
    #         for start in range(0, n_windows, batch_size):
    #             batch = tensor[start:start + batch_size]

    #             if model_type == 'eegpt':
    #                 chan_ids = list(range(batch.size(1)))
    #                 logits = model(batch, chan_ids=chan_ids)
    #             else:
    #                 logits = model(batch)

    #             probs = F.softmax(logits, dim=1)  # (B, 2)
    #             all_probs.append(probs.cpu().numpy())

    #     all_probs = np.concatenate(all_probs, axis=0)  # (N, 2)

    #     for i, probs in enumerate(all_probs):
    #         pred_idx = int(np.argmax(probs))
    #         window_preds.append({
    #             'window': i + 1,
    #             'prediction': LABELS[pred_idx],
    #             'confidence': round(float(probs[pred_idx]), 4),
    #         })

    #     # Majority vote
    #     votes = [p['prediction'] for p in window_preds]
    #     adhd_votes = votes.count('ADHD')
    #     overall_label = 'ADHD' if adhd_votes >= n_windows / 2 else 'Control'
    #     label_idx = 0 if overall_label == 'ADHD' else 1
    #     mean_conf = float(all_probs[:, label_idx].mean()) * 100

    #     return {
    #         'prediction': overall_label,
    #         'confidence': round(mean_conf, 1),
    #         'reliability': _reliability(mean_conf),
    #         'window_predictions': window_preds,
    #     }

    # # ── mock inference ────────────────────────────────────────────────────────

    # def _mock_predict(self, n_windows: int, model_type: str) -> dict:
    #     rng = random.Random()
    #     overall_adhd_prob = rng.uniform(0.55, 0.85)
    #     overall_label = 'ADHD' if overall_adhd_prob > 0.5 else 'Control'
    #     confidence = overall_adhd_prob * 100 if overall_label == 'ADHD' else (1 - overall_adhd_prob) * 100

    #     window_preds = []
    #     for i in range(n_windows):
    #         p = rng.gauss(overall_adhd_prob, 0.1)
    #         p = max(0.05, min(0.95, p))
    #         label = 'ADHD' if p > 0.5 else 'Control'
    #         window_preds.append({
    #             'window': i + 1,
    #             'prediction': label,
    #             'confidence': round(p if label == 'ADHD' else 1 - p, 4),
    #         })

    #     return {
    #         'prediction': overall_label,
    #         'confidence': round(confidence, 1),
    #         'reliability': _reliability(confidence),
    #         'window_predictions': window_preds,
    #     }
