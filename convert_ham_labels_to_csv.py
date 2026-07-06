"""
Convert DASPS_HAM_labels.mat to 276 per-trial CSV files.

Structure of DASPS_HAM_labels.mat:
  Regim_datasub
    ├── HAM     : (276,) — HAM-A score per trial (subject-level, replicated)
    ├── label   : list of 276 — SAM [Valence, Arousal] per trial
    └── trial   : list of 276 — EEG signal (14, 1920) per trial

Output CSV columns:
  AF3, AF4, F3, F4, FC5, FC6, F7, F8, T7, T8, P7, P8, O1, O2
  
Filename format:
  S{subject}t{trial}.csv  (e.g., S01t01.csv, S23t12.csv)

Metadata saved to summary/finetune/anxiety_v2_finetune_info.csv:
  subject, original_subject, trial_in_subject, valence, arousal, ham_a, label
"""

import mat73
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MAT_FILE = r"D:\EEG-FM-Bench\assets\data\raw\Anxiety\DASPS_HAM_labels.mat"
OUTPUT_ROOT = Path(r"D:\EEG-FM-Bench\assets\data\raw\Anxiety_v2")
SUBJECTS_DIR = OUTPUT_ROOT / "subjects"
SUMMARY_DIR = OUTPUT_ROOT / "summary" / "finetune"

# Emotiv EPOC+ channels (fixed order from dataset)
CHANNEL_NAMES = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8',
                 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

# Labelling threshold (SAM Arousal + Valence)
AROUSAL_THRESHOLD = 5  # arousal > 5 → high arousal
VALENCE_THRESHOLD = 5  # valence < 5 → negative valence

# ============================================================================
# LOAD MAT FILE
# ============================================================================

print(f"Loading: {MAT_FILE}")
data = mat73.loadmat(MAT_FILE)
regim = data['Regim_datasub']

n_trials = len(regim['HAM'])
print(f"Total trials: {n_trials}")
print(f"Assuming 23 subjects × 12 trials = {23 * 12}")

# ============================================================================
# CREATE OUTPUT FOLDERS
# ============================================================================

SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nOutput folder: {OUTPUT_ROOT}")

# ============================================================================
# CONVERSION
# ============================================================================

metadata_rows = []

for global_idx in range(n_trials):
    # Секој subject има 12 trials
    subject_num = (global_idx // 12) + 1        # 1..23
    trial_in_subject = (global_idx % 12) + 1    # 1..12
    
    subject_id = f"S{subject_num:02d}"
    trial_id = f"{subject_id}t{trial_in_subject:02d}"
    
    # EEG signal
    trial_data = regim['trial'][global_idx]
    # trial_data е list со 1 element, а тој element е ndarray (14, 1920)
    if isinstance(trial_data, list):
        signal = trial_data[0]
    else:
        signal = trial_data
    
    # SAM ratings [Valence, Arousal]
    label_data = regim['label'][global_idx]
    if isinstance(label_data, list):
        sam = label_data[0]
    else:
        sam = label_data
    valence = float(sam[0])
    arousal = float(sam[1])
    
    # HAM-A (subject-level)
    ham_a = float(regim['HAM'][global_idx])
    
    # Определи label
    if arousal > AROUSAL_THRESHOLD and valence < VALENCE_THRESHOLD:
        label = "Anxiety"
    else:
        label = "Control"
    
    # Transpose signal to (samples, channels)
    signal_t = signal.T  # (1920, 14)
    
    # Save as CSV
    df = pd.DataFrame(signal_t, columns=CHANNEL_NAMES)
    csv_path = SUBJECTS_DIR / f"{trial_id}.csv"
    df.to_csv(csv_path, index=False)
    
    metadata_rows.append({
        'subject': trial_id,
        'original_subject': subject_id,
        'trial_in_subject': trial_in_subject,
        'phase': 'recitation' if trial_in_subject <= 6 else 'recall',
        'valence': valence,
        'arousal': arousal,
        'ham_a': ham_a,
        'label': label
    })
    
    if global_idx % 20 == 0:
        print(f"  Processed {global_idx+1}/{n_trials}: {trial_id} "
              f"→ V={valence:.0f}, A={arousal:.0f}, HAM={ham_a:.0f}, "
              f"label={label}")

# ============================================================================
# SAVE METADATA
# ============================================================================

meta_df = pd.DataFrame(metadata_rows)
meta_path = SUMMARY_DIR / "anxiety_v2_finetune_info.csv"
meta_df.to_csv(meta_path, index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total trials converted: {len(metadata_rows)}")
print(f"Metadata saved to: {meta_path}")

# Class distribution
print(f"\nClass distribution:")
print(meta_df['label'].value_counts())

# Class distribution by phase
print(f"\nClass distribution by phase:")
print(pd.crosstab(meta_df['phase'], meta_df['label']))

# HAM-A distribution
print(f"\nHAM-A distribution per subject:")
subj_ham = meta_df.groupby('original_subject')['ham_a'].first().sort_values()
print(subj_ham)

print(f"\nDone!")