"""
Interpolation validation for the ADHD dataset (with modern channel names).
"""

import os
import glob
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBJECT_DIR = 'assets/data/raw/ADHD/subjects'  # ← ажурирaj до реален path
OUTPUT_CSV = 'interpolation_validation_results.csv'

# ⭐ АЖУРИРАНИ CHANNEL NAMES (modern 10-10 nomenclature)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                 'Fz', 'Cz', 'Pz']

SAMPLING_RATE = 128  # Hz


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_subject_raw(csv_path):
    """Load one subject's CSV and convert to MNE Raw object."""
    df = pd.read_csv(csv_path)
    
    # Провери дека сите очекувани канали се тука
    missing = [ch for ch in CHANNEL_NAMES if ch not in df.columns]
    if missing:
        raise ValueError(f"Missing channels in {csv_path}: {missing}")
    
    # Select only channel columns (не Class, не ID)
    channel_data = df[CHANNEL_NAMES].values.T  # (19 channels × N samples)
    
    # Convert μV → V for MNE
    channel_data = channel_data * 1e-6
    
    # Create MNE Raw
    info = mne.create_info(
        ch_names=CHANNEL_NAMES,
        sfreq=SAMPLING_RATE,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(channel_data, info, verbose=False)
    
    # Postavi standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn', verbose=False)
    
    return raw


def validate_interpolation_single_subject(raw):
    """
    Leave-one-channel-out validation for one subject.
    
    Returns dict: {channel_name: {'correlation': float, 'rmse_uv': float}}
    """
    ch_names = raw.ch_names
    original_data = raw.get_data()
    results = {}
    
    for i, ch in enumerate(ch_names):
        try:
            # Копирaj raw
            raw_copy = raw.copy()
            
            # Означи го тековниот канал како bad
            raw_copy.info['bads'] = [ch]
            
            # Reinterpolate
            raw_copy.interpolate_bads(
                method={'eeg': 'spline'},
                reset_bads=False,
                verbose=False
            )
            
            # Земи predictions
            predicted = raw_copy.get_data(picks=[ch])[0]
            real = original_data[i]
            
            # Метрики
            correlation, _ = pearsonr(predicted, real)
            rmse_v = np.sqrt(np.mean((predicted - real)**2))
            rmse_uv = rmse_v * 1e6  # V → μV
            
            results[ch] = {
                'correlation': correlation,
                'rmse_uv': rmse_uv
            }
        except Exception as e:
            print(f"  Warning: {ch} failed - {e}")
            results[ch] = {'correlation': np.nan, 'rmse_uv': np.nan}
    
    return results


# ============================================================================
# MAIN VALIDATION LOOP
# ============================================================================

def main():
    # Најди сите CSV файлови (различни patterns)
    patterns = ['v*.csv', 'subject_*.csv', '*.csv']
    subject_files = []
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(SUBJECT_DIR, pattern)))
        if files:
            subject_files = files
            break
    
    if len(subject_files) == 0:
        print(f"ERROR: No CSV files found in {SUBJECT_DIR}")
        return
    
    print(f"Found {len(subject_files)} subject files")
    print(f"First file: {subject_files[0]}")
    print(f"Last file: {subject_files[-1]}")
    
    # Собери резултати
    all_results = []
    failed = []
    
    for csv_path in tqdm(subject_files, desc="Validating subjects"):
        subject_id = os.path.basename(csv_path).replace('.csv', '')
        
        try:
            # Load
            raw = load_subject_raw(csv_path)
            
            # Validate
            validation = validate_interpolation_single_subject(raw)
            
            # Собери за оваа subject
            for ch, metrics in validation.items():
                all_results.append({
                    'subject_id': subject_id,
                    'channel': ch,
                    'correlation': metrics['correlation'],
                    'rmse_uv': metrics['rmse_uv']
                })
        except Exception as e:
            print(f"  Skipped {subject_id}: {e}")
            failed.append(subject_id)
            continue
    
    if failed:
        print(f"\nWarning: {len(failed)} subjects failed: {failed[:5]}...")
    
    # DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed results saved to {OUTPUT_CSV}")
    
    # ========================================================================
    # STATISTICS ЗА PAPER
    # ========================================================================
    print("\n" + "="*70)
    print("INTERPOLATION VALIDATION RESULTS")
    print("="*70)
    
    # Prov statistics — per channel
    print("\n--- Per-channel statistics (across all subjects) ---")
    channel_stats = df.groupby('channel').agg({
        'correlation': ['mean', 'std', 'min', 'max'],
        'rmse_uv': ['mean', 'std']
    }).round(3)
    print(channel_stats)
    
    # Втор statistics — per subject
    print("\n--- Per-subject statistics (average across channels) ---")
    subject_means = df.groupby('subject_id')['correlation'].mean()
    print(f"  Mean correlation: {subject_means.mean():.3f} ± {subject_means.std():.3f}")
    print(f"  Min: {subject_means.min():.3f}")
    print(f"  Max: {subject_means.max():.3f}")
    
    # Трет statistics — overall
    print("\n--- Overall statistics ---")
    correlations = df['correlation'].dropna()
    print(f"  Mean correlation (all channels × subjects): {correlations.mean():.3f} ± {correlations.std():.3f}")
    print(f"  Median: {correlations.median():.3f}")
    print(f"  25th percentile: {correlations.quantile(0.25):.3f}")
    print(f"  75th percentile: {correlations.quantile(0.75):.3f}")
    
    rmse_values = df['rmse_uv'].dropna()
    print(f"\n  Mean RMSE: {rmse_values.mean():.2f} ± {rmse_values.std():.2f} μV")
    
    # ========================================================================
    # CROWN-SPECIFIC ANALYSIS
    # ========================================================================
    print("\n--- Crown-relevant proxies ---")
    # Crown користи: F5, F6, C3, C4, CP3, CP4, PO3, PO4
    # Најблиски 10-20 electrodes:
    #   F5, F6  → F3, F4, F7, F8
    #   C3, C4  → директно измерени!
    #   CP3, CP4 → C3, C4, P3, P4
    #   PO3, PO4 → P3, P4, O1, O2
    
    crown_proxies = ['F3', 'F4', 'F7', 'F8',   # proxies for F5, F6
                     'C3', 'C4',                 # measured directly
                     'P3', 'P4',                 # proxies for CP3, CP4, PO3, PO4
                     'O1', 'O2']                 # proxies for PO3, PO4
    
    crown_df = df[df['channel'].isin(crown_proxies)]
    crown_stats = crown_df.groupby('channel')['correlation'].agg(['mean', 'std']).round(3)
    print(crown_stats)
    print(f"\n  Mean correlation for Crown-relevant electrodes: {crown_df['correlation'].mean():.3f}")
    
    # ========================================================================
    # SUMMARY ЗА PAPER
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    print(f"""
Total subjects analysed: {len(subject_means)}
Total channel-subject validations: {len(correlations)}

Overall interpolation fidelity:
  Mean correlation: {correlations.mean():.3f} ± {correlations.std():.3f}
  Median: {correlations.median():.3f}
  IQR: [{correlations.quantile(0.25):.3f}, {correlations.quantile(0.75):.3f}]
  Mean RMSE: {rmse_values.mean():.2f} μV

Interpretation:
  - Correlation > 0.85: Excellent
  - Correlation 0.70-0.85: Good
  - Correlation 0.50-0.70: Moderate
  - Correlation < 0.50: Poor
""")
    
    print("Done!")


if __name__ == "__main__":
    main()