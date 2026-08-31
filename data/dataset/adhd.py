# ============================================================
# adhd.py
# EEG-FM-Bench dataset builder for the ADHD CSV dataset
#
# Actual CSV structure (verified from real data):
#   - 2,166,383 rows total (all subjects stacked in one file)
#   - 21 columns:
#       EEG channels: Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,
#                     F7,F8,T7,T8,P7,P8,Fz,Cz,Pz
#       Label column: 'Class'  → string 'ADHD' or 'Control'
#       Subject ID:   'ID'     → arbitrary string, e.g. 'v10p', 'v107'
#                               (NO consistent suffix convention —
#                                label comes ONLY from 'Class', NOT from ID)
#   - Sampling rate: 128 Hz
#   - 121 subjects total: 61 ADHD, 60 Control
#
# ============================================================

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder

logger = logging.getLogger('preproc')

# Number of subject-level k-fold CV folds; also used by adhd_crown.py so both
# datasets partition the same 121 shared subjects into matching folds.
ADHD_N_FOLDS = 5


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class ADHDConfig(EEGConfig):

    # 'pretrain' or 'finetune' — two variants of this dataset config
    name: str = 'pretrain'

    # HuggingFace cache version — bump this number if you change preprocessing
    # so that old cached data is automatically invalidated and rebuilt
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")

    description: Optional[str] = (
        "EEG dataset for ADHD vs healthy control classification. "
        "121 children (61 ADHD, 60 Control), ages 7-12. "
        "19 channels following the 10-20 standard, recorded at 128 Hz "
        "during a visual attention task. Source: IEEE DataPort."
    )

    citation: Optional[str] = """\
    @data{bhd5-j477-22,
    doi = {10.21227/bhd5-j477},
    author = {Nasrabadi, Ali Motie},
    title = {EEG Data for ADHD / Control Children},
    publisher = {IEEE Dataport},
    year = {2020}
    }
    """

    # Notch filter at 50 Hz to remove European power-line electrical noise
    filter_notch: float = 50.0
    is_notched: bool = False

    # Key used in wrapper.py's DATASET_SELECTOR to find this builder by name
    dataset_name: Optional[str] = 'adhd'

    # CLINICAL = neurological/psychiatric classification task
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL

    # After splitting, each per-subject file will be a CSV
    file_ext: str = 'csv'

    # ------------------------------------------------------------------
    # The 19 EEG channel names, UPPERCASED.
    # These must match the actual CSV column names (case-insensitively).
    # Verified from your real data:
    #   CSV has: Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz
    #   Note: T7/T8/P7/P8 — these are the modern 10-20 names
    # ------------------------------------------------------------------
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
            'FP1', 'FP2',
            'F3',  'F4',
            'C3',  'C4',
            'P3',  'P4',
            'O1',  'O2',
            'F7',  'F8',
            'T7',  'T8',
            'P7',  'P8',
            'FZ',  'CZ',  'PZ',
        ]
    })

    # 15% of subjects for validation, 15% for test, ~70% for training
    valid_ratio: float = 0.15
    test_ratio: float = 0.15

    # Window length in seconds:
    #   pretrain = 10s = 1280 samples at 128 Hz
    #   finetune = 4s  = 512  samples at 128 Hz
    wnd_div_sec: int = 10

    # Your folder structure should look like:
    #   <raw_path>/
    #       ADHD/
    #           adhd.csv          ← the original single file you downloaded
    #           subjects/         ← will be created automatically
    #               v10p.csv
    #               v107.csv
    #               ...
    suffix_path: str = 'ADHD'
    scan_sub_dir: str = 'subjects'

    # The two class labels exactly as they appear in the CSV 'Class' column
    category: list[str] = field(default_factory=lambda: ['ADHD', 'Control'])


# ===========================================================================
# BUILDER
# ===========================================================================

class ADHDBuilder(EEGDatasetBuilder):

    BUILDER_CONFIG_CLASS = ADHDConfig

    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4),
    ] + [
        # NOTE: uses ADHDConfig directly, not BUILDER_CONFIG_CLASS -- a class-body
        # comprehension's per-item expression can't see other class attributes,
        # only true module-level names (this bit me once already for ADHD_N_FOLDS).
        ADHDConfig(
            name=f'finetune_fold{i}', is_finetune=True, wnd_div_sec=4,
            n_folds=ADHD_N_FOLDS, fold_idx=i,
        )
        for i in range(ADHD_N_FOLDS)
    ]

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    # ------------------------------------------------------------------
    # _load_meta_info
    # ------------------------------------------------------------------
    # This method does TWO things:
    #
    # 1. SPLITS adhd.csv into one file per subject.
    #    Groups all rows by the 'ID' column and saves each group as a
    #    separate CSV in the subjects/ subfolder.
    #    e.g.: all rows where ID == 'v10p' → subjects/v10p.csv
    #          all rows where ID == 'v107' → subjects/v107.csv
    #    If the per-subject files already exist, this step is skipped
    #    so you don't re-split every time you run preprocessing.
    #
    # 2. BUILDS self.sub_meta — a lookup table (pandas DataFrame):
    #    subject  | label
    #    v10p     | ADHD
    #    v107     | Control
    #    v5p      | ADHD
    #    ...
    # ------------------------------------------------------------------
    def _load_meta_info(self):

        # Path to the original downloaded file
        original_csv = os.path.join(self.config.raw_path, 'adhdata.csv')

        # Where the per-subject files will live
        subjects_dir = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        os.makedirs(subjects_dir, exist_ok=True)  # create the folder if it doesn't exist

        # Read the entire original CSV into memory
        try:
            logger.info(f"Reading original CSV: {original_csv}")
            df_all = pd.read_csv(original_csv)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find adhdata.csv at: {original_csv}\n"
                f"Please place adhdata.csv in: {self.config.raw_path}"
            )

        # Get the list of all unique subject IDs
        # These are arbitrary strings: 'v10p', 'v107', 'v5p', etc.
        unique_ids = sorted(df_all['ID'].unique())
        logger.info(f"Found {len(unique_ids)} unique subjects in adhdata.csv")

        records = []

        for subject_id in unique_ids:

            # Output file path for this subject
            # The filename is exactly the ID value, e.g. subjects/v10p.csv
            subject_file = os.path.join(subjects_dir, f'{subject_id}.csv')

            # Get all rows belonging to this subject
            df_subject = df_all[df_all['ID'] == subject_id].copy()

            # Only write to disk if the file doesn't already exist.
            # This means on the second run, splitting is skipped entirely.
            if not os.path.exists(subject_file):
                df_subject.to_csv(subject_file, index=False)
                logger.info(f"  Saved {subject_id} ({len(df_subject)} rows) → {subject_file}")

            # Read the label from the 'Class' column.
            # All rows for one subject have the same Class value.
            # We read just the first row.
            # The Class column contains the string 'ADHD' or 'Control' directly.
            # We do NOT parse the ID string — 'v107' does NOT mean Control.
            label = str(df_subject['Class'].iloc[0])  # 'ADHD' or 'Control'

            records.append({
                'subject': subject_id,   # e.g. 'v10p' or 'v107'
                'label': label,          # 'ADHD' or 'Control'
            })

        # Build the metadata lookup table from the list of dicts
        self.sub_meta = pd.DataFrame(records)

        n_adhd = len(self.sub_meta[self.sub_meta['label'] == 'ADHD'])
        n_ctrl = len(self.sub_meta[self.sub_meta['label'] == 'Control'])
        logger.info(f"Metadata ready: {n_adhd} ADHD, {n_ctrl} Control subjects")

    # ------------------------------------------------------------------
    # _walk_raw_data_files
    # ------------------------------------------------------------------
    # Returns a sorted list of all per-subject CSV file paths.
    # The parent class calls this once at startup to get the list of
    # files to process. After _load_meta_info() has run, the subjects/
    # folder already contains one CSV per subject.
    # ------------------------------------------------------------------
    def _walk_raw_data_files(self):
        subjects_dir = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []

        for fname in sorted(os.listdir(subjects_dir)):
            if fname.endswith('.csv'):
                full_path = os.path.join(subjects_dir, fname)
                raw_data_files.append(os.path.normpath(full_path))

        logger.info(f"Found {len(raw_data_files)} subject files to process.")
        return raw_data_files

    # ------------------------------------------------------------------
    # _resolve_file_name
    # ------------------------------------------------------------------
    # Extracts the subject ID from a file path.
    #
    # _extract_file_name (from parent class) removes the folder path
    # and the .csv extension, leaving just the filename stem.
    #
    # Examples:
    #   '\assets\data\raw\ADHD\subjects\v10p.csv' → subject_id = 'v10p'
    #   '\assets\data\raw\ADHD\subjects\v107.csv' → subject_id = 'v107'
    #
    # We do NOT try to parse the label from this string.
    # The label is looked up from self.sub_meta in _resolve_exp_meta_info.
    # ------------------------------------------------------------------
    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        subject_id = self._extract_file_name(file_path)  # e.g. 'v10p' or 'v107'

        return {
            'subject': subject_id,
            'session': 1,   # always 1 — each subject has exactly one recording
        }

    # ------------------------------------------------------------------
    # _resolve_exp_meta_info
    # ------------------------------------------------------------------
    # Collects all metadata for one subject's file.
    # The parent class calls this once per file during preprocessing.
    #
    # Must return a dict containing AT LEAST:
    #   subject  → unique ID string
    #   session  → integer (always 1 here)
    #   montage  → string key from self.config.montage
    #   time     → total recording duration in seconds
    # ------------------------------------------------------------------
    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:

        # Get subject ID from the filename
        info = self._resolve_file_name(file_path)
        subject_id = info['subject']   # e.g. 'v10p' or 'v107'

        # Look up the label from our metadata table.
        # self.sub_meta has columns: 'subject', 'label'
        row = self.sub_meta[self.sub_meta['subject'] == subject_id]
        label = row['label'].iloc[0] if not row.empty else 'Unknown'
        # label is 'ADHD' or 'Control'

        # Load the CSV to count timepoints → compute duration in seconds
        # duration = number_of_rows / sampling_rate
        df = pd.read_csv(file_path)
        n_samples = len(df)
        duration = n_samples / 128.0   # 128 Hz sampling rate

        info.update({
            'montage': '10_20',   # must be a key in self.config.montage
            'time': duration,     # total recording length in seconds
            'group': label,       # 'ADHD' or 'Control' — stored in final dataset
            'age': -1,            # age not available in this dataset
            'sex': 'U',           # sex not available ('U' = unknown)
        })

        return info

    # ------------------------------------------------------------------
    # _resolve_exp_events
    # ------------------------------------------------------------------
    # Tells the parent class what label to assign to the windows
    # sliced from this recording.
    #
    # Returns a list of annotation tuples: (label, start_ms, end_ms)
    #   label    = class name string
    #   start_ms = start of labeled segment in milliseconds
    #   end_ms   = end of labeled segment (-1 means "until end of file")
    #
    # Since the entire recording of one subject has one single label,
    # we return one annotation covering the whole file: start=0, end=-1.
    #
    # pretrain mode → return 'default' (labels not needed)
    # finetune mode → return the real label ('ADHD' or 'Control')
    # ------------------------------------------------------------------
    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):

        if not self.config.is_finetune:
            # Pretrain: no class labels needed, just mark the whole file
            return [('default', 0, -1)]

        # Finetune: label the entire recording with the subject's class
        group = info['group']       # 'ADHD' or 'Control'
        return [(group, 0, -1)]     # whole file gets this label

    # ------------------------------------------------------------------
    # _divide_split
    # ------------------------------------------------------------------
    # Assigns each windowed EEG sample to 'train', 'valid', or 'test'.
    # The parent class calls this after all windows have been created.
    #
    # KEY RULE: we split by SUBJECT, not by individual window.
    # This means ALL windows from subject 'v10p' go entirely into
    # one split (e.g. train) — they never appear in test as well.
    # This prevents data leakage between splits.
    #
    # _divide_label_balance_all_split (from parent) also ensures the
    # ADHD/Control ratio is roughly equal in train, valid, and test.
    # ------------------------------------------------------------------
    def _divide_split(self, df: DataFrame) -> DataFrame:
        if self.config.is_finetune:
            if self.config.n_folds and self.config.n_folds > 0:
                df = self._divide_kfold_split(df)
            else:
                # All three splits for finetune: train, valid, test
                df = self._divide_label_balance_all_split(df)
        else:
            # Pretrain only needs train and valid (no test set needed)
            df = self._divide_label_balance_all_split(df, splits_name=['train', 'valid'])
        return df

    # ------------------------------------------------------------------
    # standardize_chs_names
    # ------------------------------------------------------------------
    # Converts channel names to the unified format used across all
    # datasets in the EEG-FM-Bench benchmark.
    #
    # Our config already stores channels as uppercase strings
    # ('FP1', 'FZ', 'T7', etc.) so we just apply the parent class's
    # replace dictionary, which handles known aliases, for example:
    #   'T3' ↔ 'T7',  'T4' ↔ 'T8',  'T5' ↔ 'P7',  'T6' ↔ 'P8'
    #
    # Results are cached in _std_chs_cache so this only runs once
    # per montage name, not once per file.
    # ------------------------------------------------------------------
    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs: list[str] = self.config.montage[montage]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    # ------------------------------------------------------------------
    # _read_raw_data
    # ------------------------------------------------------------------
    # Reads one subject's CSV file and converts it into an MNE Raw object.
    # The parent class uses MNE Raw objects for all preprocessing:
    # notch filtering, bandpass filtering, resampling, and windowing.
    #
    # This method is the bridge between your CSV format and MNE.
    #
    # Steps explained:
    #   1. Read the CSV with pandas (gives a DataFrame)
    #   2. Select only the 19 EEG channel columns
    #      (drop 'Class' and 'ID' — they are not signal data)
    #   3. Transpose: CSV is (timepoints × channels),
    #                 MNE needs (channels × timepoints)
    #   4. Convert µV → V  (CSV values are in microvolts,
    #                        MNE works internally in Volts)
    #   5. Create MNE Info (stores channel names + sampling rate)
    #   6. Wrap the numpy array in MNE RawArray
    #   7. Attach standard 10-20 electrode positions to the Raw object
    # ------------------------------------------------------------------
    def _read_raw_data(self, file_path: str, preload: bool = True, verbose: bool = False):

        # Step 1: Read CSV
        df = pd.read_csv(file_path)

        # Step 2: Select the 19 EEG channel columns
        # The config has uppercase names like 'FP1', 'T7'
        # The CSV has mixed case names like 'Fp1', 'T7'
        # We build a case-insensitive mapping to find the right column
        csv_cols_upper = {c.upper(): c for c in df.columns}
        # csv_cols_upper = {'FP1': 'Fp1', 'FP2': 'Fp2', 'T7': 'T7', 'CLASS': 'Class', ...}

        channel_names_upper = self.config.montage['10_20']
        # channel_names_upper = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
        #                         'O1','O2','F7','F8','T7','T8','P7','P8','FZ','CZ','PZ']

        # For each channel name in our config, find the matching CSV column
        selected_cols = [csv_cols_upper[ch] for ch in channel_names_upper]
        # selected_cols = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
        #                   'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

        eeg_df = df[selected_cols]  # DataFrame shape: (n_timepoints, 19)

        # Step 3: Transpose to (channels × timepoints)
        data = eeg_df.values.T.astype(np.float64)
        # data shape: (19, n_timepoints) — what MNE expects

        # Step 4: Convert from microvolts to Volts
        # Raw CSV values are in µV. MNE stores everything in V.
        # 1 µV = 0.000001 V = 1×10⁻⁶ V
        data = data * 1e-6

        # Step 5: Create MNE Info object
        # This is MNE's metadata container: channel names + sampling rate
        info = mne.create_info(
            ch_names=channel_names_upper,   # ['FP1','FP2','F3',...]
            sfreq=128.0,                    # 128 samples per second
            ch_types='eeg',                 # all channels are EEG type
        )

        # Step 6: Wrap data in MNE RawArray
        # RawArray is MNE's container for continuous (non-epoched) EEG data
        raw = mne.io.RawArray(data, info, verbose=verbose)

        # Step 7: Attach standard 10-20 electrode positions
        # This tells MNE where each electrode sits physically on the scalp.
        # match_case=False: handles mismatches like 'FP1' vs 'Fp1'
        # on_missing='ignore': don't crash if a channel isn't in the standard set
        dig_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(dig_montage, match_case=False, on_missing='ignore', verbose=verbose)

        return raw


# ===========================================================================
# STANDALONE TEST
# Run this file directly to test without the full preproc.py pipeline:
#   python data/dataset/adhd.py
# ===========================================================================

if __name__ == "__main__":
    builder = ADHDBuilder("finetune")
    # Uncomment to wipe cached data and start completely fresh:
    # builder.clean_disk_cache()
    # builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
