# ============================================================
# anxiety.py
# EEG-FM-Bench dataset builder for the DASPS Anxiety dataset
#
# VERSION 2 (updated):
# - Uses official DASPS_HAM_labels.mat pre-labeled data
# - Labels: SAM Arousal > 5 AND Valence < 5 → Anxiety, else Control
# - Data source: pre-converted CSV files in subjects/ folder
# - Metadata source: anxiety_v2_finetune_info.csv
# - SUBJECT-LEVEL split to prevent data leakage
# ============================================================

import logging
import os

import torch

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


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class AnxietyConfig(EEGConfig):

    name: str = 'finetune'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "DASPS database (Database for Anxious States based on a Psychological "
        "Stimulation). 23 adult participants (ages 18-45) recorded during anxiety "
        "elicitation via face-to-face psychological stimuli. 14 EEG channels via "
        "Emotiv EPOC+ headset at 128 Hz. Per-trial labels from official DASPS+HAM "
        "labels: SAM Arousal > 5 AND Valence < 5 -> Anxiety. Total: 276 labelled trials."
    )

    citation: Optional[str] = """\
    @data{barx-we60-21,
    doi = {10.21227/barx-we60},
    author = {Baghdadi, Asma and Aribi, Yassine and Fourati, Rahma and 
              Halouani, Najla and Siarry, Patrick and Alimi, Adel M.},
    title = {DASPS database},
    publisher = {IEEE Dataport},
    year = {2021}
    }
    """

    filter_notch: float = 50.0
    is_notched: bool = False
    dataset_name: Optional[str] = 'anxiety'
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
    file_ext: str = 'csv'

    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'emotiv_14': [
            'AF3', 'AF4',
            'F3',  'F4',
            'FC5', 'FC6',
            'F7',  'F8',
            'T7',  'T8',
            'P7',  'P8',
            'O1',  'O2',
        ]
    })

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 10

    suffix_path: str = 'Anxiety'
    scan_sub_dir: str = 'subjects'
    meta_info_file: str = 'summary/finetune/anxiety_v2_finetune_info.csv'

    category: list[str] = field(default_factory=lambda: ['Anxiety', 'Control'])

    # NEW: SAM-based labelling thresholds
    arousal_threshold: float = 5.0
    valence_threshold: float = 5.0

    orig_fs: float = 128.0


# ===========================================================================
# BUILDER
# ===========================================================================

class AnxietyBuilder(EEGDatasetBuilder):

    BUILDER_CONFIG_CLASS = AnxietyConfig

    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4),
    ]

    def __init__(self, config_name='finetune', **kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        """
        Load pre-computed metadata from anxiety_v2_finetune_info.csv.
        The CSV was produced by convert_ham_labels_to_csv.py from the
        official DASPS_HAM_labels.mat file.
        """
        subjects_dir = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        meta_path = os.path.join(self.config.raw_path, self.config.meta_info_file)

        # Verify CSV files exist
        if not os.path.exists(subjects_dir):
            raise FileNotFoundError(
                f"Could not find subjects folder at: {subjects_dir}\n"
                f"Please run convert_ham_labels_to_csv.py first."
            )

        existing_csvs = [f for f in os.listdir(subjects_dir) if f.endswith('.csv')]
        expected_csvs = 23 * 12  # 276

        if len(existing_csvs) < expected_csvs:
            raise FileNotFoundError(
                f"Expected {expected_csvs} CSV files in {subjects_dir}, "
                f"found only {len(existing_csvs)}.\n"
                f"Please run convert_ham_labels_to_csv.py first."
            )

        logger.info(f"Found {len(existing_csvs)} CSV files in subjects folder.")

        # Load metadata CSV
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Could not find metadata CSV at: {meta_path}\n"
                f"Please run convert_ham_labels_to_csv.py first."
            )

        logger.info(f"Loading metadata from: {meta_path}")
        self.sub_meta = pd.read_csv(meta_path)

        # Verify metadata has required columns
        required_cols = ['subject', 'original_subject', 'valence', 'arousal', 'label']
        for col in required_cols:
            if col not in self.sub_meta.columns:
                raise ValueError(
                    f"Metadata CSV missing required column: {col}\n"
                    f"Available columns: {self.sub_meta.columns.tolist()}"
                )

        # Report statistics
        n_anx = len(self.sub_meta[self.sub_meta['label'] == 'Anxiety'])
        n_ctrl = len(self.sub_meta[self.sub_meta['label'] == 'Control'])
        n_orig = self.sub_meta['original_subject'].nunique()
        logger.info(
            f"Metadata ready: {n_anx} Anxiety, {n_ctrl} Control trials "
            f"from {n_orig} original participants "
            f"(labelling: arousal > {self.config.arousal_threshold} AND "
            f"valence < {self.config.valence_threshold})."
        )

    def _walk_raw_data_files(self):
        subjects_dir = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []

        for fname in sorted(os.listdir(subjects_dir)):
            if fname.endswith('.csv'):
                full_path = os.path.join(subjects_dir, fname)
                raw_data_files.append(os.path.normpath(full_path))

        logger.info(f"Found {len(raw_data_files)} trial files to process.")
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        subject_id = self._extract_file_name(file_path)
        return {
            'subject': subject_id,
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        subject_id = info['subject']

        row = self.sub_meta[self.sub_meta['subject'] == subject_id]
        label = row['label'].iloc[0] if not row.empty else 'Unknown'

        df = pd.read_csv(file_path)
        n_samples = len(df)
        duration = n_samples / self.config.orig_fs

        info.update({
            'montage': 'emotiv_14',
            'time': duration,
            'group': label,
            'age': -1,
            'sex': 'U',
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        group = info['group']
        return [(group, 0, -1)]

    # ==================================================================
    # CUSTOM _divide_split — prevents data leakage
    # ==================================================================
    def _divide_split(self, df: DataFrame) -> DataFrame:
        print("\n" + "=" * 70)
        print(">>> CUSTOM SUBJECT-LEVEL _divide_split IS RUNNING (v2) <<<")
        print("=" * 70)
        print(f"Input df: shape={df.shape}, columns={df.columns.tolist()}")

        if not self.config.is_finetune:
            print(">>> Pretrain mode: using default split")
            return self._divide_label_balance_all_split(df, splits=['train', 'valid'])

        # Extract original_subject from 'subject' column (S01t05 → S01)
        df = df.copy()
        df['original_subject'] = df['subject'].astype(str).str[:3]

        unique_subjects = sorted(df['original_subject'].unique())
        print(f"Found {len(unique_subjects)} unique original subjects")

        # Deterministic shuffle
        rng = np.random.RandomState(42)
        shuffled = list(unique_subjects)
        rng.shuffle(shuffled)

        # Compute split sizes (by subjects, not by trials)
        n = len(shuffled)
        n_valid = max(1, int(round(n * self.config.valid_ratio)))
        n_test = max(1, int(round(n * self.config.test_ratio)))
        n_train = n - n_valid - n_test

        train_subjects = set(shuffled[:n_train])
        valid_subjects = set(shuffled[n_train:n_train + n_valid])
        test_subjects = set(shuffled[n_train + n_valid:])

        print(f"Subject allocation:")
        print(f"  train ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  valid ({len(valid_subjects)}): {sorted(valid_subjects)}")
        print(f"  test  ({len(test_subjects)}):  {sorted(test_subjects)}")

        def assign_split(orig_sub):
            if orig_sub in train_subjects:
                return 'train'
            elif orig_sub in valid_subjects:
                return 'valid'
            else:
                return 'test'

        df['split'] = df['original_subject'].apply(assign_split)
        df = df.drop(columns=['original_subject'])

        print(f"Final split distribution:\n{df['split'].value_counts()}")
        print("=" * 70 + "\n")

        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs: list[str] = self.config.montage[montage]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = True, verbose: bool = False):
        df = pd.read_csv(file_path)

        csv_cols_upper = {c.upper(): c for c in df.columns}
        channel_names_upper = self.config.montage['emotiv_14']

        selected_cols = [csv_cols_upper[ch] for ch in channel_names_upper]
        eeg_df = df[selected_cols]

        data = eeg_df.values.T.astype(np.float64)
        data = data * 1e-6

        info = mne.create_info(
            ch_names=channel_names_upper,
            sfreq=self.config.orig_fs,
            ch_types='eeg',
        )

        raw = mne.io.RawArray(data, info, verbose=verbose)

        dig_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(dig_montage, match_case=False, on_missing='ignore', verbose=verbose)

        return raw


# ===========================================================================
# STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    builder = AnxietyBuilder("finetune")
    
    # First run or after override changes: must clean cache + preproc
    builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)