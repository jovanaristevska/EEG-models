#!/bin/bash
# Preprocesses and trains EEGPT-SCRATCH on all 23 DASPS LOSO folds (0-22).
# Mirrors run_anxiety_loso_sweep.sh (the pretrained sweep), just pointed at the
# scratch configs (pretrained_path: null, no '_pretrained' suffix) -- this is
# the scratch half of the pretrained-vs-scratch comparison on DASPS.
#
# For each fold this:
#   1. Deletes the HF `datasets` Arrow cache for that fold's dataset_config
#      (NOT cleared by clean_middle_cache/clean_shared_info -- see comments in
#      run_anxiety_loso_sweep.sh for why this matters).
#   2. Rebuilds the summary CSV + mid-level parquet cache from scratch.
#   3. Trains EEGPT-scratch on that fold.
# Continues to the next fold even if one fails, so one bad fold doesn't stall
# the rest of the sweep overnight.

for i in $(seq 0 22); do
  echo "=================================================="
  echo "=== Fold $i: preprocessing at $(date) ==="
  echo "=================================================="

  rm -rf "assets/data/processed/fs_256/anxiety/finetune_loso${i}"

  python preproc.py finetune_datasets.anxiety=finetune_loso${i} \
    clean_middle_cache=true clean_shared_info=true

  echo "=== Fold $i: training at $(date) ==="
  python -m baseline.eegpt.eegpt_trainer \
    assets/conf/baseline/eegpt/eegpt_anxiety_loso${i}.yaml

  echo "=== Fold $i: finished (trainer exit code $?) at $(date) ==="
done
