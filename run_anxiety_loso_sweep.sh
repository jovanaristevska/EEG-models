#!/bin/bash
# Preprocesses and trains EEGPT-pretrained on all remaining DASPS LOSO folds (1-22).
# Fold 0 is already done -- start range is 1 here on purpose.
#
# For each fold this:
#   1. Deletes the HF `datasets` Arrow cache for that fold's dataset_config
#      (NOT cleared by clean_middle_cache/clean_shared_info -- see the config's
#      own comment for why this is needed every time the split logic changes,
#      though for a first run per fold it's just a no-op if nothing exists yet).
#   2. Rebuilds the summary CSV + mid-level parquet cache from scratch.
#   3. Trains EEGPT-pretrained on that fold.
# Continues to the next fold even if one fails, so one bad fold doesn't stall
# the rest of the sweep overnight.

for i in $(seq 1 22); do
  echo "=================================================="
  echo "=== Fold $i: preprocessing at $(date) ==="
  echo "=================================================="

  rm -rf "assets/data/processed/fs_256/anxiety/finetune_loso${i}"

  python preproc.py finetune_datasets.anxiety=finetune_loso${i} \
    clean_middle_cache=true clean_shared_info=true

  echo "=== Fold $i: training at $(date) ==="
  python -m baseline.eegpt.eegpt_trainer \
    assets/conf/baseline/eegpt/eegpt_anxiety_loso${i}_pretrained.yaml

  echo "=== Fold $i: finished (trainer exit code $?) at $(date) ==="
done
