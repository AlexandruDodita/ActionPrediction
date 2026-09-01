#!/usr/bin/env bash
# builds pose-only features (x,y,score) for all clips, all 3 splits
# no video needed, just the pkl -> fast (seconds)
set -euo pipefail
cd "$(dirname "$0")/.."

for split in 1 2 3; do
  .venv/bin/python src/build_dataset.py \
    --pkl data/ucf101_hrnet.pkl \
    --split "$split" \
    --frames 32 \
    --flow none \
    --subsets train,test \
    --out features
done

# action prediction accuracy on split 1, pose-only
.venv/bin/python src/train_eval_quick.py \
  --train features/split1_train_T32_none.npz \
  --test features/split1_test_T32_none.npz \
  --epochs 80
