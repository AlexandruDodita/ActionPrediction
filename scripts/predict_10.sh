#!/usr/bin/env bash
# trains on the pose-only features and prints predictions for 10 test clips
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f features/split1_train_T32_none.npz ]; then
  echo "features missing, run scripts/build_pose_only.sh first"
  exit 1
fi

.venv/bin/python src/predict_demo.py \
  --train features/split1_train_T32_none.npz \
  --test features/split1_test_T32_none.npz \
  --n 10
