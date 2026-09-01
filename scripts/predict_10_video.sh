#!/usr/bin/env bash
# same as predict_10.sh but writes 10 annotated videos you can actually watch
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f features/split1_train_T32_none.npz ]; then
  echo "features missing, run scripts/build_pose_only.sh first"
  exit 1
fi

.venv/bin/python src/predict_10_video.py \
  --train features/split1_train_T32_none.npz \
  --test features/split1_test_T32_none.npz \
  --videos data/ucf101 \
  --out predictions_out \
  --n 10
