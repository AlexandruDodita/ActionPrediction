#!/usr/bin/env bash
# Precomputed HRNet 2D keypoints for all 13,320 UCF101 clips (PYSKL / MMAction2 release, ~1.07 GB).
# Same file is mirrored at https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL=https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl
OUT=data/ucf101_hrnet.pkl
EXPECTED=1070780736
if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -eq "$EXPECTED" ]; then
  echo "already present: $OUT"; exit 0
fi
curl -L --retry 3 -C - -o "$OUT" "$URL"
[ "$(stat -c %s "$OUT")" -eq "$EXPECTED" ] || { echo "size mismatch, download incomplete"; exit 1; }
echo "ok: $OUT"
