# ActionPrediction — early action anticipation on UCF101 with pose-anchored optical flow

Reproduces the core idea of **PoseOFF** (de Zoete Grundy, McCarthy, Fluke — *Pose-Anchored Optical Flow for
Low-Latency Human Action Anticipation in Human-Robot Teaming*) on UCF101: a skeleton-based action model whose
per-joint input is `(x, y, score)` **plus a small window of optical flow sampled around each joint**, evaluated
under partial observation (10 % … 100 % of the clip).

We do **not** train any pose/flow extractor. Pose comes precomputed; flow windows are the only thing we compute.

## Data

| What | Where | Size | Notes |
|---|---|---|---|
| UCF101 videos | [Kaggle: matthewjansen/ucf101-action-recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) → `archive.zip` | 7.0 GB | 13,451 `.avi` (320×240) = the 13,320 official clips, 131 of them duplicated across the Kaggle `train/val/test` folders. Only `clip_name, clip_path, label` in the CSVs — **no features shipped**. |
| 2D keypoints (all clips, all frames) | `scripts/download_keypoints.sh` → `data/ucf101_hrnet.pkl` ([PYSKL/MMAction2 release](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md)) | 1.07 GB | HRNet, 17 COCO joints, `keypoint (M,T,17,2)` + `keypoint_score (M,T,17)` per clip, up to M persons, float16. Coordinates are in a **340×256** frame (scale by 320/340, 240/256 to get video pixels). Also contains the **official split1/2/3** train/test lists. Verified: covers all 13,320 clips, 0 missing. |

Everything under `data/` and `features/` is git-ignored — do not commit the archive, the pickle, or the outputs.

### Why the Kaggle split is not used
UCF101 clips are grouped by source video (`g01`…`g25`). The official splits keep groups disjoint; the Kaggle
folders do not — **1,274 of 1,275 (class, group) pairs in Kaggle `test/` also appear in Kaggle `train/`**
(same actor, same background), and 131 clips appear in two folders. We therefore ignore the folders and use the
official split lists carried in the keypoint pickle. Results are then comparable to the paper's Table I
(UCF101 split 1/2/3, joint stream only): pose-only GCNs 47–63 %, PoseOFF 58–70 %.

## Feature representation (what `src/build_dataset.py` produces)

Per clip: `X[T, 17, 53]` float16, `T=32` uniformly sampled frames.

```
X[..., 0:3]  = (x, y, score)            HRNet keypoint of the most confident skeleton, in video pixels
X[..., 3:53] = 5×5 window of (u, v)     optical flow between sampled frame t and t+1, centred on the joint,
                                        dilation 1, border-clamped, flattened  (paper: W=5, dilation=1 on UCF101)
```

Saved as `features/split{S}_{train|test}_T{T}_{flow}.npz` with keys `X, y, names, classes, frames, window, dilation, flow`.
Class ids are alphabetical over the 101 class names (same as the pickle's `label`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# CPU:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# GPU: install torch + torchvision for your CUDA from https://pytorch.org/get-started/locally/ first, then
pip install numpy opencv-python-headless

scripts/download_keypoints.sh          # -> data/ucf101_hrnet.pkl (1.07 GB)
# put archive.zip in data/ (or anywhere; pass the path). No need to unzip — the script reads clips out of it.
```

## Building the features

```bash
# pose-only, no video needed, seconds:
python src/build_dataset.py --split 1 --frames 32 --flow none

# PoseOFF with OpenCV DIS flow (CPU). ~0.35 s CPU per clip -> ~15 min for all 13,320 clips on 8 cores:
python src/build_dataset.py --videos data/archive.zip --split 1 --frames 32 --flow dis            # --dis-preset fast (default) | medium

# PoseOFF with RAFT (what the paper used) — run this on the GPU machine:
python src/build_dataset.py --videos data/archive.zip --split 1 --frames 32 --flow raft_large --device cuda
python src/build_dataset.py --videos data/archive.zip --split 1 --frames 32 --flow raft_small --device cuda   # ~3x faster
```

Useful flags: `--limit 50` (smoke test on the first 50 clips per subset), `--subsets train` / `test`,
`--workers N` (video decoding processes; RAFT itself runs in the main process on `--device`), `--window`, `--dilation`.

Notes for the GPU run:
- Video decoding + DIS run in worker processes; RAFT runs batched (16 pairs) in the main process. If the GPU is
  starved, raise `--workers`; if it OOMs, lower `batch` in `Raft.__init__`.
- 320×240 is already divisible by 8 so RAFT needs no padding (the code pads anyway if given other sizes).
- First RAFT run downloads the torchvision weights (~20 MB small / ~80 MB large).
- Reading from the zip copies each clip to `/dev/shm` before decoding; an extracted directory works too (`--videos data/ucf101/`).

## Timings measured so far (8-core laptop CPU, no GPU, T=32)

| Step | Cost |
|---|---|
| unzip + decode one clip | ~0.2 s |
| DIS flow, 1 thread, per pair | ultrafast 2 ms · fast 4 ms · medium 22 ms |
| DIS `fast` build, whole dataset | ~15 min on 8 cores (decode-bound) |
| RAFT-small on CPU | ~10 s per clip → GPU only; expect ~1–3 ms/pair on an RTX 4050 → ≈ 15–30 min for everything incl. decode |
| pose-only build | seconds (loading the 1 GB pickle dominates) |

Smoke-tested here: `--flow dis --limit 60`, `--flow raft_small --device cpu --limit 3`, `--flow none` — all produce
`(N, 32, 17, 53|3)` with 0 errors, non-zero flow windows, keypoints in video pixel coordinates.

## Training and evaluation

Two trainers, both consuming the `.npz` features above (pose-only or PoseOFF variants of the same model,
identical settings, so the comparison is apples-to-apples):

```bash
scripts/build_pose_only.sh                  # pose-only features for all 3 splits + quick MLP baseline
python src/train_eval_quick.py --train features/split1_train_T32_dis.npz --test features/split1_test_T32_dis.npz
                                            # fast proxy: MLP on mean/std-pooled features, pose vs PoseOFF
python src/train_stgcn.py                   # the real comparison: ST-GCN backbone (paper Sec. III-B),
                                            # pose-only and PoseOFF variants (--variant pose|poseoff)
```

Demos (after features exist):

```bash
scripts/predict_10.sh                       # print predictions for 10 test clips
scripts/predict_10_video.sh                 # burn prediction + skeleton onto the clips
python src/export_fusion_demo.py && python src/fusion_demo.py   # play with pose/flow probability fusion
python src/export_video_demo_stgcn.py && python src/gen_video_demo_html.py  # browser demo (demo/video_demo.html)
```

For videos **not** in UCF101 (no HRNet pickle coverage) there is a per-clip extraction pipeline:
`src/extract_frames.py` → `src/pose_rtmpose.py` (RTMPose via `rtmlib`) → `src/optical_flow.py` (RAFT/DIS) →
`src/poseoff.py`.

## Not done yet / ideas
- Anticipation evaluation on the ST-GCN: temporal masking at observation ratios 0.1…1.0, accuracy-vs-ratio
  curve + AUC + per-class recall deltas (paper Fig. 5/6). An earlier prototype run of this (8 epochs, CPU) gave
  pose 35.5 % / PoseOFF 43.0 % at full observation, with PoseOFF matching pose's full-clip accuracy after 60 %
  of the clip — consistent with the paper's UCF101 findings.
- RAFT-flow features (GPU) vs DIS: `build_dataset.py --flow raft_large --device cuda`, then retrain.
- Splits 2 and 3 (`build_dataset.py --split 2|3`).
- Multi-person: `select_person` keeps only the top skeleton; PYSKL keeps two.

## References
- Paper: `0362_FI_flow.pdf` (PoseOFF). UCF101 numbers there used YOLO-Pose-L keypoints + RAFT flow.
- Keypoints: PYSKL — Duan et al., *Revisiting Skeleton-based Action Recognition* (PoseC3D), 2022.
- Flow: Teed & Deng, *RAFT*, ECCV 2020 (torchvision weights); Kroeger et al., *DIS optical flow*, ECCV 2016 (OpenCV).
