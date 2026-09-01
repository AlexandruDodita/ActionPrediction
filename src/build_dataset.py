#!/usr/bin/env python
"""Build PoseOFF-style features for UCF101 from precomputed HRNet keypoints + optical flow.

Output per clip: array of shape (T, V=17, C) with C = 3 + 2*W*W
    channels [0:3]  -> (x, y, score)  keypoint in video pixel coordinates
    channels [3:]   -> flattened WxW window of (u, v) optical flow centred on the joint
                       (paper: W=5, dilation=1 for UCF101  ->  50 flow channels, 53 total)

Pose comes from the PYSKL/MMAction2 release (HRNet, every frame, all 13,320 clips):
    https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl
Flow is computed here between each sampled frame and the next one:
    --flow dis         OpenCV DIS (CPU; 'fast' preset ~4 ms/pair single-thread at 320x240)
    --flow raft_small  torchvision RAFT-small (GPU recommended)
    --flow raft_large  torchvision RAFT-large (what the paper used; GPU only in practice)
    --flow none        pose-only (no video needed)

Videos can be read straight from the Kaggle archive.zip or from an extracted directory.

Example:
    python src/build_dataset.py --videos ~/Downloads/archive.zip --split 1 --frames 32 --flow dis
    python src/build_dataset.py --videos data/ucf101 --split 1 --frames 32 --flow raft_large --device cuda
"""
import argparse
import os
import pickle
import sys
import time
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)  # we parallelise over clips, not inside OpenCV

PKL_SHAPE = (256, 340)  # (h, w) frame size the HRNet keypoints were extracted at


# ----------------------------------------------------------------------------- video access
class VideoSource:
    """Find a clip by name (e.g. v_Swing_g05_c02) inside a zip archive or a directory tree."""

    def __init__(self, path):
        self.path = path
        self.is_zip = os.path.isfile(path) and path.endswith(".zip")
        self._zip = None
        self.index = {}
        if self.is_zip:
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    if n.endswith(".avi"):
                        self.index.setdefault(os.path.basename(n)[:-4], n)
        else:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".avi"):
                        self.index.setdefault(f[:-4], os.path.join(root, f))

    def _z(self):
        if self._zip is None:  # opened lazily so it is per-worker after fork
            self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def read_frames(self, name, wanted):
        """Decode frames with indices in `wanted` (sorted set). Returns dict idx -> BGR frame."""
        if self.is_zip:
            tmp = f"/dev/shm/{os.getpid()}_{name}.avi" if os.path.isdir("/dev/shm") else f"{name}.avi"
            with open(tmp, "wb") as f:
                f.write(self._z().read(self.index[name]))
            cap = cv2.VideoCapture(tmp)
        else:
            tmp = None
            cap = cv2.VideoCapture(self.index[name])
        out, i, last = {}, 0, max(wanted)
        while i <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if i in wanted:
                out[i] = fr
            i += 1
        cap.release()
        if tmp:
            os.remove(tmp)
        return out, i  # i = number of frames actually decodable


# ----------------------------------------------------------------------------- pose helpers
def select_person(kp, score):
    """(M,T,V,2),(M,T,V) -> the single most confident skeleton (T,V,3)."""
    m = int(np.argmax(score.mean(axis=(1, 2))))
    return np.concatenate([kp[m].astype(np.float32), score[m, ..., None].astype(np.float32)], axis=-1)


def sample_indices(n_frames, T):
    """T frame indices in [0, n_frames-2] so that idx+1 always exists for flow."""
    hi = max(n_frames - 2, 0)
    return np.linspace(0, hi, T).round().astype(int)  # duplicates are fine for very short clips


def window_offsets(W, dil):
    r = np.arange(W) - W // 2
    dy, dx = np.meshgrid(r * dil, r * dil, indexing="ij")
    return dy.ravel(), dx.ravel()


def crop_windows(flow, joints_xy, W, dil):
    """flow (H,W,2), joints_xy (V,2) in pixels -> (V, 2*W*W) flattened (u,v) windows, border-clamped."""
    H, Wd = flow.shape[:2]
    dy, dx = window_offsets(W, dil)
    ys = np.clip(np.round(joints_xy[:, 1:2]).astype(int) + dy[None], 0, H - 1)
    xs = np.clip(np.round(joints_xy[:, 0:1]).astype(int) + dx[None], 0, Wd - 1)
    return flow[ys, xs].reshape(len(joints_xy), -1)  # (V, W*W, 2) -> (V, 2*W*W)


# ----------------------------------------------------------------------------- per-clip work
_SRC = None
_DIS = None


DIS_PRESETS = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
               "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}


def _init_worker(video_path, dis_preset):
    global _SRC, _DIS
    _SRC = VideoSource(video_path) if video_path else None
    _DIS = cv2.DISOpticalFlow_create(DIS_PRESETS[dis_preset])


def process_clip(job):
    """Returns (name, pose (T,V,3) in video px, flow_windows (T,V,2WW) or None, rgb_pairs or None, info)."""
    name, kp, score, T, W, dil, mode = job
    pose_all = select_person(kp, score)  # (Tn, V, 3) in 340x256 coords
    n = pose_all.shape[0]
    idx = sample_indices(n, T)
    pose = pose_all[np.clip(idx, 0, n - 1)]

    if mode == "none":
        pose[..., 0] *= 320.0 / PKL_SHAPE[1]
        pose[..., 1] *= 240.0 / PKL_SHAPE[0]
        return name, pose, None, None, {"frames": n, "decoded": n}

    wanted = set(idx.tolist()) | set((idx + 1).tolist())
    frames, decoded = _SRC.read_frames(name, wanted)
    if not frames:
        return name, pose, None, None, {"frames": n, "decoded": 0, "error": "no frames decoded"}
    some = next(iter(frames.values()))
    H, Wd = some.shape[:2]
    pose[..., 0] *= Wd / PKL_SHAPE[1]
    pose[..., 1] *= H / PKL_SHAPE[0]

    def pair(i):
        a = frames.get(i, frames.get(min(frames), None))
        b = frames.get(i + 1, a)
        # fall back to nearest decoded frame if the pkl frame count exceeds the decodable count
        if a is None:
            a = frames[max(k for k in frames if k <= i)] if any(k <= i for k in frames) else some
        return a, b

    if mode == "dis":
        win = np.zeros((len(idx), pose.shape[1], 2 * W * W), np.float32)
        for t, i in enumerate(idx):
            a, b = pair(int(i))
            fl = _DIS.calc(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
            win[t] = crop_windows(fl, pose[t, :, :2], W, dil)
        return name, pose, win, None, {"frames": n, "decoded": decoded}

    # raft: hand RGB pairs back to the main process (GPU lives there)
    rgb = np.stack([np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in pair(int(i))]) for i in idx])  # (T,2,H,W,3)
    return name, pose, None, rgb, {"frames": n, "decoded": decoded}


# ----------------------------------------------------------------------------- RAFT (GPU path)
class Raft:
    def __init__(self, kind, device, batch=16):
        import torch
        from torchvision.models import optical_flow as of

        self.torch, self.device, self.batch = torch, device, batch
        if kind == "raft_small":
            w = of.Raft_Small_Weights.DEFAULT
            self.model = of.raft_small(weights=w)
        else:
            w = of.Raft_Large_Weights.DEFAULT
            self.model = of.raft_large(weights=w)
        self.tf = w.transforms()
        self.model = self.model.to(device).eval()

    def __call__(self, rgb_pairs):
        """rgb_pairs (T,2,H,W,3) uint8 -> flow (T,H,W,2) float32."""
        torch = self.torch
        x = torch.from_numpy(rgb_pairs).permute(0, 1, 4, 2, 3).float()  # (T,2,3,H,W)
        H, W = x.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8  # RAFT needs dims divisible by 8 (320x240 already is)
        out = []
        with torch.no_grad():
            for s in range(0, len(x), self.batch):
                a, b = self.tf(x[s:s + self.batch, 0], x[s:s + self.batch, 1])
                if ph or pw:
                    a = torch.nn.functional.pad(a, (0, pw, 0, ph))
                    b = torch.nn.functional.pad(b, (0, pw, 0, ph))
                fl = self.model(a.to(self.device), b.to(self.device))[-1][:, :, :H, :W]
                out.append(fl.permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="data/ucf101_hrnet.pkl")
    ap.add_argument("--videos", default=None, help="archive.zip or directory with the .avi files (not needed for --flow none)")
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3], help="official UCF101 split")
    ap.add_argument("--frames", type=int, default=32, help="T sampled frames per clip")
    ap.add_argument("--window", type=int, default=5, help="flow window size W (paper: 5)")
    ap.add_argument("--dilation", type=int, default=1, help="window dilation (paper: 1 for UCF101)")
    ap.add_argument("--flow", default="dis", choices=["dis", "raft_small", "raft_large", "none"])
    ap.add_argument("--dis-preset", default="fast", choices=list(DIS_PRESETS), help="DIS quality/speed (1-thread: ultrafast 2 ms, fast 4 ms, medium 22 ms per pair)")
    ap.add_argument("--device", default="cuda", help="device for RAFT")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default="features")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N clips per split (smoke test)")
    ap.add_argument("--subsets", default="train,test", help="comma list of which halves of the split to build")
    args = ap.parse_args()

    if args.flow != "none" and not args.videos:
        sys.exit("--videos is required unless --flow none")

    print(f"loading {args.pkl} ...", flush=True)
    d = pickle.load(open(args.pkl, "rb"))
    ann = {a["frame_dir"]: a for a in d["annotations"]}
    classes = sorted({n.split("_")[1] for n in ann})  # class name from clip name, 101 entries
    assert len(classes) == 101, len(classes)
    label_of = {n: a["label"] for n, a in ann.items()}
    # sanity: pkl labels are alphabetical class ids, same as `classes`
    for n in list(ann)[:200]:
        assert classes[label_of[n]] == n.split("_")[1], (n, label_of[n])

    src = VideoSource(args.videos) if args.flow != "none" else None
    if src:
        missing = [n for n in ann if n not in src.index]
        print(f"videos found: {len(src.index)}  missing for pkl clips: {len(missing)}", flush=True)

    raft = Raft(args.flow, args.device) if args.flow.startswith("raft") else None
    os.makedirs(args.out, exist_ok=True)
    V, C = 17, 3 + (0 if args.flow == "none" else 2 * args.window ** 2)

    for subset in args.subsets.split(","):
        names = d["split"][f"{subset}{args.split}"]
        if args.limit:
            names = names[: args.limit]
        jobs = [(n, ann[n]["keypoint"], ann[n]["keypoint_score"], args.frames, args.window, args.dilation, args.flow) for n in names]
        X = np.zeros((len(names), args.frames, V, C), np.float16)
        y = np.array([label_of[n] for n in names], np.int64)
        errors, t0 = [], time.time()
        with Pool(args.workers, initializer=_init_worker, initargs=(args.videos, args.dis_preset)) as pool:
            for k, (name, pose, win, rgb, info) in enumerate(pool.imap(process_clip, jobs, chunksize=4)):
                if "error" in info:
                    errors.append((name, info["error"]))
                    win = None
                if raft is not None and rgb is not None:
                    flow = raft(rgb)
                    win = np.stack([crop_windows(flow[t], pose[t, :, :2], args.window, args.dilation) for t in range(len(flow))])
                X[k, :, :, :3] = pose
                if win is not None:
                    X[k, :, :, 3:] = win
                if (k + 1) % 200 == 0 or k + 1 == len(names):
                    el = time.time() - t0
                    print(f"  {subset}{args.split}: {k + 1}/{len(names)}  {el / 60:.1f} min  eta {el / (k + 1) * (len(names) - k - 1) / 60:.1f} min", flush=True)
        fn = os.path.join(args.out, f"split{args.split}_{subset}_T{args.frames}_{args.flow}.npz")
        np.savez(fn, X=X, y=y, names=np.array(names), classes=np.array(classes),
                 frames=args.frames, window=args.window, dilation=args.dilation, flow=args.flow)
        print(f"saved {fn}  X{X.shape} float16  ({os.path.getsize(fn) / 1e6:.0f} MB)  errors: {len(errors)}", flush=True)
        for e in errors[:10]:
            print("   ", e)


if __name__ == "__main__":
    main()
