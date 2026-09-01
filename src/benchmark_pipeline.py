"""End-to-end throughput benchmark: decode -> PoseTracker (RTMPose) -> batched
RAFT (fp16) -> PoseOFF tensor, on real UCF101 clips. No per-frame disk writes.
"""
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from rtmlib import Body, PoseTracker
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

FLOW_BATCH = 32


def sample_flow_window(flow, x, y, n, dilation):
    h, w = flow.shape[:2]
    half = n // 2
    window = np.zeros((n, n, 2), dtype=np.float32)
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            sx = int(round(x + j * dilation))
            sy = int(round(y + i * dilation))
            if 0 <= sx < w and 0 <= sy < h:
                window[i + half, j + half] = flow[sy, sx]
    return window


def compute_flow_batched(frames_rgb, flow_model, transforms, device):
    """frames_rgb: list of (H,W,3) uint8 arrays. Returns list of (H,W,2) flow arrays
    for each consecutive pair, resized back to original resolution."""
    N = len(frames_rgb) - 1
    h, w = frames_rgb[0].shape[:2]
    flows = []
    for start in range(0, N, FLOW_BATCH):
        idxs = list(range(start, min(start + FLOW_BATCH, N)))
        t1 = torch.stack([torch.from_numpy(frames_rgb[i]).permute(2, 0, 1) for i in idxs])
        t2 = torch.stack([torch.from_numpy(frames_rgb[i + 1]).permute(2, 0, 1) for i in idxs])
        t1, t2 = transforms(t1, t2)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            batch_flow = flow_model(t1.to(device), t2.to(device))[-1]
        batch_flow = batch_flow.float().permute(0, 2, 3, 1).cpu().numpy()
        fh, fw = batch_flow.shape[1:3]
        for i in range(batch_flow.shape[0]):
            fl = cv2.resize(batch_flow[i], (w, h))
            fl[..., 0] *= w / fw
            fl[..., 1] *= h / fh
            flows.append(fl)
    return flows


def process_video(path, body, flow_model, transforms, device, n=5, dilation=1):
    body.reset()
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    T = len(frames)
    if T < 2:
        return None, T

    all_kps, all_scs = [], []
    for f in frames:
        kp, sc = body(f)
        all_kps.append(kp[0] if len(kp) > 0 else np.zeros((17, 2), dtype=np.float32))
        all_scs.append(sc[0] if len(sc) > 0 else np.zeros(17, dtype=np.float32))

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    flows = compute_flow_batched(frames_rgb, flow_model, transforms, device)

    feats = []
    for t in range(T - 1):
        kps, scs = all_kps[t], all_scs[t]
        flow = flows[t]
        joint_feats = []
        for v in range(len(kps)):
            win = sample_flow_window(flow, kps[v][0], kps[v][1], n, dilation)
            joint_feats.append(np.concatenate([[kps[v][0], kps[v][1], scs[v]], win.reshape(-1)]))
        feats.append(np.stack(joint_feats))

    tensor = np.stack(feats)[:, None, :, :]
    return tensor, T


if __name__ == "__main__":
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    random.seed(0)
    files = glob.glob("data/ucf101/train/*/*.avi") + glob.glob("data/ucf101/test/*/*.avi") + glob.glob("data/ucf101/val/*/*.avi")
    sample = random.sample(files, n_clips)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    body = PoseTracker(Body, det_frequency=10, tracking=True, mode="lightweight",
                        backend="onnxruntime", device=device)
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    flow_model = raft_small(weights=weights).to(device).eval()

    process_video(sample[0], body, flow_model, transforms, device)  # warmup

    total_frames = 0
    t0 = time.time()
    for i, path in enumerate(sample):
        vt0 = time.time()
        tensor, T = process_video(path, body, flow_model, transforms, device)
        total_frames += T
        print(f"[{i+1}/{n_clips}] {os.path.basename(path)}: {T} frames in {time.time()-vt0:.2f}s"
              f" (shape={None if tensor is None else tensor.shape})")
    total_time = time.time() - t0

    print(f"\n=== {n_clips} clips, {total_frames} frames, {total_time:.1f}s total ===")
    print(f"avg {total_time/n_clips:.2f} s/clip, {total_frames/total_time:.1f} frames/s")

    N_TOTAL_CLIPS = 13451
    est_total_s = (total_time / n_clips) * N_TOTAL_CLIPS
    print(f"\nEstimated full dataset ({N_TOTAL_CLIPS} clips): {est_total_s/3600:.2f} hours ({est_total_s/60:.1f} min)")
    print(f"Clips processable in 30 min at this rate: {int(30*60 / (total_time/n_clips))}")
