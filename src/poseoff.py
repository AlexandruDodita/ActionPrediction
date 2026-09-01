"""PoseOFF representation construction (paper Sec. III-A / Fig. 2-3):

For each pose keypoint (x, y), sample an N x N window of optical flow vectors
centred on it (with optional dilation), flatten to 2*N^2 channels, and
concatenate with the (x, y, score) keypoint channels -> C = 2*N^2 + 3 per joint.
Final tensor shape: (T, M, V, C).
"""
import argparse
import glob
import os

import cv2
import numpy as np

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


def sample_flow_window(flow: np.ndarray, x: float, y: float, n: int, dilation: int) -> np.ndarray:
    """Extract an (n, n, 2) window of flow centred at (x, y) with given dilation."""
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


def build_poseoff(keypoints: np.ndarray, scores: np.ndarray, flow: np.ndarray, n: int, dilation: int):
    """keypoints: (V, 2), scores: (V,), flow: (H, W, 2) -> per-joint feature (V, 2*n*n + 3)."""
    V = keypoints.shape[0]
    feats = []
    for v in range(V):
        x, y = keypoints[v]
        window = sample_flow_window(flow, x, y, n, dilation)
        flat = window.reshape(-1)  # 2*n*n
        pose_feat = np.array([x, y, scores[v]], dtype=np.float32)
        feats.append(np.concatenate([pose_feat, flat]))
    return np.stack(feats, axis=0)  # (V, 3 + 2*n*n)


def visualise(frame_bgr, keypoints, scores, flow, n, dilation, out_path, highlight_joints=(8, 14)):
    """Composite figure: frame+skeleton+flow, with zoomed NxN sampling windows for
    a couple of joints, mirroring Fig. 2 / Fig. 3 of the paper."""
    h, w = frame_bgr.shape[:2]

    # 1) frame with skeleton
    frame_vis = frame_bgr.copy()
    for a, b in COCO17_SKELETON:
        if scores[a] > 0.3 and scores[b] > 0.3:
            pa = tuple(keypoints[a].astype(int))
            pb = tuple(keypoints[b].astype(int))
            cv2.line(frame_vis, pa, pb, (0, 255, 255), 2)
    for v in range(len(keypoints)):
        if scores[v] > 0.3:
            cv2.circle(frame_vis, tuple(keypoints[v].astype(int)), 3, (0, 0, 255), -1)

    # 2) flow colour map
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # mark sampling window boxes for highlighted joints on the flow map
    half_extent = (n // 2) * dilation
    for v in highlight_joints:
        if scores[v] > 0.3:
            x, y = keypoints[v].astype(int)
            cv2.rectangle(flow_vis, (x - half_extent, y - half_extent), (x + half_extent, y + half_extent), (255, 255, 255), 1)
            cv2.rectangle(frame_vis, (x - half_extent, y - half_extent), (x + half_extent, y + half_extent), (255, 255, 255), 1)
            cv2.putText(frame_vis, COCO17_NAMES[v], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    top = np.hstack([cv2.resize(frame_vis, (w, h)), cv2.resize(flow_vis, (w, h))])

    # 3) zoomed-in NxN windows for the highlighted joints (upsampled for visibility)
    zoom_tiles = []
    zoom_size = 160
    for v in highlight_joints:
        window = sample_flow_window(flow, keypoints[v][0], keypoints[v][1], n, dilation)
        wmag, wang = cv2.cartToPolar(window[..., 0], window[..., 1])
        whsv = np.zeros((n, n, 3), dtype=np.uint8)
        whsv[..., 1] = 255
        whsv[..., 0] = wang * 180 / np.pi / 2
        whsv[..., 2] = cv2.normalize(wmag, None, 0, 255, cv2.NORM_MINMAX)
        wbgr = cv2.cvtColor(whsv, cv2.COLOR_HSV2BGR)
        wbgr = cv2.resize(wbgr, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
        # draw grid lines for individual cells
        cell = zoom_size // n
        for k in range(1, n):
            cv2.line(wbgr, (k * cell, 0), (k * cell, zoom_size), (80, 80, 80), 1)
            cv2.line(wbgr, (0, k * cell), (zoom_size, k * cell), (80, 80, 80), 1)
        cv2.rectangle(wbgr, (0, 0), (zoom_size - 1, zoom_size - 1), (255, 255, 255), 2)
        label = np.zeros((24, zoom_size, 3), dtype=np.uint8)
        cv2.putText(label, f"{COCO17_NAMES[v]} {n}x{n} flow window (dilation={dilation})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tile = np.vstack([label, wbgr])
        zoom_tiles.append(tile)

    bottom = np.hstack(zoom_tiles)
    # pad bottom to match top width
    if bottom.shape[1] < top.shape[1]:
        pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
        bottom = np.hstack([bottom, pad])
    elif bottom.shape[1] > top.shape[1]:
        bottom = bottom[:, :top.shape[1]]

    composite = np.vstack([top, bottom])
    cv2.imwrite(out_path, composite)
    return composite


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("pose_dir")
    ap.add_argument("flow_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dilation", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    pose_data = np.load(os.path.join(args.pose_dir, "keypoints.npz"), allow_pickle=True)
    flow_data = np.load(os.path.join(args.flow_dir, "flows.npz"), allow_pickle=True)
    flows = flow_data["flows"]
    all_kps = pose_data["keypoints"]
    all_scores = pose_data["scores"]

    all_feats = []
    for t in range(len(flows)):
        frame = cv2.imread(frame_paths[t])
        kps = all_kps[t][0] if len(all_kps[t]) > 0 else np.zeros((17, 2))
        scs = all_scores[t][0] if len(all_scores[t]) > 0 else np.zeros(17)
        flow = flows[t]

        feat = build_poseoff(kps, scs, flow, args.n, args.dilation)  # (V, C)
        all_feats.append(feat)

        out_path = os.path.join(args.out_dir, f"poseoff_{t:03d}.png")
        visualise(frame, kps, scs, flow, args.n, args.dilation, out_path)
        print(f"frame {t}: poseoff feature shape {feat.shape}")

    tensor = np.stack(all_feats, axis=0)[:, None, :, :]  # (T, M=1, V, C)
    print(f"\nFinal PoseOFF tensor shape (T, M, V, C) = {tensor.shape}")
    np.savez(os.path.join(args.out_dir, "poseoff_tensor.npz"), tensor=tensor)
