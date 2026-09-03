"""Render pose-skeleton / optical-flow / combined GIFs for 20 test clips, plus their
per-clip classifier confidence (pose-only, flow-only), for the plain HTML demo.

Trains the same pose-only and flow-only MLPs as export_fusion_demo.py (not saved to
disk elsewhere, so retrained here), picks 20 test clips, and for each one:
  - decodes every frame of the original clip
  - draws the HRNet skeleton on every frame           -> demo/videos/{name}_pose.gif
  - colorizes DIS optical flow (whole clip, consecutive pairs) -> {name}_flow.gif
  - skeleton drawn on top of the flow colors           -> {name}_combined.gif
  - PoseOFF sampling-grid view: skeleton over the dimmed flow map, with the 5x5 window
    around each confident joint at full brightness (what the model actually sees) -> {name}_grid.gif

Usage:
    python src/export_video_demo.py
    python src/export_video_demo.py --gifs-only   # re-render gifs for the clips already in
                                                  # demo/fusion_demo_data.json, without retraining
                                                  # or touching the json (keeps ST-GCN numbers)
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image

from build_dataset import VideoSource, select_person, PKL_SHAPE
from optical_flow import flow_to_color
from train_eval_quick import MLP, pool_features
from export_fusion_demo import train_and_predict

SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def draw_skeleton(frame, kp, score, thresh=0.3):
    img = frame.copy()
    for a, b in SKELETON:
        if score[a] > thresh and score[b] > thresh:
            pa, pb = tuple(kp[a].astype(int)), tuple(kp[b].astype(int))
            cv2.line(img, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)
    for i in range(len(kp)):
        if score[i] > thresh:
            cv2.circle(img, tuple(kp[i].astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def render_grid(flow_color, pose_t, n=5, dilation=1, thresh=0.3):
    """PoseOFF sampling-grid view: skeleton on the dimmed flow map, with the NxN window
    around each confident joint kept at full brightness and boxed -- the flow the model sees."""
    h, w = flow_color.shape[:2]
    out = draw_skeleton((flow_color * 0.25).astype(np.uint8), pose_t[:, :2], pose_t[:, 2])
    half = (n // 2) * dilation
    for v in range(len(pose_t)):
        if pose_t[v, 2] <= thresh:
            continue
        x, y = int(round(pose_t[v, 0])), int(round(pose_t[v, 1]))
        x0, x1 = max(x - half, 0), min(x + half + 1, w)
        y0, y1 = max(y - half, 0), min(y + half + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        out[y0:y1, x0:x1] = flow_color[y0:y1, x0:x1]
        cv2.rectangle(out, (x - half - 1, y - half - 1), (x + half + 1, y + half + 1), (255, 255, 255), 1)
    return out


def decode_all(src, name, max_frames=2000):
    """Decode every frame of a clip via VideoSource (works for both zip and directory)."""
    got, _ = src.read_frames(name, set(range(max_frames)))
    return [got[i] for i in sorted(got)]


def save_gif(frames_bgr, path, fps, stride=2):
    frames = frames_bgr[::stride]
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    dur = int(1000 / fps * stride)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/ucf101_hrnet.pkl")
    ap.add_argument("--videos", default="data/ucf101")
    ap.add_argument("--train", default="features/split1_train_T32_dis.npz")
    ap.add_argument("--test", default="features/split1_test_T32_dis.npz")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--n-demo", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--dis-preset", default="fast")
    ap.add_argument("--gif-stride", type=int, default=2, help="keep every Nth frame in the gif")
    ap.add_argument("--out-json", default="demo/fusion_demo_data.json")
    ap.add_argument("--out-videos", default="demo/videos")
    ap.add_argument("--gifs-only", action="store_true",
                    help="only (re-)render the gifs for the clips listed in --out-json; no training, json untouched")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_videos, exist_ok=True)

    print("loading pkl + features ...")
    d = pickle.load(open(args.pkl, "rb"))
    ann = {a["frame_dir"]: a for a in d["annotations"]}

    if args.gifs_only:
        names = [c["name"] for c in json.load(open(args.out_json))["demo_clips"]]
    tr = np.load(args.train, allow_pickle=True)
    te = np.load(args.test, allow_pickle=True)
    classes = [str(c) for c in tr["classes"]]
    n_classes = len(classes)
    ytr, yte = tr["y"], te["y"]

    pose_ch = [0, 1, 2]
    flow_ch = list(range(3, tr["X"].shape[-1]))

    if args.gifs_only:
        pose_probs = flow_probs = None
    else:
        print("training pose-only MLP ...")
        pose_probs = train_and_predict(
            pool_features(tr["X"], pose_ch), ytr, pool_features(te["X"], pose_ch),
            n_classes, device, args.epochs, "pose",
        )
        print("training flow-only MLP ...")
        flow_probs = train_and_predict(
            pool_features(tr["X"], flow_ch), ytr, pool_features(te["X"], flow_ch),
            n_classes, device, args.epochs, "flow",
        )
        acc_pose_only = float((pose_probs.argmax(1) == yte).mean())
        acc_flow_only = float((flow_probs.argmax(1) == yte).mean())
        print(f"pose-only {acc_pose_only*100:.1f}%  flow-only {acc_flow_only*100:.1f}%  (full {len(yte)}-clip test set)")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(yte), size=min(args.n_demo, len(yte)), replace=False)
    if args.gifs_only:
        assert [str(te["names"][i]) for i in idx] == names, "clip selection no longer matches the json"

    src = VideoSource(args.videos)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if args.dis_preset == "medium"
                                     else cv2.DISOPTICAL_FLOW_PRESET_FAST)

    demo_clips = []
    for n, i in enumerate(idx):
        name = str(te["names"][i])
        print(f"[{n+1}/{len(idx)}] {name}")
        kp = ann[name]["keypoint"]
        sc = ann[name]["keypoint_score"]
        pose_all = select_person(kp, sc)  # (T,V,3) in 340x256 coords

        frames = decode_all(src, name)
        fps = 25.0

        n_f = min(len(frames), pose_all.shape[0])
        frames = frames[:n_f]
        H, Wd = frames[0].shape[:2]
        pose = pose_all[:n_f].copy()
        pose[..., 0] *= Wd / PKL_SHAPE[1]
        pose[..., 1] *= H / PKL_SHAPE[0]

        pose_frames = [draw_skeleton(frames[t], pose[t, :, :2], pose[t, :, 2]) for t in range(n_f)]
        save_gif(pose_frames, os.path.join(args.out_videos, f"{name}_pose.gif"), fps, args.gif_stride)

        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flow_frames, combined_frames, grid_frames = [], [], []
        prev_flow_color = np.zeros_like(frames[0])
        for t in range(n_f):
            if t < n_f - 1:
                fl = dis.calc(gray[t], gray[t + 1], None)
                prev_flow_color = flow_to_color(fl)
            flow_frames.append(prev_flow_color)
            combined_frames.append(draw_skeleton(prev_flow_color, pose[t, :, :2], pose[t, :, 2]))
            grid_frames.append(render_grid(prev_flow_color, pose[t]))
        save_gif(flow_frames, os.path.join(args.out_videos, f"{name}_flow.gif"), fps, args.gif_stride)
        save_gif(combined_frames, os.path.join(args.out_videos, f"{name}_combined.gif"), fps, args.gif_stride)
        save_gif(grid_frames, os.path.join(args.out_videos, f"{name}_grid.gif"), fps, args.gif_stride)

        if args.gifs_only:
            continue
        demo_clips.append({
            "name": name,
            "true": int(yte[i]),
            "pose_probs": pose_probs[i].round(5).tolist(),
            "flow_probs": flow_probs[i].round(5).tolist(),
        })

    if args.gifs_only:
        print(f"\nre-rendered {len(idx)*4} gifs in {args.out_videos}/ (json untouched)")
        raise SystemExit(0)

    out = {
        "classes": classes,
        "n_test": int(len(yte)),
        "acc_pose_only": acc_pose_only,
        "acc_flow_only": acc_flow_only,
        "demo_clips": demo_clips,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"\nsaved {args.out_json} and {len(demo_clips)*4} gifs in {args.out_videos}/")
