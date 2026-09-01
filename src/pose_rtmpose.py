"""Human pose keypoint extraction using RTMPose (via rtmlib), analogous to the
YOLO-Pose step described in the PoseOFF paper (Sec. III-C) for UCF101, which has
no ground-truth skeleton annotations.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton


def load_body_model(device: str = "cpu"):
    # 'performance' backend uses RTMDet (detector) + RTMPose-l (17 COCO keypoints)
    return Body(
        mode="performance",
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )


def run_on_frames(frame_paths: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    body = load_body_model()

    all_keypoints = []
    for path in frame_paths:
        img = cv2.imread(path)
        keypoints, scores = body(img)
        all_keypoints.append((os.path.basename(path), keypoints, scores))

        vis = img.copy()
        if keypoints.shape[0] > 0:
            vis = draw_skeleton(vis, keypoints, scores, kpt_thr=0.3)
        out_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out_path, vis)
        n_people = keypoints.shape[0]
        print(f"{os.path.basename(path)}: {n_people} person(s) detected")

    return all_keypoints


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", type=str)
    ap.add_argument("out_dir", type=str)
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    results = run_on_frames(frame_paths, args.out_dir)

    kp_path = os.path.join(args.out_dir, "keypoints.npz")
    np.savez(
        kp_path,
        names=[r[0] for r in results],
        keypoints=np.array([r[1] for r in results], dtype=object),
        scores=np.array([r[2] for r in results], dtype=object),
    )
    print(f"saved keypoints to {kp_path}")
