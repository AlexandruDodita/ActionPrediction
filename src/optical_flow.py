"""Optical flow computation for the PoseOFF proof-of-concept.

Primary: RAFT (torchvision.models.optical_flow.raft_small, pretrained), matching
the paper's use of RAFT (Sec. III-C). Falls back to OpenCV's DISOpticalFlow when
RAFT is unavailable or too slow (e.g. no GPU / large-scale batch processing).
"""
import argparse
import glob
import os

import cv2
import numpy as np


def compute_flow_raft(frame_paths: list[str], out_dir: str, device: str = "cpu"):
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    os.makedirs(out_dir, exist_ok=True)
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=True).to(device).eval()

    flows = []
    for i in range(len(frame_paths) - 1):
        img1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(frame_paths[i + 1]), cv2.COLOR_BGR2RGB)
        h, w = img1.shape[:2]

        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).contiguous()
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).contiguous()
        t1, t2 = transforms(t1, t2)  # expects uint8 [0,255] -> normalises to [-1,1]

        with torch.no_grad():
            flow_preds = model(t1.to(device), t2.to(device))
        flow = flow_preds[-1][0].permute(1, 2, 0).cpu().numpy()  # (H', W', 2)

        # resize flow back to original frame size, scaling vectors accordingly
        fh, fw = flow.shape[:2]
        flow_resized = cv2.resize(flow, (w, h))
        flow_resized[..., 0] *= w / fw
        flow_resized[..., 1] *= h / fh

        flows.append(flow_resized)
        vis = flow_to_color(flow_resized)
        name = os.path.basename(frame_paths[i]).replace(".png", "_flow.png")
        cv2.imwrite(os.path.join(out_dir, name), vis)
        print(f"RAFT flow {i}->{i+1} done, max_mag={np.linalg.norm(flow_resized, axis=-1).max():.2f}")

    return flows


def compute_flow_dis(frame_paths: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    flows = []
    for i in range(len(frame_paths) - 1):
        g1 = cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.imread(frame_paths[i + 1]), cv2.COLOR_BGR2GRAY)
        flow = dis.calc(g1, g2, None)
        flows.append(flow)
        vis = flow_to_color(flow)
        name = os.path.basename(frame_paths[i]).replace(".png", "_flow.png")
        cv2.imwrite(os.path.join(out_dir, name), vis)
        print(f"DIS flow {i}->{i+1} done, max_mag={np.linalg.norm(flow, axis=-1).max():.2f}")

    return flows


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """HSV visualisation: hue=direction, value=magnitude (standard flow colour wheel)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", type=str)
    ap.add_argument("out_dir", type=str)
    ap.add_argument("--method", choices=["raft", "dis"], default="raft")
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))

    if args.method == "raft":
        try:
            flows = compute_flow_raft(frame_paths, args.out_dir)
        except Exception as e:
            print(f"RAFT failed ({e}), falling back to OpenCV DIS")
            flows = compute_flow_dis(frame_paths, args.out_dir)
    else:
        flows = compute_flow_dis(frame_paths, args.out_dir)

    npz_path = os.path.join(args.out_dir, "flows.npz")
    np.savez(npz_path, flows=np.array(flows))
    print(f"saved {len(flows)} flow fields to {npz_path}")
