# same as predict_demo.py but burns the prediction, the skeleton, and the sampled
# (key) frames onto the actual video so you can watch the clips instead of reading a table
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import VideoSource, select_person, sample_indices, PKL_SHAPE
from train_eval_quick import MLP, pool_features

# COCO-17 joint connections, for drawing the skeleton
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/split1_train_T32_none.npz")
    ap.add_argument("--test", default="features/split1_test_T32_none.npz")
    ap.add_argument("--videos", default="data/ucf101")
    ap.add_argument("--pkl", default="data/ucf101_hrnet.pkl")
    ap.add_argument("--out", default="predictions_out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"loading {args.pkl} ...")
    d = pickle.load(open(args.pkl, "rb"))
    ann = {a["frame_dir"]: a for a in d["annotations"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = np.load(args.train, allow_pickle=True)
    te = np.load(args.test, allow_pickle=True)
    classes = tr["classes"]

    pose_channels = [0, 1, 2]
    Xtr = pool_features(tr["X"], pose_channels)
    Xte = pool_features(te["X"], pose_channels)
    ytr, yte = tr["y"], te["y"]

    mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)

    model = MLP(Xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(yte), size=min(args.n, len(yte)), replace=False)

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t[idx]).argmax(1).cpu().numpy()

    names = te["names"][idx]
    true = yte[idx]

    src = VideoSource(args.videos)

    for i in range(len(idx)):
        name = str(names[i])
        true_label = classes[true[i]]
        pred_label = classes[pred[i]]
        ok = pred[i] == true[i]
        color = (0, 200, 0) if ok else (0, 0, 220)  # BGR: green if correct, red if wrong

        cap = cv2.VideoCapture(src.index[name])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(args.out, f"{name}_pred.avi")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

        # per-frame keypoints for the whole clip, rescaled to this video's actual size
        kp, sc = ann[name]["keypoint"], ann[name]["keypoint_score"]
        pose_all = select_person(kp, sc)  # (T_full, 17, 3) in 340x256 coords
        pose_all[..., 0] *= w / PKL_SHAPE[1]
        pose_all[..., 1] *= h / PKL_SHAPE[0]
        n_full = pose_all.shape[0]
        key_frames = set(sample_indices(n_full, 32).tolist())  # frames the model actually used

        t = 0
        while True:
            ok_read, frame = cap.read()
            if not ok_read:
                break

            if t < n_full:
                joints = pose_all[t]
                for a, b in SKELETON:
                    if joints[a, 2] > 0.3 and joints[b, 2] > 0.3:
                        pa = tuple(joints[a, :2].astype(int))
                        pb = tuple(joints[b, :2].astype(int))
                        cv2.line(frame, pa, pb, (0, 255, 255), 2)
                for j in range(joints.shape[0]):
                    if joints[j, 2] > 0.3:
                        cv2.circle(frame, tuple(joints[j, :2].astype(int)), 3, (0, 0, 255), -1)

            if t in key_frames:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 4)
                cv2.putText(frame, "KEY FRAME (used for prediction)", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(frame, f"true: {true_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"pred: {pred_label}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            writer.write(frame)
            t += 1
        cap.release()
        writer.release()
        print(f"{name}: true={true_label} pred={pred_label} {'OK' if ok else 'WRONG'} -> {out_path}")
