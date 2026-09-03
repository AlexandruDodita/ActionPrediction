"""Train a pose-only MLP and a flow-only MLP on the DIS features, then export
softmax probabilities so a static frontend can late-fuse them with an adjustable
weight (no server needed): combined = w * pose_probs + (1-w) * flow_probs.

Also sweeps the fusion weight over the full test set to find where combining
pose + optical flow actually beats either signal alone.

Usage:
    python src/export_fusion_demo.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from train_eval_quick import MLP, pool_features


def train_and_predict(Xtr, ytr, Xte, n_classes, device, epochs, name):
    mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sigma
    Xte_n = (Xte - mu) / sigma

    Xtr_t = torch.from_numpy(Xtr_n).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte_n).float().to(device)

    model = MLP(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"  [{name}] epoch {ep+1}/{epochs} loss={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=1).cpu().numpy()
    return probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/split1_train_T32_dis.npz")
    ap.add_argument("--test", default="features/split1_test_T32_dis.npz")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-demo", type=int, default=10)
    ap.add_argument("--out", default="demo/fusion_demo_data.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tr = np.load(args.train, allow_pickle=True)
    te = np.load(args.test, allow_pickle=True)
    classes = [str(c) for c in tr["classes"]]
    n_classes = len(classes)
    ytr, yte = tr["y"], te["y"]

    pose_ch = [0, 1, 2]
    flow_ch = list(range(3, tr["X"].shape[-1]))

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

    weights = np.linspace(0, 1, 51)  # weight on pose; (1-w) on flow
    accs = []
    for w in weights:
        combined = w * pose_probs + (1 - w) * flow_probs
        accs.append(float((combined.argmax(1) == yte).mean()))
    best_i = int(np.argmax(accs))

    print(f"\npose-only:  {acc_pose_only*100:.2f}%")
    print(f"flow-only:  {acc_flow_only*100:.2f}%")
    print(f"best fused: {accs[best_i]*100:.2f}% at pose weight w={weights[best_i]:.2f}")

    rng = np.random.default_rng(0)  # same 10 clips as predict_demo.py
    idx = rng.choice(len(yte), size=min(args.n_demo, len(yte)), replace=False)

    demo_clips = [
        {
            "name": str(te["names"][i]),
            "true": int(yte[i]),
            "pose_probs": pose_probs[i].round(5).tolist(),
            "flow_probs": flow_probs[i].round(5).tolist(),
        }
        for i in idx
    ]

    out = {
        "classes": classes,
        "n_test": int(len(yte)),
        "acc_pose_only": acc_pose_only,
        "acc_flow_only": acc_flow_only,
        "weight_curve": {"weights": weights.round(3).tolist(), "acc": [round(a, 5) for a in accs]},
        "demo_clips": demo_clips,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nsaved {args.out}  ({os.path.getsize(args.out)/1e3:.0f} KB)")
