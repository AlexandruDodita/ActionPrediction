"""Quick baseline: pose-only vs PoseOFF (pose + flow window) classification accuracy
on UCF101, using the (T,17,C) features from build_dataset.py.

This is a fast proxy (small MLP, mean/std pooling over T) to compare pose-only vs
PoseOFF within a tight time budget -- NOT a reproduction of the paper's InfoGCN++/
MS-G3D/ST-GCN++ results, which need full graph-conv training over many epochs.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn


def pool_features(X, channels):
    """X: (N, T, V, C) -> (N, V*len(channels)*2) via mean+std over T."""
    Xc = X[..., channels].astype(np.float32)  # (N,T,V,c)
    mean = Xc.mean(axis=1)  # (N,V,c)
    std = Xc.std(axis=1)
    feat = np.concatenate([mean, std], axis=-1)  # (N,V,2c)
    return feat.reshape(feat.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xte, yte, n_classes, device, epochs=60, name=""):
    mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sigma
    Xte = (Xte - mu) / sigma

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)

    model = MLP(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr_t)
        loss = loss_fn(out, ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).argmax(1)
            acc = (pred == yte_t).float().mean().item()
        best_acc = max(best_acc, acc)
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"  [{name}] epoch {ep+1}/{epochs} loss={loss.item():.3f} test_acc={acc*100:.2f}% best={best_acc*100:.2f}%")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = np.load(args.train, allow_pickle=True)
    te = np.load(args.test, allow_pickle=True)
    classes = tr["classes"]
    n_classes = len(classes)
    print(f"train clips: {tr['X'].shape[0]}  test clips: {te['X'].shape[0]}  classes: {n_classes}")
    print(f"present in train set: {len(set(tr['y'].tolist()))} / {n_classes} classes, "
          f"test set: {len(set(te['y'].tolist()))} / {n_classes} classes")

    results = {}
    t0 = time.time()
    n_ch = tr["X"].shape[-1]
    pose_channels = [0, 1, 2]

    acc_pose = train_eval(
        pool_features(tr["X"], pose_channels), tr["y"],
        pool_features(te["X"], pose_channels), te["y"],
        n_classes, device, args.epochs, name="pose-only",
    )
    results["pose-only (x,y,score)"] = acc_pose

    # no flow channels present (--flow none) -> nothing else to compare
    if n_ch > 3:
        flow_channels = list(range(3, n_ch))
        acc_flow = train_eval(
            pool_features(tr["X"], flow_channels), tr["y"],
            pool_features(te["X"], flow_channels), te["y"],
            n_classes, device, args.epochs, name="flow-only",
        )
        results["flow-only (optical flow)"] = acc_flow

        all_channels = list(range(n_ch))
        acc_poseoff = train_eval(
            pool_features(tr["X"], all_channels), tr["y"],
            pool_features(te["X"], all_channels), te["y"],
            n_classes, device, args.epochs, name="PoseOFF",
        )
        results["PoseOFF (pose + flow)"] = acc_poseoff

    print(f"\ndone in {time.time()-t0:.1f}s")
    print("\n=== Accuracy table (quick MLP baseline, mean/std pooled features) ===")
    print(f"{'Variant':40s} {'Test Accuracy':>15s}")
    print("-" * 56)
    for k, v in results.items():
        print(f"{k:40s} {v*100:14.2f}%")
    if "PoseOFF (pose + flow)" in results:
        delta = (results["PoseOFF (pose + flow)"] - results["pose-only (x,y,score)"]) * 100
        print("-" * 56)
        print(f"{'Delta (PoseOFF - pose-only)':40s} {delta:+14.2f}%")
