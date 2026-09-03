"""Recompute the video-demo confidence numbers using ST-GCN++ instead of the MLP:
trains a pose-only ST-GCN++ and a flow-only ST-GCN++ (PoseOFF embedding, no pose
coordinates concatenated -- a pure motion-stream GCN) on split1, gets softmax probs
for the same 20 demo clips (same rng seed/order as export_video_demo.py, so the
already-rendered gifs in demo/videos/ stay valid), and rewrites demo/fusion_demo_data.json.
Run src/gen_video_demo_html.py afterwards to regenerate demo/video_demo.html.

Usage:
    python src/export_video_demo_stgcn.py
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from train_stgcn import STGCN, FlowEmbed, normalized_adjacency, prep_tensors, W


def train_flowonly(pose_tr_unused, flow_tr, ytr, flow_te, yte, n_classes, device, epochs, batch_size, embed_dim, lr=1e-3):
    A = normalized_adjacency()
    flow_embed = FlowEmbed(embed_dim).to(device)
    model = STGCN(embed_dim, n_classes, A).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(flow_embed.parameters()), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = len(ytr)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); flow_embed.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            fb = flow_tr[idx].to(device)
            B, T, Vd = fb.shape[:3]
            emb = flow_embed(fb.reshape(B * T * Vd, 2, W, W)).reshape(B, T, Vd, -1).permute(0, 3, 1, 2).contiguous()
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(emb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  [flow-stgcn] epoch {ep+1}/{epochs} loss={tot/n:.3f} ({time.time()-t0:.0f}s)")

    model.eval(); flow_embed.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(yte), batch_size):
            fb = flow_te[s:s + batch_size].to(device)
            B, T, Vd = fb.shape[:3]
            emb = flow_embed(fb.reshape(B * T * Vd, 2, W, W)).reshape(B, T, Vd, -1).permute(0, 3, 1, 2).contiguous()
            probs.append(torch.softmax(model(emb), 1).cpu())
    return torch.cat(probs).numpy()


def train_poseonly(pose_tr, ytr, pose_te, yte, n_classes, device, epochs, batch_size, lr=1e-3):
    A = normalized_adjacency()
    model = STGCN(3, n_classes, A).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = len(ytr)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            pb = pose_tr[idx].to(device).permute(0, 3, 1, 2).contiguous()
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(pb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  [pose-stgcn] epoch {ep+1}/{epochs} loss={tot/n:.3f} ({time.time()-t0:.0f}s)")

    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(yte), batch_size):
            pb = pose_te[s:s + batch_size].to(device).permute(0, 3, 1, 2).contiguous()
            probs.append(torch.softmax(model(pb), 1).cpu())
    return torch.cat(probs).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/split1_train_T32_dis.npz")
    ap.add_argument("--test", default="features/split1_test_T32_dis.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--n-demo", type=int, default=20)
    ap.add_argument("--out-json", default="demo/fusion_demo_data.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tr = np.load(args.train, allow_pickle=True)
    te = np.load(args.test, allow_pickle=True)
    classes = [str(c) for c in tr["classes"]]
    n_classes = len(classes)
    ytr_np, yte_np = tr["y"], te["y"]
    ytr, yte = torch.from_numpy(ytr_np).long(), torch.from_numpy(yte_np).long()

    pose_tr, flow_tr = prep_tensors(tr["X"], "poseoff")
    pose_te, flow_te = prep_tensors(te["X"], "poseoff")

    print("training pose-only ST-GCN++ ...")
    pose_probs = train_poseonly(pose_tr, ytr, pose_te, yte, n_classes, device, args.epochs, args.batch_size)
    print("training flow-only ST-GCN++ (PoseOFF embedding, no pose coords) ...")
    flow_probs = train_flowonly(pose_tr, flow_tr, ytr, flow_te, yte, n_classes, device, args.epochs, args.batch_size, args.embed_dim)

    acc_pose_only = float((pose_probs.argmax(1) == yte_np).mean())
    acc_flow_only = float((flow_probs.argmax(1) == yte_np).mean())
    print(f"\npose-only ST-GCN++: {acc_pose_only*100:.2f}%")
    print(f"flow-only ST-GCN++: {acc_flow_only*100:.2f}%")

    rng = np.random.default_rng(0)  # same selection as export_video_demo.py -> same 20 clips/gifs
    idx = rng.choice(len(yte_np), size=min(args.n_demo, len(yte_np)), replace=False)

    old = json.load(open(args.out_json))
    old_names = [c["name"] for c in old["demo_clips"]]
    new_names = [str(te["names"][i]) for i in idx]
    assert old_names == new_names, "clip selection changed -- gifs in demo/videos/ would no longer match"

    demo_clips = [
        {"name": str(te["names"][i]), "true": int(yte_np[i]),
         "pose_probs": pose_probs[i].round(5).tolist(), "flow_probs": flow_probs[i].round(5).tolist()}
        for i in idx
    ]

    out = {"classes": classes, "n_test": int(len(yte_np)), "acc_pose_only": acc_pose_only,
           "acc_flow_only": acc_flow_only, "demo_clips": demo_clips, "model": "ST-GCN++ (simplified)"}
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"\nsaved {args.out_json} (gifs in demo/videos/ reused, unchanged)")
