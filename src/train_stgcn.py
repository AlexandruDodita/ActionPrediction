"""ST-GCN++-style spatial-temporal graph conv net (simplified, uniform spatial
partition) for UCF101 split1, in pose-only and PoseOFF-augmented variants -- the
same comparison as train_eval_quick.py, but with the paper's actual backbone
family (Sec. III-B) instead of a pooled-feature MLP.

PoseOFF embedding (Fig. 4 of the paper): a small CNN reads the 5x5 DIS optical-flow
window around each joint and produces a learned embedding, concatenated with the
(x, y, score) pose channels as extra input channels to the GCN -- no change to the
backbone itself.

Not a full reproduction of PYSKL's ST-GCN++ (no multi-branch bag-of-tricks); this
is the core spatial-configuration-free (uniform) ST-GCN mechanism: per-block
spatial graph conv (1x1 conv + adjacency mix) + temporal conv (9x1) + residual,
stacked and stride-2 downsampled twice, global-average-pooled, linear head.

Usage:
    python src/train_stgcn.py                     # trains pose-only then poseoff, prints both
    python src/train_stgcn.py --variant pose
    python src/train_stgcn.py --variant poseoff
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

V = 17
W = 5  # PoseOFF flow-window side (paper: 5 for UCF101)
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def normalized_adjacency():
    A = np.eye(V, dtype=np.float32)
    for a, b in SKELETON:
        A[a, b] = A[b, a] = 1.0
    d = A.sum(1)
    d_inv_sqrt = np.diag(d ** -0.5)
    return d_inv_sqrt @ A @ d_inv_sqrt  # (V,V)


class SpatialGC(nn.Module):
    """1x1 conv (channel mix) then mix across joints via the normalised adjacency."""

    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A))
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):  # x: (N,C,T,V)
        x = self.conv(x)
        return torch.einsum("nctv,vw->nctw", x, self.A)


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, residual=True, dropout=0.3):
        super().__init__()
        self.gcn = SpatialGC(in_c, out_c, A)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_c),
        )
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.res = None
        elif in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        res = 0 if self.res is None else self.res(x)
        x = self.relu(self.bn1(self.gcn(x)))
        x = self.drop(self.tcn(x))
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_channels, n_classes, A, dropout=0.3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        chans = [(in_channels, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
                 (64, 128, 2, True), (128, 128, 1, True),
                 (128, 256, 2, True), (256, 256, 1, True)]
        self.blocks = nn.ModuleList([STGCNBlock(i, o, A, s, r, dropout) for i, o, s, r in chans])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):  # x: (N,C,T,V)
        N, C, T, Vd = x.shape
        z = x.permute(0, 3, 1, 2).reshape(N, Vd * C, T)
        z = self.data_bn(z)
        z = z.reshape(N, Vd, C, T).permute(0, 2, 3, 1)
        for b in self.blocks:
            z = b(z)
        z = z.mean(dim=[2, 3])
        return self.fc(self.drop(z))


class FlowEmbed(nn.Module):
    """PoseOFF embedding layer (Fig. 4): CNN on the (2,W,W) flow window -> embed_dim vector."""

    def __init__(self, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):  # x: (B,2,W,W)
        return self.fc(self.net(x).flatten(1))


def prep_tensors(X, variant):
    """X: (N,T,V,C) float16 -> pose (N,T,V,3) float32, flow_win (N,T,V,2,W,W) float32 or None.
    Raw tensors only -- the flow window is embedded per-batch inside the train/eval loop so the
    PoseOFF CNN (Fig. 4) actually receives gradients instead of acting as a frozen random projector."""
    pose = torch.from_numpy(X[..., :3].astype(np.float32))
    if variant == "pose":
        return pose, None
    flow = X[..., 3:].astype(np.float32).reshape(*X.shape[:3], W, W, 2)  # (N,T,V,W,W,2)
    flow = torch.from_numpy(flow).permute(0, 1, 2, 5, 3, 4).contiguous()  # (N,T,V,2,W,W)
    return pose, flow


def build_input(pose_b, flow_b, flow_embed, variant="poseoff"):
    """pose_b: (B,T,V,3), flow_b: (B,T,V,2,W,W) or None -> (B,C',T,V) for STGCN.
    variant "pose": pose channels only. "flow": PoseOFF embedding only (no joint coords --
    a pure motion-stream GCN). "poseoff": both concatenated."""
    if flow_b is None:
        z = pose_b
    else:
        B, T, Vd = flow_b.shape[:3]
        embed = flow_embed(flow_b.reshape(B * T * Vd, 2, W, W)).reshape(B, T, Vd, -1)
        z = embed if variant == "flow" else torch.cat([pose_b, embed], dim=-1)
    return z.permute(0, 3, 1, 2).contiguous()  # (B,C',T,V)


def run(variant, train_npz, test_npz, device, epochs, batch_size, embed_dim=16, lr=1e-3, dropout=0.3):
    tr = np.load(train_npz, allow_pickle=True)
    te = np.load(test_npz, allow_pickle=True)
    classes = tr["classes"]
    n_classes = len(classes)
    ytr = torch.from_numpy(tr["y"]).long()
    yte = torch.from_numpy(te["y"]).long()

    A = normalized_adjacency()
    flow_embed = FlowEmbed(embed_dim).to(device) if variant in ("flow", "poseoff") else None
    in_c = {"pose": 3, "flow": embed_dim, "poseoff": 3 + embed_dim}[variant]
    model = STGCN(in_c, n_classes, A, dropout=dropout).to(device)

    params = list(model.parameters()) + (list(flow_embed.parameters()) if flow_embed else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"[{variant}] preparing tensors ...")
    pose_tr, flow_tr = prep_tensors(tr["X"], variant)
    pose_te, flow_te = prep_tensors(te["X"], variant)
    print(f"[{variant}] {len(ytr)} train / {len(yte)} test clips  in_channels={in_c}")

    n = len(ytr)
    best_acc = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if flow_embed is not None:
            flow_embed.train()
        perm = torch.randperm(n)
        tot_loss = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            pb = pose_tr[idx].to(device)
            fb = flow_tr[idx].to(device) if flow_tr is not None else None
            yb = ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(build_input(pb, fb, flow_embed, variant)), yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        if flow_embed is not None:
            flow_embed.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(yte), batch_size):
                pb = pose_te[s:s + batch_size].to(device)
                fb = flow_te[s:s + batch_size].to(device) if flow_te is not None else None
                pred = model(build_input(pb, fb, flow_embed, variant)).argmax(1).cpu()
                correct += (pred == yte[s:s + batch_size]).sum().item()
        acc = correct / len(yte)
        best_acc = max(best_acc, acc)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  [{variant}] epoch {ep+1}/{epochs}  loss={tot_loss/n:.3f}  test_acc={acc*100:.2f}%  "
                  f"best={best_acc*100:.2f}%  ({time.time()-t0:.0f}s)")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/split1_train_T32_dis.npz")
    ap.add_argument("--test", default="features/split1_test_T32_dis.npz")
    ap.add_argument("--variant", choices=["pose", "flow", "poseoff", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    variants = ["pose", "flow", "poseoff"] if args.variant == "all" else [args.variant]
    results = {}
    for v in variants:
        results[v] = run(v, args.train, args.test, device, args.epochs, args.batch_size, args.embed_dim,
                          dropout=args.dropout)

    print("\n=== ST-GCN++ (simplified) accuracy ===")
    for v, acc in results.items():
        print(f"{v:10s} {acc*100:.2f}%")
    if "pose" in results and "poseoff" in results:
        print(f"delta (poseoff - pose): {(results['poseoff']-results['pose'])*100:+.2f} pts")
