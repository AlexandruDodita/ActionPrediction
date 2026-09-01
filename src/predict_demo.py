# trains the quick pose-only model, then shows predictions on N real test clips
import argparse

import numpy as np
import torch

from train_eval_quick import MLP, pool_features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/split1_train_T32_none.npz")
    ap.add_argument("--test", default="features/split1_test_T32_none.npz")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

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

    # pick N random test clips and show real predictions
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yte), size=min(args.n, len(yte)), replace=False)

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t[idx]).argmax(1).cpu().numpy()

    names = te["names"][idx]
    true = yte[idx]

    print(f"{'clip':35s} {'true':20s} {'predicted':20s} {'ok'}")
    print("-" * 85)
    correct = 0
    for i in range(len(idx)):
        ok = pred[i] == true[i]
        correct += ok
        print(f"{str(names[i]):35s} {classes[true[i]]:20s} {classes[pred[i]]:20s} {'yes' if ok else 'no'}")
    print("-" * 85)
    print(f"{correct}/{len(idx)} correct on these {len(idx)} clips")
