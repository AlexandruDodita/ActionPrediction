"""Bare CLI: play with the pose/flow fusion weight on the 10 demo clips.

Loads the probabilities exported by export_fusion_demo.py (no retraining, no GPU).
combined = w * pose_probs + (1-w) * flow_probs, argmax -> predicted class.

Usage:
    python src/fusion_demo.py --weight 0.58   # one-shot
    python src/fusion_demo.py                 # interactive: type a weight, repeat, q to quit
"""
import argparse
import json

import numpy as np


def show(data, w):
    classes = data["classes"]
    weights = np.array(data["weight_curve"]["weights"])
    accs = data["weight_curve"]["acc"]
    curve_acc = accs[int(np.argmin(np.abs(weights - w)))]

    print(f"\nw={w:.2f} (pose)  {1-w:.2f} (flow)   "
          f"pose-only={data['acc_pose_only']*100:.1f}%  flow-only={data['acc_flow_only']*100:.1f}%  "
          f"fused@w={curve_acc*100:.1f}%  [{data['n_test']} test clips]")

    print(f"{'clip':28s} {'true':16s} {'pose':16s} {'flow':16s} {'combined':16s}")
    n_pose = n_flow = n_comb = 0
    for c in data["demo_clips"]:
        pose_p = np.array(c["pose_probs"])
        flow_p = np.array(c["flow_probs"])
        comb_p = w * pose_p + (1 - w) * flow_p
        true_i, pose_i, flow_i, comb_i = c["true"], int(pose_p.argmax()), int(flow_p.argmax()), int(comb_p.argmax())
        n_pose += pose_i == true_i
        n_flow += flow_i == true_i
        n_comb += comb_i == true_i
        fmt = lambda i: f"{classes[i]}{'*' if i == true_i else ''}"
        print(f"{c['name']:28s} {classes[true_i]:16s} {fmt(pose_i):16s} {fmt(flow_i):16s} {fmt(comb_i):16s}")
    print(f"correct/10: pose={n_pose} flow={n_flow} combined={n_comb}  (* = correct)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo/fusion_demo_data.json")
    ap.add_argument("--weight", type=float, default=None, help="pose weight 0..1; omit for interactive mode")
    args = ap.parse_args()

    data = json.load(open(args.data))

    if args.weight is not None:
        show(data, max(0.0, min(1.0, args.weight)))
    else:
        print("enter pose weight 0..1 (flow gets 1-w), q to quit")
        while True:
            s = input("w> ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                w = max(0.0, min(1.0, float(s)))
            except ValueError:
                print("not a number")
                continue
            show(data, w)
