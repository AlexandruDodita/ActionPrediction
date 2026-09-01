"""Extract RGB frames from a UCF101 clip for the PoseOFF proof-of-concept."""
import argparse
import os

import cv2


def extract_frames(
    video_path: str, out_dir: str, max_frames: int = 12, start: int = 0, step: int | None = None
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video={video_path} total_frames={total} fps={fps:.2f}")

    if step is None:
        step = max(1, total // max_frames)  # spread across whole clip

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    paths = []
    idx = start
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:03d}.png")
            cv2.imwrite(out_path, frame)
            paths.append(out_path)
            saved += 1
        idx += 1
    cap.release()
    print(f"saved {len(paths)} frames to {out_dir}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("out_dir", type=str)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    extract_frames(args.video, args.out_dir, args.max_frames, args.start, args.step)
