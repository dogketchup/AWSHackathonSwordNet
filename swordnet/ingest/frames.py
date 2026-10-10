"""Frame extractor: sparring video -> still images in INGEST_DIR.

Consecutive frames are near-duplicates, so we sample every `every_s` seconds. Every frame
keeps `video:<name>#<frame>` as its origin; export.py uses that to keep one video entirely in
train or entirely in val (otherwise val mAP is inflated by frames the model has nearly seen).
"""
import argparse
from pathlib import Path

import cv2

from swordnet import config


def extract(video_path, dataset, every_s=0.5, max_frames=2000, out_dir=None, register=True):
    """Sample frames from a video.

    register=True: write them into INGEST_DIR and the local store; returns sample ids.
    register=False: only write the files into out_dir; returns [(path, origin)] for the caller
    to send to the team database.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir or config.INGEST_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_s))
    config.ensure_dirs()
    ids, idx = [], 0
    while len(ids) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            path = out_dir / f"{video_path.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            origin = f"video:{video_path.name}#{idx}"
            if not register:
                ids.append((path, origin))
            else:
                from swordnet.storage import store
                sid = store.add_image(path, dataset, origin)
                if sid:  # None: the same video (or a frame of it) was uploaded before
                    ids.append(sid)
        idx += 1
    cap.release()
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="+")
    p.add_argument("--dataset", default="default")
    p.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    a = p.parse_args()
    for v in a.videos:
        print(f"{v}: {len(extract(v, a.dataset, a.every))} frames")


if __name__ == "__main__":
    main()
