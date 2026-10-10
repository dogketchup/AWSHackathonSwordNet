"""Frame extractor: sparring video -> still images in INGEST_DIR.

Consecutive frames are near-duplicates, so we sample every `every_s` seconds. Every frame
keeps `video:<name>#<frame>` as its origin; export.py uses that to keep one video entirely in
train or entirely in val (otherwise val mAP is inflated by frames the model has nearly seen).
"""
import argparse
from pathlib import Path

import cv2

from swordnet import config
from swordnet.storage import db


def extract(video_path, dataset, every_s=0.5, max_frames=2000):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_s))
    config.ensure_dirs()
    ids, idx = [], 0
    with db.session() as con:
        while len(ids) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"{video_path.stem}_f{idx:06d}.jpg"
                cv2.imwrite(str(config.INGEST_DIR / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                h, w = frame.shape[:2]
                ids.append(db.add_image(con, dataset, name, f"video:{video_path.name}#{idx}", w, h))
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
