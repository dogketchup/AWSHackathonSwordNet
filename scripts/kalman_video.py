"""Sword boxes on a video, with a Kalman filter that carries each sword through missed frames.

The detector finds swords; ByteTrack gives each one an ID; a constant-velocity Kalman filter per ID
(state: centre x, y, width, height and their speeds) smooths the box and, when a frame is blurry and
the detector misses the sword, predicts where it is for up to --coast frames. Predicted boxes are
drawn dashed and orange; detected ones solid and green.

    python scripts/kalman_video.py VIDEO --weights best.pt [--out out.mp4] [--coast 15] [--conf 0.25]
    python scripts/kalman_video.py VIDEO --weights yoloe-11l-seg.pt --prompts "a sword|a hockey stick|a weapon"
Works with the team's YOLO / YOLO-OBB weights (rotated boxes are drawn as their upright bounds) and YOLOE.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


class Kalman:
    """Constant-velocity filter on [cx, cy, w, h]."""

    def __init__(self, box):
        k = cv2.KalmanFilter(8, 4)
        k.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            k.transitionMatrix[i, i + 4] = 1.0
            k.transitionMatrix[i + 4, i + 4] = 0.8  # speed fades while coasting, so boxes don't fly off
        k.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        k.processNoiseCov = np.diag([1, 1, 1, 1, 10, 10, 5, 5]).astype(np.float32) * 1e-1
        k.measurementNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        k.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        k.statePost = np.array([*box, 0, 0, 0, 0], np.float32).reshape(8, 1)
        self.k, self.missed = k, 0

    def predict(self):
        return self.k.predict()[:4, 0]

    def correct(self, box):
        self.missed = 0
        return self.k.correct(np.array(box, np.float32).reshape(4, 1))[:4, 0]


def boxes_of(r):
    """(track ids, [cx, cy, w, h]) from a tracking result, for detect or OBB models."""
    b = r.obb if getattr(r, "obb", None) is not None else r.boxes
    if b is None or b.id is None:
        return [], []
    if getattr(r, "obb", None) is not None:
        xyxy = b.xyxyxyxy.cpu().numpy()  # 4 corners -> upright bounds
        xyxy = np.stack([xyxy[:, :, 0].min(1), xyxy[:, :, 1].min(1), xyxy[:, :, 0].max(1), xyxy[:, :, 1].max(1)], 1)
    else:
        xyxy = b.xyxy.cpu().numpy()
    cxcywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                       xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
    return b.id.int().tolist(), cxcywh.tolist()


def dashed_rect(img, p1, p2, color, t=3, dash=14):
    (x1, y1), (x2, y2) = p1, p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, color, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompts", help="YOLOE only: phrases separated by |")
    ap.add_argument("--out")
    ap.add_argument("--coast", type=int, default=6, help="frames to keep predicting a missed sword (6 = 0.2 s at 30 fps)")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    from ultralytics import YOLO, YOLOE
    if a.prompts:
        model = YOLOE(a.weights)
        texts = a.prompts.split("|")
        model.set_classes(texts, model.get_text_pe(texts))
    else:
        model = YOLO(a.weights)
    out = Path(a.out or Path(a.video).with_name(Path(a.video).stem + "_kalman.mp4"))
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracks, n, with_det, with_any = {}, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, conf=a.conf, tracker="bytetrack.yaml", verbose=False)[0]
        ids, meas = boxes_of(r)
        for tid in list(tracks):
            tracks[tid].pred = tracks[tid].predict()
        for tid, m in zip(ids, meas):
            if tid not in tracks:
                tracks[tid] = Kalman(m)
                tracks[tid].predict()
            box, solid = tracks[tid].correct(m), True
            cx, cy, bw, bh = box
            cv2.rectangle(frame, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (60, 200, 60), 3)
        seen = set(ids)
        for tid in list(tracks):
            if tid in seen:
                continue
            t = tracks[tid]
            t.missed += 1
            if t.missed > a.coast:
                del tracks[tid]
                continue
            cx, cy, bw, bh = t.pred
            dashed_rect(frame, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (0, 165, 255))
        shown = len(seen) + sum(1 for tid in tracks if tid not in seen)
        with_det += bool(seen)
        with_any += bool(shown)
        cv2.putText(frame, f"frame {n}: {len(seen)} detected, {shown - len(seen)} predicted", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(frame)
        n += 1
    vw.release()
    print(f"{n} frames: a sword box in {with_det} frames from detections alone, {with_any} with the Kalman filter -> {out}")


if __name__ == "__main__":
    main()
