"""Local deployment: run SwordNet weights on a webcam, video, image or folder.

    python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source 0           # webcam
    python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source fight.mp4 --save
    python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source photos/ --save

Draws the oriented boxes and shows FPS. --save writes annotated output to runs/predict/.
Needs `pip install ultralytics` (get_weights.sh / .ps1 does it for you).
"""
import argparse
import time

import cv2
import numpy as np

COLORS = {0: (58, 102, 224), 1: (191, 143, 47)}  # BGR: sword orange, stick blue


def draw(frame, result, fps=None):
    if result.obb is not None:
        for poly, cls, conf in zip(result.obb.xyxyxyxy.cpu().numpy(), result.obb.cls.tolist(), result.obb.conf.tolist()):
            pts = poly.astype(np.int32)
            c = COLORS.get(int(cls), (0, 255, 0))
            cv2.polylines(frame, [pts], True, c, 2)
            x, y = pts[np.argmin(pts[:, 1])]
            cv2.putText(frame, f"{result.names[int(cls)]} {conf:.2f}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    if fps:
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/swordnet-best.pt")
    p.add_argument("--source", default="0", help="webcam index, video, image or folder")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    if model.task != "obb":
        raise SystemExit(f"{a.weights} is a {model.task} model, not an OBB SwordNet model")

    source = int(a.source) if a.source.isdigit() else a.source
    if isinstance(source, str) and not source.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        # images / folders: let ultralytics handle batching and saving
        for r in model.predict(source, conf=a.conf, imgsz=a.imgsz, save=a.save, stream=True, verbose=False):
            print(f"{r.path}: {0 if r.obb is None else len(r.obb)} objects")
        return

    cap = cv2.VideoCapture(source)
    writer, t_prev, fps = None, time.perf_counter(), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        now = time.perf_counter()
        fps = 1 / (now - t_prev) if fps is None else 0.9 * fps + 0.1 / (now - t_prev)
        t_prev = now
        out = draw(frame, r, fps)
        if a.save:
            if writer is None:
                import os
                os.makedirs("runs/predict", exist_ok=True)
                h, w = out.shape[:2]
                writer = cv2.VideoWriter("runs/predict/swordnet.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(out)
        if not a.no_show:
            cv2.imshow("SwordNet (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
        print("saved runs/predict/swordnet.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
