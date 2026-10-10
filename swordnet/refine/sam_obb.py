"""SAM + contours: turn a loose box into a tight oriented box (OBB).

A sword held diagonally fills maybe 10% of its axis-aligned box; the rest is background that
teaches YOLO the wrong thing. For each box we:
    1. prompt SAM with the box -> a pixel mask of the object
    2. clean the mask (morphological open/close, keep the largest contour)
    3. cv2.minAreaRect on that contour -> 4 rotated corners

If SAM's mask is implausible (empty, or covering most of the box, e.g. it grabbed the arm) we
keep the original box and say why, rather than silently writing a worse label.

Runs on a seat pod through the bridge: images dir + labels JSON in -> refined labels JSON out.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_FILL = 0.01  # mask covers less than 1% of the box: SAM found nothing
MAX_FILL = 0.90  # mask fills the box: SAM segmented background or the person, not the blade


def mask_to_obb(mask):
    """Binary mask -> 4 corners [[x, y]]*4 of the min-area rotated rect, or None."""
    m = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).tolist()


def bbox_of(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


class Refiner:
    def __init__(self, weights="sam2.1_b.pt", device=None):
        from ultralytics import SAM
        self.sam = SAM(weights)
        self.device = device

    def refine(self, image_path, labels):
        """labels: [{cls, points, ...}] -> same list, with points tightened where SAM agrees."""
        if not labels:
            return []
        boxes = [bbox_of(lb["points"]) for lb in labels]
        res = self.sam(str(image_path), bboxes=boxes, device=self.device, verbose=False)[0]
        masks = res.masks.data.cpu().numpy() if res.masks is not None else []
        out = []
        for lb, box, mask in zip(labels, boxes, masks):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            box_area = max(1, (x2 - x1) * (y2 - y1))
            fill = float(mask[max(0, y1):y2, max(0, x1):x2].sum()) / box_area
            obb = mask_to_obb(mask) if MIN_FILL <= fill <= MAX_FILL else None
            new = dict(lb)
            if obb is None:
                new["refine_note"] = f"kept original box: SAM mask fill {fill:.0%} of box"
            else:
                new["points"], new["source"] = obb, f"{lb.get('source', 'manual')}+sam"
            out.append(new)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True, help="JSON {filename: [label, ...]}")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="sam2.1_b.pt")
    a = p.parse_args()

    refiner = Refiner(a.weights)
    todo = json.loads(Path(a.labels).read_text())
    done = {}
    for i, (name, labels) in enumerate(todo.items(), 1):
        done[name] = refiner.refine(Path(a.images) / name, labels)
        kept = sum("refine_note" in lb for lb in done[name])
        print(f"[{i}/{len(todo)}] {name}: {len(labels) - kept} refined, {kept} kept", flush=True)
    Path(a.out).write_text(json.dumps(done))


if __name__ == "__main__":
    main()
