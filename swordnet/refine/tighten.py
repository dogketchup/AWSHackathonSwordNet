"""Annotator, pod side: tighten each human box into an oriented box (scheduler plan, Part 0).

For each human box:
    1. SAM gets the box as its prompt and returns a mask of the main thing inside it.
    2. The mask is cut to the box grown by MARGIN, so SAM cannot spill onto the arm or background.
    3. If the mask covers less than MIN_FILL or more than MAX_FILL of the box, SAM failed: keep the
       human box and say why.
    4. Morphological open + close, then keep only the biggest contour.
    5. cv2.minAreaRect on that contour -> 4 corners. These replace the box's shape.

The class is never touched, and no box is ever dropped: every input label id appears in the
output, either tightened or kept with a note.

    python tighten.py --images DIR --labels in.json --out out.json [--weights sam2.1_b.pt]
    in.json:  {filename: [{"id": 7, "cls": 0, "points": [[x, y]] * 4}, ...]}
    out.json: {filename: {"7": {"points": [[x, y]] * 4, "tightened": true, "note": ""}}}
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_FILL = 0.01
MAX_FILL = 0.90
MARGIN = 0.05


def bounds(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _grow(box, margin, w, h):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
            min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy))))


def clip_to_box(mask, box, margin=MARGIN):
    """Zero the mask outside the box grown by `margin` (a fraction of the box size)."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = _grow(box, margin, w, h)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0)
    return out


def mask_to_obb(mask):
    """Biggest contour of the cleaned mask -> 4 corners of its min-area rotated rect, or None."""
    m = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 4:
        return None
    return [[float(x), float(y)] for x, y in cv2.boxPoints(cv2.minAreaRect(biggest))]


def tighten_box(mask, points, margin=MARGIN, min_fill=MIN_FILL, max_fill=MAX_FILL):
    """One human box + SAM's mask for it -> {"points", "tightened", "note"}. Never None."""
    keep = {"points": [[float(x), float(y)] for x, y in points], "tightened": False}
    if mask is None:
        return {**keep, "note": "kept human box: SAM returned no mask"}
    box = bounds(points)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    area = max(1, (x2 - x1) * (y2 - y1))
    m = clip_to_box(mask, box, margin)
    fill = float(m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum()) / area
    if fill < min_fill:
        return {**keep, "note": f"kept human box: SAM mask covers {fill:.1%} of it (found nothing)"}
    if fill > max_fill:
        return {**keep, "note": f"kept human box: SAM mask covers {fill:.0%} of it (grabbed background or a person)"}
    obb = mask_to_obb(m)
    if obb is None:
        return {**keep, "note": "kept human box: no usable contour after cleaning"}
    return {"points": obb, "tightened": True, "note": ""}


def _fit(mask, h, w):
    """SAM masks should already be image-sized; resize defensively if not."""
    mask = np.asarray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def tighten_image(image_path, labels, segment, margin=MARGIN):
    """labels: [{"id", "cls", "points"}]. segment(image_path, boxes) -> list of HxW masks, one per
    box in order (SAM). Returns {str(label id): result}; every label id is present."""
    if not labels:
        return {}
    img = cv2.imread(str(image_path))
    if img is None:
        return {str(lb["id"]): {"points": lb["points"], "tightened": False,
                                "note": "kept human box: image could not be read"} for lb in labels}
    h, w = img.shape[:2]
    boxes = [list(bounds(lb["points"])) for lb in labels]
    try:
        masks = list(segment(str(image_path), boxes))
    except Exception as e:  # SAM crashed on this image: keep every human box
        return {str(lb["id"]): {"points": lb["points"], "tightened": False,
                                "note": f"kept human box: SAM error {type(e).__name__}: {str(e)[:120]}"}
                for lb in labels}
    out = {}
    for i, lb in enumerate(labels):
        mask = _fit(masks[i], h, w) if i < len(masks) and masks[i] is not None else None
        out[str(lb["id"])] = tighten_box(mask, lb["points"], margin)
    return out


def sam_segmenter(weights="sam2.1_b.pt", device=None):
    """Ultralytics SAM as a `segment` callable: one mask per box prompt, in order."""
    from ultralytics import SAM

    sam = SAM(weights)

    def segment(image_path, boxes):
        res = sam(image_path, bboxes=boxes, device=device, verbose=False)[0]
        if res.masks is None:
            return []
        return list(res.masks.data.cpu().numpy())

    return segment


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="sam2.1_b.pt")
    p.add_argument("--margin", type=float, default=MARGIN)
    a = p.parse_args()
    todo = json.loads(Path(a.labels).read_text())
    segment = sam_segmenter(a.weights)
    done = {}
    for i, (name, labels) in enumerate(todo.items(), 1):
        done[name] = tighten_image(Path(a.images) / name, labels, segment, a.margin)
        n = sum(r["tightened"] for r in done[name].values())
        print(f"[{i}/{len(todo)}] {name}: {n} tightened, {len(labels) - n} kept", flush=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(done))
    tmp.replace(a.out)


if __name__ == "__main__":
    main()
