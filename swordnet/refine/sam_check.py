"""SAM check: see what the SAM annotator would do to the human boxes, without changing any label.

For each labeled image, every human box goes through the annotator's tightening step
(refine/tighten.py: SAM with the box as prompt -> mask cut to the box -> biggest contour ->
cv2.minAreaRect). Results are drawn on the image and listed in the web UI's "SAM check" tab:
    yellow   the person's box
    cyan     SAM's mask, cut to the box
    green    the tightened oriented box
    red      SAM failed for this box, so the annotator would keep the person's box (reason shown)

Read-only: labels in the store are never written. Output goes to data/samcheck/.

Where the human boxes come from (pick one):
    python -m swordnet.refine.sam_check --store [--limit 50]        labeled images in the FiftyOne store
    python -m swordnet.refine.sam_check --snapshot DIR [--limit 50] a YOLO-OBB snapshot (images/*/, labels/*/),
                                                                    e.g. one pulled back from a trainer seat
Runs on the laptop's CPU by default (a few seconds per image with sam2.1_b).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from swordnet import config
from swordnet.refine import tighten

OUT_DIR = config.HOME / "samcheck"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def from_store(limit):
    from swordnet.storage import store

    out = []
    for info in store.list_images(status="labeled", limit=limit, newest_first=True):
        s = store.get_sample(info["id"])
        labels = [{"id": i, "cls": lb["cls"], "points": lb["points"]} for i, lb in enumerate(store.get_labels(s))]
        out.append({"name": info["filename"], "path": info["filepath"], "labels": labels})
    return out


def from_snapshot(root, limit):
    """YOLO-OBB snapshot: images/<split>/x.jpg with labels/<split>/x.txt lines 'cls x1 y1 ... x4 y4' (0..1)."""
    root = Path(root)
    out = []
    for img in sorted(p for p in root.glob("images/*/*") if p.suffix.lower() in IMAGE_EXTS):
        txt = root / "labels" / img.parent.name / (img.stem + ".txt")
        if not txt.exists():
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        labels = []
        for i, line in enumerate(l for l in txt.read_text().splitlines() if l.strip()):
            v = line.split()
            if len(v) != 9:
                continue
            c = [float(x) for x in v[1:]]
            labels.append({"id": i, "cls": int(v[0]), "points": [[c[k] * w, c[k + 1] * h] for k in range(0, 8, 2)]})
        out.append({"name": f"{img.parent.name}/{img.name}", "path": str(img), "labels": labels})
        if len(out) >= limit:
            break
    return out


def draw(image_path, labels, masks, results):
    img = cv2.imread(str(image_path))
    overlay = img.copy()
    for lb, mask in zip(labels, masks):
        if mask is not None:
            m = tighten.clip_to_box(tighten._fit(mask, *img.shape[:2]), tighten.bounds(lb["points"]))
            overlay[m > 0] = (255, 255, 0)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    t = max(2, round(max(img.shape[:2]) / 400))
    for lb in labels:
        r = results[str(lb["id"])]
        human = np.array(lb["points"], np.int32)
        cv2.polylines(img, [human], True, (0, 215, 255), t)                    # yellow: the person's box
        if r["tightened"]:
            cv2.polylines(img, [np.array(r["points"], np.int32)], True, (60, 200, 60), t + 1)   # green
        else:
            cv2.polylines(img, [human], True, (40, 40, 220), t + 1)           # red: kept
        x, y = human.min(axis=0)
        cv2.putText(img, config.CLASSES[lb["cls"]], (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return img


def run(items, weights="sam2.1_b.pt", device="cpu", out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    segment = tighten.sam_segmenter(weights, device)
    results = []
    for k, it in enumerate(items, 1):
        t0 = time.perf_counter()
        boxes = [list(tighten.bounds(lb["points"])) for lb in it["labels"]]
        try:
            masks = list(segment(it["path"], boxes)) if boxes else []
        except Exception as e:
            masks, err = [], f"{type(e).__name__}: {e}"
        else:
            err = None
        masks = masks + [None] * (len(it["labels"]) - len(masks))
        img = cv2.imread(str(it["path"]))
        per = {}
        if img is not None:
            h, w = img.shape[:2]
            for lb, mask in zip(it["labels"], masks):
                per[str(lb["id"])] = tighten.tighten_box(tighten._fit(mask, h, w) if mask is not None else None, lb["points"])
        png = f"{k:04d}.png"
        if img is not None:
            cv2.imwrite(str(out_dir / "img" / png), draw(it["path"], it["labels"], masks, per))
        results.append({
            "name": it["name"], "png": png if img is not None else None, "seconds": round(time.perf_counter() - t0, 2),
            "error": err or (None if img is not None else "image could not be read"),
            "boxes": [{"cls": config.CLASSES[lb["cls"]], **per.get(str(lb["id"]), {"tightened": False, "note": "not run"})}
                      for lb in it["labels"]],
        })
        n_t = sum(b["tightened"] for b in results[-1]["boxes"])
        print(f"[{k}/{len(items)}] {it['name']}: {n_t}/{len(it['labels'])} boxes tightened "
              f"({results[-1]['seconds']} s){' ERROR ' + err if err else ''}", flush=True)
    summary = {"time": time.time(), "weights": weights, "images": len(results),
               "boxes": sum(len(r["boxes"]) for r in results),
               "tightened": sum(b["tightened"] for r in results for b in r["boxes"]), "results": results}
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out_dir / "results.json")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", action="store_true", help="labeled images in the FiftyOne store")
    src.add_argument("--snapshot", help="a YOLO-OBB snapshot folder")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--weights", default="sam2.1_b.pt")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    items = from_store(a.limit) if a.store else from_snapshot(a.snapshot, a.limit)
    if not items:
        raise SystemExit("no labeled images with boxes found")
    s = run(items, a.weights, a.device)
    print(f"done: {s['tightened']}/{s['boxes']} boxes tightened on {s['images']} images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
