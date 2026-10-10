"""Auto-labeler: plain upright boxes around swords and sticks, written the way the web UI writes them.

Runs INSIDE the agent seat pod, in the team project (projects/03-swordnet), using its own code:
    ingest   core.media.ingest_upload          same path as the web UI's Upload button
    claim    the same queue update as core.db.claim, for one specific pending image
    label    core.db.submit                    same as "Save & mark processed" (ground_truth, -> labeled/)
    unsure   core.db.release + core.db.set_prefill   back in the queue with pre-filled boxes for a person
Detector: Ultralytics YOLOE (open-vocabulary, text prompts = the project's AUTOLABEL_PROMPTS),
class-agnostic NMS, then one box per object: weak extras (< half the best score) and boxes lying
mostly inside a stronger box are dropped. Boxes are axis-aligned (no SAM, no rotation).

Decision per image:
    every box >= --submit-conf and no
    nested/duplicate boxes were seen  -> submitted as labels (labeler "autobox")
    any box, but one below that       -> pre-filled, back in the queue for a person to confirm
    no box                            -> left in the queue untouched (never saved as "no sword here")
Images a person holds are never touched: only free queue entries are claimed (pending, or a
claim whose lease expired, exactly as core.db.claim treats them).

    python3 autobox.py [--ingest DIR] [--limit 200] [--submit-conf 0.35] [--min-conf 0.15] [--dry-run]
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ.get("SWORDNET_PROJECT", "/workspace/projects/03-swordnet"))
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

from pymongo import ReturnDocument  # noqa: E402

from core import db, media  # noqa: E402
from core.config import S  # noqa: E402
from core.geometry import BoxError, validate_obb  # noqa: E402

LABELER = "autobox"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ingest(folder: Path) -> dict:
    out = {"added": 0, "duplicates": 0, "errors": 0}
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT | set(getattr(media, "VIDEO_EXT", ())):
            continue
        try:
            for r in media.ingest_upload(p.read_bytes(), p.name, source="operator", uploader=LABELER):
                out["duplicates" if r.get("duplicate") else "added"] += 1
        except media.IngestError as e:
            out["errors"] += 1
            print(f"skip {p.name}: {e}", file=sys.stderr)
    return out


def claim_one(sid: str) -> bool:
    """db.claim's queue update, restricted to this one image and only if nobody holds it."""
    t = db.now()
    doc = db.mongo()[db.QUEUE].find_one_and_update(
        {"_id": sid, "$or": [{"state": "pending"}, {"state": "claimed", "lease_until": {"$lt": t}}]},
        {"$set": {"state": "claimed", "claimed_by": LABELER, "claimed_at": t,
                  "lease_until": t + dt.timedelta(minutes=S.lease_min)}, "$inc": {"claims": 1}},
        return_document=ReturnDocument.AFTER)
    if doc is None:
        return False
    with db._lock:
        s = db.get(sid)
        s["status"] = "claimed"
        s["claimed_by"] = LABELER
        s.save()
    return True


def one_per_object(xyxy, scores, classes, rel=0.2, inside=0.6, encloses=0.85):
    """One box per sword: drop boxes scoring under `rel` x the image's best, then drop any box that
    lies mostly (> `inside` of its area) within a stronger kept box, or that wraps (> `encloses`)
    a stronger kept box. Crossed swords overlap but don't wrap each other, so both survive."""
    one_per_object.overlapped = False
    dets = sorted(zip(xyxy, scores, classes), key=lambda d: -d[1])
    if not dets:
        return []
    top = dets[0][1]
    kept = []
    for b, sc, c in dets:
        if sc < rel * top:
            continue
        area = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k, _, _ in kept:
            iw, ih = min(b[2], k[2]) - max(b[0], k[0]), min(b[3], k[3]) - max(b[1], k[1])
            k_area = max(1e-9, (k[2] - k[0]) * (k[3] - k[1]))
            if iw > 0 and ih > 0 and (iw * ih / area > inside or iw * ih / k_area > encloses):
                dup = True
                break
        if dup:
            one_per_object.overlapped = True  # the detector saw nested boxes: count is unsure
        else:
            kept.append((b, sc, c))
    return kept


def detector(min_conf: float):
    from ultralytics import YOLOE
    prompts = S.prompts()                      # {"sword": "a sword", "stick": "a wooden stick"}
    # Extra phrasings find more blades in multi-fencer frames. The last three are what YOLOE's
    # prompt-free model called the swords it missed. On a 194-frame sparring clip: frames with no box
    # 96 -> 20, frames with two boxes 23 -> 90. All are labeled with the first class.
    extra = ["a fencing sword", "a long thin blade", "a metal rod", "a hockey stick", "a weapon", "a cane"]
    classes = list(prompts) + [list(prompts)[0]] * len(extra)
    texts = list(prompts.values()) + extra
    model = YOLOE(os.environ.get("AUTOBOX_MODEL", "yoloe-11l-seg.pt"))
    model.set_classes(texts, model.get_text_pe(texts))

    def detect(path: str, w: int, h: int) -> list:
        r = model.predict(path, conf=min_conf, iou=0.7, agnostic_nms=True, verbose=False)[0]
        boxes = []
        for (x1, y1, x2, y2), score, c in one_per_object(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            pts = [[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]
            try:
                boxes.append({**validate_obb({"cls": classes[int(c)], "points": pts}, S.classes), "score": round(score, 4)})
            except BoxError:
                continue
        detect.unsure = one_per_object.overlapped
        return boxes
    detect.unsure = False
    return detect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", help="folder of images/videos to upload first (the normal ingest path)")
    ap.add_argument("--limit", type=int, default=200, help="max queue images to label this run")
    ap.add_argument("--submit-conf", type=float, default=0.35)
    ap.add_argument("--min-conf", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true", help="detect and report only; change nothing")
    ap.add_argument("--speed", action="store_true",
                    help="save every image with at least one box (no human check); each box's score is kept in "
                         "the sample's autolabel_raw field so a filter can drop weak labels afterwards")
    a = ap.parse_args()

    stats = {"ingest": ingest(Path(a.ingest)) if a.ingest and not a.dry_run else None,
             "submitted": 0, "prefilled": 0, "no_boxes": 0, "skipped_busy": 0, "errors": 0, "boxes": 0}
    detect = detector(a.min_conf)
    if a.dry_run and a.ingest:  # preview the folder's photos without uploading anything
        import cv2
        for p in sorted(Path(a.ingest).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT and (im := cv2.imread(str(p))) is not None:
                boxes = detect(str(p), im.shape[1], im.shape[0])
                sure = bool(boxes) and (a.speed or (min(b["score"] for b in boxes) >= a.submit_conf and not detect.unsure))
                print(json.dumps({"image": p.name, "boxes": [(b["cls"], b["score"]) for b in boxes],
                                  "action": "submit" if sure else ("prefill" if boxes else "leave")}), flush=True)
        print("SUMMARY dry run of the folder only; nothing was uploaded or labeled", flush=True)
        return
    q = db.mongo()[db.QUEUE]
    free = {"$or": [{"state": "pending"}, {"state": "claimed", "lease_until": {"$lt": db.now()}}]}  # as db.claim
    pending = [d["_id"] for d in q.find(free).sort([("priority", -1), ("created_at", 1)]).limit(a.limit)]
    t0 = time.time()
    for sid in pending:
        try:
            s = db.get(sid)
            boxes = detect(s.filepath, s.metadata.width, s.metadata.height)
        except Exception as e:
            stats["errors"] += 1
            print(f"{sid}: detect failed: {e}", file=sys.stderr)
            continue
        sure = bool(boxes) and (a.speed or (min(b["score"] for b in boxes) >= a.submit_conf and not detect.unsure))
        print(json.dumps({"image": Path(s.filepath).name, "boxes": [(b["cls"], b["score"]) for b in boxes],
                          "action": "submit" if sure else ("prefill" if boxes else "leave")}), flush=True)
        if a.dry_run:
            continue
        if not boxes:
            stats["no_boxes"] += 1
            continue
        if not claim_one(sid):
            stats["skipped_busy"] += 1
            continue
        clean = [{"cls": b["cls"], "points": b["points"]} for b in boxes]
        try:
            if sure:
                db.submit(sid, LABELER, clean)
                with db._lock:  # keep the scores for the end-of-run filter
                    s2 = db.get(sid)
                    s2["autolabel_raw"] = [{**b, "engine": "yoloe", "unsure_count": detect.unsure} for b in boxes]
                    s2.save()
                stats["submitted"] += 1
            else:
                db.release(sid, LABELER)
                db.set_prefill(sid, clean, raw=[{**b, "engine": "yoloe"} for b in boxes])
                stats["prefilled"] += 1
            stats["boxes"] += len(clean)
        except Exception as e:
            stats["errors"] += 1
            print(f"{sid}: write failed: {e}", file=sys.stderr)
            try:
                db.release(sid, LABELER)
            except Exception:
                pass
    stats["seconds"] = round(time.time() - t0, 1)
    print("SUMMARY " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
