"""Annotator, laptop side: the side process that moves images from labeled to annotated
(scheduler plan, Part 0).

Each pass:
    1. take up to BATCH labeled images (oldest labeled first) with fewer than MAX_TRIES attempts
    2. images with no boxes go straight to annotated (nothing to tighten)
    3. snapshot each image's labels exactly as they are, count one attempt
    4. hand the images + labels to a worker (the aux seat runs tighten.py), get results back
    5. for each image, in one atomic step: write the tightened points and move it to annotated
       ONLY IF it is still labeled and its labels still equal the snapshot; otherwise drop the
       results (a person edited it; it will be picked up again)

Storage and the worker are interfaces, so this file does not depend on any database or on
kubectl. `MemoryStore` and `LocalWorker` are working implementations for tests and local runs.
"""
import json
import threading
import time
import traceback
from typing import Protocol

BATCH = 8
EVERY = 120      # seconds between passes
MAX_TRIES = 2


class Store(Protocol):
    def list_to_annotate(self, limit: int, max_tries: int) -> list:
        """Labeled images with tries < max_tries, oldest labeled first: [{"id", "filename", "path"}].
        Filter BEFORE applying the limit (review aafd970 N7: limiting first starves newer images)."""
    def labels(self, image_id) -> list:
        """[{"id", "cls", "points"}] for one image."""
    def count_try(self, image_id) -> None: ...
    def apply_if_unchanged(self, image_id, snapshot, new_points: dict) -> bool:
        """Atomically: if the image is still labeled and snapshot_of(labels) == snapshot, set each
        label's points from new_points {label_id: points}, keep the original points, and move the
        image to annotated. Return whether it was written."""
    def mark_annotated(self, image_id) -> bool:
        """Atomically move a still-labeled image with no labels to annotated."""
    def stuck(self, max_tries: int) -> list:
        """Labeled images that used up their tries. They stay labeled (so training never sees an
        untightened box) and are shown on the dashboard for a person to look at."""


class Worker(Protocol):
    def run(self, images: list, labels: dict) -> dict:
        """images: [{"id", "filename", "path"}]; labels: {filename: [label]}.
        Returns {filename: {str(label id): {"points", "tightened", "note"}}}."""


def snapshot_of(labels):
    """Canonical form used for the unchanged check: sorted (id, cls, points rounded to 0.01 px)."""
    return sorted([str(lb["id"]), int(lb["cls"]), [[round(float(x), 2), round(float(y), 2)] for x, y in lb["points"]]]
                  for lb in labels)


def run_pass(store, worker, batch=BATCH, max_tries=MAX_TRIES, log=print):
    """One pass. Returns counts: annotated, empty, dropped, failed, kept_boxes, tightened_boxes."""
    stats = {"annotated": 0, "empty": 0, "dropped": 0, "failed": 0, "tightened_boxes": 0, "kept_boxes": 0}
    imgs = store.list_to_annotate(batch, max_tries)
    if not imgs:
        return stats
    todo, snaps, labels = [], {}, {}
    for img in imgs:
        lbs = store.labels(img["id"])
        if not lbs:
            if store.mark_annotated(img["id"]):
                stats["empty"] += 1
            continue
        snaps[img["id"]] = snapshot_of(lbs)
        labels[img["filename"]] = [{"id": lb["id"], "cls": lb["cls"], "points": lb["points"]} for lb in lbs]
        store.count_try(img["id"])
        todo.append(img)
    if not todo:
        return stats
    try:
        results = worker.run(todo, labels)
    except Exception:
        log(f"annotator: worker failed for {len(todo)} images; they stay labeled and are retried "
            f"until {max_tries} tries:\n{traceback.format_exc()[-1500:]}")
        stats["failed"] += len(todo)
        return stats
    for img in todo:
        res = results.get(img["filename"])
        if res is None:
            stats["failed"] += 1
            log(f"annotator: no result for {img['filename']}; it stays labeled")
            continue
        new_points = {lid: r["points"] for lid, r in res.items() if r.get("tightened")}
        if store.apply_if_unchanged(img["id"], snaps[img["id"]], new_points):
            stats["annotated"] += 1
            stats["tightened_boxes"] += len(new_points)
            stats["kept_boxes"] += len(res) - len(new_points)
        else:
            stats["dropped"] += 1
            log(f"annotator: {img['filename']} changed while it was being annotated; results dropped")
    return stats


def run_forever(store, worker, every=EVERY, batch=BATCH, max_tries=MAX_TRIES, log=print, stop=None):
    """The side process: a pass every `every` seconds until `stop` (a threading.Event) is set."""
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            s = run_pass(store, worker, batch, max_tries, log)
            if any(s.values()):
                log(f"annotator: {json.dumps(s)}")
        except Exception:
            log(f"annotator: pass failed, will retry:\n{traceback.format_exc()[-1500:]}")
        stop.wait(every)


# --- working implementations for tests and local runs -------------------------

class MemoryStore:
    """In-memory Store with the same atomic rules a real database adapter must keep."""

    def __init__(self):
        self.lock = threading.Lock()
        self.images = {}   # id -> {"id", "filename", "path", "status", "labeled_at", "tries"}
        self._labels = {}  # image id -> [{"id", "cls", "points", "orig_points", "source"}]
        self._next = 1

    def add_image(self, image_id, filename, path="", status="labeled", labeled_at=None, labels=()):
        with self.lock:
            self.images[image_id] = {"id": image_id, "filename": filename, "path": path, "status": status,
                                     "labeled_at": labeled_at if labeled_at is not None else time.time(), "tries": 0}
            self._labels[image_id] = []
        for cls, pts in labels:
            self.add_label(image_id, cls, pts)

    def add_label(self, image_id, cls, points, source="manual"):
        with self.lock:
            lid = self._next
            self._next += 1
            self._labels[image_id].append({"id": lid, "cls": cls, "points": [list(p) for p in points],
                                           "orig_points": None, "source": source})
            return lid

    def human_save(self, image_id, labels):
        """What the web UI's save does: replace all labels; an annotated image goes back to labeled."""
        with self.lock:
            self._labels[image_id] = []
        for cls, pts in labels:
            self.add_label(image_id, cls, pts)
        with self.lock:
            img = self.images[image_id]
            if img["status"] == "annotated":
                img["status"], img["tries"] = "labeled", 0

    def list_to_annotate(self, limit, max_tries):
        with self.lock:
            imgs = [i for i in self.images.values() if i["status"] == "labeled" and i["tries"] < max_tries]
            imgs.sort(key=lambda i: (i["labeled_at"], str(i["id"])))
            return [dict(i) for i in imgs[:limit]]

    def labels(self, image_id):
        with self.lock:
            return [dict(lb) for lb in self._labels.get(image_id, [])]

    def count_try(self, image_id):
        with self.lock:
            self.images[image_id]["tries"] += 1

    def apply_if_unchanged(self, image_id, snapshot, new_points):
        with self.lock:
            img = self.images.get(image_id)
            if not img or img["status"] != "labeled" or snapshot_of(self._labels[image_id]) != snapshot:
                return False
            for lb in self._labels[image_id]:
                pts = new_points.get(str(lb["id"]))
                if pts is not None and len(pts) == 4:
                    lb["orig_points"] = lb["orig_points"] or lb["points"]
                    lb["points"] = [list(p) for p in pts]
                    if not lb["source"].endswith("+sam"):
                        lb["source"] += "+sam"
            img["status"] = "annotated"
            return True

    def mark_annotated(self, image_id):
        with self.lock:
            img = self.images.get(image_id)
            if not img or img["status"] != "labeled" or self._labels[image_id]:
                return False
            img["status"] = "annotated"
            return True

    def stuck(self, max_tries):
        with self.lock:
            return [dict(i) for i in self.images.values() if i["status"] == "labeled" and i["tries"] >= max_tries]


class LocalWorker:
    """Runs tighten.py's logic in this process (on the laptop, or inside a pod)."""

    def __init__(self, segment, margin=None):
        self.segment = segment
        self.margin = margin

    def run(self, images, labels):
        import tighten
        out = {}
        for img in images:
            kw = {"margin": self.margin} if self.margin is not None else {}
            out[img["filename"]] = tighten.tighten_image(img["path"], labels[img["filename"]], self.segment, **kw)
        return out
