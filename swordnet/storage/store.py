"""FiftyOne-backed store: images, labels, the labeling queue, training runs and logs.

Everything lives in FiftyOne's MongoDB (started automatically by `import fiftyone`):
  - FiftyOne dataset "swordnet": one sample per image.
        status        "ingest" | "labeled"
        collection    which upload set it came from (has its own auto-label toggle)
        origin        upload:<name> | scrape:<url> | video:<name>#<frame>
        ground_truth  fo.Polylines: human-reviewed oriented boxes
        autolabels    fo.Polylines: auto-labeler proposals, never trained on directly
        claimed_by / claim_expires    the labeling queue (see claim_next)
    Each box is a closed 4-point fo.Polyline in normalized [0, 1] coords, label = class name,
    with `source` (manual | auto:<model> | ...+sam) and `score` attributes.
    Open it in the FiftyOne App to browse:  python -m swordnet.storage.store app
  - Mongo collections next to it (see coll()): swordnet_collections (auto-label toggles),
    swordnet_runs (training runs), swordnet_deploy (promoted run), swordnet_logs (swordnet/logs.py).

The web UI works in pixels; conversion happens here and nowhere else.
"""
import os
import shutil
import sys
import time

import fiftyone as fo
from bson import ObjectId
from pymongo import ReturnDocument

from swordnet import config

# Tests set SWORDNET_FO_DATASET so they never touch the real data.
DATASET = os.environ.get("SWORDNET_FO_DATASET", "swordnet")
CLAIM_SECONDS = 10 * 60  # a labeler holds an image this long without saving before it is re-queued


# --- handles ------------------------------------------------------------------

_FIELDS = (("status", fo.StringField), ("collection", fo.StringField), ("origin", fo.StringField),
           ("claimed_by", fo.StringField), ("claim_expires", fo.FloatField),
           ("ingested_at", fo.FloatField), ("labeled_at", fo.FloatField), ("labeled_by", fo.StringField))
_schema_ok = False


def dataset():
    global _schema_ok
    if fo.dataset_exists(DATASET):
        ds = fo.load_dataset(DATASET)
    else:
        ds = fo.Dataset(DATASET, persistent=True)
        ds.default_classes = config.CLASSES
        ds.save()
    if not _schema_ok:  # idempotent, so a dataset from an older version gains new fields
        schema = ds.get_field_schema()
        for name, ftype in _FIELDS:
            if name not in schema:
                ds.add_sample_field(name, ftype)
        for name in ("ground_truth", "autolabels"):
            if name not in schema:
                ds.add_sample_field(name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Polylines)
        _schema_ok = True
    return ds


def mongo():
    return fo.core.odm.get_db_conn()


def coll(name):
    """Our own Mongo collections, named after the dataset: swordnet_runs, swordnet_logs, ..."""
    return mongo()[f"{DATASET}_{name}"]


def _samples_coll(ds=None):
    return mongo()[(ds or dataset())._sample_collection_name]


# --- collections (upload sets) and the auto-label toggle --------------------------

def set_autolabel(name, enabled):
    coll("collections").update_one({"_id": name}, {"$set": {"autolabel": bool(enabled)}}, upsert=True)


def autolabel_collections():
    return [d["_id"] for d in coll("collections").find({"autolabel": True})]


def list_collections():
    ds = dataset()
    toggles = {d["_id"]: d.get("autolabel", False) for d in coll("collections").find()}
    counts = {}
    for c, s in zip(ds.values("collection"), ds.values("status")):
        counts.setdefault(c, {"n_ingest": 0, "n_labeled": 0})[f"n_{s}"] += 1
    names = sorted(set(counts) | set(toggles))
    return [{"name": n, "autolabel": toggles.get(n, False), **counts.get(n, {"n_ingest": 0, "n_labeled": 0})}
            for n in names]


# --- images -----------------------------------------------------------------

def add_image(path, collection, origin):
    """Register a file already in INGEST_DIR. Returns the sample id."""
    ds = dataset()
    s = fo.Sample(filepath=str(path), status="ingest", collection=collection, origin=origin,
                  ingested_at=time.time())
    s.metadata = fo.ImageMetadata.build_for(str(path))
    ds.add_sample(s)
    coll("collections").update_one({"_id": collection}, {"$setOnInsert": {"autolabel": False}}, upsert=True)
    return s.id


def get_sample(sample_id):
    try:
        s = dataset()[sample_id]
    except KeyError:
        return None
    s.reload()  # the queue updates claim fields with raw Mongo writes
    return s


STATES = ("pending", "annotating", "labeled")


def image_state(s, now=None):
    """pending: waiting in the queue · annotating: a labeler holds an unexpired claim · labeled: done."""
    if s.status == "labeled":
        return "labeled"
    if s.claimed_by and (s.claim_expires or 0) > (now or time.time()):
        return "annotating"
    return "pending"


def image_info(s, now=None):
    state = image_state(s, now)
    return {"id": s.id, "filename": s.filename, "filepath": s.filepath, "status": s.status, "state": state,
            "collection": s.collection, "origin": s.origin, "width": s.metadata.width,
            "height": s.metadata.height, "claimed_by": s.claimed_by if state == "annotating" else None,
            "claim_expires": s.claim_expires if state == "annotating" else None,
            "labeled_by": s.labeled_by, "labeled_at": s.labeled_at, "ingested_at": s.ingested_at,
            "n_labels": len(s.ground_truth.polylines) if s.ground_truth else 0,
            "n_auto": len(s.autolabels.polylines) if s.autolabels else 0}


def list_images(status=None, collection=None, state=None, limit=500, newest_first=False):
    F, now = fo.ViewField, time.time()
    view = dataset().view()
    if state == "labeled" or status == "labeled":
        view = view.match(F("status") == "labeled")
    elif state == "annotating":
        view = view.match((F("status") == "ingest") & (F("claim_expires") > now))
    elif state == "pending":
        view = view.match((F("status") == "ingest") & ~(F("claim_expires") > now))
    elif status:
        view = view.match(F("status") == status)
    if collection:
        view = view.match(F("collection") == collection)
    view = view.sort_by("ingested_at", reverse=newest_first).limit(limit)
    return [image_info(s, now) for s in view]


def state_counts():
    q = queue_stats()
    return {"pending": q["ingest"] - q["claimed"], "annotating": q["claimed"], "labeled": q["labeled"]}


def mark_processed(sample_id, user="anonymous"):
    """Labeler is done: move the file ingest -> labeled, flip status, release the claim."""
    s = get_sample(sample_id)
    if s is None or s.status == "labeled":
        return s
    dst = config.LABELED_DIR / s.filename
    if s.filepath != str(dst):
        shutil.move(s.filepath, dst)
    s.filepath = str(dst)
    s.status, s.labeled_at, s.labeled_by = "labeled", time.time(), user
    s.claimed_by, s.claim_expires = None, None
    s.save()
    return s


def delete_image(sample_id):
    s = get_sample(sample_id)
    if s is None:
        return False
    from pathlib import Path
    Path(s.filepath).unlink(missing_ok=True)
    dataset().delete_samples([sample_id])
    return True


# --- the labeling queue ----------------------------------------------------------

def claim_next(user, collection=None, skip=()):
    """Atomically hand `user` the oldest ingest image nobody else holds.

    A single Mongo find_one_and_update, so two labelers asking at the same moment can never
    get the same image. Claims expire after CLAIM_SECONDS, so an abandoned tab re-queues its
    image. A user asking again gets their own unexpired claim back first.
    """
    now = time.time()
    coll = _samples_coll()
    mine = coll.find_one({"status": "ingest", "claimed_by": user, "claim_expires": {"$gt": now},
                          "_id": {"$nin": [ObjectId(i) for i in skip]}})
    if mine:
        return str(mine["_id"])
    q = {"status": "ingest",
         "$or": [{"claimed_by": None}, {"claim_expires": {"$lt": now}}],
         "_id": {"$nin": [ObjectId(i) for i in skip]}}
    if collection:
        q["collection"] = collection
    doc = coll.find_one_and_update(q, {"$set": {"claimed_by": user, "claim_expires": now + CLAIM_SECONDS}},
                                   sort=[("ingested_at", 1)], return_document=ReturnDocument.AFTER)
    return str(doc["_id"]) if doc else None


def renew_claim(sample_id, user):
    r = _samples_coll().update_one({"_id": ObjectId(sample_id), "claimed_by": user},
                                   {"$set": {"claim_expires": time.time() + CLAIM_SECONDS}})
    return r.modified_count == 1


def release_claim(sample_id, user):
    _samples_coll().update_one({"_id": ObjectId(sample_id), "claimed_by": user},
                               {"$set": {"claimed_by": None, "claim_expires": None}})


def queue_stats():
    now = time.time()
    coll = _samples_coll()
    return {"ingest": coll.count_documents({"status": "ingest"}),
            "claimed": coll.count_documents({"status": "ingest", "claim_expires": {"$gt": now}}),
            "labeled": coll.count_documents({"status": "labeled"}),
            "labelers": sorted(coll.distinct("claimed_by", {"claim_expires": {"$gt": now}}))}


# --- labels (pixel coords at the API boundary) ------------------------------------

def _to_polyline(lb, w, h):
    pts = [(float(x) / w, float(y) / h) for x, y in lb["points"]]
    return fo.Polyline(label=config.CLASSES[int(lb["cls"])], points=[pts], closed=True, filled=False,
                       source=lb.get("source", "manual"), score=lb.get("score"))


def _from_polyline(p, w, h):
    return {"cls": config.CLASSES.index(p.label), "points": [[x * w, y * h] for x, y in p.points[0]],
            "source": p.get_field("source") if p.has_field("source") else "manual",
            "score": p.get_field("score") if p.has_field("score") else None}


def get_labels(s, field="ground_truth"):
    lbls = s[field]
    return [_from_polyline(p, s.metadata.width, s.metadata.height) for p in lbls.polylines] if lbls else []


def set_labels(s, labels, field="ground_truth"):
    s[field] = fo.Polylines(polylines=[_to_polyline(lb, s.metadata.width, s.metadata.height) for lb in labels])
    s.save()


# --- runs -------------------------------------------------------------------

def _next_id(name):
    return coll("counters").find_one_and_update(
        {"_id": name}, {"$inc": {"n": 1}}, upsert=True, return_document=ReturnDocument.AFTER)["n"]


def create_run(model, epochs, cfg, **fields):
    run_id = _next_id("runs")
    coll("runs").insert_one({"_id": run_id, "model": model, "epochs": epochs, "config": cfg,
                                      "status": "queued", "metrics": None, **fields})
    return run_id


def update_run(run_id, **fields):
    coll("runs").update_one({"_id": run_id}, {"$set": fields})


def _run_out(d):
    d["id"] = d.pop("_id")
    return d


def list_runs(limit=200):
    return [_run_out(d) for d in coll("runs").find().sort("_id", -1).limit(limit)]


def get_run(run_id):
    d = coll("runs").find_one({"_id": run_id})
    return _run_out(d) if d else None


if __name__ == "__main__":
    if sys.argv[1:] == ["app"]:
        session = fo.launch_app(dataset(), address="0.0.0.0")
        session.wait()
    else:
        print(dataset())
        print(queue_stats())
