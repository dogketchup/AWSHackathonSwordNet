"""The team database's front door. Runs ON seat-145, one operation per `kubectl exec`:

    kubectl exec -i seat-145 -- env <POD_ENV> python3 -m swordnet.storage.pod_api <op> < request.json

It runs the same storage code as everywhere else (swordnet/storage/store.py) against the one
FiftyOne/MongoDB on the pod, so the queue, duplicate check and labels behave exactly as tested.
Prints one JSON object; failures come back as {"error": "..."} for the client to show.

Pod environment (set by the client, see remote.py):
    FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017   the MongoDB the team already runs there
    SWORDNET_FO_DATASET=swordnet_team                 our dataset, separate from anything else in it
    SWORDNET_HOME=/workspace/swordnet-team            image files: ingest/, labeled/, incoming/
"""
import base64
import json
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path

from swordnet import config, logs
from swordnet.storage import store

INLINE_MAX = 6 * 1024 * 1024  # images up to this size travel inside the JSON answer


def _info(s):
    return store.image_info(s) if s is not None else None


def _sample(req):
    s = store.get_sample(req["id"])
    if s is None:
        raise LookupError("image no longer exists (someone deleted it)")
    return s


def _with_file(s, out):
    """Attach the image bytes when small enough, so showing an image costs one exec, not three."""
    p = Path(s.filepath)
    if p.exists() and p.stat().st_size <= INLINE_MAX:
        out["file_b64"] = base64.b64encode(p.read_bytes()).decode()
        out["file_ext"] = p.suffix.lower()
    return out


# --- images and labeling ----------------------------------------------------------

def op_ingest(req):
    """Register files pushed into incoming/<batch>/ (with optional labels and state)."""
    batch = config.HOME / "incoming" / Path(req["batch"]).name
    added, dupes = [], []
    for it in req["items"]:
        src = batch / Path(it["file"]).name
        if not src.exists():
            continue
        dst = config.INGEST_DIR / f"{uuid.uuid4().hex[:12]}{src.suffix.lower()}"
        shutil.move(str(src), dst)
        sid = store.add_image(dst, it.get("collection") or "default", it.get("origin") or "upload")
        if sid is None:
            dupes.append(it["file"])
            continue
        s = store.get_sample(sid)
        if it.get("ingested_at"):
            s.ingested_at = it["ingested_at"]
            s.save()
        if it.get("labels"):
            store.set_labels(s, it["labels"])
        if it.get("autolabels"):
            store.set_labels(s, it["autolabels"], field="autolabels")
        if it.get("status") == "labeled":
            s = store.mark_processed(sid, user=it.get("labeled_by") or "?")
            if it.get("labeled_at"):
                s.labeled_at = it["labeled_at"]
                s.save()
        added.append(sid)
    shutil.rmtree(batch, ignore_errors=True)
    return {"added": added, "duplicates": dupes}


def op_list_images(req):
    return {"images": store.list_images(**{k: req[k] for k in ("status", "collection", "state", "limit",
                                                               "newest_first") if k in req})}


def op_summary(req):
    """Header numbers + datasets in one call."""
    return {"queue": store.queue_stats(), "states": store.state_counts(), "datasets": store.list_collections()}


def op_set_autolabel(req):
    store.set_autolabel(req["collection"], req["enabled"])
    return {"ok": True}


def op_next(req):
    """Claim the next image for a labeler and return it with its boxes and its bytes."""
    sid = store.claim_next(req["user"], collection=req.get("collection"), skip=req.get("skip", []))
    if sid is None:
        return {"image": None}
    return op_open({"id": sid})


def op_open(req):
    s = _sample(req)
    labels = store.get_labels(s) or store.get_labels(s, field="autolabels")
    return _with_file(s, {"image": _info(s), "labels": labels})


def op_release(req):
    store.release_claim(req["id"], req["user"])
    return {"ok": True}


def op_save(req):
    """Save boxes; `finish` also marks the image labeled (moves its file) and ends the claim."""
    s = _sample(req)
    if s.status == "ingest" and s.claimed_by and s.claim_expires and s.claim_expires > time.time() \
            and s.claimed_by != req["user"]:
        return {"error": f"{s.claimed_by} is labeling this image", "conflict": True}
    store.set_labels(s, req["labels"])
    if req.get("finish"):
        s = store.mark_processed(req["id"], user=req["user"])
        logs.log("webui", f"{req['user']} labeled {s.filename}: {len(req['labels'])} box(es)",
                 user=req["user"], sample=req["id"])
    else:
        store.renew_claim(req["id"], req["user"])
    return {"image": _info(store.get_sample(req["id"]))}


def op_set_autolabels(req):
    s = _sample(req)
    if store.get_labels(s):
        return {"skipped": "already has human labels"}
    store.set_labels(s, req["autolabels"], field="autolabels")
    return {"ok": True}


def op_delete(req):
    s = _sample(req)
    name = s.filename
    store.delete_image(req["id"])
    logs.log("webui", f"{req.get('user', '?')} rejected {name}", user=req.get("user"))
    return {"ok": True}


def op_pack(req):
    """Tar up image files (named <id><ext>) for the client to download in verified pieces."""
    out = config.HOME / "incoming" / f"pack-{uuid.uuid4().hex}.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w") as tar:
        for sid in req["ids"]:
            s = store.get_sample(sid)
            if s is not None and Path(s.filepath).exists():
                tar.add(s.filepath, arcname=f"{sid}{Path(s.filepath).suffix.lower()}")
    return {"path": str(out)}


def op_samples_for(req):
    """Samples with full label data, for the bridge (export, auto-label, refine)."""
    view = store.dataset().view()
    F = store.fo.ViewField
    if req.get("status"):
        view = view.match(F("status") == req["status"])
    if req.get("collections") is not None:
        view = view.match(F("collection").is_in(req["collections"]))
    return {"samples": [{**store.image_info(s), "labels": store.get_labels(s),
                         "autolabels": store.get_labels(s, field="autolabels")}
                        for s in view.limit(req.get("limit", 100000))]}


def op_export(req):
    """Write a YOLO-OBB training snapshot on this pod (see train/export.py)."""
    from swordnet.train import export
    out = config.DATASETS_DIR / Path(req["name"]).name
    return {"manifest": export.export(out), "path": str(out)}


# --- runs, deploy, logs (the bridge's history) ------------------------------------

def op_create_run(req):
    return {"id": store.create_run(req["model"], req["epochs"], req["config"], **req.get("fields", {}))}


def op_update_run(req):
    store.update_run(req["id"], **req["fields"])
    return {"ok": True}


def op_list_runs(req):
    return {"runs": store.list_runs(limit=req.get("limit", 200))}


def op_get_run(req):
    return {"run": store.get_run(req["id"])}


def op_deploy_get(req):
    return {"deploy": store.coll("deploy").find_one({"_id": "current"})}


def op_deploy_set(req):
    store.coll("deploy").replace_one({"_id": "current"}, {"_id": "current", **req["deploy"]}, upsert=True)
    return {"ok": True}


def op_log(req):
    entries = req["entries"]
    if entries:
        store.coll("logs").insert_many(entries)
    return {"ok": True}


def op_logs(req):
    return {"sources": logs.sources(), "entries": logs.query(**{k: req.get(k) for k in
                                                              ("source", "run_id", "level", "after", "text", "limit")
                                                              if req.get(k) is not None})}


OPS = {name[3:]: fn for name, fn in globals().items() if name.startswith("op_")}

if __name__ == "__main__":
    config.ensure_dirs()
    op = sys.argv[1]
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        res = OPS[op](json.loads(raw) if raw.strip() else {})
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}"}
    print("\n" + json.dumps(res, default=str))
