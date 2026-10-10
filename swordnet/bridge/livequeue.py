"""Live labeling queue and logs from the team database on the cluster (read-only).

Reads through the thin client (storage/remote.py -> storage/pod_api.py on the database seat), so
it shows the team dataset every laptop works on:
    annotating   images someone holds right now: who, and until when
    next         images waiting, in the order the queue hands them out (oldest first)
    done         labeled images, newest first: who, when
    logs         the latest entries of the team log
"""
import datetime

from swordnet.storage import remote


def _t(v):
    return datetime.datetime.fromtimestamp(v).isoformat(timespec="seconds") if isinstance(v, (int, float)) else v


def live():
    imgs = remote.call("list_images", {"limit": 10000})["images"]
    annotating = [{"file": i["filename"], "by": i.get("claimed_by"), "until": _t(i.get("claim_expires"))}
                  for i in imgs if i.get("state") == "annotating"]
    waiting = sorted((i for i in imgs if i.get("state") == "pending"), key=lambda i: i.get("ingested_at") or 0)
    done = sorted((i for i in imgs if i.get("state") == "labeled"), key=lambda i: -(i.get("labeled_at") or 0))
    entries = remote.call("logs", {"limit": 60})["entries"]
    return {
        "seat": remote.SEAT, "time": datetime.datetime.now().timestamp(),
        "counts": {"images": len(imgs), "annotating": len(annotating), "next": len(waiting), "done": len(done)},
        "annotating": annotating,
        "next": [{"file": i["filename"], "waiting_since": _t(i.get("ingested_at"))} for i in waiting],
        "done": [{"file": i["filename"], "by": i.get("labeled_by"), "at": _t(i.get("labeled_at")),
                  "boxes": i.get("n_labels")} for i in done],
        "logs": {"team": [f"{_t(e.get('ts'))} {e.get('source', '')} {e.get('level', '')}: {e.get('msg', '')}" for e in entries]},
    }
