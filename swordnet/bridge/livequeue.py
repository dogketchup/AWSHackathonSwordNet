"""Live labeling queue and logs from the team database on the cluster (read-only).

The FiftyOne database and the newer web UI run on the agent seat (projects/03-swordnet). One
`kubectl exec` runs a small read-only query there and returns:
    annotating   images someone holds right now: who, since when, lease end, whether it expired
    next         images waiting, in the order the queue hands them out
    done         annotated images, newest first: who, when
    logs         the last lines of the pod's api / agent / import logs
Nothing is written on the pod.
"""
import json

from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS

PROJECT = "/workspace/projects/03-swordnet"

_QUERY = r'''
import json, os, time, datetime, pymongo
def iso(v):
    return v.isoformat(timespec="seconds") if isinstance(v, datetime.datetime) else v
db = pymongo.MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=4000)["fiftyone"]
ds = db.datasets.find_one({"name": "swordnet"})
samples = db[ds["sample_collection_name"]] if ds else None
files = {str(d["_id"]): os.path.basename(d.get("filepath", "")) for d in samples.find({}, {"filepath": 1})} if samples is not None else {}
q = db["swordnet_queue__swordnet"]
now = datetime.datetime.utcnow()
out = {"annotating": [], "next": [], "done": [], "other": {}}
for d in q.find().sort([("priority", -1), ("created_at", 1)]):
    f = files.get(str(d["_id"]), str(d["_id"]))
    st = d.get("state")
    if st == "claimed":
        lease = d.get("lease_until")
        out["annotating"].append({"file": f, "by": d.get("claimed_by"), "since": iso(d.get("claimed_at")),
                                  "lease_until": iso(lease), "expired": bool(lease and lease < now)})
    elif st == "done":
        out["done"].append({"file": f, "by": d.get("claimed_by"), "at": iso(d.get("finished_at"))})
    elif st in (None, "pending", "queued", "ready"):
        out["next"].append({"file": f, "waiting_since": iso(d.get("created_at"))})
    else:
        out["other"].setdefault(st, []).append(f)
out["done"].sort(key=lambda x: x["at"] or "", reverse=True)
out["counts"] = {k: len(out[k]) for k in ("annotating", "next", "done")}
out["counts"]["images"] = len(files)
logs = {}
for name in ("api", "agent", "import"):
    p = os.path.join(PROJECT, "data", "logs", name + ".log")
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2); fh.seek(max(0, fh.tell() - 6000))
            logs[name] = fh.read().decode("utf-8", "replace").splitlines()[-25:]
    except OSError:
        logs[name] = []
out["logs"] = logs
out["time"] = time.time()
print(json.dumps(out))
'''


def live(seat=None):
    seat = seat or SEATS["agent"]
    script = "PROJECT=" + json.dumps(PROJECT) + "\n" + _QUERY
    raw = kube.sh(seat, f"/opt/conda/bin/python - <<'PY'\n{script}\nPY", timeout=60)
    data = json.loads(raw.strip().splitlines()[-1])
    data["seat"] = seat
    return data
