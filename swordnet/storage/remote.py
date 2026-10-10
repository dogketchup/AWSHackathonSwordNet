"""Thin client for the team database on the cluster (pod_api.py runs there).

Laptops keep no database: every read and write is one `kubectl exec` into the database seat.
The only local state is a cache of image files (data/cache/), so the web UI can show an image
without fetching it again.
"""
import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from swordnet import config
from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS

SEAT = SEATS.get("database", SEATS["agent"])
TEAM_HOME = "/workspace/swordnet-team"
POD_ENV = ("FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017 SWORDNET_FO_DATASET=swordnet_team "
           f"SWORDNET_HOME={TEAM_HOME} SWORDNET_SHARED=0 FIFTYONE_DO_NOT_TRACK=true")
CACHE = config.HOME / "cache"


class ClusterError(RuntimeError):
    """The team database refused the request or could not be reached; the text says which."""


def call(op, req=None, timeout=180):
    script = f"cd {kube.APP_DIR} && {POD_ENV} python3 -m swordnet.storage.pod_api {op}"
    try:
        out = kube.sh(SEAT, script, input=json.dumps(req or {}), timeout=timeout)
    except Exception as e:
        raise ClusterError(f"cluster unreachable ({str(e).strip().splitlines()[-1][:160]}); try again")
    lines = [l for l in out.strip().splitlines() if l.startswith("{")]
    if not lines:
        raise ClusterError(f"no answer from the team database: {out.strip()[-200:]}")
    res = json.loads(lines[-1])
    if "error" in res:
        raise ClusterError(res["error"])
    return res


def cache_file(sample_id, b64=None, ext=None):
    """Path of a cached image, writing it first if its bytes came inline with an answer."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if b64 is not None:
        path = CACHE / f"{sample_id}{ext}"
        path.write_bytes(base64.b64decode(b64))
        return path
    hits = list(CACHE.glob(f"{sample_id}.*"))
    return hits[0] if hits else None


def prefetch(ids):
    """Download any of these images not cached yet, in one bundle."""
    missing = [i for i in ids if cache_file(i) is None]
    if not missing:
        return 0
    pack = call("pack", {"ids": missing})["path"]
    try:
        data = kube.fetch_bytes(SEAT, pack)
    finally:
        kube.sh(SEAT, f"rm -f {pack}", check=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(CACHE, filter="data")
    return len(missing)


def ingest(items, files):
    """Upload image files plus their records (labels, state) and register them in the database.

    items: [{file, collection, origin, status, labels, autolabels, labeled_by, labeled_at, ...}]
    files: {file name in items -> local path}
    """
    if not items:
        return {"added": [], "duplicates": []}
    batch = f"b{int(time.time() * 1000)}"
    stage = Path(tempfile.mkdtemp(prefix="swordnet-in-")) / batch
    stage.mkdir()
    for it in items:
        shutil.copy2(files[it["file"]], stage / Path(it["file"]).name)
    try:
        kube.push(SEAT, stage, f"{TEAM_HOME}/incoming")
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)
    return call("ingest", {"batch": batch, "items": items}, timeout=600)
