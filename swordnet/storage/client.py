"""The one way the web UI and the bridge reach the team database.

    SWORDNET_DB=cluster (default)  every operation is a `kubectl exec` into the database seat
                                   (pod_api.py there). Nothing on this computer imports FiftyOne,
                                   so no MongoDB starts here. Only image files are cached.
    SWORDNET_DB=local              the same operations run in this process against a local
                                   FiftyOne. Used by the pod itself and by the tests.

Read-only calls the UI polls (summary, run lists, logs) are cached for a few seconds, because each
exec costs ~5 s on this cluster.
"""
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from swordnet import config

MODE = os.environ.get("SWORDNET_DB", "cluster")
_cache, _cache_lock = {}, threading.Lock()


class DBError(RuntimeError):
    """The team database refused the request or could not be reached; the text says which."""


def call(op, req=None, ttl=0):
    """Run one database operation. With ttl > 0, a recent identical answer is reused."""
    key = (op, repr(sorted((req or {}).items()))) if ttl else None
    if key:
        with _cache_lock:
            hit = _cache.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
    if MODE == "local":
        from swordnet.storage import pod_api
        try:
            res = pod_api.OPS[op](req or {})
        except LookupError as e:
            raise DBError(str(e))
        if isinstance(res, dict) and "error" in res:
            raise DBError(res["error"])
    else:
        from swordnet.storage import remote
        try:
            res = remote.call(op, req)
        except remote.ClusterError as e:
            raise DBError(str(e))
    if key:
        with _cache_lock:
            _cache[key] = (time.time(), res)
    return res


def forget():
    """Drop cached reads after a write, so the next poll shows it."""
    with _cache_lock:
        _cache.clear()


def image_file(sample_id, answer=None):
    """Local path to show an image. In cluster mode, files are fetched once and cached."""
    if MODE == "local":
        from swordnet.storage import store
        s = store.get_sample(sample_id)
        return Path(s.filepath) if s is not None else None
    from swordnet.storage import remote
    if answer and answer.get("file_b64"):
        return remote.cache_file(sample_id, answer["file_b64"], answer["file_ext"])
    path = remote.cache_file(sample_id)
    if path is None:
        with _fetch_lock:
            path = remote.cache_file(sample_id)
            if path is None:
                remote.prefetch([sample_id, *_wanted])
                _wanted.clear()
                path = remote.cache_file(sample_id)
    return path


_fetch_lock, _wanted = threading.Lock(), set()


def want_files(ids):
    """Images a page is about to show: the next fetch bundles them all into one download."""
    if MODE != "local":
        from swordnet.storage import remote
        _wanted.update(i for i in ids if remote.cache_file(i) is None)


def ingest(items, files):
    """Add image files with their records. items[i]['file'] names a key of `files` (local paths)."""
    if MODE == "local":
        batch = f"b{uuid.uuid4().hex[:10]}"
        dst = config.HOME / "incoming" / batch
        dst.mkdir(parents=True, exist_ok=True)
        for it in items:
            shutil.copy2(files[it["file"]], dst / Path(it["file"]).name)
        res = call("ingest", {"batch": batch, "items": items})
    else:
        from swordnet.storage import remote
        try:
            res = remote.ingest(items, files)
        except remote.ClusterError as e:
            raise DBError(str(e))
    forget()
    return res


def stage_dir():
    """A scratch folder for files on their way into the database."""
    return Path(tempfile.mkdtemp(prefix="swordnet-stage-"))
