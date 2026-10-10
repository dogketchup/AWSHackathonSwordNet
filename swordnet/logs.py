"""Logging layer: one searchable log for every component, in the team database on the cluster.

    from swordnet.logs import log
    log("bridge", "run 7 started on seat-146", run_id=7, seat="seat-146")
    log("webui", "upload failed", level="error", filename=name)

On a laptop (SWORDNET_DB=cluster) log() queues entries and a background thread sends them in
batches, one exec every few seconds; queries go through the same front door as everything else
(storage/client.py). On the database seat itself (SWORDNET_DB=local) entries are written directly.

Each entry: {ts, level, source, msg, run_id?, seat?, ...extra}. Sources used so far:
  webui      uploads, labeling (who labeled what), rejects
  bridge     setup, job starts/finishes, plans, pulls
  seat-NNN   lines tailed from a pod's job logs (train.log, autolabel, refine)
  agent      planner decisions and validator rejections, report generation
The web UI's Logs tab reads them (GET /api/logs). Logs never block work: if a write fails, the
entry is kept for the next try (laptop) or printed to stderr (pod).
"""
import os
import re
import sys
import threading
import time

LEVELS = ("debug", "info", "warning", "error")
LOCAL = os.environ.get("SWORDNET_DB", "cluster") == "local"
_pending, _pending_lock, _sender = [], threading.Lock(), None


def _coll():
    from swordnet.storage import store
    return store.coll("logs")


def flush():
    """Send queued entries to the team database now (the bridge calls this before exiting)."""
    with _pending_lock:
        batch = _pending[:500]
        del _pending[:500]
    if not batch:
        return
    try:
        from swordnet.storage import client
        client.call("log", {"entries": batch})
    except Exception as e:  # keep them for the next try; never break the caller
        with _pending_lock:
            _pending[:0] = batch[-5000:]
        print(f"[log send failed, will retry: {str(e)[:120]}]", file=sys.stderr)


def _send_loop():
    while True:
        time.sleep(3)
        flush()


def _store(docs):
    global _sender
    if LOCAL:
        try:
            _coll().insert_many(docs)
        except Exception as e:  # never let logging break the caller
            print(f"[log write failed: {e}] {len(docs)} entries", file=sys.stderr)
        return
    with _pending_lock:
        _pending.extend(docs)
        if _sender is None:
            _sender = threading.Thread(target=_send_loop, daemon=True, name="log-sender")
            _sender.start()


def log(source, msg, level="info", **fields):
    _store([{"ts": time.time(), "level": level, "source": source, "msg": str(msg), **fields}])
    if level in ("warning", "error") or source == "bridge":
        print(f"{time.strftime('%H:%M:%S')} {level.upper():7} {source}: {msg}", file=sys.stderr)


def log_lines(source, lines, **fields):
    """Bulk insert lines tailed from a pod log."""
    now = time.time()
    docs = [{"ts": now, "level": _guess_level(l), "source": source, "msg": l, **fields} for l in lines if l.strip()]
    if docs:
        _store(docs)


def _guess_level(line):
    low = line.lower()
    if "traceback" in low or "error" in low or "exception" in low:
        return "error"
    if "warning" in low:
        return "warning"
    return "info"


def query(source=None, run_id=None, level=None, after=None, text=None, limit=500):
    if not LOCAL:
        from swordnet.storage import client
        return client.call("logs", {"source": source, "run_id": run_id, "level": level, "after": after,
                                    "text": text, "limit": limit}, ttl=3)["entries"]
    q = {}
    if source:
        q["source"] = source
    if run_id is not None:
        q["run_id"] = run_id
    if level:
        q["level"] = {"$in": list(LEVELS[LEVELS.index(level):])}
    if after:
        q["ts"] = {"$gt": after}
    if text:
        q["msg"] = {"$regex": re.escape(text), "$options": "i"}
    docs = list(_coll().find(q, {"_id": 0}).sort("ts", -1).limit(limit))
    return docs[::-1]


def sources():
    if not LOCAL:
        from swordnet.storage import client
        return client.call("logs", {"limit": 1}, ttl=30)["sources"]
    return sorted(_coll().distinct("source"))
