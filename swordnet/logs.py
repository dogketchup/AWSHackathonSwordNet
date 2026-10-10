"""Logging layer: one searchable log for every component, in the same FiftyOne MongoDB.

    from swordnet.logs import log
    log("bridge", "run 7 started on seat-146", run_id=7, seat="seat-146")
    log("webui", "upload failed", level="error", filename=name)

Each entry: {ts, level, source, msg, run_id?, seat?, ...extra}. Sources used so far:
  webui      uploads, labeling (who labeled what), rejects
  bridge     setup, job starts/finishes, plans, pulls
  seat-NNN   lines tailed from a pod's job logs (train.log, autolabel, refine)
  agent      planner decisions and validator rejections, report generation
The web UI's Logs tab reads them (GET /api/logs). Logs never block work: if the DB write
fails, the entry goes to stderr instead.
"""
import re
import sys
import time

LEVELS = ("debug", "info", "warning", "error")


def _coll():
    from swordnet.storage import store
    return store.coll("logs")


def log(source, msg, level="info", **fields):
    entry = {"ts": time.time(), "level": level, "source": source, "msg": str(msg), **fields}
    try:
        _coll().insert_one(entry)
    except Exception as e:  # never let logging break the caller
        print(f"[log write failed: {e}] {entry}", file=sys.stderr)
    if level in ("warning", "error") or source == "bridge":
        print(f"{time.strftime('%H:%M:%S')} {level.upper():7} {source}: {msg}", file=sys.stderr)


def log_lines(source, lines, **fields):
    """Bulk insert lines tailed from a pod log."""
    now = time.time()
    docs = [{"ts": now, "level": _guess_level(l), "source": source, "msg": l, **fields} for l in lines if l.strip()]
    if docs:
        try:
            _coll().insert_many(docs)
        except Exception as e:
            print(f"[log write failed: {e}] {len(docs)} lines from {source}", file=sys.stderr)


def _guess_level(line):
    low = line.lower()
    if "traceback" in low or "error" in low or "exception" in low:
        return "error"
    if "warning" in low:
        return "warning"
    return "info"


def query(source=None, run_id=None, level=None, after=None, text=None, limit=500):
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
    return sorted(_coll().distinct("source"))
