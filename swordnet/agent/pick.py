"""Best model picker: chooses which trained model to download, from the runs already logged.
No model is re-run.

Runs come from the team model registry on the cluster (bridge/registry.py), so every teammate's
published runs count. If the cluster can't be reached, the runs on this laptop are used instead.

    1. numbers per run (code): exact metrics from each finished run, ~150 tokens each
    2. compact logs (gpt-oss): each run's log (or its published report.md if this laptop has no log
       for it), progress bars dropped, split into chunks under
       gpt-oss's input limit, summarized; summaries are cached per run in metrics/log_summaries.json
    3. rank (code): keep runs that meet the target and fit the speed/size budget; the code's pick is
       the smallest model within `noise` of the most accurate one
    4. decide (gpt-oss): gets the numbers, summaries and ranking, returns {"pick", "why"}; a pick
       that is not one of the candidates is replaced by the code's pick
Changing the budget only re-runs step 3 (`current`); step 2 and 4 run when someone presses Analyze.
"""
import json
import re
from pathlib import Path

from swordnet import config
from swordnet.agent import llm

SUMMARIES = config.METRICS_DIR / "log_summaries.json"
DECISION = config.METRICS_DIR / "best_model.json"
CHUNK_CHARS = 12000  # well under gpt-oss's 8192-token input limit, with room for the prompt
DEFAULTS = {"target": 0.5, "ms": 150.0, "mb": 20.0, "noise": 0.02}  # team sets these on the tab


def _num(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else None


def _local_runs():
    """Finished runs in this laptop's store, keyed like the registry (run-<id>)."""
    from swordnet.storage import store

    out = []
    for r in store.list_runs(limit=1000):
        m = r.get("metrics") or {}
        if r.get("status") != "done" or not m:
            continue
        w = r.get("weights")
        out.append({"key": f"run-{r['id']}", "run": r["id"], "model": r.get("model"),
                    "n_train": r.get("n_train") or (m.get("dataset") or {}).get("n_train"),
                    **{k: m.get(k) for k in ("best_mAP50", "best_mAP50_95", "ms_per_image", "best_epoch", "epochs_run")},
                    "findings": m.get("findings") or [],
                    "weights_mb": round(Path(w).stat().st_size / 1e6, 1) if w and Path(w).exists() else None})
    return out


def runs():
    """Finished runs as flat rows, plus where they came from ("registry" or "this laptop")."""
    try:
        from swordnet.bridge import registry
        entries, source = registry.models()[0], "registry"
    except Exception:
        entries, source = [], "registry unreachable"
    if not entries:  # cluster unreachable, or nothing published yet: use this laptop's runs
        entries, source = _local_runs(), "this laptop" + (" (registry unreachable)" if source != "registry" else "")
    rows = [{
        "id": str(e["key"]), "run": e.get("run"), "model": e.get("model"), "n_train": e.get("n_train"),
        "mAP50": _num(e.get("best_mAP50")), "mAP50_95": _num(e.get("best_mAP50_95")),
        "ms": _num(e.get("ms_per_image")), "mb": _num(e.get("weights_mb")),
        "best_epoch": e.get("best_epoch"), "epochs_run": e.get("epochs_run"),
        "findings": (e.get("findings") or [])[:3], "published_by": e.get("published_by"),
    } for e in entries]
    return rows, source


def numbers_text(rows):
    keys = ("id", "model", "n_train", "mAP50", "mAP50_95", "ms", "mb", "best_epoch", "epochs_run")
    return "\n".join(json.dumps({**{k: r[k] for k in keys}, "findings": [f[:120] for f in r["findings"]]}) for r in rows)


def rank(rows, target, ms, mb, noise):
    """Candidates that meet the target and budget (most accurate first) and the code's pick."""
    def why_not(r):
        out = []
        if r["mAP50"] is None or r["mAP50"] < target:
            out.append(f"mAP50 {r['mAP50']} < {target}")
        if r["ms"] is None or r["ms"] > ms:
            out.append(f"{r['ms']} ms > {ms}")
        if r["mb"] is None or r["mb"] > mb:
            out.append(f"{r['mb']} MB > {mb}" if r["mb"] is not None else "weights size unknown")
        return out

    fits = sorted((r for r in rows if not why_not(r)), key=lambda r: -r["mAP50_95"])
    table = [{**{k: r[k] for k in ("id", "model", "n_train", "mAP50", "mAP50_95", "ms", "mb", "published_by")},
              "fits": not why_not(r), "why_not": ", ".join(why_not(r))} for r in rows]
    if not fits:
        return {"candidates": [], "code_pick": None, "table": table,
                "reason": "no run meets the target and fits the budget"}
    best = fits[0]["mAP50_95"]
    pick = min((r for r in fits if r["mAP50_95"] >= best - noise), key=lambda r: (r["mb"], r["ms"]))
    return {"candidates": [r["id"] for r in fits], "code_pick": pick["id"], "table": table,
            "reason": f"{pick['id']} is the smallest within {noise} mAP50-95 of the best that fits"}


def _load(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _log_text(row):
    """This laptop's log lines for the run; else the run's report.md from the registry."""
    from swordnet import logs

    lines = []
    if row.get("run") is not None:
        lines = [str(e.get("msg") or "") for e in logs.query(run_id=row["run"], limit=5000)]
    if not any(l.strip() for l in lines):
        try:
            from swordnet.bridge import kube, registry
            lines = kube.fetch_bytes(registry.SEAT, f"{registry.REG}/runs/{row['id']}/report.md").decode(errors="replace").splitlines()
        except Exception:
            lines = []
    return "\n".join(l for l in lines if l.strip() and "━" not in l and "%|" not in l)  # drop progress-bar redraws


def summarize_log(row):
    """Compact one run's log with gpt-oss: chunk, summarize each chunk, merge if more than one."""
    text = _log_text(row)
    if not text:
        return "no log or report found"
    chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    parts = [llm.chat(f"Summarize this YOLO training log in at most 60 words: warnings, errors and anything unusual.\n\n{c}",
                      which="gptoss", max_tokens=1500) for c in chunks]
    if len(parts) == 1:
        return parts[0]
    return llm.chat("Merge these notes about one training run into at most 60 words.\n\n" + "\n\n".join(parts),
                    which="gptoss", max_tokens=1500)


def analyze(target, ms, mb, noise):
    """Steps 1-4. Summarizes only runs without a cached summary."""
    rows, source = runs()
    ranking = rank(rows, target, ms, mb, noise)
    summaries = _load(SUMMARIES, {})
    for r in rows:
        if str(r["id"]) not in summaries:
            try:
                summaries[str(r["id"])] = summarize_log(r)
            except Exception as e:
                summaries[str(r["id"])] = f"(log summary failed: {e})"
            _save(SUMMARIES, summaries)
    decision = {"pick": ranking["code_pick"], "why": ranking["reason"], "by": "code",
                "budget": {"target": target, "ms": ms, "mb": mb, "noise": noise}}
    if ranking["candidates"]:
        prompt = (f"Choose which SwordNet model (sword and stick detector for a phone) to ship.\n"
                  f"Runs (numbers are exact):\n{numbers_text(rows)}\n\n"
                  f"Log notes: {json.dumps({k: v for k, v in summaries.items() if k in [r['id'] for r in rows]})}\n\n"
                  f"Candidates that meet the target and budget: {ranking['candidates']}. "
                  f"Code ranking says: {ranking['reason']}.\n"
                  'Reply with JSON only: {"pick": "<run id from the candidates>", "why": "<2 sentences>"}')
        try:
            ans = llm.extract_json(llm.chat(prompt, which="gptoss", max_tokens=3000))
            if str(ans.get("pick")) in ranking["candidates"]:
                decision.update(pick=str(ans["pick"]), why=str(ans.get("why", ""))[:600], by="gpt-oss")
            else:
                decision["why"] += f" (gpt-oss picked {ans.get('pick')!r}, not a candidate; using the code's pick)"
        except Exception as e:
            decision["why"] += f" (gpt-oss failed: {type(e).__name__}: {str(e)[:200]}; using the code's pick)"
    _save(DECISION, decision)
    return current(target, ms, mb, noise, (rows, source))


def current(target, ms, mb, noise, rows_source=None):
    """Step 3 only, plus the last decision. No gpt-oss call."""
    rows, source = rows_source or runs()
    ranking = rank(rows, target, ms, mb, noise)
    decision = _load(DECISION, None)
    if decision and decision.get("pick") not in ranking["candidates"]:
        decision = {**decision, "stale": True}
    return {**ranking, "decision": decision, "summaries": _load(SUMMARIES, {}), "source": source}


def download(key):
    """The run's best.pt: from the registry (verified chunks) into weights/, or this laptop's copy."""
    try:
        from swordnet.bridge import registry
        return registry.download(key, out=str(config.HOME / "weights"))[1]
    except (Exception, SystemExit):
        from swordnet.storage import store
        r = store.get_run(int(key.removeprefix("run-"))) if key.startswith("run-") and key[4:].isdigit() else None
        w = (r or {}).get("weights")
        return Path(w) if w and Path(w).exists() else None
