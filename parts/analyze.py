"""Analyzer + report: answers the two questions from the logged runs (scheduler plan, Part 4).

    1. How many annotated images does each model size need to reach the "decent" target?
    2. Which model is the best balance of small, fast and accurate for a phone?

Everything here is computed in code, the same way every time. gpt-oss-20b only writes the
explanation of a summary it is handed, and the report is complete without it.

Team settings (placeholders until the team picks them):
    TARGET_MAP50   held-out mAP50 that counts as "decent"
    BUDGET_MS      max CPU ms per image (pod CPU, a stand-in for a phone)
    BUDGET_MB      max phone file size (8-bit TFLite; best.pt size if the export failed)
    NOISE_FALLBACK used only until the seed-noise runs have finished
"""
import json
import random
from collections import defaultdict

TARGET_MAP50 = 0.70
BUDGET_MS = 150.0
BUDGET_MB = 20.0
NOISE_FALLBACK = 0.02

SIZE_ORDER = ("nano", "medium", "large")


def _done(records, kinds):
    return [r for r in records if r.get("status") == "done" and r.get("kind") in kinds
            and r.get("mAP50_95") is not None and r.get("mAP50") is not None]


def noise(records, fallback=NOISE_FALLBACK):
    """Spread (max - min) of the same job run with different seeds, for mAP50 and mAP50-95."""
    groups = defaultdict(list)
    for r in _done(records, ("grid", "noise")):
        groups[(r["level_id"], r["size"], json.dumps({k: v for k, v in (r.get("settings") or {}).items() if k != "seed"}, sort_keys=True))].append(r)
    spreads50, spreads = [], []
    for rs in groups.values():
        if len({r.get("seed") for r in rs}) >= 2:
            spreads50.append(max(r["mAP50"] for r in rs) - min(r["mAP50"] for r in rs))
            spreads.append(max(r["mAP50_95"] for r in rs) - min(r["mAP50_95"] for r in rs))
    if not spreads:
        return {"mAP50": fallback, "mAP50_95": fallback, "measured": False}
    return {"mAP50": round(max(spreads50), 4), "mAP50_95": round(max(spreads), 4), "measured": True}


def curves(records):
    """{size: [{"level_id", "n_images", "mAP50", "mAP50_95", "runs"}]} sorted by n_images (seeds averaged)."""
    pts = defaultdict(list)
    for r in _done(records, ("grid", "noise")):
        pts[(r["size"], r["level_id"])].append(r)
    out = defaultdict(list)
    for (size, level), rs in pts.items():
        out[size].append({"level_id": level, "n_images": rs[0].get("n_train_images"),
                          "mAP50": round(sum(r["mAP50"] for r in rs) / len(rs), 4),
                          "mAP50_95": round(sum(r["mAP50_95"] for r in rs) / len(rs), 4), "runs": len(rs)})
    return {s: sorted(v, key=lambda p: (p["n_images"] or 0)) for s, v in out.items()}


def images_needed(curve, target, noise50):
    """Smallest n where mAP50 >= target and every larger level stays >= target - noise; else None."""
    for i, p in enumerate(curve):
        if p["mAP50"] >= target and all(q["mAP50"] >= target - noise50 for q in curve[i + 1:]):
            return p["n_images"]
    return None


def _size_mb(r):
    return (r.get("phone_mb"), "phone") if r.get("phone_mb") is not None else (r.get("file_mb"), "best.pt")


def frontier(cands):
    """Candidates not beaten on all of: higher mAP50-95, lower cpu_ms, smaller size."""
    def beats(a, b):
        sa, sb = _size_mb(a)[0], _size_mb(b)[0]
        if None in (a.get("cpu_ms"), b.get("cpu_ms"), sa, sb):
            return False
        no_worse = a["mAP50_95"] >= b["mAP50_95"] and a["cpu_ms"] <= b["cpu_ms"] and sa <= sb
        better = a["mAP50_95"] > b["mAP50_95"] or a["cpu_ms"] < b["cpu_ms"] or sa < sb
        return no_worse and better
    return [c for c in cands if not any(beats(o, c) for o in cands if o is not c)]


def analyze(records, target=TARGET_MAP50, budget_ms=BUDGET_MS, budget_mb=BUDGET_MB, noise_fallback=NOISE_FALLBACK):
    nz = noise(records, noise_fallback)
    cv = curves(records)
    needed = {s: images_needed(c, target, nz["mAP50"]) for s, c in cv.items()}
    grid = _done(records, ("grid",))
    sizes = sorted({r["size"] for r in grid}, key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 99)
    common = [lv for lv in sorted({r["level_id"] for r in grid}, reverse=True)
              if all(any(r["level_id"] == lv and r["size"] == s for r in grid) for s in sizes)]
    summary = {"target_mAP50": target, "budget_ms": budget_ms, "budget_mb": budget_mb, "noise": nz,
               "curves": cv, "images_needed": needed, "level": None, "candidates": [], "frontier": [],
               "recommendation": None, "reason": None,
               "runs": {"done": len([r for r in records if r.get("status") == "done"]),
                        "failed": len([r for r in records if r.get("status") in ("failed", "timeout", "stopped")])}}
    if not grid:
        summary["reason"] = "no finished grid runs yet"
        return summary
    if not common:
        summary["reason"] = "no data level has finished runs for every size yet"
        return summary
    level = common[0]
    cands = []
    for s in sizes:
        rs = [r for r in grid if r["level_id"] == level and r["size"] == s and r.get("seed", 0) == 0] or \
             [r for r in grid if r["level_id"] == level and r["size"] == s]
        r = rs[-1]
        mb, src = _size_mb(r)
        cands.append({"size": s, "model": r.get("model"), "run_id": r["run_id"], "mAP50": r["mAP50"],
                      "mAP50_95": r["mAP50_95"], "cpu_ms": r.get("cpu_ms"), "size_mb": mb, "size_source": src,
                      "phone_mb": r.get("phone_mb"), "file_mb": r.get("file_mb"), "params": r.get("params"),
                      "phone_error": r.get("phone_error")})
    front = frontier(cands)
    summary.update(level=level, candidates=cands, frontier=[c["size"] for c in front])

    def fails(c):
        why = []
        if c["mAP50"] < target:
            why.append(f"accuracy {c['mAP50']:.2f} < target {target:.2f}")
        if c["cpu_ms"] is None or c["cpu_ms"] > budget_ms:
            why.append(f"speed {c['cpu_ms']} ms > {budget_ms} ms" if c["cpu_ms"] is not None else "speed not measured")
        if c["size_mb"] is None or c["size_mb"] > budget_mb:
            why.append(f"size {c['size_mb']} MB > {budget_mb} MB" if c["size_mb"] is not None else "size not measured")
        return why

    fits = [c for c in front if not fails(c)]
    if not fits:
        summary["reason"] = "nothing fits yet: " + "; ".join(f"{c['size']}: {', '.join(fails(c))}" for c in front)
        return summary
    best = max(c["mAP50_95"] for c in fits)
    within = [c for c in fits if c["mAP50_95"] >= best - nz["mAP50_95"]]
    pick = min(within, key=lambda c: (c["params"] if c["params"] is not None else float("inf"), c["cpu_ms"]))
    top = max(fits, key=lambda c: c["mAP50_95"])
    faster = (top["cpu_ms"] / pick["cpu_ms"]) if pick["cpu_ms"] else None
    summary["recommendation"] = pick
    summary["reason"] = (f"{pick['size']} is within noise ({nz['mAP50_95']:.3f}"
                         f"{'' if nz['measured'] else ', assumed'}) of the most accurate model that fits"
                         + (f", and {faster:.1f}x faster than {top['size']}" if faster and top is not pick else "")
                         + (f"; its size is best.pt, because phone export failed" if pick["size_source"] != "phone" else ""))
    return summary


# --- report -------------------------------------------------------------------

REPORT_PROMPT = """Here is the summary of a day of YOLO training runs for SwordNet, a sword and stick detector meant to run on a phone. All numbers were computed in code and are correct; scores are on a held-out set the trainer never saw.

{summary}

In 6 to 10 short bullet points, explain what the runs show about how many images each model size needed, which model the team should ship and why, and the single most useful thing to do next. Cite the numbers."""


def numbers_markdown(s):
    lines = ["# SwordNet: images needed and model choice", "",
             f"Target: held-out mAP50 ≥ {s['target_mAP50']}. Phone budget: ≤ {s['budget_ms']} ms per image "
             f"(pod CPU), ≤ {s['budget_mb']} MB. Run-to-run noise: ±{s['noise']['mAP50_95']} mAP50-95"
             f"{'' if s['noise']['measured'] else ' (assumed; seed runs not finished)'}.", "",
             f"Runs: {s['runs']['done']} finished, {s['runs']['failed']} failed or stopped.", "",
             "## Images needed to reach the target", "", "| size | images |", "|---|---|"]
    lines += [f"| {k} | {v if v is not None else 'not reached yet'} |" for k, v in s["images_needed"].items()]
    lines += ["", "## Learning curves (held-out)", "", "| size | images | mAP50 | mAP50-95 | runs |", "|---|---|---|---|---|"]
    for size, pts in s["curves"].items():
        lines += [f"| {size} | {p['n_images']} | {p['mAP50']} | {p['mAP50_95']} | {p['runs']} |" for p in pts]
    if s["candidates"]:
        lines += ["", f"## Size, speed and accuracy on level {s['level']}", "",
                  "| size | mAP50 | mAP50-95 | CPU ms | size MB | params | on frontier |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {c['size']} | {c['mAP50']} | {c['mAP50_95']} | {c['cpu_ms']} | {c['size_mb']} ({c['size_source']}) | "
                  f"{c['params']} | {'yes' if c['size'] in s['frontier'] else 'no'} |" for c in s["candidates"]]
    rec = s["recommendation"]
    lines += ["", "## Recommendation", "", (f"**{rec['size']}** ({rec['model']}, run {rec['run_id']}). " if rec else "**None yet.** ")
              + (s["reason"] or "")]
    return "\n".join(lines)


def write_report(summary, chat=None):
    """Numbers section always; gpt-oss analysis if `chat` (e.g. gptoss_chat) works, else the reason it didn't."""
    md = numbers_markdown(summary)
    if chat is None:
        return md + "\n\n## Analysis (gpt-oss-20b)\n\n_Not requested._\n"
    try:
        text = chat(REPORT_PROMPT.format(summary=json.dumps(summary)))
        return md + "\n\n## Analysis (gpt-oss-20b)\n\n" + text + "\n"
    except Exception as e:
        return md + f"\n\n## Analysis (gpt-oss-20b)\n\n_gpt-oss failed: {type(e).__name__}: {e}_\n"


class GptOssError(RuntimeError):
    pass


def gptoss_chat(base_url, max_tokens=3000, timeout=300):
    """A `chat` callable for the shared gpt-oss-20b (trainium-agent-labs/gptoss/README.md):
    two deployments under one URL (/agg/v1, /disagg/v1) with failover on connection errors and 5xx;
    the certificate does not match the host, so verify=False (review aafd970 N3); it reasons before
    answering, so empty content with finish_reason=length is an error to show, not 'unavailable'."""
    import httpx

    if not base_url:
        raise GptOssError("GPTOSS_BASE_URL is not set")

    def chat(prompt):
        paths = ["/agg/v1", "/disagg/v1"]
        random.shuffle(paths)
        body = {"model": "gpt-oss-20b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        errors = []
        with httpx.Client(verify=False, timeout=timeout) as client:
            for path in paths:
                try:
                    r = client.post(f"{base_url.rstrip('/')}{path}/chat/completions", json=body)
                except httpx.HTTPError as e:
                    errors.append(f"{path}: {type(e).__name__}: {e}")
                    continue
                if r.status_code >= 500:
                    errors.append(f"{path}: HTTP {r.status_code}")
                    continue
                if r.status_code >= 400:
                    raise GptOssError(f"{path}: HTTP {r.status_code}: {r.text[:300]}")
                choice = r.json()["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                if not text:
                    raise GptOssError(f"empty answer (finish_reason={choice.get('finish_reason')}); "
                                      "the prompt is likely too long; shorten it rather than raising max_tokens")
                if choice.get("finish_reason") == "length":
                    text += "\n\n_(cut off at the token limit)_"
                return text
        raise GptOssError("unreachable: " + "; ".join(errors) +
                          " (a hang with no HTTP response usually means this network is not allowlisted)")

    return chat
