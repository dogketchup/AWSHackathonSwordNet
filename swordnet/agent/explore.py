"""Explorer: gpt-oss reads the finished runs, decides whether the network should scale up or down,
and proposes the next round of training runs as hypotheses. The bridge's scheduler launches them.

    1. numbers (code): the same compact run rows the Best model tab uses (agent/pick.py)
    2. propose (gpt-oss): {"scale": "up"|"down"|"keep", "why", "runs": [one per idle seat]}
    3. check (code): plan.validate (seat/model/limits) plus "not already tried"; the reasons for a
       rejection go back to gpt-oss, up to 3 tries, then the planner's fallback schedule
    4. launch (bridge): cli.cycle with this planner, on idle trainer seats only, epochs capped

One press of the button is one round; nothing loops on its own.
"""
import argparse
import json
import threading
import time

from swordnet import config
from swordnet.agent import llm, pick, plan

STATE = config.METRICS_DIR / "explore.json"
SCALES = ("up", "down", "keep")
_lock = threading.Lock()

PROMPT = """You tune SwordNet, a YOLO-OBB detector for swords and sticks that must run on a phone.
Goal: mAP50 >= {target}, at most {ms} ms/image and {mb} MB of weights.

Finished runs (exact numbers):
{numbers}

Configs already tried (model, imgsz, epochs), do not repeat them: {tried}

Idle seats this round and the model each one trains ("flex" = you pick from {flex}):
{seats}
Dataset: {dataset}

Decide whether the network should scale up (bigger model, larger imgsz, more epochs) because accuracy
is short of the goal, scale down because a smaller model would still meet it and be faster, or keep.
Then give exactly one run per seat; each run tests one hypothesis.
Limits: epochs {epochs}, imgsz {imgsz} (multiple of 32), batch {batch}.

Answer with JSON only:
{{"scale": "up|down|keep", "why": "two sentences",
  "runs": [{{"seat": "...", "model": "...", "epochs": N, "imgsz": N, "batch": N, "reason": "the hypothesis"}}]}}
{feedback}"""


def _key(model, cfg):
    return (model, cfg.get("imgsz", 640), cfg.get("epochs"))


def tried(history):
    return {_key(r.get("model"), r.get("config") or {}) for r in history if r.get("model")}


def check(p, seats, history, n_train=None):
    """plan.validate's problems, plus a missing scale and repeats of runs already tried."""
    errs = plan.validate(p, seats, n_train)
    if p.get("scale") not in SCALES:
        errs.append(f'"scale" must be one of {list(SCALES)}')
    done = tried(history)
    seen = set()
    for r in p.get("runs") or []:
        if not isinstance(r, dict):
            continue
        k = _key(r.get("model"), r)
        if k in done:
            errs.append(f"{r.get('seat')}: {k} was already tried; change the hypothesis")
        elif k in seen:
            errs.append(f"{r.get('seat')}: {k} repeats another run in this plan")
        seen.add(k)
    return errs


def propose(seats, manifest, history, target, ms, mb, tries=3):
    """A checked plan for these seats, or the planner's fallback (marked by "by")."""
    rows, _ = pick.runs()
    feedback = ""
    for attempt in range(tries):
        prompt = PROMPT.format(target=target, ms=ms, mb=mb, numbers=pick.numbers_text(rows) or "none yet",
                               tried=sorted(map(list, tried(history))) or "none", flex=plan.FLEX_MODELS,
                               seats=json.dumps(seats), dataset=json.dumps(manifest)[:600],
                               epochs=plan.LIMITS["epochs"], imgsz=plan.LIMITS["imgsz"],
                               batch=plan.LIMITS["batch"], feedback=feedback)
        try:
            p = llm.extract_json(llm.chat(prompt, which="gptoss", max_tokens=3000))
        except Exception as e:
            feedback = f"\nYour previous answer failed ({type(e).__name__}: {str(e)[:200]}). Reply with the JSON object only."
            continue
        errs = check(p, seats, history, manifest.get("n_train"))
        if not errs:
            allowed = set(plan.LIMITS) | plan.OPTIONAL | {"seat", "model", "reason"}
            return {"scale": p["scale"], "why": str(p.get("why", ""))[:600], "by": "gpt-oss",
                    "attempts": attempt + 1, "runs": [{k: v for k, v in r.items() if k in allowed} for r in p["runs"]]}
        feedback = "\nYour previous plan was rejected:\n- " + "\n- ".join(errs) + "\nFix exactly these."
    out = plan.fallback(seats, history)
    return {"scale": "keep", "why": f"gpt-oss gave no valid plan; fallback schedule. {feedback.strip()}"[:600],
            "by": "fallback", "runs": out["runs"]}


def _save(data):
    pick._save(STATE, {**data, "time": time.time()})


def last():
    return pick._load(STATE, None)


def launch(target, ms, mb, max_epochs=60):
    """One round: cli.cycle on the idle trainer seats with gpt-oss as the planner. Returns False if a
    round is already starting or nothing could start."""
    if not _lock.acquire(blocking=False):
        return False
    try:
        from swordnet.bridge import cli

        def planner(seats, manifest, history):
            p = propose(seats, manifest, history, target, ms, mb)
            _save({"status": "launching", **p, "max_epochs": max_epochs})
            return {"runs": [dict(r) for r in p["runs"]], "notes": f"explore ({p['by']}): scale {p['scale']}: {p['why']}"}

        _save({"status": "planning"})
        ok = cli.cycle(argparse.Namespace(planner=planner, no_wait=True, max_epochs=max_epochs, poll=60))
        _save({**(last() or {}), "status": "launched" if ok else "nothing started (no labeled data or no idle seat)"})
        return ok
    except Exception as e:
        _save({**(last() or {}), "status": f"failed: {type(e).__name__}: {str(e)[:300]}"})
        return False
    finally:
        _lock.release()
