"""Planner: the agent picks the next training run for each trainer seat.

Runs on the agent seat (seat-145), called by the bridge:
    kubectl exec -i seat-145 -- python -m swordnet.agent.plan < request.json > plan.json

request.json: {"seats": {"seat-146": "yolo11s-obb", ...}, "dataset": manifest, "history": [run, ...]}
plan.json:    {"runs": [{"seat", "model", "epochs", "imgsz", "batch", "lr0", ..., "reason"}], "notes"}

This follows the starter kit's loop: the model proposes, a checker (validate()) grades, and
the *reason* for a rejection goes back into the next prompt. If the model cannot produce a
valid plan in 3 tries, or the server is down, we fall back to a deterministic schedule so
training never stalls on the agent.
"""
import json
import sys

from swordnet.agent import llm

LIMITS = {"epochs": (5, 300), "imgsz": (320, 1280), "batch": (2, 64), "lr0": (1e-5, 0.1)}
OPTIONAL = {"lr0", "mosaic", "mixup", "degrees", "fliplr", "scale", "cos_lr", "patience", "weight_decay"}

PROMPT = """You plan YOLO-OBB training runs for SwordNet, a detector for swords and sticks
(oriented boxes). Training is on CPU (11 cores per seat), so a run's cost grows with
epochs x images x model size. Each seat has a fixed model size.

Seats and their models:
{seats}

Current dataset snapshot:
{dataset}

Previous runs, newest first (config, key metrics, evaluator findings):
{history}

Choose ONE run per seat. Use the history: if a model overfit, lower epochs or add augmentation;
if it was still improving, raise epochs; if the val set is small, do not chase tiny mAP changes.
Keep imgsz a multiple of 32. Allowed keys: epochs, imgsz, batch, lr0, mosaic, mixup, degrees,
fliplr, scale, cos_lr, patience, weight_decay.

Answer with JSON only:
{{"runs": [{{"seat": "...", "model": "...", "epochs": N, "imgsz": N, "batch": N, "reason": "one sentence"}}],
  "notes": "what you are testing across these runs"}}
{feedback}"""


def summarize_history(history, n=8):
    out = []
    for r in history[:n]:
        m = r.get("metrics") or {}
        out.append({
            "run": r.get("id"), "model": r.get("model"), "config": r.get("config"),
            "n_train": r.get("n_train"), "status": r.get("status"),
            "best_mAP50_95": m.get("best_mAP50_95"), "best_epoch": m.get("best_epoch"),
            "epochs_run": m.get("epochs_run"), "gap_ratio": m.get("gap_ratio"), "fps": m.get("fps"),
            "findings": m.get("findings", [])[:4],
        })
    return out


def validate(plan, seats):
    """Return a list of concrete problems; empty means the plan is usable."""
    errs = []
    runs = plan.get("runs") if isinstance(plan, dict) else None
    if not isinstance(runs, list):
        return ['top-level "runs" must be a list']
    got = {r.get("seat") for r in runs if isinstance(r, dict)}
    for seat in seats:
        if seat not in got:
            errs.append(f"missing a run for {seat}")
    for r in runs:
        if not isinstance(r, dict):
            errs.append(f"run {r!r} is not an object"); continue
        seat = r.get("seat")
        if seat not in seats:
            errs.append(f"unknown seat {seat!r}; seats are {list(seats)}"); continue
        if r.get("model") != seats[seat]:
            errs.append(f"{seat} runs {seats[seat]}, not {r.get('model')!r}")
        for k, (lo, hi) in LIMITS.items():
            if k not in r:
                if k != "lr0":
                    errs.append(f"{seat}: missing {k}")
                continue
            v = r[k]
            if not isinstance(v, (int, float)) or not lo <= v <= hi:
                errs.append(f"{seat}: {k}={v!r} must be a number in [{lo}, {hi}]")
        if isinstance(r.get("imgsz"), int) and r["imgsz"] % 32:
            errs.append(f"{seat}: imgsz={r['imgsz']} is not a multiple of 32")
    return errs


def fallback(seats, history):
    """Deterministic schedule: epochs grow with each round, capped by size."""
    rounds = len({r.get("n_train") for r in history})
    base = {"n": 120, "s": 100, "m": 80, "l": 60, "x": 50}
    runs = []
    for seat, model in seats.items():
        size = model.removeprefix("yolo11").split("-")[0]
        runs.append({"seat": seat, "model": model, "epochs": min(base.get(size, 60), 30 + 15 * rounds),
                     "imgsz": 640, "batch": 16 if size in "nsm" else 8, "reason": "fallback schedule"})
    return {"runs": runs, "notes": "fallback: agent unavailable or no valid plan"}


def plan(request, tries=3):
    seats = request["seats"]
    history = summarize_history(request.get("history", []))
    if not llm.available():
        return fallback(seats, request.get("history", []))
    feedback = ""
    for attempt in range(tries):
        prompt = PROMPT.format(seats=json.dumps(seats, indent=1), dataset=json.dumps(request.get("dataset", {})),
                               history=json.dumps(history, indent=1) if history else "none yet",
                               feedback=feedback)
        reply = llm.chat(prompt)
        try:
            p = llm.extract_json(reply)
        except ValueError as e:
            feedback = f"\nYour previous answer was not valid JSON ({e}). Reply with the JSON object only."
            continue
        errs = validate(p, seats)
        if not errs:
            allowed = set(LIMITS) | OPTIONAL | {"seat", "model", "reason"}
            p["runs"] = [{k: v for k, v in r.items() if k in allowed} for r in p["runs"]]
            p["attempts"] = attempt + 1
            return p
        feedback = "\nYour previous plan was rejected:\n- " + "\n- ".join(errs) + "\nFix exactly these."
    out = fallback(seats, request.get("history", []))
    out["notes"] += f"; last rejection: {feedback.strip()}"
    return out


if __name__ == "__main__":
    print(json.dumps(plan(json.load(sys.stdin)), indent=2))
