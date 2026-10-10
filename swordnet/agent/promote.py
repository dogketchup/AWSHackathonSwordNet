"""Promoter: Qwen3 picks which finished run to deploy. Runs on the agent seat via the bridge:
    kubectl exec -i seat-145 -- python -m swordnet.agent.promote < candidates.json

candidates.json: {"runs": [run summaries], "current": run_id | null, "target": "laptop CPU" }
output:          {"run": id | null, "reason": "..."}

Same loop as the planner: the choice is checked (must be a finished run with weights, must
not be worse than the current deployment on mAP50-95 by more than val noise unless the reason
says why) and rejected choices go back with the reason. Fallback: highest best_mAP50_95.
"""
import json
import sys

from swordnet.agent import llm

PROMPT = """You choose which trained SwordNet detector (oriented boxes around swords)
to deploy on {target}. Candidates (finished runs):
{runs}
Currently deployed run: {current}

Weigh accuracy (best_mAP50_95) against speed (fps): a model too slow for live video on the
target is not useful. Differences under about 0.02 mAP on a small val set are noise; prefer the
faster model then. Smoke-test failures disqualify a run.

Answer with JSON only: {{"run": <id>, "reason": "one or two sentences"}}
{feedback}"""


def fallback(runs):
    ok = [r for r in runs if not r.get("smoke_failed")]
    best = max(ok, key=lambda r: r.get("best_mAP50_95") or 0, default=None)
    return {"run": best["id"] if best else None, "reason": "fallback: highest mAP50-95", "agent": False}


def validate(choice, runs, current):
    by_id = {r["id"]: r for r in runs}
    rid = choice.get("run") if isinstance(choice, dict) else None
    if rid not in by_id:
        return [f"run must be one of {sorted(by_id)}, got {rid!r}"]
    if by_id[rid].get("smoke_failed"):
        return [f"run {rid} failed its smoke test"]
    if current in by_id and rid != current:
        drop = (by_id[current].get("best_mAP50_95") or 0) - (by_id[rid].get("best_mAP50_95") or 0)
        if drop > 0.05 and "speed" not in choice.get("reason", "").lower() and "fps" not in choice.get("reason", "").lower():
            return [f"run {rid} is {drop:.3f} mAP worse than the deployed run {current}; "
                    "only switch for a stated speed reason"]
    return []


def promote(req, tries=3):
    runs, current = req["runs"], req.get("current")
    if not runs:
        return {"run": None, "reason": "no finished runs"}
    if not llm.available("qwen"):
        return fallback(runs)
    feedback = ""
    for _ in range(tries):
        reply = llm.chat(PROMPT.format(target=req.get("target", "a laptop CPU"), runs=json.dumps(runs, indent=1),
                                       current=current, feedback=feedback), which="qwen", max_tokens=400)
        try:
            choice = llm.extract_json(reply)
        except ValueError:
            feedback = "\nYour previous answer was not valid JSON. Reply with the JSON object only."
            continue
        errs = validate(choice, runs, current)
        if not errs:
            return {"run": choice["run"], "reason": choice.get("reason", ""), "agent": True}
        feedback = "\nRejected: " + "; ".join(errs)
    return fallback(runs) | {"reason": f"fallback after rejections ({feedback.strip()})"}


if __name__ == "__main__":
    print(json.dumps(promote(json.load(sys.stdin))))
