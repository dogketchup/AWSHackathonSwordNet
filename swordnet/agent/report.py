"""Report generator: one Markdown report per training run.

Runs on the agent seat, called by the bridge after a run finishes:
    kubectl exec -i seat-145 -- python -m swordnet.agent.report < run.json > report.md

The numbers and findings sections are generated directly from metrics/evaluate.py's output,
so they are always right. An agent writes only the analysis section, and it is given the
findings rather than asked to discover them. gpt-oss-20b writes it when GPTOSS_BASE_URL is set
(the bridge then renders the report on the laptop); otherwise the bridge runs this module on
the agent seat and Qwen3 writes it. If neither answers, the report is still complete, just
without that section.
"""
import json
import sys

from swordnet.agent import llm

ANALYSIS_PROMPT = """You are reviewing a YOLO-OBB training run for SwordNet (detects swords).
Run {run_no}: model {model}, config {config}.
Dataset: {dataset}
Metrics: {metrics}
Automatic findings: {findings}
Earlier runs for comparison: {previous}

Write 5-10 short bullet points for the people labeling data and planning runs:
- Is this run better than earlier ones, and is the difference bigger than val-set noise?
- Overfitting or underfitting, and what to change next run.
- What the humans may have missed: label quality, val leakage, class balance, image size vs
  thin objects like blades, speed vs accuracy for deployment, whether more data or more epochs
  is the better next investment.
Rules: only call a run overfitting, underfitting or converged if the automatic findings say so;
if there are no such findings, say the curves are inconclusive. With fewer than 10 epochs, say it
is too short to judge convergence. Thin objects like blades get *more* pixels with a larger imgsz.
Be concrete and cite numbers. No preamble."""


def _fmt(v):
    return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def render(run, previous=(), which="qwen"):
    m = run.get("metrics") or {}
    cfg = run.get("config") or {}
    ds = m.get("dataset") or {}
    lines = [
        f"# Run {run['id']}: {run['model']}",
        "",
        f"Status **{run.get('status')}** on `{run.get('seat', '?')}`, "
        f"{ds.get('n_train', run.get('n_train', '?'))} train / {ds.get('n_val', run.get('n_val', '?'))} val images.",
        "",
        "## Config",
        "",
        "| key | value |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in cfg.items()],
        "",
        "## Results",
        "",
        "| metric | value |", "|---|---|",
        *[f"| {k} | {_fmt(m.get(k))} |" for k in (
            "epochs_run", "best_epoch", "best_mAP50", "best_mAP50_95", "final_mAP50", "final_mAP50_95",
            "final_precision", "final_recall", "train_loss_final", "val_loss_final", "gap_ratio",
            "ms_per_image", "fps")],
        "",
        "## Findings",
        "",
        *([f"- {f}" for f in m.get("findings", [])] or ["- none flagged"]),
        "",
        "## Dataset",
        "",
        f"- objects: `{json.dumps(ds.get('objects', {}))}`",
        f"- label sources: `{json.dumps(ds.get('label_sources', {}))}`",
    ]
    if previous:
        lines += ["", "## Compared with earlier runs", "",
                  "| run | model | epochs | train imgs | best mAP50-95 | fps |", "|---|---|---|---|---|---|"]
        for p in previous:
            pm = p.get("metrics") or {}
            lines.append(f"| {p['id']} | {p['model']} | {p['epochs']} | {_fmt(p.get('n_train'))} | "
                         f"{_fmt(pm.get('best_mAP50_95'))} | {_fmt(pm.get('fps'))} |")

    analysis = None
    if m and llm.available(which):
        try:
            analysis = llm.chat(ANALYSIS_PROMPT.format(
                run_no=run["id"], model=run["model"], config=json.dumps(cfg), dataset=json.dumps(ds),
                metrics=json.dumps({k: v for k, v in m.items() if k not in ("history", "dataset", "findings")}),
                findings=json.dumps(m.get("findings", [])),
                previous=json.dumps([{"id": p["id"], "model": p["model"], "n_train": p.get("n_train"),
                                      "best_mAP50_95": (p.get("metrics") or {}).get("best_mAP50_95")}
                                     for p in previous])), which=which, max_tokens=1500)
        except Exception as e:
            analysis = f"_Agent analysis failed: {e}_"
    who = {"qwen": "Qwen3-8B on our Trainium chip", "gptoss": "gpt-oss-20b on Trainium"}[which]
    lines += ["", f"## Agent analysis ({who})", "",
              analysis or "_Agent unavailable; numbers and findings above are complete._", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(render(req["run"], req.get("previous", []), req.get("llm", "qwen")))
