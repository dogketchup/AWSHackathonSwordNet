"""The bridge: run from your laptop, drives the team's seat pods over `kubectl exec`.

    python -m swordnet.bridge.cli setup          # push code, install deps, start Qwen
    python -m swordnet.bridge.cli status         # what each seat is doing
    python -m swordnet.bridge.cli autolabel      # propose boxes for ingest images (toggle-gated)
    python -m swordnet.bridge.cli refine         # SAM-tighten boxes on ingest images into OBBs
    python -m swordnet.bridge.cli cycle          # one training round on every trainer seat
    python -m swordnet.bridge.cli loop           # rolling rounds as new labels arrive
    python -m swordnet.bridge.cli pull-weights   # best weights -> ./weights/
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from swordnet import config
from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS, all_seats
from swordnet.storage import db
from swordnet.train import export

REPO = Path(__file__).resolve().parents[2]
TRAIN_DEPS = "ultralytics opencv-python-headless pillow pyyaml httpx"
AUX_DEPS = TRAIN_DEPS + " transformers"


# --- setup ------------------------------------------------------------------

def push_code(seats=None):
    for seat in seats or all_seats():
        kube.sh(seat, f"rm -rf {kube.APP_DIR}/swordnet")
        kube.push(seat, REPO / "swordnet", kube.APP_DIR, exclude=("__pycache__", "/webui/static"))
        print(f"{seat}: code pushed to {kube.APP_DIR}")


def setup(a):
    push_code()
    for seat in [*SEATS["train"], SEATS["aux"]]:
        deps = AUX_DEPS if seat == SEATS["aux"] else TRAIN_DEPS
        # CPU torch from the PyTorch index: small, and leaves the image's vLLM torch untouched.
        kube.spawn(seat, f"""
            set -e
            test -x {kube.VENV_PY} || python3 -m venv {kube.VENV}
            {kube.VENV_PY} -m pip install -q --upgrade pip
            {kube.VENV_PY} -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
            {kube.VENV_PY} -m pip install -q {deps}
            echo SETUP_OK
        """, log=f"{kube.DATA_DIR}/setup.log")
        print(f"{seat}: installing deps in the background (log {kube.DATA_DIR}/setup.log)")
    kube.spawn(SEATS["agent"], "cd /workspace && MAX_MODEL_LEN=8192 ./serve.sh", log="/tmp/serve-wrapper.log")
    print(f"{SEATS['agent']}: starting Qwen3 (./serve.sh). Run `status` until every seat says ready.")


def status(a):
    agent = SEATS["agent"]
    up = kube.sh(agent, "curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8000/health", check=False)
    print(f"{agent:10} agent  Qwen3 {'ready' if up == '200' else 'NOT ready (' + up + ')'}")
    for seat in [SEATS["aux"], *SEATS["train"]]:
        role = SEATS["train"].get(seat, "aux")
        out = kube.sh(seat, f"""
            grep -q SETUP_OK {kube.DATA_DIR}/setup.log 2>/dev/null && echo deps=ready || echo deps=installing
            for s in {kube.DATA_DIR}/runs/*/status; do [ -f "$s" ] && echo "$(basename $(dirname $s))=$(cat $s)"; done
        """, check=False).split()
        running = [x for x in out if x.endswith("=running")]
        print(f"{seat:10} {role:12} {out[0] if out else '?'}  {' '.join(running) or 'idle'}")


# --- labeling helpers on the aux seat ----------------------------------------

def _stage_images(images, with_labels=False):
    """Copy images (and their labels) into a temp dir for upload."""
    tmp = Path(tempfile.mkdtemp(prefix="swordnet-"))
    (tmp / "imgs").mkdir()
    labels = {}
    with db.session() as con:
        for img in images:
            shutil.copy2(db.image_path(img), tmp / "imgs" / img["filename"])
            if with_labels:
                labels[img["filename"]] = [{"cls": l["cls"], "points": l["points"], "source": l["source"]}
                                           for l in db.get_labels(con, img["id"])]
    (tmp / "imgs" / "labels.json").write_text(json.dumps(labels))
    return tmp


def _remote_job(seat, local_stage, module, args, result_name, wait_s):
    """Upload a staged dir, run module on it detached, wait, pull the JSON result."""
    job = f"{kube.DATA_DIR}/jobs/{module.rsplit('.', 1)[1]}-{int(time.time())}"
    kube.sh(seat, f"mkdir -p {job}")
    kube.push(seat, local_stage / "imgs", job)
    kube.spawn(seat, f"cd {kube.APP_DIR} && {kube.VENV_PY} -m {module} {args.format(job=job)} && touch {job}/DONE",
               log=f"{job}/log.txt")
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if kube.sh(seat, f"test -f {job}/DONE && echo y || true").strip() == "y":
            return json.loads(kube.read(seat, f"{job}/{result_name}"))
        if kube.sh(seat, f"pgrep -f {job} >/dev/null && echo y || true").strip() != "y":
            raise RuntimeError(f"{seat}: job died:\n" + kube.read(seat, f"{job}/log.txt")[-2000:])
        time.sleep(10)
    raise TimeoutError(f"{seat}: {module} still running after {wait_s}s; log {job}/log.txt")


def autolabel(a):
    with db.session() as con:
        enabled = [d["name"] for d in db.list_datasets(con) if d["autolabel"]]
        todo = [i for d in enabled for i in db.list_images(con, status="ingest", dataset=d, limit=a.limit)
                if not db.get_labels(con, i["id"])]
    if not todo:
        print("nothing to auto-label (toggle is off, or every ingest image already has labels)")
        return
    print(f"auto-labeling {len(todo)} images from {enabled} on {SEATS['aux']} with {a.model}")
    stage = _stage_images(todo)
    res = _remote_job(SEATS["aux"], stage, "swordnet.ingest.autolabel",
                      f"--images {{job}}/imgs --out {{job}}/out.json --model {a.model} --threshold {a.threshold}",
                      "out.json", a.wait)
    by_name = {i["filename"]: i for i in todo}
    with db.session() as con:
        for name, dets in res.items():
            for d in dets:
                db.add_label(con, by_name[name]["id"], d["cls"], d["points"], d["source"], d["score"])
    print(f"added {sum(map(len, res.values()))} proposed boxes")


def refine(a):
    with db.session() as con:
        todo = [i for i in db.list_images(con, status="ingest", limit=a.limit)
                if any("sam" not in l["source"] for l in db.get_labels(con, i["id"]))]
    if not todo:
        print("nothing to refine")
        return
    print(f"SAM-refining {len(todo)} images on {SEATS['aux']}")
    stage = _stage_images(todo, with_labels=True)
    res = _remote_job(SEATS["aux"], stage, "swordnet.refine.sam_obb",
                      "--images {job}/imgs --labels {job}/imgs/labels.json --out {job}/out.json",
                      "out.json", a.wait)
    by_name = {i["filename"]: i for i in todo}
    with db.session() as con:
        for name, labels in res.items():
            db.replace_labels(con, by_name[name]["id"], labels)
    notes = [l["refine_note"] for ls in res.values() for l in ls if "refine_note" in l]
    print(f"refined {sum(map(len, res.values())) - len(notes)} boxes; kept {len(notes)} as drawn")


# --- training ---------------------------------------------------------------

def cycle(a):
    snap = f"snap-{time.strftime('%Y%m%d-%H%M%S')}"
    local = config.DATASETS_DIR / snap
    manifest = export.export(local)
    if manifest["n_train"] == 0 or manifest["n_val"] == 0:
        print(f"not enough labeled data yet: {manifest['n_train']} train / {manifest['n_val']} val")
        return False
    print(f"snapshot {snap}: {manifest['n_train']} train / {manifest['n_val']} val")

    with db.session() as con:
        history = db.list_runs(con, limit=30)
    plan = json.loads(kube.py(SEATS["agent"], "swordnet.agent.plan", stdin_json={
        "seats": SEATS["train"], "dataset": manifest, "history": history}))
    print(f"plan ({plan.get('notes', '')}):")

    started = []
    for r in plan["runs"]:
        seat = r.pop("seat")
        reason = r.pop("reason", "")
        with db.session() as con:
            run_id = db.create_run(con, r["model"], r["epochs"], r)
            db.update_run(con, run_id, dataset_dir=snap, n_train=manifest["n_train"], n_val=manifest["n_val"],
                          status="running", started_at=time.time())
        kube.push(seat, local, f"{kube.DATA_DIR}/datasets")
        rdir = f"{kube.DATA_DIR}/runs/run-{run_id}"
        kube.spawn(seat, f"cd {kube.APP_DIR} && {kube.VENV_PY} -m swordnet.train.train "
                         f"--dataset {kube.DATA_DIR}/datasets/{snap} --run-dir {rdir} "
                         f"--config '{json.dumps(r)}'", log=f"{rdir}.log")
        print(f"  run {run_id} on {seat}: {r} -- {reason}")
        started.append((run_id, seat))

    if a.no_wait:
        return True
    pending = dict(started)
    while pending:
        time.sleep(a.poll)
        for run_id, seat in list(pending.items()):
            st = kube.read(seat, f"{kube.DATA_DIR}/runs/run-{run_id}/status")
            if st in ("done", "failed") or (not st and "Traceback" in kube.read(seat, f"{kube.DATA_DIR}/runs/run-{run_id}.log")):
                finish_run(run_id, seat, st or "failed")
                del pending[run_id]
    return True


def finish_run(run_id, seat, st):
    rdir = f"{kube.DATA_DIR}/runs/run-{run_id}"
    kube.pull(seat, rdir, config.RUNS_DIR)
    kube.pull(seat, f"{rdir}.log", config.RUNS_DIR)
    local = config.RUNS_DIR / f"run-{run_id}"
    local.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((local / "metrics.json").read_text()) if (local / "metrics.json").exists() else None
    best = local / "weights" / "best.pt"
    with db.session() as con:
        db.update_run(con, run_id, status=st, metrics=metrics, finished_at=time.time(),
                      weights=str(best) if best.exists() else None)
        run = db.get_run(con, run_id)
        previous = [r for r in db.list_runs(con, limit=20) if r["id"] != run_id and r["status"] == "done"]

    def slim(r):  # per-epoch history is large and the report doesn't need it
        return r | {"metrics": {k: v for k, v in (r["metrics"] or {}).items() if k != "history"}}

    report = kube.py(SEATS["agent"], "swordnet.agent.report",
                     stdin_json={"run": slim(run) | {"seat": seat}, "previous": [slim(p) for p in previous]})
    (local / "report.md").write_text(report)
    with db.session() as con:
        db.update_run(con, run_id, report_path=str(local / "report.md"))
    print(f"  run {run_id} {st}; report {local / 'report.md'}")


def loop(a):
    last = -1
    while True:
        with db.session() as con:
            n = con.execute("SELECT COUNT(*) FROM images WHERE status='labeled'").fetchone()[0]
        if n - max(last, 0) >= a.min_new or last < 0:
            print(f"[{time.strftime('%H:%M:%S')}] {n} labeled images; starting a round")
            if cycle(a):
                last = n
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {n} labeled, need {a.min_new} new since last round ({last})")
        time.sleep(a.every)


def pull_weights(a):
    with db.session() as con:
        runs = [r for r in db.list_runs(con, limit=1000) if r["status"] == "done" and r["metrics"]]
    if a.run:
        runs = [r for r in runs if r["id"] == a.run]
    if not runs:
        print("no finished runs")
        return
    best = max(runs, key=lambda r: r["metrics"].get("best_mAP50_95") or 0)
    src = config.RUNS_DIR / f"run-{best['id']}" / "weights" / "best.pt"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"swordnet-run{best['id']}-{best['model']}.pt"
    shutil.copy2(src, dst)
    shutil.copy2(src, out / "swordnet-best.pt")
    print(f"run {best['id']} ({best['model']}, mAP50-95 {best['metrics'].get('best_mAP50_95')}) -> {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=setup)
    sub.add_parser("push-code").set_defaults(fn=lambda a: push_code())
    sub.add_parser("status").set_defaults(fn=status)
    s = sub.add_parser("autolabel"); s.set_defaults(fn=autolabel)
    s.add_argument("--model", default="owlv2", choices=["owlv2", "groundingdino"])
    s.add_argument("--threshold", type=float, default=0.25)
    s.add_argument("--limit", type=int, default=500)
    s.add_argument("--wait", type=int, default=3600)
    s = sub.add_parser("refine"); s.set_defaults(fn=refine)
    s.add_argument("--limit", type=int, default=500)
    s.add_argument("--wait", type=int, default=3600)
    for name, fn in (("cycle", cycle), ("loop", loop)):
        s = sub.add_parser(name); s.set_defaults(fn=fn)
        s.add_argument("--poll", type=int, default=60, help="seconds between status checks")
        s.add_argument("--no-wait", action="store_true", help="start runs and return")
        if name == "loop":
            s.add_argument("--min-new", type=int, default=50, help="new labeled images per round")
            s.add_argument("--every", type=int, default=300, help="seconds between checks")
    s = sub.add_parser("pull-weights"); s.set_defaults(fn=pull_weights)
    s.add_argument("--run", type=int)
    s.add_argument("--out", default="weights")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
