"""The bridge: run from your laptop, drives the team's seat pods over `kubectl exec`.

    python -m swordnet.bridge.cli setup          # push code, install deps, start Qwen
    python -m swordnet.bridge.cli status         # what each seat is doing
    python -m swordnet.bridge.cli autolabel      # propose boxes for ingest images (toggle-gated)
    python -m swordnet.bridge.cli refine         # SAM-tighten boxes on ingest images into OBBs
    python -m swordnet.bridge.cli cycle          # one training round on every trainer seat
    python -m swordnet.bridge.cli loop           # rolling rounds as new labels arrive
    python -m swordnet.bridge.cli collect        # pull back runs that finished while the bridge was away
    python -m swordnet.bridge.cli promote        # Qwen picks the run to deploy; smoke-tests it
    python -m swordnet.bridge.cli pull-weights   # promoted (or best) weights -> ./weights/

  Shared with every teammate through the registry on the cluster (swordnet/bridge/registry.py):
    python -m swordnet.bridge.cli publish        # runs + labeled dataset + media -> registry
    python -m swordnet.bridge.cli models         # every published model, best first
    python -m swordnet.bridge.cli download       # best (or --run N) weights from the registry
    python -m swordnet.bridge.cli pull-dataset   # import the published dataset into your FiftyOne

Everything it does is logged (swordnet.logs), including the pods' own job logs, which it tails
while it waits. The web UI's Logs tab shows them.
"""
import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swordnet import config
from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS, all_seats
from swordnet.logs import log, log_lines
from swordnet.storage import store
from swordnet.train import export

REPO = config.REPO
TRAIN_DEPS = "ultralytics opencv-python-headless pillow pyyaml httpx"
AUX_DEPS = TRAIN_DEPS + " transformers"


# --- pod log tailing -----------------------------------------------------------

_offsets = {}


def tail_into_logs(seat, remote_log, **fields):
    """Copy lines added to a pod log since the last call into the central log."""
    key = (seat, remote_log)
    start = _offsets.get(key, 0)
    out = kube.sh(seat, f"test -f {remote_log} && tail -c +{start + 1} {remote_log} | head -c 1000000 || true",
                  check=False)
    if out:
        _offsets[key] = start + len(out.encode())
        # ultralytics redraws progress bars with \r; keep only each line's final state
        lines = [l.rsplit("\r", 1)[-1] for l in out.splitlines()]
        log_lines(seat, lines, **fields)


# --- setup ------------------------------------------------------------------

def push_code(seats=None):
    for seat in seats or all_seats():
        kube.sh(seat, f"rm -rf {kube.APP_DIR}/swordnet")
        kube.push(seat, REPO / "swordnet", kube.APP_DIR, exclude=("__pycache__", "/webui"))
        log("bridge", f"code pushed to {seat}:{kube.APP_DIR}", seat=seat)


def setup(a):
    push_code()
    for seat in dict.fromkeys([*SEATS["train"], SEATS["aux"]]):
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
        log("bridge", f"installing deps on {seat} (log {kube.DATA_DIR}/setup.log)", seat=seat)
    kube.spawn(SEATS["agent"], "cd /workspace && MAX_MODEL_LEN=8192 ./serve.sh", log="/tmp/serve-wrapper.log")
    log("bridge", f"starting Qwen3 on {SEATS['agent']}; run `status` until every seat is ready")


def status(a):
    agent = SEATS["agent"]
    up = kube.sh(agent, "curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8000/health", check=False)
    print(f"{agent:10} agent        Qwen3 {'ready' if up == '200' else 'NOT ready (' + up + ')'}")
    for seat in dict.fromkeys([*SEATS["train"], SEATS["aux"]]):
        role = SEATS["train"].get(seat, "")
        role = ("agent+" if seat == SEATS["agent"] else "") + ("aux+" if seat == SEATS["aux"] else "") + role
        out = kube.sh(seat, f"""
            grep -q SETUP_OK {kube.DATA_DIR}/setup.log 2>/dev/null && echo deps=ready || echo deps=installing
            for s in {kube.DATA_DIR}/runs/*/status; do [ -f "$s" ] && echo "$(basename $(dirname $s))=$(cat $s)"; done
        """, check=False).split()
        running = [x for x in out if x.endswith("=running")]
        print(f"{seat:10} {role:12} {out[0] if out else '?'}  {' '.join(running) or 'idle'}")
    print(f"gpt-oss    reports      {'configured' if config.LLMS['gptoss']['base_url'] else 'GPTOSS_BASE_URL not set in .env'}")
    q = store.queue_stats()
    print(f"queue      {q['ingest']} to label ({q['claimed']} claimed by {q['labelers'] or 'nobody'}), {q['labeled']} labeled")


# --- labeling helpers on the aux seat ----------------------------------------

def _stage(samples, with_labels=False):
    tmp = Path(tempfile.mkdtemp(prefix="swordnet-"))
    (tmp / "imgs").mkdir()
    labels = {}
    for s in samples:
        shutil.copy2(s.filepath, tmp / "imgs" / s.filename)
        if with_labels:
            labels[s.filename] = store.get_labels(s)
    (tmp / "imgs" / "labels.json").write_text(json.dumps(labels))
    return tmp


def _remote_job(seat, stage, module, args, wait_s):
    """Upload a staged dir, run module on it detached, tail its log, pull out.json."""
    name = module.rsplit(".", 1)[1]
    job = f"{kube.DATA_DIR}/jobs/{name}-{int(time.time())}"
    kube.sh(seat, f"mkdir -p {job}")
    kube.push(seat, stage / "imgs", job)
    kube.spawn(seat, f"cd {kube.APP_DIR} && {kube.VENV_PY} -m {module} {args.format(job=job)} && touch {job}/DONE",
               log=f"{job}/log.txt")
    log("bridge", f"{name} job started on {seat}: {job}", seat=seat)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(10)
        tail_into_logs(seat, f"{job}/log.txt", job=name)
        if kube.sh(seat, f"test -f {job}/DONE && echo y || true").strip() == "y":
            return json.loads(kube.read(seat, f"{job}/out.json"))
        if not kube.running(seat, job):
            tail_into_logs(seat, f"{job}/log.txt", job=name)
            log("bridge", f"{name} job on {seat} died; see its log lines", level="error", seat=seat)
            raise SystemExit(1)
    raise TimeoutError(f"{seat}: {module} still running after {wait_s}s; log {job}/log.txt")


def autolabel(a):
    F = store.fo.ViewField
    cols = store.autolabel_collections()
    view = store.dataset().match((F("status") == "ingest") & F("collection").is_in(cols))
    todo = [s for s in view.limit(a.limit) if not s.autolabels and not s.ground_truth]
    if not todo:
        print("nothing to auto-label (no collection has auto-label on, or every ingest image has labels)")
        return
    log("bridge", f"auto-labeling {len(todo)} images from {cols} on {SEATS['aux']} with {a.model}")
    res = _remote_job(SEATS["aux"], _stage(todo), "swordnet.ingest.autolabel",
                      f"--images {{job}}/imgs --out {{job}}/out.json --model {a.model} --threshold {a.threshold}", a.wait)
    by_name = {s.filename: s for s in todo}
    for name, dets in res.items():
        store.set_labels(by_name[name], dets, field="autolabels")
    log("bridge", f"auto-labeler proposed {sum(map(len, res.values()))} boxes on {len(res)} images")


def refine(a):
    view = store.dataset().match(store.fo.ViewField("status") == "ingest")
    todo = [s for s in view.limit(a.limit) if any("sam" not in l["source"] for l in store.get_labels(s))]
    if not todo:
        print("nothing to refine (no ingest image has unrefined boxes)")
        return
    log("bridge", f"SAM-refining {len(todo)} images on {SEATS['aux']}")
    res = _remote_job(SEATS["aux"], _stage(todo, with_labels=True), "swordnet.refine.sam_obb",
                      "--images {job}/imgs --labels {job}/imgs/labels.json --out {job}/out.json", a.wait)
    by_name = {s.filename: s for s in todo}
    for name, labels in res.items():
        store.set_labels(by_name[name], labels)
    kept = sum("refine_note" in l for ls in res.values() for l in ls)
    log("bridge", f"SAM refined {sum(map(len, res.values())) - kept} boxes; kept {kept} as drawn")


# --- training ---------------------------------------------------------------

def _slim(r):  # per-epoch history is large; the agents don't need it
    return r | {"metrics": {k: v for k, v in (r.get("metrics") or {}).items() if k != "history"}}


def cycle(a):
    snap = f"snap-{time.strftime('%Y%m%d-%H%M%S')}"
    local = config.DATASETS_DIR / snap
    manifest = export.export(local)
    if manifest["n_train"] == 0 or manifest["n_val"] == 0:
        log("bridge", f"not enough labeled data yet: {manifest['n_train']} train / {manifest['n_val']} val",
            level="warning")
        return False
    log("bridge", f"snapshot {snap}: {manifest['n_train']} train / {manifest['n_val']} val", snapshot=snap)

    # A seat still busy (a long l run from the last round, or an auto-label job on the aux seat)
    # sits this round out instead of being double-booked.
    busy = {seat for seat in SEATS["train"]
            if kube.running(seat, f"swordnet.train.train|{kube.DATA_DIR}/jobs/")}
    seats = {s: m for s, m in SEATS["train"].items() if s not in busy}
    if busy:
        log("bridge", f"skipping busy seats this round: {sorted(busy)}", level="warning")
    if not seats:
        log("bridge", "every trainer seat is busy; try again later", level="warning")
        return False
    history = [_slim(r) for r in store.list_runs(limit=30)]
    plan = json.loads(kube.py(SEATS["agent"], "swordnet.agent.plan",
                              stdin_json={"seats": seats, "dataset": manifest, "history": history}))
    log("agent", f"plan: {plan.get('notes', '')}", plan=plan)

    # Every exec call costs ~5 s on this cluster and a start needs several in a row, so seats are
    # started (and later polled) in parallel rather than one after another.
    jobs = []
    for r in plan["runs"]:
        seat, reason = r.pop("seat"), r.pop("reason", "")
        if a.max_epochs and r["epochs"] > a.max_epochs:
            reason += f" [epochs capped {r['epochs']} -> {a.max_epochs} by --max-epochs]"
            r["epochs"] = a.max_epochs
        run_id = store.create_run(r["model"], r["epochs"], r, seat=seat, dataset_dir=snap, reason=reason,
                                  n_train=manifest["n_train"], n_val=manifest["n_val"])
        jobs.append((run_id, seat, r, reason))

    def start(job):
        run_id, seat, r, reason = job
        try:
            kube.push(seat, local, f"{kube.DATA_DIR}/datasets")
            rdir = f"{kube.DATA_DIR}/runs/run-{run_id}"
            threads = SEATS.get("train_threads", {}).get(seat)
            env = f"SWORDNET_TORCH_THREADS={threads} " if threads else ""
            kube.spawn(seat, f"cd {kube.APP_DIR} && {env}{kube.VENV_PY} -m swordnet.train.train "
                             f"--dataset {kube.DATA_DIR}/datasets/{snap} --run-dir {rdir} "
                             f"--config '{json.dumps(r)}'", log=f"{rdir}.log")
        except Exception as e:
            store.update_run(run_id, status="failed", finished_at=time.time())
            log("bridge", f"run {run_id} could not start on {seat}: {e}", level="error", run_id=run_id, seat=seat)
            return None
        store.update_run(run_id, status="running", started_at=time.time())
        log("bridge", f"run {run_id} started on {seat}: {r} ({reason})", run_id=run_id, seat=seat)
        return run_id, seat

    with ThreadPoolExecutor(len(jobs)) as pool:
        started = [x for x in pool.map(start, jobs) if x]

    if a.no_wait:
        return True

    def check(job):
        try:
            return _check_run(*job)
        except Exception as e:  # e.g. the laptop's network dropped; try again next poll
            log("bridge", f"run {job[0]} on {job[1]}: check failed, will retry ({str(e)[-200:]})",
                level="warning", run_id=job[0])
            return False

    pending = list(started)
    while pending:
        time.sleep(a.poll)
        with ThreadPoolExecutor(len(pending)) as pool:
            finished = list(pool.map(check, pending))
        pending = [j for j, f in zip(pending, finished) if not f]
    return True


def _check_run(run_id, seat):
    """True once the run has ended on its pod and has been collected."""
    rdir = f"{kube.DATA_DIR}/runs/run-{run_id}"
    tail_into_logs(seat, f"{rdir}.log", run_id=run_id)
    st = kube.read(seat, f"{rdir}/status")
    if st in ("done", "failed") or not kube.running(seat, f"run-dir {rdir}"):
        finish_run(run_id, seat, st if st in ("done", "failed") else "failed")
        return True
    return False


def collect(a):
    """Collect runs the DB still shows as running: finished on the pod but never pulled back
    (the bridge was stopped, or the laptop lost its network)."""
    runs = [r for r in store.list_runs(limit=1000) if r["status"] == "running"]
    if not runs:
        print("nothing to collect")
        return
    for r in runs:
        try:
            done = _check_run(r["id"], r["seat"])
            print(f"run {r['id']} on {r['seat']}: {'collected' if done else 'still training'}")
        except Exception as e:
            log("bridge", f"run {r['id']} on {r['seat']}: collect failed ({str(e)[-200:]})", level="error",
                run_id=r["id"])


def finish_run(run_id, seat, st):
    rdir = f"{kube.DATA_DIR}/runs/run-{run_id}"
    tail_into_logs(seat, f"{rdir}.log", run_id=run_id)
    # last.pt duplicates best.pt in size and is never deployed; it stays on the pod
    kube.pull(seat, rdir, config.RUNS_DIR, exclude=("last.pt",))
    kube.pull(seat, f"{rdir}.log", config.RUNS_DIR)
    local = config.RUNS_DIR / f"run-{run_id}"
    local.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((local / "metrics.json").read_text()) if (local / "metrics.json").exists() else None
    if metrics:
        (config.METRICS_DIR / f"run-{run_id}.json").write_text(json.dumps(metrics, indent=2))
    best = local / "weights" / "best.pt"
    store.update_run(run_id, status=st, metrics=metrics, finished_at=time.time(),
                     weights=str(best) if best.exists() else None)
    log("bridge", f"run {run_id} {st} on {seat}" + (f": best mAP50-95 {metrics.get('best_mAP50_95')}" if metrics else ""),
        level="info" if st == "done" else "error", run_id=run_id, seat=seat)
    write_report(run_id)
    if st == "done" and best.exists():
        try:
            from swordnet.bridge import registry
            registry.publish_run(store.get_run(run_id))
        except Exception as e:  # sharing must never lose a collected run
            log("bridge", f"run {run_id}: publish to registry failed, retry with `bridge publish` ({e})",
                level="warning", run_id=run_id)


def write_report(run_id):
    run = _slim(store.get_run(run_id))
    previous = [_slim(r) for r in store.list_runs(limit=20) if r["id"] != run_id and r["status"] == "done"]
    req = {"run": run, "previous": previous}
    if config.LLMS["gptoss"]["base_url"]:
        from swordnet.agent import report  # gpt-oss is reachable from here; no need to go via a pod
        md = report.render(run, previous, which="gptoss")
    else:
        md = kube.py(SEATS["agent"], "swordnet.agent.report", stdin_json=req | {"llm": "qwen"})
    path = config.RUNS_DIR / f"run-{run_id}" / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    store.update_run(run_id, report_path=str(path))
    log("agent", f"report written for run {run_id}", run_id=run_id, path=str(path))


def loop(a):
    last = -1
    while True:
        n = store.queue_stats()["labeled"]
        if last < 0 or n - last >= a.min_new:
            log("bridge", f"{n} labeled images; starting a training round")
            if cycle(a):
                last = n
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {n} labeled, waiting for {a.min_new - (n - last)} more")
        time.sleep(a.every)


# --- deploy -----------------------------------------------------------------

def promote(a):
    runs = [r for r in store.list_runs(limit=1000) if r["status"] == "done" and r.get("weights")]
    if not runs:
        print("no finished runs with weights")
        return
    current = (store.coll("deploy").find_one({"_id": "current"}) or {}).get("run")
    cands = [{"id": r["id"], "model": r["model"], "epochs": r["epochs"], "n_train": r.get("n_train"),
              **{k: (r["metrics"] or {}).get(k) for k in ("best_mAP50", "best_mAP50_95", "fps", "ms_per_image",
                                                          "final_recall", "final_precision")},
              "smoke_failed": r.get("smoke", {}).get("pass") is False} for r in runs]
    choice = json.loads(kube.py(SEATS["agent"], "swordnet.agent.promote",
                                stdin_json={"runs": cands, "current": current, "target": a.target}))
    log("agent", f"promote: run {choice['run']} ({choice['reason']})", run_id=choice["run"])
    if choice["run"] is None:
        return
    run = store.get_run(choice["run"])
    seat, rdir = run["seat"], f"{kube.DATA_DIR}/runs/run-{run['id']}"
    out = f"{rdir}/smoke.json"
    kube.sh(seat, f"cd {kube.APP_DIR} && {kube.VENV_PY} -m swordnet.metrics.smoketest --weights {rdir}/weights/best.pt "
                  f"--images {kube.DATA_DIR}/datasets/{run['dataset_dir']}/images/val --out {out} > {rdir}/smoke.log 2>&1",
            check=False, timeout=1800)
    tail_into_logs(seat, f"{rdir}/smoke.log", run_id=run["id"], job="smoketest")
    smoke = json.loads(kube.read(seat, out) or '{"pass": false, "checks": {"ran": {"pass": false, "detail": "no output"}}}')
    store.update_run(run["id"], smoke=smoke)
    (config.METRICS_DIR / f"run-{run['id']}-smoke.json").write_text(json.dumps(smoke, indent=2))
    if smoke["pass"]:
        store.coll("deploy").replace_one(
            {"_id": "current"}, {"_id": "current", "run": run["id"], "reason": choice["reason"], "ts": time.time()},
            upsert=True)
        log("agent", f"run {run['id']} promoted: smoke test passed", run_id=run["id"])
    else:
        failed = {k: c["detail"] for k, c in smoke["checks"].items() if not c["pass"]}
        log("agent", f"run {run['id']} NOT promoted: smoke test failed {failed}", level="error", run_id=run["id"])


def pull_weights(a):
    if a.run:
        run = store.get_run(a.run)
    else:
        cur = store.coll("deploy").find_one({"_id": "current"})
        runs = [r for r in store.list_runs(limit=1000) if r["status"] == "done" and r.get("weights")]
        run = store.get_run(cur["run"]) if cur else max(
            runs, key=lambda r: (r["metrics"] or {}).get("best_mAP50_95") or 0, default=None)
    if not run or not run.get("weights") or not Path(run["weights"]).exists():
        print("no weights available (run `cycle` first)")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"swordnet-run{run['id']}-{run['model']}.pt"
    shutil.copy2(run["weights"], dst)
    shutil.copy2(run["weights"], out / "swordnet-best.pt")
    log("bridge", f"weights of run {run['id']} ({run['model']}) -> {dst}", run_id=run["id"])
    print(dst)


def publish(a):
    from swordnet.bridge import registry
    index = registry.read_index()
    for r in store.list_runs(limit=1000):
        if r["status"] == "done" and r.get("weights") and (a.all or f"run-{r['id']}" not in index):
            registry.publish_run(r)
    if not a.runs_only:
        registry.publish_dataset()
    models(a)


def models(a):
    from swordnet.bridge import registry
    runs, ds = registry.models()
    if not runs:
        print("registry is empty")
    else:
        print(f"{'model':<10} {'run':<8} {'imgsz':>5} {'train':>5} {'mAP50':>6} {'mAP50-95':>8} {'FPS':>5} {'MB':>5}  by")
        for r in runs:
            f = lambda k, d=3: "-" if r.get(k) is None else f"{r[k]:.{d}f}"
            print(f"{r['model'].removesuffix('-obb'):<10} {r['key']:<8} {(r.get('config') or {}).get('imgsz', '-'):>5} "
                  f"{r.get('n_train') or '-':>5} {f('best_mAP50'):>6} {f('best_mAP50_95'):>8} {f('fps', 1):>5} "
                  f"{r.get('weights_mb', '-'):>5}  {r.get('published_by', '?')}")
    if ds:
        print(f"dataset: {ds['images']} labeled images, {ds['media_files']} media files, "
              f"published by {ds['published_by']} {time.strftime('%Y-%m-%d %H:%M', time.localtime(ds['published_at']))}")


def download(a):
    from swordnet.bridge import registry
    run, dst = registry.download(f"run-{a.run}" if a.run else None, a.out)
    print(f"{run['key']} ({run['model']}, mAP50-95 {run.get('best_mAP50_95')}, {run.get('fps')} FPS) -> {dst}")


def pull_dataset(a):
    from swordnet.bridge import registry
    ds = registry.pull_dataset(a.name)
    print(f"imported {len(ds)} images as FiftyOne dataset '{ds.name}'")


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
        s.add_argument("--max-epochs", type=int, help="cap the agent's epochs (quick test rounds)")
        if name == "loop":
            s.add_argument("--min-new", type=int, default=50, help="new labeled images per round")
            s.add_argument("--every", type=int, default=300, help="seconds between checks")
    sub.add_parser("collect").set_defaults(fn=collect)
    s = sub.add_parser("report"); s.set_defaults(fn=lambda a: write_report(a.run))
    s.add_argument("run", type=int)
    s = sub.add_parser("promote"); s.set_defaults(fn=promote)
    s.add_argument("--target", default="a laptop CPU")
    s = sub.add_parser("publish"); s.set_defaults(fn=publish)
    s.add_argument("--all", action="store_true", help="republish runs already in the registry")
    s.add_argument("--runs-only", action="store_true", help="skip the dataset and media")
    sub.add_parser("models").set_defaults(fn=models)
    s = sub.add_parser("download"); s.set_defaults(fn=download)
    s.add_argument("--run", type=int, help="run number (default: best mAP50-95 in the registry)")
    s.add_argument("--out", default="weights")
    s = sub.add_parser("pull-dataset"); s.set_defaults(fn=pull_dataset)
    s.add_argument("--name", default="swordnet_shared", help="FiftyOne dataset name to import into")
    s = sub.add_parser("pull-weights"); s.set_defaults(fn=pull_weights)
    s.add_argument("--run", type=int)
    s.add_argument("--out", default="weights")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
