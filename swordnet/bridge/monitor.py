"""Live cluster status for the web UI's Cluster tab.

One `kubectl get pods` for pod health plus one exec per seat (in parallel) running a small
probe script, refreshed in a background thread so the web UI never waits on the cluster.
"""
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from swordnet import config
from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS, all_seats

REFRESH_S = 20

# Runs inside each seat pod with the image's python. Prints one JSON object.
PROBE = r'''
import glob, json, os, subprocess, time, urllib.request
D = "/workspace/swordnet-data"
out = {"load": [float(x) for x in open("/proc/loadavg").read().split()[:3]]}
try:
    out["cpus"] = int(open("/sys/fs/cgroup/cpu.max").read().split()[0]) // 100000
except Exception:
    out["cpus"] = os.cpu_count()
try:
    out["deps"] = "SETUP_OK" in open(D + "/setup.log").read()
except OSError:
    out["deps"] = False
procs = subprocess.run(["pgrep", "-af", r"swordnet\.(train|ingest|refine|metrics)"],
                       capture_output=True, text=True).stdout.splitlines()
out["procs"] = [p.split(" ", 1)[1][:200] for p in procs if " " in p]
runs = []
for st in sorted(glob.glob(D + "/runs/run-*/status"), key=os.path.getmtime, reverse=True)[:3]:
    rd = os.path.dirname(st)
    r = {"run": os.path.basename(rd), "status": open(st).read().strip(), "age_s": int(time.time() - os.path.getmtime(st))}
    try:
        cfg = json.load(open(rd + "/config.json"))
        r.update(model=cfg.get("model"), epochs=cfg.get("epochs"), imgsz=cfg.get("imgsz"))
    except Exception:
        pass
    try:
        r["epochs_done"] = max(0, sum(1 for _ in open(rd + "/results.csv")) - 1)
    except OSError:
        r["epochs_done"] = 0
    runs.append(r)
out["runs"] = runs
jobs = []
for j in sorted(glob.glob(D + "/jobs/*"), key=os.path.getmtime, reverse=True)[:2]:
    jobs.append({"job": os.path.basename(j), "done": os.path.exists(j + "/DONE"),
                 "age_s": int(time.time() - os.path.getmtime(j))})
out["jobs"] = jobs
try:
    urllib.request.urlopen("http://localhost:8000/health", timeout=3)
    out["llm"] = True
except Exception:
    out["llm"] = False
print(json.dumps(out))
'''


def _pods():
    p = subprocess.run(["kubectl", "get", "pods", *all_seats(), "-o", "json"], capture_output=True, text=True,
                       timeout=30)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[-300:]}
    pods = {}
    for item in json.loads(p.stdout)["items"]:
        cs = (item["status"].get("containerStatuses") or [{}])[0]
        pods[item["metadata"]["name"]] = {
            "phase": item["status"].get("phase"), "ready": cs.get("ready", False),
            "restarts": cs.get("restartCount", 0), "node": item["spec"].get("nodeName", ""),
            "started": item["status"].get("startTime")}
    return pods


def _probe(seat):
    try:
        return json.loads(kube.sh(seat, "python3 -", input=PROBE, timeout=40))
    except Exception as e:
        return {"error": str(e).splitlines()[-1][:300]}


def _role(seat):
    roles = []
    if seat == SEATS["agent"]:
        roles.append("agent (Qwen3 on Trainium)")
    if seat == SEATS["aux"]:
        roles.append("auto-label + SAM")
    model = SEATS["train"].get(seat)
    if model:
        roles.append("trainer: " + ("flex (Qwen picks)" if model == "flex" else model))
    return roles


def snapshot():
    """Gather everything once. Takes a few seconds; call from a background thread."""
    t0 = time.time()
    pods = _pods()
    with ThreadPoolExecutor(len(all_seats())) as pool:
        probes = dict(zip(all_seats(), pool.map(_probe, all_seats())))
    seats = []
    for seat in all_seats():
        seats.append({"seat": seat, "roles": _role(seat), "is_agent": seat == SEATS["agent"],
                      "pod": pods.get(seat) if isinstance(pods, dict) and "error" not in pods else None,
                      **probes[seat]})
    return {"ts": time.time(), "took_s": round(time.time() - t0, 1), "seats": seats,
            "pods_error": pods.get("error") if isinstance(pods, dict) else None,
            "gptoss": bool(config.LLMS["gptoss"]["base_url"]),
            "train_threads": SEATS.get("train_threads", {})}


class Monitor:
    """Keeps the latest snapshot fresh in a daemon thread; started on first use."""

    def __init__(self):
        self.latest, self.error = None, None
        self._lock = threading.Lock()
        self._thread = None

    def _loop(self):
        while True:
            try:
                snap = snapshot()
                with self._lock:
                    self.latest, self.error = snap, None
            except Exception as e:  # keep the last good snapshot, report the failure
                with self._lock:
                    self.error = str(e)[:300]
            time.sleep(REFRESH_S)

    def get(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="cluster-monitor")
            self._thread.start()
        with self._lock:
            return {"status": self.latest, "error": self.error, "refresh_s": REFRESH_S}


MONITOR = Monitor()
