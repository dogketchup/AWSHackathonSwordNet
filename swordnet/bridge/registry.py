"""Model registry on the cluster: the one place every teammate's client can reach.

Nobody can port-forward, and our laptops can't reach each other, but everyone with the workshop
credentials can `kubectl exec` into the team seats. So shared results live on a seat pod:

    seat-145:/workspace/swordnet-registry/
        index.json                      every published run: model, metrics, who/when, files
        runs/run-<id>/best.pt           weights
        runs/run-<id>/metrics.json      evaluator output (swordnet/metrics/evaluate.py)
        runs/run-<id>/report.md         the run report
        runs/run-<id>/results.png       training curves
        dataset/swordnet.tgz            FiftyOne export of the labeled data (images + labels)
        media/                          raw videos and images, as uploaded

Publishing happens on the laptop that runs the bridge (automatically after each run, or with
`bridge publish`). Any client reads it with `bridge models`, `bridge download`, `bridge pull-dataset`.
"""
import getpass
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from swordnet import config
from swordnet.bridge import kube
from swordnet.bridge.seats import SEATS
from swordnet.logs import log

REG = "/workspace/swordnet-registry"
SEAT = SEATS.get("registry", SEATS["agent"])
RUN_FILES = ("weights/best.pt", "metrics.json", "report.md", "results.png", "config.json")


def _publisher():
    return os.environ.get("SWORDNET_USER") or getpass.getuser()


def read_index():
    raw = kube.sh(SEAT, f"cat {REG}/index.json 2>/dev/null || echo '{{}}'").strip()
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as e:  # a half-read index must not be overwritten with less
        raise RuntimeError(f"registry index on {SEAT} is unreadable: {e}")


def _write_index(index):
    kube.send_bytes(SEAT, json.dumps(index, indent=1, sort_keys=True).encode(), f"{REG}/index.json")


def publish_run(run):
    """Copy one finished run's files to the registry and add it to index.json."""
    local = config.RUNS_DIR / f"run-{run['id']}"
    if not (local / "weights" / "best.pt").exists():
        raise FileNotFoundError(f"run {run['id']} has no weights to publish")
    key = f"run-{run['id']}"
    stage = Path(tempfile.mkdtemp(prefix="swordnet-pub-")) / key
    for f in RUN_FILES:
        if (local / f).exists():
            (stage / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local / f, stage / f)
    kube.sh(SEAT, f"rm -rf {REG}/runs/{key} && mkdir -p {REG}/runs")
    kube.push(SEAT, stage, f"{REG}/runs")
    shutil.rmtree(stage.parent, ignore_errors=True)
    m = run.get("metrics") or {}
    index = read_index()
    index[key] = {
        "run": run["id"], "model": run["model"], "seat": run.get("seat"), "config": run.get("config"),
        "n_train": run.get("n_train"), "n_val": run.get("n_val"),
        **{k: m.get(k) for k in ("best_mAP50", "best_mAP50_95", "final_precision", "final_recall",
                                 "fps", "ms_per_image", "best_epoch", "epochs_run")},
        "findings": m.get("findings", [])[:6],
        "weights": f"{REG}/runs/{key}/weights/best.pt",
        "weights_mb": round((local / "weights" / "best.pt").stat().st_size / 1e6, 1),
        "published_by": _publisher(), "published_at": time.time(),
    }
    _write_index(index)
    log("bridge", f"published {key} ({run['model']}) to the registry on {SEAT}", run_id=run["id"])
    return key


def publish_dataset():
    """Export the labeled data (FiftyOne format, images included) and raw media to the registry."""
    from swordnet.storage import store
    view = store.dataset().match(store.fo.ViewField("status") == "labeled")
    tmp = Path(tempfile.mkdtemp(prefix="swordnet-ds-"))
    view.export(export_dir=str(tmp / "swordnet"), dataset_type=store.fo.types.FiftyOneDataset, export_media=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(tmp / "swordnet", arcname="swordnet")
    shutil.rmtree(tmp, ignore_errors=True)
    kube.send_bytes(SEAT, buf.getvalue(), f"{REG}/dataset/swordnet.tgz")
    n_media = 0
    for src in (config.VIDEO_DIR, config.REPO / "images"):
        if src.exists() and any(src.iterdir()):
            kube.push(SEAT, src, f"{REG}/media", arcname=src.name)
            n_media += sum(1 for _ in src.iterdir())
    index = read_index()
    index["_dataset"] = {"images": len(view), "media_files": n_media, "bytes": buf.tell(),
                         "published_by": _publisher(), "published_at": time.time()}
    _write_index(index)
    log("bridge", f"published dataset ({len(view)} labeled images, {n_media} media files) to {SEAT}")
    return len(view)


def models():
    """Published runs, best first."""
    index = read_index()
    runs = [v | {"key": k} for k, v in index.items() if not k.startswith("_")]
    return sorted(runs, key=lambda r: -(r.get("best_mAP50_95") or 0)), index.get("_dataset")


def download(key=None, out="weights"):
    """Fetch a run's best.pt from the registry (the best one if key is None)."""
    runs, _ = models()
    if not runs:
        raise SystemExit("the registry is empty: no run has been published yet")
    run = next((r for r in runs if r["key"] == key), None) if key else runs[0]
    if run is None:
        raise SystemExit(f"{key} is not in the registry; `bridge models` lists what is")
    data = kube.fetch_bytes(SEAT, run["weights"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"swordnet-{run['key']}-{run['model']}.pt"
    dst.write_bytes(data)
    (out / "swordnet-best.pt").write_bytes(data)
    return run, dst


def pull_dataset(name="swordnet_shared"):
    """Import the published dataset into this machine's FiftyOne as `name` (never over your own)."""
    import fiftyone as fo
    data = kube.fetch_bytes(SEAT, f"{REG}/dataset/swordnet.tgz")
    dest = config.HOME / "shared"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset.from_dir(str(dest / "swordnet"), dataset_type=fo.types.FiftyOneDataset, name=name)
    ds.persistent = True
    return ds
