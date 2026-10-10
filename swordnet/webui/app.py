"""SwordNet web UI: Upload, Images, Annotate, Reports, Cluster and Logs tabs.

    uvicorn swordnet.webui.app:app --host 127.0.0.1 --port 8080

Each teammate runs their own copy, for themselves only. It holds no database: every read and
write goes to the team database on the cluster through storage/client.py (one `kubectl exec`
per operation), so everyone sees the same images, queue, labels, runs and logs. This laptop only
caches image files it has shown.

Labelers identify themselves with a name (the X-Labeler header, set by the page). The Annotate
tab pulls work from the queue (/api/queue/next), so two labelers never get the same image.
"""
import base64
import json
import os
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from swordnet import config, logs
from swordnet.ingest import frames, scraper
from swordnet.logs import log
from swordnet.storage import client

app = FastAPI(title="SwordNet")
STATIC = Path(__file__).parent / "static"
STATES = ("pending", "annotating", "labeled")

# Optional team password from .env (HTTP basic auth, any username). Unset = no password.
UI_PASSWORD = os.environ.get("SWORDNET_UI_PASSWORD", "")


@app.middleware("http")
async def team_password(request: Request, call_next):
    if UI_PASSWORD:
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(auth[6:]).decode().partition(":")
                ok = secrets.compare_digest(given.encode(), UI_PASSWORD.encode())
            except Exception:
                ok = False
        if not ok:
            return Response("SwordNet: team password required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="SwordNet"'})
    return await call_next(request)


@app.exception_handler(client.DBError)
def _db_error(request, exc):
    text = str(exc)
    status = 409 if "is labeling" in text else 404 if "no longer exists" in text else 503
    return JSONResponse({"detail": text}, status_code=status)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def ui_config():
    return {"classes": config.CLASSES, "database": client.MODE}


def _summary():
    return client.call("summary", ttl=5)


# --- collections & stats ---------------------------------------------------------

@app.get("/api/datasets")
def datasets():
    return _summary()["datasets"]


class Toggle(BaseModel):
    enabled: bool


@app.post("/api/datasets/{name}/autolabel")
def toggle_autolabel(name: str, body: Toggle, x_labeler: str = Header("anonymous")):
    client.call("set_autolabel", {"collection": name, "enabled": body.enabled})
    client.forget()
    log("webui", f"auto-label {'on' if body.enabled else 'off'} for {name}", user=x_labeler)
    return {"name": name, "autolabel": body.enabled}


@app.get("/api/queue")
def queue():
    return _summary()["queue"]


# --- ingest -----------------------------------------------------------------

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), dataset: str = Form("default"),
                 every: float = Form(0.5), x_labeler: str = Header("anonymous")):
    stage = client.stage_dir()
    items, paths, videos, skipped = [], {}, [], []
    try:
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext in config.IMAGE_EXTS:
                name = f"up{uuid.uuid4().hex[:12]}{ext}"
                dst = stage / name
                with dst.open("wb") as out:
                    shutil.copyfileobj(f.file, out)
                try:
                    Image.open(dst).verify()
                except Exception:
                    skipped.append(f.filename)
                    continue
                items.append({"file": name, "collection": dataset, "origin": f"upload:{f.filename}",
                              "ingested_at": time.time()})
                paths[name] = dst
            elif ext in config.VIDEO_EXTS:
                vid = stage / f"{Path(f.filename).stem}-{uuid.uuid4().hex[:6]}{ext}"
                with vid.open("wb") as out:
                    shutil.copyfileobj(f.file, out)
                got = frames.extract(vid, dataset, every_s=every, out_dir=stage / "frames", register=False)
                for path, origin in got:
                    items.append({"file": path.name, "collection": dataset, "origin": origin,
                                  "ingested_at": time.time()})
                    paths[path.name] = path
                videos.append({"video": f.filename, "frames": len(got)})
            else:
                skipped.append(f.filename)
        res = client.ingest(items, paths) if items else {"added": [], "duplicates": []}
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    dupes = [next((i["origin"].split(":", 1)[-1] for i in items if i["file"] == d), d) for d in res["duplicates"]]
    log("webui", f"upload to {dataset}: {len(res['added'])} images" + (f", {len(videos)} videos" if videos else "")
        + (f", {len(dupes)} duplicates skipped" if dupes else "") + (f", unreadable {skipped}" if skipped else ""),
        user=x_labeler, level="warning" if skipped else "info")
    return {"added": len(res["added"]), "videos": videos, "skipped": skipped, "duplicates": dupes}


class ScrapeReq(BaseModel):
    urls: list[str]
    dataset: str = "default"
    limit: int = 100


@app.post("/api/scrape")
def scrape(req: ScrapeReq, x_labeler: str = Header("anonymous")):
    stage = client.stage_dir()
    try:
        got = scraper.scrape(req.urls, req.dataset, req.limit, out_dir=stage, register=False)
        items = [{"file": p.name, "collection": req.dataset, "origin": origin, "ingested_at": time.time()}
                 for p, origin in got]
        res = client.ingest(items, {p.name: p for p, _ in got}) if items else {"added": []}
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    log("webui", f"scraped {len(res['added'])} images into {req.dataset} from {len(req.urls)} url(s)",
        user=x_labeler, urls=req.urls)
    return {"added": len(res["added"])}


# --- annotate ---------------------------------------------------------------
# Opening an image costs one exec: the answer carries the image's record, its boxes and its bytes.
# The page then asks for /labels and /file separately; both are served from what that call
# brought back, so they cost nothing extra.

_opened, _opened_lock = {}, threading.Lock()


def _remember(answer):
    sid = answer["image"]["id"]
    client.image_file(sid, answer)
    with _opened_lock:
        _opened[sid] = (time.time(), answer)
    return answer


def _open(sample_id):
    with _opened_lock:
        hit = _opened.pop(sample_id, None)
    if hit and time.time() - hit[0] < 120:
        return hit[1]
    return _remember(client.call("open", {"id": sample_id}))


@app.post("/api/queue/next")
def queue_next(skip: str = "", collection: str | None = None, x_labeler: str = Header("anonymous")):
    ans = client.call("next", {"user": x_labeler, "collection": collection,
                               "skip": [s for s in skip.split(",") if s]})
    client.forget()
    if not ans.get("image"):
        return {"image": None}
    return {"image": _remember(ans)["image"]}


@app.post("/api/images/{sample_id}/release")
def release(sample_id: str, who: str | None = None, x_labeler: str = Header("anonymous")):
    # sendBeacon (tab closing) cannot set headers, so the page passes ?who= instead.
    client.call("release", {"id": sample_id, "user": who if x_labeler == "anonymous" and who else x_labeler})
    client.forget()
    return {"released": sample_id}


@app.get("/api/images")
def images(status: str | None = None, dataset: str | None = None, state: str | None = None,
           limit: int = 500, newest_first: bool = False):
    if state and state not in STATES:
        raise HTTPException(422, f"state must be one of {STATES}")
    req = {k: v for k, v in {"status": status, "collection": dataset, "state": state, "limit": limit,
                             "newest_first": newest_first}.items() if v is not None}
    out = client.call("list_images", req, ttl=5)["images"]
    client.want_files([i["id"] for i in out])  # thumbnails arrive as one download, not one each
    return out


@app.get("/api/images/states")
def image_states():
    return _summary()["states"]


@app.get("/api/images/{sample_id}/file")
def image_file(sample_id: str):
    path = client.image_file(sample_id)
    if path is None or not Path(path).exists():
        raise HTTPException(404, "image file not found")
    return FileResponse(path)


@app.get("/api/images/{sample_id}/labels")
def get_labels(sample_id: str):
    ans = _open(sample_id)
    return {"image": ans["image"], "labels": ans["labels"]}


class Label(BaseModel):
    cls: int
    points: list[list[float]]
    source: str = "manual"
    score: float | None = None


class LabelSet(BaseModel):
    labels: list[Label]


def _check(labels):
    for lb in labels:
        if not 0 <= lb.cls < len(config.CLASSES) or len(lb.points) != 4:
            raise HTTPException(422, f"bad label {lb}")
    return [lb.model_dump() for lb in labels]


@app.put("/api/images/{sample_id}/labels")
def put_labels(sample_id: str, body: LabelSet, x_labeler: str = Header("anonymous")):
    labels = _check(body.labels)
    client.call("save", {"id": sample_id, "user": x_labeler, "labels": labels})
    return {"saved": len(labels)}


@app.post("/api/images/{sample_id}/processed")
def processed(sample_id: str, body: LabelSet | None = None, x_labeler: str = Header("anonymous")):
    """Finish an image. With a body, its boxes are saved in the same call."""
    labels = _check(body.labels) if body else _open(sample_id)["labels"]
    res = client.call("save", {"id": sample_id, "user": x_labeler, "labels": labels, "finish": True})
    client.forget()
    return res["image"]


@app.delete("/api/images/{sample_id}")
def delete_image(sample_id: str, x_labeler: str = Header("anonymous")):
    client.call("delete", {"id": sample_id, "user": x_labeler})
    client.forget()
    return {"deleted": sample_id}


# --- reports & metrics (run records in the database; files in the registry) --------

def _deployed():
    return (client.call("deploy_get", ttl=10).get("deploy") or {}).get("run")


@app.get("/api/runs")
def runs():
    deployed = _deployed()
    out = client.call("list_runs", {}, ttl=5)["runs"]
    for r in out:
        r["deployed"] = r["id"] == deployed
        if r.get("metrics"):
            r["metrics"] = {k: v for k, v in r["metrics"].items() if k not in ("history", "dataset")}
    return out


def _registry_text(path):
    from swordnet.bridge import kube, registry
    return kube.sh(registry.SEAT, f"cat {registry.REG}/{path} 2>/dev/null || true")


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: int):
    local = config.RUNS_DIR / f"run-{run_id}" / "report.md"
    text = local.read_text() if local.exists() else _registry_text(f"runs/run-{run_id}/report.md")
    if not text.strip():
        raise HTTPException(404, "no report yet")
    return text


@app.get("/api/runs/{run_id}/plot/{name}")
def run_plot(run_id: int, name: str):
    name = Path(name).name
    if not name.endswith(".png"):
        raise HTTPException(404)
    local = config.RUNS_DIR / f"run-{run_id}" / name
    if not local.exists():
        from swordnet.bridge import kube, registry
        try:
            data = kube.fetch_bytes(registry.SEAT, f"{registry.REG}/runs/run-{run_id}/{name}")
        except Exception:
            raise HTTPException(404)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return FileResponse(local)


@app.get("/api/metrics")
def metrics_files():
    """Evaluator output per run: from the registry on the cluster (or the local metrics/ folder
    when the database is local, as in tests)."""
    if client.MODE == "local":
        config.ensure_dirs()
        return sorted(p.name for p in config.METRICS_DIR.glob("*.json"))
    from swordnet.bridge import registry
    runs, _ = registry.models()
    return sorted(f"{r['key']}.json" for r in runs)


@app.get("/api/metrics/{name}")
def metrics_file(name: str):
    key = Path(name).stem
    if client.MODE == "local":
        path = config.METRICS_DIR / f"{key}.json"
        if not path.exists():
            raise HTTPException(404)
        return json.loads(path.read_text())
    if not key.startswith("run-") or not key[4:].isdigit():
        raise HTTPException(404)
    text = _registry_text(f"runs/{key}/metrics.json")
    if not text.strip():
        raise HTTPException(404)
    return json.loads(text)


# --- cluster ----------------------------------------------------------------

@app.get("/api/cluster")
def cluster():
    """Seat pods (refreshed in the background every 20 s), plus pipeline state from the database."""
    from swordnet.bridge.monitor import MONITOR
    runs = client.call("list_runs", {"limit": 50}, ttl=10)["runs"]
    recent = [e for e in logs.query(limit=300) if e["source"] in ("bridge", "agent")][-12:]
    return {**MONITOR.get(), "queue": _summary()["queue"], "deployed": _deployed(),
            "active_runs": [{"id": r["id"], "model": r["model"], "seat": r.get("seat"), "epochs": r["epochs"],
                             "started_at": r.get("started_at")} for r in runs if r["status"] == "running"],
            "done_runs": sum(r["status"] == "done" for r in runs),
            "failed_runs": sum(r["status"] == "failed" for r in runs),
            "events": recent}


# --- logs -------------------------------------------------------------------

@app.get("/api/logs")
def get_logs(source: str | None = None, run_id: int | None = None, level: str | None = None,
             after: float | None = None, q: str | None = None, limit: int = 500):
    if level and level not in logs.LEVELS:
        raise HTTPException(422, f"level must be one of {logs.LEVELS}")
    return {"sources": logs.sources(),
            "entries": logs.query(source=source, run_id=run_id, level=level, after=after, text=q, limit=limit)}


# --- best model (agent/pick.py): ranks logged runs; no model is re-run ---------

@app.get("/api/best")
def best_current(target: float = 0.5, ms: float = 150, mb: float = 20, noise: float = 0.02):
    from swordnet.agent import pick
    return pick.current(target, ms, mb, noise)


@app.post("/api/best/analyze")
def best_analyze(target: float = 0.5, ms: float = 150, mb: float = 20, noise: float = 0.02):
    from swordnet.agent import pick
    return pick.analyze(target, ms, mb, noise)


@app.post("/api/best/explore")
def best_explore(target: float = 0.5, ms: float = 150, mb: float = 20, max_epochs: int = 60):
    """One round of gpt-oss hypotheses on the idle trainer seats (agent/explore.py), in the background."""
    from swordnet.agent import explore
    threading.Thread(target=explore.launch, args=(target, ms, mb, max_epochs), daemon=True).start()
    return {"started": True}


@app.get("/api/best/explore")
def best_explore_state():
    from swordnet.agent import explore
    return explore.last() or {}


@app.get("/api/best/weights/{key}")
def best_weights(key: str):
    from swordnet.agent import pick
    path = pick.download(key)
    if not path:
        raise HTTPException(404, f"{key}: not in the registry and not on this laptop")
    return FileResponse(path, filename=Path(path).name)


@app.get("/api/live")
def live_queue():
    """Labeling queue and pod logs, read live from the team database on the cluster (read-only)."""
    from swordnet.bridge import livequeue
    try:
        return livequeue.live()
    except Exception as e:
        raise HTTPException(502, f"could not read the cluster: {e}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
