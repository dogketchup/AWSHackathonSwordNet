"""SwordNet web UI: Upload, Annotate, Reports and Logs tabs over the FiftyOne store.

    uvicorn swordnet.webui.app:app --host 0.0.0.0 --port 8080

Runs on the laptop (port-forward to the pods is blocked). Teammates on the same network use
http://<laptop-ip>:8080; scraping agents POST images to /api/upload.

Labelers identify themselves with a name (the X-Labeler header, set by the page). The Annotate
tab pulls work from the queue (/api/queue/next), so two labelers never get the same image.
"""
import base64
import json
import os
import secrets
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi import Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from swordnet import config, logs
from swordnet.ingest import frames, scraper
from swordnet.logs import log
from swordnet.storage import store

app = FastAPI(title="SwordNet")
STATIC = Path(__file__).parent / "static"

# One shared team password, from .env. When it is set, every page and API call needs it
# (browsers ask once; scripts use `curl -u team:<password>`). Unset = open, as before.
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


def _sample_or_404(sample_id):
    s = store.get_sample(sample_id)
    if s is None:
        raise HTTPException(404, "no such image")
    return s


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def ui_config():
    return {"classes": config.CLASSES}


# --- collections & stats ---------------------------------------------------------

@app.get("/api/datasets")
def datasets():
    return store.list_collections()


class Toggle(BaseModel):
    enabled: bool


@app.post("/api/datasets/{name}/autolabel")
def toggle_autolabel(name: str, body: Toggle, x_labeler: str = Header("anonymous")):
    store.set_autolabel(name, body.enabled)
    log("webui", f"auto-label {'on' if body.enabled else 'off'} for {name}", user=x_labeler)
    return {"name": name, "autolabel": body.enabled}


@app.get("/api/queue")
def queue():
    return store.queue_stats()


# --- ingest -----------------------------------------------------------------

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), dataset: str = Form("default"),
                 every: float = Form(0.5), x_labeler: str = Header("anonymous")):
    config.ensure_dirs()
    added, videos, skipped, dupes = [], [], [], []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext in config.IMAGE_EXTS:
            dst = config.INGEST_DIR / f"up{uuid.uuid4().hex[:12]}{ext}"
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                Image.open(dst).verify()
            except Exception:
                dst.unlink()
                skipped.append(f.filename)
                continue
            sid = store.add_image(dst, dataset, f"upload:{f.filename}")
            if sid is None:
                dupes.append(f.filename)
            else:
                added.append(sid)
        elif ext in config.VIDEO_EXTS:
            dst = config.VIDEO_DIR / f"{Path(f.filename).stem}-{uuid.uuid4().hex[:6]}{ext}"
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            ids = frames.extract(dst, dataset, every_s=every)
            videos.append({"video": f.filename, "frames": len(ids)})
            added += ids
        else:
            skipped.append(f.filename)
    log("webui", f"upload to {dataset}: {len(added)} images" + (f", {len(videos)} videos" if videos else "")
        + (f", {len(dupes)} duplicates skipped" if dupes else "") + (f", unreadable {skipped}" if skipped else ""),
        user=x_labeler, level="warning" if skipped else "info")
    return {"added": len(added), "videos": videos, "skipped": skipped, "duplicates": dupes}


class ScrapeReq(BaseModel):
    urls: list[str]
    dataset: str = "default"
    limit: int = 100


@app.post("/api/scrape")
def scrape(req: ScrapeReq, x_labeler: str = Header("anonymous")):
    ids = scraper.scrape(req.urls, req.dataset, req.limit)
    log("webui", f"scraped {len(ids)} images into {req.dataset} from {len(req.urls)} url(s)", user=x_labeler,
        urls=req.urls)
    return {"added": len(ids)}


# --- annotate ---------------------------------------------------------------

@app.post("/api/queue/next")
def queue_next(skip: str = "", collection: str | None = None, x_labeler: str = Header("anonymous")):
    sid = store.claim_next(x_labeler, collection=collection, skip=[s for s in skip.split(",") if s])
    if sid is None:
        return {"image": None}
    return {"image": store.image_info(store.get_sample(sid))}


@app.post("/api/images/{sample_id}/release")
def release(sample_id: str, who: str | None = None, x_labeler: str = Header("anonymous")):
    # sendBeacon (tab closing) cannot set headers, so the page passes ?who= instead.
    store.release_claim(sample_id, who if x_labeler == "anonymous" and who else x_labeler)
    return {"released": sample_id}


@app.get("/api/images")
def images(status: str | None = None, dataset: str | None = None, state: str | None = None,
           limit: int = 500, newest_first: bool = False):
    if state and state not in store.STATES:
        raise HTTPException(422, f"state must be one of {store.STATES}")
    return store.list_images(status=status, collection=dataset, state=state, limit=limit,
                             newest_first=newest_first)


@app.get("/api/images/states")
def image_states():
    return store.state_counts()


@app.get("/api/images/{sample_id}/file")
def image_file(sample_id: str):
    s = _sample_or_404(sample_id)
    if not Path(s.filepath).exists():
        raise HTTPException(404, "file missing")
    return FileResponse(s.filepath)


@app.get("/api/images/{sample_id}/labels")
def get_labels(sample_id: str):
    s = _sample_or_404(sample_id)
    labels = store.get_labels(s)
    # Nothing reviewed yet: start from the auto-labeler's proposals (shown dashed).
    if not labels:
        labels = store.get_labels(s, field="autolabels")
    return {"image": store.image_info(s), "labels": labels}


class Label(BaseModel):
    cls: int
    points: list[list[float]]
    source: str = "manual"
    score: float | None = None


class LabelSet(BaseModel):
    labels: list[Label]


@app.put("/api/images/{sample_id}/labels")
def put_labels(sample_id: str, body: LabelSet, x_labeler: str = Header("anonymous")):
    for lb in body.labels:
        if not 0 <= lb.cls < len(config.CLASSES) or len(lb.points) != 4:
            raise HTTPException(422, f"bad label {lb}")
    s = _sample_or_404(sample_id)
    if s.status == "ingest" and s.claimed_by not in (None, x_labeler):
        raise HTTPException(409, f"{s.claimed_by} is labeling this image")
    store.set_labels(s, [lb.model_dump() for lb in body.labels])
    store.renew_claim(sample_id, x_labeler)
    return {"saved": len(body.labels)}


@app.post("/api/images/{sample_id}/processed")
def processed(sample_id: str, x_labeler: str = Header("anonymous")):
    s = _sample_or_404(sample_id)
    s = store.mark_processed(sample_id, user=x_labeler)
    n = len(store.get_labels(s))
    log("webui", f"{x_labeler} labeled {s.filename}: {n} box(es)", user=x_labeler, sample=sample_id)
    return store.image_info(s)


@app.delete("/api/images/{sample_id}")
def delete_image(sample_id: str, x_labeler: str = Header("anonymous")):
    s = _sample_or_404(sample_id)
    name = s.filename
    store.delete_image(sample_id)
    log("webui", f"{x_labeler} rejected {name}", user=x_labeler)
    return {"deleted": sample_id}


# --- reports & metrics --------------------------------------------------------

@app.get("/api/runs")
def runs():
    deployed = (store.coll("deploy").find_one({"_id": "current"}) or {}).get("run")
    out = store.list_runs()
    for r in out:
        r["deployed"] = r["id"] == deployed
        if r.get("metrics"):
            r["metrics"] = {k: v for k, v in r["metrics"].items() if k not in ("history", "dataset")}
    return out


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: int):
    r = store.get_run(run_id)
    if not r or not r.get("report_path") or not Path(r["report_path"]).exists():
        raise HTTPException(404, "no report yet")
    return Path(r["report_path"]).read_text()


@app.get("/api/runs/{run_id}/plot/{name}")
def run_plot(run_id: int, name: str):
    path = config.RUNS_DIR / f"run-{run_id}" / Path(name).name
    if path.suffix != ".png" or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/metrics")
def metrics_files():
    """The metrics/ directory: evaluator output and smoke tests, one JSON per file."""
    config.ensure_dirs()
    return sorted(p.name for p in config.METRICS_DIR.glob("*.json"))


@app.get("/api/metrics/{name}")
def metrics_file(name: str):
    path = config.METRICS_DIR / Path(name).name
    if path.suffix != ".json" or not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())


# --- cluster ----------------------------------------------------------------

@app.get("/api/cluster")
def cluster():
    """Seat pods (refreshed in the background every 20 s), plus pipeline state from the DB."""
    from swordnet.bridge.monitor import MONITOR
    runs = store.list_runs(limit=50)
    deployed = (store.coll("deploy").find_one({"_id": "current"}) or {}).get("run")
    recent = [e for e in logs.query(limit=300) if e["source"] in ("bridge", "agent")][-12:]
    return {**MONITOR.get(), "queue": store.queue_stats(), "deployed": deployed,
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


app.mount("/static", StaticFiles(directory=STATIC), name="static")
