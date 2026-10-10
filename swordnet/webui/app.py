"""SwordNet web UI: Upload, Annotate and Reports tabs over the label DB.

    uvicorn swordnet.webui.app:app --host 0.0.0.0 --port 8080

Runs on the laptop (port-forward to the pods is blocked). Other people on the same network can
use it at http://<laptop-ip>:8080, and scraping agents can POST images to /api/upload.
"""
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from swordnet import config
from swordnet.ingest import frames, scraper
from swordnet.storage import db

app = FastAPI(title="SwordNet")
STATIC = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# --- datasets & stats ---------------------------------------------------------

@app.get("/api/datasets")
def datasets():
    with db.session() as con:
        return db.list_datasets(con)


class Toggle(BaseModel):
    enabled: bool


@app.post("/api/datasets/{name}/autolabel")
def toggle_autolabel(name: str, body: Toggle):
    with db.session() as con:
        db.set_autolabel(con, name, body.enabled)
    return {"name": name, "autolabel": body.enabled}


@app.get("/api/config")
def ui_config():
    return {"classes": config.CLASSES}


# --- ingest -----------------------------------------------------------------

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), dataset: str = Form("default"),
                 every: float = Form(0.5)):
    config.ensure_dirs()
    added, videos = [], []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext in config.IMAGE_EXTS:
            name = f"up{uuid.uuid4().hex[:12]}{ext}"
            dst = config.INGEST_DIR / name
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            im = cv2.imread(str(dst))
            if im is None:
                dst.unlink()
                continue
            with db.session() as con:
                added.append(db.add_image(con, dataset, name, f"upload:{f.filename}", im.shape[1], im.shape[0]))
        elif ext in config.VIDEO_EXTS:
            dst = config.VIDEO_DIR / f"{Path(f.filename).stem}-{uuid.uuid4().hex[:6]}{ext}"
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            ids = frames.extract(dst, dataset, every_s=every)
            videos.append({"video": f.filename, "frames": len(ids)})
            added += ids
    return {"added": len(added), "videos": videos}


class ScrapeReq(BaseModel):
    urls: list[str]
    dataset: str = "default"
    limit: int = 100


@app.post("/api/scrape")
def scrape(req: ScrapeReq):
    ids = scraper.scrape(req.urls, req.dataset, req.limit)
    return {"added": len(ids)}


# --- annotate ---------------------------------------------------------------

@app.get("/api/images")
def images(status: str | None = None, dataset: str | None = None, limit: int = 500):
    with db.session() as con:
        out = db.list_images(con, status=status, dataset=dataset, limit=limit)
        for img in out:
            labels = db.get_labels(con, img["id"])
            img["n_labels"] = len(labels)
            img["n_auto"] = sum(l["source"].startswith("auto") for l in labels)
        return out


@app.get("/api/images/{image_id}/file")
def image_file(image_id: int):
    with db.session() as con:
        img = db.get_image(con, image_id)
    if not img or not db.image_path(img).exists():
        raise HTTPException(404)
    return FileResponse(db.image_path(img))


@app.get("/api/images/{image_id}/labels")
def get_labels(image_id: int):
    with db.session() as con:
        img = db.get_image(con, image_id)
        if not img:
            raise HTTPException(404)
        return {"image": img, "labels": db.get_labels(con, image_id)}


class Label(BaseModel):
    cls: int
    points: list[list[float]]
    source: str = "manual"
    score: float | None = None


class LabelSet(BaseModel):
    labels: list[Label]


@app.put("/api/images/{image_id}/labels")
def put_labels(image_id: int, body: LabelSet):
    for lb in body.labels:
        if not 0 <= lb.cls < len(config.CLASSES) or len(lb.points) != 4:
            raise HTTPException(422, f"bad label {lb}")
    with db.session() as con:
        if not db.get_image(con, image_id):
            raise HTTPException(404)
        db.replace_labels(con, image_id, [lb.model_dump() for lb in body.labels])
    return {"saved": len(body.labels)}


@app.post("/api/images/{image_id}/processed")
def processed(image_id: int):
    with db.session() as con:
        img = db.mark_processed(con, image_id)
    if not img:
        raise HTTPException(404)
    return img


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    """Reject an image (not a sword/stick photo, broken, duplicate)."""
    with db.session() as con:
        img = db.get_image(con, image_id)
        if not img:
            raise HTTPException(404)
        db.image_path(img).unlink(missing_ok=True)
        con.execute("DELETE FROM images WHERE id=?", (image_id,))
    return {"deleted": image_id}


# --- reports ----------------------------------------------------------------

@app.get("/api/runs")
def runs():
    with db.session() as con:
        out = db.list_runs(con)
    for r in out:
        if r["metrics"]:
            r["metrics"] = {k: v for k, v in r["metrics"].items() if k not in ("history", "dataset")}
    return out


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: int):
    with db.session() as con:
        r = db.get_run(con, run_id)
    if not r or not r["report_path"] or not Path(r["report_path"]).exists():
        raise HTTPException(404, "no report yet")
    return Path(r["report_path"]).read_text()


@app.get("/api/runs/{run_id}/plot/{name}")
def run_plot(run_id: int, name: str):
    path = config.RUNS_DIR / f"run-{run_id}" / Path(name).name
    if path.suffix != ".png" or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
