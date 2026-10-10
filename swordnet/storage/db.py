"""SQLite store: datasets, images, labels (with source tag) and training runs.

One process (the web UI) owns writes in normal use; the agent and trainer open their own
connections. WAL mode lets them read while the UI writes.

A label is always stored as an oriented box: four (x, y) corners in pixel coordinates.
An axis-aligned box from the editor is just the special case with right angles.
"""
import json
import shutil
import sqlite3
import time
from contextlib import contextmanager

from swordnet import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name       TEXT PRIMARY KEY,
    autolabel  INTEGER NOT NULL DEFAULT 0,      -- Web UI toggle; auto-labeler runs only if 1
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY,
    dataset    TEXT NOT NULL REFERENCES datasets(name),
    filename   TEXT NOT NULL UNIQUE,            -- basename inside INGEST_DIR or LABELED_DIR
    status     TEXT NOT NULL DEFAULT 'ingest',  -- ingest | labeled
    origin     TEXT NOT NULL,                   -- upload | scrape:<url> | video:<name>#<frame>
    width      INTEGER,
    height     INTEGER,
    created_at REAL NOT NULL,
    labeled_at REAL
);
CREATE TABLE IF NOT EXISTS labels (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    cls        INTEGER NOT NULL,                -- index into config.CLASSES
    points     TEXT NOT NULL,                   -- JSON [[x,y]*4], pixels
    source     TEXT NOT NULL,                   -- manual | auto:<model> | sam
    score      REAL,
    reviewed   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    model       TEXT NOT NULL,                  -- e.g. yolo11s-obb
    epochs      INTEGER NOT NULL,
    config      TEXT NOT NULL,                  -- JSON of every train() kwarg
    dataset_dir TEXT,
    n_train     INTEGER,
    n_val       INTEGER,
    status      TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed
    metrics     TEXT,                           -- JSON from evaluate.py
    report_path TEXT,
    weights     TEXT,
    started_at  REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);
"""


def connect():
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


# --- datasets ---------------------------------------------------------------

def ensure_dataset(con, name):
    con.execute("INSERT OR IGNORE INTO datasets(name, created_at) VALUES (?, ?)", (name, time.time()))


def set_autolabel(con, name, enabled):
    ensure_dataset(con, name)
    con.execute("UPDATE datasets SET autolabel=? WHERE name=?", (int(bool(enabled)), name))


def autolabel_enabled(con, name):
    row = con.execute("SELECT autolabel FROM datasets WHERE name=?", (name,)).fetchone()
    return bool(row and row["autolabel"])


def list_datasets(con):
    return [dict(r) for r in con.execute("""
        SELECT d.name, d.autolabel,
               SUM(i.status='ingest') AS n_ingest, SUM(i.status='labeled') AS n_labeled
        FROM datasets d LEFT JOIN images i ON i.dataset = d.name
        GROUP BY d.name ORDER BY d.name""")]


# --- images -----------------------------------------------------------------

def add_image(con, dataset, filename, origin, width=None, height=None):
    ensure_dataset(con, dataset)
    cur = con.execute(
        "INSERT INTO images(dataset, filename, origin, width, height, created_at) VALUES (?,?,?,?,?,?)",
        (dataset, filename, origin, width, height, time.time()))
    return cur.lastrowid


def get_image(con, image_id):
    row = con.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    return dict(row) if row else None


def image_path(img):
    base = config.LABELED_DIR if img["status"] == "labeled" else config.INGEST_DIR
    return base / img["filename"]


def list_images(con, status=None, dataset=None, limit=500):
    q, args = "SELECT * FROM images WHERE 1=1", []
    if status:
        q += " AND status=?"; args.append(status)
    if dataset:
        q += " AND dataset=?"; args.append(dataset)
    q += " ORDER BY id LIMIT ?"; args.append(limit)
    return [dict(r) for r in con.execute(q, args)]


def mark_processed(con, image_id):
    """Labeler is done with an image: move the file ingest -> labeled and flip its status.

    Every label on it counts as reviewed from here on, including auto-labels the human kept.
    """
    img = get_image(con, image_id)
    if img is None or img["status"] == "labeled":
        return img
    src = config.INGEST_DIR / img["filename"]
    dst = config.LABELED_DIR / img["filename"]
    if src.exists():
        shutil.move(str(src), str(dst))
    con.execute("UPDATE images SET status='labeled', labeled_at=? WHERE id=?", (time.time(), image_id))
    con.execute("UPDATE labels SET reviewed=1 WHERE image_id=?", (image_id,))
    return get_image(con, image_id)


# --- labels -----------------------------------------------------------------

def add_label(con, image_id, cls, points, source, score=None):
    if len(points) != 4:
        raise ValueError(f"label needs 4 corners, got {len(points)}")
    cur = con.execute(
        "INSERT INTO labels(image_id, cls, points, source, score, created_at) VALUES (?,?,?,?,?,?)",
        (image_id, int(cls), json.dumps([[float(x), float(y)] for x, y in points]), source, score, time.time()))
    return cur.lastrowid


def replace_labels(con, image_id, labels):
    """Editor save: the client sends the full label set for one image."""
    con.execute("DELETE FROM labels WHERE image_id=?", (image_id,))
    for lb in labels:
        add_label(con, image_id, lb["cls"], lb["points"], lb.get("source", "manual"), lb.get("score"))


def get_labels(con, image_id):
    out = []
    for r in con.execute("SELECT * FROM labels WHERE image_id=? ORDER BY id", (image_id,)):
        d = dict(r)
        d["points"] = json.loads(d["points"])
        out.append(d)
    return out


# --- runs -------------------------------------------------------------------

def create_run(con, model, epochs, cfg):
    cur = con.execute("INSERT INTO runs(model, epochs, config) VALUES (?,?,?)",
                      (model, epochs, json.dumps(cfg)))
    return cur.lastrowid


def update_run(con, run_id, **fields):
    for k in ("config", "metrics"):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    sets = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))


def list_runs(con, limit=200):
    out = []
    for r in con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)):
        d = dict(r)
        d["config"] = json.loads(d["config"]) if d["config"] else {}
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else None
        out.append(d)
    return out


def get_run(con, run_id):
    runs = [r for r in list_runs(con, limit=10_000) if r["id"] == run_id]
    return runs[0] if runs else None
