"""Paths and constants shared by every component.

Everything lives under SWORDNET_HOME (default ./data). On the Trainium seat pod that is
/workspace/AWSHackathonSwordNet/data, so it survives as long as the pod does.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_env(path=REPO / ".env"):
    """Secrets live in .env (gitignored). Values already in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

HOME = Path(os.environ.get("SWORDNET_HOME", "data")).resolve()

INGEST_DIR = HOME / "ingest"      # new images waiting for labels
LABELED_DIR = HOME / "labeled"    # images a labeler marked as processed
VIDEO_DIR = HOME / "videos"       # raw uploaded videos (frames go to INGEST_DIR)
DATASETS_DIR = HOME / "datasets"  # YOLO-OBB snapshots, one per training run
RUNS_DIR = HOME / "runs"          # ultralytics output + reports, one dir per run
DB_PATH = HOME / "swordnet.db"

CLASSES = ["sword", "stick"]
VAL_FRACTION = 0.20

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# LLM on the Trainium chip (the starter kit's vLLM server, OpenAI-compatible).
LLM_BASE_URL = os.environ.get("SWORDNET_LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("SWORDNET_LLM_MODEL", "Qwen/Qwen3-8B")


def ensure_dirs():
    for d in (INGEST_DIR, LABELED_DIR, VIDEO_DIR, DATASETS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
