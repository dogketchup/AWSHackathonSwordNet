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

# One class: the footage has only steel swords. Add classes here (and to autolabel.PROMPTS) if
# that changes; label files and the editor follow this list.
CLASSES = ["sword"]
VAL_FRACTION = 0.20

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

METRICS_DIR = HOME / "metrics"    # evaluator output per run (metrics.json, smoke tests), read by the web UI

# Two agents, both OpenAI-compatible:
#   qwen    Qwen3-8B on our own Trainium chip (seat-145, vLLM): plans runs, promotes and tests weights
#   gptoss  the organisers' shared gpt-oss-20b on Trainium: writes the report analysis
LLMS = {
    "qwen": {"base_url": os.environ.get("SWORDNET_QWEN_BASE_URL", "http://localhost:8000/v1"),
             "model": os.environ.get("SWORDNET_QWEN_MODEL", "Qwen/Qwen3-8B")},
    "gptoss": {"base_url": os.environ.get("GPTOSS_BASE_URL", "").rstrip("/") + "/agg/v1"
               if os.environ.get("GPTOSS_BASE_URL") else "",
               "model": os.environ.get("GPTOSS_MODEL", "gpt-oss-20b")},
}


def ensure_dirs():
    for d in (INGEST_DIR, LABELED_DIR, VIDEO_DIR, DATASETS_DIR, RUNS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
