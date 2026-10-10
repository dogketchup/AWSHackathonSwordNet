"""Which team seat does what. Override with SWORDNET_SEATS, e.g.
    SWORDNET_SEATS='{"agent": "seat-145", "aux": "seat-149", "train": {"seat-146": "yolo11s-obb"}}'
"""
import json
import os

DEFAULT = {
    "agent": "seat-145",   # Qwen3 on Trainium: plans runs, writes reports
    "aux": "seat-149",     # auto-labeler + SAM refine
    "train": {             # one model size per seat, trained in parallel on CPU
        "seat-146": "yolo11s-obb",
        "seat-147": "yolo11m-obb",
        "seat-148": "yolo11l-obb",
    },
}

SEATS = json.loads(os.environ["SWORDNET_SEATS"]) if os.environ.get("SWORDNET_SEATS") else DEFAULT


def all_seats():
    return [SEATS["agent"], SEATS["aux"], *SEATS["train"]]
