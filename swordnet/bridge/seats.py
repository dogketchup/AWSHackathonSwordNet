"""Which team seat does what. Override with SWORDNET_SEATS, e.g.
    SWORDNET_SEATS='{"agent": "seat-145", "aux": "seat-149", "train": {"seat-146": "yolo11s-obb"}}'
"""
import json
import os

FLEX = "flex"  # the planner picks the model size for this seat each round

DEFAULT = {
    "agent": "seat-145",   # Qwen3 on Trainium: plans runs, writes reports
    "aux": "seat-149",     # auto-labeler + SAM refine
    # Every seat trains on its CPU, in parallel. The Qwen chip and the aux jobs leave the CPUs of
    # seat-145 and seat-149 mostly idle, so they become "flex" trainers the planner uses for
    # experiments (another size, image size or augmentation).
    "train": {
        "seat-146": "yolo11s-obb",
        "seat-147": "yolo11m-obb",
        "seat-148": "yolo11l-obb",
        "seat-145": FLEX,
        "seat-149": FLEX,
    },
    # CPU threads for training where the seat has another job; the rest use all 11 cores.
    "train_threads": {"seat-145": 8},
}

SEATS = json.loads(os.environ["SWORDNET_SEATS"]) if os.environ.get("SWORDNET_SEATS") else DEFAULT


def all_seats():
    return list(dict.fromkeys([SEATS["agent"], SEATS["aux"], *SEATS["train"]]))
