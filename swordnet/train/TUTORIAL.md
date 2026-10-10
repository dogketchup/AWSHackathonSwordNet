# Train: rolling YOLO-OBB runs on the seat pods

## One round (`bridge cycle`)

1. **Export** ([`export.py`](export.py)): every image with `status=labeled` → a snapshot folder in
   YOLO-OBB format (`images/{train,val}`, `labels/{train,val}` with `cls x1 y1 … x4 y4` normalized,
   `data.yaml`, `manifest.json`). *Each new run grabs all labeled images*, not just new ones.
2. **Plan**: Qwen on seat-145 chooses epochs / image size / batch / augmentation per seat, from the
   history of earlier runs (see `swordnet/agent`).
3. **Train** ([`train.py`](train.py)): the snapshot is copied to each trainer seat, and one run per
   seat starts detached (it survives the bridge disconnecting). All five seats train:

   | seat | model | also does |
   |---|---|---|
   | seat-146 | `yolo11s-obb` (small) | |
   | seat-147 | `yolo11m-obb` (medium) | |
   | seat-148 | `yolo11l-obb` (large) | |
   | seat-145 | flex: Qwen picks | runs Qwen on the chip; trains with 8 CPU threads |
   | seat-149 | flex: Qwen picks | auto-labeler + SAM |

   A seat that is still busy (a slow `l` run from the last round, or an auto-label job) sits the
   round out instead of being double-booked. Seat roles live in `swordnet/bridge/seats.py`.

4. **Evaluate**: when training ends, `swordnet/metrics/evaluate.py` runs on the same pod.
5. **Collect**: the bridge pulls the run folder (weights, `results.csv`, plots, metrics) to
   `data/runs/run-N/`, copies metrics to `data/metrics/`, and writes the report.

## Rolling training

```bash
.venv/bin/python -m swordnet.bridge.cli loop --min-new 50     # a new round per 50 new labels
.venv/bin/python -m swordnet.bridge.cli cycle --max-epochs 3  # quick test round
```

## The 80/20 split

Decided by hashing a group key, not drawn at random:
- **stable**: an image never moves from val to train as data grows, so runs over time are compared
  on a val set that only ever gains images;
- **leak-free**: every frame of a video shares one key, so a clip is entirely train or entirely val.

## Why CPU

The seat pods have a Trainium2 chip but no CUDA GPU, and ultralytics trains on CUDA, Apple MPS or CPU
only. Each pod gives 11 CPU cores. Measured at imgsz 640 (one epoch, including validation):

| model | per image per epoch | 300 images x 50 epochs | 1,000 images x 50 epochs |
|---|---|---|---|
| s | 0.23 s | ~1 h | ~3 h |
| m | 0.56 s | ~2.3 h | ~8 h |
| l | 0.61 s | ~2.5 h | ~8.5 h |

imgsz 480 is roughly 1.8x faster. If the organisers give us a GPU node, `train.py` uses it
automatically. Nothing else changes.

## Dependencies on the pods

`bridge setup` makes `/workspace/swordnet-venv` on each trainer seat with CPU PyTorch + ultralytics,
kept apart from the image's own PyTorch so the Qwen/vLLM server is never disturbed.
