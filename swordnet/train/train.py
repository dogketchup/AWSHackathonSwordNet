"""One YOLO-OBB training run. Runs on a trainer seat pod, started by the bridge.

    python -m swordnet.train.train --dataset DIR --run-dir DIR --config '{"model": "yolo11s-obb", "epochs": 50}'

Writes into --run-dir:
    status          queued -> running -> done | failed   (the bridge polls this file)
    train.log       ultralytics output
    weights/best.pt, weights/last.pt, results.csv, ...   (ultralytics)
    metrics.json    evaluate.py's summary (mAP, losses, overfitting signals, FPS)

The seat pods have Trainium, not CUDA, and ultralytics trains on CUDA or CPU only, so this
trains on the pod's CPUs (device=cpu) unless a GPU is present.
"""
import argparse
import json
import os
import traceback
from pathlib import Path

ALLOWED_MODELS = {"yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb"}
# train() kwargs the agent may set; anything else in the config is ignored.
TRAIN_KEYS = {"epochs", "imgsz", "batch", "lr0", "lrf", "patience", "optimizer", "cos_lr",
              "mosaic", "mixup", "degrees", "fliplr", "scale", "close_mosaic", "weight_decay", "dropout"}


def run(dataset, run_dir, cfg):
    import torch
    from ultralytics import YOLO

    from swordnet.metrics import evaluate

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / "status"
    status.write_text("running")

    # data.yaml was written on the laptop; point it at where the snapshot lives on this pod.
    data_yaml = Path(dataset).resolve() / "data.yaml"
    lines = data_yaml.read_text().splitlines()
    data_yaml.write_text("\n".join(f"path: {data_yaml.parent}" if l.startswith("path:") else l for l in lines))

    model_name = cfg.get("model", "yolo11s-obb")
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"model {model_name!r} not in {sorted(ALLOWED_MODELS)}")
    kwargs = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    kwargs.setdefault("epochs", 50)
    kwargs.setdefault("imgsz", 640)
    kwargs.setdefault("patience", 20)
    device = 0 if torch.cuda.is_available() else "cpu"
    # On seats with another job (seat-145 serves Qwen), the bridge caps our CPU threads.
    threads = int(os.environ.get("SWORDNET_TORCH_THREADS", 0)) or None
    if threads:
        torch.set_num_threads(threads)
    workers = min(8, threads or os.cpu_count() or 4)
    # Ultralytics reshuffles the training images every epoch, but its fixed default seed (0) deals
    # every run the same sequence of shuffles. A fresh seed per run, saved in config.json, makes each
    # run's order different and still reproducible.
    kwargs.setdefault("seed", int.from_bytes(os.urandom(4), "big") % 2**31)
    (run_dir / "config.json").write_text(json.dumps({"model": model_name, **kwargs, "device": str(device)}, indent=2))

    model = YOLO(f"{model_name}.pt")
    model.train(data=str(data_yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                device=device, workers=workers, plots=True, verbose=True, **kwargs)

    metrics = evaluate.evaluate(run_dir, data_yaml, device=device)
    metrics["config"] = {"model": model_name, **kwargs}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    status.write_text("done")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", required=True, help="JSON object")
    a = p.parse_args()
    try:
        run(a.dataset, a.run_dir, json.loads(a.config))
    except Exception:
        traceback.print_exc()
        Path(a.run_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.run_dir) / "status").write_text("failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
