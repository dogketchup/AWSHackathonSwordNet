"""Evaluator: everything a run report needs, computed from numbers, not opinions.

The agent on Trainium writes the prose; this module is its checker. Each signal below comes
with a plain-language reason, so the report (and the next run's plan) can act on it.

Signals:
  - final and best mAP50 / mAP50-95, precision, recall (on val)
  - best epoch vs last epoch: training past the best epoch was wasted (or overfit)
  - overfitting: val loss rising while train loss keeps falling over the last third of epochs
  - underfitting: train loss still falling steeply at the end -> more epochs would help
  - generalization gap: val/train loss ratio at the end
  - inference speed: ms/image and FPS on this machine, measured on val images
  - dataset warnings: tiny val set, class imbalance, too few objects per class
"""
import csv
import json
import time
from pathlib import Path


def _read_results(run_dir):
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): _num(v) for k, v in r.items()} for r in rows]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _col(rows, *names):
    """First matching column; ultralytics names vary a little between tasks and versions."""
    if not rows:
        return []
    for n in names:
        if n in rows[0]:
            return [r[n] for r in rows]
    return []


def _slope(ys):
    """Least-squares slope per epoch."""
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, sum(ys) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den


def _total_loss(rows, prefix):
    cols = [_col(rows, f"{prefix}/{n}") for n in ("box_loss", "cls_loss", "dfl_loss", "angle_loss")]
    cols = [c for c in cols if c]
    return [sum(v) for v in zip(*cols)] if cols else []


def curve_signals(rows):
    train_loss, val_loss = _total_loss(rows, "train"), _total_loss(rows, "val")
    map50 = _col(rows, "metrics/mAP50(B)", "metrics/mAP50(OBB)", "metrics/mAP50")
    map5095 = _col(rows, "metrics/mAP50-95(B)", "metrics/mAP50-95(OBB)", "metrics/mAP50-95")
    prec = _col(rows, "metrics/precision(B)", "metrics/precision(OBB)", "metrics/precision")
    rec = _col(rows, "metrics/recall(B)", "metrics/recall(OBB)", "metrics/recall")

    s = {"epochs_run": len(rows)}
    if not map5095:
        return s
    best = max(range(len(map5095)), key=lambda i: map5095[i])
    s.update({
        "best_epoch": best + 1,
        "best_mAP50_95": round(map5095[best], 4),
        "best_mAP50": round(map50[best], 4) if map50 else None,
        "final_mAP50_95": round(map5095[-1], 4),
        "final_mAP50": round(map50[-1], 4) if map50 else None,
        "final_precision": round(prec[-1], 4) if prec else None,
        "final_recall": round(rec[-1], 4) if rec else None,
        "epochs_after_best": len(rows) - (best + 1),
    })
    findings = []
    tail = max(3, len(rows) // 3)
    if train_loss and val_loss and len(rows) >= 6:
        tr_slope, va_slope = _slope(train_loss[-tail:]), _slope(val_loss[-tail:])
        s["train_loss_final"] = round(train_loss[-1], 4)
        s["val_loss_final"] = round(val_loss[-1], 4)
        s["val_loss_min_epoch"] = min(range(len(val_loss)), key=lambda i: val_loss[i]) + 1
        s["gap_ratio"] = round(val_loss[-1] / train_loss[-1], 3) if train_loss[-1] else None
        s["train_loss_slope_tail"] = round(tr_slope, 5)
        s["val_loss_slope_tail"] = round(va_slope, 5)
        if tr_slope < 0 and va_slope > 0:
            findings.append(
                f"OVERFITTING: over the last {tail} epochs train loss fell ({tr_slope:+.4f}/epoch) while "
                f"val loss rose ({va_slope:+.4f}/epoch); val loss was lowest at epoch {s['val_loss_min_epoch']}. "
                "Fewer epochs, more augmentation or more data.")
        rel = -tr_slope / train_loss[-1] if train_loss[-1] else 0
        if rel > 0.01 and va_slope <= 0:
            findings.append(
                f"UNDERFITTING: train and val loss were both still falling at the end "
                f"({rel:.1%} of train loss per epoch). More epochs would likely help.")
    if s["epochs_after_best"] > max(10, len(rows) // 3):
        findings.append(
            f"WASTED EPOCHS: best mAP50-95 was at epoch {best + 1} of {len(rows)}; "
            f"{s['epochs_after_best']} epochs after it did not improve it.")
    if len(map5095) >= 2 and map5095[-1] < 0.9 * map5095[best]:
        findings.append(
            f"REGRESSION: final mAP50-95 {map5095[-1]:.3f} is >10% below the best {map5095[best]:.3f}; "
            "use best.pt, not last.pt.")
    s["findings"] = findings
    return s


def dataset_signals(manifest):
    findings = []
    if manifest.get("n_val", 0) < 30:
        findings.append(f"SMALL VAL SET: {manifest.get('n_val', 0)} images; mAP can swing by several points "
                        "from a single image. Treat run-to-run differences under ~5 points as noise.")
    train_objs = manifest.get("objects", {}).get("train", {})
    if train_objs:
        lo, hi = min(train_objs.values()), max(train_objs.values())
        if hi and lo / hi < 0.25:
            findings.append(f"CLASS IMBALANCE in train: {train_objs}. The rare class will have low recall.")
        for cls, n in train_objs.items():
            if n < 50:
                findings.append(f"FEW EXAMPLES: only {n} '{cls}' objects in train.")
    srcs = manifest.get("label_sources", {})
    if srcs.get("auto", 0) > srcs.get("manual", 0):
        findings.append("MOSTLY AUTO-LABELS: more boxes came from the auto-labeler than from people; "
                        "label noise may cap mAP. Review a sample.")
    return findings


def speed(weights, val_images, device="cpu", n=50):
    from ultralytics import YOLO

    imgs = sorted(Path(val_images).glob("*"))[:n]
    if not imgs:
        return {}
    model = YOLO(str(weights))
    model.predict(str(imgs[0]), device=device, verbose=False)  # warm-up
    t = time.perf_counter()
    for im in imgs:
        model.predict(str(im), device=device, verbose=False)
    ms = (time.perf_counter() - t) * 1000 / len(imgs)
    return {"ms_per_image": round(ms, 1), "fps": round(1000 / ms, 1), "speed_device": str(device)}


def evaluate(run_dir, data_yaml, device="cpu"):
    run_dir = Path(run_dir)
    rows = _read_results(run_dir)
    m = curve_signals(rows)
    manifest_path = Path(data_yaml).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    m["dataset"] = manifest
    m.setdefault("findings", []).extend(dataset_signals(manifest))
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        m.update(speed(best, Path(data_yaml).parent / "images" / "val", device=device))
    m["history"] = rows
    return m


if __name__ == "__main__":
    import sys
    print(json.dumps({k: v for k, v in curve_signals(_read_results(sys.argv[1])).items()}, indent=2))
