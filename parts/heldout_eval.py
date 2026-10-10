"""Evaluator: scores one trained model on the frozen held-out set (scheduler plan, Part 2).

Runs on the evaluator seat, the only place the held-out images live.

    python heldout_eval.py --weights best.pt --data heldout/data.yaml --out DIR [--dev-best 0.41]

Writes DIR/heldout.json:
    mAP50, mAP50_95, precision, recall      overall, on the held-out set
    per_class                               {class: {p, r, mAP50, mAP50_95, instances}}
    confusion                               {"true sword -> predicted stick": n, ...}
    params, file_mb                         model size
    cpu_ms, fps                             per-image time on this machine's CPU (a stand-in for a phone)
    phone_format, phone_mb | phone_error    8-bit TFLite export size, or why it failed
    findings                                plain-language problems, each naming what to change
On a crash it writes {"error": ...} instead, so the caller always gets a file.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

CLASSES = ("sword", "stick")
SMALL_HELDOUT = 30


def _r(v):
    return None if v is None else round(float(v), 4)


def size_mb(path):
    path = Path(path)
    if path.is_dir():
        return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)
    return round(path.stat().st_size / 1e6, 2)


def per_class(metrics, names):
    """Ultralytics metrics -> {class name: {p, r, mAP50, mAP50_95, instances}}, present classes only."""
    out = {}
    idx = list(getattr(metrics, "ap_class_index", []) or [])
    nt = getattr(metrics, "nt_per_class", None)
    for i, c in enumerate(idx):
        p, r, ap50, ap = metrics.box.class_result(i)
        out[names[int(c)]] = {"p": _r(p), "r": _r(r), "mAP50": _r(ap50), "mAP50_95": _r(ap),
                              "instances": int(nt[int(c)]) if nt is not None else None}
    return out


def confusion(matrix, names):
    """Ultralytics confusion matrix (rows = predicted, columns = true, last = background) -> counts."""
    if matrix is None:
        return None
    labels = [names[i] for i in sorted(names)] + ["background"]
    bg = len(labels) - 1
    return {f"true {labels[t]} -> predicted {labels[p]}": int(matrix[p][t])
            for t in range(len(labels)) for p in range(len(labels)) if not (p == bg and t == bg)}


def findings(result, dev_best=None, classes=CLASSES):
    out = []
    pc = result.get("per_class") or {}
    for c in classes:
        if c not in pc:
            out.append(f"MISSING CLASS: no '{c}' in the held-out set, so its accuracy is unmeasured. "
                       f"Annotate more '{c}' sessions.")
    rec = {c: v["r"] for c, v in pc.items() if v.get("r") is not None}
    if len(rec) >= 2:
        (lo_c, lo), (hi_c, hi) = min(rec.items(), key=lambda kv: kv[1]), max(rec.items(), key=lambda kv: kv[1])
        if hi - lo >= 0.2:
            out.append(f"WEAK CLASS: '{lo_c}' recall {lo:.2f} vs '{hi_c}' {hi:.2f}. "
                       f"Add more '{lo_c}' images, or try a larger image size for thin objects.")
    conf = result.get("confusion") or {}
    for a in classes:
        for b in classes:
            if a == b:
                continue
            n, hit = conf.get(f"true {a} -> predicted {b}", 0), conf.get(f"true {a} -> predicted {a}", 0)
            if n and n >= 0.2 * (n + hit):
                out.append(f"CONFUSION: {n} true '{a}' predicted as '{b}' ({hit} correct). "
                           "Add examples that separate the two.")
    held = result.get("mAP50_95")
    if dev_best is not None and held is not None and dev_best - held > 0.1:
        out.append(f"DEV/HELD-OUT GAP: mAP50-95 {dev_best:.3f} on dev but {held:.3f} held out. "
                   "The model fits the training sessions; add new sessions rather than more epochs.")
    n_val = result.get("n_val")
    if n_val is not None and n_val < SMALL_HELDOUT:
        out.append(f"SMALL HELD-OUT SET: {n_val} images; treat differences under ~5 points as noise.")
    return out


def cpu_speed(model, images, n=50):
    imgs = sorted(p for p in Path(images).glob("*") if p.is_file())[:n]
    if not imgs:
        return {}
    model.predict(str(imgs[0]), device="cpu", verbose=False)  # warm-up
    t = time.perf_counter()
    for im in imgs:
        model.predict(str(im), device="cpu", verbose=False)
    ms = (time.perf_counter() - t) * 1000 / len(imgs)
    return {"cpu_ms": round(ms, 1), "fps": round(1000 / ms, 1)}


def phone_export(weights, imgsz=None):
    """8-bit TFLite. Newer ultralytics: format="litert", quantize=8; older: "tflite", int8=True."""
    from ultralytics import YOLO

    errors = []
    for fmt, q in (("litert", {"quantize": 8}), ("tflite", {"int8": True})):
        try:
            kw = dict(format=fmt, **q, **({"imgsz": imgsz} if imgsz else {}))
            out = YOLO(str(weights)).export(**kw)
            return {"phone_format": f"{fmt} {q}", "phone_mb": size_mb(out)}
        except Exception as e:
            errors.append(f"{fmt}: {type(e).__name__}: {str(e)[:200]}")
    return {"phone_error": " | ".join(errors)}


def evaluate(weights, data_yaml, out_dir, dev_best=None, phone=True):
    from ultralytics import YOLO

    out_dir = Path(out_dir).resolve()
    data_yaml = Path(data_yaml).resolve()
    # The snapshot was written elsewhere: point data.yaml's path at where it is now.
    lines = data_yaml.read_text().splitlines()
    data_yaml.write_text("\n".join(f"path: {data_yaml.parent}" if l.startswith("path:") else l for l in lines) + "\n")
    manifest = data_yaml.parent / "manifest.json"
    n_val = json.loads(manifest.read_text()).get("n_val") if manifest.exists() else None

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split="val", device="cpu", plots=True,
                  project=str(out_dir.parent), name=out_dir.name, exist_ok=True, verbose=False)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    cm = getattr(getattr(m, "confusion_matrix", None), "matrix", None)
    result = {
        "split": "heldout", "n_val": n_val,
        "mAP50": _r(m.box.map50), "mAP50_95": _r(m.box.map), "precision": _r(m.box.mp), "recall": _r(m.box.mr),
        "per_class": per_class(m, names), "confusion": confusion(cm, names),
        "params": int(sum(p.numel() for p in model.model.parameters())), "file_mb": size_mb(weights),
    }
    result.update(cpu_speed(model, data_yaml.parent / "images" / "val"))
    if phone:
        result.update(phone_export(weights, imgsz=(getattr(model, "overrides", None) or {}).get("imgsz")))
    result["findings"] = findings(result, dev_best, tuple(names[i] for i in sorted(names)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dev-best", type=float, default=None)
    p.add_argument("--no-phone", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        res = evaluate(a.weights, a.data, out, a.dev_best, phone=not a.no_phone)
    except Exception:
        res = {"split": "heldout", "error": traceback.format_exc()[-2000:]}
    tmp = out / "heldout.json.tmp"
    tmp.write_text(json.dumps(res, indent=2))
    tmp.replace(out / "heldout.json")
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
