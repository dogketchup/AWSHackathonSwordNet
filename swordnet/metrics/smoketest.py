"""Deployment smoke test for a trained checkpoint. Runs on the trainer seat that has the weights.

    python -m swordnet.metrics.smoketest --weights run-7/weights/best.pt --images DIR --out smoke.json

Checks that would break a local deployment, each reported pass/fail with a reason:
  loads        the checkpoint loads in ultralytics and is an OBB model with our class names
  predicts     inference runs on real val images without error
  sane_boxes   predicted boxes are finite, inside the image, with positive area
  speed        ms/image on this machine (CPU), so you know what the laptop will roughly see
"""
import argparse
import json
import math
import time
from pathlib import Path

from swordnet import config


def smoketest(weights, images, n=20):
    from ultralytics import YOLO

    checks, out = {}, {"weights": str(weights)}
    try:
        model = YOLO(str(weights))
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        ok = model.task == "obb" and names == config.CLASSES
        checks["loads"] = {"pass": ok, "detail": f"task={model.task}, names={names}"}
    except Exception as e:
        checks["loads"] = {"pass": False, "detail": repr(e)}
        return out | {"checks": checks, "pass": False}

    imgs = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in config.IMAGE_EXTS)[:n]
    bad, n_boxes, t = [], 0, 0.0
    try:
        model.predict(str(imgs[0]), verbose=False) if imgs else None  # warm-up
        for p in imgs:
            t0 = time.perf_counter()
            r = model.predict(str(p), verbose=False)[0]
            t += time.perf_counter() - t0
            h, w = r.orig_shape
            if r.obb is None:
                continue
            for poly in r.obb.xyxyxyxy.tolist():
                n_boxes += 1
                xs, ys = [q[0] for q in poly], [q[1] for q in poly]
                if not all(map(math.isfinite, xs + ys)) or min(xs) < -0.05 * w or max(xs) > 1.05 * w \
                        or min(ys) < -0.05 * h or max(ys) > 1.05 * h or (max(xs) - min(xs)) * (max(ys) - min(ys)) <= 0:
                    bad.append(p.name)
        checks["predicts"] = {"pass": bool(imgs), "detail": f"{len(imgs)} images, {n_boxes} boxes"}
        checks["sane_boxes"] = {"pass": not bad, "detail": f"{len(bad)} bad boxes" + (f" e.g. in {bad[:3]}" if bad else "")}
        if imgs:
            ms = t * 1000 / len(imgs)
            checks["speed"] = {"pass": True, "detail": f"{ms:.0f} ms/image ({1000 / ms:.1f} FPS) on CPU"}
            out["ms_per_image"] = round(ms, 1)
    except Exception as e:
        checks["predicts"] = {"pass": False, "detail": repr(e)}
    return out | {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = smoketest(a.weights, a.images)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
