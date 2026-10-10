"""Labeled images in the DB -> a YOLO-OBB dataset snapshot on disk.

Layout written (what ultralytics expects for task=obb):
    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt     one line per object: cls x1 y1 x2 y2 x3 y3 x4 y4 (0..1)
    <out>/data.yaml
    <out>/manifest.json                counts and per-class stats, read by the report

The 80/20 split is decided by hashing a *group key*, not drawn at random:
  - stable: an image never moves from val to train as the dataset grows, so runs over time
    are compared on a val set that only ever gains images;
  - leak-free: every frame from one video shares a group (the video), so near-identical
    frames cannot sit on both sides of the split.
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from swordnet import config
from swordnet.storage import db


def group_key(img):
    origin = img["origin"]
    if origin.startswith("video:"):
        return origin.split("#", 1)[0]
    return img["filename"]


def split_of(img, val_fraction=config.VAL_FRACTION):
    h = int(hashlib.sha1(group_key(img).encode()).hexdigest()[:8], 16)
    return "val" if h / 0xFFFFFFFF < val_fraction else "train"


def label_line(lb, w, h):
    coords = []
    for x, y in lb["points"]:
        coords += [min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)]
    return f"{lb['cls']} " + " ".join(f"{c:.6f}" for c in coords)


def export(out_dir, val_fraction=config.VAL_FRACTION):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    counts = {"train": Counter(), "val": Counter()}
    n_img = Counter()
    sources = Counter()
    with db.session() as con:
        for img in db.list_images(con, status="labeled", limit=10**9):
            src = db.image_path(img)
            if not src.exists() or not img["width"]:
                continue
            split = split_of(img, val_fraction)
            labels = db.get_labels(con, img["id"])
            shutil.copy2(src, out / "images" / split / img["filename"])
            lines = [label_line(lb, img["width"], img["height"]) for lb in labels]
            # An image with no objects is a valid negative example: empty label file.
            (out / "labels" / split / (Path(img["filename"]).stem + ".txt")).write_text("\n".join(lines))
            n_img[split] += 1
            for lb in labels:
                counts[split][config.CLASSES[lb["cls"]]] += 1
                sources[lb["source"].split(":")[0].split("+")[0]] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(config.CLASSES)))
    manifest = {
        "n_train": n_img["train"], "n_val": n_img["val"],
        "objects": {s: dict(c) for s, c in counts.items()},
        "label_sources": dict(sources),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(config.DATASETS_DIR / "latest"))
    a = p.parse_args()
    print(json.dumps(export(a.out), indent=2))


if __name__ == "__main__":
    main()
