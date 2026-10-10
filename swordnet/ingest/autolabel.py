"""Auto-labeler: zero-shot detectors propose sword/stick boxes.

Runs on a seat pod (seat-149 by default) through the bridge:
    images dir in  ->  JSON {filename: [{cls, points, score, source}]} out
The bridge only sends images from datasets whose auto-label toggle is on, and imports the
results with source "auto:<model>" so the label editor shows them as unreviewed proposals.

Models (both via Hugging Face transformers):
    owlv2         google/owlv2-base-patch16-ensemble
    groundingdino IDEA-Research/grounding-dino-tiny
"""
import argparse
import json
from pathlib import Path

from swordnet import config

MODELS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "groundingdino": "IDEA-Research/grounding-dino-tiny",
}
# Prompt phrases per class; several phrasings help zero-shot recall.
PROMPTS = {
    0: ["a sword", "a longsword", "a fencing sword", "a katana"],
    1: ["a stick", "a wooden staff", "a bamboo sword"],
}


def _box_to_points(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class AutoLabeler:
    def __init__(self, model="owlv2", device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODELS[model])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODELS[model]).to(self.device).eval()
        self.phrases = [(cls, p) for cls, ps in PROMPTS.items() for p in ps]

    def __call__(self, image, threshold=0.25):
        import torch

        size = [image.size[::-1]]
        if self.name == "groundingdino":
            # Grounding DINO takes one lowercase string, phrases separated by ". "
            text = ". ".join(p for _, p in self.phrases) + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=size)[0]
            labels = res.get("text_labels", res.get("labels"))
            phrase_cls = {p.lower(): c for c, p in self.phrases}
            cls_of = lambda lab: next((c for p, c in phrase_cls.items() if p in str(lab) or str(lab) in p), None)
        else:
            text = [[p for _, p in self.phrases]]
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                out, threshold=threshold, target_sizes=size)[0]
            labels = res["labels"]
            cls_of = lambda lab: self.phrases[int(lab)][0]

        dets = []
        for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            cls = cls_of(lab)
            if cls is not None:
                dets.append({"cls": cls, "points": _box_to_points(*box), "score": round(score, 4),
                             "source": f"auto:{self.name}"})
        return _nms(dets)


def _nms(dets, iou_thr=0.5):
    """Several prompts fire on the same object; keep the best-scoring box per object."""
    def iou(a, b):
        ax1, ay1 = a[0]; ax2, ay2 = a[2]
        bx1, by1 = b[0]; bx2, by2 = b[2]
        iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0
    kept = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["points"], k["points"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


def main():
    from PIL import Image

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="directory of images")
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--model", choices=MODELS, default="owlv2")
    p.add_argument("--threshold", type=float, default=0.25)
    a = p.parse_args()

    labeler = AutoLabeler(a.model)
    results = {}
    files = sorted(f for f in Path(a.images).iterdir() if f.suffix.lower() in config.IMAGE_EXTS)
    for i, f in enumerate(files, 1):
        results[f.name] = labeler(Image.open(f).convert("RGB"), a.threshold)
        print(f"[{i}/{len(files)}] {f.name}: {len(results[f.name])} boxes", flush=True)
    Path(a.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
