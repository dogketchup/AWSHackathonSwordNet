# Refine: SAM + OpenCV turn loose boxes into tight oriented boxes

A sword held diagonally fills perhaps 10% of an upright box; the rest is background that teaches
YOLO the wrong thing. [`sam_obb.py`](sam_obb.py) fixes that for every box on unprocessed images:

1. **SAM** (Segment Anything 2.1, via ultralytics) is prompted with the box → a pixel mask of the object.
2. **Clean** the mask: morphological open + close, keep the largest contour.
3. **`cv2.minAreaRect`** on that contour → the smallest rotated rectangle → 4 corners.

```bash
.venv/bin/python -m swordnet.bridge.cli refine      # runs on seat-149
```

Refined boxes get `+sam` in their source and a dot on their corner in the editor. The labeler still
reviews them.

## When SAM is ignored, on purpose

If the mask covers under 1% of the box (SAM found nothing) or over 90% (it grabbed the arm or the
background), the original box is kept and the reason recorded as `refine_note`. A wrong "tight" box
is worse than an honest loose one. Tune `MIN_FILL` / `MAX_FILL` in `sam_obb.py` if your footage needs it.

`mask_to_obb()` is pure OpenCV and unit-tested (`tests/test_core.py::test_mask_to_obb_tightens_diagonal_blade`).
