import cv2
import numpy as np

import tighten

H, W = 200, 300


def diagonal_blade(thickness=6):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (60, 160), (220, 40), 1, thickness)
    return m


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def area(pts):
    return cv2.contourArea(np.array(pts, np.float32))


def test_diagonal_blade_becomes_a_thin_rotated_box():
    human = box(50, 30, 230, 170)
    r = tighten.tighten_box(diagonal_blade(), human)
    assert r["tightened"] and len(r["points"]) == 4
    assert area(r["points"]) < 0.25 * area(human)


def test_mask_outside_the_box_is_ignored():
    m = diagonal_blade()
    m[:, 260:] = 1                       # SAM spilled onto something right of the box (an arm)
    r = tighten.tighten_box(m, box(50, 30, 230, 170))
    xs = [p[0] for p in r["points"]]
    assert r["tightened"] and max(xs) < 245   # box + 5% margin, not the spill


def test_biggest_contour_wins():
    m = diagonal_blade()
    cv2.circle(m, (70, 45), 4, 1, -1)    # a small second blob inside the box
    r = tighten.tighten_box(m, box(50, 30, 230, 170))
    assert r["tightened"] and area(r["points"]) < 0.25 * area(box(50, 30, 230, 170))


def test_empty_and_full_masks_keep_the_human_box():
    human = box(50, 30, 230, 170)
    empty = tighten.tighten_box(np.zeros((H, W), np.uint8), human)
    full = tighten.tighten_box(np.ones((H, W), np.uint8), human)
    none = tighten.tighten_box(None, human)
    for r in (empty, full, none):
        assert not r["tightened"] and r["points"] == [[float(x), float(y)] for x, y in human] and r["note"]


def test_no_box_is_ever_dropped(tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((H, W, 3), np.uint8))
    labels = [{"id": 1, "cls": 0, "points": box(50, 30, 230, 170)},
              {"id": 2, "cls": 1, "points": box(10, 10, 40, 40)},
              {"id": 3, "cls": 0, "points": box(100, 100, 150, 150)}]
    only_one_mask = lambda path, boxes: [diagonal_blade()]          # SAM returned fewer masks (H2)
    out = tighten.tighten_image(img, labels, only_one_mask)
    assert set(out) == {"1", "2", "3"} and out["1"]["tightened"] and not out["2"]["tightened"]

    def crash(path, boxes):
        raise RuntimeError("boom")
    out = tighten.tighten_image(img, labels, crash)
    assert set(out) == {"1", "2", "3"} and not any(r["tightened"] for r in out.values())

    none_masks = tighten.tighten_image(img, labels, lambda p, b: [])  # res.masks is None
    assert set(none_masks) == {"1", "2", "3"}


def test_missing_image_keeps_boxes(tmp_path):
    labels = [{"id": 9, "cls": 0, "points": box(0, 0, 10, 10)}]
    out = tighten.tighten_image(tmp_path / "gone.jpg", labels, lambda p, b: [])
    assert not out["9"]["tightened"] and "could not be read" in out["9"]["note"]


def test_wrong_size_mask_is_resized(tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((H, W, 3), np.uint8))
    small = cv2.resize(diagonal_blade(10), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    out = tighten.tighten_image(img, [{"id": 1, "cls": 0, "points": box(50, 30, 230, 170)}], lambda p, b: [small])
    assert out["1"]["tightened"]
