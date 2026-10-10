import annotator
from annotator import MemoryStore, run_pass, snapshot_of


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


TIGHT = [[1, 1], [2, 1], [2, 2], [1, 2]]


class FakeWorker:
    """Tightens every box to TIGHT; can run a hook mid-job (a person editing) or fail."""

    def __init__(self, during=None, fail=False, omit=()):
        self.during, self.fail, self.omit, self.calls = during, fail, set(omit), 0

    def run(self, images, labels):
        self.calls += 1
        if self.fail:
            raise ConnectionError("exec stream dropped")
        if self.during:
            self.during()
        return {img["filename"]: {str(lb["id"]): {"points": TIGHT, "tightened": True, "note": ""}
                                  for lb in labels[img["filename"]]}
                for img in images if img["filename"] not in self.omit}


def store_with(n=3):
    s = MemoryStore()
    for i in range(n):
        s.add_image(i, f"{i}.jpg", labeled_at=i, labels=[(0, box(0, 0, 10, 10))])
    return s


def quiet(*a):
    pass


def test_tightens_and_moves_to_annotated_keeping_original_and_class():
    s = store_with(1)
    st = run_pass(s, FakeWorker(), log=quiet)
    lb = s.labels(0)[0]
    assert st["annotated"] == 1 and s.images[0]["status"] == "annotated"
    assert lb["points"] == TIGHT and lb["orig_points"] == box(0, 0, 10, 10) and lb["cls"] == 0
    assert lb["source"] == "manual+sam"


def test_person_edit_during_job_wins():
    s = store_with(1)
    st = run_pass(s, FakeWorker(during=lambda: s.human_save(0, [(1, box(5, 5, 50, 50)), (0, box(0, 0, 9, 9))])), log=quiet)
    assert st["dropped"] == 1 and s.images[0]["status"] == "labeled"
    assert [l["points"] for l in s.labels(0)] == [box(5, 5, 50, 50), box(0, 0, 9, 9)]


def test_person_finishing_a_different_image_does_not_block_others():
    s = store_with(2)
    st = run_pass(s, FakeWorker(during=lambda: s.human_save(1, [(0, box(1, 1, 3, 3))])), log=quiet)
    assert st["annotated"] == 1 and st["dropped"] == 1


def test_images_without_boxes_go_straight_through_without_the_worker():
    s = MemoryStore()
    s.add_image(0, "neg.jpg")
    w = FakeWorker()
    st = run_pass(s, w, log=quiet)
    assert st["empty"] == 1 and s.images[0]["status"] == "annotated" and w.calls == 0


def test_worker_failure_keeps_images_labeled_then_stuck_after_max_tries():
    s = store_with(1)
    for _ in range(3):
        run_pass(s, FakeWorker(fail=True), max_tries=2, log=quiet)
    assert s.images[0]["status"] == "labeled" and s.images[0]["tries"] == 2
    assert [i["id"] for i in s.stuck(2)] == [0]
    assert s.list_to_annotate(8, 2) == []


def test_missing_result_for_one_image():
    s = store_with(2)
    st = run_pass(s, FakeWorker(omit={"1.jpg"}), log=quiet)
    assert st["annotated"] == 1 and st["failed"] == 1 and s.images[1]["status"] == "labeled"


def test_editing_an_annotated_image_sends_it_back():
    s = store_with(1)
    run_pass(s, FakeWorker(), log=quiet)
    s.human_save(0, [(0, box(0, 0, 20, 20))])
    assert s.images[0]["status"] == "labeled" and s.images[0]["tries"] == 0
    run_pass(s, FakeWorker(), log=quiet)
    assert s.images[0]["status"] == "annotated"


def test_oldest_first_and_filter_before_limit():
    s = MemoryStore()
    for i in range(5):
        s.add_image(i, f"{i}.jpg", labeled_at=10 - i, labels=[(0, box(0, 0, 5, 5))])
    s.images[4]["tries"] = 2          # used up: must not take a slot (N7)
    picked = [i["id"] for i in s.list_to_annotate(2, 2)]
    assert picked == [3, 2]


def test_untightened_boxes_keep_human_points():
    s = store_with(1)

    class Keep:
        def run(self, images, labels):
            return {i["filename"]: {str(lb["id"]): {"points": TIGHT, "tightened": False, "note": "kept"}
                                    for lb in labels[i["filename"]]} for i in images}
    st = run_pass(s, Keep(), log=quiet)
    assert st["annotated"] == 1 and st["kept_boxes"] == 1
    assert s.labels(0)[0]["points"] == box(0, 0, 10, 10) and s.labels(0)[0]["source"] == "manual"


def test_snapshot_ignores_tiny_float_noise_but_not_real_moves():
    a = [{"id": 1, "cls": 0, "points": box(0, 0, 10, 10)}]
    b = [{"id": 1, "cls": 0, "points": [[0.001, 0], [10, 0], [10, 10], [0, 10]]}]
    c = [{"id": 1, "cls": 0, "points": box(0, 0, 11, 10)}]
    assert snapshot_of(a) == snapshot_of(b) != snapshot_of(c)


def test_local_worker_runs_tighten(tmp_path):
    import cv2
    import numpy as np
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((100, 100, 3), np.uint8))
    s = MemoryStore()
    s.add_image(0, "a.jpg", path=str(img), labels=[(0, box(10, 10, 90, 90))])
    st = run_pass(s, annotator.LocalWorker(segment=lambda p, b: []), log=quiet)
    assert st["annotated"] == 1 and st["kept_boxes"] == 1
