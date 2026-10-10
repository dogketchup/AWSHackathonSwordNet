"""Core behaviour that runs on the laptop: store, queue, export, SAM->OBB geometry, evaluator,
agent validators, logs and the web API. Nothing here needs a pod, a GPU or a model."""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from swordnet import config, logs
from swordnet.agent import plan, promote
from swordnet.metrics import evaluate
from swordnet.refine.sam_obb import mask_to_obb
from swordnet.storage import store
from swordnet.train import export

BOX = [[10, 20], [110, 20], [110, 70], [10, 70]]


# --- store & labels ------------------------------------------------------------

def test_labels_roundtrip_in_pixels(make_image):
    sid = make_image()
    s = store.get_sample(sid)
    store.set_labels(s, [{"cls": 0, "points": BOX, "source": "manual"}])
    got = store.get_labels(store.get_sample(sid))
    assert got[0]["cls"] == 0 and got[0]["source"] == "manual"
    assert np.allclose(got[0]["points"], BOX)


def test_mark_processed_moves_file(make_image):
    sid = make_image()
    old = Path(store.get_sample(sid).filepath)
    s = store.mark_processed(sid, user="ana")
    assert s.status == "labeled" and s.labeled_by == "ana"
    assert not old.exists() and Path(s.filepath).parent == config.LABELED_DIR and Path(s.filepath).exists()


# --- queue ---------------------------------------------------------------------

def test_queue_never_hands_one_image_to_two_labelers(make_image):
    col = "queue-race"
    ids = {make_image(collection=col) for _ in range(6)}
    got, lock = [], threading.Lock()

    def worker(name):
        while (sid := store.claim_next(name, collection=col)) is not None:
            with lock:
                got.append(sid)
            store.mark_processed(sid, user=name)

    threads = [threading.Thread(target=worker, args=(f"labeler{i}",)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(got) == sorted(ids), "every image handed out exactly once"


def test_queue_returns_own_claim_and_respects_skip(make_image):
    col = "queue-own"
    a, b = make_image(collection=col), make_image(collection=col)
    first = store.claim_next("bo", collection=col)
    assert store.claim_next("bo", collection=col) == first          # same tab reload: same image
    other = store.claim_next("cy", collection=col)
    assert other != first and other in (a, b)
    store.release_claim(first, "bo")
    assert store.claim_next("bo", collection=col, skip=[first]) is None  # skipped, and the other is cy's


def test_expired_claim_is_requeued(make_image, monkeypatch):
    col = "queue-expire"
    sid = make_image(collection=col)
    assert store.claim_next("dee", collection=col) == sid
    monkeypatch.setattr(store, "CLAIM_SECONDS", -1)
    store.renew_claim(sid, "dee")                                    # now expired
    assert store.claim_next("eve", collection=col) == sid


# --- export --------------------------------------------------------------------

def test_split_is_stable_and_groups_video_frames():
    frames = [{"filename": f"v_f{i:06d}.jpg", "origin": f"video:clip.mp4#{i}"} for i in range(50)]
    assert len({export.split_of(f) for f in frames}) == 1, "one video never straddles train/val"
    imgs = [{"filename": f"img{i}.jpg", "origin": "upload:x"} for i in range(2000)]
    val = sum(export.split_of(i) == "val" for i in imgs) / len(imgs)
    assert 0.17 < val < 0.23
    assert [export.split_of(i) for i in imgs[:20]] == [export.split_of(i) for i in imgs[:20]]


def test_export_writes_yolo_obb(make_image, tmp_path):
    sid = make_image(w=200, h=100)
    store.set_labels(store.get_sample(sid), [{"cls": 0, "points": [[20, 10], [180, 10], [180, 90], [20, 90]]}])
    store.mark_processed(sid)
    m = export.export(tmp_path / "snap")
    assert m["n_train"] + m["n_val"] >= 1
    name = Path(store.get_sample(sid).filepath).stem
    lbl = next((tmp_path / "snap" / "labels").rglob(f"{name}.txt")).read_text().split()
    assert lbl[0] == "0" and np.allclose([float(v) for v in lbl[1:]], [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
    assert "names:" in (tmp_path / "snap" / "data.yaml").read_text()


# --- SAM mask -> OBB -------------------------------------------------------------

def test_mask_to_obb_tightens_diagonal_blade():
    import cv2
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (20, 180), (180, 20), 1, 5)
    pts = np.array(mask_to_obb(mask))
    w = np.linalg.norm(pts[1] - pts[0]); h = np.linalg.norm(pts[2] - pts[1])
    assert min(w, h) < 12 and max(w, h) > 200            # thin and long, not a 160x160 square
    assert mask_to_obb(np.zeros((50, 50), np.uint8)) is None


# --- evaluator -------------------------------------------------------------------

def _rows(train, val, maps):
    return [{"train/box_loss": t, "val/box_loss": v, "metrics/mAP50-95(B)": m, "metrics/mAP50(B)": m + 0.2}
            for t, v, m in zip(train, val, maps)]


def test_evaluator_flags_overfitting():
    n = 30
    train = [2 - i * 0.05 for i in range(n)]
    val = [2 - i * 0.05 if i < 15 else 1.25 + (i - 15) * 0.04 for i in range(n)]
    maps = [min(i, 15) * 0.03 - max(0, i - 15) * 0.01 for i in range(n)]
    s = evaluate.curve_signals(_rows(train, val, maps))
    assert s["best_epoch"] == 16
    assert any(f.startswith("OVERFITTING") for f in s["findings"])


def test_evaluator_flags_underfitting():
    n = 12
    s = evaluate.curve_signals(_rows([3 - i * 0.2 for i in range(n)], [3 - i * 0.18 for i in range(n)],
                                     [i * 0.02 for i in range(n)]))
    assert any(f.startswith("UNDERFITTING") for f in s["findings"])


def test_dataset_signals():
    f = evaluate.dataset_signals({"n_val": 8, "objects": {"train": {"sword": 400, "stick": 20}},
                                  "label_sources": {"auto": 50, "manual": 10}})
    text = " ".join(f)
    assert "SMALL VAL SET" in text and "CLASS IMBALANCE" in text and "MOSTLY AUTO-LABELS" in text


# --- agent checkers ---------------------------------------------------------------

SEATS = {"seat-146": "yolo11s-obb", "seat-147": "yolo11m-obb"}


def test_plan_validator_names_the_problem():
    good = {"runs": [{"seat": "seat-146", "model": "yolo11s-obb", "epochs": 50, "imgsz": 640, "batch": 16},
                     {"seat": "seat-147", "model": "yolo11m-obb", "epochs": 40, "imgsz": 640, "batch": 8}]}
    assert plan.validate(good, SEATS) == []
    bad = json.loads(json.dumps(good))
    bad["runs"][0]["imgsz"] = 650
    bad["runs"][1]["model"] = "yolo11x-obb"
    errs = plan.validate(bad, SEATS)
    assert any("multiple of 32" in e for e in errs) and any("seat-147 runs yolo11m-obb" in e for e in errs)
    assert any("too large for 40" in e for e in plan.validate(
        {"runs": [dict(r, batch=64) for r in good["runs"]]}, SEATS, n_train=40))
    assert plan.fallback(SEATS, [])["runs"][0]["model"] == "yolo11s-obb"
    flex = {"seat-145": "flex"}
    assert plan.validate({"runs": [{"seat": "seat-145", "model": "yolo11l-obb", "epochs": 30, "imgsz": 640,
                                    "batch": 8}]}, flex) == []
    assert plan.validate({"runs": [{"seat": "seat-145", "model": "yolov8-huge", "epochs": 30, "imgsz": 640,
                                    "batch": 8}]}, flex)
    assert plan.fallback(flex, [])["runs"][0]["model"] == "yolo11s-obb"


def test_promote_validator():
    runs = [{"id": 1, "best_mAP50_95": 0.60}, {"id": 2, "best_mAP50_95": 0.50}, {"id": 3, "smoke_failed": True}]
    assert promote.validate({"run": 9}, runs, None)
    assert promote.validate({"run": 3}, runs, None)
    assert promote.validate({"run": 2, "reason": "smaller"}, runs, current=1)       # worse, no speed reason
    assert not promote.validate({"run": 2, "reason": "3x the fps"}, runs, current=1)
    assert promote.fallback(runs)["run"] == 1


# --- logs ------------------------------------------------------------------------

def test_logs_query_filters():
    logs.log("unit", "plain info", run_id=99)
    logs.log("unit", "something broke (a+b", level="error", run_id=99)
    logs.log_lines("seat-test", ["Epoch 1/3", "Traceback (most recent call last):"], run_id=99)
    assert len(logs.query(run_id=99)) == 4
    assert {e["msg"] for e in logs.query(run_id=99, level="error")} == {"something broke (a+b",
                                                                         "Traceback (most recent call last):"}
    assert len(logs.query(text="(a+b")) == 1                              # regex chars are literal


# --- web API ---------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from swordnet.webui.app import app
    return TestClient(app)


def test_api_label_flow(client, tmp_path):
    import cv2
    img = tmp_path / "s.jpg"
    cv2.imwrite(str(img), np.full((100, 150, 3), 120, np.uint8))
    r = client.post("/api/upload", files=[("files", ("s.jpg", img.read_bytes(), "image/jpeg"))],
                    data={"dataset": "api"}, headers={"X-Labeler": "fay"})
    assert r.json()["added"] == 1

    h = {"X-Labeler": "fay"}
    nxt = client.post("/api/queue/next", params={"collection": "api"}, headers=h).json()["image"]
    assert nxt and nxt["claimed_by"] == "fay"
    # someone else cannot overwrite fay's claimed image
    r = client.put(f"/api/images/{nxt['id']}/labels", json={"labels": []}, headers={"X-Labeler": "gus"})
    assert r.status_code == 409
    r = client.put(f"/api/images/{nxt['id']}/labels", json={"labels": [{"cls": 0, "points": BOX}]}, headers=h)
    assert r.json()["saved"] == 1
    assert client.post(f"/api/images/{nxt['id']}/processed", headers=h).json()["status"] == "labeled"
    assert client.get(f"/api/images/{nxt['id']}/file").status_code == 200
    assert any("fay labeled" in e["msg"] for e in client.get("/api/logs", params={"source": "webui"}).json()["entries"])


def test_api_metrics_dir(client):
    config.ensure_dirs()
    (config.METRICS_DIR / "run-1.json").write_text(json.dumps({"best_mAP50_95": 0.5, "history": []}))
    assert "run-1.json" in client.get("/api/metrics").json()
    assert client.get("/api/metrics/run-1.json").json()["best_mAP50_95"] == 0.5
    assert client.get("/api/metrics/..%2Fswordnet.json").status_code == 404


def test_image_states(make_image, client):
    col = "states"
    pending, annotating, labeled = (make_image(collection=col) for _ in range(3))
    assert store.claim_next("hal", collection=col, skip=[pending, labeled]) == annotating
    store.mark_processed(labeled, user="ivy")
    by_state = {i["id"]: i for i in client.get("/api/images", params={"dataset": col}).json()}
    assert by_state[pending]["state"] == "pending" and by_state[pending]["claimed_by"] is None
    assert by_state[annotating]["state"] == "annotating" and by_state[annotating]["claimed_by"] == "hal"
    assert by_state[labeled]["state"] == "labeled" and by_state[labeled]["labeled_by"] == "ivy"
    only = client.get("/api/images", params={"dataset": col, "state": "annotating"}).json()
    assert [i["id"] for i in only] == [annotating]
    counts = client.get("/api/images/states").json()
    assert counts["annotating"] >= 1 and counts["pending"] >= 1 and counts["labeled"] >= 1


def test_upload_skips_duplicates(client, tmp_path):
    import cv2
    img = tmp_path / "d.jpg"
    cv2.imwrite(str(img), np.random.default_rng(7).integers(0, 255, (80, 120, 3), dtype=np.uint8))
    files = lambda: [("files", ("d.jpg", img.read_bytes(), "image/jpeg"))]
    first = client.post("/api/upload", files=files(), data={"dataset": "dup"}).json()
    n_files = len(list(config.INGEST_DIR.iterdir()))
    again = client.post("/api/upload", files=files(), data={"dataset": "dup"}).json()
    assert first["added"] == 1 and first["duplicates"] == []
    assert again["added"] == 0 and again["duplicates"] == ["d.jpg"]
    assert len(list(config.INGEST_DIR.iterdir())) == n_files, "the rejected copy's file was deleted"
