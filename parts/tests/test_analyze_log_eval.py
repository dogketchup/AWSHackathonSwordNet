import json

import pytest

import analyze
import heldout_eval
from run_log import RunLog


def rec(size, level, n, m50, m5095, ms, mb, params, seed=0, kind="grid", status="done", phone=True):
    return {"run_id": f"{size}-{level}-{seed}", "kind": kind, "size": size, "model": f"yolo11{size[0]}-obb",
            "level_id": level, "n_train_images": n, "seed": seed, "settings": {"epochs": 30, "seed": seed},
            "status": status, "mAP50": m50, "mAP50_95": m5095, "cpu_ms": ms,
            "phone_mb": mb if phone else None, "file_mb": mb * 2, "params": params}


def day():
    out = []
    for lvl, n in ((0, 100), (1, 300), (2, 600)):
        out += [rec("nano", lvl, n, 0.55 + 0.1 * lvl, 0.35 + 0.08 * lvl, 50, 3, 2.6e6),
                rec("medium", lvl, n, 0.60 + 0.1 * lvl, 0.38 + 0.08 * lvl, 200, 20, 20e6),
                rec("large", lvl, n, 0.61 + 0.1 * lvl, 0.385 + 0.08 * lvl, 400, 26, 25e6)]
    out += [rec("nano", 0, 100, 0.56, 0.36, 50, 3, 2.6e6, seed=1, kind="noise"),
            rec("nano", 0, 100, 0.54, 0.345, 50, 3, 2.6e6, seed=2, kind="noise")]
    return out


def test_noise_from_seed_runs_and_fallback():
    assert analyze.noise(day())["measured"] and analyze.noise(day())["mAP50_95"] == pytest.approx(0.015)
    assert analyze.noise([])["measured"] is False


def test_images_needed_requires_staying_above_target():
    curve = [{"n_images": 100, "mAP50": 0.72}, {"n_images": 300, "mAP50": 0.60}, {"n_images": 600, "mAP50": 0.75}]
    assert analyze.images_needed(curve, 0.70, 0.02) == 600
    assert analyze.images_needed(curve[:2], 0.70, 0.02) is None


def test_picks_smallest_within_noise_that_fits():
    s = analyze.analyze(day(), target=0.70, budget_ms=500, budget_mb=30)
    assert s["level"] == 2
    assert s["recommendation"]["size"] in ("nano", "medium", "large")
    # nano is 0.025 below large (> noise 0.015), so the pick is the smallest within noise of the best
    best = max(c["mAP50_95"] for c in s["candidates"])
    assert s["recommendation"]["mAP50_95"] >= best - s["noise"]["mAP50_95"]
    assert s["images_needed"]["large"] == 300


def test_phone_budget_excludes_big_models():
    s = analyze.analyze(day(), target=0.70, budget_ms=100, budget_mb=10)
    assert s["recommendation"]["size"] == "nano"


def test_nothing_fits_says_why():
    s = analyze.analyze(day(), target=0.99, budget_ms=10, budget_mb=1)
    assert s["recommendation"] is None and "accuracy" in s["reason"] and "speed" in s["reason"]


def test_needs_a_common_level():
    recs = [rec("nano", 1, 300, 0.7, 0.5, 50, 3, 2.6e6), rec("large", 0, 100, 0.7, 0.5, 400, 26, 25e6)]
    assert "every size" in analyze.analyze(recs)["reason"]


def test_failed_runs_are_counted_not_used():
    recs = day() + [rec("large", 2, 600, 0.99, 0.99, 1, 1, 1, status="failed", seed=5)]
    s = analyze.analyze(recs, target=0.7, budget_ms=500, budget_mb=30)
    assert s["runs"]["failed"] == 1 and max(c["mAP50_95"] for c in s["candidates"]) < 0.99


def test_phone_export_failure_falls_back_to_file_size_and_says_so():
    recs = [rec("nano", 0, 100, 0.8, 0.6, 50, 3, 2.6e6, phone=False)]
    s = analyze.analyze(recs, target=0.7, budget_ms=500, budget_mb=30)
    assert s["candidates"][0]["size_source"] == "best.pt" and "phone export failed" in s["reason"]


def test_report_without_and_with_failing_llm():
    s = analyze.analyze(day(), target=0.7, budget_ms=500, budget_mb=30)
    assert "Recommendation" in analyze.write_report(s)

    def broken(prompt):
        raise analyze.GptOssError("unreachable")
    md = analyze.write_report(s, chat=broken)
    assert "gpt-oss failed" in md and "unreachable" in md


def test_gptoss_client_handles_tls_failover_and_empty(monkeypatch):
    import httpx
    calls = []

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b, self.text = code, body, json.dumps(body)

        def json(self):
            return self._b

    class Client:
        def __init__(self, verify, timeout):
            assert verify is False                      # cert does not match the host (N3)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            calls.append(url)
            if "/agg/" in url:
                raise httpx.ConnectError("down")
            return Resp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})

    monkeypatch.setattr(httpx, "Client", Client)
    chat = analyze.gptoss_chat("https://example.invalid")
    with pytest.raises(analyze.GptOssError, match="empty answer"):
        chat("hi")
    assert any("/disagg/" in c for c in calls)


def test_run_log_roundtrip_and_torn_line(tmp_path):
    log = RunLog(tmp_path)
    log.record("r1", {"kind": "grid", "size": "nano", "model": "m", "level_id": 0}, "done",
               heldout={"mAP50": 0.7, "mAP50_95": 0.5, "error": None, "findings": ["x"]},
               train_metrics={"best_mAP50_95": 0.6, "history": [1, 2], "findings": ["y"]})
    with (tmp_path / "runs.jsonl").open("a") as f:
        f.write('{"run_id": "torn')
    recs = log.records()
    assert len(recs) == 1 and recs[0]["findings"] == ["x", "y"] and recs[0]["dev_best_mAP50_95"] == 0.6
    assert "history" not in json.loads((tmp_path / "run-r1.json").read_text())["train_metrics"]
    log.write_seats([{"seat": "s1", "state": "idle"}])
    assert log.read("seats.json")["seats"][0]["seat"] == "s1"


def test_run_log_records_heldout_error_line(tmp_path):
    log = RunLog(tmp_path)
    r = log.record("r2", {"kind": "grid"}, "done", heldout={"error": "Traceback...\nValueError: bad yaml\n"})
    assert r["heldout_error"] == "ValueError: bad yaml"


def test_heldout_confusion_and_findings():
    names = {0: "sword", 1: "stick"}
    # rows predicted, cols true: 8 swords right, 5 swords called stick, 6 sticks right, 1 stick missed
    m = [[8, 0, 1], [5, 6, 0], [0, 1, 0]]
    conf = heldout_eval.confusion(m, names)
    assert conf["true sword -> predicted stick"] == 5 and conf["true stick -> predicted background"] == 1
    assert "true background -> predicted background" not in conf
    res = {"per_class": {"sword": {"r": 0.5}, "stick": {"r": 0.85}}, "confusion": conf, "mAP50_95": 0.3, "n_val": 12}
    f = heldout_eval.findings(res, dev_best=0.5)
    text = " ".join(f)
    assert "WEAK CLASS" in text and "CONFUSION" in text and "DEV/HELD-OUT GAP" in text and "SMALL HELD-OUT" in text


def test_heldout_missing_class_flagged():
    f = heldout_eval.findings({"per_class": {"sword": {"r": 0.9}}, "n_val": 100})
    assert any("MISSING CLASS" in x and "stick" in x for x in f)
