"""agent/explore.py: gpt-oss hypotheses are checked before anything launches, bad answers fall back,
and cli.cycle uses the explorer as its planner. No pod or model is touched."""
import json

from swordnet.agent import explore, pick

SEATS = {"seat-146": "yolo11s-obb", "seat-145": "flex"}
HISTORY = [{"model": "yolo11s-obb", "config": {"imgsz": 640, "epochs": 60}}]


def _run(seat, model, imgsz=640, epochs=40):
    return {"seat": seat, "model": model, "epochs": epochs, "imgsz": imgsz, "batch": 8, "reason": "h"}


def good():
    return {"scale": "up", "why": "short of target",
            "runs": [_run("seat-146", "yolo11s-obb", 800), _run("seat-145", "yolo11m-obb")]}


def test_check_accepts_a_valid_plan():
    assert explore.check(good(), SEATS, HISTORY, n_train=40) == []


def test_check_rejects_repeats_bad_scale_and_wrong_model():
    p = {"scale": "bigger", "runs": [_run("seat-146", "yolo11s-obb", 640, 60), _run("seat-145", "yolo11s-obb", 640, 60)]}
    errs = explore.check(p, SEATS, HISTORY)
    assert any('"scale"' in e for e in errs)
    assert any("already tried" in e for e in errs)
    p = {"scale": "up", "runs": [_run("seat-146", "yolo11l-obb"), _run("seat-145", "yolo11n-obb")]}
    assert any("runs yolo11s-obb" in e for e in explore.check(p, SEATS, []))
    p = {"scale": "up", "runs": [_run("seat-146", "yolo11s-obb"), _run("seat-145", "yolo11s-obb")]}
    assert any("repeats another run" in e for e in explore.check(p, SEATS, []))


def _patch(monkeypatch, replies, tmp_path):
    calls = []

    def chat(prompt, **kw):
        calls.append(prompt)
        return replies.pop(0)
    monkeypatch.setattr(explore.llm, "chat", chat)
    monkeypatch.setattr(pick, "runs", lambda: ([], "test"))
    monkeypatch.setattr(explore, "STATE", tmp_path / "explore.json")
    return calls


def test_propose_feeds_rejections_back(monkeypatch, tmp_path):
    bad = {"scale": "up", "runs": [_run("seat-146", "yolo11s-obb", 640, 60), _run("seat-145", "yolo11m-obb")]}
    calls = _patch(monkeypatch, [json.dumps(bad), json.dumps(good())], tmp_path)
    p = explore.propose(SEATS, {"n_train": 40}, HISTORY, 0.5, 150, 20)
    assert p["by"] == "gpt-oss" and p["attempts"] == 2 and p["scale"] == "up"
    assert "already tried" in calls[1]


def test_propose_falls_back_after_three_bad_answers(monkeypatch, tmp_path):
    _patch(monkeypatch, ["not json", "{}", '{"scale": "up"}'], tmp_path)
    p = explore.propose(SEATS, {"n_train": 40}, [], 0.5, 150, 20)
    assert p["by"] == "fallback" and {r["seat"] for r in p["runs"]} == set(SEATS)


def test_cycle_uses_the_explorer_as_planner(monkeypatch, tmp_path):
    from swordnet.bridge import cli
    _patch(monkeypatch, [json.dumps(good())], tmp_path)
    started = []
    ids = iter(range(100, 110))
    team_db = {"export": lambda r: {"manifest": {"n_train": 40, "n_val": 8}, "path": f"/x/{r['name']}"},
               "list_runs": lambda r: {"runs": HISTORY}, "create_run": lambda r: {"id": next(ids)},
               "update_run": lambda r: {"ok": True}}
    monkeypatch.setattr(cli.client, "call", lambda op, req=None, ttl=0: team_db[op](req))
    monkeypatch.setattr(cli, "SEATS", {"agent": "seat-145", "train": SEATS})
    monkeypatch.setattr(cli.kube, "running", lambda seat, pat: False)
    monkeypatch.setattr(cli.kube, "py", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Qwen planner called")))
    monkeypatch.setattr(cli.kube, "pull", lambda *a, **k: True)
    monkeypatch.setattr(cli.kube, "push", lambda *a, **k: None)
    monkeypatch.setattr(cli.kube, "spawn", lambda seat, script, log: started.append((seat, script)))

    assert explore.launch(0.5, 150, 20, max_epochs=10) is True
    assert sorted(s for s, _ in started) == ["seat-145", "seat-146"]
    assert all('"epochs": 10' in script for _, script in started)  # capped
    state = explore.last()
    assert state["status"] == "launched" and state["scale"] == "up" and state["by"] == "gpt-oss"


def test_explore_endpoint_reports_last_round(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from swordnet.webui.app import app
    monkeypatch.setattr(explore, "STATE", tmp_path / "explore.json")
    c = TestClient(app)
    assert c.get("/api/best/explore").json() == {}
    explore._save({"status": "launched", "scale": "down"})
    assert c.get("/api/best/explore").json()["scale"] == "down"
