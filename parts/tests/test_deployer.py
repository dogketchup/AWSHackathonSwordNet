import pytest

from deployer import Deployer
from run_log import RunLog


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeData:
    def __init__(self, n=300):
        self.images = [{"id": f"i{k}", "filename": f"i{k}.jpg", "origin": "upload"} for k in range(n)]
        self.exports = []

    def annotated_images(self):
        return list(self.images)

    def add(self, n):
        start = len(self.images)
        self.images += [{"id": f"i{k}", "filename": f"i{k}.jpg", "origin": "upload"} for k in range(start, start + n)]

    def export_level(self, level, heldout_ids):
        self.exports.append((level.level_id, heldout_ids is not None))


class FakeCluster:
    """Runs take `secs_per_img_epoch * n_images * epochs` of fake time. Knobs simulate failures."""

    def __init__(self, clock, secs_per_img_epoch=0.1, dead=(), flaky_reads=0, pull_fail=0, hang=()):
        self.clock, self.k = clock, secs_per_img_epoch
        self.runs = {}            # run_id -> dict
        self.dead, self.hang = set(dead), set(hang)
        self.flaky_reads, self.pull_fail = flaky_reads, pull_fail
        self.levels, self.heldout_on, self.stopped, self.dropped, self.started = set(), set(), [], [], []

    def push_level(self, seat, level_id):
        self.levels.add((seat, level_id))

    def push_heldout(self, seat):
        self.heldout_on.add(seat)

    def start(self, seat, job, run_id):
        assert (seat, job.level_id) in self.levels, "level must be pushed before start"
        self.started.append((seat, job.kind, job.size, job.level_id, job.seed))
        speed = {"nano": 1, "medium": 4, "large": 10}[job.size]   # bigger models train slower
        dur = self.k * speed * max(1, job.n_train_images) * job.settings["epochs"]
        self.runs[run_id] = {"seat": seat, "job": job, "end": self.clock() + dur}

    def _read(self):
        if self.flaky_reads:
            self.flaky_reads -= 1
            raise ConnectionError("exec stream dropped")

    def status(self, seat, run_id):
        self._read()
        r = self.runs[run_id]
        if r["job"].size in self.dead or r["job"].size in self.hang:
            return None if r["job"].size in self.dead else "running"
        return "done" if self.clock() >= r["end"] else "running"

    def alive(self, seat, run_id):
        self._read()
        r = self.runs[run_id]
        if r["job"].size in self.dead:
            return False
        return r["job"].size in self.hang or self.clock() < r["end"]

    def stop(self, seat, run_id):
        self.stopped.append(run_id)

    def collect(self, seat, run_id):
        if self.pull_fail:
            self.pull_fail -= 1
            raise ConnectionError("pull failed")
        return {"best_mAP50_95": 0.5, "epochs_run": 30, "findings": []}

    def evaluate(self, seat, run_id, dev_best):
        assert seat in self.heldout_on, "evaluated before the held-out set was pushed"
        job = self.runs[run_id]["job"]
        base = {"nano": 0.60, "medium": 0.70, "large": 0.72}[job.size] + 0.0001 * job.n_train_images / 10
        return {"mAP50": base + 0.1, "mAP50_95": base, "precision": 0.8, "recall": 0.7,
                "cpu_ms": {"nano": 50, "medium": 200, "large": 400}[job.size],
                "phone_mb": {"nano": 3, "medium": 20, "large": 26}[job.size], "file_mb": 5,
                "params": {"nano": 2.6e6, "medium": 20e6, "large": 25e6}[job.size], "findings": []}

    def drop_level(self, seat, level_id):
        self.dropped.append((seat, level_id))


def make(tmp_path, cluster_kw=None, data=None, deadline=200_000, **kw):
    clock = Clock()
    cluster = FakeCluster(clock, **(cluster_kw or {}))
    data = data or FakeData()
    d = Deployer(cluster, data, RunLog(tmp_path / "metrics"), trainers=["s1", "s2", "s3"], evaluator="ev",
                 deadline=deadline, min_new=50, min_heldout=10, poll=60, clock=clock, sleep=clock.sleep,
                 say=lambda *a: None, state_path=tmp_path / "state.json", **kw)
    return d, cluster, data, clock


def run_steps(d, clock, n):
    for _ in range(n):
        d.step()
        clock.sleep(d.poll)


def test_calibrates_then_runs_noise_first_then_grid(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 1)
    assert set(d.epoch_time) == {"nano", "medium", "large"}
    assert all(t is not None for t in d.epoch_time.values())
    first = [s[1:] for s in c.started if s[1] != "calibrate"][:3]
    assert first[0][0] == "noise" and first[1][0] == "noise"
    run_steps(d, clock, 400)
    recs = [r for r in d.log.records() if r["status"] == "done" and r["kind"] in ("grid", "noise")]
    assert {r["size"] for r in recs if r["kind"] == "grid"} == {"nano", "medium", "large"}
    assert all(r["mAP50_95"] is not None for r in recs)
    assert ("ev" in c.heldout_on)


def test_trainers_never_get_heldout(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 3)
    assert data.exports[0] == (0, True)            # held-out snapshot written once, at level 0
    assert c.heldout_on == {"ev"}                   # and pushed only to the evaluator


def test_new_level_only_after_min_new(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 2)
    data.add(20)
    run_steps(d, clock, 2)
    assert len(d.book.levels) == 1
    data.add(80)
    run_steps(d, clock, 2)
    assert len(d.book.levels) == 2
    assert sum(1 for j in d.queue if j.level_id == 1) + sum(1 for s in c.started if s[3] == 1) >= 3


def test_one_dropped_read_does_not_fail_a_live_run(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 2)
    c.flaky_reads = 2
    run_steps(d, clock, 1)
    assert not [r for r in d.log.records() if r["status"] == "failed"]
    run_steps(d, clock, 600)
    assert not [r for r in d.log.records() if r["status"] == "failed"]


def test_really_dead_run_is_failed_after_three_checks(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 1)                     # calibration done with healthy runs
    c.dead = {"medium"}
    run_steps(d, clock, 600)
    failed = [r for r in d.log.records() if r["status"] == "failed" and r["size"] == "medium"]
    assert failed and all(r["note"] == "no process and no status" for r in failed)


def test_hung_run_is_stopped_as_timeout(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 1)
    c.hang = {"large"}
    run_steps(d, clock, 1500)
    assert [r for r in d.log.records() if r["status"] == "timeout" and r["size"] == "large"]
    assert c.stopped


def test_pull_failure_retries_then_succeeds(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 1)
    c.pull_fail = 2
    run_steps(d, clock, 600)
    assert not [r for r in d.log.records() if r["status"] == "failed"]


def test_jobs_that_dont_fit_in_time_are_skipped(tmp_path):
    d, c, data, clock = make(tmp_path, deadline=4000)
    run_steps(d, clock, 80)
    started = {s[2] for s in c.started if s[1] != "calibrate"}
    assert "large" not in started and "nano" in started


def test_old_level_snapshots_are_dropped(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 300)
    data.add(100)
    run_steps(d, clock, 900)
    assert any(lid == 0 for _, lid in c.dropped)
    assert all(lid != d.book.latest().level_id for _, lid in c.dropped)


def test_state_survives_a_restart(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 2)
    running = {s: r.run_id for s, r in d.running.items()}
    d2 = Deployer(c, data, RunLog(tmp_path / "metrics"), trainers=["s1", "s2", "s3"], evaluator="ev",
                  deadline=200_000, clock=clock, sleep=clock.sleep, say=lambda *a: None,
                  state_path=tmp_path / "state.json")
    assert {s: r.run_id for s, r in d2.running.items()} == running and d2.epoch_time == d.epoch_time


def test_run_ids_are_unique(tmp_path):
    d, c, data, clock = make(tmp_path)
    run_steps(d, clock, 300)
    ids = [r["run_id"] for r in d.log.records()]
    assert len(ids) == len(set(ids))


def test_run_finishes_and_writes_summary(tmp_path):
    from analyze import analyze
    d, c, data, clock = make(tmp_path, deadline=20_000, analyze=analyze)
    d.run(final_report=lambda s: "# report")
    assert (tmp_path / "metrics" / "summary.json").exists()
    assert (tmp_path / "metrics" / "final_report.md").read_text() == "# report"
    assert not d.running


def test_hard_stop_stops_running_jobs(tmp_path):
    d, c, data, clock = make(tmp_path, deadline=1000, hard_stop=1500)
    c.hang = {"nano", "medium", "large"}
    d.run()
    assert [r for r in d.log.records() if r["status"] == "stopped"] or [r for r in d.log.records() if r["status"] == "timeout"]
    assert not d.running


def test_a_crashing_part_does_not_stop_the_loop(tmp_path):
    d, c, data, clock = make(tmp_path)

    def boom():
        raise RuntimeError("storage down")
    data.annotated_images = boom
    run_steps(d, clock, 3)                      # must not raise
    assert not d.book.levels
