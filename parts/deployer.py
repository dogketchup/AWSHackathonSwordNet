"""Deployer (training scheduler): turns annotated images into YOLO training runs on the cluster,
watches them, and brings their results back (scheduler plan, Part 1).

Each pass (every POLL seconds):
    1. new data -> new level: when MIN_NEW more annotated training images exist, snapshot every
       annotated train/dev image as a new data level and queue one grid job per model size.
       The first level also freezes the held-out set (pushed once to the evaluator seat) and puts
       two extra nano jobs (seeds 1, 2) at the front, to measure run-to-run noise.
    2. collect: for each running job, read its status. A job counts as dead only after DEAD_AFTER
       checks in a row find no process AND no status (review aafd970 N1: one dropped exec must not
       fail a live run). Finished runs are pulled back, scored on the held-out set by the evaluator
       and logged; failed, timed-out and stopped runs are logged too.
    3. hand out work (until the deadline): each idle trainer seat gets the queued job that fits in
       the time left: noise jobs first, then the newest level, then the cheapest size.
    4. housekeeping: snapshots of levels no job needs any more are deleted from the seats (N8),
       seats.json and the state file are rewritten, and the analyzer refreshes the dashboard summary.
Before the first job, one 1-epoch run per size measures seconds per image per epoch, which is
what `estimate` uses.

Cluster and data access are interfaces (`Cluster`, `Data`), so this file has no kubectl or
database code. `tests/test_deployer.py` drives it with a fake cluster and a fake clock.
"""
import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from splits import LevelBook

SIZES = [("nano", "yolo11n-obb"), ("medium", "yolo11m-obb"), ("large", "yolo11l-obb")]  # cheapest first
FIXED_SETTINGS = {"epochs": 30, "imgsz": 640}   # identical for every grid job (team sets)
MIN_NEW = 50
MIN_HELDOUT = 30
EVAL_OVERHEAD = 300   # seconds added to each estimate for pull + held-out scoring
DEAD_AFTER = 3        # consecutive "no process, no status" checks before a run counts as dead
CALIBRATE_LIMIT = 3 * 3600  # seconds a 1-epoch calibration run may take
PULL_TRIES = 5        # failed pulls/evaluations of a finished run before it is logged as failed
POLL = 60


@dataclass
class Job:
    kind: str            # grid | noise | tune | calibrate
    size: str
    model: str
    level_id: int
    seed: int = 0
    settings: dict = field(default_factory=dict)
    n_train_images: int = 0


@dataclass
class Running:
    job: Job
    run_id: str
    seat: str
    started: float
    misses: int = 0
    pull_failures: int = 0


class Cluster(Protocol):
    def push_level(self, seat: str, level_id: int) -> None:
        """Copy that level's train+dev snapshot to the seat (verified). Raise on failure."""
    def push_heldout(self, seat: str) -> None:
        """Copy the frozen held-out snapshot to the evaluator seat. Raise on failure."""
    def start(self, seat: str, job: Job, run_id: str) -> None:
        """Start training detached on the seat. Raise on failure."""
    def status(self, seat: str, run_id: str):
        """The run's status file: "running" | "done" | "failed" | None (no file). Raise if the seat can't be read."""
    def alive(self, seat: str, run_id: str) -> bool:
        """Is the training process for exactly this run id running? (match the whole id, not a
        prefix: review aafd970 N12). Raise if the seat can't be read."""
    def stop(self, seat: str, run_id: str) -> None: ...
    def collect(self, seat: str, run_id: str) -> dict:
        """Pull the run folder back (verified); return the trainer's own metrics (dev curves). Raise on failure."""
    def evaluate(self, seat: str, run_id: str, dev_best) -> dict:
        """Score the run's best.pt on the frozen held-out set on the evaluator seat; heldout.json contents."""
    def drop_level(self, seat: str, level_id: int) -> None:
        """Delete that level's snapshot from the seat."""


class Data(Protocol):
    def annotated_images(self) -> list:
        """Every annotated image: [{"id", "filename", "origin"}]."""
    def export_level(self, level, heldout_ids) -> None:
        """Write the level's train/dev snapshot; with heldout_ids (first level only), also the
        frozen held-out snapshot. Images must carry their annotated (tightened) boxes."""


class Deployer:
    def __init__(self, cluster, data, log, trainers, evaluator, deadline, hard_stop=None,
                 settings=None, sizes=SIZES, min_new=MIN_NEW, min_heldout=MIN_HELDOUT,
                 eval_overhead=EVAL_OVERHEAD, dead_after=DEAD_AFTER, poll=POLL, state_path=None,
                 analyze=None, tune=None, clock=time.time, sleep=time.sleep, say=print):
        self.cluster, self.data, self.log = cluster, data, log
        self.trainers, self.evaluator = list(trainers), evaluator
        self.deadline = deadline
        self.hard_stop = hard_stop if hard_stop is not None else deadline + 1800
        self.settings = dict(settings or FIXED_SETTINGS)
        self.sizes = list(sizes)
        self.rank = {s: i for i, (s, _) in enumerate(self.sizes)}
        self.model_of = dict(self.sizes)
        self.eval_overhead, self.dead_after, self.poll = eval_overhead, dead_after, poll
        self.analyze, self.tune = analyze, tune
        self.clock, self.sleep, self.say = clock, sleep, say
        self.state_path = Path(state_path) if state_path else None

        self.book = LevelBook(min_new=min_new, min_heldout=min_heldout)
        self.queue = []            # [Job]
        self.running = {}          # seat -> Running
        self.epoch_time = {}       # size -> seconds per image per epoch; None = calibration failed
        self.pushed = set()        # (seat, level_id) snapshots on seats
        self.heldout_pushed = False
        self.counter = 0
        self.last_reason = None
        if self.state_path and self.state_path.exists():
            self._load()

    # --- helpers ---------------------------------------------------------------

    def _new_run_id(self, job):
        # Never reused, even after a state reset (scaffold review L5).
        self.counter += 1
        return f"r{self.counter:04d}-L{job.level_id}-{job.size}-s{job.seed}-{uuid.uuid4().hex[:4]}"

    def _job(self, kind, size, level, seed=0, settings=None):
        return Job(kind=kind, size=size, model=self.model_of[size], level_id=level.level_id, seed=seed,
                   settings={**(settings or self.settings), "seed": seed}, n_train_images=level.n_train)

    def estimate(self, job):
        """Seconds for the job, or None if its size has no calibration."""
        t = self.epoch_time.get(job.size)
        if t is None:
            return None
        return t * max(1, job.n_train_images) * job.settings.get("epochs", 1) + self.eval_overhead

    def pick_job(self, time_left):
        """Noise jobs first, then newest level, then cheapest size; skip what can't finish in time."""
        order = sorted(self.queue, key=lambda j: (j.kind != "noise", -j.level_id, self.rank.get(j.size, 99)))
        for job in order:
            est = self.estimate(job)
            if est is not None and est <= time_left:
                self.queue.remove(job)
                return job
        return None

    def _poll(self, r):
        """running | done | failed | unknown. Never "failed" because of one bad read (N1)."""
        try:
            st = self.cluster.status(r.seat, r.run_id)
            if st in ("done", "failed"):
                return st
            if self.cluster.alive(r.seat, r.run_id):
                r.misses = 0
                return "running"
            st = self.cluster.status(r.seat, r.run_id)  # it may have finished between the two reads
            if st in ("done", "failed"):
                return st
        except Exception as e:
            self.say(f"deployer: could not read {r.seat} for {r.run_id} ({type(e).__name__}); will retry")
            return "unknown"
        r.misses += 1
        return "failed" if r.misses >= self.dead_after else "unknown"

    def _max_wall(self, job):
        est = self.estimate(job)
        return None if est is None else 2 * est + 1800

    # --- calibration -----------------------------------------------------------

    def calibrate(self):
        """One 1-epoch run per size on the first level, in parallel across trainer seats."""
        level = self.book.levels[0]
        todo = [s for s, _ in self.sizes if s not in self.epoch_time]
        self.say(f"deployer: calibrating {todo} on level 0 ({level.n_train} images)")
        while todo:
            batch = {}
            for seat in self.trainers:
                if not todo:
                    break
                size = todo.pop(0)
                job = self._job("calibrate", size, level, settings={**self.settings, "epochs": 1})
                try:
                    self._ensure_level(seat, level.level_id)
                    run_id = self._new_run_id(job)
                    self.cluster.start(seat, job, run_id)
                    batch[seat] = Running(job, run_id, seat, self.clock())
                except Exception as e:
                    self.epoch_time[size] = None
                    self.log.record(f"calibrate-{size}", asdict(job), "failed", seat=seat,
                                    note=f"could not start: {type(e).__name__}: {e}")
            while batch:
                self.sleep(self.poll)
                for seat, r in list(batch.items()):
                    out = self._poll(r)
                    # A 1-epoch run that is still going (or unreadable) after CALIBRATE_LIMIT, or at
                    # the hard stop, is stopped; its size is then not scheduled.
                    if out in ("running", "unknown") and (self.clock() - r.started > CALIBRATE_LIMIT
                                                          or self.clock() >= self.hard_stop):
                        self._stop(r)
                        out = "timeout"
                    if out in ("running", "unknown"):
                        continue
                    secs = self.clock() - r.started
                    if out == "done":
                        self.epoch_time[r.job.size] = secs / max(1, r.job.n_train_images)
                    else:
                        self.epoch_time[r.job.size] = None
                    self.log.record(r.run_id, asdict(r.job), out, seat=seat, seconds=secs,
                                    note="calibration (1 epoch); not evaluated")
                    del batch[seat]
            self._save()
        failed = [s for s, t in self.epoch_time.items() if t is None]
        if failed:
            self.say(f"deployer: calibration failed for {failed}; those sizes will not be scheduled")

    # --- the pass --------------------------------------------------------------

    def _ensure_level(self, seat, level_id):
        if (seat, level_id) not in self.pushed:
            self.cluster.push_level(seat, level_id)
            self.pushed.add((seat, level_id))

    def _stop(self, r):
        try:
            self.cluster.stop(r.seat, r.run_id)
        except Exception as e:
            self.say(f"deployer: could not stop {r.run_id} on {r.seat}: {e}")

    def step_levels(self):
        level, reason = self.book.maybe_new_level(self.data.annotated_images(), now=self.clock())
        if reason != self.last_reason:
            self.say(f"deployer: {reason}")
            self.last_reason = reason
        if level is None:
            return None
        first = level.level_id == 0
        self.data.export_level(level, self.book.heldout if first else None)
        if first:
            self.queue[:0] = [self._job("noise", "nano", level, seed=s) for s in (1, 2)]
        self.queue += [self._job("grid", size, level) for size, _ in self.sizes]
        return level

    def step_calibrate(self):
        """Runs once, before any job is handed out (pseudocode START), as soon as level 0 exists."""
        if self.book.levels and len(self.epoch_time) < len(self.sizes) and self.clock() < self.deadline:
            self.calibrate()

    def step_heldout(self):
        if self.book.levels and not self.heldout_pushed:
            try:
                self.cluster.push_heldout(self.evaluator)
                self.heldout_pushed = True
            except Exception as e:
                self.say(f"deployer: could not push the held-out set to {self.evaluator} ({e}); will retry")

    def step_collect(self):
        for seat, r in list(self.running.items()):
            out = self._poll(r)
            limit = self._max_wall(r.job)
            if out in ("running", "unknown") and limit and self.clock() - r.started > limit:
                self._stop(r)
                out = "timeout"
            if out in ("running", "unknown"):
                continue
            secs = self.clock() - r.started
            if out != "done":
                self.log.record(r.run_id, asdict(r.job), out, seat=seat, seconds=secs,
                                note="no process and no status" if out == "failed" else "ran past 2x its estimate")
                del self.running[seat]
                continue
            if not self.heldout_pushed:
                continue  # can't score yet; keep it until the held-out set is on the evaluator
            try:
                train_metrics = self.cluster.collect(seat, r.run_id)
            except Exception as e:
                r.pull_failures += 1
                if r.pull_failures < PULL_TRIES:
                    self.say(f"deployer: pulling {r.run_id} failed ({e}); will retry")
                    continue
                self.log.record(r.run_id, asdict(r.job), "failed", seat=seat, seconds=secs,
                                note=f"finished but could not be pulled after {PULL_TRIES} tries: {e}")
                del self.running[seat]
                continue
            try:
                heldout = self.cluster.evaluate(self.evaluator, r.run_id, (train_metrics or {}).get("best_mAP50_95"))
            except Exception:
                heldout = {"error": traceback.format_exc()[-1500:]}
            self.log.record(r.run_id, asdict(r.job), "done", seat=seat, seconds=secs,
                            train_metrics=train_metrics, heldout=heldout)
            del self.running[seat]

    def step_hand_out(self):
        if self.clock() >= self.deadline:
            return
        for seat in self.trainers:
            if seat in self.running:
                continue
            job = self.pick_job(self.deadline - self.clock())
            if job is None and self.tune:
                job = self._tune_job()
            if job is None:
                continue
            try:
                self._ensure_level(seat, job.level_id)
                run_id = self._new_run_id(job)
                self.cluster.start(seat, job, run_id)
            except Exception as e:
                self.say(f"deployer: could not start {job.kind} {job.size} L{job.level_id} on {seat} ({e}); requeued")
                self.queue.insert(0, job)
                continue
            self.running[seat] = Running(job, run_id, seat, self.clock())

    def _tune_job(self):
        """Optional: ask the tuner (e.g. Qwen3) for settings for the best size on the newest level."""
        level = self.book.latest()
        recs = [x for x in self.log.records() if x.get("status") == "done" and x.get("mAP50_95") is not None]
        if not level or not recs:
            return None
        best = max((x for x in recs if x["kind"] == "grid"), key=lambda x: x["mAP50_95"], default=None)
        if best is None:
            return None
        try:
            settings = self.tune(recs, best["size"])
        except Exception as e:
            self.say(f"deployer: tuner failed ({e}); no tuning job this pass")
            return None
        if not settings:
            return None
        tried = [x.get("settings") for x in recs if x.get("kind") == "tune"]
        settings = {**self.settings, **settings, "seed": 0}
        if settings in tried or settings == {**self.settings, "seed": 0}:
            return None  # duplicate: already tried, or the grid's own settings
        job = self._job("tune", best["size"], level, settings=settings)
        return job if (self.estimate(job) or float("inf")) <= self.deadline - self.clock() else None

    def step_cleanup(self):
        """Delete snapshots of levels no job needs from the seats (N8). The newest level stays."""
        needed = {j.level_id for j in self.queue} | {r.job.level_id for r in self.running.values()}
        latest = self.book.latest().level_id if self.book.levels else None
        for seat, lid in sorted(self.pushed):
            if lid not in needed and lid != latest:
                try:
                    self.cluster.drop_level(seat, lid)
                    self.pushed.discard((seat, lid))
                except Exception as e:
                    self.say(f"deployer: could not delete level {lid} from {seat} ({e}); will retry")

    def step(self):
        """One pass. Each part is guarded, so one error is logged and the loop carries on (M3)."""
        for part in (self.step_levels, self.step_calibrate, self.step_heldout, self.step_collect,
                     self.step_hand_out, self.step_cleanup):
            try:
                part()
            except Exception:
                self.say(f"deployer: {part.__name__} failed, continuing:\n{traceback.format_exc()[-1500:]}")
        self.log.write_seats([{"seat": s, "state": "running" if s in self.running else "idle",
                               **({"run_id": self.running[s].run_id, "kind": self.running[s].job.kind,
                                   "size": self.running[s].job.size, "level_id": self.running[s].job.level_id,
                                   "since": self.running[s].started} if s in self.running else {})}
                              for s in self.trainers])
        if self.analyze:
            try:
                self.log.write_summary(self.analyze(self.log.records()))
            except Exception:
                self.say(f"deployer: analyzer failed:\n{traceback.format_exc()[-800:]}")
        self._save()

    def run(self, final_report=None):
        """Loop until the deadline and all runs are collected, or the hard stop."""
        while True:
            now = self.clock()
            if now >= self.hard_stop:
                for seat, r in list(self.running.items()):
                    self._stop(r)
                    self.log.record(r.run_id, asdict(r.job), "stopped", seat=seat,
                                    seconds=now - r.started, note="hard stop reached")
                    del self.running[seat]
                break
            if now >= self.deadline and not self.running:
                break
            self.step()
            self.sleep(self.poll)
        self._save()
        if self.analyze:
            summary = self.analyze(self.log.records())
            self.log.write_summary(summary)
            if final_report:
                try:
                    self.log.write_final_report(final_report(summary))
                except Exception:
                    self.say(f"deployer: final report failed:\n{traceback.format_exc()[-800:]}")

    # --- state file (so a laptop restart does not orphan runs) -------------------

    def _save(self):
        if not self.state_path:
            return
        state = {"book": self.book.to_json(), "queue": [asdict(j) for j in self.queue],
                 "running": {s: {**asdict(r), "job": asdict(r.job)} for s, r in self.running.items()},
                 "epoch_time": self.epoch_time, "pushed": sorted(self.pushed),
                 "heldout_pushed": self.heldout_pushed, "counter": self.counter}
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)

    def _load(self):
        st = json.loads(self.state_path.read_text())
        self.book = LevelBook.from_json(st["book"])
        self.queue = [Job(**j) for j in st["queue"]]
        self.running = {s: Running(**{**r, "job": Job(**r["job"])}) for s, r in st["running"].items()}
        self.epoch_time = st["epoch_time"]
        self.pushed = {tuple(p) for p in st["pushed"]}
        self.heldout_pushed = st["heldout_pushed"]
        self.counter = st["counter"]
