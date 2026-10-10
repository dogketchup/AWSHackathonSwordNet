"""Logger: one record per finished, failed or stopped job, in a metrics folder the dashboard reads
(scheduler plan, Part 3).

    runs.jsonl        one line per job, failures included
    run-<id>.json     the full record for one run (training curves + held-out results)
    seats.json        what each seat is doing right now
    summary.json      the analyzer's latest output
    final_report.md   the analyzer's report, written at the end

Whole files are written to a temp name and renamed, so a reader never sees half a file.
runs.jsonl is append-only; a line cut short by a crash is skipped on read.
"""
import json
import os
import threading
import time
from pathlib import Path

HELDOUT_KEYS = ("mAP50", "mAP50_95", "precision", "recall", "per_class", "confusion", "params",
                "file_mb", "phone_mb", "phone_format", "phone_error", "cpu_ms", "fps", "n_val")


class RunLog:
    def __init__(self, metrics_dir):
        self.dir = Path(metrics_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, name, text):
        path = self.dir / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def record(self, run_id, job, status, seat=None, seconds=None, train_metrics=None, heldout=None, note=None):
        """job: {"kind", "size", "model", "level_id", "seed", "settings", "n_train_images"}."""
        train_metrics = train_metrics or {}
        heldout = heldout or {}
        err = heldout.get("error")
        rec = {
            "run_id": run_id, "time": time.time(), "seat": seat, "status": status, "seconds": seconds,
            "kind": job.get("kind"), "size": job.get("size"), "model": job.get("model"),
            "level_id": job.get("level_id"), "n_train_images": job.get("n_train_images"),
            "seed": job.get("seed", 0), "settings": job.get("settings", {}),
            **{k: heldout.get(k) for k in HELDOUT_KEYS},
            "heldout_error": err.strip().splitlines()[-1] if err else None,
            "findings": [*heldout.get("findings", []), *train_metrics.get("findings", [])],
            "dev_best_mAP50_95": train_metrics.get("best_mAP50_95"),
            "epochs_run": train_metrics.get("epochs_run"),
            "note": note,
        }
        with self._lock:
            with (self.dir / "runs.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            full = {**rec, "train_metrics": {k: v for k, v in train_metrics.items() if k != "history"},
                    "heldout": heldout}
            self._write(f"run-{run_id}.json", json.dumps(full, indent=2))
        return rec

    def records(self):
        path = self.dir / "runs.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def write_seats(self, seats):
        self._write("seats.json", json.dumps({"time": time.time(), "seats": seats}, indent=2))

    def write_summary(self, summary):
        self._write("summary.json", json.dumps(summary, indent=2))

    def write_final_report(self, markdown):
        self._write("final_report.md", markdown)

    def read(self, name):
        path = self.dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
