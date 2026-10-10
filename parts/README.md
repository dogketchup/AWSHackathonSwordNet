# SwordNet parts: standalone, not yet in the repo

Each file here implements one part of the team scheduler plan. It follows that plan's pseudocode and doesn't import or change the SwordNet repo. Anything that touches the cluster, the label storage or SAM goes through a small interface, so the logic can be tested now and plugged into wukongxzero's code later.

| File | Plan part | What it does |
|---|---|---|
| `splits.py` | 1 | Train / dev / held-out split by video group; nested data levels; held-out set frozen at the first level |
| `tighten.py` | 0 (pod side) | SAM + OpenCV: human box → mask cut to the box → biggest contour → `minAreaRect`. Never drops a box. Runs as a script on the aux seat |
| `annotator.py` | 0 (laptop side) | The side process: labeled → annotated, a few images per pass, saving only if a person hasn't changed the image meanwhile |
| `deployer.py` | 1 | The training scheduler: levels, calibration, job queue, nano/medium/large, collect, evaluate, log |
| `heldout_eval.py` | 2 | Scores one model on the frozen held-out set: accuracy, per class, confusion, size, CPU ms, phone export |
| `run_log.py` | 3 | `runs.jsonl`, `run-<id>.json`, `seats.json`, `summary.json`, written so readers never see half a file |
| `analyze.py` | 4 | Images needed per size, size/speed/accuracy pick, report; gpt-oss client |
| `dashboard.html` | 3 | Read-only page over the metrics folder: recommendation, seats, curves, trade-off chart, runs |

Tests: `python -m pytest -q parts/tests` from the repo root (54 tests, about 5 s). They use fakes: a fake cluster with a fake clock for the deployer, synthetic masks for SAM, and an in-memory store for the annotator. Nothing here has run against a real pod, real SAM or real YOLO yet.

## What integration needs (later)

- **`annotator.Store` on wukongxzero's FiftyOne storage.** The image stages become pending/annotating (a person's claim) → labeled → **annotated**. "Annotated" is a new state and name; their `annotating` state means a labeler holds the image, a different thing. `apply_if_unchanged` must be atomic: compare the labels and write in one step. `list_to_annotate` must filter before it limits. Saving an annotated image in the web UI must send it back to labeled.
- **`annotator.Worker` over the bridge.** Push the batch's images plus a labels JSON to the aux seat, run `tighten.py`, pull `out.json`. Skip files that vanished while staging.
- **`deployer.Cluster` over `kube.py`.** `alive` must match the exact run id. `status`, `collect` and `evaluate` must raise on a failed exec instead of returning "nothing". Transfers get verified and retried.
- **`deployer.Data`.** `export_level` writes the level's train/dev snapshot from annotated images only, and at level 0 the separate held-out snapshot.
- **Team settings to fill in:** `FIXED_SETTINGS` (epochs, image size), `MIN_NEW`, `MIN_HELDOUT`, and in `analyze.py` `TARGET_MAP50`, `BUDGET_MS`, `BUDGET_MB`. The current values are placeholders.

## Review findings these parts handle

From the first review and the aafd970 review, only the findings in areas these parts cover:

| Finding | What went wrong there | How it's handled here | Tested |
|---|---|---|---|
| Gap 1 | The trainer sees the 20% validation set | Held-out set frozen once, only exported for and pushed to the evaluator; trainers get train + dev | `test_splits`, `test_trainers_never_get_heldout` |
| N2 | SAM never runs on human boxes | The annotator works on labeled images; training uses annotated only | `test_annotator` |
| H2(1) | SAM returning no masks wiped all labels | Every label id always comes back, kept with a note if SAM failed | `test_no_box_is_ever_dropped` |
| H2(2) | Refine overwrote a person's edits | Snapshot, then compare-and-write atomically; edits win, the image is retried | `test_person_edit_during_job_wins` |
| Risky: refine geometry | Biggest contour of the whole mask, not inside the box | Mask cut to the box plus 5% before the contour | `test_mask_outside_the_box_is_ignored` |
| N7 | Limit applied before filtering starves newer images | Interface requires filter, then limit; memory store does it | `test_oldest_first_and_filter_before_limit` |
| N9 | Files vanishing mid-job crash the batch | Unreadable image → boxes kept with a note; missing result → that image retried | `test_missing_image_keeps_boxes`, `test_missing_result_for_one_image` |
| N1 | One dropped exec marks a live run failed | Dead only after 3 checks in a row with no process and no status, and status re-read after a "no process" | `test_one_dropped_read_does_not_fail_a_live_run`, `test_really_dead_run_is_failed_after_three_checks` |
| H1 / risky: no deadline | A dead or hung run polls forever | Dead detection above; runs past 2× their estimate are stopped as `timeout`; calibration has its own limit; hard stop at the end | `test_hung_run_is_stopped_as_timeout`, `test_hard_stop_stops_running_jobs` |
| M3 | One LLM or exec error kills the loop | Every part of a pass is guarded; errors are logged and the loop carries on | `test_a_crashing_part_does_not_stop_the_loop` |
| N8 | A full export on every attempt; snapshots never deleted | Export only when a new level is made; old levels deleted from seats once no job needs them | `test_new_level_only_after_min_new`, `test_old_level_snapshots_are_dropped` |
| L5 | Run ids reused after a reset | Counter plus random suffix, saved in the state file | `test_run_ids_are_unique`, `test_state_survives_a_restart` |
| N3 | gpt-oss TLS failure looks like "unavailable" | `verify=False`, both endpoints with failover, empty answers and real errors shown in the report | `test_gptoss_client_handles_tls_failover_and_empty`, `test_report_without_and_with_failing_llm` |
| M9 | A class missing from evaluation isn't flagged | `MISSING CLASS` finding on the held-out set | `test_heldout_missing_class_flagged` |
| M4 | Group split can land far from 20% | Not changed (moving groups would break stability); a warning is printed when the share is outside 10–35% | read only |

Not handled here because they're in other parts of the system: N4 (`get_weights.sh`), N5 and N13 (labeler claims), N6 (log tailing), N10 (API validation), N11 (transfer retries inside `kube.py`, which the `Cluster` adapter should use), M6 (no auth).

## Known limits

- Calibration time includes model download and per-epoch validation, so estimates run high, which is the safe direction.
- Pod CPU time stands in for phone speed. Real phone speed needs a phone.
- Tightened boxes aren't spot-checked by a person unless the team adds that view.
