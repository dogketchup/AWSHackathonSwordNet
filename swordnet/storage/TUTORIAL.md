# Storage: the FiftyOne database

Everything SwordNet remembers lives in **FiftyOne** (Voxel51's dataset database for computer vision),
which runs its own MongoDB on the laptop the first time anything does `import fiftyone`.
Code: [`store.py`](store.py). Nothing else touches the database directly.

## What is stored where

| Thing | Where | Notes |
|---|---|---|
| Images | FiftyOne dataset `swordnet`, one sample each | `filepath` points into `data/ingest/` or `data/labeled/` |
| Human labels | `sample.ground_truth`: `fo.Polylines` | closed 4-point polylines, normalized 0–1, `label` = class, attrs `source`, `score` |
| Auto-label proposals | `sample.autolabels` | kept separate so they are never trained on until a human accepts them |
| Status | `sample.status` | `ingest` → `labeled` (file moves with it) |
| Queue claim | `sample.claimed_by`, `claim_expires` | see below |
| Upload sets + auto-label toggle | Mongo `swordnet_collections` | |
| Training runs | Mongo `swordnet_runs` | config, metrics, report path, weights path, smoke test |
| Deployed run | Mongo `swordnet_deploy` | set by `bridge promote` |
| Logs | Mongo `swordnet_logs` | see `swordnet/logs.py` |

Why oriented boxes are polylines: FiftyOne's `Detection` is axis-aligned only. A closed 4-point
polyline is FiftyOne's usual way to hold a rotated box, and the FiftyOne App draws it correctly.

## The labeling queue

`claim_next(user)` is one atomic Mongo `find_one_and_update`: "give me the oldest ingest image that
nobody holds, and mark it mine for 10 minutes." Two labelers asking at the same instant can never
get the same image (`tests/test_core.py::test_queue_never_hands_one_image_to_two_labelers` runs 4
threads against it). Saving renews the claim; closing the tab or pressing *Skip* releases it; an
abandoned claim simply expires.

## Looking at the data yourself

```bash
.venv/bin/python -m swordnet.storage.store          # counts + queue
.venv/bin/python -m swordnet.storage.store app      # FiftyOne App on http://localhost:5151
```
In the App, color by `ground_truth` vs `autolabels` to see where the auto-labeler disagrees with people.

## Backups

The data is the `data/` folder plus FiftyOne's database. To snapshot labels portably:
```python
import fiftyone as fo
fo.load_dataset("swordnet").export("backup/", dataset_type=fo.types.FiftyOneDataset)
```
