# Metrics: the numbers behind every report

Code that measures lives here; what it measures lands in **`data/metrics/`** (one JSON per run, plus
`run-N-smoke.json` per smoke test). The web UI lists that folder under Reports → *Metrics files*.

## [`evaluate.py`](evaluate.py): runs after every training run

| Signal | What it tells you |
|---|---|
| best / final mAP50, mAP50-95, precision, recall | accuracy on the val split |
| `best_epoch`, `epochs_after_best` | training past the best epoch was wasted |
| **OVERFITTING** | train loss still falling while val loss rises over the last third of training |
| **UNDERFITTING** | both losses still falling steeply at the end: more epochs would help |
| `gap_ratio` | val loss / train loss at the end |
| **REGRESSION** | final mAP > 10% below best: deploy `best.pt`, not `last.pt` |
| `ms_per_image`, `fps` | speed on the pod's CPU |
| **SMALL VAL SET** | under 30 val images: mAP differences of a few points are noise |
| **CLASS IMBALANCE**, **FEW EXAMPLES** | the rare class will have poor recall |
| **MOSTLY AUTO-LABELS** | label noise may be capping mAP |

Each finding is written as a sentence with the numbers in it, because the agents act on these
sentences. Vague "it's worse" messages make the agent loop useless.

## [`smoketest.py`](smoketest.py): before a model is deployed

Loads the checkpoint, checks it is an OBB model with our class names, runs it on real val images,
checks every box is finite, inside the image and non-degenerate, and times it. `bridge promote` only
marks a run *deployed* if every check passes.

Both are unit-tested against synthetic training curves (`tests/test_core.py`).
