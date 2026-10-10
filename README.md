# SwordNet: project overview

**Goal:** Detect swords in sparring footage with tight, oriented bounding boxes (YOLO11-OBB). Data, labels
and run history live in one FiftyOne database on AWS Trainium hardware; teammates reach it from a local web
UI through a Kubernetes bridge (`kubectl exec` only).

**Hand-ins:** [`submission/`](submission/): the checker ([`CHECKER.md`](submission/CHECKER.md)), the attempt
log ([`attempt_log.csv`](submission/attempt_log.csv)), the one-page note ([`NOTE.md`](submission/NOTE.md)) and
a highlight reel.

## Results

| Round | Data | Runs | mAP50-95 |
|---|---|---|---|
| 1. Human labels | 13 train / 2 val images | 5 (YOLO11 s, m, l; 3 image sizes) | 0.084–0.296, mean 0.134 |
| 2. SAM-oriented labels | 46 sparring clips, 1,305 frames → 290 train / 72 val | 4 (YOLO11s at 4 image sizes) | 0.173–0.197, mean 0.188 |
| 3. Continued training | same as round 2, from each seat's best checkpoint | 3 | 0.122–0.183, mean 0.157 |

**Best run:** `seat-148/obb-416` (YOLO11s, image size 416): mAP50 0.398, mAP50-95 0.197. On a held-out
sparring video it found a sword in 156 of 194 frames, and 74% of its boxes were rotated along the blade.
Round 1's high 0.296 came from a 2-image validation set, so it is mostly noise. Round-2 labels were made by
the detector plus SAM, not checked by a person. Details and caveats: [`submission/NOTE.md`](submission/NOTE.md).

## How it is built

| Where | What runs there |
|---|---|
| Each teammate's laptop | The web UI (`http://localhost:8080`, for that person only) and the bridge CLI. No database: every read and write is a `kubectl exec` into the cluster |
| seat-145 (Trainium2) | Qwen3-8B (vLLM): plans training rounds and picks the model to deploy. The team database (MongoDB/FiftyOne). The model registry (weights, metrics and reports of every published run) |
| seat-146 (Trainium2) | gpt-oss-20b (vLLM, BF16): writes the analysis section of each run report |
| seats 145–149 (CPUs) | YOLO11-OBB training, five runs in parallel (YOLO cannot train on Trainium) |
| seat-149 | Auto-labeler (OWLv2 / Grounding DINO) and SAM box refinement |

Our cluster role allows only `get pods` and `exec`: no port-forward and no new services. So everything goes
through `kubectl exec`, and files move in checksummed 1 MB pieces because the cluster drops large streams.

## 1. Data and storage

- One FiftyOne database on seat-145 holds all images, labels, the labeling queue, training runs and logs.
- Laptops only cache image files they have displayed.
- Secrets (AWS keys, tokens) live only in a `.env` file (see `.env.example`). Nothing secret is committed.

## 2. Web UI (runs locally)

- **Upload:** photos and videos (videos become one frame every 0.5 s). Exact duplicates are skipped.
- **Images:** every image with its state: pending, being annotated (by whom), or labeled.
- **Annotate:** a queue hands each labeler an image nobody else holds; draw oriented boxes, then
  *Save + mark processed*.
- **Reports, Cluster, Logs:** run reports and training curves, live status of the five seats and both model
  servers, and one searchable log from every component.
- **Built-in tools:** a web scraper, and an auto-labeler (Grounding DINO or OWLv2) whose suggestions a person
  reviews.

## 3. Label refinement

- SAM makes a mask for each box; OpenCV fits the tightest rotated rectangle to it. If SAM's mask is implausible,
  the human box is kept and the reason recorded.

## 4. Training

- `bridge cycle` exports every labeled image with an 80/20 split grouped by video (one clip never sits on both
  sides), then trains on all five seats. `bridge loop` repeats as new labels arrive.
- Each run is scored on mAP, speed, and checks for overfitting, underfitting, wasted epochs, a small
  validation set and class imbalance.

## 5. Agents

- **Qwen3-8B** proposes each round's runs; a checker accepts or rejects them with reasons that go back into
  the next attempt (3 tries, then a fixed schedule). It also picks the model to deploy, which must pass a
  smoke test.
- **gpt-oss-20b** writes each report's analysis from the computed metrics and findings.

## 6. Metrics and reports

- Evaluator and smoke-test code live in `swordnet/metrics/`; their output is readable in the web UI.
- Each report covers the run's settings (model size, epochs, image size), its scores, overfitting and the
  other findings, a comparison with earlier runs, and the agent's analysis.

## 7. Using the trained model

- `scripts/get_weights.sh` (Linux/macOS) and `scripts/get_weights.ps1` (Windows) download the best published
  weights from the registry; `--serve` / `-Serve` opens them on a webcam or video.
- Video demos with a Kalman-filtered sword tracker and clash detection are on the `best-model` branch
  (`scripts/kalman_video.py`, `scripts/clash_demo.py`).

## 8. Getting started

```bash
cp .env.example .env                 # fill in the workshop AWS block
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
.venv/bin/uvicorn swordnet.webui.app:app --host 127.0.0.1 --port 8080
```

Each component has a `TUTORIAL.md` explaining how it works and where it fits; start with
[`TUTORIAL.md`](TUTORIAL.md).
