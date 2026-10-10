# SwordNet: how the whole thing fits together

SwordNet detects swords and sticks in photos and sparring videos as **oriented boxes** (OBB): a
rotated rectangle that hugs a diagonal blade instead of a big upright box that is mostly background.
Humans label, a cluster of Trainium seat pods trains, two LLM agents plan and review, and the best
weights come back to a laptop for live use.

Every component has its own `TUTORIAL.md` next to its code; this one is the map.

```
 laptop                                              AWS (team seats, EKS cluster hack-hyd)
 ───────────────────────────────────────            ───────────────────────────────────────────
 Web UI :8080  (Upload · Annotate · Reports · Logs)
     │
 FiftyOne DB  (images, labels, queue, runs, logs)    seat-145  Qwen3-8B on Trainium (vLLM)
     │                                                          plan runs · promote · test
     │                                                          + flex YOLO trainer on its CPU
 bridge  ── kubectl exec (verified 1 MB pieces) ───► seat-149  auto-labeler · SAM refine
     │                                                          + flex YOLO trainer on its CPU
     │                                                seat-146  YOLO11s-OBB trainer (CPU)
     │                                                seat-147  YOLO11m-OBB trainer (CPU)
     └── gpt-oss-20b (organisers' endpoint) ◄──       seat-148  YOLO11l-OBB trainer (CPU)
          writes report analysis
```

## Why it is shaped like this

| Constraint we found | Consequence |
|---|---|
| Our cluster role allows only `get pods` and `exec` (no port-forward, no new pods or deployments) | Web UI + DB live on the laptop; the **bridge** talks to pods purely over `kubectl exec`. We have exactly 5 seats (145–149) and cannot create more |
| One exec stream moves ~1 MB/s and is cut off at random, sometimes silently | Files move as 1 MB pieces, 6 at a time, each SHA-256-checked and retried. Expect ~1 min per 30 MB |
| Each seat is capped at 11 CPU cores | All 5 seats train in parallel (55 cores); seat-145 trains with 8 threads so Qwen stays responsive |
| Seat pods have a Trainium chip, no CUDA; ultralytics YOLO cannot train on Trainium | YOLO trains on each pod's 11 CPU cores; the chip runs the Qwen agent, which is what it is good at |
| Seat pods may not reach each other | Each pod gets its own copy of the dataset snapshot from the laptop |
| Secrets must never reach git | Everything secret is in `.env` (gitignored). `.env.example` lists what to fill in |

## The data flow, end to end

1. **Ingest** ([swordnet/ingest](swordnet/ingest/TUTORIAL.md)): uploads, video frames (every 0.5 s),
   scraped pages, or any agent POSTing to `/api/upload`. Files land in `data/ingest/` and in FiftyOne
   with `status=ingest`.
2. **Auto-label** (optional, per upload set): `bridge autolabel` sends new images to seat-149, where
   OWLv2 or Grounding DINO propose boxes. Proposals are stored separately (`autolabels`) and never
   trained on until a human accepts them.
3. **Refine**: `bridge refine` runs SAM on seat-149: box → mask → OpenCV `minAreaRect` → tight OBB
   ([swordnet/refine](swordnet/refine/TUTORIAL.md)).
4. **Label** ([swordnet/webui](swordnet/webui/TUTORIAL.md)): the Annotate tab pulls from a **queue**:
   each labeler is handed an image nobody else holds. *Save + mark processed* moves the file to
   `data/labeled/`.
5. **Train** ([swordnet/train](swordnet/train/TUTORIAL.md)): `bridge cycle` (or `bridge loop` for
   rolling rounds) exports every labeled image to a YOLO-OBB snapshot with a stable, leak-free 80/20
   split, asks Qwen for one run per seat, and trains on all five seats in parallel: s / m / l on
   146–148, plus two "flex" experiments of Qwen's choosing on 145 and 149.
6. **Measure** ([swordnet/metrics](swordnet/metrics/TUTORIAL.md)): each run ends with mAP, losses,
   overfitting / underfitting / wasted-epoch checks, dataset warnings and FPS → `data/metrics/`.
7. **Report** ([swordnet/agent](swordnet/agent/TUTORIAL.md)): one Markdown report per run: numbers
   and findings computed directly, analysis written by gpt-oss-20b (or Qwen if gpt-oss isn't configured).
8. **Deploy** ([swordnet/deploy](swordnet/deploy/TUTORIAL.md)): `bridge promote` has Qwen pick the run
   to deploy, then smoke-tests it; `scripts/get_weights.sh` / `.ps1` pull weights and open a webcam.

Every step writes to one central log (FiftyOne's MongoDB), including the pods' own training output,
which the bridge tails while it waits. The web UI's **Logs** tab filters it by source, run and level.

## First-time setup (laptop)

```bash
# 1. Tools: AWS CLI + kubectl (see trainium-agent-labs/README.md, Part 1, for Mac/Windows)
# 2. Secrets
cp .env.example .env        # fill in the AWS_* block from the organisers' channel, and GPTOSS_BASE_URL
# 3. Point kubectl at the cluster (also re-run with fresh credentials when they expire)
set -a; . ./.env; set +a
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
kubectl get pod seat-145 seat-146 seat-147 seat-148 seat-149     # all Running 1/1
# 4. Python env (FiftyOne starts its own MongoDB on first import)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 5. Pods: push code, install deps, start Qwen (~5 min the first time)
.venv/bin/python -m swordnet.bridge.cli setup
.venv/bin/python -m swordnet.bridge.cli status                   # wait until every seat is ready
# 6. Web UI for the labelers
.venv/bin/uvicorn swordnet.webui.app:app --host 0.0.0.0 --port 8080
```

Teammates on the same network open `http://<laptop-ip>:8080`. Each enters a name once; the queue
uses it so two people never label the same image.

## Daily commands

```bash
B=".venv/bin/python -m swordnet.bridge.cli"
$B status                         # seats, Qwen, gpt-oss, queue
$B autolabel                      # only for upload sets with the Auto-label toggle on
$B refine                         # SAM-tighten drawn boxes on unprocessed images
$B cycle                          # one training round on s/m/l
$B loop --min-new 50              # a new round every 50 newly labeled images
$B promote                        # Qwen picks the run to deploy, smoke test decides
./scripts/get_weights.sh --serve  # best weights -> weights/, then webcam
python -m swordnet.storage.store app   # browse everything in the FiftyOne App (:5151)
```

## Credentials expire

When kubectl says `ExpiredToken`, paste the newest AWS block from the channel into `.env` and re-run
`aws eks update-kubeconfig ...` (step 3). Pods and data are untouched.

## Tests

```bash
.venv/bin/python -m pytest tests      # uses a throwaway FiftyOne dataset; no pods needed
```
If ROS (or anything else) is on your `PYTHONPATH`, run `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests`.
