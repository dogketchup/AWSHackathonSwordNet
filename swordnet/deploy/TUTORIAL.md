# Deploy: weights onto a laptop

## 1. Pick the model (`bridge promote`)

Qwen looks at every finished run (mAP, FPS, recall) and picks the one to deploy on the target
(default "a laptop CPU"). It must justify trading accuracy for speed, and its pick is smoke-tested
on the pod (`swordnet/metrics/smoketest.py`) before it is marked deployed.

```bash
.venv/bin/python -m swordnet.bridge.cli promote
```

## 2. Get the weights and run them

macOS / Linux:
```bash
./scripts/get_weights.sh                 # best run on any trainer seat -> weights/swordnet-best.pt
./scripts/get_weights.sh --run 7         # a specific run
./scripts/get_weights.sh --serve         # ...and open the webcam
./scripts/get_weights.sh --serve fight.mp4
```
Windows (PowerShell):
```powershell
.\scripts\get_weights.ps1
.\scripts\get_weights.ps1 -Serve -Source fight.mp4
```
The scripts talk to the pods directly with kubectl, so any teammate with cluster access can use
them. They do not need the FiftyOne database. They pick the highest mAP50-95 run; for the run
Qwen promoted, use `bridge pull-weights` on the laptop that runs the bridge.

## 3. Run it any time

```bash
python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source 0            # webcam
python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source clip.mp4 --save
python -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source photos/ --save
```
Needs `pip install ultralytics`. Press `q` to quit. Next step on the roadmap: the hit classifier
(video + sound cues) consumes these boxes.
