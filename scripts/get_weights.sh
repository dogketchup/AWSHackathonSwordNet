#!/usr/bin/env bash
# Grab the best SwordNet weights from the team's trainer pods and run them locally.
#
#   ./scripts/get_weights.sh                 # best run across all trainer seats -> weights/
#   ./scripts/get_weights.sh --run 7         # a specific run
#   ./scripts/get_weights.sh --serve         # ...then open the webcam with it
#   ./scripts/get_weights.sh --serve fight.mp4
#
# Works on any laptop with kubectl access to the cluster; it does not need the FiftyOne DB.
# Credentials come from .env (AWS_*), or whatever your shell already has.
set -euo pipefail
cd "$(dirname "$0")/.."

SEATS=${SWORDNET_TRAIN_SEATS:-"seat-145 seat-146 seat-147 seat-148 seat-149"}
DATA=/workspace/swordnet-data
RUN="" SERVE="" SOURCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --serve) SERVE=1; shift; if [ $# -gt 0 ] && [[ $1 != --* ]]; then SOURCE=$1; shift; fi ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -f .env ]; then set -a; . ./.env; set +a; fi
command -v kubectl >/dev/null || { echo "kubectl not found; see TUTORIAL.md" >&2; exit 1; }

# Each seat prints "<mAP50-95> <seat> <run dir>" for every finished run it has.
best_line=""
for seat in $SEATS; do
  lines=$(kubectl exec "$seat" -c app -- python3 - "$DATA" "$RUN" <<'PY' 2>/dev/null || true
import json, sys, pathlib
data, want = sys.argv[1], sys.argv[2]
for m in pathlib.Path(data, "runs").glob("run-*/metrics.json"):
    run = m.parent
    if want and run.name != f"run-{want}":
        continue
    if (run / "weights" / "best.pt").exists():
        print(json.loads(m.read_text()).get("best_mAP50_95") or 0, run)
PY
)
  while read -r score dir; do
    [ -z "${score:-}" ] && continue
    echo "  $seat ${dir##*/}  mAP50-95=$score"
    if [ -z "$best_line" ] || awk "BEGIN{exit !($score > ${best_line%% *})}"; then
      best_line="$score $seat $dir"
    fi
  done <<< "$lines"
done

[ -n "$best_line" ] || { echo "No finished runs with weights found${RUN:+ for run $RUN}." >&2; exit 1; }
read -r score seat dir <<< "$best_line"
name=${dir##*/}
mkdir -p weights
echo "Downloading $name from $seat (mAP50-95 $score)..."
kubectl exec "$seat" -c app -- cat "$dir/weights/best.pt" > "weights/swordnet-$name.pt"
cp "weights/swordnet-$name.pt" weights/swordnet-best.pt
echo "Saved weights/swordnet-$name.pt and weights/swordnet-best.pt"

if [ -n "$SERVE" ]; then
  PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
  $PY -c "import ultralytics" 2>/dev/null || $PY -m pip install -q ultralytics
  exec $PY -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source "$SOURCE"
fi
