#!/usr/bin/env bash
# Download SwordNet weights from the team registry on the cluster and run them locally.
#
#   ./scripts/get_weights.sh                 # list published models, download the best (mAP50-95)
#   ./scripts/get_weights.sh --run 7         # a specific run
#   ./scripts/get_weights.sh --serve         # ...then open the webcam with it
#   ./scripts/get_weights.sh --serve fight.mp4
#
# Works for any teammate with kubectl access and this repo; it does not need the FiftyOne DB.
# The registry lives on seat-145 (see swordnet/bridge/registry.py). Transfers are chunked and
# checksummed because this cluster drops large exec streams.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="" SERVE="" SOURCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2 ;;
    --serve) SERVE=1; shift; if [ $# -gt 0 ] && [[ $1 != --* ]]; then SOURCE=$1; shift; fi ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

command -v kubectl >/dev/null || { echo "kubectl not found; see TUTORIAL.md" >&2; exit 1; }
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
run() { env -u PYTHONPATH "$PY" -m swordnet.bridge.cli "$@"; }

run models
run download ${RUN:+--run "$RUN"} --out weights

if [ -n "$SERVE" ]; then
  env -u PYTHONPATH "$PY" -c "import ultralytics" 2>/dev/null || "$PY" -m pip install -q ultralytics
  exec env -u PYTHONPATH "$PY" -m swordnet.deploy.predict --weights weights/swordnet-best.pt --source "$SOURCE"
fi
