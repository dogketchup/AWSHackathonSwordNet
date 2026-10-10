#!/usr/bin/env bash
# Run the auto-labeler on the agent seat, from this laptop.
#   scripts/autobox.sh                      label what's waiting in the queue
#   scripts/autobox.sh --dry-run            show what it would do, change nothing
#   scripts/autobox.sh --ingest ~/photos    upload a local folder first (normal ingest path), then label
# Other flags go straight to autobox.py (--limit, --submit-conf, --min-conf).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
SEAT=${SEAT:-seat-145}
TOOLS=/workspace/swordnet-tools
args=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--ingest" ]; then
    dir="$2"; shift 2
    stamp="incoming-$(date +%s)"
    echo "uploading $dir to $SEAT ..."
    tar czf - -C "$dir" . | kubectl exec -i "$SEAT" -c app -- bash -c "mkdir -p $TOOLS/$stamp && tar xzf - -C $TOOLS/$stamp"
    args+=(--ingest "$TOOLS/$stamp")
  else
    args+=("$1"); shift
  fi
done
kubectl exec -i "$SEAT" -c app -- bash -c "mkdir -p $TOOLS && cat > $TOOLS/autobox.py" < scripts/autobox.py
kubectl exec "$SEAT" -c app -- bash -c "cd /workspace/projects/03-swordnet && python3 $TOOLS/autobox.py ${args[*]:-}"
