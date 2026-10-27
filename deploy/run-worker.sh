#!/usr/bin/env bash
# Starts a render worker with the built-in defaults (fp32, sequential offload, DPM++ 12 steps);
# worker-config.json, if present, overrides them for this node.
# Coordinator address comes from $COORDINATOR (see /etc/default/multitraktion).
cd "$(dirname "$0")/.."
args=(--coordinator "${COORDINATOR:-http://192.168.1.65:8765}")
[ -f worker-config.json ] && args+=(--config worker-config.json)
exec "$HOME/sdvenv/bin/python" fleet/worker.py "${args[@]}"
