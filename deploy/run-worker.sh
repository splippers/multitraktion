#!/usr/bin/env bash
# Starts a render worker; uses bench-best.json if this node has been tuned.
# Coordinator address comes from $COORDINATOR (see /etc/default/multitraktion).
cd "$(dirname "$0")/.."
args=(--coordinator "${COORDINATOR:-http://192.168.1.65:8765}")
[ -f bench-best.json ] && args+=(--config bench-best.json)
exec "$HOME/sdvenv/bin/python" fleet/worker.py "${args[@]}"
