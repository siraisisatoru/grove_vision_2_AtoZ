#!/usr/bin/env bash
# Stop the camera_relay receiver started by start.sh.
set -euo pipefail
cd "$(dirname "$0")"

PIDS=$(pgrep -f "receiver\.py" || true)
if [[ -z "$PIDS" ]]; then
    echo "receiver.py is not running"
    rm -f receiver.pid
    exit 0
fi

echo "stopping receiver.py (pid(s): $PIDS)"
kill $PIDS 2>/dev/null || true

for _ in $(seq 1 10); do
    sleep 0.5
    pgrep -f "receiver\.py" > /dev/null 2>&1 || break
done

if pgrep -f "receiver\.py" > /dev/null 2>&1; then
    echo "still running after SIGTERM, sending SIGKILL"
    pkill -9 -f "receiver\.py" 2>/dev/null || true
fi

rm -f receiver.pid
echo "stopped"
