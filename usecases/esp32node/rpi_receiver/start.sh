#!/usr/bin/env bash
# Check for the venv/dependencies receiver.py needs, installing them if
# missing, then start it in the background so it survives the SSH session
# that launched it (setsid, not just nohup -- see esp_node/README.md §10.3).
#
# Usage: ./start.sh [-- extra receiver.py args, e.g. --labels classes.txt]
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -f "receiver\.py" > /dev/null 2>&1; then
    echo "receiver.py already running (pid $(pgrep -f 'receiver\.py' | tr '\n' ' '))" >&2
    echo "run ./stop.sh first" >&2
    exit 1
fi

if [[ ! -d venv ]]; then
    echo "==> no venv found, creating one"
    python3 -m venv venv
fi

if ./venv/bin/python3 -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "==> dependencies already satisfied"
else
    # opencv-python-headless is a ~40 MB wheel. If this Pi's only route to
    # PyPI is through a phone-tethered hotspot (macOS marks those
    # "constrained" and can throttle/cut large transfers), the download can
    # come back truncated -- pip still "succeeds" but cv2 then fails to
    # import with a confusing ImportError (e.g. libGL.so.1 missing) that
    # looks like a missing system library but isn't. --no-cache-dir forces a
    # real re-download on retry instead of reusing the corrupted copy.
    installed=0
    for attempt in 1 2 3; do
        echo "==> dependencies missing, installing from requirements.txt (attempt $attempt/3)"
        if [[ $attempt -eq 1 ]]; then
            ./venv/bin/pip install -q -r requirements.txt
        else
            ./venv/bin/pip install -q --no-cache-dir --force-reinstall -r requirements.txt
        fi
        if ./venv/bin/python3 -c "import cv2, numpy" >/dev/null 2>&1; then
            installed=1
            break
        fi
    done
    if [[ $installed -ne 1 ]]; then
        echo "ERROR: cv2/numpy still fail to import after 3 install attempts." >&2
        echo "       This usually means the download of the ~40 MB opencv wheel keeps" >&2
        echo "       getting truncated over a slow/constrained network path (e.g. a" >&2
        echo "       phone hotspot), not a missing system package. Check connectivity" >&2
        echo "       and retry. The actual import error:" >&2
        ./venv/bin/python3 -c "import cv2, numpy"
        exit 1
    fi
fi

echo "==> starting receiver.py"
setsid nohup ./venv/bin/python3 -u receiver.py --mjpeg-port 8080 "$@" > receiver.log 2>&1 < /dev/null &
sleep 1

PID=$(pgrep -f "receiver\.py" | head -1 || true)
if [[ -n "$PID" ]]; then
    echo "$PID" > receiver.pid
    echo "started: pid $PID, log: receiver.log, mjpeg: http://$(hostname -I | awk '{print $1}'):8080/"
else
    echo "ERROR: receiver.py did not stay up, check receiver.log" >&2
    tail -20 receiver.log >&2 || true
    exit 1
fi
