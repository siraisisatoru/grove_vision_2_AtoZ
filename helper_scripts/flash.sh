#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "Error: venv not found at $SCRIPT_DIR/.venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi
FW_DIR="$SCRIPT_DIR/grove_vision_fw"

DEFAULT_IMG="$FW_DIR/grove_vision_ai_v2_20250102.img"
YOLO26_IMG="$FW_DIR/yolo26_support.img"
OUTPUT_IMG="$FW_DIR/output.img"

usage() {
    echo "Usage: $0 --default | --yolo26"
    echo ""
    echo "  --default   Flash the Grove Vision AI V2 stock firmware"
    echo "  --yolo26    Flash the YOLO v2.6 support firmware"
    echo "  --output    Flash the YOLO v2.6 support firmware"
    exit 1
}

case "${1:-}" in
    --default) IMG="$DEFAULT_IMG" ;;
    --yolo26)  IMG="$YOLO26_IMG" ;;
    --output)  IMG="$OUTPUT_IMG" ;;
    *)         usage ;;
esac

if [ ! -f "$IMG" ]; then
    echo "Error: firmware image not found: $IMG"
    exit 1
fi

# Auto-detect port using the project's device module (CH343 VID/PID 0x1A86:0x55D3, with glob fallback)
PORT=$("$PYTHON" -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR')
from grove_vision_ai.device import find_port
p = find_port()
print(p or '')
" 2>/dev/null) || true

if [ -z "$PORT" ]; then
    echo "Error: Could not auto-detect the board's serial port."
    echo "Make sure the device is connected, then retry."
    echo "To list ports manually: python3 flash_fw.py --list-ports"
    exit 1
fi

echo "Detected port: $PORT"
echo "Firmware:      $(basename "$IMG")"

# Dual firmware A/B slot gotcha: the bootloader alternates the active slot on
# reset independent of what was just flashed, so a single flash can leave the
# *other* slot holding stale (possibly broken/incompatible) firmware — the
# board then boots that stale slot with no indication anything is wrong
# (symptom: total AT-command silence, no boot banner at all). Flashing twice
# in a row guarantees both slots hold the same image.
for pass in 1 2; do
  echo ""
  echo "── Flash pass $pass/2 (dual A/B slot — both must match) ──"
  "$PYTHON" "$SCRIPT_DIR/flash_fw.py" --port "$PORT" --firmware "$IMG"
done
