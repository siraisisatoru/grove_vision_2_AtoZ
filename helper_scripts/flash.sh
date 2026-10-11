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
    echo "Usage: $0 --default | --yolo26 | --output | --face"
    echo ""
    echo "  --default   Flash the Grove Vision AI V2 stock firmware"
    echo "  --yolo26    Flash the YOLO v2.6 support firmware"
    echo "  --output    Flash a locally built output.img"
    echo "  --face      Flash the face-recognition firmware + both face models"
    exit 1
}

FACE=0
FACE_IMG="$FW_DIR/face_recognition.img"
SCRFD_MODEL="$FW_DIR/face_models/scrfd_500m_kps_int8_vela.tflite"
MFN_MODEL="$FW_DIR/face_models/mobilefacenet_128d_int8_vela.tflite"
case "${1:-}" in
    --default) IMG="$DEFAULT_IMG" ;;
    --yolo26)  IMG="$YOLO26_IMG" ;;
    --output)  IMG="$OUTPUT_IMG" ;;
    --face)    IMG="$FACE_IMG"; FACE=1 ;;
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
if [ "$FACE" = "1" ]; then
  # Face recognition also needs the SCRFD + MobileFaceNet models. They sit at
  # fixed flash addresses (not the A/B slots), so flash them once in pass 1; the
  # firmware still needs both slots. flash_model.py flashes fw + N models in one
  # pass via a software reset (no BOOT button).
  for m in "$SCRFD_MODEL" "$MFN_MODEL"; do
    [ -f "$m" ] || { echo "Error: face model not found: $m"; exit 1; }
  done
  echo ""
  echo "── Flash pass 1/2 (firmware + SCRFD + MobileFaceNet) ──"
  "$PYTHON" "$SCRIPT_DIR/flash_model.py" --port "$PORT" --software-reset \
    --firmware "$IMG" \
    --model-at "$SCRFD_MODEL" 0x200000 \
    --model-at "$MFN_MODEL" 0x400000
  echo ""
  echo "── Flash pass 2/2 (firmware, other A/B slot) ──"
  "$PYTHON" "$SCRIPT_DIR/flash_model.py" --port "$PORT" --software-reset --firmware "$IMG"
else
  for pass in 1 2; do
    echo ""
    echo "── Flash pass $pass/2 (dual A/B slot — both must match) ──"
    "$PYTHON" "$SCRIPT_DIR/flash_fw.py" --port "$PORT" --firmware "$IMG"
  done
fi
