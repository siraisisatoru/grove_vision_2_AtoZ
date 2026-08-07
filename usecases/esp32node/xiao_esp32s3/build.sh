#!/usr/bin/env bash
# Build and optionally flash the XIAO ESP32-S3 companion node.
#
#   ./build.sh              compile only
#   ./build.sh --upload     compile and flash (auto-detects the port)
#   ./build.sh --upload --port /dev/cu.usbmodem1101
#   ./build.sh --monitor    open the serial monitor after flashing
#
# Requires: arduino-cli with the esp32:esp32 core.
#   arduino-cli core install esp32:esp32
set -euo pipefail

cd "$(dirname "$0")"

# OPI PSRAM is mandatory -- the frame buffers live there.
# USB CDC on boot keeps the monitor on native USB so GPIO43/44 stay free for
# Serial1, which is the link to the Grove board.
FQBN="esp32:esp32:XIAO_ESP32S3:PSRAM=opi,USBMode=hwcdc,CDCOnBoot=cdc"

UPLOAD=0
MONITOR=0
PORT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)  UPLOAD=1; shift ;;
        --monitor) MONITOR=1; shift ;;
        --port)    PORT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f config.h ]]; then
    echo "error: config.h is missing." >&2
    echo "       cp config.h.example config.h  and fill in WiFi + RPi host." >&2
    exit 1
fi

if ! command -v arduino-cli >/dev/null; then
    echo "error: arduino-cli not found (brew install arduino-cli)" >&2
    exit 1
fi

echo "==> compiling for ${FQBN}"
arduino-cli compile --fqbn "$FQBN" .

if [[ $UPLOAD -eq 0 ]]; then
    exit 0
fi

if [[ -z "$PORT" ]]; then
    # XIAO ESP32-S3 in native-USB mode enumerates as a usbmodem on macOS.
    PORT=$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)
    [[ -z "$PORT" ]] && PORT=$(ls /dev/ttyACM* 2>/dev/null | head -1 || true)
fi
if [[ -z "$PORT" ]]; then
    echo "error: no serial port found; pass --port explicitly." >&2
    echo "       If the board is not enumerating, hold BOOT and tap RESET." >&2
    exit 1
fi

echo "==> uploading to ${PORT}"
arduino-cli upload --fqbn "$FQBN" --port "$PORT" .

if [[ $MONITOR -eq 1 ]]; then
    echo "==> monitoring ${PORT} (ctrl-C to exit)"
    arduino-cli monitor --port "$PORT" --config baudrate=115200
fi
