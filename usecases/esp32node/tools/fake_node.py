#!/usr/bin/env python3
"""Stand in for the ESP32 so the Pi side can be developed without hardware.

Emits the same wire format xiao_esp32s3.ino does, with either synthetic frames or
frames replayed from real AT event lines captured off the board.

    # synthetic: moving boxes over a gradient
    python3 fake_node.py --host 127.0.0.1 --port 5010 --fps 5

    # replay: feed it a file of raw AT lines (one JSON object per line, as the
    # firmware emits them) captured with `screen`/`cat /dev/tty... > capture.txt`
    python3 fake_node.py --replay capture.txt --loop
"""

import argparse
import base64
import json
import math
import socket
import struct
import sys
import time

HEADER = struct.Struct("<4sII")


def send_message(sock, meta_json_bytes, jpeg_bytes):
    sock.sendall(HEADER.pack(b"SSC1", len(meta_json_bytes), len(jpeg_bytes)))
    sock.sendall(meta_json_bytes)
    if jpeg_bytes:
        sock.sendall(jpeg_bytes)


def synth_frame(n, width, height):
    """Builds one (meta, jpeg) pair with two boxes orbiting the centre."""
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(30, 120, width, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(20, 90, height, dtype=np.uint8)[:, None]
    cv2.putText(img, f"synthetic #{n}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    boxes = []
    for i in range(2):
        phase = n / 12.0 + i * math.pi
        cx = int(width / 2 + math.cos(phase) * width * 0.28)
        cy = int(height / 2 + math.sin(phase) * height * 0.28)
        bw, bh = width // 5, height // 4
        # [x, y, w, h, score, target] with x,y as the CENTRE, matching the firmware.
        boxes.append([cx, cy, bw, bh, 70 + (n + i * 7) % 30, i])

    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    meta = {
        "type": 1,
        "name": "INVOKE",
        "code": 0,
        "data": {
            "count": n,
            "image": "",
            "boxes": boxes,
            "perf": [3, 41, 2],
            "rotation": 0,
            "width": width,
            "height": height,
        },
    }
    return json.dumps(meta, separators=(",", ":")).encode(), encoded.tobytes()


def replay_lines(path):
    """Yields (meta, jpeg) from captured AT event lines, splitting out the image."""
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.strip(b"\r\n\x00 \t")
            if not line.startswith(b"{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("name") != "INVOKE" or msg.get("type") != 1:
                continue
            data = msg.setdefault("data", {})
            b64 = data.get("image") or ""
            data["image"] = ""
            jpeg = base64.b64decode(b64, validate=False) if b64 else b""
            yield json.dumps(msg, separators=(",", ":")).encode(), jpeg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5010)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--results-only", action="store_true",
                    help="omit the JPEG, as AT+INVOKE=-1,1 does")
    ap.add_argument("--replay", default=None, help="file of captured AT event lines")
    ap.add_argument("--loop", action="store_true", help="repeat the replay file")
    args = ap.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[fake] connected to {args.host}:{args.port}")

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    sent = 0
    try:
        while True:
            if args.replay:
                source = replay_lines(args.replay)
            else:
                source = (synth_frame(i, args.width, args.height)
                          for i in range(sent, sent + 10_000_000))

            for meta, jpeg in source:
                send_message(sock, meta, b"" if args.results_only else jpeg)
                sent += 1
                if args.frames and sent >= args.frames:
                    print(f"[fake] sent {sent} frames")
                    return 0
                if period:
                    time.sleep(period)

            if not (args.replay and args.loop):
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    print(f"[fake] sent {sent} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
