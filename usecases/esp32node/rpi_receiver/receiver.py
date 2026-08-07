#!/usr/bin/env python3
"""Receive detections + JPEG frames from the XIAO ESP32-S3 bridge and display them.

Listens for the TCP stream produced by esp_node/xiao_esp32s3. Wire format,
little-endian:

    "SSC1" | u32 json_len | u32 jpeg_len | json bytes | jpeg bytes

`json` is the firmware's AT event line with the base64 image value emptied, so
boxes/perf/width/height/rotation arrive exactly as the HX6538 produced them.
`jpeg_len` is 0 in results-only mode, in which case boxes are drawn on a blank
canvas sized from data.width/data.height.

    python3 receiver.py --mjpeg-port 8080     # view at http://<pi>:8080/
    python3 receiver.py --display             # local OpenCV window
    python3 receiver.py --save-dir ./captures # write annotated frames to disk
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MAGIC = b"SSC1"
HEADER = struct.Struct("<4sII")

# Distinct hues that survive being drawn over arbitrary camera imagery, in BGR.
PALETTE = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (153, 153, 153),
]


class LatestFrame:
    """Single-slot mailbox: viewers always get the newest frame, never a backlog."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._seq = 0
        self._updated_at = None
        self._cond = threading.Condition(self._lock)

    def put(self, jpeg_bytes):
        with self._cond:
            self._jpeg = jpeg_bytes
            self._seq += 1
            self._updated_at = time.time()
            self._cond.notify_all()

    def get(self, last_seq, timeout=5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def status(self):
        """Real, server-side freshness -- independent of how any particular
        browser handles multipart <img> load/error events, which turned out
        not to be reliable enough to drive a "live" indicator on their own."""
        with self._lock:
            seq, updated_at = self._seq, self._updated_at
        age = (time.time() - updated_at) if updated_at is not None else None
        return {"frames": seq, "age_sec": age}


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(sock, max_json, max_jpeg):
    """Reads one framed message. Returns (meta_dict, jpeg_bytes|None) or None on EOF."""
    head = recv_exact(sock, HEADER.size)
    if head is None:
        return None
    magic, json_len, jpeg_len = HEADER.unpack(head)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r} -- stream desynchronised")
    if json_len > max_json or jpeg_len > max_jpeg:
        raise ValueError(f"implausible lengths json={json_len} jpeg={jpeg_len}")

    raw_json = recv_exact(sock, json_len)
    if raw_json is None:
        return None
    jpeg = None
    if jpeg_len:
        jpeg = recv_exact(sock, jpeg_len)
        if jpeg is None:
            return None

    try:
        msg = json.loads(raw_json.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed json: {exc}") from None
    return msg, jpeg


def annotate(msg, jpeg, labels, fps=None):
    """Decodes and draws. Returns the annotated BGR image."""
    data = msg.get("data", {}) or {}
    src_w = int(data.get("width") or 0)
    src_h = int(data.get("height") or 0)

    if jpeg:
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
    else:
        # Results-only mode: draw on a blank canvas the size the boxes were
        # scaled to, so geometry still reads correctly.
        img = np.zeros((src_h or 480, src_w or 640, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    # Boxes are in the raw frame's pixel space; the JPEG is a separate capture
    # and is not guaranteed to be the same size. Always rescale.
    sx = w / src_w if src_w else 1.0
    sy = h / src_h if src_h else 1.0

    for box in data.get("boxes") or []:
        if len(box) < 6:
            continue
        cx, cy, bw, bh, score, target = box[:6]
        # x,y are the box CENTRE (see refactor_required.hpp), not a corner.
        x1 = int(round((cx - bw / 2) * sx))
        y1 = int(round((cy - bh / 2) * sy))
        x2 = int(round((cx + bw / 2) * sx))
        y2 = int(round((cy + bh / 2) * sy))

        color = PALETTE[int(target) % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = labels[int(target)] if 0 <= int(target) < len(labels) else str(target)
        caption = f"{name} {int(score)}%"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, caption, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    perf = data.get("perf") or []
    hud_parts = []
    if perf:
        hud_parts.append("pre {} / inf {} / post {} ms".format(*(list(perf) + [0, 0, 0])[:3]))
    if fps is not None:
        # Actual end-to-end rate the receiver sees frames arrive at, not the
        # firmware's own inference-only timing above -- this also reflects
        # link/decode overhead, so it's normally lower than 1000/inf.
        hud_parts.append(f"{fps:.1f} fps")
    if hud_parts:
        cv2.putText(img, " | ".join(hud_parts), (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img


# Chrome (and recent Safari) no longer render a directly-navigated
# multipart/x-mixed-replace response -- the tab just spins forever even
# though frames are arriving over the wire. They still render it fine inside
# an <img> tag, though, so "/" serves a real HTML document with one, and the
# raw stream moves to /stream.mjpg.
#
# The status line does NOT use img.onload/onerror: those events turned out to
# be unreliable for a multipart-replace <img> in practice (onload may never
# fire again after the first part, even while frames keep arriving and
# rendering correctly -- there's no DOM signal that fires per-frame). Instead
# it polls the /stats JSON endpoint, which reports real server-side data: how
# long ago the mailbox last got a frame from the ESP32. That's independent of
# any particular browser's handling of the image events.
INDEX_HTML = b"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Grove Vision AI V2</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; }
  .wrap {
    display: flex; align-items: center; justify-content: center;
    height: 100%;
  }
  img { max-width: 100vw; max-height: 100vh; }
  #status {
    position: fixed; top: 8px; left: 8px;
    font: 12px monospace; color: #9f9; background: #0008; padding: 2px 6px;
  }
  #status.stale { color: #f99; }
</style>
</head>
<body>
<div id="status">connecting&hellip;</div>
<div class="wrap"><img id="feed" src="/stream.mjpg"></div>
<script>
  var img = document.getElementById('feed');
  // NOT named `status`: that collides with the browser's built-in global
  // `window.status` (legacy status-bar text, a string-typed property). A
  // plain `var status = document.getElementById(...)` at script scope gets
  // silently coerced to a string by that native setter -- every subsequent
  // `status.textContent = ...` then writes to a throwaway boxed string and
  // never touches the real element, with no error thrown anywhere. Confirmed
  // via `typeof status` reporting "string" instead of "object" when this bug
  // was in place.
  var statusEl = document.getElementById('status');

  img.onerror = function() {
    setTimeout(function() {
      img.src = '/stream.mjpg?t=' + Date.now();
    }, 1000);
  };

  function poll() {
    fetch('/stats', {cache: 'no-store'})
      .then(function(r) { return r.json(); })
      .then(function(s) {
        if (s.age_sec === null) {
          statusEl.textContent = 'waiting for first frame...';
          statusEl.className = 'stale';
        } else if (s.age_sec > 3) {
          statusEl.textContent = 'stale -- last frame ' + s.age_sec.toFixed(1) + 's ago';
          statusEl.className = 'stale';
        } else {
          statusEl.textContent = 'live (' + s.frames + ' frames)';
          statusEl.className = '';
        }
      })
      .catch(function() {
        statusEl.textContent = 'lost contact with receiver';
        statusEl.className = 'stale';
      });
  }
  poll();
  setInterval(poll, 1000);
</script>
</body>
</html>
"""


class MJPEGHandler(BaseHTTPRequestHandler):
    mailbox = None  # set by serve_mjpeg

    def log_message(self, *args):
        pass  # the receiver prints its own stats

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return
        if self.path == "/stats":
            body = json.dumps(self.mailbox.status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path not in ("/stream", "/stream.mjpg") and not self.path.startswith("/stream.mjpg?"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frameboundary")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpeg, seq = self.mailbox.get(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frameboundary\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve_mjpeg(mailbox, port):
    MJPEGHandler.mailbox = mailbox
    server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[mjpeg] http://0.0.0.0:{port}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5010,
                    help="must match RPI_PORT in the ESP32 config.h")
    ap.add_argument("--display", action="store_true", help="local OpenCV window")
    ap.add_argument("--mjpeg-port", type=int, default=None,
                    help="also serve an MJPEG stream on this port")
    ap.add_argument("--save-dir", default=None, help="write annotated JPEGs here")
    ap.add_argument("--labels", default=None,
                    help="text file, one class name per line, indexed by target id")
    ap.add_argument("--max-json", type=int, default=1 << 20)
    ap.add_argument("--max-jpeg", type=int, default=4 << 20)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    labels = []
    if args.labels:
        with open(args.labels) as fh:
            labels = [line.strip() for line in fh if line.strip()]

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    mailbox = LatestFrame()
    if args.mjpeg_port:
        serve_mjpeg(mailbox, args.mjpeg_port)

    # Only one camera node is ever expected, but it reconnects on its own
    # schedule (reboot, WiFi drop, retry) and the old socket rarely gets a
    # clean FIN when that happens -- the peer just vanishes. A single
    # blocking accept()/recv() loop would get stuck on that dead socket
    # forever and never notice the new connection sitting unaccepted behind
    # it. So: accept in a loop, and whenever a new connection arrives, close
    # whatever the previous one was (unblocking its handler) before handing
    # the new one to a fresh handler thread. Newest connection always wins.
    current_lock = threading.Lock()
    current_conn = [None]

    def handle_connection(conn, addr):
        print(f"[recv] connected from {addr[0]}:{addr[1]}", flush=True)
        frames = 0
        bytes_in = 0
        t0 = time.time()
        t_report = t0
        last_frame_ts = None
        fps_ema = None
        try:
            while True:
                result = read_message(conn, args.max_json, args.max_jpeg)
                if result is None:
                    print(f"[recv] {addr[0]}:{addr[1]} peer closed", flush=True)
                    break
                msg, jpeg = result
                frames += 1
                bytes_in += len(jpeg or b"")

                # Smoothed actual arrival rate, drawn into the HUD -- distinct
                # from the firmware's own pre/inf/post timing, since it also
                # captures link and JPEG-decode overhead between frames.
                frame_ts = time.time()
                if last_frame_ts is not None and frame_ts > last_frame_ts:
                    inst_fps = 1.0 / (frame_ts - last_frame_ts)
                    fps_ema = inst_fps if fps_ema is None else fps_ema * 0.8 + inst_fps * 0.2
                last_frame_ts = frame_ts

                img = annotate(msg, jpeg, labels, fps=fps_ema)
                if img is None:
                    continue

                ok, encoded = cv2.imencode(".jpg", img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    mailbox.put(encoded.tobytes())
                    if args.save_dir:
                        count = (msg.get("data") or {}).get("count", frames)
                        with open(f"{args.save_dir}/frame_{count:08d}.jpg", "wb") as fh:
                            fh.write(encoded.tobytes())

                if args.display:
                    cv2.imshow("grove vision ai v2", img)
                    cv2.waitKey(1)

                now = time.time()
                if not args.quiet and now - t_report >= 5.0:
                    dt = now - t_report
                    print(f"[recv] {frames} frames, "
                          f"{frames / (now - t0):.2f} fps avg, "
                          f"{bytes_in / dt / 1024:.1f} kB/s jpeg", flush=True)
                    bytes_in = 0
                    t_report = now
        except (ValueError, ConnectionResetError, OSError) as exc:
            print(f"[recv] dropping connection: {exc}", flush=True)
        finally:
            conn.close()
            with current_lock:
                if current_conn[0] is conn:
                    current_conn[0] = None

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen(2)
    print(f"[recv] listening on {args.bind}:{args.port}", flush=True)

    while True:
        conn, addr = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with current_lock:
            stale = current_conn[0]
            current_conn[0] = conn
        if stale is not None:
            print("[recv] new connection arrived, dropping the previous one",
                  flush=True)
            stale.close()  # unblocks the old handler's recv(), it exits and cleans up
        threading.Thread(target=handle_connection, args=(conn, addr),
                          daemon=True).start()

    return 0


if __name__ == "__main__":
    sys.exit(main())
