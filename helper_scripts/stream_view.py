#!/usr/bin/env python3
"""
Live camera viewer for Grove Vision AI V2.

Displays bounding boxes from the device's YOLO inference stream.
Press Q or Esc to quit.

Usage:
    python stream_view.py [--port /dev/cu.usbmodem...] [--labels Deer] [--score-threshold 25] [--scale 3]

For a multi-class COCO model:
    python stream_view.py --labels "person,bicycle,car,motorcycle,..."
"""
import argparse
import base64
import io
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from grove_vision_ai.device import (GroveVisionAI, find_port, COCO_LABELS,
                                    model_type_name)

SCALE = 2

COLORS = [
    (0, 220, 0),    # green
    (220, 60, 0),   # blue-ish
    (0, 60, 220),   # red-orange
    (220, 0, 220),  # purple
    (0, 220, 220),  # yellow
]


def draw_boxes(frame_bgr, boxes, class_names, scale, coord_scale_x=1.0, coord_scale_y=1.0):
    """Draw boxes on an upscaled frame.

    boxes: list of [cx, cy, bw, bh, score_pct, class_id] (center-based, model-input space)
    coord_scale_x/y: multiply coords to convert from model space → source image space
    """
    fh, fw = frame_bgr.shape[:2]
    nc = len(class_names)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    pad = 4

    for box in boxes:
        cx, cy, bw, bh, score_pct, cls = box
        cls = int(cls)
        if cls >= nc:
            continue
        name = class_names[cls]
        label = f"{name} {score_pct}%"
        color = COLORS[cls % len(COLORS)]

        x1 = int((cx - bw / 2) * coord_scale_x * scale)
        y1 = int((cy - bh / 2) * coord_scale_y * scale)
        x2 = int((cx + bw / 2) * coord_scale_x * scale)
        y2 = int((cy + bh / 2) * coord_scale_y * scale)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_h = th + baseline + pad * 2
        label_w = tw + pad * 2

        # Place label above the box; flip below if it would clip the top edge
        if y1 - label_h >= 0:
            bg_y1, bg_y2 = y1 - label_h, y1
        else:
            bg_y1, bg_y2 = y1, y1 + label_h

        # Clamp to right edge
        bg_x1 = max(0, min(x1, fw - label_w))
        bg_x2 = bg_x1 + label_w

        cv2.rectangle(frame_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        # baseline at bg_y2 - baseline - pad puts pad pixels below descenders and above ascenders
        cv2.putText(frame_bgr, label, (bg_x1 + pad, bg_y2 - baseline - pad),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Live viewer for Grove Vision AI V2")
    ap.add_argument("--port", default=None, help="Serial port (auto-detected)")
    ap.add_argument("--scale", type=int, default=SCALE,
                    help=f"Display upscale factor (default {SCALE})")
    ap.add_argument("--score-threshold", type=int, default=25,
                    help="Min detection score 0-100 (default 25)")
    ap.add_argument("--labels", default=",".join(COCO_LABELS),
                    help="Comma-separated class names (default: 80-class COCO)")
    ap.add_argument("--algo", type=int, default=0,
                    help="AT+ALGO value before streaming (default 0 = decoded-box for no_post=False model)")
    ap.add_argument("--model-size", type=int, default=192, metavar="PX",
                    help="Model input resolution in pixels (default 192). Used to scale "
                         "coords from model space → camera space when SSCMA metadata is absent.")
    ap.add_argument("--debug", action="store_true",
                    help="Print raw box values and image size for the first detection frame, then exit.")
    ap.add_argument("--log-detections", action="store_true",
                    help="Print a timestamped line to stdout for every detection.")
    args = ap.parse_args()

    class_names = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not class_names:
        class_names = COCO_LABELS

    port = args.port or find_port()
    if not port:
        sys.exit("No Grove Vision AI V2 found. Pass --port explicitly.")

    window = f"Grove Vision AI V2  [{port}]"
    print(f"Connecting to {port}  (scale={args.scale}x, threshold={args.score_threshold}%)")
    print("Press Q or Esc to quit.\n")

    def log_detections(boxes):
        if not (args.log_detections and boxes):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for b in boxes:
            cx, cy, bw, bh, score_pct, cls = b
            cls = int(cls)
            name = class_names[cls] if cls < nc else f"cls{cls}"
            print(f"[{ts}] {name} score={score_pct}% "
                  f"box=(cx={cx}, cy={cy}, w={bw}, h={bh})", flush=True)

    dev = GroveVisionAI()
    dev.set_raw_callback(lambda line: print(f"  [boot] {line}", flush=True))
    dev.connect(port)

    # Ensure the model at 0x400000 is loaded — device may default to a different slot on boot.
    models = dev.list_models()
    slot1 = next((m for m in models if m.get("address") == 0x400000), None)
    if slot1:
        dev.load_model(slot1["id"])

    # ALGO=0: decoded-box NMS. kris-himax fork exports with no_post=False by default —
    # DFL decode is baked into the TFLite graph; firmware only applies NMS.
    if not dev.set_algo(args.algo):
        print(f"Warning: AT+ALGO={args.algo} not accepted — detections may not work.")

    # AT+TSCORE is not supported in firmware 2025.01.02 — ignore failures silently.
    dev.set_score_threshold(args.score_threshold)

    nc = len(class_names)
    # "seq" increments once per frame actually received from the device. The display
    # loop below spins far faster than frames arrive (it is paced by cv2.waitKey(1)),
    # so counting loop iterations measures the redraw rate, not the device — it read
    # ~70 while the board was delivering ~10, and kept reading ~70 even when the board
    # had stopped sending altogether.
    latest = {"frame": None, "boxes": [], "perf": [0, 0, 0], "seq": 0,
              "lock": threading.Lock()}

    def on_frame(msg):
        data    = msg.get("data", {})
        img_b64 = data.get("image", "")
        raw_boxes = data.get("boxes", [])
        perf    = data.get("perf", [0, 0, 0])
        if not img_b64:
            return
        try:
            pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        except Exception:
            return

        src_w, src_h = pil_img.size
        # SSCMA maps box coords to sensor image space before returning them — the firmware
        # already accounts for the model→sensor transform, so no rescaling is needed here.
        # (Using src_w/model_size caused progressive drift: error grew with distance from origin.)
        cx_scale = 1.0
        cy_scale = 1.0

        if args.debug and raw_boxes:
            print("\n── DEBUG: first detection frame ──────────────────")
            print(f"  image size from device : {src_w} × {src_h} px")
            print(f"  --model-size           : {args.model_size} px")
            print(f"  cx_scale / cy_scale    : {cx_scale:.3f} / {cy_scale:.3f}")
            print(f"  raw boxes (first 3)    : {raw_boxes[:3]}")
            print("  box fields expected    : [cx, cy, w, h, score%, class]")
            print("──────────────────────────────────────────────────")
            import os; os._exit(0)

        boxes = raw_boxes
        log_detections(boxes)

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (src_w * args.scale, src_h * args.scale),
                         interpolation=cv2.INTER_NEAREST)
        draw_boxes(bgr, boxes, class_names, args.scale, cx_scale, cy_scale)

        with latest["lock"]:
            latest["frame"] = bgr
            latest["boxes"] = boxes
            latest["perf"]  = perf
            latest["seq"]  += 1

    dev.set_callback(on_frame)
    ok = dev.start_invoke(n_times=-1, result_only=0)
    if not ok:
        dev.disconnect()
        sys.exit("Failed to start stream. Is a model loaded?")

    # Architecture reported BY THE BOARD, not inferred from a CLI flag: the firmware
    # tells us which decoder ModelFactory bound to the flashed model. If this says
    # YOLO11 when you flashed a YOLO26 model (or vice versa), the export shape did not
    # match that decoder's isValid() and it silently fell through to another one.
    algo_name = model_type_name(dev.algorithm_type)
    print(f"Device reports algorithm: {algo_name} (ma_model_type_t={dev.algorithm_type})")
    window = f"{window}  —  {algo_name}"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # fps_seq: the "seq" value at the last measurement, so fps counts device frames.
    fps_t, fps_seq, fps = time.time(), 0, 0.0

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            with latest["lock"]:
                frame = latest["frame"]
                boxes = latest["boxes"]
                perf  = latest["perf"]
                seq   = latest["seq"]

            # Recompute once a second from frames the DEVICE delivered, regardless of
            # how many times this loop redrew them. Updated before the `frame is None`
            # check so a stalled board decays to 0.0 instead of freezing at its last
            # healthy value.
            now = time.time()
            if now - fps_t >= 1.0:
                fps = (seq - fps_seq) / (now - fps_t)
                fps_t, fps_seq = now, seq

            if frame is None:
                time.sleep(0.01)
                continue

            disp = frame.copy()
            inf_ms = perf[1] if len(perf) > 1 else 0
            if boxes:
                det_parts = []
                for b in boxes[:3]:
                    cls = 0 if nc == 1 else int(b[5])
                    name = class_names[cls] if cls < nc else f"cls{cls}"
                    det_parts.append(f"{name}:{b[4]}%")
                det_str = ", ".join(det_parts)
            else:
                det_str = "no detections"
            hud = f"{algo_name}  |  FPS {fps:.1f}  |  NPU {inf_ms}ms  |  {det_str}"
            cv2.putText(disp, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, disp)

    finally:
        dev.stop_stream()
        dev.disconnect()
        cv2.destroyAllWindows()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
