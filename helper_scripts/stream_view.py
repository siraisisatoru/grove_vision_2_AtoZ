#!/usr/bin/env python3
"""
Live camera viewer for Grove Vision AI V2.

Displays bounding boxes from the device's YOLO inference stream (--task detect,
the default), or top-class overlays from a classifier stream (--task classify).
Press Q or Esc to quit.

Usage:
    python stream_view.py [--port /dev/cu.usbmodem...] [--labels Deer] [--score-threshold 25] [--scale 3]

For a multi-class COCO model:
    python stream_view.py --labels "person,bicycle,car,motorcycle,..."

For a classifier:
    python stream_view.py --task classify --labels "cat,dog"
"""
import argparse
import base64
import io
import os
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


def cls_score_pct(score):
    """Class score -> percent, robust to the firmware's two unit conventions.

    ma::model::Classifier::postprocess() dequantizes the output and then
    applies a heuristic: quant scale < 0.1 means the tensor really is 0..1
    probabilities (our int8 Softmax exports: scale = 1/256), anything else is
    assumed to be percent and divided by 100. A model whose output scale
    lands on the wrong side of that heuristic arrives here in the other unit
    — the giveaway is nonsense like "8700%" or every class pinned at "0%".
    Values <= 1.0 are treated as fractions, anything larger as percent
    already, so both conventions display correctly.
    """
    return score * 100.0 if score <= 1.0 else float(score)


def draw_classes(frame_bgr, classes, class_names, top_n=3, threshold_pct=0):
    """Overlay the top-N classification results as text (no boxes to draw —
    a classifier has no spatial localization).

    classes: list of [score, class_id], already sorted descending by
    ma::model::Classifier::postprocess() on the device (score units vary —
    see cls_score_pct).

    threshold_pct: the same 0-100 threshold sent to the device via AT+TSCORE.
    The firmware already drops classes below its own threshold, so normally
    everything arriving here passes — but older firmware ignores AT+TSCORE
    (keeping its 0.5 default or stored value), so the client-side check makes
    the "unknown" semantics hold regardless of what the device filtered. When
    nothing clears the threshold, an explicit grey "unknown" is drawn rather
    than showing nothing: for recognition use cases (face_recognition), "a
    face I don't know" is a result, not an absence of one.
    """
    nc = len(class_names)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    pad = 4
    y = 40

    entries = [(cls_score_pct(score), int(cls)) for score, cls in classes[:top_n]]
    if not entries or entries[0][0] < threshold_pct:
        top = f" (best {entries[0][0]:.0f}%)" if entries else ""
        label = f"unknown{top}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame_bgr, (4, y - th - pad), (4 + tw + pad * 2, y + baseline + pad), (128, 128, 128), -1)
        cv2.putText(frame_bgr, label, (4 + pad, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + baseline + pad * 2 + 4

    for pct, cls in entries:
        name = class_names[cls] if cls < nc else f"class_{cls}"
        label = f"{name}: {pct:.0f}%"
        color = (0, 220, 0) if pct >= threshold_pct else (160, 160, 160)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame_bgr, (4, y - th - pad), (4 + tw + pad * 2, y + baseline + pad), color, -1)
        cv2.putText(frame_bgr, label, (4 + pad, y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += th + baseline + pad * 2 + 4


# ── Face recognition (--task face) ─────────────────────────────────────────────
# The face firmware (tflm_face_embedding) free-runs and streams FACE_RESULT
# events whose data.faces[] carry a bbox, 5 landmarks and a 128-D embedding.
# Recognition matches each embedding against an enrolled gallery (the same
# sqlite faces.db the enrollment tools write); with no gallery it degrades to a
# detection-only view. The gallery reader is inlined here so this viewer has no
# hard dependency on the experimental face-recognition package.
def load_gallery(path):
    """-> list of (name, L2-normalized 128-D embedding). Empty if absent."""
    import sqlite3
    out = []
    try:
        conn = sqlite3.connect(path)
        rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
        conn.close()
    except Exception:
        return out
    for name, blob in rows:
        e = np.frombuffer(blob, dtype=np.float32)
        n = np.linalg.norm(e)
        if e.size and n > 0:
            out.append((name, e / n))
    return out


def recognize(emb, gallery, threshold):
    """-> (name or None, best_similarity). name is None if below threshold."""
    e = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(e)
    if n == 0 or not gallery:
        return None, 0.0
    e = e / n
    best_name, best = None, -1.0
    for name, g in gallery:
        s = float(np.dot(e, g))
        if s > best:
            best_name, best = name, s
    return (best_name if best >= threshold else None), best


def decode_embedding(face):
    emb = face.get("embedding")
    if isinstance(emb, str) and emb and face.get("embedding_format") == "float32_base64":
        return np.frombuffer(base64.b64decode(emb + "=" * ((4 - len(emb) % 4) % 4)), dtype=np.float32)
    if isinstance(emb, list) and emb:
        return np.array(emb, dtype=np.float32)
    return None


def draw_faces(frame_bgr, faces, gallery, scale, threshold):
    """Draw each face's box + landmarks, labelled with the recognized name
    (green), 'unknown' (red, when a gallery exists) or the raw confidence
    (green, detection-only when there is no gallery)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f in faces:
        bx = f.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = (int(v * scale) for v in bx)
        emb = decode_embedding(f)
        name, sim = (recognize(emb, gallery, threshold) if emb is not None else (None, 0.0))
        if name:
            color, label = (0, 220, 0), f"{name} {sim:.2f}"
        elif gallery:
            color, label = (0, 60, 220), f"unknown {sim:.2f}"
        else:
            color, label = (0, 220, 0), f"face {f.get('confidence', 0):.2f}"
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        ly = y - 6 if y - 6 > 12 else y + h + 18
        cv2.putText(frame_bgr, label, (x + 3, ly), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (x + 3, ly), font, 0.6, color, 1, cv2.LINE_AA)
        for lm in f.get("landmarks") or []:
            cv2.circle(frame_bgr, (int(lm[0] * scale), int(lm[1] * scale)), 2, color, -1)


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
    ap.add_argument("--task", choices=["detect", "classify", "face"], default="detect",
                    help="detect (default): draw boxes from data.boxes. classify: "
                         "overlay top-N results from data.classes. face: draw faces "
                         "from data.faces (bbox + landmarks) and recognize each "
                         "embedding against --gallery (face-recognition firmware).")
    ap.add_argument("--gallery", default=None,
                    help="Path to the enrolled-faces sqlite db for --task face "
                         "(default: output/face_recognition/faces.db — enroll.py's "
                         "gallery — falling back to the legacy "
                         "../experimental/face_recognition_2/faces.db). "
                         "With no gallery, faces are shown as detections only.")
    ap.add_argument("--recognize-threshold", type=float, default=0.6,
                    help="Cosine-similarity threshold for a face match (--task face, default 0.6)")
    ap.add_argument("--model-size", type=int, default=192, metavar="PX",
                    help="Model input resolution in pixels (default 192). Used to scale "
                         "coords from model space → camera space when SSCMA metadata is absent.")
    ap.add_argument("--debug", action="store_true",
                    help="Print raw box values and image size for the first detection frame, then exit.")
    ap.add_argument("--log-detections", action="store_true",
                    help="Print a timestamped line to stdout for every detection.")
    ap.add_argument("--save-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "output", "face_recognition"),
                    metavar="DIR",
                    help="Where the S key saves clean camera frames (default "
                         "output/face_recognition/, beside the enrollment db). "
                         "Frames are saved at the sensor's "
                         "native stream resolution with no overlays — intended "
                         "as enrollment sources: frames captured by the SAME "
                         "sensor/optics the model will see at inference close "
                         "the LFW-photo-vs-board-camera domain gap that makes "
                         "photo-enrolled people score poorly on-device.")
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

    nc = len(class_names)
    is_classify = args.task == "classify"
    is_face = args.task == "face"

    # Face recognition uses the free-running tflm_face_embedding firmware, which
    # streams FACE_RESULT from boot and has no AT command parser — so skip the
    # AT model/algo/threshold setup (each would just time out) and load the
    # enrollment gallery instead.
    gallery = []
    if is_face:
        gpath = args.gallery
        if gpath is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            for _default in (
                os.path.join(_here, "output", "face_recognition", "faces.db"),
                os.path.join(_here, "..", "experimental", "face_recognition_2", "faces.db"),
            ):
                if os.path.isfile(_default):
                    gpath = _default
                    break
        if gpath and os.path.isfile(gpath):
            gallery = load_gallery(gpath)
            print(f"gallery: {len(gallery)} enrolled {[n for n, _ in gallery]}  ({gpath})")
        else:
            print("gallery: none found — showing face detections only "
                  "(pass --gallery PATH to recognize names).")

    dev = GroveVisionAI()
    dev.set_raw_callback(lambda line: print(f"  [boot] {line}", flush=True))
    # The face firmware streams from boot; connect()'s post-boot reset_input_buffer
    # (boot_wait > 0) would race the reader on the live stream and kill it, so open
    # with boot_wait=0 for face. The AT firmwares need the boot wait after the
    # RTS-induced reset, so keep the default there.
    dev.connect(port, boot_wait=(0 if is_face else 4.0))

    if not is_face:
        # Ensure the model at 0x400000 is loaded — device may default to a different slot on boot.
        models = dev.list_models()
        slot1 = next((m for m in models if m.get("address") == 0x400000), None)
        if slot1:
            dev.load_model(slot1["id"])

        # ALGO=0: decoded-box NMS. kris-himax fork exports with no_post=False by default —
        # DFL decode is baked into the TFLite graph; firmware only applies NMS.
        if not dev.set_algo(args.algo):
            print(f"Warning: AT+ALGO={args.algo} not accepted — detections may not work.")

        # AT+TSCORE (0-100, firmware divides by 100 and applies it via
        # setConfig(MA_MODEL_CFG_OPT_THRESHOLD) at invoke setup) IS supported by
        # firmware built from the current sscma-example-we2 tree (yolo26_support.img);
        # the older stock 2025.01.02 image ignores it and keeps its stored/default
        # threshold (0.5 for classifiers) — failures stay silent for that case, and
        # draw_classes() re-applies the threshold client-side so the behavior is
        # consistent either way.
        dev.set_score_threshold(args.score_threshold)

    def log_classes(classes):
        if not (args.log_detections and classes):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for score, cls in classes:
            cls = int(cls)
            name = class_names[cls] if cls < nc else f"class_{cls}"
            print(f"[{ts}] {name} score={cls_score_pct(score):.0f}%", flush=True)

    # "seq" increments once per frame actually received from the device. The display
    # loop below spins far faster than frames arrive (it is paced by cv2.waitKey(1)),
    # so counting loop iterations measures the redraw rate, not the device — it read
    # ~70 while the board was delivering ~10, and kept reading ~70 even when the board
    # had stopped sending altogether.
    # "clean" holds the frame BEFORE overlays are drawn (native resolution) —
    # the S key must save what the sensor delivered, not the HUD-decorated
    # upscale, or the saved "enrollment photos" would carry green label boxes.
    latest = {"frame": None, "clean": None, "boxes": [], "classes": [], "faces": [],
              "perf": [0, 0, 0], "seq": 0, "lock": threading.Lock()}

    def on_frame(msg):
        data    = msg.get("data", {})
        img_b64 = data.get("image", "")
        raw_boxes = data.get("boxes", [])
        raw_classes = data.get("classes", [])
        raw_faces = data.get("faces", []) if is_face else []
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

        if args.debug and (raw_boxes or raw_classes):
            print("\n── DEBUG: first inference frame ──────────────────")
            print(f"  image size from device : {src_w} × {src_h} px")
            print(f"  --model-size           : {args.model_size} px")
            print(f"  cx_scale / cy_scale    : {cx_scale:.3f} / {cy_scale:.3f}")
            if is_classify:
                print(f"  raw classes            : {raw_classes}")
                print("  class fields expected  : [score(0-1), class_id]")
            else:
                print(f"  raw boxes (first 3)    : {raw_boxes[:3]}")
                print("  box fields expected    : [cx, cy, w, h, score%, class]")
            print("──────────────────────────────────────────────────")
            os._exit(0)

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        clean = bgr.copy()  # native size, no overlays — see latest["clean"]
        bgr = cv2.resize(bgr, (src_w * args.scale, src_h * args.scale),
                         interpolation=cv2.INTER_NEAREST)

        if is_face:
            draw_faces(bgr, raw_faces, gallery, args.scale, args.recognize_threshold)
        elif is_classify:
            log_classes(raw_classes)
            draw_classes(bgr, raw_classes, class_names,
                         threshold_pct=args.score_threshold)
        else:
            log_detections(raw_boxes)
            draw_boxes(bgr, raw_boxes, class_names, args.scale, cx_scale, cy_scale)

        with latest["lock"]:
            latest["frame"] = bgr
            latest["clean"] = clean
            latest["boxes"] = raw_boxes
            latest["classes"] = raw_classes
            latest["faces"] = raw_faces
            latest["perf"]  = perf
            latest["seq"]  += 1

    dev.set_callback(on_frame)
    if is_face:
        # The face firmware streams FACE_RESULT continuously from boot; there is
        # no AT+INVOKE to start (and calling it would just time out).
        print("Face recognition: reading the free-running FACE_RESULT stream (no AT+INVOKE).")
        algo_name = "Face Recognition"
    else:
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

    print(f"Keys: Q/Esc quit  |  S save clean frame to {args.save_dir}/")
    n_saved = 0
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            with latest["lock"]:
                frame   = latest["frame"]
                clean   = latest["clean"]
                boxes   = latest["boxes"]
                classes = latest["classes"]
                perf    = latest["perf"]
                seq     = latest["seq"]

            if key in (ord("s"), ord("S")) and clean is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = os.path.join(args.save_dir, f"frame_{ts}.png")
                cv2.imwrite(path, clean)
                n_saved += 1
                print(f"  saved {path}  ({n_saved} this session)", flush=True)

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
            if is_face:
                with latest["lock"]:
                    faces = latest["faces"]
                if faces:
                    parts = []
                    for f in faces[:3]:
                        emb = decode_embedding(f)
                        name, sim = (recognize(emb, gallery, args.recognize_threshold)
                                     if emb is not None else (None, 0.0))
                        parts.append(f"{name} {sim:.2f}" if name
                                     else (f"unknown {sim:.2f}" if gallery
                                           else f"face {f.get('confidence', 0):.2f}"))
                    det_str = ", ".join(parts)
                else:
                    det_str = "no face"
            elif is_classify:
                if classes:
                    det_parts = []
                    for score, cls in classes[:3]:
                        cls = int(cls)
                        name = class_names[cls] if cls < nc else f"class_{cls}"
                        det_parts.append(f"{name}:{cls_score_pct(score):.0f}%")
                    det_str = ", ".join(det_parts)
                    if cls_score_pct(classes[0][0]) < args.score_threshold:
                        det_str = f"unknown  ({det_str})"
                else:
                    det_str = "unknown (no class above device threshold)"
            elif boxes:
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
