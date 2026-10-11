#!/usr/bin/env python3
"""Enroll a face into the gallery from a single board capture, with live preview.

Run with the helper_scripts venv:
    .venv/bin/python3 enroll.py

Shows the camera's live view (face box + landmarks + current recognition).
Press SPACE to capture the frame on screen; the preview freezes so you see
exactly what was captured, then the terminal asks for the person's name and
the face goes into the gallery. The enrolled embedding is the one the BOARD
computed for that frame — board-live embeddings match board-live recognition
at ~0.9 cosine, whereas embeddings from external photos land ~0.45-0.5
(measured: the camera/lighting domain gap, not image quality) — so enrollment
must come from the device. After enrolling, the live view reloads the gallery
and should label you by name right away.

Usage (with the helper_scripts venv):
    .venv/bin/python3 enroll.py                        # preview, SPACE to capture, name prompted
    .venv/bin/python3 enroll.py --name siraisisatoru   # same, but the name is preset
    .venv/bin/python3 enroll.py --list                 # show enrolled names
    .venv/bin/python3 enroll.py --remove siraisisatoru # remove an entry

Keys in the preview window:
    SPACE  capture the current frame (needs a visible face)
    Q/Esc  quit

Requires the face firmware (flash.sh --face); the board streams FACE_RESULT
events from boot with bbox, landmarks and a 128-D embedding per face.
"""

import argparse
import os
import sqlite3
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grove_vision_ai.device import GroveVisionAI, find_port  # noqa: E402
from stream_view import decode_embedding, draw_faces, load_gallery, recognize  # noqa: E402

HERE = Path(__file__).resolve().parent
# Shared output location: gallery db + captured enrollment frames both live in
# output/face_recognition/ (gitignored). stream_view.py --task face reads the
# same db by default, so enrollment and the viewer share one gallery.
OUTPUT_DIR = HERE / "output" / "face_recognition"
DEFAULT_GALLERY = OUTPUT_DIR / "faces.db"


# ── Gallery writes (schema-compatible with face_recognition_2/lib/gallery.py:
# faces(id, name UNIQUE, embedding BLOB float32 L2-normalized, sample_count,
# created_at); inlined like stream_view's reader so there is no dependency on
# the experimental package) ──────────────────────────────────────────────────

def _open_db(path):
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            sample_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    return conn


def gallery_add(path, name, embedding, sample_count=1):
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    conn = _open_db(path)
    conn.execute(
        "INSERT OR REPLACE INTO faces (name, embedding, sample_count) VALUES (?, ?, ?)",
        (name, emb.tobytes(), sample_count),
    )
    conn.commit()
    conn.close()


def gallery_remove(path, name):
    conn = _open_db(path)
    cur = conn.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


# ── Capture UI ───────────────────────────────────────────────────────────────

def best_face(faces):
    """Highest-confidence face that carries an embedding, or None."""
    best, best_conf = None, -1.0
    for f in faces or []:
        if decode_embedding(f) is None:
            continue
        c = float(f.get("confidence", 0))
        if c > best_conf:
            best, best_conf = f, c
    return best


def put_hud(frame, text, color=(255, 255, 255)):
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


def prompt_name_async(preset):
    """Read the name from the terminal on a thread so the cv2 window keeps
    pumping events (a blocked event loop greys the window on macOS).
    -> (thread, result dict with 'name' once done)."""
    result = {}

    def _read():
        if preset:
            result["name"] = preset
            return
        try:
            result["name"] = input("Name for this face (empty = retake): ").strip()
        except EOFError:
            result["name"] = ""

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    return t, result


def run_capture(args):
    port = args.port or find_port()
    if not port:
        sys.exit("No Grove Vision AI V2 found. Pass --port explicitly.")

    gallery = load_gallery(args.gallery)
    print(f"gallery: {len(gallery)} enrolled {[n for n, _ in gallery]}  ({args.gallery})")
    print("Keys: SPACE capture   Q/Esc quit\n")

    latest = {"clean": None, "faces": [], "lock": threading.Lock(), "seq": 0}
    # Recent embeddings for --avg: (monotonic seq, embedding) newest-last.
    recent = deque(maxlen=max(args.avg, 1))

    def on_frame(msg):
        data = msg.get("data", {})
        img_b64 = data.get("image", "")
        if not img_b64:
            return
        import base64
        import io
        from PIL import Image
        try:
            pil = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        except Exception:
            return
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        faces = data.get("faces", [])
        f = best_face(faces)
        with latest["lock"]:
            latest["clean"] = bgr
            latest["faces"] = faces
            latest["seq"] += 1
            if f is not None:
                recent.append(decode_embedding(f))

    dev = GroveVisionAI()
    dev.set_raw_callback(lambda line: None)  # silence non-JSON boot lines
    dev.set_callback(on_frame)
    # Face firmware free-runs from boot; boot_wait>0 would race the live stream.
    dev.connect(port, boot_wait=0)

    window = f"Enroll — Grove Vision AI V2 [{port}]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state = "live"          # live | naming
    frozen = None           # frozen display frame (scaled, with overlays)
    cap = None              # dict: clean frame + face + embedding at capture
    name_thread = None
    name_result = None
    flash_until = 0.0       # "enrolled X" banner deadline
    flash_text = ""

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            with latest["lock"]:
                clean = latest["clean"]
                faces = list(latest["faces"])

            if state == "live":
                if clean is None:
                    time.sleep(0.01)
                    continue
                disp = cv2.resize(
                    clean, (clean.shape[1] * args.scale, clean.shape[0] * args.scale),
                    interpolation=cv2.INTER_NEAREST)
                draw_faces(disp, faces, gallery, args.scale, args.recognize_threshold)
                f = best_face(faces)
                if time.time() < flash_until:
                    put_hud(disp, flash_text, (0, 220, 0))
                elif f is None:
                    put_hud(disp, "no face — SPACE disabled", (0, 60, 220))
                else:
                    put_hud(disp, f"face {float(f.get('confidence', 0)):.2f} — SPACE to capture")
                cv2.imshow(window, disp)

                if key == ord(" "):
                    if f is None:
                        print("no face in the current frame — not captured")
                        continue
                    emb = decode_embedding(f)
                    if args.avg > 1 and len(recent) > 1:
                        embs = [e / np.linalg.norm(e) for e in recent if e is not None
                                and np.linalg.norm(e) > 0]
                        emb = np.mean(embs, axis=0)
                        n_avg = len(embs)
                    else:
                        n_avg = 1
                    cap = {"clean": clean.copy(), "faces": faces, "face": f,
                           "embedding": emb, "n_avg": n_avg}
                    frozen = cv2.resize(
                        cap["clean"],
                        (clean.shape[1] * args.scale, clean.shape[0] * args.scale),
                        interpolation=cv2.INTER_NEAREST)
                    draw_faces(frozen, [f], [], args.scale, args.recognize_threshold)
                    put_hud(frozen, "captured — type the name in the terminal")
                    conf = float(f.get("confidence", 0))
                    avg_note = f", embedding averaged over {n_avg} frames" if n_avg > 1 else ""
                    print(f"captured (face confidence {conf:.2f}{avg_note})")
                    name_thread, name_result = prompt_name_async(args.name)
                    state = "naming"

            elif state == "naming":
                cv2.imshow(window, frozen)
                if name_thread.is_alive():
                    continue
                name = (name_result.get("name") or "").strip()
                if not name:
                    print("no name given — back to live view (retake)")
                    state = "live"
                    continue

                # Similarity to existing entries BEFORE saving — surfaces
                # duplicates ("bob 0.91" while enrolling "alice" means the
                # gallery will confuse the two).
                match_name, match_sim = recognize(cap["embedding"], gallery,
                                                  args.recognize_threshold)
                if gallery:
                    print(f"closest existing entry: "
                          f"{match_name or 'none above threshold'} ({match_sim:.2f})")

                gallery_add(args.gallery, name, cap["embedding"], cap["n_avg"])

                os.makedirs(args.save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(args.save_dir, f"enroll_{name}_{ts}.png")
                cv2.imwrite(img_path, cap["clean"])

                gallery = load_gallery(args.gallery)  # live view now knows the name
                print(f"enrolled '{name}'  ({len(gallery)} in gallery)  capture: {img_path}")
                print("the live view should now label this face — Q to quit, "
                      "SPACE to enroll another\n")
                flash_text = f"enrolled '{name}'"
                flash_until = time.time() + 3.0
                if args.name:      # preset name is one-shot: prompt next time
                    args.name = None
                state = "live"
    finally:
        dev.disconnect()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="name to enroll (prompted after capture if omitted)")
    ap.add_argument("--port", default=None, help="serial port (auto-detected)")
    ap.add_argument("--gallery", default=str(DEFAULT_GALLERY.resolve()),
                    help="faces.db path (default: the shared face_recognition_2 gallery)")
    ap.add_argument("--scale", type=int, default=2, help="display upscale (default 2)")
    ap.add_argument("--recognize-threshold", type=float, default=0.6,
                    help="cosine threshold for the live recognition overlay (default 0.6)")
    ap.add_argument("--avg", type=int, default=1, metavar="N",
                    help="average the embeddings of the last N face frames at capture "
                         "(default 1 = exactly the frame on screen; ~5 smooths the "
                         "~0.1 frame-to-frame jitter)")
    ap.add_argument("--save-dir", default=str(OUTPUT_DIR),
                    help="where captured enrollment frames are saved as PNG "
                         "(default: output/face_recognition/, beside the db)")
    ap.add_argument("--list", action="store_true", help="list enrolled names and exit")
    ap.add_argument("--remove", metavar="NAME", help="remove an enrolled name and exit")
    args = ap.parse_args()

    if args.list:
        rows = load_gallery(args.gallery)
        if not rows:
            print("gallery is empty")
        for n, e in rows:
            print(f"{n}  (dim={e.size})")
        return
    if args.remove:
        print("removed" if gallery_remove(args.gallery, args.remove)
              else f"no entry named {args.remove!r}")
        return

    run_capture(args)


if __name__ == "__main__":
    main()
