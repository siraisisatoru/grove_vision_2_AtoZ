#!/usr/bin/env python3
"""Build datasets/detect/hagrid_gestures/ — 5-class hand-gesture detection from HaGRID.

Run from the repo root, after get_dataset.sh has populated datasets/_hagrid_raw/:

    custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
    python3 custom_dataset_usecases/detect/hagrid_gestures/build_dataset.py

datasets/ is gitignored, so this script is the reproducible source of truth. It
is idempotent: it rebuilds datasets/detect/hagrid_gestures/ from scratch every run and
never touches the raw cache.

WHAT IT DOES
    HaGRID ships images as <gesture>/<uid>.jpg plus one JSON per gesture holding
    every annotation for that gesture's folder:

        "0534147c-...": {
            "bboxes": [[0.380, 0.740, 0.083, 0.091]],   # x_tl y_tl w h, normalised
            "labels": ["call"],
            "leading_hand": "right",
            "user_id": "bb138d5db2..."
        }

    We keep the five GESTURES below, convert the boxes to YOLO's centre format,
    and split train/valid by user_id so the same person never appears in both.

    The raw layout has changed between HaGRID releases (v1 vs HaGRIDv2-512 vs the
    30k subsample), so nothing here hardcodes a directory shape: annotation JSONs
    are found by filename anywhere under the raw cache, and images are located
    through a uid -> path index built by walking it once.
"""

import json
import shutil
import zlib
from collections import defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Five gestures picked to be separable from a 288 px frame by a nano model, i.e.
# distinct in gross silhouette rather than in finger detail:
#
#   palm   five fingers spread, frontal — the largest, most open silhouette
#   fist   closed hand — the most compact silhouette
#   peace  two adjacent fingers up in a V
#   like   thumbs up — hand in profile, one stubby vertical protrusion
#   call   "shaka", thumb and pinky extended — the widest span
#
# Deliberately avoided: palm/stop and peace/peace_inverted and three/three2 are
# near-identical pairs in HaGRID, and like/dislike differ only by rotation.
#
# The weakest pair here is fist vs like (a "like" is a fist plus a thumb, and the
# thumb is a handful of pixels at 288). If the confusion matrix in
# runs/train/confusion_matrix.png shows them mixing, swap `like` for `ok` (a
# thumb-index loop with three fingers up) — change the list here and the GESTURES
# array in get_dataset.sh, re-run both scripts.
#
# Order defines the class indices, which the firmware and stream_view consume
# positionally via LABELS. Do not reorder after training a model.
GESTURES = ["palm", "fist", "peace", "like", "call"]

# Cap per gesture, applied to images that survive filtering, taken in sorted-uid
# order so it is deterministic. Keeps the classes balanced and the run finishable:
# 2000 x 5 = 10k images is ~625 optimizer steps/epoch at batch 16, roughly
# 2 min/epoch on MPS at 288. Raise once a first run proves the recipe.
MAX_PER_CLASS = 2000

# Fraction of images sent to valid/, split on user_id via CRC32 so that (a) it is
# stable across runs and machines — unlike Python's salted hash() — and (b) no
# subject's hands appear in both splits. Subject leakage is the classic way a
# gesture dataset reports val mAP it cannot reproduce on new people.
VALID_EVERY = 10

# What to do with an image containing a box whose label is not one of GESTURES —
# in practice a second, idle hand annotated "no_gesture":
#
#   "skip_image"  drop the whole image. Every hand in every kept image is
#                 labelled, so the model is never penalised for detecting a hand
#                 that the labels call background. This is the default because
#                 an unlabelled hand is the most harmful noise possible for this
#                 task, and HaGRID has plenty of single-hand images to spare.
#   "drop_box"    keep the image, drop just that box. Maximises data, teaches the
#                 model to suppress hands that are not making a target gesture,
#                 and accepts the gradient conflict that comes with it.
#
# The summary prints how many images this costs, so switch on evidence.
UNKNOWN_BOX_POLICY = "skip_image"

# Minimum box area as a fraction of the frame; boxes below it are dropped, and an
# image left with no boxes is dropped entirely. 0.0 keeps everything.
#
# This is the knob that mattered most on coco_val_person: 45% of its person boxes
# were under ~29 px at 288, unresolvable, and acted as label noise that no amount
# of extra data fixed. Hands in HaGRID are much larger in frame than COCO people,
# so the default here is 0.0 — but the summary prints the real area distribution
# for this build. If a large tail sits under ~1% of frame (~29 px at 288), set
# this to 0.01-0.03 and rebuild.
MIN_BOX_AREA = 0.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

repo = Path(__file__).resolve().parents[3]
raw = repo / "datasets" / "_hagrid_raw"
dst = repo / "datasets" / "detect" / "hagrid_gestures"

CLASS_INDEX = {name: i for i, name in enumerate(GESTURES)}


# ── Locating the raw pieces ───────────────────────────────────────────────────

def find_annotations() -> dict[str, dict]:
    """Merge every <gesture>.json found under the raw cache, keyed by gesture.

    HaGRID splits annotations across ann_train_val/, ann_subsample/, or
    annotations/{train,val,test}/ depending on the release. All of them name the
    file after the gesture, so match on the stem and merge whatever turns up —
    duplicate uids across splits collapse harmlessly.
    """
    merged: dict[str, dict] = {g: {} for g in GESTURES}
    found: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(raw.rglob("*.json")):
        if path.stem in CLASS_INDEX:
            found[path.stem].append(path)
            with path.open() as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                merged[path.stem].update(data)

    missing = [g for g in GESTURES if not merged[g]]
    if missing:
        raise SystemExit(
            f"No annotations found for: {', '.join(missing)}\n"
            f"Looked for <gesture>.json anywhere under {raw}\n"
            f"Run custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh first."
        )

    for g in GESTURES:
        srcs = ", ".join(str(p.relative_to(raw)) for p in found[g])
        print(f"  {g:6s} {len(merged[g]):6d} annotations  ({srcs})")
    return merged


def index_images() -> dict[str, Path]:
    """Map uid -> image path by walking the raw cache once.

    Sorted so that if both the subsample and the 512p release are cached, the
    same one always wins and rebuilds stay reproducible.
    """
    index: dict[str, Path] = {}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    if not index:
        raise SystemExit(f"No images found under {raw}. Run get_dataset.sh first.")
    print(f"  indexed {len(index)} images")
    return index


# ── Conversion ────────────────────────────────────────────────────────────────

def to_yolo_lines(entry: dict, areas: list[float]) -> list[str] | None:
    """YOLO label lines for one annotation entry, or None to drop the image.

    HaGRID boxes are [x_top_left, y_top_left, w, h] normalised; YOLO wants
    [cx, cy, w, h] normalised. Appends every kept box's area to `areas` so the
    caller can report the distribution.
    """
    bboxes = entry.get("bboxes") or []
    labels = entry.get("labels") or []
    if len(bboxes) != len(labels) or not bboxes:
        return None

    lines: list[str] = []
    for box, label in zip(bboxes, labels):
        if label not in CLASS_INDEX:
            if UNKNOWN_BOX_POLICY == "skip_image":
                return None
            continue  # "drop_box"
        x, y, w, h = (float(v) for v in box[:4])
        if w <= 0 or h <= 0:
            continue
        if w * h < MIN_BOX_AREA:
            continue
        # Clamp: a few HaGRID boxes run a hair outside the frame, and
        # Ultralytics rejects labels outside [0, 1].
        cx = min(max(x + w / 2, 0.0), 1.0)
        cy = min(max(y + h / 2, 0.0), 1.0)
        w = min(w, 2 * min(cx, 1 - cx))
        h = min(h, 2 * min(cy, 1 - cy))
        if w <= 0 or h <= 0:
            continue
        areas.append(w * h)
        lines.append(f"{CLASS_INDEX[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines or None


def split_for(entry: dict, uid: str) -> str:
    """'train' or 'valid', decided by user_id so subjects never straddle splits."""
    key = entry.get("user_id") or uid
    return "valid" if zlib.crc32(str(key).encode()) % VALID_EVERY == 0 else "train"


def pct(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


# ── Build ─────────────────────────────────────────────────────────────────────

print(f"Raw cache: {raw}")
print("\nAnnotations:")
annotations = find_annotations()
print("\nImages:")
images = index_images()

# Rebuild from scratch so a re-run can never leave stale files behind.
if dst.exists():
    shutil.rmtree(dst)
for split in ("train", "valid"):
    (dst / split / "images").mkdir(parents=True)
    (dst / split / "labels").mkdir(parents=True)
(dst / "test" / "images").mkdir(parents=True)
(dst / "runs").mkdir(parents=True)

kept = {s: defaultdict(int) for s in ("train", "valid")}
boxes = {s: defaultdict(int) for s in ("train", "valid")}
areas: list[float] = []
skipped_missing_image = 0
skipped_unknown_box = 0
skipped_no_box = 0
capped = 0

print("\nConverting:")
for gesture in GESTURES:
    taken = 0
    for uid in sorted(annotations[gesture]):
        if taken >= MAX_PER_CLASS:
            capped += 1
            continue
        entry = annotations[gesture][uid]

        img = images.get(uid)
        if img is None:
            skipped_missing_image += 1
            continue

        before = len(areas)
        lines = to_yolo_lines(entry, areas)
        if lines is None:
            # Distinguish the two rejection reasons for the summary. An entry
            # holding a non-target label is an unknown-box skip; anything else
            # (no boxes, all boxes below MIN_BOX_AREA) is a no-box skip.
            del areas[before:]
            if any(l not in CLASS_INDEX for l in (entry.get("labels") or [])):
                skipped_unknown_box += 1
            else:
                skipped_no_box += 1
            continue

        split = split_for(entry, uid)
        shutil.copy2(img, dst / split / "images" / f"{uid}{img.suffix.lower()}")
        (dst / split / "labels" / f"{uid}.txt").write_text("\n".join(lines) + "\n")
        kept[split][gesture] += 1
        boxes[split][gesture] += len(lines)
        taken += 1

    print(f"  {gesture:6s} train {kept['train'][gesture]:5d}   "
          f"valid {kept['valid'][gesture]:4d}")

names_block = "\n".join(f"  {i}: {g}" for i, g in enumerate(GESTURES))
(dst / "data.yaml").write_text(
    f"path: {dst}\n"
    f"train: {dst / 'train' / 'images'}\n"
    f"val: {dst / 'valid' / 'images'}\n"
    f"test: {dst / 'test' / 'images'}\n"
    f"nc: {len(GESTURES)}\n"
    f"names:\n{names_block}\n"
)

# ── Summary ───────────────────────────────────────────────────────────────────

train_imgs, valid_imgs = sum(kept["train"].values()), sum(kept["valid"].values())
train_box, valid_box = sum(boxes["train"].values()), sum(boxes["valid"].values())

print()
print(f"UNKNOWN_BOX_POLICY      : {UNKNOWN_BOX_POLICY}")
print(f"MIN_BOX_AREA            : {MIN_BOX_AREA}"
      f"  ({'no size filter' if MIN_BOX_AREA == 0 else 'filtering small boxes'})")
print(f"MAX_PER_CLASS           : {MAX_PER_CLASS}")
print(f"Skipped (no image file) : {skipped_missing_image}")
print(f"Skipped (unknown box)   : {skipped_unknown_box}")
print(f"Skipped (no usable box) : {skipped_no_box}")
print(f"Skipped (over cap)      : {capped}")
print(f"train : {train_imgs:5d} images, {train_box:6d} boxes")
print(f"valid : {valid_imgs:5d} images, {valid_box:6d} boxes")

if areas:
    a = sorted(areas)
    print()
    print("Box area as a fraction of frame — the MIN_BOX_AREA evidence:")
    print(f"  median {pct(a, 0.50) * 100:.2f}%   "
          f"p25 {pct(a, 0.25) * 100:.2f}%   p75 {pct(a, 0.75) * 100:.2f}%")
    for t in (0.01, 0.03, 0.05):
        n = sum(1 for v in a if v >= t)
        print(f"  boxes >= {t * 100:.0f}% of frame : {n} ({n / len(a) * 100:.0f}%)"
              f"   <- ~{int((t ** 0.5) * 288)} px at 288")

if train_imgs == 0:
    raise SystemExit("\nNo training images produced — check the skip counts above.")

# Splitting on user_id means the valid split's size depends on how many distinct
# subjects the CRC32 happens to select, not on the image count. Real HaGRID has
# ~35k subjects so this is never tight, but a small or subject-poor raw cache can
# empty the split — and Ultralytics fails deep inside the validator rather than
# here, so catch it now.
if valid_imgs == 0:
    raise SystemExit(
        "\nvalid/ split is empty — every subject landed in train.\n"
        f"Distinct subjects are too few for VALID_EVERY={VALID_EVERY}. Either use\n"
        "the full HaGRID cache, or lower VALID_EVERY."
    )
missing_in_valid = [g for g in GESTURES if kept["valid"][g] == 0]
if missing_in_valid:
    print(f"\nWARNING: no valid/ images for: {', '.join(missing_in_valid)} — "
          f"mAP for those classes will be meaningless.")

print(f"\nWrote {dst}")
print("\nNext:")
print("  cd helper_scripts && ./flow.sh --hagrid_gestures --yolo26 --skip-fetch")
