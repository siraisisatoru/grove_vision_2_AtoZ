#!/usr/bin/env python3
"""Build datasets/coco128_person/ — a single-class person dataset from coco128.

Run from the repo root:
    python3 custom_dataset_usecases/coco128_person/build_dataset.py

datasets/ is gitignored, so this script is the reproducible source of truth for
the dataset layout. It is idempotent: re-running rebuilds from scratch.

Source: datasets/coco_yolo26/ (the coco128 asset, laid out by that use case's
readme). If it is missing, this script says so rather than guessing.

What it does:
  * keeps only images containing at least one COCO class 0 (person) box
  * strips every non-person box from the labels; person stays class 0, so the
    resulting dataset is nc=1
  * splits deterministically (every 5th image to valid) so re-runs and the INT8
    calibration set are stable

Why a person-only relabel rather than training on all 80 classes: at IMGSZ=288
the frame is heavily pixelated, and 80 classes spread confidence thin over it.
One class concentrates the whole capacity of a nano model on the target.
"""

import shutil
import sys
from pathlib import Path

PERSON_CLASS = "0"   # COCO class index for person, and the only class we keep
VALID_EVERY = 5      # every Nth person image goes to valid/ (deterministic)

repo = Path(__file__).resolve().parents[2]
src = repo / "datasets" / "coco_yolo26"
dst = repo / "datasets" / "coco128_person"

src_images = src / "train" / "images"
src_labels = src / "train" / "labels"

if not src_images.is_dir() or not src_labels.is_dir():
    sys.exit(
        f"Source coco128 not found at {src}.\n"
        f"Run the one-time setup in custom_dataset_usecases/coco_yolo26/readme.md "
        f"first — this dataset is derived from it."
    )


def person_lines(label_file: Path) -> list[str]:
    """Person-only label lines, or [] if the image has no person."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == PERSON_CLASS:
            out.append(" ".join(parts))
    return out


# Rebuild from scratch so a re-run can never leave stale files behind.
if dst.exists():
    shutil.rmtree(dst)
for split in ("train", "valid"):
    (dst / split / "images").mkdir(parents=True)
    (dst / split / "labels").mkdir(parents=True)
(dst / "test" / "images").mkdir(parents=True)
(dst / "runs").mkdir(parents=True)

kept = {"train": 0, "valid": 0}
instances = {"train": 0, "valid": 0}
skipped = 0
person_idx = 0

for img in sorted(src_images.iterdir()):
    if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    label = src_labels / f"{img.stem}.txt"
    if not label.exists():
        skipped += 1
        continue
    lines = person_lines(label)
    if not lines:
        skipped += 1
        continue

    split = "valid" if person_idx % VALID_EVERY == 0 else "train"
    person_idx += 1

    shutil.copy2(img, dst / split / "images" / img.name)
    (dst / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
    kept[split] += 1
    instances[split] += len(lines)

(dst / "data.yaml").write_text(
    f"path: {dst}\n"
    f"train: {dst / 'train' / 'images'}\n"
    f"val: {dst / 'valid' / 'images'}\n"
    f"test: {dst / 'test' / 'images'}\n"
    f"nc: 1\n"
    f"names: [person]\n"
)

print(f"Source images scanned : {kept['train'] + kept['valid'] + skipped}")
print(f"Skipped (no person)   : {skipped}")
print(f"train : {kept['train']:3d} images, {instances['train']:4d} person boxes")
print(f"valid : {kept['valid']:3d} images, {instances['valid']:4d} person boxes")
print(f"\nWrote {dst}")
print("\nNOTE: this is a very small dataset. Expect overfitting; see readme.md "
      "for the larger COCO person subset alternative.")
