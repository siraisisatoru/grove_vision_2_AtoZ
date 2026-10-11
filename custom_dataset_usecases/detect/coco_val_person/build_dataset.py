#!/usr/bin/env python3
"""Build datasets/detect/coco_val_person/ — person-only detection set from COCO val2017.

Run from the repo root:
    python3 custom_dataset_usecases/detect/coco_val_person/build_dataset.py

datasets/ is gitignored, so this script is the reproducible source of truth.
It is idempotent: re-running rebuilds the dataset from the cached raw download.

Raw sources are cached in datasets/_coco_raw/ and downloaded only if absent:
  * coco2017labels.zip  (~49 MB) — Ultralytics' YOLO-format COCO labels
  * val2017.zip         (~778 MB) — the COCO val2017 images

Why val2017 and not train2017: val2017 is a 1 GB download against train2017's
19 GB. "val" is only COCO's split name, not a role — nothing stops us training
on it, we just give up the standard benchmark split (irrelevant here, since we
never compare against published COCO numbers). It yields 2693 person images,
~56x more than the coco128_person experiment. train2017 (~64k person images) is
the upgrade if this still underfits.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PERSON_CLASS = "0"   # COCO class index for person, and the only class we keep
VALID_EVERY = 10     # every Nth person image goes to valid/ (deterministic)

# Minimum box area as a fraction of frame, for dropping targets too small to
# survive the downscale to IMGSZ. 0.0 = keep every box (current setting).
#
# Measured on this set: median person box is 1.41% of frame, and 45% of boxes are
# under 1% — roughly 29 px at 288, which the model cannot resolve and which acts
# as label noise. Set this to 0.03 (38% of boxes survive) or 0.05 (30% survive)
# to train only on resolvable targets. Images left with no boxes after filtering
# are dropped entirely rather than kept as background.
MIN_BOX_AREA = 0.0

repo = Path(__file__).resolve().parents[3]
raw = repo / "datasets" / "_coco_raw"
dst = repo / "datasets" / "detect" / "coco_val_person"

DOWNLOADS = [
    ("coco2017labels.zip",
     "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"),
    ("val2017.zip",
     "http://images.cocodataset.org/zips/val2017.zip"),
]


def ensure_raw() -> tuple[Path, Path]:
    """Download + unpack the raw COCO pieces if needed. Returns (images, labels)."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS:
        archive = raw / name
        if not archive.exists():
            print(f"Downloading {name} ...")
            subprocess.run(["curl", "-L", "--progress-bar", "-o", str(archive), url],
                           check=True)
        else:
            print(f"Using cached {name} ({archive.stat().st_size / 1e6:.0f} MB)")

    labels = raw / "coco" / "labels" / "val2017"
    if not labels.is_dir():
        print("Unpacking coco2017labels.zip ...")
        with zipfile.ZipFile(raw / "coco2017labels.zip") as z:
            z.extractall(raw)

    # val2017.zip unpacks to a top-level val2017/ directory of .jpg files.
    images = raw / "val2017"
    if not images.is_dir():
        print("Unpacking val2017.zip ...")
        with zipfile.ZipFile(raw / "val2017.zip") as z:
            z.extractall(raw)

    if not labels.is_dir():
        sys.exit(f"Labels still missing at {labels} after unpacking.")
    if not images.is_dir():
        sys.exit(f"Images still missing at {images} after unpacking.")
    return images, labels


def person_lines(label_file: Path) -> list[str]:
    """Person-only label lines passing MIN_BOX_AREA, or [] if none qualify."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != PERSON_CLASS:
            continue
        if MIN_BOX_AREA > 0.0:
            # YOLO format: cls cx cy w h, all normalised to the frame
            if float(parts[3]) * float(parts[4]) < MIN_BOX_AREA:
                continue
        out.append(" ".join(parts))
    return out


src_images, src_labels = ensure_raw()

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
skipped_no_person = 0
skipped_no_label = 0
person_idx = 0

for img in sorted(src_images.iterdir()):
    if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    label = src_labels / f"{img.stem}.txt"
    if not label.exists():
        # COCO ships images with no annotations at all; they carry no label file.
        skipped_no_label += 1
        continue
    lines = person_lines(label)
    if not lines:
        skipped_no_person += 1
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

print()
print(f"Skipped (no label file) : {skipped_no_label}")
print(f"Skipped (no person)     : {skipped_no_person}")
print(f"MIN_BOX_AREA            : {MIN_BOX_AREA}  ({'no size filter' if MIN_BOX_AREA == 0 else 'filtering small boxes'})")
print(f"train : {kept['train']:5d} images, {instances['train']:6d} person boxes")
print(f"valid : {kept['valid']:5d} images, {instances['valid']:6d} person boxes")
print(f"\nWrote {dst}")
