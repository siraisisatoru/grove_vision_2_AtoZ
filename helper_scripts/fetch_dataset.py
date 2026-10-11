#!/usr/bin/env python3
"""
Download or generate the dataset for a given use-case.

Reads from environment (set by flow.sh via the use-case .env):
  ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
  DATASET_DIR   — absolute path to the download destination
  LABELS        — comma-separated class names (used by synthetic mode)
  IMGSZ         — image size for synthetic generation (default 160)
  SMOKE         — set to 1 to use synthetic data instead of Roboflow
  TASK          — 'detect' (default) or 'classify'. classify expects/produces an
                   ImageFolder layout (train/<class>/*.jpg, ...) instead of YOLO
                   bbox labels + data.yaml — see Roboflow's "folder" export format
                   (ROBOFLOW_FORMAT=folder), which is what a Classification project
                   downloads as.

Usage (via flow.sh):
  flow.sh handles calling this script automatically.

Direct usage:
  python fetch_dataset.py [--source roboflow|synthetic]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def _write_data_yaml(root: Path, names: list[str]) -> Path:
    """Write a data.yaml with absolute paths so ultralytics never misresolves them."""
    yaml_path = root / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    yaml_path.write_text(
        f"path: {root}\n"
        f"train: {root / 'train' / 'images'}\n"
        f"val: {root / 'valid' / 'images'}\n"
        f"test: {root / 'test' / 'images'}\n"
        f"nc: {len(names)}\n"
        f"names:\n{names_block}\n"
    )
    return yaml_path


def _parse_names_from_yaml(root: Path) -> list[str] | None:
    """Extract class names from a Roboflow-downloaded data.yaml."""
    p = root / "data.yaml"
    if not p.exists():
        return None
    text = p.read_text()
    if "names:" not in text:
        return None
    after = text.split("names:", 1)[1]
    line0 = after.splitlines()[0]
    names: list[str] = []
    if "[" in line0:
        inner = line0[line0.find("[") + 1 : line0.find("]")]
        names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
    else:
        for line in after.splitlines():
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() and ":" in s:
                names.append(s.split(":", 1)[1].strip())
            elif s.startswith("-"):
                names.append(s[1:].strip().strip("'\""))
            else:
                break
    return names or None


def _classify_classes(root: Path) -> list[str]:
    """Class names for an ImageFolder-layout classification dataset.

    Sorted, because Ultralytics' classify dataloader assigns class indices by
    sorted folder name — get_labels.py's --classify mode must match this order
    exactly or downstream label display will be off-by-index.
    """
    train_dir = root / "train"
    if not train_dir.is_dir():
        return []
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def download_roboflow(out_dir: Path, task: str) -> Path:
    required = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"ERROR: missing env vars: {', '.join(missing)}\n"
            "Set them in the use-case .env or use --source synthetic for an offline test."
        )

    api_key   = os.environ["ROBOFLOW_API_KEY"].strip()
    workspace = os.environ["ROBOFLOW_WORKSPACE"]
    project   = os.environ["ROBOFLOW_PROJECT"]
    version   = int(os.environ["ROBOFLOW_VERSION"])

    default_fmt = "folder" if task == "classify" else "yolov8"
    fmt = os.environ.get("ROBOFLOW_FORMAT", default_fmt)

    from roboflow import Roboflow
    print(f"Connecting to Roboflow workspace={workspace} project={project} v{version} format={fmt}")
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
          .project(project)
          .version(version)
          .download(fmt, location=str(out_dir), overwrite=True)
    )

    root = Path(dataset.location)

    if task == "classify":
        # "folder" format downloads directly as train/<class>/*.jpg — no bbox
        # labels, no data.yaml. Nothing to write; just confirm the shape.
        names = _classify_classes(root)
        if not (root / "train").is_dir() or not names:
            sys.exit(
                f"ERROR: expected an ImageFolder layout (train/<class>/*.jpg) at {root} "
                f"after downloading format={fmt!r}. Got: "
                f"{sorted(p.name for p in root.iterdir())}"
            )
        print(f"Dataset ready: {root}  classes={names}")
        return root

    names = _parse_names_from_yaml(root) or ["object"]
    yaml_path = _write_data_yaml(root, names)
    print(f"Dataset ready: {root}  classes={names}")
    return yaml_path


def _draw_blob(seed: int, img_size: int):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (img_size, img_size), (
        rng.randint(60, 120), rng.randint(90, 150), rng.randint(50, 100),
    ))
    draw = ImageDraw.Draw(img)
    boxes = []
    for _ in range(rng.randint(1, 2)):
        bw = rng.randint(30, 80)
        bh = rng.randint(40, 90)
        cx = rng.randint(bw // 2, img_size - bw // 2)
        cy = rng.randint(bh // 2, img_size - bh // 2)
        brown = (rng.randint(90, 150), rng.randint(60, 100), rng.randint(30, 60))
        draw.ellipse([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2], fill=brown)
        draw.ellipse([cx - 7, cy - bh // 2 - 12, cx + 7, cy - bh // 2 + 4], fill=brown)
        boxes.append((0, cx / img_size, cy / img_size, bw / img_size, bh / img_size))
    return img, boxes


def generate_synthetic(out_dir: Path) -> Path:
    labels_env = os.environ.get("LABELS", "object")
    names = [n.strip() for n in labels_env.split(",") if n.strip()]
    img_size = int(os.environ.get("IMGSZ", 160))

    splits = {"train": 20, "valid": 6, "test": 6}
    idx = 0
    for split, n in splits.items():
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            img, boxes = _draw_blob(seed=idx, img_size=img_size)
            stem = f"synthetic_{idx:04d}"
            img.save(img_dir / f"{stem}.jpg", quality=90)
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n"
            )
            idx += 1

    yaml_path = _write_data_yaml(out_dir, names)
    print(f"Synthetic dataset: {idx} images under {out_dir}  classes={names}")
    return yaml_path


def _draw_class_image(seed: int, img_size: int, class_idx: int, n_classes: int):
    """A distinct, deterministic-per-class solid color + shape — enough for a
    classifier to learn to discriminate without needing real data."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    # Spread base hues evenly across classes so classes are visually distinct
    # regardless of how many there are.
    hue = class_idx / max(n_classes, 1)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
    base = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.new("RGB", (img_size, img_size), base)
    draw = ImageDraw.Draw(img)
    # A random-position shape on top of the base color adds enough per-image
    # variation that a classifier can't just memorize a flat color.
    shapes = rng.randint(1, 3)
    for _ in range(shapes):
        sw = rng.randint(img_size // 6, img_size // 3)
        cx = rng.randint(sw, img_size - sw)
        cy = rng.randint(sw, img_size - sw)
        shade = tuple(max(0, min(255, c + rng.randint(-40, 40))) for c in base)
        draw.ellipse([cx - sw // 2, cy - sw // 2, cx + sw // 2, cy + sw // 2], fill=shade)
    return img


def generate_synthetic_classify(out_dir: Path) -> Path:
    """ImageFolder-layout synthetic dataset: train/<class>/*.jpg, etc."""
    labels_env = os.environ.get("LABELS", "class_a,class_b")
    names = [n.strip() for n in labels_env.split(",") if n.strip()]
    if len(names) < 2:
        names = ["class_a", "class_b"]
    img_size = int(os.environ.get("IMGSZ", 160))

    splits = {"train": 20, "valid": 6, "test": 6}
    idx = 0
    for split, n in splits.items():
        for ci, name in enumerate(names):
            class_dir = out_dir / split / name
            class_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(n):
                img = _draw_class_image(seed=idx, img_size=img_size, class_idx=ci, n_classes=len(names))
                img.save(class_dir / f"synthetic_{idx:04d}.jpg", quality=90)
                idx += 1

    print(f"Synthetic classification dataset: {idx} images under {out_dir}  classes={names}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", choices=["roboflow", "synthetic"], default=None,
        help="Data source. Defaults to 'synthetic' when SMOKE=1, otherwise 'roboflow'.",
    )
    ap.add_argument("--out", default=None, help="Output directory (overrides DATASET_DIR env var)")
    args = ap.parse_args()

    out_dir = Path(args.out or os.environ.get("DATASET_DIR", "dataset")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task = os.environ.get("TASK", "detect")

    source = args.source
    if source is None:
        source = "synthetic" if os.environ.get("SMOKE") == "1" else "roboflow"

    if source == "roboflow":
        result_path = download_roboflow(out_dir, task)
    elif task == "classify":
        print("SMOKE/synthetic mode — generating placeholder dataset (no Roboflow key needed)")
        result_path = generate_synthetic_classify(out_dir)
    else:
        print("SMOKE/synthetic mode — generating placeholder dataset (no Roboflow key needed)")
        result_path = generate_synthetic(out_dir)

    if task == "classify":
        print(f"DATASET_ROOT={result_path}")
    else:
        print(f"DATA_YAML={result_path}")


if __name__ == "__main__":
    main()
