#!/usr/bin/env python3
"""
Download or generate the dataset for a given use-case.

Reads from environment (set by flow.sh via the use-case .env):
  ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
  DATASET_DIR   — absolute path to the download destination
  LABELS        — comma-separated class names (used by synthetic mode)
  IMGSZ         — image size for synthetic generation (default 160)
  SMOKE         — set to 1 to use synthetic data instead of Roboflow

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


def download_roboflow(out_dir: Path) -> Path:
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

    fmt = os.environ.get("ROBOFLOW_FORMAT", "yolov8")

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

    source = args.source
    if source is None:
        source = "synthetic" if os.environ.get("SMOKE") == "1" else "roboflow"

    if source == "roboflow":
        yaml_path = download_roboflow(out_dir)
    else:
        print("SMOKE/synthetic mode — generating placeholder dataset (no Roboflow key needed)")
        yaml_path = generate_synthetic(out_dir)

    print(f"DATA_YAML={yaml_path}")


if __name__ == "__main__":
    main()
