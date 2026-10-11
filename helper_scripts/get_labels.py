#!/usr/bin/env python3
"""Print comma-separated class names from a YOLO data.yaml, or (with --classify)
from an ImageFolder-layout classification dataset's train/ subdirectories.

Usage: python get_labels.py <path/to/data.yaml>
       python get_labels.py --classify <path/to/dataset_root>
"""
import sys
from pathlib import Path


def classify_labels(dataset_root: Path) -> list[str]:
    """Sorted class names from train/<class>/ subdirectories.

    Must be sorted() — Ultralytics' classify dataloader assigns class indices
    by sorted folder name, so this order must match exactly or downstream
    label display (stream_view.py) will be off-by-index.
    """
    train_dir = dataset_root / "train"
    if not train_dir.is_dir():
        return []
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def parse_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        return []
    text = yaml_path.read_text()
    if "names:" not in text:
        return []
    after = text.split("names:", 1)[1]
    line0 = after.splitlines()[0]
    names: list[str] = []
    if "[" in line0:
        inner = line0[line0.find("[") + 1: line0.find("]")]
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
    return names


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--classify":
        if len(argv) < 2:
            sys.exit("Usage: get_labels.py --classify <dataset_root>")
        names = classify_labels(Path(argv[1]))
    else:
        if len(argv) < 1:
            sys.exit("Usage: get_labels.py <data.yaml>")
        names = parse_names(Path(argv[0]))
    if names:
        print(",".join(names))
