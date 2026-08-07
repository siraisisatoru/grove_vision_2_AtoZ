#!/usr/bin/env python3
"""Print comma-separated class names from a YOLO data.yaml.

Usage: python get_labels.py <path/to/data.yaml>
"""
import sys
from pathlib import Path


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
    if len(sys.argv) < 2:
        sys.exit("Usage: get_labels.py <data.yaml>")
    names = parse_names(Path(sys.argv[1]))
    if names:
        print(",".join(names))
