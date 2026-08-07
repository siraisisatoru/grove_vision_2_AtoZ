#!/usr/bin/env python3
"""
Train a YOLO model on the fetched dataset.

Reads from environment (set by flow.sh via the use-case .env):
  MODEL      — base weights, e.g. yolo11n.pt
  EPOCHS     — number of training epochs
  IMGSZ      — input image size (must match convert step)
  BATCH      — batch size
  FREEZE     — optional; freeze the first N modules (backbone) while fine-tuning.
               Unset (the default) trains the whole network, which is what every
               use case did before this knob existed. Set it for small datasets
               where a full fine-tune memorises instead of generalising.
  DATASET_DIR — absolute path to the dataset root (contains data.yaml)
  USE_CASE   — name of the use case (for labelling the run)
  SMOKE      — set to 1 to run a 1-epoch smoke test without downloading data

Saves best weights to: $DATASET_DIR/runs/train/weights/best.pt
"""

import os
import sys
from pathlib import Path


def main():
    smoke   = os.environ.get("SMOKE") == "1"
    model_name = os.environ.get("MODEL", "yolo11n.pt")
    epochs     = 1 if smoke else int(os.environ.get("EPOCHS", 100))
    imgsz      = int(os.environ.get("IMGSZ", 192))
    batch      = int(os.environ.get("BATCH", 16))
    # Absent/empty FREEZE must stay absent from the train() kwargs, not become
    # freeze=0 — Ultralytics treats 0 as "freeze module 0" rather than "freeze
    # nothing", so a default would silently change every existing use case.
    freeze_env = os.environ.get("FREEZE", "").strip()
    freeze     = int(freeze_env) if freeze_env else None
    dataset_dir = Path(os.environ.get("DATASET_DIR", "."))

    yaml_candidates = (
        list(dataset_dir.glob("data.yaml"))
        + list(dataset_dir.glob("*.yaml"))
    )
    if not yaml_candidates:
        sys.exit(f"No .yaml found in {dataset_dir}. Run fetch first (or unset --skip-fetch).")
    data_yaml = yaml_candidates[0]
    print(f"Dataset yaml: {data_yaml}")

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training device: {device}")
    if smoke:
        print("SMOKE=1 — running 1-epoch sanity check")

    if freeze is not None:
        print(f"Freezing the first {freeze} modules (backbone) — fine-tune mode")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(dataset_dir / "runs"),
        name="train",
        exist_ok=True,  # always write to runs/train/ rather than runs/train2/, train3/, …
        device=device,
        **({"freeze": freeze} if freeze is not None else {}),
    )

    trained_pt = dataset_dir / "runs" / "train" / "weights" / "best.pt"
    print(f"Trained weights: {trained_pt}")


if __name__ == "__main__":
    main()
