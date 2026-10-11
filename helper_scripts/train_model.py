#!/usr/bin/env python3
"""
Train a YOLO model on the fetched dataset.

Reads from environment (set by flow.sh via the use-case .env):
  MODEL      — base weights, e.g. yolo11n.pt (or yolo11n-cls.pt for TASK=classify)
  EPOCHS     — number of training epochs
  IMGSZ      — input image size (must match convert step)
  BATCH      — batch size
  FREEZE     — optional; freeze the first N modules (backbone) while fine-tuning.
               Unset (the default) trains the whole network, which is what every
               use case did before this knob existed. Set it for small datasets
               where a full fine-tune memorises instead of generalising.
  DATASET_DIR — absolute path to the dataset root (contains data.yaml, or for
               TASK=classify, train/<class>/ subdirectories directly)
  TASK       — 'detect' (default) or 'classify'. classify passes the dataset
               root folder to Ultralytics directly instead of a data.yaml —
               see check_cls_dataset() in ultralytics/data/utils.py, which
               resolves train/ + {val,validation,valid}/ + test/ itself.
  USE_CASE   — name of the use case (for labelling the run)
  SMOKE      — set to 1 to run a 1-epoch smoke test without downloading data

Saves best weights to: $DATASET_DIR/runs/<model-key>/weights/best.pt, where
<model-key> is MODEL with any .pt suffix stripped (e.g. yolo11n-cls.pt ->
yolo11n-cls) — namespaces the run by architecture so retraining the same use
case with a different MODEL doesn't overwrite a previous architecture's
results. flow.sh looks for the trained weights at this same path.
"""

import os
import sys
from pathlib import Path


def main():
    smoke   = os.environ.get("SMOKE") == "1"
    task       = os.environ.get("TASK", "detect")
    default_model = "yolo11n-cls.pt" if task == "classify" else "yolo11n.pt"
    model_name = os.environ.get("MODEL", default_model)
    epochs     = 1 if smoke else int(os.environ.get("EPOCHS", 100))
    imgsz      = int(os.environ.get("IMGSZ", 192))
    batch      = int(os.environ.get("BATCH", 16))
    # Absent/empty FREEZE must stay absent from the train() kwargs, not become
    # freeze=0 — Ultralytics treats 0 as "freeze module 0" rather than "freeze
    # nothing", so a default would silently change every existing use case.
    freeze_env = os.environ.get("FREEZE", "").strip()
    freeze     = int(freeze_env) if freeze_env else None
    dataset_dir = Path(os.environ.get("DATASET_DIR", "."))
    model_key  = model_name[:-3] if model_name.endswith(".pt") else model_name

    if task == "classify":
        if not (dataset_dir / "train").is_dir():
            sys.exit(f"No train/ folder found in {dataset_dir}. Run fetch first (or unset --skip-fetch).")
        data_arg = str(dataset_dir)
        print(f"Dataset root (classify): {data_arg}")
    else:
        yaml_candidates = (
            list(dataset_dir.glob("data.yaml"))
            + list(dataset_dir.glob("*.yaml"))
        )
        if not yaml_candidates:
            sys.exit(f"No .yaml found in {dataset_dir}. Run fetch first (or unset --skip-fetch).")
        data_arg = str(yaml_candidates[0])
        print(f"Dataset yaml: {data_arg}")

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
        data=data_arg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(dataset_dir / "runs"),
        name=model_key,
        exist_ok=True,  # always write to runs/<model_key>/ rather than <model_key>2/, 3/, …
        device=device,
        **({"task": "classify"} if task == "classify" else {}),
        **({"freeze": freeze} if freeze is not None else {}),
    )

    trained_pt = dataset_dir / "runs" / model_key / "weights" / "best.pt"
    print(f"Trained weights: {trained_pt}")


if __name__ == "__main__":
    main()
