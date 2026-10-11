#!/usr/bin/env python3
"""Export the trained Keras model (train.py) to a full-int8 TFLite, then Vela-compile it.

Usage (inside the grove_vision_convert Docker image — see run_in_docker.sh):
    python3 export_tflite.py [--dataset-dir DIR] [--cal-images N]

Reads:  <dataset-dir>/runs/tiny_dscnn/best.keras
Writes: <dataset-dir>/runs/tiny_dscnn/tiny_dscnn_int8.tflite
        <dataset-dir>/runs/tiny_dscnn/tiny_dscnn_int8_vela.tflite   <- deploy this to WE2

Quantization note: the model's first layer is Rescaling(1/255), so the
representative dataset below must feed RAW pixel values in [0,255] (float32),
NOT pre-divided — the Rescaling op is calibrated as part of the graph, exactly
like Ultralytics' LiteRT export does for yolo11n-cls/yolo26 (see convert_yolo_weight/
convert_tflite.py's docstring). That's what makes the resulting int8 input
quantization (scale~1/255, zero_point=-128) match the firmware's fixed
`pixel_u8 -= 128` preprocessing in Classifier::preprocess()
(sscma_micro/sscma/core/model/ma_model_classifier.cpp) — any other calibration
domain would silently feed the wrong effective values into the model on-device.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "classify" / "rock_paper_scissors"
HIMAX_VELA_INI = REPO_ROOT / "convert_yolo_weight" / "himax_vela.ini"


def load_calibration_images(train_dir: Path, n: int, imgsz: int) -> list:
    """Evenly-spaced sample of raw (un-rescaled) images across all classes."""
    paths = sorted(p for p in train_dir.rglob("*") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No images found under {train_dir}")
    step = max(1, len(paths) // n)
    selected = paths[::step][:n]
    images = []
    for p in selected:
        img = keras.utils.load_img(p, target_size=(imgsz, imgsz))
        images.append(keras.utils.img_to_array(img))  # float32, [0,255]
    print(f"Calibration set: {len(images)}/{len(paths)} images from {train_dir} at {imgsz}x{imgsz}")
    return images


def verify_boundary(tflite_path: Path) -> None:
    """Mirror ma::model::Classifier::isValid(): 1 input image tensor
    (square, >=32, %32==0, C in {1,3}), 1 output (1, num_classes>=2) tensor,
    both int8. Refuses to Vela-compile a mismatched export, same discipline as
    convert_yolo_weight/convert_tflite.py's verify_export_classifier()."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inputs = interp.get_input_details()
    outputs = interp.get_output_details()

    print("\n--- exported model boundary ---")
    for d in inputs:
        print(f"  input  {d['name']}: dtype={d['dtype'].__name__} shape={list(d['shape'])} quant={d['quantization']}")
    for d in outputs:
        print(f"  output {d['name']}: dtype={d['dtype'].__name__} shape={list(d['shape'])} quant={d['quantization']}")
    print("--- end boundary ---\n")

    problems = []
    if len(inputs) != 1 or len(outputs) != 1:
        problems.append(f"expected 1 input/1 output, got {len(inputs)}/{len(outputs)}")
    else:
        ishape = list(inputs[0]["shape"])
        if len(ishape) != 4:
            problems.append(f"input shape {ishape} is not rank-4")
        else:
            n, h, w, c = ishape  # Keras/TFLiteConverter emits NHWC natively
            if n != 1 or h != w or h < 32 or h % 32 != 0 or c not in (1, 3):
                problems.append(f"input is {h}x{w} C={c} — Classifier::isValid() needs square, >=32, %32==0, C in {{1,3}}")
        oshape = list(outputs[0]["shape"])
        if len(oshape) < 2 or oshape[0] != 1 or oshape[1] < 2:
            problems.append(f"output shape {oshape} does not match (1, num_classes>=2)")

    bad_io = [d for d in (inputs + outputs) if d["dtype"] != np.int8]
    if bad_io:
        kinds = sorted({d["dtype"].__name__ for d in bad_io})
        problems.append(f"{len(bad_io)} boundary tensors are {'/'.join(kinds)}, not int8 — not full-integer")

    if problems:
        print("EXPORT VERIFICATION FAILED:")
        for p in problems:
            print(f"  * {p}")
        raise SystemExit("Refusing to Vela-compile a mismatched export.")
    print("Export verification passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--cal-images", type=int, default=100)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = dataset_dir / "runs" / "tiny_dscnn"
    keras_path = out_dir / "best.keras"
    if not keras_path.exists():
        raise SystemExit(f"No trained model at {keras_path} — run train.py first.")

    model = keras.models.load_model(keras_path)
    # Read the resolution back from the trained model itself (input_shape is
    # (None, H, W, C)) rather than taking it as a separate flag here — that
    # would risk drifting out of sync with whatever --imgsz train.py actually
    # used to build/save this model.
    imgsz = model.input_shape[1]
    print(f"Model input resolution: {imgsz}x{imgsz} (read from {keras_path.name})")
    cal_images = load_calibration_images(dataset_dir / "train", args.cal_images, imgsz)

    def representative_dataset():
        for img in cal_images:
            yield [np.expand_dims(img, axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting to full-int8 TFLite...")
    tflite_model = converter.convert()

    tflite_path = out_dir / "tiny_dscnn_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Wrote {tflite_path} ({len(tflite_model)} bytes)")

    verify_boundary(tflite_path)

    print(f"\nRunning Vela ({HIMAX_VELA_INI.name})...")
    subprocess.run(
        [
            "vela",
            "--accelerator-config", "ethos-u55-64",
            "--config", str(HIMAX_VELA_INI),
            "--system-config", "My_Sys_Cfg",
            "--memory-mode", "My_Mem_Mode_Parent",
            "--output-dir", str(out_dir),
            str(tflite_path),
        ],
        check=True,
    )

    vela_out = out_dir / (tflite_path.stem + "_vela.tflite")
    print(f"\nDone! Deploy this file to WE2: {vela_out}")


if __name__ == "__main__":
    main()
