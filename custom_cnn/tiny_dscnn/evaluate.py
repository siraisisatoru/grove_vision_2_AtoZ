#!/usr/bin/env python3
"""Self-evaluation: does this model actually work, float AND after quantization?

Usage (inside the grove_vision_convert Docker image — see run_in_docker.sh):
    python3 evaluate.py [--dataset-dir DIR] [--model float|quantized|both]

Reports top-1 accuracy + a confusion matrix + per-class precision/recall on the
untouched test/ split, for:
  - the trained float Keras model (runs/tiny_dscnn/best.keras)
  - the exported int8 TFLite model (runs/tiny_dscnn/tiny_dscnn_int8.tflite) —
    the PRE-Vela file; a Vela-compiled model embeds the ethos-u custom op and
    cannot be allocated by a stock interpreter, same constraint documented in
    convert_yolo_weight/convert_tflite.py's _load_interpreter().

Comparing the two directly answers the actual question a "self-evaluation"
should answer for this pipeline: did int8 quantization cost real accuracy, or
is this proof-of-concept model still good after the exact transformation that
ships to the board?
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "classify" / "rock_paper_scissors"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def print_report(title: str, cm: np.ndarray, class_names: list):
    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total else 0.0

    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{n[:8]:>10}" for n in class_names)
    print(f"{'':>10}{header}")
    for i, row in enumerate(cm):
        print(f"{class_names[i][:8]:>10}" + "".join(f"{v:>10}" for v in row))

    print(f"\n{'class':>10}{'precision':>12}{'recall':>10}")
    for i, name in enumerate(class_names):
        col_sum = cm[:, i].sum()
        row_sum = cm[i, :].sum()
        precision = cm[i, i] / col_sum if col_sum else 0.0
        recall = cm[i, i] / row_sum if row_sum else 0.0
        print(f"{name[:10]:>10}{precision:>12.3f}{recall:>10.3f}")
    return acc


def load_test_set(dataset_dir: Path, class_names_expected: list, imgsz: int):
    ds = keras.utils.image_dataset_from_directory(
        dataset_dir / "test",
        image_size=(imgsz, imgsz),
        batch_size=1,
        shuffle=False,
        label_mode="int",
    )
    assert ds.class_names == class_names_expected, (
        f"test/ class order {ds.class_names} != training class order {class_names_expected} "
        f"— predictions would be scored against the wrong labels."
    )
    images, labels = [], []
    for img, label in ds.unbatch():
        images.append(img.numpy())
        labels.append(int(label.numpy()))
    return images, np.array(labels)


def evaluate_float(model, images, labels, class_names):
    batch = np.stack(images, axis=0)
    probs = model.predict(batch, verbose=0)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, len(class_names))
    return print_report("Float Keras model (best.keras)", cm, class_names)


def evaluate_quantized(out_dir: Path, images, labels, class_names):
    tflite_path = out_dir / "tiny_dscnn_int8.tflite"
    if not tflite_path.exists():
        print(f"\n(skipping quantized eval — {tflite_path} not found; run export_tflite.py first)")
        return None

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_scale, in_zero = in_detail["quantization"]

    preds = []
    for img in images:
        q = np.round(img / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_detail["index"], q[np.newaxis, ...])
        interp.invoke()
        out = interp.get_tensor(out_detail["index"])[0]
        preds.append(int(np.argmax(out)))  # argmax is scale/zero-point invariant — no need to dequantize

    cm = confusion_matrix(labels, np.array(preds), len(class_names))
    return print_report("Quantized int8 TFLite model (pre-Vela)", cm, class_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--model", choices=["float", "quantized", "both"], default="both")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = dataset_dir / "runs" / "tiny_dscnn"
    class_names = (out_dir / "classes.txt").read_text().splitlines()

    # Load the float model first (even for --model quantized) so the test set
    # gets built at the SAME resolution the model actually trained at, read
    # back from the model itself rather than duplicated/hardcoded here.
    keras_model = keras.models.load_model(out_dir / "best.keras")
    imgsz = keras_model.input_shape[1]
    print(f"Model input resolution: {imgsz}x{imgsz} (read from best.keras)")

    images, labels = load_test_set(dataset_dir, class_names, imgsz)
    print(f"Test set: {len(images)} images, classes={class_names}")

    float_acc = quant_acc = None
    if args.model in ("float", "both"):
        float_acc = evaluate_float(keras_model, images, labels, class_names)
    if args.model in ("quantized", "both"):
        quant_acc = evaluate_quantized(out_dir, images, labels, class_names)

    if float_acc is not None and quant_acc is not None:
        delta = quant_acc - float_acc
        print(f"\nFloat vs quantized accuracy delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
