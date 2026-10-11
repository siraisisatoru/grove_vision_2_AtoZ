#!/usr/bin/env python3
"""Train the hand-rolled CNN (model.py) on an ImageFolder-layout dataset.

Usage (inside the grove_vision_convert Docker image — see run_in_docker.sh):
    python3 train.py [--dataset-dir DIR] [--epochs N] [--batch-size N] [--imgsz N]

Reads:  <dataset-dir>/{train,valid}/<class>/*.jpg
Writes: <dataset-dir>/runs/tiny_dscnn/best.keras
        <dataset-dir>/runs/tiny_dscnn/classes.txt   (sorted class names, one per line —
                                                      the order Keras assigned indices in)
"""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from model import build_model

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "classify" / "rock_paper_scissors"


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # random_brightness/contrast do not clip — left alone they push pixels
    # outside the [0,255] range the model's Rescaling(1/255) layer expects,
    # feeding it out-of-distribution values during training.
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=192,
                    help="Input resolution, square (default 192). flow.sh "
                         "passes the use case's config.env IMGSZ here, so "
                         "this is the one place a use case's resolution is "
                         "actually decided — export_tflite.py/evaluate.py "
                         "read it back from the saved model instead of "
                         "needing their own copy of this flag.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = dataset_dir / "runs" / "tiny_dscnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        image_size=(args.imgsz, args.imgsz),
        batch_size=args.batch_size,
        label_mode="int",
    )
    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")
    (out_dir / "classes.txt").write_text("\n".join(class_names) + "\n")

    valid_ds = keras.utils.image_dataset_from_directory(
        dataset_dir / "valid",
        image_size=(args.imgsz, args.imgsz),
        batch_size=args.batch_size,
        label_mode="int",
    )

    # Augmentation applied only to the training tf.data pipeline — kept out of
    # the saved/exported model graph entirely.
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes=len(class_names), input_size=args.imgsz)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    best_path = out_dir / "best.keras"
    # Monitor val_loss, not val_accuracy: with only 3 classes and a handful of
    # epochs, val_accuracy can sit dead flat for several epochs in a row (an
    # all-or-nothing argmax metric on ~250 examples has limited resolution),
    # giving EarlyStopping/ModelCheckpoint no signal to pick a better epoch.
    # val_loss is continuous and keeps discriminating even when val_accuracy
    # doesn't move.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=valid_ds, epochs=args.epochs, callbacks=callbacks)

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val_accuracy: {best_val_acc:.4f}")
    print(f"Saved: {best_path}")
    print(f"Classes: {out_dir / 'classes.txt'}")


if __name__ == "__main__":
    main()
