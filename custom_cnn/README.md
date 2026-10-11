# custom_cnn — hand-rolled models, no Ultralytics

Proof-of-concept architectures built and trained from scratch (Keras/TensorFlow),
bypassing `ultralytics.YOLO()` entirely, to show the board doesn't care where a
model came from — the firmware's `Classifier::isValid()`
(`sscma_micro/sscma/core/model/ma_model_classifier.cpp`) only checks tensor shapes
(one image in, one dense `(1,num_classes)` int8 vector out), and the Ethos-U55/Vela
toolchain only cares about the op set used, not the framework that produced it.

Each subdirectory here is one architecture — its own `model.py`/`train.py`/
`evaluate.py`/`export_tflite.py`/`run_in_docker.sh`, independently runnable, so a
new idea doesn't disturb an existing one. All of them reuse the same
`grove_vision_convert:latest` Docker image (already has TensorFlow/Keras + Vela —
see `convert_yolo_weight/Dockerfile`) rather than needing their own venv.

## Models

- **[tiny_dscnn/](tiny_dscnn/)** — a tiny MobileNet-style depthwise-separable CNN
  (11,715 params), trained on `rock_paper_scissors`. **100% test accuracy**, float
  and post-quantization; Vela reports 0% CPU / 100% NPU ops, 64 KiB SRAM. See its
  README for the full architecture, a real BatchNorm/checkpoint-metric bug found
  and fixed during training, and measured results.

## Adding a new architecture

Copy the shape of `tiny_dscnn/` (same five files), give it a name that describes
the architecture rather than whatever dataset it happens to be pointed at first
(models get reused across datasets more often than the reverse). Point
`DEFAULT_DATASET_DIR` at whichever `datasets/classify/<use-case>/` you're targeting,
and give it its own `runs/<model-name>/` subdirectory under that use case so
multiple architectures trained on the same dataset don't collide.
