"""A small hand-rolled image classifier — no Ultralytics involved.

Architecture: 1 plain conv stem + 3 depthwise-separable downsampling blocks +
global average pool + dense + softmax. Every op here (Conv2D, DepthwiseConv2D,
BatchNorm, ReLU, GlobalAveragePooling2D, Dense, Softmax) is from the same op
families that already compiled to 100% Ethos-U55 NPU ops (0 CPU fallback) for
this repo's yolo11n-cls export — see custom_cnn/tiny_dscnn/README.md for the reasoning.

Softmax is included IN the graph (not left to the caller) because the WE2
firmware's Classifier::postprocess() (sscma_micro/sscma/core/model/
ma_model_classifier.cpp) thresholds the dequantized output directly against a
0..1 score threshold — it expects bounded probabilities, not raw logits.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes: int, input_size: int) -> keras.Model:
    """input_size must be square, >=32, and a multiple of 32 (Classifier::isValid()).

    No default here on purpose: config.env's IMGSZ (via flow.sh's --imgsz to
    train.py) is the actual source of truth for resolution — a module-level
    default here would just be a second copy of that number to keep in sync,
    the exact trap that briefly made changing resolution look like it required
    an edit to this file instead of config.env.
    """
    # BN_MOMENTUM lower than Keras' default (0.99): with only a few dozen short
    # epochs on a tiny dataset, the default takes too long for the running
    # mean/variance to catch up to the (fast-moving, tiny-model) batch
    # statistics — the symptom is exactly what showed up here empirically:
    # train accuracy climbing normally while eval-mode (running-stats)
    # accuracy stays pinned at a single-class collapse from epoch 1 onward,
    # because BatchNorm's inference path was still using near-initial
    # (mean=0, var=1) running stats that don't match the real activations.
    BN_MOMENTUM = 0.9

    inputs = keras.Input(shape=(input_size, input_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Stem: plain conv, stride 2
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.ReLU()(x)

    # Depthwise-separable downsampling blocks
    for channels in (32, 64, 96):
        x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(channels, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_classes)(x)
    outputs = layers.Softmax(name="scores")(logits)

    return keras.Model(inputs, outputs, name="tiny_dscnn")


if __name__ == "__main__":
    m = build_model(num_classes=3, input_size=192)
    m.summary()
