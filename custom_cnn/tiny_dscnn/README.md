# tiny_dscnn — a hand-rolled classifier, no Ultralytics

Proof of concept: does the board care that a model came from Ultralytics? No — the
firmware's `Classifier::isValid()` (`sscma_micro/sscma/core/model/ma_model_classifier.cpp`)
only checks tensor shapes (one image in, one dense `(1,num_classes)` int8 vector out).
This is a small CNN built and trained from scratch in Keras/TensorFlow — no
`ultralytics.YOLO()` anywhere in the loop — proving that contract out on the real
`rock_paper_scissors` dataset (`datasets/classify/rock_paper_scissors/`).

## Architecture (`model.py`)

A tiny MobileNet-style depthwise-separable CNN:

```
Input 192x192x3 -> Rescaling(1/255)
-> Conv2D(16, 3x3, s2) + BN + ReLU                     -> 96x96x16
-> [DepthwiseConv2D(3x3, s2) + BN + ReLU
    + Conv2D(1x1) + BN + ReLU] x3, channels 32->64->96  -> 12x12x96
-> GlobalAveragePooling2D -> Dropout(0.3)
-> Dense(3) -> Softmax
```

(Bumped from the original 96x96 to 192x192 to match the resolution the
`yolo11n-cls` baseline trains at on this same dataset — parameter count is
unchanged, since a fully-convolutional-then-GAP architecture's weights don't
depend on input size, only its compute (MACs) does.)

**11,715 params (45.76 KB)** — about 130x smaller than `yolo11n-cls`'s 1.5M params.
Softmax is included in the graph deliberately: `Classifier::postprocess()` thresholds
the dequantized output directly against a 0..1 score (default 0.5), so the exported
graph must already emit bounded probabilities, not raw logits.

Every op (Conv2D, DepthwiseConv2D, BatchNorm, ReLU, GlobalAveragePooling2D, Dense,
Softmax) is from the same op families that already compiled to 100% Ethos-U55 NPU
ops for this repo's `yolo11n-cls` export — confirmed below, this one does too.

## Environment

No new venv — `convert_yolo_weight/requirements.txt` already pins `tensorflow==2.18.0`
+ `keras==3.8.0`, and `ethos-u-vela` is already installed in the same
`grove_vision_convert:latest` Docker image `flow.sh` builds for the Ultralytics path.
`run_in_docker.sh` just runs these scripts inside that same image:

```bash
cd custom_cnn/tiny_dscnn
./run_in_docker.sh train.py
./run_in_docker.sh evaluate.py
./run_in_docker.sh export_tflite.py
```

## A real bug found and fixed along the way

The first training run collapsed completely: train accuracy climbed to 98% by
epoch 3, but `val_accuracy` was pinned at **exactly 0.3175 on every single epoch**
(6 epochs straight, bit-for-bit identical). `evaluate.py`'s confusion matrix showed
why — the model predicted "rock" for **100% of test images**, from epoch 1 onward:

```
               paper      rock  scissors
     paper         0        83         0
      rock         0        80         0
  scissors         0        89         0
```

Two compounding bugs:
1. **BatchNorm's default momentum (0.99)** left the running mean/variance too far
   behind the fast-moving batch statistics of this tiny model over just a few dozen
   short epochs — the eval-mode (running-stats) path was still effectively using
   near-initial statistics that didn't match real activations, while train-mode
   (batch-stats) looked fine. Fixed by lowering momentum to 0.9.
2. **Monitoring `val_accuracy`** for `EarlyStopping`/`ModelCheckpoint` gave no
   usable signal once it went flat — both callbacks kept "restoring" the same
   barely-trained epoch-1 weights since nothing ever technically improved on that
   metric. Switched to monitoring `val_loss` (continuous, kept discriminating
   between epochs even while `val_accuracy` was stuck).
3. (Also fixed opportunistically): `tf.image.random_brightness`/`random_contrast`
   don't clip — added `tf.clip_by_value` so augmented pixels stay in the `[0,255]`
   range the model's `Rescaling(1/255)` layer expects.

After the fix, training behaved normally (`val_accuracy` climbing to 1.0 within
~20 epochs, not stuck) — see Results below.

## Results (measured at 96x96 — stale after the 192x192 bump above; rerun and update)

**Float Keras model**, test set (252 held-out images, untouched by training):

| | accuracy | paper P/R | rock P/R | scissors P/R |
|---|---|---|---|---|
| float (`best.keras`) | **1.0000** (252/252) | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |
| int8 TFLite (pre-Vela) | **1.0000** (252/252) | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |

Quantization cost **zero accuracy** on this test set.

**Vela compile** (`ethos-u55-64` / `My_Sys_Cfg` / `My_Mem_Mode_Parent`, same
`convert_yolo_weight/himax_vela.ini` the Ultralytics path uses):

| | this model (96px) | `yolo11n-cls` (192px, Ultralytics) | `yolo26` detect (288px, Ultralytics) |
|---|---|---|---|
| CPU / NPU ops | **0% / 100%** | 0% / 100% | 0% / 100% |
| SRAM used | **64.12 KiB** | 320.39 KiB | 708.75 KiB |
| Flash used | **16.75 KiB** | 1399.23 KiB | 2200.27 KiB |
| Params | 11,715 | ~1.5M | — |
| Input quant | `scale=1.0, zero_point=-128` | `scale≈0.00387, zero_point=-128` | `scale≈0.00389, zero_point=-128` |

The `scale=1.0, zero_point=-128` input quantization is actually a cleaner match to
the firmware's fixed `pixel_u8 -= 128` preprocessing (`Classifier::preprocess()`)
than the Ultralytics path's ~1/255 scale — with scale=1.0 the firmware's raw byte
shift *exactly* reconstructs the original 0-255 pixel value at the model boundary,
with no approximation, because `Rescaling(1/255)` is an explicit layer inside this
graph rather than baked into the boundary quantization itself.

**0 CPU operators / 100% NPU operators** — same bar every other model in this repo
has cleared. Deploy file: `datasets/classify/rock_paper_scissors/runs/tiny_dscnn/tiny_dscnn_int8_vela.tflite`
(flash the same way as any other use case's Vela output — `flash_model.py
--model-file <path> --address 0x400000` — not yet flashed to real hardware in this
session).

## Files

- `model.py` — architecture (`build_model()`)
- `train.py` — trains on `datasets/classify/rock_paper_scissors/{train,valid}/`,
  saves to `<dataset>/runs/tiny_dscnn/{best.keras,classes.txt}`
- `evaluate.py` — self-evaluation: float vs quantized accuracy/confusion-matrix on
  `test/`
- `export_tflite.py` — Keras -> full-int8 TFLite -> Vela compile
- `run_in_docker.sh` — runs any of the above inside `grove_vision_convert:latest`
