# coco_val_person — person detector from COCO val2017, trained at the deploy resolution

Single-class (`nc=1`) person detection, fine-tuned from `yolo26n.pt` at
**IMGSZ=288**, on **2423 train / 270 valid images** carrying **10 777 person
boxes**.

This is the data-volume follow-up to
[coco128_person](../coco128_person/readme.md), which reached mAP50 0.51 on 48
images with both val losses still falling at the final epoch — a data-limited
run, not an epoch-limited one. Same recipe, ~56× the data.

---

## Why val2017 rather than train2017

`val` is only COCO's *split name*, not a role — nothing stops us training on it.
The choice is purely about download size:

| Split | Download | Images | Person images |
|---|---|---|---|
| val2017 | **1 GB** | 5 000 | **2 693** |
| train2017 | 19 GB | 118 287 | ~64 000 |

Training on the val split costs us the standard COCO benchmark split, which is
irrelevant here — we never compare against published COCO numbers, and the board
is the only benchmark that matters. **train2017 is the upgrade path** if 2423
images still underfits; only the source path in `build_dataset.py` changes.

---

## Setup

`datasets/` is gitignored, so build from the committed script. First run
downloads ~865 MB into `datasets/_coco_raw/` and caches it; later runs reuse it:

```bash
python3 custom_dataset_usecases/detect/coco_val_person/build_dataset.py
```

Expected output:

```
Skipped (no label file) : 48
Skipped (no person)     : 2259
MIN_BOX_AREA            : 0.0  (no size filter)
train :  2423 images,   9747 person boxes
valid :   270 images,   1030 person boxes
```

Plus the gitignored `.env` that `flow.sh` requires even with no Roboflow call:

```bash
echo 'ROBOFLOW_API_KEY=unused_no_roboflow_project' \
  > custom_dataset_usecases/detect/coco_val_person/.env
```

Raw downloads (cached, safe to delete to reclaim ~865 MB):

- `coco2017labels.zip` (49 MB) — Ultralytics' YOLO-format COCO labels
- `val2017.zip` (816 MB) — the val2017 JPEGs

### What the build script does

- Keeps images with at least one person box; **2259 of 4952 val2017 images have
  no person** and are dropped, along with 48 images that carry no label file at
  all (COCO ships some fully unannotated images).
- Strips non-person boxes, keeps person at class 0 → `nc=1`. Ultralytics rebuilds
  the classification head for the new class count (`Transferred 612/708 items`);
  backbone and neck weights still transfer.
- Deterministic split — every 10th person image to `valid/`, so re-runs are
  stable. This matters because `valid/images` doubles as the INT8 calibration set
  during conversion.

---

## The size problem this dataset does *not* fix

Person box areas as a fraction of frame, measured across all 10 777 boxes:

```
median 1.41%     p25 0.28%     p75 7.39%
boxes >= 1% of frame : 5972 (55%)     <- ~29 px at 288
boxes >= 3% of frame : 4092 (38%)
boxes >= 5% of frame : 3238 (30%)
```

**45% of person boxes are under ~29 px at 288.** The model cannot resolve those,
so they act as label noise — the network is penalised for missing targets that
carry no recoverable signal at this input size.

The current build keeps every box (`MIN_BOX_AREA = 0.0`). To train only on
resolvable targets, set that constant at the top of `build_dataset.py` to `0.03`
or `0.05` and rebuild. Expect fewer images but cleaner supervision, and a model
biased toward the large, close subjects a fixed 288 camera actually sees.

Note this is a *domain* mismatch as much as a resolution one: COCO is
photo-journalism framing — crowds, street scenes, distant bystanders. A fixed
camera watching 1–3 people at close range is a different distribution, and a few
thousand domain-matched images would beat all of COCO for that deployment.

---

## Commands

All from `helper_scripts/`. `--skip-fetch` is always required (no Roboflow
project); training is the point, so no `--skip-train` on the first run.

**Full run — train, convert, flash, stream:**

```bash
./flow.sh --detect/coco_val_person --skip-fetch
```

**Train and convert, no hardware needed:**

```bash
./flow.sh --detect/coco_val_person --skip-fetch --skip-flash
```

**Re-convert existing weights** (e.g. after changing `IMGSZ`):

```bash
./flow.sh --detect/coco_val_person --skip-fetch --skip-train
```

**Flash + stream an already-converted model** (hold BOOT, press RESET when
prompted):

```bash
./flow.sh --detect/coco_val_person --skip-fetch --skip-train --skip-convert
```

Artifacts land in `datasets/detect/coco_val_person/`: `runs/train/weights/best.pt`,
`runs/train/results.csv`, and `convert/yolo26_int8_vela.tflite`.

---

## Training configuration

Set in [config.env](config.env):

| Setting | Value | Why |
|---|---|---|
| `MODEL` | `yolo26n.pt` | selects the YOLO26 convert pipeline (`reg_max==1`, 6 raw heads, LiteRT) |
| `IMGSZ` | `288` | must match convert and be a multiple of 32; train at the resolution the board runs |
| `EPOCHS` | `60` | 152 steps/epoch, ~70 s/epoch on MPS → roughly a 70-minute run |
| `BATCH` | `16` | keeps the step count and memory in line with the other use cases |
| `FREEZE` | *unset* | deliberately **not** frozen — see below |
| `SCORE_THRESHOLD` | `40` | single class |

`coco128_person` used `FREEZE=10` because freezing a pretrained backbone is a
small-data defence: on 48 images a full fine-tune memorises. With 2423 images
that defence costs more than it saves — the backbone should adapt to 288-scale
features — so `FREEZE` is left unset, which makes `train_model.py` omit the
kwarg entirely.

---

## Result

### Training (60 epochs, 30.2 s/epoch on MPS, 30 min total)

```
Transferred 612/708 items from pretrained weights   (head rebuilt for nc=1)
no freeze — full fine-tune

BEST  epoch 59: mAP50=0.534  mAP50-95=0.306  P=0.718  R=0.459
FINAL epoch 60: mAP50=0.534  mAP50-95=0.307
```

| epoch | train box | val box | train cls | val cls | mAP50 |
|---|---|---|---|---|---|
| 1 | 1.852 | 1.881 | 2.674 | 2.172 | 0.318 |
| 21 | 1.852 | 1.973 | 1.633 | 1.638 | 0.447 |
| 41 | 1.754 | 1.811 | 1.453 | 1.413 | 0.513 |
| 60 | 1.616 | 1.784 | 1.190 | 1.389 | 0.534 |

### More data barely helped — and that is the finding

| | coco128_person | coco_val_person |
|---|---|---|
| train images | 48 | 2 423 |
| val images | 13 | 270 |
| mAP50 | 0.514 | **0.534** |
| mAP50-95 | 0.338 | 0.307 |
| P / R | 0.686 / 0.513 | 0.718 / **0.459** |

56× the training data moved mAP50 by 0.02. The two numbers are not directly
comparable — coco128_person's was measured on 13 images and 34 boxes, which is
noise — so the real conclusion is narrower but more useful:

**mAP50 ≈ 0.53 with recall 0.46 looks like the ceiling for COCO-person at 288
with a nano model, and data volume is not what is holding it there.** The model
misses over half the boxes, which lines up with the 45% of boxes under ~29 px
documented above: they are unresolvable at this input size, so no amount of extra
COCO data teaches them.

Val losses were still declining at epoch 60, so a longer run would gain a little
more — but the gap between train and val cls loss had already closed, which is
not the signature of a model starved for examples.

### Conversion

```
Total SRAM used                708.75 KiB   (of 1100 KiB arena — identical to the other 288 models)
Total Off-chip Flash used     2183.91 KiB   (vela .tflite = 2.40 MB)
inference_time  86.4 ms  ->  11.6 FPS @ 400 MHz
```

### What to try next, in order of expected payoff

1. **Filter the unresolvable boxes.** Set `MIN_BOX_AREA = 0.03` in
   `build_dataset.py` and retrain. Cleaner supervision, and the model stops being
   penalised for targets it cannot see. Cheapest experiment left.
2. **Domain-matched data.** A few thousand images from a viewpoint like the
   deployment's (fixed camera, 1–3 people, large in frame) should beat COCO
   outright. A Roboflow person project needs no new code — it runs through
   `fetch_dataset.py` as an ordinary use case.
3. **COCO train2017** (~64 k person images, 19 GB). Worth it only if the evidence
   above is wrong about data volume — try 1 and 2 first.
4. **IMGSZ=320.** ~890 KiB of the 1100 KiB arena, ~7.7 FPS. A modest detail gain
   that does not change the fundamentals.

### Detection quality on hardware

_Not evaluated — this model has not been flashed._
