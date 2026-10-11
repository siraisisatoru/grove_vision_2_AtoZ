# coco128_person — single-class person detector fine-tuned at the deploy resolution

Derived from the coco128 images already on disk for
[coco_yolo26](../coco_yolo26/readme.md): keep only images containing a person,
strip every other class, train `yolo26n.pt` at **IMGSZ=288** — the resolution the
board actually runs.

This exists to answer one question left open by `coco_yolo26`: that experiment
showed stock 640-trained weights are unusable on a pixelated 288 frame. Does
fine-tuning *at* 288 fix it?

**Read this first:** the dataset is 48 train / 13 valid images. That is tiny.
It is a controlled test of the resolution hypothesis, **not** a production person
detector — see [Getting real accuracy](#getting-real-accuracy) for what to build
once the approach is confirmed.

---

## Setup

`datasets/` is gitignored, so build the dataset from the committed script. It is
idempotent — re-running rebuilds from scratch:

```bash
# Prerequisite: datasets/detect/coco_yolo26/ must exist (see that use case's readme)
python3 custom_dataset_usecases/detect/coco128_person/build_dataset.py
```

Expected output:

```
Source images scanned : 128
Skipped (no person)   : 67
train :  48 images,  220 person boxes
valid :  13 images,   34 person boxes
```

You also need the gitignored `.env` (required by `flow.sh` even though no
Roboflow call happens):

```bash
echo 'ROBOFLOW_API_KEY=unused_no_roboflow_project' \
  > custom_dataset_usecases/detect/coco128_person/.env
```

### What the build script does, and why

- **Keeps only images with at least one person box**; 67 of the 128 coco128
  images have none.
- **Strips non-person boxes** and keeps person at class 0, so the dataset is
  `nc=1`. Ultralytics rebuilds the classification head when it sees `nc=1`
  against an 80-class checkpoint — that is expected, and the backbone/neck
  weights still transfer.
- **Deterministic split** (every 5th person image to `valid/`) so re-runs are
  stable. This matters because `valid/images` doubles as the INT8 calibration
  set during conversion.
- coco128 ships **2 label files with no matching image** (one has 6 person
  boxes) and 2 images with no label. The script skips both, which is why you get
  61 usable images rather than 62.

Why one class rather than training all 80: at 288 the frame is heavily
pixelated, and 80 classes spread a nano model's capacity thin over it. One class
puts all of it on the target.

---

## Commands

All from `helper_scripts/`. `--skip-fetch` is always required (no Roboflow
project). Unlike `coco_yolo26`, **training is the point here** — do not pass
`--skip-train` on the first run.

**Full run — train, convert, flash, stream:**

```bash
./flow.sh --detect/coco128_person --skip-fetch
```

**Train and convert, no hardware needed** (step 5 reports "No Grove Vision AI V2
found", which is harmless):

```bash
./flow.sh --detect/coco128_person --skip-fetch --skip-flash
```

**Re-convert existing weights** (after editing `IMGSZ`, say):

```bash
./flow.sh --detect/coco128_person --skip-fetch --skip-train
```

**Flash + stream an already-converted model** (hold BOOT, press RESET when
prompted):

```bash
./flow.sh --detect/coco128_person --skip-fetch --skip-train --skip-convert
```

Artifacts land in `datasets/detect/coco128_person/`: weights at
`runs/train/weights/best.pt`, training curves at `runs/train/results.csv`, and
the deployable model at `convert/yolo26_int8_vela.tflite`.

---

## Training configuration

Set in [config.env](config.env):

| Setting | Value | Why |
|---|---|---|
| `MODEL` | `yolo26n.pt` | selects the YOLO26 convert pipeline (`reg_max==1`, 6 raw heads, LiteRT) |
| `IMGSZ` | `288` | must match the convert step and be a multiple of 32; this is the whole point of the experiment |
| `EPOCHS` | `150` | 48 images converge fast; watch for overfitting |
| `BATCH` | `16` | 3 optimizer steps per epoch — do not raise |
| `FREEZE` | `10` | freeze the COCO backbone, train neck+head only |
| `SCORE_THRESHOLD` | `40` | single class, so a higher bar than `coco_yolo26`'s 25 |

`FREEZE` is a new optional knob in `helper_scripts/train_model.py`. Unset — which
is the case for every other use case — it is omitted from the `train()` call
entirely, so nothing else changes behaviour. It is not defaulted to `0` on
purpose: Ultralytics reads `freeze=0` as "freeze module 0", not "freeze nothing".

**Watch for overfitting.** With 48 images it is the main risk. Check
`runs/train/results.csv`: if `val/box_loss` turns upward while `train/box_loss`
keeps falling, cut `EPOCHS`. `best.pt` is selected on val mAP so the checkpoint
itself is reasonable, but a val set of 13 images is a noisy signal.

---

## Getting real accuracy

48 images cannot produce a good person detector no matter how it is trained. Half
the person boxes in coco128 are under 1% of image area — roughly 20 px at 288 —
so the set is both small and hard. Once this run confirms the resolution
hypothesis, replace the data, not the recipe:

1. **COCO val2017, person-filtered** — ~5 000 images, of which ~2 700 contain a
   person. About a 1 GB download and the best effort-to-accuracy trade here. The
   same `build_dataset.py` filtering logic applies unchanged; only the source
   path changes.
2. **COCO train2017, person-filtered** — ~64 000 person images, ~19 GB. This is
   the real answer for a production person detector, at the cost of a much longer
   training run.
3. **A purpose-built person/pedestrian dataset** — CrowdHuman, WiderPerson, or a
   Roboflow person project. Often better matched to a fixed-camera use case than
   COCO's photo-journalism framing, and a Roboflow project would work through the
   normal `fetch_dataset.py` path instead of needing a build script.

Whichever you pick, keep `IMGSZ` at the deploy resolution and keep it to one
class. With more data, drop `FREEZE` (a full fine-tune beats a frozen backbone
once there are thousands of images) and raise `EPOCHS`/`BATCH` accordingly.

---

## Result

### Training (150 epochs, ~2.2 s/epoch on MPS, ~6 min total)

```
Transferred 612/708 items from pretrained weights   (head rebuilt for nc=1)
freeze=10 active — model.0 … model.9 frozen

BEST  epoch 144: mAP50=0.514  mAP50-95=0.338  P=0.686  R=0.513
FINAL epoch 150: mAP50=0.506  mAP50-95=0.331
```

Loss trend — no overfitting collapse, but the gap tells the story:

| epoch | train box | val box | train cls | val cls | mAP50 |
|---|---|---|---|---|---|
| 26 | 1.917 | 1.859 | 1.807 | 2.709 | 0.206 |
| 51 | 1.748 | 1.762 | 1.427 | 2.157 | 0.433 |
| 101 | 1.534 | 1.697 | 1.163 | 1.736 | 0.454 |
| 150 | 1.414 | 1.596 | 1.159 | 1.614 | 0.506 |

Both val losses were still falling at 150, so **the run is data-limited, not
epoch-limited** — more epochs will not help much, more images will. mAP50 0.51 on
a 13-image / 34-box val set is a very noisy number; treat it as "the model
learned something real" and nothing more precise than that.

### Conversion

```
Export verification passed.
CPU operators = 0 (0.0%)      NPU operators = 421 (100.0%)
Total SRAM used                708.75 KiB   (of 1100 KiB arena)
Total Off-chip Flash used     2174.22 KiB   (vela .tflite = 2.39 MB)
inference_time  86.3 ms  ->  11.6 FPS @ 400 MHz
```

Boundary is int8 NCHW with `nc=1` cls heads, correctly interleaved:

```
input   int8 [1,3,288,288]
output0 int8 [1,4,36,36]  box @ stride 8    output1 int8 [1,1,36,36]  cls
output2 int8 [1,4,18,18]  box @ stride 16   output3 int8 [1,1,18,18]  cls
output4 int8 [1,4, 9, 9]  box @ stride 32   output5 int8 [1,1, 9, 9]  cls
```

Dropping from 80 classes to 1 barely moves the footprint — 708.75 KiB SRAM
(identical) and 2174 vs 2200 KiB flash. Cost on this platform is driven by
`IMGSZ`, not `nc`.

### On-device arena, measured

A boot log from the board confirms the arena budget directly:

```
[boot] E arena: used=726616 of 1126400 bytes (399784 free), base=0x34054020
```

726 616 bytes = 709 KiB, matching Vela's prediction exactly, with 390 KiB free.
That headroom is what makes `IMGSZ=320` (~890 KiB predicted) plausible and 384
(~1280 KiB) impossible.

### Detection quality

_Not yet evaluated on hardware — this model has not been flashed. The comparison
that matters is against the `coco_yolo26` baseline on the same camera: same board,
same 288 frame, stock 640-trained 80-class weights vs these 288-trained
single-class weights._
