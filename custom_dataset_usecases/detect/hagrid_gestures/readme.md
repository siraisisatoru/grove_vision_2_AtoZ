# hagrid_gestures — 5-class hand-gesture detector from HaGRID, trained at the deploy resolution

Five-class (`nc=5`) hand-gesture detection, fine-tuned from `yolo26n.pt` at
**IMGSZ=288** on a balanced slice of [HaGRID](https://github.com/hukenovs/hagrid)
(HAnd Gesture Recognition Image Dataset, 18 gesture classes, ~550k images).

| index | gesture | why it survives 288 px on a nano model |
|---|---|---|
| 0 | `palm` | five fingers spread, frontal — largest, most open silhouette |
| 1 | `fist` | closed hand — most compact silhouette |
| 2 | `peace` | two adjacent fingers up in a V |
| 3 | `like` | thumbs up — hand in profile, one stubby vertical protrusion |
| 4 | `call` | "shaka", thumb and pinky extended — widest span |

The selection criterion is **gross silhouette, not finger detail**. At 288 px a
hand occupies a few dozen pixels, so classes that differ only in finger count or
rotation are not separable by a nano backbone. HaGRID's near-duplicate pairs are
deliberately avoided:

- `palm` / `stop` — fingers spread vs together; nearly the same blob
- `peace` / `peace_inverted`, `three` / `three2`, `two_up` / `two_up_inverted` — rotations of each other
- `like` / `dislike` — the same shape flipped 180°
- `one` / `two_up` / `three` / `four` — a finger-counting ladder

**The weakest pair kept is `fist` vs `like`** — a `like` is a fist plus a thumb,
and the thumb is a handful of pixels at this resolution. Check
`runs/train/confusion_matrix.png` after the first run; if they mix, swap `like`
for `ok` (thumb-index loop with three fingers up) by editing `GESTURES` in
[build_dataset.py](build_dataset.py) *and* the `GESTURES` array in
[get_dataset.sh](get_dataset.sh), then re-running both.

---

## Setup

`datasets/` is gitignored, so the two scripts here are the reproducible source of
truth. Neither has been run yet — the download is yours to kick off.

### 1. Fetch the raw HaGRID pieces (cached, run once)

```bash
custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
```

Defaults to `HAGRID_SOURCE=subsample`: **one 1.01 GB zip**, 31,833 images across
19 classes *and* their annotations, at 384p. Verified reachable (HTTP 200,
`application/zip`, 1,009,688,861 bytes), no login required.

#### Why not upstream

`hukenovs/hagrid` only publishes full-size releases — there is no small official
download:

| Source | Size | Images | Notes |
|---|---|---|---|
| **[cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p)** | **1.01 GB** | 31,833 @ 384p | **the default**; ~1,768 per class, single zip, no auth |
| [cj-mills/hagrid-sample-500k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p) | 13.4 GB | 509,323 @ 384p | same layout; the upgrade if 30k underfits |
| [Kaggle innominate817](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p) | 1.01 GB | 31,833 @ 384p | the same zip, but needs a Kaggle login |
| [hukenovs/hagrid](https://github.com/hukenovs/hagrid) 512p | ~100 GB | 552,992 @ 512p | `HAGRID_SOURCE=full512` |
| hukenovs/hagrid FullHD | 723 GB | 552,992 @ 1080p | 18 classes |
| HaGRIDv2 FullHD | 1.5 TB | 1,086,158 @ 1080p | 33 classes |

**384p costs us nothing here** — we train at 288 px, so the source images are
already larger than the network's input. The annotations are identical; only the
JPEG resolution differs. 1,768 images per class also lands just under
`MAX_PER_CLASS = 2000`, so the cap effectively does not bite and the five classes
stay balanced at roughly 8,800 images total.

For the 512p release instead:

```bash
HAGRID_SOURCE=full512 custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
```

Its URLs are the **unverified** SberCloud paths from the upstream README, and
upstream re-hosts between releases. Every URL is HEAD-checked *before* any large
transfer, so a stale link fails in seconds and prints the override to use:

```bash
HAGRID_SUBSAMPLE_URL=<url> custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
```

`build_dataset.py` does not care which release you land on — it finds annotations
by filename and images by uid index, so v1, HaGRIDv2-512 and the subsample all
work.

Everything lands in `datasets/_hagrid_raw/` and is reused on re-runs. Safe to
delete afterwards to reclaim the space — `build_dataset.py` only reads it.

### 2. Build the YOLO dataset

```bash
python3 custom_dataset_usecases/detect/hagrid_gestures/build_dataset.py
```

Writes `datasets/detect/hagrid_gestures/{train,valid,test}/` plus `data.yaml`, and
prints the per-class counts, the skip reasons, and the box-area distribution.
Actual output (2026-08-07, 204 MB on disk, ~1 min):

```
Converting:
  palm   train  1247   valid  124
  fist   train  1248   valid  116
  peace  train  1272   valid  131
  like   train  1148   valid  129
  call   train  1264   valid  153

Skipped (no image file) : 131538
Skipped (unknown box)   : 1937
Skipped (no usable box) : 0
Skipped (over cap)      : 0
train :  6179 images,   6179 boxes
valid :   653 images,    653 boxes
```

**The 131,538 "no image file" skips are expected, not a failure.** The subsample
ships the *full* HaGRID annotations — ~28k entries per gesture — alongside only
~1,768 images per gesture. The vast majority of annotations therefore point at
images that only exist in the 723 GB release. The build resolves each annotation
against the uid index and silently drops the rest, which is exactly why it is
safe to point this script at any HaGRID release.

The 1,937 unknown-box skips are `UNKNOWN_BOX_POLICY = "skip_image"` doing its
job: images where a second, idle hand is annotated `no_gesture`.

`MAX_PER_CLASS = 2000` never binds — the subsample runs out of images first. Move
to the 500k release if you want the cap to matter.

### 3. The `.env` flow.sh insists on

`flow.sh` sources a gitignored `.env` even when there is no Roboflow call:

```bash
echo 'ROBOFLOW_API_KEY=unused_no_roboflow_project' \
  > custom_dataset_usecases/detect/hagrid_gestures/.env
```

---

## What the build script does, and the three decisions inside it

HaGRID stores images as `<gesture>/<uid>.jpg` and one JSON per gesture:

```json
"0534147c-...": {
  "bboxes": [[0.380, 0.740, 0.083, 0.091]],
  "labels": ["call"],
  "leading_hand": "right",
  "user_id": "bb138d5db2..."
}
```

Boxes are `[x_top_left, y_top_left, w, h]` normalised; YOLO wants
`[cx, cy, w, h]`, so `cx = x + w/2`. Boxes are clamped into the frame — a few
HaGRID boxes run a hair outside it, which Ultralytics rejects outright.

Three choices are exposed as constants at the top of the script:

**`UNKNOWN_BOX_POLICY = "skip_image"`** — a HaGRID image often carries a second,
idle hand annotated `no_gesture`. Keeping that image with the box dropped leaves
an *unlabelled hand* in frame, which trains the model to suppress exactly what it
is meant to find. The default drops the whole image instead; HaGRID has enough
single-hand images to absorb the loss, and the summary prints what it cost. Set
`"drop_box"` to trade cleanliness for volume.

**`VALID_EVERY = 10`, split on `user_id`** — not on filename. The same person's
hands appearing in both splits is the classic way a gesture dataset reports val
mAP it cannot reproduce on a new user. CRC32 rather than `hash()` because
Python's `hash()` is salted per process, so a filename split would not even be
stable across runs.

**`MIN_BOX_AREA = 0.0`** — no size filter, and the measured distribution says
that is the right call, but only just:

```
median 2.31%     p25 1.30%     p75 3.93%
boxes >= 1% of frame : 5653 (83%)   <- ~28 px at 288
boxes >= 3% of frame : 2542 (37%)   <- ~49 px at 288
boxes >= 5% of frame : 1138 (17%)   <- ~64 px at 288
```

This is the knob that dominated [coco_val_person](../coco_val_person/readme.md),
where 45% of person boxes were under ~29 px at 288 — unresolvable, pure label
noise that 56× more data did not fix. **HaGRID is much healthier: only 17% of
hand boxes fall under that bar**, so the filter stays off.

But note the median hand is 2.31% of frame ≈ 44 px at 288, which is *not* large.
HaGRID is arm's-length webcam framing, so if the first run's recall disappoints,
`MIN_BOX_AREA = 0.01` (drops 17%, keeps 5,653 boxes) is the cheapest experiment
before touching the model.

`MAX_PER_CLASS = 2000` keeps the five classes balanced and the run finishable.
Selection is by sorted uid, so it is deterministic.

---

## Commands

All from `helper_scripts/`. `--skip-fetch` is always required (no Roboflow
project).

**Full run — train, convert, flash, stream:**

```bash
./flow.sh --detect/hagrid_gestures --skip-fetch
```

**Train and convert, no hardware needed:**

```bash
./flow.sh --detect/hagrid_gestures --skip-fetch --skip-flash
```

**One-epoch smoke test** — proves the data loads, the head rebuilds for `nc=5`,
and the labels parse, in a couple of minutes:

```bash
SMOKE=1 ./flow.sh --detect/hagrid_gestures --skip-fetch --skip-convert --skip-flash
```

**Re-convert existing weights** (e.g. after changing `IMGSZ`):

```bash
./flow.sh --detect/hagrid_gestures --skip-fetch --skip-train
```

**Flash + stream an already-converted model** (hold BOOT, press RESET when
prompted):

```bash
./flow.sh --detect/hagrid_gestures --skip-fetch --skip-train --skip-convert
```

Artifacts land in `datasets/detect/hagrid_gestures/`: `runs/train/weights/best.pt`,
`runs/train/results.csv`, `runs/train/confusion_matrix.png`, and
`convert/yolo26_int8_vela.tflite`.

---

## Training configuration

Set in [config.env](config.env):

| Setting | Value | Why |
|---|---|---|
| `MODEL` | `yolo26n.pt` | selects the YOLO26 convert pipeline (`reg_max==1`, 6 raw heads, LiteRT) |
| `IMGSZ` | `288` | must match convert and be a multiple of 32; 709 KiB of the 1100 KiB arena, 0 CPU-fallback ops, ~11.4 FPS on WE2 |
| `EPOCHS` | `40` | 625 steps/epoch → ~2 min/epoch on MPS → roughly a 90-minute run |
| `BATCH` | `16` | matches every other use case, keeps the step-count comparison honest |
| `FREEZE` | *unset* | 10k images is enough for a full fine-tune; freezing is a small-data defence |
| `SCORE_THRESHOLD` | `40` | five classes compete for one hand, so per-class confidence runs lower than single-class |

`LABELS` is intentionally absent — `flow.sh` derives it from `data.yaml` in class
index order. **Never reorder `GESTURES` after training a model**: the firmware
indexes classes positionally, so a reorder silently relabels every detection on
device.

---

## Expected first-run checks

The pipeline was validated offline against a synthetic HaGRID-shaped fixture
(same directory layout, same annotation schema) before this was committed, so the
build → `data.yaml` → `train_model.py` path is known to work. A `SMOKE=1` run on
real data should reproduce these lines:

```
ultralytics.nn.modules.head.Detect  [5, 1, True, [64, 128, 256]]
Transferred 606/708 items from pretrained weights
Labels from dataset: palm,fist,peace,like,call
train: ... images, 0 backgrounds, 0 corrupt
```

- **`Detect [5, 1, ...]`** — 5 classes and `reg_max=1`. The `1` is what makes this
  weights file convertible: the YOLO26 path needs `reg_max==1` so no DFL, hence no
  `BATCH_MATMUL`, which the WE2 Ethos-U55 cannot take.
- **`606/708`** — backbone and neck transfer, classification head rebuilt for
  `nc=5`. (`coco_val_person` reports 612/708 at `nc=1`; the difference is the head.)
- **`0 corrupt`** — the box clamping in `build_dataset.py` worked. A nonzero count,
  or a different `Labels from dataset:` line, means stop and fix the build rather
  than burn 90 minutes.

---

## Result

_Not yet run._ Fill in per the pattern in
[coco_val_person](../coco_val_person/readme.md#result): training curve, mAP50 /
mAP50-95 / P / R, the confusion matrix's verdict on `fist` vs `like`, the Vela
SRAM + FPS report, and detection quality on hardware.
