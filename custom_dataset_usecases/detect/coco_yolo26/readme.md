# coco_yolo26 — stock `yolo26n.pt`, no training

A feasibility experiment, not a usable model. It takes the **pretrained
`yolo26n.pt` straight from Ultralytics** (COCO, 80 classes) and pushes it through
the normal convert → flash → stream path, to answer: can raw upstream weights be
Vela-compiled and run on the Grove board at all, without any fine-tuning?

**Answer: yes, the toolchain handles it end-to-end with zero code changes — but
the detections are bad.** See [Result](#result) before you build anything on top
of this.

This use case is special in three ways:

1. **No Roboflow project.** `datasets/detect/coco_yolo26/` is assembled by hand from the
   Ultralytics coco128 release asset. `--skip-fetch` is mandatory.
2. **No training.** `datasets/detect/coco_yolo26/runs/yolo26n/weights/best.pt` is a *copy
   of the stock `yolo26n.pt`*, standing in for the trained weights `flow.sh`
   expects (`runs/<model-key>/`, where `<model-key>` is `MODEL=yolo26n.pt` with
   the `.pt` stripped). `--skip-train` is mandatory — a training run would
   overwrite it with a 128-image COCO fit, which is far worse than the stock
   weights.
3. **The dataset is calibration data only.** Those 128 images are used solely for
   INT8 quantization calibration, never for learning.

---

## One-time setup

`datasets/` is gitignored, so the dataset directory has to be recreated after a
fresh clone. From the repo root:

```bash
# 1. Fetch the Ultralytics coco128 asset (7 MB, 128 images, all 80 classes)
curl -sL -o /tmp/coco128.zip \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip -q -o /tmp/coco128.zip -d /tmp

# 2. Lay it out the way flow.sh expects. valid/ is what the converter reads for
#    INT8 calibration; train/ is present only so the generated data.yaml is
#    well-formed.
D=datasets/detect/coco_yolo26
mkdir -p $D/train/images $D/train/labels $D/valid/images $D/valid/labels \
         $D/test/images $D/runs/yolo26n/weights
cp /tmp/coco128/images/train2017/*.jpg $D/train/images/
cp /tmp/coco128/labels/train2017/*.txt $D/train/labels/
cp /tmp/coco128/images/train2017/*.jpg $D/valid/images/
cp /tmp/coco128/labels/train2017/*.txt $D/valid/labels/

# 3. The "trained" weights = the stock checkpoint, unmodified
cp helper_scripts/yolo26n.pt $D/runs/yolo26n/weights/best.pt

# 4. data.yaml with the 80 COCO names in class-index order. flow.sh parses
#    names: [...] out of this to populate LABELS for the stream view.
helper_scripts/.venv_train/bin/python - <<'EOF'
from pathlib import Path
from ultralytics import YOLO
names = YOLO('helper_scripts/yolo26n.pt').model.names
names = [names[i] for i in range(len(names))]
d = Path('datasets/detect/coco_yolo26').resolve()
(d / 'data.yaml').write_text(
    f"path: {d}\n"
    f"train: {d/'train'/'images'}\n"
    f"val: {d/'valid'/'images'}\n"
    f"test: {d/'test'/'images'}\n"
    f"nc: {len(names)}\n"
    f"names: [{', '.join(names)}]\n"
)
EOF
```

You also need `custom_dataset_usecases/detect/coco_yolo26/.env` (gitignored, required by
`flow.sh` even though no Roboflow call is made):

```bash
echo 'ROBOFLOW_API_KEY=unused_no_roboflow_project' \
  > custom_dataset_usecases/detect/coco_yolo26/.env
```

---

## Commands

All from `helper_scripts/`. **`--skip-fetch --skip-train` are always required**
(see above); omitting them either hits an empty Roboflow config or destroys the
stock weights.

**Convert only** (produces `datasets/detect/coco_yolo26/convert/yolo26_int8_vela.tflite`;
step 5 will report "No Grove Vision AI V2 found" if no board is attached, which is
harmless):

```bash
./flow.sh --detect/coco_yolo26 --skip-fetch --skip-train --skip-flash
```

**Flash an already-converted model and view the stream** (hold BOOT, press RESET
when prompted):

```bash
./flow.sh --detect/coco_yolo26 --skip-fetch --skip-train --skip-convert
```

**Everything (convert + flash + stream):**

```bash
./flow.sh --detect/coco_yolo26 --skip-fetch --skip-train
```

Add `--rebuild-docker` if `grove_vision_convert` doesn't exist yet. Note
that the image check asks the Docker daemon directly — if Docker Desktop is still
starting up, the check can report the image as missing when it does exist. Open
the Docker UI, wait for the engine, and retry before rebuilding.

The board must already be running the `yolo26_support.img` firmware
(`helper_scripts/grove_vision_fw/`).

---

## Why `IMGSZ=288` and not 300

`Yolo26::isValid()` in the firmware rejects any input whose height is not a
multiple of 32, and the three detection grids are computed as `w>>3, w>>4, w>>5`.
300 is not a valid size. 288 is the nearest one that is (grids 36 / 18 / 9).

The upper bound is the 1100 KiB static tensor arena
(`MA_ENGINE_TFLITE_TENSOE_ARENA_SIZE`, `ma_config_board.h`). Peak co-resident
tensors are the input (CPU-written, fully resident) plus the stem conv output,
both scaling with `IMGSZ²`:

| IMGSZ | input | stem out | Vela SRAM | inference |
|---|---|---|---|---|
| 192 | 108 KB | 144 KB | 322 KiB (measured, 1-class model) | 46 ms |
| **288** | 243 KB | 324 KB | **709 KiB (measured, this model)** | **88 ms** |
| 320 | 300 KB | 400 KB | ~890 KiB (est.) | ~129 ms |
| 384 | 432 KB | 576 KB | ~1280 KiB — **over budget** | — |

320 is the practical maximum. **Vela is handed a 4 MB arena budget by
`himax_vela.ini` and does not enforce the real 1100 KiB**, so an over-budget model
compiles cleanly and then corrupts the camera framebuffer or the FreeRTOS heap on
device. Always check `sram_memory_used` in
`datasets/detect/coco_yolo26/convert/yolo26_int8_summary_My_Sys_Cfg.csv` before flashing.

---

## Result

Conversion is clean and the model runs on-device:

```
Export verification passed.
CPU operators = 0 (0.0%)      NPU operators = 427 (100.0%)
Total SRAM used                708.75 KiB   (of 1100 KiB arena)
Total Off-chip Flash used     2200.27 KiB   (vela .tflite = 2.42 MB)
inference_time  87.8 ms  ->  11.4 FPS @ 400 MHz
```

Exported boundary — int8 throughout, NCHW, 6 interleaved heads with `nc=80`,
exactly what the WE2 decoder indexes positionally:

```
input   int8 [1,3,288,288]
output0 int8 [1, 4,36,36]  box @ stride 8     output1 int8 [1,80,36,36]  cls
output2 int8 [1, 4,18,18]  box @ stride 16    output3 int8 [1,80,18,18]  cls
output4 int8 [1, 4, 9, 9]  box @ stride 32    output5 int8 [1,80, 9, 9]  cls
```

Two useful findings:

- **0 CPU ops.** The earlier poker-card model reported 98.7% NPU with 4
  `BATCH_MATMUL` + 8 `TRANSPOSE` left on the CM55. Under official ultralytics +
  LiteRT + Vela 5.1.0, C2PSA's `BATCH_MATMUL` maps to the NPU, so the
  `MA_TFLITE_OP_BATCH_MATMUL` firmware addition is no longer load-bearing
  (harmless to keep).
- **Class count costs almost nothing.** 80 classes vs 1 adds only ~83 K
  parameters (the `cv3` 1×1 heads). Flash and arena are driven by `IMGSZ`, not
  `nc`.

### Detection quality: confirmed poor

**Verified on hardware: detections are bad.** The camera frame is downscaled to
288×288 and is visibly pixelated, and `yolo26n.pt` was trained at 640 — so the
weights are being asked to work at a resolution and level of detail they never
saw. Small and medium COCO objects are missed outright; only large, close,
high-contrast subjects register at all, and those with low confidence (hence
`SCORE_THRESHOLD=25` in `config.env`).

This is a resolution/domain mismatch, not a pipeline bug — the conversion,
quantization, and on-device decode all check out. To get a usable COCO detector:

1. **Fine-tune at the deploy resolution.** Train `yolo26n.pt` on COCO (or just
   the classes you care about) with `imgsz=288`, then convert the resulting
   `best.pt`. This is the fix that actually matters; everything else is marginal.
2. **Raise `IMGSZ` to 320** for a modest detail gain (~890 KiB arena, ~7.7 FPS).
   Helps, but does not close the gap to 640.
3. **Cut the class count.** 80 classes over a pixelated 288×288 frame spreads
   confidence thin. A handful of classes trained at the deploy size performs far
   better — which is what every other use case in this directory does.
