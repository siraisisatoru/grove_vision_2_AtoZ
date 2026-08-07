# grove_vision_2_AtoZ

An end-to-end pipeline for training the latest YOLO models and running inference
on the [Grove Vision AI Module V2](https://wiki.seeedstudio.com/grove_vision_ai_v2/)
(Himax HX6538 MCU + Ethos-U55 NPU): fetch a dataset → train → convert to an
NPU-accelerated int8 TFLite → flash → view the live detection stream — all
driven by one script, per use case.

---

## Repo layout

```
helper_scripts/                  the pipeline (flow.sh) and its three Python environments
  flow.sh                        fetch → train → convert → flash → stream, one command
  train_model.py / fetch_dataset.py / flash_model.py / stream_view.py
  grove_vision_fw/                pre-built firmware images for the board
custom_dataset_usecases/<name>/  per-use-case config (config.env committed, .env secrets gitignored)
  template/                      copy this to start a new use case
datasets/<name>/                 gitignored; fetched/generated data + training runs + converted models
convert_yolo_weight/             Dockerized conversion toolchain (PyTorch → int8 TFLite → Vela)
usecases/                        things built on top of the core workflow (e.g. esp32node WiFi relay)
datasheets/                      HX6538 datasheet, Grove Vision AI V2 circuit diagram
```

## Dependencies

- **Hardware:** a Grove Vision AI Module V2 connected over USB, in BOOT mode for flashing
  (hold BOOT, press RESET).
- **Docker Desktop** (`--platform linux/amd64`) for the conversion step — PyTorch export,
  int8 quantization, and the Vela NPU compiler all run in a container so the host only needs
  Python.
- **Python 3** on the host. `flow.sh` manages three isolated environments itself, nothing to
  set up by hand beyond having `python3` on PATH:
  - `.venv` — device tools (`roboflow`, `pyserial`, `xmodem`, `opencv-python`, Flask/OpenCV
    stream viewer). See [helper_scripts/requirements.txt](helper_scripts/requirements.txt).
  - `.venv_train` — training only (`torch`, `torchvision`, `ultralytics>=8.3.209`), created
    on first run (~2 GB, Apple Silicon MPS-capable). See
    [helper_scripts/requirements_train.txt](helper_scripts/requirements_train.txt).
  - Docker — conversion only, defined by [convert_yolo_weight/Dockerfile.yolo26](convert_yolo_weight/Dockerfile.yolo26).
- A [Roboflow](https://roboflow.com/) account/API key for use cases backed by a Roboflow
  project (not needed for use cases that build their own dataset, e.g. `hagrid_gestures`).

## Firmware

The board needs the firmware from
[siraisisatoru/sscma-example-we2](https://github.com/siraisisatoru/sscma-example-we2)
(a fork of [Seeed-Studio/sscma-example-we2](https://github.com/Seeed-Studio/sscma-example-we2)
adding YOLO26 support) flashed before any model will run correctly:

- `helper_scripts/grove_vision_fw/grove_vision_ai_v2_20250102.img` — stock firmware from the
  upstream Seeed release, YOLO11-era.
- `helper_scripts/grove_vision_fw/yolo26_support.img` — build this yourself from
  `siraisisatoru/sscma-example-we2` using its Docker build method
  (`./quick_rebuild.sh`, see that repo's README) to get YOLO26 decoding support. **Required**
  for any `--yolo26` use case in this repo.

Flash a firmware image the normal way (xmodem, board in BOOT mode) before flashing a model.

## Quickstart

```bash
cd helper_scripts
./flow.sh --<use-case> [--yolo26]
```

This runs all five steps: fetch dataset → train → convert → flash → open the live stream
view. See [flow.sh](helper_scripts/flow.sh) for the full flag list
(`--skip-fetch`, `--skip-train`, `--skip-convert`, `--skip-flash`, `--rebuild-docker`).

Two concrete examples that don't need a Roboflow project (self-contained datasets):

```bash
# Stock COCO-80 weights, no training — sanity-checks the whole toolchain
./flow.sh --coco_yolo26 --yolo26 --skip-fetch --skip-train

# 5-class hand-gesture detector built from HaGRID
./flow.sh --hagrid_gestures --yolo26 --skip-fetch
```

Each use case under `custom_dataset_usecases/` has its own `readme.md` with the specifics
(dataset source, label set, resolution rationale, measured results) — start there for details
on a given model. `custom_dataset_usecases/template/` is the starting point for a new one:
copy it, fill in `config.env` (Roboflow project, `MODEL`, `IMGSZ`, `EPOCHS`) and `.env`
(`ROBOFLOW_API_KEY`, or a placeholder if the use case doesn't use Roboflow), then run
`./flow.sh --<name>` — no changes to `flow.sh` itself are needed.

### Notes on resolution (`IMGSZ`)

`IMGSZ` must be a multiple of 32 and drives arena/SRAM usage roughly with the square of the
value. 288 is the sweet spot used by most use cases here (709 KiB of the 1100 KiB arena,
~11 FPS); 320 is the practical ceiling. See
[custom_dataset_usecases/coco_yolo26/readme.md](custom_dataset_usecases/coco_yolo26/readme.md)
for the measured arena/FPS table across sizes.

## Beyond the core workflow

[usecases/](usecases/README.md) holds things built on top of this pipeline, e.g.
[esp32node](usecases/esp32node/README.md) — replacing the USB cable with a WiFi link (XIAO
ESP32-S3 → Raspberry Pi) so detections and video stream over WiFi instead of USB, verified
working end-to-end on real hardware.
