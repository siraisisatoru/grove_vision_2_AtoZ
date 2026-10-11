# template_classify — starting point for an image-classification use case

Copy this directory to start a new **classification** use case, the way
`custom_dataset_usecases/detect/template/` is the starting point for a detection
one. The difference is living under `custom_dataset_usecases/classify/` (and being
run via `--classify/<name>` instead of `--detect/<name>`), which switches the
whole `flow.sh` pipeline onto a simpler path:

| | detect (`detect/template/`) | classify (this template) |
|---|---|---|
| Dataset shape | YOLO bbox labels + `data.yaml` | `train/<class>/*.jpg` (ImageFolder), no labels file |
| Roboflow export format | `yolov8` / `yolo26` | `folder` |
| Base checkpoint | `yolo11n.pt` / `yolo26n.pt` | `yolo11n-cls.pt` / `yolo26n-cls.pt` |
| Convert pipeline | head-split + anchor decode (`convert_tflite.py --yolo26`) | plain int8 export, no head-split (`convert_tflite.py --classify`) |
| Firmware decoder | `Yolo11`/`Yolo26` (`ma_model_yolo11.cpp` etc.) | `Classifier` (`ma_model_classifier.cpp`, `MA_MODEL_TYPE_IMCLS`) — already in every existing firmware build, no rebuild needed |
| Stream view | draws boxes | overlays top-3 class + score (`stream_view.py --task classify`) |

## Setup

1. Copy this directory: `cp -r custom_dataset_usecases/classify/template_classify custom_dataset_usecases/classify/<name>`
2. Fill in `config.env`: `ROBOFLOW_WORKSPACE`/`ROBOFLOW_PROJECT`/`ROBOFLOW_VERSION`
   for a Roboflow **Classification** project, or leave as-is and use `SMOKE=1`
   for an offline sanity check against a synthetic dataset first.
3. Copy `.env.example` → `.env` and fill in `ROBOFLOW_API_KEY` (skip if using `SMOKE=1`).
4. Run:
   ```bash
   cd helper_scripts
   ./flow.sh --classify/<name>                 # real Roboflow dataset
   ./flow.sh --classify/<name> --skip-flash     # offline smoke test (set SMOKE=1 in .env), no hardware/API key needed
   ```

Architecture is picked via `MODEL=` in `config.env`, not a CLI flag —
`yolo11n-cls.pt` or `yolo26n-cls.pt` for the Ultralytics path (auto-detected from
the checkpoint), or the name of a directory under `custom_cnn/` (e.g.
`tiny_dscnn`) for a hand-rolled architecture — see
[custom_cnn/README.md](../../../custom_cnn/README.md). Every output is
namespaced by `MODEL` (with any `.pt` stripped), so retraining this use case with
a different `MODEL` never overwrites a previous architecture's results.

No changes to `flow.sh`, `train_model.py`, `convert_tflite.py`, or
`stream_view.py` are needed for a new classification use case — they all key
off the task derived from `--classify/<name>` (i.e. from living under this
`custom_dataset_usecases/classify/` directory).
