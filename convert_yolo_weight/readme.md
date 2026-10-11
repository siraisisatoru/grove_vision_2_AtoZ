# convert_yolo_weight

Dockerized conversion toolchain: PyTorch `.pt` -> int8 TFLite -> Vela, for both
the YOLO11 and YOLO26 pipelines used by `helper_scripts/flow.sh`. Normally you
don't invoke this directly — `flow.sh` builds the image on first use (or with
`--rebuild-docker`) and runs `convert_tflite.py` inside it as part of Step 3.
See the per-use-case readmes under `custom_dataset_usecases/{detect,classify}/` for
verified end-to-end results (e.g. `detect/coco_yolo26`, `detect/hagrid_gestures`,
`classify/rock_paper_scissors`).

Manual build/run, for debugging the conversion step in isolation:

```sh
docker build --platform linux/amd64 -t grove_vision_convert .
```

```sh
docker run --platform linux/amd64 \
    -v $(pwd):/app \
    grove_vision_convert \
    python convert_tflite.py --model best.pt --imgsz 288 [--yolo26]
```

`--yolo26` must match the checkpoint's architecture (`reg_max==1` for YOLO26,
`reg_max==16` for YOLO11) — `convert_tflite.py` asserts this and refuses to
convert on a mismatch. See the module docstring in
[convert_tflite.py](convert_tflite.py) for the full pipeline details.
