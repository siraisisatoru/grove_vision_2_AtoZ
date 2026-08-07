#!/usr/bin/env bash
# flow.sh — full pipeline: fetch → train → convert → flash → stream
#
# Usage:
#   ./flow.sh --<use-case>                               run all 5 steps (YOLO11 by default)
#   ./flow.sh --deer --yolo26                             run the YOLO26 pipeline instead
#   ./flow.sh --deer --skip-fetch                        skip dataset download
#   ./flow.sh --deer --skip-fetch --skip-train           convert/flash/stream only
#   ./flow.sh --deer --skip-convert --skip-flash          re-run stream view against
#                                                          an already-flashed model
#   ./flow.sh --deer --rebuild-docker                    force rebuild the Docker image
#
# Flags:
#   --<use-case>       required; selects custom_dataset_usecases/<use-case>/
#   --yolo26            use yolo26n.pt + YOLO26 convert pipeline (default: YOLO11)
#   --firmware PATH     firmware image path (currently parsed but unused downstream)
#   --skip-fetch        step 1: skip Roboflow dataset download
#   --skip-train        step 2: skip training, reuse existing best.pt
#   --skip-convert      step 3: skip conversion, reuse existing Vela .tflite output
#   --skip-flash        step 4: skip flashing the model to the board
#   --rebuild-docker    force-rebuild the conversion Docker image before step 3
#
# Adding a new use-case: create custom_dataset_usecases/<name>/{config.env,.env}
# then just run ./flow.sh --<name> — no changes to this script needed.
#
# Three isolated Python environments:
#   .venv        — device tools (flash, stream, fetch via roboflow)
#   .venv_train  — training only (torch + official ultralytics, MPS-capable)
#   Docker       — conversion (amd64 TF + Vela + kris-himax fork; grove_vision_convert_yolo26
#                  /Dockerfile.yolo26 is now used for BOTH YOLO11 and YOLO26 — see the
#                  comment above the docker build/run calls in Step 3 for why)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_DIR="$REPO_ROOT/convert_yolo_weight"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
PIP="$SCRIPT_DIR/.venv/bin/pip"
PYTHON_TRAIN="$SCRIPT_DIR/.venv_train/bin/python"
PIP_TRAIN="$SCRIPT_DIR/.venv_train/bin/pip"

# ── Parse args ─────────────────────────────────────────────────────────────────
USE_CASE=""
SKIP_FETCH=0
SKIP_TRAIN=0
SKIP_CONVERT=0
SKIP_FLASH=0
REBUILD_DOCKER=0
MODEL_FLAG=""
FIRMWARE_IMG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-fetch)      SKIP_FETCH=1 ;;
    --skip-train)      SKIP_TRAIN=1 ;;
    --skip-convert)    SKIP_CONVERT=1 ;;
    --skip-flash)      SKIP_FLASH=1 ;;
    --rebuild-docker)  REBUILD_DOCKER=1 ;;
    --yolo26)          MODEL_FLAG=yolo26 ;;
    --firmware)        FIRMWARE_IMG="$2"; shift ;;
    --*)               USE_CASE="${1#--}" ;;
    *) echo "Unknown argument: $1"
       echo "Usage: $0 --<use-case> [--yolo26] [--firmware path/to/fw.img] [--skip-fetch] [--skip-train] [--skip-convert] [--skip-flash] [--rebuild-docker]"
       exit 1 ;;
  esac
  shift
done

[[ -z "$USE_CASE" ]] && { echo "Usage: $0 --<use-case> [--yolo26]  (e.g. --deer --yolo26)"; exit 1; }

USE_CASE_DIR="$REPO_ROOT/custom_dataset_usecases/$USE_CASE"
CONFIG_FILE="$USE_CASE_DIR/config.env"
SECRETS_FILE="$USE_CASE_DIR/.env"

[[ -f "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE"; exit 1; }
[[ -f "$SECRETS_FILE" ]] || { echo "Missing secrets: $SECRETS_FILE  (copy template/.env.example → .env and fill in your API key)"; exit 1; }

# config.env first (committed defaults), .env second (secrets win on conflict)
set -a
source "$CONFIG_FILE"
source "$SECRETS_FILE"
set +a
export USE_CASE
DATASET_DIR="$REPO_ROOT/datasets/$USE_CASE"
export DATASET_DIR

# ── Model family flag overrides (applied after config so they always win) ──────
if [[ "$MODEL_FLAG" == "yolo26" ]]; then
  MODEL="yolo26n.pt"
  ROBOFLOW_FORMAT="${ROBOFLOW_FORMAT:-yolo26}"
  export MODEL ROBOFLOW_FORMAT
fi

# ── Ensure device-tools venv deps (.venv) ──────────────────────────────────────
echo "[setup] Checking device-tools venv (.venv)..."
if ! "$PYTHON" -c "import roboflow" 2>/dev/null; then
  echo "  Installing roboflow..."
  "$PIP" install -q roboflow
fi

# ── Ensure training venv (.venv_train) ─────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
  if [[ ! -x "$PYTHON_TRAIN" ]]; then
    echo "[setup] Creating training venv (.venv_train) — first time only, torch is ~2 GB..."
    python3 -m venv "$SCRIPT_DIR/.venv_train"
    "$PIP_TRAIN" install --upgrade pip -q
    "$PIP_TRAIN" install -r "$SCRIPT_DIR/requirements_train.txt"
  else
    echo "[setup] Training venv (.venv_train) ready."
  fi
fi

echo ""
echo "======================================================"
echo " grove_vision_2_AtoZ  >  $USE_CASE"
echo "======================================================"

# ── Step 1: Fetch dataset (.venv) ──────────────────────────────────────────────
if [[ $SKIP_FETCH -eq 0 ]]; then
  echo ""
  echo "[1/5] Fetching dataset from Roboflow..."
  "$PYTHON" "$SCRIPT_DIR/fetch_dataset.py"
else
  echo "[1/5] Skipping fetch (--skip-fetch)"
fi

# Resolve LABELS from data.yaml if not explicitly set in config.env
if [[ -z "${LABELS:-}" ]]; then
  _yaml="$DATASET_DIR/data.yaml"
  if [[ -f "$_yaml" ]]; then
    LABELS=$("$PYTHON" "$SCRIPT_DIR/get_labels.py" "$_yaml")
    [[ -n "$LABELS" ]] && echo "  Labels from dataset: $LABELS"
  fi
fi
export LABELS

# ── Step 2: Train (.venv_train — official ultralytics + MPS) ──────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
  echo ""
  echo "[2/5] Training ${MODEL:-yolo11n.pt} — epochs=${EPOCHS:-100} imgsz=${IMGSZ:-192} batch=${BATCH:-16}..."
  "$PYTHON_TRAIN" "$SCRIPT_DIR/train_model.py"
else
  echo "[2/5] Skipping train (--skip-train)"
fi

TRAINED_PT="$DATASET_DIR/runs/train/weights/best.pt"

# ── Step 3: Convert — Docker (amd64 TF + Vela + kris-himax fork) ──────────────
# convert_tflite.py names its output deterministically per architecture
# (yolo26_int8_vela.tflite / yolo11_int8_vela.tflite), so target the exact file.
# The old newest-match glob would happily pick up a stale artifact from a previous
# run of the OTHER architecture if this run's convert step was skipped or failed.

# Architecture selection. Must live OUTSIDE the convert block: with
# --skip-convert we still need it to pick the right pre-built Vela artifact.
# Auto-detected from the MODEL name, or forced with --yolo26.
IS_YOLO26=0
if [[ "$MODEL_FLAG" == "yolo26" ]] || [[ "${MODEL:-}" == yolo26* ]]; then
  IS_YOLO26=1
fi

# Converted weights default to living next to the training run that produced
# them ($DATASET_DIR/convert), not in convert_yolo_weight/output/. Falls back
# to the local output dir when there's no dataset directory to write into
# (e.g. --skip-fetch --skip-train on a repo with no datasets/ yet). Must also
# live OUTSIDE the convert block: with --skip-convert we still need it to
# find the pre-built Vela artifact.
if [[ -d "$DATASET_DIR" ]]; then
  OUTPUT_DIR="$DATASET_DIR/convert"
else
  OUTPUT_DIR="$CONVERT_DIR/output"
fi
mkdir -p "$OUTPUT_DIR"

if [[ $SKIP_CONVERT -eq 0 ]]; then
  echo ""
  echo "[3/5] Converting weights to Vela TFLite (imgsz=${IMGSZ:-192})..."
  [[ -f "$TRAINED_PT" ]] || { echo "Error: no trained weights at $TRAINED_PT"; exit 1; }

  # Conversion image: grove_vision_convert_yolo26 / Dockerfile.yolo26 (official
  # ultralytics + Vela 5.1.0 — matches firmware tflmtag2605_u55tag2605).
  # BOTH architectures now use this image. YOLO11 used to go through
  # grove_vision_convert (kris-himax fork 8.3.x, onnx2tf), but onnx2tf breaks the
  # WE2 contract twice over — it emits NHWC box tensors (1,24,24,64) where
  # Yolo11::isValid() requires channels-first (1,64,24,24), and it scrambles the
  # 6 output heads out of the interleaved order the firmware indexes positionally.
  # That is why YOLO11 loaded fine and detected nothing. litert-torch traces the
  # patched module directly and preserves both layout and order for either
  # architecture, so convert_tflite.py runs both through it (see _export_litert).
  # IS_YOLO26 (set above, outside this block) selects the pipeline inside
  # convert_tflite.py; it no longer selects the image.
  DOCKER_IMAGE="grove_vision_convert_yolo26"
  DOCKER_FILE="Dockerfile.yolo26"
  if [[ $REBUILD_DOCKER -eq 1 ]]; then
    echo "  Building Docker image $DOCKER_IMAGE (--rebuild-docker)..."
    docker build --platform linux/amd64 -f "$CONVERT_DIR/$DOCKER_FILE" -t "$DOCKER_IMAGE" "$CONVERT_DIR"
  elif ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "Error: Docker image $DOCKER_IMAGE not found. Run with --rebuild-docker to build it."
    exit 1
  else
    echo "  Using existing Docker image (use --rebuild-docker to force rebuild)."
  fi

  # Mount the full dataset dir at /dataset so Docker can reach the images.
  # convert_tflite.py --data-dir regenerates a fresh YAML with /dataset/... paths,
  # bypassing the host-absolute paths in the stored data.yaml.
  DATASET_MOUNT_ARGS=()
  DATASET_CONVERT_ARGS=()
  if [[ -d "$DATASET_DIR" ]]; then
    DATASET_MOUNT_ARGS=(-v "$DATASET_DIR:/dataset:ro")
    DATASET_CONVERT_ARGS=(--data-dir /dataset)
  fi

  # Pass --yolo26 down whenever the yolo26 image was selected above, so
  # --yolo26 now selects the ENTIRE YOLO26 pipeline in convert_tflite.py, not
  # just an assertion: reg_max==1 check, the 6-way detection-head split (which
  # the WE2 Yolo26 decoder requires instead of Ultralytics' single concatenated
  # tensor), and the LiteRT export path. The old separate --no-post flag is gone
  # — it is implied. Without the flag the fully isolated YOLO11 path runs
  # (reg_max==16, DFL baked in, onnx2tf). A flag/weights mismatch is a hard
  # error rather than a silent mis-convert.
  YOLO26_ARG=()
  [[ $IS_YOLO26 -eq 1 ]] && YOLO26_ARG=(--yolo26)

  docker run --rm --platform linux/amd64 \
    -v "$CONVERT_DIR:/app" \
    -v "$TRAINED_PT:/model_input/best.pt:ro" \
    -v "$OUTPUT_DIR:/output" \
    ${DATASET_MOUNT_ARGS[@]+"${DATASET_MOUNT_ARGS[@]}"} \
    "$DOCKER_IMAGE" \
    python convert_tflite.py \
      --model /model_input/best.pt \
      --imgsz "${IMGSZ:-192}" \
      --output-dir /output \
      ${DATASET_CONVERT_ARGS[@]+"${DATASET_CONVERT_ARGS[@]}"} \
      ${YOLO26_ARG[@]+"${YOLO26_ARG[@]}"}
else
  echo "[3/5] Skipping convert (--skip-convert)"
fi

VELA_ARCH=$([[ $IS_YOLO26 -eq 1 ]] && echo yolo26 || echo yolo11)
VELA_OUT="$OUTPUT_DIR/${VELA_ARCH}_int8_vela.tflite"
[[ -n "$VELA_OUT" && -f "$VELA_OUT" ]] || { echo "Error: expected Vela output not found in $OUTPUT_DIR/"; exit 1; }
echo "  Vela output: $VELA_OUT"

# ── Step 4: Flash (.venv) ──────────────────────────────────────────────────────
if [[ $SKIP_FLASH -eq 0 ]]; then
  echo ""
  echo "[4/5] Flashing model to board at 0x400000..."
  echo "      ACTION NEEDED: hold BOOT and press RESET on the device now."
  "$PYTHON" "$SCRIPT_DIR/flash_model.py" \
    --model-file "$VELA_OUT" \
    --address 0x400000
else
  echo "[4/5] Skipping flash (--skip-flash)"
fi

# ── Step 5: Stream view (.venv) ────────────────────────────────────────────────
echo ""
echo "[5/5] Opening stream view..."
STREAM_ARGS=(--model-size "${IMGSZ:-192}")
[[ -n "${LABELS:-}" ]]           && STREAM_ARGS+=(--labels "$LABELS")
[[ -n "${SCORE_THRESHOLD:-}" ]]  && STREAM_ARGS+=(--score-threshold "$SCORE_THRESHOLD")
[[ -n "${ALGO:-}" ]]             && STREAM_ARGS+=(--algo "$ALGO")

"$PYTHON" "$SCRIPT_DIR/stream_view.py" "${STREAM_ARGS[@]}"
