#!/usr/bin/env bash
# flow.sh — full pipeline: fetch → train → convert → flash → stream
#
# Usage:
#   ./flow.sh --face                                       flash face-recognition
#                                                            fw + models, open the
#                                                            recognition viewer
#                                                            (--skip-flash to just view;
#                                                             --gallery PATH for names)
#   ./flow.sh --detect/<use-case>                          run all 5 steps
#   ./flow.sh --detect/deer --skip-fetch                   skip dataset download
#   ./flow.sh --detect/deer --skip-fetch --skip-train      convert/flash/stream only
#   ./flow.sh --detect/deer --skip-convert --skip-flash    re-run stream view against
#                                                            an already-flashed model
#   ./flow.sh --detect/deer --rebuild-docker               force rebuild the Docker image
#   ./flow.sh --classify/<use-case>                        image classification instead of
#                                                            detection — see "Task" below
#
# Flags:
#   --detect/<use-case>    required (in one of these two forms); selects
#   --classify/<use-case>  custom_dataset_usecases/{detect,classify}/<use-case>/
#   --firmware PATH     firmware image path (currently parsed but unused downstream)
#   --skip-fetch        step 1: skip Roboflow dataset download
#   --skip-train        step 2: skip training, reuse existing best.pt
#   --skip-convert      step 3: skip conversion, reuse existing Vela .tflite output
#   --skip-flash        step 4: skip flashing the model to the board
#   --rebuild-docker    force-rebuild the conversion Docker image before step 3
#
# Task: detect vs classify is selected by which of --detect/<name> or
# --classify/<name> you pass (which also selects the directory the use case
# lives in — custom_dataset_usecases/detect/ or custom_dataset_usecases/classify/,
# and likewise datasets/detect/ or datasets/classify/ for its output). classify
# uses a completely different dataset shape (ImageFolder, train/<class>/*.jpg,
# no bbox labels) and a simpler convert/stream path — see
# custom_dataset_usecases/classify/template_classify/.
#
# Architecture is selected by MODEL= in the use case's config.env, not a CLI
# flag — it's a property of the use case, same as TASK:
#   MODEL=yolo11n.pt / yolo26n.pt / yolo11n-cls.pt / yolo26n-cls.pt   Ultralytics
#     checkpoint — routes through train_model.py + convert_tflite.py, exact
#     architecture (YOLO11 vs YOLO26, DFL vs no-DFL) auto-detected from the
#     checkpoint itself.
#   MODEL=<anything else, e.g. tiny_dscnn>   a hand-rolled architecture — routes
#     through custom_cnn/<MODEL>/train.py + export_tflite.py instead (see
#     custom_cnn/README.md). Classify-only for now.
# Every output path is namespaced by MODEL_KEY (MODEL with any .pt suffix
# stripped) — runs/$MODEL_KEY/ and the converted Vela file's name — so
# retraining the same use case with a different MODEL never overwrites a
# previous architecture's results.
#
# Adding a new use-case: create
# custom_dataset_usecases/{detect,classify}/<name>/{config.env,.env} then just
# run ./flow.sh --detect/<name> (or --classify/<name>) — no changes to this
# script needed.
#
# Three isolated Python environments:
#   .venv        — device tools (flash, stream, fetch via roboflow)
#   .venv_train  — training only (torch + official ultralytics, MPS-capable);
#                  not needed for custom_cnn architectures, which train inside
#                  the Docker image instead (see below)
#   Docker       — conversion (amd64 TF + Vela + official ultralytics + LiteRT;
#                  grove_vision_convert / Dockerfile, used for BOTH YOLO11 and
#                  YOLO26 — see the comment above the docker build/run calls in
#                  Step 3 for why — AND for custom_cnn training/export, since
#                  the same image already has TensorFlow/Keras + Vela)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_DIR="$REPO_ROOT/convert_yolo_weight"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
PIP="$SCRIPT_DIR/.venv/bin/pip"
PYTHON_TRAIN="$SCRIPT_DIR/.venv_train/bin/python"
PIP_TRAIN="$SCRIPT_DIR/.venv_train/bin/pip"

# ── Parse args ─────────────────────────────────────────────────────────────────
TASK=""
USE_CASE=""
SKIP_FETCH=0
SKIP_TRAIN=0
SKIP_CONVERT=0
SKIP_FLASH=0
REBUILD_DOCKER=0
FIRMWARE_IMG=""
FACE=0
GALLERY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-fetch)      SKIP_FETCH=1 ;;
    --skip-train)      SKIP_TRAIN=1 ;;
    --skip-convert)    SKIP_CONVERT=1 ;;
    --skip-flash)      SKIP_FLASH=1 ;;
    --rebuild-docker)  REBUILD_DOCKER=1 ;;
    --firmware)        FIRMWARE_IMG="$2"; shift ;;
    --face)            FACE=1 ;;
    --gallery)         GALLERY="$2"; shift ;;
    --*)
      _rest="${1#--}"
      if [[ "$_rest" == */* ]]; then
        TASK="${_rest%%/*}"
        USE_CASE="${_rest#*/}"
      else
        echo "Error: the use-case flag must include the task, e.g. --detect/${_rest} or --classify/${_rest}"
        exit 1
      fi
      ;;
    *) echo "Unknown argument: $1"
       echo "Usage: $0 --detect/<use-case>|--classify/<use-case> [--firmware path/to/fw.img] [--skip-fetch] [--skip-train] [--skip-convert] [--skip-flash] [--rebuild-docker]"
       exit 1 ;;
  esac
  shift
done

# ── Face recognition (--face) ──────────────────────────────────────────────────
# Face recognition is not a train/convert pipeline — it runs fixed SCRFD +
# MobileFaceNet models (flashed once by flash.sh --face). So --face short-circuits
# the whole use-case flow: flash the firmware + models (unless --skip-flash), then
# open the recognition viewer. Enroll faces separately; the viewer reads the
# gallery (--gallery PATH, or the default faces.db) to put names on the boxes.
# Enrollments age ("unknown" on a known face = template drift: lighting/
# appearance change dropped similarity below 0.6, measured ~0.5 after a month)
# — fix by re-enrolling: enroll.py --name <same-name> --avg 5.
if [[ $FACE -eq 1 ]]; then
  if [[ $SKIP_FLASH -eq 0 ]]; then
    echo ""
    echo "[face 1/2] Flashing face-recognition firmware + models..."
    "$SCRIPT_DIR/flash.sh" --face
  else
    echo "[face 1/2] Skipping flash (--skip-flash)"
  fi
  echo ""
  echo "[face 2/2] Opening face-recognition viewer..."
  echo "           (to enroll someone: .venv/bin/python3 enroll.py — SPACE captures,"
  echo "            name is asked in the terminal; db lives in output/face_recognition/)"
  FACE_VIEW_ARGS=(--task face)
  [[ -n "$GALLERY" ]] && FACE_VIEW_ARGS+=(--gallery "$GALLERY")
  exec "$PYTHON" "$SCRIPT_DIR/stream_view.py" "${FACE_VIEW_ARGS[@]}"
fi

[[ -z "$USE_CASE" ]] && { echo "Usage: $0 --detect/<use-case>|--classify/<use-case>  (e.g. --detect/deer, --classify/rock_paper_scissors) OR --face"; exit 1; }
[[ "$TASK" == "detect" || "$TASK" == "classify" ]] || { echo "Error: task must be 'detect' or 'classify', got '$TASK' (from --$TASK/$USE_CASE)"; exit 1; }

USE_CASE_DIR="$REPO_ROOT/custom_dataset_usecases/$TASK/$USE_CASE"
CONFIG_FILE="$USE_CASE_DIR/config.env"
SHARED_SECRETS_FILE="$REPO_ROOT/.env"
SECRETS_FILE="$USE_CASE_DIR/.env"

[[ -f "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE"; exit 1; }
if [[ ! -f "$SHARED_SECRETS_FILE" && ! -f "$SECRETS_FILE" ]]; then
  echo "Missing secrets: copy .env.example → .env at the repo root (shared across every"
  echo "use case — a Roboflow API key is tied to your account, not the individual"
  echo "project, so one key covers all of them), or copy $USE_CASE_DIR/.env.example"
  echo "→ .env for this use case only."
  exit 1
fi

# config.env first (committed defaults), then the shared repo-root .env (same
# ROBOFLOW_API_KEY for every use case), then this use case's own .env last so
# it can still override the shared value for the rare case that needs a
# different one — secrets win over config.env, and more specific wins over
# shared.
set -a
source "$CONFIG_FILE"
[[ -f "$SHARED_SECRETS_FILE" ]] && source "$SHARED_SECRETS_FILE"
[[ -f "$SECRETS_FILE" ]] && source "$SECRETS_FILE"
set +a
export USE_CASE
DATASET_DIR="$REPO_ROOT/datasets/$TASK/$USE_CASE"
export DATASET_DIR

# TASK comes from the --detect/--classify flag, not config.env — re-assert it
# here in case a use case's config.env still has a stray TASK= line, so the
# CLI-selected task always wins over anything sourced above.
export TASK

# MODEL selects both the pipeline and the architecture (see header comment).
# Default depends on TASK since detect and classify use different stock
# checkpoints. Resolved (not just a display fallback) because everything
# below — dispatch, MODEL_KEY namespacing — needs a concrete value.
if [[ -z "${MODEL:-}" ]]; then
  MODEL=$([[ "$TASK" == "classify" ]] && echo yolo11n-cls.pt || echo yolo11n.pt)
fi
export MODEL

IS_CUSTOM_ARCH=0
[[ "$MODEL" == *.pt ]] || IS_CUSTOM_ARCH=1

# Namespaces every output path (training run dir, converted Vela file) by the
# specific checkpoint/architecture, not just the use case — so retraining the
# same use case with a different MODEL never silently overwrites a previous
# architecture's results (e.g. yolo11n-cls.pt vs tiny_dscnn on the same
# classify use case).
MODEL_KEY="${MODEL%.pt}"

CUSTOM_CNN_DIR="$REPO_ROOT/custom_cnn/$MODEL"
if [[ $IS_CUSTOM_ARCH -eq 1 ]]; then
  if [[ "$TASK" != "classify" ]]; then
    echo "Error: custom architectures (MODEL=$MODEL, no .pt suffix) are classify-only for now. Use --classify/..., or pick a yolo11*.pt/yolo26*.pt checkpoint for --detect/...."
    exit 1
  fi
  [[ -d "$CUSTOM_CNN_DIR" ]] || { echo "Error: MODEL=$MODEL has no .pt suffix but $CUSTOM_CNN_DIR/ does not exist."; exit 1; }
fi

# ── Ensure device-tools venv deps (.venv) ──────────────────────────────────────
echo "[setup] Checking device-tools venv (.venv)..."
if ! "$PYTHON" -c "import roboflow" 2>/dev/null; then
  echo "  Installing roboflow..."
  "$PIP" install -q roboflow
fi

# ── Ensure training venv (.venv_train) ─────────────────────────────────────────
# Not needed for custom_cnn architectures — they train inside the Docker image
# instead (see custom_cnn/<name>/run_in_docker.sh), so skip the ~2 GB torch
# install entirely when it won't be used.
if [[ $SKIP_TRAIN -eq 0 && $IS_CUSTOM_ARCH -eq 0 ]]; then
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

# Resolve LABELS from the dataset if not explicitly set in config.env.
# classify has no data.yaml — labels come from the sorted train/<class>/ subdirs.
if [[ -z "${LABELS:-}" ]]; then
  if [[ "$TASK" == "classify" ]]; then
    if [[ -d "$DATASET_DIR/train" ]]; then
      LABELS=$("$PYTHON" "$SCRIPT_DIR/get_labels.py" --classify "$DATASET_DIR")
      [[ -n "$LABELS" ]] && echo "  Labels from dataset: $LABELS"
    fi
  else
    _yaml="$DATASET_DIR/data.yaml"
    if [[ -f "$_yaml" ]]; then
      LABELS=$("$PYTHON" "$SCRIPT_DIR/get_labels.py" "$_yaml")
      [[ -n "$LABELS" ]] && echo "  Labels from dataset: $LABELS"
    fi
  fi
fi
export LABELS

# ── Step 2: Train (.venv_train + Ultralytics, or custom_cnn's own Docker path) ─
if [[ $SKIP_TRAIN -eq 0 ]]; then
  echo ""
  echo "[2/5] Training $MODEL (task=$TASK) — epochs=${EPOCHS:-100} imgsz=${IMGSZ:-192} batch=${BATCH:-16}..."
  if [[ $IS_CUSTOM_ARCH -eq 1 ]]; then
    "$CUSTOM_CNN_DIR/run_in_docker.sh" train.py \
      --dataset-dir "$DATASET_DIR" \
      --epochs "${EPOCHS:-100}" \
      --batch-size "${BATCH:-16}" \
      --imgsz "${IMGSZ:-192}"
  else
    "$PYTHON_TRAIN" "$SCRIPT_DIR/train_model.py"
  fi
else
  echo "[2/5] Skipping train (--skip-train)"
fi

if [[ $IS_CUSTOM_ARCH -eq 1 ]]; then
  TRAINED_WEIGHTS="$DATASET_DIR/runs/$MODEL_KEY/best.keras"
else
  TRAINED_WEIGHTS="$DATASET_DIR/runs/$MODEL_KEY/weights/best.pt"
fi

# ── Step 3: Convert — Docker (amd64 TF + Vela + official ultralytics + LiteRT) ─
# convert_tflite.py names its output after MODEL_KEY (e.g.
# yolo11n-cls_int8_vela.tflite), so target the exact file. A newest-match glob
# would happily pick up a stale artifact from a previous run of a DIFFERENT
# architecture on the same use case if this run's convert step was skipped or
# failed.

# Architecture selection, auto-detected from MODEL. Must live OUTSIDE the
# convert block: with --skip-convert we still need it to pick the right
# pre-built Vela artifact. Detect-only: a classify MODEL can also start with
# "yolo26" (yolo26n-cls.pt) without meaning the YOLO26 detection pipeline.
IS_YOLO26=0
[[ "$TASK" == "detect" && "$MODEL" == yolo26* ]] && IS_YOLO26=1
IS_CLASSIFY=0
[[ "$TASK" == "classify" ]] && IS_CLASSIFY=1

# Converted weights default to living next to the training run that produced
# them ($DATASET_DIR/convert), not in convert_yolo_weight/output/. Falls back
# to the local output dir when there's no dataset directory to write into
# (e.g. --skip-fetch --skip-train on a repo with no datasets/ yet). Must also
# live OUTSIDE the convert block: with --skip-convert we still need it to
# find the pre-built Vela artifact. Not used for custom architectures — they
# write their own Vela output under runs/$MODEL_KEY/ (see custom_cnn/).
if [[ -d "$DATASET_DIR" ]]; then
  OUTPUT_DIR="$DATASET_DIR/convert"
else
  OUTPUT_DIR="$CONVERT_DIR/output"
fi
mkdir -p "$OUTPUT_DIR"

if [[ $SKIP_CONVERT -eq 0 ]]; then
  echo ""
  echo "[3/5] Converting weights to Vela TFLite (imgsz=${IMGSZ:-192})..."
  [[ -f "$TRAINED_WEIGHTS" ]] || { echo "Error: no trained weights at $TRAINED_WEIGHTS"; exit 1; }

  # Conversion image: grove_vision_convert / Dockerfile (official ultralytics +
  # Vela 5.1.0 — matches firmware tflmtag2605_u55tag2605). Every architecture,
  # Ultralytics or custom_cnn, uses this one image: onnx2tf (the old
  # kris-himax-fork path) broke the WE2 contract twice over — it emitted NHWC
  # box tensors (1,24,24,64) where Yolo11::isValid() requires channels-first
  # (1,64,24,24), and it scrambled the 6 output heads out of the interleaved
  # order the firmware indexes positionally. That is why YOLO11 loaded fine
  # and detected nothing. litert-torch traces the patched module directly and
  # preserves both layout and order for either architecture, so
  # convert_tflite.py runs both through it (see _export_litert). IS_YOLO26
  # (set above, outside this block) selects the pipeline inside
  # convert_tflite.py; it no longer selects the image.
  # Tag pinned explicitly: with Docker's containerd image store, `docker image
  # inspect <name>` (no tag) fails to resolve even when the image exists —
  # only the fully-qualified `<name>:latest` reference works.
  DOCKER_IMAGE="grove_vision_convert:latest"
  if [[ $REBUILD_DOCKER -eq 1 ]]; then
    echo "  Building Docker image $DOCKER_IMAGE (--rebuild-docker)..."
    docker build --platform linux/amd64 -f "$CONVERT_DIR/Dockerfile" -t "$DOCKER_IMAGE" "$CONVERT_DIR"
  elif ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "Error: Docker image $DOCKER_IMAGE not found. Run with --rebuild-docker to build it."
    exit 1
  else
    echo "  Using existing Docker image (use --rebuild-docker to force rebuild)."
  fi

  if [[ $IS_CUSTOM_ARCH -eq 1 ]]; then
    "$CUSTOM_CNN_DIR/run_in_docker.sh" export_tflite.py --dataset-dir "$DATASET_DIR"
  else
    # Mount the full dataset dir at /dataset so Docker can reach the images.
    # convert_tflite.py --data-dir regenerates a fresh YAML with /dataset/... paths,
    # bypassing the host-absolute paths in the stored data.yaml.
    DATASET_MOUNT_ARGS=()
    DATASET_CONVERT_ARGS=()
    if [[ -d "$DATASET_DIR" ]]; then
      DATASET_MOUNT_ARGS=(-v "$DATASET_DIR:/dataset:ro")
      DATASET_CONVERT_ARGS=(--data-dir /dataset)
    fi

    # --yolo26 selects the ENTIRE YOLO26 pipeline in convert_tflite.py, not
    # just an assertion: reg_max==1 check, the 6-way detection-head split (which
    # the WE2 Yolo26 decoder requires instead of Ultralytics' single concatenated
    # tensor), and the LiteRT export path. Without it the fully isolated YOLO11
    # path runs (reg_max==16, DFL baked in). A flag/weights mismatch is a hard
    # error rather than a silent mis-convert.
    YOLO26_ARG=()
    [[ $IS_YOLO26 -eq 1 ]] && YOLO26_ARG=(--yolo26)

    # TASK=classify selects the plain-classifier pipeline in convert_tflite.py
    # (no head-splitting, no anchor decode) instead of the YOLO11/YOLO26 branch —
    # mutually exclusive with YOLO26_ARG, enforced by convert_tflite.py itself.
    CLASSIFY_ARG=()
    [[ $IS_CLASSIFY -eq 1 ]] && CLASSIFY_ARG=(--classify)

    docker run --rm --platform linux/amd64 \
      -v "$CONVERT_DIR:/app" \
      -v "$TRAINED_WEIGHTS:/model_input/best.pt:ro" \
      -v "$OUTPUT_DIR:/output" \
      ${DATASET_MOUNT_ARGS[@]+"${DATASET_MOUNT_ARGS[@]}"} \
      "$DOCKER_IMAGE" \
      python convert_tflite.py \
        --model /model_input/best.pt \
        --imgsz "${IMGSZ:-192}" \
        --output-dir /output \
        --output-name "$MODEL_KEY" \
        ${DATASET_CONVERT_ARGS[@]+"${DATASET_CONVERT_ARGS[@]}"} \
        ${CLASSIFY_ARG[@]+"${CLASSIFY_ARG[@]}"} \
        ${YOLO26_ARG[@]+"${YOLO26_ARG[@]}"}
  fi
else
  echo "[3/5] Skipping convert (--skip-convert)"
fi

if [[ $IS_CUSTOM_ARCH -eq 1 ]]; then
  VELA_OUT="$DATASET_DIR/runs/$MODEL_KEY/${MODEL_KEY}_int8_vela.tflite"
else
  VELA_OUT="$OUTPUT_DIR/${MODEL_KEY}_int8_vela.tflite"
fi
[[ -n "$VELA_OUT" && -f "$VELA_OUT" ]] || { echo "Error: expected Vela output not found at $VELA_OUT"; exit 1; }
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
STREAM_ARGS=(--model-size "${IMGSZ:-192}" --task "$TASK")
[[ -n "${LABELS:-}" ]]           && STREAM_ARGS+=(--labels "$LABELS")
[[ -n "${SCORE_THRESHOLD:-}" ]]  && STREAM_ARGS+=(--score-threshold "$SCORE_THRESHOLD")
[[ -n "${ALGO:-}" ]]             && STREAM_ARGS+=(--algo "$ALGO")

"$PYTHON" "$SCRIPT_DIR/stream_view.py" "${STREAM_ARGS[@]}"
