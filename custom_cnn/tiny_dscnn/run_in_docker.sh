#!/usr/bin/env bash
# Run any custom_cnn/tiny_dscnn/*.py script inside the grove_vision_convert Docker
# image — it already has TensorFlow/Keras (for the int8 TFLiteConverter path) and
# Vela (convert_yolo_weight/Dockerfile), so no separate venv is needed for this
# hand-rolled-CNN proof of concept.
#
# Usage:
#   ./run_in_docker.sh train.py
#   ./run_in_docker.sh evaluate.py
#   ./run_in_docker.sh export_tflite.py
#   ./run_in_docker.sh train.py --epochs 10   # extra args pass straight through

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_IMAGE="grove_vision_convert:latest"

if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
  echo "Error: Docker image $DOCKER_IMAGE not found."
  echo "Build it first: docker build --platform linux/amd64 -t $DOCKER_IMAGE $REPO_ROOT/convert_yolo_weight"
  exit 1
fi

# Mount the whole repo at /repo. Callers outside this directory (flow.sh) pass
# --dataset-dir as a HOST absolute path (e.g. $REPO_ROOT/datasets/classify/foo)
# — rewrite any argument starting with $REPO_ROOT to start with /repo instead,
# so it still resolves once it's inside the container.
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "$REPO_ROOT"* ]]; then
    ARGS+=("/repo${arg#$REPO_ROOT}")
  else
    ARGS+=("$arg")
  fi
done

docker run --rm --platform linux/amd64 \
  -v "$REPO_ROOT:/repo" \
  -w /repo/custom_cnn/tiny_dscnn \
  "$DOCKER_IMAGE" \
  python3 "${ARGS[@]}"
