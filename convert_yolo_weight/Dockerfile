# Conversion image: PyTorch .pt -> int8 TFLite -> Vela, for both the YOLO11
# and YOLO26 pipelines in convert_tflite.py (the LiteRT export path is shared;
# --yolo26 only changes which architecture-specific branch runs). Key pinning
# decisions:
#
#   torch==2.6.0          — inherited from requirements.txt
#   litert-torch==0.9.0   — 0.9.1+ requires torchao>=0.17.0, which needs torch 2.7+
#   torchao==0.15.0       — 0.16.0+ uses torch.utils._pytree.register_constant,
#                           only present in torch 2.7+ (we're pinned to 2.6.0)
#   ethos-u-vela==5.1.0
#
# Build:
#   docker build --platform linux/amd64 -t grove_vision_convert .
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: core numeric + TF + PyTorch + ONNX stack.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: force ml_dtypes to 0.5.4 (bypasses TF's <0.5.0 cap; fine at runtime).
RUN pip install --no-deps --force-reinstall "ml_dtypes==0.5.4"

# Stage 3: latest official ultralytics from PyPI (no fork pin).
RUN pip install --no-deps ultralytics && \
    pip install pyyaml requests pillow pandas matplotlib scipy tqdm psutil py-cpuinfo

# Stage 3b: pin the LiteRT export chain — ultralytics' export_litert() auto-installs
# these on demand and will otherwise pull latest torchao, which requires torch 2.7+
# (torch.utils._pytree.register_constant, added in torchao 0.16.0).
RUN pip install "litert-torch==0.9.0" "torchao==0.15.0"

# Stage 4: pin Vela to 5.1.0.
# --no-deps avoids the flatbuffers==23.5.26 conflict with TF 2.18.0>=24.3.25.
RUN pip install --no-deps ethos-u-vela==5.1.0 && pip install lxml
