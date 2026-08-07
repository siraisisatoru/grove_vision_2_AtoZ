# Build for amd64 (x86_64) to match the Colab environment:
#   docker build --platform linux/amd64 -t grove_vision_convert .
FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: full pinned stack (torch is pinned here, so the fork below
# will NOT upgrade it — this is what keeps us on Colab's export path).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: force ml_dtypes to Colab's 0.5.4 (bypasses pip's static
# conflict with tensorflow's <0.5.0 cap; works fine at runtime).
RUN pip install --no-deps --force-reinstall "ml_dtypes==0.5.4"

# Stage 3: clone & install the kris-himax ultralytics fork.
# torch/torchvision are already satisfied, so nothing gets upgraded.
RUN git clone https://github.com/kris-himax/ultralytics /ultralytics && \
    cd /ultralytics && \
    pip install --no-deps . && \
    pip install pyyaml requests pillow pandas matplotlib scipy tqdm psutil py-cpuinfo