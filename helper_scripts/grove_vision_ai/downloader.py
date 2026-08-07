"""
Downloads latest SSCMA firmware and model files from Seeed Studio GitHub releases.

Firmware source: https://github.com/Seeed-Studio/sscma-example-we2/releases
Model zoo: https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2
"""

import os
import requests
import logging
from typing import Optional, Callable

log = logging.getLogger(__name__)

SSCMA_RELEASES_API = "https://api.github.com/repos/Seeed-Studio/sscma-example-we2/releases/latest"
HIMAX_RELEASES_API = "https://api.github.com/repos/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/releases/latest"

FIRMWARE_ASSET_PREFIX = "grove_vision_ai_v2_"


def _download_file(url: str, dest: str,
                   progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(done, total)
    return dest


def download_latest_firmware(dest_dir: str = ".",
                              progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Download the latest grove_vision_ai_v2_*.img from sscma-example-we2 releases.
    Returns the local path of the downloaded file.
    """
    log.info("Fetching latest SSCMA release info...")
    resp = requests.get(SSCMA_RELEASES_API, timeout=15)
    resp.raise_for_status()
    release = resp.json()
    tag = release["tag_name"]
    log.info("Latest SSCMA release: %s", tag)

    asset_url = None
    asset_name = None
    for asset in release.get("assets", []):
        if asset["name"].startswith(FIRMWARE_ASSET_PREFIX) and asset["name"].endswith(".img"):
            asset_url = asset["browser_download_url"]
            asset_name = asset["name"]
            break

    if not asset_url:
        # Construct URL from known pattern
        asset_name = f"{FIRMWARE_ASSET_PREFIX}{tag}.img"
        asset_url = f"https://github.com/Seeed-Studio/sscma-example-we2/releases/download/{tag}/{asset_name}"

    dest = os.path.join(dest_dir, asset_name)
    if os.path.exists(dest):
        log.info("Firmware already cached at %s", dest)
        return dest

    log.info("Downloading %s ...", asset_url)
    return _download_file(asset_url, dest, progress_cb)


def download_latest_model(model_name: str, dest_dir: str = ".",
                           progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Download a model file from the Himax model zoo.

    Common model_name values:
      "yolov8n_od_192_delete_transpose_0xB7B000.tflite"
      "yolo11n_od_192_0xB7B000.tflite"

    Returns local path.
    """
    base_url = (
        "https://raw.githubusercontent.com/"
        "HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/main/model_zoo/"
    )
    # Try to find in common subdirs
    subdirs = ["tflm_yolov8_od", "tflm_yolo11_od", "tflm_yolov8_cls", ""]
    downloaded = None

    for sub in subdirs:
        url = f"{base_url}{sub}/{model_name}" if sub else f"{base_url}{model_name}"
        dest = os.path.join(dest_dir, model_name)
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                if os.path.exists(dest):
                    log.info("Model already cached at %s", dest)
                    return dest
                log.info("Downloading model from %s", url)
                downloaded = _download_file(url, dest, progress_cb)
                break
        except requests.RequestException:
            continue

    if not downloaded:
        raise FileNotFoundError(f"Model '{model_name}' not found in Himax model zoo")
    return downloaded


def list_available_models() -> list:
    """Return a curated list of known-good models for Grove Vision AI V2."""
    return [
        {
            "name": "YOLOv8n Object Detection (192x192)",
            "filename": "yolov8n_od_192_delete_transpose_0x400000.tflite",
            "address": 0x400000,
            "task": "detection",
            "input_size": 192,
            "classes": "COCO 80",
        },
        {
            "name": "YOLO11n Object Detection (192x192)",
            "filename": "yolo11n_od_192_0x400000.tflite",
            "address": 0x400000,
            "task": "detection",
            "input_size": 192,
            "classes": "COCO 80",
        },
        {
            "name": "YOLOv8n Classification (192x192)",
            "filename": "yolov8n_cls_192_0x400000.tflite",
            "address": 0x400000,
            "task": "classification",
            "input_size": 192,
            "classes": "ImageNet 1000",
        },
        {
            "name": "YOLOv8n Pose Estimation (192x192)",
            "filename": "yolov8n_pose_192_0x400000.tflite",
            "address": 0x400000,
            "task": "pose",
            "input_size": 192,
            "classes": "COCO Person",
        },
    ]
