#!/usr/bin/env python3
"""
CLI: Flash a TFLite model onto Grove Vision AI V2.

Usage:
  # Flash a model from the Himax model zoo (auto-download):
  python flash_model.py --port /dev/cu.usbmodem5A7A0167681 --model yolov8n_od

  # Flash a custom .tflite file at a specific flash address:
  python flash_model.py --port /dev/cu.usbmodem5A7A0167681 \\
      --model-file my_model.tflite --address 0xB7B000

  # Flash firmware + several models in one pass (e.g. face recognition), no
  # BOOT button needed:
  python flash_model.py --firmware face_recognition.img --software-reset \\
      --model-at scrfd_vela.tflite 0x200000 \\
      --model-at mobilefacenet_vela.tflite 0x400000

  # List available pre-built models:
  python flash_model.py --list-models

Device MUST be in BOOT mode before running (hold BOOT + press RESET), unless
--software-reset is given (the flasher toggles RTS to enter the bootloader).

Custom model training notes:
  - Train a YOLO or MobileNet model using Edge Impulse, Roboflow, or SenseCraft AI
  - Export as TFLite int8 quantized
  - Flash at address 0xB7B000 (primary slot) or 0xA00000 (secondary)
  - The SSCMA firmware auto-detects the model type from TFLite metadata
"""

import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))
from grove_vision_ai.downloader import download_latest_model, list_available_models
from grove_vision_ai.flasher import Flasher
from grove_vision_ai.device import GroveVisionAI, find_port

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_SHORTCUTS = {
    "yolov8n_od": "yolov8n_od_192_delete_transpose_0xB7B000.tflite",
    "yolo11n_od": "yolo11n_od_192_0xB7B000.tflite",
    "yolov8n_cls": "yolov8n_cls_192_0xB7B000.tflite",
    "yolov8n_pose": "yolov8n_pose_192_0xB7B000.tflite",
}


def progress(stage, done, total, msg):
    icons = {"xmodem": "⬆", "model": "🧠", "model_preamble": "📍", "done": "✅", "error": "❌"}
    icon = icons.get(stage, "•")
    if stage == "xmodem" and total:
        bar_w = 30
        filled = int(done / total * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"\r  {icon} [{bar}] {done}/{total}", end="", flush=True)
        if done >= total:
            print()
    else:
        print(f"  {icon} [{stage}] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Flash a TFLite model to Grove Vision AI V2")
    parser.add_argument("--port", "-p", help="Serial port")
    parser.add_argument("--model", "-m", help="Model shortcut name (e.g. yolov8n_od)")
    parser.add_argument("--model-file", help="Path to custom .tflite file")
    parser.add_argument("--address", default="0x400000",
                        help="Flash address in hex (default: 0x400000 = SSCMA slot 1)")
    parser.add_argument("--model-at", nargs=2, action="append", default=[],
                        metavar=("FILE", "ADDR"),
                        help="Flash a model FILE at hex ADDR; repeat for several models "
                             "(e.g. --model-at scrfd.tflite 0x200000 --model-at mfn.tflite 0x400000)")
    parser.add_argument("--firmware", "-f",
                        help="Path to firmware .img to flash before the model(s) (required first time or after fw rebuild)")
    parser.add_argument("--offset", default="0x0", help="Model offset (default: 0x0)")
    parser.add_argument("--software-reset", action="store_true",
                        help="Enter the bootloader via an RTS toggle instead of a BOOT-button press")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--cache-dir", default="./models")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable pre-built models for Grove Vision AI V2:\n")
        for m in list_available_models():
            print(f"  Shortcut: {m['filename'].split('_0x')[0]}")
            print(f"  Name:     {m['name']}")
            print(f"  File:     {m['filename']}")
            print(f"  Address:  0x{m['address']:X}")
            print(f"  Task:     {m['task']}")
            print()
        return

    if args.list_ports:
        print("\nAvailable serial ports:")
        for p in GroveVisionAI.list_ports():
            print(f"  {p['device']:30s}  {p['description']}")
        return

    if not args.port:
        args.port = find_port()
    if not args.port:
        sys.exit("Error: no Grove Vision AI V2 detected. Pass --port explicitly.")

    offset = int(args.offset, 16)

    # Build the models list. --model-at (repeatable) takes precedence; otherwise
    # fall back to the single --model-file / --model (shortcut, may auto-download)
    # path so existing callers keep working.
    models = []
    if args.model_at:
        for file, addr in args.model_at:
            if not os.path.isfile(file):
                sys.exit(f"Error: model file not found: {file}")
            models.append((file, int(addr, 16), offset))
    elif args.model_file:
        models.append((args.model_file, int(args.address, 16), offset))
    elif args.model:
        filename = MODEL_SHORTCUTS.get(args.model, args.model)
        if not filename.endswith(".tflite"):
            filename += ".tflite"
        address = int(args.address, 16)
        if "_0x" in filename:  # address encoded in the shortcut filename
            address = int("0x" + filename.split("_0x")[1].replace(".tflite", ""), 16)
        print(f"\n📥 Downloading model: {filename}")
        os.makedirs(args.cache_dir, exist_ok=True)

        def dl_progress(done, total):
            if total:
                print(f"\r  ↓ {done/total*100:.0f}%  ({done//1024}KB/{total//1024}KB)",
                      end="", flush=True)

        model_path = download_latest_model(filename, dest_dir=args.cache_dir,
                                           progress_cb=dl_progress)
        print(f"\n  ✅ Model ready: {model_path}\n")
        models.append((model_path, address, offset))

    if not models and not args.firmware:
        sys.exit("Error: nothing to flash — pass --firmware and/or --model/--model-file/--model-at")

    print(f"\n🧠 Flashing to {args.port}")
    if args.firmware:
        print(f"   firmware: {args.firmware}")
    for file, addr, _ in models:
        print(f"   0x{addr:06X}: {file}")
    print("─" * 50)
    if not args.software_reset:
        print("ACTION NEEDED: Hold BOOT button and press RESET on the device now!")
        print("─" * 50)

    flasher = Flasher(port=args.port, baudrate=921600, progress_callback=progress)
    ok = flasher.flash(firmware_path=args.firmware or None, models=models,
                       software_reset=args.software_reset)

    if ok:
        print("\n✅ Flash complete! Device rebooting.")
    else:
        print("\n❌ Flash failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
