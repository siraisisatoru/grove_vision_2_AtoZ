#!/usr/bin/env python3
"""
CLI: Download and flash the latest SSCMA firmware onto Grove Vision AI V2.

Usage:
  python flash_fw.py --port /dev/cu.usbmodem5A7A0167681
  python flash_fw.py --port /dev/cu.usbmodem5A7A0167681 --firmware path/to/custom.img
  python flash_fw.py --list-ports

The device MUST be in BOOT mode before running:
  1. Hold the BOOT button on the device
  2. Press RESET (or re-plug USB while holding BOOT)
  3. Run this script immediately
"""

import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))
from grove_vision_ai.downloader import download_latest_firmware
from grove_vision_ai.flasher import Flasher
from grove_vision_ai.device import GroveVisionAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def progress(stage, done, total, msg):
    icons = {
        "waiting": "⏳", "firmware": "📦", "model": "🧠",
        "model_preamble": "📍", "xmodem": "⬆", "done": "✅", "error": "❌",
    }
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
    parser = argparse.ArgumentParser(description="Flash Grove Vision AI V2 firmware")
    parser.add_argument("--port", "-p", help="Serial port (e.g. /dev/cu.usbmodem...)")
    parser.add_argument("--firmware", "-f", help="Path to .img firmware file (downloads latest if omitted)")
    parser.add_argument("--baudrate", "-b", type=int, default=921600)
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    parser.add_argument("--cache-dir", default="./firmware", help="Directory to cache downloaded files")
    args = parser.parse_args()

    if args.list_ports:
        print("\nAvailable serial ports:")
        for p in GroveVisionAI.list_ports():
            print(f"  {p['device']:30s}  {p['description']}")
        return

    if not args.port:
        print("Error: --port is required. Use --list-ports to find your device.")
        sys.exit(1)

    # Download firmware if not provided
    fw_path = args.firmware
    if not fw_path:
        print(f"\n📥 Downloading latest SSCMA firmware...")
        os.makedirs(args.cache_dir, exist_ok=True)

        def dl_progress(done, total):
            if total:
                pct = done / total * 100
                print(f"\r  ↓ {pct:.0f}%  ({done//1024}KB / {total//1024}KB)", end="", flush=True)

        fw_path = download_latest_firmware(dest_dir=args.cache_dir, progress_cb=dl_progress)
        print(f"\n  ✅ Firmware ready: {fw_path}\n")

    print(f"\n🚀 Flashing firmware to {args.port}")
    print("─" * 50)
    print("ACTION NEEDED: Hold BOOT button and press RESET on the device now!")
    print("─" * 50)

    flasher = Flasher(port=args.port, baudrate=args.baudrate, progress_callback=progress)
    ok = flasher.flash(firmware_path=fw_path)

    if ok:
        print("\n✅ Flash complete! Device is rebooting into SSCMA firmware.")
        print("   Wait 3 seconds then connect with: python stream_inference.py --port", args.port)
    else:
        print("\n❌ Flash failed. Check that the device is in BOOT mode and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
