#!/usr/bin/env python3
"""
Grove Vision AI V2 — serial debugger.

Dumps all AT command responses and raw inference events to stdout so you
can diagnose why a model is not producing detections.

Usage:
    python debugger.py                          # auto-detect port
    python debugger.py --port /dev/cu.usbmodem...
    python debugger.py --raw                    # also print raw non-JSON serial bytes
    python debugger.py --invoke                 # run AT+INVOKE and dump events
    python debugger.py --invoke --algo 9        # try ALGO=9 (on-device DFL decoder)
"""
import argparse
import json
import sys
import time

from grove_vision_ai.device import GroveVisionAI, find_port


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def dump(label: str, value):
    print(f"  {label}: {json.dumps(value, indent=4) if isinstance(value, (dict, list)) else repr(value)}")


def main():
    ap = argparse.ArgumentParser(description="Grove Vision AI V2 serial debugger")
    ap.add_argument("--port", default=None, help="Serial port (auto-detected if omitted)")
    ap.add_argument("--raw", action="store_true", help="Print raw non-JSON serial lines")
    ap.add_argument("--invoke", action="store_true", help="Run AT+INVOKE and print events")
    ap.add_argument("--algo", type=int, default=None,
                    help="Override ALGO before invoking (e.g. 3 or 9)")
    ap.add_argument("--events", type=int, default=5,
                    help="Number of inference events to capture (default 5)")
    ap.add_argument("--score-threshold", type=int, default=10,
                    help="Score threshold for invoke run (default 10 — very permissive)")
    args = ap.parse_args()

    port = args.port or find_port()
    if not port:
        sys.exit("No Grove Vision AI V2 found. Pass --port explicitly.")

    print(f"\nConnecting to {port} ...")
    dev = GroveVisionAI()

    # Capture raw non-JSON boot/status lines
    raw_lines: list[str] = []
    if args.raw:
        def on_raw(line: str):
            ts = time.strftime("%H:%M:%S")
            print(f"  [RAW {ts}] {line}")
        dev.set_raw_callback(on_raw)

    dev.connect(port)
    print("Connected (boot wait done).")

    # ── Device info ────────────────────────────────────────────────────────────
    section("Device info")
    dump("ID",      dev.get_id())
    dump("Name",    dev.get_name())
    dump("Version", dev.get_version())
    dump("Status",  dev.get_status())

    # ── Models ─────────────────────────────────────────────────────────────────
    section("Models (AT+MODELS?)")
    models = dev.list_models()
    if models:
        for m in models:
            print(f"  {json.dumps(m)}")
    else:
        print("  (no models returned — device may have no flashed model)")

    section("Current model (AT+MODEL?)")
    cur = dev.get_current_model()
    dump("current", cur)

    # ── Sensors ────────────────────────────────────────────────────────────────
    section("Sensors (AT+SENSORS?)")
    sensors = dev.list_sensors()
    for s in sensors:
        print(f"  {json.dumps(s)}")
    section("Current sensor (AT+SENSOR?)")
    dump("sensor", dev.get_current_sensor())

    # ── Algo / thresholds ──────────────────────────────────────────────────────
    section("Algorithm / thresholds")
    dump("ALGO",   dev.get_algo())
    dump("TSCORE", dev.get_score_threshold())
    dump("TIOU",   dev.get_iou_threshold())

    # ── Optional invoke run ────────────────────────────────────────────────────
    if not args.invoke:
        print("\nTip: re-run with --invoke to capture live inference events.\n")
        dev.disconnect()
        return

    section("Setting up invoke run")

    # Apply algo override if requested
    algo_to_use = args.algo
    if algo_to_use is not None:
        ok = dev.set_algo(algo_to_use)
        print(f"  AT+ALGO={algo_to_use} → {'OK' if ok else 'FAILED'}")
    else:
        algo_to_use = dev.get_algo()
        print(f"  Using existing ALGO={algo_to_use}")

    ok = dev.set_score_threshold(args.score_threshold)
    print(f"  AT+TSCORE={args.score_threshold} → {'OK' if ok else 'FAILED (firmware may not support it)'}")

    # Capture N inference events
    section(f"AT+INVOKE (capturing {args.events} events, score threshold={args.score_threshold}%)")

    captured = []

    def on_event(msg: dict):
        data = msg.get("data", {})
        ts   = time.strftime("%H:%M:%S")
        idx  = len(captured) + 1

        boxes   = data.get("boxes", [])
        perf    = data.get("perf", [])
        has_img = bool(data.get("image", ""))

        print(f"\n  [{ts}] Event #{idx}")
        print(f"    perf (pre/inf/post ms): {perf}")
        print(f"    has_image: {has_img}")
        print(f"    boxes ({len(boxes)}):")
        if boxes:
            for b in boxes:
                print(f"      {b}")
        else:
            print("      (none)")

        # Also dump other keys (classes, keypoints, count, etc.)
        for key, val in data.items():
            if key not in ("boxes", "perf", "image"):
                print(f"    {key}: {val}")

        captured.append(data)

    dev.set_callback(on_event)
    ok = dev.start_invoke(n_times=-1, result_only=0)
    if not ok:
        print("  AT+INVOKE failed — is a model loaded?")
        dev.disconnect()
        return

    print(f"  Streaming... waiting for {args.events} events (Ctrl-C to stop early)\n")
    try:
        deadline = time.time() + 60
        while len(captured) < args.events and time.time() < deadline:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    dev.stop_stream()
    dev.disconnect()

    section("Summary")
    print(f"  Events received: {len(captured)}")
    total_boxes = sum(len(e.get("boxes", [])) for e in captured)
    print(f"  Total boxes across all events: {total_boxes}")
    if total_boxes == 0:
        print("\n  *** No detections. Possible causes:")
        print("      1. Wrong ALGO — try --algo 9 (on-device DFL) or --algo 3 (decoded-box)")
        print("      2. Model not loaded at the expected address (check AT+MODELS? above)")
        print("      3. Score threshold too high — this run used", args.score_threshold)
        print("      4. Model output tensor shape mismatch with firmware parser")
        print("      5. Model flashed to wrong address (device loaded a different slot)")
    print()


if __name__ == "__main__":
    main()
