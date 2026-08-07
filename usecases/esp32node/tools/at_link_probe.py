#!/usr/bin/env python3
"""Measure what an AT link out of the Grove Vision AI V2 can actually sustain.

Point this at either transport and it drives AT+INVOKE and reports throughput:

  * the CH343 USB console (UART_0) -- e.g. /dev/tty.usbmodem*, 921600
  * UART_1 on the XIAO header (PB6/PB7 = D6/D7) via a USB-TTL adapter, 921600

Same firmware, same protocol on both, so the USB console gives you the upper bound
for free and the USB-TTL run tells you whether UART_1 matches it. Run --mode image
against UART_1 both before and after applying firmware_patch/0001-*.patch: unpatched,
the stream is expected to stall after a handful of large frames.

    python3 at_link_probe.py --port /dev/tty.usbmodem1101 --mode boxes
    python3 at_link_probe.py --port /dev/ttyUSB0 --mode image --seconds 20
"""

import argparse
import base64
import json
import statistics
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial is required:  pip install pyserial")


def drain(port, seconds=0.3):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if port.in_waiting:
            port.read(port.in_waiting)
        else:
            time.sleep(0.01)


def send_at(port, cmd):
    port.write((cmd + "\r\n").encode())
    port.flush()


def summarize(name, values, unit, fmt="{:.1f}"):
    if not values:
        return f"  {name:<22} (none)"
    body = "  ".join(
        f"{label}={fmt.format(v)}"
        for label, v in (
            ("min", min(values)),
            ("med", statistics.median(values)),
            ("max", max(values)),
        )
    )
    return f"  {name:<22} {body} {unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial device")
    ap.add_argument("--baudrate", type=int, default=921600)
    ap.add_argument("--mode", choices=["boxes", "image"], default="boxes",
                    help="boxes = AT+INVOKE=-1,1 (results only); image = AT+INVOKE=-1,0")
    ap.add_argument("--seconds", type=float, default=15.0, help="measurement window")
    ap.add_argument("--sensor-preset", type=int, default=None,
                    help="issue AT+SENSOR=1,1,<idx> first to change resolution")
    ap.add_argument("--stall-timeout", type=float, default=5.0,
                    help="declare a stall after this many seconds with no complete line")
    ap.add_argument("--verbose", action="store_true", help="print each frame's stats")
    args = ap.parse_args()

    results_only = 1 if args.mode == "boxes" else 0

    with serial.Serial(args.port, args.baudrate, timeout=0.05) as port:
        drain(port)
        send_at(port, "AT+BREAK")
        drain(port, 0.5)

        if args.sensor_preset is not None:
            send_at(port, f"AT+SENSOR=1,1,{args.sensor_preset}")
            drain(port, 1.0)

        send_at(port, f"AT+INVOKE=-1,{results_only}")

        buf = bytearray()
        payload_bytes = []
        jpeg_bytes = []
        box_counts = []
        perf_rows = []
        frame_times = []
        total_bytes = 0
        parse_errors = 0
        oversize_drops = 0

        start = time.time()
        last_line = start
        deadline = start + args.seconds
        stalled = False

        while time.time() < deadline:
            chunk = port.read(4096)
            if chunk:
                total_bytes += len(chunk)
                buf.extend(chunk)
            elif time.time() - last_line > args.stall_timeout:
                stalled = True
                break

            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                raw = bytes(buf[:nl])
                del buf[:nl + 1]
                last_line = time.time()

                line = raw.strip(b"\r\x00 \t")
                if not line.startswith(b"{"):
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                # type 1 == MA_MSG_TYPE_EVT
                if msg.get("name") != "INVOKE" or msg.get("type") != 1:
                    continue

                data = msg.get("data", {})
                payload_bytes.append(len(line))
                box_counts.append(len(data.get("boxes") or []))
                frame_times.append(last_line)
                if data.get("perf"):
                    perf_rows.append(data["perf"])

                b64 = data.get("image")
                if b64:
                    try:
                        jpeg_bytes.append(len(base64.b64decode(b64, validate=False)))
                    except Exception:
                        parse_errors += 1

                if args.verbose:
                    print(f"    frame #{data.get('count')} line={len(line)}B "
                          f"boxes={box_counts[-1]} perf={data.get('perf')}")

            if len(buf) > 4 * 1024 * 1024:
                # No newline in 4 MB means we are not framed correctly; do not grow forever.
                oversize_drops += 1
                buf.clear()

        send_at(port, "AT+BREAK")
        drain(port, 0.3)

    elapsed = time.time() - start
    n = len(frame_times)

    print()
    print(f"port          {args.port} @ {args.baudrate} 8N1")
    print(f"mode          {args.mode}  (AT+INVOKE=-1,{results_only})")
    print(f"window        {elapsed:.1f} s")
    print()
    print(f"frames        {n}   ({n / elapsed:.2f} fps)" if elapsed else "frames        0")
    print(f"link          {total_bytes / elapsed / 1024:.1f} kB/s "
          f"({total_bytes / elapsed / (args.baudrate / 10) * 100:.0f}% of line rate)"
          if elapsed else "")
    print(summarize("json line", payload_bytes, "B", "{:.0f}"))
    if jpeg_bytes:
        print(summarize("jpeg (decoded)", jpeg_bytes, "B", "{:.0f}"))
    print(summarize("boxes/frame", box_counts, "", "{:.0f}"))
    if perf_rows:
        cols = list(zip(*perf_rows))
        labels = ["perf preprocess", "perf inference", "perf postprocess"]
        for label, col in zip(labels, cols):
            print(summarize(label, list(col), "ms", "{:.0f}"))

    if n >= 2:
        gaps = [b - a for a, b in zip(frame_times, frame_times[1:])]
        print(summarize("inter-frame gap", [g * 1000 for g in gaps], "ms"))

    if parse_errors:
        print(f"\n  !! {parse_errors} malformed/truncated line(s)")
    if oversize_drops:
        print(f"  !! {oversize_drops} unframed 4 MB buffer flush(es)")
    if stalled:
        print(f"\n  !! STALL: no complete line for {args.stall_timeout:.0f}s after "
              f"{n} frame(s).")
        print("     On UART_1 in --mode image this is the known Serial TX bug; see "
              "esp_node/firmware_patch/README.md")
        return 1
    if n == 0:
        print("\n  !! no INVOKE events. Check baudrate, wiring, and that a model is loaded "
              "(AT+MODEL?).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
