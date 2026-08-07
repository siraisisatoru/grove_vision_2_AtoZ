# esp_node — XIAO ESP32-S3 companion link for Grove Vision AI V2

**Question:** can this firmware talk to an ESP32-S3 mounted on the Grove board, so the
ESP32 replaces the USB cable and forwards boxes + JPEG frames over WiFi to a Raspberry Pi?

**Answer: yes, and no new firmware transport is needed.** The link already exists in the
firmware you are running today. One latent bug in it has to be fixed before the *image*
path is reliable; the *boxes-only* path works as-is. Details and measurements below.

```
Grove Vision AI V2 (HX6538)                XIAO ESP32-S3            Raspberry Pi
┌──────────────────────────┐              ┌──────────────┐         ┌──────────────┐
│ sscma app                │  UART1       │ AT client    │  WiFi   │ receiver.py  │
│ AT server ──► Serial ────┼──921600 8N1──┤ line reader  ├──TCP────┤ cv2 draw +   │
│         └──► Console ────┼──USB (CH343) │ JSON split   │         │ MJPEG server │
└──────────────────────────┘   (debug)    │ base64→JPEG  │         └──────────────┘
                                          └──────────────┘
```

---

## 1. Feasibility: what the hardware actually wires up

From `Grove_Vision_AI_Module_V2_Circuit_Diagram.pdf` (XIAO connector `M1`, HX6538 `U1`):

| XIAO pin | Net on the Grove board | HX6538 ball | Notes |
|---|---|---|---|
| D6 / TX  | `PB6/UART1_RX/3V3`  | N1 | XIAO **TX** → Himax UART1 **RX** — correct crossover, no swap |
| D7 / RX  | `PB7/UART1_TX/3V3`  | M1 | XIAO **RX** ← Himax UART1 **TX** |
| D4 / SDA | `PA3/I2CS0_SDA/3V3` | B4 | Himax I2C **slave** 0 |
| D5 / SCL | `PA2/I2CS0_SCL/3V3` | A4 | Himax I2C **slave** 0 |
| D8/D9/D10 | `PB4/SSPIM_SCLK`, `PB3/SSPIM_DI`, `PB2/SSPIM_DO` | | Himax SPI **master** — wrong direction for this use |
| 3V3 / GND / VUSB | | | shared rails, both boards can power each other |

All XIAO-facing signals are 3V3, matching the ESP32-S3. Nothing needs level shifting.

## 2. Feasibility: what the firmware already exposes

`Device::init()` registers three AT transports
([ma_device.cpp:84-99](../EPII_CM55M_APP_S/library/sscma_micro_porting/porting/himax/we2/ma_device.cpp#L84-L99)):

| Transport | Peripheral | Pins | State in this build |
|---|---|---|---|
| `Console` | UART_0 | CH343 USB bridge | always on — the USB cable you use today |
| `Serial`  | UART_1 | **PB6/PB7 = the XIAO header** | **always on** |
| `I2C`     | I2CS_0 | PA2/PA3 = XIAO D4/D5 | compiled out (`MA_CONFIG_BOARD_I2C_SLAVE` is commented out in `ma_config_board.h:99`) |

The AT server polls *every* registered transport each tick and replies on whichever one
received the command ([ma_server_at.cpp:88-98](../EPII_CM55M_APP_S/library/sscma_micro/sscma/server/at/ma_server_at.cpp#L88-L98)).
An `AT+INVOKE` issued on UART1 gets its event stream on UART1. So the ESP32 sees exactly
the same protocol the USB console sees — it is a drop-in replacement for the cable, and the
USB console stays live at the same time for debugging.

**UART1 is fixed at 921600 8N1** and is not runtime-configurable
([ma_transport_serial.cpp:84](../EPII_CM55M_APP_S/library/sscma_micro_porting/porting/himax/we2/ma_transport_serial.cpp#L84)).

## 3. Wire format the ESP32 has to parse

Each AT message is `\r` + one line of compact JSON + `\n`
([ma_codec_json.cpp:172-175](../EPII_CM55M_APP_S/library/sscma_micro/sscma/server/at/codec/ma_codec_json.cpp#L172-L175)).
An `AT+INVOKE=-1,0` event looks like:

```json
{"type":1,"name":"INVOKE","code":0,"data":{
  "count":42,
  "image":"<base64 JPEG>",
  "boxes":[[x,y,w,h,score,target], ...],
  "perf":[prepocess_ms, inference_ms, postprocess_ms],
  "rotation":0, "width":640, "height":480 }}
```

- `boxes` entries are `[x, y, w, h, score, target]` with **x,y = box centre**, in pixels of
  `data.width` × `data.height`, and `score` as an integer percent
  ([refactor_required.hpp:114-122](../EPII_CM55M_APP_S/library/sscma_micro/sscma/server/at/callback/refactor_required.hpp#L114-L122)).
  The receiver rescales them onto the decoded JPEG rather than trusting the two to match.
- `image` is base64 of a hardware-encoded JPEG. It is omitted entirely when `RESULTS_ONLY=1`.
- Command form is `AT+INVOKE=N_TIMES,RESULTS_ONLY` — `-1` for a continuous stream.

## 4. Feasibility: the bandwidth budget

UART1 at 921600 8N1 = **92.2 kB/s** of line rate. Base64 inflates the JPEG by 4/3.

| Mode | Payload / frame | Link time / frame | Link-limited FPS |
|---|---|---|---|
| `AT+INVOKE=-1,1` (boxes only) | 0.2–0.5 kB | ~5 ms | >100 — inference-bound, not link-bound |
| 320×240 JPEG + boxes | ~8–13 kB b64 | 90–140 ms | **7–11** |
| 640×480 JPEG + boxes | ~27–47 kB b64 | 300–510 ms | **2–3** |

So: **boxes at full inference rate, or video at a few FPS.** Drop the sensor preset with
`AT+SENSOR=1,1,<preset_idx>` if you want smoother video — resolution is the only real lever.

For comparison, the I2C slave path (disabled here) moves 256 bytes per polled transaction
at 400 kHz, roughly 20–25 kB/s effective — under 1 FPS with images. **Use UART1.** The only
reason to enable I2C is if you also need D6/D7 for something else.

## 5. The one firmware bug in the way

`Console` (UART_0) was hardened for large payloads; `Serial` (UART_1) was not, because at
the time nothing streamed images over it. Three defects remain in `Serial`:

1. **Non-atomic TX-DMA claim.** `send()` does a bare `if (!_tx_busy) { _tx_busy = true; … }`
   from task context while the DMA ISR clears `_tx_busy` whenever it finds the ring empty.
   Lose that race and bytes sit in the ring with no DMA in flight — the stream stops dead.
   `Console` fixed this with an interrupt-disabled `_tx_kick()`; `Serial` still has the old
   code ([ma_transport_serial.cpp:176-190](../EPII_CM55M_APP_S/library/sscma_micro_porting/porting/himax/we2/ma_transport_serial.cpp#L176-L190)).
2. **`flush()` is a no-op.** The invoke loop calls `_transport->flush()` after every frame
   specifically to pace itself to the link so the ring never fills. On `Serial` that returns
   immediately, so the pacing does nothing.
3. **4 kB TX ring** vs `Console`'s 48 kB. Every image frame is larger than the whole ring,
   so every frame goes through the ring-full path — the exact path defect 1 lives in.

Together these mean boxes-only streaming works today, and image streaming will wedge after
a handful of frames. All three are fixed in `sscma_micro_porting/porting/himax/we2/` — committed
first-party source, not applied by a patch (see §10.1). The ring size becomes
`MA_TRANSPORT_SERIAL_TX_RING_SIZE`, defaulting to 4 kB and set to 24 kB in `ma_config_board.h`.
See [patches/README.md](../patches/README.md) for the fix writeup.

**Verdict: feasible.** Boxes-only needs zero firmware changes. Image streaming needs the fixes
above, which are already in the tree.

---

## 6. Contents

```
esp_node/
├── README.md
├── xiao_esp32s3/                            arduino-cli sketch for the companion MCU
│   ├── xiao_esp32s3.ino                     reader task, sender task, AT control
│   ├── at_frame.h                           AT line parsing (host-testable, no Arduino deps)
│   ├── config.h.example                     copy to config.h, gitignored
│   └── build.sh                             compile / upload / monitor
├── rpi_receiver/
│   ├── receiver.py                          TCP receiver, box drawing, MJPEG + OpenCV
│   ├── start.sh                              checks/installs deps, launches receiver.py detached
│   ├── stop.sh                               stops it
│   └── requirements.txt
├── mac_receiver/                            run receiver.py on a Mac instead of a Pi
│   ├── start.sh                              venv + plain opencv-python, runs in the foreground
│   └── requirements.txt
└── tools/
    ├── at_link_probe.py                     measure the link before building anything
    ├── fake_node.py                         stand in for the ESP32 while developing the Pi side
    └── test_at_frame.cpp                    host tests for at_frame.h
```

Build tooling is `arduino-cli` with the `esp32:esp32` core (already installed here, board
`XIAO_ESP32S3`), not PlatformIO. The sketch has **no external library dependencies** — it
never parses the detection JSON, it only excises the base64 image and forwards both halves,
so the Pi does the parsing where memory is free.

### What has been verified, and what has not

| | Status |
|---|---|
| Pin mapping D6/D7 → PB6/PB7 | read from the schematic netlist and `pins_arduino.h` (D6=GPIO43, D7=GPIO44) |
| `Serial` transport registered and polled on UART1 | read from firmware sources |
| Sketch compiles for `XIAO_ESP32S3` | ✅ `arduino-cli compile` clean, 25% flash / 14% DRAM |
| AT line parsing (`at_frame.h`) | ✅ `tools/test_at_frame.cpp`, all checks pass |
| Receiver end-to-end (framing, boxes, MJPEG, save) | ✅ against `tools/fake_node.py` |
| Anything on real hardware | ❌ not run — start with §7 step 1 |

## 7. Bring-up order

Do these in order; each step is independently observable.

### Step 1 — measure the link, no ESP32 involved

Prove the numbers in §4 on your hardware before committing to the design. Works over the
USB console (identical protocol) or over a USB-TTL adapter clipped to D6/D7:

```bash
python3 esp_node/tools/at_link_probe.py --port /dev/tty.usbmodem* --mode boxes
python3 esp_node/tools/at_link_probe.py --port /dev/tty.usbmodem* --mode image --seconds 20
```

It reports frames/s, bytes/s, payload size distribution, and the `perf` triple, so you can
see immediately whether you are inference-bound or link-bound. Run `--mode image` against
**UART1** (USB-TTL on D6/D7) to reproduce the §5 stall before patching, and again after.

### Step 2 — reflash the Grove board (image mode only)

The UART1 fixes are already part of `patches/sscma_micro_full.patch`, which `quick_rebuild.sh`
applies automatically — nothing to patch by hand. For image streaming, raise the ring:
```bash
# add -DMA_TRANSPORT_SERIAL_TX_RING_SIZE=24576 to APPL_DEFINES, or #define it
# in ma_config_board.h, then:
./build_and_flash.sh
```

### Step 3 — bring up the Pi side with no hardware in the loop

`fake_node.py` speaks the ESP32's wire format, so the receiver can be finished and debugged
before the XIAO exists:

```bash
cd esp_node/rpi_receiver
pip install -r requirements.txt
python3 receiver.py --mjpeg-port 8080 &        # browse http://<host>:8080/
python3 ../tools/fake_node.py --fps 5          # synthetic moving boxes
```

`fake_node.py --replay <capture.txt>` will also replay real AT event lines captured off the
board in step 1, which is the closest you can get to the real stream without the ESP32.

### Step 4 — flash the XIAO

```bash
cd esp_node/xiao_esp32s3
cp config.h.example config.h        # fill in WiFi + RPi host
./build.sh --upload --monitor
```

The monitor prints a `[stat]` line every 5 s with input fps, kB/s, dropped and oversize
frame counts, and free heap/PSRAM — compare it against what `at_link_probe.py` measured in
step 1. A gap between the two means the ESP32 is not keeping the UART ring drained.

If you change `at_frame.h`, re-run its tests:

```bash
cd esp_node/tools
c++ -std=c++17 -Wall -Wextra -o /tmp/test_at_frame test_at_frame.cpp && /tmp/test_at_frame
```

## 8. Powering the stack

`VUSB`, `3V3` and `GND` are common across the XIAO connector, so either board's USB-C can
power both. Powering from the XIAO is the "replace the cable" configuration. Note the Grove
board's own USB-C still enumerates the CH343 console independently — keep it plugged in
during bring-up, it costs nothing and gives you the `Console` transport for debugging while
the ESP32 drives `Serial`.

## 9. Known caveats

- **Verified working end-to-end on real hardware as of 2026-08-04.** See §10 for the exact
  setup, the two real bugs that turned up during bring-up (both fixed), and how to reproduce
  it. Section §7 below is the original, pre-bring-up bring-up plan; §10 is what actually
  happened when it was followed.
- The ESP32 must drain UART1 faster than 92 kB/s or the driver ring overflows mid-frame and
  the JSON line is truncated. `main.cpp` uses a dedicated high-priority reader task and a
  32 kB driver ring for this; truncated lines are counted and dropped, not parsed.
- `data.width`/`data.height` describe the *raw* frame the boxes were scaled to. The JPEG is
  a separate `retrieveFrame(MA_PIXEL_FORMAT_JPEG)` and is not guaranteed to be the same
  size, so the receiver always rescales.
- Nothing here uses TLS or authentication. It is a lab link on a trusted LAN.

## 10. Verified working configuration (2026-08-04)

This section documents the exact setup that was built, flashed, and confirmed showing a
live annotated video feed in a browser. Use it as the reference configuration; §§1–9 above
are the design rationale and original bring-up plan that led here.

### 10.1 Grove board firmware patch

The Grove board runs the normal `sscma` scenario app. Two separate things have to be in place, and
they are deliberately *not* delivered the same way:

1. **`sscma_micro` submodule fixes** — the WE2 compatibility set (see `patches/README.md`),
   delivered as `patches/sscma_micro_full.patch`. The submodule is kept pristine upstream and
   repatched on every build, so this content lives in the patch file and nowhere else.
2. **UART1 (`Serial` transport) TX fixes** — described in §5: atomic `_tx_kick()`, a `flush()` that
   actually waits for drain, a bounded `send()` that drops a frame after 200k stalled yields
   instead of wedging forever, and a configurable `MA_TRANSPORT_SERIAL_TX_RING_SIZE`. These are in
   `sscma_micro_porting/`, which is ordinary first-party source, so they are **committed directly**
   — nothing to apply.

These were briefly merged into one patch file spanning both trees (commit `7307f34`). That did not
work and was reverted in `b43c65f`: mixing "reapplied every build" content with "committed once"
content meant no single tree state satisfied both the forward and the reverse apply check, so
`quick_rebuild.sh` failed with `does not apply cleanly` on a clean checkout. Keep the boundary —
submodule fixes in the patch, everything else committed.

`quick_rebuild.sh` applies the patch automatically from inside the submodule (idempotent — skips if
already applied, errors loudly if it can't apply cleanly):

```bash
./quick_rebuild.sh
```

This produces `we2_image_gen_local/output_case1_sec_wlcsp/output.img`, flashed the normal way (see
the top-level `CLAUDE.md` / repo README for the xmodem flashing steps).

**TX ring for the image path.** `AT+INVOKE=-1,0` sends a base64 JPEG per frame, which is bigger
than the transport's 4 KB default ring, so every frame takes `send()`'s ring-full path (§5).
`ma_config_board.h` sets `MA_TRANSPORT_SERIAL_TX_RING_SIZE` to 24 KB for this reason. Do **not**
try to set it as `./quick_rebuild.sh APPL_DEFINES+=-DMA_TRANSPORT_SERIAL_TX_RING_SIZE=24576`, as an
earlier revision of this section advised — a command-line `APPL_DEFINES` assignment makes GNU Make
ignore every `APPL_DEFINES +=` in the makefiles, silently dropping `-DSSCMA`, `-DIC_PACKAGE_WLCSP65`
and the rest of the app's defines. See `patches/README.md` for the measurement.

### 10.2 ESP32 (XIAO ESP32-S3) — code and dependencies

**Toolchain:** `arduino-cli` (Homebrew) with the `esp32:esp32` core installed
(`arduino-cli core install esp32:esp32`). No other libraries — the sketch is
dependency-free by design (§6).

**Board / FQBN** (set in `esp_node/xiao_esp32s3/build.sh`, do not change casually):
```
esp32:esp32:XIAO_ESP32S3:PSRAM=opi,USBMode=hwcdc,CDCOnBoot=cdc
```
OPI PSRAM is mandatory (frame buffers live there); `USBMode=hwcdc` keeps the monitor on the
native USB peripheral so GPIO43/44 stay free for `Serial1` (the Grove link).

**Files** (`esp_node/xiao_esp32s3/`):
- `xiao_esp32s3.ino` — reader task (UART1 → framed lines, pinned to core 1, high priority),
  sender task (lines → TCP, core 0), AT control (`AT+BREAK`, optional `AT+SENSOR`,
  `AT+INVOKE`), and the WiFi/TCP reconnect state machine.
- `at_frame.h` — AT line parsing (base64 image extraction, in-place decode), host-testable
  with no Arduino dependencies. Re-run its tests after any change:
  ```bash
  cd esp_node/tools && c++ -std=c++17 -Wall -Wextra -o /tmp/test_at_frame test_at_frame.cpp && /tmp/test_at_frame
  ```
- `config.h` (gitignored — copy from `config.h.example`) — WiFi credentials, the Pi's host
  and port, results-only vs. image mode, sensor preset, and buffer sizes. See §10.4 for the
  actual values used.

**Build and flash:**
```bash
cd esp_node/xiao_esp32s3
cp config.h.example config.h   # then edit — see §10.4 for the values that were used
./build.sh --upload --port /dev/cu.usbmodem1101
```

**Bug found and fixed during bring-up — WiFi reconnect storm.** The original `loop()` called
`WiFi.reconnect()` unconditionally whenever `WiFi.status() != WL_CONNECTED`, once per second
(`delay(1000)` at the top of `loop()`). The Arduino-ESP32 core already auto-reconnects on
disconnect by default, so this was both redundant and actively harmful: if `WiFi.status()`
hadn't yet settled to `WL_CONNECTED` (e.g. mid-association or mid-DHCP), this tore the
in-progress connection down and restarted it — forever, once a second. Confirmed on the Pi
side via `journalctl`: `wpa_supplicant` showed `EAPOL-4WAY-HS-COMPLETED` (auth succeeding)
immediately followed by `AP-STA-DISCONNECTED`, on an exact 1-second cadence matching the
sketch's `delay(1000)`. Fixed by debouncing: only force a reconnect after 8 seconds of
sustained disconnection, not on every 1-second sample. The fix is in `xiao_esp32s3.ino`'s
`loop()` — grep for `last_connected_ms`.

### 10.3 Raspberry Pi — code, dependencies, and structure

**Layout on the Pi** (`/home/pi/camera_relay/`, deployed via `scp` from
`esp_node/rpi_receiver/` and `esp_node/tools/` — `start.sh`/`stop.sh`/`receiver.py`/
`fake_node.py`/`requirements.txt` are all committed to this repo under
`esp_node/rpi_receiver/`; the rest is created at runtime on the Pi):
```
camera_relay/
├── venv/                # created by start.sh (python3 -m venv venv); NOT committed, Pi-local
├── receiver.py           # TCP listener (port 5010) + MJPEG HTTP server (port 8080)
├── start.sh              # checks/installs deps, then launches receiver.py detached
├── stop.sh               # stops it
├── fake_node.py          # synthetic-frame generator, stands in for the ESP32 (testing only)
├── requirements.txt       # opencv-python-headless, numpy
├── receiver.pid           # written by start.sh
└── receiver.log           # start.sh's stdout+stderr redirect target
```

**Dependencies:** `opencv-python-headless` (this Pi is headless — no X11/display) and
`numpy`, per `requirements.txt`. `start.sh` handles the venv and the install itself (see
below) — you normally don't run pip by hand.

**Running it:**
```bash
cd /home/pi/camera_relay
./start.sh          # checks deps, installs if missing, starts receiver.py detached
./stop.sh           # stops it
```
`start.sh` refuses to run if `receiver.py` is already running (checked via `pgrep`), checks
whether `cv2`/`numpy` actually import before deciding to install anything, and launches with
`setsid nohup` (see the process-survival note below) so it outlives the SSH session that
started it. `stop.sh` sends SIGTERM, waits up to 5s, then SIGKILL if needed.

View the live annotated feed at `http://<pi-ip>:8080/` (e.g. `http://192.168.2.2:8080/` over
the USB-ethernet link from the Mac, or `http://10.42.0.1:8080/` from anything on the AP).

**Bug found and fixed during bring-up — wrong OpenCV package in requirements.txt.**
`requirements.txt` listed plain `opencv-python`, with only a *comment* recommending
`opencv-python-headless` for a headless Pi — the dependency line itself was never updated to
match the comment. Plain `opencv-python` bundles Qt/GTK GUI bindings that dynamically link
against `libGL.so.1`, which this Pi doesn't have, so a clean `pip install -r
requirements.txt` on a fresh venv reproducibly failed with `ImportError: libGL.so.1: cannot
open shared object file`. This looked like a corrupted/truncated download at first (the
~40 MB wheel over a phone-tethered hotspot uplink is a plausible failure mode too, and
`start.sh` retries + `--no-cache-dir` for exactly that reason) — but it reproduced 100% of
the time even with forced fresh re-downloads, which pointed at a real package difference
instead. Comparing `pip list` between the working (manually-installed, months-old) venv and
a fresh one confirmed it: working venv had `opencv-python-headless`, fresh venv had
`opencv-python`, both installed from what should have been the same requirements. Fixed by
actually changing `requirements.txt`'s dependency line to `opencv-python-headless`.

**Bug found and fixed during bring-up — single-connection receiver.** The original
`receiver.py` called `listener.accept()` once, then blocked forever inside that one
connection's read loop. If the ESP32 rebooted (new firmware, WiFi drop, etc.), its *old* TCP
connection rarely got a clean FIN — the peer just vanished — so the receiver stayed stuck
reading a dead socket forever, never noticing a new connection sitting unaccepted behind it.
Symptom: the MJPEG page in Chrome would hang on "loading" forever, because the shared
frame mailbox was never being updated. Fixed by restructuring `main()` to accept
continuously in a loop, spawning a handler thread per connection and closing whichever
connection was previously active whenever a new one arrives — newest connection always wins,
which is correct for a single-camera link.

**Process survival over SSH.** Plain `nohup ... &` was intermittently killed if the SSH
channel dropped mid-command, before the child had fully detached. `setsid nohup ... &`
(full session detach, immune to the closing channel's signals) fixed it reliably. This is
*not* a systemd `KillUserProcesses` issue — that's disabled on this Pi
(`/etc/systemd/logind.conf`) — it was specifically about the launching SSH command's own
channel closing uncleanly.

Note: none of this (venv, `setsid`, log rotation) is a permanent service. A Pi reboot will
not bring `receiver.py` back up; there's no systemd unit for it yet. Add one if persistence
across reboots is needed.

### 10.4 Network setup — Pi as WiFi AP

The Pi's only network path from the Mac is a private USB-ethernet link (`192.168.2.0/24`,
not shared with WiFi), so the ESP32 can't reach it over the Mac's own WiFi. Instead, the Pi
runs its own WiFi AP via NetworkManager (Debian/Raspberry Pi OS Bookworm+; no hostapd/dnsmasq
config needed, `nmcli` drives it directly):

```bash
sudo nmcli radio wifi on
sudo nmcli device wifi hotspot ifname wlan0 con-name <your-ap-name> ssid <your-ssid> password <your-password>
sudo nmcli connection modify <your-ap-name> connection.autoconnect yes   # survives reboot
```

This gives `wlan0` the address `10.42.0.1/24` (NetworkManager's hotspot default) and hands
out DHCP leases to clients — confirmed via `nmcli device show wlan0` and `journalctl`
(`dnsmasq-dhcp` / `wpa_supplicant` `AP-STA-CONNECTED` lines).

**`config.h` values used** for this network (see §10.2 for the file's location):
```c
#define WIFI_SSID     "<your ssid>"
#define WIFI_PASSWORD "<your password>"
#define WIFI_HOSTNAME "grove-esp-node"
#define RPI_HOST      "10.42.0.1"     // the Pi AP's own gateway IP
#define RPI_PORT      5010
#define INVOKE_RESULTS_ONLY 0          // boxes + JPEG; needs the UART1 patch (§10.1)
#define SENSOR_PRESET_INDEX -1
#define FRAME_BUF_SIZE (96 * 1024)
#define UART_RX_RING_SIZE (32 * 1024)
#define INVOKE_WATCHDOG_MS 8000
```

### 10.5 Alternative: receiver on a Mac, network via phone hotspot

`receiver.py` has no Pi-specific dependency -- it is plain Python + OpenCV + numpy, so it
runs unmodified on macOS. `mac_receiver/` swaps only what actually differs from §10.3:

- **No WiFi AP setup on the Mac.** macOS cannot rebroadcast its own WiFi as a hotspot (it
  can only *share* a *different* interface, e.g. Ethernet, out over WiFi). The simplest
  network for a WiFi-only Mac is a phone's mobile hotspot: the Mac and the ESP32 both join
  it as ordinary WiFi clients, and the ESP32 connects to the Mac's IP on that network. No
  NetworkManager/nmcli step is needed.
- **Plain `opencv-python`, not headless.** The Mac has a real display, so
  `mac_receiver/start.sh` runs with `--display` (a local OpenCV window) in addition to
  `--mjpeg-port 8080`. `mac_receiver/requirements.txt` installs the GUI-enabled build
  accordingly.
- **Foreground, not a service.** It runs directly in Terminal (`Ctrl+C` to stop) -- no
  `setsid`/`systemd` machinery, since there is no SSH session to survive.

**Setup:**

1. Join the phone's hotspot from both the Mac (WiFi menu) and, via `config.h`, the ESP32:
   ```c
   #define WIFI_SSID     "<hotspot ssid>"
   #define WIFI_PASSWORD "<hotspot password>"
   ```
2. Start the receiver -- it prints the Mac's IP on that network:
   ```bash
   cd usecases/esp32node/mac_receiver
   ./start.sh
   ```
3. Put the printed IP in `config.h`'s `RPI_HOST`, then build/flash as in §7 step 4 /
   §10.2. A phone hotspot commonly reassigns client IPs on reconnect, so re-check this if
   the ESP32 stops finding the Mac after either device rejoins.
4. View the feed at `http://<mac-ip>:8080/`, or in the OpenCV window `start.sh` opens
   directly on the Mac.

### 10.6 Result

With all of the above in place: Grove board camera → UART1 (patched `Serial` transport) →
ESP32 (`xiao_esp32s3.ino`) → WiFi (Pi's own hotspot) → TCP → `receiver.py` → MJPEG →
browser, sustaining roughly **7 fps at 17–30 kB/s** with real detections and live JPEG
frames, confirmed rendering correctly in Chrome at `http://192.168.2.2:8080/`.
