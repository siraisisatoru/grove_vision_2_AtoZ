"""
Grove Vision AI V2 - AT command interface over serial.

AT command protocol:
  Send:    AT+<CMD>\r
  Receive: \r{"type":<0|1>,"name":"<CMD>","code":<int>,"data":<any>}\n

  type 0 = operation response (reply to our command)
  type 1 = event response (async inference/sample result)
"""

import glob
import serial
import serial.tools.list_ports
import json
import threading
import time
import logging
from typing import Callable, Dict, List, Optional, Any

log = logging.getLogger(__name__)

# Must match board.c's console_setup() call on the firmware side. A lower
# rate (460800) was tried to work around image transport corruption during
# streaming and made it measurably worse — reverted.
BAUDRATE = 921600
READ_TIMEOUT = 5.0

# Firmware bug workaround (sscma_face_recognition <= v1.2, likely all DMA-TX
# console firmwares): the WE2's shared DMAC3 interrupt can lose a UART TX-done
# event under load, silencing the stream even though the board keeps running.
# Any byte SENT to the board makes the driver ISR re-scan its status and
# dispatch the overdue completion — proven on hardware to revive the stream
# instantly. A lone b"\n" is consumed silently by the firmware's AT parser
# (single-char lines are dropped before execute()), so it costs no response
# traffic. Enabled per-connection via connect(..., keepalive=True).
KEEPALIVE_INTERVAL = 0.2  # seconds

# CH343 USB-serial bridge used on Grove Vision AI V2
_CH343_VID = 0x1A86
_CH343_PID = 0x55D3


def find_port() -> Optional[str]:
    """Auto-detect the Grove Vision AI V2 serial port by CH343 VID/PID."""
    try:
        for p in serial.tools.list_ports.comports():
            if p.vid == _CH343_VID and p.pid == _CH343_PID:
                return p.device
    except Exception:
        pass
    # Fallback: first usbmodem / ttyACM / ttyUSB
    for pattern in ("/dev/cu.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None

# ma_model_type_t, verbatim from sscma_micro/sscma/core/ma_types.h. This is the
# decoder class ModelFactory actually bound to the loaded .tflite, reported by the
# device as data.algorithm.type in the AT+INVOKE response.
#
# NOTE: do NOT read the type from AT+MODELS? / AT+MODEL? — those report the model's
# metadata header, which is 0 for every model we build (Vela output carries no sscma
# metadata). AT+ALGO? is also 0. The INVOKE response is the only place the real bound
# type appears. (The previous table here mapped 1..4 to FOMO/PFLD/YOLO/YOLO Pose,
# which matches no firmware enum and was unused.)
MODEL_TYPE = {
    0:  "undefined",
    1:  "FOMO",
    2:  "PFLD",
    3:  "YOLOv5",
    4:  "IMCLS",
    5:  "YOLOv8 Pose",
    6:  "YOLOv8",
    7:  "NVIDIA Det",
    8:  "YOLO World",
    9:  "YOLO11",
    10: "YOLO11 Pose",
    11: "YOLO11 Seg",
    12: "YOLOv8 Seg",
    13: "RTMDet",
    14: "YOLO26",
    15: "YOLO26 Pose",
    16: "YOLO26 Seg",
    17: "BiSeNetV2",
}


def model_type_name(type_id) -> str:
    """Human-readable name for a ma_model_type_t value."""
    if type_id is None:
        return "unknown"
    return MODEL_TYPE.get(type_id, f"type {type_id}")
SENSOR_TYPE = {1: "Camera", 2: "IMU"}

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


class GroveVisionAI:
    """
    Thread-safe interface to Grove Vision AI V2 via the SSCMA AT command protocol.

    Usage:
        device = GroveVisionAI()
        device.connect("/dev/cu.usbmodem...")
        device.set_callback(lambda msg: print(msg))
        device.start_invoke()   # streams type-1 events to callback
        ...
        device.stop_stream()
        device.disconnect()
    """

    def __init__(self):
        self._ser: Optional[serial.Serial] = None
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        self._write_lock = threading.Lock()
        self._resp_ready = threading.Event()
        self._last_resp: Optional[Dict] = None
        self._event_callback: Optional[Callable[[Dict], None]] = None
        self._raw_callback: Optional[Callable[[str], None]] = None
        self.port: Optional[str] = None
        self.is_streaming = False
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()
        self._rx_seen = threading.Event()  # set by reader on first received data
        # ma_model_type_t of the decoder the firmware bound to the loaded model.
        # Populated by start_invoke() from the device's own INVOKE response; stays
        # None until then. See MODEL_TYPE above for why this is the only reliable source.
        self.algorithm_type: Optional[int] = None

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self, port: str, baudrate: int = BAUDRATE, boot_wait: float = 4.0,
                keepalive: bool = False) -> bool:
        if self._ser and self._ser.is_open:
            self.disconnect()
        self._rx_seen.clear()
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            xonxoff=False,
            rtscts=False,
            timeout=READ_TIMEOUT,
        )
        self._ser.flushInput()
        self._ser.flushOutput()
        self.port = port
        self._running = True
        self._read_thread = threading.Thread(target=self._reader, daemon=True, name="sscma-reader")
        self._read_thread.start()
        log.info("Connected to %s @ %d baud", port, baudrate)
        # Opening the port resets the WE2 board via RTS (CH343 bridge behaviour on macOS).
        # Wait for the bootloader + firmware to finish starting before sending AT commands.
        if boot_wait > 0:
            time.sleep(boot_wait)
            self._ser.reset_input_buffer()
        if keepalive:
            self.start_keepalive()
        return True

    def start_keepalive(self, interval: float = KEEPALIVE_INTERVAL):
        """Periodically poke the board with a benign byte so a lost UART TX-done
        interrupt (firmware DMAC3 bug — see KEEPALIVE_INTERVAL) is re-dispatched
        by the RX interrupt within ~interval seconds. Reuses the already-open
        port object; NEVER reopens (open() toggles RTS and hard-resets the board).
        """
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, args=(interval,),
            daemon=True, name="sscma-keepalive")
        self._keepalive_thread.start()

    def stop_keepalive(self):
        self._keepalive_stop.set()
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=1.0)
            self._keepalive_thread = None

    def _keepalive_loop(self, interval: float):
        while not self._keepalive_stop.wait(interval):
            if not (self._running and self._ser and self._ser.is_open):
                continue
            # Don't poke the bootloader window right after the RTS reset:
            # only send once the firmware has said something.
            if not self._rx_seen.is_set():
                continue
            try:
                with self._write_lock:  # never interleave with _cmd() writes
                    self._ser.write(b"\n")
            except Exception as e:
                log.debug("Keepalive write error: %s", e)

    def disconnect(self):
        self.stop_keepalive()
        self._running = False
        self.is_streaming = False
        if self._read_thread:
            self._read_thread.join(timeout=3.0)
            self._read_thread = None
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None
        self.port = None
        log.info("Disconnected")

    @property
    def is_connected(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_callback(self, fn: Callable[[Dict], None]):
        """Called for every type-1 (event) response — inference/sample results."""
        self._event_callback = fn

    def set_raw_callback(self, fn: Callable[[str], None]):
        """Called for every non-JSON line received (boot messages, etc.)."""
        self._raw_callback = fn

    # -------------------------------------------------------------------------
    # Reader thread — routes all incoming serial data
    # -------------------------------------------------------------------------

    def _reader(self):
        # Read one byte to block until data arrives (respecting READ_TIMEOUT),
        # then drain whatever else is already buffered in a single syscall.
        # A naive read(1)-per-iteration loop with `buf += chunk` is O(n^2) for
        # a large message (each += copies the whole buffer again, since bytes
        # are immutable) and issues one syscall per byte — for a ~20-30 KB
        # base64 image frame, that combined overhead can make this thread fall
        # behind the incoming 921600-baud stream, risking dropped/reordered
        # bytes at the OS serial-buffer level and corrupting message framing
        # (seen as garbage lines landing in the raw callback instead of a
        # clean JSON event). A bigger/more detailed frame (e.g. a real subject
        # in view vs. a blank background) compresses to a larger JPEG, making
        # this more likely to bite exactly when real detections start.
        buf = bytearray()
        while self._running and self._ser and self._ser.is_open:
            try:
                chunk = self._ser.read(1)
                if not chunk:
                    continue
                waiting = self._ser.in_waiting
                if waiting:
                    chunk += self._ser.read(waiting)
                buf.extend(chunk)
                self._rx_seen.set()
                while True:
                    idx = buf.find(b"\n")
                    if idx < 0:
                        break
                    raw = bytes(buf[:idx])
                    del buf[:idx + 1]
                    self._dispatch(raw)
            except serial.SerialException as e:
                if "Bad file descriptor" in str(e):
                    # Transient EBADF during post-reset USB re-enumerate — retry briefly
                    time.sleep(0.1)
                    continue
                log.error("Serial read error: %s", e)
                break
            except Exception as e:
                log.debug("Reader exception: %s", e)

    def _dispatch(self, raw: bytes):
        text = raw.decode("ascii", errors="ignore").strip().lstrip("\r").strip()
        if not text:
            return
        if text.startswith("{"):
            try:
                msg: Dict = json.loads(text)
            except json.JSONDecodeError:
                return
            if msg.get("type") == 1:
                if self._event_callback:
                    try:
                        self._event_callback(msg)
                    except Exception as e:
                        log.debug("Event callback error: %s", e)
            else:
                self._last_resp = msg
                self._resp_ready.set()
        else:
            if self._raw_callback:
                try:
                    self._raw_callback(text)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Command send/receive
    # -------------------------------------------------------------------------

    def _cmd(self, body: str, timeout: float = 8.0) -> Optional[Dict]:
        """Send AT+<body>\r and wait for the type-0 operation response."""
        if not self.is_connected:
            raise ConnectionError("Device not connected")
        with self._write_lock:
            self._resp_ready.clear()
            self._last_resp = None
            self._ser.write(f"AT+{body}\r".encode("ascii"))
        got = self._resp_ready.wait(timeout=timeout)
        if not got:
            log.warning("Timeout waiting for response to AT+%s", body)
            return None
        return self._last_resp

    def _cmd_ok(self, body: str, timeout: float = 8.0) -> bool:
        r = self._cmd(body, timeout)
        return r is not None and r.get("code") == 0

    # -------------------------------------------------------------------------
    # Device info
    # -------------------------------------------------------------------------

    def get_id(self) -> str:
        r = self._cmd("ID?")
        return r["data"] if r else ""

    def get_name(self) -> str:
        r = self._cmd("NAME?")
        return r["data"] if r else ""

    def get_version(self) -> Dict:
        r = self._cmd("VER?")
        return r["data"] if r else {}

    def get_status(self) -> Dict:
        r = self._cmd("STAT?")
        return r["data"] if r else {}

    def get_device_info(self) -> Dict:
        return {
            "id": self.get_id(),
            "name": self.get_name(),
            "version": self.get_version(),
            "status": self.get_status(),
            "port": self.port,
        }

    # -------------------------------------------------------------------------
    # Models & sensors
    # -------------------------------------------------------------------------

    @staticmethod
    def _unwrap_list(data, key: str) -> List[Dict]:
        """Unwrap the firmware's v1 ATServer envelope.

        EncoderJSON::write(vector<T>) always nests under data[key] (e.g.
        data.models, data.sensors) — including for "current" (singular)
        queries, which wrap the one item in a 1-element vector rather than
        using the dedicated single-object writer. Handles both shapes
        defensively in case a future firmware build fixes this.
        """
        if isinstance(data, dict):
            return data.get(key, [])
        if isinstance(data, list):
            return data
        return []

    def list_models(self) -> List[Dict]:
        r = self._cmd("MODELS?")
        if not r or r.get("code") != 0:
            return []
        return self._unwrap_list(r.get("data"), "models")

    def get_current_model(self) -> Optional[Dict]:
        r = self._cmd("MODEL?")
        if not r or r.get("code") != 0:
            return None
        data = r.get("data")
        if isinstance(data, dict) and "model" in data:
            return data["model"]
        models = self._unwrap_list(data, "models")
        return models[0] if models else None

    def load_model(self, model_id: int) -> bool:
        return self._cmd_ok(f"MODEL={model_id}", timeout=10.0)

    def list_sensors(self) -> List[Dict]:
        r = self._cmd("SENSORS?")
        if not r or r.get("code") != 0:
            return []
        return self._unwrap_list(r.get("data"), "sensors")

    def get_current_sensor(self) -> Optional[Dict]:
        r = self._cmd("SENSOR?")
        if not r or r.get("code") != 0:
            return None
        data = r.get("data")
        if isinstance(data, dict) and "sensor" in data:
            return data["sensor"]
        sensors = self._unwrap_list(data, "sensors")
        return sensors[0] if sensors else None

    def set_sensor(self, sensor_id: int, enable: int = 1, opt_id: int = 0) -> bool:
        return self._cmd_ok(f"SENSOR={sensor_id},{enable},{opt_id}")

    # -------------------------------------------------------------------------
    # Thresholds
    # -------------------------------------------------------------------------

    def get_score_threshold(self) -> int:
        r = self._cmd("TSCORE?")
        if r and r.get("code") == 0 and isinstance(r.get("data"), (int, float)):
            return int(r["data"])
        return 60

    def get_iou_threshold(self) -> int:
        r = self._cmd("TIOU?")
        if r and r.get("code") == 0 and isinstance(r.get("data"), (int, float)):
            return int(r["data"])
        return 50

    def set_score_threshold(self, score: int) -> bool:
        return self._cmd_ok(f"TSCORE={score}")

    def set_iou_threshold(self, iou: int) -> bool:
        return self._cmd_ok(f"TIOU={iou}")

    def get_algo(self) -> Optional[int]:
        r = self._cmd("ALGO?")
        if r and r.get("code") == 0 and isinstance(r.get("data"), (int, float)):
            return int(r["data"])
        return None

    def set_algo(self, algo: int) -> bool:
        """Set the post-processing algorithm (AT+ALGO=<n>).

        ALGO=0: decoded-box detection (use with no_post=False TFLite — DFL decode in model).
        ALGO=9: YOLO11 DFL decoder on-device (use with no_post=True raw tensors).
        The correct value depends on how the model was exported.
        """
        return self._cmd_ok(f"ALGO={algo}")

    # -------------------------------------------------------------------------
    # Streaming — invoke / sample
    # -------------------------------------------------------------------------

    def start_invoke(self, n_times: int = -1, differed: int = 0, result_only: int = 0) -> bool:
        """
        Start inference loop.

        n_times: -1 = continuous, N = run N times
        differed: unused positional slot, always send 0 — see below
        result_only: 0 = include JPEG image, 1 = results only (faster)

        The AT+INVOKE argument count is NOT the same across firmware builds:
        stock firmware (grove_vision_ai_v2_20250102.img, MA_VERSION 2025.01.02)
        registers INVOKE with argc=3 (N_TIMES,DIFFERED,RESULTS_ONLY) and its
        AT parser hard-rejects a call with fewer args ("Command INVOKE got
        wrong arguments", verified live against the device). The newer
        yolo26_support firmware build (MA_VERSION 2025.02.11) dropped the
        "differed" param — addService("INVOKE", ..., "N_TIMES,RESULTS_ONLY")
        gives argc=2 — but ma_server_at.cpp's arg-splitting loop stops as soon
        as it has collected argc+1 tokens (cmd + argc values) rather than
        erroring on extras, so a trailing 3rd value is just silently ignored,
        not rejected.
        Sending 3 args (n_times, differed=0, result_only) therefore works on
        BOTH firmwares: satisfies the strict stock parser, and on the newer
        firmware the harmless "differed" value lands in the RESULTS_ONLY slot
        (fine since it's always 0, matching the common result_only=0 case) —
        every current caller in this codebase only ever needs result_only=0.
        Only sending 2 args (no differed) breaks the stock firmware outright,
        which is what happened here — don't drop this again without a real
        firmware-version check for both builds.

        Events arrive via the callback set with set_callback().
        data keys: count, perf, boxes, classes, keypoints, image (if result_only=0)
        boxes format: [x, y, w, h, score, class_id]
        """
        r = self._cmd(f"INVOKE={n_times},{differed},{result_only}", timeout=8.0)
        if r and r.get("code") == 0:
            # The response reports the decoder the firmware actually bound
            # (_algorithm->getType() in invoke.hpp's directReply). Record it so
            # callers can display the real architecture instead of guessing.
            algo = (r.get("data") or {}).get("algorithm") or {}
            self.algorithm_type = algo.get("type")
            self.is_streaming = True
            return True
        return False

    def start_sample(self, n_times: int = -1) -> bool:
        """
        Capture camera frames without inference.
        Events have: count, image (base64 JPEG)
        """
        r = self._cmd(f"SAMPLE={n_times}", timeout=8.0)
        if r and r.get("code") == 0:
            self.is_streaming = True
            return True
        return False

    def stop_stream(self):
        """Stop an active SAMPLE/INVOKE stream cleanly using AT+BREAK."""
        self.is_streaming = False
        if not self.is_connected:
            return
        try:
            with self._write_lock:
                self._ser.write(b"AT+BREAK\r")
            # Drain any remaining event frames
            time.sleep(0.3)
            if self._ser and self._ser.is_open:
                self._ser.flushInput()
        except Exception as e:
            log.debug("Stop stream error: %s", e)

    # -------------------------------------------------------------------------
    # Single-shot capture
    # -------------------------------------------------------------------------

    def capture_frame(self, timeout: float = 10.0) -> Optional[Dict]:
        """Capture a single camera frame. Returns event data dict or None."""
        result: list = []
        evt = threading.Event()

        orig_cb = self._event_callback

        def _once(msg):
            if msg.get("name") in ("SAMPLE", "INVOKE"):
                result.append(msg.get("data", {}))
                evt.set()

        self._event_callback = _once
        try:
            self._cmd("SAMPLE=1", timeout=5.0)
            evt.wait(timeout=timeout)
        finally:
            self._event_callback = orig_cb

        return result[0] if result else None

    def infer_once(self, timeout: float = 15.0) -> Optional[Dict]:
        """Run inference once and return the result."""
        result: list = []
        evt = threading.Event()
        orig_cb = self._event_callback

        def _once(msg):
            if msg.get("name") == "INVOKE":
                result.append(msg.get("data", {}))
                evt.set()

        self._event_callback = _once
        try:
            self._cmd("INVOKE=1,0,0", timeout=5.0)
            evt.wait(timeout=timeout)
        finally:
            self._event_callback = orig_cb

        return result[0] if result else None

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------

    def reset(self):
        """Soft reset via AT+RST! (firmware reboots, stays in firmware mode)."""
        if not self.is_connected:
            return
        with self._write_lock:
            self._ser.write(b"AT+RST!\r")
        time.sleep(2.0)
        if self._ser and self._ser.is_open:
            self._ser.flushInput()

    def hard_reset(self, boot_wait: float = 2.0):
        """Hardware reset via RTS toggle on the CH343 bridge, then wait for firmware."""
        if not self.is_connected:
            return
        try:
            self._ser.setRTS(False)
            time.sleep(0.1)
            self._ser.setRTS(True)
            time.sleep(boot_wait)
            self._ser.flushInput()
        except Exception as e:
            log.debug("Hard reset error: %s", e)

    @staticmethod
    def list_ports() -> List[Dict]:
        return [
            {"device": p.device, "description": p.description, "hwid": p.hwid}
            for p in serial.tools.list_ports.comports()
        ]

    @staticmethod
    def label_for(class_id: int, labels: Optional[List[str]] = None) -> str:
        lst = labels or COCO_LABELS
        if 0 <= class_id < len(lst):
            return lst[class_id]
        return f"class_{class_id}"
