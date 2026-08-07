"""
Grove Vision AI V2 - XMODEM firmware and model flasher.

The board uses a CH343 USB-serial bridge (VID=0x1A86, PID=0x55D3). RTS is
wired to the WE2's RESET pin, so toggling RTS causes an instant hardware reset
without any USB re-enumeration — the serial port stays open throughout.

Software reset sequence (no physical button needed):
  1. Toggle RTS low → high (hardware reset via CH343).
  2. Spam '1\\r' while re-toggling RTS every 2.5 s to keep creating fresh
     boot windows (the bootloader accepts input for only ~100 ms per boot).
  3. When bootloader prints the xmodem banner, settle 1 s, flush stale echoes,
     confirm with '1\\r', then start XMODEM.

Manual fallback: hold BOOT + press RESET — the same '1\\r' spam catches it.

Flashing sequence:
  - [Optional] firmware .img via XMODEM
  - For each model:
      a. 128-byte address preamble via XMODEM (SOH, always 128 B)
      b. "reboot?" → 'n'
      c. .tflite payload via XMODEM (SOH or 1K)
      d. "reboot?" → 'n' (if more models) or 'y' (last)
"""

import os
import time
import serial
import logging
from typing import Callable, Optional, List, Tuple

log = logging.getLogger(__name__)

# Bootloader detection strings — matched case-insensitively
_BANNER_NEEDLE = "send data using the xmodem protocol"
_REBOOT_NEEDLE = "do you want to end file transmission and reboot system"

# XMODEM control bytes
_SOH = 0x01   # 128-byte block
_STX = 0x02   # 1024-byte block
_EOT = 0x04   # end of transmission
_ACK = 0x06   # acknowledge
_NAK = 0x15   # negative acknowledge
_CAN = 0x18   # cancel
_CRC = 0x43   # 'C' — CRC-mode start signal from receiver

PREAMBLE_SIZE = 128  # address preamble is always a single 128-byte SOH block


def _crc16(data: bytes) -> int:
    crc = 0
    for ch in data:
        crc ^= ch << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _build_preamble(address: int, offset: int = 0) -> bytes:
    """128-byte header that tells the bootloader where to write the next file."""
    hdr = bytearray([0xFF] * PREAMBLE_SIZE)
    hdr[0] = 0xC0
    hdr[1] = 0x5A
    hdr[2:6]  = address.to_bytes(4, "little")
    hdr[6:10] = offset.to_bytes(4, "little")
    hdr[10] = 0x5A
    hdr[11] = 0xC0
    return bytes(hdr)


class Flasher:
    """
    Flash firmware (.img) and/or TFLite models onto Grove Vision AI V2.

    progress_callback(stage, done, total, message)
        stage: "waiting" | "firmware" | "model_preamble" | "model" | "xmodem"
               | "done" | "error"
        done/total: block counts (0/0 when not applicable)
    """

    def __init__(self, port: str, baudrate: int = 921600,
                 progress_callback: Optional[Callable] = None):
        self.port = port
        self.baudrate = baudrate
        self._cb = progress_callback or (lambda *a: None)
        self._ser: Optional[serial.Serial] = None

    def _progress(self, stage: str, done: int = 0, total: int = 0, msg: str = ""):
        try:
            self._cb(stage, done, total, msg)
        except Exception:
            pass

    # ── Serial helpers ─────────────────────────────────────────────────────

    def _open(self):
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            xonxoff=False,
            rtscts=False,
            timeout=0.2,
        )
        self._ser.flushInput()
        self._ser.flushOutput()

    def _close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def _toggle_rts(self):
        """Hardware-reset the WE2 via the CH343's RTS→RESET wiring."""
        try:
            self._ser.setRTS(False)
            time.sleep(0.08)
            self._ser.setRTS(True)
        except Exception:
            pass

    def _read_until(self, wanted: tuple, timeout: float) -> int:
        """Read bytes until one of the wanted control bytes appears, or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            for c in chunk:
                if c in wanted:
                    return c
        return -1

    # ── Boot detection ─────────────────────────────────────────────────────

    def _wait_for_banner(self, timeout: float = 25.0, software_reset: bool = False) -> bool:
        """
        Spam '1\\r' until the xmodem-ready banner appears.

        With software_reset=True: toggle RTS immediately, then re-toggle every
        2.5 s to keep creating fresh ~100 ms boot windows.
        With software_reset=False: user must hold BOOT+RESET; loop still catches it.
        """
        if software_reset:
            self._progress("waiting", 0, 0, "Resetting device into bootloader via RTS...")
            self._toggle_rts()
            self._ser.reset_input_buffer()
        else:
            self._progress("waiting", 0, 0, "Hold BOOT + press RESET on the device...")

        deadline    = time.time() + timeout
        buf         = bytearray()
        last_reset  = time.time()
        hinted      = False

        while time.time() < deadline:
            self._ser.write(b"1\r")
            self._ser.flush()
            time.sleep(0.01)

            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                text = bytes(buf).decode("latin-1", "ignore").lower()
                if _BANNER_NEEDLE in text:
                    # Banner found — settle, flush stale echoes, send final '1\r'
                    time.sleep(1.0)
                    self._ser.reset_input_buffer()
                    self._ser.write(b"1\r")
                    self._ser.flush()
                    log.info("Bootloader banner received, XMODEM mode starting")
                    return True
                if len(buf) > 8192:
                    del buf[:-1024]

            now = time.time()
            # Re-trigger RTS periodically to generate new boot windows
            if software_reset and (now - last_reset) > 2.5:
                self._toggle_rts()
                last_reset = now

            if not hinted and (now - (deadline - timeout)) > 6.0:
                msg = ("Still waiting... try pressing physical RESET on the board."
                       if software_reset else
                       "Still waiting — keep holding BOOT while pressing RESET.")
                self._progress("waiting", 0, 0, msg)
                hinted = True

        return False

    # ── Reboot prompt ──────────────────────────────────────────────────────

    def _wait_for_reboot_prompt(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        buf = bytearray()
        while time.time() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                if _REBOOT_NEEDLE in bytes(buf).decode("latin-1", "ignore").lower():
                    return True
                if len(buf) > 4096:
                    del buf[:-512]
            else:
                time.sleep(0.01)
        return False

    # ── Hand-rolled XMODEM-CRC ─────────────────────────────────────────────

    def _xmodem_send(self, data: bytes, label: str, block_size: int = 128) -> bool:
        """
        XMODEM-CRC transfer.

        Waits explicitly for the receiver's 'C' before sending block 1. The WE2
        bootloader emits 'C' only ~once/second; the xmodem library races this
        cadence and fails on block 1 — waiting for it explicitly is reliable.
        """
        self._progress("xmodem", 0, 0, f"Waiting for XMODEM start ({label})...")
        if self._read_until((_CRC,), 8.0) < 0:
            self._progress("error", 0, 0, "Device never sent 'C' to start XMODEM")
            return False

        hdr_byte = _STX if block_size == 1024 else _SOH
        total    = (len(data) + block_size - 1) // block_size
        blk      = 1

        for off in range(0, len(data), block_size):
            chunk = data[off: off + block_size]
            if len(chunk) < block_size:
                chunk = chunk + bytes([0x1A] * (block_size - len(chunk)))  # SUB pad
            crc    = _crc16(chunk)
            packet = (
                bytes([hdr_byte, blk & 0xFF, 0xFF - (blk & 0xFF)])
                + chunk
                + bytes([(crc >> 8) & 0xFF, crc & 0xFF])
            )
            acked = False
            for _ in range(16):
                self._ser.write(packet)
                self._ser.flush()
                # 30 s timeout: WE2 NOR flash sector erase can take >4 s at
                # 64 KB boundaries; retrying too soon causes duplicate blocks.
                resp = self._read_until((_ACK, _NAK, _CAN), 30.0)
                if resp == _ACK:
                    acked = True
                    break
                if resp == _CAN:
                    self._progress("error", 0, 0, "Transfer cancelled by device")
                    return False
                # NAK or timeout → resend this block
            if not acked:
                self._progress("error", 0, 0, f"Block {blk} not acknowledged after retries")
                return False

            self._progress("xmodem", blk, total, f"{blk}/{total}  ({label})")
            blk += 1

        # End of transmission
        for _ in range(10):
            self._ser.write(bytes([_EOT]))
            self._ser.flush()
            if self._read_until((_ACK,), 3.0) == _ACK:
                return True
        return False

    def _skip_firmware_slot(self) -> bool:
        """
        Skip the firmware XMODEM slot when flashing model-only.

        After the bootloader banner the WE2 waits for XMODEM data (firmware).
        Sending EOT immediately tells it "no more data" → it shows the
        reboot prompt → caller answers 'n' → model slot opens.
        """
        self._progress("waiting", 0, 0, "Skipping firmware slot (sending EOT)...")
        got = self._read_until((_CRC,), 8.0)
        if got < 0:
            self._progress("error", 0, 0, "Bootloader never sent 'C' for firmware slot")
            return False
        self._ser.write(bytes([_EOT]))
        self._ser.flush()
        # Bootloader ACKs the EOT and then shows the reboot prompt
        self._read_until((_ACK,), 5.0)
        return True

    # ── Public API ─────────────────────────────────────────────────────────

    def flash(self,
              firmware_path: Optional[str] = None,
              models: Optional[List[Tuple[str, int, int]]] = None,
              protocol: str = "xmodem1k",
              software_reset: bool = False) -> bool:
        """
        Flash firmware and/or models.

        firmware_path : path to .img file (omit to flash models only)
        models        : list of (tflite_path, flash_address, offset)
                        e.g. [("model.tflite", 0x400000, 0x0)]
        protocol      : "xmodem1k" (1 KB blocks, default) or "xmodem" (128 B)
        software_reset: True → RTS-toggle reset (no button press needed)
                        False → user must hold BOOT + press RESET manually

        Returns True on success.
        """
        if not firmware_path and not models:
            raise ValueError("Provide at least firmware_path or models")

        payload_bs = 1024 if protocol == "xmodem1k" else 128

        self._progress("waiting", 0, 0, "Opening serial port...")
        try:
            self._open()
        except Exception as e:
            self._progress("error", 0, 0, f"Cannot open {self.port}: {e}")
            return False

        try:
            if not self._wait_for_banner(timeout=25.0, software_reset=software_reset):
                hint = ("RTS reset did not trigger the bootloader — "
                        "try manually: hold BOOT + press RESET, then retry."
                        if software_reset else
                        "Timed out waiting for boot. Hold BOOT and press RESET.")
                self._progress("error", 0, 0, hint)
                return False

            self._progress("waiting", 0, 0, "Bootloader ready, starting transfer...")

            # ── Firmware ──────────────────────────────────────────────────
            if firmware_path:
                fw = open(firmware_path, "rb").read()
                self._progress("firmware", 0, 0,
                    f"Sending firmware: {os.path.basename(firmware_path)} ({len(fw):,} B)")
                if not self._xmodem_send(fw, os.path.basename(firmware_path), payload_bs):
                    self._progress("error", 0, 0, "Firmware transfer failed")
                    return False
                self._progress("firmware", 1, 1, "Firmware sent!")
            elif models:
                # No firmware — skip the firmware XMODEM slot with EOT so the
                # bootloader advances to the reboot prompt and then model slot.
                if not self._skip_firmware_slot():
                    return False

            # ── Models ────────────────────────────────────────────────────
            if models:
                for idx, (model_path, address, offset) in enumerate(models):
                    # Wait for reboot prompt → 'n' (stay in flash mode)
                    self._progress("model_preamble", 0, 0,
                        f"Waiting for prompt before model {idx + 1}...")
                    if not self._wait_for_reboot_prompt(timeout=60):
                        self._progress("error", 0, 0, "Timed out waiting for reboot prompt")
                        return False
                    time.sleep(0.5)
                    self._ser.reset_input_buffer()
                    self._ser.write(b"n")
                    self._ser.flush()

                    # Preamble — always 128-byte SOH regardless of payload block size
                    preamble = _build_preamble(address, offset)
                    self._progress("model_preamble", 0, 1,
                        f"Sending address preamble @ {hex(address)}...")
                    if not self._xmodem_send(preamble, f"preamble {idx + 1}", block_size=128):
                        self._progress("error", 0, 0, "Preamble transfer failed")
                        return False

                    # Wait again → 'n' → send model
                    if not self._wait_for_reboot_prompt(timeout=60):
                        self._progress("error", 0, 0,
                            "Timed out waiting for prompt after preamble")
                        return False
                    time.sleep(0.5)
                    self._ser.reset_input_buffer()
                    self._ser.write(b"n")
                    self._ser.flush()

                    mdata = open(model_path, "rb").read()
                    self._progress("model", 0, 0,
                        f"Sending {os.path.basename(model_path)} ({len(mdata):,} B)...")
                    if not self._xmodem_send(mdata, os.path.basename(model_path), payload_bs):
                        self._progress("error", 0, 0, "Model transfer failed")
                        return False
                    self._progress("model", 1, 1, f"Model {idx + 1} sent!")

            # ── Final reboot ──────────────────────────────────────────────
            self._progress("done", 0, 0, "Waiting for final reboot prompt...")
            if not self._wait_for_reboot_prompt(timeout=60):
                self._progress("error", 0, 0, "Timed out waiting for final reboot prompt")
                return False
            time.sleep(0.5)
            self._ser.reset_input_buffer()
            self._ser.write(b"y")
            self._ser.flush()
            time.sleep(2.0)
            self._progress("done", 1, 1, "Flash complete! Device rebooting...")
            return True

        except Exception as e:
            self._progress("error", 0, 0, str(e))
            log.exception("Flash error")
            return False
        finally:
            self._close()
