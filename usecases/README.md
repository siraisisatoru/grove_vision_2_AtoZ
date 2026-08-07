# Use cases

Reference builds on top of Grove Vision AI V2, beyond the core firmware/model workflow.

## [esp32node](esp32node/README.md)

Replaces the USB cable with a WiFi link: a XIAO ESP32-S3 mounted on the Grove board's
XIAO connector reads the existing UART1 AT transport and forwards detection boxes (and
optionally JPEG frames) over WiFi to a Raspberry Pi, which parses them and serves a live
annotated MJPEG stream in a browser.

Verified working end-to-end on real hardware. See the linked README for the hardware
feasibility analysis, wire format, bandwidth budget, and step-by-step bring-up
instructions.
