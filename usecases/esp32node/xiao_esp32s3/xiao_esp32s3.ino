// Grove Vision AI V2 -> WiFi bridge for the XIAO ESP32-S3.
//
// Mounts on the Grove board's XIAO connector. D6/D7 land on the HX6538's
// PB6/PB7 = UART1 = the sscma firmware's `Serial` AT transport, which speaks the
// same protocol as the USB console. See ../README.md for the schematic mapping,
// the bandwidth budget, and the firmware patch the image path needs.
//
// This node deliberately does NOT parse the detection JSON. It only has to find
// the base64 image inside the AT event line, decode it, and forward the two
// pieces; the Raspberry Pi does the parsing where memory is free. That keeps the
// hot path allocation-free and dependency-free.
//
// Wire format to the Pi, little-endian, over one long-lived TCP connection:
//
//     "SSC1" | u32 json_len | u32 jpeg_len | json bytes | jpeg bytes
//
// json is the AT event line with the image value emptied ("image":""), so the
// Pi still sees boxes/perf/width/height/rotation exactly as the firmware sent
// them. jpeg_len is 0 in results-only mode.

#include <WiFi.h>

#include "at_frame.h"
#include "config.h"

#define MA_STR_(x) #x
#define MA_STR(x) MA_STR_(x)

// ---------------------------------------------------------------------------
// Link to the Grove board
// ---------------------------------------------------------------------------

// XIAO D6 = GPIO43 = our TX -> Grove PB6 (UART1_RX)
// XIAO D7 = GPIO44 = our RX <- Grove PB7 (UART1_TX)
static constexpr int kGroveTxPin = D6;
static constexpr int kGroveRxPin = D7;
static constexpr uint32_t kGroveBaud = 921600;  // fixed in firmware, not negotiable

#define GroveSerial Serial1

// ---------------------------------------------------------------------------
// Frame buffers and handoff
// ---------------------------------------------------------------------------

struct Frame {
    char* data;
    size_t len;
};

static Frame g_slots[2];
static char* g_carry = nullptr;  // bytes read past a line terminator

static QueueHandle_t g_free_q;   // Frame* the reader may fill
static QueueHandle_t g_ready_q;  // Frame* waiting to go out (depth 1, freshest wins)

// Counters, printed periodically over USB CDC.
static volatile uint32_t g_frames_rx = 0;
static volatile uint32_t g_frames_tx = 0;
static volatile uint32_t g_frames_dropped = 0;   // no free buffer / sender busy
static volatile uint32_t g_frames_oversize = 0;  // line longer than FRAME_BUF_SIZE
static volatile uint32_t g_bytes_rx = 0;
static volatile uint32_t g_last_frame_ms = 0;

// ---------------------------------------------------------------------------
// Reader task: UART1 -> framed lines
// ---------------------------------------------------------------------------

// Hands a completed line to the sender. Never blocks: if the sender is still
// busy with the previous frame, the older frame is recycled and the newer one
// takes its place. For a live view, freshest-wins beats a backlog.
static void publish(Frame* f) {
    if (xQueueSend(g_ready_q, &f, 0) == pdTRUE) {
        return;
    }
    Frame* stale = nullptr;
    if (xQueueReceive(g_ready_q, &stale, 0) == pdTRUE) {
        ++g_frames_dropped;
        xQueueSend(g_free_q, &stale, 0);
    }
    if (xQueueSend(g_ready_q, &f, 0) != pdTRUE) {
        ++g_frames_dropped;
        xQueueSend(g_free_q, &f, 0);
    }
}

static Frame* acquire_slot() {
    Frame* f = nullptr;
    xQueueReceive(g_free_q, &f, 0);
    return f;
}

static void reader_task(void*) {
    Frame* cur = acquire_slot();
    size_t len = 0;

    for (;;) {
        if (cur == nullptr) {
            // No buffer free. Keep draining the UART so the driver ring never
            // overflows mid-frame, and resync on the next line terminator.
            const int avail = GroveSerial.available();
            if (avail <= 0) {
                vTaskDelay(1);
                continue;
            }
            uint8_t scratch[512];
            const int n = GroveSerial.read(scratch, min<int>(avail, sizeof(scratch)));
            g_bytes_rx += n;
            if (memchr(scratch, '\n', n) != nullptr) {
                ++g_frames_dropped;
                cur = acquire_slot();
                len = 0;
            }
            continue;
        }

        const int avail = GroveSerial.available();
        if (avail <= 0) {
            vTaskDelay(1);
            continue;
        }

        size_t space = FRAME_BUF_SIZE - 1 - len;
        if (space == 0) {
            // Overlong line: nothing sane to do but discard it and resync.
            ++g_frames_oversize;
            len = 0;
            space = FRAME_BUF_SIZE - 1;
        }

        const int n = GroveSerial.read(reinterpret_cast<uint8_t*>(cur->data) + len,
                                       min<size_t>(static_cast<size_t>(avail), space));
        if (n <= 0) {
            continue;
        }
        g_bytes_rx += n;
        len += n;

        // A single read can straddle several lines.
        for (;;) {
            char* nl = static_cast<char*>(memchr(cur->data, '\n', len));
            if (nl == nullptr) {
                break;
            }
            const size_t line_len = static_cast<size_t>(nl - cur->data);
            const size_t tail_len = len - (line_len + 1);

            // Copy the carry out before handing the buffer to the sender.
            if (tail_len) {
                memcpy(g_carry, nl + 1, tail_len);
            }

            cur->len = line_len;
            cur->data[line_len] = '\0';
            ++g_frames_rx;
            g_last_frame_ms = millis();
            publish(cur);

            cur = acquire_slot();
            len = tail_len;
            if (cur == nullptr) {
                // Carry is lost with no buffer to put it in; the discard path
                // above will resync on the next terminator.
                len = 0;
                break;
            }
            if (tail_len) {
                memcpy(cur->data, g_carry, tail_len);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sender task: framed lines -> TCP
// ---------------------------------------------------------------------------

static WiFiClient g_client;

static bool ensure_connected() {
    if (g_client.connected()) {
        return true;
    }
    g_client.stop();
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }
    if (!g_client.connect(RPI_HOST, RPI_PORT, 3000)) {
        return false;
    }
    g_client.setNoDelay(true);
    Serial.printf("[net] connected to %s:%d\n", RPI_HOST, RPI_PORT);
    return true;
}

static bool write_all(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    size_t sent = 0;
    while (sent < n) {
        const int w = g_client.write(b + sent, n - sent);
        if (w <= 0) {
            return false;
        }
        sent += static_cast<size_t>(w);
    }
    return true;
}

static void send_frame(Frame* f) {
    char* line = f->data;
    size_t len = f->len;

    if (!at_frame::trim_line(&line, &len)) {
        return;  // banner text, AT echo, or a truncated line
    }

    size_t b64_start = 0;
    size_t b64_end = 0;
    size_t jpeg_len = 0;
    const bool has_image = at_frame::find_image_value(line, len, &b64_start, &b64_end);

    if (has_image && b64_end > b64_start) {
        const size_t decoded =
            at_frame::b64_decode_inplace(line + b64_start, b64_end - b64_start);
        if (decoded == SIZE_MAX) {
            return;  // corrupt payload, drop the frame rather than ship garbage
        }
        jpeg_len = decoded;
    }

    // JSON is sent as prefix + suffix around the (now emptied) image value, so
    // no copy is needed: the wire sees ..."image":"","boxes":[...]...
    const size_t json_head = has_image ? b64_start : len;
    const size_t json_tail = has_image ? (len - b64_end) : 0;
    const uint32_t json_len = static_cast<uint32_t>(json_head + json_tail);

    uint8_t hdr[12];
    memcpy(hdr, "SSC1", 4);
    memcpy(hdr + 4, &json_len, 4);
    const uint32_t jpeg_len_le = static_cast<uint32_t>(jpeg_len);
    memcpy(hdr + 8, &jpeg_len_le, 4);

    const bool ok = write_all(hdr, sizeof(hdr)) &&
                    write_all(line, json_head) &&
                    (json_tail == 0 || write_all(line + b64_end, json_tail)) &&
                    (jpeg_len == 0 || write_all(line + b64_start, jpeg_len));
    if (!ok) {
        Serial.println("[net] write failed, dropping connection");
        g_client.stop();
        return;
    }
    ++g_frames_tx;
}

static void sender_task(void*) {
    for (;;) {
        Frame* f = nullptr;
        if (xQueueReceive(g_ready_q, &f, pdMS_TO_TICKS(200)) != pdTRUE) {
            ensure_connected();
            continue;
        }
        if (ensure_connected()) {
            send_frame(f);
        } else {
            ++g_frames_dropped;
        }
        xQueueSend(g_free_q, &f, 0);
    }
}

// ---------------------------------------------------------------------------
// AT control
// ---------------------------------------------------------------------------

static void grove_at(const char* cmd) {
    Serial.printf("[at ] > %s\n", cmd);
    GroveSerial.print(cmd);
    GroveSerial.print("\r\n");
    GroveSerial.flush();
}

static void start_invoke() {
    grove_at("AT+BREAK");
    delay(200);
#if SENSOR_PRESET_INDEX >= 0
    char buf[40];
    snprintf(buf, sizeof(buf), "AT+SENSOR=1,1,%d", SENSOR_PRESET_INDEX);
    grove_at(buf);
    delay(800);
#endif
    grove_at("AT+INVOKE=-1," MA_STR(INVOKE_RESULTS_ONLY));
    g_last_frame_ms = millis();
}

// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("\n[boot] grove -> wifi bridge");

    if (!psramFound()) {
        Serial.println("[boot] FATAL: no PSRAM. Enable OPI PSRAM in the board options.");
        for (;;) delay(1000);
    }

    for (int i = 0; i < 2; ++i) {
        g_slots[i].data = static_cast<char*>(ps_malloc(FRAME_BUF_SIZE));
        g_slots[i].len = 0;
    }
    g_carry = static_cast<char*>(ps_malloc(FRAME_BUF_SIZE));
    if (!g_slots[0].data || !g_slots[1].data || !g_carry) {
        Serial.println("[boot] FATAL: PSRAM allocation failed");
        for (;;) delay(1000);
    }

    g_free_q = xQueueCreate(2, sizeof(Frame*));
    g_ready_q = xQueueCreate(1, sizeof(Frame*));
    for (int i = 0; i < 2; ++i) {
        Frame* f = &g_slots[i];
        xQueueSend(g_free_q, &f, 0);
    }

    GroveSerial.setRxBufferSize(UART_RX_RING_SIZE);
    GroveSerial.begin(kGroveBaud, SERIAL_8N1, kGroveRxPin, kGroveTxPin);

    WiFi.mode(WIFI_STA);
    WiFi.setHostname(WIFI_HOSTNAME);
    WiFi.setSleep(false);  // sleep adds latency the video path cannot absorb
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[wifi] connecting");
    for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; ++i) {
        delay(500);
        Serial.print('.');
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[wifi] %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("[wifi] not connected; will keep retrying in the background");
    }

    // Reader pinned to core 1 at high priority: it must keep the 32 kB UART ring
    // drained at 92 kB/s or frames arrive truncated. Sender goes on core 0 with
    // the WiFi stack, where blocking socket writes belong.
    xTaskCreatePinnedToCore(reader_task, "grove_rx", 4096, nullptr, 10, nullptr, 1);
    xTaskCreatePinnedToCore(sender_task, "wifi_tx", 8192, nullptr, 5, nullptr, 0);

    start_invoke();
}

void loop() {
    static uint32_t last_report = 0;
    static uint32_t last_rx = 0;
    static uint32_t last_bytes = 0;
    static uint32_t last_connected_ms = 0;

    delay(1000);

    const uint32_t now = millis();

    // The core already auto-reconnects on WIFI_STA_DISCONNECTED. Forcing
    // WiFi.reconnect() here on every non-WL_CONNECTED sample fights that: if
    // status hasn't settled to WL_CONNECTED yet (association/DHCP still in
    // flight), this would tear the in-progress connection down and restart it
    // every second, forever. Only step in after a real, sustained outage.
    if (WiFi.status() == WL_CONNECTED) {
        last_connected_ms = now;
    } else if (last_connected_ms == 0) {
        last_connected_ms = now;  // don't fire before the first connect attempt settles
    } else if (now - last_connected_ms > 8000) {
        Serial.println("[wifi] no connection for 8s, forcing reconnect");
        WiFi.reconnect();
        last_connected_ms = now;
    }

    if (now - last_report >= 5000) {
        const uint32_t rx = g_frames_rx;
        const uint32_t bytes = g_bytes_rx;
        const float dt = (now - last_report) / 1000.0f;
        Serial.printf("[stat] %.1f fps in, %.1f kB/s, tx=%lu drop=%lu oversize=%lu, "
                      "heap=%u psram=%u\n",
                      (rx - last_rx) / dt, (bytes - last_bytes) / dt / 1024.0f,
                      static_cast<unsigned long>(g_frames_tx),
                      static_cast<unsigned long>(g_frames_dropped),
                      static_cast<unsigned long>(g_frames_oversize),
                      static_cast<unsigned>(ESP.getFreeHeap()),
                      static_cast<unsigned>(ESP.getFreePsram()));
        last_report = now;
        last_rx = rx;
        last_bytes = bytes;
    }

    // The Grove board reboots on its own watchdog paths; re-arm the stream if it
    // goes quiet.
    if (now - g_last_frame_ms > INVOKE_WATCHDOG_MS) {
        Serial.println("[at ] no frames, re-issuing AT+INVOKE");
        start_invoke();
    }
}
