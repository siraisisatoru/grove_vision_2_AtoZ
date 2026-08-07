// Host tests for the ESP32 node's AT line parsing.
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/test_at_frame test_at_frame.cpp && /tmp/test_at_frame
//
// The node cannot be exercised without hardware, but its parsing is pure byte
// manipulation and is where a protocol misreading would hide, so it is tested
// here against lines shaped exactly like the ones ma_codec_json.cpp emits.

#include "../xiao_esp32s3/at_frame.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

static std::string b64_encode(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < in.size(); i += 3) {
        const unsigned n = static_cast<unsigned char>(in[i]) << 16 |
                           (i + 1 < in.size() ? static_cast<unsigned char>(in[i + 1]) << 8 : 0) |
                           (i + 2 < in.size() ? static_cast<unsigned char>(in[i + 2]) : 0);
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += (i + 1 < in.size()) ? T[(n >> 6) & 63] : '=';
        out += (i + 2 < in.size()) ? T[n & 63] : '=';
    }
    return out;
}

// Mirrors what send_frame() puts on the wire, so the test covers the actual
// split rather than the helpers in isolation.
struct Split {
    bool ok = false;
    std::string json;
    std::string jpeg;
};

static Split split_like_send_frame(std::string raw) {
    Split out;
    std::vector<char> buf(raw.begin(), raw.end());
    buf.push_back('\0');

    char* line = buf.data();
    size_t len = raw.size();
    if (!at_frame::trim_line(&line, &len)) {
        return out;
    }

    size_t s = 0, e = 0;
    size_t jpeg_len = 0;
    const bool has_image = at_frame::find_image_value(line, len, &s, &e);
    if (has_image && e > s) {
        const size_t decoded = at_frame::b64_decode_inplace(line + s, e - s);
        if (decoded == SIZE_MAX) {
            return out;
        }
        jpeg_len = decoded;
    }

    const size_t head = has_image ? s : len;
    const size_t tail = has_image ? (len - e) : 0;

    out.ok = true;
    out.json.assign(line, head);
    if (tail) out.json.append(line + e, tail);
    out.jpeg.assign(line + s, jpeg_len);
    return out;
}

static void test_event_with_image() {
    std::printf("event line with a base64 image\n");
    // Not a real JPEG, but the parser is format-agnostic; what matters is that
    // the bytes survive the round trip exactly, including embedded NULs and
    // quote characters that would break a naive string-based split.
    std::string jpeg;
    for (int i = 0; i < 512; ++i) jpeg += static_cast<char>(i * 7 % 256);
    jpeg[3] = '\0';
    jpeg[4] = '"';

    const std::string raw =
        "\r{\"type\":1,\"name\":\"INVOKE\",\"code\":0,\"data\":{\"count\":42,\"image\":\"" +
        b64_encode(jpeg) +
        "\",\"boxes\":[[120,96,64,48,87,0]],\"perf\":[3,41,2],\"rotation\":0,"
        "\"width\":640,\"height\":480}}";

    const Split s = split_like_send_frame(raw);
    CHECK(s.ok);
    CHECK(s.jpeg == jpeg);
    CHECK(s.json ==
          "{\"type\":1,\"name\":\"INVOKE\",\"code\":0,\"data\":{\"count\":42,\"image\":\"\","
          "\"boxes\":[[120,96,64,48,87,0]],\"perf\":[3,41,2],\"rotation\":0,"
          "\"width\":640,\"height\":480}}");
}

static void test_results_only() {
    std::printf("results-only event line (no image key)\n");
    const std::string raw =
        "\r{\"type\":1,\"name\":\"INVOKE\",\"code\":0,\"data\":{\"count\":7,"
        "\"boxes\":[[10,20,30,40,55,2]],\"perf\":[2,38,1],\"width\":192,\"height\":192}}";
    const Split s = split_like_send_frame(raw);
    CHECK(s.ok);
    CHECK(s.jpeg.empty());
    CHECK(s.json == raw.substr(1));
}

static void test_empty_image_value() {
    std::printf("image key present but empty\n");
    const std::string raw = "\r{\"data\":{\"image\":\"\",\"boxes\":[]}}";
    const Split s = split_like_send_frame(raw);
    CHECK(s.ok);
    CHECK(s.jpeg.empty());
    CHECK(s.json == "{\"data\":{\"image\":\"\",\"boxes\":[]}}");
}

static void test_key_appearing_in_a_string_value_first() {
    std::printf("a decoy \\\"image\\\" substring earlier in the line\n");
    // The scan must anchor on the full key including the colon and opening
    // quote, not on the bare word.
    const std::string raw =
        "\r{\"name\":\"image thing\",\"data\":{\"image\":\"" + b64_encode("abc") + "\"}}";
    const Split s = split_like_send_frame(raw);
    CHECK(s.ok);
    CHECK(s.jpeg == "abc");
    CHECK(s.json == "{\"name\":\"image thing\",\"data\":{\"image\":\"\"}}");
}

static void test_rejects_non_json() {
    std::printf("non-JSON lines are rejected\n");
    CHECK(!split_like_send_frame("AT+INVOKE=-1,0").ok);
    CHECK(!split_like_send_frame("\r\n").ok);
    CHECK(!split_like_send_frame("").ok);
    // Truncated line: the reader can emit one if the UART ring overflowed.
    CHECK(!split_like_send_frame("\r{\"data\":{\"image\":\"AAAA").ok);
}

static void test_rejects_corrupt_base64() {
    std::printf("corrupt base64 is rejected rather than shipped\n");
    const std::string raw = "\r{\"data\":{\"image\":\"AA*A\"}}";
    CHECK(!split_like_send_frame(raw).ok);
}

static void test_decode_lengths() {
    std::printf("base64 length arithmetic across all padding cases\n");
    for (size_t n = 0; n < 200; ++n) {
        std::string payload(n, '\0');
        for (size_t i = 0; i < n; ++i) payload[i] = static_cast<char>((i * 31 + 7) & 0xFF);
        std::string enc = b64_encode(payload);
        std::vector<char> buf(enc.begin(), enc.end());
        const size_t got = at_frame::b64_decode_inplace(buf.data(), buf.size());
        CHECK(got == n);
        CHECK(std::memcmp(buf.data(), payload.data(), n) == 0);
    }
}

int main() {
    test_event_with_image();
    test_results_only();
    test_empty_image_value();
    test_key_appearing_in_a_string_value_first();
    test_rejects_non_json();
    test_rejects_corrupt_base64();
    test_decode_lengths();

    if (g_failures) {
        std::printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
