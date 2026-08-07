// Parsing for one AT event line from the Grove Vision AI V2.
//
// Deliberately free of Arduino/ESP32 dependencies so it can be compiled and
// tested on the host: see ../tools/test_at_frame.cpp.
//
// The node never parses the detection JSON. It only splits the line into
// (metadata JSON, decoded JPEG) and forwards both; the Raspberry Pi parses. The
// image value is base64, which contains no quotes and no escapes, so locating
// it is an unambiguous byte scan and needs no JSON state machine.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace at_frame {

inline int8_t b64_val(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return static_cast<int8_t>(c - 'A');
    if (c >= 'a' && c <= 'z') return static_cast<int8_t>(c - 'a' + 26);
    if (c >= '0' && c <= '9') return static_cast<int8_t>(c - '0' + 52);
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Decodes in place. Safe because the output advances 3 bytes per 4 input bytes,
// so the write cursor never overtakes the read cursor. Returns the decoded
// length, or SIZE_MAX if a character is neither base64 nor benign whitespace.
inline size_t b64_decode_inplace(char* buf, size_t len) {
    size_t w = 0;
    uint32_t acc = 0;
    int nbits = 0;
    for (size_t r = 0; r < len; ++r) {
        const uint8_t c = static_cast<uint8_t>(buf[r]);
        if (c == '=') break;
        const int8_t v = b64_val(c);
        if (v < 0) {
            if (c == '\r' || c == '\n' || c == ' ') continue;
            return SIZE_MAX;
        }
        acc = (acc << 6) | static_cast<uint32_t>(v);
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            buf[w++] = static_cast<char>((acc >> nbits) & 0xFF);
        }
    }
    return w;
}

// Locates the base64 image value inside an AT event line. On success sets
// [*start, *end) to the value's byte range within `line`. `memmem` is a GNU
// extension that newlib-nano does not reliably provide, hence the manual scan.
inline bool find_image_value(const char* line, size_t len, size_t* start, size_t* end) {
    static const char kKey[] = "\"image\":\"";
    const size_t kKeyLen = sizeof(kKey) - 1;

    if (len < kKeyLen) {
        return false;
    }
    const char* k = nullptr;
    for (const char* p = line;
         (p = static_cast<const char*>(
              memchr(p, '"', static_cast<size_t>(line + len - p)))) != nullptr;
         ++p) {
        if (static_cast<size_t>(line + len - p) < kKeyLen) {
            break;
        }
        if (memcmp(p, kKey, kKeyLen) == 0) {
            k = p;
            break;
        }
    }
    if (k == nullptr) {
        return false;
    }
    const size_t s = static_cast<size_t>(k - line) + kKeyLen;
    const char* q = static_cast<const char*>(memchr(line + s, '"', len - s));
    if (q == nullptr) {
        return false;
    }
    *start = s;
    *end = static_cast<size_t>(q - line);
    return true;
}

// Strips the firmware's '\r' prefix and any trailing '\r'/NUL padding, and
// rejects anything that is not a JSON object (AT echoes, boot banners,
// truncated lines). Returns false if the line should be dropped.
inline bool trim_line(char** line, size_t* len) {
    char* p = *line;
    size_t n = *len;
    while (n && (*p == '\r' || *p == '\n' || *p == '\0')) {
        ++p;
        --n;
    }
    while (n && (p[n - 1] == '\r' || p[n - 1] == '\n' || p[n - 1] == '\0')) {
        --n;
    }
    *line = p;
    *len = n;
    return n >= 2 && p[0] == '{' && p[n - 1] == '}';
}

}  // namespace at_frame
