#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROR((x), 2) ^ ROR((x), 13) ^ ROR((x), 22))
#define SIGMA1(x) (ROR((x), 6) ^ ROR((x), 11) ^ ROR((x), 25))
#define GAMMA0(x) (ROR((x), 7) ^ ROR((x), 18) ^ ((x) >> 3))
#define GAMMA1(x) (ROR((x), 17) ^ ROR((x), 19) ^ ((x) >> 10))

__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint8_t* g_charset = nullptr;
static uint64_t g_charset_len = 0;
static uint8_t* g_charset_host = nullptr;
static uint64_t g_charset_host_cap = 0;
static uint32_t* g_target_hash = nullptr;
static uint32_t g_target_hash_host[8];
static int g_target_hash_valid = 0;
static uint32_t* g_result_found = nullptr;
static uint8_t* g_result_word = nullptr;
static uint32_t* g_result_found_host = nullptr;
static uint8_t* g_result_word_host = nullptr;
static hipStream_t g_stream = nullptr;
static int g_device_initialized = 0;

struct HipDeviceInfo {
    int available;
    int device_count;
    int device_id;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    uint64_t total_global_mem;
    int clock_rate_khz;
    uint8_t name[256];
};

static int hip_fail(hipError_t err, const char* stage) {
    if (err == hipSuccess) {
        return 0;
    }
    printf("[HIP ERROR] %s: %s\n", stage, hipGetErrorString(err));
    return static_cast<int>(err);
}

static int ensure_device_initialized() {
    if (g_device_initialized != 0) {
        return 0;
    }

    hipError_t err = hipSetDevice(0);
    int rc = hip_fail(err, "hipSetDevice");
    if (rc != 0) {
        return rc;
    }

    err = hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking);
    rc = hip_fail(err, "hipStreamCreateWithFlags");
    if (rc != 0) {
        return rc;
    }

    err = hipHostMalloc(reinterpret_cast<void**>(&g_result_found_host), sizeof(uint32_t), hipHostMallocDefault);
    rc = hip_fail(err, "hipHostMalloc(result_found_host)");
    if (rc != 0) {
        return rc;
    }

    err = hipHostMalloc(reinterpret_cast<void**>(&g_result_word_host), 64, hipHostMallocDefault);
    rc = hip_fail(err, "hipHostMalloc(result_word_host)");
    if (rc != 0) {
        return rc;
    }

    g_device_initialized = 1;
    return 0;
}

__device__ __forceinline__ uint32_t pack_word(const uint8_t* pass, uint32_t len, int word_index) {
    int idx = word_index * 4;
    uint32_t word = 0;

    if (idx < static_cast<int>(len)) word |= static_cast<uint32_t>(pass[idx]) << 24;
    else if (idx == static_cast<int>(len)) word |= 0x80000000u;

    if (idx + 1 < static_cast<int>(len)) word |= static_cast<uint32_t>(pass[idx + 1]) << 16;
    else if (idx + 1 == static_cast<int>(len)) word |= 0x00800000u;

    if (idx + 2 < static_cast<int>(len)) word |= static_cast<uint32_t>(pass[idx + 2]) << 8;
    else if (idx + 2 == static_cast<int>(len)) word |= 0x00008000u;

    if (idx + 3 < static_cast<int>(len)) word |= static_cast<uint32_t>(pass[idx + 3]);
    else if (idx + 3 == static_cast<int>(len)) word |= 0x00000080u;

    return word;
}

__device__ __forceinline__ bool sha256_compress_matches(
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    uint32_t w3,
    uint32_t w4,
    uint32_t w5,
    uint32_t w6,
    uint32_t w7,
    uint32_t w8,
    uint32_t w9,
    uint32_t wa,
    uint32_t wb,
    uint32_t wc,
    uint32_t wd,
    uint32_t we,
    uint32_t wf,
    const uint32_t* target_hash
) {
    uint32_t a = 0x6a09e667;
    uint32_t b = 0xbb67ae85;
    uint32_t c = 0x3c6ef372;
    uint32_t d = 0xa54ff53a;
    uint32_t e = 0x510e527f;
    uint32_t f = 0x9b05688c;
    uint32_t g = 0x1f83d9ab;
    uint32_t h = 0x5be0cd19;

#define SHA256_STEP(a, b, c, d, e, f, g, h, wv, kv) \
    do { \
        uint32_t t1 = (h) + SIGMA1(e) + CH(e, f, g) + (kv) + (wv); \
        uint32_t t2 = SIGMA0(a) + MAJ(a, b, c); \
        (d) += t1; \
        (h) = t1 + t2; \
    } while (0)

    SHA256_STEP(a, b, c, d, e, f, g, h, w0, K[0x00]);
    SHA256_STEP(h, a, b, c, d, e, f, g, w1, K[0x01]);
    SHA256_STEP(g, h, a, b, c, d, e, f, w2, K[0x02]);
    SHA256_STEP(f, g, h, a, b, c, d, e, w3, K[0x03]);
    SHA256_STEP(e, f, g, h, a, b, c, d, w4, K[0x04]);
    SHA256_STEP(d, e, f, g, h, a, b, c, w5, K[0x05]);
    SHA256_STEP(c, d, e, f, g, h, a, b, w6, K[0x06]);
    SHA256_STEP(b, c, d, e, f, g, h, a, w7, K[0x07]);
    SHA256_STEP(a, b, c, d, e, f, g, h, w8, K[0x08]);
    SHA256_STEP(h, a, b, c, d, e, f, g, w9, K[0x09]);
    SHA256_STEP(g, h, a, b, c, d, e, f, wa, K[0x0a]);
    SHA256_STEP(f, g, h, a, b, c, d, e, wb, K[0x0b]);
    SHA256_STEP(e, f, g, h, a, b, c, d, wc, K[0x0c]);
    SHA256_STEP(d, e, f, g, h, a, b, c, wd, K[0x0d]);
    SHA256_STEP(c, d, e, f, g, h, a, b, we, K[0x0e]);
    SHA256_STEP(b, c, d, e, f, g, h, a, wf, K[0x0f]);

    w0 = GAMMA1(we) + w9 + GAMMA0(w1) + w0; SHA256_STEP(a, b, c, d, e, f, g, h, w0, K[0x10]);
    w1 = GAMMA1(wf) + wa + GAMMA0(w2) + w1; SHA256_STEP(h, a, b, c, d, e, f, g, w1, K[0x11]);
    w2 = GAMMA1(w0) + wb + GAMMA0(w3) + w2; SHA256_STEP(g, h, a, b, c, d, e, f, w2, K[0x12]);
    w3 = GAMMA1(w1) + wc + GAMMA0(w4) + w3; SHA256_STEP(f, g, h, a, b, c, d, e, w3, K[0x13]);
    w4 = GAMMA1(w2) + wd + GAMMA0(w5) + w4; SHA256_STEP(e, f, g, h, a, b, c, d, w4, K[0x14]);
    w5 = GAMMA1(w3) + we + GAMMA0(w6) + w5; SHA256_STEP(d, e, f, g, h, a, b, c, w5, K[0x15]);
    w6 = GAMMA1(w4) + wf + GAMMA0(w7) + w6; SHA256_STEP(c, d, e, f, g, h, a, b, w6, K[0x16]);
    w7 = GAMMA1(w5) + w0 + GAMMA0(w8) + w7; SHA256_STEP(b, c, d, e, f, g, h, a, w7, K[0x17]);
    w8 = GAMMA1(w6) + w1 + GAMMA0(w9) + w8; SHA256_STEP(a, b, c, d, e, f, g, h, w8, K[0x18]);
    w9 = GAMMA1(w7) + w2 + GAMMA0(wa) + w9; SHA256_STEP(h, a, b, c, d, e, f, g, w9, K[0x19]);
    wa = GAMMA1(w8) + w3 + GAMMA0(wb) + wa; SHA256_STEP(g, h, a, b, c, d, e, f, wa, K[0x1a]);
    wb = GAMMA1(w9) + w4 + GAMMA0(wc) + wb; SHA256_STEP(f, g, h, a, b, c, d, e, wb, K[0x1b]);
    wc = GAMMA1(wa) + w5 + GAMMA0(wd) + wc; SHA256_STEP(e, f, g, h, a, b, c, d, wc, K[0x1c]);
    wd = GAMMA1(wb) + w6 + GAMMA0(we) + wd; SHA256_STEP(d, e, f, g, h, a, b, c, wd, K[0x1d]);
    we = GAMMA1(wc) + w7 + GAMMA0(wf) + we; SHA256_STEP(c, d, e, f, g, h, a, b, we, K[0x1e]);
    wf = GAMMA1(wd) + w8 + GAMMA0(w0) + wf; SHA256_STEP(b, c, d, e, f, g, h, a, wf, K[0x1f]);

    w0 = GAMMA1(we) + w9 + GAMMA0(w1) + w0; SHA256_STEP(a, b, c, d, e, f, g, h, w0, K[0x20]);
    w1 = GAMMA1(wf) + wa + GAMMA0(w2) + w1; SHA256_STEP(h, a, b, c, d, e, f, g, w1, K[0x21]);
    w2 = GAMMA1(w0) + wb + GAMMA0(w3) + w2; SHA256_STEP(g, h, a, b, c, d, e, f, w2, K[0x22]);
    w3 = GAMMA1(w1) + wc + GAMMA0(w4) + w3; SHA256_STEP(f, g, h, a, b, c, d, e, w3, K[0x23]);
    w4 = GAMMA1(w2) + wd + GAMMA0(w5) + w4; SHA256_STEP(e, f, g, h, a, b, c, d, w4, K[0x24]);
    w5 = GAMMA1(w3) + we + GAMMA0(w6) + w5; SHA256_STEP(d, e, f, g, h, a, b, c, w5, K[0x25]);
    w6 = GAMMA1(w4) + wf + GAMMA0(w7) + w6; SHA256_STEP(c, d, e, f, g, h, a, b, w6, K[0x26]);
    w7 = GAMMA1(w5) + w0 + GAMMA0(w8) + w7; SHA256_STEP(b, c, d, e, f, g, h, a, w7, K[0x27]);
    w8 = GAMMA1(w6) + w1 + GAMMA0(w9) + w8; SHA256_STEP(a, b, c, d, e, f, g, h, w8, K[0x28]);
    w9 = GAMMA1(w7) + w2 + GAMMA0(wa) + w9; SHA256_STEP(h, a, b, c, d, e, f, g, w9, K[0x29]);
    wa = GAMMA1(w8) + w3 + GAMMA0(wb) + wa; SHA256_STEP(g, h, a, b, c, d, e, f, wa, K[0x2a]);
    wb = GAMMA1(w9) + w4 + GAMMA0(wc) + wb; SHA256_STEP(f, g, h, a, b, c, d, e, wb, K[0x2b]);
    wc = GAMMA1(wa) + w5 + GAMMA0(wd) + wc; SHA256_STEP(e, f, g, h, a, b, c, d, wc, K[0x2c]);
    wd = GAMMA1(wb) + w6 + GAMMA0(we) + wd; SHA256_STEP(d, e, f, g, h, a, b, c, wd, K[0x2d]);
    we = GAMMA1(wc) + w7 + GAMMA0(wf) + we; SHA256_STEP(c, d, e, f, g, h, a, b, we, K[0x2e]);
    wf = GAMMA1(wd) + w8 + GAMMA0(w0) + wf; SHA256_STEP(b, c, d, e, f, g, h, a, wf, K[0x2f]);

    w0 = GAMMA1(we) + w9 + GAMMA0(w1) + w0; SHA256_STEP(a, b, c, d, e, f, g, h, w0, K[0x30]);
    w1 = GAMMA1(wf) + wa + GAMMA0(w2) + w1; SHA256_STEP(h, a, b, c, d, e, f, g, w1, K[0x31]);
    w2 = GAMMA1(w0) + wb + GAMMA0(w3) + w2; SHA256_STEP(g, h, a, b, c, d, e, f, w2, K[0x32]);
    w3 = GAMMA1(w1) + wc + GAMMA0(w4) + w3; SHA256_STEP(f, g, h, a, b, c, d, e, w3, K[0x33]);
    w4 = GAMMA1(w2) + wd + GAMMA0(w5) + w4; SHA256_STEP(e, f, g, h, a, b, c, d, w4, K[0x34]);
    w5 = GAMMA1(w3) + we + GAMMA0(w6) + w5; SHA256_STEP(d, e, f, g, h, a, b, c, w5, K[0x35]);
    w6 = GAMMA1(w4) + wf + GAMMA0(w7) + w6; SHA256_STEP(c, d, e, f, g, h, a, b, w6, K[0x36]);
    w7 = GAMMA1(w5) + w0 + GAMMA0(w8) + w7; SHA256_STEP(b, c, d, e, f, g, h, a, w7, K[0x37]);
    w8 = GAMMA1(w6) + w1 + GAMMA0(w9) + w8; SHA256_STEP(a, b, c, d, e, f, g, h, w8, K[0x38]);
    w9 = GAMMA1(w7) + w2 + GAMMA0(wa) + w9; SHA256_STEP(h, a, b, c, d, e, f, g, w9, K[0x39]);
    wa = GAMMA1(w8) + w3 + GAMMA0(wb) + wa; SHA256_STEP(g, h, a, b, c, d, e, f, wa, K[0x3a]);
    wb = GAMMA1(w9) + w4 + GAMMA0(wc) + wb; SHA256_STEP(f, g, h, a, b, c, d, e, wb, K[0x3b]);
    wc = GAMMA1(wa) + w5 + GAMMA0(wd) + wc; SHA256_STEP(e, f, g, h, a, b, c, d, wc, K[0x3c]);
    wd = GAMMA1(wb) + w6 + GAMMA0(we) + wd; SHA256_STEP(d, e, f, g, h, a, b, c, wd, K[0x3d]);
    we = GAMMA1(wc) + w7 + GAMMA0(wf) + we; SHA256_STEP(c, d, e, f, g, h, a, b, we, K[0x3e]);
    wf = GAMMA1(wd) + w8 + GAMMA0(w0) + wf; SHA256_STEP(b, c, d, e, f, g, h, a, wf, K[0x3f]);

#undef SHA256_STEP

    return (a + 0x6a09e667) == target_hash[0]
        && (b + 0xbb67ae85) == target_hash[1]
        && (c + 0x3c6ef372) == target_hash[2]
        && (d + 0xa54ff53a) == target_hash[3]
        && (e + 0x510e527f) == target_hash[4]
        && (f + 0x9b05688c) == target_hash[5]
        && (g + 0x1f83d9ab) == target_hash[6]
        && (h + 0x5be0cd19) == target_hash[7];
}

__device__ __forceinline__ bool sha256_candidate_matches(
    const uint8_t* pass,
    uint32_t len,
    const uint32_t* target_hash
) {
    uint32_t w0 = pack_word(pass, len, 0);
    uint32_t w1 = pack_word(pass, len, 1);
    uint32_t w2 = pack_word(pass, len, 2);
    uint32_t w3 = pack_word(pass, len, 3);
    uint32_t w4 = pack_word(pass, len, 4);
    uint32_t w5 = pack_word(pass, len, 5);
    uint32_t w6 = pack_word(pass, len, 6);
    uint32_t w7 = pack_word(pass, len, 7);
    uint32_t w8 = pack_word(pass, len, 8);
    uint32_t w9 = pack_word(pass, len, 9);
    uint32_t wa = pack_word(pass, len, 10);
    uint32_t wb = pack_word(pass, len, 11);
    uint32_t wc = pack_word(pass, len, 12);
    uint32_t wd = pack_word(pass, len, 13);
    return sha256_compress_matches(
        w0, w1, w2, w3, w4, w5, w6, w7,
        w8, w9, wa, wb, wc, wd, 0, len * 8,
        target_hash
    );
}

template <uint32_t LEN>
__device__ __forceinline__ bool sha256_candidate_matches_fixed(
    const uint8_t* pass,
    const uint32_t* target_hash
);

template <>
__device__ __forceinline__ bool sha256_candidate_matches_fixed<6>(
    const uint8_t* pass,
    const uint32_t* target_hash
) {
    uint32_t w0 = (static_cast<uint32_t>(pass[0]) << 24) |
                  (static_cast<uint32_t>(pass[1]) << 16) |
                  (static_cast<uint32_t>(pass[2]) << 8) |
                  static_cast<uint32_t>(pass[3]);
    uint32_t w1 = (static_cast<uint32_t>(pass[4]) << 24) |
                  (static_cast<uint32_t>(pass[5]) << 16) |
                  0x00008000u;
    return sha256_compress_matches(
        w0, w1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 48,
        target_hash
    );
}

template <>
__device__ __forceinline__ bool sha256_candidate_matches_fixed<7>(
    const uint8_t* pass,
    const uint32_t* target_hash
) {
    uint32_t w0 = (static_cast<uint32_t>(pass[0]) << 24) |
                  (static_cast<uint32_t>(pass[1]) << 16) |
                  (static_cast<uint32_t>(pass[2]) << 8) |
                  static_cast<uint32_t>(pass[3]);
    uint32_t w1 = (static_cast<uint32_t>(pass[4]) << 24) |
                  (static_cast<uint32_t>(pass[5]) << 16) |
                  (static_cast<uint32_t>(pass[6]) << 8) |
                  0x00000080u;
    return sha256_compress_matches(
        w0, w1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 56,
        target_hash
    );
}

template <>
__device__ __forceinline__ bool sha256_candidate_matches_fixed<8>(
    const uint8_t* pass,
    const uint32_t* target_hash
) {
    uint32_t w0 = (static_cast<uint32_t>(pass[0]) << 24) |
                  (static_cast<uint32_t>(pass[1]) << 16) |
                  (static_cast<uint32_t>(pass[2]) << 8) |
                  static_cast<uint32_t>(pass[3]);
    uint32_t w1 = (static_cast<uint32_t>(pass[4]) << 24) |
                  (static_cast<uint32_t>(pass[5]) << 16) |
                  (static_cast<uint32_t>(pass[6]) << 8) |
                  static_cast<uint32_t>(pass[7]);
    return sha256_compress_matches(
        w0, w1, 0x80000000u, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 64,
        target_hash
    );
}

template <>
__device__ __forceinline__ bool sha256_candidate_matches_fixed<9>(
    const uint8_t* pass,
    const uint32_t* target_hash
) {
    uint32_t w0 = (static_cast<uint32_t>(pass[0]) << 24) |
                  (static_cast<uint32_t>(pass[1]) << 16) |
                  (static_cast<uint32_t>(pass[2]) << 8) |
                  static_cast<uint32_t>(pass[3]);
    uint32_t w1 = (static_cast<uint32_t>(pass[4]) << 24) |
                  (static_cast<uint32_t>(pass[5]) << 16) |
                  (static_cast<uint32_t>(pass[6]) << 8) |
                  static_cast<uint32_t>(pass[7]);
    uint32_t w2 = (static_cast<uint32_t>(pass[8]) << 24) |
                  0x00800000u;
    return sha256_compress_matches(
        w0, w1, w2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 72,
        target_hash
    );
}

template <>
__device__ __forceinline__ bool sha256_candidate_matches_fixed<10>(
    const uint8_t* pass,
    const uint32_t* target_hash
) {
    uint32_t w0 = (static_cast<uint32_t>(pass[0]) << 24) |
                  (static_cast<uint32_t>(pass[1]) << 16) |
                  (static_cast<uint32_t>(pass[2]) << 8) |
                  static_cast<uint32_t>(pass[3]);
    uint32_t w1 = (static_cast<uint32_t>(pass[4]) << 24) |
                  (static_cast<uint32_t>(pass[5]) << 16) |
                  (static_cast<uint32_t>(pass[6]) << 8) |
                  static_cast<uint32_t>(pass[7]);
    uint32_t w2 = (static_cast<uint32_t>(pass[8]) << 24) |
                  (static_cast<uint32_t>(pass[9]) << 16) |
                  0x00008000u;
    return sha256_compress_matches(
        w0, w1, w2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 80,
        target_hash
    );
}

__device__ __forceinline__ void init_candidate_from_index(
    uint64_t absolute_index,
    uint32_t len,
    const uint8_t* charset,
    uint64_t charset_len,
    uint16_t* digits,
    uint8_t* pass
) {
    uint64_t temp = absolute_index;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        pass[i] = 0;
        digits[i] = 0;
    }

    for (int pos = static_cast<int>(len) - 1; pos >= 0; pos--) {
        const uint16_t digit = static_cast<uint16_t>(temp % charset_len);
        temp /= charset_len;
        digits[pos] = digit;
        pass[pos] = charset[digit];
    }
}

__device__ __forceinline__ bool increment_candidate(
    uint32_t len,
    const uint8_t* charset,
    uint64_t charset_len,
    uint16_t* digits,
    uint8_t* pass
) {
    for (int pos = static_cast<int>(len) - 1; pos >= 0; pos--) {
        const uint16_t next = static_cast<uint16_t>(digits[pos] + 1);

        if (next < charset_len) {
            digits[pos] = next;
            pass[pos] = charset[next];
            return true;
        }

        digits[pos] = 0;
        pass[pos] = charset[0];
    }

    return false;
}

template <uint32_t LEN>
__device__ __forceinline__ void init_candidate_from_index_fixed(
    uint64_t absolute_index,
    const uint8_t* charset,
    uint64_t charset_len,
    uint16_t* digits,
    uint8_t* pass
) {
    uint64_t temp = absolute_index;

    #pragma unroll
    for (uint32_t i = 0; i < LEN; i++) {
        pass[i] = 0;
        digits[i] = 0;
    }

    #pragma unroll
    for (int pos = static_cast<int>(LEN) - 1; pos >= 0; pos--) {
        const uint16_t digit = static_cast<uint16_t>(temp % charset_len);
        temp /= charset_len;
        digits[pos] = digit;
        pass[pos] = charset[digit];
    }
}

template <uint32_t LEN>
__device__ __forceinline__ bool increment_candidate_fixed(
    const uint8_t* charset,
    uint64_t charset_len,
    uint16_t* digits,
    uint8_t* pass
) {
    #pragma unroll
    for (int pos = static_cast<int>(LEN) - 1; pos >= 0; pos--) {
        const uint16_t next = static_cast<uint16_t>(digits[pos] + 1);

        if (next < charset_len) {
            digits[pos] = next;
            pass[pos] = charset[next];
            return true;
        }

        digits[pos] = 0;
        pass[pos] = charset[0];
    }

    return false;
}

__device__ __forceinline__ void advance_candidate_by_digits(
    uint32_t len,
    const uint8_t* charset,
    uint64_t charset_len,
    const uint16_t* stride_digits,
    uint16_t* digits,
    uint8_t* pass
) {
    uint32_t carry = 0;
    for (int pos = static_cast<int>(len) - 1; pos >= 0; pos--) {
        uint32_t sum = static_cast<uint32_t>(digits[pos]) + static_cast<uint32_t>(stride_digits[pos]) + carry;
        if (sum >= charset_len) {
            sum -= static_cast<uint32_t>(charset_len);
            carry = 1;
        } else {
            carry = 0;
        }
        const uint16_t next = static_cast<uint16_t>(sum);
        digits[pos] = next;
        pass[pos] = charset[next];
    }
}

template <uint32_t LEN>
__device__ __forceinline__ void advance_candidate_by_digits_fixed(
    const uint8_t* charset,
    uint64_t charset_len,
    const uint16_t* stride_digits,
    uint16_t* digits,
    uint8_t* pass
) {
    uint32_t carry = 0;
    #pragma unroll
    for (int pos = static_cast<int>(LEN) - 1; pos >= 0; pos--) {
        uint32_t sum = static_cast<uint32_t>(digits[pos]) + static_cast<uint32_t>(stride_digits[pos]) + carry;
        if (sum >= charset_len) {
            sum -= static_cast<uint32_t>(charset_len);
            carry = 1;
        } else {
            carry = 0;
        }
        const uint16_t next = static_cast<uint16_t>(sum);
        digits[pos] = next;
        pass[pos] = charset[next];
    }
}

template <uint32_t ITEMS_PER_THREAD>
__global__ __launch_bounds__(256, 2) void brute_force_hip_kernel(
    const uint8_t* __restrict__ charset,
    uint64_t charset_len,
    uint64_t start_offset,
    uint32_t len,
    uint64_t batch_size,
    const uint32_t* __restrict__ target_hash,
    uint32_t* __restrict__ result_found,
    uint8_t* __restrict__ result_word
) {
    uint64_t thread_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t thread_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x * ITEMS_PER_THREAD;
    uint64_t candidate_base = thread_index * ITEMS_PER_THREAD;
    if (candidate_base >= batch_size) {
        return;
    }

    uint8_t pass[16];
    uint16_t digits[16];
    uint16_t stride_digits[16];
    init_candidate_from_index(start_offset + candidate_base, len, charset, charset_len, digits, pass);
    uint8_t stride_pass_scratch[16];
    init_candidate_from_index(thread_stride, len, charset, charset_len, stride_digits, stride_pass_scratch);

    uint32_t loop_iter = 0;
    for (;;) {
        if ((loop_iter & 7u) == 0u && __builtin_amdgcn_readfirstlane(*result_found) != 0) {
            return;
        }

        #pragma unroll
        for (uint64_t item = 0; item < ITEMS_PER_THREAD; item++) {
            const uint64_t candidate_index = candidate_base + item;
            if (candidate_index >= batch_size) {
                break;
            }

            const bool match = sha256_candidate_matches(pass, len, target_hash);

            if (match && atomicExch(result_found, 1) == 0) {
                for (uint32_t i = 0; i < len; i++) {
                    result_word[i] = pass[i];
                }
                result_word[len] = 0;
                return;
            }

            if ((item + 1) < ITEMS_PER_THREAD && !increment_candidate(len, charset, charset_len, digits, pass)) {
                break;
            }
        }

        const uint64_t next_base = candidate_base + thread_stride;
        if (next_base >= batch_size) {
            break;
        }
        advance_candidate_by_digits(len, charset, charset_len, stride_digits, digits, pass);
        candidate_base = next_base;
        loop_iter++;
    }
}

template <uint32_t ITEMS_PER_THREAD, uint32_t LEN>
__global__ __launch_bounds__(256, 2) void brute_force_hip_kernel_fixed(
    const uint8_t* __restrict__ charset,
    uint64_t charset_len,
    uint64_t start_offset,
    uint64_t batch_size,
    const uint32_t* __restrict__ target_hash,
    uint32_t* __restrict__ result_found,
    uint8_t* __restrict__ result_word
) {
    uint64_t thread_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t thread_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x * ITEMS_PER_THREAD;
    uint64_t candidate_base = thread_index * ITEMS_PER_THREAD;
    if (candidate_base >= batch_size) {
        return;
    }

    uint8_t pass[LEN];
    uint16_t digits[LEN];
    uint16_t stride_digits[LEN];
    init_candidate_from_index_fixed<LEN>(start_offset + candidate_base, charset, charset_len, digits, pass);
    uint8_t stride_pass_scratch[LEN];
    init_candidate_from_index_fixed<LEN>(thread_stride, charset, charset_len, stride_digits, stride_pass_scratch);

    uint32_t loop_iter = 0;
    for (;;) {
        if ((loop_iter & 7u) == 0u && __builtin_amdgcn_readfirstlane(*result_found) != 0) {
            return;
        }

        #pragma unroll
        for (uint64_t item = 0; item < ITEMS_PER_THREAD; item++) {
            const uint64_t candidate_index = candidate_base + item;
            if (candidate_index >= batch_size) {
                break;
            }

            const bool match = sha256_candidate_matches_fixed<LEN>(pass, target_hash);

            if (match && atomicExch(result_found, 1) == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < LEN; i++) {
                    result_word[i] = pass[i];
                }
                result_word[LEN] = 0;
                return;
            }

            if ((item + 1) < ITEMS_PER_THREAD &&
                !increment_candidate_fixed<LEN>(charset, charset_len, digits, pass)) {
                break;
            }
        }

        const uint64_t next_base = candidate_base + thread_stride;
        if (next_base >= batch_size) {
            break;
        }
        advance_candidate_by_digits_fixed<LEN>(charset, charset_len, stride_digits, digits, pass);
        candidate_base = next_base;
        loop_iter++;
    }
}

#define LAUNCH_GENERIC_KERNEL(ITEMS) \
    hipLaunchKernelGGL( \
        HIP_KERNEL_NAME(brute_force_hip_kernel<ITEMS>), \
        dim3(blocks), \
        dim3(threads), \
        0, \
        g_stream, \
        g_charset, \
        charset_len, \
        start_offset, \
        len, \
        batch_size, \
        g_target_hash, \
        g_result_found, \
        g_result_word \
    )

#define LAUNCH_FIXED_KERNEL(ITEMS, FIXED_LEN) \
    hipLaunchKernelGGL( \
        HIP_KERNEL_NAME(brute_force_hip_kernel_fixed<ITEMS, FIXED_LEN>), \
        dim3(blocks), \
        dim3(threads), \
        0, \
        g_stream, \
        g_charset, \
        charset_len, \
        start_offset, \
        batch_size, \
        g_target_hash, \
        g_result_found, \
        g_result_word \
    )

template <uint32_t ITEMS>
static int launch_items_kernel(
    uint32_t len,
    int blocks,
    int threads,
    uint64_t charset_len,
    uint64_t start_offset,
    uint64_t batch_size
) {
    switch (len) {
        case 6:  LAUNCH_FIXED_KERNEL(ITEMS, 6);  break;
        case 7:  LAUNCH_FIXED_KERNEL(ITEMS, 7);  break;
        case 8:  LAUNCH_FIXED_KERNEL(ITEMS, 8);  break;
        case 9:  LAUNCH_FIXED_KERNEL(ITEMS, 9);  break;
        case 10: LAUNCH_FIXED_KERNEL(ITEMS, 10); break;
        default: LAUNCH_GENERIC_KERNEL(ITEMS);    break;
    }
    return 0;
}

static int launch_dispatched_kernel(
    uint32_t items_per_thread,
    uint32_t len,
    int blocks,
    int threads,
    uint64_t charset_len,
    uint64_t start_offset,
    uint64_t batch_size
) {
    switch (items_per_thread) {
        case 1:  return launch_items_kernel<1>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 2:  return launch_items_kernel<2>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 4:  return launch_items_kernel<4>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 8:  return launch_items_kernel<8>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 16: return launch_items_kernel<16>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 32: return launch_items_kernel<32>(len, blocks, threads, charset_len, start_offset, batch_size);
        case 64: return launch_items_kernel<64>(len, blocks, threads, charset_len, start_offset, batch_size);
        default: return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int launch_brute_force_hip(
    const uint8_t* charset,
    uint64_t charset_len,
    uint64_t start_offset,
    uint32_t len,
    const uint32_t* target_hash,
    uint32_t* result_found,
    uint8_t* result_word,
    uint64_t batch_size,
    int blocks,
    int threads,
    uint32_t items_per_thread
) {
    if (charset == nullptr || target_hash == nullptr || result_found == nullptr || result_word == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    int rc = ensure_device_initialized();
    if (rc != 0) {
        return rc;
    }

    if (g_charset == nullptr || g_charset_len != charset_len) {
        if (g_charset != nullptr) {
            hipError_t free_status = hipFree(g_charset);
            int free_rc = hip_fail(free_status, "hipFree(g_charset)");
            if (free_rc != 0) {
                return free_rc;
            }
        }
        hipError_t alloc_status = hipMalloc(reinterpret_cast<void**>(&g_charset), charset_len);
        int alloc_rc = hip_fail(alloc_status, "hipMalloc(g_charset)");
        if (alloc_rc != 0) {
            return alloc_rc;
        }
        g_charset_len = charset_len;

        if (g_charset_host_cap < charset_len) {
            uint8_t* new_host = reinterpret_cast<uint8_t*>(realloc(g_charset_host, charset_len));
            if (new_host == nullptr) {
                return static_cast<int>(hipErrorOutOfMemory);
            }
            g_charset_host = new_host;
            g_charset_host_cap = charset_len;
        }

        hipError_t copy_charset = hipMemcpyAsync(g_charset, charset, charset_len, hipMemcpyHostToDevice, g_stream);
        int copy_charset_rc = hip_fail(copy_charset, "hipMemcpy(charset)");
        if (copy_charset_rc != 0) {
            return copy_charset_rc;
        }
        memcpy(g_charset_host, charset, charset_len);
    } else if (memcmp(g_charset_host, charset, charset_len) != 0) {
        hipError_t copy_charset = hipMemcpyAsync(g_charset, charset, charset_len, hipMemcpyHostToDevice, g_stream);
        int copy_charset_rc = hip_fail(copy_charset, "hipMemcpy(charset)");
        if (copy_charset_rc != 0) {
            return copy_charset_rc;
        }
        memcpy(g_charset_host, charset, charset_len);
    }

    if (g_target_hash == nullptr) {
        hipError_t alloc_status = hipMalloc(reinterpret_cast<void**>(&g_target_hash), 8 * sizeof(uint32_t));
        int alloc_rc = hip_fail(alloc_status, "hipMalloc(g_target_hash)");
        if (alloc_rc != 0) {
            return alloc_rc;
        }
    }

    if (g_result_found == nullptr) {
        hipError_t alloc_status = hipMalloc(reinterpret_cast<void**>(&g_result_found), sizeof(uint32_t));
        int alloc_rc = hip_fail(alloc_status, "hipMalloc(g_result_found)");
        if (alloc_rc != 0) {
            return alloc_rc;
        }
    }

    if (g_result_word == nullptr) {
        hipError_t alloc_status = hipMalloc(reinterpret_cast<void**>(&g_result_word), 64);
        int alloc_rc = hip_fail(alloc_status, "hipMalloc(g_result_word)");
        if (alloc_rc != 0) {
            return alloc_rc;
        }
    }

    if (g_target_hash_valid == 0 || memcmp(g_target_hash_host, target_hash, 8 * sizeof(uint32_t)) != 0) {
        hipError_t copy_target = hipMemcpyAsync(g_target_hash, target_hash, 8 * sizeof(uint32_t), hipMemcpyHostToDevice, g_stream);
        int copy_target_rc = hip_fail(copy_target, "hipMemcpy(target_hash)");
        if (copy_target_rc != 0) {
            return copy_target_rc;
        }
        memcpy(g_target_hash_host, target_hash, 8 * sizeof(uint32_t));
        g_target_hash_valid = 1;
    }

    uint32_t zero = 0;
    hipError_t err = hipMemcpyAsync(g_result_found, &zero, sizeof(uint32_t), hipMemcpyHostToDevice, g_stream);
    rc = hip_fail(err, "hipMemcpy(reset result_found)");
    if (rc != 0) {
        return rc;
    }

    rc = launch_dispatched_kernel(
        items_per_thread,
        len,
        blocks,
        threads,
        charset_len,
        start_offset,
        batch_size
    );
    if (rc != 0) {
        return rc;
    }

    err = hipGetLastError();
    rc = hip_fail(err, "hipGetLastError");
    if (rc != 0) {
        return rc;
    }

    err = hipMemcpyAsync(g_result_found_host, g_result_found, sizeof(uint32_t), hipMemcpyDeviceToHost, g_stream);
    rc = hip_fail(err, "hipMemcpyAsync(result_found)");
    if (rc != 0) {
        return rc;
    }

    err = hipStreamSynchronize(g_stream);
    rc = hip_fail(err, "hipStreamSynchronize(result_found)");
    if (rc != 0) {
        return rc;
    }

    *result_found = *g_result_found_host;

    if (*result_found != 0) {
        err = hipMemcpyAsync(g_result_word_host, g_result_word, 64, hipMemcpyDeviceToHost, g_stream);
        rc = hip_fail(err, "hipMemcpyAsync(result_word)");
        if (rc != 0) {
            return rc;
        }
        err = hipStreamSynchronize(g_stream);
        rc = hip_fail(err, "hipStreamSynchronize(result_word)");
        if (rc != 0) {
            return rc;
        }
        memcpy(result_word, g_result_word_host, 64);
    }

    return 0;
}

extern "C" int get_hip_device_info(HipDeviceInfo* info) {
    if (info == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    memset(info, 0, sizeof(HipDeviceInfo));

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    int rc = hip_fail(err, "hipGetDeviceCount");
    if (rc != 0) {
        return rc;
    }
    if (device_count <= 0) {
        return static_cast<int>(hipErrorNoDevice);
    }

    rc = ensure_device_initialized();
    if (rc != 0) {
        return rc;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, 0);
    rc = hip_fail(err, "hipGetDeviceProperties");
    if (rc != 0) {
        return rc;
    }

    info->available = 1;
    info->device_count = device_count;
    info->device_id = 0;
    info->multiprocessor_count = props.multiProcessorCount;
    info->max_threads_per_block = props.maxThreadsPerBlock;
    info->warp_size = props.warpSize;
    info->total_global_mem = props.totalGlobalMem;
    info->clock_rate_khz = props.clockRate;
    strncpy(reinterpret_cast<char*>(info->name), props.name, sizeof(info->name) - 1);

    return 0;
}

extern "C" void cleanup_brute_force_hip() {
    if (g_charset != nullptr) {
        hipError_t free_status = hipFree(g_charset);
        (void)free_status;
        g_charset = nullptr;
        g_charset_len = 0;
    }
    if (g_target_hash != nullptr) {
        hipError_t free_status = hipFree(g_target_hash);
        (void)free_status;
        g_target_hash = nullptr;
    }
    if (g_result_found != nullptr) {
        hipError_t free_status = hipFree(g_result_found);
        (void)free_status;
        g_result_found = nullptr;
    }
    if (g_result_word != nullptr) {
        hipError_t free_status = hipFree(g_result_word);
        (void)free_status;
        g_result_word = nullptr;
    }
    if (g_result_found_host != nullptr) {
        hipError_t free_status = hipHostFree(g_result_found_host);
        (void)free_status;
        g_result_found_host = nullptr;
    }
    if (g_result_word_host != nullptr) {
        hipError_t free_status = hipHostFree(g_result_word_host);
        (void)free_status;
        g_result_word_host = nullptr;
    }
    if (g_stream != nullptr) {
        hipError_t destroy_status = hipStreamDestroy(g_stream);
        (void)destroy_status;
        g_stream = nullptr;
    }
    if (g_charset_host != nullptr) {
        free(g_charset_host);
        g_charset_host = nullptr;
        g_charset_host_cap = 0;
    }
    g_target_hash_valid = 0;
    g_device_initialized = 0;
}

#undef LAUNCH_FIXED_KERNEL
#undef LAUNCH_GENERIC_KERNEL
