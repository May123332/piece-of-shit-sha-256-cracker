#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
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
static uint32_t* g_target_hash = nullptr;
static uint32_t* g_result_found = nullptr;
static uint8_t* g_result_word = nullptr;

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

__device__ __forceinline__ void sha256_candidate(const uint8_t* pass, uint32_t len, uint32_t digest[8]) {
    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = pack_word(pass, len, i);
    }
    w[14] = 0;
    w[15] = len * 8;

    for (int i = 16; i < 64; i++) {
        w[i] = GAMMA1(w[i - 2]) + w[i - 7] + GAMMA0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = 0x6a09e667;
    uint32_t b = 0xbb67ae85;
    uint32_t c = 0x3c6ef372;
    uint32_t d = 0xa54ff53a;
    uint32_t e = 0x510e527f;
    uint32_t f = 0x9b05688c;
    uint32_t g = 0x1f83d9ab;
    uint32_t h = 0x5be0cd19;

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + w[i];
        uint32_t t2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    digest[0] = a + 0x6a09e667;
    digest[1] = b + 0xbb67ae85;
    digest[2] = c + 0x3c6ef372;
    digest[3] = d + 0xa54ff53a;
    digest[4] = e + 0x510e527f;
    digest[5] = f + 0x9b05688c;
    digest[6] = g + 0x1f83d9ab;
    digest[7] = h + 0x5be0cd19;
}

extern "C" __global__ __launch_bounds__(256, 2) void brute_force_hip(
    const uint8_t* charset,
    uint64_t charset_len,
    uint64_t start_offset,
    uint32_t len,
    uint64_t batch_size,
    const uint32_t* target_hash,
    uint32_t* result_found,
    uint8_t* result_word
) {
    uint64_t thread_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t thread_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t candidate_index = thread_index; candidate_index < batch_size; candidate_index += thread_stride) {
        if (*result_found != 0) {
            return;
        }

        uint8_t pass[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            pass[i] = 0;
        }

        uint64_t temp = start_offset + candidate_index;
        for (int i = static_cast<int>(len) - 1; i >= 0; i--) {
            pass[i] = charset[temp % charset_len];
            temp /= charset_len;
        }

        uint32_t digest[8];
        sha256_candidate(pass, len, digest);

        bool match = true;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (digest[i] != target_hash[i]) {
                match = false;
            }
        }

        if (match && atomicExch(result_found, 1) == 0) {
            for (uint32_t i = 0; i < len; i++) {
                result_word[i] = pass[i];
            }
            result_word[len] = 0;
            return;
        }
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
    int threads
) {
    if (charset == nullptr || target_hash == nullptr || result_found == nullptr || result_word == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    hipError_t err = hipSetDevice(0);
    int rc = hip_fail(err, "hipSetDevice");
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

    err = hipMemcpy(g_charset, charset, charset_len, hipMemcpyHostToDevice);
    rc = hip_fail(err, "hipMemcpy(charset)");
    if (rc != 0) {
        return rc;
    }

    err = hipMemcpy(g_target_hash, target_hash, 8 * sizeof(uint32_t), hipMemcpyHostToDevice);
    rc = hip_fail(err, "hipMemcpy(target_hash)");
    if (rc != 0) {
        return rc;
    }

    uint32_t zero = 0;
    err = hipMemcpy(g_result_found, &zero, sizeof(uint32_t), hipMemcpyHostToDevice);
    rc = hip_fail(err, "hipMemcpy(reset result_found)");
    if (rc != 0) {
        return rc;
    }

    hipLaunchKernelGGL(
        brute_force_hip,
        dim3(blocks),
        dim3(threads),
        0,
        0,
        g_charset,
        charset_len,
        start_offset,
        len,
        batch_size,
        g_target_hash,
        g_result_found,
        g_result_word
    );

    err = hipGetLastError();
    rc = hip_fail(err, "hipGetLastError");
    if (rc != 0) {
        return rc;
    }

    err = hipDeviceSynchronize();
    rc = hip_fail(err, "hipDeviceSynchronize");
    if (rc != 0) {
        return rc;
    }

    err = hipMemcpy(result_found, g_result_found, sizeof(uint32_t), hipMemcpyDeviceToHost);
    rc = hip_fail(err, "hipMemcpy(result_found)");
    if (rc != 0) {
        return rc;
    }

    if (*result_found != 0) {
        err = hipMemcpy(result_word, g_result_word, 64, hipMemcpyDeviceToHost);
        rc = hip_fail(err, "hipMemcpy(result_word)");
        if (rc != 0) {
            return rc;
        }
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

    err = hipSetDevice(0);
    rc = hip_fail(err, "hipSetDevice");
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
}
