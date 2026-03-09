// ============================================================
// RX 6800 / RDNA2 (gfx1030) Tuning Notes
// ------------------------------------------------------------
// - SIMD count: 60 CUs * 2 SIMDs = 120 SIMDs
// - Max wavefronts per SIMD: 16
// - VGPRs per SIMD: 1024 (RDNA2)
// - FULL occupancy requires ≤ 64 VGPRs per thread
// - SHA256 W[64] alone = 64 VGPRs → terrible occupancy
// - Fix: rolling W[16] cuts ~48 VGPRs, targets ≤60 total
// - uint8 vectorization is an NVIDIA pattern; AMD scalarizes
//   it and blows up register pressure. Use scalar + more threads.
// - LWS=256 = 4 wavefronts per workgroup, good for RDNA2 scheduler
// - Target throughput: ~1.5-2 GH/s on RX 6800 for raw SHA256
// ============================================================

// Batch/dispatch tuning for RX 6800
// Target ~50ms per dispatch (GPU TDR is usually 2s, 50ms gives headroom)
// At ~1.5 GH/s: 1.5e9 * 0.05 = 75M candidates per batch
pub const RX6800_OPTIMAL_LWS: usize = 256;
pub const RX6800_BATCH_SIZE: u64 = 64 * 1024 * 1024; // 67M, aligned to LWS

pub const SHA256_KERNEL: &str = r#"
// ============================================================
// SHA-256 OPTIMIZED FOR AMD RDNA2
// Key optimizations:
//   1. Rolling W[16] message schedule (saves ~48 VGPRs vs W[64])
//   2. State stored in 8 scalar registers (no rotation array)
//   3. Unrolled rounds for compiler scheduling
//   4. __local target hash cache (saves constant cache pressure)
//   5. One candidate per thread (not uint8 - that's NV-only)
//   6. Pre-parsed uint8[8] target hash (no per-thread uchar decode)
//   7. LWS=256 for both kernels (4 wavefronts/workgroup on RDNA2)
// ============================================================

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// Fast rotate using hardware rotate instruction (AMD supports this)
#define ROR(x, n)  (rotate((uint)(x), (uint)(32u - (n))))

// SHA-256 round functions
#define CH(e,f,g)    ((g) ^ ((e) & ((f) ^ (g))))
#define MAJ(a,b,c)   (((a) & (b)) | ((c) & ((a) | (b))))
#define EP0(a)       (ROR(a, 2u)  ^ ROR(a, 13u) ^ ROR(a, 22u))
#define EP1(e)       (ROR(e, 6u)  ^ ROR(e, 11u) ^ ROR(e, 25u))
#define SIG0(x)      (ROR(x, 7u)  ^ ROR(x, 18u) ^ ((x) >> 3u))
#define SIG1(x)      (ROR(x, 17u) ^ ROR(x, 19u) ^ ((x) >> 10u))

// One round of SHA-256. h is added to produce new a; d gets t1 added.
// Using explicit variable names avoids the rotation array and saves registers.
#define SHA256_ROUND(a,b,c,d,e,f,g,h,w,k) { \
    uint _t = (h) + EP1(e) + CH(e,f,g) + (k) + (w); \
    (d) += _t; \
    (h) = EP0(a) + MAJ(a,b,c) + _t; \
}

__constant uint K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// ============================================================
// CORE SHA-256 - rolling W[16], single block only (msg ≤ 55B)
// Targeting ≤ 64 VGPRs: 8 state + 16 schedule + ~20 temps ≈ 44
// ============================================================
inline bool sha256_match(
    const uchar* msg,
    uint         msg_len,
    const uint*  target   // 8 pre-parsed uint words (big-endian)
) {
    uint W[16];

    // --- Pack message into W[0..15] with SHA-256 big-endian padding ---
    // Unrolled manually because #pragma unroll on index-computed
    // byte access causes spills on some AMD drivers.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int b = i << 2;
        uint w = 0;
        if      ((uint)b     < msg_len) w  = ((uint)msg[b])     << 24;
        else if ((uint)b     == msg_len) w  = 0x80000000u;
        if      ((uint)(b+1) < msg_len) w |= ((uint)msg[b+1])   << 16;
        else if ((uint)(b+1) == msg_len) w |= 0x00800000u;
        if      ((uint)(b+2) < msg_len) w |= ((uint)msg[b+2])   <<  8;
        else if ((uint)(b+2) == msg_len) w |= 0x00008000u;
        if      ((uint)(b+3) < msg_len) w |= ((uint)msg[b+3]);
        else if ((uint)(b+3) == msg_len) w |= 0x00000080u;
        W[i] = w;
    }
    W[14] = 0u;
    W[15] = msg_len * 8u;

    // --- Initial hash state ---
    uint a = 0x6a09e667u, b2 = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f  = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

    // --- Rounds 0-15 (W already loaded) ---
    SHA256_ROUND(a,b2,c,d,e,f,g,h, W[ 0], K[ 0])
    SHA256_ROUND(h,a,b2,c,d,e,f,g, W[ 1], K[ 1])
    SHA256_ROUND(g,h,a,b2,c,d,e,f, W[ 2], K[ 2])
    SHA256_ROUND(f,g,h,a,b2,c,d,e, W[ 3], K[ 3])
    SHA256_ROUND(e,f,g,h,a,b2,c,d, W[ 4], K[ 4])
    SHA256_ROUND(d,e,f,g,h,a,b2,c, W[ 5], K[ 5])
    SHA256_ROUND(c,d,e,f,g,h,a,b2, W[ 6], K[ 6])
    SHA256_ROUND(b2,c,d,e,f,g,h,a, W[ 7], K[ 7])
    SHA256_ROUND(a,b2,c,d,e,f,g,h, W[ 8], K[ 8])
    SHA256_ROUND(h,a,b2,c,d,e,f,g, W[ 9], K[ 9])
    SHA256_ROUND(g,h,a,b2,c,d,e,f, W[10], K[10])
    SHA256_ROUND(f,g,h,a,b2,c,d,e, W[11], K[11])
    SHA256_ROUND(e,f,g,h,a,b2,c,d, W[12], K[12])
    SHA256_ROUND(d,e,f,g,h,a,b2,c, W[13], K[13])
    SHA256_ROUND(c,d,e,f,g,h,a,b2, W[14], K[14])
    SHA256_ROUND(b2,c,d,e,f,g,h,a, W[15], K[15])

    // --- Rounds 16-63 (rolling window, expand W in-place) ---
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        int j = i & 15;
        W[j] = SIG1(W[(i-2)  & 15])
             + W[(i-7)  & 15]
             + SIG0(W[(i-15) & 15])
             + W[(i-16) & 15];
        // Rotate state variables via compile-time constant indexing.
        // We map each round to the right state variable by using the
        // offset into the a,b,c,d,e,f,g,h rotation cycle.
        switch (i & 7) {
            case 0: SHA256_ROUND(a,b2,c,d,e,f,g,h, W[j], K[i]); break;
            case 1: SHA256_ROUND(h,a,b2,c,d,e,f,g, W[j], K[i]); break;
            case 2: SHA256_ROUND(g,h,a,b2,c,d,e,f, W[j], K[i]); break;
            case 3: SHA256_ROUND(f,g,h,a,b2,c,d,e, W[j], K[i]); break;
            case 4: SHA256_ROUND(e,f,g,h,a,b2,c,d, W[j], K[i]); break;
            case 5: SHA256_ROUND(d,e,f,g,h,a,b2,c, W[j], K[i]); break;
            case 6: SHA256_ROUND(c,d,e,f,g,h,a,b2, W[j], K[i]); break;
            case 7: SHA256_ROUND(b2,c,d,e,f,g,h,a, W[j], K[i]); break;
        }
    }

    // --- Add IV and compare ---
    // Using short-circuit: bail out as soon as any word mismatches.
    // This saves ~87.5% of the comparison work on non-matches.
    if ((0x6a09e667u + a) != target[0]) return false;
    if ((0xbb67ae85u + b2) != target[1]) return false;
    if ((0x3c6ef372u + c) != target[2]) return false;
    if ((0xa54ff53au + d) != target[3]) return false;
    if ((0x510e527fu + e) != target[4]) return false;
    if ((0x9b05688cu + f) != target[5]) return false;
    if ((0x1f83d9abu + g) != target[6]) return false;
    return (0x5be0cd19u + h) == target[7];
}

// ============================================================
// DICTIONARY ATTACK KERNEL
// Optimizations:
//   - LWS=256 for 4 wavefronts/workgroup → better scheduler fill
//   - __local target hash: loaded once per workgroup, not per thread
//   - scalar SHA256 with rolling W[16]
//   - salt variants: raw, word+salt, salt+word
// ============================================================
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void dictionary_attack(
    __global const uchar* wordlist_data,
    __global const uint*  offsets,       // byte offsets into wordlist_data
    uint                  count,         // total word count
    __constant uint*      target_hash_w, // pre-parsed 8x uint32 (big-endian)
    __constant uchar*     salt,
    uint                  salt_len,
    __global uint*        result_found,
    __global uint*        result_index
) {
    // Cache target hash in LDS once per workgroup (saves constant cache bandwidth)
    __local uint local_target[8];
    if (get_local_id(0) < 8) {
        local_target[get_local_id(0)] = target_hash_w[get_local_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint gid = get_global_id(0);
    if (gid >= count) return;
    if (*result_found) return;

    uint word_start = offsets[gid];
    uint word_len   = offsets[gid + 1] - word_start;
    if (word_len > 55u) return; // single-block SHA256 limit

    uchar word_buf[56]; // 55 bytes max + 1 safety
    for (uint i = 0; i < word_len; i++)
        word_buf[i] = wordlist_data[word_start + i];

    // Variant 1: raw word
    if (sha256_match(word_buf, word_len, local_target)) {
        atom_cmpxchg(result_found, 0u, 1u);
        *result_index = gid;
        return;
    }

    if (salt_len > 0u) {
        uchar salted[56];
        uint combined = word_len + salt_len;
        if (combined <= 55u) {
            // Variant 2: word + salt
            for (uint i = 0; i < word_len; i++)  salted[i]           = word_buf[i];
            for (uint i = 0; i < salt_len; i++)  salted[word_len+i]  = salt[i];
            if (sha256_match(salted, combined, local_target)) {
                atom_cmpxchg(result_found, 0u, 1u);
                *result_index = gid;
                return;
            }

            // Variant 3: salt + word
            for (uint i = 0; i < salt_len; i++)  salted[i]           = salt[i];
            for (uint i = 0; i < word_len; i++)  salted[salt_len+i]  = word_buf[i];
            if (sha256_match(salted, combined, local_target)) {
                atom_cmpxchg(result_found, 0u, 1u);
                *result_index = gid;
                return;
            }
        }
    }
}

// ============================================================
// BRUTE FORCE KERNEL
// Optimizations:
//   - One candidate per thread (AMD doesn't benefit from uint8/uint4)
//   - Scalar SHA256 with rolling W[16]
//   - LWS=256 for 4 wavefronts/workgroup
//   - __local target hash cache
//   - atom_cmpxchg for result (avoids write races)
//   - Candidates generated from index via charset modular decode
// ============================================================
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void brute_force_attack(
    __constant uchar* charset,
    ulong             charset_len,
    ulong             start_offset,
    uint              suffix_len,
    __constant uint*  target_hash_w,  // pre-parsed 8x uint32
    __constant uchar* salt,
    uint              salt_len,
    __global uint*    result_found,
    __global uchar*   result_word,
    __constant uchar* prefix,
    uint              prefix_len
) {
    __local uint local_target[8];
    if (get_local_id(0) < 8) {
        local_target[get_local_id(0)] = target_hash_w[get_local_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ulong gid      = get_global_id(0);
    ulong cand_idx = start_offset + gid;
    uint  total_len = prefix_len + suffix_len;

    if (*result_found) return;
    if (total_len > 55u) return;

    // Build candidate: prefix + charset-decoded suffix
    uchar cand[56];
    for (uint i = 0; i < prefix_len; i++) cand[i] = prefix[i];

    ulong t = cand_idx;
    for (int i = (int)suffix_len - 1; i >= 0; i--) {
        cand[prefix_len + i] = charset[t % charset_len];
        t /= charset_len;
    }

    if (sha256_match(cand, total_len, local_target)) {
        if (atom_cmpxchg(result_found, 0u, 1u) == 0u) {
            for (uint i = 0; i < total_len; i++) result_word[i] = cand[i];
            result_word[total_len] = 0;
        }
    }
}
"#;
