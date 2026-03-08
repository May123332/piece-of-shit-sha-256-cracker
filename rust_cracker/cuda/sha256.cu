#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

__constant__ uint32_t c_target[8];

// PTX Optimized Rotates
#define ROR(x, n) (__funnelshift_r((x), (x), (n)))

// Optimized Logic (LOP3 candidates)
#define CH(x, y, z) (z ^ (x & (y ^ z)))
#define MAJ(x, y, z) ((x & y) | (z & (x | y)))

#define SIGMA0(x) (ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22))
#define SIGMA1(x) (ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25))
#define GAMMA0(x) (ROR(x, 7) ^ ROR(x, 18) ^ (x >> 3))
#define GAMMA1(x) (ROR(x, 17) ^ ROR(x, 19) ^ (x >> 10))
#define SWAP(x) (__byte_perm((x), 0, 0x0123))

// SHA-256 Constants (Hardcoded for register/cache speed)
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

// LAUNCH BOUNDS: Critical for Ampere
// 128 threads/block, Min 4 blocks/SM -> Max 128 registers/thread.
extern "C" __global__ void __launch_bounds__(128, 4) brute_force_cuda(
    const uint8_t* charset, 
    uint64_t charset_len, 
    uint64_t start_offset, 
    uint32_t len,
    uint32_t* result_found, 
    uint8_t* result_word
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No early exit check to avoid latency

    // --- INTERLEAVED x4 GENERATION ---
    // Registers: ~32-40 used here
    uint8_t pass[4][16]; 
    uint64_t base_id = start_offset + (idx * 4);
    
    #pragma unroll
    for(int k=0; k<4; k++) {
        uint64_t temp = base_id + k;
        for(int i = len - 1; i >= 0; i--) {
            pass[k][i] = charset[temp % charset_len];
            temp /= charset_len;
        }
    }

    // --- PACKING ---
    // Reusing 'w' as circular buffer to save registers.
    // Instead of w[4][64], we use w[4][16] and overwrite.
    uint32_t w[4][16]; 
    
    #pragma unroll
    for(int k=0; k<4; k++) {
        #pragma unroll
        for(int i=0; i<16; i++) w[k][i] = 0;
        #pragma unroll
        for(int i=0; i<len; i++) ((uint8_t*)&w[k][0])[i] = pass[k][i];
        ((uint8_t*)&w[k][0])[len] = 0x80;
        w[k][15] = __byte_perm((uint32_t)(len*8), 0, 0x0123);
        #pragma unroll
        for(int i=0; i<15; i++) w[k][i] = SWAP(w[k][i]);
    }

    // --- HASHING (x4 Interleaved) ---
    // Registers: 8 state * 4 = 32. 
    // Plus w buffer = 64. Total ~96. Fits in 128!
    uint32_t a[4], b[4], c[4], d[4], e[4], f[4], g[4], h[4];
    
    #pragma unroll
    for(int k=0; k<4; k++) {
        a[k]=0x6a09e667; b[k]=0xbb67ae85; c[k]=0x3c6ef372; d[k]=0xa54ff53a;
        e[k]=0x510e527f; f[k]=0x9b05688c; g[k]=0x1f83d9ab; h[k]=0x5be0cd19;
    }

    uint32_t t1, t2;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        // Just-In-Time Schedule Generation (Circular Buffer)
        // We overwrite w[k][i%16] with the NEXT word needed for future rounds
        // But for the current round 'i', we need w[k][i] (or its computed value).
        // Standard SHA256: w[i] = GAMMA1(w[i-2]) + w[i-7] + GAMMA0(w[i-15]) + w[i-16] 
        
        // MAPPING: 
        // We keep 16 words active. w[0]..w[15].
        // Round 0-15: Use w[i].
        // Round 16: Compute new w[0] based on w[14], w[9], w[1], w[0]. 
        // This effectively simulates the sliding window without 64 registers.
        
        uint32_t w_val[4];
        uint32_t k_val = K[i];

        if (i < 16) {
            #pragma unroll
            for(int k=0; k<4; k++) w_val[k] = w[k][i];
        } else {
            #pragma unroll
            for(int k=0; k<4; k++) {
                // Circular addressing logic
                // w[i&15] is the oldest word (w[i-16]), which we are about to overwrite with w[i]
                uint32_t w_m16 = w[k][i & 15]; 
                uint32_t w_m15 = w[k][(i + 1) & 15]; // (i-15) maps to (i+1)%16? Check math. 
                // Index mapping: idx = (i - offset) % 16.
                // If current buffer head is i%16.
                // Wait, simpler:
                // w[0]..w[15] are loaded.
                // i=16. we need w[14], w[9], w[1], w[0]. compute new w[16]. store in w[0].
                // i=17. we need w[15], w[10], w[2], w[1]. compute new w[17]. store in w[1].
                // So w[i&15] is the target.
                // w[(i+14)&15] is w[i-2].
                // w[(i+9)&15] is w[i-7].
                // w[(i+1)&15] is w[i-15].
                
                uint32_t n = w_m16 + GAMMA0(w[k][(i + 1) & 15]) + w[k][(i + 9) & 15] + GAMMA1(w[k][(i + 14) & 15]);
                w[k][i & 15] = n;
                w_val[k] = n;
            }
        }

        // Interleaved Compression (4-way parallel)
        #pragma unroll
        for(int k=0; k<4; k++) {
            t1 = h[k] + SIGMA1(e[k]) + CH(e[k], f[k], g[k]) + k_val + w_val[k];
            t2 = SIGMA0(a[k]) + MAJ(a[k], b[k], c[k]);
            h[k] = g[k];
            g[k] = f[k];
            f[k] = e[k];
            e[k] = d[k] + t1;
            d[k] = c[k];
            c[k] = b[k];
            b[k] = a[k];
            a[k] = t1 + t2;
        }
    }

    // --- CHECK ---
    #pragma unroll
    for(int k=0; k<4; k++) {
        if ((a[k] + 0x6a09e667) != c_target[0]) continue;
        if ((b[k] + 0xbb67ae85) != c_target[1]) continue;
        if ((c[k] + 0x3c6ef372) != c_target[2]) continue;
        if ((d[k] + 0xa54ff53a) != c_target[3]) continue;
        if ((e[k] + 0x510e527f) != c_target[4]) continue;
        if ((f[k] + 0x9b05688c) != c_target[5]) continue;
        if ((g[k] + 0x1f83d9ab) != c_target[6]) continue;
        if ((h[k] + 0x5be0cd19) != c_target[7]) continue;

        // Match Found (Atomic Exchange only on success)
        if (atomicExch(result_found, 1) == 0) {
            for(int i=0; i<len; i++) result_word[i] = pass[k][i];
            result_word[len] = 0;
        }
        return;
    }
}

extern "C" void launch_brute_force_cuda(
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
    cudaMemcpyToSymbol(c_target, target_hash, 32);
    brute_force_cuda<<<blocks, threads>>>(charset, charset_len, start_offset, len, result_found, result_word);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("[CUDA ERROR] Launch: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("[CUDA ERROR] Sync: %s\n", cudaGetErrorString(err));
}