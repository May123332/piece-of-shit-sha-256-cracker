pub const SHA256_KERNEL: &str = r#"
// ==========================================
// SHA-256 VECTORIZED (UINT4) + LOP3 OPTIMIZED + PREFIX SUPPORT
// ==========================================

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// Vector Rotate
#define ROR(x, n) rotate(x, (uint4)(32 - n))

// Optimized Logic for LOP3
#define CH(x, y, z) (z ^ (x & (y ^ z)))
#define MAJ(x, y, z) ((x & y) | (z & (x | y)))

#define SIGMA0(x) (ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22))
#define SIGMA1(x) (ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25))
#define GAMMA0(x) (ROR(x, 7) ^ ROR(x, 18) ^ (x >> 3))
#define GAMMA1(x) (ROR(x, 17) ^ ROR(x, 19) ^ (x >> 10))

__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void sha256_transform_vector(uint4* state, const uint4* W_schedule) {
    uint4 a = state[0], b = state[1], c = state[2], d = state[3];
    uint4 e = state[4], f = state[5], g = state[6], h = state[7];
    uint4 t1, t2;

    #pragma unroll
    for(int i=0; i<64; i++) {
        t1 = h + SIGMA1(e) + CH(e, f, g) + (uint4)(K[i]) + W_schedule[i];
        t2 = SIGMA0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void sha256_hash_vector(const uchar* c0, const uchar* c1, const uchar* c2, const uchar* c3, uint len, uint4* digest_out) {
    uint4 state[8];
    state[0] = (uint4)(0x6a09e667); state[1] = (uint4)(0xbb67ae85);
    state[2] = (uint4)(0x3c6ef372); state[3] = (uint4)(0xa54ff53a);
    state[4] = (uint4)(0x510e527f); state[5] = (uint4)(0x9b05688c);
    state[6] = (uint4)(0x1f83d9ab); state[7] = (uint4)(0x5be0cd19);
    
    uint4 W[64];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = i * 4;
        uint4 w = (uint4)(0);
        
        // Manual packing of 4 lanes
        // Note: This is optimized for single block (<56 bytes)
        // Lane 0
        uint v = 0;
        if(idx < len) v |= ((uint)c0[idx]) << 24; else if(idx == len) v |= 0x80000000;
        if(idx+1 < len) v |= ((uint)c0[idx+1]) << 16; else if(idx+1 == len) v |= 0x00800000;
        if(idx+2 < len) v |= ((uint)c0[idx+2]) << 8; else if(idx+2 == len) v |= 0x00008000;
        if(idx+3 < len) v |= ((uint)c0[idx+3]); else if(idx+3 == len) v |= 0x00000080;
        w.x = v;

        // Lane 1
        v = 0;
        if(idx < len) v |= ((uint)c1[idx]) << 24; else if(idx == len) v |= 0x80000000;
        if(idx+1 < len) v |= ((uint)c1[idx+1]) << 16; else if(idx+1 == len) v |= 0x00800000;
        if(idx+2 < len) v |= ((uint)c1[idx+2]) << 8; else if(idx+2 == len) v |= 0x00008000;
        if(idx+3 < len) v |= ((uint)c1[idx+3]); else if(idx+3 == len) v |= 0x00000080;
        w.y = v;

        // Lane 2
        v = 0;
        if(idx < len) v |= ((uint)c2[idx]) << 24; else if(idx == len) v |= 0x80000000;
        if(idx+1 < len) v |= ((uint)c2[idx+1]) << 16; else if(idx+1 == len) v |= 0x00800000;
        if(idx+2 < len) v |= ((uint)c2[idx+2]) << 8; else if(idx+2 == len) v |= 0x00008000;
        if(idx+3 < len) v |= ((uint)c2[idx+3]); else if(idx+3 == len) v |= 0x00000080;
        w.z = v;

        // Lane 3
        v = 0;
        if(idx < len) v |= ((uint)c3[idx]) << 24; else if(idx == len) v |= 0x80000000;
        if(idx+1 < len) v |= ((uint)c3[idx+1]) << 16; else if(idx+1 == len) v |= 0x00800000;
        if(idx+2 < len) v |= ((uint)c3[idx+2]) << 8; else if(idx+2 == len) v |= 0x00008000;
        if(idx+3 < len) v |= ((uint)c3[idx+3]); else if(idx+3 == len) v |= 0x00000080;
        w.w = v;

        W[i] = w;
    }
    
    uint bit_len = len * 8;
    W[15] = (uint4)(bit_len); 
    W[14] = (uint4)(0);       
    
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = GAMMA1(W[i - 2]) + W[i - 7] + GAMMA0(W[i - 15]) + W[i - 16];
    }
    
    sha256_transform_vector(state, W);
    
    for(int i=0; i<8; i++) digest_out[i] = state[i];
}

__kernel void brute_force_attack(
    __constant uchar* charset,
    ulong charset_len,
    ulong start_offset, 
    uint suffix_len, // Length of the part we generate (total_len - prefix_len)
    __constant uchar* target_hash,
    __constant uchar* salt,
    uint salt_len,
    __global uint* result_found,
    __global uchar* result_word,
    __constant uchar* prefix, // NEW: Prefix buffer
    uint prefix_len           // NEW: Prefix length
) {
    ulong gid = get_global_id(0);
    if (*result_found > 0) return;

    ulong base_id = start_offset + (gid * 4);
    uint total_len = prefix_len + suffix_len;
    
    // Generate 4 Candidates (Prefix + Generated Suffix)
    uchar c0[64], c1[64], c2[64], c3[64];
    
    // Copy Prefix first
    for(int i=0; i<prefix_len; i++) {
        c0[i] = prefix[i]; c1[i] = prefix[i]; c2[i] = prefix[i]; c3[i] = prefix[i];
    }
    
    // Gen Suffixes
    // Lane 0
    ulong t = base_id;
    for(int i=suffix_len-1; i>=0; i--) { c0[prefix_len + i] = charset[t % charset_len]; t /= charset_len; }
    
    // Lane 1
    t = base_id + 1;
    for(int i=suffix_len-1; i>=0; i--) { c1[prefix_len + i] = charset[t % charset_len]; t /= charset_len; }
    
    // Lane 2
    t = base_id + 2;
    for(int i=suffix_len-1; i>=0; i--) { c2[prefix_len + i] = charset[t % charset_len]; t /= charset_len; }
    
    // Lane 3
    t = base_id + 3;
    for(int i=suffix_len-1; i>=0; i--) { c3[prefix_len + i] = charset[t % charset_len]; t /= charset_len; }
    
    // Hash All 4
    uint4 digest[8]; 
    sha256_hash_vector(c0, c1, c2, c3, total_len, digest);
    
    // Check Matches
    uint target_w[8];
    #pragma unroll
    for(int i=0; i<8; i++) {
        target_w[i] = (target_hash[i*4]<<24) | (target_hash[i*4+1]<<16) | (target_hash[i*4+2]<<8) | target_hash[i*4+3];
    }
    
    // Lane 0
    bool m = true; for(int i=0; i<8; i++) if(digest[i].x != target_w[i]) m = false;
    if(m) { *result_found = 1; for(int i=0; i<total_len; i++) result_word[i] = c0[i]; result_word[total_len]=0; return; }
    
    // Lane 1
    m = true; for(int i=0; i<8; i++) if(digest[i].y != target_w[i]) m = false;
    if(m) { *result_found = 1; for(int i=0; i<total_len; i++) result_word[i] = c1[i]; result_word[total_len]=0; return; }
    
    // Lane 2
    m = true; for(int i=0; i<8; i++) if(digest[i].z != target_w[i]) m = false;
    if(m) { *result_found = 1; for(int i=0; i<total_len; i++) result_word[i] = c2[i]; result_word[total_len]=0; return; }
    
    // Lane 3
    m = true; for(int i=0; i<8; i++) if(digest[i].w != target_w[i]) m = false;
    if(m) { *result_found = 1; for(int i=0; i<total_len; i++) result_word[i] = c3[i]; result_word[total_len]=0; return; }
}

__kernel void dictionary_attack(
    __global const uchar* wordlist_data,
    __global const uint* offsets,
    uint count,
    __constant uchar* target_hash, 
    __constant uchar* salt,       
    uint salt_len,
    __global uint* result_found,
    __global uint* result_index
) {
    uint gid = get_global_id(0);
}
"#;
