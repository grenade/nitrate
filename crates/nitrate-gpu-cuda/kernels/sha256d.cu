/**
 * Minimal CUDA "hello" kernel to validate module load and launch.
 *
 * This kernel writes a simple deterministic pattern into the output buffer:
 *   out[i] = seed ^ i
 *
 * You can launch it with a grid/block configuration large enough to cover `len`.
 * If `len` is larger than the total number of threads, the remaining elements
 * will be left untouched.
 *
 * Expected host-side usage (pseudocode):
 *   - Allocate a device buffer `out` with `len` elements (u32).
 *   - Load the PTX module and get the function "hello_kernel".
 *   - Launch with parameters: (out_device_ptr, len, seed).
 *   - Copy the buffer back and verify a few positions.
 */

#include <stdint.h>

extern "C" __global__
void hello_kernel(uint32_t* __restrict__ out, uint32_t len, uint32_t seed) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < len) {
        out[gid] = seed ^ gid;
    }
}

// ========== SHA256d midstate scan kernel with candidate ring ==========

#include <stddef.h>

// SHA-256 constants in constant memory for initialization
__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// Declare shared memory for K constants
__shared__ uint32_t K_shared[64];

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    // Use CUDA intrinsic for faster rotation
    return __funnelshift_r(x, x, n);
}
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
__device__ __forceinline__ uint32_t Sig0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}
__device__ __forceinline__ uint32_t Sig1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}
__device__ __forceinline__ uint32_t sig0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t sig1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ void sha256_iv(uint32_t st[8]) {
    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;
    st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;
}

__device__ __forceinline__ uint32_t load_be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

__device__ __forceinline__ void sha256_compress(uint32_t st[8], const uint8_t block[64]) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be32(&block[i * 4]);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + Sig1(e) + Ch(e, f, g) + K_shared[i] + w[i];
        uint32_t t2 = Sig0(a) + Maj(a, b, c);
        h = g; g = f; f = e;
        e = d + t1;
        d = c; c = b; b = a;
        a = t1 + t2;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

__device__ __forceinline__ void store_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

__device__ __forceinline__ int hash_leq_target(const uint8_t h[32], const uint8_t t[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (h[i] < t[i]) return 1;
        if (h[i] > t[i]) return 0;
    }
    return 1;
}

typedef struct {
    uint32_t nonce;
    uint8_t hash_be[32];
    unsigned long long generation;
} Candidate;

// Optimized kernel: scan nonces using provided midstate and header tail, write candidates into a ring buffer
extern "C" __global__
void sha256d_scan_kernel(
    const uint32_t* __restrict__ midstate_be8,  // 8 words, big-endian SHA-256 state
    const uint8_t* __restrict__ tail16,         // 16 bytes: merkle_tail(4) | ntime(4 LE) | nbits(4 LE) | nonce placeholder(4 LE)
    const uint8_t* __restrict__ target_be32,    // 32-byte big-endian target
    uint32_t start_nonce,
    uint32_t nonce_count,
    unsigned long long generation,
    unsigned int ring_capacity,
    unsigned int* __restrict__ ring_write_idx,
    Candidate* __restrict__ ring_out
) {
    // Load SHA256 K constants into shared memory cooperatively
    if (threadIdx.x < 64) {
        K_shared[threadIdx.x] = SHA256_K[threadIdx.x];
    }
    __syncthreads();
    
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    // Process 4 nonces per thread for better throughput
    const uint32_t NONCES_PER_THREAD = 4;
    
    for (uint64_t base_i = gid * NONCES_PER_THREAD; base_i < (uint64_t)nonce_count; base_i += stride * NONCES_PER_THREAD) {
        
        #pragma unroll
        for (uint32_t j = 0; j < NONCES_PER_THREAD; j++) {
            uint64_t i = base_i + j;
            if (i >= (uint64_t)nonce_count) break;
            
            const uint32_t nonce = start_nonce + (uint32_t)i;

        // Build second 64-byte block for first SHA-256 (header bytes 64..79 + padding)
        uint8_t blk2[64];
        // Copy merkle_tail(4) + ntime(4 LE) + nbits(4 LE)
        #pragma unroll
        for (int j = 0; j < 12; ++j) {
            blk2[j] = tail16[j];
        }
        // Nonce (LE)
        blk2[12] = (uint8_t)(nonce);
        blk2[13] = (uint8_t)(nonce >> 8);
        blk2[14] = (uint8_t)(nonce >> 16);
        blk2[15] = (uint8_t)(nonce >> 24);
        // Padding
        blk2[16] = 0x80;
        #pragma unroll
        for (int j = 17; j < 56; ++j) {
            blk2[j] = 0;
        }
        // Length = 80 * 8 = 640 bits (64-bit BE)
        blk2[56] = 0x00; blk2[57] = 0x00; blk2[58] = 0x00; blk2[59] = 0x00;
        blk2[60] = 0x00; blk2[61] = 0x00; blk2[62] = 0x02; blk2[63] = 0x80;

        // First SHA-256 starting from provided midstate
        uint32_t st1[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) st1[k] = midstate_be8[k];
        sha256_compress(st1, blk2);

        // Produce first hash (32 bytes, big-endian)
        uint8_t h1[32];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            store_be32(&h1[k * 4], st1[k]);
        }

        // Second SHA-256 over 32-byte h1
        uint8_t blk3[64];
        // 32B data
        #pragma unroll
        for (int j = 0; j < 32; ++j) blk3[j] = h1[j];
        blk3[32] = 0x80;
        #pragma unroll
        for (int j = 33; j < 56; ++j) blk3[j] = 0;
        // Length = 32 * 8 = 256 bits
        blk3[56] = 0x00; blk3[57] = 0x00; blk3[58] = 0x00; blk3[59] = 0x00;
        blk3[60] = 0x00; blk3[61] = 0x00; blk3[62] = 0x01; blk3[63] = 0x00;

        uint32_t st2[8];
        sha256_iv(st2);
        sha256_compress(st2, blk3);

        uint8_t h2[32];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            store_be32(&h2[k * 4], st2[k]);
        }

        // Compare with target; if hash <= target, write candidate
        if (hash_leq_target(h2, target_be32)) {
            unsigned int idx = atomicAdd(ring_write_idx, 1u);
            // Place into a circular ring
            unsigned int slot = (ring_capacity > 0u) ? (idx % ring_capacity) : 0u;
            Candidate c;
            c.nonce = nonce;
            #pragma unroll
            for (int k = 0; k < 32; ++k) c.hash_be[k] = h2[k];
            c.generation = generation;
            ring_out[slot] = c;
            }
        }
    }
}