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