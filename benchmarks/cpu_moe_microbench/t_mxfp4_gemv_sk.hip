// HIP MXFP4 GEMV kernel (port of t_mxfp4_gemv_sk.hlsl, 2026-08-30).
//
// Computes outv[row] = (sum over 32-elem blocks of (wsum + bb) * bs2) * gbl[row] + rowBias[row]
// where wsum = sum over 32 nibbles of (decoded FP4 value * act[idx]).
//
// One thread block per row (256 threads). Wavefront = 32 threads on RDNA 3.
//
// Inputs:
//   packed  : [M, K/8] uint32   (4-bit nibbles, packed 8 per uint)
//   scl     : [M, K/32] float32 (per-32-block scale, e8m0)
//   bias_pb : [M, K/32] float32 (per-32-block bias)
//   act     : [K] float32       (shared across all rows in FC_CALL mode)
//   gbl     : [M] float32       (per-row global scale)
//   rowBias : [M] float32       (per-row bias)
//
// Output:
//   outv    : [M] float32

#include <hip/hip_runtime.h>

#define WARP 32
#define BLOCK 256

__device__ __forceinline__ float fp4_to_float(unsigned n) {
    // Qwen3.6 MXFP4 export is uint4-AFFINE (python/freetoken/checkpoint/quantize.py):
    // value = nibble(0..15) * scale + bias per 32-block. NOT e2m1-coded. The old
    // e2m1 table here decoded cos=-0.33 anti-correlated weights, poisoning the FC
    // output and collapsing MTP acceptance to 0 (mtp.py _UAFF16 comment, 2026-09-02).
    return float(n);
}

extern "C" __global__ __launch_bounds__(BLOCK) void mxfp4_gemv_fc(
    const unsigned* __restrict__ packed,
    const float*    __restrict__ scl,
    const float*    __restrict__ bias_pb,
    const float*    __restrict__ act,
    const float*    __restrict__ gbl,
    const float*    __restrict__ rowBias,
    float*          __restrict__ outv,
    unsigned K,
    unsigned nbPerRow,
    unsigned nsPerRow
) {
    unsigned row = blockIdx.x;
    unsigned t   = threadIdx.x;
    float acc = 0.0f;

    for (unsigned b = t; b < nbPerRow; b += BLOCK) {
        unsigned pbase = row * nbPerRow + b;
        unsigned sbase = row * nsPerRow + b;
        float bs2 = scl[sbase];
        float bb = bias_pb[sbase];

        float wsum = 0.0f;
        float asum = 0.0f;   // sum of activations over this 32-block (for the affine bias)
        unsigned ai_base = b * 32u;

        // Process 4 uints = 32 nibbles
        #pragma unroll
        for (int wordIdx = 0; wordIdx < 4; wordIdx++) {
            unsigned w = packed[pbase * 4u + (unsigned)wordIdx];
            unsigned ai = ai_base + (unsigned)wordIdx * 8u;
            unsigned n0 = w & 0xFu; unsigned n1 = (w >> 4) & 0xFu;
            unsigned n2 = (w >> 8) & 0xFu; unsigned n3 = (w >> 12) & 0xFu;
            unsigned n4 = (w >> 16) & 0xFu; unsigned n5 = (w >> 20) & 0xFu;
            unsigned n6 = (w >> 24) & 0xFu; unsigned n7 = (w >> 28) & 0xFu;

            float a0 = act[ai + 0u];
            float a1 = act[ai + 1u];
            float a2 = act[ai + 2u];
            float a3 = act[ai + 3u];
            float a4 = act[ai + 4u];
            float a5 = act[ai + 5u];
            float a6 = act[ai + 6u];
            float a7 = act[ai + 7u];

            wsum += fp4_to_float(n0) * a0;
            wsum += fp4_to_float(n1) * a1;
            wsum += fp4_to_float(n2) * a2;
            wsum += fp4_to_float(n3) * a3;
            wsum += fp4_to_float(n4) * a4;
            wsum += fp4_to_float(n5) * a5;
            wsum += fp4_to_float(n6) * a6;
            wsum += fp4_to_float(n7) * a7;
            asum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        }
        // exact affine: w = n*s + b  =>  dot_block = s*sum(n*a) + b*sum(a)
        acc += wsum * bs2 + bb * asum;
    }

    __shared__ float sh[BLOCK];
    sh[t] = acc;
    __syncthreads();
    // Reduction in BLOCK threads
    for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0) outv[row] = sh[0] * gbl[row] + rowBias[row];
}

extern "C" hipError_t launch_mxfp4_gemv_fc(
    const unsigned* d_packed, const float* d_scl, const float* d_bias_pb,
    const float* d_act, const float* d_gbl, const float* d_rowBias,
    float* d_outv, unsigned M, unsigned K, unsigned nbPerRow, unsigned nsPerRow,
    hipStream_t stream
) {
    void* args[] = {
        (void*)&d_packed, (void*)&d_scl, (void*)&d_bias_pb,
        (void*)&d_act, (void*)&d_gbl, (void*)&d_rowBias,
        (void*)&d_outv, (void*)&K, (void*)&nbPerRow, (void*)&nsPerRow,
    };
    hipLaunchKernel((void*)mxfp4_gemv_fc,
                    dim3(M, 1, 1), dim3(BLOCK, 1, 1),
                    args, 0, stream);
    return hipGetLastError();
}
