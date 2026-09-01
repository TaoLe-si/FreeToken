// HIP route kernel (port of t_mtp_moe_route.hlsl, 2026-08-30).
//
// Computes: hidden_f32[H] @ routerW_f32[E, H].T -> top-8 idx + softmax weights.
// One thread block of 32 threads; each thread processes E/32 logits in parallel.
// Then a bitonic top-8 reduction in shared memory + softmax.
//
// ROCm 6.4 / RDNA 3 (AMD Radeon 780M, 12 CUs).
// Wavefront = 32 threads on RDNA (vs NVIDIA 64), LDS bank = 32 dwords wide.

#include <hip/hip_runtime.h>

#define HIP_CHECK(call) do { hipError_t err = call; if (err != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); return err; } } while(0)

// Wavefront (RDNA) = 32 threads
#define WARP 32

// Top-K (k=8) via bitonic sort in shared memory.
// Each thread holds a (logit, idx) pair. We sort and keep top-8.
__device__ __forceinline__ void bitonic_sort_step(float& v, unsigned& i, float& nv, unsigned& ni,
                                                  bool asc) {
    // If ordering is wrong, swap with neighbor
    bool swap = asc ? (v > nv) : (v < nv);
    if (swap) {
        float tv = v; v = nv; nv = tv;
        unsigned ti = i; i = ni; ni = ti;
    }
}

// One thread block = 32 threads. Each thread holds E/32 logits in registers.
// For E=256 -> 8 logits per thread.
__global__ __launch_bounds__(32) void moe_route_kernel(
    const float* __restrict__ routerW,  // [E, H]
    const float* __restrict__ hidden,    // [H]
    unsigned* __restrict__ top8_idx,     // [8] output
    float* __restrict__ top8_w,          // [8] output (softmax)
    int E, int H
) {
    int tid = threadIdx.x;  // 0..31
    int n_per_thread = (E + WARP - 1) / WARP;  // 8 for E=256

    // Each thread computes n_per_thread logits via dot product with hidden
    float local_logits[16];  // up to 16 per thread (E/32 <= 16 for E<=512)
    unsigned local_idx[16];
    for (int i = 0; i < n_per_thread; ++i) {
        int e = tid * n_per_thread + i;
        if (e >= E) { local_logits[i] = -INFINITY; local_idx[i] = e; continue; }
        float sum = 0.0f;
        // Dot product: sum over H
        // Hidden is shared across all threads -- LDS unroll not needed
        for (int k = 0; k < H; ++k) {
            sum += routerW[e * H + k] * hidden[k];
        }
        local_logits[i] = sum;
        local_idx[i] = e;
    }

    // Bitonic top-K reduction across the 32 threads.
    // For E=256, total elements = 256 (32 threads x 8 logits each).
    // Simple approach: each thread maintains a local top-1, then warp-wide reduce.
    // For simplicity: each thread does serial sort of its n_per_thread logits,
    // then warp shuffles to find global top-8.
    // Sort local n_per_thread logits descending.
    for (int i = 1; i < n_per_thread; ++i) {
        for (int j = i; j > 0 && local_logits[j] > local_logits[j-1]; --j) {
            float tv = local_logits[j]; local_logits[j] = local_logits[j-1]; local_logits[j-1] = tv;
            unsigned ti = local_idx[j]; local_idx[j] = local_idx[j-1]; local_idx[j-1] = ti;
        }
    }
    // local_logits[0] is thread's max.

    // Now find top-8 across all 32 threads via 32-wide reductions.
    // Step 1: each thread's max -> find max across warp
    float warp_max = local_logits[0];
    unsigned warp_max_idx = local_idx[0];
    // 32 threads -> need 5 reduction steps
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor(warp_max, offset);
        unsigned other_idx = __shfl_xor(warp_max_idx, offset);
        if (other > warp_max) {
            warp_max = other;
            warp_max_idx = other_idx;
        }
    }
    // After warp reduction, lane 0 has the max -- but other lanes still have stale data.
    // For top-K, broadcast and repeat.
    // Actually for k=8, let's do K iterations of warp reduction.

    // Simpler: each iteration finds the current max, sets it to -inf in the source,
    // and collects it as the next top-K element.
    __shared__ float shared_max[8];
    __shared__ unsigned shared_idx[8];

    for (int k = 0; k < 8; ++k) {
        // Find warp max
        float wmax = local_logits[0];
        unsigned widx = local_idx[0];
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_xor(wmax, offset);
            unsigned other_idx = __shfl_xor(widx, offset);
            if (other > wmax) {
                wmax = other;
                widx = other_idx;
            }
        }
        // Lane 0 has the max.
        if (tid == 0) {
            shared_max[k] = wmax;
            shared_idx[k] = widx;
        }
        // Now set the winning lane's max to -inf in the source so next iteration finds next max.
        // But the winning lane's local_logits[0] is the max. We need to invalidate it.
        // The winner's lane id is whoever's local_logits[0] == wmax. Use broadcast.
        unsigned winner = __shfl(tid, (widx == local_idx[0]) ? tid : 0);
        // Simpler: if this thread has the max, set it to -inf.
        if (local_idx[0] == shared_idx[k]) {
            local_logits[0] = -INFINITY;
            // Re-sort to bring next value to position 0
            for (int j = 1; j < n_per_thread; ++j) {
                if (local_logits[j] > local_logits[0]) {
                    float tv = local_logits[0]; local_logits[0] = local_logits[j]; local_logits[j] = tv;
                    unsigned ti = local_idx[0]; local_idx[0] = local_idx[j]; local_idx[j] = ti;
                } else {
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Softmax: exp(x - max) / sum. Max of 8 values.
    if (tid < 8) {
        // Compute max for numerical stability
        float m = shared_max[0];
        for (int i = 1; i < 8; ++i) {
            if (shared_max[i] > m) m = shared_max[i];
        }
        float e = expf(shared_max[tid] - m);
        // Stash in shared_max (reuse for exp values)
        shared_max[tid] = e;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += shared_max[i];
        for (int i = 0; i < 8; ++i) {
            top8_idx[i] = shared_idx[i];
            top8_w[i] = shared_max[i] / sum;
        }
    }
}

extern "C" {

// Launch wrapper exposed via hiprtc/host API.
hipError_t launch_moe_route(
    const float* d_routerW,  // [E, H]
    const float* d_hidden,    // [H]
    unsigned* d_top8_idx,     // [8]
    float* d_top8_w,          // [8]
    int E, int H,
    hipStream_t stream
) {
    dim3 block(WARP, 1, 1);
    dim3 grid(1, 1, 1);
    void* args[] = { (void*)&d_routerW, (void*)&d_hidden, (void*)&d_top8_idx, (void*)&d_top8_w, (void*)&E, (void*)&H };
    hipLaunchKernel((void*)moe_route_kernel, grid, block,
                    args, 0, stream);
    return hipGetLastError();
}


}  // extern "C"
