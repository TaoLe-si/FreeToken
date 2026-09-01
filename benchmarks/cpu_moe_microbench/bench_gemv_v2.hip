// Optimized BF16 GEMV for RDNA 3 (gfx1103) using packed math + vectorized loads.
//
// Algorithm:
//   One block per OUTPUT TILE (e.g. 8 consecutive rows).
//   Block size 256 threads = 8 warps.
//   Each warp handles 1 row.
//   Within a warp: each thread loads 8 bf16 per iteration via uint4 (16 bytes).
//   Uses packed BF16 multiply (compiler lowers to V_DOT2 / packed math).
//
// Memory bandwidth target: > 40 GB/s (DDR5 ceiling ~50).
// Compute: negligible at M=1.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>

#define CK(call) do { hipError_t e = call; if (e != hipSuccess) { printf("HIP err %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while(0)

using clk = std::chrono::high_resolution_clock;


// V2 GEMV: each block handles ROWS_PER_BLOCK output rows.
// Threads in a warp cooperate on one row.
// BLOCK_THREADS = 256 = 8 warps.
// Each warp = 1 row, so ROWS_PER_BLOCK = 8.
//
// Per row: K elements, K/8 elements per thread (uint4 loads).
// Each thread does (K/8) dot products of length 8.

template<int ROWS_PER_BLOCK, int K>
__global__ __launch_bounds__(256) void gemv_bf16_v2(
    const __bf16* __restrict__ W,  // [N, K]
    const float* __restrict__ act,  // [K]
    float* __restrict__ out,        // [N]
    int N
) {
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;  // 8
    static_assert(WARPS_PER_BLOCK == ROWS_PER_BLOCK, "1 warp per row");
    constexpr int VEC = 8;  // bf16 per uint4
    constexpr int K_VEC = K / VEC;  // total vector ops per row
    constexpr int K_PER_THREAD = K_VEC / 32;  // each thread does this many

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= N) return;

    // Cooperative load of activation into LDS (256 threads × 8 bf16 = 2048 = K)
    // For dynamic K, fall through to scalar load.
    __shared__ float s_act[K];
    {
        // Vectorized load: each thread loads 8 bf16 (= 16 bytes) as uint4
        // Total: 256 threads × 8 bf16 = 2048 elements. Perfect.
        const uint4* act_v = reinterpret_cast<const uint4*>(act);
        uint4* s_act_v = reinterpret_cast<uint4*>(s_act);
        s_act_v[threadIdx.x] = act_v[threadIdx.x];
    }
    __syncthreads();

    const __bf16* w_row = W + row * K;

    // Load 8 bf16 at a time, multiply with 8 act values
    float sum = 0.0f;
    #pragma unroll
    for (int v = 0; v < K_VEC / 32; v++) {
        int vec_idx = v * 32 + lane;
        // Load 8 bf16 weights (uint4)
        uint4 w_vec = *reinterpret_cast<const uint4*>(w_row + vec_idx * 8);
        // Load 8 bf16 acts (uint4)
        uint4 a_vec = *reinterpret_cast<const uint4*>(s_act + vec_idx * 8);
        // Reinterpret as __bf16[8]
        __bf16* w8 = reinterpret_cast<__bf16*>(&w_vec);
        __bf16* a8 = reinterpret_cast<__bf16*>(&a_vec);
        // 4 packed multiplies via built-in dot2
        // pair (w8[0],a8[0]), (w8[1],a8[1])
        // pair (w8[2],a8[2]), (w8[3],a8[3])
        // ...
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            sum += (float)w8[p*2] * (float)a8[p*2] + (float)w8[p*2+1] * (float)a8[p*2+1];
        }
    }

    // Warp reduction
    sum += __shfl_xor(sum, 16);
    sum += __shfl_xor(sum, 8);
    sum += __shfl_xor(sum, 4);
    sum += __shfl_xor(sum, 2);
    sum += __shfl_xor(sum, 1);

    if (lane == 0) out[row] = sum;
}

// Generic version for dynamic K (not template specialized)
__global__ __launch_bounds__(256) void gemv_bf16_v2_dyn(
    const __bf16* __restrict__ W,  // [N, K]
    const float* __restrict__ act,  // [K]
    float* __restrict__ out,        // [N]
    int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    extern __shared__ float s_act_dyn[];
    int K_PER_THREAD = (K + 255) / 256;
    int K_VEC_PER_THREAD = (K_PER_THREAD + 3) / 4;
    const uint4* act_v = reinterpret_cast<const uint4*>(act);
    uint4* s_act_v = reinterpret_cast<uint4*>(s_act_dyn);
    for (int i = 0; i < K_VEC_PER_THREAD; i++) {
        int idx = threadIdx.x + i * 256;
        if (idx * 4 < K) {
            s_act_v[idx] = act_v[idx];
        }
    }
    __syncthreads();

    const __bf16* w_row = W + row * K;

    int K_VEC = K / 8;
    float sum = 0.0f;
    for (int v = lane; v < K_VEC; v += 32) {
        uint4 w_vec = *reinterpret_cast<const uint4*>(w_row + v * 8);
        uint4 a_vec = *reinterpret_cast<const uint4*>(s_act_dyn + v * 8);
        __bf16* w8 = reinterpret_cast<__bf16*>(&w_vec);
        __bf16* a8 = reinterpret_cast<__bf16*>(&a_vec);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            sum += (float)w8[p*2] * (float)a8[p*2] + (float)w8[p*2+1] * (float)a8[p*2+1];
        }
    }

    sum += __shfl_xor(sum, 16);
    sum += __shfl_xor(sum, 8);
    sum += __shfl_xor(sum, 4);
    sum += __shfl_xor(sum, 2);
    sum += __shfl_xor(sum, 1);

    if (lane == 0) out[row] = sum;
}

double bench_v2(const char* name, int N, int K, void* d_W, void* d_act, void* d_out, hipStream_t stream) {
    int blocks = (N + 7) / 8;
    int smem = K * sizeof(float);
    
    void* args[] = {(void*)&d_W, (void*)&d_act, (void*)&d_out, (void*)&N, (void*)&K};
    
    for (int i = 0; i < 5; i++) {
        hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks, 1, 1), dim3(256, 1, 1),
                        args, smem, stream);
    }
    hipDeviceSynchronize();
    
    const int ITER = 30;
    auto t0 = clk::now();
    for (int i = 0; i < ITER; i++) {
        hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks, 1, 1), dim3(256, 1, 1),
                        args, smem, stream);
    }
    hipDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / ITER;
    
    double bytes = (double)N * K * sizeof(__bf16) + (double)K * sizeof(float);
    double bw = bytes / (ms * 1e-3) / 1e9;
    printf("  %-40s  %7.3f ms  (%6.1f GB/s)\n", name, ms, bw);
    return ms;
}

int main() {
    hipStream_t stream;
    hipStreamCreate(&stream);

    printf("=== iGPU GEMV v2 (RDNA 3 packed BF16 math, gfx1103) ===\n\n");
    
    // Setup shapes
    struct Shape { const char* name; int N, K; };
    Shape shapes[] = {
        {"qkv_proj (1,2048 -> 1,5120)",       5120,  2048},
        {"o_proj (1,4096 -> 1,2048)",          2048,  4096},
        {"moe_gate (1,2048 -> 1,256)",         256,   2048},
        {"shared_gate/up (1,2048 -> 1,512)",   512,   2048},
        {"shared_down (1,512 -> 1,2048)",      2048,  512},
        {"switch_gate/up (1,2048 -> 1,512)",   512,   2048},
        {"switch_down (1,512 -> 1,2048)",      2048,  512},
        {"lm_head (1,2048 -> 1,248320)",       248320, 2048},
    };
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    double total_step = 0;
    for (auto& s : shapes) {
        __bf16* d_W;
        float* d_act;
        float* d_out;
        
        hipMalloc(&d_W, s.N * s.K * sizeof(__bf16));
        hipMalloc(&d_act, s.K * sizeof(float));
        hipMalloc(&d_out, s.N * sizeof(float));
        
        std::vector<__bf16> h_W(s.N * s.K);
        std::vector<float> h_act(s.K);
        for (auto& v : h_act) v = dist(rng);
        for (auto& v : h_W) v = (__bf16)dist(rng);
        
        hipMemcpy(d_W, h_W.data(), s.N * s.K * sizeof(__bf16), hipMemcpyHostToDevice);
        hipMemcpy(d_act, h_act.data(), s.K * sizeof(float), hipMemcpyHostToDevice);
        
        double ms = bench_v2(s.name, s.N, s.K, d_W, d_act, d_out, stream);
        if (strstr(s.name, "lm_head") == nullptr &&
            strstr(s.name, "shared_gate") == nullptr &&
            strstr(s.name, "shared_down") == nullptr &&
            strstr(s.name, "switch_") == nullptr) {
            total_step += ms;
        }
        if (strstr(s.name, "switch")) {
            total_step += ms * 8;
        }
        if (strstr(s.name, "shared")) {
            total_step += ms;
        }
        
        hipFree(d_W); hipFree(d_act); hipFree(d_out);
    }
    
    printf("\n=== Aggregated MTP step ===\n");
    printf("Total per draft step: %.2f ms (8x switch experts aggregated)\n", total_step);
    printf("K=3 draft: %.2f ms (naive, weights re-read each call)\n", total_step * 3);
    
    hipStreamDestroy(stream);
    return 0;
}
