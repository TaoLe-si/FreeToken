// V3 GEMV: simple, correct
// Each block handles ROWS_PER_BLOCK output rows.
// Each warp handles 1 row.
// 32 threads cooperate: thread t handles K indices starting at t, t+32, t+64, ...
//
// No LDS for activation - just read from global memory each time.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdlib>

using clk = std::chrono::high_resolution_clock;

__global__ __launch_bounds__(256) void gemv_v3(
    const __bf16* __restrict__ W, const float* __restrict__ a,
    float* __restrict__ out, int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;
    
    const __bf16* w_row = W + row * K;
    
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) {
        sum += (float)w_row[k] * a[k];
    }
    
    // Warp reduction
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
    
    if (lane == 0) out[row] = sum;
}

// CPU reference
void cpu_ref(const __bf16* W, const float* a, float* out, int N, int K) {
    for (int n = 0; n < N; n++) {
        float s = 0;
        for (int k = 0; k < K; k++) s += (float)W[n*K+k] * a[k];
        out[n] = s;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s N K\n", argv[0]); return 1; }
    int N = atoi(argv[1]); int K = atoi(argv[2]);
    
    std::vector<__bf16> h_W(N * K);
    std::vector<float> h_a(K);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : h_W) v = (__bf16)dist(rng);
    for (auto& v : h_a) v = dist(rng);
    
    __bf16* d_W; float* d_a; float* d_out;
    hipMalloc(&d_W, N*K*sizeof(__bf16));
    hipMalloc(&d_a, K*sizeof(float));
    hipMalloc(&d_out, N*sizeof(float));
    hipMemcpy(d_W, h_W.data(), N*K*sizeof(__bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_a, h_a.data(), K*sizeof(float), hipMemcpyHostToDevice);
    
    int blocks = (N + 7) / 8;
    void* args[] = {(void*)&d_W, (void*)&d_a, (void*)&d_out, (void*)&N, (void*)&K};
    for (int i = 0; i < 5; i++) {
        hipLaunchKernel((void*)gemv_v3, dim3(blocks, 1, 1), dim3(256, 1, 1), args, 0, 0);
    }
    hipDeviceSynchronize();
    
    hipLaunchKernel((void*)gemv_v3, dim3(blocks, 1, 1), dim3(256, 1, 1), args, 0, 0);
    hipDeviceSynchronize();
    
    std::vector<float> gpu(N);
    hipMemcpy(gpu.data(), d_out, N*sizeof(float), hipMemcpyDeviceToHost);
    
    std::vector<float> cpu(N);
    cpu_ref(h_W.data(), h_a.data(), cpu.data(), N, K);
    
    int err = 0;
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::abs(gpu[i] - cpu[i]);
        max_err = std::max(max_err, diff);
        if (std::isnan(gpu[i]) || diff > 1.0f) {
            if (err < 5) printf("  out[%d] = %.4f (gpu) vs %.4f (cpu), diff=%.4f\n", i, gpu[i], cpu[i], diff);
            err++;
        }
    }
    printf("=== V3 GEMV (no LDS, just FMA) ===\n");
    printf("N=%d, K=%d\n", N, K);
    printf("Errors > 1.0: %d / %d\n", err, N);
    printf("Max error: %.4f\n", max_err);
    printf("First 5: gpu=%.2f %.2f %.2f %.2f %.2f cpu=%.2f %.2f %.2f %.2f %.2f\n",
        gpu[0], gpu[1], gpu[2], gpu[3], gpu[4],
        cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
    
    // Speed
    auto t0 = clk::now();
    for (int i = 0; i < 30; i++) {
        hipLaunchKernel((void*)gemv_v3, dim3(blocks, 1, 1), dim3(256, 1, 1), args, 0, 0);
    }
    hipDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 30;
    double bytes = (double)N * K * 2 + (double)K * 4;
    printf("Speed: %.3f ms (%.1f GB/s)\n", ms, bytes/(ms*1e-3)/1e9);
    return 0;
}
