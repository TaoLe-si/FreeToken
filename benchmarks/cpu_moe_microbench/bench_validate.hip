// V2 GEMV numerical validation
// Compares iGPU V2 GEMV vs CPU reference for correctness, then measures speed.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

using clk = std::chrono::high_resolution_clock;

// === V2 GEMV kernel (from bench_gemv_v2.hip) ===
__global__ __launch_bounds__(256) void gemv_bf16_v2_dyn(
    const __bf16* __restrict__ W, const float* __restrict__ act,
    float* __restrict__ out, int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    extern __shared__ float s_act_dyn[];
    // Cooperative load: K floats distributed across 256 threads
    // Each thread loads (K + 255) / 256 floats via uint4 (4 floats per load)
    int K_PER_THREAD = (K + 255) / 256;  // ceil
    int K_VEC_PER_THREAD = (K_PER_THREAD + 3) / 4;  // uint4 count
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

// CPU reference: out[n] = sum_k (float)W[n,k] * (float)act[k]
void cpu_reference(const __bf16* W, const float* act, float* out, int N, int K) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)W[n * K + k] * act[k];
        }
        out[n] = sum;
    }
}

int main(int argc, char** argv) {
    int N = 1024;  // outputs
    int K = 2048;  // reduction
    
    if (argc >= 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }
    
    printf("=== iGPU V2 GEMV numerical validation (N=%d, K=%d) ===\n\n", N, K);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    // Allocate host
    std::vector<__bf16> h_W(N * K);
    std::vector<float> h_act(K);
    std::vector<float> h_out_gpu(N);
    std::vector<float> h_out_cpu(N);
    
    for (auto& v : h_act) v = dist(rng);
    for (auto& v : h_W) v = (__bf16)dist(rng);
    
    // Allocate device
    __bf16* d_W;
    float* d_act;
    float* d_out;
    hipMalloc(&d_W, N * K * sizeof(__bf16));
    hipMalloc(&d_act, K * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));
    hipMemcpy(d_W, h_W.data(), N * K * sizeof(__bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_act, h_act.data(), K * sizeof(float), hipMemcpyHostToDevice);
    
    // Run GPU
    int blocks = (N + 7) / 8;
    int smem = K * sizeof(float);
    printf("blocks=%d, smem=%d, N=%d, K=%d\n", blocks, smem, N, K);
    void* args[] = {(void*)&d_W, (void*)&d_act, (void*)&d_out, (void*)&N, (void*)&K};
    for (int i = 0; i < 5; i++) {
        hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks, 1, 1), dim3(256, 1, 1),
                        args, smem, 0);
    }
    hipError_t e = hipDeviceSynchronize();
    printf("warmup sync err=%d (%s)\n", e, hipGetErrorString(e));
    
    hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks, 1, 1), dim3(256, 1, 1),
                    args, smem, 0);
    hipDeviceSynchronize();
    hipMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost);
    
    // Run CPU
    auto t_cpu = clk::now();
    cpu_reference(h_W.data(), h_act.data(), h_out_cpu.data(), N, K);
    double cpu_ms = std::chrono::duration<double, std::milli>(clk::now() - t_cpu).count();
    
    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int n_big_err = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        float ref = std::abs(h_out_cpu[i]);
        float rel_err = (ref > 1e-6f) ? diff / ref : diff;
        max_abs_err = std::max(max_abs_err, diff);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (diff > 0.5f) n_big_err++;
    }
    
    printf("Numerical accuracy:\n");
    printf("  Max abs error: %.4f\n", max_abs_err);
    printf("  Max rel error: %.4f\n", max_rel_err);
    printf("  Outputs with abs err > 0.5: %d / %d\n", n_big_err, N);
    printf("  First 5 GPU outputs: ");
    for (int i = 0; i < 5 && i < N; i++) printf("%.2f ", h_out_gpu[i]);
    printf("\n  First 5 CPU outputs: ");
    for (int i = 0; i < 5 && i < N; i++) printf("%.2f ", h_out_cpu[i]);
    printf("\n\n");
    
    // Speed test
    const int ITER = 30;
    auto t0 = clk::now();
    for (int i = 0; i < ITER; i++) {
        hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks, 1, 1), dim3(256, 1, 1),
                        args, smem, 0);
    }
    hipDeviceSynchronize();
    double gpu_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / ITER;
    
    double bytes = (double)N * K * sizeof(__bf16) + (double)K * sizeof(float);
    printf("Speed:\n");
    printf("  GPU V2 GEMV: %.3f ms (%.1f GB/s)\n", gpu_ms, bytes / (gpu_ms * 1e-3) / 1e9);
    printf("  CPU reference: %.1f ms (%.2f x slower)\n", cpu_ms, cpu_ms / gpu_ms);
    
    // Compare to MTP-relevant sizes
    if (K == 2048) {
        // Estimate full MTP step time
        double qkv_ms = 0.604;  // 5120 outputs
        double o_proj_ms = 0.462;  // 2048 outputs
        double moe_gate_ms = 0.032;  // 256 outputs
        double shared_ms = 0.037 + 0.025;  // 512 + 2048 outputs
        double switch_ms = (0.021 + 0.024) * 8;  // 8 experts
        double lm_head_ms = 29.4;  // 248320 outputs
        double total_step = qkv_ms + o_proj_ms + moe_gate_ms + shared_ms + switch_ms + lm_head_ms;
        printf("\nFull MTP step projection (just linear layers on iGPU):\n");
        printf("  qkv:     %.2f ms\n", qkv_ms);
        printf("  o_proj:  %.2f ms\n", o_proj_ms);
        printf("  moe:     %.2f ms (gate + shared + 8 switch)\n", moe_gate_ms + shared_ms + switch_ms);
        printf("  lm_head: %.2f ms\n", lm_head_ms);
        printf("  Total:   %.2f ms per step\n", total_step);
        printf("  K=3:     %.2f ms\n", total_step * 3);
    }
    
    hipFree(d_W); hipFree(d_act); hipFree(d_out);
    return 0;
}
