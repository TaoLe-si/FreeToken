// iGPU GEMV benchmark with hand-written kernels (no hipBLASLt)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>

#define CK(call) do { hipError_t e = call; if (e != hipSuccess) { printf("HIP err %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while(0)

using clk = std::chrono::high_resolution_clock;

// Generic bf16 GEMV kernel: out[n] = sum_k act[k] * w[n,k]
// One block per output row, threads cooperate on K reduction.
extern "C" __global__ void gemv_bf16(
    const __bf16* __restrict__ W,  // [N, K]
    const float* __restrict__ act,          // [K]
    float* __restrict__ out,                // [N]
    int N, int K
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    if (row >= N) return;
    
    const __bf16* w_row = W + row * K;
    
    float sum = 0.0f;
    for (int k = tid; k < K; k += blockSize) {
        sum += act[k] * (float)(w_row[k]);
    }
    
    // Warp-level reduction (32 threads per warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }
    
    // Block-level reduction across warps
    __shared__ float warp_sums[8];  // up to 8 warps per block
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    
    if (warp_id == 0 && lane < 8) {
        sum = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor(sum, offset);
        }
        if (lane == 0) out[row] = sum;
    }
}

// Optimized: cache act in shared memory
extern "C" __global__ void gemv_bf16_smem(
    const __bf16* __restrict__ W,  // [N, K]
    const float* __restrict__ act,          // [K]
    float* __restrict__ out,                // [N]
    int N, int K
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    if (row >= N) return;
    
    extern __shared__ float s_act[];
    
    // Cooperative load of act into shared memory
    for (int k = tid; k < K; k += blockSize) {
        s_act[k] = act[k];
    }
    __syncthreads();
    
    const __bf16* w_row = W + row * K;
    
    float sum = 0.0f;
    for (int k = tid; k < K; k += blockSize) {
        sum += s_act[k] * (float)(w_row[k]);
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }
    
    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    
    if (warp_id == 0 && lane < 8) {
        sum = warp_sums[lane];
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor(sum, offset);
        }
        if (lane == 0) out[row] = sum;
    }
}

double bench(const char* name, int N, int K, bool use_smem) {
    __bf16* d_W;
    float* d_act;
    float* d_out;
    
    CK(hipMalloc(&d_W, N * K * sizeof(__bf16)));
    CK(hipMalloc(&d_act, K * sizeof(float)));
    CK(hipMalloc(&d_out, N * sizeof(float)));
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    std::vector<float> h_W(N * K);
    std::vector<float> h_act(K);
    for (auto& v : h_W) v = dist(rng);
    for (auto& v : h_act) v = dist(rng);
    
    // Convert to bf16
    std::vector<__bf16> h_W_bf(N * K);
    for (size_t i = 0; i < h_W.size(); i++) h_W_bf[i] = (__bf16)(h_W[i]);
    
    CK(hipMemcpy(d_W, h_W_bf.data(), N * K * sizeof(__bf16), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_act, h_act.data(), K * sizeof(float), hipMemcpyHostToDevice));
    
    int block_size = 256;
    int smem_bytes = use_smem ? K * sizeof(float) : 0;
    void* args[] = {(void*)&d_W, (void*)&d_act, (void*)&d_out, (void*)&N, (void*)&K};
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (use_smem) hipLaunchKernel((void*)gemv_bf16_smem, dim3(N, 1, 1), dim3(block_size, 1, 1), args, smem_bytes, 0);
        else hipLaunchKernel((void*)gemv_bf16, dim3(N, 1, 1), dim3(block_size, 1, 1), args, 0, 0);
    }
    hipDeviceSynchronize();
    
    // Measure
    const int ITER = 30;
    auto t0 = clk::now();
    for (int i = 0; i < ITER; i++) {
        if (use_smem) hipLaunchKernel((void*)gemv_bf16_smem, dim3(N, 1, 1), dim3(block_size, 1, 1), args, smem_bytes, 0);
        else hipLaunchKernel((void*)gemv_bf16, dim3(N, 1, 1), dim3(block_size, 1, 1), args, 0, 0);
    }
    hipDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / ITER;
    
    // Memory bandwidth check
    double bytes_read_w = (double)N * K * sizeof(__bf16);
    double bytes_read_a = (double)K * sizeof(float);
    double bytes_w = bytes_read_w + bytes_read_a;
    double bw_gbps = bytes_w / (ms * 1e-3) / 1e9;
    
    printf("  %-40s  %7.3f ms  (%6.1f GB/s)\n", name, ms, bw_gbps);
    
    hipFree(d_W); hipFree(d_act); hipFree(d_out);
    return ms;
}

int main() {
    printf("=== iGPU GEMV benchmark (gfx1103, bf16 weight, fp32 act) ===\n");
    printf("M=1 for all. Block size = 256.\n\n");
    printf("MTP layer shapes:\n");
    bench("qkv_proj (1,2048 -> 1,5120)",       5120, 2048, true);
    bench("o_proj (1,4096 -> 1,2048)",          2048, 4096, true);
    bench("moe_gate (1,2048 -> 1,256)",         256,  2048, false);
    bench("shared_gate/up (1,2048 -> 1,512)",   512,  2048, true);
    bench("shared_down (1,512 -> 1,2048)",      2048, 512,  false);
    bench("switch_gate/up (1,2048 -> 1,512)",   512,  2048, true);
    bench("switch_down (1,512 -> 1,2048)",      2048, 512,  false);
    bench("lm_head (1,2048 -> 1,248320)",       248320, 2048, true);
    
    printf("\nMTP step cost (1 token draft, all on iGPU):\n");
    double total = 0;
    total += bench("(dummy)  ", 100, 100, false);  // warmup
    total = 0;
    total += bench("qkv_proj", 5120, 2048, true);
    total += bench("o_proj", 2048, 4096, true);
    total += bench("moe_gate", 256, 2048, false);
    total += bench("shared_gate/up", 512, 2048, true);
    total += bench("shared_down", 2048, 512, false);
    // switch experts: 8 experts, each x3 GEMV
    total += bench("switch_expert (x1 of 8)", 512, 2048, true);
    total += 8 * bench("switch_down (x1 of 8)", 2048, 512, false);
    total += bench("lm_head", 248320, 2048, true);
    printf("\n  GEMV subtotal: %.2f ms (per draft step)\n", total);
    printf("  K=3 draft subtotal: %.2f ms\n", total * 3);
    
    return 0;
}
