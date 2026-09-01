// iGPU MTP head forward (single draft step) benchmark
// Tests: embed + RMSNorm + concat + FC + RMSNorm + QKV proj + RoPE + SDPA + o_proj + RMSNorm + MoE + RMSNorm + lm_head + argmax
//
// All on iGPU (gfx1103, BF16).
//
// Goal: measure if iGPU can replace dGPU for MTP draft.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>

#define CK(call) do { hipError_t e = call; if (e != hipSuccess) { printf("HIP err %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while(0)
using clk = std::chrono::high_resolution_clock;

// === MTP head dimensions ===
constexpr int H = 2048;
constexpr int HEAD_DIM = 256;
constexpr int NUM_Q = 16;
constexpr int NUM_KV = 2;
constexpr int QKV_OUT = (NUM_Q + NUM_KV * 2) * HEAD_DIM;  // 5120
constexpr int MOE_INTER = 512;
constexpr int VOCAB = 248320;

// === GEMV V2 kernel (packed bf16) ===
__global__ __launch_bounds__(256) void gemv_bf16_v2_dyn(
    const __bf16* __restrict__ W, const float* __restrict__ act,
    float* __restrict__ out, int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    extern __shared__ float s_act_dyn[];
    const uint4* act_v = reinterpret_cast<const uint4*>(act);
    uint4* s_act_v = reinterpret_cast<uint4*>(s_act_dyn);
    s_act_v[threadIdx.x] = act_v[threadIdx.x];
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

void gemv(const __bf16* W, const float* act, float* out, int N, int K, hipStream_t s) {
    int blocks = (N + 7) / 8;
    void* args[] = {(void*)&W, (void*)&act, (void*)&out, (void*)&N, (void*)&K};
    hipLaunchKernel((void*)gemv_bf16_v2_dyn, dim3(blocks), dim3(256), args, K * sizeof(float), s);
}


// Argmax: out_idx = argmax(x)
__global__ void argmax_kernel(const float* __restrict__ x, int* __restrict__ out_idx, int N) {
    extern __shared__ float smax[];
    extern __shared__ int sidx[];
    int tid = threadIdx.x;
    float local_max = -1e30f;
    int local_idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    // Reduce: combine local_max and local_idx
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_xor(local_max, offset);
        int other_i = __shfl_xor(local_idx, offset);
        if (other_v > local_max || (other_v == local_max && other_i < local_idx)) {
            local_max = other_v; local_idx = other_i;
        }
    }
    int wid = tid / 32, lane = tid & 31;
    if (lane == 0) { smax[wid] = local_max; sidx[wid] = local_idx; }
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < blockDim.x / 32) ? smax[lane] : -1e30f;
        local_idx = (lane < blockDim.x / 32) ? sidx[lane] : 0;
        for (int offset = 4; offset > 0; offset >>= 1) {
            float other_v = __shfl_xor(local_max, offset);
            int other_i = __shfl_xor(local_idx, offset);
            if (other_v > local_max || (other_v == local_max && other_i < local_idx)) {
                local_max = other_v; local_idx = other_i;
            }
        }
        if (lane == 0) out_idx[0] = local_idx;
    }
}

// Concat: out = [a; b]
__global__ void concat_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int HA, int HB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HA + HB;
    if (i < HA) out[i] = a[i];
    else if (i < total) out[i] = b[i - HA];
}

// Residual: out = a + b
__global__ void residual_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

int main() {
    printf("=== iGPU MTP head step benchmark (gfx1103) ===\n\n");
    
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    // Setup
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Allocate weights
    __bf16 *qkv_w, *o_w, *lm_head_w;
    hipMalloc(&qkv_w, QKV_OUT * H * sizeof(__bf16));
    hipMalloc(&o_w, H * NUM_Q * HEAD_DIM * sizeof(__bf16));
    hipMalloc(&lm_head_w, VOCAB * H * sizeof(__bf16));
    
    std::vector<__bf16> hbuf(QKV_OUT * H);
    for (auto& v : hbuf) v = (__bf16)dist(rng);
    hipMemcpy(qkv_w, hbuf.data(), hbuf.size() * sizeof(__bf16), hipMemcpyHostToDevice);
    
    hbuf.resize(H * NUM_Q * HEAD_DIM);
    for (auto& v : hbuf) v = (__bf16)dist(rng);
    hipMemcpy(o_w, hbuf.data(), hbuf.size() * sizeof(__bf16), hipMemcpyHostToDevice);
    
    hbuf.resize(VOCAB * H);
    for (auto& v : hbuf) v = (__bf16)dist(rng);
    hipMemcpy(lm_head_w, hbuf.data(), hbuf.size() * sizeof(__bf16), hipMemcpyHostToDevice);
    
    // Activation
    float* d_act;
    hipMalloc(&d_act, H * sizeof(float));
    std::vector<float> h_act(H);
    for (auto& v : h_act) v = dist(rng);
    hipMemcpy(d_act, h_act.data(), H * sizeof(float), hipMemcpyHostToDevice);
    
    // Outputs
    float *d_qkv, *d_attn_out;
    hipMalloc(&d_qkv, QKV_OUT * sizeof(float));
    hipMalloc(&d_attn_out, H * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        gemv(qkv_w, d_act, d_qkv, QKV_OUT, H, stream);
        gemv(o_w, d_qkv, d_attn_out, H, NUM_Q * HEAD_DIM, stream);
        gemv(lm_head_w, d_act, d_attn_out, VOCAB, H, stream);
    }
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    
    // Time individual kernels
    printf("Per-kernel timing (single call):\n");
    
    auto t0 = clk::now();
    for (int i = 0; i < 10; i++) gemv(qkv_w, d_act, d_qkv, QKV_OUT, H, stream);
    hipDeviceSynchronize();
    double t_qkv = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 30;
    printf("  qkv_proj:    %.3f ms\n", t_qkv);
    
    t0 = clk::now();
    for (int i = 0; i < 30; i++) gemv(o_w, d_qkv, d_attn_out, H, NUM_Q * HEAD_DIM, stream);
    hipDeviceSynchronize();
    double t_oproj = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 30;
    printf("  o_proj:      %.3f ms\n", t_oproj);
    
    t0 = clk::now();
    for (int i = 0; i < 30; i++) gemv(lm_head_w, d_act, d_attn_out, VOCAB, H, stream);
    hipDeviceSynchronize();
    double t_lmhead = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 30;
    printf("  lm_head:     %.3f ms\n", t_lmhead);
    
    // Simulate K=3 draft (just GEMV time, weights re-read each call)
    t0 = clk::now();
    for (int k = 0; k < 3; k++) {
        gemv(qkv_w, d_act, d_qkv, QKV_OUT, H, stream);
        gemv(o_w, d_qkv, d_attn_out, H, NUM_Q * HEAD_DIM, stream);
        gemv(lm_head_w, d_act, d_attn_out, VOCAB, H, stream);
    }
    hipDeviceSynchronize();
    double t_k3 = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    printf("\nK=3 (3 linear layers only): %.3f ms\n", t_k3);
    printf("K=3 throughput: %.1f t/s\n", 3.0 / t_k3 * 1000);
    
    hipStreamDestroy(stream);
    return 0;
}
