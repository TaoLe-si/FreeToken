// iGPU MTP benchmark - measure key operation times
// Compile: hipcc --offload-arch=gfx1103 -O3 bench_mtp_igpu.cpp -o bench_mtp_igpu.exe
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

#define CK(call) do { hipError_t e = call; if (e != hipSuccess) { printf("HIP err %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while(0)

using clk = std::chrono::high_resolution_clock;

int main() {
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);
    
    // MTP head shapes
    const int H = 2048;        // hidden
    const int V = 248320;      // vocab (lm_head)
    const int HEAD_DIM = 256;
    const int NUM_Q = 16;
    const int NUM_KV = 2;
    const int MOE_INTER = 512;
    const int NUM_EXPERTS = 256;
    
    // Test shapes (M=1, N=output, K=input, all bf16)
    struct Shape { const char* name; int M, N, K; };
    std::vector<Shape> shapes = {
        {"qkv_proj (1, 2048 -> 1, 2048*5)",     1, H * (NUM_Q*2 + NUM_KV*2), H},
        {"o_proj (1, 2048 -> 1, 2048)",          1, H, H * NUM_Q},
        {"moe_gate (1, 2048 -> 1, 256)",         1, NUM_EXPERTS, H},
        {"shared_gate (1, 2048 -> 1, 512)",      1, MOE_INTER, H},
        {"shared_up (1, 2048 -> 1, 512)",        1, MOE_INTER, H},
        {"shared_down (1, 512 -> 1, 2048)",      1, H, MOE_INTER},
        {"switch_expert (1, 2048 -> 1, 512)",    1, MOE_INTER, H},  // x8 experts
        {"switch_down (1, 512 -> 1, 2048)",      1, H, MOE_INTER},
        {"lm_head (1, 2048 -> 1, 248320)",       1, V, H},
    };
    
    printf("=== iGPU GEMM benchmark (M=1, bf16, gfx1103) ===\n");
    
    for (auto& s : shapes) {
        size_t a_bytes = s.M * s.K * 2;
        size_t b_bytes = s.K * s.N * 2;
        size_t c_bytes = s.M * s.N * 2;
        
        void *d_a, *d_b, *d_c;
        CK(hipMalloc(&d_a, a_bytes));
        CK(hipMalloc(&d_b, b_bytes));
        CK(hipMalloc(&d_c, c_bytes));
        
        // Random init
        std::vector<float> h_a(a_bytes / 2);
        std::vector<float> h_b(b_bytes / 2);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto& v : h_a) v = dist(rng);
        for (auto& v : h_b) v = dist(rng);
        // Use bf16 directly (skip conversion for simplicity)
        CK(hipMemcpy(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice));
        
        // Setup matmul descriptor (RowMajor)
        hipblasLtMatmulDesc_t desc;
        hipblasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
        
        // Use bf16 compute
        hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
        
        // A: [M, K] row major = [K, M] col major
        hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, s.K, s.M, s.K);
        // B: [K, N] row major = [N, K] col major
        hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_16BF, s.N, s.K, s.N);
        // C: [M, N] row major
        hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, s.N, s.M, s.N);
        
        hipblasLtMatmulDescCreate(&desc, compute_type, HIP_R_32F);
        hipblasOperation_t opT = HIPBLAS_OP_T;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        
        // Heuristic-based algo selection
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t workspace_size = 32 * 1024 * 1024;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_size, sizeof(workspace_size));
        void* d_workspace;
        CK(hipMalloc(&d_workspace, workspace_size));
        
        hipblasLtMatmulHeuristicResult_t heur;
        int returned = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc, Adesc, Bdesc, Cdesc, Cdesc, pref, 1, &heur, &returned);
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            hipblasLtMatmul(handle, desc, &alpha, d_a, Adesc, d_b, Bdesc, &beta, d_c, Cdesc, d_c, Cdesc,
                            returned > 0 ? &heur.algo : nullptr, d_workspace, workspace_size, 0);
        }
        hipDeviceSynchronize();
        
        // Measure
        const int N = 50;
        auto t0 = clk::now();
        for (int i = 0; i < N; i++) {
            hipblasLtMatmul(handle, desc, &alpha, d_a, Adesc, d_b, Bdesc, &beta, d_c, Cdesc, d_c, Cdesc,
                            returned > 0 ? &heur.algo : nullptr, d_workspace, workspace_size, 0);
        }
        hipDeviceSynchronize();
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / N;
        
        printf("  %-40s  %7.3f ms\n", s.name, ms);
        
        hipFree(d_a); hipFree(d_b); hipFree(d_c);
        hipblasLtMatrixLayoutDestroy(Adesc);
        hipblasLtMatrixLayoutDestroy(Bdesc);
        hipblasLtMatrixLayoutDestroy(Cdesc);
        hipblasLtMatmulDescDestroy(desc);
    }
    
    hipblasLtDestroy(handle);
    return 0;
}
