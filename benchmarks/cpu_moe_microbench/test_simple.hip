// Minimal GEMV test - no LDS, no shfl
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

__global__ void simple_gemv(const __bf16* W, const float* a, float* out, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += (float)W[row * K + k] * a[k];
    }
    out[row] = sum;
}

int main() {
    int N = 64;
    int K = 256;
    std::vector<__bf16> h_W(N * K, (__bf16)0.1f);
    std::vector<float> h_a(K, 0.5f);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            h_W[i * K + k] = (__bf16)((i + k) % 7 * 0.1f);
        }
    }
    __bf16* d_W; float* d_a; float* d_out;
    hipMalloc(&d_W, N*K*sizeof(__bf16));
    hipMalloc(&d_a, K*sizeof(float));
    hipMalloc(&d_out, N*sizeof(float));
    hipMemcpy(d_W, h_W.data(), N*K*sizeof(__bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_a, h_a.data(), K*sizeof(float), hipMemcpyHostToDevice);
    
    simple_gemv<<<(N+63)/64, 64>>>(d_W, d_a, d_out, N, K);
    hipDeviceSynchronize();
    
    std::vector<float> h_out(N);
    hipMemcpy(h_out.data(), d_out, N*sizeof(float), hipMemcpyDeviceToHost);
    
    // CPU ref
    std::vector<float> ref(N);
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int k = 0; k < K; k++) sum += (float)h_W[i*K+k] * h_a[k];
        ref[i] = sum;
    }
    
    int err = 0;
    for (int i = 0; i < N; i++) {
        float diff = std::abs(h_out[i] - ref[i]);
        if (diff > 0.1f || std::isnan(h_out[i])) {
            if (err < 5) printf("out[%d] = %.4f (gpu) vs %.4f (cpu), diff=%.4f\n", i, h_out[i], ref[i], diff);
            err++;
        }
    }
    printf("Mismatches: %d / %d\n", err, N);
    if (err == 0) printf("First 5 outputs: %.2f %.2f %.2f %.2f %.2f\n", h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);
    return 0;
}
