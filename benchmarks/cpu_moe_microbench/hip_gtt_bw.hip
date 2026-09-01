
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void read_sum(const float* __restrict__ p, size_t n, float* __restrict__ out) {
    float acc = 0.f;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) acc += p[i];
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){ printf("  HIP ERR %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e)); return 1; } } while(0)

int main() {
    CK(hipInit(0));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("totalGlobalMem: %.1f GB\n", prop.totalGlobalMem / 1073741824.0);

    // 1) 容量阶梯: 尝试分配越来越大
    std::vector<void*> chunks;
    const size_t GB = 1024ull*1024*1024;
    double cap_gb = 0;
    // 每次尝试 1GB 块, 最多 20GB
    for (int i = 0; i < 20; i++) {
        void* p = nullptr;
        hipError_t rc = hipMalloc(&p, GB);
        if (rc != hipSuccess) { printf("hipMalloc 1GB #%d failed: %s\n", i, hipGetErrorString(rc)); break; }
        chunks.push_back(p);
        cap_gb += 1.0;
    }
    printf("TOTAL hipMalloc capacity: %.0f GB\n", cap_gb);

    if (cap_gb >= 1) {
        // 2) kernel 读带宽: 用第一个 1GB 块
        float* p = (float*)chunks[0];
        size_t n = GB / 4;
        CK(hipMemset(p, 1, GB));
        float* d_out; CK(hipMalloc(&d_out, 4));
        hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
        // 预热
        hipLaunchKernelGGL(read_sum, dim3(1024), dim3(256), 0, 0, p, n, d_out);
        CK(hipStreamSynchronize(0));
        float ms_total = 0;
        for (int it = 0; it < 5; it++) {
            CK(hipEventRecord(t0));
            hipLaunchKernelGGL(read_sum, dim3(1024), dim3(256), 0, 0, p, n, d_out);
            CK(hipEventRecord(t1));
            CK(hipEventSynchronize(t1));
            float ms; CK(hipEventElapsedTime(&ms, t0, t1));
            ms_total += ms;
        }
        float avg = ms_total / 5;
        printf("kernel GTT read: %.1f GB in %.2f ms = %.1f GB/s\n", 1.0, avg, 1.0 / (avg / 1000.0));

        // 3) H2D memcpy 带宽 (pinned -> device)
        float* h; CK(hipHostMalloc(&h, GB, hipHostMallocMapped));
        hipEventRecord(t0);
        for (int it = 0; it < 5; it++) CK(hipMemcpy(p, h, GB, hipMemcpyHostToDevice));
        hipEventRecord(t1);
        CK(hipEventSynchronize(t1));
        float cms; CK(hipEventElapsedTime(&cms, t0, t1));
        printf("H2D memcpy: %.1f GB x5 in %.2f ms = %.1f GB/s\n", 1.0, cms/5, 5.0 / (cms / 1000.0));

        // 4) 小拷贝开销: 8 x 1.69MB (单专家块)
        hipEventRecord(t0);
        for (int e = 0; e < 8; e++) CK(hipMemcpy(p, h, 1690000, hipMemcpyHostToDevice));
        hipEventRecord(t1);
        CK(hipEventSynchronize(t1));
        float sms; CK(hipEventElapsedTime(&sms, t0, t1));
        printf("8 x 1.69MB copies: %.3f ms total (%.4f ms each)\n", sms, sms / 8.0);
        CK(hipFree(d_out)); CK(hipHostFree(h));
    }
    for (void* p : chunks) hipFree(p);
    printf("DONE\n");
    return 0;
}
