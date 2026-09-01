#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){printf("HIP ERR %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e_)); exit(1);} } while(0)


__global__ void read_seq(const float4* __restrict__ src, size_t n4, float* out) {
    float4 acc = {0.f,0.f,0.f,0.f};
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
    if (lane == 0) sh[wid] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (blockDim.x >> 5); i++) t += sh[i];
        atomicAdd(out, t);
    }
}

__global__ void read_blocks(const unsigned char* __restrict__ base, const unsigned long long* __restrict__ offs,
                            unsigned blockBytes, float* out) {
    const unsigned char* p = base + offs[blockIdx.x];
    unsigned n4 = blockBytes >> 4;
    float4 acc = {0.f,0.f,0.f,0.f};
    for (unsigned i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = *(const float4*)(p + (((size_t)i) << 4));
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
    if (lane == 0) sh[wid] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (blockDim.x >> 5); i++) t += sh[i];
        atomicAdd(out, t);
    }
}

static float time_kernel(const char* tag, int iters, size_t bytes, void (*launch)(void*), void* ctx) {
    hipEvent_t a, b;
    CHK(hipEventCreate(&a)); CHK(hipEventCreate(&b));
    launch(ctx); CHK(hipDeviceSynchronize());          // warm
    float best = 1e30f, acc_ms = 0.f;
    for (int i = 0; i < iters; i++) {
        CHK(hipEventRecord(a));
        launch(ctx);
        CHK(hipEventRecord(b));
        CHK(hipEventSynchronize(b));
        float ms; CHK(hipEventElapsedTime(&ms, a, b));
        best = std::min(best, ms); acc_ms += ms;
    }
    float avg = acc_ms / iters;
    if (tag) printf("%s: avg %.3f ms best %.3f ms -> %.1f GB/s (best)\n", tag, avg, best, bytes / (best / 1e3) / 1e9);
    CHK(hipEventDestroy(a)); CHK(hipEventDestroy(b));
    return best;
}

struct SeqCtx { const float4* src; size_t n4; float* out; };
static void launch_seq(void* p) {
    SeqCtx* c = (SeqCtx*)p;
    read_seq<<<24, 1024>>>(c->src, c->n4, c->out);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) { printf("launch err: %s\n", hipGetErrorString(le)); exit(1); }
}

struct BlkCtx { const unsigned char* base; const unsigned long long* offs; unsigned bb; float* out; };
static void launch_blk(void* p) {
    BlkCtx* c = (BlkCtx*)p;
    read_blocks<<<320, 1024>>>(c->base, c->offs, c->bb, c->out);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) { printf("launch err: %s\n", hipGetErrorString(le)); exit(1); }
}

int main() {
    CHK(hipInit(0));
    int n = 0; CHK(hipGetDeviceCount(&n));
    if (n <= 0) { printf("no HIP device\n"); return 1; }
    CHK(hipSetDevice(0));
    hipDeviceProp_t pr; CHK(hipGetDeviceProperties(&pr, 0));
    printf("dev: %s CU=%d totalMem=%.1fGB mappedHost=%d gcn=%s\n", pr.name, pr.multiProcessorCount,
           pr.totalGlobalMem / 1e9, (int)pr.canMapHostMemory, pr.gcnArchName);

    const size_t GB1 = 1ull << 30;
    float chk = 0.f;

    // 1) device-local sequential read
    {
        float* dev; CHK(hipMalloc(&dev, GB1)); CHK(hipMemset(dev, 1, GB1));
        float* dout; CHK(hipMalloc(&dout, 4)); CHK(hipMemset(dout, 0, 4));
        SeqCtx c{(const float4*)dev, GB1 >> 4, dout};
        time_kernel("A device hipMalloc 1GB seq-read", 10, GB1, launch_seq, &c);
        CHK(hipMemcpy(&chk, dout, 4, hipMemcpyDeviceToHost));
        printf("   checksum=%.3e (must be >0)\n", chk);
        CHK(hipFree(dev)); CHK(hipFree(dout));
    }
    // 2) pinned host zero-copy read (the shared-pool path)
    {
        float* host; CHK(hipHostMalloc(&host, GB1, hipHostMallocDefault)); CHK(hipMemset(host, 1, GB1));
        float* dout; CHK(hipMalloc(&dout, 4)); CHK(hipMemset(dout, 0, 4));
        SeqCtx c{(const float4*)host, GB1 >> 4, dout};
        time_kernel("B pinned hipHostMalloc 1GB seq-read (zero-copy)", 10, GB1, launch_seq, &c);
        CHK(hipMemcpy(&chk, dout, 4, hipMemcpyDeviceToHost));
        printf("   checksum=%.3e (must be >0)\n", chk);
        CHK(hipHostFree(host)); CHK(hipFree(dout));
    }
    // 3) MoE token pattern: 320 blocks x 1.61MB from a 2GB pinned pool
    {
        const size_t POOL = 2ull << 30;
        const unsigned BB = 1690000u & ~15u;
        float* pool; CHK(hipHostMalloc(&pool, POOL, hipHostMallocDefault)); CHK(hipMemset(pool, 1, POOL));
        unsigned long long* offs; CHK(hipHostMalloc((void**)&offs, 320 * 8, hipHostMallocDefault));
        float* dout; CHK(hipMalloc(&dout, 4));
        srand(42);
        double worst = 1e30; int tokens = 20;
        std::vector<float> times(tokens);
        for (int t = 0; t < tokens; t++) {
            for (int i = 0; i < 320; i++) {
                unsigned long long r = ((unsigned long long)rand() << 17) ^ rand();
                offs[i] = r % ((POOL - BB - 16) >> 4) << 4;
            }
            BlkCtx c{(const unsigned char*)pool, offs, BB, dout};
            char tagbuf[64];
            sprintf(tagbuf, "C moe-token %d", t);
            float ms = time_kernel(t == 0 ? "C moe-token read 515MB (320 x 1.61MB)" : nullptr, 5, (size_t)BB * 320, launch_blk, &c);
            times[t] = ms; worst = std::min(worst, (double)ms);
        }
        std::sort(times.begin(), times.end());
        float med = times[tokens / 2];
        printf("   token-read: best %.3f ms (%.1f GB/s) median %.3f ms (%.1f GB/s)\n",
               worst, (double)BB * 320 / (worst / 1e3) / 1e9, med, (double)BB * 320 / (med / 1e3) / 1e9);
        printf("   => MoE-only decode ceiling: %.1f t/s (median)\n", 1000.0 / med);
        CHK(hipHostFree(pool)); CHK(hipHostFree(offs)); CHK(hipFree(dout));
    }
    printf("DONE\n");
    return 0;
}
