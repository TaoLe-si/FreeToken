#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s | SMs %d | clock %.0f MHz | mem %.1f GB | PCIe bus %02x:%02x.%x | integrated=%d | L2 %.1f MB\n",
         p.name, p.multiProcessorCount, p.clockRate / 1000.0, p.totalGlobalMem / 1e9,
         p.pciBusID, p.pciDeviceID, p.pciDomainID, p.integrated, p.l2CacheSize / 1e6);
  const size_t maxb = 512ull << 20;
  char* h = nullptr; char* d = nullptr;
  CK(cudaMallocHost(&h, maxb));
  CK(cudaMalloc(&d, maxb));
  for (size_t i = 0; i < maxb; i += 4096) h[i] = 1;
  cudaStream_t s; CK(cudaStreamCreate(&s));
  const size_t sizes[] = {8ull << 20, 64ull << 20, 256ull << 20, 512ull << 20};
  for (size_t sz : sizes) {
    int iters = (int)((512ull << 20) / sz);
    if (iters < 3) iters = 3;
    if (iters > 64) iters = 64;
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0, s));
    for (int i = 0; i < iters; ++i) CK(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, s));
    CK(cudaEventRecord(t1, s)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("H2D %4zu MB x %2d: %7.1f GB/s\n", sz >> 20, iters, (double)sz * iters / ms / 1e6);
    CK(cudaEventRecord(t0, s));
    for (int i = 0; i < iters; ++i) CK(cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, s));
    CK(cudaEventRecord(t1, s)); CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("D2H %4zu MB x %2d: %7.1f GB/s\n", sz >> 20, iters, (double)sz * iters / ms / 1e6);
  }
  CK(cudaFreeHost(h)); CK(cudaFree(d));
  return 0;
}
