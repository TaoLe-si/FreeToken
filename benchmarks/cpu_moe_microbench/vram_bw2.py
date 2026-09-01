# VRAM residency limit probe: D2D bw vs allocation size
from cuda.bindings import runtime as cudart
import time

def ok(r):
    assert r[0] == cudart.cudaError_t.cudaSuccess, r

ok(cudart.cudaSetDevice(0))
err, s = cudart.cudaStreamCreate()
ok((err,))

def d2d(gb1, gb2, iters=5):
    SIZE1 = gb1 << 30; SIZE2 = gb2 << 30
    err, d1 = cudart.cudaMalloc(SIZE1); ok((err,))
    err, d2 = cudart.cudaMalloc(SIZE2); ok((err,))
    ok(cudart.cudaMemcpyAsync(d2, d1, min(SIZE1, SIZE2), cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, s))
    cudart.cudaStreamSynchronize(s)
    sz = min(SIZE1, SIZE2)
    t0 = time.perf_counter()
    for _ in range(iters):
        ok(cudart.cudaMemcpyAsync(d2, d1, sz, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, s))
    cudart.cudaStreamSynchronize(s)
    dt = time.perf_counter() - t0
    print(f"alloc {gb1:2d}GB + {gb2:2d}GB, copy {sz>>30:2d}GB x{iters}: {sz*iters/dt/1e9:7.1f} GB/s")
    cudart.cudaFree(d1); cudart.cudaFree(d2)

d2d(2, 2)
d2d(3, 3)
d2d(4, 2)
d2d(4, 3)
# 报告可用显存
err, f = cudart.cudaMemGetInfo()
print("free/total:", f[1]/1e9, f[2]/1e9, "GB")
