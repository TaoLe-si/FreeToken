# dGPU VRAM D2D bandwidth (RTX 4070 Laptop GDDR6)
from cuda.bindings import runtime as cudart
import time

def ok(r):
    assert r[0] == cudart.cudaError_t.cudaSuccess, r

ok(cudart.cudaSetDevice(0))
err, s = cudart.cudaStreamCreate()
ok((err,))

for gb in [1, 2, 4]:
    SIZE = gb << 30
    err, d1 = cudart.cudaMalloc(SIZE)
    ok((err,))
    err, d2 = cudart.cudaMalloc(SIZE)
    ok((err,))
    # warmup
    ok(cudart.cudaMemcpyAsync(d2, d1, SIZE, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, s))
    cudart.cudaStreamSynchronize(s)
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        ok(cudart.cudaMemcpyAsync(d2, d1, SIZE, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, s))
    cudart.cudaStreamSynchronize(s)
    dt = time.perf_counter() - t0
    print(f"D2D {gb:2d} GB x{iters}: {SIZE*iters/dt/1e9:7.1f} GB/s")
    cudart.cudaFree(d1); cudart.cudaFree(d2)
