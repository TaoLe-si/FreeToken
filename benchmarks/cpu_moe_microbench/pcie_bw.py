from cuda.bindings import runtime as cudart
import ctypes, time

def ok(r):
    assert r[0] == cudart.cudaError_t.cudaSuccess, r

ok(cudart.cudaSetDevice(0))
err, props = cudart.cudaGetDeviceProperties(0)
ok((err,))
name = props.name.decode() if isinstance(props.name, bytes) else props.name
print(f"GPU: {name} | SMs {props.multiProcessorCount} | mem {props.totalGlobalMem/1e9:.1f} GB | pciBus {props.pciBusID}")

MAXB = 512 << 20
err, h = cudart.cudaHostAlloc(MAXB, 0)
ok((err,))
err, d = cudart.cudaMalloc(MAXB)
ok((err,))
ctypes.memset(ctypes.c_void_p(h), 1, MAXB)
err, stream = cudart.cudaStreamCreate()
ok((err,))

for sz in [8 << 20, 64 << 20, 256 << 20, 512 << 20]:
    iters = max(3, min(64, (512 << 20) // sz))
    cudart.cudaDeviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ok(cudart.cudaMemcpyAsync(d, h, sz, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    cudart.cudaStreamSynchronize(stream)
    dt = time.perf_counter() - t0
    print(f"H2D {sz>>20:4d} MB x{iters:2d}: {sz*iters/dt/1e9:7.1f} GB/s")
    t0 = time.perf_counter()
    for _ in range(iters):
        ok(cudart.cudaMemcpyAsync(h, d, sz, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    cudart.cudaStreamSynchronize(stream)
    dt = time.perf_counter() - t0
    print(f"D2H {sz>>20:4d} MB x{iters:2d}: {sz*iters/dt/1e9:7.1f} GB/s")
