
import importlib.util, numpy as np, os, time
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
svc = mod.IgpuGemvService(); svc._load_dll()
M, K = 4096, 4096; NB = K // 16
rng = np.random.default_rng(0)
pk = np.ascontiguousarray(rng.integers(0, 256, M*NB*8, dtype=np.uint8))
sc = np.ascontiguousarray(rng.integers(0, 128, M*NB, dtype=np.uint32))
ac = np.ascontiguousarray(rng.integers(-127, 128, NB*16, dtype=np.int32))
ab = np.ascontiguousarray(rng.random(NB).astype(np.float32))
gb = np.ascontiguousarray(rng.random(M).astype(np.float32))
svc.gemv(pk, sc, ac, ab, gb, M, K)  # warm
t0 = time.perf_counter()
for _ in range(20):
    svc.gemv(pk, sc, ac, ab, gb, M, K)
dt = (time.perf_counter() - t0) / 20
wbytes = M*NB*8 + M*NB*4 + NB*16*4 + NB*4 + M*4
print(f"bandwidth: {dt*1000:.2f}ms -> {wbytes/dt/1e9:.1f} GB/s")
svc.close()
print("DLL BW OK")
