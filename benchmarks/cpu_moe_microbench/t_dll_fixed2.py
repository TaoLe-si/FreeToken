
import importlib.util, numpy as np, os, time
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
svc = mod.IgpuGemvService(); ok = svc._load_dll()
print("dll loaded:", ok, "handle:", hex(svc._dll_handle or 0))
KE = mod._K_E2M1X2
def test(M, K, seed):
    NB = K // 16
    rng = np.random.default_rng(seed)
    packed = np.ascontiguousarray(rng.integers(0, 256, M*NB*8, dtype=np.uint8))
    scl = np.ascontiguousarray(rng.integers(0, 128, M*NB, dtype=np.uint32))
    act = np.ascontiguousarray(rng.integers(-127, 128, NB*16, dtype=np.int32))
    asb = np.ascontiguousarray(rng.random(NB).astype(np.float32))
    gbl = np.ascontiguousarray(rng.random(M).astype(np.float32))
    t0 = time.perf_counter()
    out = svc.gemv(packed, scl, act, asb, gbl, M, K)
    dt = time.perf_counter() - t0
    pk = packed.reshape(M, NB, 8).astype(np.int32)
    sc = (scl.reshape(M, NB) & 0xFF).astype(np.float64)
    ac = act.reshape(NB, 16).astype(np.int32)
    low = KE[pk & 0x0F]; high = KE[(pk >> 4) & 0x0F]
    wsum = (low.astype(np.int64) * ac[None,:,:8]).sum(axis=2) + (high.astype(np.int64) * ac[None,:,8:]).sum(axis=2)
    ref = (wsum.astype(np.float64) * 0.01 * sc + asb[None,:]).sum(axis=1) * 0.25 * gbl
    err = float(np.abs(out - ref).max())
    rel = err / (float(np.abs(ref).max()) + 1e-6)
    print(f"M={M} K={K}: {dt*1000:.2f}ms maxerr={err:.4g} rel={rel:.3g} {'OK' if rel < 1e-3 else 'FAIL'}")
test(1024, 4096, 7)
test(4096, 4096, 7)
test(4096, 4096, 42)
test(2048, 2048, 1)
# 连续带宽
t0 = time.perf_counter()
for _ in range(20):
    svc.gemv(svc._gemv_via_dll.args[0] if False else None, None, None, None, None, 4096, 4096)  # skip
# 实际带宽测试
M, K = 4096, 4096; NB = K // 16
rng = np.random.default_rng(0)
pk = rng.integers(0, 256, M*NB*8, dtype=np.uint8); sc = rng.integers(0, 128, M*NB, dtype=np.uint32)
ac = rng.integers(-127, 128, NB*16, dtype=np.int32); ab = rng.random(NB).astype(np.float32); gb = rng.random(M).astype(np.float32)
svc.gemv(pk, sc, ac, ab, gb, M, K)  # warm
t0 = time.perf_counter()
for _ in range(20):
    svc.gemv(pk, sc, ac, ab, gb, M, K)
dt = (time.perf_counter() - t0) / 20
wbytes = M*NB*8 + M*NB*4 + NB*16*4 + NB*4 + M*4
print(f"bandwidth: {dt*1000:.2f}ms -> {wbytes/dt/1e9:.1f} GB/s")
svc.close()
print("DLL OK")
