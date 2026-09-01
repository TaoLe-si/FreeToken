
import importlib.util, numpy as np, os, time, ctypes
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
svc = mod.IgpuGemvService(); svc._load_dll()
KE = mod._K_E2M1X2
def test(M, K, seed):
    NB = K // 16
    rng = np.random.default_rng(seed)
    packed = np.ascontiguousarray(rng.integers(0, 256, M*NB*8, dtype=np.uint8))
    scl = np.ascontiguousarray(rng.integers(0, 128, M*NB, dtype=np.uint32))
    act = np.ascontiguousarray(rng.integers(-127, 128, NB*16, dtype=np.int32))
    asb = np.ascontiguousarray(rng.random(NB).astype(np.float32))
    gbl = np.ascontiguousarray(rng.random(M).astype(np.float32))
    out = svc.gemv(packed, scl, act, asb, gbl, M, K)
    pk = packed.reshape(M, NB, 8).astype(np.int32)
    sc = (scl.reshape(M, NB) & 0xFF).astype(np.float64)
    ac = act.reshape(NB, 16).astype(np.int32)
    low = KE[pk & 0x0F]; high = KE[(pk >> 4) & 0x0F]
    wsum = (low.astype(np.int64) * ac[None,:,:8]).sum(axis=2) + (high.astype(np.int64) * ac[None,:,8:]).sum(axis=2)
    ref = (wsum.astype(np.float64) * 0.01 * sc + asb[None,:]).sum(axis=1) * 0.25 * gbl
    err = float(np.abs(out - ref).max())
    rel = err / (float(np.abs(ref).max()) + 1e-6)
    print(f"M={M} K={K} seed={seed}: maxerr={err:.4g} rel={rel:.3g} {'OK' if rel < 1e-3 else 'FAIL'}")
test(1024, 4096, 7)
test(4096, 4096, 7)
test(4096, 4096, 42)
test(2048, 2048, 1)
test(512, 512, 2)
svc.close()
