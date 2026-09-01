
import importlib.util, numpy as np, time, os
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
svc = mod.IgpuGemvService()
print("dll default path:", svc._default_dll_path())
ok = svc._load_dll()
print("dll load:", ok, "handle:", hex(svc._dll_handle or 0))
M, K = 4096, 4096
NB = K // 16
rng = np.random.default_rng(42)
packed = rng.integers(0, 256, M*NB*8, dtype=np.uint8)
scl = rng.integers(0, 128, M*NB, dtype=np.uint32)
act = rng.integers(-127, 128, NB*16, dtype=np.int32)
asb = rng.random(NB).astype(np.float32).copy()
gbl = rng.random(M).astype(np.float32).copy()
t0 = time.perf_counter()
out = svc.gemv(packed, scl, act, asb, gbl, M, K)
dt = time.perf_counter() - t0
print("first gemv: %.2f ms" % (dt*1000))
# 连续调用带宽
t0 = time.perf_counter()
for _ in range(20):
    out = svc.gemv(packed, scl, act, asb, gbl, M, K)
dt = (time.perf_counter() - t0)/20
wbytes = M*NB*8 + M*NB*4 + NB*16*4 + NB*4 + M*4
print(f"dll-inline gemv: {dt*1000:.2f} ms -> {wbytes/dt/1e9:.1f} GB/s")
# numpy 参考（同数学，随机数据）
KE = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
pk = packed.astype(np.int32).reshape(M, NB, 8)
lo = KE[pk & 0x0F]; hi = KE[(pk >> 4) & 0x0F]
ac = act.reshape(NB, 16).astype(np.int32)
wsum = (lo.astype(np.int64) * ac[None,:,:8]).sum(axis=2) + (hi.astype(np.int64) * ac[None,:,8:]).sum(axis=2)
ref = (wsum * 0.01 * scl.astype(np.int64).reshape(M, NB) + asb[None,:]*0.25).sum(axis=1).astype(np.float32) * gbl * 0.25
# 服务数学: out = (wsum*0.01*sb + asb[b])*0.25*gbl[r]
# 修正参考: (wsum*0.01*scl + asb)*0.25*gbl (asb 加到每块再乘0.25？看 HLSL: acc += (wsum*0.01*sb + asb[b])... gs=0.25, out=acc*gs*gbl)
ref2 = ((wsum.astype(np.float64) * 0.01 * scl.reshape(M, NB).astype(np.float64) + asb[None,:]).sum(axis=1) * 0.25 * gbl).astype(np.float32)
err = float(np.abs(out - ref2).max())
rel = err / (float(np.abs(ref2).max()) + 1)
print("vs numpy ref maxerr:", err, "rel:", rel, "OK" if rel < 1e-3 else "MISMATCH")
svc.close()
print("done");
