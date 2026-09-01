
import ctypes, numpy as np, time, os
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\d3d12_gemv.dll")
P = ctypes.c_void_p
dll.igpu_create.restype = P
dll.igpu_gemv.restype = ctypes.c_int
dll.igpu_gemv.argtypes = [P, ctypes.c_int, ctypes.c_int, P, P, P, P, P, P]
dll.igpu_destroy.restype = None
dll.igpu_destroy.argtypes = [P]
dll.igpu_errmsg.restype = ctypes.c_char_p
dll.igpu_errmsg.argtypes = [P]
h = dll.igpu_create()
print("handle:", hex(h or 0))
if not h:
    print("err:", dll.igpu_errmsg(None)); raise SystemExit(1)
M, K = 4096, 4096
NB = K // 16
rng = np.random.default_rng(42)
packed = np.ascontiguousarray(rng.integers(0, 256, M*NB*8, dtype=np.uint8))
scl = np.ascontiguousarray(rng.integers(0, 128, M*NB, dtype=np.uint32))
act = np.ascontiguousarray(rng.integers(-127, 128, NB*16, dtype=np.int32))
asb = np.ascontiguousarray(rng.random(NB).astype(np.float32))
gbl = np.ascontiguousarray(rng.random(M).astype(np.float32))
out = np.zeros(M, dtype=np.float32)
pk_p = packed.ctypes.data_as(P); sc_p = scl.ctypes.data_as(P)
ac_p = act.ctypes.data_as(P); as_p = asb.ctypes.data_as(P)
gb_p = gbl.ctypes.data_as(P); ou_p = out.ctypes.data_as(P)
rc = dll.igpu_gemv(h, M, K, pk_p, sc_p, ac_p, as_p, gb_p, ou_p)
print("warm rc:", rc)
t0 = time.perf_counter()
for _ in range(20):
    rc = dll.igpu_gemv(h, M, K, pk_p, sc_p, ac_p, as_p, gb_p, ou_p)
dt = (time.perf_counter() - t0) / 20
wbytes = M*NB*8 + M*NB*4 + NB*16*4 + NB*4 + M*4
print(f"dll gemv: {dt*1000:.2f} ms -> {wbytes/dt/1e9:.1f} GB/s, rc={rc}")
print("out[0..3]:", out[:4])
# 与 stdio 服务对比正确性（调用服务）
dll.igpu_destroy(h)
print("DLL OK")
