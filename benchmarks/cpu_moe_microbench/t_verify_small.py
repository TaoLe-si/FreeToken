
import importlib.util, numpy as np, time, sys
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
E, H, I = 4, 512, 256
rng = np.random.default_rng(3)
svc = mod.IgpuGemvService()
t0 = time.perf_counter()
svc.start()
print("started in %.2fs, adapter: %s" % (time.perf_counter()-t0, svc.adapter_desc), flush=True)
# 小 M 调用
xq, xasb = mod._quantize_w4a8(rng.standard_normal(H).astype(np.float32), H)
pk = rng.integers(0, 256, (2*I, H//2), dtype=np.uint8)
sc = rng.integers(0, 128, (2*I, H//16), dtype=np.uint32)
gb = rng.random(2*I).astype(np.float32)
t0 = time.perf_counter()
out = svc.gemv(pk, sc, xq, xasb, gb, 2*I, H)
print("gemv M=512 K=512 in %.3fs out[0..3]=%s" % (time.perf_counter()-t0, out[:4]), flush=True)
svc.close()
print("done", flush=True)
