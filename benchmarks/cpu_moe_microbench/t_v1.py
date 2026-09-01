
import importlib.util, numpy as np, time
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("module %.2fs" % (time.perf_counter()-t0), flush=True)
E, H, I = 4, 512, 256
rng = np.random.default_rng(3)
class FakeBanks:
    def __init__(s):
        s.quant_format = "nvfp4"; s.num_layers = 1; s.num_experts = E
        s.bank_sources = {
            "gate_up_packed": [rng.integers(0, 256, (E, 2*I, H//2), dtype=np.uint8)],
            "gate_up_scale": [rng.integers(0, 128, (E, 2*I, H//16), dtype=np.uint8)],
            "gate_up_global": [rng.random((E, 2*I)).astype(np.float16)],
            "down_packed": [rng.integers(0, 256, (E, H, I//2), dtype=np.uint8)],
            "down_scale": [rng.integers(0, 128, (E, H, I//16), dtype=np.uint8)],
            "down_global": [rng.random((E, H)).astype(np.float16)],
        }
cache = FakeBanks()
ex = mod.IgpuMoeExecutor(cache, top_k=2, activation="silu", apply_router_weight_on_input=False, service=mod.IgpuGemvService(), max_tokens=1, device=None)
print("executor %.2fs H=%d I=%d" % (time.perf_counter()-t0, ex.H, ex.I), flush=True)
# 单次投影（无 torch）
x = rng.standard_normal(H).astype(np.float32)
xq, xasb = mod._quantize_w4a8(x, H)
pk = ex._banks["gate_up_packed"][0][1]
sc = ex._banks["gate_up_scale"][0][1]
gb = ex._banks["gate_up_global"][0][1].astype(np.float32)
t1 = time.perf_counter()
gu = ex._project(pk, sc, gb, xq, xasb, 2*I, H)
print("project %.3fs out[0..3]=%s" % (time.perf_counter()-t1, gu[:4]), flush=True)
ex.service.close()
print("PASS", flush=True)
