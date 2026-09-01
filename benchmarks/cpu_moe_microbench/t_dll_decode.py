
import importlib.util, numpy as np, torch, time, os
os.chdir(r"E:\FreeToken\benchmarks\cpu_moe_microbench")
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
t0 = time.perf_counter()
ok, msg = mod.igpu_available()
print("igpu_available:", ok, msg, "(%.2fs)" % (time.perf_counter()-t0))
# decode 走 DLL（IgpuMoeExecutor 会用 IgpuGemvService -> DLL 优先）
E, H, I = 2, 128, 64
rng = np.random.default_rng(0)
class FakeBanks:
    def __init__(s):
        s.quant_format = "nvfp4"; s.num_layers = 1; s.num_experts = E
        s.bank_sources = {
            "gate_up_packed": [rng.integers(0, 16, (E, 2*I, H//2), dtype=np.uint8)],
            "gate_up_scale": [rng.integers(1, 8, (E, 2*I, H//16), dtype=np.uint8)],
            "gate_up_global": [np.full((E, 2*I), 0.5, dtype=np.float16)],
            "down_packed": [rng.integers(0, 16, (E, H, I//2), dtype=np.uint8)],
            "down_scale": [rng.integers(1, 8, (E, H, I//16), dtype=np.uint8)],
            "down_global": [np.full((E, H), 0.5, dtype=np.float16)],
        }
cache = FakeBanks()
ex = mod.IgpuMoeExecutor(cache, top_k=2, activation="silu", apply_router_weight_on_input=False, service=mod.IgpuGemvService(), max_tokens=1, device=None)
hidden = torch.randn(2, H, dtype=torch.float32) * 0.5
w = torch.tensor([0.6, 0.4, 0.7, 0.3], dtype=torch.float32)
ids = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
t0 = time.perf_counter()
out = ex.decode(0, hidden, w, ids)
dt = time.perf_counter() - t0
print("decode via DLL: %.2fs out=%s finite=%s" % (dt, tuple(out.shape), bool(torch.isfinite(out).all())))
print("ALL OK")
