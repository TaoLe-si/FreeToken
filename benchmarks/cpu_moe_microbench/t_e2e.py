
import importlib.util, numpy as np, torch, time
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ok, msg = mod.igpu_available()
print("igpu_available:", ok, msg)
if not ok: raise SystemExit(1)
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
print("decode B=2 K=2:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()), "norm:", round(float(out.norm()), 2), "time: %.2fs" % dt)
print("ALL OK")
