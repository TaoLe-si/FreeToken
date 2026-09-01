
import importlib.util, numpy as np, time, sys
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("module loaded %.2fs" % (time.perf_counter()-t0), flush=True)
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
print("executor built %.2fs" % (time.perf_counter()-t0), flush=True)
import torch
print("torch imported %.2fs" % (time.perf_counter()-t0), flush=True)
hidden = torch.randn(1, H, dtype=torch.float32)
w = torch.tensor([0.5, 0.5], dtype=torch.float32)
ids = torch.tensor([1, 3], dtype=torch.int64)
out = ex.decode(0, hidden, w, ids)
print("decode done %.2fs out=%s norm=%.3f" % (time.perf_counter()-t0, tuple(out.shape), float(out.norm())), flush=True)
print("PASS", flush=True)
