
import importlib.util, numpy as np, sys, time
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
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
import torch
hidden = torch.randn(1, H, dtype=torch.float32)
w = torch.tensor([0.5, 0.5], dtype=torch.float32)
ids = torch.tensor([1, 3], dtype=torch.int64)
t0 = time.perf_counter()
out = ex.decode(0, hidden, w, ids)
dt = time.perf_counter() - t0
print("H I:", ex.H, ex.I)
print("decode out:", tuple(out.shape), out.dtype, "finite:", bool(torch.isfinite(out).all()), "norm:", round(float(out.norm()), 3), "time: %.2fs" % dt)
# 验证投影正确性（单次 _project vs numpy）
e = 1
xq, xasb = mod._quantize_w4a8(hidden.numpy()[0], H)
gu_p = ex._banks["gate_up_packed"][0][e].astype(np.int32)
gu_s = ex._banks["gate_up_scale"][0][e]
gu_g = ex._banks["gate_up_global"][0][e].astype(np.float32)
gu = ex._project(gu_p, gu_s, gu_g, xq, xasb, 2*I, H)
# numpy 参考（紧凑向量化）
NB = H // 16
K_E = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
pk = gu_p.reshape(2*I, NB, 8)
lo = K_E[pk & 0x0F]; hi = K_E[(pk >> 4) & 0x0F]
ac = xq.reshape(NB, 16).astype(np.int32)
wsum = (lo * ac[None, :, :8]).sum(axis=2) + (hi * ac[None, :, 8:]).sum(axis=2)  # (2I, NB)
ref = (wsum.astype(np.float64) * 0.01 * (gu_s.astype(np.uint32).reshape(2*I, NB) & 0xFF) + xasb[None, :]).sum(axis=1)
ref = ref.astype(np.float32) * 0.25 * gu_g
err = float(np.abs(gu - ref).max())
print("project maxerr:", err, "OK" if err < 0.01 * (float(np.abs(ref).max()) + 1) else "MISMATCH")
