
import importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# 小规模 + 小权重值 -> 数值可控
E, H, I = 2, 128, 64
NB = H // 16
rng = np.random.default_rng(0)
# 稀疏小值权重（避免溢出）
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
ex = mod.IgpuMoeExecutor(cache, top_k=1, activation="silu", apply_router_weight_on_input=False, service=mod.IgpuGemvService(), max_tokens=1, device=None)
hidden = torch.randn(1, H, dtype=torch.float32) * 0.5
w = torch.tensor([1.0], dtype=torch.float32)
ids = torch.tensor([1], dtype=torch.int64)
out = ex.decode(0, hidden, w, ids).numpy()  # (1, H)

# numpy 参考（同一近似数学）：只算 expert 1 的 gate_up -> silu -> down
K_E = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
x = hidden.numpy()[0]
xq, xasb = mod._quantize_w4a8(x, H)
gu_p = cache.bank_sources["gate_up_packed"][0][1].astype(np.int32)
gu_s = cache.bank_sources["gate_up_scale"][0][1].astype(np.int32)
ac = xq.reshape(NB, 16).astype(np.int32)
pk = gu_p.reshape(2*I, NB, 8)
lo = K_E[pk & 0x0F]; hi = K_E[(pk >> 4) & 0x0F]
wsum = (lo * ac[None,:,:8]).sum(axis=2) + (hi * ac[None,:,8:]).sum(axis=2)  # (2I, NB)
gu = (wsum.astype(np.float64) * 0.01 * gu_s + xasb[None,:]).sum(axis=1) * 0.25 * 0.5
gu = gu.astype(np.float32)
act = gu[:I] / (1 + np.exp(-gu[:I])) * gu[I:]
aq, aasb = mod._quantize_w4a8(act.astype(np.float32), I)
dn_p = cache.bank_sources["down_packed"][0][1].astype(np.int32)
dn_s = cache.bank_sources["down_scale"][0][1].astype(np.int32)
NB2 = I // 16
ac2 = aq.reshape(NB2, 16).astype(np.int32)
pk2 = dn_p.reshape(H, NB2, 8)
lo2 = K_E[pk2 & 0x0F]; hi2 = K_E[(pk2 >> 4) & 0x0F]
wsum2 = (lo2 * ac2[None,:,:8]).sum(axis=2) + (hi2 * ac2[None,:,8:]).sum(axis=2)
ref = (wsum2.astype(np.float64) * 0.01 * dn_s + aasb[None,:]).sum(axis=1) * 0.25 * 0.5
ref = ref.astype(np.float32)
err = float(np.abs(out[0] - ref).max())
rel = err / (float(np.abs(ref).max()) + 1)
print("decode maxerr:", err, "rel:", rel, "OK" if rel < 1e-2 else "MISMATCH")
print("out[0..5]:", out[0][:6], "ref[0..5]:", ref[:6])
