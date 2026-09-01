
import importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("igpu_backend", r"E:\FreeToken\python\freetoken\moe\igpu_backend.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
E, H, I = 2, 128, 64
NB = H // 16
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
ex = mod.IgpuMoeExecutor(cache, top_k=1, activation="silu", apply_router_weight_on_input=False, service=mod.IgpuGemvService(), max_tokens=1, device=None)
hidden = torch.randn(1, H, dtype=torch.float32) * 0.5
x = hidden.numpy()[0]
xq, xasb = mod._quantize_w4a8(x, H)
# gate_up 服务（已验证一致）
gu = ex._project(cache.bank_sources["gate_up_packed"][0][1], cache.bank_sources["gate_up_scale"][0][1], cache.bank_sources["gate_up_global"][0][1].astype(np.float32), xq, xasb, 2*I, H)
# activation: silu（与 decode 的 _activation 相同数学）
with np.errstate(over="ignore"):
    act = gu[:I] / (1 + np.exp(-gu[:I])) * gu[I:]
aq, aasb = mod._quantize_w4a8(act.astype(np.float32), I)
# down 服务
dn_svc = ex._project(cache.bank_sources["down_packed"][0][1], cache.bank_sources["down_scale"][0][1], cache.bank_sources["down_global"][0][1].astype(np.float32), aq, aasb, H, I)
# numpy 参考 down
K_E = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
NB2 = I // 16
ac2 = aq.reshape(NB2, 16).astype(np.int32)
pki = cache.bank_sources["down_packed"][0][1].astype(np.int32).reshape(H, NB2, 8)
lo = K_E[pki & 0x0F]; hi = K_E[(pki >> 4) & 0x0F]
wsum = (lo * ac2[None,:,:8]).sum(axis=2) + (hi * ac2[None,:,8:]).sum(axis=2)
sci = cache.bank_sources["down_scale"][0][1].astype(np.int64)
ref_dn = (wsum.astype(np.float64) * 0.01 * sci + aasb[None,:]).sum(axis=1) * 0.25 * 0.5
print("down maxerr:", float(np.abs(dn_svc - ref_dn).max()))
print("dn_svc[0..5]:", dn_svc[:6])
print("ref_dn[0..5]:", ref_dn[:6].astype(np.float32))
