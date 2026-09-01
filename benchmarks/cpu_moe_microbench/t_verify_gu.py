
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
pk = cache.bank_sources["gate_up_packed"][0][1]  # (2I, H//2) uint8
sc = cache.bank_sources["gate_up_scale"][0][1]   # (2I, H//16) uint8
gb = cache.bank_sources["gate_up_global"][0][1].astype(np.float32)
gu_svc = ex._project(pk, sc, gb, xq, xasb, 2*I, H)  # 服务结果 (2I,)

# numpy 参考 gate_up
K_E = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
ac = xq.reshape(NB, 16).astype(np.int32)
pki = pk.astype(np.int32).reshape(2*I, NB, 8)
lo = K_E[pki & 0x0F]; hi = K_E[(pki >> 4) & 0x0F]
wsum = (lo * ac[None,:,:8]).sum(axis=2) + (hi * ac[None,:,8:]).sum(axis=2)  # (2I, NB)
sci = sc.astype(np.int64)
ref_gu = (wsum.astype(np.float64) * 0.01 * sci + xasb[None,:]).sum(axis=1) * 0.25 * gb.astype(np.float64)
gu_err = float(np.abs(gu_svc - ref_gu).max())
print("gate_up maxerr:", gu_err)
# 分解：检查 wsum 是否正确（服务内部 wsum 不可见；但 scale/asb 路径可以验证）
print("xq[:16]:", xq[:16], "xasb[:3]:", xasb[:3])
# 服务结果的 vs 参考结构性对比
print("gu_svc[0..3]:", gu_svc[:4], "ref_gu[0..3]:", ref_gu[:4].astype(np.float32))
