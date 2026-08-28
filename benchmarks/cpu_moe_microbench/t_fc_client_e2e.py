"""端到端: 新 IgpuFcSticky (FC_LOAD/FC_CALL) + torch 桥接, 真实 mtp.fc 全量 2048 行."""
import sys, time
sys.path.insert(0, r"E:\FreeToken\python")
import numpy as np
import json, safetensors.torch

bench = r"E:\FreeToken\benchmarks\cpu_moe_microbench"
mdl = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(mdl + "/model.safetensors.index.json") as f: idx = json.load(f)
st = safetensors.torch.load_file(mdl + "/" + idx["weight_map"]["mtp.fc.weight"])
fc_w = st["mtp.fc.weight"].cpu().numpy()
fc_b = st["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = st["mtp.fc.scales"].cpu().numpy().astype(np.float32)
M, K = fc_w.shape[0], 4096

from freetoken.kernel.igpu_fc import IgpuFcSticky

rng = np.random.default_rng(42)
act = rng.standard_normal(K).astype(np.float32)

# numpy 参考
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)
w = fc_w[:, :, None]
shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
vals = kE2M1[((w >> shifts) & 0xF).astype(np.int64)].reshape(M, K)
prod = (vals * act[None, :]).reshape(M, K // 32, 32)
ref = ((prod.sum(axis=2) + fc_b) * fc_s).sum(axis=1).astype(np.float32)

# ===== numpy 接口 =====
print("===== numpy 接口 =====")
t0 = time.time()
fc = IgpuFcSticky(fc_w, K, scales_f32=fc_s, biases_f32=fc_b)
print(f"  构造+FC_LOAD: {time.time()-t0:.2f}s")
out = fc(act)
d = np.abs(out - ref)
print(f"  numpy call: max|diff|={d.max():.3e} NaN={int(np.isnan(out).sum())}  {'PASS' if d.max()<1e-3 else 'FAIL'}")
print(f"  out[:4]={out[:4]}")

# ===== torch 桥接 (CPU) =====
print("\n===== torch 桥接 (CPU) =====")
import torch
tf = fc.torch()
x = torch.from_numpy(act)
t_out = tf(x)
print(f"  输入: {tuple(x.shape)} {x.dtype} → 输出: {tuple(t_out.shape)} {t_out.dtype}")
d2 = np.abs(t_out.squeeze(0).numpy() - ref)
print(f"  max|diff|={d2.max():.3e}  {'PASS' if d2.max()<1e-3 else 'FAIL'}")

# ===== torch 桥接 (CUDA, 若可用) =====
if torch.cuda.is_available():
    print("\n===== torch 桥接 (CUDA) =====")
    xc = x.to(torch.bfloat16).cuda()
    t_out2 = tf(xc)
    print(f"  输入: {tuple(xc.shape)} {xc.dtype}@cuda → 输出: {tuple(t_out2.shape)} {t_out2.dtype}@{t_out2.device}")
    d3 = np.abs(t_out2.squeeze(0).float().cpu().numpy() - ref)
    print(f"  max|diff|={d3.max():.3e}  {'PASS' if d3.max()<1e-2 else 'FAIL'}  (bf16 输入有量化损失)")

    # CUDA 上的延迟 (含 H2D/D2H + IPC)
    ts = []
    for _ in range(20):
        t0 = time.time(); tf(xc); ts.append((time.time()-t0)*1000)
    print(f"  torch(cuda,bf16) 20 次: median={np.median(ts):.2f}ms p90={np.percentile(ts,90):.2f}ms")
else:
    print("\n(CUDA 不可用, 跳过)")

# ===== numpy 延迟 =====
ts = []
for _ in range(30):
    t0 = time.time(); fc(act); ts.append((time.time()-t0)*1000)
print(f"\nnumpy 接口 30 次: median={np.median(ts):.2f}ms p90={np.percentile(ts,90):.2f}ms")

fc.close()
print("\n=== E2E DONE ===")
