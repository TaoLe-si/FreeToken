"""Realistic MTP K=3 tok/s projection with iGPU FC.

Uses:
  - Actual MTP head (loaded from real 35B safetensors)
  - iGPU FC server (0.215ms per fc call)
  - dGPU attn/MoE (PyTorch native, ~7-10ms each)
  - Cost model for main model verify (5ms small token prefill on real 35B)

The main model step cost on a 35B model at ~60 tok/s baseline = 16.7ms per step.
MTP K=3 = MTP head K=3 drafts + main model verify (5ms extra for K+1 tokens, ~5ms total).
"""
import sys, time, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[env] torch={torch.__version__} device={DEV}')

# 直接按文件路径加载 mtp 模块, 绕过 freetoken 包 __init__ (flashlib 依赖)
import importlib.util as _ilu
def _load_mod(name, path):
    _s = _ilu.spec_from_file_location(name, path)
    _m = _ilu.module_from_spec(_s)
    sys.modules[name] = _m
    _s.loader.exec_module(_m)
    return _m
_mtp = _load_mod('freetoken_mtp_real', r'E:\FreeToken\python\freetoken\models\qwen3_5_moe\mtp.py')
MtpHeadConfig = _mtp.MtpHeadConfig
load_mtp_head_from_safetensors = _mtp.load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky
import safetensors.torch
import numpy as np

# MTP head config (from real model)
cfg = MtpHeadConfig(
    hidden_size=2048, vocab_size=248320,
    num_experts=256, num_experts_per_tok=8,
    moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2,
    partial_rotary_factor=0.25, rms_norm_eps=1e-6,
)
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(DEV).to(torch.bfloat16)
igpu = IgpuFcClient()
class MyLM(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.proj = nn.Linear(H, V, bias=False).to(torch.bfloat16)
    def forward(self, x): return self.proj(x)
lm = MyLM(cfg.vocab_size, cfg.hidden_size).to(DEV)

# Load MTP head with iGPU FC
print('Loading MTP head with iGPU FC...')
t0 = time.time()
head = load_mtp_head_from_safetensors(
    r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP', cfg, embed, lm,
    igpu_fc=None, device=DEV, dtype=torch.bfloat16,
)
# Wire iGPU FC — full (2048, 512) weight matrix via FC_LOAD/FC_CALL sticky protocol
import numpy as np
fc_w = head._packed_mxfp4['fc.weight'].numpy().astype(np.uint32)      # (2048, 512)
fc_s_t = head._packed_mxfp4.get('fc.scales')
fc_b_t = head._packed_mxfp4.get('fc.biases')
ns_fc = fc_w.shape[1] * 8 // 32
fc_s = fc_s_t.numpy().astype(np.float32) if fc_s_t is not None else np.zeros((fc_w.shape[0], ns_fc), dtype=np.float32)
fc_b = fc_b_t.numpy().astype(np.float32) if fc_b_t is not None else np.zeros((fc_w.shape[0], ns_fc), dtype=np.float32)
K_fc = fc_w.shape[1] * 8
print(f'  iGPU FC: weight={fc_w.shape} K={K_fc} scales={fc_s.shape} biases={fc_b.shape}')
igpu_fc_sticky = IgpuFcSticky(fc_w, K_fc, scales_f32=fc_s, biases_f32=fc_b)
igpu.close()   # close stateless client; sticky owns its own server process
head.igpu_fc = igpu_fc_sticky.torch()
head.eval()
print(f'  Loaded in {time.time()-t0:.1f}s')

# ===== 数值验证: iGPU FC 路径 vs torch NVFP4 参考路径 =====
print('\n===== Numerical verification =====')
_kE2M1 = torch.tensor([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=torch.float32)
_fc_w = torch.from_numpy(fc_w.copy())          # (2048, 512) uint32
_fc_s = torch.from_numpy(fc_s.copy())          # (2048, 128)
_fc_b = torch.from_numpy(fc_b.copy())          # (2048, 128)
M_fc, K_fc2 = _fc_w.shape[0], _fc_w.shape[1] * 8
_shifts = (torch.arange(8, dtype=torch.int32) * 4).view(1, 1, 8)
def _ref_fc(cat_flat):
    # cat_flat: (4096,) fp32 -> (1, 2048) fp32  (NVFP4 公式, torch 参考)
    x = cat_flat.detach().to(torch.float32)
    w = _fc_w[:, :, None].to(torch.int64)
    nibs = (w >> _shifts) & 0xF               # (M, 512, 8)
    vals = _kE2M1[nibs.reshape(-1)].reshape(M_fc, K_fc2)  # (M, 4096)
    prod = (vals * x[None, :]).reshape(M_fc, -1, 32)
    out = ((prod.sum(dim=2) + _fc_b) * _fc_s).sum(dim=1)  # (M,)
    return out.unsqueeze(0)

torch.manual_seed(7)
prev_token = torch.tensor([12345], device=DEV, dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device=DEV, dtype=torch.bfloat16)
# 用同一输入跑两条路径, 比较最终 logits
head.igpu_fc = igpu_fc_sticky.torch()
logits_igpu = head(prev_token, prev_hidden)
head.igpu_fc = _ref_fc
logits_ref = head(prev_token, prev_hidden)
head.igpu_fc = igpu_fc_sticky.torch()   # 恢复
l1 = logits_igpu.detach().to(torch.float32)
l2 = logits_ref.detach().to(torch.float32)
d = (l1 - l2).abs()
rel = d / (l2.abs() + 1e-6)
print(f'  logits shape: {tuple(l1.shape)}')
print(f'  max|diff|={d.max().item():.3e}  mean|diff|={d.mean().item():.3e}')
print(f'  max rel diff={rel.max().item():.3e}')
print(f'  argmax igpu={l1.argmax().item()} ref={l2.argmax().item()}  {"MATCH" if l1.argmax().item()==l2.argmax().item() else "MISMATCH"}')
ok = d.max().item() < 5e-3 and l1.argmax().item() == l2.argmax().item()
print(f'  >>> {"PASS" if ok else "FAIL"}: iGPU FC 全头前向与 torch 参考一致' if ok else f'  >>> FAIL')

# FC 输出级对比 (更精细)
cat2 = head.__dict__.get('_last_cat', None)

# Warmup
prev_token = torch.tensor([12345], device=DEV, dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device=DEV, dtype=torch.bfloat16)
for _ in range(3): _ = head(prev_token, prev_hidden)
if DEV == "cuda": torch.cuda.synchronize()

# Measure MTP head forward (single token)
N = 50
ts = []
for _ in range(N):
    t0 = time.time()
    _ = head(prev_token, prev_hidden)
    if DEV == "cuda": torch.cuda.synchronize()
    ts.append((time.time() - t0) * 1000)
import numpy as np
mtp_head_ms = np.median(ts[5:])  # skip first few
print(f'\\nMTP head forward (real weights, iGPU FC): {mtp_head_ms:.2f}ms (median)')

# Project MTP K=3 speedup on real 35B
# Main model baseline: ~16.7ms per token (60 tok/s for 35B at decode)
# Main model verify K+1 tokens: ~17ms (small extra for K more tokens)
# MTP K=3 = 3 * mtp_head_ms + main_verify_ms = 3*10 + 17 = 47ms for up to 4 tokens
# Acceptance 60% -> 2.8 tokens per step -> 2.8/0.047 = 60 tok/s  (similar to baseline?)
# Actually: baseline 16.7ms for 1 token = 60 tok/s
# MTP K=3: 47ms for 2.8 tokens = 60 tok/s (same!)
# Acceptance 80% -> 3.4 tokens / 47ms = 72 tok/s
# Acceptance 100% -> 4 tokens / 47ms = 85 tok/s
# Speedup at 80%: 72/60 = 1.20x
# Speedup at 100%: 85/60 = 1.42x

# For 35B with iGPU FC: mtp_head ~10ms means 3 drafts = 30ms + main_verify 17ms = 47ms
# Without iGPU FC: MTP head would be 7.88ms * 3 = 23.6ms + 17 = 40.6ms (faster on dGPU!)
# Wait - dGPU MTP head is 7.88ms (P1c) but iGPU MTP head is 9.74ms (P1e) due to Python IPC overhead
# iGPU wins for the FC op (0.22ms vs ~1-2ms dGPU FC) but the rest (attn/MoE) is on dGPU

# Let me re-measure with actual timings
mtp_k3_total_ms = 3 * mtp_head_ms + 17  # 3 drafts + main verify
print(f'\\nProjection on 35B (assuming 16.7ms baseline main model):')
print(f'  MTP K=3 step time: 3 * {mtp_head_ms:.2f}ms (MTP head) + 17ms (main verify) = {mtp_k3_total_ms:.2f}ms')
print(f'  Baseline: 16.7ms per 1 token')
for accept in [0.5, 0.7, 0.8, 1.0]:
    avg_tok = 1 + accept * 3
    mtp_throughput = avg_tok / (mtp_k3_total_ms / 1000)
    baseline_throughput = 1 / 0.0167
    print(f'  Accept {accept*100:.0f}%: {avg_tok:.1f} tok/step -> {mtp_throughput:.1f} tok/s, speedup {mtp_throughput/baseline_throughput:.2f}x')

# Note: this projection is conservative because it assumes MTP head 9.74ms (with IPC).
# Once we move attn+MoE to iGPU, MTP head drops to ~3-4ms, making MTP much more attractive.
