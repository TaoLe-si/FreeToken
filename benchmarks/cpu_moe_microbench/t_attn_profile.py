import sys, time, os
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
import safetensors.torch, numpy as np

cfg = MtpHeadConfig(
    hidden_size=2048, vocab_size=248320,
    num_experts=256, num_experts_per_tok=8,
    moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2,
    partial_rotary_factor=0.25, rms_norm_eps=1e-6,
)
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)
class MyLM(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.proj = nn.Linear(H, V, bias=False).to(torch.bfloat16)
    def forward(self, x): return self.proj(x)
lm = MyLM(cfg.vocab_size, cfg.hidden_size).cuda()
head = load_mtp_head_from_safetensors(
    r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP', cfg, embed, lm,
    igpu_fc=None, device='cuda', dtype=torch.bfloat16,
)

# Profile attn sub-components
h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
N = 50

# qkv_proj
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): qkv = head.attn.qkv_proj(h)
torch.cuda.synchronize()
t_qkv = (time.time()-t0)*1000/N

# split
qkv = head.attn.qkv_proj(h)
qg, k, v = torch.split(qkv, head.attn.split_sizes, dim=-1)
qo_dim = head.attn.qo_dim
kv_dim = head.attn.kv_dim
head_dim = head.attn.head_dim
num_q = head.attn.num_q
num_kv = head.attn.num_kv

# rmsnorm
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    qg2 = qg.view(-1, num_q, head_dim*2)
    q2 = qg2[..., :head_dim]
    gate2 = qg2[..., head_dim:]
    k2 = k.view(-1, num_kv, head_dim)
    v2 = v.view(-1, num_kv, head_dim)
    import torch.nn.functional as F
    from freetoken.models.qwen3_5_moe.mtp import _rmsnorm
    qn = _rmsnorm(q2, head.attn.q_norm).reshape(-1, qo_dim)
    kn = _rmsnorm(k2, head.attn.k_norm).reshape(-1, kv_dim)
torch.cuda.synchronize()
t_norm = (time.time()-t0)*1000/N

# rope
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    from freetoken.models.qwen3_5_moe.mtp import _neox_rope
    positions = torch.zeros(1, dtype=torch.long, device='cuda')
    rotary_dim = int(head_dim * cfg.partial_rotary_factor)
    q3 = qn.view(-1, num_q, head_dim)
    k3 = kn.view(-1, num_kv, head_dim)
    qr, kr = _neox_rope(q3, k3, positions, rotary_dim)
torch.cuda.synchronize()
t_rope = (time.time()-t0)*1000/N

# attention compute
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    q4 = qr
    k4 = kr
    scale = 1.0 / (head_dim ** 0.5)
    attn_logits = torch.einsum('nqd,nkd->nqk', q4, k4) * scale
    attn = torch.softmax(attn_logits.float(), dim=-1).to(q4.dtype)
    out_attn = torch.einsum('nqk,nkd->nqd', attn, v2)
    out_attn = out_attn * torch.sigmoid(gate2)
    out_flat = out_attn.reshape(-1, qo_dim)
torch.cuda.synchronize()
t_attn_compute = (time.time()-t0)*1000/N

# o_proj
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): o_out = head.attn.o_proj(out_flat)
torch.cuda.synchronize()
t_o = (time.time()-t0)*1000/N

print(f'attn breakdown (M=1):')
print(f'  qkv_proj:    {t_qkv:.3f}ms')
print(f'  rmsnorm:      {t_norm:.3f}ms')
print(f'  rope:         {t_rope:.3f}ms')
print(f'  attn compute: {t_attn_compute:.3f}ms')
print(f'  o_proj:       {t_o:.3f}ms')
print(f'  Sum:          {t_qkv+t_norm+t_rope+t_attn_compute+t_o:.3f}ms')
print(f'  full attn:    {7.989:.3f}ms')
print()
print('qkv_proj is the biggest attn op (~2-3ms iGPU+IPC could replace)')
print('o_proj is also large (~2-3ms iGPU could replace)')
print('softmax/einsum is fast (custom kernel might be even faster)')
