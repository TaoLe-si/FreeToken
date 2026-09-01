
import sys, time, os
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
import safetensors.torch, numpy as np
import torch.nn as nn

cfg = MtpHeadConfig(
    hidden_size=2048, vocab_size=248320,
    num_experts=256, num_experts_per_tok=8,
    moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2,
    partial_rotary_factor=0.25, rms_norm_eps=1e-6,
)

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

# Profile each component
N = 50

prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    emb = head.embed_table(prev_token)
    h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.long, device='cuda')
    out = head.attn(h, positions)
    out = head.mlp(h)
    out = head.lm_head(h)
torch.cuda.synchronize()

# 1) embed lookup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): emb = head.embed_table(prev_token)
torch.cuda.synchronize()
t_emb = (time.time()-t0)*1000/N

# 2) attn
h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
positions = torch.zeros(1, dtype=torch.long, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): out = head.attn(h, positions)
torch.cuda.synchronize()
t_attn = (time.time()-t0)*1000/N

# 3) MoE
h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): out = head.mlp(h)
torch.cuda.synchronize()
t_moe = (time.time()-t0)*1000/N

# 4) lm_head
h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): out = head.lm_head(h)
torch.cuda.synchronize()
t_lm = (time.time()-t0)*1000/N

# 5) Full draft step (K=3)
print(f'\n=== Component timings (dGPU) ===')
print(f'  embed:   {t_emb:.3f}ms')
print(f'  attn:    {t_attn:.3f}ms')
print(f'  MoE:     {t_moe:.3f}ms')
print(f'  lm_head: {t_lm:.3f}ms')
print(f'  Sum (1 step): {t_emb+t_attn+t_moe+t_lm:.3f}ms')
print(f'  Per draft (K=3): {3*(t_emb+t_attn+t_moe+t_lm):.3f}ms')
print(f'  + RMSNorm x3, FC, gate, o_proj, residual ops (estimate ~3ms): {3*(t_emb+t_attn+t_moe+t_lm)+3:.3f}ms')
print(f'\nBottleneck: MoE and lm_head dominate')

# Time full forward_with_state
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): logits, state = head.forward_with_state(prev_token, prev_hidden, position=0)
torch.cuda.synchronize()
t_full = (time.time()-t0)*1000/N
print(f'\n  full forward_with_state (1 step): {t_full:.3f}ms')
print(f'  3 draft steps: {3*t_full:.3f}ms')
print(f'  -> implies MTP could potentially run at {1000/(3*t_full):.1f} steps/s with K=3')
