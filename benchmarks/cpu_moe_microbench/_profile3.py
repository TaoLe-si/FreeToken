
import sys, time, os
sys.path.insert(0, r'E:\FreeToken\python')
import torch
torch.set_grad_enabled(False)
import numpy as np
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import make_igpu_fc_sticky
import safetensors.torch, torch.nn as nn

cfg = MtpHeadConfig(hidden_size=2048, vocab_size=248320, num_experts=256,
    num_experts_per_tok=8, moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2, partial_rotary_factor=0.25,
    rms_norm_eps=1e-6)

embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)
lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).cuda().to(torch.bfloat16)
head = load_mtp_head_from_safetensors(
    r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP', cfg, embed, lm_head,
    igpu_fc=None, device='cuda', dtype=torch.bfloat16,
)

# Pre-populate KV cache with N=512 rows (typical real context)
N_cache = 512
tokens = torch.randint(0, cfg.vocab_size, (N_cache,), device='cuda', dtype=torch.long)
hiddens = torch.randn(N_cache, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
# Manual extend: use torch FC
head.attn.reset_draft_cache()
fc_packed = head._packed_mxfp4['fc.weight']
fc_scales = head._packed_mxfp4['fc.scales']
fc_biases = head._packed_mxfp4['fc.biases']
from freetoken.models.qwen3_5_moe.mtp import _dequant_mxfp4_affine, _rmsnorm
w_fc = _dequant_mxfp4_affine(fc_packed, fc_scales, fc_biases).to('cuda')  # (M, K) fp32
emb = head.embed_table(tokens)
emb_n = _rmsnorm(emb, head.pre_fc_norm_embedding)
hid_n = _rmsnorm(hiddens, head.pre_fc_norm_hidden)
cat = torch.cat([emb_n, hid_n], dim=-1)
fc_out = (cat.float() @ w_fc.t()).to(torch.bfloat16)
h = fc_out
h = _rmsnorm(h, head.input_layernorm)
positions_all = torch.arange(1, N_cache + 1, device='cuda', dtype=torch.long)
head.attn.append_rows(h, positions_all)
print(f'KV cache populated: {head.attn.kv_len()} rows')

# Profile each component with realistic KV size
N = 30
prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
position = N_cache

# Warmup
for _ in range(10):
    emb = head.embed_table(prev_token)
    h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
    positions = torch.tensor([position], dtype=torch.long, device='cuda')
    out = head.attn(h, positions)
    out = head.mlp(h)
    out = head.lm_head(h)
torch.cuda.synchronize()

# 1) embed
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): emb = head.embed_table(prev_token)
torch.cuda.synchronize()
t_emb = (time.time()-t0)*1000/N

# 2) attn (with KV cache of N_cache=512)
h = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
positions = torch.tensor([position], dtype=torch.long, device='cuda')
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

# 5) FC iGPU
fc_packed = head._packed_mxfp4['fc.weight'].cpu().numpy().astype('uint32')
K_fc = fc_packed.shape[1] * 8
ns_fc = K_fc // 32
fc_scales = head._packed_mxfp4['fc.scales'].cpu().numpy().astype('float32')
fc_biases = head._packed_mxfp4['fc.biases'].cpu().numpy().astype('float32')
sticky = make_igpu_fc_sticky(fc_packed, K_fc, scales_f32=fc_scales, biases_f32=fc_biases)
act = np.random.randn(K_fc).astype(np.float32)
sticky(act)
N_fc = 100
t0 = time.time()
for _ in range(N_fc): out = sticky(act)
t_fc = (time.time()-t0)*1000/N_fc

print(f'\n=== Component timings (KV cache: {N_cache} rows) ===')
print(f'  embed:        {t_emb:.3f}ms')
print(f'  attn:         {t_attn:.3f}ms')
print(f'  MoE:          {t_moe:.3f}ms')
print(f'  lm_head:      {t_lm:.3f}ms')
print(f'  FC (iGPU):    {t_fc:.3f}ms')
total_per_step = t_emb + t_attn + t_moe + t_lm + t_fc
print(f'  Total/step:   {total_per_step:.3f}ms')
print(f'  Per K=3:      {3*total_per_step:.3f}ms')

# Time forward_with_state (real step)
prev_hidden_d = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
# Warmup forward
for _ in range(10):
    head.attn.reset_draft_cache()
    head.extend_context(tokens, hiddens, start_pos=0)
    logits, state = head.forward_with_state(prev_token, prev_hidden_d, position=position)
torch.cuda.synchronize()

# Time forward step
head.attn.reset_draft_cache()
head.extend_context(tokens, hiddens, start_pos=0)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    logits, state = head.forward_with_state(prev_token, prev_hidden_d, position=position)
torch.cuda.synchronize()
t_full = (time.time()-t0)*1000/N

print(f'\n  Full forward step: {t_full:.3f}ms')
print(f'  3 draft steps:     {3*t_full:.3f}ms')
print(f'  Time for K=3 (excluding verify): {3*t_full:.1f}ms')
print(f'  Implies MTP can do {1000/t_full:.0f} draft steps/s on this hardware')
print(f'  Main verify forward: ~75ms for 4 tokens')
print(f'  Per round (verify + 3 draft): ~{75 + 3*t_full:.0f}ms')
print(f'  If 100% accepted: {4 / (75 + 3*t_full) * 1000:.1f} t/s')
print(f'  Actual measured: 7 t/s -> {1000/7:.0f}ms/token = {(75 + 3*t_full) / 7:.0f}ms over budget')
