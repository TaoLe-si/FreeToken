import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
import numpy as np
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient
import safetensors.torch

cfg = MtpHeadConfig(
    hidden_size=2048, vocab_size=248320,
    num_experts=256, num_experts_per_tok=8,
    moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2,
    partial_rotary_factor=0.25, rms_norm_eps=1e-6,
)
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)
igpu = IgpuFcClient()
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

state23 = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00023-of-00023.safetensors')
q_w = state23['mtp.layers.0.self_attn.q_proj.weight'].numpy()  # [8192, 256]
k_w = state23['mtp.layers.0.self_attn.k_proj.weight'].numpy()  # [512, 256]
v_w = state23['mtp.layers.0.self_attn.v_proj.weight'].numpy()  # [512, 256]
o_w = state23['mtp.layers.0.self_attn.o_proj.weight'].numpy()  # [2048, 512]
fc_w = head._packed_mxfp4['fc.weight'].numpy()  # [2048, 512]
print(f'q_w: {q_w.shape}, k_w: {k_w.shape}, v_w: {v_w.shape}, o_w: {o_w.shape}, fc_w: {fc_w.shape}')

# Concatenate q+k+v vertically for batched M=9216 call
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]
print(f'qkv_w: {qkv_w.shape}')

act = np.random.randn(2048).astype(np.float32)

# First call: warmup (will be slow due to realloc)
print('warmup batched qkv...')
t0 = time.time()
qkv_out = igpu.forward(qkv_w, act.view(np.int32))
t1 = time.time()
print(f'  warmup: {(t1-t0)*1000:.1f}ms, outv shape: {qkv_out.shape}')

# Steady state
N = 50
t0 = time.time()
for _ in range(N): qkv_out = igpu.forward(qkv_w, act.view(np.int32))
t1 = time.time()
print(f'batched qkv steady state: {(t1-t0)*1000/N:.3f}ms/iter')

# Same for o and fc (each M=1)
act_o = np.random.randn(4096).astype(np.float32)
o_w_1 = o_w[:1]
t0 = time.time()
for _ in range(N): o_out = igpu.forward(o_w_1, act_o.view(np.int32))
t1 = time.time()
print(f'o_proj (M=1, K=4096): {(t1-t0)*1000/N:.3f}ms/iter')

fc_w_1 = fc_w[:1]
t0 = time.time()
for _ in range(N): fc_out = igpu.forward(fc_w_1, act_o.view(np.int32))
t1 = time.time()
print(f'fc (M=1, K=4096): {(t1-t0)*1000/N:.3f}ms/iter')

# Now full MTP head manual forward
def mtp_with_igpu(head, prev_token, prev_hidden):
    emb = head.embed_table(prev_token)
    emb_n = (emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.pre_fc_norm_embedding)
    hid_n = (prev_hidden * torch.rsqrt(prev_hidden.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.pre_fc_norm_hidden)
    cat = torch.cat([emb_n, hid_n], dim=-1)
    cat_flat = cat.view(-1).to(torch.float32).cpu().numpy().astype(np.float32)
    fc_out_np = igpu.forward(fc_w_1, cat_flat.view(np.int32))
    fc_out = torch.from_numpy(fc_out_np.copy()).cuda().to(torch.bfloat16).view(1, -1)
    h = fc_out + prev_hidden
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.input_layernorm)
    h_flat = h.view(-1).detach().to(torch.float32).cpu().numpy().astype(np.float32)
    qkv_out = igpu.forward(qkv_w, h_flat.view(np.int32))
    qkv = torch.from_numpy(qkv_out.copy()).cuda().to(torch.bfloat16).view(1, -1)
    qg, k, v = torch.split(qkv, head.attn.split_sizes, dim=-1)
    qg = qg.view(-1, head.attn.num_q, head.attn.head_dim * 2)
    q = qg[..., :head.attn.head_dim]
    gate = qg[..., head.attn.head_dim:]
    k = k.view(-1, head.attn.num_kv, head.attn.head_dim)
    v = v.view(-1, head.attn.num_kv, head.attn.head_dim)
    from freetoken.models.qwen3_5_moe.mtp import _rmsnorm, _neox_rope
    q = _rmsnorm(q, head.attn.q_norm).reshape(-1, head.attn.qo_dim)
    k = _rmsnorm(k, head.attn.k_norm).reshape(-1, head.attn.kv_dim)
    positions = torch.zeros(1, dtype=torch.long, device=h.device)
    rotary_dim = int(head.attn.head_dim * 0.25)
    q = q.view(-1, head.attn.num_q, head.attn.head_dim)
    k = k.view(-1, head.attn.num_kv, head.attn.head_dim)
    q, k = _neox_rope(q, k, positions, rotary_dim)
    scale = 1.0 / (head.attn.head_dim ** 0.5)
    attn_logits = torch.einsum('nqd,nkd->nqk', q, k) * scale
    attn = torch.softmax(attn_logits.float(), dim=-1).to(q.dtype)
    out_attn = torch.einsum('nqk,nkd->nqd', attn, v)
    out_attn = out_attn * torch.sigmoid(gate)
    out_attn = out_attn.reshape(-1, head.attn.qo_dim)
    h = head.attn.o_proj(out_attn) + h
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.post_attention_layernorm)
    h = head.mlp(h) + h
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.mtp_norm)
    return head.lm_head(h)

prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
for _ in range(3): logits = mtp_with_igpu(head, prev_token, prev_hidden)
torch.cuda.synchronize()
N = 20
t0 = time.time()
for i in range(N): logits = mtp_with_igpu(head, prev_token, prev_hidden)
torch.cuda.synchronize()
t1 = time.time()
print(f'MTP head (iGPU fc + iGPU qkv + dGPU rest): {(t1-t0)*1000/N:.2f}ms/iter')
print(f'  vs dGPU: 7.88ms')
print(f'logits[:5]: {logits[0, :5].float().cpu().tolist()}')
