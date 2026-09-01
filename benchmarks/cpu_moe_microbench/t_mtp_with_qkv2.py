import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
import numpy as np
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky
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
q_w = state23['mtp.layers.0.self_attn.q_proj.weight'].numpy()
k_w = state23['mtp.layers.0.self_attn.k_proj.weight'].numpy()
v_w = state23['mtp.layers.0.self_attn.v_proj.weight'].numpy()

q_sticky = IgpuFcSticky(igpu, q_w[:1], 2048)
k_sticky = IgpuFcSticky(igpu, k_w[:1], 2048)
v_sticky = IgpuFcSticky(igpu, v_w[:1], 2048)
fc_packed_1 = head._packed_mxfp4['fc.weight'][0:1].numpy().astype(np.uint32)
fc_sticky = IgpuFcSticky(igpu, fc_packed_1, 4096)

# Warmup
act_test = np.random.randn(2048).astype(np.float32)
q_sticky(act_test); k_sticky(act_test); v_sticky(act_test)
fc_sticky(np.random.randn(4096).astype(np.float32))

# Patch head manually (no nn.Module assignment)
class IgpuFcAdapter:
    def __init__(self, sticky): self.sticky = sticky
    def __call__(self, act_flat):
        act_fp32 = act_flat.detach().to(torch.float32) if act_flat.dtype == torch.bfloat16 else act_flat.detach()
        act_np = act_fp32.cpu().numpy().astype(np.float32)
        outv = self.sticky(act_np)
        return torch.from_numpy(outv.copy()).cuda().to(torch.bfloat16)

# Override qkv_proj's forward by monkey-patching
orig_qkv = head.attn.qkv_proj
class IgpuQkvWrapper:
    def __init__(self, q_s, k_s, v_s, head_ref):
        self.q_s, self.k_s, self.v_s = q_s, k_s, v_s
    def __call__(self, x):
        x_flat = x.view(-1).detach().to(torch.float32).cpu().numpy().astype(np.float32)
        if x_flat.shape[0] == 2048:
            q = self.q_s(x_flat); k = self.k_s(x_flat); v = self.v_s(x_flat)
        else: raise ValueError(f'unexpected K={x_flat.shape[0]}')
        qkv = np.concatenate([q, k, v])
        return torch.from_numpy(qkv.copy()).cuda().to(torch.bfloat16).view(1, -1)

head.igpu_fc = IgpuFcAdapter(fc_sticky)
# Don't replace qkv_proj (which is nn.Linear). Instead, intercept by overriding the call in MtpHeadAttention.forward
# Simpler: just measure the timing using a manual forward
prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)

# Manual forward using iGPU qkv
def mtp_with_igpu_qkv(head, prev_token, prev_hidden):
    emb = head.embed_table(prev_token)
    emb_n = (emb * torch.rsqrt(emb.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.pre_fc_norm_embedding)
    hid_n = (prev_hidden * torch.rsqrt(prev_hidden.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.pre_fc_norm_hidden)
    cat = torch.cat([emb_n, hid_n], dim=-1)
    cat_flat = cat.view(-1).to(torch.float32)
    fc_out = head.igpu_fc(cat_flat).view(1, -1).to(torch.bfloat16)
    h = fc_out + prev_hidden
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + head.input_layernorm)
    # iGPU qkv
    qkv = IgpuQkvWrapper(q_sticky, k_sticky, v_sticky, head)(h)
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

# Warmup
for _ in range(3): logits = mtp_with_igpu_qkv(head, prev_token, prev_hidden)
torch.cuda.synchronize()
N = 20
t0 = time.time()
for i in range(N): logits = mtp_with_igpu_qkv(head, prev_token, prev_hidden)
torch.cuda.synchronize()
t1 = time.time()
print(f'MTP head (iGPU fc + iGPU qkv + dGPU rest): {(t1-t0)*1000/N:.2f}ms/iter')
print(f'  vs dGPU: 7.88ms, vs iGPU fc only: 9.74ms')
print(f'logits[:5]: {logits[0, :5].float().cpu().tolist()}')
