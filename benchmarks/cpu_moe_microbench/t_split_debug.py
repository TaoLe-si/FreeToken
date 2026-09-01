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

# Test call with real h
act_test = np.random.randn(2048).astype(np.float32)
q_sticky(act_test); k_sticky(act_test); v_sticky(act_test)

class IgpuQkvWrapper:
    def __init__(self, q_s, k_s, v_s):
        self.q_s, self.k_s, self.v_s = q_s, k_s, v_s
    def __call__(self, x):
        x_flat = x.view(-1).detach().to(torch.float32).cpu().numpy().astype(np.float32)
        if x_flat.shape[0] == 2048:
            q = self.q_s(x_flat); k = self.k_s(x_flat); v = self.v_s(x_flat)
        else: raise ValueError(f'unexpected K={x_flat.shape[0]}')
        qkv = np.concatenate([q, k, v])
        print(f'qkv after concat: {qkv.shape}')
        out = torch.from_numpy(qkv.copy()).cuda().to(torch.bfloat16)
        print(f'after from_numpy: {out.shape}')
        out_view = out.view(1, -1)
        print(f'after view(1,-1): {out_view.shape}')
        return out_view

h_test = torch.randn(1, 2048, device='cuda', dtype=torch.bfloat16)
qkv_test = IgpuQkvWrapper(q_sticky, k_sticky, v_sticky)(h_test)
print(f'qkv_test: {qkv_test.shape}')
# Try split
print('Trying split...')
qg, k, v = torch.split(qkv_test, [8192, 512, 512], dim=-1)
print(f'OK: qg {qg.shape}, k {k.shape}, v {v.shape}')
