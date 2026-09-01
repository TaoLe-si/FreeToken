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
print(f'Loaded in {time.time()-0:.1f}s')

# Get q/k/v weights
state23 = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00023-of-00023.safetensors')
q_w = state23['mtp.layers.0.self_attn.q_proj.weight'].numpy()
k_w = state23['mtp.layers.0.self_attn.k_proj.weight'].numpy()
v_w = state23['mtp.layers.0.self_attn.v_proj.weight'].numpy()
o_w = state23['mtp.layers.0.self_attn.o_proj.weight'].numpy()

# Pre-build M=1 sticky wrappers (3 of them)
q_sticky = IgpuFcSticky(igpu, q_w[:1], 2048)
k_sticky = IgpuFcSticky(igpu, k_w[:1], 2048)
v_sticky = IgpuFcSticky(igpu, v_w[:1], 2048)
o_sticky = IgpuFcSticky(igpu, o_w[:1], 4096)
fc_packed_1 = head._packed_mxfp4['fc.weight'][0:1].numpy().astype(np.uint32)
fc_sticky = IgpuFcSticky(igpu, fc_packed_1, 4096)

# Warmup all
act_test = np.random.randn(2048).astype(np.float32)
q_sticky(act_test)
k_sticky(act_test)
v_sticky(act_test)
o_sticky(np.random.randn(4096).astype(np.float32))
fc_sticky(np.random.randn(4096).astype(np.float32))

class IgpuFcAdapter:
    def __init__(self, sticky): self.sticky = sticky
    def __call__(self, act_flat):
        act_fp32 = act_flat.detach().to(torch.float32) if act_flat.dtype == torch.bfloat16 else act_flat.detach()
        act_np = act_fp32.cpu().numpy().astype(np.float32)
        outv = self.sticky(act_np)
        return torch.from_numpy(outv.copy()).cuda().to(torch.bfloat16)

class IgpuQkvAdapter:
    """Replaces head.attn.qkv_proj with 3 iGPU calls."""
    def __init__(self, q_s, k_s, v_s, head_ref):
        self.q_s, self.k_s, self.v_s = q_s, k_s, v_s
        self.split_sizes = head_ref.attn.split_sizes
        self.num_q = head_ref.attn.num_q
        self.num_kv = head_ref.attn.num_kv
        self.head_dim = head_ref.attn.head_dim
        self.qo_dim = head_ref.attn.qo_dim
        self.kv_dim = head_ref.attn.kv_dim
    def __call__(self, x):
        # x: [N, H] bf16 -> flat float32
        x_flat = x.view(-1).detach().to(torch.float32).cpu().numpy().astype(np.float32)
        if x_flat.shape[0] == 2048:
            q = self.q_s(x_flat)
            k = self.k_s(x_flat)
            v = self.v_s(x_flat)
        else:
            raise ValueError(f'unexpected K={x_flat.shape[0]}')
        qkv = np.concatenate([q, k, v])
        return torch.from_numpy(qkv.copy()).cuda().to(torch.bfloat16).view(1, -1)

# Patch
head.igpu_fc = IgpuFcAdapter(fc_sticky)
head.attn.qkv_proj = IgpuQkvAdapter(q_sticky, k_sticky, v_sticky, head)

# Warmup full head
prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
for _ in range(3): logits = head(prev_token, prev_hidden)
torch.cuda.synchronize()
N = 20
t0 = time.time()
for i in range(N): logits = head(prev_token, prev_hidden)
torch.cuda.synchronize()
t1 = time.time()
print(f'MTP head (iGPU fc + iGPU qkv): {(t1-t0)*1000/N:.2f}ms/iter')
print(f'  vs dGPU: 7.88ms, vs iGPU fc only: 9.74ms')
print(f'logits[:5]: {logits[0, :5].float().cpu().tolist()}')
