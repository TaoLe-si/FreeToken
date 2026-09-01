import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky
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

igpu = IgpuFcClient()

class DummyLMHead:
    def forward(self, x): return torch.zeros(1, cfg.vocab_size, device=x.device, dtype=x.dtype)
lm = DummyLMHead()

t0 = time.time()
head = load_mtp_head_from_safetensors(
    r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP', cfg, embed, lm,
    igpu_fc=None, device='cuda', dtype=torch.bfloat16,
)
t1 = time.time()
print(f'Loaded MTP head (no iGPU) in {t1-t0:.1f}s')

# Replace the head's cat fc with our IgpuFcSticky using packed fc weight
fc_packed = head._packed_mxfp4['fc.weight']  # [1, 2048] uint32 (K=2048 -> K/8=256)... wait K=4096 actually
K = head._packed_mxfp4['fc.weight'].shape[1] * 8
print(f'fc packed shape: {head._packed_mxfp4["fc.weight"].shape}, K={K}')
# fc was [1, 2048] uint32 = 16384 nibbles? no [1, 2048] uint32 = 2048*4=8192 nibbles -> K=8192. That's wrong.
# Actually fc output is 2048 and fc input K = 2*2048 = 4096 (concat of embed+hidden)
# packed = [out=2048, K/8=512] uint32. shape (2048, 512) -> M=2048, K=4096
fc_packed_2d = head._packed_mxfp4['fc.weight']  # [2048, 512]
# We need M=1, so take row 0
fc_packed_1 = fc_packed_2d[0:1].numpy().astype(np.uint32)  # [1, 512]
K = 4096
igpu_fc_sticky = IgpuFcSticky(igpu, fc_packed_1, K)
head.igpu_fc = igpu_fc_sticky

# Warmup
prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
for _ in range(3):
    logits = head(prev_token, prev_hidden)
torch.cuda.synchronize()
N = 20
t0 = time.time()
for i in range(N):
    logits = head(prev_token, prev_hidden)
torch.cuda.synchronize()
t1 = time.time()
print(f'MTP head forward (iGPU fc): {(t1-t0)*1000/N:.2f}ms/iter, logits shape {logits.shape}')
print(f'logits sample: {logits[0, :5].float().cpu().tolist()}')
print(f'Server log: {igpu.get_log(3)}')
