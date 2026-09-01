import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky
import safetensors.torch, numpy as np
torch.set_grad_enabled(False)

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
print(f'Loaded in {time.time()-t0:.1f}s')

fc_packed_1 = head._packed_mxfp4['fc.weight'][0:1].numpy().astype(np.uint32)

class IgpuFcAdapter:
    def __init__(self, sticky):
        self.sticky = sticky
    def __call__(self, act_flat):
        act_fp32 = act_flat.detach().to(torch.float32) if act_flat.dtype == torch.bfloat16 else act_flat.detach()
        act_np = act_fp32.cpu().numpy().astype(np.float32)
        outv = self.sticky(act_np)
        return torch.from_numpy(outv.copy()).to(torch.bfloat16)
head.igpu_fc = IgpuFcAdapter(IgpuFcSticky(igpu, fc_packed_1, 4096))

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
print(f'MTP head (iGPU fc): {(t1-t0)*1000/N:.2f}ms/iter, logits {logits.shape}')
print(f'logits[:5]: {logits[0, :5].float().cpu().tolist()}')
print(f'Server log: {igpu.get_log(3)}')
