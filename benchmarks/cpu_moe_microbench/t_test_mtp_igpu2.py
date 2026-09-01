import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient

cfg = MtpHeadConfig(
    hidden_size=2048, num_qo_heads=16, num_kv_heads=4, head_dim=128,
    num_experts=256, num_experts_per_tok=8, moe_intermediate=512,
    shared_expert_intermediate=512, rms_norm_eps=1e-6, vocab_size=248320,
)
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)

# iGPU client
igpu = IgpuFcClient()

class DummyLMHead:
    def forward(self, x): return torch.zeros(1, cfg.vocab_size, device=x.device, dtype=x.dtype)
lm = DummyLMHead()

# Load head with iGPU FC
t0 = time.time()
head = load_mtp_head_from_safetensors(
    r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP', cfg, embed, lm,
    igpu_fc=igpu, device='cuda', dtype=torch.bfloat16,
)
t1 = time.time()
print(f'Loaded MTP head in {t1-t0:.1f}s')

# Warmup
prev_token = torch.tensor([12345], device='cuda', dtype=torch.long)
prev_hidden = torch.randn(1, cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
t0 = time.time()
logits = head(prev_token, prev_hidden)
t1 = time.time()
print(f'MTP head forward (iGPU fc): {(t1-t0)*1000:.2f}ms, logits shape {logits.shape}')
print(f'logits sample: {logits[0, :5].float().cpu().tolist()}')
print(f'Server log: {igpu.get_log(5)}')
