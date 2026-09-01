
import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, Qwen3_5MtpHead, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient

cfg = MtpHeadConfig(
    hidden_size=2048, num_qo_heads=16, num_kv_heads=4, head_dim=128,
    num_experts=256, top_k=8, num_shared_experts=1, moe_intermediate_size=512,
    moe_ffn_size_per_expert=512, rms_norm_eps=1e-6, vocab_size=248320,
)

# Build head without iGPU first
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)
class DummyLMHead:
    def forward(self, x): return torch.zeros(1, cfg.vocab_size, device=x.device, dtype=x.dtype)
lm = DummyLMHead()
