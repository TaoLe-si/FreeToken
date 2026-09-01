
import sys, time, os
sys.path.insert(0, r'E:\FreeToken\python')
import torch
torch.set_grad_enabled(False)
import numpy as np
from freetoken.kernel.igpu_fc import make_igpu_fc_sticky
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig
import safetensors.torch

cfg = MtpHeadConfig(hidden_size=2048, vocab_size=248320, num_experts=256,
    num_experts_per_tok=8, moe_intermediate=512, shared_expert_intermediate=512,
    head_dim=256, num_qo_heads=16, num_kv_heads=2, partial_rotary_factor=0.25,
    rms_norm_eps=1e-6)

# Load real FC weights from checkpoint
import safetensors
from safetensors import safe_open
path = r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP'
# Need to find MTP head files
import glob
files = sorted(glob.glob(os.path.join(path, '*.safetensors')))
# Find MTP files
for f in files:
    with safe_open(f, framework='pt') as g:
        keys = list(g.keys())
        if any('mtp' in k.lower() for k in keys):
            print(f'  {os.path.basename(f)}: {len(keys)} keys, mtp keys: {sum(1 for k in keys if "mtp" in k.lower())}')

# Load MTP fc
print('\nLoading MTP head...')
import torch.nn as nn
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().to(torch.bfloat16)
lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).cuda().to(torch.bfloat16)
from freetoken.models.qwen3_5_moe.mtp import load_mtp_head_from_safetensors
head = load_mtp_head_from_safetensors(
    path, cfg, embed, lm_head,
    igpu_fc=None, device='cuda', dtype=torch.bfloat16,
)
print('head loaded')

fc_packed = head._packed_mxfp4['fc.weight'].cpu().numpy().astype('uint32')
M, nb = fc_packed.shape
K = nb * 8
ns = K // 32
fc_scales = head._packed_mxfp4['fc.scales'].cpu().numpy().astype('float32')
fc_biases = head._packed_mxfp4['fc.biases'].cpu().numpy().astype('float32')
print(f'fc shape: M={M} K={K}')

# Build HIP sticky
sticky = make_igpu_fc_sticky(fc_packed, K, scales_f32=fc_scales, biases_f32=fc_biases)
print('sticky:', type(sticky).__name__)

# Measure FC call
act = np.random.randn(K).astype(np.float32)
sticky(act)  # warmup
N = 200
t0 = time.time()
for _ in range(N): out = sticky(act)
t_fc = (time.time()-t0)*1000/N
print(f'\nFC iGPU call (HIP): {t_fc:.3f}ms per call')
print(f'For K=3 draft: {3*t_fc:.3f}ms')
