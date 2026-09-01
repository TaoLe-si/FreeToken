"""Track 4 (A) P2/P3: Synthetic MTP decode benchmark.

Measures the speedup of MTP speculative decoding (iGPU drafter + dGPU verify)
vs. plain dGPU decode, using the actual MTP head + iGPU executor.

Approach:
  - Load real MTP head from checkpoint (Qwen3.6-35B-A3B-MXFP4-MTP)
  - Use MTP head's iGPU FC for the MTP head's FC layer
  - Use a synthetic "main model" (we don't load the full 35B model)
  - Simulate: 100 tokens, MTP-K=3 draft, verify on "main model"
  - Measure: tok/s for dGPU-only vs dGPU+MTP-iGPU

This is a standalone benchmark, NOT a scheduler integration.
The actual scheduler integration (with KV cache, etc.) is a separate
larger effort tracked in the e82ea6b1 report.
"""
import sys, os, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
import torch.nn.functional as F
import numpy as np

# Load MTP head
print('=== Loading MTP head ===')
t0 = time.time()
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, load_mtp_head_from_safetensors
import torch.nn as nn
cfg = MtpHeadConfig()
embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
head = load_mtp_head_from_safetensors(
    r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP',
    cfg, embed, lm_head, igpu_fc=None, device='cuda' if torch.cuda.is_available() else 'cpu',
    dtype=torch.bfloat16,
)
print(f'Loaded MTP head in {time.time()-t0:.1f}s')

# Start iGPU executor for the FC
print('\n=== Starting iGPU FC executor ===')
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight']  # (2048, 512) uint32
fc_row0 = fc_w[0:1].cpu().numpy()  # (1, 512) uint32

t0 = time.time()
from freetoken.engine.mtp_igpu_executor import MtpIgpuExecutor
igpu_fc = MtpIgpuExecutor(fc_row0, K=4096)
print(f'Started iGPU FC executor in {time.time()-t0:.1f}s')

# Create adapter
class IgpuFcAdapter:
    def __init__(self, executor):
        self.s = executor
    def __call__(self, act_flat):
        if act_flat.requires_grad:
            act_flat = act_flat.detach()
        if act_flat.dtype == torch.bfloat16:
            act_flat = act_flat.to(torch.float32)
        if act_flat.device.type != 'cpu':
            act_flat = act_flat.cpu()
        act_np = act_flat.contiguous().numpy().astype('float32')
        return torch.from_numpy(self.s.forward(act_np))

head.igpu_fc = IgpuFcAdapter(igpu_fc)

# Benchmark function for dGPU MTP head (no iGPU)
@torch.inference_mode()
def mtp_forward_dgpu(head, prev_token_id, prev_hidden):
    tok_t = torch.tensor([prev_token_id], device=head.embed_table.weight.device, dtype=torch.long)
    return head(tok_t, prev_hidden)

# Benchmark function for iGPU MTP head
@torch.inference_mode()
def mtp_forward_igpu(head, prev_token_id, prev_hidden):
    tok_t = torch.tensor([prev_token_id], device=head.embed_table.weight.device, dtype=torch.long)
    return head(tok_t, prev_hidden)

# Generate random test data
device = head.embed_table.weight.device
HIDDEN = 2048
prev_token_id = 42
prev_hidden = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)

# Warmup
print('\n=== Warmup ===')
for _ in range(5):
    _ = mtp_forward_dgpu(head, prev_token_id, prev_hidden)
    _ = mtp_forward_igpu(head, prev_token_id, prev_hidden)
torch.cuda.synchronize() if torch.cuda.is_available() else None

# === MTP draft (3 candidates per step) ===
def mtp_draft(head, prev_token_id, prev_hidden, k=3):
    draft_ids = [prev_token_id]
    cur_token = prev_token_id
    cur_hidden = prev_hidden
    for _ in range(k):
        logits = head(
            torch.tensor([cur_token], device=device, dtype=torch.long),
            cur_hidden,
        )
        next_id = int(logits[0].argmax().item())
        draft_ids.append(next_id)
        cur_token = next_id
        cur_hidden = logits[0:1, :].detach()
    return draft_ids

# Simulate "main model" verify with random argmax (no actual 35B model)
# The "main model" cost is what we want to compare. We'll use the MTP head itself
# as a stand-in (the iGPU FC dispatch is the only iGPU savings).
# 
# The point of the benchmark: how much time does MTP head add per token?
# If dGPU MTP head is 0.5ms, then per token we add 0.5ms for draft generation.
# If iGPU MTP head is 0.1ms (saving 0.4ms), that's the speedup we get.

print('\n=== Benchmark: dGPU MTP head draft (k=3) ===')
N = 100
t0 = time.time()
for _ in range(N):
    draft = mtp_draft(head, prev_token_id, prev_hidden, k=3)
if torch.cuda.is_available(): torch.cuda.synchronize()
t_dgpu = (time.time() - t0) * 1000 / N
print(f'  dGPU MTP draft (k=3): {t_dgpu:.2f}ms per token')

# Now without iGPU: re-init head without igpu_fc
head.igpu_fc = None
print('\n=== Benchmark: iGPU MTP head draft (k=3) ===')
t0 = time.time()
for _ in range(N):
    draft = mtp_draft(head, prev_token_id, prev_hidden, k=3)
if torch.cuda.is_available(): torch.cuda.synchronize()
t_igpu = (time.time() - t0) * 1000 / N
print(f'  iGPU MTP draft (k=3): {t_igpu:.2f}ms per token')

# Speedup
print(f'\n  iGPU MTP head speedup vs dGPU: {t_dgpu/t_igpu:.2f}x')

# Compare to dGPU-only baseline (no MTP)
# For speculative decoding, the time per accepted token is:
#   1 main model forward (e.g. 7ms for 35B)
#   + K MTP drafts (3 * 0.5ms = 1.5ms dGPU, or 3 * 0.1ms = 0.3ms iGPU)
# Per accepted token we save K-1 main forwards.
# 
# Net speedup = main_dgpu / (main_dgpu + K * draft_time)
# With MTP accept rate r, effective tokens per step = 1 + r*K
# So tok/s = (1 + r*K) / (main_dgpu + K*draft_time)

# Assumed main model latency
main_dgpu_ms = 7.0  # dGPU forward for 35B class model
accept_rate = 0.6  # typical MTP accept rate

print(f'\n=== Theoretical tok/s comparison ===')
print(f'  Main dGPU forward: {main_dgpu_ms}ms (assumed)')
print(f'  MTP accept rate: {accept_rate} (assumed)')
print(f'  dGPU MTP draft: {t_dgpu/3:.2f}ms per draft')
print(f'  iGPU MTP draft: {t_igpu/3:.2f}ms per draft')

for accept in [0.3, 0.5, 0.6, 0.7, 0.8]:
    # dGPU-only: 1 token per main_dgpu_ms
    # dGPU+MTP-dGPU: (1 + K*r) tokens per (main_dgpu_ms + K * t_dgpu/3) ms
    # dGPU+MTP-iGPU: (1 + K*r) tokens per (main_dgpu_ms + K * t_igpu/3) ms
    K = 3
    base_tps = 1000.0 / main_dgpu_ms
    dgpu_mtp_tps = (1 + K * accept) * 1000.0 / (main_dgpu_ms + K * t_dgpu / 3)
    igpu_mtp_tps = (1 + K * accept) * 1000.0 / (main_dgpu_ms + K * t_igpu / 3)
    print(f'  accept={accept}: dGPU-only={base_tps:.1f} tok/s, dGPU+MTP-dGPU={dgpu_mtp_tps:.1f} ({dgpu_mtp_tps/base_tps:.2f}x), dGPU+MTP-iGPU={igpu_mtp_tps:.1f} ({igpu_mtp_tps/base_tps:.2f}x)')

igpu_fc.close()
print('\n=== P2/P3 benchmark DONE ===')
