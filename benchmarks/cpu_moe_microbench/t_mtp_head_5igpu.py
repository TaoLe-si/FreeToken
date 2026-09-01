"""MTP head with 5 iGPU calls (fc + q + k + v + o) via sticky server.

This is the production path: pre-load all 5 weights once, then per-token
just send 5 act payloads (8-16KB each).
"""
import sys, time, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
import numpy as np
import safetensors.torch, glob
from freetoken.kernel.igpu_fc import IgpuMultiClient

state_path = r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP'
files = sorted(glob.glob(state_path + '/model-*.safetensors'))
all_state = {}
for p in files:
    all_state.update(safetensors.torch.load_file(p))
fc_w = all_state['mtp.fc.weight'].numpy()
q_w = all_state['mtp.layers.0.self_attn.q_proj.weight'].numpy()
k_w = all_state['mtp.layers.0.self_attn.k_proj.weight'].numpy()
v_w = all_state['mtp.layers.0.self_attn.v_proj.weight'].numpy()
o_w = all_state['mtp.layers.0.self_attn.o_proj.weight'].numpy()

# Pre-load all 5 weights
client = IgpuMultiClient()
print('Pre-loading 5 weights...')
t0 = time.time()
client.load('fc', fc_w[0:1].astype('uint32'), 4096)
client.load('q', q_w[0:1].astype('uint32'), 2048)
client.load('k', k_w[0:1].astype('uint32'), 2048)
client.load('v', v_w[0:1].astype('uint32'), 2048)
client.load('o', o_w[0:1].astype('uint32'), 4096)
print(f'  loaded in {time.time()-t0:.2f}s')

act_fc = np.random.randn(4096).astype('float32')
act_attn = np.random.randn(2048).astype('float32')

# Warmup
for _ in range(5):
    client.call('fc', act_fc.view(np.int32))
    client.call('q', act_attn.view(np.int32))
    client.call('k', act_attn.view(np.int32))
    client.call('v', act_attn.view(np.int32))
    client.call('o', act_fc.view(np.int32))
torch.cuda.synchronize()

# Bench 5-call sequence
N = 100
ts = []
for _ in range(N):
    t0 = time.time()
    client.call('fc', act_fc.view(np.int32))
    client.call('q', act_attn.view(np.int32))
    client.call('k', act_attn.view(np.int32))
    client.call('v', act_attn.view(np.int32))
    client.call('o', act_fc.view(np.int32))
    ts.append((time.time()-t0)*1000)
import numpy as np
ts = np.array(ts[5:])
print(f'\\nMTP head 5 iGPU calls (fc + qkv + o): {ts.mean():.3f}ms (p50 {np.median(ts):.3f}ms)')
print(f'  per call: {ts.mean()/5:.3f}ms')
print(f'\\nProjection for full MTP head iGPU offload:')
print(f'  iGPU fc+attn+moe: ~{ts.mean():.1f}ms (5 calls)')
print(f'  dGPU rmsnorm+attn_compute+MoE: ~2ms (PyTorch ops)')
print(f'  Total MTP head: ~{ts.mean()+2:.1f}ms')
print(f'  vs dGPU 7.88ms (P1c baseline): {7.88/(ts.mean()+2):.2f}x speedup')
print(f'\\nMTP K=3 with this MTP head:')
main_model_ms = 16.7
for accept in [0.5, 0.7, 0.8, 0.9, 1.0]:
    mtp_step_ms = 3*ts.mean() + main_model_ms  # 3 MTP drafts + 1 main verify
    avg_tok = 1 + accept * 3
    mtp_throughput = avg_tok / (mtp_step_ms / 1000)
    baseline_throughput = 1 / 0.0167
    print(f'  Accept {accept*100:.0f}%: MTP step {mtp_step_ms:.1f}ms, {mtp_throughput:.0f} tok/s, speedup {mtp_throughput/baseline_throughput:.2f}x')
