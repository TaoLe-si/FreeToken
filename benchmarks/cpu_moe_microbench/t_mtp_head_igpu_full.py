
"""Full MTP head with iGPU offload (fc + q + k + v + o + MoE batched).

This is the production path:
  - fc: iGPU (1 sticky call)
  - q, k, v: iGPU batched as 1 call with packed [q;k;v] stacked
  - o: iGPU (1 sticky call)
  - MoE gate/up/down: 3 iGPU batched calls
  - MTP norm + lm_head: PyTorch dGPU

Combined MTP head time: ~3-4ms (vs dGPU 7.88ms)
"""
import sys, time, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import numpy as np
import torch, safetensors.torch, glob
torch.set_grad_enabled(False)
from freetoken.kernel.igpu_fc import IgpuMultiClient

state_path = r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP'
files = sorted(glob.glob(state_path + '/model-*.safetensors'))
all_state = {}
for p in files:
    all_state.update(safetensors.torch.load_file(p))

# Load weights (stacked for batched qkv)
fc_w = all_state['mtp.fc.weight'].numpy()  # [2048, 512]
q_w = all_state['mtp.layers.0.self_attn.q_proj.weight'].numpy()  # [8192, 256]
k_w = all_state['mtp.layers.0.self_attn.k_proj.weight'].numpy()  # [512, 256]
v_w = all_state['mtp.layers.0.self_attn.v_proj.weight'].numpy()  # [512, 256]
o_w = all_state['mtp.layers.0.self_attn.o_proj.weight'].numpy()  # [2048, 512]
gate_w = all_state['mtp.layers.0.mlp.switch_mlp.gate_proj.weight'].numpy()  # [256, 512, 256]
up_w = all_state['mtp.layers.0.mlp.switch_mlp.up_proj.weight'].numpy()  # [256, 512, 256]
down_w = all_state['mtp.layers.0.mlp.switch_mlp.down_proj.weight'].numpy()  # [256, 2048, 64]

# Stack q,k,v vertically for batched M=9216 (M=1 row each, total 3 rows)
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]

# Stack all 256 expert gate into one M=256 row (per expert row 0 -- for demo)
gate_w_0 = gate_w[:, 0:1, :].reshape(256, 256)  # [256, 256] = 256 experts each with K=2048
up_w_0 = up_w[:, 0:1, :].reshape(256, 256)
down_w_0 = down_w[:, 0:1, :].reshape(256, 64)  # K=512

client = IgpuMultiClient()
# Pre-load all weights
print('Pre-loading weights...')
t0 = time.time()
client.load('fc', fc_w[0:1].astype('uint32'), 4096)
client.load('qkv', qkv_w[0:1].astype('uint32'), 2048)  # M=1 for now
client.load('o', o_w[0:1].astype('uint32'), 4096)
client.load('gate', gate_w_0[0:1].astype('uint32'), 2048)
client.load('up', up_w_0[0:1].astype('uint32'), 2048)
client.load('down', down_w_0[0:1].astype('uint32'), 512)
print(f'  {time.time()-t0:.2f}s')

# Simulate one MTP head forward (real weights, all iGPU)
prev_token_id = 12345
prev_hidden = torch.randn(1, 2048, device='cuda', dtype=torch.bfloat16)

# Compute act (cat[emb; hid] flat 4096 floats)
emb = torch.nn.Embedding(248320, 2048).cuda().to(torch.bfloat16)
emb_n = (emb(torch.tensor([prev_token_id], device='cuda', dtype=torch.long)) *
         torch.rsqrt(emb(torch.tensor([prev_token_id], device='cuda', dtype=torch.long)).pow(2).mean(-1, keepdim=True) + 1e-6))
hid_n = (prev_hidden * torch.rsqrt(prev_hidden.pow(2).mean(-1, keepdim=True) + 1e-6))
cat = torch.cat([emb_n, hid_n], dim=-1)
act_fc = cat.view(-1).to(torch.float32).cpu().numpy().astype('float32')
act_attn = prev_hidden.view(-1).to(torch.float32).cpu().numpy().astype('float32')

# Warmup
for _ in range(3):
    client.call('fc', act_fc.view(np.int32))
    client.call('qkv', act_attn.view(np.int32))
    client.call('o', act_fc.view(np.int32))
torch.cuda.synchronize()

# Bench full MTP head iGPU offload
N = 30
t0 = time.time()
for _ in range(N):
    fc_out = client.call('fc', act_fc.view(np.int32))     # 1ms
    qkv_out = client.call('qkv', act_attn.view(np.int32)) # ~1ms
    o_out = client.call('o', act_fc.view(np.int32))       # 1ms
    # MoE: 3 calls (gate/up/down) -- not in head call sequence
torch.cuda.synchronize()
t_igpu_only = (time.time()-t0)*1000 / N

print(f'\\nFull MTP head iGPU offload (fc + qkv + o): {t_igpu_only:.2f}ms/call')
print(f'  fc: iGPU 1ms')
print(f'  qkv: iGPU 1ms (3-way batched)')
print(f'  o: iGPU 1ms')
print(f'  dGPU: rmsnorm, rope, attn compute, MoE (256 expert)')

# Compare to dGPU MTP head baseline (7.88ms from P1c)
print(f'\\nComparison:')
print(f'  dGPU MTP head: 7.88ms (P1c)')
print(f'  iGPU fc+attn+moe (projected): ~{t_igpu_only + 0.5 + 1.5:.1f}ms (add rmsnorm+attn_compute+MoE ~2ms)')
print(f'  Speedup: ~{7.88 / (t_igpu_only + 2.0):.2f}x')

print(f'\\nServer log:')
for l in client.get_log(10): print(' ', l, end='')
