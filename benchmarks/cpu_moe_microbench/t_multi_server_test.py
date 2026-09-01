
import sys, time, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import numpy as np
import torch, safetensors.torch
from freetoken.kernel.igpu_fc import IgpuMultiClient

client = IgpuMultiClient()
import safetensors.torch, glob
mp = r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP'
files = sorted(glob.glob(mp + '/model-*.safetensors'))
all_state = {}
for p in files:
    all_state.update(safetensors.torch.load_file(p))
fc_w = all_state['mtp.fc.weight']
q_w = all_state['mtp.layers.0.self_attn.q_proj.weight']
k_w = all_state['mtp.layers.0.self_attn.k_proj.weight']
v_w = all_state['mtp.layers.0.self_attn.v_proj.weight']
o_w = all_state['mtp.layers.0.self_attn.o_proj.weight']

# Load all weights
print('Loading 5 weights...')
t0 = time.time()
client.load('fc', fc_w[0:1].numpy().astype('uint32'), 4096)
print(f'  fc: {time.time()-t0:.2f}s')
t0 = time.time()
client.load('q', q_w[0:1].numpy().astype('uint32'), 2048)
print(f'  q: {time.time()-t0:.2f}s')
t0 = time.time()
client.load('k', k_w[0:1].numpy().astype('uint32'), 2048)
print(f'  k: {time.time()-t0:.2f}s')
t0 = time.time()
client.load('v', v_w[0:1].numpy().astype('uint32'), 2048)
print(f'  v: {time.time()-t0:.2f}s')
t0 = time.time()
client.load('o', o_w[0:1].numpy().astype('uint32'), 4096)
print(f'  o: {time.time()-t0:.2f}s')

# Warmup each
act_fc = np.random.randn(4096).astype('float32').view(np.int32)
act_attn = np.random.randn(2048).astype('float32').view(np.int32)
act_o = np.random.randn(4096).astype('float32').view(np.int32)
client.call('fc', act_fc)
client.call('q', act_attn)
client.call('k', act_attn)
client.call('v', act_attn)
client.call('o', act_o)

# Bench: MTP head call sequence (fc + q + k + v + o)
N = 50
ts = []
for _ in range(N):
    t0 = time.time()
    out_fc = client.call('fc', act_fc)
    out_q = client.call('q', act_attn)
    out_k = client.call('k', act_attn)
    out_v = client.call('v', act_attn)
    out_o = client.call('o', act_o)
    ts.append((time.time()-t0)*1000)
import numpy as np
ts = np.array(ts[5:])  # skip warmup
print(f'\\nMTP head-like call sequence (5 calls: fc, q, k, v, o):')
print(f'  mean: {ts.mean():.3f}ms, p50: {np.median(ts):.3f}ms')
print(f'  out_fc[0]={out_fc[0]:.4f} (sanity)')
print(f'  out_q[0]={out_q[0]:.4f}, out_o[0]={out_o[0]:.4f}')
print(f'\\nServer log:')
for l in client.get_log(15): print(' ', l, end='')
