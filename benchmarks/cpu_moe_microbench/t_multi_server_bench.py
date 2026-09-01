
import sys, time, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import numpy as np
import torch, safetensors.torch, glob
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuMultiClient

# Use original stateless server (one big payload per call - faster)
state_path = r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP'
files = sorted(glob.glob(state_path + '/model-*.safetensors'))
all_state = {}
for p in files:
    all_state.update(safetensors.torch.load_file(p))
fc_w = all_state['mtp.fc.weight']
q_w = all_state['mtp.layers.0.self_attn.q_proj.weight']
k_w = all_state['mtp.layers.0.self_attn.k_proj.weight']
v_w = all_state['mtp.layers.0.self_attn.v_proj.weight']
o_w = all_state['mtp.layers.0.self_attn.o_proj.weight']

# Use stateless client (single big call per op)
client = IgpuFcClient()
act_fc = np.random.randn(4096).astype('float32')
act_attn = np.random.randn(2048).astype('float32')

# Warmup
client.forward(fc_w[0:1].numpy().astype('uint32'), act_fc.view(np.int32))
client.forward(q_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
client.forward(k_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
client.forward(v_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
client.forward(o_w[0:1].numpy().astype('uint32'), act_fc.view(np.int32))

# Bench MTP head sequence (5 calls)
N = 100
ts = []
for _ in range(N):
    t0 = time.time()
    out_fc = client.forward(fc_w[0:1].numpy().astype('uint32'), act_fc.view(np.int32))
    out_q = client.forward(q_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
    out_k = client.forward(k_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
    out_v = client.forward(v_w[0:1].numpy().astype('uint32'), act_attn.view(np.int32))
    out_o = client.forward(o_w[0:1].numpy().astype('uint32'), act_fc.view(np.int32))
    ts.append((time.time()-t0)*1000)

# Now compare to using multi server (sticky)
client2 = IgpuMultiClient()
client2.load('fc', fc_w[0:1].numpy().astype('uint32'), 4096)
client2.load('q', q_w[0:1].numpy().astype('uint32'), 2048)
client2.load('k', k_w[0:1].numpy().astype('uint32'), 2048)
client2.load('v', v_w[0:1].numpy().astype('uint32'), 2048)
client2.load('o', o_w[0:1].numpy().astype('uint32'), 4096)
client2.call('fc', act_fc.view(np.int32))
client2.call('q', act_attn.view(np.int32))
client2.call('k', act_attn.view(np.int32))
client2.call('v', act_attn.view(np.int32))
client2.call('o', act_fc.view(np.int32))

ts2 = []
for _ in range(N):
    t0 = time.time()
    client2.call('fc', act_fc.view(np.int32))
    client2.call('q', act_attn.view(np.int32))
    client2.call('k', act_attn.view(np.int32))
    client2.call('v', act_attn.view(np.int32))
    client2.call('o', act_fc.view(np.int32))
    ts2.append((time.time()-t0)*1000)

import numpy as np
ts = np.array(ts[10:])
ts2 = np.array(ts2[10:])
print(f'\\nMTP head call sequence (5 iGPU calls, real 35B weights):')
print(f'  Stateless (re-uploads weight each call):')
print(f'    mean: {ts.mean():.3f}ms, p50: {np.median(ts):.3f}ms')
print(f'  Sticky (weight pre-loaded):')
print(f'    mean: {ts2.mean():.3f}ms, p50: {np.median(ts2):.3f}ms')
print(f'  Speedup: {ts.mean()/ts2.mean():.2f}x')
print(f'\\nPer-call latency (5 calls each):')
print(f'  Stateless: {ts.mean()/5:.3f}ms/call')
print(f'  Sticky:    {ts2.mean()/5:.3f}ms/call')
print(f'\\nServer log:')
for l in client2.get_log(10): print(' ', l, end='')
