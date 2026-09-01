
import sys, time, os, threading
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import numpy as np
import torch, safetensors.torch, glob
from freetoken.kernel.igpu_fc import IgpuMultiClient

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

# Use 3 separate server instances for parallel calls (3 GPU queues serialize otherwise)
# In production we'd want 1 server with parallel dispatch queue
servers = [IgpuMultiClient() for _ in range(3)]
for i, s in enumerate(servers):
    s.load('fc', fc_w[0:1].numpy().astype('uint32'), 4096)
    s.load('q', q_w[0:1].numpy().astype('uint32'), 2048)
    s.load('k', k_w[0:1].numpy().astype('uint32'), 2048)
    s.load('v', v_w[0:1].numpy().astype('uint32'), 2048)
    s.load('o', o_w[0:1].numpy().astype('uint32'), 4096)

act_fc = np.random.randn(4096).astype('float32')
act_attn = np.random.randn(2048).astype('float32')

# Warmup
for s in servers:
    s.call('fc', act_fc.view(np.int32))

N = 100

# 1) Single server, sequential calls
ts1 = []
for _ in range(N):
    t0 = time.time()
    servers[0].call('fc', act_fc.view(np.int32))
    servers[0].call('q', act_attn.view(np.int32))
    servers[0].call('k', act_attn.view(np.int32))
    servers[0].call('v', act_attn.view(np.int32))
    servers[0].call('o', act_fc.view(np.int32))
    ts1.append((time.time()-t0)*1000)

# 2) 3 servers, parallel via threads (q+k+v parallel; fc and o sequential)
ts2 = []
for _ in range(N):
    t0 = time.time()
    # Issue q,k,v in parallel
    out_fc = servers[0].call('fc', act_fc.view(np.int32))
    out_o = servers[0].call('o', act_fc.view(np.int32))
    # q/k/v in parallel
    results = [None, None, None]
    def call_qkv(idx, name, results, results_idx):
        results[results_idx] = servers[idx%3].call(name, act_attn.view(np.int32))
    t1 = [threading.Thread(target=call_qkv, args=(0, 'q', results, 0)),
          threading.Thread(target=call_qkv, args=(1, 'k', results, 1)),
          threading.Thread(target=call_qkv, args=(2, 'v', results, 2))]
    for t1_th in t1: t1_th.start()
    for t1_th in t1: t1_th.join()
    ts2.append((time.time()-t0)*1000)

import numpy as np
ts1 = np.array(ts1[10:]); ts2 = np.array(ts2[10:])
print(f'\\nMTP head call sequence (5 iGPU calls, 3 parallel servers):')
print(f'  Single server sequential:')
print(f'    mean: {ts1.mean():.3f}ms, p50: {np.median(ts1):.3f}ms')
print(f'  3 servers parallel (q/k/v parallel):')
print(f'    mean: {ts2.mean():.3f}ms, p50: {np.median(ts2):.3f}ms')
print(f'  Speedup: {ts1.mean()/ts2.mean():.2f}x')

# Server GPU dispatch times (from server log)
print(f'\\nServer log:')
for s in servers[:1]:
    for l in s.get_log(15): print(' ', l, end='')
