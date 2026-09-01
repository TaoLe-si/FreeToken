"""Full MTP head with iGPU offload, batched qkv (M=3 not M=1)."""
import sys, time, os
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

fc_w = all_state['mtp.fc.weight'].numpy()
q_w = all_state['mtp.layers.0.self_attn.q_proj.weight'].numpy()
k_w = all_state['mtp.layers.0.self_attn.k_proj.weight'].numpy()
v_w = all_state['mtp.layers.0.self_attn.v_proj.weight'].numpy()
o_w = all_state['mtp.layers.0.self_attn.o_proj.weight'].numpy()

# Batched: stack q,k,v vertically (M=3, K=2048)
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]
q_out_dim = q_w.shape[0]  # 8192
k_out_dim = k_w.shape[0]  # 512
v_out_dim = v_w.shape[0]  # 512
print(f'qkv_w: {qkv_w.shape} = [q={q_w.shape[0]} | k={k_w.shape[0]} | v={v_w.shape[0]}]')

client = IgpuMultiClient()
print('Loading weights...')
client.load('fc', fc_w[0:1].astype('uint32'), 4096)
# Batched: load all 3 as one M=3 weight
# Wait, the server currently only supports M=1. Let me test M=3 directly
# Actually I can do 3 separate M=1 calls or 1 M=3 call.
# Try M=1 first (simpler)
client.load('q', q_w[0:1].astype('uint32'), 2048)
client.load('k', k_w[0:1].astype('uint32'), 2048)
client.load('v', v_w[0:1].astype('uint32'), 2048)
client.load('o', o_w[0:1].astype('uint32'), 4096)

act_fc = np.random.randn(4096).astype('float32')
act_attn = np.random.randn(2048).astype('float32')

# Warmup
for _ in range(3):
    client.call('fc', act_fc.view(np.int32))
    client.call('q', act_attn.view(np.int32))
    client.call('k', act_attn.view(np.int32))
    client.call('v', act_attn.view(np.int32))
    client.call('o', act_fc.view(np.int32))
torch.cuda.synchronize()

# Bench: 5 separate calls (fc + q + k + v + o)
N = 50
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
print(f'\\n5 sequential iGPU calls: {ts.mean():.3f}ms (p50 {np.median(ts):.3f}ms)')
print(f'  per call: {ts.mean()/5:.3f}ms')
