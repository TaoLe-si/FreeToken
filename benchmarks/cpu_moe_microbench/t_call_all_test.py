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

client = IgpuMultiClient()
print('Pre-loading 5 weights...')
client.load('fc', fc_w[0:1].astype('uint32'), 4096)
client.load('q', q_w[0:1].astype('uint32'), 2048)
client.load('k', k_w[0:1].astype('uint32'), 2048)
client.load('v', v_w[0:1].astype('uint32'), 2048)
client.load('o', o_w[0:1].astype('uint32'), 4096)

# Prepare acts
act_fc = np.random.randn(4096).astype('float32')
act_attn = np.random.randn(2048).astype('float32')
acts = [act_fc.view(np.int32), act_attn.view(np.int32), act_attn.view(np.int32), act_attn.view(np.int32), act_fc.view(np.int32)]

# Warmup
for _ in range(3):
    res = client.call_all(acts)
torch.cuda.synchronize()

# Bench ALL
N = 100
ts = []
for _ in range(N):
    t0 = time.time()
    res = client.call_all(acts)
    ts.append((time.time()-t0)*1000)
import numpy as np
ts = np.array(ts[5:])
print(f'\\nALL (5 iGPU calls in 1 server frame): {ts.mean():.3f}ms (p50 {np.median(ts):.3f}ms)')
print(f'  vs 5 separate calls (5.5ms): speedup {5.5/ts.mean():.2f}x')
print(f'  results: {[(k, v[0]) for k, v in res.items()]}')
