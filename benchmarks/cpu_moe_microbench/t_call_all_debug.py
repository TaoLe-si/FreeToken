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

client = IgpuMultiClient()
client.load('fc', fc_w[0:1].astype('uint32'), 4096)
# Test with a specific non-zero act
act = np.full(4096, 0.05, dtype='float32').view(np.int32)
print(f'act bytes head: {act[:3].tobytes().hex()}')
res = client.call('fc', act)
print(f'fc result[0] = {res[0]}')
res = client.call_all([act])
print(f'ALL result[fc] = {res["fc"][0]}')
