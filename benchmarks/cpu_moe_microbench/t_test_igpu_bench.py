
import sys
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
from freetoken.kernel.igpu_fc import IgpuFcClient
import numpy as np
import safetensors.torch
import torch, time
client = IgpuFcClient()
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = state['mtp.fc.weight']
np.random.seed(0)
act_real = np.random.randn(4096).astype(np.float32)
fcW_1 = fcW[0:1].numpy().astype(np.uint32)
act_int = act_real.view(np.int32)

# Warmup
client.forward(fcW_1, act_int)

# 100 iters
N = 100
ts = []
for i in range(N):
    t0 = time.time()
    outv = client.forward(fcW_1, act_int)
    t1 = time.time()
    ts.append((t1-t0)*1000)
ts = np.array(ts)
print(f'N={N} M=1 K=4096 iGPU forward latency (Python overhead + IPC + GPU):')
print(f'  mean: {ts.mean():.3f}ms')
print(f'  p50:  {np.median(ts):.3f}ms')
print(f'  p99:  {np.percentile(ts, 99):.3f}ms')
print(f'  outv[0]={outv[0]:.4f} (sanity)')
print(f'  Server log: {client.get_log(3)}')
