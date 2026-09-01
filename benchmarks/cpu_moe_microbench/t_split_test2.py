import sys
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
import numpy as np

# Simulate what IgpuQkvWrapper does
q = np.random.randn(8192).astype(np.float32)
k = np.random.randn(512).astype(np.float32)
v = np.random.randn(512).astype(np.float32)
qkv = np.concatenate([q, k, v])  # (9216,)
t = torch.from_numpy(qkv.copy()).cuda().to(torch.bfloat16)
print(f't shape: {t.shape}')
v2 = t.view(1, -1)
print(f'v2 shape: {v2.shape}')
# split
qg, k2, v3 = torch.split(v2, [8192, 512, 512], dim=-1)
print(f'qg {qg.shape}, k2 {k2.shape}, v3 {v3.shape}')
