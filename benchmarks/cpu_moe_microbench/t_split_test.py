import sys
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig
cfg = MtpHeadConfig(head_dim=256, num_qo_heads=16, num_kv_heads=2)
print('split_sizes =', cfg.split_sizes if hasattr(cfg, 'split_sizes') else 'N/A')
qo = 16*256; kv = 2*256
print(f'qo_dim = {qo}, kv_dim = {kv}, sum = {qo*2 + kv + kv}')
import numpy as np
v = np.zeros(9216, dtype=np.float32)
print(f'qkv shape: {v.shape}')
import torch
t = torch.from_numpy(v).view(1, -1)
print(f'qkv tensor shape: {t.shape}')
qg, k, v2 = torch.split(t, [8192, 512, 512], dim=-1)
print(f'qg {qg.shape}, k {k.shape}, v2 {v2.shape}')
