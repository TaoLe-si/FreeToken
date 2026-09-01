import sys, os
sys.path.insert(0, r'E:\FreeToken\python')
import torch
from freetoken.kernel.store import store_cache
k_cache = torch.zeros(10, 2, 4, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(10, 2, 4, device='cuda', dtype=torch.float16)
indices = torch.tensor([0,2,5], device='cuda', dtype=torch.long)
k = torch.randn(3, 2, 4, device='cuda', dtype=torch.float16)
v = torch.randn(3, 2, 4, device='cuda', dtype=torch.float16)
store_cache(k_cache, v_cache, indices, k, v)
print('store OK')
print('k_cache[0]:', k_cache[0])