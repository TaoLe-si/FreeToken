import sys, os
sys.path.insert(0, r'E:\FreeToken\python')
# 检查 CUDA_HOME 设置
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'unset'))
import torch
from freetoken.kernel.index import indexing
w = torch.randn(1024, 256, dtype=torch.float16, device='cuda')
idx = torch.randint(0, 1024, (8,), device='cuda', dtype=torch.long)
out = torch.empty(8, 256, dtype=torch.float16, device='cuda')
result = indexing(w, idx, output=out)
print('indexing OK, shape:', result.shape)