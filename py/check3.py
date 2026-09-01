import sys
sys.path.insert(0, r"E:\FreeToken\python")
import freetoken.kernel.pinned as p
print("pinned:", p.__file__)
from freetoken.kernel import _cpu_moe, _pinned_tensor
print("_cpu_moe:", _cpu_moe.__file__)
print("_pinned_tensor ok")
from freetoken.kernel.pinned import alloc_pinned_tensor
import torch
t = alloc_pinned_tensor(4, dtype=torch.int64)
print("alloc ok")
