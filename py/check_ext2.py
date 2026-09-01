import sys
sys.path.insert(0, r"E:\FreeToken\python")
import freetoken.kernel.pinned as p
print("pinned module:", p.__file__)
from freetoken.kernel.pinned import alloc_pinned_tensor
t = alloc_pinned_tensor(4, dtype=__import__("torch").int64)
print("pinned ok:", t[0].dtype)
from freetoken.kernel import _cpu_moe
print("_cpu_moe ok:", _cpu_moe.__file__)
