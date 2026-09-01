import sys
sys.path.insert(0, r"E:\FreeToken\python")
from freetoken.kernel.pinned import _load_pinned_extension, alloc_pinned_tensor
ext = _load_pinned_extension()
print("ext ok:", ext)
t = alloc_pinned_tensor(4, dtype=__import__("torch").int64)
t[0] = 42
print("alloc ok:", t[0].item())
