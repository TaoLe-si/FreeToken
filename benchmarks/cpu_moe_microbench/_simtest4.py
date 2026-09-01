
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
x = torch.randn(8, device="cuda:0"); torch.cuda.synchronize()
# 模拟 banks: 2GB cudaHostAlloc pinned
pinned = [torch.zeros(256*1024*1024//4, dtype=torch.float32).pin_memory() for _ in range(8)]
print("pinned banks allocated", flush=True)
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int
print("igpu_init:", dll.igpu_init())
dll.igpu_hostmalloc.restype = ctypes.c_void_p
dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
h = dll.igpu_hostmalloc(8192)
data = np.arange(2048, dtype=np.float32) * 0.01
ctypes.memmove(ctypes.c_void_p(h), data.ctypes.data, 8192)
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(h), 0)
print("host:", hex(h), "alias:", hex(dp.value or 0))
chk_h = dll.igpu_hostmalloc(64)
chk_dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(chk_dp), ctypes.c_void_p(chk_h), 0)
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
rc = hip.hipMemcpy(chk_dp, dp, ctypes.c_size_t(32), 2)
val = np.frombuffer(ctypes.string_at(ctypes.c_void_p(chk_h), 32), dtype=np.float32)
print("memcpy rc:", rc, "readback:", val[:4], "expect", data[:4])
