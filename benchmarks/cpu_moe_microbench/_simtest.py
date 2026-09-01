
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
# 模拟引擎: 先 CUDA init (RTX 4070)
x = torch.randn(8, device="cuda:0")
torch.cuda.synchronize()
print("cuda ok:", x.sum().item() != 0)
# HIP init
dll.igpu_init.restype = ctypes.c_int
print("igpu_init:", dll.igpu_init())
dll.igpu_hostmalloc.restype = ctypes.c_void_p
dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
# hidden 8KB hostMalloc
h = dll.igpu_hostmalloc(8192)
hid = np.arange(2048, dtype=np.float32) * 0.01
ctypes.memmove(ctypes.c_void_p(h), hid.ctypes.data, 8192)
print("host readback:", np.frombuffer(ctypes.string_at(ctypes.c_void_p(h), 32), dtype=np.float32)[:4])
dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(h), 0)
print("host ptr:", hex(h), "alias:", hex(dp.value or 0))
# hipMemcpy 读回
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
chk_h = dll.igpu_hostmalloc(64)
chk_dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(chk_dp), ctypes.c_void_p(chk_h), 0)
rc = hip.hipMemcpy(chk_dp, dp, ctypes.c_size_t(32), 2)
val = np.frombuffer(ctypes.string_at(ctypes.c_void_p(chk_h), 32), dtype=np.float32)
print("memcpy rc:", rc, "alias readback:", val[:4])
