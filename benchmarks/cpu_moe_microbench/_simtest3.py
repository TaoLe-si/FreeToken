
import torch, ctypes, numpy as np, os, sys
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
# 父进程: CUDA init
x = torch.randn(8, device="cuda:0"); torch.cuda.synchronize()
print("parent cuda ok", flush=True)
pid = os.fork() if hasattr(os, "fork") else None
# Windows 没有 fork! 引擎是 spawn 不是 fork (multiprocessing spawn_main)
# spawn = 全新进程, 重新 import... 那 scheduler 里 CUDA/HIP 顺序由 engine 代码决定
# scheduler: engine init -> torch cuda 大量使用 -> igpu executor (HIP init) -> decode
# 与 simtest2 相同顺序... 剩余差异: FT_KV 等 torch 大量 cudaMalloc + graphs?
# 直接测: 分配 2GB CUDA 显存 + cublas 等后再 HIP init
import torch
a = torch.randn(1024, 1024, device="cuda:0")
b = a @ a  # cublas init
big = [torch.randn(100*1024*1024//4, device="cuda:0") for _ in range(16)]  # 1.6GB
torch.cuda.synchronize()
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
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
