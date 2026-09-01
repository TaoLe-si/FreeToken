
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipInit(0); hip.hipSetDevice(0)

# 256MB, 按 64MB 切块注册每块
SZ = 256 << 20; CH = 64 << 20
big = torch.zeros(SZ // 4, dtype=torch.float32).pin_memory()
big[0] = 42.0
big[SZ//4 - 1] = 77.0
# 逐 64MB 块注册
n_chunks = SZ // CH
dev_ptrs = []
for c in range(n_chunks):
    p = big.data_ptr() + c * CH
    rc = hip.hipHostRegister(ctypes.c_void_p(p), CH, 0)
    dp = ctypes.c_void_p()
    hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(p), 0)
    dev_ptrs.append((rc, dp.value))
print("chunk reg results:", [(rc, hex(dp) if dp else None) for rc, dp in dev_ptrs])

# 读首块头
out = torch.zeros(1, dtype=torch.float32).pin_memory()
hip.hipHostRegister(ctypes.c_void_p(out.data_ptr()), 4, 0)
rc = hip.hipMemcpy(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(dev_ptrs[0][1]), 4, 2)
print("head val:", out.numpy()[0], "expect 42, rc", rc)
# 读末块尾
rc = hip.hipMemcpy(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(dev_ptrs[-1][1] + CH - 4), 4, 2)
print("tail val:", out.numpy()[0], "expect 77, rc", rc)
