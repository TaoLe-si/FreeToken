
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostRegister.restype = ctypes.c_int
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipInit(0); hip.hipSetDevice(0)

def readback(t, nbytes):
    dp = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(t.data_ptr()), 0)
    out = torch.zeros(nbytes//4, dtype=torch.float32).pin_memory()
    hip.hipHostRegister(ctypes.c_void_p(out.data_ptr()), nbytes, 0)
    rc2 = hip.hipMemcpy(ctypes.c_void_p(out.data_ptr()), dp, ctypes.c_size_t(nbytes), 2)
    return rc, rc2, out.numpy()

# 1) 256MB 连续 pinned，读首尾
big = torch.zeros(256*1024*1024//4, dtype=torch.float32).pin_memory()
big[0] = 42.0; big[-1] = 77.0
rc, rc2, o = readback(big, 4096)
print("256MB head: reg was 0, memcpy rc=", rc2, "out[0]=", o[0], "expect 42")
# 读尾部: 别名 + 偏移
dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(big.data_ptr()), 0)
out2 = torch.zeros(4, dtype=torch.float32).pin_memory()
hip.hipHostRegister(ctypes.c_void_p(out2.data_ptr()), 16, 0)
rc3 = hip.hipMemcpy(ctypes.c_void_p(out2.data_ptr()), ctypes.c_void_p(dp.value + big.numel()*4 - 16), 16, 2)
print("256MB tail via alias+offset: rc=", rc3, "out[0]=", out2.numpy()[0], "expect 77")

# 2) 子范围注册: 大分配的中间偏移视图
base = torch.zeros(64*1024*1024//4, dtype=torch.float32).pin_memory()  # 64MB
base[1024] = 555.0
off_t = base[1024:1024+256]  # 偏移视图
rc4 = hip.hipHostRegister(ctypes.c_void_p(off_t.data_ptr()), 256*4, 0)
dp2 = ctypes.c_void_p()
rc5 = hip.hipHostGetDevicePointer(ctypes.byref(dp2), ctypes.c_void_p(off_t.data_ptr()), 0)
out3 = torch.zeros(256, dtype=torch.float32).pin_memory()
hip.hipHostRegister(ctypes.c_void_p(out3.data_ptr()), 256*4, 0)
rc6 = hip.hipMemcpy(ctypes.c_void_p(out3.data_ptr()), dp2, 256*4, 2)
print("subrange: reg rc=", rc4, "getptr rc=", rc5, "memcpy rc=", rc6, "out[0]=", out3.numpy()[0], "expect 555")
