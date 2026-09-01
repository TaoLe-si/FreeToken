
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostRegister.restype = ctypes.c_int
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipInit(0); hip.hipSetDevice(0)

# 测 3 种分配方式
N = 1024
data = np.arange(N, dtype=np.float32)  # 0..1023 可识别

def test(name, t):
    ptr = t.data_ptr()
    rc = hip.hipHostRegister(ctypes.c_void_p(ptr), N*4, 0)
    dp = ctypes.c_void_p()
    rc2 = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(ptr), 0)
    print(f"{name}: reg rc={rc} getptr rc={rc2} dev={hex(dp.value or 0)} host={hex(ptr)} same={dp.value==ptr}")
    return t, dp

# 1) torch pin_memory (cudaHostAlloc default)
t1 = torch.from_numpy(data.copy()).pin_memory()
r1 = test("torch.pin_memory", t1)

# 2) 普通 pageable + hipHostRegister
t2 = torch.from_numpy(data.copy())
r2 = test("pageable+reg", t2)

# 3) 大块 (256MB, 模拟 bank)
big = torch.zeros(256*1024*1024//4, dtype=torch.float32).pin_memory()
big.copy_(torch.arange(big.numel(), dtype=torch.float32))
rc = hip.hipHostRegister(ctypes.c_void_p(big.data_ptr()), big.numel()*4, 0)
print("256MB pin_memory reg rc:", rc)

# 关键测试: HIP kernel 读 registered pinned 内存
# 用 DLL 的 dot kernel 太复杂, 直接用 hipMemcpy D2H 验证设备别名可读性
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
out = np.zeros(N, dtype=np.float32)
t_out = torch.from_numpy(out).pin_memory()
# hipMemcpyDeviceToHost=2, hipMemcpyHostToDevice=1
# device alias -> host
rc = hip.hipMemcpy(t_out.data_ptr(), r1[1].value, N*4, 2)
print("D2H via device alias rc:", rc, "out[0:5]:", t_out.numpy()[:5], "expect 0..4")
print("out[-3:]:", t_out.numpy()[-3:], "expect 1021..1023")
