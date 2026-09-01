
import torch, ctypes, struct

# 1) torch 分配 pinned host tensor（模拟 engine 的 bank 分配）
bank = torch.empty(256 * 1024, dtype=torch.float32, pin_memory=True)
bank.fill_(42.0)
ptr = bank.data_ptr()
nbytes = bank.nelement() * bank.element_size()
print(f"torch pinned: ptr=0x{ptr:x} nbytes={nbytes} val0={bank[0].item()}")

# 2) 加载 HIP runtime
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipInit.argtypes = [ctypes.c_int]
hip.hipSetDevice.argtypes = [ctypes.c_int]
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostRegister.restype = ctypes.c_int
hip.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
hip.hipHostGetDevicePointer.restype = ctypes.c_int

print("hipInit:", hip.hipInit(0))
print("hipSetDevice:", hip.hipSetDevice(0))

# 3) hipHostRegister 同一指针
flags = 0  # hipHostRegisterDefault
r = hip.hipHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags))
print(f"hipHostRegister(cudaHostAlloc ptr): rc={r} {'OK' if r==0 else 'FAIL'}")

if r == 0:
    # 4) 获取 device pointer（zero-copy）
    devptr = ctypes.c_void_p(0)
    r2 = hip.hipHostGetDevicePointer(ctypes.byref(devptr), ctypes.c_void_p(ptr), 0)
    print(f"hipHostGetDevicePointer: rc={r2} devptr=0x{devptr.value or 0:x}")
    print("ZERO-COPY BANK REGISTRATION: OK" if r2 == 0 else "FAIL")
else:
    # 试 hipHostRegisterMapped flag
    r3 = hip.hipHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(2))
    print(f"retry with Mapped flag: rc={r3}")
    print("ZERO-COPY BANK REGISTRATION: FAIL" if r3 != 0 else "OK with Mapped")
