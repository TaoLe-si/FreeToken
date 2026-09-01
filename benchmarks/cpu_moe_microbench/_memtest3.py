
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipHostMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipInit(0); hip.hipSetDevice(0)

def readback_reg(t, nbytes):
    dp = ctypes.c_void_p()
    rc0 = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(t.data_ptr()), 0)
    out = torch.zeros(1, dtype=torch.float32).pin_memory()
    hip.hipHostRegister(ctypes.c_void_p(out.data_ptr()), 4, 0)
    rc2 = hip.hipMemcpy(ctypes.c_void_p(out.data_ptr()), dp, 4, 2)
    return rc0, rc2, out.numpy()[0]

sizes = [1<<20, 4<<20, 16<<20, 32<<20, 64<<20, 96<<20, 128<<20, 192<<20, 256<<20]
for sz in sizes:
    n = sz // 4
    t = torch.zeros(n, dtype=torch.float32).pin_memory()
    t[0] = 42.0
    rc = hip.hipHostRegister(ctypes.c_void_p(t.data_ptr()), sz, 0)
    if rc != 0:
        print(f"{sz>>20}MB: register rc={rc} FAIL")
        continue
    rc0, rc2, v = readback_reg(t, sz)
    print(f"{sz>>20}MB: reg=0 getptr_rc={rc0} memcpy_rc={rc2} val={v:.0f} {'OK' if v==42.0 else 'FAIL'}")
