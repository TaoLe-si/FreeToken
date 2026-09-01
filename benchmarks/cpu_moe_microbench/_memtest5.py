
import ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipHostMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipInit(0); hip.hipSetDevice(0)
hipMemmapAllocationFailure = 0

for SZ_MB in (64, 128, 256):
    SZ = SZ_MB << 20
    h = ctypes.c_void_p()
    rc = hip.hipHostMalloc(ctypes.byref(h), ctypes.c_size_t(SZ), 0)  # Mapped 默认
    if rc != 0:
        print(f"{SZ_MB}MB hipHostMalloc rc={rc} FAIL")
        continue
    dp = ctypes.c_void_p()
    rc2 = hip.hipHostGetDevicePointer(ctypes.byref(dp), h, 0)
    # 写标记
    import ctypes as c
    buf = (c.c_char * 8).from_address(h.value)
    buf[0] = b"\x2a"
    # 读回 via alias
    out = ctypes.create_string_buffer(8)
    rc3 = hip.hipMemcpy(out, dp, 8, 2)
    val = int.from_bytes(out.raw[:4], "little")
    print(f"{SZ_MB}MB hostMalloc: rc={rc} getptr={rc2} memcpy={rc3} val={val} expect 42")
