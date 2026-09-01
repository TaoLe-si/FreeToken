
import ctypes, os, sys
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_hostmalloc.restype = ctypes.c_void_p
dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
hip.hipInit(0); hip.hipSetDevice(0)
h = dll.igpu_hostmalloc(64)
buf = (ctypes.c_char * 64).from_address(h)
for i in range(64): buf[i] = bytes([i & 0xFF])
dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(h), 0)
chk_h = dll.igpu_hostmalloc(64)
chk_dp = ctypes.c_void_p()
hip.hipHostGetDevicePointer(ctypes.byref(chk_dp), ctypes.c_void_p(chk_h), 0)
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
rc = hip.hipMemcpy(chk_dp, dp, ctypes.c_size_t(64), 2)
print("memcpy rc:", rc, "match:", bytes(buf) == ctypes.string_at(chk_h, 64))
