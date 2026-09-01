
import torch
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
torch.cuda.init()
p = torch.cuda.get_device_properties(0)
print("CUDA dev0:", p.name, str(p.total_memory//2**20)+"MB")
x = torch.randn(1024, device="cuda"); y = (x*x).sum().item()
print("CUDA kernel OK sum=", round(y,1))

import ctypes
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
hip.hipInit.argtypes = [ctypes.c_int]
hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
hip.hipGetDeviceProperties.restype = ctypes.c_int
print("hipInit rc=", hip.hipInit(0))
n = ctypes.c_int(0)
print("hipGetDeviceCount rc=", hip.hipGetDeviceCount(ctypes.byref(n)), "n=", n.value)
class P(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char*256), ("totalGlobalMem", ctypes.c_size_t),
                ("multiProcessorCount", ctypes.c_int), ("isIntegrated", ctypes.c_int),
                ("canMapHostMemory", ctypes.c_int), ("gcnArchName", ctypes.c_char*256)]
prop = P()
r3 = hip.hipGetDeviceProperties(ctypes.byref(prop), 0)
if r3 == 0:
    print("HIP dev0:", prop.name.decode(), str(prop.totalGlobalMem//2**20)+"MB CU", prop.multiProcessorCount, prop.gcnArchName.decode(), "integrated", prop.isIntegrated)
print("COEXISTENCE:", "OK" if n.value > 0 else "FAIL")
