import ctypes
d3d12 = ctypes.WinDLL("d3d12.dll")
c_void_p = ctypes.c_void_p; c_uint32 = ctypes.c_uint32; c_uint64 = ctypes.c_uint64
class GUID(ctypes.Structure):
    _fields_ = [("Data1", c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_uint8 * 8)]
    @classmethod
    def from_str(cls, s):
        s = s.strip("{}"); parts = s.split("-")
        g = cls(); g.Data1 = int(parts[0], 16); g.Data2 = int(parts[1], 16); g.Data3 = int(parts[2], 16)
        g.Data4 = (ctypes.c_uint8 * 8).from_buffer_copy(bytes.fromhex(parts[3] + parts[4]))
        return g
IID_D = GUID.from_str("189819F1-1DB6-4B57-BE54-1821339B85F7")
IID_R = GUID.from_str("696442BE-A72E-4059-BC79-5B5C98040FAD")
IID_H = GUID.from_str("6B3B2502-6E51-45B3-90EE-9884265E8DF3")
def vfn(obj, idx, restype, *argtypes):
    vp = c_void_p(); ctypes.memmove(ctypes.byref(vp), obj, 8)
    arr = (c_void_p * (idx + 1)).from_address(vp.value)
    fn = arr[idx]
    if not fn: raise RuntimeError("vtbl[%d] NULL" % idx)
    return ctypes.cast(fn, ctypes.CFUNCTYPE(restype, c_void_p, *argtypes))
d3d12.D3D12CreateDevice.restype = ctypes.HRESULT
d3d12.D3D12CreateDevice.argtypes = [c_void_p, ctypes.c_int, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p)]
dev = c_void_p()
hr = d3d12.D3D12CreateDevice(None, 0xC000, ctypes.byref(IID_D), ctypes.byref(dev))
class D3D12_HEAP_PROPERTIES(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("CPUPageProperty", ctypes.c_int), ("MemoryPoolPreference", ctypes.c_int), ("CreationNodeMask", c_uint32), ("VisibleNodeMask", c_uint32)]
class DXGI_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", c_uint32), ("Quality", c_uint32)]
class D3D12_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [("Dimension", ctypes.c_int), ("Alignment", ctypes.c_uint64), ("Width", ctypes.c_uint64), ("Height", c_uint32),
                ("DepthOrArraySize", ctypes.c_uint16), ("MipLevels", ctypes.c_uint16), ("Format", ctypes.c_int),
                ("SampleDesc", DXGI_SAMPLE_DESC), ("Layout", ctypes.c_int), ("Flags", c_uint32)]
class D3D12_CLEAR_VALUE(ctypes.Structure):
    _fields_ = [("Format", ctypes.c_int), ("Color", ctypes.c_float * 4)]
class D3D12_HEAP_DESC(ctypes.Structure):
    _fields_ = [("SizeInBytes", ctypes.c_uint64), ("Properties", D3D12_HEAP_PROPERTIES), ("Alignment", ctypes.c_uint64), ("Flags", c_uint32)]
hp = D3D12_HEAP_PROPERTIES(Type=1, CPUPageProperty=0, MemoryPoolPreference=0, CreationNodeMask=0, VisibleNodeMask=0)
rd = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=64, Height=1, DepthOrArraySize=1, MipLevels=1, Format=0, SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=0, Flags=0x2)
cv = D3D12_CLEAR_VALUE(Format=0, Color=(0.0, 0.0, 0.0, 0.0))
hd = D3D12_HEAP_DESC(SizeInBytes=4 << 20, Properties=hp, Alignment=0, Flags=0)
heap = c_void_p()
hr = vfn(dev.value, 28, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_DESC), ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(dev.value, ctypes.byref(hd), ctypes.byref(IID_H), ctypes.byref(heap))
print("CreateHeap:", hex(hr & 0xFFFFFFFF), "heap:", hex(heap.value or 0))
res = c_void_p()
for label, state, clr in [("state=UAV clear=NULL", 0x8, None), ("state=UAV clear=CV", 0x8, ctypes.byref(cv)), ("state=COMMON clear=NULL", 0, None)]:
    try:
        hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
                 ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
            dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), state, clr, ctypes.byref(IID_R), ctypes.byref(res))
        print("CCR(27)", label, ":", hex(hr & 0xFFFFFFFF))
    except OSError as e:
        print("CCR(27)", label, "EXC:", e)
res3 = c_void_p()
for label, state, clr in [("state=UAV clear=CV", 0x8, ctypes.byref(cv)), ("state=COMMON clear=NULL", 0, None)]:
    try:
        hr = vfn(dev.value, 29, ctypes.HRESULT, c_void_p, ctypes.c_uint64, ctypes.POINTER(D3D12_RESOURCE_DESC),
                 ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
            dev.value, heap.value, 0, ctypes.byref(rd), state, clr, ctypes.byref(IID_R), ctypes.byref(res3))
        print("CPR(29)", label, ":", hex(hr & 0xFFFFFFFF))
    except OSError as e:
        print("CPR(29)", label, "EXC:", e)
print("done")

# 64 位整数参数变体（x64 传参高位清零）
res64 = c_void_p()
try:
    hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), ctypes.c_uint64, ctypes.POINTER(D3D12_RESOURCE_DESC),
             ctypes.c_uint64, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
        dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), 0x8, None, ctypes.byref(IID_R), ctypes.byref(res64))
    print("CCR(27) 64bit-args:", hex(hr & 0xFFFFFFFF))
except OSError as e:
    print("CCR(27) 64bit-args EXC:", e)
# 用 HeapOffset=0 的 64 位变体
res65 = c_void_p()
try:
    hr = vfn(dev.value, 29, ctypes.HRESULT, c_void_p, ctypes.c_uint64, ctypes.POINTER(D3D12_RESOURCE_DESC),
             ctypes.c_uint64, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
        dev.value, heap.value, 0, ctypes.byref(rd), 0x8, None, ctypes.byref(IID_R), ctypes.byref(res65))
    print("CPR(29) 64bit-args:", hex(hr & 0xFFFFFFFF))
except OSError as e:
    print("CPR(29) 64bit-args EXC:", e)

# Layout=ROW_MAJOR(1) 变体
rd1 = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=64, Height=1, DepthOrArraySize=1, MipLevels=1, Format=0, SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=1, Flags=0x2)
res7 = c_void_p()
try:
    hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
             ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
        dev.value, ctypes.byref(hp), 0, ctypes.byref(rd1), 0x8, None, ctypes.byref(IID_R), ctypes.byref(res7))
    print("CCR(27) Layout=ROW_MAJOR:", hex(hr & 0xFFFFFFFF))
except OSError as e:
    print("CCR(27) Layout=ROW_MAJOR EXC:", e)
