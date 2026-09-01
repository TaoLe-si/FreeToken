import ctypes, sys
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
print("dev hr=%08X" % (hr & 0xFFFFFFFF))
# GetDeviceRemovedReason = 37
GetDRR = vfn(dev.value, 37, ctypes.HRESULT)
print("GetDeviceRemovedReason:", hex(GetDRR(dev.value) & 0xFFFFFFFF))
# CreateHeap = 28（验证 vtable 区域）D3D12_HEAP_DESC{SizeInBytes u64, Properties(20), Alignment u64, Flags u32}
class D3D12_HEAP_PROPERTIES(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("CPUPageProperty", ctypes.c_int), ("MemoryPoolPreference", ctypes.c_int), ("CreationNodeMask", c_uint32), ("VisibleNodeMask", c_uint32)]
class D3D12_HEAP_DESC(ctypes.Structure):
    _fields_ = [("SizeInBytes", ctypes.c_uint64), ("Properties", D3D12_HEAP_PROPERTIES), ("Alignment", ctypes.c_uint64), ("Flags", c_uint32)]
print("HEAP_DESC sizeof:", ctypes.sizeof(D3D12_HEAP_DESC), "expect 40")
hd = D3D12_HEAP_DESC(SizeInBytes=4 << 20, Properties=D3D12_HEAP_PROPERTIES(Type=1), Alignment=0, Flags=0)
heap = c_void_p()
hr = vfn(dev.value, 28, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_DESC), ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(dev.value, ctypes.byref(hd), ctypes.byref(IID_H), ctypes.byref(heap))
print("CreateHeap(28):", hex(hr & 0xFFFFFFFF), "heap:", hex(heap.value or 0))
# CreateCommittedResource = 27（完整参数）
class DXGI_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", c_uint32), ("Quality", c_uint32)]
class D3D12_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [("Dimension", ctypes.c_int), ("Alignment", ctypes.c_uint64), ("Width", ctypes.c_uint64), ("Height", c_uint32),
                ("DepthOrArraySize", ctypes.c_uint16), ("MipLevels", ctypes.c_uint16), ("Format", ctypes.c_int),
                ("SampleDesc", DXGI_SAMPLE_DESC), ("Layout", ctypes.c_int), ("Flags", c_uint32)]
hp = D3D12_HEAP_PROPERTIES(Type=1, CPUPageProperty=0, MemoryPoolPreference=0, CreationNodeMask=0, VisibleNodeMask=0)
rd = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=64, Height=1, DepthOrArraySize=1, MipLevels=1,
                         Format=0, SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=0, Flags=0x2)
print("RESOURCE_DESC sizeof:", ctypes.sizeof(D3D12_RESOURCE_DESC), "expect 56")
res = c_void_p()
hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
         ctypes.c_int, c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
    dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), 0x8, None, ctypes.byref(IID_R), ctypes.byref(res))
print("CreateCommittedResource(27) w=64:", hex(hr & 0xFFFFFFFF))
# 试 InitialState=0
hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
         ctypes.c_int, c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
    dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), 0, None, ctypes.byref(IID_R), ctypes.byref(res))
print("CreateCommittedResource(27) state=COMMON:", hex(hr & 0xFFFFFFFF))
# 试 CreateHeap 布局验证后 PlacedResource？——先看 heap 是否成功
print("done")

# 追加：pClearValue 非 NULL + PlacedResource
class D3D12_CLEAR_VALUE(ctypes.Structure):
    _fields_ = [("Format", ctypes.c_int), ("Color", ctypes.c_float * 4)]
cv = D3D12_CLEAR_VALUE(Format=0, Color=(0.0, 0.0, 0.0, 0.0))
res2 = c_void_p()
try:
    hr = vfn(dev.value, 27, ctypes.HRESULT, ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
             ctypes.c_int, ctypes.POINTER(D3D12_CLEAR_VALUE), ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
        dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), 0x8, ctypes.byref(cv), ctypes.byref(IID_R), ctypes.byref(res2))
    print("CreateCommittedResource(27) clear!=NULL:", hex(hr & 0xFFFFFFFF))
except OSError as e:
    print("CreateCommittedResource(27) clear!=NULL EXC:", e)
# PlacedResource(29)
res3 = c_void_p()
try:
    hr = vfn(dev.value, 29, ctypes.HRESULT, c_void_p, ctypes.c_uint64, ctypes.POINTER(D3D12_RESOURCE_DESC),
             ctypes.c_int, ctypes.POINTER(D3D12_CLEAR_VALUE), ctypes.POINTER(GUID), ctypes.POINTER(c_void_p))(
        dev.value, heap.value, 0, ctypes.byref(rd), 0x8, ctypes.byref(cv), ctypes.byref(IID_R), ctypes.byref(res3))
    print("CreatePlacedResource(29) clear!=NULL:", hex(hr & 0xFFFFFFFF))
except OSError as e:
    print("CreatePlacedResource(29) EXC:", e)
