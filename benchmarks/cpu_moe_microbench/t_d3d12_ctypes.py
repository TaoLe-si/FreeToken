# D3D12 compute 最小测试（ctypes 直调）：写 buffer → dispatch → 读回
# 目的：验证 780M 的 D3D12 驱动是否消除 compute shader 写（对照 LLPC DCE）
import ctypes, struct, time, sys
from ctypes import wintypes
c_uint32 = ctypes.c_uint32; c_uint16 = ctypes.c_uint16; c_uint8 = ctypes.c_uint8
c_int = ctypes.c_int; c_void_p = ctypes.c_void_p; c_size_t = ctypes.c_size_t; c_uint64 = ctypes.c_uint64

d3d12 = ctypes.WinDLL("d3d12.dll")
dxgi = ctypes.WinDLL("dxgi.dll")
kernel32 = ctypes.WinDLL("kernel32.dll")

class GUID(ctypes.Structure):
    _fields_ = [("Data1", c_uint32), ("Data2", c_uint16), ("Data3", c_uint16), ("Data4", c_uint8 * 8)]
    @classmethod
    def from_str(cls, s):
        s = s.strip("{}"); parts = s.split("-")
        g = cls(); g.Data1 = int(parts[0], 16); g.Data2 = int(parts[1], 16); g.Data3 = int(parts[2], 16)
        b = bytes.fromhex(parts[3] + parts[4]); g.Data4 = (c_uint8 * 8).from_buffer_copy(b)
        return g

IID_IDXGIFactory1 = GUID.from_str("770AAE78-F26F-4DBA-A829-253C83D1B387")
IID_ID3D12Device = GUID.from_str("189819F1-1DB6-4B57-BE54-1821339B85F7")
IID_ID3D12CommandQueue = GUID.from_str("0EC870A6-5D7E-4C22-8CFC-5BAAE07616ED")
IID_ID3D12CommandAllocator = GUID.from_str("6102DEE4-AF59-4B09-B999-B44D73F09B24")
IID_ID3D12GraphicsCommandList = GUID.from_str("5B160D0F-AC1B-4185-8BA8-B3AE42A5A455")
IID_ID3D12Fence = GUID.from_str("0A753DCF-C4D8-4B91-ADF6-BE5A60D95A76")
IID_ID3D12DescriptorHeap = GUID.from_str("8EFB471D-616C-4F49-90F7-127BB763FA51")
IID_ID3D12RootSignature = GUID.from_str("C54A6B66-72DF-4EE8-8BE5-A946A1429214")
IID_ID3D12PipelineState = GUID.from_str("765A30F3-F624-4C6F-A828-ACE948622445")
IID_ID3D12Resource = GUID.from_str("696442BE-A72E-4059-BC79-5B5C98040FAD")

class DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", c_uint32), ("DeviceId", c_uint32),
                ("SubSysId", c_uint32), ("Revision", c_uint32), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_uint64 * 2)]

# ---- vtable helper ----
def vfn(obj, idx, restype, *argtypes):
    vp = ctypes.c_void_p()
    ctypes.memmove(ctypes.byref(vp), obj, 8)  # 对象 offset 0 = vtable 指针
    arr = (ctypes.c_void_p * (idx + 1)).from_address(vp.value)
    fn = arr[idx]
    if not fn:
        raise RuntimeError("vtbl[%d] is NULL (obj=%#x vtable=%#x)" % (idx, obj, vp.value))
    return ctypes.cast(fn, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes))

# ---- 1. 默认适配器（跳过 DXGI 枚举）----
amd_adapter = None
print("using default adapter")
# ---- 2. D3D12CreateDevice ----
D3D12CreateDevice = d3d12.D3D12CreateDevice
D3D12CreateDevice.restype = ctypes.HRESULT
D3D12CreateDevice.argtypes = [ctypes.c_void_p, c_int, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
dev = ctypes.c_void_p()
hr = D3D12CreateDevice(amd_adapter, 0xC000, ctypes.byref(IID_ID3D12Device), ctypes.byref(dev))  # D3D_FEATURE_LEVEL_12_0 = 0xC000
assert hr == 0, "D3D12CreateDevice %08X" % hr
print("device created (default adapter), dev=%#x" % dev.value)
dbg = open(r"E:\FreeToken\benchmarks\cpu_moe_microbench\t_d3d12_dbg.txt", "a")
try:
    vp = ctypes.c_void_p(); ctypes.memmove(ctypes.byref(vp), dev.value, 8)
    arr = (ctypes.c_void_p * 12).from_address(vp.value)
    dbg.write("dev vtable=%#x vtbl[0..11]: " % vp.value + str([hex(arr[i]) for i in range(12)]) + "\n")
except Exception as e:
    dbg.write("dev vtbl read err: " + str(e) + "\n")
dbg.flush()

# ---- 3. CommandQueue (COMPUTE=2) ----
class D3D12_COMMAND_QUEUE_DESC(ctypes.Structure):
    _fields_ = [("Type", c_int), ("Priority", c_int), ("Flags", c_uint32), ("NodeMask", c_uint32)]
cqd = D3D12_COMMAND_QUEUE_DESC(Type=2, Priority=0, Flags=0, NodeMask=0)
queue = ctypes.c_void_p()
hr = vfn(dev.value, 8, ctypes.HRESULT, ctypes.POINTER(D3D12_COMMAND_QUEUE_DESC), ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, ctypes.byref(cqd), ctypes.byref(IID_ID3D12CommandQueue), ctypes.byref(queue))
assert hr == 0, "CreateCommandQueue %08X" % hr

N = 1 << 20  # 1M floats
# ---- 4. UAV 资源（默认堆）+ readback ----
class D3D12_HEAP_PROPERTIES(ctypes.Structure):
    _fields_ = [("Type", c_int), ("CPUPageProperty", c_int), ("MemoryPoolPreference", c_int), ("CreationNodeMask", c_uint32), ("VisibleNodeMask", c_uint32)]
class DXGI_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", c_uint32), ("Quality", c_uint32)]
class D3D12_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [("Dimension", c_int), ("Alignment", ctypes.c_uint64), ("Width", ctypes.c_uint64), ("Height", c_uint32),
                ("DepthOrArraySize", ctypes.c_uint16), ("MipLevels", ctypes.c_uint16), ("Format", c_int),
                ("SampleDesc", DXGI_SAMPLE_DESC), ("Layout", c_int), ("Flags", c_uint32)]
CreateCommittedResource = vfn(dev.value, 27, ctypes.HRESULT,
    ctypes.POINTER(D3D12_HEAP_PROPERTIES), c_uint32, ctypes.POINTER(D3D12_RESOURCE_DESC),
    c_int, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
def make_buf(heap_type, state, flags, size):
    hp = D3D12_HEAP_PROPERTIES(Type=heap_type, CPUPageProperty=0, MemoryPoolPreference=0, CreationNodeMask=0, VisibleNodeMask=0)
    rd = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=size, Height=1, DepthOrArraySize=1, MipLevels=1,
                             Format=0, SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=0, Flags=flags)
    res = ctypes.c_void_p()
    hr = CreateCommittedResource(dev.value, ctypes.byref(hp), 0, ctypes.byref(rd), state, None, ctypes.byref(IID_ID3D12Resource), ctypes.byref(res))
    assert hr == 0, "CreateCommittedResource %08X" % hr
    return res
uav = make_buf(1, 0x8, 0x2, N * 4)        # DEFAULT, UNORDERED_ACCESS, ALLOW_UNORDERED_ACCESS
rbuf = make_buf(3, 0x400, 0, N * 4)      # READBACK, COPY_DEST

# ---- 5. 描述符堆 + UAV 视图 ----
class D3D12_DESCRIPTOR_HEAP_DESC(ctypes.Structure):
    _fields_ = [("Type", c_int), ("NumDescriptors", c_uint32), ("Flags", c_uint32), ("NodeMask", c_uint32)]
heap = ctypes.c_void_p()
hdesc = D3D12_DESCRIPTOR_HEAP_DESC(Type=0, NumDescriptors=1, Flags=1, NodeMask=0)
hr = vfn(dev.value, 14, ctypes.HRESULT, ctypes.POINTER(D3D12_DESCRIPTOR_HEAP_DESC), ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, ctypes.byref(hdesc), ctypes.byref(IID_ID3D12DescriptorHeap), ctypes.byref(heap))
assert hr == 0, "CreateDescriptorHeap %08X" % hr
class D3D12_CPU_DESCRIPTOR_HANDLE(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_size_t)]
class D3D12_GPU_DESCRIPTOR_HANDLE(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64)]
cpu_h = D3D12_CPU_DESCRIPTOR_HANDLE()
gpu_h = D3D12_GPU_DESCRIPTOR_HANDLE()
vfn(heap.value, 8, None, ctypes.POINTER(D3D12_CPU_DESCRIPTOR_HANDLE))(heap.value, ctypes.byref(cpu_h))
vfn(heap.value, 9, None, ctypes.POINTER(D3D12_GPU_DESCRIPTOR_HANDLE))(heap.value, ctypes.byref(gpu_h))
class D3D12_BUFFER_UAV(ctypes.Structure):
    _fields_ = [("FirstElement", ctypes.c_uint64), ("NumElements", c_uint32), ("StructureByteStride", c_uint32), ("CounterOffsetInBytes", c_uint32), ("Flags", c_uint32)]
class D3D12_UNORDERED_ACCESS_VIEW_DESC(ctypes.Structure):
    _fields_ = [("Format", c_int), ("ViewDimension", c_int), ("Buffer", D3D12_BUFFER_UAV)]
uavd = D3D12_UNORDERED_ACCESS_VIEW_DESC(Format=0, ViewDimension=1, Buffer=D3D12_BUFFER_UAV(0, N, 4, 0, 0))
vfn(dev.value, 19, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(D3D12_UNORDERED_ACCESS_VIEW_DESC), D3D12_CPU_DESCRIPTOR_HANDLE)(dev.value, uav.value, None, ctypes.byref(uavd), cpu_h)

# ---- 6. 根签名 ----
class D3D12_DESCRIPTOR_RANGE(ctypes.Structure):
    _fields_ = [("RangeType", c_int), ("NumDescriptors", c_uint32), ("BaseShaderRegister", c_uint32), ("RegisterSpace", c_uint32), ("OffsetInDescriptorsFromTableStart", c_uint32)]
class D3D12_ROOT_DESCRIPTOR_TABLE(ctypes.Structure):
    _fields_ = [("NumDescriptorRanges", c_uint32), ("pDescriptorRanges", ctypes.POINTER(D3D12_DESCRIPTOR_RANGE))]
class D3D12_ROOT_PARAMETER(ctypes.Union):
    _fields_ = [("ParameterType", c_int), ("DescriptorTable", D3D12_ROOT_DESCRIPTOR_TABLE)]
class D3D12_ROOT_SIGNATURE_DESC(ctypes.Structure):
    _fields_ = [("NumParameters", c_uint32), ("pParameters", ctypes.POINTER(D3D12_ROOT_PARAMETER)),
                ("NumStaticSamplers", c_uint32), ("pStaticSamplers", c_void_p), ("Flags", c_uint32)]
range1 = D3D12_DESCRIPTOR_RANGE(RangeType=1, NumDescriptors=1, BaseShaderRegister=0, RegisterSpace=0, OffsetInDescriptorsFromTableStart=0)
param = D3D12_ROOT_PARAMETER(ParameterType=0, DescriptorTable=D3D12_ROOT_DESCRIPTOR_TABLE(1, ctypes.pointer(range1)))
rsd = D3D12_ROOT_SIGNATURE_DESC(NumParameters=1, pParameters=ctypes.pointer(param), NumStaticSamplers=0, pStaticSamplers=None, Flags=0)
class ID3DBlob(ctypes.Structure):
    _fields_ = [("vtbl", ctypes.c_void_p)]
blob = ctypes.POINTER(ID3DBlob)()
errblob = ctypes.POINTER(ID3DBlob)()
SerializeRootSignature = d3d12.D3D12SerializeRootSignature
SerializeRootSignature.restype = ctypes.HRESULT
SerializeRootSignature.argtypes = [ctypes.POINTER(D3D12_ROOT_SIGNATURE_DESC), c_int, ctypes.POINTER(ctypes.POINTER(ID3DBlob)), ctypes.POINTER(ctypes.POINTER(ID3DBlob))]
hr = SerializeRootSignature(ctypes.byref(rsd), 1, ctypes.byref(blob), ctypes.byref(errblob))
assert hr == 0, "SerializeRootSignature %08X" % hr
GetBufferPointer = vfn(blob, 3, ctypes.c_void_p)
GetBufferSize = vfn(blob, 4, ctypes.c_size_t)
rs = ctypes.c_void_p()
hr = vfn(dev.value, 16, ctypes.HRESULT, c_uint32, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, 0, GetBufferPointer(blob), GetBufferSize(blob), ctypes.byref(IID_ID3D12RootSignature), ctypes.byref(rs))
assert hr == 0, "CreateRootSignature %08X" % hr
print("root signature created")

# ---- 7. PSO（DXIL）----
class D3D12_SHADER_BYTECODE(ctypes.Structure):
    _fields_ = [("pShaderBytecode", c_void_p), ("BytecodeLength", ctypes.c_size_t)]
class D3D12_CACHED_PIPELINE_STATE(ctypes.Structure):
    _fields_ = [("pCachedBlob", c_void_p), ("CachedBlobSizeInBytes", ctypes.c_size_t)]
class D3D12_COMPUTE_PIPELINE_STATE_DESC(ctypes.Structure):
    _fields_ = [("pRootSignature", c_void_p), ("CS", D3D12_SHADER_BYTECODE), ("NodeMask", c_uint32), ("CachedPSO", D3D12_CACHED_PIPELINE_STATE), ("Flags", c_uint32)]
dxil = open(r"E:\FreeToken\benchmarks\cpu_moe_microbench\t_d3d12.dxil", "rb").read()
buf = ctypes.create_string_buffer(dxil)
csd = D3D12_COMPUTE_PIPELINE_STATE_DESC(pRootSignature=rs.value, CS=D3D12_SHADER_BYTECODE(ctypes.cast(buf, c_void_p), len(dxil)), NodeMask=0, CachedPSO=D3D12_CACHED_PIPELINE_STATE(None, 0), Flags=0)
pso = ctypes.c_void_p()
hr = vfn(dev.value, 11, ctypes.HRESULT, ctypes.POINTER(D3D12_COMPUTE_PIPELINE_STATE_DESC), ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, ctypes.byref(csd), ctypes.byref(IID_ID3D12PipelineState), ctypes.byref(pso))
assert hr == 0, "CreateComputePipelineState %08X" % hr
print("pso created")

# ---- 8. 命令列表 ----
alloc = ctypes.c_void_p()
hr = vfn(dev.value, 9, ctypes.HRESULT, c_int, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, 2, ctypes.byref(IID_ID3D12CommandAllocator), ctypes.byref(alloc))
assert hr == 0, "CreateCommandAllocator %08X" % hr
cl = ctypes.c_void_p()
hr = vfn(dev.value, 12, ctypes.HRESULT, c_uint32, c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, 0, 2, alloc.value, None, ctypes.byref(IID_ID3D12GraphicsCommandList), ctypes.byref(cl))
assert hr == 0, "CreateCommandList %08X" % hr
vfn(cl.value, 9, ctypes.HRESULT, ctypes.c_void_p, ctypes.c_void_p)(cl.value, alloc.value, None)  # Reset
vfn(cl.value, 19, None, ctypes.c_void_p)(cl.value, pso.value)          # SetPipelineState
vfn(cl.value, 23, None, ctypes.c_void_p)(cl.value, rs.value)           # SetComputeRootSignature
heaps = (ctypes.c_void_p * 1)(heap.value)
vfn(cl.value, 22, None, c_uint32, ctypes.POINTER(ctypes.c_void_p))(cl.value, 1, heaps)  # SetDescriptorHeaps
vfn(cl.value, 24, None, c_uint32, D3D12_GPU_DESCRIPTOR_HANDLE)(cl.value, 0, gpu_h)     # SetComputeRootDescriptorTable(0, gpu)
vfn(cl.value, 13, None, c_uint32, c_uint32, c_uint32)(cl.value, N // 256, 1, 1)        # Dispatch
class D3D12_RESOURCE_TRANSITION_BARRIER(ctypes.Structure):
    _fields_ = [("pResource", c_void_p), ("Subresource", c_uint32), ("StateBefore", c_int), ("StateAfter", c_int)]
class D3D12_RESOURCE_BARRIER(ctypes.Structure):
    _fields_ = [("Type", c_int), ("Flags", c_uint32), ("Transition", D3D12_RESOURCE_TRANSITION_BARRIER)]
bar = D3D12_RESOURCE_BARRIER(Type=0, Flags=0, Transition=D3D12_RESOURCE_TRANSITION_BARRIER(uav.value, 0, 0x8, 0x800))
vfn(cl.value, 20, None, c_uint32, ctypes.POINTER(D3D12_RESOURCE_BARRIER))(cl.value, 1, ctypes.byref(bar))  # ResourceBarrier UAV->COPY_SOURCE
vfn(cl.value, 17, None, ctypes.c_void_p, ctypes.c_void_p)(cl.value, rbuf.value, uav.value)  # CopyResource
vfn(cl.value, 8, ctypes.HRESULT)(cl.value)  # Close

# ---- 9. 提交 + fence ----
fence = ctypes.c_void_p()
hr = vfn(dev.value, 36, ctypes.HRESULT, ctypes.c_uint64, c_uint32, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(dev.value, 0, 0, ctypes.byref(IID_ID3D12Fence), ctypes.byref(fence))
assert hr == 0, "CreateFence %08X" % hr
ev = kernel32.CreateEventW(None, False, False, None)
clist = (ctypes.c_void_p * 1)(cl.value)
def submit_and_wait():
    vfn(queue.value, 10, None, c_uint32, ctypes.POINTER(ctypes.c_void_p))(queue.value, 1, clist)  # ExecuteCommandLists
    vfn(queue.value, 14, ctypes.HRESULT, ctypes.c_void_p, ctypes.c_uint64)(queue.value, fence.value, 1)  # Signal
    vfn(fence.value, 8, ctypes.HRESULT, ctypes.c_uint64, ctypes.c_void_p)(fence.value, 1, ev)  # SetEventOnCompletion
    kernel32.WaitForSingleObject(ev, 30000)

# ---- 10. 时间测量（多轮）----
def run_rounds(n):
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter()
        submit_and_wait()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best

# 先跑一轮（正确性）
submit_and_wait()
Map = vfn(rbuf.value, 9, ctypes.HRESULT, c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
ptr = ctypes.c_void_p()
hr = Map(rbuf.value, 0, None, ctypes.byref(ptr))
assert hr == 0, "Map %08X" % hr
raw = ctypes.string_at(ptr.value, 16)
vals = struct.unpack("<4f", raw)
print("D3D12 outv[0..3] =", vals)
print("  expected float(gx)*1.0001+0.5:", [g * 1.0001 + 0.5 for g in range(4)])
ok = abs(vals[0] - 0.5) < 1e-3 and vals[3] > 3.0
print("RESULT:", "WRITE SURVIVES (no DCE)" if ok else "WRITE ELIMINATED or wrong")
best = run_rounds(20)
print(f"D3D12 dispatch+wait: {best*1000:.3f} ms  → {N*4/best/1e9:.1f} GB/s 写")
print("done")
