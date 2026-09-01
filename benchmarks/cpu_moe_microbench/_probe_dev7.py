import torch, ctypes, os, time
os.add_dll_directory(r'C:\Program Files\AMD\ROCm\6.4\bin')
hip = ctypes.CDLL('amdhip64_6.dll')
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
x = torch.randn(1000, 1000, device="cuda:0")
chunks = []
for i in range(16):
    chunks.append(torch.empty(512*1024**2, dtype=torch.uint8, pin_memory=True))
print('8GB pinned done')
dll = ctypes.CDLL(r'E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll')
dll.igpu_init.restype = ctypes.c_int
dll.igpu_devmalloc.restype = ctypes.c_void_p
dll.igpu_devmalloc.argtypes = [ctypes.c_size_t]
dll.igpu_init()
ptrs = []
for i in range(453):
    d = dll.igpu_devmalloc(ctypes.c_size_t(433000000))
    if not d:
        print("big alloc", i, "FAILED (null)")
        break
    ptrs.append(d)
print('migrated', len(ptrs), 'banks, last', hex(ptrs[-1] or 0))
d = dll.igpu_devmalloc(ctypes.c_size_t(8192))
print('small staging ptr:', hex(d or 0))
src = bytes([0x41]*8192)
rc = hip.hipMemcpy(ctypes.c_void_p(d), src, 8192, 1) if d else -1
dst = ctypes.create_string_buffer(8192)
rc2 = hip.hipMemcpy(dst, ctypes.c_void_p(d), 8192, 2) if d else -1
print('H2D', rc, 'D2H', rc2, 'match:', dst.raw[:8] == b'AAAAAAAA' if d else 'no')