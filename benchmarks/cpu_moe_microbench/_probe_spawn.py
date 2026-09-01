import torch, ctypes, os, numpy as np
import multiprocessing as mp

def child():
    os.add_dll_directory(r'C:\Program Files\AMD\ROCm\6.4\bin')
    hip = ctypes.CDLL('amdhip64_6.dll')
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    dll = ctypes.CDLL(r'E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll')
    dll.igpu_init.restype = ctypes.c_int
    dll.igpu_devmalloc.restype = ctypes.c_void_p
    dll.igpu_devmalloc.argtypes = [ctypes.c_size_t]
    dll.igpu_init()
    x = torch.randn(1000, 1000, device="cuda:0")
    chunks = []
    for i in range(12):
        chunks.append(torch.empty(1024*1024*1024, dtype=torch.uint8, pin_memory=True))
    print("12GB pinned done", flush=True)
    d = dll.igpu_devmalloc(ctypes.c_size_t(268435456))
    print("dev ptr:", hex(d or 0), flush=True)
    plain = np.zeros(268435456, dtype=np.uint8)
    rc = hip.hipMemcpy(ctypes.c_void_p(d), plain.ctypes.data, 268435456, 1)
    print("H2D rc:", rc, flush=True)
    dst = ctypes.create_string_buffer(64)
    rc2 = hip.hipMemcpy(dst, ctypes.c_void_p(d), 64, 2)
    print("D2H rc:", rc2, flush=True)

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=child)
    p.start()
    p.join(240)
    print("child exit:", p.exitcode)