
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
x = torch.randn(8, device="cuda:0"); torch.cuda.synchronize()
dll.igpu_init.restype = ctypes.c_int
print("igpu_init:", dll.igpu_init())
dll.igpu_hostmalloc.restype = ctypes.c_void_p
dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
H, I, E = 2048, 512, 8
def alloc(nbytes):
    return dll.igpu_hostmalloc(nbytes)
rng = np.random.default_rng(3)
# 8 专家 bank
gp = alloc(E*H*1024); gs = alloc(E*H*128); gg = alloc(E*H*2)
dp_ = alloc(E*I*512); ds = alloc(E*I*64); dg = alloc(E*I*2)
def wr(h, a): ctypes.memmove(ctypes.c_void_p(h), a.ctypes.data, a.nbytes)
wr(gp, rng.integers(0,256,E*H*1024,dtype=np.uint8))
wr(gs, rng.integers(1,120,E*H*128,dtype=np.uint8))
gg_a = rng.integers(10000,20000,E*H,dtype=np.uint16); wr(gg, gg_a)
wr(dp_, rng.integers(0,256,E*I*512,dtype=np.uint8))
wr(ds, rng.integers(1,120,E*I*64,dtype=np.uint8))
dg_a = rng.integers(10000,20000,E*I,dtype=np.uint16); wr(dg, dg_a)
hid = alloc(H*4); wr(hid, rng.standard_normal(H).astype(np.float32))
ids = alloc(E*4); wr(ids, np.arange(E, dtype=np.int32))
tkw = alloc(E*4); wr(tkw, np.ones(E, dtype=np.float32)*0.1)
out = alloc(H*4); ctypes.memset(ctypes.c_void_p(out), 0, H*4)
dll.igpu_register_layer.restype = ctypes.c_int
dll.igpu_register_layer.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
rc = dll.igpu_register_layer(0, gp, gs, gg, dp_, ds, dg)
print("register:", rc)
dll.igpu_moe_decode.restype = ctypes.c_int
dll.igpu_moe_decode.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4
rc2 = dll.igpu_moe_decode(0, hid, ids, tkw, out)
o = np.frombuffer(ctypes.string_at(ctypes.c_void_p(out), H*4), dtype=np.float32)
print("decode rc:", rc2, "out norm:", np.linalg.norm(o), "nonzero:", (o!=0).any())
