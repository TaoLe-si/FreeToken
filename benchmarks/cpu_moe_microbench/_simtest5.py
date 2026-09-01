
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
# 完整模拟引擎 decode 数据流
x = torch.randn(8, device="cuda:0"); torch.cuda.synchronize()
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int
print("igpu_init:", dll.igpu_init())
dll.igpu_hostmalloc.restype = ctypes.c_void_p
dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
H, I, E = 2048, 512, 8
# IO: hostMalloc (同引擎新 _io_for)
h_hid = dll.igpu_hostmalloc(H*4)
h_ids = dll.igpu_hostmalloc(E*4)
h_tkw = dll.igpu_hostmalloc(E*4)
h_out = dll.igpu_hostmalloc(H*4)
hid_view = torch.frombuffer((ctypes.c_char*(H*4)).from_address(h_hid), dtype=torch.float32).view(1, H)
ids_view = torch.frombuffer((ctypes.c_char*(E*4)).from_address(h_ids), dtype=torch.int32).view(1, E)
w_view = torch.frombuffer((ctypes.c_char*(E*4)).from_address(h_tkw), dtype=torch.float32).view(1, E)
out_view = torch.frombuffer((ctypes.c_char*(H*4)).from_address(h_out), dtype=torch.float32)
# 模拟 bank (hostMalloc) + register
gp = dll.igpu_hostmalloc(E*H*1024); gs = dll.igpu_hostmalloc(E*H*128); gg = dll.igpu_hostmalloc(E*H*2)
dp_ = dll.igpu_hostmalloc(E*I*512); ds = dll.igpu_hostmalloc(E*I*64); dg = dll.igpu_hostmalloc(E*I*2)
rng = np.random.default_rng(5)
def wr(h, a): ctypes.memmove(ctypes.c_void_p(h), a.ctypes.data, a.nbytes)
wr(gp, rng.integers(0,256,E*H*1024,dtype=np.uint8))
wr(gs, rng.integers(1,120,E*H*128,dtype=np.uint8))
wr(gg, rng.integers(10000,20000,E*H,dtype=np.uint16))
wr(dp_, rng.integers(0,256,E*I*512,dtype=np.uint8))
wr(ds, rng.integers(1,120,E*I*64,dtype=np.uint8))
wr(dg, rng.integers(10000,20000,E*I,dtype=np.uint16))
dll.igpu_register_layer.restype = ctypes.c_int
dll.igpu_register_layer.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
print("register:", dll.igpu_register_layer(0, gp, gs, gg, dp_, ds, dg))
# === 引擎 decode 数据流 ===
hidden_gpu = torch.randn(1, H, device="cuda:0", dtype=torch.bfloat16)
ids_gpu = torch.randint(0, E, (1, E), device="cuda:0", dtype=torch.int64)
w_gpu = torch.rand(1, E, device="cuda:0")
stream = torch.cuda.current_stream()
hid_view.copy_(hidden_gpu, non_blocking=True)   # bf16->f32 D2H 到 hostMalloc
ids_view.copy_(ids_gpu.to(torch.int32), non_blocking=True)
w_view.copy_(w_gpu.to(torch.float32), non_blocking=True)
stream.synchronize()
print("after copy hid_view[0,:4]:", hid_view[0,:4].tolist())
dll.igpu_moe_decode.restype = ctypes.c_int
dll.igpu_moe_decode.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4
rc = dll.igpu_moe_decode(0, ctypes.c_void_p(h_hid), ctypes.c_void_p(h_ids), ctypes.c_void_p(h_tkw), ctypes.c_void_p(h_out))
print("decode rc:", rc, "out nonzero:", out_view.abs().sum().item() != 0, "norm:", out_view.norm().item())

# === bisect: hidden pin_memory+register ===
hid_pin = hid_view.clone().pin_memory()
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
rc_reg = hip.hipHostRegister(ctypes.c_void_p(hid_pin.data_ptr()), H*4, 0)
ctypes.memmove(ctypes.c_void_p(h_out), b"\x00" * (H*4), H*4)
rc2 = dll.igpu_moe_decode(0, ctypes.c_void_p(hid_pin.data_ptr()), ctypes.c_void_p(h_ids), ctypes.c_void_p(h_tkw), ctypes.c_void_p(h_out))
print("PIN-hidden decode rc:", rc2, "out norm:", out_view.norm().item())
ids_pin = ids_view.clone().contiguous().pin_memory()
w_pin = w_view.clone().contiguous().pin_memory()
hip.hipHostRegister(ctypes.c_void_p(ids_pin.data_ptr()), E*4, 0)
hip.hipHostRegister(ctypes.c_void_p(w_pin.data_ptr()), E*4, 0)
ctypes.memmove(ctypes.c_void_p(h_out), b"\x00" * (H*4), H*4)
rc3 = dll.igpu_moe_decode(0, ctypes.c_void_p(hid_pin.data_ptr()), ctypes.c_void_p(ids_pin.data_ptr()), ctypes.c_void_p(w_pin.data_ptr()), ctypes.c_void_p(h_out))
print("ALL-PIN decode rc:", rc3, "out norm:", out_view.norm().item())

