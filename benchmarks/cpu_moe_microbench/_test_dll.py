
import torch, ctypes, numpy as np, time, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")

NUM_EXPERTS, E, GU_ROWS, GU_K, INTER, DN_ROWS, DN_K = 256, 8, 1024, 2048, 512, 2048, 512
GU_KB, GU_SC, DN_KB, DN_SC = GU_K//2, GU_K//16, DN_K//2, DN_K//16
rng = np.random.default_rng(42)
# 256 专家 bank
gu_pack = rng.integers(0, 256, NUM_EXPERTS*GU_ROWS*GU_KB, dtype=np.uint8)
gu_scale = rng.integers(1, 200, NUM_EXPERTS*GU_ROWS*GU_SC, dtype=np.uint8)
gu_global = rng.integers(100, 30000, NUM_EXPERTS*GU_ROWS, dtype=np.uint16)
dn_pack = rng.integers(0, 256, NUM_EXPERTS*DN_ROWS*DN_KB, dtype=np.uint8)
dn_scale = rng.integers(1, 200, NUM_EXPERTS*DN_ROWS*DN_SC, dtype=np.uint8)
dn_global = rng.integers(100, 30000, NUM_EXPERTS*DN_ROWS, dtype=np.uint16)
hidden = rng.standard_normal(GU_K).astype(np.float32)
topk_ids = np.array([7, 42, 99, 155, 200, 3, 88, 254], dtype=np.int32)  # 随机路由
topkw = rng.standard_normal(E).astype(np.float32)

dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int; dll.igpu_init.argtypes = []
dll.igpu_register_layer.restype = ctypes.c_int
dll.igpu_register_layer.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
dll.igpu_moe_decode.restype = ctypes.c_int
dll.igpu_moe_decode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
dll.igpu_version.restype = ctypes.c_char_p; dll.igpu_version.argtypes = []
print("version:", dll.igpu_version().decode())
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]; hip.hipHostRegister.restype = ctypes.c_int
hip.hipInit(0); hip.hipSetDevice(0)
print("igpu_init:", dll.igpu_init())

def pin_reg(np_arr):
    t = torch.from_numpy(np_arr.copy()).pin_memory()
    r = hip.hipHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.nelement()*t.element_size()), 0)
    return t

banks = [pin_reg(gu_pack), pin_reg(gu_scale), pin_reg(gu_global),
         pin_reg(dn_pack), pin_reg(dn_scale), pin_reg(dn_global)]
t_hid = pin_reg(hidden); t_tkw = pin_reg(topkw); t_ids = pin_reg(topk_ids)
t_out = torch.zeros(DN_ROWS, dtype=torch.float32).pin_memory()
hip.hipHostRegister(ctypes.c_void_p(t_out.data_ptr()), ctypes.c_size_t(DN_ROWS*4), 0)
print("register:", dll.igpu_register_layer(0, *[b.data_ptr() for b in banks]))
r = dll.igpu_moe_decode(0, t_hid.data_ptr(), t_ids.data_ptr(), t_tkw.data_ptr(), t_out.data_ptr())
print("decode rc:", r)
out = t_out.numpy()
print("out[0:5]:", out[:5])
print("finite:", np.isfinite(out).all(), "nonzero:", np.any(out != 0))
times = []
for _ in range(20):
    t0 = time.perf_counter()
    dll.igpu_moe_decode(0, t_hid.data_ptr(), t_ids.data_ptr(), t_tkw.data_ptr(), t_out.data_ptr())
    times.append(time.perf_counter() - t0)
times.sort()
best = times[0] * 1000
print(f"per-layer: {best:.3f} ms best, projected 40-layer: {best*40:.1f} ms -> {1000/(best*40):.0f} t/s")
print("DLL TEST:", "PASS" if r == 0 and np.isfinite(out).all() and np.any(out != 0) else "FAIL")
