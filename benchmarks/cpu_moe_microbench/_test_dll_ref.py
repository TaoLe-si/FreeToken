
import torch, ctypes, numpy as np, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
NUM_EXPERTS, E, H, I = 256, 8, 2048, 512
GU_KB, GU_SC, DN_KB, DN_SC = H//2, H//16, I//2, I//16
rng = np.random.default_rng(7)
gu_pack = rng.integers(0, 256, NUM_EXPERTS*(2*I)*GU_KB, dtype=np.uint8).reshape(NUM_EXPERTS, 2*I, GU_KB)
gu_scale = rng.integers(1, 120, NUM_EXPERTS*(2*I)*GU_SC, dtype=np.uint8).reshape(NUM_EXPERTS, 2*I, GU_SC)
gu_global = rng.integers(10000, 30000, NUM_EXPERTS*(2*I), dtype=np.uint16).reshape(NUM_EXPERTS, 2*I)
dn_pack = rng.integers(0, 256, NUM_EXPERTS*H*DN_KB, dtype=np.uint8).reshape(NUM_EXPERTS, H, DN_KB)
dn_scale = rng.integers(1, 120, NUM_EXPERTS*H*DN_SC, dtype=np.uint8).reshape(NUM_EXPERTS, H, DN_SC)
dn_global = rng.integers(10000, 30000, NUM_EXPERTS*H, dtype=np.uint16).reshape(NUM_EXPERTS, H)
hidden = rng.standard_normal(H).astype(np.float32)
ids = np.array([3, 17, 99, 155, 200, 42, 88, 254], dtype=np.int32)
topkw = np.abs(rng.standard_normal(E)).astype(np.float32)

# ---- CPU 参考: NVFP4 dequant ----
# e2m1: 16 个 4-bit 码点
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
def e4m3_to_f(u8):
    # e4m3: sign(1) exp(4) mantissa(3), bias 7
    s = np.where(u8 & 0x80, -1.0, 1.0).astype(np.float32)
    e = ((u8 >> 3) & 0xF).astype(np.int32)
    m = (u8 & 0x7).astype(np.int32)
    val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0**(e - 7).astype(np.float32))
    # NaN (0x7F/0xFF) 忽略, 随机数据没有
    return (s * val).astype(np.float32)
def f16_to_f(u16):
    return torch.from_numpy(u16.astype(np.uint16)).view(torch.float16).float().numpy()

def dequant_w(packed, scale, glb, N, K):
    # packed [N, K/2] u8; 低 nibble = k 偶数, 高 nibble = k 奇数 (待验证)
    lo = (packed & 0xF).astype(np.int32)
    hi = (packed >> 4).astype(np.int32)
    codes = np.empty((N, K), dtype=np.int32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    w = E2M1[codes]  # [N, K]
    sc = e4m3_to_f(scale)  # [N, K/16]
    sc_exp = np.repeat(sc, 16, axis=1)  # [N, K]
    return w * sc_exp * glb[:, None]

out_ref = np.zeros(H, dtype=np.float32)
for i, eid in enumerate(ids):
    gu_w = dequant_w(gu_pack[eid], gu_scale[eid], f16_to_f(gu_global[eid]), 2*I, H)  # [1024, 2048]
    gu = gu_w @ hidden  # [1024]
    gate, up = gu[:I], gu[I:2*I]
    act = (gate / (1 + np.exp(-np.clip(gate, -30, 30)))) * up  # silu(gate)*up
    dn_w = dequant_w(dn_pack[eid], dn_scale[eid], f16_to_f(dn_global[eid]), H, I)  # [2048, 512]
    out_ref += topkw[i] * (dn_w @ act)

# ---- DLL ----
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int
dll.igpu_register_layer.restype = ctypes.c_int
dll.igpu_register_layer.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
dll.igpu_moe_decode.restype = ctypes.c_int
dll.igpu_moe_decode.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipInit(0); hip.hipSetDevice(0)
def pin(a):
    t = torch.from_numpy(a.copy()).pin_memory()
    assert hip.hipHostRegister(ctypes.c_void_p(t.data_ptr()), t.numel()*t.element_size(), 0) == 0
    return t
ts = [pin(gu_pack.ravel()), pin(gu_scale.ravel()), pin(gu_global.ravel()),
      pin(dn_pack.ravel()), pin(dn_scale.ravel()), pin(dn_global.ravel())]
t_h = pin(hidden); t_i = pin(ids); t_w = pin(topkw); t_o = torch.zeros(H).pin_memory()
hip.hipHostRegister(ctypes.c_void_p(t_o.data_ptr()), H*4, 0)
print("init:", dll.igpu_init(), "reg:", dll.igpu_register_layer(0, *[x.data_ptr() for x in ts]))
rc = dll.igpu_moe_decode(0, t_h.data_ptr(), t_i.data_ptr(), t_w.data_ptr(), t_o.data_ptr())
out_dll = t_o.numpy()
print("rc:", rc)
print("ref[0:4]:", out_ref[:4])
print("dll[0:4]:", out_dll[:4])
rel = np.abs(out_dll - out_ref) / (np.abs(out_ref) + 1e-9)
print("max rel err:", rel.max(), " median:", np.median(rel))
# nibble 顺序备选: 高 nibble = 偶数
lo = (gu_pack[3] & 0xF).astype(np.int32); hi = (gu_pack[3] >> 4).astype(np.int32)
print("pack sample lo:", lo[:4], "hi:", hi[:4])
