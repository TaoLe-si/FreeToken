
import torch, ctypes, numpy as np, os, time
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int
assert dll.igpu_init() == 0
dll.igpu_devmalloc.restype = ctypes.c_void_p
dll.igpu_devmalloc.argtypes = [ctypes.c_size_t]
dll.igpu_register_layer_dev.restype = ctypes.c_int
dll.igpu_register_layer_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
dll.igpu_moe_decode_dev.restype = ctypes.c_int
dll.igpu_moe_decode_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

H, I, E = 2048, 512, 256
rng = np.random.default_rng(11)

# 40 层设备 banks
bank_ptrs = []
for L in range(40):
    gp = dll.igpu_devmalloc(E*1024*1024)
    gs = dll.igpu_devmalloc(E*1024*128)
    gg = dll.igpu_devmalloc(E*1024*2)
    dp = dll.igpu_devmalloc(E*2048*256)
    ds = dll.igpu_devmalloc(E*2048*32)
    dg = dll.igpu_devmalloc(E*2048*2)
    assert all([gp, gs, gg, dp, ds, dg]), L
    bank_ptrs.append((gp, gs, gg, dp, ds, dg))
for L, ptrs in enumerate(bank_ptrs):
    assert dll.igpu_register_layer_dev(L, *ptrs) == 0, L
print("40 layers allocated+registered")

# 只全量填充 layer0 (正确性用)
gp_h = rng.integers(0, 256, E*1024*1024, dtype=np.uint8)
gs_h = rng.integers(1, 24, E*1024*128, dtype=np.uint8)
gg_h = np.full(E*1024, 15360, dtype=np.uint16)  # fp16 1.0
dp_h = rng.integers(0, 256, E*2048*256, dtype=np.uint8)
ds_h = rng.integers(1, 24, E*2048*32, dtype=np.uint8)
dg_h = np.full(E*2048, 15360, dtype=np.uint16)  # fp16 1.0
gp, gs, gg, dp_, ds_, dg_ = bank_ptrs[0]
for h, d in ((gp_h, gp), (gs_h, gs), (gg_h, gg), (dp_h, dp_), (ds_h, ds_), (dg_h, dg_)):
    hip.hipMemcpy(ctypes.c_void_p(d), h.ctypes.data, h.nbytes, 1)
print("layer0 filled:", gp_h.nbytes + gs_h.nbytes + dp_h.nbytes + ds_h.nbytes, "bytes")

# 设备 IO 缓冲
d_hid = dll.igpu_devmalloc(H*4)
d_ids = dll.igpu_devmalloc(8*4)
d_tkw = dll.igpu_devmalloc(8*4)
d_out = dll.igpu_devmalloc(H*4)

ids = np.array([7, 42, 99, 155, 200, 3, 88, 254], dtype=np.int32)
tkw = (np.ones(8, dtype=np.float32) * 0.125)
hidden = (rng.standard_normal(H).astype(np.float32) * 0.5)
hip.hipMemcpy(ctypes.c_void_p(d_ids), ids.ctypes.data, 32, 1)
hip.hipMemcpy(ctypes.c_void_p(d_tkw), tkw.ctypes.data, 32, 1)
hip.hipMemcpy(ctypes.c_void_p(d_hid), hidden.ctypes.data, H*4, 1)

# === 正确性: CPU 参考 vs 设备 decode (layer0) ===
E2M1 = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=np.float32)
def e4m3(u8):
    s = np.where(u8 & 0x80, -1.0, 1.0).astype(np.float32)
    e = ((u8 >> 3) & 0xF).astype(np.int32); m = (u8 & 7).astype(np.int32)
    val = np.where(e == 0, m/8.0*np.float32(2.0**-6), (1+m/8.0)*np.float32(2.0)**(e-7).astype(np.float32))
    return s*val
def dequant(packed, scale, glb, N, K):
    lo = (packed & 0xF).astype(np.int32); hi = (packed >> 4).astype(np.int32)
    codes = np.empty((N, K), dtype=np.int32); codes[:, 0::2] = lo; codes[:, 1::2] = hi
    gl16 = torch.from_numpy(glb.astype(np.uint16)).view(torch.float16).float().numpy()
    return (E2M1[codes] * np.repeat(e4m3(scale), 16, axis=1) * gl16[:, None]).astype(np.float32)
out_ref = np.zeros(H, dtype=np.float32)
for i, eid in enumerate(ids):
    gu = dequant(gp_h.reshape(E,1024,1024)[eid], gs_h.reshape(E,1024,128)[eid], gg_h.reshape(E,1024)[eid], 1024, 2048) @ hidden
    gate, up = gu[:512], gu[512:]
    act = (gate / (1 + np.exp(-np.clip(gate, -30, 30)))) * up
    out_ref += tkw[i] * (dequant(dp_h.reshape(E,2048,256)[eid], ds_h.reshape(E,2048,32)[eid], dg_h.reshape(E,2048)[eid], 2048, 512) @ act)

rc = dll.igpu_moe_decode_dev(0, ctypes.c_void_p(d_hid), ctypes.c_void_p(d_ids), ctypes.c_void_p(d_tkw), ctypes.c_void_p(d_out))
out_dev = np.zeros(H, dtype=np.float32)
hip.hipMemcpy(out_dev.ctypes.data, ctypes.c_void_p(d_out), H*4, 2)
print("decode rc:", rc)
rel = np.abs(out_dev - out_ref) / (np.abs(out_ref) + 1e-9)
print("ref norm:", np.linalg.norm(out_ref), "dev norm:", np.linalg.norm(out_dev))
print("max rel err:", rel.max(), "median:", np.median(rel))

# === 速度: 纯 kernel (无宿主 IO 开销) ===
t0 = time.perf_counter()
for _ in range(50):
    rc = dll.igpu_moe_decode_dev(0, ctypes.c_void_p(d_hid), ctypes.c_void_p(d_ids), ctypes.c_void_p(d_tkw), ctypes.c_void_p(d_out))
    assert rc == 0
wall = (time.perf_counter() - t0) / 50
print(f"pure kernel per layer: {wall*1000:.3f} ms -> 40-layer {wall*4000:.1f} ms -> {1000/(wall*40):.1f} t/s")

# === 速度: 含每步 IO (hidden H2D 每层 8KB + 输出 8KB 回读) ===
hbuf = np.zeros(H, dtype=np.float32)
t0 = time.perf_counter()
for _ in range(20):
    for L in range(40):
        hip.hipMemcpy(ctypes.c_void_p(d_hid), hbuf.ctypes.data, H*4, 1)  # 模拟每层 hidden 上行
        rc = dll.igpu_moe_decode_dev(L if L < 40 else 0, ctypes.c_void_p(d_hid), ctypes.c_void_p(d_ids), ctypes.c_void_p(d_tkw), ctypes.c_void_p(d_out))
        # 注意: 层 1-39 bank 是未初始化显存, 结果无效但时序真实
        hip.hipMemcpy(hbuf.ctypes.data, ctypes.c_void_p(d_out), H*4, 2)  # 模拟结果回读
wall = (time.perf_counter() - t0) / 20
print(f"kernel + per-layer IO: {wall*1000:.1f} ms/token -> {1000/(wall*1000):.1f} t/s")
