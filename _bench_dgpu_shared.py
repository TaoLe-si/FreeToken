"""
对比 benchmark: dGPU (RTX 4070) 共享内存 vs iGPU (780M) GTT
目的: 看看 dGPU WDDM shared pool 能不能直接当大模型权重存放池
"""
import os, sys, time, ctypes
sys.path.insert(0, r"E:\\FreeToken\\python")
import torch

print(f"[dGPU = NVIDIA RTX 4070 Laptop] {torch.cuda.get_device_name(0)}")
print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB\n")

def time_ms(fn, n=10, warmup=2):
    """测时, 返回 ms/调用"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ============================================================
# A. dGPU VRAM 基线 (用 torch 在 dGPU 上 alloc)
# ============================================================
print("="*60)
print("[A] dGPU VRAM 基线 (8GB 专用显存, fast path)")
print("="*60)

def dgpu_vram_alloc():
    return torch.empty(64*1024*1024, dtype=torch.float32, device='cuda')

ms = time_ms(dgpu_vram_alloc)
print(f"  dGPU VRAM alloc 64MB: {ms:.3f} ms")

def dgpu_vram_read():
    x = torch.randn(64*1024*1024, dtype=torch.float32, device='cuda')
    return x.sum().item()

ms = time_ms(dgpu_vram_read, n=5)
print(f"  dGPU VRAM sum 64MB: {ms:.3f} ms => {256/ms*1000:.1f} GB/s")


# ============================================================
# B. dGPU WDDM 共享池 (大块 alloc 触发 shared pool)
# ============================================================
print("\n" + "="*60)
print("[B] dGPU WDDM shared pool (>8GB 触发)")
print("="*60)

print("\n[B.1] Alloc 14 GB pinned (触发 WDDM shared pool)...")
t0 = time.perf_counter()
wddm_buf = torch.empty(14*1024*1024*1024, dtype=torch.uint8, pin_memory=True)
print(f"  alloc done in {time.perf_counter()-t0:.2f}s, size={wddm_buf.numel()/1e9:.1f} GB")

print("\n[B.2] First access (page fault fill)...")
t0 = time.perf_counter()
wddm_buf.zero_()
torch.cuda.synchronize()
print(f"  zero 14 GB (warm fault-fill): {time.perf_counter()-t0:.2f}s")

print("\n[B.3] Sequential read 1 GB (warm)...")
def wddm_seq_read():
    chunk = wddm_buf[0:1*1024*1024*1024]
    return chunk.sum().item()
ms = time_ms(wddm_seq_read, n=5)
print(f"  WDDM seq read 1 GB: {ms:.3f} ms => {1000/ms*1000:.1f} GB/s")

print("\n[B.4] Sequential write 1 GB (warm)...")
def wddm_seq_write():
    wddm_buf[0:1*1024*1024*1024] = 0
ms = time_ms(wddm_seq_write, n=5)
print(f"  WDDM seq write 1 GB: {ms:.3f} ms => {1000/ms*1000:.1f} GB/s")

print("\n[B.5] Decode-like random access (16KB every 300MB)...")
def decode_random_pattern():
    # 40 层, 每层 16 KB, 间隔 300 MB (模拟 MoE bank 布局)
    for layer in range(40):
        offset = layer * 300 * 1024 * 1024
        _ = wddm_buf[offset:offset+16*1024].sum().item()
ms = time_ms(decode_random_pattern, n=5)
print(f"  WDDM random 40×16KB: {ms:.3f} ms ({ms*1000/40:.1f} us/layer)")


# ============================================================
# C. dGPU <-> WDDM 之间的 D2H / H2D (走 PCIe)
# ============================================================
print("\n" + "="*60)
print("[C] dGPU VRAM <-> WDDM pinned (走 PCIe)")
print("="*60)

x_vram = torch.randn(64*1024*1024, dtype=torch.float32, device='cuda')
wddm_small = torch.empty(64*1024*1024, dtype=torch.float32, pin_memory=True)

def d2h_wddm():
    wddm_small.copy_(x_vram, non_blocking=True)
    torch.cuda.synchronize()
ms = time_ms(d2h_wddm, n=20)
print(f"  D2H dGPU -> WDDM pinned 64MB: {ms:.3f} ms => {256/ms*1000:.1f} GB/s")

def h2d_wddm():
    x_vram.copy_(wddm_small, non_blocking=True)
    torch.cuda.synchronize()
ms = time_ms(h2d_wddm, n=20)
print(f"  H2D WDDM pinned -> dGPU 64MB: {ms:.3f} ms => {256/ms*1000:.1f} GB/s")


# ============================================================
# D. iGPU GTT 对比 (用 HIP)
# ============================================================
print("\n" + "="*60)
print("[D] iGPU GTT (780M via HIP)")
print("="*60)
try:
    rocm = r"C:\\Program Files\\AMD\\ROCm\\6.4\\bin\\amdhip64_6.dll"
    hip = ctypes.CDLL(rocm)
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    n = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(n))
    hip.hipSetDevice(0)
    fb = ctypes.c_size_t(0); tb = ctypes.c_size_t(0)
    hip.hipMemGetInfo(ctypes.byref(fb), ctypes.byref(tb))
    print(f"  iGPU devices: {n.value}, total mem: {tb.value/1e9:.2f} GB")
    
    d_buf = ctypes.c_void_p(0)
    rc = hip.hipMalloc(ctypes.byref(d_buf), 64*1024*1024)
    print(f"  iGPU hipMalloc 64MB: rc={rc}, ptr=0x{d_buf.value:x}")
    
    src_buf = (ctypes.c_float * (16*1024*1024))(*([0.5]*16*1024*1024))
    def hip_h2d():
        hip.hipMemcpy(d_buf, ctypes.cast(src_buf, ctypes.c_void_p), 64*1024*1024, 1)
    times=[]
    for _ in range(3):
        t0=time.perf_counter()
        for _ in range(50): hip_h2d()
        times.append((time.perf_counter()-t0)/50)
    ms = min(times)*1000
    print(f"  iGPU H2D 64MB: {ms:.3f} ms => {256/ms*1000:.1f} GB/s")
except Exception as e:
    print(f"  iGPU test skipped: {e}")

print("\n" + "="*60)
print("总结 (理论上限对比)")
print("="*60)
print(f"  dGPU VRAM 带宽 (理论): 256 GB/s")
print(f"  PCIe 3.0 (理论):       16 GB/s")
print(f"  iGPU APU coherent (理论): 80-100 GB/s")
print(f"  iGPU GTT 实际 (Form-2): 26 GB/s (commit 55af654 实测)")
print(f"  dGPU WDDM shared (本测试): 看上面数字")
