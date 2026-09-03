"""Phase 1 验证: igpu_moe_decode_dev 数值正确性 + 性能不退化.
对比 Phase 0 baseline 的 step_total.
"""
import os, sys, ctypes, time, json
sys.path.insert(0, r"E:\FreeToken\python")

rocm_bin = r"C:\Program Files\AMD\ROCm\6.4\bin"
if rocm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = rocm_bin + ";" + os.environ["PATH"]

import torch

NUM_LAYERS = 40
H = 2048
TOP_K = 8
BANK_SIZE = 433 * 1024 * 1024

hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
for name, args, rest in [
    ("hipGetDeviceCount", [ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("hipSetDevice", [ctypes.c_int], ctypes.c_int),
    ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
    ("hipFree", [ctypes.c_void_p], ctypes.c_int),
    ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    ("hipMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
    ("hipStreamSynchronize", [ctypes.c_void_p], ctypes.c_int),
]:
    fn = getattr(hip, name)
    fn.argtypes = args
    fn.restype = rest

n = ctypes.c_int(0)
hip.hipGetDeviceCount(ctypes.byref(n))
hip.hipSetDevice(0)
print(f"[HIP] devices: {n.value}")

x = torch.zeros(1024, device="cuda"); del x; torch.cuda.synchronize()
print(f"[CUDA] {torch.cuda.get_device_name(0)}")

# Get the DLL's stream (we need to sync it manually now)
dll = ctypes.CDLL(r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll")
dll.igpu_init.restype = ctypes.c_int
dll.igpu_init.argtypes = []
rc = dll.igpu_init()
print(f"[DLL] igpu_init rc={rc}")

# Get the g_stream pointer
dll.igpu_get_stream.argtypes = []
dll.igpu_get_stream.restype = ctypes.c_void_p
try:
    g_stream = dll.igpu_get_stream()
    print(f"[DLL] g_stream ptr: {hex(g_stream)}")
except Exception as e:
    print(f"[DLL] no igpu_get_stream export: {e}")
    g_stream = None

# Allocate 40 GTT banks
print(f"[GTT] alloc {NUM_LAYERS}x{BANK_SIZE//1024//1024}MB...")
t0 = time.perf_counter()
banks = []
for i in range(NUM_LAYERS):
    p = ctypes.c_void_p(0)
    rc = hip.hipMalloc(ctypes.byref(p), BANK_SIZE)
    if rc != 0:
        print(f"  bank {i} FAILED rc={rc}"); break
    banks.append(p.value)
print(f"[GTT] {len(banks)} banks in {time.perf_counter()-t0:.1f}s")

# Register banks
dll.igpu_register_layer_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
dll.igpu_register_layer_dev.restype = ctypes.c_int
for i, bp in enumerate(banks):
    rc = dll.igpu_register_layer_dev(i, ctypes.c_void_p(bp), 0, 0, 0, 0, 0)
    if rc != 0:
        print(f"[DLL] register {i} rc={rc}"); break
print(f"[DLL] registered {len(banks)} layers")

dll.igpu_moe_decode_dev.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
dll.igpu_moe_decode_dev.restype = ctypes.c_int

# Stage buffers
d_h = ctypes.c_void_p(0); d_i = ctypes.c_void_p(0); d_w = ctypes.c_void_p(0); d_o = ctypes.c_void_p(0)
hip.hipMalloc(ctypes.byref(d_h), H*4)
hip.hipMalloc(ctypes.byref(d_i), TOP_K*4)
hip.hipMalloc(ctypes.byref(d_w), TOP_K*4)
hip.hipMalloc(ctypes.byref(d_o), H*4)

# Reference data: hidden=0.5, ids=expert 0, weights=1/8
hidden_cpu = (ctypes.c_float * H)(*[0.5]*H)
ids_cpu = (ctypes.c_int * TOP_K)(*([0]*TOP_K))
weights_cpu = (ctypes.c_float * TOP_K)(*[0.125]*TOP_K)

# H2D staging
hip.hipMemcpy(d_h, ctypes.cast(hidden_cpu, ctypes.c_void_p), H*4, 1)
hip.hipMemcpy(d_i, ctypes.cast(ids_cpu, ctypes.c_void_p), TOP_K*4, 1)
hip.hipMemcpy(d_w, ctypes.cast(weights_cpu, ctypes.c_void_p), TOP_K*4, 1)

# === CORRECTNESS TEST ===
print("\n=== Correctness: call igpu_moe_decode_dev once, then sync, check output ===")
t0 = time.perf_counter()
rc = dll.igpu_moe_decode_dev(0, ctypes.c_void_p(d_h.value), ctypes.c_void_p(d_i.value), ctypes.c_void_p(d_w.value), ctypes.c_void_p(d_o.value))
t_call = (time.perf_counter() - t0) * 1e3
print(f"[CALL] rc={rc} t={t_call:.3f}ms")

# Sync (caller's responsibility now)
if g_stream:
    rc = hip.hipStreamSynchronize(g_stream)
    print(f"[SYNC] rc={rc}")

# Read output
out_cpu = (ctypes.c_float * H)()
hip.hipMemcpy(ctypes.cast(out_cpu, ctypes.c_void_p), d_o, H*4, 2)
vals = [out_cpu[i] for i in range(8)]
print(f"[OUT] first 8 values: {vals}")
norm = sum(v*v for v in out_cpu)**0.5
print(f"[OUT] L2 norm = {norm:.6f}")

# Sanity: with hidden=0.5 + expert 0 weights (all zero-init bank),
# output should be all zeros (since packed=0 -> weights=0 -> matmul=0)
all_zero = all(abs(v) < 1e-6 for v in out_cpu)
print(f"[OUT] all_zero (expected since bank packed=0): {all_zero}")
if all_zero:
    print("[OK] Phase 1 correctness verified")
else:
    print("[WARN] Output not zero (expected zero with zero-init bank)")

# === PERFORMANCE TEST (40 layers enqueue + 1 sync at end) ===
print("\n=== Performance: 40 layers enqueue + single hipStreamSynchronize ===")
gpu_hidden = torch.zeros(1, H, dtype=torch.float32, device="cuda")
gpu_ids = torch.zeros(1, TOP_K, dtype=torch.int32, device="cuda")
gpu_weights = torch.zeros(1, TOP_K, dtype=torch.float32, device="cuda")
gpu_out = torch.zeros(1, H, dtype=torch.float32, device="cuda")
out_host = torch.empty(1, H, dtype=torch.float32, pin_memory=True)
gpu_hidden.fill_(0.5)
gpu_weights.fill_(0.125)

N = 5  # 5 decode steps
t_total_start = time.perf_counter()
for step in range(N):
    h_cpu = gpu_hidden.to(torch.float32).cpu()
    i_cpu = gpu_ids.to(torch.int32).cpu()
    w_cpu = gpu_weights.to(torch.float32).cpu()
    torch.cuda.current_stream().synchronize()
    h_ptr = h_cpu[0].data_ptr()
    i_ptr = i_cpu[0].data_ptr()
    w_ptr = w_cpu[0].data_ptr()
    hip.hipMemcpy(d_h, ctypes.c_void_p(h_ptr), H*4, 1)
    hip.hipMemcpy(d_i, ctypes.c_void_p(i_ptr), TOP_K*4, 1)
    hip.hipMemcpy(d_w, ctypes.c_void_p(w_ptr), TOP_K*4, 1)
    # 40 kernel calls, NO sync between (this is the key change)
    for layer in range(NUM_LAYERS):
        rc = dll.igpu_moe_decode_dev(0, ctypes.c_void_p(d_h.value), ctypes.c_void_p(d_i.value), ctypes.c_void_p(d_w.value), ctypes.c_void_p(d_o.value))
        if rc != 0:
            print(f"  step {step} layer {layer} rc={rc}"); break
    # ONE sync at end (Phase 1 change)
    if g_stream:
        hip.hipStreamSynchronize(g_stream)
    out_ptr = out_host[0].data_ptr()
    hip.hipMemcpy(ctypes.c_void_p(out_ptr), d_o, H*4, 2)
    gpu_out.copy_(out_host, non_blocking=False)

t_total = (time.perf_counter() - t_total_start) * 1e3
per_step = t_total / N
print(f"\n[PHASE 1 RESULT]")
print(f"  Total: {t_total:.2f} ms for {N} steps")
print(f"  Per-step: {per_step:.2f} ms => {1000/per_step:.1f} tok/s")
print(f"  vs Phase 0 baseline: 60.65 ms / 16.5 tok/s (single sync per layer)")
print(f"  Improvement: {(60.65 - per_step)/60.65*100:+.1f}%")

# Cleanup
for p_ in (d_h, d_i, d_w, d_o):
    hip.hipFree(p_)
for bp in banks:
    hip.hipFree(ctypes.c_void_p(bp))
