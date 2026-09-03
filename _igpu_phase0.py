"""
Phase 0: iGPU decode() 耗时分布测量.
独立脚本: 不动源码, 复刻每一步, 测时输出.
用法: python _igpu_phase0.py [--steps N]
"""
import os, sys, ctypes, time, json, argparse
sys.path.insert(0, r"E:\FreeToken\python")

rocm_bin = r"C:\Program Files\AMD\ROCm\6.4\bin"
if rocm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = rocm_bin + ";" + os.environ["PATH"]

import torch

NUM_LAYERS = 40
H = 2048
I = 512
TOP_K = 8
BANK_SIZE = 433 * 1024 * 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--bs", type=int, default=1)
    args = p.parse_args()

    hip_path = r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll"
    hip = ctypes.CDLL(hip_path)
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipSetDevice.restype = ctypes.c_int
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMalloc.restype = ctypes.c_int
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipFree.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemcpy.restype = ctypes.c_int
    hip.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    hip.hipMemGetInfo.restype = ctypes.c_int

    n = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(n))
    hip.hipSetDevice(0)
    print(f"[HIP] device count: {n.value}")

    print("[CUDA] init...")
    x = torch.zeros(1024, device="cuda")
    del x
    torch.cuda.synchronize()
    print(f"[CUDA] device: {torch.cuda.get_device_name(0)}")

    print(f"[GTT] allocating {NUM_LAYERS}x{BANK_SIZE//1024//1024}MB banks...")
    t0 = time.perf_counter()
    banks = []
    for i in range(NUM_LAYERS):
        p_ = ctypes.c_void_p(0)
        rc = hip.hipMalloc(ctypes.byref(p_), BANK_SIZE)
        if rc != 0:
            print(f"  bank {i} alloc FAILED rc={rc}")
            break
        banks.append(p_.value)
    alloc_s = time.perf_counter() - t0
    print(f"[GTT] {len(banks)} banks in {alloc_s:.1f}s")

    fb = ctypes.c_size_t(0)
    tb = ctypes.c_size_t(0)
    hip.hipMemGetInfo(ctypes.byref(fb), ctypes.byref(tb))
    print(f"[GTT] meminfo: free={fb.value/2**30:.2f}GB total={tb.value/2**30:.2f}GB")

    print(f"\n[PHASE 0] profiling {args.steps} decode steps, bs={args.bs}")
    print("  simulating 40 layers × 6 sync calls (real decode() flow)")

    d_h = ctypes.c_void_p(0)
    d_i = ctypes.c_void_p(0)
    d_w = ctypes.c_void_p(0)
    d_o = ctypes.c_void_p(0)
    hip.hipMalloc(ctypes.byref(d_h), args.bs * H * 4)
    hip.hipMalloc(ctypes.byref(d_i), args.bs * TOP_K * 4)
    hip.hipMalloc(ctypes.byref(d_w), args.bs * TOP_K * 4)
    hip.hipMalloc(ctypes.byref(d_o), args.bs * H * 4)
    print(f"[STAGING] d_h={hex(d_h.value)} d_i={hex(d_i.value)} d_w={hex(d_w.value)} d_o={hex(d_o.value)}")

    gpu_hidden = torch.zeros(args.bs, H, dtype=torch.float32, device="cuda")
    gpu_ids = torch.zeros(args.bs, TOP_K, dtype=torch.int32, device="cuda")
    gpu_weights = torch.zeros(args.bs, TOP_K, dtype=torch.float32, device="cuda")
    gpu_out = torch.zeros(args.bs, H, dtype=torch.float32, device="cuda")

    out_host = torch.empty(args.bs, H, dtype=torch.float32, pin_memory=True)

    timings = {
        "d2h_hidden": [], "d2h_ids": [], "d2h_weights": [], "sync": [],
        "h2d_staging_hidden": [], "h2d_staging_ids": [], "h2d_staging_weights": [],
        "hip_kernel": [], "d2h_out": [], "h2d_out": [],
        "step_total": []
    }

    dll_path = r"E:\FreeToken\benchmarks\cpu_moe_microbench\hip_moe_dll.dll"
    dll = ctypes.CDLL(dll_path)
    dll.igpu_init.restype = ctypes.c_int
    rc = dll.igpu_init()
    print(f"[DLL] igpu_init rc={rc}")
    dll.igpu_register_layer_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6
    dll.igpu_register_layer_dev.restype = ctypes.c_int
    for i, bp in enumerate(banks):
        rc = dll.igpu_register_layer_dev(i, ctypes.c_void_p(bp), 0, 0, 0, 0, 0)
        if rc != 0:
            print(f"[DLL] register layer {i} rc={rc}")
            break
    print(f"[DLL] registered {len(banks)} layers")

    dll.igpu_moe_decode_dev.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    dll.igpu_moe_decode_dev.restype = ctypes.c_int

    gpu_hidden.fill_(0.5)
    gpu_ids.fill_(0)
    gpu_weights.fill_(0.125)

    H2D_const = 1
    D2H_const = 2

    for step in range(args.steps):
        t_step_start = time.perf_counter()

        t0 = time.perf_counter()
        h_cpu = gpu_hidden.to(torch.float32).cpu()
        timings["d2h_hidden"].append((time.perf_counter()-t0)*1e3)
        t0 = time.perf_counter()
        i_cpu = gpu_ids.to(torch.int32).cpu()
        timings["d2h_ids"].append((time.perf_counter()-t0)*1e3)
        t0 = time.perf_counter()
        w_cpu = gpu_weights.to(torch.float32).cpu()
        timings["d2h_weights"].append((time.perf_counter()-t0)*1e3)

        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        timings["sync"].append((time.perf_counter()-t0)*1e3)

        h_ptr = h_cpu[0].data_ptr()
        i_ptr = i_cpu[0].data_ptr()
        w_ptr = w_cpu[0].data_ptr()
        t0 = time.perf_counter()
        hip.hipMemcpy(d_h, ctypes.c_void_p(h_ptr), ctypes.c_size_t(H*4), H2D_const)
        timings["h2d_staging_hidden"].append((time.perf_counter()-t0)*1e3)
        t0 = time.perf_counter()
        hip.hipMemcpy(d_i, ctypes.c_void_p(i_ptr), ctypes.c_size_t(TOP_K*4), H2D_const)
        timings["h2d_staging_ids"].append((time.perf_counter()-t0)*1e3)
        t0 = time.perf_counter()
        hip.hipMemcpy(d_w, ctypes.c_void_p(w_ptr), ctypes.c_size_t(TOP_K*4), H2D_const)
        timings["h2d_staging_weights"].append((time.perf_counter()-t0)*1e3)

        t0 = time.perf_counter()
        for layer in range(NUM_LAYERS):
            rc = dll.igpu_moe_decode_dev(0, ctypes.c_void_p(d_h.value), ctypes.c_void_p(d_i.value), ctypes.c_void_p(d_w.value), ctypes.c_void_p(d_o.value))
            if rc != 0:
                print(f"[KERNEL] layer {layer} rc={rc}")
                break
        timings["hip_kernel"].append((time.perf_counter()-t0)*1e3)

        out_ptr = out_host[0].data_ptr()
        t0 = time.perf_counter()
        hip.hipMemcpy(ctypes.c_void_p(out_ptr), d_o, ctypes.c_size_t(H*4), D2H_const)
        timings["d2h_out"].append((time.perf_counter()-t0)*1e3)

        t0 = time.perf_counter()
        gpu_out.copy_(out_host, non_blocking=False)
        timings["h2d_out"].append((time.perf_counter()-t0)*1e3)

        timings["step_total"].append((time.perf_counter()-t_step_start)*1e3)

    print("\n" + "="*70)
    print(f"PHASE 0 RESULTS: {args.steps} decode steps (mimicking real decode())")
    print("="*70)
    for k, vs in timings.items():
        if not vs:
            continue
        avg = sum(vs)/len(vs)
        mn = min(vs)
        mx = max(vs)
        print(f"  {k:25s} avg={avg:8.3f} ms  min={mn:8.3f}  max={mx:8.3f}")

    total_avg = sum(timings["step_total"]) / len(timings["step_total"])
    print(f"\n  Step total: avg={total_avg:.2f} ms => {1000/total_avg:.1f} tok/s")

    out = {"steps": args.steps, "bs": args.bs, "timings_ms": timings, "num_layers": NUM_LAYERS}
    with open("_igpu_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: _igpu_baseline.json")

    for p_ in (d_h, d_i, d_w, d_o):
        hip.hipFree(p_)
    for bp in banks:
        hip.hipFree(ctypes.c_void_p(bp))


if __name__ == "__main__":
    main()
