# iGPU (Radeon 780M) shared-DRAM read bandwidth test via OpenCL ctypes
import ctypes, time, sys

lib = ctypes.CDLL("OpenCL.dll")
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_TYPE_CPU = 2
CL_MEM_READ_ONLY = 1
CL_MEM_WRITE_ONLY = 2
CL_MEM_READ_WRITE = 4

def errcheck(ret, name):
    if ret != 0:
        raise RuntimeError(f"{name} failed: {ret}")

# --- API signatures ---
lib.clGetPlatformIDs.restype = ctypes.c_int32
lib.clGetDeviceIDs.restype = ctypes.c_int32
lib.clCreateContext.restype = ctypes.c_void_p
lib.clCreateCommandQueue.restype = ctypes.c_void_p
lib.clCreateBuffer.restype = ctypes.c_void_p
lib.clCreateProgramWithSource.restype = ctypes.c_void_p
lib.clCreateKernel.restype = ctypes.c_void_p
lib.clGetDeviceInfo.restype = ctypes.c_int32
lib.clBuildProgram.restype = ctypes.c_int32
lib.clEnqueueNDRangeKernel.restype = ctypes.c_int32
lib.clFinish.restype = ctypes.c_int32

np_ = ctypes.c_uint32(0)
errcheck(lib.clGetPlatformIDs(0, None, ctypes.byref(np_)), "clGetPlatformIDs")
print(f"platforms: {np_.value}")
plats = (ctypes.c_void_p * 8)()
errcheck(lib.clGetPlatformIDs(8, plats, None), "clGetPlatformIDs2")
dev = None
devname = ""
for pi in range(np_.value):
    p = plats[pi]
    nd_ = ctypes.c_uint32(0)
    devs = (ctypes.c_void_p * 16)()
    errcheck(lib.clGetDeviceIDs(ctypes.c_void_p(p), 0xFFFFFFFF, 16, devs, ctypes.byref(nd_)), "clGetDeviceIDs")
    for i in range(nd_.value):
        buf = ctypes.create_string_buffer(256)
        lib.clGetDeviceInfo(ctypes.c_void_p(devs[i]), 0x102B, 256, buf, None)
        name = buf.value.decode(errors="replace")
        cu = ctypes.c_uint32(0); lib.clGetDeviceInfo(ctypes.c_void_p(devs[i]), 0x1002, 4, ctypes.byref(cu), None)
        clk = ctypes.c_uint32(0); lib.clGetDeviceInfo(ctypes.c_void_p(devs[i]), 0x103C, 4, ctypes.byref(clk), None)
        gm = ctypes.c_uint64(0); lib.clGetDeviceInfo(ctypes.c_void_p(devs[i]), 0x1020, 8, ctypes.byref(gm), None)
        print(f"  plat[{pi}] dev[{i}] {name} | CUs {cu.value} | clock {clk.value} MHz | global mem {gm.value/1e9:.1f} GB")
        if "Radeon" in name or "780M" in name:
            dev = ctypes.c_void_p(devs[i]); devname = name
if dev is None:
    raise SystemExit("no AMD iGPU device found via OpenCL")
print("using:", devname)
ctx = lib.clCreateContext(None, 1, ctypes.byref(dev), None, None, None)
assert ctx, "clCreateContext failed"
q = lib.clCreateCommandQueue(ctx, dev, 0, None)
assert q, "clCreateCommandQueue failed"
KERNEL_SRC = r"""
__kernel void read_bw(__global const uchar* src, __global uchar* dst, ulong bytes) {
  ulong gid = get_global_id(0);
  ulong n = get_global_size(0);
  ulong per = (bytes + n - 1) / n;
  ulong start = gid * per;
  ulong end = start + per;
  if (end > bytes) end = bytes;
  uchar acc = 0;
  for (ulong i = start; i + 16 <= end; i += 16) {
    uchar16 v = vload16(0, src + i);
    acc ^= v.s0 ^ v.s1 ^ v.s2 ^ v.s3 ^ v.s4 ^ v.s5 ^ v.s6 ^ v.s7;
  }
  if (acc == 0x55) dst[gid & 0xFFFF] = 1;
}
"""
src_c = ctypes.c_char_p(KERNEL_SRC.encode())
prog = lib.clCreateProgramWithSource(ctx, 1, ctypes.byref(src_c), None, None)
assert prog, "clCreateProgramWithSource failed"
rc = lib.clBuildProgram(prog, 1, ctypes.byref(dev), None, None, None)
if rc != 0:
    log = ctypes.create_string_buffer(8192)
    lib.clGetProgramBuildInfo(prog, dev, 0x1183, 8192, log, None)  # CL_PROGRAM_BUILD_LOG
    raise RuntimeError("build failed:\n" + log.value.decode(errors="replace"))
ker = lib.clCreateKernel(prog, b"read_bw", None)
assert ker, "clCreateKernel failed"

def bench(buf, nbytes, gsize, iters=5):
    dst = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 65536, None, None)
    arg_bytes = ctypes.c_uint64(nbytes)
    lib.clSetKernelArg(ker, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(ctypes.c_void_p(buf)))
    lib.clSetKernelArg(ker, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(ctypes.c_void_p(dst)))
    lib.clSetKernelArg(ker, 2, ctypes.sizeof(arg_bytes), ctypes.byref(arg_bytes))
    gsz = (ctypes.c_size_t * 1)(gsize)
    best = float("inf")
    for _ in range(iters):
        lib.clFinish(q)
        t0 = time.perf_counter()
        rc = lib.clEnqueueNDRangeKernel(q, ker, 1, None, gsz, None, 0, None, None)
        if rc != 0: raise RuntimeError(f"enqueue {rc}")
        lib.clFinish(q)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return nbytes / best / 1e9

SIZE = 512 << 20
buf = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, SIZE, None, None)
assert buf, "clCreateBuffer failed"
print(f"\n-- iGPU read-only DRAM bandwidth ({SIZE>>20} MB, min of 5) --")
for gsize in [16384, 65536, 262144, 1048576]:
    gbps = bench(buf, SIZE, gsize)
    print(f"  work-items {gsize:>8d} ({(SIZE/gsize/1024):.1f} KB/wi): {gbps:7.1f} GB/s")
