"""P0 diagnostic 3: each call uses fresh server. NVFP4 protocol."""
import numpy as np
import os
import struct
import subprocess
import sys
import time

base = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

def call_one(M, K, packed, scl, act, bias_pb, tag):
    """NVFP4 protocol: packed (M*nb uint32) + scl (M*ns float) + act (K int32) + bias_pb (M*ns float)."""
    proc = subprocess.Popen(
        [exe],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, cwd=base,
    )
    time.sleep(1.5)
    szP = packed.size * 4       # M*nb*4
    szS = scl.size * 4          # M*ns*4 (NVFP4 float scale)
    szA = act.size * 4          # K*4 (int32, server 端转 float)
    szB = bias_pb.size * 4      # M*ns*4 (per-block bias float)
    cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)).encode()
    proc.stdin.write(cmd)
    proc.stdin.write(packed.tobytes())
    proc.stdin.write(scl.tobytes())
    proc.stdin.write(act.tobytes())
    proc.stdin.write(bias_pb.tobytes())
    proc.stdin.flush()
    rl = proc.stdout.read(4)
    if len(rl) < 4:
        out, _ = proc.communicate(timeout=1)
        print("  [%s] server dead, stderr: %s" % (tag, out.decode(errors='replace')[-500:]))
        return None
    sz = struct.unpack('<I', rl)[0]
    data = proc.stdout.read(sz)
    if len(data) < sz:
        out, _ = proc.communicate(timeout=1)
        print("  [%s] short read, stderr: %s" % (tag, out.decode(errors='replace')[-500:]))
        return None
    outv = np.frombuffer(data, dtype=np.float32)
    proc.stdin.write(b"QUIT" + b"\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=1)
    except:
        proc.kill()
    return outv

# Test all-zero
M, K = 1, 32
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
scl = np.zeros((M, ns), dtype=np.float32)  # NVFP4 fp16 scale (as float32)
act = np.zeros(K, dtype=np.float32).view(np.int32)
bias_pb = np.zeros((M, ns), dtype=np.float32)  # per-block bias

# T1: zero everything, bias=0
outv = call_one(M, K, packed, scl, act, bias_pb, "T1-zeros")
print("T1 all-zero bias=0: outv=%s (expect 0)" % outv)

# T2: zero packed/scl/act, bias_pb=5
bias_pb[:] = 5.0
outv = call_one(M, K, packed, scl, act, bias_pb, "T2-bias5")
print("T2 zero-input bias=5: outv=%s (expect 5 if bias is read correctly)" % outv)

# T3: full data, bias_pb=0
packed[:] = 0x11111111
scl[:] = 1.0  # NVFP4 scale = 1.0
act[:] = 1
bias_pb[:] = 0.0
outv = call_one(M, K, packed, scl, act, bias_pb, "T3-full")
print("T3 packed=1 scale=1 act=1 bias=0: outv=%s (expect 32)" % outv)

# T4: full data, bias_pb=5
bias_pb[:] = 5.0
outv = call_one(M, K, packed, scl, act, bias_pb, "T4-full-bias5")
print("T4 packed=1 scale=1 act=1 bias=5: outv=%s (expect 37 if bias is read)" % outv)

# T5: only bias_pb=5, packed/scl/act all zero
packed[:] = 0
scl[:] = 0
act[:] = 0
bias_pb[:] = 5.0
outv = call_one(M, K, packed, scl, act, bias_pb, "T5-zero-bias5")
print("T5 all-zero bias=5: outv=%s (expect 5 if bias is read)" % outv)

# T6: M=4, K=32
M, K = 4, 32
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
packed[:] = 0x11111111
scl = np.ones((M, ns), dtype=np.float32)
act = np.ones(K, dtype=np.float32).view(np.int32)
bias_pb = np.zeros((M, ns), dtype=np.float32)
outv = call_one(M, K, packed, scl, act, bias_pb, "T6-m4")
print("T6 M=4 K=32 packed=1 scale=1 act=1 bias=0: outv=%s (expect [32, 32, 32, 32])" % outv)

bias_pb[:] = 1.0
outv = call_one(M, K, packed, scl, act, bias_pb, "T7-m4-bias")
print("T7 M=4 K=32 + bias=1: outv=%s (expect [33, 33, 33, 33] since per-block bias adds 1.0 per block)" % outv)
