"""P0 test - NVFP4 format expected by t_nvfp4_gemv_sk.dxil.
Protocol:
  cmd: "STATELESS M K szP szS szA szB\n"
  body: packed (szP bytes) + scales (szS bytes) + act (szA bytes) + per-block bias (szB bytes)
  per-row bias (rR) hardcoded to 0 in server
  gbl hardcoded to 1.0 in server
"""
import numpy as np
import os
import struct
import subprocess
import time

base = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

# Test: M=1 K=32, simple NVFP4 GEMV
# Expected: outv[0] = sum_k (W[k] * act[k] + bias_b) * scale_b
#          = 32 * (1 * 1 + 0) * 1.0 = 32
M, K = 1, 32
nb = K // 8   # 4 uints = 16 bytes
ns = K // 32  # 1 micro-block

# packed: each uint32 = 8 nibbles. For weight=1, nibble=1, byte=0x11, uint=0x11111111
packed = np.array([[0x11111111] * nb], dtype=np.uint32)
szP = packed.size * 4  # 16 bytes

# scales: NVFP4 fp16 scale (here converted to float32, M*ns*4 bytes)
# server 期望 M*ns 个 float, 直接 memcpy 到 rS
scales = np.ones((M, ns), dtype=np.float32)
szS = scales.size * 4  # M*ns*4 bytes = 4 bytes

# act: K int32 = 1 each (server 端转 float)
act = np.ones(K, dtype=np.int32)
szA = act.size * 4  # 128 bytes

# bias: NVFP4 per-block bias (M*ns floats)
bias_pb = np.zeros((M, ns), dtype=np.float32)
szB = bias_pb.size * 4  # M*ns*4 = 4 bytes

print("M=%d K=%d szP=%d szS=%d szA=%d szB=%d" % (M, K, szP, szS, szA, szB))

proc = subprocess.Popen(
    [exe],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    bufsize=0, cwd=base,
)
time.sleep(2.0)

import threading
def drain():
    while True:
        try:
            l = proc.stderr.readline()
        except Exception:
            return
        if not l: return
        print("[server]", l.decode(errors="replace").rstrip())
threading.Thread(target=drain, daemon=True).start()

cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)).encode()
proc.stdin.write(cmd)
print("Sent cmd:", cmd.rstrip())
proc.stdin.flush()

proc.stdin.write(packed.tobytes())
proc.stdin.write(scales.tobytes())
proc.stdin.write(act.tobytes())
proc.stdin.write(bias_pb.tobytes())
proc.stdin.flush()
print("Sent body: packed(16) + scales(4) + act(128) + bias_pb(4)")

rl = proc.stdout.read(4)
print("Response len header:", rl)
if len(rl) < 4:
    print("ERROR: short read")
    proc.kill()
    import sys
    sys.exit(1)
sz = struct.unpack('<I', rl)[0]
print("Response data len:", sz)
data = proc.stdout.read(sz)
print("Read data bytes:", len(data))
outv = np.frombuffer(data, dtype=np.float32) if len(data) >= sz else None
print("outv =", outv)
if outv is not None:
    print("diff vs 32:", outv[0] - 32.0)
    print("PASS" if abs(outv[0] - 32.0) < 0.5 else "FAIL")

proc.stdin.write(b"QUIT" + b"\n")
proc.stdin.flush()
time.sleep(0.3)
proc.kill()
