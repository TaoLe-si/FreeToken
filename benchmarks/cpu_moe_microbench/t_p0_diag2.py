"""P0 diagnostic 2: what value does the v3 server return when bias is set?"""
import numpy as np
import os
import struct
import subprocess
import sys
import time
import threading

base = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

proc = subprocess.Popen(
    [exe],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    bufsize=0, cwd=base,
)
time.sleep(2.0)

def drain():
    while True:
        try:
            l = proc.stderr.readline()
        except Exception:
            return
        if not l: return
        print("[server]", l.decode(errors="replace").rstrip())

threading.Thread(target=drain, daemon=True).start()

def call_stateless(M, K, packed, scl, act, bias):
    szP = packed.size * 4
    szS = scl.size * 4
    szA = act.size * 4
    szB = bias.size * 4
    cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)).encode()
    proc.stdin.write(cmd)
    proc.stdin.write(packed.tobytes())
    proc.stdin.write(scl.tobytes())
    proc.stdin.write(act.tobytes())
    proc.stdin.write(bias.tobytes())
    proc.stdin.flush()
    rl = proc.stdout.read(4)
    if len(rl) < 4: return None
    sz = struct.unpack('<I', rl)[0]
    data = proc.stdout.read(sz)
    if len(data) < sz: return None
    return np.frombuffer(data, dtype=np.float32)

# Test 1: all zeros, bias=0
M, K = 1, 32
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
scl = np.zeros((M, ns), dtype=np.uint32)
act = np.zeros(K, dtype=np.int32)
bias = np.zeros(M, dtype=np.float32)
outv = call_stateless(M, K, packed, scl, act, bias)
print("T1 all-zero bias=0: outv=%s (expect 0)" % outv)

# Test 2: all zeros, bias=5
bias2 = np.array([5.0], dtype=np.float32)
outv = call_stateless(M, K, packed, scl, act, bias2)
print("T2 all-zero bias=5: outv=%s (expect 5 if shader reads bias correctly)" % outv)

# Test 3: packed=0x11, scl=127, act=1, bias=0
packed[:] = 0x11111111
scl[:] = 127
act[:] = 1
outv = call_stateless(M, K, packed, scl, act, bias)
print("T3 nibble=1, scale=1, act=1, bias=0: outv=%s (expect 32)" % outv)

# Test 4: same as T3 but with bias=5
outv = call_stateless(M, K, packed, scl, act, bias2)
print("T4 nibble=1, scale=1, act=1, bias=5: outv=%s (expect 37)" % outv)

# Test 5: same as T3 but gbl... wait, gbl is hardcoded to 1 in server (line 218)
# gbl=1 baked-in, so cannot test gbl. But bias goes into bias resource.

# Test 6: bigger K, see if still 0
M, K = 1, 256
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
packed[:] = 0x11111111
scl = np.zeros((M, ns), dtype=np.uint32)
# Set all bytes in scl to 127 (e8m0 scale=1)
scl_bytes = np.full(ns * 4, 127, dtype=np.uint8).view(np.uint32)
scl[:] = scl_bytes
act = np.ones(K, dtype=np.int32)
outv = call_stateless(M, K, packed, scl, act, bias)
print("T6 M=1 K=256 nibble=1 scale=1 act=1 bias=0: outv=%s (expect 256)" % outv)

outv = call_stateless(M, K, packed, scl, act, bias2)
print("T7 M=1 K=256 nibble=1 scale=1 act=1 bias=5: outv=%s (expect 261)" % outv)

proc.stdin.write(b"QUIT" + b"\n")
proc.stdin.flush()
time.sleep(0.3)
proc.kill()
