"""Test simple shader: outv[0] = bias[0] + 1.0"""
import numpy as np
import os
import struct
import subprocess
import time

base = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

def call(M, K, packed, scl, act, bias, tag):
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, cwd=base)
    time.sleep(1.5)
    szP = packed.size * 4
    szS = len(scl) if isinstance(scl, bytes) else scl.size
    szA = act.size * 4
    szB = bias.size * 4
    cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)).encode()
    proc.stdin.write(cmd)
    proc.stdin.write(packed.tobytes())
    proc.stdin.write(scl if isinstance(scl, bytes) else scl.tobytes())
    proc.stdin.write(act.tobytes())
    proc.stdin.write(bias.tobytes())
    proc.stdin.flush()
    rl = proc.stdout.read(4)
    if len(rl) < 4:
        return None, "short read"
    sz = struct.unpack('<I', rl)[0]
    data = proc.stdout.read(sz)
    outv = np.frombuffer(data, dtype=np.float32)
    proc.stdin.write(b"QUIT" + b"\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=1)
    except:
        proc.kill()
    return outv, None

M, K = 1, 32
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
scl = bytes([0])  # 1 byte scale = 0
act = np.zeros(K, dtype=np.int32)
bias = np.zeros(M, dtype=np.float32)
outv, err = call(M, K, packed, scl, act, bias, "T1")
print("T1 bias=0: outv=%s (expect 1.0)" % outv)

bias = np.array([5.0], dtype=np.float32)
outv, err = call(M, K, packed, scl, act, bias, "T2")
print("T2 bias=5: outv=%s (expect 6.0)" % outv)

M, K = 4, 32
nb = K // 8
ns = K // 32
packed = np.zeros((M, nb), dtype=np.uint32)
scl = bytes([0]*ns*M)  # 4 bytes
act = np.zeros(K, dtype=np.int32)
bias = np.array([1., 2., 3., 4.], dtype=np.float32)
outv, err = call(M, K, packed, scl, act, bias, "T3")
print("T3 M=4 bias=[1,2,3,4]: outv=%s (expect [2,3,4,5])" % outv)
