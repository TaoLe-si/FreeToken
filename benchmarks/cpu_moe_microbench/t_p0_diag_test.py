"""P0 diagnostic: verify v3 server STATELESS path returns correct value.

v3 server protocol:
  cmd line: "STATELESS M K szP szS szA szB\n"
  body: packed(szP) + scales(szS) + act(szA) + bias(szB)
  out:   4-byte uint32 len + M float32
"""
import numpy as np
import os
import struct
import subprocess
import sys
import time
import threading

base = os.path.dirname(os.path.abspath(__file__))
exe  = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

# Test 1: M=1 K=32, simple GEMV
# Expected: 32 * 1 * 1 * 1 + 0 + 0 = 32
M, K = 1, 32
nbPerRow = K // 8   # 4
nsPerRow = K // 32  # 1

packed = np.zeros((M, nbPerRow), dtype=np.uint32)
packed[:] = 0x11111111  # each nibble = 1
scl = np.zeros((M, nsPerRow), dtype=np.uint32)
scl[0, 0] = 127  # e8m0 byte 0 = 127 (scale=1)

act = np.ones(K, dtype=np.int32)
bias = np.zeros(M, dtype=np.float32)

szP = packed.size * 4
szS = scl.size * 4
szA = act.size * 4
szB = bias.size * 4

print("=== STATELESS M=1 K=32 ===")
print("  M=%d K=%d szP=%d szS=%d szA=%d szB=%d" % (M, K, szP, szS, szA, szB))
print("  Expected outv[0] = 32")

# Start server
proc = subprocess.Popen(
    [exe],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0,
    cwd=base,
)
time.sleep(2.0)

def drain():
    while True:
        try:
            l = proc.stderr.readline()
        except Exception:
            return
        if not l:
            return
        print("[server]", l.decode(errors="replace").rstrip())

threading.Thread(target=drain, daemon=True).start()

# Send STATELESS line command
cmd = "STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)
print("  cmd: %r" % cmd.rstrip())
proc.stdin.write(cmd.encode())
proc.stdin.flush()

# Body: packed + scales + act + bias
proc.stdin.write(packed.tobytes())
proc.stdin.write(scl.tobytes())
proc.stdin.write(act.tobytes())
proc.stdin.write(bias.tobytes())
proc.stdin.flush()

# Read response: 4 byte len + M float32
rl = proc.stdout.read(4)
if len(rl) < 4:
    print("ERROR: server returned %d bytes for len header" % len(rl))
    proc.kill()
    sys.exit(1)
sz = struct.unpack('<I', rl)[0]
print("  Response len: %d bytes (expected %d)" % (sz, M*4))
data = proc.stdout.read(sz)
if len(data) < sz:
    print("ERROR: short read, got %d/%d" % (len(data), sz))
    proc.kill()
    sys.exit(1)
outv = np.frombuffer(data, dtype=np.float32)
print("  outv = %s" % outv)
print("  diff vs expected (32): %.6e" % (outv[0] - 32.0))

if abs(outv[0] - 32.0) < 0.01:
    print("  PASS")
else:
    print("  FAIL")

# Test 2: M=1 K=4096 (real MTP fc shape) with bit-exact verification
print("\n=== STATELESS M=1 K=4096 (real MTP fc) ===")
M, K = 1, 4096
np.random.seed(42)
nbPerRow = K // 8
nsPerRow = K // 32
packed = np.random.randint(0, 16, size=(M, nbPerRow), dtype=np.uint32)  # random nibbles
scl = np.zeros((M, nsPerRow), dtype=np.uint32)
sc = np.random.randint(120, 130, size=nsPerRow, dtype=np.int32)
for i in range(nsPerRow):
    scl[0, i // 4] |= (sc[i] & 0xFF) << ((i & 3) * 8)
act = np.random.randint(-5, 5, size=K, dtype=np.int32)
bias = np.random.randn(M).astype(np.float32)

szP = packed.size * 4
szS = scl.size * 4
szA = act.size * 4
szB = bias.size * 4

cmd = "STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)
proc.stdin.write(cmd.encode())
proc.stdin.write(packed.tobytes())
proc.stdin.write(scl.tobytes())
proc.stdin.write(act.tobytes())
proc.stdin.write(bias.tobytes())
proc.stdin.flush()

rl = proc.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
data = proc.stdout.read(sz)
outv = np.frombuffer(data, dtype=np.float32)

# CPU reference
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)
nibbles_packed = np.zeros((M, K), dtype=np.int8)
for r in range(M):
    for k in range(K):
        uint_idx = k // 8
        nib_idx = k & 7
        nibbles_packed[r, k] = (packed[r, uint_idx] >> (nib_idx * 4)) & 0xF
weights = kE2M1[nibbles_packed.astype(np.int32)]  # (M, K)

# Extract e8m0 scales
scales = np.zeros((M, K), dtype=np.float32)
for r in range(M):
    for b in range(K // 32):
        packIdx = b >> 2
        byteIdx = b & 3
        sb = (scl[r, packIdx] >> (byteIdx * 8)) & 0xFF
        bs = 0.0 if sb == 0 else 2 ** (int(sb) - 127)
        scales[r, b*32:(b+1)*32] = bs

w_scaled = weights * scales
ref = (w_scaled * act.astype(np.float32)).sum(axis=1) + bias
print("  v3 outv[0]: %.6f" % outv[0])
print("  ref:       %.6f" % ref[0])
print("  abs diff:  %.6e" % abs(outv[0] - ref[0]))
print("  rel diff:  %.6e" % (abs(outv[0] - ref[0]) / max(abs(ref[0]), 1e-6)))
if abs(outv[0] - ref[0]) / max(abs(ref[0]), 1e-6) < 1e-3:
    print("  PASS (rel < 0.1%)")
else:
    print("  FAIL")

# Cleanup
proc.stdin.write(b"QUIT" + b"\n")
proc.stdin.flush()
time.sleep(0.5)
proc.kill()
