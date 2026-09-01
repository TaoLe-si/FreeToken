"""Manual reference compute."""
import struct, numpy as np

with open("E:\\FreeToken\\benchmarks\\cpu_moe_microbench\\t_mtp_fc_with_act.bin", "rb") as f:
    M, K, nb, ns = struct.unpack("IIII", f.read(16))
    packed = np.frombuffer(f.read(M * nb * 4), dtype=np.uint32).reshape(M, nb)
    bias_pb = np.frombuffer(f.read(M * ns * 4), dtype=np.float32).reshape(M, ns)
    scl = np.frombuffer(f.read(M * ns * 4), dtype=np.float32).reshape(M, ns)
    act = np.frombuffer(f.read(K * 4), dtype=np.float32)

# e2m1 decode
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)

# Row 0
r = 0
acc = 0.0
for b in range(ns):
    p = packed[r, b*4:(b+1)*4]
    nibbles = np.zeros(32, dtype=np.int32)
    for i in range(4):
        w = int(p[i])
        for j in range(8):
            nib = (w >> (4*j)) & 0xF
            nibbles[i*8 + j] = kE2M1[nib]
    act_b = act[b*32:(b+1)*32]
    wsum = float((nibbles.astype(np.float32) * act_b).sum())
    bb = bias_pb[r, b]
    bs = scl[r, b]
    contrib = (wsum + bb) * bs
    acc += contrib
    if b < 3:
        print(f"  b={b}: wsum={wsum:.4f} bb={bb:.4f} bs={bs:.4f} contrib={contrib:.4f}")

print(f"Total: {acc:.6f}")
print(f"Reference: -1.711113")
print(f"Diff: {acc - (-1.711113):.6f}")
