import numpy as np, struct, os
out = r"E:\FreeToken\benchmarks\cpu_moe_microbench"
with open(out + r"\t_mtp_fc_with_act.bin", "rb") as f:
    M, K, nbPerRow, nsPerRow = struct.unpack("IIII", f.read(16))
    fc_w = np.frombuffer(f.read(M*nbPerRow*4), dtype=np.uint32).reshape(M, nbPerRow)
    fc_b = np.frombuffer(f.read(M*nsPerRow*4), dtype=np.float32).reshape(M, nsPerRow)
    fc_s = np.frombuffer(f.read(M*nsPerRow*4), dtype=np.float32).reshape(M, nsPerRow)
    act = np.frombuffer(f.read(K*4), dtype=np.float32)

kE2M1 = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
def unpack(packed_row):
    packed_bytes = packed_row.view(np.uint8)
    n = np.arange(len(packed_row)*8, dtype=np.int64)
    uint_idx = n // 8; byte_idx = (n // 2) % 4; bit = (n % 2) * 4
    flat = uint_idx * 4 + byte_idx
    b = packed_bytes[flat]
    nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
    return kE2M1[nibble.astype(np.int64)]

W = unpack(fc_w[0]).astype(np.float32)
S_bytes = fc_s[0].view(np.uint8)
nb = K // 32
m = np.arange(nb)
flat_idx = m // 4 * 4 + (m % 4)
sb = S_bytes[flat_idx].astype(np.int32)
S = np.where(sb == 0, 0.0, np.exp2(sb.astype(np.float32) - 127.0)).astype(np.float32)
S_full = np.repeat(S, 32)
outv = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
    outv += (wsum + fc_b[0, b]) * S[b]
print("CPU ref outv[0] =", outv)
print("fcW[0:5]=", fc_w[0,:5].tolist())
print("fcB[0,:5]=", fc_b[0,:5].tolist())
print("fcS[0,:5]=", fc_s[0,:5].tolist())
print("act[:5]=", act[:5].tolist())
print("W[:5]=", W[:5])
