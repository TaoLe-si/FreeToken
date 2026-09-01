
import os, struct, time, subprocess, torch, safetensors.torch, numpy as np
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
state = safetensors.torch.load_file(r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP\\model-00022-of-00023.safetensors')
# fc: [4096, 512]? check actual shape
fc_w = state['mtp.fc.weight'].contiguous()   # [out, K/8] uint32 packed
fc_s = state['mtp.fc.scales'].contiguous()   # [out, K/32] bf16?
fc_b = state['mtp.fc.biases'].contiguous()   # [out, K/32] bf16?
print('fc_w', tuple(fc_w.shape), fc_w.dtype)
print('fc_s', tuple(fc_s.shape), fc_s.dtype)
print('fc_b', tuple(fc_b.shape), fc_b.dtype)
K = fc_w.shape[1] * 8
print('K =', K, 'out =', fc_w.shape[0])
np.random.seed(0)
act = (np.random.randn(K) * 0.1).astype(np.float32)

# CPU reference with real dequant (reuse t_mxfp4_dequant logic inline)
LUT = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
w = fc_w[0].view(torch.uint8).numpy()  # bytes
s = fc_s[0].view(torch.uint8).numpy()
b = fc_b[0].view(torch.uint8).numpy()
# packed: uints little-endian, nibbles low-first
wu = w.view(np.uint32).copy()  # NOT view - copy as uint8 view first
print('w bytes first 16:', w[:16].tolist())
# dequant per 32-block
nb = K // 8  # uints per row
ns = K // 32
acc = 0.0
for blk in range(ns):
    base = blk * 32
    # scale: read 2 bytes bf16 from fc_s[blk]
    sb = s[2*blk:2*blk+2]
    scale_bits = sb[0] | (sb[1] << 8)
    # bf16 -> float32
    f = struct.unpack('<f', struct.pack('<I', scale_bits << 16))[0]
    # bias bf16
    bb = b[2*blk:2*blk+2]
    bias_bits = bb[0] | (bb[1] << 8)
    bias_f = struct.unpack('<f', struct.pack('<I', bias_bits << 16))[0]
    nsum = 0
    for j in range(8):
        u = wu[base//4 + j]
        nib = [(u >> (4*k)) & 0xF for k in range(8)]
        vals = LUT[nib]
        nsum += int(np.dot(vals, act[base + j*8 : base + j*8 + 8]))
    acc += nsum * f
acc += bias_f  # note: bias per block -> actually bias[row] per output row? unclear; try without first
# The shader does (sh[0] + bias[row]) * gbl[row] -- so bias added once per row, not per block
# The dequant in python: out = sum_blk (sum_j nibbles * act * scale_blk) + bias? Let's just compute both and compare
print('CPU ref (per-block bias):', acc)

# Reference B: real dequant function
from t_mxfp4_dequant import dequant_mxfp4_weight_v2
row_w = torch.from_numpy(wu.copy()).view(torch.uint32).reshape(1, -1)
row_s = torch.from_numpy(s.copy()).view(torch.uint16).reshape(1, -1)
row_b = torch.from_numpy(b.copy()).view(torch.uint16).reshape(1, -1)
print('shapes:', row_w.shape, row_s.shape, row_b.shape)
