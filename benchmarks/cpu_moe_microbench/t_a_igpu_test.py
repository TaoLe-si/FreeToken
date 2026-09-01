"""Test iGPU FC client + sticky + compare with PyTorch ref."""
import numpy as np
import os
import sys
import struct
import torch
import time

# Add FreeToken Python to path
sys.path.insert(0, "E:\\FreeToken\\python")
sys.path.insert(0, "E:\\FreeToken\\benchmarks\\cpu_moe_microbench")

from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky

base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"

# Load real fc weights
import safetensors.torch
import json as _json
with open(os.path.join(base, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()  # (M, K/8) uint32 packed
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)  # (M, K/32) float32 NVFP4 bias
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)  # (M, K/32) float32 NVFP4 scale

M, K = fc_w.shape[0], fc_w.shape[1] * 8
print(f"Loaded MTP fc: M={M}, K={K}, packed.shape={fc_w.shape}, scales.shape={fc_s.shape}, biases.shape={fc_b.shape}")
print(f"fc_w[0, :2] = {fc_w[0, :2]}")
print(f"fc_s[0, :3] = {fc_s[0, :3]}")
print(f"fc_b[0, :3] = {fc_b[0, :3]}")

# Use first row
fc_packed = fc_w[0:1].copy()  # (1, K/8)
scales_f32 = fc_s[0:1].copy()  # (1, K/32)
biases_f32 = fc_b[0:1].copy()  # (1, K/32)

# Load real act from bin
with open(os.path.join(out, "t_mtp_fc_with_act.bin"), "rb") as f:
    fileM, fileK, nb, ns = struct.unpack("IIII", f.read(16))
    f.read(fileM * nb * 4)
    f.read(fileM * ns * 4)
    f.read(fileM * ns * 4)
    act = np.frombuffer(f.read(fileK * 4), dtype=np.float32)
print(f"act: shape={act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")

# PyTorch reference (NVFP4 formula)
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
def unpack_row(packed_row_u32, K):
    packed_bytes = packed_row_u32.view(np.uint8)
    n = np.arange(K, dtype=np.int64)
    uint_idx = n // 8
    byte_idx = (n // 2) % 4
    bit = (n % 2) * 4
    flat = uint_idx * 4 + byte_idx
    b = packed_bytes[flat]
    nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
    return kE2M1[nibble.astype(np.int64)]

ref_outv = np.zeros(M, dtype=np.float32)
for r in range(M):
    W = unpack_row(fc_w[r], K).astype(np.float32)
    for b in range(ns):
        kstart = b * 32
        wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
        ref_outv[r] += (wsum + fc_b[r, b]) * fc_s[r, b]
print(f"\\nPyTorch ref: outv[0:3] = {ref_outv[:3]}")
print(f"  outv[0] = {ref_outv[0]:.6f}")

# iGPU test
print("\\n=== iGPU GEMV test ===")
t0 = time.time()
client = IgpuFcClient()
print(f"Client opened in {time.time()-t0:.2f}s")
t0 = time.time()
outv_gpu = client.forward(fc_packed, act.view(np.int32), scales_f32, biases_f32)
print(f"iGPU forward: {(time.time()-t0)*1000:.1f}ms")
print(f"iGPU outv[0:3] = {outv_gpu[:3]}")
print(f"  outv[0] = {outv_gpu[0]:.6f}")
print(f"\\nDiff: max abs = {np.abs(ref_outv[:1] - outv_gpu).max():.4e}")
print(f"Rel err: max = {np.abs((ref_outv[:1] - outv_gpu) / (ref_outv[:1] + 1e-9)).max():.4e}")
