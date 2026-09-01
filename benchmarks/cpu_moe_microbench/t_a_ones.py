"""iGPU single-row test with REAL weights, but verify bit-exact."""
import numpy as np
import os
import sys
import struct
import torch

sys.path.insert(0, "E:\\FreeToken\\python")

from freetoken.kernel.igpu_fc import IgpuFcClient

base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"

import safetensors.torch
import json as _json
with open(os.path.join(base, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()  # (2048, 512) uint32 packed
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)  # (2048, 128) float32
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)  # (2048, 128) float32

# Use first row
fc_packed = fc_w[0:1].copy()
scales_f32 = fc_s[0:1].copy()
biases_f32 = fc_b[0:1].copy()

# Use ZERO act (not random) to debug
K = 4096
act_zero = np.zeros(K, dtype=np.int32)
print("act_zero: all 0")

# iGPU test
client = IgpuFcClient()
outv_zero = client.forward(fc_packed, act_zero, scales_f32, biases_f32)
print(f"iGPU outv_zero[0] = {outv_zero[0]:.6f}")
# Expected: 0 (because act is 0, wsum = 0, (0+fc_b)*fc_s = fc_b*fc_s != 0)
# But PyTorch ref would be sum_b (0 + fc_b[0,b]) * fc_s[0,b] = sum_b fc_b * fc_s

# iGPU test with constant 1.0 act (int32 0x3F800000 = 1065353216)
act_ones = np.full(K, 1065353216, dtype=np.int32)
print("act_ones: 0x3F800000 (1.0)")
outv_ones = client.forward(fc_packed, act_ones, scales_f32, biases_f32)
print(f"iGPU outv_ones[0] = {outv_ones[0]:.6f}")
