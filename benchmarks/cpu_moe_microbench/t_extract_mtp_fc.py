"""Extract MTP fc weights from Qwen3.6 model and test FC GEMV with float activation."""
import numpy as np
import torch
import safetensors.torch
import os
import struct
import sys

base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"

# Find which safetensors file has mtp.fc.weight
import json
with open(os.path.join(base, "model.safetensors.index.json"), "r") as f:
    idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
fc_path = os.path.join(base, fc_file)
print(f"fc in {fc_path}")

# Load the safetensors file
state = safetensors.torch.load_file(fc_path)
fc_w_packed = state["mtp.fc.weight"].cpu().numpy()  # uint32 [2048, 512]
fc_w_biases = state["mtp.fc.biases"].cpu().numpy()  # fp16 [2048, 128]
fc_w_scales = state["mtp.fc.scales"].cpu().numpy()  # fp16 [2048, 128]

# Convert biases/scales to fp32
fc_w_biases_f32 = fc_w_biases.astype(np.float32)
fc_w_scales_f32 = fc_w_scales.astype(np.float32)

# The packed format: each uint32 contains 8 nibbles (4-bit e2m1 weights).
# Each row [out_dim] has K=4096 weights = 512 uints.
# Biases/scales: each row has 128 micro-blocks (K/32). Each micro-block has 1 bias + 1 scale (fp16).
# Layout: biases[r, m] and scales[r, m] for micro-block m of row r.

# Save to binary for cpp host
bin_path = os.path.join(out, "t_mtp_fc_weights.bin")
with open(bin_path, "wb") as f:
    # Header: M K nbPerRow nsPerRow
    M, K = 2048, 4096
    nbPerRow = K // 8  # 512
    nsPerRow = K // 32  # 128
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(fc_w_packed.tobytes())
    f.write(fc_w_biases_f32.tobytes())  # 2048*128 fp32
    f.write(fc_w_scales_f32.tobytes())  # 2048*128 fp32
print(f"Saved {bin_path}, packed={fc_w_packed.shape} biases={fc_w_biases.shape} scales={fc_w_scales.shape}")
print(f"packed[0,:8] = {fc_w_packed[0,:8]}")
print(f"biases[0,:4] = {fc_w_biases[0,:4]}")
print(f"scales[0,:4] = {fc_w_scales[0,:4]}")
