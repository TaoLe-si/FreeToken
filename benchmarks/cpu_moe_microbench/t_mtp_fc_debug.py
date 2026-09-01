"""Debug: check if the cpp is feeding bias/scale to the right shader slots."""
import numpy as np
import struct
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(out + "/t_mtp_fc_weights.bin", "rb") as f:
    M, K, nbPerRow, nsPerRow = struct.unpack("IIII", f.read(16))
    f.read(M * nbPerRow * 4)  # skip packed
    biases = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.float32).reshape(M, nsPerRow)
    scales = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.float32).reshape(M, nsPerRow)
print(f"biases[0, :5] = {biases[0,:5]}")
print(f"scales[0, :5] = {scales[0,:5]}")
# If biases look like "tiny post-RMSNorm offsets", and scales look like "0.002 per-micro-block multipliers"
# then cpp should send biases to t2 (bias reg) and scales to t1 (scl reg).
# That's what my cpp does. Let me verify the reference matches D3D12 if I do it the OTHER way (swapped).
import os
d3d_out = np.fromfile(out + "/t_mtp_fc_output.bin", dtype=np.float32)
print(f"\nD3D12[0] = {d3d_out[0]}")

# Compute manually row 0 with current order (biases to t2, scales to t1)
import torch, safetensors.torch, json
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
state = safetensors.torch.load_file(os.path.join(base, fc_file))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

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

rng = np.random.default_rng(42)
act = rng.normal(0, 1, size=4096).astype(np.float32)

# Compute row 0 using my python reference (matches what's already in ref)
W = unpack_row(fc_w[0], 4096).astype(np.float32)
out0_ref = 0.0
for b in range(128):
    kstart = b * 32
    wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
    out0_ref += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"python ref row 0 = {out0_ref}")
print(f"D3D12 row 0 = {d3d_out[0]}")
print(f"Difference = {d3d_out[0] - out0_ref}")
