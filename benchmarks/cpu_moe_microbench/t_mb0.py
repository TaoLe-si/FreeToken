"""Compute each iter's round value."""
import numpy as np, struct, os, json, torch, safetensors.torch
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

with open(out + "/t_mtp_fc_with_act.bin", "rb") as f:
    M, K, nbPerRow, nsPerRow = struct.unpack("IIII", f.read(16))
    f.read(M * nbPerRow * 4)
    f.read(M * nsPerRow * 4)
    f.read(M * nsPerRow * 4)
    act = np.frombuffer(f.read(K * 4), dtype=np.float32)

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

W = unpack_row(fc_w[0], K).astype(np.float32)

# Test various wsum formulas
# 1. Float sum (no round)
outv_f = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
    outv_f += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"outv (float sum) = {outv_f}")

# 2. 4-iters int round (matches shader)
outv_4i = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = 0
    for j in range(4):
        seg = W[kstart + j*8 : kstart + j*8 + 8] * act[kstart + j*8 : kstart + j*8 + 8]
        partial = seg.sum()
        wsum += int(round(partial))
    outv_4i += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"outv (4-iter int round) = {outv_4i}")

# 3. Single round (32 elements)
outv_1i = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = int(round((W[kstart:kstart+32] * act[kstart:kstart+32]).sum()))
    outv_1i += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"outv (1-iter int round) = {outv_1i}")

# 4. Truncate (no round)
outv_t = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = int((W[kstart:kstart+32] * act[kstart:kstart+32]).sum())  # int truncates
    outv_t += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"outv (truncate) = {outv_t}")

# 5. C-style trunc with banker's
outv_b = 0.0
import math
for b in range(nsPerRow):
    kstart = b * 32
    partial = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
    wsum = int(math.floor(partial + 0.5)) if partial >= 0 else -int(math.floor(-partial + 0.5))
    outv_b += (wsum + fc_b[0, b]) * fc_s[0, b]
print(f"outv (banker's) = {outv_b}")

print(f"\nD3D12 = -1.6923")
