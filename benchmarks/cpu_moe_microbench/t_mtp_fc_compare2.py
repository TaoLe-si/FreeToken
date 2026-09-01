"""Ref WITHOUT rounding, to see if matches shader output."""
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

# Test float sum (no int round)
r = 0
W = unpack_row(fc_w[r], K).astype(np.float32)
outv_f = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()  # float
    outv_f += (wsum + fc_b[r, b]) * fc_s[r, b]

# Test with int round (matches shader)
outv_i = 0.0
for b in range(nsPerRow):
    kstart = b * 32
    wsum = np.round((W[kstart:kstart+32] * act[kstart:kstart+32]).sum())
    outv_i += (wsum + fc_b[r, b]) * fc_s[r, b]

d3d = np.fromfile(out + "/t_mtp_fc_clean_output.bin", dtype=np.float32)
print(f"ref (float) = {outv_f}")
print(f"ref (int round) = {outv_i}")
print(f"D3D12 = {d3d[0]}")
print(f"abs diff (float) = {abs(d3d[0] - outv_f):.4e}")
print(f"abs diff (int) = {abs(d3d[0] - outv_i):.4e}")
