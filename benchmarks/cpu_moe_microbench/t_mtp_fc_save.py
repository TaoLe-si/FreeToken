"""Save MTP fc weights AND random act to a single binary for cpp."""
import numpy as np, os, json, torch, safetensors.torch, struct
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
state = safetensors.torch.load_file(os.path.join(base, fc_file))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

M, K = 2048, 4096
nbPerRow = K // 8  # 512
nsPerRow = K // 32  # 128

rng = np.random.default_rng(42)
act = rng.normal(0, 1, size=K).astype(np.float32)

with open(out + "/t_mtp_fc_with_act.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(fc_w.tobytes())
    f.write(fc_b.tobytes())  # biases
    f.write(fc_s.tobytes())  # scales
    f.write(act.tobytes())    # act
print(f"Saved {out}/t_mtp_fc_with_act.bin")

# Compute reference using both formulas and report which matches
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

outv1 = np.zeros(M, dtype=np.float32)
outv2 = np.zeros(M, dtype=np.float32)
outv3 = np.zeros(M, dtype=np.float32)
for r in range(M):
    W = unpack_row(fc_w[r], K).astype(np.float32)
    for b in range(nsPerRow):
        kstart = b * 32
        wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
        outv1[r] += wsum * fc_s[r, b]  # no bias
        outv2[r] += (wsum + fc_b[r, b]) * fc_s[r, b]  # add bias then scale
        outv3[r] += wsum * fc_s[r, b] + fc_b[r, b]  # scale then add bias

# Read D3D12 output
d3d = np.fromfile(out + "/t_mtp_fc_output.bin", dtype=np.float32)
# Skip garbage row 0
print(f"\nref1[:5]={outv1[1:6]}\nref2[:5]={outv2[1:6]}\nref3[:5]={outv3[1:6]}\nD3D12[1:6]={d3d[1:6]}")
for name, ref in [("no bias", outv1), ("add_bias*scale", outv2), ("wsum*scale+add_bias", outv3)]:
    diff = d3d[1:] - ref[1:]
    print(f"{name:30s}: max_diff={np.abs(diff).max():.4e}  mean={np.abs(diff).mean():.4e}")
