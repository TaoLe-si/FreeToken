"""Single-row NVFP4 GEMV test including activation."""
import numpy as np, os, json, torch, safetensors.torch, struct

base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
state = safetensors.torch.load_file(os.path.join(base, fc_file))
fc_w_0 = state["mtp.fc.weight"][0].cpu().numpy().astype(np.uint32)
fc_b_0 = state["mtp.fc.biases"][0].cpu().numpy().astype(np.float32)
fc_s_0 = state["mtp.fc.scales"][0].cpu().numpy().astype(np.float32)

w_packed = fc_w_0[:4]
b_ones = fc_b_0[:1]
s_ones = fc_s_0[:1]
K = 32
M = 1
nbPerRow = 1
nsPerRow = 1

rng = np.random.default_rng(42)
act = rng.normal(0, 1, size=K).astype(np.float32)

with open(out + "/t_mtp_fc_single.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(w_packed.tobytes())
    f.write(b_ones.tobytes())
    f.write(s_ones.tobytes())
    f.write(act.tobytes())  # ADD act

# Compute reference
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
packed_bytes = w_packed.view(np.uint8)
n = np.arange(K, dtype=np.int64)
uint_idx = n // 8
byte_idx = (n // 2) % 4
bit = (n % 2) * 4
flat = uint_idx * 4 + byte_idx
b = packed_bytes[flat]
nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
W = kE2M1[nibble.astype(np.int64)].astype(np.float32)
wsum = (W * act).sum()
ref = (wsum + b_ones[0]) * s_ones[0]
print(f"act = {act}")
print(f"ref = {ref}")
np.save(out + "/t_mtp_fc_single_ref.npy", np.array([ref], dtype=np.float32))
