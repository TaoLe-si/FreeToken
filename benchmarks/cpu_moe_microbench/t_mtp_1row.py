"""Single-row NVFP4 GEMV debug. M=1, K=32."""
import os, struct, json, torch, safetensors.torch
import numpy as np
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
state = safetensors.torch.load_file(os.path.join(base, fc_file))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

# Row 0, micro-block 0 (32 elements)
w_packed = fc_w[0, :4]  # 4 uints
b_one = fc_b[0, 0]
s_one = fc_s[0, 0]

# Simple act: all 1s
act = np.ones(32, dtype=np.float32)

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
packed_bytes = w_packed.view(np.uint8)
n = np.arange(32, dtype=np.int64)
uint_idx = n // 8
byte_idx = (n // 2) % 4
bit = (n % 2) * 4
flat = uint_idx * 4 + byte_idx
b = packed_bytes[flat]
nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
W = kE2M1[nibble.astype(np.int64)]
print("W =", W)
wsum = (W.astype(np.float32) * act).sum()
print(f"wsum = {wsum}")
# With bias=-0.024 and scale=0.0027:
result = (wsum + b_one) * s_one
print(f"with b={b_one:.5f}, s={s_one:.5f}: result = {result}")

# Save for cpp single-row test
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
M, K = 1, 32
nbPerRow = 1
nsPerRow = 1
with open(out + "/t_mtp_fc_1row.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(w_packed.tobytes())
    f.write(np.array([b_one], dtype=np.float32).tobytes())
    f.write(np.array([s_one], dtype=np.float32).tobytes())
    f.write(act.tobytes())
print(f"Saved t_mtp_fc_1row.bin")
print(f"\nExpected: {result}")
