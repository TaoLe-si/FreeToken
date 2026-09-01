import os, struct, numpy as np, json, torch, safetensors.torch
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

M, K = 128, 4096
nbPerRow = K // 8
nsPerRow = K // 32
rng = np.random.default_rng(42)
act = rng.normal(0, 1, size=K).astype(np.float32)

with open(out + "/t_p1b_m128.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(fc_w[:M].tobytes())  # first M rows
    f.write(fc_b[:M].tobytes())
    f.write(fc_s[:M].tobytes())
    f.write(act.tobytes())
print(f"Saved M={M} test data")
