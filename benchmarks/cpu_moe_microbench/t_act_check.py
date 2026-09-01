"""Check file act for NaN/Inf."""
import struct, numpy as np
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(out + "/t_mtp_fc_with_act.bin", "rb") as f:
    M, K, nb, ns = struct.unpack("IIII", f.read(16))
    f.read(M * nb * 4)
    f.read(M * ns * 4)
    f.read(M * ns * 4)
    act = np.frombuffer(f.read(K * 4), dtype=np.float32)

print(f"act.shape={act.shape}")
print(f"any nan: {np.isnan(act).any()}, count: {np.isnan(act).sum()}")
print(f"any inf: {np.isinf(act).any()}, count: {np.isinf(act).sum()}")
print(f"min={act.min()}, max={act.max()}, mean={act.mean()}, std={act.std()}")
print(f"first nan idx: {np.argwhere(np.isnan(act)).flatten()[:10] if np.isnan(act).any() else 'none'}")

# Check fc_w[0]
import os, json, safetensors.torch, torch
mdl = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(mdl, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(mdl, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()
print(f"fc_w shape={fc_w.shape}")
print(f"fc_w dtype={fc_w.dtype}")
# uint32 has no nan, but check for zero packed
print(f"fc_w[0] all zeros: {(fc_w[0] == 0).all()}")
print(f"fc_w[0, 0] = 0x{fc_w[0, 0]:08X}")
