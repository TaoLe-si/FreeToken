"""CPU MXFP4 GEMV timing for M=2048 K=4096."""
import time, os, struct, json, torch, safetensors.torch, numpy as np
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

with open(out + "/t_mtp_fc_with_act.bin", "rb") as f:
    M, K, nbPerRow, nsPerRow = struct.unpack("IIII", f.read(16))
    f.read(M * nbPerRow * 4); f.read(M * nsPerRow * 4); f.read(M * nsPerRow * 4)
    act = np.frombuffer(f.read(K * 4), dtype=np.float32)

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
def unpack_row(packed_row_u32, K):
    packed_bytes = packed_row_u32.view(np.uint8)
    n = np.arange(K, dtype=np.int64)
    uint_idx = n // 8; byte_idx = (n // 2) % 4; bit = (n % 2) * 4
    flat = uint_idx * 4 + byte_idx
    b = packed_bytes[flat]
    nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
    return kE2M1[nibble.astype(np.int64)]

# Warmup
for _ in range(3):
    outv = np.zeros(M, dtype=np.float32)
    for r in range(M):
        W = unpack_row(fc_w[r], K).astype(np.float32)
        for b in range(nsPerRow):
            kstart = b * 32
            wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
            outv[r] += (wsum + fc_b[r, b]) * fc_s[r, b]
    outv = outv + 0  # prevent optimization

import time
times = []
for _ in range(50):
    t0 = time.perf_counter()
    outv = np.zeros(M, dtype=np.float32)
    for r in range(M):
        W = unpack_row(fc_w[r], K).astype(np.float32)
        for b in range(nsPerRow):
            kstart = b * 32
            wsum = (W[kstart:kstart+32] * act[kstart:kstart+32]).sum()
            outv[r] += (wsum + fc_b[r, b]) * fc_s[r, b]
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
times.sort()
print(f"CPU M=2048 K=4096: p50={times[25]:.3f}ms p90={times[45]:.3f}ms")
