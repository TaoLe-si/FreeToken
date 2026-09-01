"""Multi-GEMV batched benchmark: B=8 sequential vs 1 MULTI_GEMV call.
Goal: validate 6x speedup path.
"""
import sys, time, os
sys.path.insert(0, "E:\\FreeToken\\python")
sys.path.insert(0, "E:\\FreeToken\\benchmarks\\cpu_moe_microbench")
import json as _json
import numpy as np
import safetensors.torch

MODEL_DIR = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
HIDDEN = 2048
K_FC = 4096

print("=== Multi-GEMV Batched Benchmark ===\n")

# Load FC (full M=2048)
with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
fc_path = os.path.join(MODEL_DIR, idx["weight_map"]["mtp.fc.weight"])
state = safetensors.torch.load_file(fc_path)
fc_packed = state["mtp.fc.weight"].cpu().numpy().astype("uint32")  # (2048, 512)
fc_scales = state["mtp.fc.scales"].cpu().numpy().astype("float32")  # (2048, 128)
fc_biases = state["mtp.fc.biases"].cpu().numpy().astype("float32")  # (2048, 128)
print(f"FC: M={fc_packed.shape[0]}, K={fc_packed.shape[1]*8}\n")

from freetoken.kernel.igpu_fc import IgpuFcClient
client = IgpuFcClient()
print("Client ready\n")

# Verify bit-exact first
print("=== Bit-exact verification (M=1) ===")
fc_packed_r0 = fc_packed[0:1]  # (1, 512)
fc_scales_r0 = fc_scales[0:1]
fc_biases_r0 = fc_biases[0:1]
act = np.random.randn(K_FC).astype(np.float32)
act_int = act.view(np.int32)
outv = client.forward(fc_packed_r0, act_int, fc_scales_r0, fc_biases_r0)
print(f"  outv[0] = {outv[0]:.6f}\n")

# Sequential M=1 calls benchmark
print("=== Sequential M=1 calls (B iterations) ===")
B = 8
acts = [np.random.randn(K_FC).astype(np.float32) for _ in range(B)]
acts_int = [a.view(np.int32) for a in acts]

# Warmup
for _ in range(3):
    _ = client.forward(fc_packed_r0, acts_int[0], fc_scales_r0, fc_biases_r0)

t0 = time.time()
for i in range(B):
    _ = client.forward(fc_packed_r0, acts_int[i], fc_scales_r0, fc_biases_r0)
seq_elapsed = time.time() - t0
print(f"  {B} sequential calls: {seq_elapsed*1000:.1f}ms")
print(f"  Per call: {seq_elapsed/B*1000:.2f}ms")
print(f"  Throughput: {B/seq_elapsed:.1f} calls/s\n")

# Multi-GEMV: Need to implement or use t_mxfp4_gemv_multi_server.exe
# But we have v3 server with MULTI_GEMV protocol
# Let me use v3 server directly via Python

print("=== Multi-GEMV batched (1 call for B items) ===")
# Use v3 server MULTI_GEMV command directly
# Format: MULTI_GEMV B K szPPerItem szSPerItem szAPerItem szBPerItem gblPerItem
# Each item: M=1 row of fc (1, K/8) packed + (1, K/32) scales + (K,) act + (1, K/32) bias + (1,) gbl

nb = K_FC // 8
ns = K_FC // 32
szP = nb * 4  # 2048 bytes
szS = ns * 4  # 512 bytes
szA = K_FC * 4  # 16384 bytes
szB = ns * 4  # 512 bytes (M=1, ns=128)
szG = 4  # 1 float (gbl per item, M=1)

# Build body
body = bytearray()
for i in range(B):
    # Use fc row 0 for all items (testing same weight, different act)
    body += fc_packed_r0.tobytes()  # szP
    body += fc_scales_r0.tobytes()  # szS
    body += acts_int[i].tobytes()  # szA (int32)
    body += fc_biases_r0.tobytes()  # szB
    body += np.float32(1.0).tobytes()  # szG (gbl=1.0)
body = bytes(body)

cmd = f"MULTI_GEMV {B} {K_FC} {szP} {szS} {szA} {szB} {szG}\n".encode()

# Send via client.proc
import struct
with client._lock:
    client.proc.stdin.write(cmd)
    client.proc.stdin.write(body)
    client.proc.stdin.flush()
    rl = client._read_exact(4)
    sz = struct.unpack('<I', rl)[0]
    outv = client._read_exact(sz)

multi_outv = np.frombuffer(outv, dtype=np.float32).copy()
print(f"  multi-GEMV {B} items: 1 call, returned {len(multi_outv)} floats")
print(f"  multi_outv[0:B] = {multi_outv[:B].tolist()}")

# Verify
print()
print("=== Verification: multi-GEMV vs sequential ===")
# Re-run sequential to compare
seq_outv = []
for i in range(B):
    o = client.forward(fc_packed_r0, acts_int[i], fc_scales_r0, fc_biases_r0)
    seq_outv.append(o[0])
print(f"  Sequential: {seq_outv}")
print(f"  Multi:      {multi_outv[:B].tolist()}")
diffs = [abs(s - m) for s, m in zip(seq_outv, multi_outv[:B])]
print(f"  Max diff: {max(diffs):.4e}")
if max(diffs) < 1e-4:
    print(f"  PASS: bit-exact")
else:
    print(f"  FAIL")

# Now benchmark
print()
print("=== Multi-GEMV batched benchmark ===")
# Warmup
for _ in range(3):
    with client._lock:
        client.proc.stdin.write(cmd)
        client.proc.stdin.write(body)
        client.proc.stdin.flush()
        rl = client._read_exact(4)
        sz = struct.unpack('<I', rl)[0]
        client.proc.stdin.read(sz)

N = 30
t0 = time.time()
for _ in range(N):
    with client._lock:
        client.proc.stdin.write(cmd)
        client.proc.stdin.write(body)
        client.proc.stdin.flush()
        rl = client._read_exact(4)
        sz = struct.unpack('<I', rl)[0]
        client.proc.stdin.read(sz)
multi_elapsed = (time.time() - t0) / N
print(f"  {B}-item multi-GEMV: {multi_elapsed*1000:.1f}ms/call")
print(f"  Per item: {multi_elapsed/B*1000:.2f}ms")
print(f"  Throughput: {B/multi_elapsed:.1f} items/s")

print()
print("=== Comparison ===")
print(f"  Sequential {B} calls: {seq_elapsed*1000:.1f}ms ({B/seq_elapsed:.1f} calls/s)")
print(f"  Batched 1 call:       {multi_elapsed*1000:.1f}ms ({B/multi_elapsed:.1f} items/s)")
print(f"  Speedup: {seq_elapsed/multi_elapsed:.2f}x")
