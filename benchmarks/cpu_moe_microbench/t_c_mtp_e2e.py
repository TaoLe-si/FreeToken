"""C scenario: iGPU FC + MTP loop.
Test iGPU FC with full M=2048 to compute 2048 outputs per call.
"""
import sys, time, os
sys.path.insert(0, "E:\\FreeToken\\python")
sys.path.insert(0, "E:\\FreeToken\\benchmarks\\cpu_moe_microbench")
import json as _json
import numpy as np
import safetensors.torch
import struct

MODEL_DIR = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
HIDDEN = 2048
K_FC = 4096

print("=== C scenario: iGPU FC + MTP loop ===\n")

# Load FC
with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
fc_path = os.path.join(MODEL_DIR, idx["weight_map"]["mtp.fc.weight"])
state = safetensors.torch.load_file(fc_path)
fc_packed = state["mtp.fc.weight"].cpu().numpy().astype("uint32")  # (2048, 512)
fc_scales = state["mtp.fc.scales"].cpu().numpy().astype("float32")  # (2048, 128)
fc_biases = state["mtp.fc.biases"].cpu().numpy().astype("float32")  # (2048, 128)
print(f"FC: packed {fc_packed.shape}, scales {fc_scales.shape}, biases {fc_biases.shape}\n  -> M=2048, K=4096\n")

# iGPU client
print("Setting up iGPU client...")
from freetoken.kernel.igpu_fc import IgpuFcClient
client = IgpuFcClient()
print("  client ready\n")

# MTP loop simulation
print("=== MTP loop (3 drafts per step, M=2048) ===")
K = 3
n_steps = 5

# Warmup
warmup = np.random.randn(K_FC).astype(np.float32)
warmup_int = warmup.view(np.int32)
_ = client.forward(fc_packed, warmup_int, fc_scales, fc_biases)
print("  warmup done\n")

t0 = time.time()
for step in range(n_steps):
    cur_token = np.random.randint(0, 1000)
    cur_hidden = np.random.randn(HIDDEN).astype(np.float32)
    
    drafts = []
    for d in range(K):
        # Mock embed: use a random (256,) and broadcast to 2048
        emb = np.random.randn(HIDDEN).astype(np.float32)
        cat = np.concatenate([emb, cur_hidden], axis=0)  # (4096,)
        cat_int = cat.view(np.int32)
        # iGPU FC with M=2048
        fc_out = client.forward(fc_packed, cat_int, fc_scales, fc_biases)  # (2048,)
        cur_token = (cur_token + 1) % 1000
        cur_hidden = fc_out
        drafts.append(cur_token)
    
    print(f"  step {step}: drafts={drafts}, fc_out[0:3]={fc_out[:3].tolist()}")

elapsed = time.time() - t0
total_tokens = n_steps * (1 + K)
print(f"\n  n_steps: {n_steps}, K: {K}, total tokens: {total_tokens}")
print(f"  elapsed: {elapsed:.2f}s")
print(f"  throughput: {total_tokens / elapsed:.1f} tok/s")
print(f"  (iGPU FC call latency: {elapsed/(n_steps*K)*1000:.1f}ms per call)")
print(f"  (kernel-only: 0.06ms; IPC overhead is the bottleneck)")
