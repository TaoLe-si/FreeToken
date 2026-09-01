"""Benchmark MXFP4 dequant for 256 experts."""
import sys, time, torch
sys.path.insert(0, r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench')
import safetensors.torch, os
from t_mxfp4_dequant import dequant_mxfp4_expert_block

base = r"E:\\\\models\\\\Qwen3.6-35B-A3B-MXFP4-MTP"
state = safetensors.torch.load_file(os.path.join(base, "model-00022-of-00023.safetensors"))

# 256 experts gate_proj: [256, 512, 256] uint32
gate_packed = state["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]
gate_scales = state["mtp.layers.0.mlp.switch_mlp.gate_proj.scales"]
gate_biases = state["mtp.layers.0.mlp.switch_mlp.gate_proj.biases"]
print(f"gate_packed: {gate_packed.shape} {gate_packed.dtype}")
print(f"gate_scales: {gate_scales.shape} {gate_scales.dtype}")
print(f"gate_biases: {gate_biases.shape} {gate_biases.dtype}")

# Dequant all 256 experts
t0 = time.perf_counter()
gate_dequant = dequant_mxfp4_expert_block(gate_packed, gate_scales, gate_biases)
t1 = time.perf_counter()
print(f"Dequant 256 experts gate: {(t1-t0)*1000:.0f}ms, shape={gate_dequant.shape}, dtype={gate_dequant.dtype}")
print(f"Output bytes: {gate_dequant.numel() * 4 / 1024 / 1024:.0f}MB (float32) or {gate_dequant.numel() * 2 / 1024 / 1024:.0f}MB (bf16)")

# Same for up
up_packed = state["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]
up_scales = state["mtp.layers.0.mlp.switch_mlp.up_proj.scales"]
up_biases = state["mtp.layers.0.mlp.switch_mlp.up_proj.biases"]
t0 = time.perf_counter()
up_dequant = dequant_mxfp4_expert_block(up_packed, up_scales, up_biases)
t1 = time.perf_counter()
print(f"Dequant 256 experts up: {(t1-t0)*1000:.0f}ms")

# Same for down
down_packed = state["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]
down_scales = state["mtp.layers.0.mlp.switch_mlp.down_proj.scales"]
down_biases = state["mtp.layers.0.mlp.switch_mlp.down_proj.biases"]
t0 = time.perf_counter()
down_dequant = dequant_mxfp4_expert_block(down_packed, down_scales, down_biases)
t1 = time.perf_counter()
print(f"Dequant 256 experts down: {(t1-t0)*1000:.0f}ms")

# Total
total_t = (t1 - time.perf_counter()) + 0  # ignore
print(f"\\nNote: this is one-time dequant. After this, weights are bf16 in CPU RAM.")
print(f"Total RAM needed: 3 experts * 256 * 4096 * 4 bytes = {(3 * 256 * 4096 * 4) / 1024 / 1024:.0f}MB (bf16: 2 bytes/elem)")
